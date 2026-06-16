"""
Pipeline principal de contagem e classificação de veículos em rodovias.

Orquestra VideoCapture (thread Producer), loop de inferência (thread main),
CrossingCounter e OverlayRenderer com shutdown gracioso via SIGINT/SIGTERM.
"""

from __future__ import annotations

import warnings
warnings.filterwarnings("ignore", category=FutureWarning, module="supervision")

import argparse
import collections
import logging
import queue
import signal
import sys
import threading
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np

from src.capture.video_capture import VideoCapture
from src.config import Settings, load_settings
from src.counting.crossing_logic import CrossingCounter
from src.database.db_writer import DbWriter
from src.database.models import create_sqlite_engine, init_db
from src.detection.yolo_detector import YoloDetector
from src.ocr.plate_ocr import OCRWorker, PlateOCR
from src.rendering.overlay_renderer import OverlayRenderer
from src.tracking.bytetrack_wrapper import ByteTrackWrapper

logger = logging.getLogger(__name__)


# ── Motor de FPS ─────────────────────────────────────────────────────────────

class _FpsMeter:
    """Calcula FPS em janela deslizante usando perf_counter.

    Args:
        window: Número de amostras na janela deslizante (padrão 30).
    """

    def __init__(self, window: int = 30) -> None:
        self._timestamps: collections.deque[float] = collections.deque(maxlen=window)

    def tick(self) -> None:
        """Registra o timestamp do frame atual."""
        self._timestamps.append(time.perf_counter())

    @property
    def fps(self) -> float:
        """FPS médio na janela atual. Retorna 0.0 se dados insuficientes."""
        if len(self._timestamps) < 2:
            return 0.0
        return (len(self._timestamps) - 1) / (self._timestamps[-1] - self._timestamps[0])


# ── Helpers ───────────────────────────────────────────────────────────────────

def _build_writer(
    output_path: Path,
    fps: float,
    frame: np.ndarray,
) -> cv2.VideoWriter:
    """Instancia VideoWriter com as dimensões exatas do primeiro frame recebido.

    Args:
        output_path: Caminho do arquivo de saída.
        fps: Taxa de quadros da fonte de vídeo.
        frame: Primeiro frame real — determina (width, height) do writer.

    Returns:
        cv2.VideoWriter pronto para receber frames.
    """
    h, w = frame.shape[:2]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    return cv2.VideoWriter(str(output_path), fourcc, fps, (w, h))


def _print_summary(
    duration: float,
    total_frames: int,
    avg_fps: float,
    total_count: int,
    class_counts: dict[str, int],
    output_path: str,
) -> None:
    """Registra tabela de estatísticas finais via logger."""
    lines: list[str] = [
        "=" * 52,
        "  PIPELINE ENCERRADO — ESTATÍSTICAS FINAIS",
        "=" * 52,
        f"  Duração da sessão  : {duration:.1f} s",
        f"  Frames avaliados   : {total_frames}",
        f"  FPS médio          : {avg_fps:.1f}",
        f"  Total de veículos  : {total_count}",
        f"  Vídeo anotado      : {output_path}",
        "",
    ]
    if class_counts:
        lines.append("  Contagem por classe:")
        for cls, cnt in sorted(class_counts.items()):
            lines.append(f"    {cls:<22} {cnt}")
    else:
        lines.append("  Nenhum veículo contado.")
    lines.append("=" * 52)
    logger.info("\n" + "\n".join(lines))


# ── Pipeline principal ────────────────────────────────────────────────────────

def main() -> None:
    """Ponto de entrada do pipeline de contagem de veículos.

    Inicializa e orquestra todos os módulos do pipeline em threads separadas.
    Suporta shutdown gracioso via SIGINT (Ctrl+C) e SIGTERM (docker stop).
    Toda a configuração vem exclusivamente do arquivo YAML especificado via --config.

    Raises:
        SystemExit: Se a configuração for inválida ou a fonte de vídeo não puder ser aberta.
    """
    # ── CLI ──────────────────────────────────────────────────────────────
    parser = argparse.ArgumentParser(
        description="Pipeline de contagem e classificação de veículos em rodovias"
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("config/settings.yaml"),
        help="Caminho para config/settings.yaml (padrão: config/settings.yaml)",
    )
    args = parser.parse_args()

    # ── Logging ──────────────────────────────────────────────────────────
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        stream=sys.stdout,
    )

    # ── Configuração ─────────────────────────────────────────────────────
    settings: Settings = load_settings(args.config)
    logger.info("Configuração carregada: %s", args.config)

    # ── Stop event — única fonte de verdade para encerramento ─────────────
    stop_event = threading.Event()

    def _signal_handler(signum: int, _frame: object) -> None:
        """Captura SIGINT/SIGTERM e aciona o stop_event sem join nem I/O."""
        logger.info("Sinal %d recebido — iniciando shutdown gracioso", signum)
        stop_event.set()

    signal.signal(signal.SIGINT, _signal_handler)
    signal.signal(signal.SIGTERM, _signal_handler)

    # ── Identidade da sessão ──────────────────────────────────────────────
    session_id: str = str(uuid.uuid4())
    logger.info("Sessão iniciada: %s", session_id)

    # ── Filas ─────────────────────────────────────────────────────────────
    frame_queue: queue.Queue[np.ndarray] = queue.Queue(maxsize=3)
    ocr_queue: queue.Queue = queue.Queue(maxsize=50)
    db_queue: queue.Queue = queue.Queue(maxsize=200)

    # ── Módulos do pipeline (todos no escopo de main) ─────────────────────
    cap = VideoCapture(
        source=settings.video.source,
        frame_queue=frame_queue,
        settings=settings.video,
    )
    detector = YoloDetector(settings=settings.model)
    tracker = ByteTrackWrapper(settings=settings.tracking)
    counter = CrossingCounter(
        line_points=settings.counting.line_points,
        direction=settings.counting.direction,
        min_displacement_px=settings.counting.min_displacement_px,
        class_vote_window=settings.counting.class_vote_window,
        suv_aspect_ratio_threshold=settings.counting.suv_aspect_ratio_threshold,
        truck_area_threshold=settings.counting.truck_area_threshold,
    )
    renderer = OverlayRenderer(
        settings=settings.rendering,
        line_points=settings.counting.line_points,
    )
    fps_meter = _FpsMeter(window=30)

    # ── Banco de dados ────────────────────────────────────────────────────
    db_path = Path(settings.database.sqlite_path)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    engine = create_sqlite_engine(db_path)
    init_db(engine)
    db_writer = DbWriter(db_queue=db_queue, engine=engine)
    db_writer.start()

    # ── OCR (opcional) ────────────────────────────────────────────────────
    ocr_worker: OCRWorker | None = None
    if settings.ocr.enabled:
        if settings.ocr.engine == "fast_alpr":
            # Engine ONNX end-to-end — localiza e lê placa em um único passo
            from src.ocr.fast_alpr_worker import FastAlprWorker
            ocr_worker = FastAlprWorker(
                ocr_queue=ocr_queue,
                db_queue=db_queue,
                stop_event=stop_event,
                settings=settings.ocr,
            )
            logger.info("Engine OCR: fast-alpr (ONNX)")
        else:
            # Engine padrão: EasyOCR + PlateDetector (dois estágios)
            use_gpu = settings.model.device == "cuda"
            plate_ocr = PlateOCR(
                languages=settings.ocr.languages,
                gpu=use_gpu,
                confidence_threshold=settings.ocr.confidence_threshold,
            )

            # Estágio 1: detector de placa dedicado
            plate_detector = None
            if settings.ocr.plate_detector_enabled:
                pd_weights = Path(settings.ocr.plate_detector_weights)
                if pd_weights.exists():
                    from src.ocr.plate_detector import PlateDetector
                    plate_detector = PlateDetector(
                        weights_path=str(pd_weights),
                        conf_threshold=settings.ocr.plate_detector_conf,
                    )
                else:
                    logger.warning(
                        "PlateDetector habilitado mas modelo não encontrado: %s — "
                        "OCR de dois estágios desabilitado; execute "
                        "python scripts/download_plate_model.py para baixar o modelo",
                        pd_weights,
                    )

            ocr_worker = OCRWorker(
                ocr_queue=ocr_queue,
                db_queue=db_queue,
                plate_ocr=plate_ocr,
                plate_detector=plate_detector,
            )
            logger.info("Engine OCR: easyocr (dois estágios)")

        ocr_worker.start()

    # ── Estado de runtime ─────────────────────────────────────────────────
    writer: cv2.VideoWriter | None = None
    total_frames: int = 0
    session_start: float = time.perf_counter()
    warmed_up: bool = False

    # ── Iniciar thread de captura ─────────────────────────────────────────
    cap.start()

    try:
        while not stop_event.is_set():
            # Consumir fila com timeout — não bloqueia se a fila estiver vazia
            try:
                frame = frame_queue.get(timeout=0.1)
            except queue.Empty:
                # EOF: thread de captura encerrou e não há mais frames pendentes
                if not cap.is_alive() and frame_queue.empty():
                    logger.info("Fim do stream detectado — encerrando pipeline")
                    stop_event.set()
                continue

            # Warmup único com o shape real do vídeo antes do loop de inferência
            if not warmed_up:
                detector.warmup(frame.shape)
                warmed_up = True

            # Instanciar VideoWriter dinamicamente: herda resolução e FPS reais
            if writer is None:
                writer = _build_writer(Path(settings.video.output), cap.fps, frame)
                logger.info("VideoWriter inicializado — saída: %s", settings.video.output)

            total_frames += 1

            # ── Inferência + Rastreamento + Contagem ──────────────────────
            detections = detector.detect(frame)
            tracks = tracker.update(detections, frame)
            crossed_ids = counter.update(tracks, frame)

            for tid in crossed_ids:
                cls = counter.get_vehicle_class(tid)
                logger.info(
                    "Cruzamento: track_id=%d classe=%s total=%d",
                    tid, cls, counter.count,
                )

                event_meta = {
                    "track_id": tid,
                    "vehicle_class": cls,
                    "frame_number": total_frames,
                    "timestamp": datetime.now(timezone.utc),
                    "session_id": session_id,
                }

                crop = counter.get_best_crop(tid)
                # Fallback: usa o crop do frame atual quando best_crop ainda não
                # foi capturado (ex.: primeiro frame do veículo é o cruzamento).
                if crop is None or crop.size == 0:
                    track_at_crossing = next(
                        (t for t in tracks if t.track_id == tid), None
                    )
                    if track_at_crossing is not None:
                        x1_, y1_, x2_, y2_ = track_at_crossing.bbox_xyxy
                        x1_ = max(0, int(x1_))
                        y1_ = max(0, int(y1_))
                        x2_ = min(frame.shape[1], int(x2_))
                        y2_ = min(frame.shape[0], int(y2_))
                        if x2_ > x1_ and y2_ > y1_:
                            crop = frame[y1_:y2_, x1_:x2_]

                frame_area = cap.frame_width * cap.frame_height
                crop_qualifies = (
                    crop is not None
                    and crop.size > 0
                    and (crop.shape[0] * crop.shape[1])
                    >= settings.ocr.min_bbox_area_ratio * frame_area
                )

                if settings.ocr.enabled and ocr_worker is not None and crop_qualifies:
                    try:
                        ocr_queue.put_nowait((tid, crop, event_meta))
                    except queue.Full:
                        logger.warning(
                            "ocr_queue cheia — crop de track_id=%d descartado; "
                            "evento salvo sem placa",
                            tid,
                        )
                        event_meta["plate_text"] = None
                        event_meta["plate_confidence"] = None
                        try:
                            db_queue.put_nowait(event_meta)
                        except queue.Full:
                            logger.warning(
                                "db_queue cheia — evento track_id=%d descartado", tid
                            )
                else:
                    event_meta["plate_text"] = None
                    event_meta["plate_confidence"] = None
                    try:
                        db_queue.put_nowait(event_meta)
                    except queue.Full:
                        logger.warning(
                            "db_queue cheia — evento track_id=%d descartado", tid
                        )

            # ── Renderização + Gravação ───────────────────────────────────
            fps_meter.tick()
            # Mapeia track_id → classe refinada para o renderer colorir bboxes
            # com a paleta de classes de negócio (não as COCO brutas do detector).
            class_render_map = {
                t.track_id: counter.get_vehicle_class(t.track_id) for t in tracks
            }
            annotated = renderer.draw(
                frame, tracks, counter.count, fps_meter.fps, class_render_map
            )
            writer.write(annotated)

    except Exception:
        logger.error("Erro crítico no loop principal", exc_info=True)
        stop_event.set()

    finally:
        session_end = time.perf_counter()

        # Ordem inegociável: stop_event → captura → OCR → DB → VideoWriter
        stop_event.set()
        cap.stop()
        cap.join(timeout=5.0)

        if ocr_worker is not None:
            ocr_worker.stop_and_join(timeout=10.0)

        db_writer.flush_and_close(timeout=5.0)

        if writer is not None:
            writer.release()
            logger.info("VideoWriter liberado")

        _print_summary(
            duration=session_end - session_start,
            total_frames=total_frames,
            avg_fps=fps_meter.fps,
            total_count=counter.count,
            class_counts=counter.get_class_counts(),
            output_path=settings.video.output,
        )


if __name__ == "__main__":
    main()
