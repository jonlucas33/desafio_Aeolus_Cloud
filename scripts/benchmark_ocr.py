"""
Benchmark comparativo de engines OCR: EasyOCR vs fast-alpr.

Roda o pipeline completo (detector + tracker + counter + OCR) no vídeo inteiro
para cada engine, mede placas lidas, confiança e FPS, e salva resultado em JSON.

Uso:
    python scripts/benchmark_ocr.py --config config/settings.yaml
"""
from __future__ import annotations

import argparse
import json
import logging
import queue
import sys
import threading
import time
from pathlib import Path

import cv2

# Garante que src.* é importável ao rodar o script diretamente
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.config import Settings, load_settings
from src.counting.crossing_logic import CrossingCounter
from src.database.db_writer import DbWriter
from src.database.models import create_sqlite_engine, init_db
from src.detection.yolo_detector import YoloDetector
from src.tracking.bytetrack_wrapper import ByteTrackWrapper

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    stream=sys.stdout,
)
logger = logging.getLogger(__name__)

_OCR_ENGINES = ["easyocr", "fast_alpr"]


# ── Helpers ───────────────────────────────────────────────────────────────────

def _resize_frame(frame: "cv2.Mat", resize_width: int) -> "cv2.Mat":
    h, w = frame.shape[:2]
    if w <= resize_width:
        return frame
    scale = resize_width / w
    return cv2.resize(frame, (resize_width, int(h * scale)), interpolation=cv2.INTER_LINEAR)


def _build_ocr_worker(
    engine: str,
    settings: Settings,
    ocr_queue: queue.Queue,
    db_queue: queue.Queue,
    stop_event: threading.Event,
):
    """Instancia o OCR worker correto com base no engine solicitado."""
    if engine == "fast_alpr":
        from src.ocr.fast_alpr_worker import FastAlprWorker
        worker = FastAlprWorker(
            ocr_queue=ocr_queue,
            db_queue=db_queue,
            stop_event=stop_event,
            settings=settings.ocr,
        )
        logger.info("Engine: fast-alpr (ONNX)")
        return worker

    # EasyOCR (padrão)
    from src.ocr.plate_ocr import OCRWorker, PlateOCR
    use_gpu = settings.model.device == "cuda"
    plate_ocr = PlateOCR(
        languages=settings.ocr.languages,
        gpu=use_gpu,
        confidence_threshold=settings.ocr.confidence_threshold,
    )
    plate_detector = None
    if settings.ocr.plate_detector_enabled:
        pd_weights = Path(settings.ocr.plate_detector_weights)
        if pd_weights.exists():
            from src.ocr.plate_detector import PlateDetector
            plate_detector = PlateDetector(
                weights_path=str(pd_weights),
                conf_threshold=settings.ocr.plate_detector_conf,
            )
    worker = OCRWorker(
        ocr_queue=ocr_queue,
        db_queue=db_queue,
        plate_ocr=plate_ocr,
        plate_detector=plate_detector,
    )
    logger.info("Engine: easyocr (dois estágios)")
    return worker


# ── Núcleo do benchmark ───────────────────────────────────────────────────────

def run_single_ocr_benchmark(
    engine: str,
    settings: Settings,
    video_path: Path,
) -> dict:
    """Executa o pipeline completo com um engine de OCR e coleta métricas.

    Roda o vídeo inteiro. OCR é assíncrono — FPS do pipeline principal não
    deve ser afetado pela lentidão do OCR (garantido pela arquitetura de fila).

    Args:
        engine: "easyocr" ou "fast_alpr".
        settings: Configurações globais.
        video_path: Caminho para o vídeo de entrada.

    Returns:
        Dicionário com métricas da sessão.
    """
    logger.info("─── Iniciando benchmark OCR: %s ───", engine)

    # Filas isoladas por execução
    frame_queue: queue.Queue = queue.Queue(maxsize=3)
    ocr_queue: queue.Queue = queue.Queue(maxsize=10)
    db_queue: queue.Queue = queue.Queue(maxsize=500)  # maior para coleta de eventos
    stop_event = threading.Event()

    # Banco de dados em memória — apenas para contabilizar eventos
    import tempfile, uuid
    tmp_db = Path(tempfile.gettempdir()) / f"benchmark_ocr_{uuid.uuid4().hex}.db"
    engine_obj = create_sqlite_engine(tmp_db)
    init_db(engine_obj)
    db_writer = DbWriter(db_queue=db_queue, engine=engine_obj)
    db_writer.start()

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

    ocr_worker = _build_ocr_worker(engine, settings, ocr_queue, db_queue, stop_event)
    ocr_worker.start()

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Não foi possível abrir o vídeo: {video_path}")

    # Warmup
    ok, peek = cap.read()
    if ok:
        peek = _resize_frame(peek, settings.video.resize_width)
        detector.warmup(peek.shape)
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    # Coleta de eventos de placa diretamente da db_queue interceptada
    plates_collected: list[dict] = []
    frame_wall_times: list[float] = []
    frame_count = 0
    import datetime, uuid as _uuid
    session_id = str(_uuid.uuid4())

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame = _resize_frame(frame, settings.video.resize_width)

        t0 = time.perf_counter()
        detections = detector.detect(frame)
        tracks = tracker.update(detections, frame)
        crossed_ids = counter.update(tracks, frame)

        for tid in crossed_ids:
            cls = counter.get_vehicle_class(tid)
            crop = counter.get_best_crop(tid)

            frame_area = cap.get(cv2.CAP_PROP_FRAME_WIDTH) * cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
            crop_ok = (
                crop is not None
                and crop.size > 0
                and (crop.shape[0] * crop.shape[1]) >= settings.ocr.min_bbox_area_ratio * frame_area
            )

            event_meta = {
                "track_id": tid,
                "vehicle_class": cls,
                "frame_number": frame_count,
                "timestamp": datetime.datetime.now(datetime.timezone.utc),
                "session_id": session_id,
            }

            if crop_ok:
                try:
                    ocr_queue.put_nowait((tid, crop, event_meta))
                except queue.Full:
                    event_meta["plate_text"] = None
                    event_meta["plate_confidence"] = None
                    try:
                        db_queue.put_nowait(event_meta)
                    except queue.Full:
                        pass
            else:
                event_meta["plate_text"] = None
                event_meta["plate_confidence"] = None
                try:
                    db_queue.put_nowait(event_meta)
                except queue.Full:
                    pass

        frame_wall_times.append(time.perf_counter() - t0)
        frame_count += 1

        if frame_count % 500 == 0:
            elapsed = sum(frame_wall_times)
            logger.info(
                "  %d frames | FPS: %.1f | Veículos: %d",
                frame_count, frame_count / elapsed if elapsed else 0, counter.count,
            )

    cap.release()
    stop_event.set()

    # Aguarda worker OCR processar fila pendente
    if hasattr(ocr_worker, "stop_and_join"):
        ocr_worker.stop_and_join(timeout=30.0)
    else:
        ocr_worker.join(timeout=30.0)

    db_writer.flush_and_close(timeout=10.0)

    # Lê eventos do banco em memória para extrair métricas de placa
    import sqlalchemy
    with engine_obj.connect() as conn:
        rows = conn.execute(
            sqlalchemy.text(
                "SELECT track_id, plate_text, plate_confidence, frame_number "
                "FROM vehicle_events WHERE plate_text IS NOT NULL AND session_id = :sid"
            ),
            {"sid": session_id},
        ).fetchall()

    valid_plates = [
        {"track_id": r[0], "plate_text": r[1], "confidence": r[2], "frame_number": r[3]}
        for r in rows
    ]

    fps = frame_count / sum(frame_wall_times) if frame_wall_times else 0.0
    confidences = [p["confidence"] for p in valid_plates if p["confidence"] is not None]

    # Limpeza do banco temporário
    try:
        tmp_db.unlink(missing_ok=True)
    except Exception:
        pass

    return {
        "engine": engine,
        "frames_processed": frame_count,
        "total_vehicles": counter.count,
        "fps_avg": round(fps, 2),
        "plates_valid": len(valid_plates),
        "read_rate_pct": round(len(valid_plates) / max(counter.count, 1) * 100, 1),
        "confidence_avg": round(sum(confidences) / len(confidences), 3) if confidences else None,
        "confidence_min": round(min(confidences), 3) if confidences else None,
        "confidence_max": round(max(confidences), 3) if confidences else None,
        "plates_detail": valid_plates,
    }


# ── Tabela e saída ────────────────────────────────────────────────────────────

def _print_table(results: list[dict], video_path: Path) -> None:
    if len(results) < 2:
        return

    a, b = results[0], results[1]

    def _fmt(v):
        if v is None:
            return "N/A"
        return str(v)

    def _delta(va, vb):
        if va is None or vb is None:
            return "N/A"
        diff = vb - va
        sign = "+" if diff >= 0 else ""
        if isinstance(va, float):
            return f"{sign}{diff:.1f}"
        return f"{sign}{int(diff)}"

    col = 14
    sep = "─" * (28 + col * 3)

    rows = [
        ("Placas lidas (válidas)", a["plates_valid"],    b["plates_valid"]),
        ("Taxa de leitura (%)",   a["read_rate_pct"],   b["read_rate_pct"]),
        ("Confiança média",       a["confidence_avg"],  b["confidence_avg"]),
        ("Confiança mínima",      a["confidence_min"],  b["confidence_min"]),
        ("FPS pipeline",          a["fps_avg"],          b["fps_avg"]),
    ]

    lines = [
        "",
        "=== BENCHMARK DE OCR ===",
        f"Vídeo: {video_path} | Frames: {a['frames_processed']} | Veículos: {a['total_vehicles']}",
        "",
        f"{'Métrica':<28} {a['engine']:<{col}} {b['engine']:<{col}} Delta",
        sep,
    ]
    for label, va, vb in rows:
        lines.append(
            f"{label:<28} {_fmt(va):<{col}} {_fmt(vb):<{col}} {_delta(va, vb)}"
        )
    lines.append(sep)
    lines.append("")
    lines.append("Placas detectadas:")
    for res in results:
        lines.append(f"  {res['engine']}:")
        if res["plates_detail"]:
            for p in res["plates_detail"]:
                lines.append(
                    f"    track_id={p['track_id']} | {p['plate_text']} "
                    f"| conf={p['confidence']:.2f} | frame={p['frame_number']}"
                )
        else:
            lines.append("    (nenhuma)")
    lines.append(sep)
    lines.append("")

    print("\n".join(lines))


# ── Ponto de entrada ──────────────────────────────────────────────────────────

def main() -> None:
    """Executa benchmark de EasyOCR vs fast-alpr e salva resultado em JSON."""
    parser = argparse.ArgumentParser(
        description="Benchmark comparativo: EasyOCR vs fast-alpr"
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("config/settings.yaml"),
        help="Caminho para config/settings.yaml",
    )
    args = parser.parse_args()

    settings = load_settings(args.config)
    video_path = Path(settings.video.source)

    if not video_path.exists():
        logger.error("Vídeo não encontrado: %s", video_path)
        sys.exit(1)

    results: list[dict] = []
    for engine in _OCR_ENGINES:
        result = run_single_ocr_benchmark(engine, settings, video_path)
        results.append(result)
        logger.info(
            "Concluído %s → placas=%d | FPS=%.1f | conf_avg=%s",
            engine,
            result["plates_valid"],
            result["fps_avg"],
            result["confidence_avg"],
        )

    _print_table(results, video_path)

    output_path = Path("data/outputs/benchmark_ocr.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as fh:
        json.dump(
            {
                "config": str(args.config),
                "video": str(video_path),
                "results": results,
            },
            fh,
            indent=2,
            ensure_ascii=False,
            default=str,  # serializa datetime e outros tipos não-serializáveis
        )
    logger.info("Resultado salvo em %s", output_path)


if __name__ == "__main__":
    main()
