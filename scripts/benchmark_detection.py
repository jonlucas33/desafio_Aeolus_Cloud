"""
Benchmark comparativo de modelos de detecção: YOLOv8n vs YOLOv8s.

Roda o loop de inferência + tracking + contagem nos primeiros N frames do vídeo
de entrada para cada modelo, isola a métrica de detecção sem OCR nem banco de dados
e imprime tabela comparativa.

Uso:
    python scripts/benchmark_detection.py --config config/settings.yaml --frames 300
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import cv2

# Garante que src.* é importável ao rodar o script diretamente
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.config import Settings, load_settings
from src.counting.crossing_logic import CrossingCounter
from src.detection.yolo_detector import YoloDetector
from src.tracking.bytetrack_wrapper import ByteTrackWrapper

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    stream=sys.stdout,
)
logger = logging.getLogger(__name__)

_BENCHMARK_MODELS = ["models/yolov8n.pt", "models/yolov8s.pt"]
_CLASSES_ORDER = ["sedan_hatch", "suv_pickup", "truck_bus", "motorcycle"]


# ── Helpers ───────────────────────────────────────────────────────────────────

def _resize_frame(frame: "cv2.Mat", resize_width: int) -> "cv2.Mat":
    """Redimensiona o frame para a largura alvo mantendo proporção."""
    h, w = frame.shape[:2]
    if w <= resize_width:
        return frame
    scale = resize_width / w
    return cv2.resize(frame, (resize_width, int(h * scale)), interpolation=cv2.INTER_LINEAR)


def _class_distribution(
    counter: CrossingCounter,
    unique_track_ids: set[int],
) -> dict[str, int]:
    """Conta quantos tracks únicos têm cada classe como votação majoritária."""
    dist: dict[str, int] = {cls: 0 for cls in _CLASSES_ORDER}
    for tid in unique_track_ids:
        cls = counter.get_vehicle_class(tid)
        if cls in dist:
            dist[cls] += 1
    return dist


def _fmt_delta(va: float | int, vb: float | int) -> str:
    diff = vb - va
    sign = "+" if diff >= 0 else ""
    if isinstance(va, float):
        return f"{sign}{diff:.1f}"
    return f"{sign}{int(diff)}"


def _print_table(results: list[dict], video_path: Path, max_frames: int) -> None:
    """Imprime tabela comparativa alinhada no terminal."""
    if len(results) < 2:
        logger.warning("Tabela comparativa requer ao menos dois resultados.")
        return

    a, b = results[0], results[1]
    a_name = Path(a["model"]).stem   # "yolov8n"
    b_name = Path(b["model"]).stem   # "yolov8s"

    col_w = 13
    sep = "─" * (28 + col_w * 3)

    rows: list[tuple[str, float | int, float | int]] = [
        ("FPS médio",             a["fps_avg"],              b["fps_avg"]),
        ("Inferência média (ms)", a["inference_avg_ms"],     b["inference_avg_ms"]),
        ("Track IDs únicos",      a["unique_track_ids"],     b["unique_track_ids"]),
    ]
    for cls in _CLASSES_ORDER:
        rows.append((
            f"{cls} detectados",
            a["class_distribution"][cls],
            b["class_distribution"][cls],
        ))

    lines = [
        "",
        "=== BENCHMARK DE DETECÇÃO ===",
        f"Frames analisados: {max_frames} | Vídeo: {video_path}",
        "",
        f"{'Métrica':<28} {a_name:<{col_w}} {b_name:<{col_w}} Delta",
        sep,
    ]
    for label, va, vb in rows:
        if isinstance(va, float):
            lines.append(
                f"{label:<28} {va:<{col_w}.1f} {vb:<{col_w}.1f} {_fmt_delta(va, vb)}"
            )
        else:
            lines.append(
                f"{label:<28} {va:<{col_w}} {vb:<{col_w}} {_fmt_delta(va, vb)}"
            )
    lines.append(sep)
    lines.append("")

    print("\n".join(lines))


# ── Núcleo do benchmark ───────────────────────────────────────────────────────

def run_single_benchmark(
    model_weights: str,
    settings: Settings,
    video_path: Path,
    max_frames: int,
) -> dict:
    """Executa o loop de inferência para um modelo e retorna as métricas coletadas.

    Instancia detector, tracker e counter do zero para garantir isolamento
    entre as duas execuções. OCR e banco de dados são deliberadamente omitidos
    para medir apenas o custo do pipeline de visão computacional.

    Args:
        model_weights: Caminho relativo para o arquivo .pt (ex: "models/yolov8n.pt").
        settings: Configurações globais carregadas do YAML.
        video_path: Caminho para o arquivo de vídeo.
        max_frames: Número máximo de frames a processar.

    Returns:
        Dicionário com métricas da execução.
    """
    logger.info("─── Iniciando: %s (%d frames) ───", model_weights, max_frames)

    # Override de pesos via Pydantic model_copy — sem mutação das settings globais
    model_settings = settings.model.model_copy(update={"weights": model_weights})

    detector = YoloDetector(settings=model_settings)
    tracker = ByteTrackWrapper(settings=settings.tracking)
    counter = CrossingCounter(
        line_points=settings.counting.line_points,
        direction=settings.counting.direction,
        min_displacement_px=settings.counting.min_displacement_px,
        class_vote_window=settings.counting.class_vote_window,
        suv_aspect_ratio_threshold=settings.counting.suv_aspect_ratio_threshold,
        truck_area_threshold=settings.counting.truck_area_threshold,
    )

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Não foi possível abrir o vídeo: {video_path}")

    # Lê primeiro frame apenas para obter o shape real e aquece o detector
    ok, peek = cap.read()
    if not ok:
        cap.release()
        raise RuntimeError(f"Vídeo vazio ou ilegível: {video_path}")
    peek = _resize_frame(peek, settings.video.resize_width)
    detector.warmup(peek.shape)

    # Reseta para o início — o primeiro frame fará parte da medição
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    inference_times_ms: list[float] = []
    frame_wall_times: list[float] = []
    unique_track_ids: set[int] = set()
    frame_count = 0

    while frame_count < max_frames:
        ok, frame = cap.read()
        if not ok:
            logger.info("  Fim do vídeo após %d frames", frame_count)
            break

        frame = _resize_frame(frame, settings.video.resize_width)

        t_frame = time.perf_counter()

        # Inferência — cronometrada separadamente para isolar o custo do modelo
        t0 = time.perf_counter()
        detections = detector.detect(frame)
        inference_times_ms.append((time.perf_counter() - t0) * 1000.0)

        tracks = tracker.update(detections, frame)
        counter.update(tracks, frame)

        for track in tracks:
            unique_track_ids.add(track.track_id)

        frame_wall_times.append(time.perf_counter() - t_frame)
        frame_count += 1

        if frame_count % 100 == 0:
            elapsed = sum(frame_wall_times)
            live_fps = frame_count / elapsed if elapsed > 0 else 0.0
            logger.info("  %d/%d frames | FPS parcial: %.1f", frame_count, max_frames, live_fps)

    cap.release()

    fps = frame_count / sum(frame_wall_times) if frame_wall_times else 0.0
    avg_ms = sum(inference_times_ms) / len(inference_times_ms) if inference_times_ms else 0.0

    return {
        "model": model_weights,
        "frames_processed": frame_count,
        "fps_avg": round(fps, 2),
        "inference_avg_ms": round(avg_ms, 2),
        "unique_track_ids": len(unique_track_ids),
        "class_distribution": _class_distribution(counter, unique_track_ids),
        "total_crossings": counter.count,
    }


# ── Ponto de entrada ──────────────────────────────────────────────────────────

def main() -> None:
    """Executa benchmark de YOLOv8n vs YOLOv8s e salva resultado em JSON."""
    parser = argparse.ArgumentParser(
        description="Benchmark comparativo: YOLOv8n vs YOLOv8s"
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("config/settings.yaml"),
        help="Caminho para config/settings.yaml",
    )
    parser.add_argument(
        "--frames",
        type=int,
        default=300,
        help="Número de frames a processar por modelo (padrão: 300)",
    )
    args = parser.parse_args()

    settings = load_settings(args.config)
    video_path = Path(settings.video.source)

    if not video_path.exists():
        logger.error("Vídeo não encontrado: %s", video_path)
        sys.exit(1)

    # Valida modelos antes de iniciar
    missing = [m for m in _BENCHMARK_MODELS if not Path(m).exists()]
    if missing:
        logger.error(
            "Modelo(s) não encontrado(s): %s\n"
            "  Execute: python scripts/download_models.py",
            ", ".join(missing),
        )
        sys.exit(1)

    results: list[dict] = []
    for model_path in _BENCHMARK_MODELS:
        result = run_single_benchmark(model_path, settings, video_path, args.frames)
        results.append(result)
        logger.info(
            "Concluído %s → FPS=%.1f | Infer=%.1f ms | Tracks=%d",
            model_path,
            result["fps_avg"],
            result["inference_avg_ms"],
            result["unique_track_ids"],
        )

    _print_table(results, video_path, args.frames)

    output_path = Path("data/outputs/benchmark_detection.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as fh:
        json.dump(
            {
                "config": str(args.config),
                "video": str(video_path),
                "frames": args.frames,
                "results": results,
            },
            fh,
            indent=2,
            ensure_ascii=False,
        )
    logger.info("Resultado salvo em %s", output_path)


if __name__ == "__main__":
    main()
