"""
Teste de integração Fase 1: VideoCapture → YoloDetector (mock) → ByteTrackWrapper.

Simula o fluxo do main.py sem modelo YOLO real nem arquivo de vídeo:
  - cv2.VideoCapture mockado → frames np.zeros
  - YoloDetector.detect mockado → detecções sintéticas com deslocamento crescente
  - ByteTrackWrapper real (supervision ByteTrack) → List[Track]

Assert crítico: o track_id criado no frame 1 deve ser mantido até o frame 5,
provando que ByteTrack mantém identidade de um veículo em movimento contínuo.
"""
from __future__ import annotations

import itertools
import queue
import time
from unittest.mock import MagicMock, patch

import cv2
import numpy as np

from src.capture.video_capture import VideoCapture
from src.config import ModelSettings, TrackingSettings, VideoSettings
from src.detection.yolo_detector import YoloDetector
from src.domain import Detection
from src.tracking.bytetrack_wrapper import ByteTrackWrapper

# ── Constantes do cenário sintético ─────────────────────────────────────────
_N_FRAMES = 5
_FRAME_W, _FRAME_H = 640, 480
# Deslocamento de 5 px/frame mantém IoU ≈ 0.91 entre frames consecutivos,
# acima do minimum_matching_threshold=0.8 do ByteTrack.
_BBOX_INITIAL: list[float] = [100.0, 150.0, 200.0, 250.0]
_BBOX_STEP_PX: float = 5.0


# ── Helpers de fixture ───────────────────────────────────────────────────────

def _mock_cv2_cap() -> MagicMock:
    """Cria mock de cv2.VideoCapture que devolve frames de zeros."""
    cap = MagicMock()
    cap.isOpened.return_value = True
    cap.read.return_value = (True, np.zeros((_FRAME_H, _FRAME_W, 3), dtype=np.uint8))
    cap.get.side_effect = lambda prop: {
        cv2.CAP_PROP_FPS: 30.0,
        cv2.CAP_PROP_FRAME_WIDTH: float(_FRAME_W),
        cv2.CAP_PROP_FRAME_HEIGHT: float(_FRAME_H),
    }.get(prop, 0.0)
    return cap


def _make_detect_side_effect() -> object:
    """Retorna side_effect que produz Detection com deslocamento X crescente por frame."""
    counter = itertools.count()

    def _detect(frame: np.ndarray) -> list[Detection]:
        n = next(counter)
        x1, y1, x2, y2 = _BBOX_INITIAL
        x_off = n * _BBOX_STEP_PX
        return [Detection(
            bbox_xyxy=np.array([x1 + x_off, y1, x2 + x_off, y2], dtype=np.float32),
            confidence=0.85,
            class_id=2,
            class_name="car",
        )]

    return _detect


def _model_settings() -> ModelSettings:
    return ModelSettings(
        weights="dummy.pt",
        confidence_threshold=0.45,
        iou_threshold=0.5,
        device="cpu",
        fp16=False,
    )


def _video_settings() -> VideoSettings:
    return VideoSettings(source="dummy.mp4", output="out.mp4", resize_width=640)


# ── Testes ───────────────────────────────────────────────────────────────────

def test_frames_arrive_in_queue_and_detector_is_called_per_frame() -> None:
    """VideoCapture thread alimenta a fila e o detector é chamado para cada frame."""
    with patch("cv2.VideoCapture", return_value=_mock_cv2_cap()), \
         patch("src.detection.yolo_detector.YOLO") as mock_yolo_cls:

        mock_yolo_cls.return_value = MagicMock()
        detector = YoloDetector(_model_settings())
        frame_queue: queue.Queue[np.ndarray] = queue.Queue(maxsize=3)
        cap = VideoCapture("dummy.mp4", frame_queue, _video_settings())
        cap.start()

        detections_per_frame: list[list[Detection]] = []
        try:
            with patch.object(YoloDetector, "detect", side_effect=_make_detect_side_effect()):
                for _ in range(_N_FRAMES):
                    frame = frame_queue.get(timeout=2.0)
                    detections_per_frame.append(detector.detect(frame))
        finally:
            cap.stop()

    assert len(detections_per_frame) == _N_FRAMES, (
        f"Esperado {_N_FRAMES} conjuntos de detecções, obtido {len(detections_per_frame)}"
    )
    assert all(len(d) == 1 for d in detections_per_frame), (
        "Cada frame deve ter exatamente 1 detecção sintética"
    )
    assert all(isinstance(d[0], Detection) for d in detections_per_frame), (
        "Os objetos retornados devem ser instâncias de Detection"
    )


def test_track_id_remains_consistent_across_5_consecutive_frames() -> None:
    """track_id criado no frame 1 deve ser o mesmo track_id no frame 5.

    Um veículo com deslocamento de 5 px/frame (IoU ≈ 0.91 entre frames
    consecutivos) deve receber um único track_id estável em todos os 5 frames.
    O ByteTrackWrapper usa o ByteTrack real — sem mock da lógica de tracking.
    """
    tracker = ByteTrackWrapper(TrackingSettings(track_buffer=30, min_box_area=100))

    with patch("cv2.VideoCapture", return_value=_mock_cv2_cap()), \
         patch("src.detection.yolo_detector.YOLO") as mock_yolo_cls:

        mock_yolo_cls.return_value = MagicMock()
        detector = YoloDetector(_model_settings())
        frame_queue: queue.Queue[np.ndarray] = queue.Queue(maxsize=3)
        cap = VideoCapture("dummy.mp4", frame_queue, _video_settings())
        cap.start()

        track_ids_per_frame: list[list[int]] = []
        t_start = time.perf_counter()

        try:
            with patch.object(YoloDetector, "detect", side_effect=_make_detect_side_effect()):
                for _ in range(_N_FRAMES):
                    frame = frame_queue.get(timeout=2.0)
                    detections = detector.detect(frame)
                    tracks = tracker.update(detections, frame)
                    track_ids_per_frame.append([t.track_id for t in tracks])
        finally:
            cap.stop()  # sempre chamado, mesmo em caso de falha

    elapsed = time.perf_counter() - t_start
    fps = _N_FRAMES / elapsed
    print(f"\n[1.7] FPS médio ({_N_FRAMES} frames sintéticos): {fps:.1f} fps")

    # ── Assertion 1: todos os frames geraram tracks ──────────────────────
    assert len(track_ids_per_frame) == _N_FRAMES, (
        f"Deve ter processado {_N_FRAMES} frames, processou {len(track_ids_per_frame)}"
    )
    assert all(len(ids) >= 1 for ids in track_ids_per_frame), (
        f"Cada frame deve ter ao menos 1 track ativo. "
        f"Tracks por frame: {track_ids_per_frame}"
    )

    # ── Assertion crítica: track_id estável em todos os frames ───────────
    first_track_id = track_ids_per_frame[0][0]
    all_ids = [ids[0] for ids in track_ids_per_frame]
    assert all(tid == first_track_id for tid in all_ids), (
        f"track_id deve ser consistente do frame 1 ao frame {_N_FRAMES}. "
        f"Esperado: {first_track_id} em todos. "
        f"Obtido por frame: {all_ids}"
    )
