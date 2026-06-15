"""
Testes unitários para o wrapper YOLOv8.

ultralytics.YOLO é mockado — nenhum arquivo .pt é necessário.
"""
from __future__ import annotations

from unittest.mock import MagicMock

import numpy as np
import pytest

from src.config import ModelSettings


def _make_settings(
    device: str = "cpu",
    base_conf_threshold: float = 0.35,
    default_class_threshold: float = 0.45,
    motorcycle_threshold: float = 0.35,
) -> ModelSettings:
    return ModelSettings(
        weights="models/yolov8s.pt",
        confidence_threshold=0.45,
        iou_threshold=0.5,
        device=device,
        fp16=False,
        base_conf_threshold=base_conf_threshold,
        default_class_threshold=default_class_threshold,
        motorcycle_threshold=motorcycle_threshold,
    )


def _tensor_mock(data: list, dtype=np.float32) -> MagicMock:
    """Mock que imita tensor PyTorch com .cpu().numpy()."""
    m = MagicMock()
    m.cpu.return_value.numpy.return_value = np.array(data, dtype=dtype)
    return m


def _make_yolo_result(boxes: list, confs: list, clss: list) -> MagicMock:
    """Cria resultado mock de ultralytics.YOLO com boxes/confs/cls."""
    result = MagicMock()
    result.boxes.xyxy = _tensor_mock(boxes)
    result.boxes.conf = _tensor_mock(confs)
    result.boxes.cls = _tensor_mock(clss)
    return result


# ---------------------------------------------------------------------------

def test_detect_returns_detection_for_car_class(mocker) -> None:
    """detect() deve retornar Detection quando YOLO detectar car (class_id=2)."""
    mock_yolo_cls = mocker.patch("src.detection.yolo_detector.YOLO")
    mock_model = MagicMock()
    mock_yolo_cls.return_value = mock_model
    mock_model.return_value = [_make_yolo_result(
        boxes=[[10.0, 20.0, 110.0, 120.0]],
        confs=[0.9],
        clss=[2.0],
    )]

    from src.detection.yolo_detector import YoloDetector
    from src.domain import Detection

    detector = YoloDetector(_make_settings())
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    result = detector.detect(frame)

    assert len(result) == 1
    assert isinstance(result[0], Detection)
    assert result[0].class_id == 2
    assert result[0].class_name == "car"
    assert result[0].confidence == pytest.approx(0.9)
    assert result[0].bbox_xyxy.shape == (4,)


def test_detect_filters_out_non_vehicle_classes(mocker) -> None:
    """detect() deve descartar person(0), bicycle(1), chair(56), etc."""
    mock_yolo_cls = mocker.patch("src.detection.yolo_detector.YOLO")
    mock_model = MagicMock()
    mock_yolo_cls.return_value = mock_model
    mock_model.return_value = [_make_yolo_result(
        boxes=[[0, 0, 50, 50], [10, 20, 100, 200], [0, 0, 30, 30]],
        confs=[0.92, 0.85, 0.7],
        clss=[0.0, 2.0, 56.0],  # person, car, chair
    )]

    from src.detection.yolo_detector import YoloDetector

    detector = YoloDetector(_make_settings())
    detections = detector.detect(np.zeros((480, 640, 3), dtype=np.uint8))

    assert len(detections) == 1
    assert detections[0].class_id == 2


def test_detect_maps_all_four_vehicle_coco_classes(mocker) -> None:
    """car(2), motorcycle(3), bus(5), truck(7) devem todos ser mapeados."""
    mock_yolo_cls = mocker.patch("src.detection.yolo_detector.YOLO")
    mock_model = MagicMock()
    mock_yolo_cls.return_value = mock_model
    mock_model.return_value = [_make_yolo_result(
        boxes=[[0, 0, 10, 10]] * 4,
        confs=[0.8, 0.8, 0.8, 0.8],
        clss=[2.0, 3.0, 5.0, 7.0],  # car, motorcycle, bus, truck
    )]

    from src.detection.yolo_detector import YoloDetector

    detector = YoloDetector(_make_settings())
    detections = detector.detect(np.zeros((480, 640, 3), dtype=np.uint8))

    class_names = {d.class_name for d in detections}
    assert class_names == {"car", "motorcycle", "bus", "truck"}


def test_detect_returns_empty_list_when_no_vehicles(mocker) -> None:
    """detect() deve retornar lista vazia quando não há veículos."""
    mock_yolo_cls = mocker.patch("src.detection.yolo_detector.YOLO")
    mock_model = MagicMock()
    mock_yolo_cls.return_value = mock_model
    mock_model.return_value = [_make_yolo_result(
        boxes=[[0, 0, 50, 50]],
        confs=[0.9],
        clss=[0.0],  # apenas person
    )]

    from src.detection.yolo_detector import YoloDetector

    detector = YoloDetector(_make_settings())
    detections = detector.detect(np.zeros((480, 640, 3), dtype=np.uint8))

    assert detections == []


def test_asymmetric_conf_filter_keeps_motorcycle_removes_low_conf_car(mocker) -> None:
    """Filtro assimétrico por classe: carro a 0.40 removido, moto a 0.40 mantida.

    Configuração:
        default_class_threshold = 0.45  → carro(0.40) < 0.45 → DESCARTADO
        motorcycle_threshold    = 0.35  → moto (0.40) >= 0.35 → MANTIDA
    """
    mock_yolo_cls = mocker.patch("src.detection.yolo_detector.YOLO")
    mock_model = MagicMock()
    mock_yolo_cls.return_value = mock_model
    mock_model.return_value = [_make_yolo_result(
        boxes=[[0.0, 0.0, 100.0, 100.0], [0.0, 0.0, 80.0, 80.0]],
        confs=[0.40, 0.40],
        clss=[2.0, 3.0],  # car a 0.40, motorcycle a 0.40
    )]

    from src.detection.yolo_detector import YoloDetector

    detector = YoloDetector(_make_settings(
        base_conf_threshold=0.35,
        default_class_threshold=0.45,
        motorcycle_threshold=0.35,
    ))
    detections = detector.detect(np.zeros((480, 640, 3), dtype=np.uint8))

    assert len(detections) == 1, (
        f"Esperado 1 detecção (moto), obtidas {len(detections)}: "
        f"{[(d.class_name, d.confidence) for d in detections]}"
    )
    assert detections[0].class_id == 3
    assert detections[0].class_name == "motorcycle"
    assert detections[0].confidence == pytest.approx(0.40)


def test_warmup_calls_model_n_times(mocker) -> None:
    """warmup() deve chamar o modelo exatamente n_frames vezes."""
    mock_yolo_cls = mocker.patch("src.detection.yolo_detector.YOLO")
    mock_model = MagicMock()
    mock_yolo_cls.return_value = mock_model
    mock_model.return_value = []

    from src.detection.yolo_detector import YoloDetector

    detector = YoloDetector(_make_settings())
    detector.warmup(frame_shape=(720, 1280, 3), n_frames=3)

    assert mock_model.call_count == 3


def test_warmup_passes_frame_with_exact_shape(mocker) -> None:
    """warmup() deve passar frames com o shape exato fornecido — nunca shape arbitrário."""
    mock_yolo_cls = mocker.patch("src.detection.yolo_detector.YOLO")
    mock_model = MagicMock()
    mock_yolo_cls.return_value = mock_model
    mock_model.return_value = []

    from src.detection.yolo_detector import YoloDetector

    expected_shape = (540, 960, 3)
    detector = YoloDetector(_make_settings())
    detector.warmup(frame_shape=expected_shape, n_frames=2)

    for call in mock_model.call_args_list:
        frame_arg = call.args[0]
        assert frame_arg.shape == expected_shape, (
            f"warmup deve usar shape {expected_shape}, mas passou {frame_arg.shape}"
        )
