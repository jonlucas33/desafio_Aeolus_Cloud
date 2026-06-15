"""
Testes unitários para PlateDetector.

Usa mocks limpos para a engine YOLO — sem inferência real.
Cobre: sem detecção, crop correto, margem de 4px, upscale de input pequeno,
seleção por maior confiança e rejeição abaixo do threshold.
"""
from __future__ import annotations

import numpy as np
import pytest
from unittest.mock import MagicMock


# ── Helpers de mock ───────────────────────────────────────────────────────────

def _make_boxes_mock(
    xyxy: list[list[float]],
    confs: list[float],
) -> MagicMock:
    """Cria mock de result.boxes no formato Ultralytics."""
    boxes = MagicMock()
    xyxy_tensor = MagicMock()
    xyxy_tensor.cpu.return_value.numpy.return_value = (
        np.array(xyxy, dtype=np.float32) if xyxy else np.zeros((0, 4), dtype=np.float32)
    )
    conf_tensor = MagicMock()
    conf_tensor.cpu.return_value.numpy.return_value = (
        np.array(confs, dtype=np.float32) if confs else np.zeros(0, dtype=np.float32)
    )
    boxes.xyxy = xyxy_tensor
    boxes.conf = conf_tensor
    return boxes


def _make_result(
    xyxy: list[list[float]] | None = None,
    confs: list[float] | None = None,
) -> MagicMock:
    result = MagicMock()
    result.boxes = _make_boxes_mock(xyxy or [], confs or [])
    return result


# ── Testes ───────────────────────────────────────────────────────────────────

def test_detect_returns_none_when_no_plates_detected(mocker) -> None:
    """detect() deve retornar None quando o YOLO não encontra nenhuma placa."""
    mock_model = MagicMock(return_value=[_make_result()])
    mocker.patch("src.ocr.plate_detector.YOLO", return_value=mock_model)

    from src.ocr.plate_detector import PlateDetector
    detector = PlateDetector(weights_path="models/fake.pt", conf_threshold=0.5)

    crop = np.zeros((100, 200, 3), dtype=np.uint8)
    result = detector.detect(crop)

    assert result is None


def test_detect_returns_plate_crop_from_vehicle_crop(mocker) -> None:
    """detect() deve retornar array 3D recortado da região da placa no crop veicular."""
    mock_model = MagicMock(return_value=[
        _make_result(xyxy=[[20.0, 10.0, 80.0, 40.0]], confs=[0.9])
    ])
    mocker.patch("src.ocr.plate_detector.YOLO", return_value=mock_model)

    from src.ocr.plate_detector import PlateDetector
    detector = PlateDetector(weights_path="models/fake.pt", conf_threshold=0.5)

    crop = np.ones((100, 200, 3), dtype=np.uint8) * 128
    result = detector.detect(crop)

    assert result is not None
    assert result.ndim == 3, "Resultado deve ser imagem BGR (3 dimensões)"


def test_detect_applies_4px_safety_margin_to_plate_crop(mocker) -> None:
    """A margem de 4px deve ser aplicada ao redor do bbox detectado.

    Bbox detectada em [10, 5, 90, 25] sobre crop 200×100 (sem upscale).
    Região esperada com margem: x1=6, y1=1, x2=94, y2=29.
    Dimensões esperadas: altura=28, largura=88.
    """
    mock_model = MagicMock(return_value=[
        _make_result(xyxy=[[10.0, 5.0, 90.0, 25.0]], confs=[0.85])
    ])
    mocker.patch("src.ocr.plate_detector.YOLO", return_value=mock_model)

    from src.ocr.plate_detector import PlateDetector
    detector = PlateDetector(weights_path="models/fake.pt", conf_threshold=0.5)

    crop = np.zeros((100, 200, 3), dtype=np.uint8)
    result = detector.detect(crop)

    assert result is not None
    assert result.shape[0] == 28, f"Altura esperada 28 px, obteve {result.shape[0]}"
    assert result.shape[1] == 88, f"Largura esperada 88 px, obteve {result.shape[1]}"


def test_detect_upscales_small_input_before_inference(mocker) -> None:
    """crop com height < 64 px deve ser upscalado antes do YOLO.

    Verifica que o modelo recebe frame maior que o input original.
    O crop retornado é extraído do input ORIGINAL (não do upscalado).
    """
    received_frames: list[np.ndarray] = []

    def _capture_call(frame: np.ndarray, **kwargs) -> list:
        received_frames.append(frame.copy())
        h, w = frame.shape[:2]
        return [_make_result(
            xyxy=[[w * 0.25, h * 0.25, w * 0.75, h * 0.75]],
            confs=[0.9],
        )]

    mock_model = MagicMock(side_effect=_capture_call)
    mocker.patch("src.ocr.plate_detector.YOLO", return_value=mock_model)

    from src.ocr.plate_detector import PlateDetector
    detector = PlateDetector(weights_path="models/fake.pt", conf_threshold=0.5)

    small_crop = np.ones((30, 80, 3), dtype=np.uint8) * 77  # height=30 < 64
    result = detector.detect(small_crop)

    assert len(received_frames) == 1, "YOLO deve ter sido chamado exatamente uma vez"
    assert received_frames[0].shape[0] > 30, (
        f"YOLO deve receber frame upscalado (>30 px), "
        f"recebeu height={received_frames[0].shape[0]}"
    )
    assert result is not None, "Deve retornar crop mesmo com input pequeno"


def test_detect_picks_highest_confidence_among_multiple_detections(mocker) -> None:
    """Quando múltiplas placas são detectadas, a de maior confiança deve ser usada.

    Detecções: bbox A (conf=0.60), bbox B (conf=0.95, ← esperada), bbox C (conf=0.70).
    A região de B é marcada com pixels brancos para distinguir os casos.
    """
    mock_model = MagicMock(return_value=[
        _make_result(
            xyxy=[
                [5.0, 5.0, 30.0, 20.0],    # conf=0.60
                [40.0, 5.0, 90.0, 30.0],   # conf=0.95 ← deve ser selecionada
                [95.0, 5.0, 150.0, 20.0],  # conf=0.70
            ],
            confs=[0.60, 0.95, 0.70],
        )
    ])
    mocker.patch("src.ocr.plate_detector.YOLO", return_value=mock_model)

    from src.ocr.plate_detector import PlateDetector
    detector = PlateDetector(weights_path="models/fake.pt", conf_threshold=0.5)

    crop = np.zeros((100, 200, 3), dtype=np.uint8)
    # Marca a região da detecção de maior confiança (bbox B)
    crop[5:30, 40:90] = 255

    result = detector.detect(crop)

    assert result is not None
    assert result.max() == 255, (
        "O crop deve vir da detecção de maior confiança — "
        "bbox B tem pixels brancos, as outras não"
    )


def test_detect_returns_none_below_confidence_threshold(mocker) -> None:
    """Detecção com confiança abaixo do threshold deve ser ignorada e retornar None."""
    mock_model = MagicMock(return_value=[
        _make_result(xyxy=[[10.0, 5.0, 80.0, 30.0]], confs=[0.3])
    ])
    mocker.patch("src.ocr.plate_detector.YOLO", return_value=mock_model)

    from src.ocr.plate_detector import PlateDetector
    detector = PlateDetector(weights_path="models/fake.pt", conf_threshold=0.5)

    crop = np.zeros((100, 200, 3), dtype=np.uint8)
    result = detector.detect(crop)

    assert result is None, (
        f"Confiança 0.3 < threshold 0.5 deve retornar None, obteve {result}"
    )
