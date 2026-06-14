"""
Testes unitários para src/ocr/plate_ocr.py.

_validate_plate é testada como função pura — sem mocks.
PlateOCR.read() e OCRWorker são testados com easyocr.Reader mockado.
"""
from __future__ import annotations

import queue
import time
from unittest.mock import MagicMock

import numpy as np
import pytest


# ── _validate_plate (função pura, zero mocks) ────────────────────────────────

def test_validate_plate_accepts_mercosul_format() -> None:
    """Placa Mercosul ABC1D23 deve ser aceita e retornada uppercase."""
    from src.ocr.plate_ocr import _validate_plate
    assert _validate_plate("ABC1D23") == "ABC1D23"


def test_validate_plate_accepts_old_format() -> None:
    """Placa no formato antigo ABC1234 deve ser aceita."""
    from src.ocr.plate_ocr import _validate_plate
    assert _validate_plate("ABC1234") == "ABC1234"


def test_validate_plate_rejects_invalid_format() -> None:
    """Strings sem padrão de placa brasileira devem retornar None."""
    from src.ocr.plate_ocr import _validate_plate
    assert _validate_plate("INVALID") is None
    assert _validate_plate("1234567") is None
    assert _validate_plate("") is None
    assert _validate_plate("AB12345") is None


def test_validate_plate_normalizes_lowercase_to_uppercase() -> None:
    """Texto em minúsculas deve ser normalizado antes da validação."""
    from src.ocr.plate_ocr import _validate_plate
    assert _validate_plate("abc1d23") == "ABC1D23"


def test_validate_plate_strips_hyphens_and_spaces() -> None:
    """Hifens e espaços comuns em saídas de OCR devem ser removidos."""
    from src.ocr.plate_ocr import _validate_plate
    assert _validate_plate("ABC-1234") == "ABC1234"
    assert _validate_plate("ABC 1234") == "ABC1234"


# ── PlateOCR.read() (easyocr.Reader mockado) ─────────────────────────────────

def test_plate_ocr_returns_valid_mercosul_plate(mocker) -> None:
    """PlateOCR deve retornar placa e confiança quando EasyOCR encontra texto válido."""
    mock_reader = MagicMock()
    mock_reader.readtext.return_value = [
        ([[0, 0], [100, 0], [100, 30], [0, 30]], "ABC1D23", 0.95)
    ]
    mocker.patch("easyocr.Reader", return_value=mock_reader)

    from src.ocr.plate_ocr import PlateOCR
    crop = np.zeros((50, 200, 3), dtype=np.uint8)
    plate, confidence = PlateOCR().read(crop)

    assert plate == "ABC1D23"
    assert confidence == pytest.approx(0.95)


def test_plate_ocr_returns_none_when_confidence_below_threshold(mocker) -> None:
    """Confiança < 0.3 deve ser descartada e retornar (None, None)."""
    mock_reader = MagicMock()
    mock_reader.readtext.return_value = [
        ([[0, 0], [100, 0], [100, 30], [0, 30]], "ABC1234", 0.2)
    ]
    mocker.patch("easyocr.Reader", return_value=mock_reader)

    from src.ocr.plate_ocr import PlateOCR
    crop = np.zeros((50, 200, 3), dtype=np.uint8)
    plate, confidence = PlateOCR().read(crop)

    assert plate is None
    assert confidence is None


def test_plate_ocr_returns_none_for_invalid_plate_format(mocker) -> None:
    """Texto com alta confiança mas sem formato de placa deve ser descartado."""
    mock_reader = MagicMock()
    mock_reader.readtext.return_value = [
        ([[0, 0], [100, 0], [100, 30], [0, 30]], "NOTAPLATE", 0.99)
    ]
    mocker.patch("easyocr.Reader", return_value=mock_reader)

    from src.ocr.plate_ocr import PlateOCR
    crop = np.zeros((50, 200, 3), dtype=np.uint8)
    plate, confidence = PlateOCR().read(crop)

    assert plate is None
    assert confidence is None


def test_plate_ocr_returns_none_when_no_text_found(mocker) -> None:
    """Lista vazia do EasyOCR deve retornar (None, None) sem exceção."""
    mock_reader = MagicMock()
    mock_reader.readtext.return_value = []
    mocker.patch("easyocr.Reader", return_value=mock_reader)

    from src.ocr.plate_ocr import PlateOCR
    crop = np.zeros((50, 200, 3), dtype=np.uint8)
    plate, confidence = PlateOCR().read(crop)

    assert plate is None
    assert confidence is None


def test_plate_ocr_picks_result_with_highest_confidence(mocker) -> None:
    """Quando EasyOCR retorna múltiplos resultados, o de maior confiança deve ser usado."""
    mock_reader = MagicMock()
    mock_reader.readtext.return_value = [
        ([[0, 0], [50, 0], [50, 30], [0, 30]], "GARBAGE", 0.4),
        ([[0, 0], [100, 0], [100, 30], [0, 30]], "ABC1234", 0.91),
        ([[0, 0], [80, 0], [80, 30], [0, 30]], "TRASH01", 0.6),
    ]
    mocker.patch("easyocr.Reader", return_value=mock_reader)

    from src.ocr.plate_ocr import PlateOCR
    crop = np.zeros((50, 200, 3), dtype=np.uint8)
    plate, confidence = PlateOCR().read(crop)

    assert plate == "ABC1234"
    assert confidence == pytest.approx(0.91)


# ── OCRWorker ────────────────────────────────────────────────────────────────

def test_ocr_worker_forwards_valid_event_to_db_queue(mocker) -> None:
    """OCRWorker deve processar crop, adicionar placa e encaminhar para db_queue."""
    mock_ocr = MagicMock()
    mock_ocr.read.return_value = ("ABC1234", 0.85)

    ocr_q: queue.Queue = queue.Queue()
    db_q: queue.Queue = queue.Queue()

    from src.ocr.plate_ocr import OCRWorker
    worker = OCRWorker(ocr_queue=ocr_q, db_queue=db_q, plate_ocr=mock_ocr)
    worker.start()

    crop = np.zeros((100, 200, 3), dtype=np.uint8)
    event_meta = {
        "track_id": 42,
        "vehicle_class": "car",
        "frame_number": 500,
        "timestamp": "2026-01-01T00:00:00Z",
        "session_id": "sess-test",
    }
    ocr_q.put((42, crop, event_meta))

    time.sleep(0.3)
    worker.stop_and_join(timeout=2.0)

    assert not db_q.empty(), "db_queue deve ter recebido o evento processado"
    result = db_q.get_nowait()
    assert result["track_id"] == 42
    assert result["plate_text"] == "ABC1234"
    assert result["plate_confidence"] == pytest.approx(0.85)


def test_ocr_worker_does_not_block_on_full_db_queue(mocker) -> None:
    """OCRWorker não deve travar quando db_queue está cheia — descarta e continua."""
    mock_ocr = MagicMock()
    mock_ocr.read.return_value = (None, None)

    ocr_q: queue.Queue = queue.Queue()
    db_q: queue.Queue = queue.Queue(maxsize=1)
    db_q.put({"dummy": True})  # fila cheia

    from src.ocr.plate_ocr import OCRWorker
    worker = OCRWorker(ocr_queue=ocr_q, db_queue=db_q, plate_ocr=mock_ocr)
    worker.start()

    crop = np.zeros((100, 200, 3), dtype=np.uint8)
    ocr_q.put((1, crop, {"track_id": 1, "vehicle_class": "car",
                         "frame_number": 1, "timestamp": "now", "session_id": "s"}))

    time.sleep(0.3)
    worker.stop_and_join(timeout=2.0)

    assert not worker._thread.is_alive(), "OCRWorker deve ter encerrado sem travar"
