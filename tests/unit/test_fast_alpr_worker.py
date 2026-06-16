"""
Testes unitários para FastAlprWorker.

ALPR é mockado via pytest-mock — zero download de modelos durante os testes.
Cobre: processamento de placa válida, descarte de placa inválida,
encerramento via stop_event e recuperação de exceções do ALPR.
"""
from __future__ import annotations

import queue
import threading
import time
from unittest.mock import MagicMock

import numpy as np
import pytest


# ── Helpers ───────────────────────────────────────────────────────────────────

def _make_alpr_result(text: str, confidence: float) -> MagicMock:
    """Cria mock de ALPRResult com OCR."""
    ocr = MagicMock()
    ocr.text = text
    ocr.confidence = confidence
    result = MagicMock()
    result.ocr = ocr
    return result


def _make_event_meta(track_id: int = 1) -> dict:
    return {
        "track_id": track_id,
        "vehicle_class": "sedan_hatch",
        "frame_number": 100,
        "timestamp": "2026-01-01T00:00:00Z",
        "session_id": "bench-session",
    }


# ── Testes ───────────────────────────────────────────────────────────────────

def test_worker_processes_valid_plate_and_writes_to_db_queue(mocker) -> None:
    """Worker deve ler placa válida do ALPR e encaminhar evento para db_queue."""
    mock_alpr_instance = MagicMock()
    mock_alpr_instance.predict.return_value = [_make_alpr_result("ABC1234", 0.88)]
    mocker.patch("src.ocr.fast_alpr_worker.ALPR", return_value=mock_alpr_instance)

    from src.ocr.fast_alpr_worker import FastAlprWorker

    ocr_q: queue.Queue = queue.Queue()
    db_q: queue.Queue = queue.Queue()
    stop_event = threading.Event()
    settings = MagicMock()
    settings.alpr_detector_model = "yolo-v9-t-384-license-plate-end2end"
    settings.alpr_ocr_model = "cct-xs-v2-global-model"

    worker = FastAlprWorker(
        ocr_queue=ocr_q,
        db_queue=db_q,
        stop_event=stop_event,
        settings=settings,
    )
    worker.start()

    crop = np.zeros((100, 200, 3), dtype=np.uint8)
    ocr_q.put((1, crop, _make_event_meta(1)))

    time.sleep(0.4)
    stop_event.set()
    worker.join(timeout=2.0)

    assert not db_q.empty(), "db_queue deve ter recebido o evento"
    event = db_q.get_nowait()
    assert event["plate_text"] == "ABC1234"
    assert event["plate_confidence"] == pytest.approx(0.88)


def test_worker_discards_invalid_plate_format(mocker) -> None:
    """Worker deve ignorar texto que não passa na validação regex de placa.

    ALPR retorna "NOTAPLATE" com alta confiança → plate_text deve ser None.
    """
    mock_alpr_instance = MagicMock()
    mock_alpr_instance.predict.return_value = [_make_alpr_result("NOTAPLATE", 0.95)]
    mocker.patch("src.ocr.fast_alpr_worker.ALPR", return_value=mock_alpr_instance)

    from src.ocr.fast_alpr_worker import FastAlprWorker

    ocr_q: queue.Queue = queue.Queue()
    db_q: queue.Queue = queue.Queue()
    stop_event = threading.Event()
    settings = MagicMock()

    worker = FastAlprWorker(
        ocr_queue=ocr_q,
        db_queue=db_q,
        stop_event=stop_event,
        settings=settings,
    )
    worker.start()

    crop = np.zeros((100, 200, 3), dtype=np.uint8)
    ocr_q.put((2, crop, _make_event_meta(2)))

    time.sleep(0.4)
    stop_event.set()
    worker.join(timeout=2.0)

    assert not db_q.empty()
    event = db_q.get_nowait()
    assert event["plate_text"] is None
    assert event["plate_confidence"] is None


def test_worker_stops_within_timeout_when_stop_event_is_set(mocker) -> None:
    """Worker deve encerrar em até 2 s após stop_event ser acionado."""
    mock_alpr_instance = MagicMock()
    mock_alpr_instance.predict.return_value = []
    mocker.patch("src.ocr.fast_alpr_worker.ALPR", return_value=mock_alpr_instance)

    from src.ocr.fast_alpr_worker import FastAlprWorker

    ocr_q: queue.Queue = queue.Queue()
    db_q: queue.Queue = queue.Queue()
    stop_event = threading.Event()
    settings = MagicMock()

    worker = FastAlprWorker(
        ocr_queue=ocr_q,
        db_queue=db_q,
        stop_event=stop_event,
        settings=settings,
    )
    worker.start()

    stop_event.set()
    worker.join(timeout=2.0)

    assert not worker.is_alive(), "Worker deve ter encerrado dentro do timeout de 2 s"


def test_worker_handles_alpr_exception_without_crashing(mocker) -> None:
    """Exceção do ALPR não deve derrubar o worker — evento vai para db_queue com plate_text=None."""
    mock_alpr_instance = MagicMock()
    mock_alpr_instance.predict.side_effect = RuntimeError("ONNX runtime error")
    mocker.patch("src.ocr.fast_alpr_worker.ALPR", return_value=mock_alpr_instance)

    from src.ocr.fast_alpr_worker import FastAlprWorker

    ocr_q: queue.Queue = queue.Queue()
    db_q: queue.Queue = queue.Queue()
    stop_event = threading.Event()
    settings = MagicMock()

    worker = FastAlprWorker(
        ocr_queue=ocr_q,
        db_queue=db_q,
        stop_event=stop_event,
        settings=settings,
    )
    worker.start()

    crop = np.zeros((100, 200, 3), dtype=np.uint8)
    ocr_q.put((3, crop, _make_event_meta(3)))

    time.sleep(0.4)
    stop_event.set()
    worker.join(timeout=2.0)

    assert not worker.is_alive(), "Worker não deve travar após exceção do ALPR"
    assert not db_q.empty(), "Evento deve chegar ao db_queue mesmo com exceção"
    event = db_q.get_nowait()
    assert event["plate_text"] is None
    assert event["plate_confidence"] is None
