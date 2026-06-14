"""
Testes unitários para a thread de captura de vídeo.

cv2.VideoCapture é mockado — nenhum arquivo de vídeo real é necessário.
"""
from __future__ import annotations

import queue
import time
from unittest.mock import MagicMock

import cv2
import numpy as np
import pytest

from src.config import VideoSettings


def _make_mock_cap(mocker, *, fps: float = 30.0, width: float = 640.0,
                   height: float = 480.0, frame: np.ndarray | None = None) -> MagicMock:
    """Fábrica de mock para cv2.VideoCapture."""
    if frame is None:
        frame = np.zeros((int(height), int(width), 3), dtype=np.uint8)

    mock_cap = MagicMock()
    mock_cap.isOpened.return_value = True
    mock_cap.read.return_value = (True, frame.copy())
    mock_cap.get.side_effect = lambda prop: {
        cv2.CAP_PROP_FPS: fps,
        cv2.CAP_PROP_FRAME_WIDTH: width,
        cv2.CAP_PROP_FRAME_HEIGHT: height,
    }.get(prop, 0.0)

    mocker.patch("cv2.VideoCapture", return_value=mock_cap)
    return mock_cap


def _make_settings(realtime: bool = False) -> VideoSettings:
    return VideoSettings(source="dummy.mp4", output="out.mp4", resize_width=640, realtime=realtime)


# ---------------------------------------------------------------------------
# Testes
# ---------------------------------------------------------------------------

def test_frames_arrive_in_queue_after_start(mocker) -> None:
    """Após start(), frames devem aparecer na fila dentro de 200 ms."""
    _make_mock_cap(mocker)

    from src.capture.video_capture import VideoCapture

    q: queue.Queue[np.ndarray] = queue.Queue(maxsize=3)
    cap = VideoCapture("dummy.mp4", q, _make_settings())
    cap.start()
    time.sleep(0.2)
    cap.stop()

    assert not q.empty(), "A fila deve conter pelo menos um frame após iniciar a captura"


def test_stop_cleans_up_without_deadlock(mocker) -> None:
    """stop() deve encerrar a thread sem travar (timeout de 2 s)."""
    _make_mock_cap(mocker)

    from src.capture.video_capture import VideoCapture

    q: queue.Queue[np.ndarray] = queue.Queue(maxsize=3)
    cap = VideoCapture("dummy.mp4", q, _make_settings())
    cap.start()
    time.sleep(0.05)
    cap.stop()

    assert not cap.is_alive(), "A thread de captura deve ter encerrado após stop()"


def test_frame_drop_discards_oldest_when_queue_is_full(mocker) -> None:
    """realtime=True: quando a fila está cheia, descarta o mais antigo e insere o novo."""
    old_frame = np.zeros((480, 640, 3), dtype=np.uint8)
    new_frame = np.full((480, 640, 3), 128, dtype=np.uint8)

    q: queue.Queue[np.ndarray] = queue.Queue(maxsize=1)
    q.put(old_frame)

    mock_cap = MagicMock()
    mock_cap.isOpened.return_value = True
    mock_cap.read.return_value = (True, new_frame.copy())
    mock_cap.get.side_effect = lambda prop: {
        cv2.CAP_PROP_FPS: 30.0,
        cv2.CAP_PROP_FRAME_WIDTH: 640.0,
        cv2.CAP_PROP_FRAME_HEIGHT: 480.0,
    }.get(prop, 0.0)
    mocker.patch("cv2.VideoCapture", return_value=mock_cap)

    from src.capture.video_capture import VideoCapture

    cap = VideoCapture("dummy.mp4", q, _make_settings(realtime=True))
    cap.start()
    time.sleep(0.1)
    cap.stop()

    assert q.qsize() == 1, "A fila deve continuar com exatamente 1 item"
    item = q.get_nowait()
    assert np.array_equal(item, new_frame), (
        "O frame antigo deve ter sido descartado e o novo deve estar na fila"
    )


def test_fps_property_returns_source_fps(mocker) -> None:
    """A propriedade fps deve refletir o valor de CAP_PROP_FPS da fonte."""
    _make_mock_cap(mocker, fps=25.0)

    from src.capture.video_capture import VideoCapture

    q: queue.Queue[np.ndarray] = queue.Queue(maxsize=3)
    cap = VideoCapture("dummy.mp4", q, _make_settings())

    assert cap.fps == 25.0


def test_frame_width_and_height_properties(mocker) -> None:
    """frame_width e frame_height devem refletir as dimensões da fonte de vídeo."""
    _make_mock_cap(mocker, width=1280.0, height=720.0)

    from src.capture.video_capture import VideoCapture

    q: queue.Queue[np.ndarray] = queue.Queue(maxsize=3)
    cap = VideoCapture("dummy.mp4", q, _make_settings())

    assert cap.frame_width == 1280
    assert cap.frame_height == 720


def test_corrupted_frame_is_silently_discarded(mocker) -> None:
    """Quando read() retorna False, o frame é descartado sem colocar None na fila."""
    mock_cap = MagicMock()
    mock_cap.isOpened.return_value = True
    mock_cap.read.return_value = (False, None)  # frame corrompido/fim de stream
    mock_cap.get.return_value = 30.0
    mocker.patch("cv2.VideoCapture", return_value=mock_cap)

    from src.capture.video_capture import VideoCapture

    q: queue.Queue[np.ndarray] = queue.Queue(maxsize=3)
    cap = VideoCapture("dummy.mp4", q, _make_settings())
    cap.start()
    time.sleep(0.1)
    cap.stop()

    assert q.empty(), "Frames corrompidos não devem ser colocados na fila"


def test_realtime_false_waits_for_space_without_dropping(mocker) -> None:
    """realtime=False: com fila cheia, a thread bloqueia sem descartar o frame antigo."""
    sentinel = np.zeros((480, 640, 3), dtype=np.uint8)
    new_frame = np.full((480, 640, 3), 99, dtype=np.uint8)

    mock_cap = MagicMock()
    mock_cap.isOpened.return_value = True
    mock_cap.read.return_value = (True, new_frame.copy())
    mock_cap.get.side_effect = lambda prop: {
        cv2.CAP_PROP_FPS: 30.0,
        cv2.CAP_PROP_FRAME_WIDTH: 640.0,
        cv2.CAP_PROP_FRAME_HEIGHT: 480.0,
    }.get(prop, 0.0)
    mocker.patch("cv2.VideoCapture", return_value=mock_cap)

    q: queue.Queue[np.ndarray] = queue.Queue(maxsize=1)
    q.put(sentinel)  # fila cheia com frame sentinela

    from src.capture.video_capture import VideoCapture

    cap = VideoCapture("dummy.mp4", q, _make_settings(realtime=False))
    cap.start()

    # Thread deve estar bloqueada aguardando espaço — sentinel intacto
    time.sleep(0.3)
    assert q.qsize() == 1, "Fila deve permanecer cheia: backpressure não descarta"

    # Liberar espaço: consumir o sentinel
    item = q.get_nowait()
    assert np.array_equal(item, sentinel), "sentinel deve ser o primeiro a sair, não foi descartado"

    # Thread agora consegue inserir new_frame
    time.sleep(0.2)
    cap.stop()

    assert not q.empty(), "new_frame deve ter sido inserido após liberar espaço"
    result = q.get_nowait()
    assert np.array_equal(result, new_frame), "frame inserido deve ser new_frame"


def test_queue_put_timeout_does_not_increment_eof_counter(mocker) -> None:
    """Timeout de put na fila (realtime=False) nunca deve acionar detecção de EOF.

    Se o contador de falhas fosse incrementado por timeouts de fila,
    5 timeouts × 0.1 s = 0.5 s bastariam para encerrar a thread.
    Após 0.9 s com fila cheia e leituras sempre bem-sucedidas, a thread
    deve continuar viva — provando que os contadores estão separados.
    """
    good_frame = np.full((480, 640, 3), 7, dtype=np.uint8)

    mock_cap = MagicMock()
    mock_cap.isOpened.return_value = True
    mock_cap.read.return_value = (True, good_frame.copy())  # codec nunca falha
    mock_cap.get.side_effect = lambda prop: {
        cv2.CAP_PROP_FPS: 30.0,
        cv2.CAP_PROP_FRAME_WIDTH: 640.0,
        cv2.CAP_PROP_FRAME_HEIGHT: 480.0,
    }.get(prop, 0.0)
    mocker.patch("cv2.VideoCapture", return_value=mock_cap)

    q: queue.Queue[np.ndarray] = queue.Queue(maxsize=1)
    q.put(good_frame)  # mantém fila cheia durante todo o teste

    from src.capture.video_capture import VideoCapture

    cap = VideoCapture("dummy.mp4", q, _make_settings(realtime=False))
    cap.start()

    # 0.9 s >> 5 × 0.1 s (limiar de EOF se o bug existisse)
    time.sleep(0.9)

    assert cap.is_alive(), (
        "Thread não deve encerrar por EOF quando a fila está cheia: "
        "timeouts de put != falhas de leitura do codec"
    )
    cap.stop()
