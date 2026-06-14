"""
Thread Producer de captura de vídeo.

Responsabilidade única: ler frames de uma fonte de vídeo de forma não-bloqueante
e manter frame_queue preenchida com os frames mais recentes.
"""
from __future__ import annotations

import logging
import queue
import sys
import threading

import cv2
import numpy as np

from src.config import VideoSettings

logger = logging.getLogger(__name__)


class VideoCapture:
    """Thread Producer que lê frames e os coloca em frame_queue.

    Quando a fila está cheia, o frame mais antigo é descartado para inserir
    o novo — nunca acumula latência.

    Args:
        source: Caminho de arquivo ou índice de câmera (0, 1, ...).
        frame_queue: Fila compartilhada com o loop de inferência.
        settings: Configurações de vídeo (VideoSettings).
    """

    def __init__(
        self,
        source: str | int,
        frame_queue: queue.Queue[np.ndarray],
        settings: VideoSettings,
    ) -> None:
        self._source = source
        self._queue = frame_queue
        self._settings = settings
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None

        self._cap = cv2.VideoCapture(source)
        if not self._cap.isOpened():
            logger.error("Não foi possível abrir a fonte de vídeo: %s", source)
            sys.exit(1)

        self._fps: float = float(self._cap.get(cv2.CAP_PROP_FPS))
        self._frame_width: int = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self._frame_height: int = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # ------------------------------------------------------------------
    # Interface pública
    # ------------------------------------------------------------------

    def start(self) -> None:
        """Inicia a thread daemon de captura."""
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._run, daemon=True, name="VideoCapture")
        self._thread.start()
        logger.info("Thread de captura iniciada (fonte=%s, fps=%.1f)", self._source, self._fps)

    def stop(self) -> None:
        """Sinaliza parada e aguarda o encerramento da thread (timeout 2 s)."""
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
        self._cap.release()
        logger.info("Thread de captura encerrada.")

    def join(self, timeout: float | None = None) -> None:
        """Aguarda o encerramento da thread de captura com timeout configurável.

        Complementa stop(): permite ao caller controlar o timeout de espera
        independentemente do join interno de 2 s embutido em stop().

        Args:
            timeout: Segundos máximos de espera. None = espera indefinida.
        """
        if self._thread is not None:
            self._thread.join(timeout=timeout)

    def is_alive(self) -> bool:
        """Retorna True se a thread de captura ainda está em execução."""
        return self._thread is not None and self._thread.is_alive()

    @property
    def fps(self) -> float:
        """FPS real da fonte de vídeo."""
        return self._fps

    @property
    def frame_width(self) -> int:
        """Largura dos frames da fonte de vídeo em pixels."""
        return self._frame_width

    @property
    def frame_height(self) -> int:
        """Altura dos frames da fonte de vídeo em pixels."""
        return self._frame_height

    # ------------------------------------------------------------------
    # Loop interno da thread
    # ------------------------------------------------------------------

    def _run(self) -> None:
        """Loop de leitura executado na thread daemon.

        Detecta EOF quando ≥ 5 leituras consecutivas falham, sinalizando
        stop_event para encerramento limpo do pipeline principal.
        """
        _EOF_THRESHOLD = 5
        consecutive_failures = 0

        while not self._stop_event.is_set():
            ok, frame = self._cap.read()
            if not ok:
                consecutive_failures += 1
                logger.debug("Frame inválido ou fim de stream — descartado.")
                if consecutive_failures >= _EOF_THRESHOLD:
                    logger.info(
                        "Fim do stream detectado (%d falhas consecutivas) — encerrando captura",
                        consecutive_failures,
                    )
                    self._stop_event.set()
                    break
                continue

            consecutive_failures = 0
            self._put_frame(frame)

    def _put_frame(self, frame: np.ndarray) -> None:
        """Insere frame na fila, descartando o mais antigo se ela estiver cheia."""
        try:
            self._queue.put_nowait(frame)
        except queue.Full:
            try:
                self._queue.get_nowait()
            except queue.Empty:
                pass
            self._queue.put_nowait(frame)
