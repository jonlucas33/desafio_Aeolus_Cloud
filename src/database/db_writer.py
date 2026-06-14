"""
Thread consumidora da fila de banco de dados.

Responsabilidade única: ler dicts de evento de db_queue e persistir em batch
no banco, com commit automático a cada 10 eventos ou a cada 5 segundos.
"""
from __future__ import annotations

import logging
import queue
import threading
import time

from sqlalchemy.orm import Session

from src.database.models import VehicleEventModel

logger = logging.getLogger(__name__)

_BATCH_SIZE = 10
_FLUSH_INTERVAL_S = 5.0


class DbWriter:
    """Thread Consumer que persiste eventos de veículos em batch no banco.

    Cada item de db_queue é um dict com os campos de VehicleEventModel
    (exceto id, que é autoincrement). O commit ocorre automaticamente
    ao acumular _BATCH_SIZE itens ou após _FLUSH_INTERVAL_S segundos.

    Args:
        db_queue: Fila compartilhada com o pipeline principal.
        engine: Engine SQLAlchemy (deve ter check_same_thread=False para SQLite).
    """

    def __init__(self, db_queue: queue.Queue, engine) -> None:
        self._queue = db_queue
        self._engine = engine
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        """Inicia a thread daemon de escrita no banco."""
        self._thread = threading.Thread(
            target=self._run, daemon=True, name="DbWriter"
        )
        self._thread.start()
        logger.info("DbWriter iniciado")

    def flush_and_close(self, timeout: float = 5.0) -> None:
        """Sinaliza parada, aguarda flush dos itens pendentes e encerra a thread.

        Args:
            timeout: Segundos máximos de espera pelo encerramento.
        """
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=timeout)
        logger.info("DbWriter encerrado")

    def _run(self) -> None:
        """Loop da thread: drena a fila em batches com flush temporizado."""
        batch: list[dict] = []
        last_flush = time.monotonic()

        while not self._stop_event.is_set() or not self._queue.empty():
            try:
                item = self._queue.get(timeout=0.1)
                batch.append(item)

                if len(batch) >= _BATCH_SIZE:
                    self._commit_batch(batch)
                    batch.clear()
                    last_flush = time.monotonic()

            except queue.Empty:
                pass

            if batch and (time.monotonic() - last_flush >= _FLUSH_INTERVAL_S):
                self._commit_batch(batch)
                batch.clear()
                last_flush = time.monotonic()

        # Flush final ao encerrar
        if batch:
            self._commit_batch(batch)

    def _commit_batch(self, batch: list[dict]) -> None:
        """Persiste batch no banco. Erros são logados sem propagar."""
        try:
            with Session(self._engine) as session:
                session.add_all([VehicleEventModel(**item) for item in batch])
                session.commit()
            logger.debug("DbWriter: %d evento(s) persistido(s)", len(batch))
        except Exception:
            logger.warning(
                "DbWriter: falha ao inserir batch de %d evento(s)", len(batch),
                exc_info=True,
            )
