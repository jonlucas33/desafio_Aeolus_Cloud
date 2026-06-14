"""
Modelos SQLAlchemy e engine factory para persistência de eventos de veículos.

Responsabilidade única: definir o schema do banco e fornecer função de criação
de engine com acesso multithread seguro e WAL mode ativado.
"""
from __future__ import annotations

import logging
from datetime import datetime, timezone
from pathlib import Path

from sqlalchemy import DateTime, Float, Integer, String, event
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column

logger = logging.getLogger(__name__)


class Base(DeclarativeBase):
    pass


class VehicleEventModel(Base):
    """Registro de um cruzamento de linha virtual por um veículo rastreado.

    Attributes:
        id: Chave primária autoincrementada.
        track_id: ID do track atribuído pelo ByteTrack.
        vehicle_class: Classe do veículo (car, truck, bus, motorcycle).
        plate_text: Placa lida via OCR; None se não detectada.
        plate_confidence: Confiança do OCR; None se não detectada.
        frame_number: Frame do vídeo em que ocorreu o cruzamento.
        timestamp: Instante UTC do cruzamento.
        session_id: UUID da execução do pipeline que gerou o evento.
    """

    __tablename__ = "vehicle_events"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    track_id: Mapped[int] = mapped_column(Integer, nullable=False, index=True)
    vehicle_class: Mapped[str] = mapped_column(String(50), nullable=False)
    plate_text: Mapped[str | None] = mapped_column(String(20), nullable=True)
    plate_confidence: Mapped[float | None] = mapped_column(Float, nullable=True)
    frame_number: Mapped[int] = mapped_column(Integer, nullable=False)
    timestamp: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False
    )
    session_id: Mapped[str] = mapped_column(String(36), nullable=False, index=True)


def create_sqlite_engine(path: str | Path):
    """Cria engine SQLite com acesso multithread seguro e WAL mode.

    Args:
        path: Caminho do arquivo SQLite ou ":memory:" para banco em memória.

    Returns:
        Engine SQLAlchemy configurada para uso em threads concorrentes.
    """
    from sqlalchemy import create_engine

    engine = create_engine(
        f"sqlite:///{path}",
        connect_args={"check_same_thread": False},
    )

    @event.listens_for(engine, "connect")
    def _set_wal_pragmas(dbapi_conn, _connection_record) -> None:
        cursor = dbapi_conn.cursor()
        cursor.execute("PRAGMA journal_mode=WAL")
        cursor.execute("PRAGMA synchronous=NORMAL")
        cursor.close()

    return engine


def init_db(engine) -> None:
    """Cria todas as tabelas definidas em Base.metadata (idempotente).

    Args:
        engine: Engine SQLAlchemy a usar para criação das tabelas.
    """
    Base.metadata.create_all(engine)
    logger.info("Banco de dados inicializado")
