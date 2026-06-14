"""
Testes unitários para src/database/models.py e src/database/db_writer.py.

Usa SQLite em memória — sem arquivo em disco, sem estado persistido entre testes.
"""
from __future__ import annotations

import queue
import threading
import time
from datetime import datetime, timezone

import os
import tempfile

import pytest
from sqlalchemy import create_engine, text
from sqlalchemy.orm import Session
from sqlalchemy.pool import StaticPool


# ── Helpers ──────────────────────────────────────────────────────────────────

def _make_event_dict(track_id: int, session_id: str = "test-session") -> dict:
    return {
        "track_id": track_id,
        "vehicle_class": "car",
        "plate_text": None,
        "plate_confidence": None,
        "frame_number": track_id * 10,
        "timestamp": datetime.now(timezone.utc),
        "session_id": session_id,
    }


# ── VehicleEventModel ─────────────────────────────────────────────────────────

def test_vehicle_event_model_can_be_persisted() -> None:
    """VehicleEventModel deve persistir em SQLite e receber ID automático."""
    from src.database.models import Base, VehicleEventModel

    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)

    with Session(engine) as session:
        event = VehicleEventModel(
            track_id=1,
            vehicle_class="car",
            plate_text=None,
            plate_confidence=None,
            frame_number=100,
            timestamp=datetime.now(timezone.utc),
            session_id="test-session-id",
        )
        session.add(event)
        session.commit()
        session.refresh(event)

    assert event.id is not None, "Chave primária deve ser atribuída pelo banco"


def test_vehicle_event_model_nullable_plate_fields() -> None:
    """plate_text e plate_confidence nulos devem ser armazenados e recuperados como None."""
    from src.database.models import Base, VehicleEventModel

    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)

    with Session(engine) as session:
        event = VehicleEventModel(
            track_id=5,
            vehicle_class="truck",
            plate_text=None,
            plate_confidence=None,
            frame_number=200,
            timestamp=datetime.now(timezone.utc),
            session_id="s",
        )
        session.add(event)
        session.commit()
        stored_id = event.id

    with Session(engine) as session:
        retrieved = session.get(VehicleEventModel, stored_id)
        assert retrieved.plate_text is None
        assert retrieved.plate_confidence is None


def test_vehicle_event_model_with_plate_data() -> None:
    """plate_text e plate_confidence com valores devem ser armazenados corretamente."""
    from src.database.models import Base, VehicleEventModel

    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)

    with Session(engine) as session:
        event = VehicleEventModel(
            track_id=3,
            vehicle_class="car",
            plate_text="ABC1D23",
            plate_confidence=0.92,
            frame_number=50,
            timestamp=datetime.now(timezone.utc),
            session_id="s",
        )
        session.add(event)
        session.commit()
        stored_id = event.id

    with Session(engine) as session:
        retrieved = session.get(VehicleEventModel, stored_id)
        assert retrieved.plate_text == "ABC1D23"
        assert retrieved.plate_confidence == pytest.approx(0.92)


def test_create_sqlite_engine_allows_multithread_access() -> None:
    """create_sqlite_engine deve criar engine com check_same_thread=False.

    Usa arquivo temporário real porque SQLite :memory: cria uma base distinta
    por conexão — threads receberiam uma base vazia se usassem o pool padrão.
    """
    from src.database.models import Base, VehicleEventModel, create_sqlite_engine

    fd, db_path = tempfile.mkstemp(suffix=".db")
    os.close(fd)

    try:
        engine = create_sqlite_engine(db_path)
        Base.metadata.create_all(engine)

        errors: list[Exception] = []

        def write_from_thread() -> None:
            try:
                with Session(engine) as session:
                    session.add(VehicleEventModel(
                        track_id=99,
                        vehicle_class="bus",
                        plate_text=None,
                        plate_confidence=None,
                        frame_number=1,
                        timestamp=datetime.now(timezone.utc),
                        session_id="thread-test",
                    ))
                    session.commit()
            except Exception as exc:
                errors.append(exc)

        t = threading.Thread(target=write_from_thread)
        t.start()
        t.join(timeout=2.0)

        assert not errors, f"Acesso multithread gerou erro: {errors}"
    finally:
        engine.dispose()
        os.unlink(db_path)


# ── DbWriter ─────────────────────────────────────────────────────────────────

def test_db_writer_inserts_events_from_queue() -> None:
    """DbWriter deve persistir todos os eventos colocados na fila.

    StaticPool garante que main thread e DbWriter thread usem a MESMA conexão
    in-memory — sem StaticPool cada thread veria uma base vazia diferente.
    """
    from sqlalchemy import create_engine
    from src.database.models import Base, VehicleEventModel
    from src.database.db_writer import DbWriter

    engine = create_engine(
        "sqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    Base.metadata.create_all(engine)

    db_q: queue.Queue = queue.Queue()
    for i in range(3):
        db_q.put(_make_event_dict(i))

    writer = DbWriter(db_queue=db_q, engine=engine)
    writer.start()
    time.sleep(0.5)
    writer.flush_and_close()

    with Session(engine) as session:
        count = session.query(VehicleEventModel).count()

    assert count == 3, f"Esperado 3 eventos, encontrado {count}"


def test_db_writer_batch_commits_at_10_events() -> None:
    """DbWriter deve commitar automaticamente quando batch atinge 10 eventos."""
    from sqlalchemy import create_engine
    from src.database.models import Base, VehicleEventModel
    from src.database.db_writer import DbWriter

    engine = create_engine(
        "sqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    Base.metadata.create_all(engine)

    db_q: queue.Queue = queue.Queue()
    writer = DbWriter(db_queue=db_q, engine=engine)
    writer.start()

    for i in range(10):
        db_q.put(_make_event_dict(i))

    time.sleep(0.5)  # aguarda DbWriter processar o batch completo

    with Session(engine) as session:
        count = session.query(VehicleEventModel).count()

    writer.flush_and_close()
    assert count == 10, f"Esperado 10 após batch completo, encontrado {count}"


def test_db_writer_flushes_remaining_on_close() -> None:
    """flush_and_close deve persistir itens parciais antes de encerrar."""
    from sqlalchemy import create_engine
    from src.database.models import Base, VehicleEventModel
    from src.database.db_writer import DbWriter

    engine = create_engine(
        "sqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    Base.metadata.create_all(engine)

    db_q: queue.Queue = queue.Queue()
    writer = DbWriter(db_queue=db_q, engine=engine)
    writer.start()

    # Menos de 10 eventos — não dispara batch automático
    for i in range(4):
        db_q.put(_make_event_dict(i))

    time.sleep(0.2)
    writer.flush_and_close()  # deve forçar flush dos 4 eventos pendentes

    with Session(engine) as session:
        count = session.query(VehicleEventModel).count()

    assert count == 4, f"Esperado 4 eventos após flush_and_close, encontrado {count}"
