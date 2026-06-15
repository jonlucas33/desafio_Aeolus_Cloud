"""
Consulta ao banco SQLite de eventos de veículos.

Descobre automaticamente o session_id mais recente e filtra todas as
queries por ele, evitando mistura de contagens entre execuções distintas.
"""
from __future__ import annotations

import sqlite3
import sys
from pathlib import Path

import pandas as pd


def check_database(db_path: str = "data/outputs/events.db") -> None:
    """Exibe estatísticas da execução mais recente registrada no banco.

    Args:
        db_path: Caminho para o arquivo .db do SQLite.
    """
    path = Path(db_path)
    if not path.exists():
        print(f"Banco de dados não encontrado: {path}")
        sys.exit(1)

    with sqlite3.connect(db_path) as conn:
        # ── Descobrir sessão mais recente ─────────────────────────────────
        row = conn.execute(
            "SELECT session_id FROM vehicle_events ORDER BY timestamp DESC LIMIT 1"
        ).fetchone()

        if row is None:
            print("Banco de dados vazio — nenhuma execução registrada.")
            return

        session_id: str = row[0]
        print(f"\n{'='*56}")
        print(f"  Sessão analisada: {session_id}")
        print(f"{'='*56}")

        # ── 1. Contagem por classe ────────────────────────────────────────
        print("\n=== 1. CONTAGEM POR CLASSE DE VEÍCULO ===")
        df1 = pd.read_sql_query(
            """
            SELECT vehicle_class, COUNT(*) AS total
            FROM vehicle_events
            WHERE session_id = ?
            GROUP BY vehicle_class
            ORDER BY total DESC
            """,
            conn,
            params=(session_id,),
        )
        print(df1.to_string(index=False) if not df1.empty else "  (sem dados)")

        # ── 2. Placas detectadas ──────────────────────────────────────────
        print("\n=== 2. PLACAS DETECTADAS ===")
        df2 = pd.read_sql_query(
            """
            SELECT track_id, vehicle_class, plate_text, plate_confidence, frame_number
            FROM vehicle_events
            WHERE session_id = ?
              AND plate_text IS NOT NULL
            ORDER BY frame_number
            """,
            conn,
            params=(session_id,),
        )
        if df2.empty:
            print("  Nenhuma placa registrada nesta sessão.")
        else:
            print(df2.to_string(index=False))

        # ── 3. Total de eventos na sessão ─────────────────────────────────
        total = conn.execute(
            "SELECT COUNT(*) FROM vehicle_events WHERE session_id = ?",
            (session_id,),
        ).fetchone()[0]
        print(f"\n  Total de eventos nesta sessão: {total}")
        print(f"{'='*56}\n")


if __name__ == "__main__":
    db = sys.argv[1] if len(sys.argv) > 1 else "data/outputs/events.db"
    check_database(db)
