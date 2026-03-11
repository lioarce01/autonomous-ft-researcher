"""SQLite wrapper for experiment tracking."""
import sqlite3
import os
from datetime import datetime

DB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "experiments.db")


def get_conn():
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    with get_conn() as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS experiments (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                name        TEXT    NOT NULL,
                accuracy    REAL    NOT NULL,
                notes       TEXT    DEFAULT '',
                hypothesis  TEXT    DEFAULT '',
                kept        INTEGER DEFAULT 0,
                timestamp   TEXT    NOT NULL
            )
        """)
        conn.commit()


def insert_experiment(name: str, accuracy: float, notes: str = "", hypothesis: str = "") -> int:
    init_db()
    ts = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")
    with get_conn() as conn:
        cur = conn.execute(
            "INSERT INTO experiments (name, accuracy, notes, hypothesis, kept, timestamp) VALUES (?,?,?,?,0,?)",
            (name, accuracy, notes, hypothesis, ts),
        )
        conn.commit()
        return cur.lastrowid


def get_best_kept_accuracy() -> float | None:
    init_db()
    with get_conn() as conn:
        row = conn.execute(
            "SELECT MAX(accuracy) as best FROM experiments WHERE kept=1"
        ).fetchone()
        return row["best"] if row and row["best"] is not None else None


def set_kept(experiment_id: int):
    with get_conn() as conn:
        conn.execute("UPDATE experiments SET kept=1 WHERE id=?", (experiment_id,))
        conn.commit()


def get_all_experiments() -> list[dict]:
    init_db()
    with get_conn() as conn:
        rows = conn.execute(
            "SELECT * FROM experiments ORDER BY accuracy DESC, timestamp DESC"
        ).fetchall()
        return [dict(r) for r in rows]


def get_recent_not_kept(n: int = 3) -> list[dict]:
    init_db()
    with get_conn() as conn:
        rows = conn.execute(
            "SELECT * FROM experiments WHERE kept=0 ORDER BY timestamp DESC LIMIT ?", (n,)
        ).fetchall()
        return [dict(r) for r in rows]
