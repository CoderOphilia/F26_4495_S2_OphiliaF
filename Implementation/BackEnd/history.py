"""Store classification summaries in a local SQLite database."""

from contextlib import closing
from datetime import datetime, timezone
from pathlib import Path
import sqlite3


SCHEMA = """
CREATE TABLE IF NOT EXISTS classification_history (
    id INTEGER PRIMARY KEY,
    analyzed_at_utc TEXT NOT NULL,
    prediction TEXT NOT NULL CHECK (
        prediction IN ('normal', 'sag', 'swell', 'interruption', 'harmonic_distortion')
    ),
    model TEXT NOT NULL CHECK (model IN ('random_forest', 'svm')),
    added_noise_snr_db INTEGER CHECK (added_noise_snr_db IN (20, 30, 40)),
    analysis_time_ms REAL NOT NULL CHECK (analysis_time_ms >= 0)
)
"""


def save_analysis(database_path, *, prediction, model, added_noise_snr_db, analysis_time_ms):
    """Commit one complete result using a separate connection per operation."""
    database_path = Path(database_path)
    database_path.parent.mkdir(parents=True, exist_ok=True)
    with closing(sqlite3.connect(database_path, timeout=5)) as connection:
        with connection:
            connection.execute("BEGIN IMMEDIATE")
            connection.execute(SCHEMA)
            connection.execute(
                """INSERT INTO classification_history
                   (analyzed_at_utc, prediction, model, added_noise_snr_db, analysis_time_ms)
                   VALUES (?, ?, ?, ?, ?)""",
                (
                    datetime.now(timezone.utc).isoformat(timespec="milliseconds"),
                    prediction, model, added_noise_snr_db, analysis_time_ms,
                ),
            )


def read_history(database_path, limit=20):
    """Return the most recently saved results first, up to 100 records."""
    return read_history_page(database_path, limit)["history"]


def read_history_page(database_path, limit=20, before_id=None):
    """Read one page using an ID cursor that is stable when new results arrive."""
    if isinstance(limit, bool) or not isinstance(limit, int) or not 1 <= limit <= 100:
        raise ValueError("History limit must be a whole number between 1 and 100.")
    if before_id is not None and (
        isinstance(before_id, bool)
        or not isinstance(before_id, int)
        or not 1 <= before_id <= 9223372036854775807
    ):
        raise ValueError("History cursor must be a positive SQLite record ID.")
    database_path = Path(database_path)
    if not database_path.exists():
        return {"history": [], "has_more": False, "next_before_id": None}
    where_clause = ""
    parameters = (limit + 1,)
    if before_id is not None:
        where_clause = "WHERE id < ?"
        parameters = (before_id, limit + 1)
    with closing(sqlite3.connect(database_path, timeout=5)) as connection:
        connection.row_factory = sqlite3.Row
        rows = connection.execute(
            f"""SELECT id, analyzed_at_utc, prediction, model,
                      added_noise_snr_db, analysis_time_ms
               FROM classification_history {where_clause} ORDER BY id DESC LIMIT ?""",
            parameters,
        ).fetchall()
    records = [dict(row) for row in rows[:limit]]
    has_more = len(rows) > limit
    return {
        "history": records,
        "has_more": has_more,
        "next_before_id": str(records[-1]["id"]) if has_more else None,
    }
