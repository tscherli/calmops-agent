"""Local CSV backend for mood log rows.

No external service, no credentials -- appends structured rows to a local
CSV file. Any write failure falls back to a separate JSONL file rather than
losing the entry; append_mood_row/log_mood never raise past this module's
boundary -- callers get a structured status instead.

Replaced the original Google Sheets/OAuth backend (see DEVELOPMENT_LOG.md,
"Stage 3 revision") after real end-to-end OAuth setup turned out to be far
more friction than a personal tool should require, and after weighing
several alternatives. This keeps the log_mood tool's contract identical --
agent/graph.py needed only a one-line import change.
"""
from __future__ import annotations

import csv
import json
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

from langchain_core.tools import tool

LOG_COLUMNS = [
    "timestamp",
    "row_id",
    "valence",
    "energy_level",
    "circumstances",
    "measure_taken",
    "helpfulness",
    "note",
]

BASE_DIR = Path(__file__).resolve().parent.parent
LOG_PATH = BASE_DIR / "data" / "mood_log.csv"
PENDING_LOG_PATH = BASE_DIR / "state" / "pending_sheet_log.jsonl"

Valence = Literal["good", "neutral", "bad"]
EnergyLevel = Literal["low", "medium", "high"]


def _write_pending_fallback(row: dict[str, Any]) -> None:
    PENDING_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    with PENDING_LOG_PATH.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row) + "\n")


def _build_note(situation: str | None, thoughts: str | None, need: str | None, next_step: str | None) -> str:
    parts = []
    if situation:
        parts.append(f"situation: {situation}")
    if thoughts:
        parts.append(f"thoughts: {thoughts}")
    if need:
        parts.append(f"need: {need}")
    if next_step:
        parts.append(f"next_step: {next_step}")
    return " | ".join(parts)


def append_mood_row(
    valence: Valence,
    circumstances: str,
    energy_level: EnergyLevel | None = None,
    measure_taken: str | None = None,
    helpfulness: str | None = None,
    situation: str | None = None,
    thoughts: str | None = None,
    need: str | None = None,
    next_step: str | None = None,
    *,
    path: Path | None = None,
) -> dict[str, Any]:
    """Append one structured mood row to the local CSV log. Never raises --
    any failure (e.g. a disk I/O error) falls back to a local JSONL file
    and returns status="error" instead.

    energy_level is optional and orthogonal to valence -- it captures
    fatigue/exhaustion (an emotional-exhaustion-adjacent signal from burnout
    research) as its own trend, distinct from day-to-day mood. See README
    "Why energy_level exists" for the non-clinical framing/sources."""
    row_id = str(uuid.uuid4())
    row = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "row_id": row_id,
        "valence": valence,
        "energy_level": energy_level or "",
        "circumstances": circumstances,
        "measure_taken": measure_taken or "",
        "helpfulness": helpfulness or "",
        "note": _build_note(situation, thoughts, need, next_step),
    }

    log_path = path or LOG_PATH
    try:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        write_header = not log_path.exists() or log_path.stat().st_size == 0
        with log_path.open("a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=LOG_COLUMNS)
            if write_header:
                writer.writeheader()
            writer.writerow(row)
        return {"status": "ok", "row_id": row_id}
    except Exception as exc:
        _write_pending_fallback(row)
        return {"status": "error", "row_id": row_id, "error": str(exc)}


@tool
def log_mood(
    valence: Valence,
    circumstances: str,
    energy_level: str = "",
    measure_taken: str = "",
    helpfulness: str = "",
    situation: str = "",
    thoughts: str = "",
    need: str = "",
    next_step: str = "",
) -> dict[str, Any]:
    """Log a structured mood entry to the user's local mood log.

    valence: "good", "neutral", or "bad".
    circumstances: brief description of what's going on.
    energy_level: "low", "medium", or "high", only if fatigue/energy came up -- distinct from
    valence (you can be low-energy-but-content, or high-energy-but-stressed).
    measure_taken / helpfulness: only if the user named a coping action and how well it worked.
    situation / thoughts / need / next_step: deeper-structuring fields, only for heavier check-ins
    that went through the structuring sequence.
    """
    return append_mood_row(
        valence=valence,
        circumstances=circumstances,
        energy_level=energy_level or None,
        measure_taken=measure_taken or None,
        helpfulness=helpfulness or None,
        situation=situation or None,
        thoughts=thoughts or None,
        need=need or None,
        next_step=next_step or None,
    )
