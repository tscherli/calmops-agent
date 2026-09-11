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

Stage-14 redesign changes (see the redesign plan's "New/changed components"):
- `valence` -> `mood` (values bad/neutral/good) and `energy_level` ->
  `energy` (values exhausted/calm/wired). This is a real vocabulary change
  the LLM's tool-calling depends on, not just a storage rename.
- `situation`/`thoughts`/`need`/`next_step` are real columns now, instead of
  being pipe-joined into one packed `note` string.
- New `tags` column for semantic classification (see tools/classify.py).

IMPORTANT -- who fills in `tags`: the `log_mood` @tool below deliberately
does NOT accept a `tags` argument, because the LLM does not invent tags.
In the real agent path, agent/graph.py intercepts the model's `log_mood`
tool call, runs `classify_entry` first, and then calls `append_mood_row`
itself with the resulting tags -- so the row is written to disk exactly
once, already tagged, and never rewritten afterwards. Callers that bypass
the graph (the quick-select buttons in app.py, which are deliberately
LLM-free) write untagged rows, exactly the way `energy` is already left
empty when nothing implied it.
"""
from __future__ import annotations

import csv
import json
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Literal

from langchain_core.tools import tool

LOG_COLUMNS = [
    "timestamp",
    "row_id",
    "mood",
    "energy",
    "circumstances",
    "measure_taken",
    "helpfulness",
    "situation",
    "thoughts",
    "need",
    "next_step",
    "tags",
]

BASE_DIR = Path(__file__).resolve().parent.parent
LOG_PATH = BASE_DIR / "data" / "mood_log.csv"
PENDING_LOG_PATH = BASE_DIR / "state" / "pending_sheet_log.jsonl"

Mood = Literal["bad", "neutral", "good"]
Energy = Literal["exhausted", "calm", "wired"]

# Tags are stored as a single ";"-joined string in one CSV column. Semicolon
# rather than comma purely so the stored value stays readable when the file is
# opened in a spreadsheet -- csv quoting would handle commas correctly either
# way, this is about human legibility, not correctness.
TAG_SEPARATOR = ";"


def _write_pending_fallback(row: dict[str, Any]) -> None:
    PENDING_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    with PENDING_LOG_PATH.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row) + "\n")


def format_tags(tags: Iterable[str] | None) -> str:
    """Join tags into the single stored CSV field. Empty/blank tags are
    dropped so a failed classification stores "" rather than junk."""
    if not tags:
        return ""
    return TAG_SEPARATOR.join(t.strip() for t in tags if t and t.strip())


def parse_tags(raw: str | None) -> list[str]:
    """Inverse of format_tags -- used by tools/log_query.py when scanning."""
    if not raw:
        return []
    return [t.strip() for t in raw.split(TAG_SEPARATOR) if t.strip()]


def append_mood_row(
    mood: Mood,
    circumstances: str,
    energy: Energy | None = None,
    measure_taken: str | None = None,
    helpfulness: str | None = None,
    situation: str | None = None,
    thoughts: str | None = None,
    need: str | None = None,
    next_step: str | None = None,
    tags: Iterable[str] | None = None,
    *,
    path: Path | None = None,
) -> dict[str, Any]:
    """Append one structured mood row to the local CSV log. Never raises --
    any failure (e.g. a disk I/O error) falls back to a local JSONL file
    and returns status="error" instead.

    mood and energy are two independent signals, not one scale: `mood` is
    valence (bad/neutral/good), `energy` is activation (exhausted/calm/wired).
    Someone can be tired regardless of mood, and a bad day can be either
    depleted (bad+exhausted) or agitated (bad+wired) -- those usually want
    different responses, which is the whole reason they stay separate
    columns. See README "Why mood and energy are separate."
    """
    row_id = str(uuid.uuid4())
    row = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "row_id": row_id,
        "mood": mood,
        "energy": energy or "",
        "circumstances": circumstances,
        "measure_taken": measure_taken or "",
        "helpfulness": helpfulness or "",
        "situation": situation or "",
        "thoughts": thoughts or "",
        "need": need or "",
        "next_step": next_step or "",
        "tags": format_tags(tags),
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
    mood: Mood,
    circumstances: str,
    energy: str = "",
    measure_taken: str = "",
    helpfulness: str = "",
    situation: str = "",
    thoughts: str = "",
    need: str = "",
    next_step: str = "",
) -> dict[str, Any]:
    """Log a structured mood entry to the user's local mood log.

    mood: "bad", "neutral", or "good" -- overall valence.
    circumstances: brief description of what's going on.
    energy: "exhausted", "calm", or "wired" -- how activated the user is, independent of mood
    (someone can be tired regardless of mood; "wired" means racing/restless/keyed up, whether
    from excitement or anxiety). Only fill this in when it's actually implied; never ask directly.
    measure_taken / helpfulness: only if the user named a coping action and how well it worked.
    situation / thoughts / need / next_step: deeper-structuring fields, only for heavier check-ins
    that went through the structuring sequence.
    """
    return append_mood_row(
        mood=mood,
        circumstances=circumstances,
        energy=energy or None,
        measure_taken=measure_taken or None,
        helpfulness=helpfulness or None,
        situation=situation or None,
        thoughts=thoughts or None,
        need=need or None,
        next_step=next_step or None,
    )
