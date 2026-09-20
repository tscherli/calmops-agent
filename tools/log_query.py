"""Read-only, deterministic aggregation over the check-in log.

Plain typed Python scanning CSV rows -- no query language, no LLM-generated
code touching the file, nothing the model can inject into. This is the same
"bounded tool surface" reasoning that kept logging a single fixed-schema tool
instead of a general code-execution one (see DEVELOPMENT_LOG.md, "Stage 3
revision", on why `create_csv_agent` was rejected).

`analyze_stress_patterns` is not agent-facing at all: scheduler.py calls it to
build the weekly digest. It still counts the columns the previous log schema
had, so against a check-in log its tag and mood tallies come back empty; the
digest moves to the new basis -- measures and how they were rated -- with the
rest of that subsystem.

It never raises: a missing or malformed log reads as "no data", because a
digest failing should never take down anything else.
"""
from __future__ import annotations

import csv
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from tools.checkin_log import CHECKIN_LOG_PATH

TAG_SEPARATOR = "|"


def parse_tags(raw: str | None) -> list[str]:
    if not raw:
        return []
    return [t.strip() for t in raw.split(TAG_SEPARATOR) if t.strip()]


def _read_rows(path: Path | None = None) -> list[dict[str, str]]:
    log_path = path or CHECKIN_LOG_PATH
    try:
        if not log_path.exists():
            return []
        with log_path.open(newline="", encoding="utf-8") as f:
            return list(csv.DictReader(f))
    except Exception:
        return []


def _parse_timestamp(raw: str | None) -> datetime | None:
    if not raw:
        return None
    try:
        parsed = datetime.fromisoformat(raw)
    except (ValueError, TypeError):
        return None
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=timezone.utc)
    return parsed


def analyze_stress_patterns(
    window_days: int = 7,
    *,
    path: Path | None = None,
    now: datetime | None = None,
) -> dict[str, Any]:
    """Aggregate tag/mood/energy counts over the last `window_days`.

    Pure Python, no LLM and no code execution -- scheduler.py takes this
    summary dict and makes exactly one LLM call to phrase it conversationally.
    `now` is injectable so the windowing is unit-testable without freezing
    the clock, per this project's DI convention.
    """
    reference = now or datetime.now(timezone.utc)
    if reference.tzinfo is None:
        reference = reference.replace(tzinfo=timezone.utc)
    cutoff = reference - timedelta(days=window_days)

    tag_counts: dict[str, int] = {}
    mood_counts: dict[str, int] = {}
    energy_counts: dict[str, int] = {}
    helpful_measures: list[dict[str, str]] = []
    entry_count = 0

    for row in _read_rows(path):
        timestamp = _parse_timestamp(row.get("timestamp"))
        if timestamp is None or timestamp < cutoff:
            continue
        entry_count += 1

        for tag in parse_tags(row.get("tags")):
            tag_counts[tag] = tag_counts.get(tag, 0) + 1
        mood = (row.get("mood") or "").strip()
        if mood:
            mood_counts[mood] = mood_counts.get(mood, 0) + 1
        energy = (row.get("energy") or "").strip()
        if energy:
            energy_counts[energy] = energy_counts.get(energy, 0) + 1

        measure = (row.get("measure_taken") or "").strip()
        if measure:
            helpful_measures.append(
                {"measure_taken": measure, "helpfulness": (row.get("helpfulness") or "").strip()}
            )

    # Two entries is the floor for calling anything a "pattern" -- below that
    # the honest answer is "not enough data", not a trend extrapolated from
    # a single row.
    return {
        "window_days": window_days,
        "entry_count": entry_count,
        "enough_data": entry_count >= 2,
        "top_tags": sorted(tag_counts.items(), key=lambda kv: (-kv[1], kv[0])),
        "mood_counts": mood_counts,
        "energy_counts": energy_counts,
        "measures_taken": helpful_measures,
    }
