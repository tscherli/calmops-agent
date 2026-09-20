"""Read-only, deterministic aggregation over the check-in log.

Plain typed Python scanning CSV rows -- no query language, no LLM-generated
code touching the file, nothing the model can inject into. This is the same
"bounded tool surface" reasoning that kept logging a single fixed-schema tool
instead of a general code-execution one (see DEVELOPMENT_LOG.md, "Stage 3
revision", on why `create_csv_agent` was rejected).

`analyze_stress_patterns` is not agent-facing at all: scheduler.py calls it to
build the weekly digest. What it reports is now **measures and how they were
rated**, because that is what this log records. The previous basis was tags
and moods, and neither is a column any more.

The two kinds of row are counted separately and deliberately. A check-in says
what was going on and which measure was settled on; a rating says how that
measure went, and arrives later. Counting a rating as an entry would inflate
a quiet week into a busy one, and the "enough data" floor exists precisely so
a thin week is reported as thin.

It never raises: a missing or malformed log reads as "no data", because a
digest failing should never take down anything else.
"""
from __future__ import annotations

import csv
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from tools.checkin_log import CHECKIN_LOG_PATH, is_rating_row
from tools.measures import read_measures

# Two entries is the floor for calling anything a "pattern". Below that the
# honest answer is "not enough data", not a trend extrapolated from one row.
ENOUGH_DATA_MIN_ENTRIES = 2


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


def measure_labels(path: Path | str | None = None) -> dict[str, str]:
    """measure_id -> the sentence the user would recognise.

    The log stores ids because they are stable; a digest that said "sw_02
    came up three times" would be useless to the person reading it.
    """
    return {row["measure_id"]: row.get("measure", "") for row in read_measures(path)}


def _label_for(row: dict[str, str], labels: dict[str, str]) -> str:
    """What to call the measure in this row.

    An id missing from the list falls back to the id itself rather than being
    dropped: the row is real, and silently omitting it would make the counts
    disagree with the file. A measure the user described in their own words
    has no id and is named by that text.
    """
    measure_id = (row.get("measure_id") or "").strip()
    if measure_id:
        return labels.get(measure_id) or measure_id
    return (row.get("measure_taken") or "").strip()


def analyze_stress_patterns(
    window_days: int = 7,
    *,
    path: Path | None = None,
    now: datetime | None = None,
    measures_path: Path | str | None = None,
) -> dict[str, Any]:
    """Aggregate the last `window_days` of check-ins and ratings.

    Pure Python, no LLM and no code execution -- scheduler.py takes this
    summary dict and makes exactly one LLM call to phrase it conversationally.
    `now` is injectable so the windowing is unit-testable without freezing
    the clock, per this project's DI convention.

    Ratings are windowed by their own timestamp, not by the check-in they
    refer to. A rating written this week about last week's measure is this
    week's news, which is what the digest is for.
    """
    reference = now or datetime.now(timezone.utc)
    if reference.tzinfo is None:
        reference = reference.replace(tzinfo=timezone.utc)
    cutoff = reference - timedelta(days=window_days)

    labels = measure_labels(measures_path)
    energy_counts: dict[str, int] = {}
    measure_counts: dict[str, int] = {}
    helpfulness_counts: dict[str, dict[str, int]] = {}
    entry_count = 0
    rating_count = 0

    for row in _read_rows(path):
        timestamp = _parse_timestamp(row.get("timestamp"))
        if timestamp is None or timestamp < cutoff:
            continue
        label = _label_for(row, labels)

        if is_rating_row(row):
            rating_count += 1
            if label:
                verdict = (row.get("helpfulness") or "").strip()
                counts = helpfulness_counts.setdefault(label, {})
                counts[verdict] = counts.get(verdict, 0) + 1
            continue

        entry_count += 1
        energy = (row.get("energy") or "").strip()
        if energy:
            energy_counts[energy] = energy_counts.get(energy, 0) + 1
        if label:
            measure_counts[label] = measure_counts.get(label, 0) + 1

    # Ties break alphabetically so the digest phrasing is deterministic --
    # the same week must not produce a different opener on a re-run.
    top_measures = sorted(measure_counts.items(), key=lambda kv: (-kv[1], kv[0]))

    return {
        "window_days": window_days,
        "entry_count": entry_count,
        "rating_count": rating_count,
        "enough_data": entry_count >= ENOUGH_DATA_MIN_ENTRIES,
        "energy_counts": energy_counts,
        "top_measures": top_measures,
        "helpfulness_counts": helpfulness_counts,
        "best_measures": _best_measures(helpfulness_counts),
    }


def _best_measures(helpfulness_counts: dict[str, dict[str, int]]) -> list[str]:
    """Measures the user's own ratings say worked, best first.

    "Worked" is more helped than didnt_help, counted from the user's own
    verdicts -- not a judgement this code or the model is making. `somewhat`
    deliberately breaks no tie either way: it is the answer people give when
    something was not really the thing, and treating it as a success is how a
    digest starts recommending what did not help.
    """
    scored = []
    for label, counts in helpfulness_counts.items():
        helped = counts.get("helped", 0)
        if helped > counts.get("didnt_help", 0):
            scored.append((helped, label))
    return [label for _, label in sorted(scored, key=lambda pair: (-pair[0], pair[1]))]
