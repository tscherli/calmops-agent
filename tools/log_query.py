"""Read-only, deterministic lookups over data/mood_log.csv.

Plain typed Python functions scanning CSV rows -- no query language, no
LLM-generated code touching the file, nothing the model can inject into.
This is the same "bounded tool surface" reasoning that kept `log_mood` a
single fixed-schema tool instead of a general code-execution one (see
DEVELOPMENT_LOG.md, "Stage 3 revision", on why `create_csv_agent` was
rejected): the semantic work happens once, up front, in tools/classify.py;
everything here is a group-by over the strings that call already stored.

Two consumers, two different shapes:
- `find_similar_entries` is exposed to the agent as a real bound tool
  (`find_similar_entries_tool`), AND called deterministically by the graph
  right after an entry is logged. Both are intended -- see the redesign
  plan's resolved open decisions.
- `analyze_stress_patterns` is not agent-facing at all; scheduler.py calls
  it to build the weekly digest.

Neither function raises: a missing or malformed log reads as "no data",
because a history lookup failing should never take down a check-in.
"""
from __future__ import annotations

import csv
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from langchain_core.tools import tool

from tools.csv_log import LOG_PATH, parse_tags

# Fields echoed back for a matched past entry. Deliberately a subset --
# `thoughts` and `need` are the most personal free text in a row, and a
# similar-situations lookup doesn't need to re-surface them to answer
# "what did I do about this last time."
SIMILAR_ENTRY_FIELDS = (
    "timestamp",
    "mood",
    "energy",
    "circumstances",
    "measure_taken",
    "helpfulness",
    "next_step",
)


def _read_rows(path: Path | None = None) -> list[dict[str, str]]:
    log_path = path or LOG_PATH
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


def known_tags(limit: int = 40, *, path: Path | None = None) -> list[str]:
    """Tags already used in this log, most frequent first.

    Fed back into tools/classify.py so the classifier can reuse established
    vocabulary instead of reinventing it. Without this the open vocabulary
    drifts: the same input produced both `workload_spike` and
    `high_workload` across two runs, which doesn't break the similarity
    lookup outright (it matches on any overlap) but does fragment
    `analyze_stress_patterns`' counts -- two entries about one problem
    tallying as two separate one-off concepts, which is exactly what the
    weekly digest is supposed to notice.

    Frequency-ordered rather than recency-ordered: the point is to surface
    the *established* vocabulary, and a tag used six times is a better
    consolidation target than yesterday's one-off. Ties break
    alphabetically so the prompt stays deterministic.
    """
    counts: dict[str, int] = {}
    for row in _read_rows(path):
        for tag in parse_tags(row.get("tags")):
            counts[tag] = counts.get(tag, 0) + 1
    ordered = sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))
    return [tag for tag, _ in ordered[: max(0, limit)]]


def find_similar_entries(
    tags: list[str],
    limit: int = 5,
    *,
    path: Path | None = None,
    exclude_row_id: str | None = None,
) -> list[dict[str, Any]]:
    """Return past entries sharing at least one tag with `tags`.

    Ranked by number of overlapping tags first, then most recent. Returns
    [] when there's no log, no tags to match on, or nothing overlaps --
    an empty result is a real, reportable answer ("this hasn't come up
    before"), not an error to paper over.

    `exclude_row_id` keeps an entry from matching itself. The graph looks up
    history immediately *after* appending the new row, so without this the
    just-written entry comes back as its own precedent and the agent would
    report a "pattern" invented from one row -- caught by a test, not by
    inspection.
    """
    wanted = {t.strip().lower() for t in (tags or []) if t and t.strip()}
    if not wanted:
        return []

    matches: list[dict[str, Any]] = []
    for row in _read_rows(path):
        if exclude_row_id and row.get("row_id") == exclude_row_id:
            continue
        row_tags = {t.lower() for t in parse_tags(row.get("tags"))}
        overlap = wanted & row_tags
        if not overlap:
            continue
        entry = {field: (row.get(field) or "") for field in SIMILAR_ENTRY_FIELDS}
        entry["tags"] = sorted(row_tags)
        entry["matched_tags"] = sorted(overlap)
        matches.append(entry)

    matches.sort(
        key=lambda e: (
            len(e["matched_tags"]),
            _parse_timestamp(e["timestamp"]) or datetime.min.replace(tzinfo=timezone.utc),
        ),
        reverse=True,
    )
    return matches[: max(0, limit)]


@tool("find_similar_entries")
def find_similar_entries_tool(tags: list[str], limit: int = 5) -> list[dict[str, Any]]:
    """Look up past log entries about a similar kind of situation.

    tags: semantic tags to match on (e.g. ["deadline_pressure", "context_switching"]).
    Pass the tags of the entry just logged, or tags describing what the user is
    describing now.
    limit: how many past entries to return, most relevant first.

    Returns a list of past entries, each with what the user did about it
    (measure_taken) and how well that worked (helpfulness). An empty list means
    nothing similar is on record -- report that honestly rather than implying a
    pattern exists.
    """
    return find_similar_entries(tags, limit)


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
