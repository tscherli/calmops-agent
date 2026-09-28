"""The check-in log: what happened, which measure was settled on, and later
whether it helped.

Two kinds of row, one file, append-only:

    check-in   energy, the situation sentence the user confirmed, the measure
    rating     the measure and helpfulness, nothing else

A measure is named by its measure_id, or by the user's own words
(measure_taken) when it is not in the list.

A rating arrives later than the check-in it is about, so it cannot be an edit
of the earlier row without giving up append-only. The duplication of the
situation sentence across several measures tried for the same situation is
accepted (see the plan's parked list).

The rules below are enforced here rather than in a prompt: the model decides
what to write, code decides what a row may contain.
"""
from __future__ import annotations

import csv
import json
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Annotated, Any, Literal

from langchain_core.messages import ToolMessage
from langchain_core.tools import InjectedToolCallId, tool
from langgraph.types import Command

from tools.measures import measure_ids

CHECKIN_COLUMNS = [
    "timestamp",
    "row_id",
    "energy",
    "circumstances",
    "measure_id",
    "measure_taken",
    "helpfulness",
]

BASE_DIR = Path(__file__).resolve().parent.parent
CHECKIN_LOG_PATH = BASE_DIR / "data" / "checkin_log.csv"
PENDING_LOG_PATH = BASE_DIR / "state" / "pending_sheet_log.jsonl"

Energy = Literal["exhausted", "calm", "wired"]
Helpfulness = Literal["helped", "somewhat", "didnt_help"]


def _write_pending_fallback(row: dict[str, Any]) -> None:
    PENDING_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    with PENDING_LOG_PATH.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row) + "\n")


def is_rating_row(row: dict[str, str]) -> bool:
    """A rating says how a measure went; a check-in says what was going on.

    Kept here so the log reader, the digest and the graph cannot disagree
    about which rows are entries and which are verdicts.
    """
    return bool(row.get("helpfulness"))


def _measure_key(measure_id: str, measure_taken: str) -> str:
    """What a measure is called when two rows are compared.

    Its id when it has one; otherwise the user's own words, without case or
    extra spaces, because the model retypes them from one call to the next.
    """
    if measure_id:
        return f"id:{measure_id}"
    return "own:" + " ".join(measure_taken.split()).casefold()


def _row_key(row: dict[str, str]) -> str:
    return _measure_key(row.get("measure_id") or "", row.get("measure_taken") or "")


def _already_rated(key: str, helpfulness: str, rows: list[dict[str, str]]) -> bool:
    """Is the very same verdict for this measure already the latest rating?

    A model logged "instrumental_music_break helped" twice in one conversation,
    which the digest counts as two successes. A *different* verdict is a
    correction and is written: the newer row wins by being later.
    """
    for row in reversed(rows):
        if not is_rating_row(row):
            return False
        if _row_key(row) == key:
            return row.get("helpfulness") == helpfulness
    return False


def append_checkin_row(
    *,
    energy: str = "",
    circumstances: str = "",
    measure_id: str = "",
    measure_taken: str = "",
    helpfulness: str = "",
    path: Path | str | None = None,
) -> dict[str, Any]:
    """Append one row. Never raises: a failure falls back to a local JSONL
    file and returns status="error".

    Enforced here, not in the prompt:
    - a rating row (helpfulness set) carries only the measure -- its id, or
      the user's own words when it has none; energy and the situation belong
      to the check-in row that came before it
    - a rating that arrives with a situation and has no check-in waiting for it
      writes that check-in first, rather than dropping what the user said
    - the same measure is not rated twice in a row
    - no row is written without a measure: a check-in exists to record one the
      user settled on, and a rating says which one it is about
    """
    rating = bool(helpfulness)
    if not (measure_id or measure_taken):
        return {
            "status": "skipped",
            "reason": (
                "no measure named: a rating says which measure it is about"
                if rating
                else "no measure yet: log once the user has settled on one"
            ),
        }
    rows = read_checkin_rows(path) if rating else []
    key = _measure_key(measure_id, measure_taken)
    if rating and _already_rated(key, helpfulness, rows):
        return {
            "status": "skipped",
            "reason": f"{measure_id or measure_taken} already has this verdict; nothing new to record",
        }

    wrote_checkin = ""
    if rating and (energy or circumstances) and not _awaiting_rating(key, rows):
        # The user said how it went and what was going on in one breath. The
        # rating row has no room for the situation, so it gets its own row
        # first -- a hand test lost "feeling very tired at work" this way.
        first = append_checkin_row(
            energy=energy,
            circumstances=circumstances,
            measure_id=measure_id,
            measure_taken=measure_taken,
            path=path,
        )
        if first["status"] != "ok":
            return first
        wrote_checkin = first["row_id"]

    row = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "row_id": str(uuid.uuid4()),
        "energy": "" if rating else energy,
        "circumstances": "" if rating else circumstances,
        "measure_id": measure_id,
        "measure_taken": "" if rating and measure_id else measure_taken,
        "helpfulness": helpfulness,
    }

    log_path = Path(path) if path is not None else CHECKIN_LOG_PATH
    try:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        write_header = not log_path.exists() or log_path.stat().st_size == 0
        with log_path.open("a", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=CHECKIN_COLUMNS)
            if write_header:
                writer.writeheader()
            writer.writerow(row)
        return {
            "status": "ok",
            "row_id": row["row_id"],
            "kind": "rating" if rating else "checkin",
            "checkin_row_id": wrote_checkin,
        }
    except Exception as exc:  # noqa: BLE001 - a logging failure must never end the turn
        _write_pending_fallback(row)
        return {"status": "error", "row_id": row["row_id"], "error": str(exc)}


def measures_awaiting_rating(path: Path | str | None = None) -> list[str]:
    """Every measure from the last run of check-ins that has no rating yet.

    One turn can log more than one: the user settles on two measures and both
    get a row. Each of them is worth its own rating, so rating one leaves the
    other waiting instead of closing the whole batch.

    A rating row written after the run began is what takes a measure out of
    the list. Measures logged in the user's own words carry no measure_id and
    never appear: this list feeds the rating buttons, which need an id to
    attach a rating to.
    """
    return measures_awaiting_rating_in(read_checkin_rows(path))


def _last_run(rows: list[dict[str, str]]) -> tuple[int, int] | None:
    """Where the latest run of check-ins starts and ends; None without one."""
    last = None
    for index, row in enumerate(rows):
        if not is_rating_row(row):
            last = index
    if last is None:
        return None
    start = last
    while start > 0 and not is_rating_row(rows[start - 1]):
        start -= 1
    return start, last


def _awaiting_rating(key: str, rows: list[dict[str, str]]) -> bool:
    """Is a check-in for this one measure in the latest run still unrated?

    The question measures_awaiting_rating_in answers for the buttons, asked
    about one measure -- any of the waiting ones, not only the first -- and
    answered for a measure in the user's own words too.
    """
    run = _last_run(rows)
    if run is None:
        return False
    start, last = run
    rated = {_row_key(row) for row in rows[start:] if is_rating_row(row)}
    return key not in rated and any(_row_key(row) == key for row in rows[start : last + 1])


def measures_awaiting_rating_in(rows: list[dict[str, str]]) -> list[str]:
    """The same answer, from rows already in hand.

    The eval harness hands its expectations a case's rows rather than the
    temporary file they came from, so an assertion about what is still waiting
    has no path to open.
    """
    run = _last_run(rows)
    if run is None:
        return []
    start, last = run
    rated = {row.get("measure_id") for row in rows[start:] if is_rating_row(row)}
    pending: list[str] = []
    for row in rows[start : last + 1]:
        measure_id = row.get("measure_id") or ""
        if measure_id and measure_id not in rated and measure_id not in pending:
            pending.append(measure_id)
    return pending


def measure_awaiting_rating(path: Path | str | None = None) -> str:
    """The next measure waiting for a rating, or "" when none is.

    Lets the rating buttons still mean something after the window is closed
    and reopened -- "come back later and say how it went" is the whole point
    of a rating being its own row, and it would not survive a restart if the
    id only lived in memory.

    Empty when the last check-in named no measure_id (the user described one
    in their own words, and the buttons have no id to attach a rating to) or
    when a rating has already been written since.
    """
    pending = measures_awaiting_rating(path)
    return pending[0] if pending else ""


def read_checkin_rows(path: Path | str | None = None) -> list[dict[str, str]]:
    log_path = Path(path) if path is not None else CHECKIN_LOG_PATH
    if not log_path.exists():
        return []
    with log_path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def make_log_checkin(log_path: Path | str | None = None, measures_path: Path | str | None = None):
    """The logging tool, bound to one log and one measure list."""

    @tool
    def log_checkin(
        tool_call_id: Annotated[str, InjectedToolCallId],
        energy: str = "",
        circumstances: str = "",
        measure_id: str = "",
        measure_taken: str = "",
        helpfulness: str = "",
    ) -> Command:
        """Append one row to the user's check-in log. One row per event.

        Write the check-in row once the user has settled on a measure: their
        energy, the one situation sentence they confirmed, and the measure_id.
        When they later say how it went, call this again with only the
        measure_id and the helpfulness (helped, somewhat or didnt_help).

        circumstances: their situation only - what happened and what it does to
        them, never the measure or how it went.
        measure_taken: something they did that is not in the measure list.
        """
        if measure_id and measure_id not in measure_ids(measures_path):
            return _reply(
                f"Not logged: {measure_id} is not in the measure list. Use measure_taken instead.",
                tool_call_id,
                status="error",
            )

        result = append_checkin_row(
            energy=energy,
            circumstances=circumstances,
            measure_id=measure_id,
            measure_taken=measure_taken,
            helpfulness=helpfulness,
            path=log_path,
        )
        status = result["status"]
        if status == "skipped":
            return _reply(f"Not logged: {result['reason']}.", tool_call_id, status="skipped")
        if status == "error":
            return _reply(
                f"Not logged: the write failed ({result['error']}); saved to the pending file.",
                tool_call_id,
                status="error",
            )

        is_checkin = result["kind"] == "checkin" or bool(result.get("checkin_row_id"))
        update: dict[str, Any] = {"rejections": 0}
        if is_checkin:
            # the signal the clients use to start the next conversation fresh
            update["checkin_logged"] = True
            # What the rating buttons attach to later. Reported by the write
            # rather than read back out of the model's tool call, for the same
            # reason the receipts are: this is what actually reached the file.
            # Empty for a measure the user described in their own words: the
            # buttons need an id to attach a rating to, and a rating typed in
            # words names the measure itself.
            update["last_measure_id"] = measure_id
        return _reply(
            json.dumps({"status": "ok", "row_id": result["row_id"], "kind": result["kind"]}),
            tool_call_id,
            detail=f"{result['kind']} row {result['row_id'][:8]}",
            **update,
        )

    return log_checkin


def _reply(
    text: str,
    tool_call_id: str,
    *,
    status: str = "ok",
    detail: str | None = None,
    **update: Any,
) -> Command:
    """A tool result plus the state it writes.

    The receipt itself is set by the tools node; the hint travels with the
    message so the node can report honestly what happened.
    """
    message = ToolMessage(
        text,
        tool_call_id=tool_call_id,
        name="log_checkin",
        status="error" if status == "error" else "success",
    )
    message.additional_kwargs["receipt"] = {
        "status": status,
        "detail": detail if detail is not None else text[:60],
    }
    return Command(update={**update, "messages": [message]})
