"""The curated measure list, and the two tools that read and grow it.

The list is data the user owns: `tools/measures.csv`, editable by hand. Code
filters it; the model only chooses among what the filter returned. That split
is why a suggestion can never be a measure the user does not have.

Columns
    measure_id   short lowercase id; log rows refer to it, so never rename one
                 that has been logged
    type         recover (feel better) or act (change the cause)
    measure      one sentence, plain text
    fits_energy  ";"-separated subset of exhausted/calm/wired
    effort       minutes / hours / days
    source       stretchly:... | user | agent | web:<url>

Stretchly rows are BSD 2-Clause; the licence and notice live in
third_party/stretchly/ and must travel with the data.
"""
from __future__ import annotations

import csv
import re
from pathlib import Path
from typing import Annotated, Any, Literal

from langchain_core.tools import InjectedToolCallId, tool
from langgraph.prebuilt import InjectedState
from langgraph.types import Command

from agent.common import as_tool_content

BASE_DIR = Path(__file__).resolve().parent.parent
MEASURES_PATH = BASE_DIR / "tools" / "measures.csv"

MEASURE_COLUMNS = ["measure_id", "type", "measure", "fits_energy", "effort", "source"]

Energy = Literal["exhausted", "calm", "wired"]
Goal = Literal["calm down now", "plan recovery", "change the situation"]
MeasureType = Literal["recover", "act"]
Effort = Literal["minutes", "hours", "days"]

# One lookup returns a handful, not the whole list: the model picks two from
# what it sees, and a later lookup in the same conversation returns the next
# few rather than repeating them.
FIND_MEASURES_BATCH = 4

MEASURE_ID_PATTERN = re.compile(r"^[a-z0-9_]{3,40}$")
MAX_MEASURE_CHARS = 200


def read_measures(path: Path | str | None = None) -> list[dict[str, str]]:
    """Every row of the measure list, or an empty list if the file is missing."""
    measures_path = Path(path) if path is not None else MEASURES_PATH
    if not measures_path.exists():
        return []
    with measures_path.open(encoding="utf-8", newline="") as handle:
        return [row for row in csv.DictReader(handle) if row.get("measure_id")]


def measure_ids(path: Path | str | None = None) -> set[str]:
    return {row["measure_id"] for row in read_measures(path)}


def fits_goal(row: dict[str, str], goal: str) -> bool:
    """The goal is a filter over two existing columns, not a column of its own.

    calm down now      -> something to do in minutes that helps you recover
    plan recovery      -> recovery that takes hours or days
    change the situation -> the measures that act on the cause
    """
    if goal == "calm down now":
        return row["type"] == "recover" and row["effort"] == "minutes"
    if goal == "plan recovery":
        return row["type"] == "recover" and row["effort"] in {"hours", "days"}
    return row["type"] == "act"


def matching_measures(
    energy: str, goal: str, *, exclude: set[str] | None = None, path: Path | str | None = None
) -> list[dict[str, str]]:
    """Measures for this energy and goal that have not been shown yet."""
    seen = exclude or set()
    return [
        {
            "measure_id": row["measure_id"],
            "type": row["type"],
            "effort": row["effort"],
            "measure": row["measure"],
        }
        for row in read_measures(path)
        if energy in row["fits_energy"].split(";")
        and fits_goal(row, goal)
        and row["measure_id"] not in seen
    ][:FIND_MEASURES_BATCH]


def append_measure(row: dict[str, str], path: Path | str | None = None) -> None:
    measures_path = Path(path) if path is not None else MEASURES_PATH
    is_new = not measures_path.exists() or measures_path.stat().st_size == 0
    with measures_path.open("a", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=MEASURE_COLUMNS, lineterminator="\n")
        if is_new:
            writer.writeheader()
        writer.writerow(row)


def _clean_measure_text(text: str) -> str:
    """One line, no control characters, capped.

    Text reaching this function may come from a web page the model read, so it
    is treated as data to be sanitised, not as something to trust.
    """
    flat = " ".join(str(text).split())
    return flat[:MAX_MEASURE_CHARS]


def make_find_measures(path: Path | str | None = None):
    """The lookup tool, bound to one measure list (tests and evals pass their own)."""

    @tool
    def find_measures(
        state: Annotated[dict, InjectedState],
        tool_call_id: Annotated[str, InjectedToolCallId],
    ) -> Command:
        """Look up measures from the user's own list for their recorded energy and goal.

        Returns up to four measures that have not been shown in this
        conversation yet. An empty list means the list has nothing left for
        this combination; say so and search the web instead of inventing one.
        """
        energy, goal = state.get("energy"), state.get("goal")
        offered = list(state.get("offered") or [])
        if not (energy and goal):
            return _reply(
                "Energy and goal are not recorded yet; ask the user first.",
                tool_call_id,
                "find_measures",
                status="skipped",
            )
        found = matching_measures(energy, goal, exclude=set(offered), path=path)
        detail = f"{len(found)} measures" if found else "no measures left for this"
        return _reply(
            as_tool_content(found),
            tool_call_id,
            "find_measures",
            detail=detail,
            offered=offered + [m["measure_id"] for m in found],
        )

    return find_measures


def make_add_measure(path: Path | str | None = None):
    """The tool that grows the list. Validates before writing, because the text
    can come from a web page and the row is read back in every later
    conversation."""

    @tool
    def add_measure(
        measure_id: str,
        measure: str,
        measure_type: MeasureType,
        fits_energy: list[Energy],
        effort: Effort,
        tool_call_id: Annotated[str, InjectedToolCallId],
        source_url: str = "",
        from_user: bool = False,
    ) -> Command:
        """Add one new measure to the user's list. Only when the user asked for it.

        measure_id: short lowercase name with underscores, e.g. "box_breathing".
        measure: one short sentence in your own words.
        measure_type: "recover" if it helps the user feel better, "act" if it
        changes the stressful situation.
        fits_energy: the energy levels it suits.
        effort: how long it takes.
        source_url: the page the idea came from; empty if it is your own idea.
        from_user: true when it is the user's own idea.
        """
        clean_id = str(measure_id).strip().lower()
        if not MEASURE_ID_PATTERN.match(clean_id):
            return _reply(
                f"Not added: {measure_id!r} is not a valid id (lowercase letters, digits, underscores).",
                tool_call_id,
                "add_measure",
                status="error",
            )
        if clean_id in measure_ids(path):
            return _reply(
                f"Not added: {clean_id} already exists. Pick another id or reuse that measure.",
                tool_call_id,
                "add_measure",
                status="error",
            )
        text = _clean_measure_text(measure)
        if not text:
            return _reply(
                "Not added: the measure text is empty.", tool_call_id, "add_measure", status="error"
            )
        source = "user" if from_user else (f"web:{source_url.strip()}" if source_url.strip() else "agent")
        append_measure(
            {
                "measure_id": clean_id,
                "type": measure_type,
                "measure": text,
                "fits_energy": ";".join(dict.fromkeys(fits_energy)),
                "effort": effort,
                "source": source,
            },
            path,
        )
        return _reply(
            f"Added {clean_id} to the measure list.",
            tool_call_id,
            "add_measure",
            detail=f"{clean_id} ({source})",
        )

    return add_measure


def _reply(
    text: str,
    tool_call_id: str,
    name: str,
    *,
    status: str = "ok",
    detail: str | None = None,
    **update: Any,
) -> Command:
    """A tool result plus the state it writes.

    No receipt here on purpose: receipts are set by the tools node, in one
    place the model cannot reach. The status and detail travel as part of the
    message text the node reads back.
    """
    from langchain_core.messages import ToolMessage

    message = ToolMessage(
        text,
        tool_call_id=tool_call_id,
        name=name,
        status="error" if status == "error" else "success",
    )
    message.additional_kwargs["receipt"] = {
        "status": status,
        "detail": detail if detail is not None else text[:60],
    }
    return Command(update={**update, "messages": [message]})
