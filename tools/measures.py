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
import unicodedata
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
GOALS: tuple[str, ...] = ("calm down now", "plan recovery", "change the situation")

# A lookup hands over every measure for the energy and goal, so the model can
# pick the one that fits what the user described.
#
# It used to return the first four rows that had not been shown yet. That made
# recency the selection rule: rows come out of the file in the order they were
# added, so each lookup walked further down the list, and after two turns the
# only measures left were the newest ones -- the ones a web search had just
# appended -- whatever the user had said. It also spent rows unseen, because
# the prompt shows at most two of the four.
#
# The whole slice is affordable while the list is small: the widest one
# (exhausted, calm down now) is about 1100 tokens. The limit below is a safety
# valve, not a selection rule -- a list long enough to hit it needs ranking,
# not a window.
LOOKUP_ROW_LIMIT = 40

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
    energy: str,
    goal: str | None,
    *,
    path: Path | str | None = None,
) -> list[dict[str, str]]:
    """Every measure in the list for this energy and goal.

    Nothing is held back for being shown already: the model has the
    conversation in front of it and knows what it suggested, and a measure that
    fits the situation best should not be hidden because it came up earlier.

    Without a goal, every goal is allowed: someone who does not know yet what
    would help still gets something from their own list, not an invented idea.
    """
    rows = [
        {
            "measure_id": row["measure_id"],
            "type": row["type"],
            "effort": row["effort"],
            "measure": row["measure"],
        }
        for row in read_measures(path)
        if energy in row["fits_energy"].split(";")
        and (goal is None or fits_goal(row, goal))
    ]
    if goal is None:
        rows = _one_goal_at_a_time(rows)
    return rows[:LOOKUP_ROW_LIMIT]


def _one_goal_at_a_time(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    """Take turns between the goals, so a batch spans them.

    In file order the first batch was four quick recoveries: the real list has
    fourteen of them for "exhausted" before any act or plan row.
    """
    by_goal = [[r for r in rows if fits_goal(r, g)] for g in GOALS]
    mixed = []
    for turn in range(max((len(group) for group in by_goal), default=0)):
        mixed.extend(group[turn] for group in by_goal if turn < len(group))
    return mixed


SEARCH_RESULTS = 5

# Words that would match half the list and say nothing about what was asked.
_SEARCH_STOPWORDS = frozenset(
    "the and for with your you this that what into from about list measure measures already "
    "there have does something".split()
)


def search_measures_by_words(words: str, path: Path | str | None = None) -> list[dict[str, str]]:
    """Rows whose id or text share a word stem with the query.

    A stem is the first four letters, so "walking" finds "walk" and "breathe"
    finds "breaths" -- crude on purpose: the question it answers is "is
    something like this already in my list?", and a near miss shown to the user
    is better than a real match missed.
    """
    stems = {
        word[:4]
        for word in re.findall(r"[a-z]+", words.lower())
        if len(word) >= 3 and word not in _SEARCH_STOPWORDS
    }
    if not stems:
        return []
    hits = []
    for row in read_measures(path):
        row_words = re.findall(r"[a-z]+", f"{row['measure_id']} {row['measure']}".lower())
        if any(row_word.startswith(stem) for stem in stems for row_word in row_words):
            hits.append(
                {
                    "measure_id": row["measure_id"],
                    "type": row["type"],
                    "effort": row["effort"],
                    "fits_energy": row["fits_energy"],
                    "measure": row["measure"],
                }
            )
    return hits[:SEARCH_RESULTS]


def append_measure(row: dict[str, str], path: Path | str | None = None) -> None:
    measures_path = Path(path) if path is not None else MEASURES_PATH
    is_new = not measures_path.exists() or measures_path.stat().st_size == 0
    with measures_path.open("a", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=MEASURE_COLUMNS, lineterminator="\n")
        if is_new:
            writer.writeheader()
        writer.writerow(row)


# What a web page's punctuation becomes. The list is a plain ASCII file that
# is read in a terminal and edited by hand; a measure copied from a page
# arrived as "rest your legs up the wall for 5-15 minutes" with an en dash.
_PUNCTUATION = {
    0x2010: "-", 0x2011: "-", 0x2012: "-", 0x2013: "-", 0x2014: "-", 0x2015: "-",
    0x2018: "'", 0x2019: "'", 0x201A: "'", 0x201B: "'",
    0x201C: '"', 0x201D: '"', 0x201E: '"', 0x201F: '"',
    0x2026: "...", 0x00A0: " ", 0x2007: " ", 0x202F: " ", 0x2212: "-",
}


def _clean_measure_text(text: str) -> str:
    """One line of plain ASCII, no control characters, capped.

    Text reaching this function may come from a web page the model read, so it
    is treated as data to be sanitised, not as something to trust.
    """
    flat = " ".join(str(text).translate(_PUNCTUATION).split())
    ascii_only = unicodedata.normalize("NFKD", flat).encode("ascii", "ignore").decode("ascii")
    return " ".join(ascii_only.split())[:MAX_MEASURE_CHARS]


def make_find_measures(path: Path | str | None = None):
    """The lookup tool, bound to one measure list (tests and evals pass their own)."""

    @tool
    def find_measures(
        state: Annotated[dict, InjectedState],
        tool_call_id: Annotated[str, InjectedToolCallId],
    ) -> Command:
        """Look up measures from the user's own list for their recorded energy and goal.

        Returns every measure in their list for the energy and goal that are
        recorded, so pick the ones that fit what they described rather than the
        first rows of the result. If no goal is recorded yet, it returns
        measures for their energy across every goal. An empty list means their
        list holds nothing at all for this; say so instead of inventing one.
        """
        energy, goal = state.get("energy"), state.get("goal")
        offered = list(state.get("offered") or [])
        if not energy:
            return _reply(
                "Energy is not recorded yet; ask the user first.",
                tool_call_id,
                "find_measures",
                status="skipped",
            )
        found = matching_measures(energy, goal or None, path=path)
        detail = f"{len(found)} measures" if found else "nothing in the list for this"
        if not goal:
            detail += " (any goal)"
        return _reply(
            as_tool_content(found),
            tool_call_id,
            "find_measures",
            detail=detail,
            offered=offered + [m["measure_id"] for m in found],
        )

    return find_measures


def make_search_measures(path: Path | str | None = None):
    """The search tool, bound to one measure list (tests and evals pass their own)."""

    @tool
    def search_measures(words: str) -> list[dict[str, str]]:
        """Search the user's own measure list by a few words.

        Use it to answer whether something is already in their list, for
        example "walk" or "breathing". Returns the matching rows; an empty list
        means nothing in their list matches. It only reads: it changes nothing.

        words: a few words for what you are looking for.
        """
        return search_measures_by_words(words, path)

    return search_measures


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
        from_user: true when it is the user's own idea. A source_url wins:
        an idea from a web page is saved as coming from the web.
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
        # A link decides it: an idea with a page behind it came from the web,
        # whatever the model says. Web ideas that reached the debate station
        # were added with from_user true -- its prompt assumes every idea there
        # is the user's -- and saved as "user" with their link thrown away.
        url = source_url.strip()
        source = f"web:{url}" if url else ("user" if from_user else "agent")
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
