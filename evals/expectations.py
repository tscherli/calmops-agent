"""The `expect` block: what a case asserts about what actually happened.

Every other evaluator in this suite grades one fixed property of every case.
An expect block is the opposite -- it is the case's own claim, written next to
the turns that produce it, and it is graded against the machine-set record of
the run rather than against the reply text. `tool_events`, the rows in the
case's log, the cards the user was shown and the ids added to the case's
measure list are all set by code the model cannot reach, which is what makes
these assertions worth having.

One feedback key per field, never one verdict for the block. A case asserting
five things and failing as a single red line tells you nothing about which of
the five broke, and the remaining four are never exercised.

Row assertions are scored None when the case never reached a write. "Did not
get there" and "wrote the wrong thing" are different failures with different
fixes, and collapsing them hides which one happened.
"""
from __future__ import annotations

import re

from evals.outputs import card_sentences, lookup_results, user_turns
from tools.checkin_log import measures_awaiting_rating_in
from tools.measures import matching_measures

# Every field a case may use. An unknown one fails the run rather than being
# skipped: a typo in an assertion is a case that silently asserts nothing.
EXPECT_FIELDS = frozenset(
    {
        "tools_called",
        "tools_not_called",
        "lookup_rows",
        "log_rows",
        "circumstances",
        "pending_ratings_at_end",
        "measures_added",
        "reached_write",
    }
)

# The circumstances check false-fails on paraphrase -- "slept badly" against a
# user who wrote "bad sleep". It runs report-only until the first three runs
# have shown how often that happens; until then it states a verdict in its
# comment and scores None, so it cannot fail a case on a defect of its own.
CIRCUMSTANCES_REPORT_ONLY = True

# Deliberately short. Every word this list excuses is a word the agent may put
# in the user's mouth for free, so it holds function words only -- nothing
# that could carry a fact about the situation.
_FUNCTION_WORDS = frozenset(
    {
        "that", "this", "with", "from", "have", "been", "were", "they", "them",
        "their", "there", "then", "than", "when", "what", "will", "would",
        "could", "should", "about", "into", "your", "yours", "just", "also",
        "some", "because", "which", "while", "after", "before", "today",
        "and", "the", "for", "but", "was", "are", "you", "her", "him", "his",
    }
)

_WORD = re.compile(r"[a-z0-9]+")


def _feedback(key: str, score: bool | None, comment: str | None = None) -> dict:
    result: dict = {"key": f"expect_{key}", "score": score}
    if comment:
        result["comment"] = comment
    return result


def _tools_used(outputs: dict) -> list[str]:
    return [event.get("tool", "") for event in (outputs.get("tool_events") or [])]


def _user_text(outputs: dict) -> str:
    return " ".join(turn.lower() for turn in user_turns(outputs))


def _row_matches(expected: dict, actual: dict) -> bool:
    """A row matches on the fields the case names, and only those.

    Cases assert energy, measure_id and helpfulness -- the fields with a small
    known vocabulary. Free text is not asserted here; the circumstances
    sentence is checked on the card, which is the only place an edit turn has
    not already overwritten.
    """
    return all(str(actual.get(field, "")) == str(value) for field, value in expected.items())


def _check_tools_called(outputs: dict, wanted: list) -> dict:
    used = _tools_used(outputs)
    missing = [name for name in wanted if name not in used]
    return _feedback(
        "tools_called",
        not missing,
        None
        if not missing
        else f"never called: {', '.join(missing)} (called: {', '.join(used) or 'nothing'})",
    )


def _check_tools_not_called(outputs: dict, unwanted: list) -> dict:
    used = _tools_used(outputs)
    called = [name for name in unwanted if name in used]
    return _feedback(
        "tools_not_called",
        not called,
        None if not called else f"called anyway: {', '.join(called)}",
    )


def _check_lookup_rows(outputs: dict, spec: dict) -> dict:
    """The only assertion that catches a lookup handing back a window.

    A window over the list in file order is how the measures that fitted best
    stayed invisible: the rows existed, the model never saw them. Nothing in
    the reply shows that, and no other field here would go red for it -- the
    agent suggests something plausible either way.

    The case states the energy and goal because find_measures takes neither as
    an argument (it reads them off the state), so the call itself records
    nothing about which slice was asked for.
    """
    energy = spec.get("energy")
    goal = spec.get("goal") or None
    if not energy:
        return _feedback(
            "lookup_rows", False, "the case must state the energy its slice was looked up for"
        )
    results = lookup_results(outputs)
    if not results:
        return _feedback("lookup_rows", None, "the case never looked a measure up")
    # Never fall back to the shipped list. matching_measures(path=None) reads
    # the user's own measures.csv, so a missing path here would quietly grade
    # the run against a file it never opened -- and pass or fail on rows that
    # were not in play.
    if not outputs.get("measures_path"):
        return _feedback(
            "lookup_rows", False, "the run did not report which measure list the case used"
        )
    expected = {
        row["measure_id"]
        for row in matching_measures(energy, goal, path=outputs.get("measures_path"))
    }
    for index, rows in enumerate(results):
        got = {row.get("measure_id") for row in rows if isinstance(row, dict)}
        if got != expected:
            missing = sorted(expected - got)
            extra = sorted(got - expected)
            return _feedback(
                "lookup_rows",
                False,
                f"lookup {index + 1} was not the whole slice: {len(got)} of {len(expected)} rows"
                + (f", missing {', '.join(missing[:5])}" if missing else "")
                + (f", unexpected {', '.join(extra[:5])}" if extra else ""),
            )
    return _feedback("lookup_rows", True)


def _check_log_rows(outputs: dict, spec: dict) -> dict:
    """Rows by shape, unordered, each expected row matched to its own actual row.

    Unordered because two measures settled in one turn are written in whatever
    order the model names them, and that order is not a guarantee. By shape
    because the row's free text is the model's sentence, not a fixed string.
    """
    rows = outputs.get("log_rows") or []
    wanted_rows = spec.get("rows") or []
    count = spec.get("count")
    if (wanted_rows or count) and not outputs.get("reached_write") and not rows:
        return _feedback(
            "log_rows", None, "no write happened in this case -- nothing was logged to check"
        )
    if count is not None and len(rows) != count:
        return _feedback("log_rows", False, f"expected {count} rows in the log, found {len(rows)}")
    remaining = list(rows)
    for expected in wanted_rows:
        match = next((row for row in remaining if _row_matches(expected, row)), None)
        if match is None:
            return _feedback("log_rows", False, f"no row matching {expected} among {len(rows)} rows")
        remaining.remove(match)
    return _feedback("log_rows", True)


def _check_circumstances(outputs: dict, mode: str) -> dict:
    """The sentence the user was shown, built only from words the user used.

    Checked on the card rather than on the row: an edit turn overwrites the
    row with the user's own wording, so a row can look faultless after the
    agent proposed something the user never said.
    """
    if mode != "empty_or_user_words":
        return _feedback("circumstances", False, f"unknown circumstances mode: {mode!r}")
    said = _user_text(outputs)
    invented: list[str] = []
    for sentence in card_sentences(outputs):
        for word in _WORD.findall(sentence.lower()):
            if len(word) >= 4 and word not in _FUNCTION_WORDS and word not in said:
                invented.append(word)
    verdict = not invented
    comment = None if verdict else f"words no turn used: {', '.join(sorted(set(invented)))}"
    if CIRCUMSTANCES_REPORT_ONLY:
        passed = "passed" if verdict else "failed"
        return _feedback(
            "circumstances",
            None,
            f"report-only: would have {passed}" + (f" -- {comment}" if comment else ""),
        )
    return _feedback("circumstances", verdict, comment)


def _check_pending_ratings(outputs: dict, wanted: int) -> dict:
    """How many measures are still waiting to be rated, not which ones.

    Which measures the agent settled on is its own judgement and moves run to
    run; that every logged measure gets its own rating prompt does not.
    """
    pending = measures_awaiting_rating_in(outputs.get("log_rows") or [])
    return _feedback(
        "pending_ratings_at_end",
        len(pending) == wanted,
        None
        if len(pending) == wanted
        else f"expected {wanted} waiting, found {len(pending)}: {pending}",
    )


def _check_measures_added(outputs: dict, wanted) -> dict:
    added = outputs.get("measures_added") or []
    if isinstance(wanted, int):
        return _feedback(
            "measures_added",
            len(added) == wanted,
            None if len(added) == wanted else f"expected {wanted} new measures, got {added}",
        )
    if sorted(wanted) == sorted(added):
        return _feedback("measures_added", True)
    return _feedback("measures_added", False, f"expected {sorted(wanted)}, got {added}")


def _check_reached_write(outputs: dict, wanted: bool) -> dict:
    reached = bool(outputs.get("reached_write"))
    return _feedback(
        "reached_write",
        reached == bool(wanted),
        None if reached == bool(wanted) else f"expected reached_write={bool(wanted)}, got {reached}",
    )


_CHECKS = {
    "tools_called": _check_tools_called,
    "tools_not_called": _check_tools_not_called,
    "lookup_rows": _check_lookup_rows,
    "log_rows": _check_log_rows,
    "circumstances": _check_circumstances,
    "pending_ratings_at_end": _check_pending_ratings,
    "measures_added": _check_measures_added,
    "reached_write": _check_reached_write,
}


def check_expectations(outputs: dict, expect: dict | None) -> list[dict]:
    """One feedback entry per field the case asserted."""
    if not expect:
        return []
    results = []
    for field, value in expect.items():
        if field not in _CHECKS:
            results.append(_feedback("schema", False, f"unknown expect field: {field!r}"))
            continue
        results.append(_CHECKS[field](outputs, value))
    return results


def expectation_evaluator(outputs: dict, reference_outputs: dict) -> dict:
    """The (outputs, reference_outputs) form LangSmith binds by parameter name.

    Returns several feedback entries at once -- `{"results": [...]}` is the
    shape evaluate() accepts from a multi-key evaluator, and run_local.py
    unwraps the same shape.
    """
    return {"results": check_expectations(outputs, reference_outputs.get("expect"))}


__all__ = [
    "CIRCUMSTANCES_REPORT_ONLY",
    "EXPECT_FIELDS",
    "check_expectations",
    "expectation_evaluator",
]
