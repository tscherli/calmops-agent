"""Tests for evals/judge.py -- the two narrow questions.

The underlying agentevals callables are stubbed throughout: no Gemini call,
no network. What is checked here is what each question is *shown*, and when
it refuses to ask at all. A judge handed the wrong evidence answers
confidently anyway, and nothing downstream would look wrong.
"""
import json

import pytest

import evals.judge as judge_module
from evals.judge import OFFERED_KEY, SITUATION_KEY, judge_evaluator


@pytest.fixture
def asked(monkeypatch):
    """Records what each question was shown, and answers 1."""
    seen = {}

    def offered(**kwargs):
        seen["offered"] = kwargs
        return {"key": "ignored", "score": 1, "comment": "fine"}

    def situation(**kwargs):
        seen["situation"] = kwargs
        return {"key": "ignored", "score": 1, "comment": "fine"}

    monkeypatch.setattr(judge_module, "_offered_judge", offered)
    monkeypatch.setattr(judge_module, "_situation_judge", situation)
    return seen


@pytest.fixture
def never_asked(monkeypatch):
    def boom(**kwargs):
        raise AssertionError("the judge must not be called here")

    monkeypatch.setattr(judge_module, "_offered_judge", boom)
    monkeypatch.setattr(judge_module, "_situation_judge", boom)


def _lookup(*measure_ids):
    return {
        "role": "tool",
        "name": "find_measures",
        "content": json.dumps([{"measure_id": m, "measure": f"Do {m}."} for m in measure_ids]),
    }


def _outputs(**overrides):
    outputs = {
        "messages": [
            {"role": "human", "content": "wired after a long call"},
            _lookup("stand_up", "breathe"),
            {"role": "ai", "content": "Two from your list: stand up, or breathe.", "tool_calls": None},
        ],
        "approvals": [{"call_id": "w1", "args": {"circumstances": "a long call"}, "decision": "approve"}],
    }
    outputs.update(overrides)
    return outputs


def _scores(result):
    return {r["key"]: r["score"] for r in result["results"]}


def test_both_questions_are_asked_and_keyed_separately(asked):
    result = judge_evaluator(outputs=_outputs(), reference_outputs={})

    assert _scores(result) == {OFFERED_KEY: 1, SITUATION_KEY: 1}


def test_the_offered_question_is_shown_the_list_and_the_reply(asked):
    """It is answering "did the reply go outside this list", so it has to be
    given the list. Handed the trajectory alone it would have to dig the rows
    out of a JSON tool message, and a judge that has to parse is a judge that
    sometimes parses wrong."""
    judge_evaluator(outputs=_outputs(), reference_outputs={})

    shown = asked["offered"]["available_measures"]
    assert "stand_up" in shown and "breathe" in shown
    assert asked["offered"]["agent_reply"] == "Two from your list: stand up, or breathe."


def test_rows_from_several_lookups_all_count_as_the_list(asked):
    """A measure shown after the second lookup came from the list just as
    much as one shown after the first."""
    outputs = _outputs(
        messages=[
            {"role": "human", "content": "wired"},
            _lookup("stand_up"),
            {"role": "human", "content": "something else"},
            _lookup("breathe"),
            {"role": "ai", "content": "Try breathing.", "tool_calls": None},
        ]
    )

    judge_evaluator(outputs=outputs, reference_outputs={})

    # Counted by line, not by substring: the stub's measure text repeats the
    # id, so a substring count says two for a single row.
    lines = asked["offered"]["available_measures"].splitlines()
    assert [line for line in lines if line.startswith("- stand_up:")] == ["- stand_up: Do stand_up."]
    assert any(line.startswith("- breathe:") for line in lines)


def test_the_situation_question_is_shown_the_turns_and_the_card(asked):
    judge_evaluator(outputs=_outputs(), reference_outputs={})

    assert "wired after a long call" in asked["situation"]["user_said"]
    assert asked["situation"]["card_sentence"] == "a long call"


def test_the_situation_question_reads_the_card_not_the_written_row(asked):
    """An edit turn overwrites the row with the user's own wording, so the row
    can look faultless after the agent proposed something invented."""
    outputs = _outputs(
        approvals=[{"call_id": "w1", "args": {"circumstances": "a fortnight of appraisals"}}],
        log_rows=[{"circumstances": "a long call"}],
    )

    judge_evaluator(outputs=outputs, reference_outputs={})

    assert asked["situation"]["card_sentence"] == "a fortnight of appraisals"


# -- when the question refuses to be asked ------------------------------------


def test_a_crisis_case_costs_nothing_to_judge(never_asked):
    """The gate intercepts before any agent turn, so the trajectory is empty.
    The old judge graded that against a paragraph of expected behaviour and
    predictably failed every crisis case, deflating the headline on a
    comparison that was never meaningful."""
    result = judge_evaluator(outputs={"messages": [], "approvals": []}, reference_outputs={})

    assert _scores(result) == {OFFERED_KEY: None, SITUATION_KEY: None}


def test_the_offered_question_is_skipped_when_nothing_was_looked_up(never_asked):
    outputs = _outputs(
        messages=[
            {"role": "human", "content": "morning"},
            {"role": "ai", "content": "Morning.", "tool_calls": None},
        ],
        approvals=[],
    )

    result = judge_evaluator(outputs=outputs, reference_outputs={})

    assert _scores(result)[OFFERED_KEY] is None
    assert "no list to depart from" in result["results"][0]["comment"]


def test_the_offered_question_is_skipped_when_the_agent_said_nothing(never_asked):
    outputs = _outputs(
        messages=[{"role": "human", "content": "wired"}, _lookup("stand_up")], approvals=[]
    )

    assert _scores(judge_evaluator(outputs=outputs, reference_outputs={}))[OFFERED_KEY] is None


def test_the_situation_question_is_skipped_when_no_card_proposed_one(asked):
    outputs = _outputs(approvals=[{"call_id": "w1", "args": {"measure_id": "stand_up"}}])

    result = judge_evaluator(outputs=outputs, reference_outputs={})

    assert _scores(result)[SITUATION_KEY] is None
    assert "situation" not in asked


def test_one_question_skipping_does_not_stop_the_other(asked):
    outputs = _outputs(approvals=[])

    assert _scores(judge_evaluator(outputs=outputs, reference_outputs={})) == {
        OFFERED_KEY: 1,
        SITUATION_KEY: None,
    }
