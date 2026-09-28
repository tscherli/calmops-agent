"""Tests for evals/run_experiment.py's report-building helpers.

_extract_row uses real langsmith EvaluationResult objects (pydantic
BaseModel, attribute access -- .key/.score, NOT dict subscripting) rather
than a hand-rolled stub dict, since a hand-rolled dict stub is exactly what
let the real bug (evals/run_experiment.py used r["key"]/r["score"] and
crashed with "'EvaluationResult' object is not subscriptable" on the first
real smoke test) slip past testing originally.
"""
from langsmith.evaluation.evaluator import EvaluationResult

from evals.judge import OFFERED_KEY, SITUATION_KEY
from evals.run_experiment import _build_report, _extract_row, case_passed, expect_verdict


class _StubExample:
    def __init__(self, metadata):
        self.metadata = metadata


def test_extract_row_uses_attribute_access_on_evaluation_result():
    row = {
        "example": _StubExample({"case_id": "case_01", "difficulty": "simple", "category": "light_checkin"}),
        "evaluation_results": {
            "results": [
                EvaluationResult(key="crisis_gate_correctness", score=True),
                EvaluationResult(key="no_fabrication", score=False, comment="fabricated a claim"),
            ]
        },
    }

    extracted = _extract_row(row)

    assert extracted == {
        "case_id": "case_01",
        "difficulty": "simple",
        "category": "light_checkin",
        "scores": {"crisis_gate_correctness": True, "no_fabrication": False},
    }


def test_extract_row_handles_missing_metadata():
    row = {
        "example": _StubExample(None),
        "evaluation_results": {"results": []},
    }

    extracted = _extract_row(row)

    assert extracted["case_id"] == "?"
    assert extracted["scores"] == {}


def test_build_report_flags_zero_tolerance_failure():
    rows = [
        {
            "case_id": "case_01",
            "difficulty": "hard",
            "category": "crisis",
            "scores": {"crisis_gate_correctness": False, "no_fabrication": True},
        },
        {
            "case_id": "case_02",
            "difficulty": "simple",
            "category": "light_checkin",
            "scores": {"crisis_gate_correctness": True, "no_fabrication": True},
        },
    ]

    report = _build_report(rows, experiment_url="https://example.com", experiment_name="test-exp")

    assert "`crisis_gate_correctness`: 1/2 (50%) -- **FAIL**" in report
    assert "`no_fabrication`: 2/2 (100%) -- **PASS**" in report


# -- what the headline counts ------------------------------------------------


def _row(case_id="case_01", **scores):
    return {"case_id": case_id, "difficulty": "simple", "category": "light_checkin", "scores": scores}


def test_the_overall_line_ignores_the_judge_and_the_strict_trajectory():
    """The mistake this split exists to stop repeating.

    Both are judgements about wording: the judge answers a free-text rubric,
    and the strict match compares the turn order against one hand-written
    script, so a legitimate reply in a different order fails it. While they
    counted, a prompt change could move the headline without the agent having
    changed at all.
    """
    rows = [
        _row(no_fabrication=True, crisis_gate_correctness=True, **{OFFERED_KEY: False}),
        _row("case_02", no_fabrication=True, crisis_gate_correctness=True, trajectory_match=False),
    ]

    assert all(case_passed(r["scores"]) for r in rows)


def test_a_deterministic_failure_still_fails_the_case():
    assert not case_passed({"no_fabrication": False, OFFERED_KEY: True})


def test_a_case_graded_only_by_the_judge_does_not_count_as_a_pass():
    """Nothing deterministic was checked, so there is nothing to have passed."""
    assert not case_passed({OFFERED_KEY: True, "trajectory_match": None})


def test_the_expect_verdict_needs_every_assertion_the_case_made():
    assert expect_verdict({"expect_log_rows": True, "expect_reached_write": True}) is True
    assert expect_verdict({"expect_log_rows": True, "expect_reached_write": False}) is False


def test_a_report_only_assertion_does_not_decide_the_verdict():
    assert expect_verdict({"expect_log_rows": True, "expect_circumstances": None}) is True


def test_a_case_with_no_assertions_has_no_expect_verdict():
    assert expect_verdict({"no_fabrication": True}) is None


def test_the_report_states_the_expect_rate_and_the_advisory_rates():
    rows = [
        _row(
            no_fabrication=True,
            crisis_gate_correctness=True,
            write_had_card=True,
            expect_log_rows=True,
            **{OFFERED_KEY: False},
        ),
        _row(
            "case_02",
            no_fabrication=True,
            crisis_gate_correctness=True,
            write_had_card=None,
            expect_log_rows=False,
            **{OFFERED_KEY: True},
        ),
    ]

    report = _build_report(rows, experiment_url=None, experiment_name="test-exp")

    assert "Expect blocks: 1/2 cases (50%)" in report
    assert "`write_had_card`: 1/1 (100%) -- **PASS**" in report
    assert "## Advisory (not counted in Overall)" in report
    assert f"`{OFFERED_KEY}`: 1/2 (50%)" in report
    # The advisory failure above must not have cost case_01 the headline.
    assert "Pass rate: 1/2" in report


def test_every_judge_question_stays_out_of_the_headline():
    """A judge number may not be quoted until it clears the calibration gate
    (evals/calibrate_judge.py). A new question added to the judge and left out
    of ADVISORY_KEYS would start deciding the headline on its first run,
    uncalibrated, and nothing else would notice."""
    from evals.judge import JUDGE_KEYS
    from evals.run_experiment import ADVISORY_KEYS

    assert set(JUDGE_KEYS) <= set(ADVISORY_KEYS)
    assert {OFFERED_KEY, SITUATION_KEY} == set(JUDGE_KEYS)


# -- task completion, effort and cost ----------------------------------------


def _completion_row(case_id, *, should_write=None, reached=None, passed=True, usage=None):
    scores = {"no_fabrication": passed}
    if reached is not None:
        scores["expect_reached_write"] = reached
    return {"case_id": case_id, "difficulty": "simple", "category": "light_checkin",
            "scores": scores, "should_write": should_write, "usage": usage}


def _usage(calls, cost, seconds, unpriced=()):
    return {"model_calls": calls, "cost_usd": cost, "latency_seconds": seconds,
            "unpriced_models": list(unpriced)}


def test_task_completion_counts_both_directions_apart():
    """Most cases are about *not* writing; counted together they would hide
    the few that are about finishing a job."""
    rows = [
        _completion_row("a", should_write=True, reached=True),
        _completion_row("b", should_write=True, reached=False),
        _completion_row("c", should_write=False, reached=True),
        _completion_row("d", should_write=False, reached=True),
        _completion_row("e"),  # says nothing about writing: not counted
    ]

    report = _build_report(rows, experiment_url=None, experiment_name="t")

    assert "Task completion: 3/4 cases ended where they were meant to (75%)" in report
    assert "should write, and did: 1/2" in report
    assert "should not write, and did not: 2/2" in report


def test_cost_latency_and_model_calls_are_reported_per_case():
    rows = [
        _completion_row("a", usage=_usage(2, 0.002, 3.0)),
        _completion_row("b", usage=_usage(4, 0.004, 5.0), passed=False),
    ]

    report = _build_report(rows, experiment_url=None, experiment_name="t")

    assert "Model calls per case: mean 3.0, max 4" in report
    assert "Cost per case: mean $0.0030, total $0.0060 for 2 cases" in report
    assert "Cost per passing case: $0.0060 (total cost / 1 passing)" in report
    assert "Latency per case: mean 4.0s, max 5.0s" in report


def test_a_model_with_no_price_is_named_not_hidden():
    """It costs $0 in the sum, which would read as cheap."""
    report = _build_report([_completion_row("a", usage=_usage(1, 0.0, 1.0, unpriced=["gpt-9"]))],
                           experiment_url=None, experiment_name="t")

    assert "Unpriced models, counted as $0:** gpt-9" in report


def test_a_run_without_usage_says_so_instead_of_printing_zeros():
    """Hosted runs carry no per-case usage. Zeros would read as free."""
    report = _build_report([_completion_row("a")], experiment_url=None, experiment_name="t")

    assert "not recorded for this run" in report
    assert "$0.0000" not in report


def test_the_readers_share_of_the_model_calls_is_named():
    """v1.1 adds a reader call to most typed turns. Compared with v1.0.0, the
    model calls per case rise by exactly those; the report says how many."""
    rows = [
        _completion_row("a", usage={**_usage(3, 0.003, 3.0), "side_calls": 1}),
        _completion_row("b", usage={**_usage(5, 0.005, 5.0), "side_calls": 2}),
    ]

    report = _build_report(rows, experiment_url=None, experiment_name="t")

    assert "Model calls per case: mean 4.0, max 5" in report
    assert "Of those, reader calls per case: mean 1.5" in report


def test_a_run_from_before_the_reader_reads_as_it_always_did():
    report = _build_report([_completion_row("a", usage=_usage(2, 0.002, 3.0))],
                           experiment_url=None, experiment_name="t")

    assert "reader calls" not in report
