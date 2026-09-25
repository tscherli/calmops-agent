"""Tests for evals/run_experiment.py's report-building helpers.

_extract_row uses real langsmith EvaluationResult objects (pydantic
BaseModel, attribute access -- .key/.score, NOT dict subscripting) rather
than a hand-rolled stub dict, since a hand-rolled dict stub is exactly what
let the real bug (evals/run_experiment.py used r["key"]/r["score"] and
crashed with "'EvaluationResult' object is not subscriptable" on the first
real smoke test) slip past testing originally.
"""
from langsmith.evaluation.evaluator import EvaluationResult

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
        _row(no_fabrication=True, crisis_gate_correctness=True, expected_behavior_match=False),
        _row("case_02", no_fabrication=True, crisis_gate_correctness=True, trajectory_match=False),
    ]

    assert all(case_passed(r["scores"]) for r in rows)


def test_a_deterministic_failure_still_fails_the_case():
    assert not case_passed({"no_fabrication": False, "expected_behavior_match": True})


def test_a_case_graded_only_by_the_judge_does_not_count_as_a_pass():
    """Nothing deterministic was checked, so there is nothing to have passed."""
    assert not case_passed({"expected_behavior_match": True, "trajectory_match": None})


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
            expected_behavior_match=False,
        ),
        _row(
            "case_02",
            no_fabrication=True,
            crisis_gate_correctness=True,
            write_had_card=None,
            expect_log_rows=False,
            expected_behavior_match=True,
        ),
    ]

    report = _build_report(rows, experiment_url=None, experiment_name="test-exp")

    assert "Expect blocks: 1/2 cases (50%)" in report
    assert "`write_had_card`: 1/1 (100%) -- **PASS**" in report
    assert "## Advisory (not counted in Overall)" in report
    assert "`expected_behavior_match`: 1/2 (50%)" in report
    # The advisory failure above must not have cost case_01 the headline.
    assert "Pass rate: 1/2" in report
