"""Tests for evals/run_experiment.py's report-building helpers.

_extract_row uses real langsmith EvaluationResult objects (pydantic
BaseModel, attribute access -- .key/.score, NOT dict subscripting) rather
than a hand-rolled stub dict, since a hand-rolled dict stub is exactly what
let the real bug (evals/run_experiment.py used r["key"]/r["score"] and
crashed with "'EvaluationResult' object is not subscriptable" on the first
real smoke test) slip past testing originally.
"""
from langsmith.evaluation.evaluator import EvaluationResult

from evals.run_experiment import _build_report, _extract_row


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
