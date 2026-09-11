"""Tests for evals/run_local.py -- the LangSmith-free runner.

Fully stubbed: no OpenAI, no LangSmith, no network. What matters here is
that it feeds the target the same inputs a hosted run would, and that one
broken piece can't take down the whole run (the reason this module exists
is that a transient network failure destroyed a paid run's results).
"""
import evals.run_local as run_local_module
from evals.run_local import case_inputs, evaluate_case, run_local

_CASE = {
    "id": "case_01",
    "query": "Doing okay today.",
    "difficulty": "simple",
    "category": "light_checkin",
    "expected_behavior": "Logs directly.",
    "notes": "n/a",
    "reference_trajectory": [{"role": "human"}, {"role": "ai", "tool_calls": None}],
}


def _stub_outputs(*args, **kwargs):
    return {
        "messages": [
            {"role": "human", "content": "Doing okay today."},
            {"role": "ai", "content": "Nice.", "tool_calls": None},
        ],
        "crisis_intercepted": False,
    }


def test_case_inputs_match_the_hosted_examples_shape():
    """A local run and a hosted run must feed the target identical input,
    or the two reports aren't comparable."""
    from evals.create_dataset import case_to_example

    case = {**_CASE, "simulate_write_failure": True, "seed_entries": [{"mood": "bad"}]}

    assert case_inputs(case) == case_to_example(case)["inputs"]


def test_case_inputs_wraps_single_query_as_turns():
    assert case_inputs(_CASE)["turns"] == ["Doing okay today."]
    assert "seed_entries" not in case_inputs(_CASE)


def test_multiturn_case_keeps_its_turns():
    case = {**_CASE, "turns": ["one", "two"]}
    del case["query"]
    assert case_inputs(case)["turns"] == ["one", "two"]


def test_evaluate_case_collects_scores_from_every_evaluator(monkeypatch):
    monkeypatch.setattr(run_local_module, "run_case", _stub_outputs)
    monkeypatch.setattr(
        run_local_module,
        "EVALUATORS",
        (
            lambda outputs, reference_outputs: {"key": "alpha", "score": True},
            lambda outputs, reference_outputs: {"key": "beta", "score": False},
        ),
    )

    row = evaluate_case(_CASE)

    assert row["case_id"] == "case_01"
    assert row["scores"] == {"alpha": True, "beta": False}


def test_a_raising_evaluator_does_not_lose_the_case(monkeypatch):
    """The LLM judge is known to intermittently fail parsing; one bad
    evaluator must not discard an otherwise-good result."""

    def boom(outputs, reference_outputs):
        raise RuntimeError("judge parse failure")

    monkeypatch.setattr(run_local_module, "run_case", _stub_outputs)
    monkeypatch.setattr(
        run_local_module,
        "EVALUATORS",
        (lambda outputs, reference_outputs: {"key": "alpha", "score": True}, boom),
    )

    row = evaluate_case(_CASE)

    assert row["scores"]["alpha"] is True
    assert row["case_id"] == "case_01"


def test_a_failing_target_is_recorded_not_raised(monkeypatch):
    def boom(inputs):
        raise RuntimeError("model unavailable")

    monkeypatch.setattr(run_local_module, "run_case", boom)

    row = evaluate_case(_CASE)

    assert row["scores"] == {"target_error": False}
    assert "model unavailable" in row["error"]


def test_run_local_writes_a_report_and_preserves_dataset_order(monkeypatch, tmp_path):
    monkeypatch.setattr(run_local_module, "run_case", _stub_outputs)
    monkeypatch.setattr(
        run_local_module,
        "EVALUATORS",
        (lambda outputs, reference_outputs: {"key": "alpha", "score": True},),
    )
    monkeypatch.setattr(run_local_module, "_write_report", lambda report: tmp_path / "report.md")

    cases = [{**_CASE, "id": f"case_{i:02d}"} for i in range(5)]
    result = run_local(cases, workers=2)

    assert [r["case_id"] for r in result["rows"]] == [c["id"] for c in cases]
