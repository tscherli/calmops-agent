"""Tests for evals/judge.py's wrapper -- stubs the underlying agentevals
judge callable so no real LLM call happens; only checks that
judge_evaluator wires (outputs, reference_outputs) into the expected
kwargs for the underlying judge."""
import evals.judge as judge_module


def test_judge_evaluator_passes_messages_and_expected_behavior(monkeypatch):
    captured = {}

    def _fake_judge(*, outputs, expected_behavior):
        captured["outputs"] = outputs
        captured["expected_behavior"] = expected_behavior
        return {"key": "expected_behavior_match", "score": 1}

    monkeypatch.setattr(judge_module, "_judge", _fake_judge)

    messages = [{"role": "human", "content": "hi"}]
    result = judge_module.judge_evaluator(
        outputs={"messages": messages},
        reference_outputs={"expected_behavior": "Should ask a clarifying question."},
    )

    assert captured["outputs"] == messages
    assert captured["expected_behavior"] == "Should ask a clarifying question."
    assert result == {"key": "expected_behavior_match", "score": 1}


def test_judge_evaluator_handles_missing_expected_behavior(monkeypatch):
    captured = {}

    def _fake_judge(*, outputs, expected_behavior):
        captured["expected_behavior"] = expected_behavior
        return {"key": "expected_behavior_match", "score": 0}

    monkeypatch.setattr(judge_module, "_judge", _fake_judge)

    judge_module.judge_evaluator(outputs={"messages": [{"role": "human", "content": "hi"}]}, reference_outputs={})

    assert captured["expected_behavior"] == ""


def test_judge_evaluator_skips_empty_trajectory_without_calling_judge(monkeypatch):
    # A crisis case: the gate intercepts before any agent turn, so
    # outputs["messages"] is always []. Grading that against expected_behavior
    # text describing crisis-resource behavior predictably "fails" every
    # time -- caught by the stage-10 real smoke test's report showing 0/6
    # passed for the crisis category despite crisis_gate_evaluator being
    # 100% correct. Must not call the real (costly) judge at all.
    def _fail_if_called(**kwargs):
        raise AssertionError("must not call the judge on an empty trajectory")

    monkeypatch.setattr(judge_module, "_judge", _fail_if_called)

    result = judge_module.judge_evaluator(
        outputs={"messages": []},
        reference_outputs={"expected_behavior": "Should return crisis resources.", "category": "crisis"},
    )

    assert result["key"] == "expected_behavior_match"
    assert result["score"] is None
