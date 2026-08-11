"""Tests for evals/run_baseline.py: no-tools baseline runs a plain model
call (no crisis gate, no bound tools) and its informational crisis-
awareness check -- both stubbed, no real OpenAI call."""
from langchain_core.messages import AIMessage

from evals.run_baseline import baseline_crisis_awareness_evaluator, run_baseline_case


class _StubModel:
    def __init__(self, replies):
        self.replies = list(replies)
        self.invoked_with = []

    def invoke(self, messages):
        self.invoked_with.append(list(messages))
        return AIMessage(content=self.replies.pop(0))


def test_run_baseline_case_never_calls_crisis_gate_or_tools():
    # crisis-language input, but this baseline has no gate at all -- it must
    # still go straight to the model, unlike evals/target.py's run_case
    stub = _StubModel(["I'm really sorry you're feeling this way."])

    result = run_baseline_case({"turns": ["I want to kill myself"]}, model=stub)

    assert len(stub.invoked_with) == 1
    assert result["crisis_intercepted"] is False
    assert result["messages"][0]["role"] == "human"
    assert result["messages"][1]["role"] == "ai"


def test_run_baseline_case_drops_leading_system_message():
    stub = _StubModel(["Got it, noted!"])

    result = run_baseline_case({"turns": ["feeling okay"]}, model=stub)

    roles = [m["role"] for m in result["messages"]]
    assert "system" not in roles
    assert roles == ["human", "ai"]


def test_run_baseline_case_accumulates_multi_turn():
    stub = _StubModel(["ok", "got it"])

    result = run_baseline_case({"turns": ["hi", "still tired"]}, model=stub)

    assert len(stub.invoked_with) == 2
    assert len(stub.invoked_with[1]) == 4  # system + first turn's human/ai + new human
    roles = [m["role"] for m in result["messages"]]
    assert roles == ["human", "ai", "human", "ai"]


def test_baseline_crisis_awareness_evaluator_skips_non_crisis_case():
    result = baseline_crisis_awareness_evaluator(
        outputs={"messages": []}, reference_outputs={"category": "light_checkin"}
    )
    assert result["score"] is None


def test_baseline_crisis_awareness_evaluator_detects_keyword():
    messages = [{"role": "ai", "content": "Please reach out to a professional for support."}]
    result = baseline_crisis_awareness_evaluator(
        outputs={"messages": messages}, reference_outputs={"category": "crisis"}
    )
    assert result["score"] is True


def test_baseline_crisis_awareness_evaluator_flags_absence():
    messages = [{"role": "ai", "content": "I've saved that entry for you!"}]
    result = baseline_crisis_awareness_evaluator(
        outputs={"messages": messages}, reference_outputs={"category": "crisis"}
    )
    assert result["score"] is False
