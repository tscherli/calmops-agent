"""Tests for evals/target.py: crisis gate short-circuit, message
serialization, and multi-turn accumulation -- all stubbed, no real OpenAI
call (same dependency-injection pattern as tests/test_app.py)."""
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from evals.target import run_case, serialize_messages


class _PoisonGraph:
    def invoke(self, *args, **kwargs):
        raise AssertionError("must never invoke the agent graph")


class _StubGraph:
    def __init__(self, reply_content="Stub reply", tool_call=False):
        self.reply_content = reply_content
        self.tool_call = tool_call
        self.calls = []

    def invoke(self, payload):
        self.calls.append(payload)
        messages = list(payload["messages"])
        if self.tool_call:
            messages.append(
                AIMessage(content="", tool_calls=[{"name": "log_mood", "args": {"valence": "good"}, "id": "1"}])
            )
            messages.append(ToolMessage(content="ok", name="log_mood", status="success", tool_call_id="1"))
        messages.append(AIMessage(content=self.reply_content))
        return {"messages": messages}


def test_crisis_case_never_invokes_agent():
    result = run_case({"turns": ["I want to kill myself"]}, agent_graph=_PoisonGraph())

    assert result["crisis_intercepted"] is True
    assert result["messages"] == []
    assert result["matched_pattern"] is not None


def test_non_crisis_case_invokes_agent_and_serializes_messages():
    stub = _StubGraph(reply_content="Sounds good.", tool_call=True)

    result = run_case({"turns": ["feeling good today"]}, agent_graph=stub)

    assert result["crisis_intercepted"] is False
    roles = [m["role"] for m in result["messages"]]
    assert roles == ["human", "ai", "tool", "ai"]
    assert result["messages"][1]["tool_calls"] == [{"name": "log_mood", "args": {"valence": "good"}}]
    assert result["messages"][2]["status"] == "success"
    assert result["messages"][-1]["content"] == "Sounds good."


def test_multi_turn_accumulates_across_calls():
    stub = _StubGraph(reply_content="ok")

    result = run_case({"turns": ["hi", "still not great"]}, agent_graph=stub)

    assert len(stub.calls) == 2
    # second call's messages must include the first turn's accumulated history
    assert len(stub.calls[1]["messages"]) == 3  # human, ai, + new human
    roles = [m["role"] for m in result["messages"]]
    assert roles == ["human", "ai", "human", "ai"]


def test_multi_turn_stops_at_crisis_mid_conversation():
    stub = _StubGraph(reply_content="ok")

    result = run_case({"turns": ["hi", "I want to end it all"]}, agent_graph=stub)

    assert result["crisis_intercepted"] is True
    assert len(stub.calls) == 1  # second turn never reached the agent
    roles = [m["role"] for m in result["messages"]]
    assert roles == ["human", "ai"]  # only the first turn's accumulated messages


def test_serialize_messages_role_vocabulary():
    messages = [
        HumanMessage(content="hi"),
        AIMessage(content="", tool_calls=[{"name": "log_mood", "args": {}, "id": "1"}]),
        ToolMessage(content="ok", name="log_mood", status="success", tool_call_id="1"),
        AIMessage(content="done"),
    ]

    result = serialize_messages(messages)

    assert [m["role"] for m in result] == ["human", "ai", "tool", "ai"]
    assert result[1]["tool_calls"] == [{"name": "log_mood", "args": {}}]
    assert result[3]["tool_calls"] is None
