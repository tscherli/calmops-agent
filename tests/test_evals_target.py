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


def test_eval_runs_never_write_to_the_real_mood_log(monkeypatch, tmp_path):
    """A run would otherwise inject 28 fabricated entries into the user's
    personal log -- and since the redesign classifies every entry, those
    fixtures would also enter known_tags and nudge future real
    classifications toward test vocabulary."""
    import tools.csv_log as csv_log_module

    real_log = tmp_path / "real_mood_log.csv"
    monkeypatch.setattr(csv_log_module, "LOG_PATH", real_log)

    captured = {}

    def fake_build_agent(*, log_path=None, **kwargs):
        captured["log_path"] = log_path
        return _StubGraph(tool_call=True)

    monkeypatch.setattr("evals.target.build_agent", fake_build_agent)

    run_case({"turns": ["feeling good today"]})

    assert captured["log_path"] is not None, "run_case must pass an explicit log path"
    assert captured["log_path"] != real_log
    assert not real_log.exists()


def test_simulate_write_failure_flag_produces_a_genuinely_unwritable_path(monkeypatch):
    """hard_tool_failure_10 claimed for several stages that the harness
    simulated a write failure. It never did -- the case passed on an
    ordinary successful write. This asserts the simulation is real."""
    from tools.csv_log import append_mood_row

    captured = {}

    def fake_build_agent(*, log_path=None, **kwargs):
        captured["log_path"] = log_path
        return _StubGraph(tool_call=True)

    monkeypatch.setattr("evals.target.build_agent", fake_build_agent)

    run_case({"turns": ["standard day"], "simulate_write_failure": True})

    result = append_mood_row(mood="good", circumstances="x", path=captured["log_path"])
    assert result["status"] == "error", "the simulated path must actually fail to write"


def test_seed_entries_prepopulate_an_isolated_log(monkeypatch):
    """History cases ("have I dealt with this before?", overstating a loose
    match, instruction-like text inside a past entry) are unreachable
    without seeded history -- every case otherwise runs on an empty log."""
    import csv

    captured = {}

    def fake_build_agent(*, log_path=None, **kwargs):
        captured["log_path"] = log_path
        return _StubGraph()

    monkeypatch.setattr("evals.target.build_agent", fake_build_agent)

    run_case(
        {
            "turns": ["crunch week again"],
            "seed_entries": [
                {
                    "mood": "bad",
                    "circumstances": "three deadlines at once",
                    "measure_taken": "asked for an extension",
                    "helpfulness": "helped a lot",
                    "tags": ["multiple_deadlines"],
                }
            ],
        }
    )

    seeded = captured["log_path"]
    with seeded.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    assert len(rows) == 1
    assert rows[0]["circumstances"] == "three deadlines at once"
    assert rows[0]["measure_taken"] == "asked for an extension"
    assert rows[0]["tags"] == "multiple_deadlines"


def test_seeded_entries_are_findable_by_the_history_lookup(monkeypatch):
    """Seeding is only useful if find_similar_entries can actually match it,
    so this asserts the join works end to end rather than just the write."""
    from tools.log_query import find_similar_entries

    captured = {}
    monkeypatch.setattr(
        "evals.target.build_agent",
        lambda *, log_path=None, **kw: captured.setdefault("log_path", log_path) and None or _StubGraph(),
    )

    run_case(
        {
            "turns": ["again"],
            "seed_entries": [
                {"mood": "bad", "circumstances": "deadline crunch", "tags": ["multiple_deadlines"]}
            ],
        }
    )

    found = find_similar_entries(["multiple_deadlines"], path=captured["log_path"])
    assert [e["circumstances"] for e in found] == ["deadline crunch"]


def test_explicit_log_path_is_respected(monkeypatch, tmp_path):
    captured = {}

    def fake_build_agent(*, log_path=None, **kwargs):
        captured["log_path"] = log_path
        return _StubGraph()

    monkeypatch.setattr("evals.target.build_agent", fake_build_agent)
    chosen = tmp_path / "seeded.csv"

    run_case({"turns": ["hi"]}, log_path=chosen)

    assert captured["log_path"] == chosen


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
