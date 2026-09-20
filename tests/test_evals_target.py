"""Tests for evals/target.py -- the harness that drives one eval case.

The guarantee this module exists for: **the harness runs the same system the
application runs.** It drifted from app.py once, by carrying conversation
state differently, and the result was a full eval run of plausible-looking
numbers that measured nothing (DEVELOPMENT_LOG.md, stage 26). Both sides now
go through next_turn_state, and that is asserted here rather than trusted.

The second guarantee is data safety: an eval run must not touch the user's
own check-in log or grow their measure list.
"""
import csv
from pathlib import Path

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from evals.target import run_case, serialize_messages
from tools.measures import MEASURE_COLUMNS


class _StubGraph:
    """Records every payload it is handed, and echoes a final state back."""

    def __init__(self, reply_content="Stub reply", tool_call=False, final_state=None):
        self.reply_content = reply_content
        self.tool_call = tool_call
        self.final_state = final_state or {}
        self.calls: list[dict] = []

    def invoke(self, payload):
        self.calls.append(payload)
        messages = list(payload["messages"])
        if self.tool_call:
            messages.append(
                AIMessage(
                    content="",
                    tool_calls=[{"name": "log_checkin", "args": {"energy": "wired"}, "id": "1"}],
                )
            )
            messages.append(
                ToolMessage(content="ok", name="log_checkin", status="success", tool_call_id="1")
            )
        messages.append(AIMessage(content=self.reply_content))
        return {**self.final_state, "messages": messages}


# -- the crisis gate runs before the agent, exactly as app.py does -----------


def test_crisis_case_never_invokes_agent():
    stub = _StubGraph()

    result = run_case({"turns": ["I want to kill myself"]}, agent_graph=stub)

    assert result["crisis_intercepted"] is True
    assert stub.calls == []


def test_multi_turn_stops_at_crisis_mid_conversation():
    stub = _StubGraph()

    result = run_case(
        {"turns": ["busy week", "I want to end my life", "never mind"]}, agent_graph=stub
    )

    assert result["crisis_intercepted"] is True
    assert len(stub.calls) == 1


def test_non_crisis_case_invokes_agent_and_serializes_messages():
    stub = _StubGraph(reply_content="Tell me more.", tool_call=True)

    result = run_case({"turns": ["quiet morning"]}, agent_graph=stub)

    roles = [m["role"] for m in result["messages"]]
    assert roles == ["human", "ai", "tool", "ai"]
    assert result["messages"][1]["tool_calls"] == [{"name": "log_checkin", "args": {"energy": "wired"}}]
    assert result["crisis_intercepted"] is False


# -- conversation state is carried the way the application carries it --------


def test_the_station_fields_are_carried_between_turns():
    """Passing only `messages` would reset energy and goal every turn, so the
    agent would re-ask for them forever and the suite would grade a system
    nobody runs."""
    stub = _StubGraph(final_state={"energy": "wired", "goal": "calm down now",
                                   "offered": ["short_walk"], "step": "follow_up"})

    run_case({"turns": ["I'm wired", "what else?"]}, agent_graph=stub)

    assert stub.calls[0].get("energy") is None
    assert stub.calls[1]["energy"] == "wired"
    assert stub.calls[1]["offered"] == ["short_walk"]


def test_context_resets_once_a_check_in_completes():
    """The durable memory is the CSV, so the next conversation starts cold --
    and `offered` must go with the transcript, or the next sentence lands in
    follow_up with nothing to follow up on."""
    stub = _StubGraph(final_state={"complete": True, "energy": "wired",
                                   "offered": ["short_walk"]})

    run_case({"turns": ["that helped", "hello again"]}, agent_graph=stub)

    assert stub.calls[1]["energy"] is None
    assert stub.calls[1]["offered"] == []
    assert len(stub.calls[1]["messages"]) == 1


def test_the_transcript_keeps_growing_across_a_reset():
    """`messages` is only the agent's working context; the evaluators judge
    the whole case. Everything said before a reset must still be graded."""
    stub = _StubGraph(reply_content="ok", final_state={"complete": True})

    result = run_case({"turns": ["first", "second"]}, agent_graph=stub)

    contents = [m["content"] for m in result["messages"]]
    assert contents == ["first", "ok", "second", "ok"]


def test_a_turn_can_carry_what_a_button_click_would_put_in_the_payload():
    """Three of the UI's controls write straight into the payload rather than
    going through the model. A case that cannot express a click cannot cover
    the rejection path at all."""
    stub = _StubGraph()

    run_case(
        {"turns": ["I'm wired", {"text": "no", "rejected": True}]},
        agent_graph=stub,
    )

    assert stub.calls[1]["rejected"] is True
    assert stub.calls[1]["messages"][-1].content == "no"


# -- data safety -------------------------------------------------------------


def test_eval_runs_never_write_to_the_real_check_in_log(monkeypatch):
    """A run would otherwise inject dozens of fabricated rows into the user's
    own log -- which the weekly digest then reads."""
    captured = {}

    def fake_build_agent(**kwargs):
        captured.update(kwargs)
        return _StubGraph()

    monkeypatch.setattr("evals.target.build_agent", fake_build_agent)
    run_case({"turns": ["quiet morning"]})

    log_path = Path(captured["log_path"])
    assert log_path.name == "checkin_log.csv"
    assert "data" not in log_path.parts
    assert not log_path.exists()


def test_eval_runs_get_their_own_copy_of_the_measure_list(monkeypatch):
    """add_measure writes to this file and find_measures reads it back in
    every later conversation. A shared one would let an eval run grow the
    user's real list, permanently and silently."""
    captured = {}

    def fake_build_agent(**kwargs):
        captured.update(kwargs)
        return _StubGraph()

    monkeypatch.setattr("evals.target.build_agent", fake_build_agent)
    run_case({"turns": ["quiet morning"]})

    measures_path = Path(captured["measures_path"])
    assert measures_path.name == "measures.csv"
    assert measures_path.parent != Path("tools").resolve()
    with measures_path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert rows and list(rows[0]) == MEASURE_COLUMNS


def test_each_case_gets_a_fresh_measure_copy(monkeypatch):
    """Cases must not see each other's added measures, or results depend on
    execution order."""
    seen = []

    def fake_build_agent(**kwargs):
        seen.append(kwargs["measures_path"])
        return _StubGraph()

    monkeypatch.setattr("evals.target.build_agent", fake_build_agent)
    run_case({"turns": ["a"]})
    run_case({"turns": ["b"]})

    assert seen[0] != seen[1]


def test_simulate_write_failure_flag_produces_a_genuinely_unwritable_path(monkeypatch):
    """The honesty-under-failure case has to fail for real, not be mocked."""
    from tools.checkin_log import append_checkin_row

    captured = {}

    def fake_build_agent(**kwargs):
        captured.update(kwargs)
        return _StubGraph()

    monkeypatch.setattr("evals.target.build_agent", fake_build_agent)
    run_case({"turns": ["rough day"], "simulate_write_failure": True})

    result = append_checkin_row(
        energy="wired", circumstances="x", measure_id="m01", path=captured["log_path"]
    )
    assert result["status"] == "error"


def test_explicit_log_path_is_respected(monkeypatch, tmp_path):
    captured = {}

    def fake_build_agent(**kwargs):
        captured.update(kwargs)
        return _StubGraph()

    monkeypatch.setattr("evals.target.build_agent", fake_build_agent)
    chosen = tmp_path / "chosen.csv"
    run_case({"turns": ["quiet morning"]}, log_path=chosen)

    assert captured["log_path"] == chosen


# -- serialization -----------------------------------------------------------


def test_serialize_messages_role_vocabulary():
    """The vocabulary has to match dataset.jsonl's hand-authored reference
    trajectories exactly; agentevals only compares strings."""
    messages = [
        HumanMessage(content="hi"),
        AIMessage(content="", tool_calls=[{"name": "log_checkin", "args": {}, "id": "1"}]),
        ToolMessage(content="ok", name="log_checkin", status="success", tool_call_id="1"),
        AIMessage(content="done"),
    ]

    result = serialize_messages(messages)

    assert [m["role"] for m in result] == ["human", "ai", "tool", "ai"]
    assert result[1]["tool_calls"] == [{"name": "log_checkin", "args": {}}]
    assert result[2]["name"] == "log_checkin"
