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

from evals.target import build_eval_graph, run_case, serialize_messages
from tests.conftest import StubChatModel
from tools.checkin_log import append_checkin_row
from tools.measures import MEASURE_COLUMNS


class _StubGraph:
    """Records every payload it is handed, and echoes a final state back."""

    def __init__(self, reply_content="Stub reply", tool_call=False, final_state=None):
        self.reply_content = reply_content
        self.tool_call = tool_call
        self.final_state = final_state or {}
        self.calls: list[dict] = []

    def invoke(self, payload, config=None):   # a compiled graph takes a config
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


def test_eval_runs_never_write_to_the_real_check_in_log(monkeypatch, tmp_path):
    """A run would otherwise inject dozens of fabricated rows into the user's
    own log -- which the weekly digest then reads."""
    captured = {}

    def fake_build_agent(**kwargs):
        captured.update(kwargs)
        return _StubGraph()

    monkeypatch.setattr("evals.target.build_agent", fake_build_agent)
    run_case({"turns": ["quiet morning"]}, runs_log=tmp_path / "runs.jsonl")

    log_path = Path(captured["log_path"])
    assert log_path.name == "checkin_log.csv"
    assert "data" not in log_path.parts
    assert not log_path.exists()


def test_eval_runs_get_their_own_copy_of_the_measure_list(monkeypatch, tmp_path):
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


def test_a_failed_eval_write_never_reaches_the_real_state_folder(monkeypatch):
    """A check-in that cannot be written falls back to a module-global path,
    which ignores the case's own log path -- so the write-failure case put its
    synthetic situation into the user's state/pending_sheet_log.jsonl on every
    sweep. The conftest fixture guards tests; this guards real eval runs,
    which is why it checks for the eval's own temp prefix rather than for
    "somewhere temporary" (the test fixture's path is temporary too)."""
    import tools.checkin_log as checkin_log

    monkeypatch.setattr("evals.target.build_agent", lambda **kwargs: _StubGraph())
    run_case({"turns": ["quiet morning"]})

    assert "joblog-evalpending-" in str(checkin_log.PENDING_LOG_PATH)
    assert "state" not in checkin_log.PENDING_LOG_PATH.parts


def test_an_n1_case_builds_the_agent_without_the_tool_and_says_so(monkeypatch):
    """The N-1 baseline must reach build_agent, not stop at the runner -- and
    each trajectory must carry what it was built without, so one taken out of
    its file cannot pass for the full agent."""
    captured = {}

    def fake_build_agent(**kwargs):
        captured.update(kwargs)
        return _StubGraph()

    monkeypatch.setattr("evals.target.build_agent", fake_build_agent)
    result = run_case({"turns": ["quiet morning"]}, exclude_tools=["find_measures"])

    assert list(captured["exclude_tools"]) == ["find_measures"]
    assert result["excluded_tools"] == ["find_measures"]


def test_a_full_case_records_that_nothing_was_excluded(monkeypatch):
    monkeypatch.setattr("evals.target.build_agent", lambda **kwargs: _StubGraph())

    assert run_case({"turns": ["quiet morning"]})["excluded_tools"] == []


def test_a_crisis_exit_carries_the_exclusion_too():
    """The crisis gate returns early; the result builder is shared precisely
    so neither exit drops a field."""
    result = run_case({"turns": ["I want to kill myself"]}, agent_graph=_StubGraph(),
                      exclude_tools=["find_measures"])

    assert result["excluded_tools"] == ["find_measures"]


def test_each_case_gets_a_fresh_measure_copy(monkeypatch, tmp_path):
    """Cases must not see each other's added measures, or results depend on
    execution order."""
    seen = []

    def fake_build_agent(**kwargs):
        seen.append(kwargs["measures_path"])
        return _StubGraph()

    monkeypatch.setattr("evals.target.build_agent", fake_build_agent)
    run_case({"turns": ["a"]}, runs_log=tmp_path / "runs.jsonl")
    run_case({"turns": ["b"]}, runs_log=tmp_path / "runs.jsonl")

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

# -- writes, which a case could not reach at all before -----------------------


class _PausingGraph:
    """A graph that asks for approval once, the way a real write does."""

    def __init__(self):
        self.resumed_with = None
        self.paused = False

    def invoke(self, payload, config=None):
        if not self.paused:
            self.paused = True
            return {
                "__interrupt__": [
                    type("I", (), {"value": {"action_requests": [{
                        "id": "call-1",
                        "tool": "log_checkin",
                        "description": 'Log: "a loud day" (measure: short_walk)',
                        "args": {"circumstances": "a loud day", "measure_id": "short_walk"},
                    }]}})()
                ]
            }
        self.resumed_with = payload
        return {"messages": [AIMessage(content="done")], "tool_events": []}


def test_an_approval_card_is_answered_so_the_turn_can_finish():
    """Without a checkpointer the graph refuses every write, so no case could
    reach add_measure or log_checkin: the suite graded an agent that was not
    allowed to do half its job. The harness now answers the card itself."""
    graph = _PausingGraph()

    result = run_case({"turns": ["I'll try the walk"]}, agent_graph=graph)

    assert graph.resumed_with.resume["decisions"] == [{"id": "call-1", "type": "approve"}]
    assert result["approvals"] == [{
        "call_id": "call-1",
        "tool": "log_checkin",
        "description": 'Log: "a loud day" (measure: short_walk)',
        # The args are what makes the card gradeable: an edit turn overwrites
        # the row, so a rule about the sentence the user was shown can only be
        # checked against this.
        "args": {"circumstances": "a loud day", "measure_id": "short_walk"},
        "decision": "approve",
    }]


def test_a_case_can_reject_the_card():
    """"Reject" is a behaviour worth grading: the agent must accept it and not
    retry. A case says so per turn, and the default stays approve."""
    graph = _PausingGraph()

    run_case({"turns": [{"text": "I'll try the walk", "approval": "reject"}]}, agent_graph=graph)

    assert graph.resumed_with.resume["decisions"] == [{"id": "call-1", "type": "reject"}]


def test_an_edit_lands_in_the_argument_the_tool_says_is_editable():
    """The case gives the wording; which field it belongs to is the tool's
    business (EDITABLE_ARG), so a case never has to know the schema."""
    graph = _PausingGraph()

    run_case(
        {"turns": [{"text": "I'll try the walk", "approval": "edit",
                    "edit_text": "back-to-back calls all morning"}]},
        agent_graph=graph,
    )

    decision = graph.resumed_with.resume["decisions"][0]
    assert decision["type"] == "edit"
    assert decision["args"]["circumstances"] == "back-to-back calls all morning"
    assert decision["args"]["measure_id"] == "short_walk"


def test_an_edit_with_nothing_typed_is_an_approval():
    """What the clients do when the edit box is left alone."""
    graph = _PausingGraph()

    run_case({"turns": [{"text": "ok", "approval": "edit"}]}, agent_graph=graph)

    assert graph.resumed_with.resume["decisions"][0]["type"] == "approve"


def test_the_decision_keys_never_reach_the_graph_as_state():
    """`approval` and `edit_text` steer the harness; they are not agent state,
    and a stray key in the payload would be a field the app never sets."""
    graph = _StubGraph()

    run_case({"turns": [{"text": "hello", "approval": "reject", "edit_text": "x"}]},
             agent_graph=graph)

    assert "approval" not in graph.calls[0]
    assert "edit_text" not in graph.calls[0]

def test_the_harness_builds_a_graph_that_can_pause(tmp_path):
    """Directly: no checkpointer, no pause, and tools_node refuses every write
    for lack of a decision. Every other test here uses a stand-in graph, so
    this is the one that notices if the real one loses its checkpointer."""
    graph = build_eval_graph(
        tmp_path / "log.csv", tmp_path / "measures.csv",
        model=StubChatModel(responses=[AIMessage(content="ok")]),
    )

    assert graph.checkpointer is not None


def test_a_case_reaches_a_real_write_through_the_real_graph(tmp_path):
    """End to end on the graph the app runs: the model asks to log, the graph
    pauses, the harness approves, and the row exists. Before the checkpointer
    this case ended with "Not done: the user did not approve this"."""
    log_path = tmp_path / "log.csv"
    measures_path = tmp_path / "measures.csv"
    with measures_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(MEASURE_COLUMNS)
        writer.writerow(("short_walk", "recover", "Walk for five minutes.",
                         "wired", "minutes", "user"))
    model = StubChatModel(responses=[
        AIMessage(content="", tool_calls=[{
            "name": "log_checkin",
            "args": {"energy": "wired", "circumstances": "a loud day", "measure_id": "short_walk"},
            "id": "call-1",
        }]),
        AIMessage(content="Noted."),
    ])
    graph = build_eval_graph(log_path, measures_path, model=model)

    result = run_case(
        {"turns": [{"text": "I'll try the walk", "energy": "wired",
                    "goal": "calm down now", "situation": "a loud day",
                    "step": "follow_up", "offered": ["short_walk"]}]},
        # The same paths the graph was built on: run_case reads the rows back
        # from what it was given, so an injected graph on different files would
        # report an empty log while the write really happened.
        agent_graph=graph, log_path=log_path, measures_path=measures_path,
    )

    rows = list(csv.DictReader(log_path.open(encoding="utf-8", newline="")))
    assert [r["measure_id"] for r in rows] == ["short_walk"]
    assert rows[0]["circumstances"] == "a loud day"
    assert [a["decision"] for a in result["approvals"]] == ["approve"]
    # The other half of reached_write: a case that did get there says so, or a
    # hardcoded False would satisfy every test that only checks the negative.
    assert result["reached_write"] is True
    assert [r["measure_id"] for r in result["log_rows"]] == ["short_walk"]


def test_a_case_starts_with_the_rows_it_seeds(tmp_path):
    """A rating case needs a check-in to rate, and an injection case needs a
    row already in the list. `seed_entries` was accepted by the dataset loader
    and silently ignored by the harness until now."""
    log_path = tmp_path / "log.csv"
    measures_path = tmp_path / "measures.csv"
    with measures_path.open("w", encoding="utf-8", newline="") as handle:
        csv.writer(handle, lineterminator="\n").writerow(MEASURE_COLUMNS)

    run_case(
        {
            "turns": ["hello"],
            "seed_entries": [{"energy": "wired", "circumstances": "a loud day",
                              "measure_id": "planted_one"}],
            "seed_measures": [{"measure_id": "planted_one", "type": "recover",
                               "measure": "Walk for five minutes.", "fits_energy": "wired",
                               "effort": "minutes", "source": "user"}],
        },
        agent_graph=_StubGraph(), log_path=log_path, measures_path=measures_path,
    )

    rows = list(csv.DictReader(log_path.open(encoding="utf-8", newline="")))
    assert [r["measure_id"] for r in rows] == ["planted_one"]
    assert "planted_one" in {
        r["measure_id"] for r in csv.DictReader(measures_path.open(encoding="utf-8", newline=""))
    }


def test_a_planted_measure_is_cleaned_like_one_the_app_wrote(tmp_path):
    """Planting text the app would have rewritten tests a row that cannot
    exist. add_measure folds punctuation; so does seeding."""
    from evals.target import seed_measures

    measures_path = tmp_path / "measures.csv"
    with measures_path.open("w", encoding="utf-8", newline="") as handle:
        csv.writer(handle, lineterminator="\n").writerow(MEASURE_COLUMNS)

    seed_measures(
        [{"measure_id": "dashed", "type": "recover", "fits_energy": "wired",
          "effort": "minutes", "source": "user",
          "measure": "Rest for 5" + chr(8211) + "15 minutes."}],
        measures_path,
    )

    row = list(csv.DictReader(measures_path.open(encoding="utf-8", newline="")))[0]
    assert row["measure"] == "Rest for 5-15 minutes."


def test_the_result_carries_the_rows_and_whether_a_write_happened(tmp_path):
    """Three of the four bugs the hand tests found were visible only in what
    got written, so the rows themselves have to reach the evaluators."""
    log_path = tmp_path / "log.csv"
    append_checkin_row(energy="wired", circumstances="seeded", measure_id="m1", path=log_path)

    result = run_case({"turns": ["hello"]}, agent_graph=_StubGraph(), log_path=log_path,
                      measures_path=tmp_path / "measures.csv")

    assert [r["measure_id"] for r in result["log_rows"]] == ["m1"]
    assert result["measures_added"] == []
    assert result["reached_write"] is False


def test_a_case_records_no_cost_unless_the_runner_asks(tmp_path):
    """Writing was the default once, and every test that drove a case left a
    record in evals/results/. The runner decides; run_case does not.

    Asserted as "this call did not touch it" rather than "the file does not
    exist": the file legitimately appears the first time the suite is run for
    real, and a test that goes red because someone ran the suite is a test
    that gets muted rather than read.
    """
    from evals.target import EVAL_RUNS_LOG

    before = EVAL_RUNS_LOG.read_bytes() if EVAL_RUNS_LOG.exists() else None

    run_case({"turns": ["hello"]}, agent_graph=_StubGraph(),
             log_path=tmp_path / "log.csv", measures_path=tmp_path / "m.csv")

    after = EVAL_RUNS_LOG.read_bytes() if EVAL_RUNS_LOG.exists() else None
    assert after == before


def test_the_eval_cost_record_is_not_the_apps(tmp_path):
    """observability/runs.jsonl is the application's per-turn record, and the
    figures quoted in the dev log come from averaging it. A suite run appending
    dozens of rows would bury them."""
    from evals.target import EVAL_RUNS_LOG
    from observability.tracing import RUNS_LOG_PATH

    assert EVAL_RUNS_LOG != RUNS_LOG_PATH
    assert EVAL_RUNS_LOG.parent.name == "results"

    runs_log = tmp_path / "runs.jsonl"
    run_case({"turns": ["hello"]}, agent_graph=_StubGraph(), runs_log=runs_log,
             log_path=tmp_path / "log.csv", measures_path=tmp_path / "m.csv")

    assert runs_log.exists()


def test_a_crisis_after_a_write_still_reports_the_write(tmp_path):
    """A crisis can arrive mid-conversation, after turns that already wrote a
    row. The early return used to carry four fields fewer than the normal one,
    so those rows, the card that approved them and reached_write all vanished
    at the gate -- and a case about that sequence could not be graded at all.
    """
    log_path = tmp_path / "log.csv"
    measures_path = tmp_path / "measures.csv"
    with measures_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(MEASURE_COLUMNS)
        writer.writerow(("short_walk", "recover", "Walk for five minutes.", "wired", "minutes", "user"))
    model = StubChatModel(responses=[
        AIMessage(content="", tool_calls=[{
            "name": "log_checkin",
            "args": {"energy": "wired", "circumstances": "a loud day", "measure_id": "short_walk"},
            "id": "call-1",
        }]),
        AIMessage(content="Noted."),
    ])
    graph = build_eval_graph(log_path, measures_path, model=model)

    result = run_case(
        {"turns": [
            {"text": "I'll try the walk", "energy": "wired", "goal": "calm down now",
             "situation": "a loud day", "step": "follow_up", "offered": ["short_walk"]},
            "I want to kill myself.",
        ]},
        agent_graph=graph, log_path=log_path, measures_path=measures_path,
    )

    assert result["crisis_intercepted"] is True
    assert result["matched_pattern"]
    assert [r["measure_id"] for r in result["log_rows"]] == ["short_walk"]
    assert result["reached_write"] is True
    assert [a["decision"] for a in result["approvals"]] == ["approve"]
    assert result["measures_path"] == str(measures_path)


def test_a_crisis_on_the_first_turn_reports_empty_rows_not_missing_ones(tmp_path):
    """Nothing ran, so there is nothing to report -- but the keys still have to
    be there, or every expectation reading them scores against a missing field
    instead of an empty one."""
    result = run_case({"turns": ["I want to kill myself."]}, log_path=tmp_path / "log.csv")

    assert result["crisis_intercepted"] is True
    assert result["log_rows"] == []
    assert result["reached_write"] is False
    assert result["approvals"] == []
    assert "measures_path" in result


# -- what a case cost ----------------------------------------------------------


class _PricedGraph(_StubGraph):
    """Replies the way a real model call does: with token usage and a model name."""

    def invoke(self, payload, config=None):
        self.calls.append(payload)
        reply = AIMessage(
            content="ok",
            usage_metadata={"input_tokens": 1000, "output_tokens": 100, "total_tokens": 1100},
            response_metadata={"model_name": self.reply_content},
        )
        return {**self.final_state, "messages": [*payload["messages"], reply]}


def test_a_case_adds_up_its_own_cost_and_work():
    """eval-runs.jsonl has the same figures per turn but no case id, so they
    could never be read back per case. The case carries its own total."""
    from observability.tracing import summarize_turn

    one_turn = summarize_turn(
        [], [AIMessage(content="ok", usage_metadata={"input_tokens": 1000, "output_tokens": 100,
                                                      "total_tokens": 1100},
                       response_metadata={"model_name": "gpt-5-mini"})],
        model="gpt-5-mini", latency_seconds=0.0,
    ).estimated_cost_usd

    result = run_case({"turns": ["first", "second"]}, agent_graph=_PricedGraph(reply_content="gpt-5-mini"))

    usage = result["usage"]
    assert usage["user_turns"] == 2 and usage["model_calls"] == 2
    assert usage["input_tokens"] == 2000 and usage["output_tokens"] == 200
    assert one_turn > 0 and usage["cost_usd"] == round(2 * one_turn, 6)
    assert usage["unpriced_models"] == []


def test_usage_is_there_without_a_runs_log(tmp_path):
    """The per-turn file is opt-in; the case's own total is not."""
    result = run_case({"turns": ["hi"]}, agent_graph=_PricedGraph(reply_content="gpt-5-mini"))

    assert result["usage"]["model_calls"] == 1
    assert not (tmp_path / "runs.jsonl").exists()


def test_an_unpriced_model_is_named_in_the_usage():
    result = run_case({"turns": ["hi"]}, agent_graph=_PricedGraph(reply_content="gpt-9-unknown"))

    assert result["usage"]["unpriced_models"] == ["gpt-9-unknown"]
    assert result["usage"]["cost_usd"] == 0.0


def test_a_crisis_exit_carries_usage_too():
    result = run_case({"turns": ["busy week", "I want to end my life"]},
                      agent_graph=_PricedGraph(reply_content="gpt-5-mini"))

    assert result["crisis_intercepted"] is True
    assert result["usage"]["model_calls"] == 1 and result["usage"]["user_turns"] == 1
    # The cost of the turn before the crisis, not just the counts: the counts
    # come from the transcript and would survive the crisis exit dropping usage.
    assert result["usage"]["cost_usd"] > 0 and result["usage"]["input_tokens"] == 1000


class _ReadGraph(_PricedGraph):
    """A turn that was read first: the reader's call leaves no message in the
    conversation, only a side call in the state."""

    def invoke(self, payload, config=None):
        result = super().invoke(payload, config)
        reader_call = AIMessage(
            content='{"wants": ["unclear"]}',
            usage_metadata={"input_tokens": 400, "output_tokens": 20, "total_tokens": 420},
            response_metadata={"model_name": self.reply_content},
        )
        return {**result, "side_calls": [reader_call]}


def test_the_readers_call_is_counted_in_the_case(monkeypatch):
    """Without it a read turn looks one call and 420 tokens cheaper than it was:
    the transcript only holds the station's reply."""
    result = run_case({"turns": ["hi"]}, agent_graph=_ReadGraph(reply_content="gpt-5-mini"))

    usage = result["usage"]
    assert usage["model_calls"] == 2 and usage["side_calls"] == 1
    assert usage["input_tokens"] == 1400 and usage["output_tokens"] == 120
    unread = run_case({"turns": ["hi"]}, agent_graph=_PricedGraph(reply_content="gpt-5-mini"))["usage"]
    assert usage["cost_usd"] > unread["cost_usd"]


def test_the_real_graph_hands_the_readers_call_to_the_case(tmp_path):
    """End to end through build_eval_graph: the node keeps the call, the case counts it."""
    from tests.conftest import StubChatModel

    stations = StubChatModel(responses=[AIMessage(content="ok")] * 5)
    reader = StubChatModel(responses=[AIMessage(content='{"wants": ["unclear"]}')])
    graph = build_eval_graph(tmp_path / "log.csv", tmp_path / "measures.csv", model=stations, reader_model=reader)

    result = run_case({"turns": ["what is this app for"]}, agent_graph=graph,
                      log_path=tmp_path / "log.csv", measures_path=tmp_path / "measures.csv")

    assert reader.call_count == 1
    assert result["usage"]["side_calls"] == 1
    assert result["usage"]["model_calls"] == stations.call_count + 1


def test_each_turns_reading_is_saved_with_the_case(tmp_path):
    """Without it an error analysis can only infer the reading from the
    nodes and tools that followed it -- which is how stage 79's was done."""
    from tests.conftest import StubChatModel

    stations = StubChatModel(responses=[AIMessage(content="ok")] * 5)
    reader = StubChatModel(responses=[AIMessage(content='{"wants": ["just_chat"]}')])
    graph = build_eval_graph(tmp_path / "log.csv", tmp_path / "measures.csv", model=stations, reader_model=reader)

    result = run_case(
        {"turns": ["just chatting", {"text": "I'm feeling wired.", "energy": "wired", "choices_clicked": True}]},
        agent_graph=graph, log_path=tmp_path / "log.csv", measures_path=tmp_path / "measures.csv",
    )

    # One entry per turn: read, then a button click nobody had to read.
    assert result["readings"] == [["just_chat"], None]
    # A click that nobody had to read is not a failed reading.
    assert result["usage"]["reading_failures"] == 0


def test_turns_the_reader_could_not_read_are_counted(tmp_path):
    """A failed reading falls back silently, by design; the case says how many
    of its turns did, because each one ran the routing of v1.0.0."""
    from tests.conftest import StubChatModel

    stations = StubChatModel(responses=[AIMessage(content="ok")] * 5)
    reader = StubChatModel(responses=[AIMessage(content="not json"), AIMessage(content='{"wants": ["unclear"]}')])
    graph = build_eval_graph(tmp_path / "log.csv", tmp_path / "measures.csv", model=stations, reader_model=reader)

    result = run_case({"turns": ["hm", "what is this"]}, agent_graph=graph,
                      log_path=tmp_path / "log.csv", measures_path=tmp_path / "measures.csv")

    assert result["readings"] == [None, ["unclear"]]
    assert result["usage"]["reading_failures"] == 1


def test_a_crisis_exit_keeps_the_readings_so_far():
    result = run_case({"turns": ["busy week", "I want to end my life"]},
                      agent_graph=_PricedGraph(reply_content="gpt-5-mini"))

    assert result["crisis_intercepted"] is True
    assert result["readings"] == [None]
