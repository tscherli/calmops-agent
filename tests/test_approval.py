"""The approval card: a write runs only after the user said yes.

The model may call `add_measure` and `log_checkin`; the graph pauses in
`review`, and the tools node runs a write only with an approve or edit decision
for that exact call. Executions are counted, not inferred: a double that only
raises proves nothing in a graph whose clients swallow exceptions.
"""
import csv

import pytest
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.tools import tool
from langgraph.checkpoint.memory import InMemorySaver

from agent.graph import REJECTED_TEXT, WRITE_TOOLS, action_request, build_agent
from tests.conftest import StubChatModel, invoke_approving
from tools.measures import MEASURE_COLUMNS

BASE = {"energy": "wired", "goal": "calm down now", "offered": ["breathe"], "step": "follow_up"}


def calls(*specs):
    """One AI message asking for several tools: specs are (name, args, id)."""
    return AIMessage(
        content="", tool_calls=[{"name": n, "args": a, "id": i} for n, a, i in specs]
    )


class Counting:
    """Stand-ins with the real tool names that count how often they ran."""

    def __init__(self):
        self.runs: dict[str, list[dict]] = {"find_measures": [], "add_measure": [], "log_checkin": []}

        @tool("find_measures")
        def find_measures() -> str:
            """Look up measures."""
            self.runs["find_measures"].append({})
            return "[]"

        @tool("add_measure")
        def add_measure(measure: str = "", measure_id: str = "") -> str:
            """Add a measure."""
            self.runs["add_measure"].append({"measure": measure, "measure_id": measure_id})
            return "added"

        @tool("log_checkin")
        def log_checkin(circumstances: str = "", measure_id: str = "", helpfulness: str = "") -> str:
            """Log a check-in."""
            self.runs["log_checkin"].append({"circumstances": circumstances})
            return "logged"

        self.tools = [find_measures, add_measure, log_checkin]


def graph_with(tools, *responses, checkpointer=True):
    model = StubChatModel(responses=[*responses, AIMessage(content="done")])
    return build_agent(
        model=model, tools=tools, checkpointer=InMemorySaver() if checkpointer else None
    )


def turn(graph, decide=None):
    state = {"messages": [HumanMessage(content="hi")], **BASE}
    return invoke_approving(graph, state, decide) if decide else invoke_approving(graph, state)


def reject(request):
    return {"type": "reject"}


# -- the policy --------------------------------------------------------------


def test_exactly_the_two_lasting_effects_are_write_tools():
    """Everything else changes nothing that outlives the conversation."""
    assert WRITE_TOOLS == {"add_measure", "log_checkin"}


# -- a write needs a yes ------------------------------------------------------


@pytest.mark.parametrize("name", sorted(WRITE_TOOLS))
@pytest.mark.parametrize(
    "decide",
    [
        pytest.param(reject, id="reject"),
        pytest.param(lambda r: {"type": "maybe"}, id="unknown-type"),
        pytest.param(lambda r: {"type": "approve", "id": "someone-else"}, id="wrong-id"),
    ],
)
def test_a_write_never_runs_without_a_matching_approval(name, decide):
    """A reject, an unknown answer and an approval for some other call all
    leave this call unapproved -- and an unapproved write does not run."""
    counting = Counting()
    graph = graph_with(counting.tools, calls((name, {}, "w1")))
    result = turn(graph, decide)
    assert result["pauses"] == 1
    assert counting.runs[name] == []


@pytest.mark.parametrize("name", sorted(WRITE_TOOLS))
def test_an_approved_write_runs_exactly_once(name):
    counting = Counting()
    graph = graph_with(counting.tools, calls((name, {}, "w1")))
    result = turn(graph)
    assert result["pauses"] == 1
    assert len(counting.runs[name]) == 1


def test_a_graph_that_cannot_pause_refuses_the_write():
    """Without a checkpointer there is nobody to ask -- the eval harness builds
    it that way. Nothing may be written then, and nothing may hang."""
    counting = Counting()
    graph = graph_with(counting.tools, calls(("add_measure", {}, "w1")), checkpointer=False)
    result = graph.invoke({"messages": [HumanMessage(content="hi")], **BASE})
    assert not result.get("__interrupt__")
    assert counting.runs["add_measure"] == []
    assert result["tool_events"][0]["status"] == "rejected"


# -- reads do not ask ----------------------------------------------------------


@pytest.mark.parametrize("name,args", [("find_measures", {}), ("record_choices", {"energy": "calm"})])
def test_reads_and_session_tools_never_pause(name, args):
    counting = Counting()
    from agent.graph import record_choices

    graph = graph_with([*counting.tools, record_choices], calls((name, args, "r1")))
    result = turn(graph)
    assert result["pauses"] == 0


# -- one card for many writes --------------------------------------------------


def test_five_writes_in_one_message_pause_once_and_are_matched_by_id():
    """Turn 10 of the first hand test asked for five measures at once. One
    card for all of them; each decision applies to its own call, whatever
    order the answers arrive in."""
    counting = Counting()
    specs = [("add_measure", {"measure": f"idea {i}"}, f"w{i}") for i in range(5)]
    graph = graph_with(counting.tools, calls(*specs))
    seen = []

    def decide(request):
        seen.append(request["id"])
        return {"type": "approve"} if request["id"] in {"w1", "w3"} else {"type": "reject"}

    result = turn(graph, decide)
    assert result["pauses"] == 1
    assert sorted(seen) == ["w0", "w1", "w2", "w3", "w4"]
    assert sorted(r["measure"] for r in counting.runs["add_measure"]) == ["idea 1", "idea 3"]


def test_decisions_in_reversed_order_still_land_on_their_own_calls():
    counting = Counting()
    graph = graph_with(
        counting.tools,
        calls(("add_measure", {"measure": "keep"}, "a"), ("add_measure", {"measure": "drop"}, "b")),
    )
    config = {"configurable": {"thread_id": "reversed"}}
    paused = graph.invoke({"messages": [HumanMessage(content="hi")], **BASE}, config)
    from langgraph.types import Command

    graph.invoke(
        Command(resume={"decisions": [{"id": "b", "type": "reject"}, {"id": "a", "type": "approve"}]}),
        config,
    )
    assert paused["__interrupt__"]
    assert [r["measure"] for r in counting.runs["add_measure"]] == ["keep"]


def test_resuming_does_not_run_the_reads_a_second_time():
    """A paused node re-runs from its start on resume. Were the pause inside
    the tools node, the lookup before it would run twice."""
    counting = Counting()
    graph = graph_with(
        counting.tools, calls(("find_measures", {}, "r1"), ("add_measure", {"measure": "x"}, "w1"))
    )
    turn(graph)
    assert len(counting.runs["find_measures"]) == 1
    assert len(counting.runs["add_measure"]) == 1


# -- what a rejection leaves behind --------------------------------------------


def test_a_rejection_leaves_a_receipt_and_tells_the_model():
    """The receipt says it did not run; the model reads that it was refused,
    so it cannot go on to claim it happened."""
    counting = Counting()
    graph = graph_with(counting.tools, calls(("log_checkin", {"circumstances": "x"}, "w1")))
    result = turn(graph, reject)
    event = [e for e in result["tool_events"] if e["tool"] == "log_checkin"][0]
    assert event["status"] == "rejected"
    assert event["exit_code"] is None
    refusal = [m for m in result["messages"] if isinstance(m, ToolMessage) and m.tool_call_id == "w1"][0]
    assert refusal.content == REJECTED_TEXT
    assert refusal.status == "error"


def test_a_refused_write_still_names_the_call_it_refused():
    """A refusal is a record of a call, so it names one like every other
    receipt. Without the id, "this card was rejected" and "that write went
    ahead" cannot be put side by side at all."""
    counting = Counting()
    graph = graph_with(counting.tools, calls(("log_checkin", {"circumstances": "x"}, "w1")))
    result = turn(graph, reject)
    event = [e for e in result["tool_events"] if e["tool"] == "log_checkin"][0]
    assert event["call_id"] == "w1"


# -- edit ----------------------------------------------------------------------


@pytest.fixture
def measures_file(tmp_path):
    path = tmp_path / "measures.csv"
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(MEASURE_COLUMNS)
        writer.writerow(["breathe", "recover", "Take a few slow breaths.", "exhausted;calm;wired", "minutes", "user"])
    return path


def real_add_measure_graph(measures_file, args):
    model = StubChatModel(responses=[calls(("add_measure", args, "w1")), AIMessage(content="done")])
    return build_agent(model=model, measures_path=measures_file, checkpointer=InMemorySaver())


PROPOSED = {
    "measure_id": "foam_roll",
    "measure": "Roll your back on a foam roller.",
    "measure_type": "recover",
    "fits_energy": ["exhausted"],
    "effort": "minutes",
}


def test_an_edit_writes_the_users_version(measures_file):
    graph = real_add_measure_graph(measures_file, PROPOSED)
    edited = dict(PROPOSED, measure="Roll gently, five minutes.")
    turn(graph, lambda r: {"type": "edit", "args": edited})
    rows = list(csv.DictReader(measures_file.open(encoding="utf-8", newline="")))
    assert [r["measure"] for r in rows if r["measure_id"] == "foam_roll"] == ["Roll gently, five minutes."]


def test_an_edited_value_still_goes_through_the_tools_own_checks(measures_file):
    """The user's edit is input like any other: an invalid id is refused, and
    the file is byte-for-byte what it was."""
    before = measures_file.read_bytes()
    graph = real_add_measure_graph(measures_file, PROPOSED)
    result = turn(graph, lambda r: {"type": "edit", "args": dict(PROPOSED, measure_id="gua sha_tools")})
    event = [e for e in result["tool_events"] if e["tool"] == "add_measure"][0]
    assert event["exit_code"] == 1
    assert measures_file.read_bytes() == before


def test_a_rejected_measure_leaves_the_file_untouched(measures_file):
    before = measures_file.read_bytes()
    graph = real_add_measure_graph(measures_file, PROPOSED)
    turn(graph, reject)
    assert measures_file.read_bytes() == before


# -- what the card shows ------------------------------------------------------


def test_the_card_shows_what_will_be_written_built_from_the_arguments():
    """Not the model's summary of it: an approval of a paraphrase approves
    nothing."""
    add = action_request({"name": "add_measure", "id": "a", "args": {"measure": "Stretch."}})
    log = action_request(
        {"name": "log_checkin", "id": "b", "args": {"circumstances": "Deadline week.", "measure_id": "breathe"}}
    )
    rate = action_request(
        {"name": "log_checkin", "id": "c", "args": {"measure_id": "breathe", "helpfulness": "helped"}}
    )
    assert add["description"] == 'Add to your list: "Stretch."' and add["editable"] == "measure"
    assert log["description"] == 'Log: "Deadline week." (measure: breathe)' and log["editable"] == "circumstances"
    assert "helped" in rate["description"] and rate["editable"] is None
