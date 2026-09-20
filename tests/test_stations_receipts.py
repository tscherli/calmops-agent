"""The tools node: one receipt per call, and the state the tools write.

Receipts are the project's only hard guarantee about side effects, and they
are the easiest thing to lose here: most station tools return a `Command`, and
a node that only writes receipts on the other branch fails silently -- the UI
just shows fewer lines, and the fabrication evaluator gets *stricter*, so the
eval score looks fine while the guarantee is gone.
"""
import csv

import pytest
from langchain_core.messages import AIMessage, HumanMessage

from agent.graph import build_agent
from tests.conftest import StubChatModel
from tools.measures import MEASURE_COLUMNS

MEASURE_ROWS = [
    ("short_walk", "recover", "Walk for five minutes.", "exhausted;wired", "minutes", "user"),
    ("breathe", "recover", "Take a few slow breaths.", "exhausted;calm;wired", "minutes", "user"),
    ("say_no", "act", "Say no to one request today.", "exhausted;calm;wired", "minutes", "agent"),
]


@pytest.fixture
def measures_file(tmp_path):
    path = tmp_path / "measures.csv"
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(MEASURE_COLUMNS)
        writer.writerows(MEASURE_ROWS)
    return path


@pytest.fixture
def log_file(tmp_path):
    return tmp_path / "checkin_log.csv"


def call(name, args, call_id="c1"):
    return AIMessage(content="", tool_calls=[{"name": name, "args": args, "id": call_id}])


def run(graph, state):
    return graph.invoke(state)


def agent_that_calls(measures_file, log_file, *tool_calls, then="done"):
    """A graph whose model asks for the given tools, then replies."""
    responses = [*tool_calls, AIMessage(content=then)]
    model = StubChatModel(responses=responses)
    graph = build_agent(model=model, log_path=log_file, measures_path=measures_file)
    graph.stub = model  # type: ignore[attr-defined]
    return graph


def receipts(result, tool=None):
    events = result.get("tool_events") or []
    return [e for e in events if tool is None or e["tool"] == tool]


BASE = {"energy": "wired", "goal": "calm down now", "offered": ["breathe"], "step": "follow_up"}


# -- one receipt per call ----------------------------------------------------


@pytest.mark.parametrize(
    "name,args",
    [
        ("record_choices", {"energy": "calm", "goal": "plan recovery"}),
        ("find_measures", {}),
        ("suggestion_not_helpful", {}),
        ("discuss_idea", {"idea": "pause before replying"}),
        ("idea_decided", {"outcome": "drop"}),
        ("log_checkin", {"measure_id": "breathe", "helpfulness": "helped"}),
        (
            "add_measure",
            {
                "measure_id": "new_one",
                "measure": "Try the new thing.",
                "measure_type": "act",
                "fits_energy": ["wired"],
                "effort": "minutes",
            },
        ),
    ],
)
def test_every_tool_produces_exactly_one_receipt(measures_file, log_file, name, args):
    """Most of these return a Command; a node that forgets that branch would
    drop their receipts without anything failing."""
    graph = agent_that_calls(measures_file, log_file, call(name, args))
    result = run(graph, {"messages": [HumanMessage(content="hi")], **BASE})
    assert len(receipts(result, name)) == 1


def test_two_calls_in_one_message_each_get_their_own_receipt(measures_file, log_file):
    both = AIMessage(
        content="",
        tool_calls=[
            {"name": "record_choices", "args": {"energy": "calm", "goal": "plan recovery"}, "id": "a"},
            {"name": "discuss_idea", "args": {"idea": "pause"}, "id": "b"},
        ],
    )
    graph = agent_that_calls(measures_file, log_file, both)
    result = run(graph, {"messages": [HumanMessage(content="hi")], **BASE})
    assert {e["tool"] for e in receipts(result)} == {"record_choices", "discuss_idea"}


def test_an_unknown_tool_is_an_error_receipt_not_a_crash(measures_file, log_file):
    graph = agent_that_calls(measures_file, log_file, call("no_such_tool", {}))
    result = run(graph, {"messages": [HumanMessage(content="hi")], **BASE})
    assert receipts(result, "no_such_tool")[0]["exit_code"] == 1


# -- the state the tools write -----------------------------------------------


def test_record_choices_reaches_the_state(measures_file, log_file):
    """If the Command update were dropped, the graph would ask the same
    question forever and nothing would raise."""
    graph = agent_that_calls(
        measures_file, log_file, call("record_choices", {"energy": "calm", "goal": "plan recovery"})
    )
    result = run(graph, {"messages": [HumanMessage(content="calm, plan recovery")]})
    assert result["energy"] == "calm"
    assert result["goal"] == "plan recovery"


def test_find_measures_records_what_it_showed(measures_file, log_file):
    graph = agent_that_calls(measures_file, log_file, call("find_measures", {}))
    result = run(graph, {"messages": [HumanMessage(content="ideas?")], "energy": "wired", "goal": "calm down now"})
    assert result["offered"]
    assert receipts(result, "find_measures")[0]["exit_code"] == 0


def test_an_idea_reaches_the_state_and_is_cleared_again(measures_file, log_file):
    graph = agent_that_calls(measures_file, log_file, call("discuss_idea", {"idea": "pause first"}))
    opened = run(graph, {"messages": [HumanMessage(content="what about pausing?")], **BASE})
    assert opened["idea"] == "pause first"

    graph2 = agent_that_calls(measures_file, log_file, call("idea_decided", {"outcome": "drop"}))
    closed = run(graph2, {"messages": [HumanMessage(content="drop it")], **BASE, "idea": "pause first"})
    assert closed["idea"] is None


# -- the log's receipts ------------------------------------------------------


def test_a_written_row_is_reported_with_exit_zero(measures_file, log_file):
    graph = agent_that_calls(
        measures_file,
        log_file,
        call("log_checkin", {"energy": "wired", "circumstances": "A rough one.", "measure_id": "breathe"}),
    )
    result = run(graph, {"messages": [HumanMessage(content="yes that's right")], **BASE})
    event = receipts(result, "log_checkin")[0]
    assert event["exit_code"] == 0
    assert "checkin" in event["detail"]
    written = list(csv.DictReader(log_file.open(encoding="utf-8", newline="")))
    assert [r["measure_id"] for r in written] == ["breathe"]
    assert written[0]["circumstances"] == "A rough one."


def test_an_invented_measure_id_is_refused_and_says_so(measures_file, log_file):
    graph = agent_that_calls(
        measures_file, log_file, call("log_checkin", {"measure_id": "invented", "energy": "wired"})
    )
    result = run(graph, {"messages": [HumanMessage(content="log it")], **BASE})
    assert receipts(result, "log_checkin")[0]["exit_code"] == 1
    assert not log_file.exists()


def test_a_check_in_with_no_measure_is_skipped_and_has_no_exit_code(measures_file, log_file):
    """A tool that refused to run has no exit code; reporting 0 would be a lie."""
    graph = agent_that_calls(
        measures_file, log_file, call("log_checkin", {"energy": "wired", "circumstances": "Just talking."})
    )
    result = run(graph, {"messages": [HumanMessage(content="log it")], **BASE})
    assert receipts(result, "log_checkin")[0]["exit_code"] is None


def test_a_logged_check_in_ends_the_conversation(measures_file, log_file):
    graph = agent_that_calls(
        measures_file,
        log_file,
        call("log_checkin", {"energy": "wired", "circumstances": "A rough one.", "measure_id": "breathe"}),
    )
    result = run(graph, {"messages": [HumanMessage(content="yes")], **BASE})
    assert result["complete"] is True


def test_a_rating_does_not_end_the_conversation(measures_file, log_file):
    """Saying 'that helped' mid-conversation must not wipe the context."""
    graph = agent_that_calls(
        measures_file, log_file, call("log_checkin", {"measure_id": "breathe", "helpfulness": "helped"})
    )
    result = run(graph, {"messages": [HumanMessage(content="it helped")], **BASE})
    assert result.get("complete") is not True


# -- the model cannot fake one -----------------------------------------------


def test_a_claim_in_the_prose_does_not_produce_a_receipt(measures_file, log_file):
    """The whole point: prose is checkable against the receipts."""
    model = StubChatModel(responses=[AIMessage(content="I've logged that for you.")])
    graph = build_agent(model=model, log_path=log_file, measures_path=measures_file)
    result = graph.invoke({"messages": [HumanMessage(content="hi")], **BASE})
    assert receipts(result) == []
    assert not log_file.exists()
