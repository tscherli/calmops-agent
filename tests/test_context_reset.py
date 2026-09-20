"""next_turn_state -- what a client carries from one turn into the next.

This is risk 5 of the port, and it is silent in both directions: reset too
eagerly and the next sentence starts a cold check-in with the transcript gone,
too lazily and every turn re-sends a growing history. The app and the eval
harness implemented this separately once and drifted (dev log stage 26), so
there is one function and these tests hold it to both jobs.
"""
import csv

import pytest
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from agent.graph import CARRIED_FIELDS, build_agent, next_turn_state, station_for
from tests.conftest import StubChatModel
from tools.measures import MEASURE_COLUMNS

MEASURE_ROWS = [
    ("breathe", "recover", "Take a few slow breaths.", "exhausted;calm;wired", "minutes", "user"),
    ("say_no", "act", "Say no to one request today.", "exhausted;calm;wired", "minutes", "agent"),
]

LIVE = {
    "energy": "wired",
    "goal": "calm down now",
    "offered": ["breathe"],
    "rejections": 2,
    "step": "follow_up",
    "idea": "pause before replying",
    "debate_turns": 3,
}


@pytest.fixture
def measures_file(tmp_path):
    path = tmp_path / "measures.csv"
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(MEASURE_COLUMNS)
        writer.writerows(MEASURE_ROWS)
    return path


def call(name, args, call_id="c1"):
    return AIMessage(content="", tool_calls=[{"name": name, "args": args, "id": call_id}])


def agent_that_calls(tmp_path, measures_file, tool_call):
    model = StubChatModel(responses=[tool_call, AIMessage(content="done")])
    return build_agent(
        model=model, log_path=tmp_path / "checkin_log.csv", measures_path=measures_file
    )


# -- after a completed check-in ----------------------------------------------


def test_a_completed_check_in_starts_the_next_turn_cold():
    carried = next_turn_state({"complete": True, "messages": [HumanMessage(content="hi")], **LIVE})
    assert carried["messages"] == []
    assert carried["energy"] is None and carried["goal"] is None
    assert carried["offered"] == []
    assert carried["rejections"] == 0
    assert carried["step"] is None
    assert carried["idea"] is None
    assert carried["debate_turns"] == 0


def test_the_reset_clears_every_field_the_router_reads():
    """A field the reset forgets keeps routing the next conversation: a stale
    `offered` sends the first sentence to follow_up, a stale `idea` to debate."""
    carried = next_turn_state({"complete": True, "messages": [], **LIVE})
    for field in CARRIED_FIELDS:
        assert field in carried, field
    assert station_for(carried) == "ask"


def test_the_completion_flag_itself_is_never_carried():
    """Carrying it would end the next turn before the user said anything."""
    finished = next_turn_state({"complete": True, "messages": [], **LIVE})
    running = next_turn_state({"messages": [], "checkin_logged": True, **LIVE})
    for carried in (finished, running):
        assert "complete" not in carried
        assert "checkin_logged" not in carried


def test_the_button_flag_is_never_carried():
    """`rejected` is an input from one click; carried, every later turn would
    count another rejection and the agent would keep stepping back."""
    assert "rejected" not in next_turn_state({"messages": [], "rejected": True, **LIVE})


def test_receipts_are_never_carried():
    """tool_events appends, so re-sending it would duplicate every receipt and
    the UI would report the same tool run on every turn that follows."""
    events = [{"tool": "find_measures", "status": "ok", "exit_code": 0, "detail": "2 measures"}]
    assert "tool_events" not in next_turn_state({"messages": [], "tool_events": events, **LIVE})


# -- while the conversation is still running ---------------------------------


def test_an_unfinished_turn_keeps_every_station_field():
    carried = next_turn_state({"messages": [HumanMessage(content="hi")], **LIVE})
    for field, value in LIVE.items():
        assert carried[field] == value, field


def test_only_the_fat_payload_shrinks():
    """find_measures returns the whole measure list; the log's receipt is small
    and is what the next station reads to know a row exists."""
    fat = "x" * 400
    carried = next_turn_state(
        {
            "messages": [
                ToolMessage(content=fat, name="find_measures", tool_call_id="a"),
                ToolMessage(content=fat, name="log_checkin", tool_call_id="b"),
                ToolMessage(content="short", name="find_measures", tool_call_id="c"),
            ],
            **LIVE,
        }
    )
    kept = carried["messages"]
    assert len(kept[0].content) < len(fat)
    assert kept[1].content == fat
    assert kept[2].content == "short"


def test_the_conversation_order_survives_compaction():
    carried = next_turn_state(
        {
            "messages": [
                HumanMessage(content="I am wired"),
                AIMessage(content="here are two ideas"),
                ToolMessage(content="y" * 400, name="find_measures", tool_call_id="a"),
            ],
            **LIVE,
        }
    )
    assert [type(m) for m in carried["messages"]] == [HumanMessage, AIMessage, ToolMessage]
    assert carried["messages"][0].content == "I am wired"


# -- through the graph, which is where the flag is actually set --------------


def test_a_logged_check_in_resets_and_the_next_sentence_starts_at_ask(tmp_path, measures_file):
    graph = agent_that_calls(
        tmp_path,
        measures_file,
        call("log_checkin", {"energy": "wired", "circumstances": "A rough one.", "measure_id": "breathe"}),
    )
    result = graph.invoke({"messages": [HumanMessage(content="yes, that is right")], **LIVE, "idea": None})
    assert result["complete"] is True
    assert station_for(next_turn_state(result)) == "ask"


def test_a_rating_leaves_the_conversation_where_it_was(tmp_path, measures_file):
    """Saying 'that helped' is a second row, not the end of a conversation."""
    graph = agent_that_calls(
        tmp_path, measures_file, call("log_checkin", {"measure_id": "breathe", "helpfulness": "helped"})
    )
    result = graph.invoke({"messages": [HumanMessage(content="it helped")], **LIVE, "idea": None})
    carried = next_turn_state(result)
    assert carried["energy"] == "wired" and carried["offered"] == ["breathe"]
    assert station_for(carried) == "follow_up"
