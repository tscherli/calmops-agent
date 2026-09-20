"""agent/stations.py -- the station graph: routing, bindings, canned nodes.

No model is called in this file: a stub records what it was bound to and what
it was asked. What the graph guarantees must hold without the model's help.
"""
import csv

import pytest
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from agent.stations import (
    DEBATE_TURN_CAP,
    NOT_HELPFUL_PHRASES,
    build_agent,
    next_turn_state,
    says_not_helpful,
    station_for,
)
from tests.test_agent_pipeline import StubChatModel  # records bind_tools calls
from tools.measures import MEASURE_COLUMNS

MEASURE_ROWS = [
    ("short_walk", "recover", "Walk for five minutes.", "exhausted;wired", "minutes", "user"),
    ("breathe", "recover", "Take a few slow breaths.", "exhausted;calm;wired", "minutes", "user"),
    ("say_no", "act", "Say no to one request today.", "exhausted;calm;wired", "minutes", "agent"),
    ("day_off", "recover", "Take a day off.", "exhausted;calm;wired", "days", "user"),
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
def agent(tmp_path, measures_file):
    """A graph whose model never answers: every test here stops before that."""
    model = StubChatModel(responses=[AIMessage(content="ok")] * 10)
    graph = build_agent(model=model, log_path=tmp_path / "log.csv", measures_path=measures_file)
    graph.stub = model  # type: ignore[attr-defined]
    return graph


def user(text):
    return HumanMessage(content=text)


# -- which station fits the state --------------------------------------------


@pytest.mark.parametrize(
    "state,expected",
    [
        ({}, "ask"),
        ({"energy": "wired"}, "ask"),
        ({"energy": "wired", "goal": "calm down now"}, "suggest"),
        ({"energy": "wired", "goal": "calm down now", "offered": ["breathe"]}, "follow_up"),
        ({"energy": "wired", "goal": "calm down now", "offered": ["breathe"], "idea": "pause"}, "debate"),
        ({"idea": "pause"}, "debate"),
    ],
)
def test_the_station_follows_from_the_state(state, expected):
    assert station_for(state) == expected


def test_an_open_idea_comes_before_everything_else():
    """The user brought something of their own; every other station would talk
    past it."""
    assert station_for({"idea": "wait a few seconds", "energy": "calm"}) == "debate"


# -- recognising a rejection in code -----------------------------------------


@pytest.mark.parametrize("text", ["not helpful", "that doesn't help", "Das hilft nicht", "not useful"])
def test_short_rejections_are_recognised_without_a_model(text):
    assert says_not_helpful(text) is True


def test_a_long_message_goes_to_the_model_even_if_it_contains_the_phrase():
    """A real conversation was lost this way: the message rejected a suggestion
    and added the user's own idea, and only the rejection was heard."""
    long_one = (
        "it's still not helpful but what i find helpful is to look at why my colleague reacts that way"
    )
    assert says_not_helpful(long_one) is False


def test_every_phrase_in_the_list_is_matched():
    assert all(says_not_helpful(phrase) for phrase in NOT_HELPFUL_PHRASES)


# -- the graph's shape -------------------------------------------------------


def test_the_graph_has_exactly_the_expected_nodes(agent):
    assert sorted(agent.get_graph().nodes) == [
        "__end__",
        "__start__",
        "ask",
        "ask_goal_again",
        "checkin_done",
        "count_rejection",
        "debate",
        "follow_up",
        "suggest",
        "tools",
        "web_search_ideas",
    ]


def test_the_conditional_edges_go_exactly_where_the_design_says(agent):
    """The routing is the design. An edge added by accident (a station reaching
    the search node directly, say) is a path no test would otherwise walk."""
    edges = {(e.source, e.target) for e in agent.get_graph().edges if e.conditional}
    stations = ("ask", "suggest", "follow_up", "debate")
    assert edges == {("__start__", s) for s in (*stations, "count_rejection")} | {
        (s, t) for s in stations for t in ("tools", "checkin_done", "__end__")
    } | {("tools", t) for t in (*stations, "web_search_ideas", "ask_goal_again")} | {
        ("count_rejection", "web_search_ideas"),
        ("count_rejection", "ask_goal_again"),
    }


def test_the_dead_ends_really_end(agent):
    """The three canned nodes reply and stop; a loop back would cost a model
    call to say the same sentence again."""
    plain = {(e.source, e.target) for e in agent.get_graph().edges if not e.conditional}
    assert plain == {(n, "__end__") for n in ("ask_goal_again", "web_search_ideas", "checkin_done")}


def test_the_recursion_limit_is_configured(agent):
    """The backstop against a loop that would otherwise run up a bill."""
    assert agent.config["recursion_limit"] == 25
    assert build_agent(model=StubChatModel(responses=[]), recursion_limit=7).config["recursion_limit"] == 7


# -- what each station may reach ---------------------------------------------


def test_the_ask_station_cannot_suggest_a_measure(agent):
    """Not a rule in a prompt: the tool is not bound there."""
    assert agent.station_tools["ask"] == ["record_choices", "discuss_idea"]
    assert "find_measures" not in agent.station_tools["ask"]
    assert "log_checkin" not in agent.station_tools["ask"]


def test_only_the_follow_up_station_can_write_to_the_log(agent):
    for station in ("ask", "suggest", "debate"):
        assert "log_checkin" not in agent.station_tools[station]
    assert "log_checkin" in agent.station_tools["follow_up"]


def test_the_debate_station_can_keep_an_idea_but_not_look_measures_up(agent):
    assert sorted(agent.station_tools["debate"]) == ["add_measure", "idea_decided", "record_choices"]


def test_the_stations_bind_only_their_own_tools(agent, measures_file):
    """The binding matrix is the design: everything else is commentary."""
    model = StubChatModel(responses=[AIMessage(content="hi")])
    graph = build_agent(model=model, measures_path=measures_file)
    graph.invoke({"messages": [user("I'm wired")]})
    bound = [{t.name for t in call if hasattr(t, "name")} for call in model.bind_calls]
    assert bound and bound[0] == {"record_choices", "discuss_idea"}


# -- the canned nodes cost nothing -------------------------------------------


def test_the_step_back_question_makes_no_model_call(agent):
    """Same question every time; generating it would only invite drift."""
    before = agent.stub.call_count
    result = agent.invoke(
        {
            "messages": [user("still not helpful")],
            "energy": "wired",
            "goal": "calm down now",
            "offered": ["breathe"],
            "rejections": 1,
        }
    )
    assert agent.stub.call_count == before
    assert "step back" in result["messages"][-1].content
    assert result["goal"] is None       # so the ask station really asks again
    assert result["rejections"] == 0


def test_the_first_rejection_goes_to_the_forced_search_the_second_to_the_question(agent):
    state = {
        "messages": [user("not helpful")],
        "energy": "wired",
        "goal": "calm down now",
        "offered": ["breathe"],
    }
    first = agent.invoke(dict(state, rejections=0))
    assert first["rejections"] == 1
    second = agent.invoke(dict(state, rejections=1))
    assert "step back" in second["messages"][-1].content


def test_a_button_click_is_taken_as_a_fact(agent):
    """The UI's 'not helpful' button sets a flag; no phrase matching needed."""
    result = agent.invoke(
        {
            "messages": [user("hmm")],
            "energy": "wired",
            "goal": "calm down now",
            "offered": ["breathe"],
            "rejections": 1,
            "rejected": True,
        }
    )
    assert "step back" in result["messages"][-1].content


def test_a_rejection_before_anything_was_suggested_is_not_one(agent):
    """Nothing has been offered yet, so 'not helpful' is just a sentence."""
    result = agent.invoke({"messages": [user("not helpful")]})
    assert result["step"] == "ask"


# -- what the client carries into the next turn ------------------------------


def test_a_completed_check_in_clears_the_station_fields_too():
    """Clearing only the messages would leave `offered` set, and the next
    sentence would land in follow_up with an empty transcript."""
    carried = next_turn_state(
        {
            "complete": True,
            "messages": [user("hi")],
            "energy": "wired",
            "goal": "calm down now",
            "offered": ["breathe"],
            "rejections": 2,
            "step": "follow_up",
            "idea": "pause",
        }
    )
    assert carried["messages"] == []
    assert all(carried[k] in (None, [], 0) for k in ("energy", "goal", "offered", "rejections", "step", "idea"))


def test_an_unfinished_conversation_keeps_its_state_and_compacts_the_fat_payload():
    fat = "x" * 400
    carried = next_turn_state(
        {
            "messages": [ToolMessage(content=fat, name="find_measures", tool_call_id="c1")],
            "energy": "calm",
            "goal": "plan recovery",
            "offered": ["day_off"],
            "step": "suggest",
        }
    )
    assert carried["energy"] == "calm" and carried["offered"] == ["day_off"]
    assert len(carried["messages"][0].content) < len(fat)


def test_the_debate_cap_is_a_number_the_graph_knows():
    """The discussion runs on the expensive model, so it ends in a decision."""
    assert DEBATE_TURN_CAP == 6
