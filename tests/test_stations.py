"""agent/graph.py -- the station graph: routing, bindings, canned nodes.

No model is called in this file: a stub records what it was bound to and what
it was asked. What the graph guarantees must hold without the model's help.
"""
import csv

import pytest
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from agent.graph import (
    DEBATE_TURN_CAP,
    NOT_HELPFUL_MAX_CHARS,
    NOT_HELPFUL_PHRASES,
    build_agent,
    next_turn_state,
    says_not_helpful,
    station_for,
)
from tests.conftest import StubChatModel  # records bind_tools calls
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
        # Both scale values but nothing about what is going on: still `ask`.
        # Without the situation the lookup can only filter by the two values,
        # and every measure for that energy fits equally.
        ({"energy": "wired", "goal": "calm down now"}, "ask"),
        ({"energy": "wired", "goal": "calm down now", "situation": "back to back calls"}, "suggest"),
        ({"energy": "wired", "goal": "calm down now", "situation": "not given"}, "suggest"),
        ({"energy": "wired", "goal": "calm down now", "situation": "loud day",
          "offered": ["breathe"]}, "follow_up"),
        ({"energy": "wired", "goal": "calm down now", "situation": "loud day",
          "offered": ["breathe"], "idea": "pause"}, "debate"),
        ({"idea": "pause"}, "debate"),
    ],
)
def test_the_station_follows_from_the_state(state, expected):
    assert station_for(state) == expected


def test_the_situation_is_a_gate_not_a_request(agent):
    """A suggestion is only as good as what it was chosen against. With energy
    and goal alone the lookup filters on the two scale values and every measure
    for that energy fits equally, so the model's stated reason can only repeat
    the energy word back -- which is what the hand tests of 2026-09-24 showed it
    doing. Code routes to `ask` until the situation is in."""
    assert station_for({"energy": "wired", "goal": "calm down now"}) == "ask"
    assert station_for(
        {"energy": "wired", "goal": "calm down now", "situation": "back to back calls"}
    ) == "suggest"


def test_declining_to_say_is_an_answer_not_a_gap():
    """Otherwise "rather not say" routes to `ask` forever. The recorded value
    is what distinguishes declined from unasked."""
    from agent.graph import SITUATION_DECLINED

    assert station_for(
        {"energy": "wired", "goal": "calm down now", "situation": SITUATION_DECLINED}
    ) == "suggest"


def test_two_button_clicks_do_not_jump_the_gate(agent):
    """The click path writes energy and goal straight into the payload without
    passing through `ask`, so it is the one route that could skip the question.

    It has to be asserted through a real turn: `station_for` never reads
    `choices_clicked` -- that branch lives in `route_start` -- so a test built
    on `station_for` passes whether the guard is there or not. The first
    version of this test did exactly that, and a break test caught it."""
    clicked = {
        "messages": [user("I'm feeling wired and I want to calm down now.")],
        "energy": "wired",
        "goal": "calm down now",
        "choices_clicked": True,
    }

    assert agent.invoke(clicked)["step"] == "ask"
    assert agent.invoke({**clicked, "situation": "a loud morning"})["step"] == "suggest"


def test_the_reply_to_the_situation_question_must_use_a_tool():
    """The gate would otherwise loop: the station asks, the user answers, the
    model says "got it" without recording it, and the next turn asks again."""
    from agent.graph import situation_question_unanswered

    asked = {
        "messages": [user("back to back calls since nine")],
        "energy": "wired",
        "goal": "calm down now",
        "step": "ask",
    }

    assert situation_question_unanswered(asked) is True
    assert situation_question_unanswered({**asked, "situation": "calls"}) is False
    # A tool result, not the user's reply: forcing again here would loop.
    assert situation_question_unanswered({**asked, "messages": []}) is False


def test_record_choices_carries_the_situation(agent):
    """It travels with energy and goal rather than in a tool of its own: one
    call, whatever the user happened to give."""
    from agent.graph import record_choices

    result = record_choices.invoke(
        {"type": "tool_call", "name": "record_choices", "id": "c1",
         "args": {"tool_call_id": "c1", "situation": "deadlines all week"}}
    )

    assert result.update["situation"] == "deadlines all week"


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


@pytest.mark.parametrize("text", ["that is really not helpful at all", "sorry, not what I need"])
def test_a_rejection_with_a_few_filler_words_is_still_recognised(text):
    assert says_not_helpful(text) is True


@pytest.mark.parametrize(
    "text",
    [
        "no its not useful to add it if its already there",  # a real turn
        "not useful to log it",
        "not helpful but the walk was",
        "das hilft nicht weil ich keine zeit habe",
        "not helpful, I already tried that one yesterday at work",  # no reason word, just long
    ],
)
def test_a_short_message_with_a_reason_goes_to_the_model(text):
    """Under the character limit, but it says why -- here the user liked the
    suggestion and only declined to add a duplicate. The shortcut answered it
    with the goal question; no model read it."""
    assert len(text) <= NOT_HELPFUL_MAX_CHARS
    assert says_not_helpful(text) is False


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
        "close_idea",
        "count_rejection",
        "debate",
        "follow_up",
        "listen",
        "nothing_to_save",
        "review",
        "suggest",
        "tools",
        "understand",
        "web_search_ideas",
    ]


def test_the_conditional_edges_go_exactly_where_the_design_says(agent):
    """The routing is the design. An edge added by accident (a station reaching
    the search node directly, say) is a path no test would otherwise walk."""
    edges = {(e.source, e.target) for e in agent.get_graph().edges if e.conditional}
    stations = ("ask", "suggest", "follow_up", "debate")
    assert edges == {("understand", s) for s in (*stations, "count_rejection", "listen", "nothing_to_save")} | {
        (s, t) for s in stations for t in ("review", "tools", "checkin_done", "__end__")
    } | {("tools", t) for t in (*stations, "web_search_ideas", "ask_goal_again", "close_idea")} | {
        ("count_rejection", "web_search_ideas"),
        ("count_rejection", "ask_goal_again"),
    } | {
        # Never a stop of its own: a save with nothing to save goes on to a station.
        ("nothing_to_save", s) for s in stations
    }


def test_the_dead_ends_really_end_and_a_review_always_reaches_the_tools(agent):
    """The canned nodes reply and stop; a loop back would cost a model
    call to say the same sentence again. The review has exactly one way on:
    to the tools node, which is where an unapproved write is refused."""
    plain = {(e.source, e.target) for e in agent.get_graph().edges if not e.conditional}
    assert plain == {(n, "__end__") for n in ("ask_goal_again", "web_search_ideas", "checkin_done", "close_idea")} | {
        ("review", "tools"),
        ("__start__", "understand"),
        ("listen", "__end__"),
    }


def test_the_recursion_limit_is_configured(agent):
    """The backstop against a loop that would otherwise run up a bill."""
    assert agent.config["recursion_limit"] == 25
    assert build_agent(model=StubChatModel(responses=[]), recursion_limit=7).config["recursion_limit"] == 7


# -- what each station may reach ---------------------------------------------


def test_the_ask_station_can_read_the_list_but_never_write(agent):
    """Not a rule in a prompt: no write tool is bound there. Reading is given
    on purpose -- a user who does not know what would help asked "what would
    you suggest?" four times and got the same three options back, because
    ask had no way to look in their list (third hand test)."""
    from agent.graph import WRITE_TOOLS

    assert sorted(agent.station_tools["ask"]) == [
        "discuss_idea", "find_measures", "record_choices", "search_measures"
    ]
    assert not WRITE_TOOLS & set(agent.station_tools["ask"])


def test_only_the_follow_up_station_can_write_to_the_log(agent):
    for station in ("ask", "suggest", "debate"):
        assert "log_checkin" not in agent.station_tools[station]
    assert "log_checkin" in agent.station_tools["follow_up"]


def test_the_debate_station_can_keep_an_idea_and_check_the_list(agent):
    """Asked "is this already in my list?", the debate could only say it could
    not see the list -- while the walk it was discussing was in it."""
    assert sorted(agent.station_tools["debate"]) == [
        "add_measure", "find_measures", "idea_decided", "record_choices", "search_measures"
    ]


def test_the_search_station_binds_no_local_tool_at_all(agent):
    """Two reasons. It reads untrusted web pages, so it must not be able to
    trigger a lasting effect. And its only edge is to END, so a call made here
    is never answered: gpt-5-mini called suggestion_not_helpful in a hand test,
    and the unanswered call broke every later turn with a 400."""
    assert agent.station_tools["web_search_ideas"] == []
    assert ("web_search_ideas", "tools") not in {
        (e.source, e.target) for e in agent.get_graph().edges
    }


def test_the_stations_bind_only_their_own_tools(agent, measures_file):
    """The binding matrix is the design: everything else is commentary."""
    model = StubChatModel(responses=[AIMessage(content="hi")])
    graph = build_agent(model=model, measures_path=measures_file)
    graph.invoke({"messages": [user("I'm wired")]})
    bound = [{t.name for t in call if hasattr(t, "name")} for call in model.bind_calls]
    assert bound and bound[0] == {"record_choices", "discuss_idea", "find_measures", "search_measures"}


def test_the_recorded_tool_lists_are_the_ones_actually_bound(agent):
    """station_tools used to be a second, hand-written copy of the bindings,
    free to drift from them. Now both come from the same lists."""
    bound = [sorted(t.name for t in call if hasattr(t, "name")) for call in agent.stub.bind_calls]
    for station in ("ask", "follow_up", "debate"):
        assert sorted(agent.station_tools[station]) in bound, station


def test_a_user_who_does_not_know_the_goal_gets_measures_from_the_list(measures_file):
    """End to end: energy only, the model looks up, the result is the user's
    own list -- and nothing pauses, because a read needs no approval."""
    lookup = AIMessage(content="", tool_calls=[{"name": "find_measures", "args": {}, "id": "l1"}])
    model = StubChatModel(responses=[lookup, AIMessage(content="two from your list")])
    graph = build_agent(model=model, measures_path=measures_file)
    result = graph.invoke({"messages": [user("I don't know what would help")], "energy": "exhausted"})
    event = [e for e in result["tool_events"] if e["tool"] == "find_measures"][0]
    assert event["exit_code"] == 0 and "(any goal)" in event["detail"]
    assert result["offered"] and result["step"] == "ask"
    assert not result.get("__interrupt__")


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


def test_the_web_search_is_told_what_the_list_already_holds(agent, measures_file):
    """A hand test's web search suggested a short walk that was already in
    the list. The list goes in by code, read at search time: a row added a
    moment ago is in it too."""
    with measures_file.open("a", encoding="utf-8", newline="") as handle:
        csv.writer(handle, lineterminator="\n").writerow(
            ("cold_water", "recover", "Splash cold water on your face.", "wired", "minutes", "user")
        )
    agent.invoke(
        {
            "messages": [user("not helpful")],
            "energy": "wired",
            "goal": "calm down now",
            "offered": ["breathe"],
            "rejected": True,
        }
    )
    trailing = agent.stub.received_messages[-1][-1].content
    assert "Search the web now" in trailing
    for row in [*MEASURE_ROWS, (None, None, "Splash cold water on your face.")]:
        assert row[2] in trailing


def test_an_unrated_check_in_makes_the_station_ask_how_it_went(tmp_path, measures_file):
    """The trailing note used to cover only the answer ("if the user says how
    it went, call log_checkin"). Nothing asked the question, so after a
    check-in the model asked about trying the measure instead."""
    from agent.graph import build_agent
    from tools.checkin_log import append_checkin_row

    log_path = tmp_path / "log.csv"
    append_checkin_row(
        energy="wired", circumstances="A loud day.", measure_id="short_walk", path=log_path
    )
    model = StubChatModel(responses=[AIMessage(content="ok")] * 4)
    graph = build_agent(model=model, log_path=log_path, measures_path=measures_file)

    graph.invoke(
        {
            "messages": [user("thanks")],
            "energy": "wired",
            "goal": "calm down now",
            "offered": ["short_walk"],
            "step": "follow_up",
        }
    )

    trailing = model.received_messages[-1][-1].content
    assert "short_walk" in trailing
    assert "ask them in one short sentence whether it helped" in trailing


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


# -- a button click means a fresh look in the user's list ---------------------

CLICKED_AFTER_THE_STEP_BACK = {
    "messages": [user("I'm feeling exhausted and I want to plan recovery.")],
    "energy": "exhausted",
    "goal": "plan recovery",
    "situation": "a week of deadlines",
    "offered": ["breathe"],
}


def test_a_choice_click_goes_to_the_lookup_even_with_something_offered(agent):
    """Turn 7 of the first hand test: after the goal question, the goal button
    left `offered` set, the turn reached follow_up, and the model invented
    options instead of looking in the list. A typed answer never had this
    problem -- record_choices sends it to suggest -- but the buttons skip that
    tool, so the click has to carry the same rule itself."""
    result = agent.invoke({**CLICKED_AFTER_THE_STEP_BACK, "choices_clicked": True})
    assert result["step"] == "suggest"


def test_the_same_state_without_a_click_still_goes_to_follow_up(agent):
    """The other side of the boundary: a plain message after suggestions is
    feedback, and must not trigger a new lookup."""
    result = agent.invoke(dict(CLICKED_AFTER_THE_STEP_BACK))
    assert result["step"] == "follow_up"


def test_a_click_with_only_one_value_still_asks_for_the_other(agent):
    state = {**CLICKED_AFTER_THE_STEP_BACK, "goal": None, "choices_clicked": True}
    assert agent.invoke(state)["step"] == "ask"


def test_a_click_does_not_talk_past_an_open_idea(agent):
    state = {**CLICKED_AFTER_THE_STEP_BACK, "idea": "pause first", "choices_clicked": True}
    assert agent.invoke(state)["step"] == "debate"


def test_after_a_click_the_lookup_really_runs_in_the_users_list(measures_file):
    """Routing to suggest is half of it; the lookup has to happen there and
    bring measures not shown before. (That suggest *forces* the call is
    pinned in test_agent_tool_selection.py.)"""
    lookup = AIMessage(content="", tool_calls=[{"name": "find_measures", "args": {}, "id": "l1"}])
    model = StubChatModel(responses=[lookup, AIMessage(content="two ideas")])
    graph = build_agent(model=model, measures_path=measures_file)
    result = graph.invoke({**CLICKED_AFTER_THE_STEP_BACK, "choices_clicked": True})
    assert result["step"] == "suggest", "follow_up could run a lookup too, but only suggest forces one"
    event = [e for e in result["tool_events"] if e["tool"] == "find_measures"][0]
    assert event["exit_code"] == 0
    assert len(result["offered"]) > 1 and result["offered"][0] == "breathe"


def test_the_click_flag_is_never_carried_into_the_next_turn():
    """Carried, every later message would count as a click and trigger a new
    lookup instead of being heard as feedback."""
    assert "choices_clicked" not in next_turn_state({"messages": [], "choices_clicked": True})


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
