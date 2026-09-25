"""The end of a discussed idea, and the turn after the goal question.

Both replace a model's free answer with something code decides, because a
hand test caught the model contradicting what had just happened:

- after an approved add_measure, `ask` said it "can't directly add items";
- asked to "pick one for me", `ask` offered box breathing from its own head
  although the user's list has breathing measures.
"""
import csv
from typing import Any

import pytest
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langgraph.checkpoint.memory import InMemorySaver
from pydantic import Field

from agent.graph import (
    REJECTED_TEXT,
    build_agent,
    goal_question_unanswered,
    idea_closing,
    station_for,
)
from agent.prompts import (
    IDEA_ADD_MISSING,
    IDEA_ADDED_LINE,
    IDEA_ALREADY_LISTED_LINE,
    IDEA_DROPPED,
    IDEA_NOT_APPROVED_LINE,
    IDEA_TRY,
    IDEA_TRY_WHAT_WAS_ADDED,
)
from tests.conftest import StubChatModel, invoke_approving
from tools.checkin_log import append_checkin_row, read_checkin_rows
from tools.measures import MEASURE_COLUMNS, read_measures

MEASURE_ROWS = [
    ("deep_breaths", "recover", "Take a few deep breaths.", "exhausted;wired", "minutes", "user"),
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


def add_call(measure_id="box_breathing"):
    return {
        "name": "add_measure",
        "id": "add1",
        "args": {
            "measure_id": measure_id,
            "measure": "Breathe in a box: in 4, hold 4, out 4, hold 4.",
            "measure_type": "recover",
            "fits_energy": ["exhausted"],
            "effort": "minutes",
            "from_user": True,
        },
    }


def decided(outcome):
    return {"name": "idea_decided", "id": "dec1", "args": {"outcome": outcome}}


def decide_turn(measures_file, tmp_path, *calls, decide=None):
    """One debate turn in which the model asks for `calls`, with no goal set --
    the state of the hand test, where the graph then fell back to `ask`."""
    model = StubChatModel(
        responses=[AIMessage(content="", tool_calls=list(calls)), AIMessage(content="MODEL SPOKE AGAIN")]
    )
    graph = build_agent(
        model=model, log_path=tmp_path / "log.csv", measures_path=measures_file, checkpointer=InMemorySaver()
    )
    state = {
        "messages": [HumanMessage(content="yes, add it to my list")],
        "energy": "exhausted",
        "idea": "box breathing",
        "step": "debate",
    }
    extra = {"decide": decide} if decide else {}
    return invoke_approving(graph, state, **extra), model


# -- closing an idea ---------------------------------------------------------

BOX = "Breathe in a box: in 4, hold 4, out 4, hold 4"  # add_call's text, without the full stop


def added_text(*measures):
    return "\n".join([*(IDEA_ADDED_LINE.format(measure=m) for m in measures), IDEA_TRY_WHAT_WAS_ADDED])


def test_an_approved_add_is_confirmed_by_code_not_by_a_model(measures_file, tmp_path):
    result, model = decide_turn(measures_file, tmp_path, add_call(), decided("add"))
    assert result["messages"][-1].content == added_text(BOX)
    assert model.call_count == 1  # no station got to answer after the tools
    assert "box_breathing" in {r["measure_id"] for r in read_measures(measures_file)}


def test_a_rejected_add_names_it_and_says_it_was_the_users_choice(measures_file, tmp_path):
    result, _ = decide_turn(
        measures_file, tmp_path, add_call(), decided("add"), decide=lambda r: {"type": "reject"}
    )
    assert result["messages"][-1].content == IDEA_NOT_APPROVED_LINE.format(measure=BOX)
    assert "box_breathing" not in {r["measure_id"] for r in read_measures(measures_file)}


def test_a_taken_name_is_explained_in_plain_words(measures_file, tmp_path):
    """The hand test: "Not added: five_min_mental_reset already exists. Pick
    anothe..." was the only explanation, cut off and written for the model."""
    result, _ = decide_turn(measures_file, tmp_path, add_call("deep_breaths"), decided("add"))
    text = result["messages"][-1].content
    assert text == IDEA_ALREADY_LISTED_LINE.format(measure=BOX)
    assert "note" not in text and "deep_breaths" not in text


def test_an_edited_measure_is_named_as_the_user_wrote_it(measures_file, tmp_path):
    def edit(request):
        return {"type": "edit", "args": {**add_call()["args"], "measure": "Box breathing, one minute."}}

    result, _ = decide_turn(measures_file, tmp_path, add_call(), decided("add"), decide=edit)
    assert result["messages"][-1].content == added_text("Box breathing, one minute")


def test_add_decided_without_an_add_measure_call_claims_nothing(measures_file, tmp_path):
    result, _ = decide_turn(measures_file, tmp_path, decided("add"))
    assert result["messages"][-1].content == IDEA_ADD_MISSING


@pytest.mark.parametrize("outcome,text", [("try", IDEA_TRY), ("drop", IDEA_DROPPED)])
def test_try_and_drop_get_their_own_text(measures_file, tmp_path, outcome, text):
    result, model = decide_turn(measures_file, tmp_path, decided(outcome))
    assert result["messages"][-1].content == text
    assert model.call_count == 1


def test_the_closing_hands_the_next_turn_to_follow_up(measures_file, tmp_path):
    """What usually follows is "it worked", and log_checkin lives in follow_up.
    With no station active the next turn went to `ask`, which answered a user
    who had just tried a measure with the goal question."""
    result, _ = decide_turn(measures_file, tmp_path, decided("drop"))
    assert result["step"] == "follow_up"
    assert result["idea"] is None
    assert station_for(dict(result, messages=[])) == "follow_up"


def _added(call_id="add1", **kwargs):
    return ToolMessage(content="Added it.", tool_call_id=call_id, name="add_measure", **kwargs)


def _decided_add():
    return [
        AIMessage(content="", tool_calls=[decided("add")]),
        ToolMessage(content="Decided: add.", tool_call_id="dec1", name="idea_decided"),
    ]


def _second(measure="Walk for five minutes."):
    call = add_call("other")
    return {**call, "id": "add2", "args": {**call["args"], "measure": measure}}


def test_an_add_in_an_earlier_message_of_the_same_turn_counts():
    """The hand test: two adds in one message, idea_decided in the next. Reading
    only the last batch said "Nothing was added" after both rows were written."""
    turn = [
        HumanMessage(content="log both"),
        AIMessage(content="", tool_calls=[add_call(), _second()]),
        _added("add1"),
        _added("add2"),
        *_decided_add(),
    ]
    assert idea_closing(turn) == added_text(BOX, "Walk for five minutes")


def test_an_add_from_an_earlier_turn_does_not_count():
    """An add_measure from an earlier idea must not make this one "added"."""
    history = [
        HumanMessage(content="add the first idea"),
        AIMessage(content="", tool_calls=[add_call()]),
        _added(),
        HumanMessage(content="and this one too"),
        *_decided_add(),
    ]
    assert idea_closing(history) == IDEA_ADD_MISSING


def test_one_added_one_taken_names_both():
    """The screenshot's case: one row written, one refused for its name."""
    turn = [
        HumanMessage(content="add all of them"),
        AIMessage(content="", tool_calls=[add_call(), _second()]),
        ToolMessage(
            content="Not added: other already exists. Pick another id or reuse that measure.",
            tool_call_id="add1", name="add_measure", status="error",
        ),
        _added("add2"),
        *_decided_add(),
    ]
    assert idea_closing(turn) == "\n".join([
        IDEA_ALREADY_LISTED_LINE.format(measure=BOX),
        IDEA_ADDED_LINE.format(measure="Walk for five minutes"),
        IDEA_TRY_WHAT_WAS_ADDED,
    ])


def test_one_of_two_rejected_on_the_card():
    turn = [
        HumanMessage(content="add both"),
        AIMessage(content="", tool_calls=[add_call(), _second()]),
        _added("add1"),
        ToolMessage(content=REJECTED_TEXT, tool_call_id="add2", name="add_measure", status="error"),
        *_decided_add(),
    ]
    assert IDEA_NOT_APPROVED_LINE.format(measure="Walk for five minutes") in idea_closing(turn)


def test_add_and_decide_in_separate_messages_through_the_graph(measures_file, tmp_path):
    """The same case end to end, with the approval pause in between."""
    model = StubChatModel(
        responses=[
            AIMessage(content="", tool_calls=[add_call()]),
            AIMessage(content="", tool_calls=[decided("add")]),
            AIMessage(content="MODEL SPOKE AGAIN"),
        ]
    )
    graph = build_agent(
        model=model, log_path=tmp_path / "log.csv", measures_path=measures_file, checkpointer=InMemorySaver()
    )
    state = {"messages": [HumanMessage(content="add it")], "energy": "exhausted", "idea": "box", "step": "debate"}
    result = invoke_approving(graph, state)
    assert result["messages"][-1].content == added_text(BOX)


# -- the turn after the goal question ----------------------------------------


class ChoiceRecordingStub(StubChatModel):
    """Records which tool_choice each model call was bound with. The shared
    stub returns itself from every bind, so it cannot tell bindings apart."""

    bound_choice: Any = None
    invoked_with: list = Field(default_factory=list)

    def bind_tools(self, tools: Any, *, tool_choice: Any = None, **kwargs: Any):
        super().bind_tools(tools, tool_choice=tool_choice)
        return self.model_copy(update={"bound_choice": tool_choice})

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        self.invoked_with.append(self.bound_choice)
        return super()._generate(messages, stop=stop, run_manager=run_manager, **kwargs)


def ask_turn(measures_file, tmp_path, **state):
    model = ChoiceRecordingStub(responses=[AIMessage(content="ok")])
    graph = build_agent(model=model, log_path=tmp_path / "log.csv", measures_path=measures_file)
    graph.invoke({"messages": [HumanMessage(content=state.pop("text", "pick one for me"))], **state})
    return model.invoked_with


def test_the_reply_to_the_goal_question_must_use_a_tool(measures_file, tmp_path):
    """"pick one for me": the goal recorded or the list looked up -- never an
    answer from the model's own head."""
    assert ask_turn(measures_file, tmp_path, energy="exhausted", step="ask") == ["required"]


def test_the_first_turn_is_free(measures_file, tmp_path):
    assert ask_turn(measures_file, tmp_path, text="hi") == [None]


def test_an_energy_click_before_any_question_still_gets_the_goal_question(measures_file, tmp_path):
    """No `ask` turn before it: the goal has not been asked yet."""
    assert ask_turn(measures_file, tmp_path, energy="exhausted") == [None]


def test_with_both_readings_known_the_station_still_has_to_act(measures_file, tmp_path):
    """Energy and goal are in, the situation is not, and `ask` now owns that
    question -- so the turn is forced, for the same reason the goal turn is: a
    model that answers "got it" without calling record_choices sends the next
    turn back to `ask` to ask again."""
    assert "required" in ask_turn(
        measures_file, tmp_path, energy="exhausted", goal="calm down now", step="ask"
    )


def test_with_all_three_known_ask_is_not_involved(measures_file, tmp_path):
    assert "required" not in ask_turn(
        measures_file, tmp_path, energy="exhausted", goal="calm down now",
        situation="a week of deadlines", step="ask",
    )


def test_after_a_tool_result_the_station_speaks_freely():
    """Forcing again after find_measures would loop instead of presenting it."""
    state = {
        "messages": [HumanMessage(content="pick one"), ToolMessage(content="[]", tool_call_id="c1")],
        "energy": "exhausted",
        "step": "ask",
    }
    assert goal_question_unanswered(state) is False
    state["messages"] = state["messages"][:1]
    assert goal_question_unanswered(state) is True


# -- a conversation that reached follow_up stays there ------------------------


def test_after_a_web_search_the_next_turn_can_write(measures_file, tmp_path):
    """web_search_ideas ends in follow_up. With no goal recorded the next turn
    used to go to `ask`, which has neither add_measure nor log_checkin: "add
    both of them" then only worked if the model took the discuss_idea detour,
    and once it just claimed it had added them."""
    model = StubChatModel(responses=[AIMessage(content="", tool_calls=[add_call()])])
    graph = build_agent(
        model=model, log_path=tmp_path / "log.csv", measures_path=measures_file, checkpointer=InMemorySaver()
    )
    after_search = {
        "messages": [HumanMessage(content="add both of them")],
        "energy": "exhausted",
        "offered": ["deep_breaths"],
        "rejections": 1,
        "step": "follow_up",
    }
    result = invoke_approving(graph, after_search)
    assert result["pauses"] == 1  # the card, not a detour through the debate
    assert "box_breathing" in {r["measure_id"] for r in read_measures(measures_file)}


def test_it_worked_reaches_the_station_that_can_log(measures_file, tmp_path):
    """"it worked both of it" was answered with the goal question, because the
    turn after the closing went to `ask`."""
    model = StubChatModel(responses=[AIMessage(content="how did it go?")])
    graph = build_agent(model=model, log_path=tmp_path / "log.csv", measures_path=measures_file)
    result = graph.invoke(
        {"messages": [HumanMessage(content="it worked")], "energy": "exhausted", "step": "follow_up"}
    )
    assert result["step"] == "follow_up"
    assert "log_checkin" in graph.station_tools["follow_up"]


def test_the_goal_question_still_comes_first_in_a_fresh_conversation(measures_file, tmp_path):
    """The new rule must not swallow the opening: nothing has been offered and
    no station is active, so the goal is still asked."""
    assert station_for({"energy": "exhausted"}) == "ask"
    assert station_for({"energy": "exhausted", "step": "ask", "offered": ["x"]}) == "ask"


# -- a rating typed as words reaches the card ---------------------------------


def rating_graph(measures_file, tmp_path, reply, checkpointer=True):
    """A log with one check-in waiting for its rating."""
    log = tmp_path / "log.csv"
    append_checkin_row(
        energy="exhausted", circumstances="Tired at work.", measure_id="deep_breaths", path=log
    )
    model = StubChatModel(responses=[reply])
    graph = build_agent(
        model=model,
        log_path=log,
        measures_path=measures_file,
        checkpointer=InMemorySaver() if checkpointer else None,
    )
    return graph, model, log


def test_helped_after_a_finished_conversation_reaches_the_card(measures_file, tmp_path):
    """A logged check-in ends the conversation, so "helped" arrives with an
    empty state and reached `ask`, which cannot log: the rating was lost."""
    rate = AIMessage(
        content="",
        tool_calls=[{
            "name": "log_checkin",
            "id": "r1",
            "args": {"measure_id": "deep_breaths", "helpfulness": "helped"},
        }],
    )
    graph, _, log = rating_graph(measures_file, tmp_path, rate)
    result = invoke_approving(graph, {"messages": [HumanMessage(content="helped")]})
    assert result["pauses"] == 1  # the card asked before the row was written
    assert read_checkin_rows(log)[-1]["helpfulness"] == "helped"


def test_the_station_is_told_which_measure_is_waiting(measures_file, tmp_path):
    """After the reset the transcript is gone; the log knows the measure."""
    graph, model, _ = rating_graph(
        measures_file, tmp_path, AIMessage(content="how did it go?"), checkpointer=False
    )
    graph.invoke({"messages": [HumanMessage(content="helped")]})
    sent = " ".join(str(m.content) for m in model.received_messages[-1])
    assert "deep_breaths" in sent


@pytest.mark.parametrize(
    "text,routed",
    [("helped", True), ("it worked", True), ("somewhat", True), ("didn't help", True),
     ("helped a lot, but I want to talk about my workload and what to change", False),
     ("what would help?", False)],
)
def test_only_a_short_verdict_takes_the_shortcut(text, routed):
    from agent.graph import says_rating

    assert bool(says_rating(text)) is routed


def test_a_rejection_still_wins_while_suggestions_are_on_screen(measures_file, tmp_path):
    """With suggestions shown, "didn't help" is about them, not about a measure
    waiting for a rating."""
    graph, _, _ = rating_graph(
        measures_file, tmp_path, AIMessage(content="ok"), checkpointer=False
    )
    result = graph.invoke(
        {"messages": [HumanMessage(content="didn't help")], "energy": "exhausted",
         "goal": "calm down now", "offered": ["deep_breaths"], "rejections": 1}
    )
    assert "step back" in result["messages"][-1].content
