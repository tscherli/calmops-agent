"""What the model is allowed to reach, what it is forced to reach, and how
many times it is called.

The binding matrix itself is tests/test_stations.py's job. This module covers
the three things around it that fail silently:

- **forcing.** `suggest` must call find_measures before it may present
  anything, so a suggestion can never bypass the user's own list. A dropped
  `tool_choice` looks exactly like a model that happened to comply.
- **the kill switch.** With web search off, the tool must be unbound
  *everywhere* and `web_search_ideas` must also stop forcing it -- forcing a
  tool that is not bound is a 400 from the provider, i.e. a broken turn, at
  the exact moment the user has just said nothing helped.
- **the model-call budget.** If a tool's state update is lost, the graph
  loops station -> tools -> station until the recursion limit, and the user
  sees only the same question again. Counting calls is what makes that loud.
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
]


@pytest.fixture
def measures_file(tmp_path):
    path = tmp_path / "measures.csv"
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(MEASURE_COLUMNS)
        writer.writerows(MEASURE_ROWS)
    return path


def user(text):
    return HumanMessage(content=text)


def _ready(**extra):
    """A state that routes straight to suggest: both answers in, nothing shown."""
    return {"energy": "wired", "goal": "calm down now", "offered": [], **extra}


def _names(tools):
    return {getattr(t, "name", None) or t.get("type") for t in tools}


# -- forcing the lookup ------------------------------------------------------


def test_exactly_one_binding_forces_the_measure_lookup(measures_file):
    """Without a fresh result the model does not get the option of skipping
    the user's own list -- which is the whole reason the list exists.

    Asserted on the bindings rather than on a turn, because every binding is
    built once in build_agent and this stub returns itself from bind_tools,
    so which bound object a station invoked is not observable from here.
    """
    model = StubChatModel(responses=[AIMessage(content="ok")])
    build_agent(model=model, measures_path=measures_file)

    assert model.tool_choices.count("find_measures") == 1


def test_the_lookup_is_not_repeated_once_a_result_is_on_the_table(measures_file):
    """Still forcing after the lookup and the station would call it forever;
    every turn would also cost two model calls instead of one."""
    lookup = AIMessage(
        content="",
        tool_calls=[{"name": "find_measures", "args": {}, "id": "c1", "type": "tool_call"}],
    )
    model = StubChatModel(responses=[lookup, AIMessage(content="try a short walk")])
    graph = build_agent(model=model, measures_path=measures_file)

    result = graph.invoke({**_ready(), "messages": [user("what should I do")]})

    lookups = [
        call
        for message in result["messages"]
        for call in (getattr(message, "tool_calls", None) or [])
        if call["name"] == "find_measures"
    ]
    assert len(lookups) == 1
    assert model.call_count == 2


# -- the kill switch ---------------------------------------------------------


def test_web_search_is_bound_where_it_is_allowed_by_default(measures_file):
    model = StubChatModel(responses=[AIMessage(content="ok")])
    graph = build_agent(model=model, measures_path=measures_file)

    graph.invoke({**_ready(), "messages": [user("what should I do")]})

    assert "web_search" in _names(model.bind_calls[-1])


def test_the_kill_switch_unbinds_web_search_everywhere(monkeypatch, measures_file):
    monkeypatch.setenv("OPENAI_ENABLE_WEB_SEARCH", "false")
    model = StubChatModel(responses=[AIMessage(content="ok")])
    graph = build_agent(model=model, measures_path=measures_file)

    graph.invoke({**_ready(), "messages": [user("what should I do")]})

    assert all("web_search" not in _names(call) for call in model.bind_calls)


def test_the_kill_switch_also_stops_the_search_station_forcing_it(monkeypatch, measures_file):
    """Forcing a tool that is not bound is a 400. The turn this would break
    is the one right after the user said the suggestion did not help."""
    monkeypatch.setenv("OPENAI_ENABLE_WEB_SEARCH", "false")
    model = StubChatModel(responses=[AIMessage(content="here is another idea")])
    graph = build_agent(model=model, measures_path=measures_file)

    result = graph.invoke(
        {**_ready(offered=["short_walk"]), "rejected": True, "messages": [user("no")]}
    )

    assert all(choice != {"type": "web_search"} for choice in model.tool_choices)
    assert "here is another idea" in str(result["messages"][-1].content)


# -- how often the model is called -------------------------------------------


def test_a_plain_turn_costs_exactly_one_model_call(measures_file):
    model = StubChatModel(responses=[AIMessage(content="how are you doing?")])
    graph = build_agent(model=model, measures_path=measures_file)

    graph.invoke({"messages": [user("hello")]})

    assert model.call_count == 1


def test_a_turn_that_uses_a_tool_stays_within_three_model_calls(measures_file):
    """The guard on the silent failure that costs the most: if a tool's state
    update is dropped, the station asks for the same tool again and the graph
    spins to the recursion limit while the user watches one question repeat."""
    record = AIMessage(
        content="",
        tool_calls=[
            {
                "name": "record_choices",
                "args": {"energy": "wired", "goal": "calm down now"},
                "id": "c1",
                "type": "tool_call",
            }
        ],
    )
    lookup = AIMessage(
        content="",
        tool_calls=[{"name": "find_measures", "args": {}, "id": "c2", "type": "tool_call"}],
    )
    model = StubChatModel(responses=[record, lookup, AIMessage(content="try a short walk")])
    graph = build_agent(model=model, measures_path=measures_file)

    result = graph.invoke({"messages": [user("I'm wired and need to calm down")]})

    assert model.call_count <= 3
    assert result["energy"] == "wired"


def test_a_canned_node_adds_no_model_call(measures_file):
    """count_rejection and the step-back question are code. A rejection must
    not cost what an answer costs."""
    model = StubChatModel(responses=[AIMessage(content="should not be reached")])
    graph = build_agent(model=model, measures_path=measures_file)

    result = graph.invoke(
        {
            **_ready(offered=["short_walk"], rejections=1),
            "rejected": True,
            "messages": [user("nope")],
        }
    )

    assert model.call_count == 0
    # The second rejection steps back and starts the count over, so the
    # question itself is the evidence -- not the counter, which is now 0.
    assert result["rejections"] == 0
    assert result["goal"] is None
    assert "step back" in str(result["messages"][-1].content).lower()
