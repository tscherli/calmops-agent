r"""The station graph: small nodes, each with its own short prompt and only its
own tools.

    START
      route_start (plain function, no model call)
        |- the user rejected a suggestion            -> count_rejection
        |- an idea of theirs is open                 -> debate
        |- energy or goal missing                    -> ask
        |- nothing suggested yet                     -> suggest
        \- otherwise                                 -> follow_up

    ask | suggest | follow_up | debate --(tool calls)--> tools --> back to the
                                         station, or to the rejection path
                                       --(no tool calls)--> checkin_done | END

    count_rejection -> 1st: web_search_ideas -> END
                       2nd: ask_goal_again   -> END

What a station is not given, it cannot do: `ask` has no way to suggest a
measure, because the tool is not bound there. That is the difference between a
rule the model may ignore and one it cannot reach.

The graph owns what the model would otherwise have to re-derive from the
transcript every turn: the user's energy and goal, which measures were already
shown, how many suggestions were rejected in a row, which station is active,
and whether an idea of the user's is under discussion.
"""
from __future__ import annotations

import os
from collections.abc import Sequence
from pathlib import Path
from typing import Annotated, Any, Literal, TypedDict

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, SystemMessage, ToolMessage
from langchain_core.tools import BaseTool, InjectedToolCallId, tool
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import InjectedState
from langgraph.types import Command

from agent.common import (
    as_tool_content,
    compact_spent_tool_messages,
    merge_tool_events,
    message_text,
    pending_tool_calls,
    result_detail,
    tool_event,
    web_search_ran,
)
from agent.model import (
    model_name_of,
    resolve_debate_model,
    resolve_model,
    supports_reasoning_effort,
)
from agent.prompts import (
    ASK_GOAL_AGAIN,
    SEARCH_INSTRUCTION,
    STATION_PROMPTS,
    debate_prompt,
)
from tools.checkin_log import make_log_checkin
from tools.measures import Energy, Goal, make_add_measure, make_find_measures

DEFAULT_RECURSION_LIMIT = 25
WEB_SEARCH_TOOL: dict[str, str] = {"type": "web_search"}

# The debate station runs on the stronger model, so it ends in a decision
# rather than drifting: after this many turns it asks the user to choose.
DEBATE_TURN_CAP = 6

# Reasoning is worth paying for where the job is judgement; the canned nodes
# make no model call at all.
STATION_REASONING_EFFORT: dict[str, str | None] = {
    "ask": "low",
    "suggest": "low",
    "follow_up": "low",
    "debate": "low",
    "web_search_ideas": "low",
}

# Short, unambiguous rejections are recognised in code, before any model call.
# A longer message usually carries something else as well -- the model reads
# that one and can call suggestion_not_helpful itself.
NOT_HELPFUL_MAX_CHARS = 60
NOT_HELPFUL_PHRASES = (
    "not helpful",
    "doesn't help",
    "does not help",
    "didn't help",
    "did not help",
    "not useful",
    "not feel helpful",
    "not really helpful",
    "not that helpful",
    "not what i need",
    "hilft nicht",
    "nicht hilfreich",
    "bringt nichts",
)

Station = Literal["ask", "suggest", "follow_up", "debate"]


class State(TypedDict, total=False):
    """What the graph remembers, so the model does not have to re-derive it."""

    messages: Annotated[list[BaseMessage], add_messages]
    energy: Energy | None
    goal: Goal | None
    offered: list[str]          # measure ids already shown in this conversation
    rejections: int             # suggestions rejected in a row
    step: Station | None        # the active station
    idea: str | None            # an idea of the user's, while it is discussed
    debate_turns: int
    rejected: bool              # set by the UI's "not helpful" button
    checkin_logged: bool        # a check-in row was written this turn
    last_measure_id: str        # the measure that check-in row named, for a later rating
    complete: bool              # the conversation reached its end
    tool_events: Annotated[list[dict[str, Any]], merge_tool_events]


# -- tools that only touch the state -----------------------------------------


def state_tool_reply(text: str, tool_call_id: str, name: str, **update: Any) -> Command:
    """A tool result plus the state it writes. The receipt is set by the tools
    node, which is the one place the model cannot reach."""
    return Command(
        update={**update, "messages": [ToolMessage(text, tool_call_id=tool_call_id, name=name)]}
    )


@tool
def record_choices(
    energy: Energy,
    goal: Goal,
    tool_call_id: Annotated[str, InjectedToolCallId],
) -> Command:
    """Record the user's energy and goal, or change either when they say so.

    energy: how they feel right now - exhausted, calm or wired.
    goal: what would help them most - "calm down now" (a few minutes),
    "plan recovery" (something for later, hours or days), or
    "change the situation" (act on the cause of the stress).
    Only use what the user actually said; if either is missing, ask instead.
    """
    return state_tool_reply(
        f"Noted: {energy}, {goal}.", tool_call_id, "record_choices", energy=energy, goal=goal
    )


@tool
def suggestion_not_helpful(
    state: Annotated[dict, InjectedState],
    tool_call_id: Annotated[str, InjectedToolCallId],
) -> Command:
    """Call this when the user says a suggestion didn't help or isn't what they need."""
    count = int(state.get("rejections") or 0) + 1
    return state_tool_reply(
        f"Noted: rejection {count}.", tool_call_id, "suggestion_not_helpful", rejections=count
    )


@tool
def discuss_idea(idea: str, tool_call_id: Annotated[str, InjectedToolCallId]) -> Command:
    """Call this when the user brings an idea of their own for what could help.

    idea: their idea in one short sentence, close to their own words.
    """
    return state_tool_reply(
        "Noted the user's idea; discussing it now.", tool_call_id, "discuss_idea", idea=idea
    )


@tool
def idea_decided(
    outcome: Literal["add", "try", "drop"],
    tool_call_id: Annotated[str, InjectedToolCallId],
) -> Command:
    """Call this once the user has decided what to do with their own idea."""
    return state_tool_reply(
        f"Decided: {outcome}.", tool_call_id, "idea_decided", idea=None, debate_turns=0
    )


# -- plain functions ---------------------------------------------------------


def web_search_enabled() -> bool:
    return os.environ.get("OPENAI_ENABLE_WEB_SEARCH", "true").strip().lower() not in {
        "0",
        "false",
        "no",
    }


def says_not_helpful(text: str) -> bool:
    """Is this message nothing but a rejection?

    The length limit matters as much as the phrases. A long message that
    happens to contain "not helpful" usually carries something new as well,
    and taking the shortcut would mean the model never reads it -- which is
    exactly what happened in a real conversation before the limit existed.
    """
    stripped = (text or "").strip()
    if len(stripped) > NOT_HELPFUL_MAX_CHARS:
        return False
    lowered = stripped.lower().replace("\u2019", "'")
    return any(phrase in lowered for phrase in NOT_HELPFUL_PHRASES)


def station_for(state: State) -> Station:
    """Which station fits the state.

    An open idea comes first: the user brought something of their own, and
    every other station would talk past it.
    """
    if state.get("idea"):
        return "debate"
    if not (state.get("energy") and state.get("goal")):
        return "ask"
    if not state.get("offered"):
        return "suggest"
    return "follow_up"


def last_human_text(messages: Sequence[BaseMessage]) -> str:
    for message in reversed(messages):
        if message.__class__.__name__ == "HumanMessage":
            return message_text(message)
    return ""


CARRIED_FIELDS = ("energy", "goal", "offered", "rejections", "step", "idea", "debate_turns")


def next_turn_state(result: dict[str, Any]) -> dict[str, Any]:
    """What a client carries into the next turn.

    One function for both the app and the eval harness, because they
    implemented this separately once and drifted, which produced a whole eval
    run of meaningless numbers. On completion the station fields are cleared
    too: clearing the messages alone would leave `offered` set, and the next
    sentence would land in follow_up with an empty transcript.
    """
    if result.get("complete"):
        return {
            "messages": [],
            "energy": None,
            "goal": None,
            "offered": [],
            "rejections": 0,
            "step": None,
            "idea": None,
            "debate_turns": 0,
        }
    carried: dict[str, Any] = {
        "messages": compact_spent_tool_messages(
            result.get("messages", []), tool_names=("find_measures",)
        )
    }
    carried.update({key: result.get(key) for key in CARRIED_FIELDS})
    return carried


# -- the graph ---------------------------------------------------------------


def build_agent(
    model: str | BaseChatModel | None = None,
    tools: Sequence[BaseTool] | None = None,
    *,
    debate_model: str | BaseChatModel | None = None,
    log_path: Path | str | None = None,
    measures_path: Path | str | None = None,
    recursion_limit: int = DEFAULT_RECURSION_LIMIT,
):
    """Build the station graph.

    model/tools/log_path/measures_path are injectable so tests and eval runs
    exercise the real wiring without OpenAI, the real log or the real measure
    list.
    """
    base_model = resolve_model(model)
    debate_base = resolve_debate_model(debate_model) if debate_model is not None else None
    search_on = web_search_enabled()

    find_measures = make_find_measures(measures_path)
    add_measure = make_add_measure(measures_path)
    log_checkin = make_log_checkin(log_path, measures_path)

    all_tools: list[BaseTool] = list(tools) if tools is not None else [
        record_choices,
        find_measures,
        add_measure,
        log_checkin,
        suggestion_not_helpful,
        discuss_idea,
        idea_decided,
    ]
    by_name: dict[str, BaseTool] = {t.name: t for t in all_tools}

    def pick(*names: str) -> list[Any]:
        """The tools a station may reach, plus web search where it is allowed."""
        if tools is not None:
            return list(all_tools)
        return [by_name[n] for n in names if n in by_name]

    def bind(station: str, bound: Sequence[Any], *, force: Any = None, use_debate: bool = False):
        chosen = debate_base if (use_debate and debate_base is not None) else base_model
        effort = STATION_REASONING_EFFORT.get(station)
        name = model_name_of(chosen)
        if effort and name and supports_reasoning_effort(name):
            try:
                chosen = chosen.bind(reasoning_effort=effort)
            except (AttributeError, TypeError):
                pass
        if not bound:
            return chosen
        if force is not None:
            return chosen.bind_tools(list(bound), tool_choice=force)
        return chosen.bind_tools(list(bound))

    web = [WEB_SEARCH_TOOL] if search_on else []
    ask_model = bind("ask", pick("record_choices", "discuss_idea"))
    suggest_tools = pick("find_measures", "record_choices", "discuss_idea") + web
    suggest_model = bind("suggest", suggest_tools)
    lookup_model = bind("suggest", suggest_tools, force="find_measures")
    follow_up_model = bind(
        "follow_up",
        pick(
            "record_choices",
            "find_measures",
            "add_measure",
            "log_checkin",
            "suggestion_not_helpful",
            "discuss_idea",
        )
        + web,
    )
    debate_tools = pick("add_measure", "record_choices", "idea_decided")
    debate_station_model = bind("debate", debate_tools, use_debate=True)
    search_model = bind(
        "web_search_ideas",
        list(all_tools) + web,
        # Forcing a tool that is not bound is a 400, so the kill switch also
        # turns the forcing off.
        force=WEB_SEARCH_TOOL if search_on else None,
    )

    # -- stations ------------------------------------------------------------

    def _reply(chosen, prompt: str, state: State, *, trailing: str | None = None) -> AIMessage:
        messages: list[BaseMessage] = [SystemMessage(content=prompt), *state["messages"]]
        if trailing:
            messages.append(SystemMessage(content=trailing))
        return chosen.invoke(messages)

    def ask(state: State) -> dict[str, Any]:
        """Find out energy and goal. Cannot suggest: the tool is not bound here."""
        return {"messages": [_reply(ask_model, STATION_PROMPTS["ask"], state)], "step": "ask"}

    def suggest(state: State) -> dict[str, Any]:
        """Look up measures, then present at most two.

        Without a fresh lookup the model is forced to call find_measures, so a
        suggestion can never skip the user's own list; choosing which two to
        present stays with the model.
        """
        last = state["messages"][-1] if state["messages"] else None
        has_result = isinstance(last, ToolMessage) and last.name == "find_measures"
        chosen = suggest_model if has_result else lookup_model
        return {"messages": [_reply(chosen, STATION_PROMPTS["suggest"], state)], "step": "suggest"}

    def follow_up(state: State) -> dict[str, Any]:
        """Rejection, more ideas, adding a measure, and the log."""
        response = _reply(follow_up_model, STATION_PROMPTS["follow_up"], state)
        update: dict[str, Any] = {"messages": [response], "step": "follow_up"}
        if web_search_ran(response):
            update["tool_events"] = [tool_event("web_search", "ok", "provider-reported", verified=False)]
        return update

    def debate(state: State) -> dict[str, Any]:
        """An honest sparring partner for an idea the user brought themselves."""
        turns = int(state.get("debate_turns") or 0) + 1
        prompt = debate_prompt(state.get("idea") or "", nudge=turns >= DEBATE_TURN_CAP)
        chosen = debate_station_model
        return {
            "messages": [_reply(chosen, prompt, state)],
            "step": "debate",
            "debate_turns": turns,
        }

    def web_search_ideas(state: State) -> dict[str, Any]:
        """After the first rejection: a model call that has to search."""
        response = _reply(
            search_model, STATION_PROMPTS["suggest"], state, trailing=SEARCH_INSTRUCTION
        )
        update: dict[str, Any] = {"messages": [response], "step": "follow_up"}
        if web_search_ran(response):
            update["tool_events"] = [tool_event("web_search", "ok", "provider-reported", verified=False)]
        return update

    def ask_goal_again(state: State) -> dict[str, Any]:
        """After the second rejection: a fixed question, no model call.

        The goal is cleared so the ask station really asks again instead of
        reading the old one out of the transcript, and the counter starts over.
        """
        return {
            "messages": [AIMessage(content=ASK_GOAL_AGAIN)],
            "goal": None,
            "rejections": 0,
            "step": "ask",
        }

    def count_rejection(state: State) -> dict[str, Any]:
        """Code counts the rejection; no model call is needed to notice a "no"."""
        return {"rejections": int(state.get("rejections") or 0) + 1, "rejected": False}

    def checkin_done(state: State) -> dict[str, Any]:
        """The conversation reached its end; clients start the next one fresh."""
        return {"complete": True, "checkin_logged": False}

    # -- the tools node ------------------------------------------------------

    def _invoke_tool(target: BaseTool, call: dict[str, Any], state: State) -> Any:
        """Run one tool the way the graph must: as a full tool call.

        A tool taking `tool_call_id` only accepts the whole call dict, because
        LangChain fills that argument itself. The state, which the prebuilt
        ToolNode would inject, is added here: `tool.args` is what the model may
        pass, `args_schema.model_fields` is everything the function takes, and
        the difference is exactly the injected arguments.
        """
        visible = set(target.args)
        fields = set(getattr(target.args_schema, "model_fields", {}) or {})
        args = dict(call.get("args") or {})
        if "state" in fields - visible:
            args["state"] = state
        return target.invoke(
            {"type": "tool_call", "name": call.get("name"), "id": call.get("id"), "args": args}
        )

    def tools_node(state: State) -> dict[str, Any]:
        """Run the pending tool calls and write one receipt for each.

        The receipts are set here, from what the tool actually returned -- the
        model can neither produce one nor suppress one, which is what makes a
        claim in its prose checkable.
        """
        outputs: list[BaseMessage] = []
        events: list[dict[str, Any]] = []
        update: dict[str, Any] = {}

        for call in pending_tool_calls(state["messages"]):
            name = call.get("name") or ""
            call_id = call.get("id") or ""
            target = by_name.get(name)
            if target is None:
                outputs.append(
                    ToolMessage(
                        content=f"Unknown tool: {name}", name=name, tool_call_id=call_id, status="error"
                    )
                )
                events.append(tool_event(name or "unknown", "error", "unknown tool"))
                continue
            try:
                result = _invoke_tool(target, call, state)
            except Exception as exc:  # noqa: BLE001 - a broken tool must not end the turn
                outputs.append(
                    ToolMessage(
                        content=f"Tool error: {exc}", name=name, tool_call_id=call_id, status="error"
                    )
                )
                events.append(tool_event(name, "error", str(exc)))
                continue

            if isinstance(result, Command):
                produced = dict(result.update or {})
                messages = produced.pop("messages", [])
                update.update(produced)
                outputs.extend(messages)
                receipt = {}
                for message in messages:
                    receipt = getattr(message, "additional_kwargs", {}).get("receipt", {}) or receipt
                events.append(
                    tool_event(
                        name,
                        receipt.get("status", "ok"),
                        receipt.get("detail", result_detail(messages[0].content if messages else "")),
                    )
                )
            else:
                outputs.append(
                    ToolMessage(
                        content=as_tool_content(result), name=name, tool_call_id=call_id, status="success"
                    )
                )
                events.append(tool_event(name, "ok", result_detail(result)))

        update["messages"] = outputs
        update["tool_events"] = events
        return update

    # -- routing -------------------------------------------------------------

    def after_rejection(state: State) -> str:
        """The first rejection buys a web search, the second a step back."""
        return "web_search_ideas" if int(state.get("rejections") or 0) == 1 else "ask_goal_again"

    def route_start(state: State) -> str:
        """Before any model call. A button click is a fact; a short "not
        helpful" is close enough to one; everything else goes to a station."""
        messages = state.get("messages") or []
        offered = state.get("offered")
        if state.get("rejected"):
            return "count_rejection"
        if offered and not state.get("idea") and says_not_helpful(last_human_text(messages)):
            return "count_rejection"
        return station_for(state)

    def after_tools(state: State) -> str:
        """Back to the station that asked, unless the tool changed the plan."""
        last = state["messages"][-1] if state["messages"] else None
        name = getattr(last, "name", None)
        if name == "suggestion_not_helpful":
            return after_rejection(state)
        if state.get("idea"):
            return "debate"
        if name == "record_choices":
            # New answers mean a fresh lookup rather than more of the same.
            return station_for(state) if station_for(state) == "ask" else "suggest"
        if name == "idea_decided":
            return station_for(state)
        return state.get("step") or station_for(state)

    def after_station(state: State) -> str:
        """A station either asked for tools, ended the check-in, or replied."""
        if pending_tool_calls(state["messages"]):
            return "tools"
        if state.get("checkin_logged"):
            return "checkin_done"
        return END

    graph = StateGraph(State)
    graph.add_node("ask", ask)
    graph.add_node("suggest", suggest)
    graph.add_node("follow_up", follow_up)
    graph.add_node("debate", debate)
    graph.add_node("tools", tools_node)
    graph.add_node("count_rejection", count_rejection)
    graph.add_node("web_search_ideas", web_search_ideas)
    graph.add_node("ask_goal_again", ask_goal_again)
    graph.add_node("checkin_done", checkin_done)

    graph.add_conditional_edges(
        START, route_start, ["count_rejection", "ask", "suggest", "follow_up", "debate"]
    )
    for name in ("ask", "suggest", "follow_up", "debate"):
        graph.add_conditional_edges(name, after_station, ["tools", "checkin_done", END])
    graph.add_conditional_edges(
        "tools",
        after_tools,
        ["ask", "suggest", "follow_up", "debate", "web_search_ideas", "ask_goal_again"],
    )
    graph.add_conditional_edges(
        "count_rejection", after_rejection, ["web_search_ideas", "ask_goal_again"]
    )
    graph.add_edge("web_search_ideas", END)
    graph.add_edge("ask_goal_again", END)
    graph.add_edge("checkin_done", END)

    compiled = graph.compile().with_config({"recursion_limit": recursion_limit})
    # Handy for tests and for try_graph_viz.py, which render the shape.
    compiled.station_tools = {  # type: ignore[attr-defined]
        "ask": [t.name for t in pick("record_choices", "discuss_idea")],
        "suggest": [t.name for t in pick("find_measures", "record_choices", "discuss_idea")],
        "follow_up": [
            t.name
            for t in pick(
                "record_choices",
                "find_measures",
                "add_measure",
                "log_checkin",
                "suggestion_not_helpful",
                "discuss_idea",
            )
        ],
        "debate": [t.name for t in debate_tools],
    }
    return compiled


__all__ = [
    "DEBATE_TURN_CAP",
    "DEFAULT_RECURSION_LIMIT",
    "NOT_HELPFUL_PHRASES",
    "State",
    "build_agent",
    "discuss_idea",
    "idea_decided",
    "next_turn_state",
    "record_choices",
    "says_not_helpful",
    "station_for",
    "suggestion_not_helpful",
    "web_search_enabled",
]
