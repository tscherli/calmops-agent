r"""The station graph: small nodes, each with its own short prompt and only its
own tools.

    START
      understand (one model call, only for a typed message code cannot place)
      route_start (plain function, no model call)
        |- the user rejected a suggestion            -> count_rejection
        |- a short verdict, a measure awaiting one   -> follow_up
        |- read as "not now"                         -> listen -> END
        |- read as a log request or a verdict        -> follow_up
        |- read as a rejection, something on screen  -> count_rejection
        |- read as save/rate, no measure known yet   -> nothing_to_save, then
        |                                               the station below
        |- an idea of theirs is open                 -> debate
        |- the conversation already reached follow_up-> follow_up
        |- energy or goal missing                    -> ask
        |- nothing suggested yet                     -> suggest
        \- otherwise                                 -> follow_up

    ask | suggest | follow_up | debate --(reads only)--> tools --> back to the
                                         station, or to the rejection path
                                       --(any write)--> review --> tools
                                       --(no tool calls)--> checkin_done | END

    tools --(idea_decided)--> close_idea -> END  (fixed text, from the results)

    count_rejection -> 1st: web_search_ideas -> END
                       2nd: ask_goal_again   -> END

What a station is not given, it cannot do: `ask` has no way to write
anything, because no write tool is bound there. That is the difference between
a rule the model may ignore and one it cannot reach. Reading the user's own
measure list is given freely -- it changes nothing -- so a suggestion can come
from the list even before the goal is known.

The same idea for side effects: the model may call a write tool, but the call
only runs once the user has approved it on a card. `review` pauses the graph
for that decision; the tools node refuses any write without one.

The graph owns what the model would otherwise have to re-derive from the
transcript every turn: the user's energy and goal, which measures were already
shown, how many suggestions were rejected in a row, which station is active,
and whether an idea of the user's is under discussion.
"""
from __future__ import annotations

import json
import os
import re
from collections.abc import Sequence
from pathlib import Path
from typing import Annotated, Any, Literal, TypedDict

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import BaseTool, InjectedToolCallId, tool
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import InjectedState
from langgraph.types import Command, interrupt
from pydantic import BaseModel, Field

from agent.common import (
    as_tool_content,
    compact_spent_tool_messages,
    drop_unanswered_tool_calls,
    last_ai_tool_calls,
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
    IDEA_ADD_MISSING,
    IDEA_ADDED_LINE,
    IDEA_ALREADY_LISTED_LINE,
    IDEA_DROPPED,
    IDEA_NOT_APPROVED_LINE,
    IDEA_NOT_SAVED_LINE,
    IDEA_TRY,
    IDEA_TRY_WHAT_WAS_ADDED,
    NOTHING_TO_SAVE_YET,
    READ_PROMPT,
    STATION_PROMPTS,
    debate_prompt,
    search_instruction,
)
from tools.checkin_log import make_log_checkin, measure_awaiting_rating
from tools.measures import (
    Energy,
    Goal,
    make_add_measure,
    make_find_measures,
    make_search_measures,
    read_measures,
)

DEFAULT_RECURSION_LIMIT = 25
WEB_SEARCH_TOOL: dict[str, str] = {"type": "web_search"}

# The tools whose effect outlives the conversation. The model may call them;
# they run only after the user approves on a card. Everything else -- reading
# measures, searching, recording energy and goal -- changes nothing that lasts,
# so it runs at once.
WRITE_TOOLS = frozenset({"add_measure", "log_checkin"})

# What the Edit button on the card changes, per write tool.
EDITABLE_ARG = {"add_measure": "measure", "log_checkin": "circumstances"}

REJECTED_TEXT = "Not done: the user did not approve this."

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
    "listen": "low",
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
# Besides the phrase, a bare rejection has a few words at most ("that's",
# "really", "sorry") and nothing that starts a reason or another request.
# "no its not useful to add it if its already there" fit the character limit
# and was taken for a rejection; it praised the suggestion. Missing the
# shortcut costs little -- the model reads the message and can reject itself.
NOT_HELPFUL_MAX_EXTRA_WORDS = 5
CARRIES_MORE_WORDS = frozenset(
    "but if since because to add log save".split()
)

# How a measure that is waiting for a rating can be rated in words. A check-in
# ends the conversation, so the next message starts a fresh one: "helped" then
# reached `ask`, which cannot log, and the rating was lost. Matching here only
# routes the turn -- the model still proposes the rating and the user approves
# it on the card.
RATING_PHRASES: dict[str, tuple[str, ...]] = {
    "helped": ("helped", "it worked", "that worked", "worked well"),
    "somewhat": ("somewhat", "a bit", "a little", "partly", "so-so"),
    "didnt_help": ("didn't help", "did not help", "didnt help", "no change", "nothing changed"),
}
RATING_MAX_CHARS = 40

Station = Literal["ask", "suggest", "follow_up", "debate", "listen"]

# What the user wants to happen next, as the understand node reads it. The
# routing defect of v1.0.0 was that nothing read the message before a station
# was chosen: in the release sweep the station model often said what a message
# meant ("I noted you said the ritual helped and the muscle one didn't") and
# then did what its station allowed. The reading happens before any station, in
# a call whose only job is reading.
#
# Wants, not kinds of message: the first reader returned kinds, which overlap,
# and a fixed priority chose between them without the context (stage 79).
Want = Literal[
    "log_checkin", "rate_measure", "other_suggestions", "own_idea",
    "just_chat", "checkin_answer", "pick_offered", "unclear",
]

# How much of the conversation the reader sees: enough for "yes" to have the
# question it answers, short enough to stay a cheap call.
READ_CONTEXT_MESSAGES = 6
READ_TEXT_LIMIT = 500


class Reading(BaseModel):
    """What the user wants to happen next, in the order it should happen."""

    wants: list[Want] = Field(description="What the user wants next, most urgent first.")


class State(TypedDict, total=False):
    """What the graph remembers, so the model does not have to re-derive it."""

    messages: Annotated[list[BaseMessage], add_messages]
    energy: Energy | None
    goal: Goal | None
    situation: str | None       # what is going on, in the user's words, or DECLINED
    offered: list[str]          # every measure id the lookup returned, not the two shown
    rejections: int             # suggestions rejected in a row
    step: Station | None        # the active station
    idea: str | None            # an idea of the user's, while it is discussed
    debate_turns: int
    rejected: bool              # a rejection stated as a fact, not inferred (eval harness)
    choices_clicked: bool       # set by the UI's energy/goal buttons
    checkin_logged: bool        # a check-in row was written this turn
    last_measure_id: str        # the measure that check-in row named, for a later rating
    complete: bool              # the conversation reached its end
    reading: list[str] | None   # what the understand node read the user to want, this turn
    reading_failed: bool        # the reader was asked this turn and gave no reading
    say_nothing_saved: bool     # this turn's reply opens with NOTHING_TO_SAVE_YET
    side_calls: list[AIMessage]  # model calls of this turn that left no message: the reader's
    approvals: dict[str, dict[str, Any]]  # the user's card decisions, by tool call id
    tool_events: Annotated[list[dict[str, Any]], merge_tool_events]


def _parsed(content: Any) -> Any:
    """The value inside a tool message, for its receipt: "3 results", not the
    first 60 characters of the JSON."""
    if not isinstance(content, str):
        return content
    try:
        return json.loads(content)
    except ValueError:
        return content


# -- tools that only touch the state -----------------------------------------


def state_tool_reply(text: str, tool_call_id: str, name: str, **update: Any) -> Command:
    """A tool result plus the state it writes. The receipt is set by the tools
    node, which is the one place the model cannot reach."""
    return Command(
        update={**update, "messages": [ToolMessage(text, tool_call_id=tool_call_id, name=name)]}
    )


@tool
def record_choices(
    tool_call_id: Annotated[str, InjectedToolCallId],
    energy: Energy | None = None,
    goal: Goal | None = None,
    situation: str | None = None,
) -> Command:
    """Keep the user's energy, their goal, their situation, or any one alone, for this conversation.

    This writes nothing to the log -- only a logged check-in does. Never tell
    the user it was logged, saved or recorded.

    Call this the moment the user gives you one of them. Do not wait for the
    others: telling them you noted it keeps nothing, and the next turn will
    have forgotten it.

    energy: how they feel right now - exhausted, calm or wired.
    goal: what would help them most - "calm down now" (a few minutes),
    "plan recovery" (something for later, hours or days), or
    "change the situation" (act on the cause of the stress).
    situation: what is going on for them, in one short sentence in their own
    words. Pass "not given" when they would rather not say, so they are not
    asked again.
    Only pass what the user actually said; ask for whatever is still missing.
    """
    # Only the values actually given: passing None would blank a value
    # recorded on an earlier turn.
    update = {
        name: value
        for name, value in (("energy", energy), ("goal", goal), ("situation", situation))
        if value
    }
    if not update:
        return state_tool_reply(
            "Nothing recorded: pass the energy or the goal the user gave.",
            tool_call_id,
            "record_choices",
        )
    noted = ", ".join(f"{name} {value}" for name, value in update.items())
    # The absence is said in the receipt itself. "Noted: energy exhausted." was
    # true and left the rest to the model, which in 3 of 156 case-runs of the
    # Step 7 sweeps (2026-09-27) then told the user the state had been
    # "logged" -- once answering "did it actually go into the log?" with "Yes".
    # Zero rows had been written each time.
    return state_tool_reply(
        f"Noted for this conversation: {noted}. Nothing was written to the log.",
        tool_call_id,
        "record_choices",
        **update,
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

    Not for a measure you just offered them. Choosing one of your suggestions
    is a choice, not a new idea, and sending it here has a cost that is not
    obvious: this station has no log_checkin, so "save it" once the
    conversation is in it can only become add_measure -- which is how a
    measure the agent had just called "already on your list" got written a
    second time under a different id.

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


# What `situation` holds when the user would rather not say. A value, not an
# empty string: the routing gate reads "is it set", and an empty one would ask
# again every turn.
SITUATION_DECLINED = "not given"


def situation_question_unanswered(state: State) -> bool:
    """Is this turn the user's reply to the situation question?

    Energy and goal are known, the situation is not, and the last turn ended in
    `ask`. Forcing a tool call here is what keeps the routing gate from
    becoming a loop: the station asks, the user answers, and if the model only
    says "got it" the next turn routes back to `ask` and asks again.
    """
    messages = state.get("messages") or []
    return bool(
        messages
        and isinstance(messages[-1], HumanMessage)
        and state.get("energy")
        and state.get("goal")
        and not state.get("situation")
        and state.get("step") == "ask"
    )


def goal_question_unanswered(state: State) -> bool:
    """Is this turn the user's reply to the goal question?

    The energy is known, the goal is not, the last turn ended in `ask`, and
    the station is answering the user's message, not a tool result. A first
    click on an energy button has no `ask` turn before it, so the goal is
    still asked normally.
    """
    messages = state.get("messages") or []
    return bool(
        messages
        and isinstance(messages[-1], HumanMessage)
        and state.get("energy")
        and not state.get("goal")
        and state.get("step") == "ask"
    )


def _last_tool_batch(messages: Sequence[BaseMessage]) -> list[ToolMessage]:
    """The tool results since the last model message."""
    batch: list[ToolMessage] = []
    for message in reversed(messages):
        if not isinstance(message, ToolMessage):
            break
        batch.append(message)
    return batch[::-1]


def _this_turn(messages: Sequence[BaseMessage]) -> list[BaseMessage]:
    """Everything since the user's last message, a resumed pause included."""
    for index in range(len(messages) - 1, -1, -1):
        if isinstance(messages[index], HumanMessage):
            return list(messages[index + 1 :])
    return list(messages)


def idea_closing(
    messages: Sequence[BaseMessage], approvals: dict[str, dict[str, Any]] | None = None
) -> str:
    """What to tell the user once their idea is decided, from what ran.

    It reads the whole turn, not the last batch: a model may add in one
    message and decide in the next. Reading only the last batch said
    "Nothing was added" right after two rows were written.

    Each measure is named, with the reason in plain words when it was not
    added. "Part of it is in your list now -- the notes above say what
    wasn't added" left the user to decode a cut-off note written for the
    model ("already exists. Pick anothe...").
    """
    calls = {c.get("name"): c for c in last_ai_tool_calls(messages)}
    outcome = (calls.get("idea_decided") or {}).get("args", {}).get("outcome")
    if outcome == "try":
        return IDEA_TRY
    if outcome == "drop":
        return IDEA_DROPPED
    turn = _this_turn(messages)
    results = [m for m in turn if isinstance(m, ToolMessage) and m.name == "add_measure"]
    if not results:
        return IDEA_ADD_MISSING
    asked = {
        c.get("id"): c.get("args") or {}
        for m in turn
        if isinstance(m, AIMessage)
        for c in m.tool_calls or []
        if c.get("name") == "add_measure"
    }
    lines = []
    for result in results:
        decision = (approvals or {}).get(result.tool_call_id) or {}
        args = decision.get("args") if decision.get("type") == "edit" else None
        measure = str((args or asked.get(result.tool_call_id) or {}).get("measure") or "this idea")
        if result.status != "error":
            template = IDEA_ADDED_LINE
        elif result.content == REJECTED_TEXT:
            template = IDEA_NOT_APPROVED_LINE
        elif "already exists" in str(result.content):
            template = IDEA_ALREADY_LISTED_LINE
        else:
            template = IDEA_NOT_SAVED_LINE
        lines.append(template.format(measure=measure.rstrip(".")))
    if any(m.status != "error" for m in results):
        lines.append(IDEA_TRY_WHAT_WAS_ADDED)
    return "\n".join(lines)


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
    phrase = next((p for p in NOT_HELPFUL_PHRASES if p in lowered), None)
    if phrase is None:
        return False
    extra = re.findall(r"[\w']+", lowered.replace(phrase, " ", 1))
    return len(extra) <= NOT_HELPFUL_MAX_EXTRA_WORDS and not CARRIES_MORE_WORDS & set(extra)


def says_rating(text: str) -> str:
    """The verdict a short message gives, or "" if it gives none.

    Short on purpose, like the rejection shortcut: a longer message usually
    carries something else that the model should read instead.
    """
    stripped = (text or "").strip()
    if not stripped or len(stripped) > RATING_MAX_CHARS:
        return ""
    lowered = stripped.lower().replace("\u2019", "'")
    for verdict, phrases in RATING_PHRASES.items():
        if any(phrase in lowered for phrase in phrases):
            return verdict
    return ""


def station_for(state: State) -> Station:
    """Which station fits the state.

    An open idea comes first: the user brought something of their own, and
    every other station would talk past it.

    A conversation that already reached follow_up stays there even with no
    goal recorded. Without that, every such turn went back to `ask`, which has
    neither add_measure nor log_checkin: after a web search "add both of them"
    only worked when the model took the detour through discuss_idea, and "it
    worked" after a measure was answered with the goal question instead of an
    offer to log the check-in.
    """
    if state.get("idea"):
        return "debate"
    if state.get("step") == "follow_up":
        return "follow_up"
    if not (state.get("energy") and state.get("goal") and state.get("situation")):
        return "ask"
    if not state.get("offered"):
        return "suggest"
    return "follow_up"


def last_human_text(messages: Sequence[BaseMessage]) -> str:
    for message in reversed(messages):
        if message.__class__.__name__ == "HumanMessage":
            return message_text(message)
    return ""


def _said(message: BaseMessage) -> str:
    """What a message says out loud; empty for a tool call with no text.

    Not `message_text`: that falls back to printing the content blocks, and a
    tool-calling turn would reach the reader as a dump of dicts.
    """
    text = getattr(message, "text", "")
    if not isinstance(text, str) and callable(text):
        text = text()
    return text.strip() if isinstance(text, str) else ""


# The part of the log question the model keeps when it rephrases it. The log
# prompt prescribes the question word for word (agent/prompts.py, LOG_PROMPT)
# and asks it only once the user has settled on a measure; the recorded runs
# vary the start ("check in" / "check this") but keep this phrase.
LOG_QUESTION_MARKER = "circumstances log"


def asked_to_log(messages: Sequence[BaseMessage]) -> bool:
    """Did the app's last spoken message, before the user's latest one, ask the
    log question?

    It is the one moment the app itself guarantees a measure is settled: the
    log prompt asks it "the moment" one is. A log request anywhere else may
    come before any measure was chosen -- case 36, where a forced log_checkin
    was refused by the tool and still broke the case's rule.
    """
    for message in reversed(messages[:-1]):
        if isinstance(message, HumanMessage):
            return False
        if isinstance(message, AIMessage) and _said(message):
            return LOG_QUESTION_MARKER in _said(message).lower()
    return False


def reading_input(state: State, waiting: str) -> list[BaseMessage]:
    """The reader's whole input: its prompt, a few facts from the state, the
    last few spoken turns, and the message to read.

    The facts are what a message is read against. "didn't help" is a rating
    when a row is waiting for one and a request for other suggestions when
    suggestions are on screen; "yes" answers whatever the app asked last.
    """
    messages = state.get("messages") or []
    spoken = [
        (("user" if isinstance(m, HumanMessage) else "app"), _said(m))
        for m in messages[:-1]
        if isinstance(m, (HumanMessage, AIMessage)) and _said(m)
    ][-READ_CONTEXT_MESSAGES:]
    idea = state.get("idea")
    facts = [
        f"- measures offered in this conversation: {'yes' if state.get('offered') else 'none yet'}",
        f"- a logged measure waiting for a rating: {waiting or 'none'}",
        f"- an idea of the user's under discussion: {idea if idea else 'none'}",
    ]
    history = "\n".join(f"{who}: {text[:READ_TEXT_LIMIT]}" for who, text in spoken) or "(none)"
    body = (
        "Facts from the app:\n" + "\n".join(facts)
        + "\n\nThe conversation so far, oldest first:\n" + history
        + "\n\nThe message to read:\n" + last_human_text(messages)[:READ_TEXT_LIMIT]
    )
    return [SystemMessage(content=READ_PROMPT), HumanMessage(content=body)]


def parse_reading(result: Any) -> Reading | None:
    """The reading inside whatever the structured call returned.

    With `include_raw=True` that is a dict whose `parsed` is None when the
    model's answer did not fit the schema; a scripted test model returns the
    JSON as a plain message. A reading that does not validate raises, and the
    caller falls back to the routing of v1.0.0.
    """
    if isinstance(result, dict):
        parsed = result.get("parsed")
        if isinstance(parsed, Reading):
            return parsed
        return Reading.model_validate(parsed) if isinstance(parsed, dict) else None
    if isinstance(result, Reading):
        return result
    if isinstance(result, BaseMessage):
        return Reading.model_validate_json(message_text(result))
    return None


def effective_wants(state: State) -> list[str]:
    """The reading as routing and forcing use it.

    An answer to the app's log question that says how it went -- "helped.
    literally the only thing that actually helped today" -- contains the yes:
    the reader returns a rating, and a rating given as that answer means the
    measure was done and the log is wanted first. Log first, then rate: the
    order case 44's "yes. the ritual helped ..." is already read in, with the
    yes implied instead of said. In case 39 the rating alone found no row to
    rate and the model asked which measure had helped (dev log stage 84).

    Built from two facts code already has -- the log question was asked and
    the first want is a rating -- not from the wording of the answer.
    """
    wants = list(state.get("reading") or [])
    if wants and wants[0] == "rate_measure" and asked_to_log(state.get("messages") or []):
        return ["log_checkin", *wants]
    return wants


def route_for_reading(state: State, waiting: str = "") -> str | None:
    """Where a reading sends the turn, or None to route as v1.0.0 did.

    The reader lists what the user wants in the order it should happen, so the
    first want decides; there is no priority table to overrule it. Each want
    has one place:

    - just chat -> `listen`, which has nothing to suggest with;
    - log the check-in -> `follow_up`, the only station with log_checkin --
      but only when the user is answering the app's log question, the one
      moment a measure is known to be settled;
    - rate a measure -> `follow_up`, but only when a row is waiting for a
      rating and no idea is open;
    - different suggestions -> `count_rejection`, while something is on screen
      and no idea is open (with nothing offered there is nothing to count);
    - go with their own idea -> `follow_up`, where discuss_idea is forced and
      the turn goes on to `debate`; with an idea already open, the state
      already points there.

    A want steers the route only when its preconditions are known in code. The
    two wants that lead to a write go to the station that can write only when
    the write can rightly happen; otherwise the turn goes where v1.0.0 would
    have sent it. Sending a log request there early handed the model the tool:
    in case 36 it called log_checkin before any measure was chosen, where
    v1.0.0 had kept the case in `ask`, which cannot log (dev log stage 83).

    A check-in answer, a pick from the offer and anything unclear go where the
    state points, as before.
    """
    wants = effective_wants(state)
    first = wants[0] if wants else None
    idea_open = bool(state.get("idea"))
    if first == "just_chat":
        return "listen"
    if first == "log_checkin" and asked_to_log(state.get("messages") or []):
        return "follow_up"
    if first == "rate_measure" and waiting and not idea_open:
        return "follow_up"
    if first == "other_suggestions" and state.get("offered") and not idea_open:
        return "count_rejection"
    if first == "own_idea" and not idea_open:
        return "follow_up"
    return None


def forced_tool(state: State, waiting: str) -> str | None:
    """The tool `follow_up` must call first this turn, or None.

    A want decides the action, not only the station: in the focused check the
    turn reached `follow_up` as intended and the model still picked a tool of
    its own -- a rejection on two verdicts, a lookup on "note them both down".

    Only on the turn's first call, while the station is answering the user and
    not a tool result -- the guard `ask` already uses. Otherwise a forced call
    would be forced again after its own answer, and the turn would loop.

    `log_checkin` for a log request only when all three of its preconditions
    are known in code: a settled measure -- the user is answering the app's
    log question -- and the energy and a real situation. The state lags the
    conversation (a chosen measure is never recorded in it), so the first
    version forced the log from energy and situation alone and called it
    before any measure was chosen (dev log stage 82). With a precondition
    missing the station decides, under its prompt's own rules.
    """
    messages = state.get("messages") or []
    wants = effective_wants(state)
    if not wants or not messages or not isinstance(messages[-1], HumanMessage):
        return None
    first = wants[0]
    situation = state.get("situation")
    if (
        first == "log_checkin"
        and asked_to_log(messages)
        and state.get("energy")
        and situation
        and situation != SITUATION_DECLINED
    ):
        return "log_checkin"
    if first == "rate_measure" and waiting:
        return "log_checkin"
    if first == "own_idea" and not state.get("idea"):
        return "discuss_idea"
    return None


def idea_kept(messages: Sequence[BaseMessage]) -> bool:
    """Did the user decide, in this conversation, to try or add an idea of theirs?

    `idea_decided` clears the idea from the state, so the transcript is where
    a decided one is still known. "yeah log it" right after deciding to try it
    is a log request about that idea (case 41).
    """
    return any(
        call.get("name") == "idea_decided" and (call.get("args") or {}).get("outcome") in ("try", "add")
        for message in messages
        if isinstance(message, AIMessage)
        for call in message.tool_calls or []
    )


def measure_known(state: State, waiting: str) -> bool:
    """Does the app know of any measure a save could be about?

    Measures offered from the user's list, an idea of theirs open or decided
    to try or add, a logged measure waiting for a rating, or the log question
    just asked -- the log prompt asks it only once a measure is settled.
    """
    messages = state.get("messages") or []
    return bool(
        state.get("offered")
        or state.get("idea")
        or waiting
        or asked_to_log(messages)
        or idea_kept(messages)
    )


def save_without_measure(state: State, waiting: str) -> bool:
    """Does the user want something saved -- a check-in or a rating -- that
    has nothing to be about yet?

    Then the turn opens with NOTHING_TO_SAVE_YET, said by code, and goes on to
    the station the state points to. Left to itself, a station answered this
    with a wrong reason ("there's no way for me to write it to your account
    from here", case 40), and a node that only explained would stop the
    conversation where a station goes on to offer measures (case 39's "ill do
    the first one. save that" before anything was offered).

    Only what code knows counts: a measure the user names in their own words
    in this same message is not seen, so the sentence says what the app has,
    not what the user did.
    """
    wants = effective_wants(state)
    return bool(wants) and wants[0] in ("log_checkin", "rate_measure") and not measure_known(state, waiting)


def action_request(call: dict[str, Any]) -> dict[str, Any]:
    """What the approval card shows for one pending write call.

    The description is built here from the call's own arguments, not written
    by the model: the card has to show what will actually be written, or an
    approval means nothing.
    """
    name = call.get("name") or ""
    args = dict(call.get("args") or {})
    if name == "add_measure":
        description = f'Add to your list: "{args.get("measure", "")}"'
    elif name == "log_checkin" and args.get("helpfulness"):
        measure = args.get("measure_id") or args.get("measure_taken") or "no measure"
        description = f'Rate "{measure}" as: {args["helpfulness"]}'
    elif name == "log_checkin":
        measure = args.get("measure_id") or args.get("measure_taken") or "no measure"
        description = f'Log: "{args.get("circumstances", "")}" (measure: {measure})'
    else:
        description = f"{name}: {args}"
    editable = EDITABLE_ARG.get(name)
    if name == "log_checkin" and args.get("helpfulness"):
        editable = None
    return {
        "id": call.get("id") or "",
        "name": name,
        "args": args,
        "description": description,
        "editable": editable,
    }


CARRIED_FIELDS = (
    "energy", "goal", "situation", "offered", "rejections", "step", "idea", "debate_turns"
)


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
            "situation": None,
            "offered": [],
            "rejections": 0,
            "step": None,
            "idea": None,
            "debate_turns": 0,
        }
    carried: dict[str, Any] = {
        "messages": compact_spent_tool_messages(
            drop_unanswered_tool_calls(result.get("messages", [])), tool_names=("find_measures",)
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
    reader_model: str | BaseChatModel | None = None,
    log_path: Path | str | None = None,
    measures_path: Path | str | None = None,
    recursion_limit: int = DEFAULT_RECURSION_LIMIT,
    checkpointer: BaseCheckpointSaver | None = None,
    exclude_tools: Sequence[str] = (),
):
    """Build the station graph.

    model/tools/log_path/measures_path are injectable so tests and eval runs
    exercise the real wiring without OpenAI, the real log or the real measure
    list.

    `exclude_tools` removes named tools from every station while leaving the
    per-station split as it is -- the same agent minus one capability, which
    is what the N-1 baseline measures. `tools=` cannot do that: an injected
    set is handed to *every* station (a convenience for tests), so stripping
    one tool that way would also give `ask` the write tools and measure two
    changes at once. An unknown name is an error rather than a no-op, because
    a typo would otherwise run the full agent under the baseline's name.

    `checkpointer` is what lets a turn pause for the user's approval and be
    resumed. Without one there is nobody to ask, so a write is refused rather
    than run: a graph that cannot pause must not write unapproved.
    """
    can_ask = checkpointer is not None
    base_model = resolve_model(model)
    # Nothing injected is how the real clients call this, so the debate gets
    # its own stronger model. An injected model with no debate model means a
    # test: the debate then shares it, and no real client is built behind it.
    if debate_model is not None:
        debate_base = resolve_debate_model(debate_model)
    elif model is None:
        debate_base = resolve_debate_model(None)
    else:
        debate_base = None
    # The same rule for the reader. An injected model with no reader keeps the
    # routing of v1.0.0, so a test scripting a station's replies is not handed
    # an extra call it never scripted.
    if reader_model is not None:
        reader_base: BaseChatModel | None = resolve_model(reader_model)
    elif model is None:
        reader_base = resolve_model(None)
    else:
        reader_base = None
    search_on = web_search_enabled()

    find_measures = make_find_measures(measures_path)
    search_measures = make_search_measures(measures_path)
    add_measure = make_add_measure(measures_path)
    log_checkin = make_log_checkin(log_path, measures_path)

    all_tools: list[BaseTool] = list(tools) if tools is not None else [
        record_choices,
        find_measures,
        search_measures,
        add_measure,
        log_checkin,
        suggestion_not_helpful,
        discuss_idea,
        idea_decided,
    ]
    unknown = set(exclude_tools) - {t.name for t in all_tools}
    if unknown:
        raise ValueError(f"exclude_tools names no such tool: {sorted(unknown)}")
    all_tools = [t for t in all_tools if t.name not in set(exclude_tools)]
    by_name: dict[str, BaseTool] = {t.name: t for t in all_tools}

    def pick(*names: str) -> list[Any]:
        """The tools a station may reach, plus web search where it is allowed."""
        if tools is not None:
            return list(all_tools)
        return [by_name[n] for n in names if n in by_name]

    station_models: dict[str, str] = {}

    def bind(station: str, bound: Sequence[Any], *, force: Any = None, use_debate: bool = False):
        chosen = debate_base if (use_debate and debate_base is not None) else base_model
        effort = STATION_REASONING_EFFORT.get(station)
        name = model_name_of(chosen)
        station_models[station] = name or getattr(chosen, "_llm_type", "")
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
    # Reading the user's list is allowed everywhere a user might ask about it:
    # reads change nothing, so the read/write split makes them safe to give.
    # `ask` can look measures up (with energy alone, if the goal is unknown)
    # but still cannot write.
    ask_tools = pick("record_choices", "discuss_idea", "find_measures", "search_measures")
    ask_model = bind("ask", ask_tools)
    # The turn after the goal question: a tool call is required -- record_choices
    # if they named a goal, find_measures if they did not. Left free, nano
    # answered "pick one for me" with box breathing from its own head.
    ask_must_act = bind("ask", ask_tools, force="required")
    suggest_tools = pick("find_measures", "record_choices", "discuss_idea")
    suggest_model = bind("suggest", suggest_tools + web)
    # Forcing a tool that is not bound is a 400, the same as for web search
    # below, so an agent built without the lookup is not forced to call it.
    lookup_model = bind(
        "suggest",
        suggest_tools + web,
        force="find_measures" if "find_measures" in by_name else None,
    )
    follow_up_tools = pick(
        "record_choices",
        "find_measures",
        "search_measures",
        "add_measure",
        "log_checkin",
        "suggestion_not_helpful",
        "discuss_idea",
    )
    follow_up_model = bind("follow_up", follow_up_tools + web)
    # The same station with its first call forced, one binding per tool a
    # reading can require (see forced_tool). A tool left out of the agent is
    # not forced: forcing a tool that is not bound is a 400.
    follow_up_forced = {
        name: bind("follow_up", follow_up_tools + web, force=name)
        for name in ("log_checkin", "discuss_idea")
        if name in by_name
    }
    debate_tools = pick("add_measure", "record_choices", "idea_decided", "find_measures", "search_measures")
    debate_station_model = bind("debate", debate_tools, use_debate=True)
    # This station reads untrusted web pages, so it gets no write tool at all:
    # once untrusted text is in the context, it must not be able to trigger a
    # lasting effect. The approval card would stop it too; this is the layer
    # underneath.
    #
    # It gets no local tool either, because it has no way to run one: its only
    # edge is to END, so a call made here is never answered. gpt-5-mini called
    # suggestion_not_helpful in a hand test, and the unanswered call broke every
    # later turn of that conversation with "No tool output found for function
    # call" (400).
    search_tools: list[BaseTool] = []
    search_model = bind(
        "web_search_ideas",
        search_tools + web,
        # Forcing a tool that is not bound is a 400, so the kill switch also
        # turns the forcing off.
        force=WEB_SEARCH_TOOL if search_on else None,
    )

    # No tools at all: someone who said "not now" cannot be handed a suggestion
    # or a check-in, whatever the model makes of its prompt.
    listen_tools: list[BaseTool] = []
    listen_model = bind("listen", listen_tools)

    if reader_base is not None:
        station_models["understand"] = model_name_of(reader_base) or getattr(reader_base, "_llm_type", "")
        reader = reader_base.with_structured_output(Reading, include_raw=True)
    else:
        reader = None

    # -- stations ------------------------------------------------------------

    def _reply(chosen, prompt: str, state: State, *, trailing: str | None = None) -> AIMessage:
        messages: list[BaseMessage] = [SystemMessage(content=prompt), *state["messages"]]
        if trailing:
            messages.append(SystemMessage(content=trailing))
        response = chosen.invoke(messages)
        if state.get("say_nothing_saved") and not response.tool_calls:
            # Part of the reply, not a message of its own: the clients show
            # only the last message of a turn. A new id, because the words are
            # no longer what the provider returned; the usage stays, so the
            # turn's cost still counts the call.
            said = _said(response)
            text = f"{NOTHING_TO_SAVE_YET}\n\n{said}" if said else NOTHING_TO_SAVE_YET
            response = response.model_copy(update={"content": text, "id": None})
        return response

    def ask(state: State) -> dict[str, Any]:
        """Find out energy and goal. It may look measures up in the user's list
        when asked, but cannot write: no write tool is bound here.

        Once the goal question has had its turn, the user's reply must lead to
        a tool: their goal recorded, or their own list looked up. A reply from
        the model's own head is then impossible, not just discouraged.
        """
        must_act = goal_question_unanswered(state) or situation_question_unanswered(state)
        chosen = ask_must_act if must_act else ask_model
        return {"messages": [_reply(chosen, STATION_PROMPTS["ask"], state)], "step": "ask"}

    def close_idea(state: State) -> dict[str, Any]:
        """The end of a discussed idea, said by code from what actually ran.

        After an approved add_measure the graph went back to `ask`, which has no
        write tool, and told the user it "can't directly add items" while the
        row was already in the file. Text built from the results cannot say
        that.
        """
        text = idea_closing(state["messages"], state.get("approvals"))
        # follow_up, not None: what usually comes next is "it worked", and
        # log_checkin lives there. `ask` would answer that with the goal
        # question, which is what a hand test got.
        return {"messages": [AIMessage(content=text)], "step": "follow_up"}

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
        waiting = measure_awaiting_rating(log_path)
        # Which measure a "helped" is about is a fact in the log, not something
        # to infer from a transcript the reset conversation no longer has.
        notes = []
        if waiting:
            notes.append(
                f'The situation logged for measure_id "{waiting}" has no rating yet. If the user says how it '
                "went, call log_checkin with that measure_id and the helpfulness only. If they have "
                "not said yet, ask them in one short sentence whether it helped."
            )
        force = forced_tool(state, waiting)
        if force == "log_checkin" and (effective_wants(state) or [None])[0] == "log_checkin":
            # What the log sentence is built from, put in front of the model.
            # Energy and situation can come from a button or a payload into
            # the state without ever appearing in the conversation, and a
            # model forced to log what it cannot see would have to invent it.
            notes.append(
                f"Recorded for this conversation -- energy: {state.get('energy')}; what is going on, "
                f'in the user\'s words: "{state.get("situation")}". The user asked to log the check-in: '
                "call log_checkin now, with the circumstances built only from this and the conversation."
            )
        chosen = follow_up_forced.get(force, follow_up_model) if force else follow_up_model
        response = _reply(chosen, STATION_PROMPTS["follow_up"], state, trailing="\n\n".join(notes) or None)
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
        """After the first rejection: a model call that has to search.

        It is told what the user's list already holds, read fresh here, so the
        web brings something new rather than a measure they have.
        """
        already_listed = [row["measure"] for row in read_measures(measures_path)]
        response = _reply(
            search_model,
            STATION_PROMPTS["suggest"],
            state,
            trailing=search_instruction(already_listed),
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

    def review(state: State) -> dict[str, Any]:
        """Pause for the user's decision on every pending write, all at once.

        Its own node, with no side effects, because LangGraph re-runs a paused
        node from the start on resume: inside the tools node, the reads before
        the pause would run twice. One interrupt per run, carrying every write
        of the message -- a model can ask for five at once.

        A decision counts only if it names a pending call and a known type;
        anything else leaves that call without approval, which the tools node
        treats as a rejection.
        """
        requests = [
            action_request(call)
            for call in pending_tool_calls(state["messages"])
            if call.get("name") in WRITE_TOOLS
        ]
        answer = interrupt({"action_requests": requests}) if can_ask else {}
        decisions = answer.get("decisions") if isinstance(answer, dict) else None
        approvals = dict(state.get("approvals") or {})
        pending_ids = {r["id"] for r in requests}
        for decision in decisions or []:
            if (
                isinstance(decision, dict)
                and decision.get("id") in pending_ids
                and decision.get("type") in {"approve", "edit", "reject"}
            ):
                approvals[decision["id"]] = decision
        return {"approvals": approvals}

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
        approvals = state.get("approvals") or {}

        for call in pending_tool_calls(state["messages"]):
            name = call.get("name") or ""
            call_id = call.get("id") or ""
            if name in WRITE_TOOLS:
                decision = approvals.get(call_id) or {}
                if decision.get("type") not in {"approve", "edit"}:
                    outputs.append(
                        ToolMessage(content=REJECTED_TEXT, name=name, tool_call_id=call_id, status="error")
                    )
                    events.append(tool_event(name, "rejected", "not approved", call_id=call_id))
                    continue
                if decision.get("type") == "edit" and isinstance(decision.get("args"), dict):
                    call = {**call, "args": decision["args"]}
            target = by_name.get(name)
            if target is None:
                outputs.append(
                    ToolMessage(
                        content=f"Unknown tool: {name}", name=name, tool_call_id=call_id, status="error"
                    )
                )
                events.append(tool_event(name or "unknown", "error", "unknown tool", call_id=call_id))
                continue
            try:
                # The state as of the calls already run in this batch, not as
                # of the turn. A model that records the energy and looks up in
                # one go -- which is what it does when the user gives
                # everything at once -- had the lookup injected the state from
                # before the recording, so find_measures saw no energy and
                # skipped itself. The agent then had an empty result and
                # invented measures out of its own head.
                result = _invoke_tool(target, call, {**state, **update})
            except Exception as exc:  # noqa: BLE001 - a broken tool must not end the turn
                outputs.append(
                    ToolMessage(
                        content=f"Tool error: {exc}", name=name, tool_call_id=call_id, status="error"
                    )
                )
                events.append(tool_event(name, "error", str(exc), call_id=call_id))
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
                        call_id=call_id,
                    )
                )
            else:
                if isinstance(result, ToolMessage):
                    # Invoked with the whole call, a plain-returning tool comes
                    # back as its own ToolMessage. Stringifying that handed the
                    # model "content='[...]' name=..." instead of the rows.
                    content = result.content
                    value = _parsed(content)
                else:
                    content, value = as_tool_content(result), result
                outputs.append(
                    ToolMessage(content=content, name=name, tool_call_id=call_id, status="success")
                )
                events.append(tool_event(name, "ok", result_detail(value), call_id=call_id))

        update["messages"] = outputs
        update["tool_events"] = events
        return update

    # -- routing -------------------------------------------------------------

    def after_rejection(state: State) -> str:
        """The first rejection buys a web search, the second a step back."""
        return "web_search_ideas" if int(state.get("rejections") or 0) == 1 else "ask_goal_again"

    def decided_in_code(state: State) -> str | None:
        """The routes code takes without reading the message, or None.

        A button click is a fact; a short "not helpful" is close enough to one.

        An energy/goal click with both values set means a fresh lookup in the
        user's own list, however much was offered before -- the same rule
        `after_tools` applies to record_choices. The buttons skip that tool,
        so without this a click after the goal question reached follow_up,
        where the lookup is optional, and the model invented options instead.

        A verdict on a measure that is waiting for one goes to follow_up, the
        only station with log_checkin. A logged check-in ends the conversation,
        so "helped" arrives in a fresh one and reached `ask`, which cannot log:
        the rating was lost. Rejections keep priority while suggestions are on
        screen, where "didn't help" is about the suggestion, not a measure.
        """
        messages = state.get("messages") or []
        if state.get("rejected"):
            return "count_rejection"
        if state.get("offered") and not state.get("idea") and says_not_helpful(last_human_text(messages)):
            return "count_rejection"
        if (
            not state.get("idea")
            and says_rating(last_human_text(messages))
            and measure_awaiting_rating(log_path)
        ):
            return "follow_up"
        if (
            state.get("choices_clicked")
            and state.get("energy")
            and state.get("goal")
            and state.get("situation")
            and not state.get("idea")
        ):
            return "suggest"
        return None

    def understand(state: State) -> dict[str, Any]:
        """Read the user's message before any station, once per turn.

        No model call when there is nothing to read or code already knows: a
        button click, a flag set by the harness, the two keyword shortcuts. A
        rating button and an approval never get here at all -- the first runs
        no graph, the second resumes at `review`.

        A reading that fails -- a timeout, an answer outside the schema -- is
        None, and the turn is routed as v1.0.0 routed it. A reader that is
        down must cost accuracy, not the turn.
        """
        messages = state.get("messages") or []
        if (
            reader is None
            or not messages
            or not isinstance(messages[-1], HumanMessage)
            or state.get("choices_clicked")
            or decided_in_code(state)
        ):
            return {"reading": None, "reading_failed": False, "side_calls": [], "say_nothing_saved": False}
        try:
            result = reader.invoke(reading_input(state, measure_awaiting_rating(log_path)))
        except Exception:  # noqa: BLE001 - a failed reading falls back, it does not end the turn
            result = None
        try:
            reading = parse_reading(result)
        except Exception:  # noqa: BLE001 - an answer outside the schema, the same fallback
            reading = None
        # The reader's own message, kept aside so the turn's cost includes it
        # (observability/tracing.py). An answer outside the schema was still
        # a paid call; a call that raised left no usage to count.
        raw = result.get("raw") if isinstance(result, dict) else result
        wants = list(reading.wants) if reading else None
        update: dict[str, Any] = {
            "reading": wants,
            # Said, not only survived: the fallback keeps the turn alive and
            # would otherwise hide that it ran the routing of v1.0.0. For ten
            # minutes on 2026-09-29 every reading failed, and an eval run of
            # "v1.1" measured v1.0.0's routing without a word in its report.
            "reading_failed": wants is None,
            "side_calls": [raw] if isinstance(raw, AIMessage) else [],
            # Set again only by nothing_to_save, for this turn alone.
            "say_nothing_saved": False,
        }
        # A log request that goes to follow_up -- the user answering the log
        # question -- while the user's own idea is open counts the idea as
        # tried and closes it. Left open, the turn reaches follow_up, logs --
        # and `after_tools` sends it straight back to `debate`, which has no
        # log_checkin: the log request of case 41 that was never carried out.
        effective = effective_wants({**state, "reading": wants})
        if state.get("idea") and effective and effective[0] == "log_checkin" and asked_to_log(messages):
            update.update(idea=None, debate_turns=0)
        return update

    def listen(state: State) -> dict[str, Any]:
        """For "not now": answer like a person, with nothing to suggest with."""
        return {"messages": [_reply(listen_model, STATION_PROMPTS["listen"], state)], "step": "listen"}

    def nothing_to_save(state: State) -> dict[str, Any]:
        """The user wants something saved and there is nothing to save it with.

        No model call and no message of its own: it marks the turn, and the
        station the state points to opens its reply with the fixed sentence
        (`_reply`) and goes on with its own job -- asking what is missing or
        offering measures -- so the conversation does not stop here.
        """
        return {"say_nothing_saved": True}

    def route_start(state: State) -> str:
        """Before any station: what code decides, then what the reading says,
        then where the state points -- which is all v1.0.0 had. A save with no
        measure known passes nothing_to_save on its way to that station."""
        waiting = measure_awaiting_rating(log_path)
        return (
            decided_in_code(state)
            or route_for_reading(state, waiting)
            or ("nothing_to_save" if save_without_measure(state, waiting) else None)
            or station_for(state)
        )

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
        if "idea_decided" in {m.name for m in _last_tool_batch(state["messages"])}:
            return "close_idea"
        return state.get("step") or station_for(state)

    def after_station(state: State) -> str:
        """A station either asked for tools, ended the check-in, or replied.

        A message with any write in it goes past the review first; reads alone
        go straight to the tools.
        """
        pending = pending_tool_calls(state["messages"])
        if any(call.get("name") in WRITE_TOOLS for call in pending):
            return "review"
        if pending:
            return "tools"
        if state.get("checkin_logged"):
            return "checkin_done"
        return END

    graph = StateGraph(State)
    graph.add_node("understand", understand)
    graph.add_node("listen", listen)
    graph.add_node("nothing_to_save", nothing_to_save)
    graph.add_node("ask", ask)
    graph.add_node("suggest", suggest)
    graph.add_node("follow_up", follow_up)
    graph.add_node("debate", debate)
    graph.add_node("tools", tools_node)
    graph.add_node("count_rejection", count_rejection)
    graph.add_node("web_search_ideas", web_search_ideas)
    graph.add_node("ask_goal_again", ask_goal_again)
    graph.add_node("checkin_done", checkin_done)
    graph.add_node("review", review)
    graph.add_node("close_idea", close_idea)

    graph.add_edge(START, "understand")
    graph.add_conditional_edges(
        "understand",
        route_start,
        ["count_rejection", "ask", "suggest", "follow_up", "debate", "listen", "nothing_to_save"],
    )
    graph.add_conditional_edges("nothing_to_save", station_for, ["ask", "suggest", "follow_up", "debate"])
    graph.add_edge("listen", END)
    for name in ("ask", "suggest", "follow_up", "debate"):
        graph.add_conditional_edges(name, after_station, ["review", "tools", "checkin_done", END])
    graph.add_edge("review", "tools")
    graph.add_conditional_edges(
        "tools",
        after_tools,
        ["ask", "suggest", "follow_up", "debate", "web_search_ideas", "ask_goal_again", "close_idea"],
    )
    graph.add_conditional_edges(
        "count_rejection", after_rejection, ["web_search_ideas", "ask_goal_again"]
    )
    graph.add_edge("web_search_ideas", END)
    graph.add_edge("ask_goal_again", END)
    graph.add_edge("checkin_done", END)
    graph.add_edge("close_idea", END)

    compiled = graph.compile(checkpointer=checkpointer).with_config(
        {"recursion_limit": recursion_limit}
    )
    # Handy for tests and for devtools/try_graph_viz.py, which render the shape.
    # The same lists the stations were bound with, not a second copy of them.
    compiled.station_tools = {  # type: ignore[attr-defined]
        "ask": [t.name for t in ask_tools],
        "suggest": [t.name for t in suggest_tools],
        "follow_up": [t.name for t in follow_up_tools],
        "debate": [t.name for t in debate_tools],
        "web_search_ideas": [t.name for t in search_tools],
        "listen": [t.name for t in listen_tools],
    }
    # Which model each station was built with, so a test can check the graph
    # the clients actually get rather than the resolver in isolation.
    compiled.station_models = dict(station_models)  # type: ignore[attr-defined]
    return compiled


__all__ = [
    "DEBATE_TURN_CAP",
    "DEFAULT_RECURSION_LIMIT",
    "NOT_HELPFUL_PHRASES",
    "REJECTED_TEXT",
    "Reading",
    "State",
    "WRITE_TOOLS",
    "action_request",
    "build_agent",
    "effective_wants",
    "forced_tool",
    "discuss_idea",
    "idea_decided",
    "idea_kept",
    "measure_known",
    "next_turn_state",
    "record_choices",
    "route_for_reading",
    "save_without_measure",
    "says_not_helpful",
    "station_for",
    "suggestion_not_helpful",
    "web_search_enabled",
]
