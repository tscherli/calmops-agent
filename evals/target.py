"""The eval harness's target function: runs one eval case through the same
production pipeline as app.py -- agent/safety.py's crisis gate first, then
the station graph -- for as many turns as the case provides, accumulating one
continuous conversation.

agent_graph is injectable (same dependency-injection-for-testability
pattern used throughout this project: tools/checkin_log.py's `path`,
agent/graph.py's `model`/`tools`, app.py's `agent_graph`/`recorder`) so
tests can stub it without a real OpenAI call.
"""
from __future__ import annotations

import shutil
import tempfile
import uuid
from pathlib import Path
from typing import Any

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, ToolMessage
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command

from agent.common import message_text
from agent.graph import EDITABLE_ARG, build_agent, next_turn_state
from agent.safety import screen_for_crisis
from tools.measures import MEASURES_PATH

# Role vocabulary must match dataset.jsonl's hand-authored reference_trajectory
# entries exactly ("human"/"ai"/"tool") -- agentevals' structural comparison
# only checks string equality, so both sides just need to agree.


def serialize_messages(messages: list[BaseMessage]) -> list[dict]:
    """Flatten messages for the evaluators.

    Content goes through `message_text`, not `str(m.content)`: since the
    stage-14 redesign binds OpenAI's built-in web_search, the model speaks
    Responses-API content blocks, and stringifying that list would hand the
    fabrication evaluator's keyword scan (and the LLM judge) a wall of
    reasoning metadata and base64 instead of what the agent actually said.
    """
    result = []
    for m in messages:
        if isinstance(m, HumanMessage):
            result.append({"role": "human", "content": message_text(m)})
        elif isinstance(m, ToolMessage):
            result.append(
                {
                    "role": "tool",
                    "content": message_text(m),
                    "name": m.name,
                    "status": m.status,
                }
            )
        elif isinstance(m, AIMessage):
            tool_calls = [{"name": tc["name"], "args": tc.get("args", {})} for tc in (m.tool_calls or [])]
            result.append(
                {
                    "role": "ai",
                    "content": message_text(m),
                    "tool_calls": tool_calls or None,
                }
            )
        else:
            result.append({"role": getattr(m, "type", "unknown"), "content": message_text(m)})
    return result


def _isolated_log_path() -> Path:
    """A throwaway check-in log for one eval case.

    This must never default to the real data/checkin_log.csv: a run would
    inject dozens of fabricated rows into the user's own log, and the weekly
    digest reads that file.

    Fresh per case, so cases can't see each other's rows and results don't
    depend on execution order.
    """
    return Path(tempfile.mkdtemp(prefix="joblog-eval-")) / "checkin_log.csv"


def _isolated_measures_path() -> Path:
    """A per-case copy of the curated measure list.

    `add_measure` writes to this file, so a shared one would let an eval run
    grow the user's real list -- and every later conversation reads it back.
    A copy rather than an empty file, because the cases are about choosing
    from the real list.
    """
    path = Path(tempfile.mkdtemp(prefix="joblog-evalmeasures-")) / "measures.csv"
    shutil.copyfile(MEASURES_PATH, path)
    return path


def _unwritable_log_path() -> Path:
    """A log path whose parent is a regular file, so every write genuinely
    fails. Used by the `simulate_write_failure` case to exercise the
    fallback-and-report-honestly path for real, rather than trusting that
    it works."""
    directory = Path(tempfile.mkdtemp(prefix="joblog-evalfail-"))
    blocker = directory / "not-a-directory"
    blocker.write_text("blocks the write below", encoding="utf-8")
    return blocker / "checkin_log.csv"


def build_eval_graph(log_path: Path, measures_path: Path, *, model: Any = None):
    """The agent as the harness runs it.

    The checkpointer is the whole point: without one the graph cannot pause on
    an approval card, and `tools_node` refuses every write for lack of a
    decision. The suite would then grade an agent forbidden to add a measure
    or log a check-in -- half of what it exists to do.
    """
    return build_agent(
        model=model,
        log_path=log_path,
        measures_path=measures_path,
        checkpointer=InMemorySaver(),
    )


def _decision_for(request: dict, approval: str, edit_text: str) -> dict:
    """What the user does with one approval card, as the case asked.

    An edit needs the argument the card lets you change, and which one that is
    belongs to the tool (`EDITABLE_ARG`), not to the case: a case says the new
    wording, never which field it lands in. An edit with nothing to put there
    is an approval of what was proposed, because that is what the clients do
    when the edit box is left alone.
    """
    if approval == "reject":
        return {"type": "reject"}
    if approval == "edit":
        editable = EDITABLE_ARG.get(request.get("tool") or request.get("name") or "")
        if editable and edit_text:
            return {"type": "edit", "args": {**(request.get("args") or {}), editable: edit_text}}
    return {"type": "approve"}


def run_case(
    inputs: dict,
    *,
    agent_graph: Any = None,
    log_path: Path | None = None,
    measures_path: Path | None = None,
) -> dict:
    """The function LangSmith's evaluate() calls once per Example (and the
    same logic run_baseline.py's no-tools comparison is built against).

    inputs = {"turns": [...]}. Checks the crisis gate before every turn,
    exactly like app.py's _on_send -- if it fires, stops immediately and
    returns crisis_intercepted=True with zero agent messages for that turn,
    never invoking the LLM. Otherwise runs the real agent, accumulating the
    full message list across turns (a fresh thread per turn, as the clients
    do, so this accumulation has to happen here rather than implicitly via
    agent memory).

    Writes pause on an approval card, and a case says what the user does with
    it: `"approval": "approve"` (the default), `"reject"`, or `"edit"` with
    `"edit_text"`. Without a checkpointer the graph cannot pause at all and
    refuses every write, so a case could never reach `add_measure` or
    `log_checkin` -- the eval suite graded an agent that was not allowed to
    do half of its job.
    """
    if agent_graph is not None:
        graph = agent_graph
    else:
        if log_path is None:
            log_path = (
                _unwritable_log_path()
                if inputs.get("simulate_write_failure")
                else _isolated_log_path()
            )
        if measures_path is None:
            measures_path = _isolated_measures_path()
        graph = build_eval_graph(log_path, measures_path)

    # Conversation state must be carried here exactly as app.py does, and
    # through the same function: passing only `messages` would reset the
    # station fields every turn, so the graph would re-ask for energy and goal
    # forever and the suite would measure a system the application never runs.
    # That defect produced a full run of plausible-looking but meaningless
    # numbers once already -- see DEVELOPMENT_LOG.md, stage 26.
    state: dict[str, Any] = {"messages": []}
    transcript: list[BaseMessage] = []
    # Machine-set records of what each tool actually did, accumulated across
    # the case's turns. The evaluators grade against these rather than
    # against what the reply claims -- see evals/evaluators.py.
    tool_events: list[dict] = []
    # What the user was asked to approve, and what they did about it. The
    # evaluators grade this rather than the reply: "it said it saved the
    # measure" and "the user was shown what would be written" are different
    # claims, and only the second one is a guarantee.
    approvals: list[dict] = []

    for turn in inputs["turns"]:
        turn_text, extra = (turn, {}) if isinstance(turn, str) else (turn["text"], dict(turn))
        extra.pop("text", None)
        approval = str(extra.pop("approval", "approve"))
        edit_text = extra.pop("edit_text", "")

        crisis = screen_for_crisis(turn_text)
        if crisis.triggered:
            return {
                "messages": serialize_messages(transcript),
                "crisis_intercepted": True,
                "matched_pattern": crisis.matched_pattern,
                "tool_events": tool_events,
            }
        messages_before = list(state.get("messages") or []) + [HumanMessage(content=turn_text)]
        # `extra` carries what a client would put in the payload beside the
        # message (`energy`, `goal` from the buttons; `rejected` and `idea`,
        # which no client sets any more, so a case can still state either as
        # a fact rather than hoping the model infers it).
        # One thread per turn, exactly as the clients do: the checkpointer
        # only ever has to hold the turn that is paused.
        config = {"configurable": {"thread_id": uuid.uuid4().hex}}
        result = graph.invoke({**state, **extra, "messages": messages_before}, config)
        while result.get("__interrupt__"):
            requests = result["__interrupt__"][0].value["action_requests"]
            decisions = []
            for request in requests:
                decision = _decision_for(request, approval, edit_text)
                approvals.append(
                    {
                        "tool": request.get("tool") or request.get("name") or "",
                        "description": request.get("description", ""),
                        "decision": decision["type"],
                    }
                )
                decisions.append({"id": request["id"], **decision})
            result = graph.invoke(Command(resume={"decisions": decisions}), config)

        # Accumulate only what this turn newly produced. The transcript must
        # keep growing across a context reset, or everything said before the
        # reset silently vanishes from grading -- the evaluators judge the
        # whole case, while `messages` is only the agent's working context.
        produced = result["messages"][len(messages_before):]
        transcript.append(HumanMessage(content=turn_text))
        transcript.extend(produced)

        tool_events.extend(result.get("tool_events") or [])
        state = next_turn_state(result)

    return {
        "messages": serialize_messages(transcript),
        "crisis_intercepted": False,
        "tool_events": tool_events,
        "approvals": approvals,
    }
