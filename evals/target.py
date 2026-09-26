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
import time
import uuid
from pathlib import Path
from typing import Any

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, ToolMessage
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command

from agent.common import message_text
from agent.graph import EDITABLE_ARG, WRITE_TOOLS, build_agent, next_turn_state
from agent.model import DEFAULT_MODEL
from agent.safety import screen_for_crisis
from observability.tracing import RunRecorder
from tools.checkin_log import append_checkin_row, read_checkin_rows
from tools.measures import MEASURE_COLUMNS, MEASURES_PATH, append_measure, measure_ids
from tools.measures import _clean_measure_text

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


# Eval runs keep their own cost record. observability/runs.jsonl is the app's,
# and a suite run would bury the real per-turn figures the dev log quotes.
EVAL_RUNS_LOG = Path(__file__).resolve().parent / "results" / "eval-runs.jsonl"


def seed_log(entries: list[dict], log_path: Path) -> None:
    """Rows the case starts with. A rating case needs a check-in to rate, and
    an injection case needs a row already in the log."""
    for entry in entries:
        append_checkin_row(**{**entry, "path": log_path})


def seed_measures(rows: list[dict], measures_path: Path) -> None:
    """Rows planted in the case's measure list.

    Through the same cleaning `add_measure` applies, so a planted row is one
    the app could have produced: a case that plants text the app would have
    rewritten tests a situation that cannot occur.
    """
    for row in rows:
        full = {column: "" for column in MEASURE_COLUMNS}
        full.update(row)
        full["measure"] = _clean_measure_text(full.get("measure", ""))
        append_measure(full, measures_path)


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


def _result(
    transcript: list[BaseMessage],
    tool_events: list[dict],
    approvals: list[dict],
    log_path: Path,
    measures_path: Path,
    measures_before: set[str],
) -> dict:
    """What a case reports, whichever way the turn loop ended.

    One builder for both exits. They were two literals once, and the crisis
    exit quietly carried four fields fewer -- so a case whose crisis arrives
    after a write reported no rows, no cards and no write at all.
    """
    return {
        "messages": serialize_messages(transcript),
        "tool_events": tool_events,
        "approvals": approvals,
        "log_rows": read_checkin_rows(log_path),
        "measures_added": sorted(measure_ids(measures_path) - measures_before),
        # The case's own list, seeds included. An expectation about the lookup
        # has to recompute the full slice from the list the case actually ran
        # against, and no other output says which list that was.
        "measures_path": str(measures_path),
        # Its own key, so "never got there" stays separate from "wrote the
        # wrong thing": about half the model-driven cases assert a write, and
        # one prompt quirk would otherwise fail them all as wrong content.
        "reached_write": any(
            e.get("tool") in WRITE_TOOLS and e.get("exit_code") == 0 for e in tool_events
        ),
    }


def run_case(
    inputs: dict,
    *,
    agent_graph: Any = None,
    log_path: Path | None = None,
    measures_path: Path | None = None,
    runs_log: Path | None = None,
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

    `log_path` and `measures_path` are where the rows are read back from at the
    end, so a caller injecting its own `agent_graph` has to pass the same paths
    that graph was built on, or the result reports an empty log for a write that
    really happened.

    Writes pause on an approval card, and a case says what the user does with
    it: `"approval": "approve"` (the default), `"reject"`, or `"edit"` with
    `"edit_text"`. Without a checkpointer the graph cannot pause at all and
    refuses every write, so a case could never reach `add_measure` or
    `log_checkin` -- the eval suite graded an agent that was not allowed to
    do half of its job.
    """
    # Paths are resolved even when a graph is injected: the case's rows are
    # read back from them at the end, and seeding needs somewhere to write.
    if log_path is None:
        log_path = (
            _unwritable_log_path()
            if inputs.get("simulate_write_failure")
            else _isolated_log_path()
        )
    if measures_path is None:
        measures_path = _isolated_measures_path()
    graph = agent_graph if agent_graph is not None else build_eval_graph(log_path, measures_path)

    if inputs.get("seed_entries"):
        seed_log(inputs["seed_entries"], log_path)
    if inputs.get("seed_measures"):
        seed_measures(inputs["seed_measures"], measures_path)
    measures_before = measure_ids(measures_path)

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
    # Opt-in: the runner says where cost goes. A default would have every test
    # that drives a case append a record to the repo, which is how this was
    # first written and what the untracked file it left behind showed.
    recorder = RunRecorder(log_path=runs_log) if runs_log else None

    for turn in inputs["turns"]:
        turn_text, extra = (turn, {}) if isinstance(turn, str) else (turn["text"], dict(turn))
        extra.pop("text", None)
        approval = str(extra.pop("approval", "approve"))
        edit_text = extra.pop("edit_text", "")

        crisis = screen_for_crisis(turn_text)
        if crisis.triggered:
            # The same result fields as a normal return, not a shorter dict.
            # A crisis can arrive mid-conversation, after turns that already
            # looked something up or wrote a row, and a case about that is
            # ungradable if the rows and cards vanish at the gate.
            return {
                **_result(transcript, tool_events, approvals, log_path, measures_path, measures_before),
                "crisis_intercepted": True,
                "matched_pattern": crisis.matched_pattern,
            }
        messages_before = list(state.get("messages") or []) + [HumanMessage(content=turn_text)]
        # `extra` carries what a client would put in the payload beside the
        # message (`energy`, `goal` from the buttons; `rejected` and `idea`,
        # which no client sets any more, so a case can still state either as
        # a fact rather than hoping the model infers it).
        # One thread per turn, exactly as the clients do: the checkpointer
        # only ever has to hold the turn that is paused.
        config = {"configurable": {"thread_id": uuid.uuid4().hex}}
        started = time.monotonic()
        result = graph.invoke({**state, **extra, "messages": messages_before}, config)
        while result.get("__interrupt__"):
            requests = result["__interrupt__"][0].value["action_requests"]
            decisions = []
            for request in requests:
                decision = _decision_for(request, approval, edit_text)
                approvals.append(
                    {
                        "call_id": request.get("id", ""),
                        "tool": request.get("tool") or request.get("name") or "",
                        "description": request.get("description", ""),
                        # What the card actually proposed. An edit turn
                        # overwrites the row, so a rule about the sentence the
                        # user was shown can only be checked here.
                        "args": dict(decision.get("args") or request.get("args") or {}),
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
        if recorder is not None:
            try:
                recorder.record_turn(
                    messages_before, result.get("messages") or [],
                    model=DEFAULT_MODEL, latency_seconds=time.monotonic() - started,
                )
            except Exception:  # a cost record must never fail a case
                pass
        state = next_turn_state(result)

    return {**_result(transcript, tool_events, approvals, log_path, measures_path, measures_before),
            "crisis_intercepted": False}
