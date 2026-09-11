"""The eval harness's target function: runs one eval case through the same
production pipeline as app.py -- agent/safety.py's crisis gate first, then
the real agent -- for as many turns as the case provides, accumulating one
continuous conversation.

agent_graph is injectable (same dependency-injection-for-testability
pattern used throughout this project: tools/csv_log.py's `path`,
agent/graph.py's `model`/`tools`, app.py's `agent_graph`/`recorder`) so
tests can stub it without a real OpenAI call.
"""
from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Any

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, ToolMessage

from agent.graph import build_agent, message_text
from agent.safety import screen_for_crisis

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


def _content_str(content: Any) -> str:
    return content if isinstance(content, str) else str(content)


def _isolated_log_path() -> Path:
    """A throwaway mood log for one eval case.

    Two reasons this must never default to the real data/mood_log.csv:
    a run would inject 28 fabricated entries into the user's personal log,
    and -- since the stage-14 redesign classifies every entry -- those
    fixtures would also enter `known_tags`, nudging future *real*
    classifications toward vocabulary invented by test data.

    Fresh per case, so cases can't see each other's entries and results
    don't depend on execution order. Cases that need seeded history pass
    their own prepared `log_path` instead.
    """
    return Path(tempfile.mkdtemp(prefix="joblog-eval-")) / "mood_log.csv"


def _seeded_log_path(seed_entries: list[dict]) -> Path:
    """An isolated log pre-populated with fixture entries.

    Needed for any case about *history*: "have I dealt with this before?",
    whether the agent overstates a loose match, and whether instruction-like
    text sitting in a past entry gets obeyed when the history lookup
    surfaces it. Without seeding, every eval case runs against an empty log
    and none of that is reachable.

    Written through the real append_mood_row so fixtures go through exactly
    the same schema and tag encoding as production rows.
    """
    from tools.csv_log import append_mood_row

    path = Path(tempfile.mkdtemp(prefix="joblog-evalseed-")) / "mood_log.csv"
    for entry in seed_entries:
        append_mood_row(
            mood=entry.get("mood", "neutral"),
            circumstances=entry.get("circumstances", ""),
            energy=entry.get("energy") or None,
            measure_taken=entry.get("measure_taken") or None,
            helpfulness=entry.get("helpfulness") or None,
            situation=entry.get("situation") or None,
            thoughts=entry.get("thoughts") or None,
            need=entry.get("need") or None,
            next_step=entry.get("next_step") or None,
            tags=entry.get("tags") or None,
            path=path,
        )
    return path


def _unwritable_log_path() -> Path:
    """A log path whose parent is a regular file, so every write genuinely
    fails. Used by the `simulate_write_failure` case to exercise the
    fallback-and-report-honestly path for real, rather than trusting that
    it works."""
    directory = Path(tempfile.mkdtemp(prefix="joblog-evalfail-"))
    blocker = directory / "not-a-directory"
    blocker.write_text("blocks the write below", encoding="utf-8")
    return blocker / "mood_log.csv"


def run_case(inputs: dict, *, agent_graph: Any = None, log_path: Path | None = None) -> dict:
    """The function LangSmith's evaluate() calls once per Example (and the
    same logic run_baseline.py's no-tools comparison is built against).

    inputs = {"turns": [...]}. Checks the crisis gate before every turn,
    exactly like app.py's _on_send -- if it fires, stops immediately and
    returns crisis_intercepted=True with zero agent messages for that turn,
    never invoking the LLM. Otherwise runs the real agent, accumulating the
    full message list across turns (the agent has no checkpointer -- see
    agent/graph.py -- so this accumulation has to happen here, not
    implicitly via agent memory).
    """
    if agent_graph is not None:
        graph = agent_graph
    else:
        if log_path is None:
            if inputs.get("simulate_write_failure"):
                log_path = _unwritable_log_path()
            elif inputs.get("seed_entries"):
                log_path = _seeded_log_path(inputs["seed_entries"])
            else:
                log_path = _isolated_log_path()
        graph = build_agent(log_path=log_path)

    messages: list[BaseMessage] = []

    for turn_text in inputs["turns"]:
        crisis = screen_for_crisis(turn_text)
        if crisis.triggered:
            return {
                "messages": serialize_messages(messages),
                "crisis_intercepted": True,
                "matched_pattern": crisis.matched_pattern,
            }
        messages_before = messages + [HumanMessage(content=turn_text)]
        result = graph.invoke({"messages": messages_before})
        messages = result["messages"]

    return {"messages": serialize_messages(messages), "crisis_intercepted": False}
