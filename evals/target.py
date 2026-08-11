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

from typing import Any

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, ToolMessage

from agent.graph import build_agent
from agent.safety import screen_for_crisis

# Role vocabulary must match dataset.jsonl's hand-authored reference_trajectory
# entries exactly ("human"/"ai"/"tool") -- agentevals' structural comparison
# only checks string equality, so both sides just need to agree.


def serialize_messages(messages: list[BaseMessage]) -> list[dict]:
    result = []
    for m in messages:
        if isinstance(m, HumanMessage):
            result.append({"role": "human", "content": _content_str(m.content)})
        elif isinstance(m, ToolMessage):
            result.append(
                {
                    "role": "tool",
                    "content": _content_str(m.content),
                    "name": m.name,
                    "status": m.status,
                }
            )
        elif isinstance(m, AIMessage):
            tool_calls = [{"name": tc["name"], "args": tc.get("args", {})} for tc in (m.tool_calls or [])]
            result.append(
                {
                    "role": "ai",
                    "content": _content_str(m.content),
                    "tool_calls": tool_calls or None,
                }
            )
        else:
            result.append({"role": getattr(m, "type", "unknown"), "content": _content_str(m.content)})
    return result


def _content_str(content: Any) -> str:
    return content if isinstance(content, str) else str(content)


def run_case(inputs: dict, *, agent_graph: Any = None) -> dict:
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
    graph = agent_graph if agent_graph is not None else build_agent()
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
