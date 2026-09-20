"""Helpers shared by every version of the graph: receipts, message plumbing.

These outlive individual node designs. Receipts in particular are the
project's one hard guarantee about side effects -- they are set by the graph
from a tool's real return value, never by the model -- so they live in their
own module rather than inside whichever graph happens to be current.
"""
from __future__ import annotations

import json
from collections.abc import Sequence
from typing import Any

from langchain_core.messages import AIMessage, BaseMessage, ToolMessage

# -- tool receipts -----------------------------------------------------------


def merge_tool_events(
    left: list[dict[str, Any]] | None, right: list[dict[str, Any]] | None
) -> list[dict[str, Any]]:
    """Accumulate tool events across the nodes of one turn.

    A plain append reducer, not a replace: a single turn can run several
    tool-bearing nodes, and the UI reports each one as it finishes. Clients do
    not pass `tool_events` back in on the next turn, so each turn starts empty.
    """
    return (left or []) + (right or [])


def tool_event(tool: str, status: str, detail: str = "", *, verified: bool = True) -> dict[str, Any]:
    """One machine-set record of what a tool actually did.

    Set by the graph from the tool's real return value, never by the model.
    That is the whole point: the agent cannot emit one of these, and cannot
    suppress one, so a claim in its prose can be checked against them.

    `exit_code` is deliberately None for "skipped". A tool that never ran has
    no exit code, and reporting it as 0 would be its own small lie.

    `verified=False` marks an event we did NOT observe in our own code --
    currently only OpenAI's server-side web_search, whose execution is
    reported by the provider rather than witnessed here.
    """
    return {
        "tool": tool,
        "status": status,
        "exit_code": {"ok": 0, "error": 1}.get(status),
        "detail": detail,
        "verified": verified,
    }


def result_detail(result: Any) -> str:
    """A short, honest summary of what a tool returned."""
    if isinstance(result, list):
        return f"{len(result)} results" if result else "no matches"
    if isinstance(result, dict):
        return str(result.get("status") or f"{len(result)} fields")
    text = str(result)
    return text if len(text) <= 60 else text[:57] + "..."


def web_search_ran(message: BaseMessage) -> bool:
    """Did OpenAI's server-side web_search run on this response?

    Read off the Responses-API content blocks, which is the only evidence
    available: the search executes on OpenAI's infrastructure and its payload
    never passes through our code. That is why the event it produces is marked
    verified=False -- we are relaying the provider's report, not witnessing the
    call the way we witness a local write.
    """
    content = getattr(message, "content", None)
    if not isinstance(content, list):
        return False
    for block in content:
        block_type = block.get("type") if isinstance(block, dict) else getattr(block, "type", None)
        if isinstance(block_type, str) and "web_search" in block_type:
            return True
    return False


# -- messages ----------------------------------------------------------------


def message_text(message: BaseMessage) -> str:
    """Plain display text for a message, whatever shape its content is in.

    With the Responses API, `AIMessage.content` is a list of content blocks
    rather than a string -- so reading `.content` straight into the UI prints
    raw dicts and base64 at the user. `.text` is clean in both modes.
    """
    text = getattr(message, "text", None)
    # isinstance first: in langchain-core 1.5.3 `.text` is a property returning
    # a str that is *also* callable, so checking callable() first would take
    # the deprecated method path on every message.
    if not isinstance(text, str) and callable(text):
        text = text()
    if isinstance(text, str) and text:
        return text
    content = message.content
    return content if isinstance(content, str) else str(content)


def as_tool_content(result: Any) -> str:
    if isinstance(result, str):
        return result
    try:
        return json.dumps(result, ensure_ascii=False, default=str)
    except (TypeError, ValueError):
        return str(result)


def last_ai_tool_calls(messages: Sequence[BaseMessage]) -> list[dict[str, Any]]:
    for message in reversed(messages):
        if isinstance(message, AIMessage):
            return list(message.tool_calls or [])
    return []


def answered_tool_call_ids(messages: Sequence[BaseMessage]) -> set[str]:
    return {m.tool_call_id for m in messages if isinstance(m, ToolMessage) and m.tool_call_id}


def pending_tool_calls(messages: Sequence[BaseMessage]) -> list[dict[str, Any]]:
    """Tool calls from the last AI message that nothing has answered yet.

    A model can ask for several tools in one message; every call needs exactly
    one ToolMessage back, or the next request is rejected.
    """
    answered = answered_tool_call_ids(messages)
    return [call for call in last_ai_tool_calls(messages) if call.get("id") not in answered]


def compact_spent_tool_messages(
    messages: list[BaseMessage], tool_names: Sequence[str] = ("log_mood",), limit: int = 200
) -> list[BaseMessage]:
    """Shrink already-used tool results to a marker.

    A fat payload is needed for the reply on its own turn and then re-sent on
    every later turn of the conversation. `tool_names` says which tools produce
    one; the caller decides, because that changes with the graph.
    """
    out: list[BaseMessage] = []
    for message in messages:
        if (
            isinstance(message, ToolMessage)
            and message.name in tool_names
            and len(message.content or "") > limit
        ):
            try:
                data = json.loads(message.content)
                slim = {"status": data.get("status"), "row_id": data.get("row_id")}
            except (ValueError, TypeError):
                slim = {"status": "ok"}
            out.append(
                ToolMessage(
                    content=as_tool_content(slim),
                    name=message.name,
                    tool_call_id=message.tool_call_id,
                    status=message.status,
                )
            )
        else:
            out.append(message)
    return out
