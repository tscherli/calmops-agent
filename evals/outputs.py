"""Readers over what the harness returns.

Both graders ask questions about the same three things: what the lookup gave
the model, what the card proposed, and what the user actually said. The
deterministic expect block and the LLM judge have to read them the same way --
two readings of "what the lookup returned" that drift apart would put the two
graders quietly at odds, and the report would show one passing and the other
failing with nothing to explain it.

Nothing here judges anything. These functions only say what happened.
"""
from __future__ import annotations

import json


def lookup_results(outputs: dict) -> list[list[dict]]:
    """The rows every find_measures call returned, as the tool returned them.

    The tool answers with a JSON list (tools/measures.py builds it through
    as_tool_content), so this reads the real result rather than the model's
    account of it. A result that does not parse is reported as an empty
    lookup -- correctly: whatever the model was handed, it was not the list.
    """
    results = []
    for message in outputs.get("messages") or []:
        if message.get("role") == "tool" and message.get("name") == "find_measures":
            try:
                parsed = json.loads(message.get("content") or "")
            except (TypeError, ValueError):
                parsed = []
            results.append(parsed if isinstance(parsed, list) else [])
    return results


def offered_measures(outputs: dict) -> list[dict]:
    """Every row any lookup in this case returned, without repeats.

    A case can look up more than once, and a measure shown after the second
    lookup came from the list just as much as one shown after the first.
    """
    seen: dict[str, dict] = {}
    for rows in lookup_results(outputs):
        for row in rows:
            if isinstance(row, dict) and row.get("measure_id"):
                seen.setdefault(row["measure_id"], row)
    return list(seen.values())


def card_sentences(outputs: dict) -> list[str]:
    """The situation sentence on every approval card that carried one.

    Read from the card rather than from the written row: an edit turn
    overwrites the row with the user's own wording, so a row can look
    faultless after the agent proposed something the user never said.
    """
    sentences = []
    for approval in outputs.get("approvals") or []:
        sentence = str((approval.get("args") or {}).get("circumstances") or "")
        if sentence:
            sentences.append(sentence)
    return sentences


def user_turns(outputs: dict) -> list[str]:
    """Everything the user said, one entry per turn."""
    return [
        (m.get("content") or "")
        for m in (outputs.get("messages") or [])
        if m.get("role") == "human"
    ]


def final_reply(outputs: dict) -> str:
    """The last thing the agent said with no tool call attached to it.

    That is the message the user reads, and the only place a measure can be
    "presented" to them.
    """
    replies = [
        m for m in (outputs.get("messages") or [])
        if m.get("role") == "ai" and not m.get("tool_calls")
    ]
    return (replies[-1].get("content") or "") if replies else ""


__all__ = [
    "card_sentences",
    "final_reply",
    "lookup_results",
    "offered_measures",
    "user_turns",
]
