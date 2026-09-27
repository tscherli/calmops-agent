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

from agent.graph import SITUATION_DECLINED

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


def conversation(outputs: dict) -> list[str]:
    """What a person would have seen on screen, in order.

    Their own messages and the agent's spoken replies. Tool calls, tool
    results and reasoning blocks are not shown to a user and must not be shown
    to someone reading the exchange back.

    Why this exists rather than a list of user turns: a labelling file built
    from `user_turns` plus the *final* reply drops every agent turn in
    between, so "yeah go ahead" sits in the evidence with nothing visible to
    be agreeing to. The first labelling session hit that on its second item.
    The judge never had the problem -- it is handed the whole message list as
    context -- which made this an asymmetry between the two raters rather than
    a missing convenience.
    """
    lines = []
    for message in outputs.get("messages") or []:
        text = str(message.get("content") or "").strip()
        if not text:
            continue
        if message.get("role") == "human":
            lines.append(f"user: {text}")
        elif message.get("role") == "ai" and not message.get("tool_calls"):
            lines.append(f"agent: {text}")
    return lines


def situation_sentences(outputs: dict) -> list[str]:
    """Every sentence the agent wrote about what is going on for the user.

    Two places, and both are the agent's own words rather than the user's:
    the `circumstances` it proposed on an approval card, and the `situation`
    it recorded through record_choices. Either can assert something no turn
    said, and the recorded one is the more consequential of the two -- it is
    carried into every later turn and shapes what gets suggested, while the
    card's only reaches the log.

    Why this is not just `card_sentences`: the judge's situation question was
    graded on cards alone, and across a full sweep of the suite exactly two
    trajectories reached one. Twenty labels of it would have taken ten sweeps
    of the same two cases, which is not twenty observations -- so the question
    could never have been calibrated, and an uncalibratable question is worse
    than a slightly broader one. Recorded situations bring it within reach.

    The declined marker is not a sentence about anything: "not given" is the
    value that means the user would rather not say, and grading it against
    what they said would fail every case that respected the refusal.
    """
    sentences: list[str] = []
    for approval in outputs.get("approvals") or []:
        sentence = str((approval.get("args") or {}).get("circumstances") or "").strip()
        if sentence:
            sentences.append(sentence)
    for message in outputs.get("messages") or []:
        for call in message.get("tool_calls") or []:
            if call.get("name") != "record_choices":
                continue
            sentence = str((call.get("args") or {}).get("situation") or "").strip()
            if sentence and sentence.lower() != SITUATION_DECLINED:
                sentences.append(sentence)
    return list(dict.fromkeys(sentences))


def user_turns(outputs: dict) -> list[str]:
    """Everything the user said, one entry per turn."""
    return [
        (m.get("content") or "")
        for m in (outputs.get("messages") or [])
        if m.get("role") == "human"
    ]


def searched_the_web(outputs: dict) -> bool:
    """Did the provider's web search run anywhere in this conversation?

    Read from `tool_events`, because that is the only place it is recorded.
    The search runs on OpenAI's side and comes back as a content block, which
    evals/target.py flattens to text with the rest of the message -- so the
    message list the judge is handed as context carries no trace of it.

    The judge's grounding question allows ideas found on the web, and from the
    day that allowance was written (2026-09-26) it could never apply it: it was
    told "if the trajectory shows a web search" and the trajectory never did. The labelling file read
    this flag and the judge did not, so on hard_tool_failure_write_39 the human
    correctly allowed a cited web idea and the judge failed it. One reader,
    used by both, is what keeps the two raters on the same evidence.
    """
    return any(event.get("tool") == "web_search" for event in outputs.get("tool_events") or [])


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
    "searched_the_web",
    "user_turns",
]
