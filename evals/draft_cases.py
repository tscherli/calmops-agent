"""Draft the user's side of an eval case by playing it against the real agent.

**Play once, then freeze.** A model is given a persona and a hidden goal and
converses with the real `gpt-5-mini` agent for one pass. The user's turns are
kept, the agent's are thrown away, and what is left is read, edited by hand and
frozen into `evals/dataset.jsonl` as a fixed script.

Why not draft turns blind from a written brief, which costs nothing: turns
written without seeing the agent are written against an *imagined* agent, and
they stop making sense the moment the real one asks its questions in another
order. Turns captured from a real exchange answer what the agent actually
asks. The agent's behaviour is also visible at the moment the `expect` block
is written, which is exactly when it matters for telling a guarantee from a
description of what happened.

Why this is not a simulated user at run time, which the plan rejects: nothing
here runs during an eval. The script is frozen, every run replays the same
turns, and run-to-run variance is unchanged.

**This writes turns and nothing else.** No ids, no tiers, no `expect`, no
`expected_behavior`. The drafter is asked for a person, not for a test, and
`_reject_assertions` refuses output that carries assertion keys -- a generated
assertion that is wrong does not fail the suite, it passes it, which is the
one failure mode a golden dataset cannot survive.

Usage:

    python -m evals.draft_cases --brief evals/briefs/measure_choice.json
    python -m evals.draft_cases --brief ... --turns 6 --out -

Nothing it writes reaches the dataset without being read. That is the point of
the "then by hand" half of the name.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

load_dotenv()

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage

from agent.common import message_text
from agent.graph import CARRIED_FIELDS, next_turn_state
from agent.safety import screen_for_crisis
from evals.target import (
    _isolated_log_path,
    _isolated_measures_path,
    _unwritable_log_path,
    build_eval_graph,
    seed_log,
    seed_measures,
)

# The drafter. Pro rather than the judge's flash tier: writing a person who is
# terse, changes their mind and says no is the one job here where capability
# shows, and it runs a few dozen times offline rather than on every case of
# every run. Its output is one sentence per call, so the expensive output rate
# barely applies -- the cost is almost all input.
#
# A `-preview` model is acceptable for this and would not be for the judge.
# evals/judge.py pins a stable id because a judge on a moving target silently
# changes recorded numbers between runs; the drafter runs once, offline, and
# its output is frozen by hand. If the preview is withdrawn, the dataset is
# already written.
AUTHOR_MODEL = os.environ.get("EVAL_AUTHOR_MODEL", "google_genai:gemini-3.1-pro-preview")

DEFAULT_TURNS = 6

# Keys a drafter must never return. Asked for a test case rather than a person,
# a capable model will happily invent assertions, and a wrong assertion does
# not go red -- it goes green while checking something nobody intended.
ASSERTION_KEYS = frozenset(
    {"expect", "expected_behavior", "id", "tier", "covers", "difficulty", "notes", "category"}
)

PERSONA_PROMPT = """You are playing a person using a private check-in app about how work is going. \
You are NOT the assistant. You write only what the person types.

<Who you are right now>
{persona}
</Who you are right now>

<What you want from this conversation>
{goal}
</What you want from this conversation>

<What the app already knows about you>
{known}
</What the app already knows about you>

<Measures that exist in your own list>
These are things you wrote down previously. You may refer to them the way a person refers to their \
own notes -- vaguely, by half the name, or not at all. Never quote this list verbatim.
{measures}
</Measures that exist in your own list>

<How to write>
- One short message at a time. Most real messages are under fifteen words.
- Answer the question you were actually asked, or don't. People often answer a \
different question, or ignore one and volunteer something else.
- You are allowed to be unhelpful: one-word replies, "no", "not that one", changing your mind, \
saying you already tried something, going quiet on a question you find annoying.
- Lowercase, missing punctuation and typos are normal. Do not write like a customer service email.
- Never describe yourself in the third person, never narrate ("[user pauses]"), never explain what \
you are doing. Just type the message.
- Never mention that this is a test, a scenario or a simulation.

<The conversation so far>
{transcript}
</The conversation so far>

Write only the person's next message, as plain text. No quotes around it, no name prefix, nothing \
else."""


def load_brief(path: Path) -> dict:
    """A case brief: who the person is, what they want, what to plant.

    Hand-written, always. The scenario comes from a hand test, a trace or an
    observed problem -- never from the drafter, which is given a person and
    asked to speak, not asked what to test.
    """
    brief = json.loads(path.read_text(encoding="utf-8"))
    for field in ("persona", "goal"):
        if not brief.get(field):
            raise SystemExit(f"{path}: brief needs a non-empty {field!r}")
    leaked = ASSERTION_KEYS & set(brief)
    if leaked:
        raise SystemExit(
            f"{path}: a brief describes a person, not a test. Remove {sorted(leaked)} -- "
            "those are written by hand after reading the transcript."
        )
    approval = brief.get("approval", "approve")
    if approval not in ("approve", "reject", "edit"):
        raise SystemExit(
            f"{path}: approval must be approve, reject or edit -- {approval!r} is none of them, "
            "and an unrecognised value silently approves, which is the one outcome a brief "
            "about a declined card must not get."
        )
    unknown = set(brief.get("state") or {}) - set(CARRIED_FIELDS)
    if unknown:
        raise SystemExit(
            f"{path}: {sorted(unknown)} is not carried between turns. A brief may set only "
            f"{list(CARRIED_FIELDS)}, because that is what a client puts in the payload; "
            "anything else is silently ignored by the graph and the pass would play from a "
            "state the case can never reproduce."
        )
    return brief


def _reject_assertions(text: str) -> str:
    """Refuse a drafter that has started writing the test instead of the turn.

    Loud rather than quiet: a returned `expect` block that got silently
    dropped would look identical to one that was never produced, and the next
    person would not know the prompt had stopped working.
    """
    stripped = text.strip()
    if stripped.startswith("{") or stripped.startswith("["):
        raise SystemExit(
            "The drafter returned structured data instead of a message. It is asked for "
            "one line of what the person types; the case around it is written by hand."
        )
    lowered = stripped.lower()
    for key in sorted(ASSERTION_KEYS):
        if f'"{key}"' in lowered or f"'{key}'" in lowered:
            raise SystemExit(f"The drafter returned an assertion key ({key!r}). Rejected.")
    return stripped


def _format_measures(rows: list[dict]) -> str:
    return "\n".join(f"- {r['measure_id']}: {r['measure']}" for r in rows) or "- (nothing yet)"


def _card_line(request: dict, approval: str) -> str:
    """The approval card, as the person saw it.

    A card is an interrupt rather than a message, so nothing in the transcript
    shows it. But it is the most visible thing that happens in the app -- the
    conversation stops and asks a yes/no question -- and a drafter that cannot
    see it writes the next turn as though nothing had been proposed. Rendering
    it is also the only way the drafter learns that a rejected card was
    rejected.
    """
    verb = {"reject": "declined it", "edit": "edited the wording and confirmed"}.get(
        approval, "confirmed it"
    )
    return f"[a card asked you to confirm: {request.get('description', '')}] You {verb}."


def _describe_state(state: dict) -> str:
    """What the app already knows, told to the drafter in the person's own terms.

    A brief may open the pass mid-conversation, the way a case opens with
    `energy` and `goal` in its first turn (the payload shortcut,
    agent/graph.py:512). Without telling the drafter, the pass plays from a
    state the person is unaware of: it writes an opening message that answers
    a question already answered, and the frozen turns read as a non-sequitur
    to anyone who later reads the case.
    """
    labels = {
        "energy": "Your energy today, which you have already given it:",
        "goal": "What you said you want from today:",
        "situation": "What you already told it is going on:",
        "idea": "An idea of your own you have already put forward:",
    }
    lines = [f"- {labels[key]} {state[key]}" for key in labels if state.get(key)]
    return "\n".join(lines) or "Nothing yet. This is the first thing you have said today."


def _is_user_visible(message: BaseMessage) -> bool:
    """Would the person see this in the window?

    Only their own messages and the agent's spoken replies. Tool results,
    tool-call stubs and reasoning blocks are not shown to a user and must not
    be shown to someone playing one -- the first pass handed the drafter all
    31 rows of the measure list rendered as JSON, which wastes most of the
    prompt and contradicts the short list it is given on purpose.
    """
    if isinstance(message, HumanMessage):
        return True
    if isinstance(message, AIMessage) and not message.tool_calls:
        return bool(message_text(message).strip())
    return False


def _format_transcript(messages: list[BaseMessage]) -> str:
    visible = [m for m in messages if _is_user_visible(m)]
    if not visible:
        return "(nothing yet -- write the opening message)"
    return "\n".join(
        f"{'You' if isinstance(m, HumanMessage) else 'The app'}: {message_text(m).strip()}"
        for m in visible
    )


def build_author(model: str = AUTHOR_MODEL):
    from langchain.chat_models import init_chat_model

    return init_chat_model(model)


def draft(
    brief: dict,
    *,
    turns: int = DEFAULT_TURNS,
    author: Any = None,
    agent_graph: Any = None,
    measures_path: Path | None = None,
    log_path: Path | None = None,
) -> dict:
    """One pass: the drafter speaks, the real agent answers, repeat.

    Returns the user's turns and the transcript that produced them. The
    transcript is for reading while the `expect` block is written; only the
    turns go into the dataset.
    """
    # The same switch a case carries (evals/target.py:237). A brief about what
    # the agent says when a write fails has to play against a write that really
    # fails: drafted over a working log, every turn after the write answers a
    # success the case will never produce.
    if log_path is None:
        log_path = (
            _unwritable_log_path() if brief.get("simulate_write_failure") else _isolated_log_path()
        )
    measures_path = measures_path or _isolated_measures_path()
    if brief.get("seed_measures"):
        seed_measures(brief["seed_measures"], measures_path)
    if brief.get("seed_entries"):
        seed_log(brief["seed_entries"], log_path)

    author = author if author is not None else build_author()
    graph = agent_graph if agent_graph is not None else build_eval_graph(log_path, measures_path)

    from tools.measures import read_measures

    measures = _format_measures(read_measures(measures_path))

    import uuid

    from langgraph.types import Command

    from evals.target import _decision_for

    # The payload shortcut, the same one a case's first turn uses: whatever the
    # brief sets is carried into the first invoke and then maintained by
    # next_turn_state, so the pass plays from the state the case will start in.
    opening_state = {key: value for key, value in (brief.get("state") or {}).items()}
    approval = str(brief.get("approval") or "approve")
    edit_text = str(brief.get("edit_text") or "")
    state: dict[str, Any] = {"messages": [], **opening_state}
    transcript: list[BaseMessage] = []
    user_turns: list[str] = []

    for _ in range(turns):
        prompt = PERSONA_PROMPT.format(
            persona=brief["persona"],
            goal=brief["goal"],
            known=_describe_state(opening_state),
            measures=measures,
            transcript=_format_transcript(transcript),
        )
        said = _reject_assertions(message_text(author.invoke(prompt)))
        user_turns.append(said)
        transcript.append(HumanMessage(content=said))

        # The gate ends the conversation exactly as it would in the app. A
        # drafter that wanders into crisis language stops the pass rather than
        # producing turns the harness could never replay.
        if screen_for_crisis(said).triggered:
            break

        messages_before = list(state.get("messages") or []) + [HumanMessage(content=said)]
        config = {"configurable": {"thread_id": uuid.uuid4().hex}}
        result = graph.invoke({**state, "messages": messages_before}, config)
        while result.get("__interrupt__"):
            requests = result["__interrupt__"][0].value["action_requests"]
            # Drafting approves by default. What the user does with a card is
            # the case's decision, and refusing everywhere would end most
            # conversations before they reach a write. A brief may say
            # otherwise, because a case about what the agent says *after* a
            # rejected card cannot be drafted from a pass that approved it:
            # the turns would answer a reply the case never produces.
            decisions = [{"id": r["id"], **_decision_for(r, approval, edit_text)} for r in requests]
            transcript.extend(AIMessage(content=_card_line(r, approval)) for r in requests)
            result = graph.invoke(Command(resume={"decisions": decisions}), config)

        produced = result["messages"][len(messages_before):]
        transcript.extend(produced)
        state = next_turn_state(result)
        if result.get("complete"):
            break

    return {
        "turns": user_turns,
        # Two views, because they have different readers. `conversation` is
        # what the person saw, and it is what you read while deciding what the
        # case should assert. `tools` is what actually ran, which is the other
        # half of writing an honest expect block.
        "conversation": [
            {"who": "user" if isinstance(m, HumanMessage) else "agent", "text": message_text(m).strip()}
            for m in transcript
            if _is_user_visible(m)
        ],
        "tools": [
            {"name": tc["name"], "args": tc.get("args", {})}
            for m in transcript
            if isinstance(m, AIMessage)
            for tc in (m.tool_calls or [])
        ],
        "measures_path": str(measures_path),
        "log_path": str(log_path),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Draft a case's user turns by playing them against the real agent"
    )
    parser.add_argument("--brief", type=Path, required=True, help="A hand-written case brief.")
    parser.add_argument("--turns", type=int, default=DEFAULT_TURNS, help="Maximum user turns.")
    parser.add_argument("--model", default=AUTHOR_MODEL)
    parser.add_argument("--out", default="-", help="Where to write the result; - for stdout.")
    args = parser.parse_args()

    result = draft(load_brief(args.brief), turns=args.turns, author=build_author(args.model))
    payload = json.dumps(result, indent=2, ensure_ascii=False)
    if args.out == "-":
        print(payload)
    else:
        Path(args.out).write_text(payload, encoding="utf-8")
        print(f"Wrote {len(result['turns'])} turns to {args.out}")
    print(
        "\nNow by hand: read every turn, cut what the person would not say, write the expect "
        "block, assign the id and the suite tier, name what it covers, register it in "
        "evals/case_ids.txt."
    )


if __name__ == "__main__":
    main()
