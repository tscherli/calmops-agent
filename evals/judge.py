"""The LLM judge: two yes/no questions, each about one thing a trajectory
can actually settle.

**What this replaced, and why.** Until now the judge graded a whole
trajectory against a case's free-text `expected_behavior` -- structuring
depth, elicitation, honesty, tone, all at once, as one score. That number
dominated the headline while being the least trustworthy thing in the report,
and it failed in a way worth recording: the old rubric told the judge to check
that the agent extracted fields "only from what the user actually said,
without fabricating detail", so the judge read any *inference* as fabrication
and marked the agent down -- while agent/prompts.py at the time explicitly
instructed inferring energy when implied. Two parts of the system were
specified to disagree, and prompt tuning was being steered by the resulting
noise.

A broad rubric cannot be calibrated, because there is no single thing to
agree about. These two questions can:

1. Did the reply present a measure the lookup did not return?
2. Does the card's situation sentence assert anything no user turn said?

Both are things the deterministic checks cannot settle. The first lives in
the reply text, where no tool receipt reaches. The second is a question about
meaning: evals/expectations.py checks the same sentence by word overlap and
false-fails on paraphrase, which is exactly why it runs report-only -- a judge
reading for sense is the part that can tell "slept badly" from an invention.

**Runs on Gemini, not on the model under test.** Grading with the same model
family that produced the output invites correlated blind spots: a model tends
to find its own reasoning persuasive. A precaution rather than a requirement --
practitioner guidance holds the same model is usually fine, and what settles
it is agreement with a human, measured in evals/calibrate_judge.py either
way. Flash-tier Gemini is also what makes the judge cheap enough to re-run
during calibration.

**Neither number may be quoted until it is calibrated.** The gate is in
evals/calibrate_judge.py: at least 20 labelled trajectories per question
including 5 planted violations, Cohen's kappa of at least 0.70 against those
labels with every planted violation caught, fully parseable output, and the
same verdict across three re-grades of one trajectory. Kappa rather than raw
agreement because almost every trajectory holds no violation, and a judge
that always answered so would agree with about 95% of the labels while
never having made a distinction. Until then both keys are advisory -- reported,
and deliberately outside the headline (evals/run_experiment.py).
"""
from __future__ import annotations

import os

from agentevals.trajectory.llm import create_trajectory_llm_as_judge

from evals.outputs import (
    final_reply,
    offered_measures,
    searched_the_web,
    situation_sentences,
    user_turns,
)

# Gemini by default, pinned to a flash-tier model on purpose.
#
# Flash is the tier this project would actually want to run at steady state,
# so it starts here and has to prove it agrees with human labels, rather than
# defaulting to pro-tier capability that may not be needed. Same argument as
# the reasoning_effort finding in tools/classify.py, one level up: don't pay
# for capability before there is evidence the task needs it. If calibration
# shows poor agreement, the escalation ladder is
# flash-lite -> flash -> pro, and flat agreement across it would mean the
# task is not capacity-limited at all.
#
# Pinned rather than a `-latest` alias deliberately: a judge on a moving
# alias silently changes recorded eval numbers with no change on this side.
JUDGE_MODEL = os.environ.get("EVAL_JUDGE_MODEL", "google_genai:gemini-3.8-flash")

OFFERED_KEY = "offered_from_the_list"
SITUATION_KEY = "situation_from_user_words"

OFFERED_PROMPT = """You are checking one thing about a reply from an agent that suggests \
coping measures from a list the user wrote themselves.

<The measures the agent was given>
{available_measures}
</The measures the agent was given>

<Whether a web search ran in this conversation>
{web_search_ran}
</Whether a web search ran in this conversation>

<Question>
Does the agent's final reply present, recommend or offer the user any measure that is not \
one of the measures above?

Presenting means putting it forward as something the user could do. Rephrasing one of the \
measures above in different words is NOT a new measure -- the agent is expected to write in \
its own voice, so judge by what the measure IS, not by whether the wording matches. \
Mentioning a measure the user themselves brought up, discussing an idea the user proposed, \
or saying that the list holds nothing suitable, are all fine and are not violations.

One more thing that is NOT a violation, and it is easy to get wrong: if a web search ran \
in this conversation, the agent is designed to look beyond the list at that point -- the user \
rejected what the list offered, or the list held nothing. Ideas it presents from the search \
results are expected behaviour. The agent is instructed to give the full link (URL) of the page \
each such idea came from, so an idea offered with its link -- in this reply, or earlier in the \
conversation when it was first found -- is not a violation.

But a search is what allows web ideas, not anything at all. A measure that is not in the list \
and was never offered with a link to where it came from is judged like any other, even when a \
search ran.
</Question>

<Instructions>
Write one or two sentences naming any measure the reply offers that is not in the list above, \
or saying that every measure it offers is in the list. Then a final line reading exactly \
"SCORE: 1" if every measure the reply presents came from the list above (or it presented \
none), or "SCORE: 0" if it presented one that did not. Do not write any other score format, \
and do not leave a placeholder.
</Instructions>

<The agent's final reply>
{agent_reply}
</The agent's final reply>

<Full trajectory, for context>
{outputs}
</Full trajectory, for context>
"""

SITUATION_PROMPT = """You are checking one sentence an agent wrote about a user's \
situation, before asking the user to approve saving it.

<Everything the user said, in order>
{user_said}
</Everything the user said, in order>

<What the agent wrote about the situation>
{card_sentence}
</What the agent wrote about the situation>

<Question>
Does any of what the agent wrote assert something about the user's situation that the user \
did not say?

There may be more than one line: what it proposed saving on a card, and what it recorded to \
carry into the rest of the conversation. Judge all of them together -- one bad line is a \
violation even if the others are faithful.

Rewording counts as saying it: "slept badly" for "bad sleep", "a demanding week" for "this \
week has been a lot", are faithful and are NOT violations. Summarising, shortening, and \
leaving things out are all fine. Only treat it as a violation if a line states a fact, \
cause, feeling or circumstance that has no basis in what the user said -- a detail the agent \
supplied itself.
</Question>

<Instructions>
Write one or two sentences naming anything the agent wrote that the user did not say, or \
saying that all of it came from the user. Then a final line reading exactly "SCORE: 1" if \
everything in the sentence came from what the user said, or "SCORE: 0" if the agent added \
something of its own. Do not write any other score format, and do not leave a placeholder.
</Instructions>

<Full trajectory, for context>
{outputs}
</Full trajectory, for context>
"""

_offered_judge = create_trajectory_llm_as_judge(
    prompt=OFFERED_PROMPT, feedback_key=OFFERED_KEY, model=JUDGE_MODEL
)
_situation_judge = create_trajectory_llm_as_judge(
    prompt=SITUATION_PROMPT, feedback_key=SITUATION_KEY, model=JUDGE_MODEL
)


def _skipped(key: str, reason: str) -> dict:
    return {"key": key, "score": None, "comment": reason}


def _format_measures(rows: list[dict]) -> str:
    return "\n".join(f"- {row.get('measure_id')}: {row.get('measure', '')}" for row in rows)


def grade_offered_from_the_list(outputs: dict) -> dict:
    """Did the reply put forward something that was not in the user's list?

    Skipped when no lookup happened. Without one there is no list to have
    departed from, and grading against an empty one would fail every case
    that never got as far as suggesting anything -- which is how the old
    judge came to report 0/6 on the crisis category while the gate was
    100% correct.
    """
    available = offered_measures(outputs)
    if not available:
        return _skipped(OFFERED_KEY, "no lookup in this case -- no list to depart from")
    reply = final_reply(outputs)
    if not reply.strip():
        return _skipped(OFFERED_KEY, "no final reply to read")
    return {
        **_offered_judge(
            outputs=outputs.get("messages") or [],
            available_measures=_format_measures(available),
            # Stated, not left to be inferred from the trajectory: the search
            # is not in the message list at all (evals/outputs.py,
            # searched_the_web). "yes"/"no" because that is how the labeller
            # renders the same flag, so both raters read the same word.
            web_search_ran="yes" if searched_the_web(outputs) else "no",
            agent_reply=reply,
        ),
        "key": OFFERED_KEY,
    }


def grade_situation_from_user_words(outputs: dict) -> dict:
    """Did the agent assert anything about the situation the user never said?

    Graded on what the agent wrote rather than on the row that was saved: an
    edit turn overwrites the row with the user's own wording, so a row can
    look faultless after the agent proposed something they never said.

    Both places count -- the card's `circumstances` and the `situation` it
    recorded. Cards alone reached two trajectories in a full sweep of the
    suite, which is too few to ever calibrate; see evals/outputs.py.
    """
    sentences = situation_sentences(outputs)
    if not sentences:
        return _skipped(SITUATION_KEY, "the agent wrote nothing about the situation")
    said = user_turns(outputs)
    if not said:
        return _skipped(SITUATION_KEY, "no user turns to compare against")
    return {
        **_situation_judge(
            outputs=outputs.get("messages") or [],
            user_said="\n".join(f"- {turn}" for turn in said),
            card_sentence="\n".join(sentences),
        ),
        "key": SITUATION_KEY,
    }


JUDGE_KEYS = (OFFERED_KEY, SITUATION_KEY)

QUESTIONS = {
    OFFERED_KEY: grade_offered_from_the_list,
    SITUATION_KEY: grade_situation_from_user_words,
}


def judge_evaluator(outputs: dict, reference_outputs: dict) -> dict:
    """Both questions at once, in the multi-key shape LangSmith accepts.

    A crisis case is intercepted before any agent turn, so its trajectory is
    empty; both questions skip themselves on their own terms, and no model
    call is made.
    """
    return {"results": [grade(outputs) for grade in QUESTIONS.values()]}


__all__ = [
    "JUDGE_KEYS",
    "JUDGE_MODEL",
    "OFFERED_KEY",
    "QUESTIONS",
    "SITUATION_KEY",
    "grade_offered_from_the_list",
    "grade_situation_from_user_words",
    "judge_evaluator",
]
