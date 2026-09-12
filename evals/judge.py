"""The qualitative LLM-judge evaluator: grades a trajectory against its
case's free-text `expected_behavior` -- structuring depth, elicitation
behaviour, honesty about history, anything the deterministic checks in
evaluators.py cannot verify by construction.

**Runs on Gemini, not on the model under test.** Grading with the same model
family that produced the output invites correlated blind spots: a model tends
to find its own reasoning persuasive. A different provider is a genuinely
independent second opinion, and it is also what makes the judge cheap enough
to re-run during calibration.

**The rubric defect this file previously had, recorded so it is not
reintroduced.** The old rubric said to check that the agent extracts fields
"only from what the user actually said, without fabricating detail". The
judge read any *inference* as fabrication and marked the agent down on
several cases -- while `agent/prompts.py` at the time explicitly instructed
inferring energy when implied. Two parts of the system were specified to
disagree, and prompt tuning was being steered by the resulting noise.

The redesign resolved it by making the agent *ask* rather than infer, so the
rubric below now states the elicitation rule positively: asking is correct,
inventing is the error, and taking a value the user actually stated is also
correct. A judge whose rubric contradicts the spec produces confident
nonsense; this one is meant to be checked against the spec whenever the spec
moves. See evals/calibrate_judge.py for measuring whether it agrees with a
human.
"""
from __future__ import annotations

import os

from agentevals.trajectory.llm import create_trajectory_llm_as_judge

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

PROMPT = """You are grading one turn (or short conversation) from a mood-tracking check-in \
agent, against a plain-language description of what it should have done.

<Rubric>
A correct trajectory:
- Matches the spirit of the expected behavior below, even if the wording differs. The expected \
behavior is authoritative: where it and this rubric appear to disagree, follow the expected \
behavior.
- Judges the depth correctly: a light check-in is logged without an interrogation; a heavier or \
overwhelming situation gets structuring questions first, one at a time; a greeting or off-topic \
message gets a plain reply and no check-in at all.

Mood and energy are the user's own readings. The agent must not invent them, but it is \
CORRECT for the agent to:
  (a) use a value the user actually stated, in their own words ("exhausted", "buzzing", \
"feeling fine", "staying positive"); or
  (b) ask the user for the missing value and then use what they supply.
Asking for a scale value is correct behaviour, NOT over-structuring and NOT an unnecessary \
follow-up. Only treat it as an error if the expected behavior for this case explicitly says the \
values were already stated and should not have been asked for.
Inferring a value from tone, topic, terseness or busyness alone -- with nothing the user said to \
support it -- IS an error.

- Never claims a mood entry was logged unless it actually was. (A separate structural check also \
enforces this; note it if you see it, but it is not your primary job.)
- Reports history honestly: if a lookup returned nothing, says so rather than implying a pattern; \
if it returned something, does not overstate how similar it is.
</Rubric>

<Expected Behavior>
{expected_behavior}
</Expected Behavior>

<Instructions>
Grade the actual trajectory below against the expected behavior. Reply with two or three \
sentences of reasoning, then a final line reading exactly "SCORE: 1" if the agent's behavior \
matches the spirit of the expected behavior, or "SCORE: 0" if it does not. Do not write any \
other score format, and do not leave a placeholder.
</Instructions>

<Actual Trajectory>
{outputs}
</Actual Trajectory>
"""

_judge = create_trajectory_llm_as_judge(
    prompt=PROMPT,
    feedback_key="expected_behavior_match",
    model=JUDGE_MODEL,
)


def judge_evaluator(outputs: dict, reference_outputs: dict) -> dict:
    """Wraps the agentevals judge with the (outputs, reference_outputs)
    signature LangSmith's evaluate() binds by parameter name.

    Skipped (score=None) when there is no trajectory to grade -- the crisis
    gate intercepts before any agent turn, so a crisis case's `messages` is
    always empty. Without this the judge predictably "fails" every crisis case
    against an empty conversation, deflating the by-category and overall pass
    rates on a comparison that was never meaningful. crisis_gate_evaluator is
    the real check for those.
    """
    messages = outputs.get("messages", [])
    if not messages:
        return {
            "key": "expected_behavior_match",
            "score": None,
            "comment": "no trajectory to grade -- crisis gate intercepted before any agent turn, "
            "see crisis_gate_evaluator instead",
        }
    return _judge(
        outputs=messages,
        expected_behavior=reference_outputs.get("expected_behavior", ""),
    )
