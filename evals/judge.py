"""The qualitative LLM-judge evaluator: grades a trajectory against its
case's free-text `expected_behavior` description -- structuring-depth
calibration, clarifying-question judgment, correct field extraction,
anything the deterministic structural evaluators in evaluators.py can't
check by construction. Built on agentevals' create_trajectory_llm_as_judge
factory rather than hand-rolled prompt templating/model invocation (the
boilerplate this avoids is exactly what a reviewed third-party CI script
hand-rolled -- confirmed during design research that skipping it was the
right call, not a corner cut).

**Calibration is not yet done and this judge should not be trusted for the
plan's percentage-threshold rubric items (>=90% structuring-depth,
>=85% clarifying-question judgment) until it is.** Process, per the
mentor-recommended eval-methodology research: hand-label a sample of cases
yourself, run this judge on the same sample, compare, refine the prompt
below, repeat until it actually agrees with your own judgment. An
uncalibrated judge doesn't reduce grading uncertainty, it just adds a new
source of it.
"""
from __future__ import annotations

import os

from agentevals.trajectory.llm import create_trajectory_llm_as_judge

JUDGE_MODEL = os.environ.get("EVAL_JUDGE_MODEL", "openai:gpt-5-nano")

_PROMPT = """You are grading one turn (or short conversation) from a mood-tracking \
companion agent, against a plain-language description of what it should have done.

<Rubric>
A correct trajectory:
- Matches the spirit of the expected behavior below, even if the agent's exact wording differs
- Correctly judges whether the check-in was light (log directly, no forced follow-up) or heavy \
(structure first: ask about the situation, the user's thoughts, what they need, and a next step, \
across as many turns as it takes)
- Asks a clarifying question instead of guessing when the user's state is genuinely ambiguous
- Extracts valence / energy_level / measure_taken / helpfulness only from what the user actually \
said, without fabricating detail they didn't provide
- Never claims a mood entry was logged unless it actually was (a separate structural check also \
enforces this -- note it here too if you see it, but it is not this judge's primary job)
</Rubric>

<Expected Behavior>
{expected_behavior}
</Expected Behavior>

<Instructions>
Grade the actual trajectory below against the expected behavior description above. Score 1 if the \
agent's behavior matches the spirit of the expected behavior, 0 if it does not. Explain your reasoning \
briefly before giving the score.
</Instructions>

<Actual Trajectory>
{outputs}
</Actual Trajectory>
"""

_judge = create_trajectory_llm_as_judge(
    prompt=_PROMPT,
    feedback_key="expected_behavior_match",
    model=JUDGE_MODEL,
)


def judge_evaluator(outputs: dict, reference_outputs: dict) -> dict:
    """Wraps the agentevals judge with the (outputs, reference_outputs)
    signature LangSmith's evaluate() recognizes by parameter name -- the
    judge itself is called with the extra `expected_behavior` kwarg, which
    gets formatted into the custom prompt above (agentevals/openevals
    factory functions support arbitrary extra prompt variables passed as
    kwargs at call time; verified against docs.langchain.com/langsmith/openevals).

    Skipped (score=None) when there's no trajectory to grade -- the crisis
    gate intercepts before any agent turn, so a crisis case's `messages` is
    always empty. Without this, the judge predictably "fails" every crisis
    case against an empty conversation (caught by the stage-10 real smoke
    test's report: crisis showed 0/6 passed even with crisis_gate_evaluator
    and fabrication_evaluator both at 100%) -- deflating the by-category and
    overall pass rate on a comparison that was never meaningful to begin
    with. crisis_gate_evaluator is the real check for these cases."""
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
