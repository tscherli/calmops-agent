"""Tests for the conversation state machine in agent/graph.py.

The redesign's safety properties are all of the form "state X cannot do Y".
That is checkable with no LLM involved -- an unreachable edge and an empty
tool list are both plain assertions -- which is most of why the state machine
was preferred over prose rules in the prompt.

tests/test_agent_pipeline.py covers the deterministic log pipeline and the
per-state tool surface; this file covers routing and transitions.
"""
from typing import Any, Optional

import pytest
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from pydantic import Field

from agent.graph import CollectedFields, TurnDecision, build_agent, message_text
from agent.prompts import SCALE_QUESTION


class DecidingModel(BaseChatModel):
    """Returns scripted TurnDecision objects for structured-output states and
    scripted AIMessages for tool-calling ones."""

    decisions: list[Any] = Field(default_factory=list)
    call_count: int = 0
    states_seen: list[str] = Field(default_factory=list)

    def bind_tools(self, tools: Any, *, tool_choice: Optional[str] = None, **kwargs: Any):
        return self

    def bind(self, **kwargs: Any):
        return self

    def with_structured_output(self, schema: Any, **kwargs: Any):
        return self

    def invoke(self, input: Any, config: Any = None, **kwargs: Any):
        item = self.decisions[min(self.call_count, len(self.decisions) - 1)]
        self.call_count += 1
        return item

    def _generate(self, messages, stop=None, run_manager=None, **kwargs) -> ChatResult:
        return ChatResult(generations=[ChatGeneration(message=AIMessage(content="unused"))])

    @property
    def _llm_type(self) -> str:
        return "deciding-stub"


def _graph(model, tmp_path):
    return build_agent(
        model=model,
        classifier=lambda **kw: {"tags": [], "status": "skipped"},
        log_path=tmp_path / "mood_log.csv",
    )


def _run(graph, text, step="", collected=None):
    return graph.invoke(
        {
            "messages": [HumanMessage(content=text)],
            "current_step": step,
            "collected": dict(collected or {}),
        }
    )


# -- triage ------------------------------------------------------------------


def test_small_talk_replies_and_ends_the_conversation(tmp_path):
    """A greeting must not become a check-in. current_step is cleared, so the
    next message is triaged fresh rather than resuming anything."""
    model = DecidingModel(decisions=[TurnDecision(route="small_talk", reply="Good thanks!")])
    result = _run(_graph(model, tmp_path), "hey, how's it going?")

    assert message_text(result["messages"][-1]) == "Good thanks!"
    assert result["current_step"] == ""
    assert model.call_count == 1, "small talk should cost exactly one model call"


def test_heavy_signal_asks_one_question_and_waits(tmp_path):
    model = DecidingModel(
        decisions=[TurnDecision(route="structuring", reply="What's been going on?")]
    )
    result = _run(_graph(model, tmp_path), "everything is too much right now")

    assert message_text(result["messages"][-1]) == "What's been going on?"
    assert result["current_step"] == "structuring"
    assert model.call_count == 1


def test_unparseable_triage_falls_back_to_small_talk_not_logging(tmp_path):
    """The conservative fallback matters: defaulting toward logging would
    write an entry from a turn the model failed to understand."""
    model = DecidingModel(decisions=[AIMessage(content="...")])
    result = _run(_graph(model, tmp_path), "hello")

    assert result["current_step"] == ""
    assert not list((tmp_path / "mood_log.csv").parent.glob("mood_log.csv"))


# -- structuring -------------------------------------------------------------


def test_structuring_resumes_without_re_triaging(tmp_path):
    """The saving that makes the state machine affordable: a continuing turn
    skips triage entirely, so it costs one call, not two."""
    model = DecidingModel(
        decisions=[TurnDecision(route="structuring", reply="And how did that land?")]
    )
    result = _run(_graph(model, tmp_path), "three deadlines collided", step="structuring")

    assert result["current_step"] == "structuring"
    assert model.call_count == 1


def test_structuring_keeps_asking_until_the_gate_fields_are_present(tmp_path):
    """The model saying "ready" is not enough, and that is the point: it said
    the equivalent zero times in 37 eval cases while its prompt told it three
    ways to be thorough. The gate is now checked in Python."""
    model = DecidingModel(
        decisions=[
            TurnDecision(
                route="ready",
                reply="What's been the hardest part?",
                collected=CollectedFields(circumstances="behind on everything"),
            )
        ]
    )
    result = _run(_graph(model, tmp_path), "I'm swamped", step="structuring")

    # situation and measure_taken are still missing, so it must not move on
    # however confidently the model claimed otherwise.
    assert result["current_step"] == "structuring"


def test_structuring_hands_off_to_light_once_the_gate_is_satisfied(tmp_path):
    """And it hands off to `light`, not to the scale question -- `light`
    already knows how to tell conveyed values from missing ones, so that
    judgement lives in one place instead of two."""
    model = DecidingModel(
        decisions=[
            TurnDecision(
                route="ready",
                reply="That sounds rough.",
                collected=CollectedFields(
                    circumstances="three deadlines collided",
                    situation="all due the same afternoon",
                    measure_taken="nothing yet",
                ),
            ),
            # `light` runs in the same turn and finds no conveyed values.
            AIMessage(content="Got it."),
        ]
    )
    result = _run(_graph(model, tmp_path), "I'll pick one and commit", step="structuring")

    assert result["current_step"] == "awaiting_scale"
    assert SCALE_QUESTION in message_text(result["messages"][-1])


def test_a_field_established_early_is_not_lost_when_a_later_turn_omits_it(tmp_path):
    """Turn 4's report restating only what came up in turn 4 must not erase
    what turn 2 established."""
    model = DecidingModel(
        decisions=[
            TurnDecision(
                route="structuring",
                reply="And what have you tried?",
                collected=CollectedFields(situation="all due the same afternoon"),
            )
        ]
    )
    result = _run(
        _graph(model, tmp_path),
        "they're all due at once",
        step="structuring",
        collected={"circumstances": "behind on everything"},
    )

    assert result["collected"]["circumstances"] == "behind on everything"
    assert result["collected"]["situation"] == "all due the same afternoon"


def test_structuring_cannot_log_no_matter_what_it_returns(tmp_path):
    """No edge runs from structuring to classify, so logging is unreachable
    from here even if the model tries."""
    model = DecidingModel(
        decisions=[TurnDecision(route="ready", reply="ok"), AIMessage(content="ok")]
    )
    _run(_graph(model, tmp_path), "anything", step="structuring")

    assert not (tmp_path / "mood_log.csv").exists()


# -- awaiting_scale ----------------------------------------------------------


def test_scale_question_costs_no_model_call(tmp_path):
    """E1. The question is a constant; generating it would cost a call and
    let the scale words drift."""
    model = DecidingModel(decisions=[TurnDecision(route="ready", reply="Got it.")])
    _run(_graph(model, tmp_path), "done", step="structuring")

    assert model.call_count == 1, "only the structuring turn should call the model"


def test_scale_question_lists_the_exact_vocabulary():
    for value in ("bad", "neutral", "good", "exhausted", "calm", "wired"):
        assert value in SCALE_QUESTION


def test_supplied_scale_values_win_over_anything_the_model_says(tmp_path):
    """The user's own reading is authoritative -- that is the entire point of
    asking rather than inferring."""
    import csv

    log_call = AIMessage(
        content="",
        tool_calls=[
            {
                "name": "log_mood",
                "args": {"mood": "good", "energy": "calm", "circumstances": "crunch"},
                "id": "c1",
                "type": "tool_call",
            }
        ],
    )
    model = DecidingModel(decisions=[log_call, AIMessage(content="Logged.")])
    graph = _graph(model, tmp_path)
    graph.invoke(
        {
            "messages": [HumanMessage(content="mood: bad, energy: wired")],
            "current_step": "awaiting_scale",
            "scale_values": {"mood": "bad", "energy": "wired"},
        }
    )

    with (tmp_path / "mood_log.csv").open(newline="", encoding="utf-8") as f:
        row = list(csv.DictReader(f))[0]
    assert row["mood"] == "bad"
    assert row["energy"] == "wired"


# -- light -------------------------------------------------------------------


def test_light_defers_to_the_scale_question_when_values_are_missing(tmp_path):
    """Never guess. A light check-in with nothing stated asks rather than
    inventing an energy value."""
    model = DecidingModel(decisions=[AIMessage(content="Nice one.")])
    result = _run(_graph(model, tmp_path), "quiet morning", step="light")

    assert result["current_step"] == "awaiting_scale"
    assert not (tmp_path / "mood_log.csv").exists()


def test_a_placeholder_does_not_open_the_gate(tmp_path):
    """Observed in the first live run of the heavy path: the model wrote
    `measure_taken: "not provided yet"` and walked straight through a gate
    that only checked non-emptiness -- opening it without ever asking, and
    putting a stand-in into the user's log as though it were something they
    said."""
    model = DecidingModel(
        decisions=[
            TurnDecision(
                route="ready",
                reply="Let's log it.",
                collected=CollectedFields(
                    circumstances="behind on everything",
                    situation="three projects at once",
                    measure_taken="not provided yet",
                ),
            )
        ]
    )
    result = _run(_graph(model, tmp_path), "swamped", step="structuring")

    assert result["current_step"] == "structuring"


def test_nothing_yet_is_a_real_answer_and_does_open_the_gate(tmp_path):
    """The counter-check that keeps the placeholder rule from over-reaching:
    a user who says they have tried nothing has answered the question."""
    model = DecidingModel(
        decisions=[
            TurnDecision(
                route="ready",
                reply="Let's log it.",
                collected=CollectedFields(
                    circumstances="behind on everything",
                    situation="three projects at once",
                    measure_taken="nothing yet",
                ),
            ),
            AIMessage(content="Got it."),
        ]
    )
    result = _run(_graph(model, tmp_path), "swamped", step="structuring")

    assert result["current_step"] == "awaiting_scale"
