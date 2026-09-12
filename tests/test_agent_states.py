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

from agent.graph import TurnDecision, build_agent, message_text
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


def _run(graph, text, step=""):
    return graph.invoke({"messages": [HumanMessage(content=text)], "current_step": step})


# -- triage ------------------------------------------------------------------


def test_small_talk_replies_and_ends_the_conversation(tmp_path):
    """A greeting must not become a check-in. current_step is cleared, so the
    next message is triaged fresh rather than resuming anything."""
    model = DecidingModel(decisions=[TurnDecision(next_step="small_talk", reply="Good thanks!")])
    result = _run(_graph(model, tmp_path), "hey, how's it going?")

    assert message_text(result["messages"][-1]) == "Good thanks!"
    assert result["current_step"] == ""
    assert model.call_count == 1, "small talk should cost exactly one model call"


def test_heavy_signal_asks_one_question_and_waits(tmp_path):
    model = DecidingModel(
        decisions=[TurnDecision(next_step="structuring", reply="What's been going on?")]
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
        decisions=[TurnDecision(next_step="structuring", reply="And how did that land?")]
    )
    result = _run(_graph(model, tmp_path), "three deadlines collided", step="structuring")

    assert result["current_step"] == "structuring"
    assert model.call_count == 1


def test_structuring_ready_hands_off_to_the_scale_question(tmp_path):
    model = DecidingModel(decisions=[TurnDecision(next_step="ready", reply="That sounds rough.")])
    result = _run(_graph(model, tmp_path), "I'll pick one and commit", step="structuring")

    assert result["current_step"] == "awaiting_scale"
    assert SCALE_QUESTION in message_text(result["messages"][-1])


def test_structuring_cannot_log_no_matter_what_it_returns(tmp_path):
    """No edge runs from structuring to classify, so logging is unreachable
    from here even if the model tries."""
    model = DecidingModel(
        decisions=[TurnDecision(next_step="ready", reply="ok"), AIMessage(content="ok")]
    )
    _run(_graph(model, tmp_path), "anything", step="structuring")

    assert not (tmp_path / "mood_log.csv").exists()


# -- awaiting_scale ----------------------------------------------------------


def test_scale_question_costs_no_model_call(tmp_path):
    """E1. The question is a constant; generating it would cost a call and
    let the scale words drift."""
    model = DecidingModel(decisions=[TurnDecision(next_step="ready", reply="Got it.")])
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
