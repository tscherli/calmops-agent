"""Tests for scheduler.py's digest opener.

The aggregation itself is deterministic (tools/log_query.py, tested in
test_log_query.py); the only LLM involvement is phrasing one sentence around
numbers that are already true. These tests pin the two things that actually
matter for honesty: a thin week must not be dressed up as a pattern, and a
failed LLM call must not silently cancel the digest.
"""
from typing import Any, Optional

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from pydantic import Field

from scheduler import OPENER_MESSAGE, build_digest_opener, opener_for
from scheduler_config import Fire, SchedulerConfig


class StubChatModel(BaseChatModel):
    reply: str = "Here's your week."
    error: Optional[str] = None
    received_messages: list = Field(default_factory=list)

    def bind_tools(self, tools: Any, **kwargs: Any):
        return self

    def _generate(self, messages, stop=None, run_manager=None, **kwargs) -> ChatResult:
        self.received_messages.append(messages)
        if self.error:
            raise RuntimeError(self.error)
        return ChatResult(generations=[ChatGeneration(message=AIMessage(content=self.reply))])

    @property
    def _llm_type(self) -> str:
        return "stub-chat-model"


def _summary(**overrides) -> dict:
    base = {
        "window_days": 7,
        "entry_count": 4,
        "enough_data": True,
        "top_tags": [("deadline_pressure", 4), ("context_switching", 2)],
        "mood_counts": {"bad": 3, "neutral": 1},
        "energy_counts": {"wired": 3},
        "measures_taken": [],
    }
    base.update(overrides)
    return base


def test_uses_the_model_phrasing_when_there_is_enough_data():
    model = StubChatModel(reply="You logged deadline_pressure 4 times this week.")

    opener = build_digest_opener(_summary(), model=model)

    assert opener == "You logged deadline_pressure 4 times this week."


def test_thin_week_says_not_enough_data_without_calling_the_model():
    """N=1 must not become a trend -- and shouldn't cost an LLM call either."""
    model = StubChatModel()

    opener = build_digest_opener(_summary(entry_count=1, enough_data=False), model=model)

    assert "enough" in opener.lower()
    assert model.received_messages == []


def test_llm_failure_falls_back_to_a_factual_opener():
    """A failed phrasing call must not mean the digest silently doesn't fire."""
    model = StubChatModel(error="simulated outage")

    opener = build_digest_opener(_summary(), model=model)

    assert "deadline_pressure" in opener
    assert "4 times" in opener


def test_empty_model_reply_falls_back_too():
    opener = build_digest_opener(_summary(), model=StubChatModel(reply="   "))

    assert "deadline_pressure" in opener


def test_real_counts_are_what_gets_sent_to_the_model():
    model = StubChatModel()

    build_digest_opener(_summary(), model=model)

    sent = str(model.received_messages[0][1].content)
    assert "deadline_pressure (4x)" in sent
    assert "'bad': 3" in sent


def test_checkin_fire_uses_the_plain_opener():
    fire = Fire(when=None, kind="checkin")

    assert opener_for(fire, SchedulerConfig()) == OPENER_MESSAGE


def test_digest_fire_builds_a_digest_opener(monkeypatch):
    monkeypatch.setattr(
        "scheduler.analyze_stress_patterns",
        lambda window_days: _summary(enough_data=False, entry_count=0),
    )

    opener = opener_for(Fire(when=None, kind="digest"), SchedulerConfig())

    assert opener != OPENER_MESSAGE
    assert "enough" in opener.lower()
