"""Tests for scheduler.py's digest opener.

The aggregation itself is deterministic (tools/log_query.py, tested in
test_log_query.py); the only LLM involvement is phrasing one sentence around
numbers that are already true. These tests pin the two things that actually
matter for honesty: a thin week must not be dressed up as a pattern, and a
failed LLM call must not silently cancel the digest.
"""
from typing import Any

import pytest
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from pydantic import Field

from scheduler import OPENER_MESSAGE, build_digest_opener, opener_for
from scheduler_config import Fire, SchedulerConfig


class StubChatModel(BaseChatModel):
    reply: str = "Here's your week."
    error: str | None = None
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
        "rating_count": 2,
        "enough_data": True,
        "energy_counts": {"wired": 3},
        "top_measures": [("Walk for five minutes.", 4), ("Say no to one request today.", 2)],
        "helpfulness_counts": {"Walk for five minutes.": {"helped": 2}},
        "best_measures": ["Walk for five minutes."],
    }
    base.update(overrides)
    return base


def test_uses_the_model_phrasing_when_there_is_enough_data():
    model = StubChatModel(reply="A five-minute walk came up four times this week.")

    opener = build_digest_opener(_summary(), model=model)

    assert opener == "A five-minute walk came up four times this week."


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

    assert "Walk for five minutes." in opener
    assert "helped" in opener


def test_the_fallback_reports_a_count_when_nothing_has_been_rated():
    """With no verdicts in, the honest thing to say is what they did, not
    what worked."""
    summary = _summary(best_measures=[], helpfulness_counts={}, rating_count=0)

    opener = build_digest_opener(summary, model=StubChatModel(error="outage"))

    assert "Walk for five minutes." in opener
    assert "4 times" in opener
    assert "helped" not in opener


def test_empty_model_reply_falls_back_too():
    opener = build_digest_opener(_summary(), model=StubChatModel(reply="   "))

    assert "Walk for five minutes." in opener


def test_real_counts_are_what_gets_sent_to_the_model():
    model = StubChatModel()

    build_digest_opener(_summary(), model=model)

    sent = str(model.received_messages[0][1].content)
    assert "Walk for five minutes. (4x)" in sent
    assert "'wired': 3" in sent
    assert "check-ins in the last 7 days: 4" in sent


def test_what_worked_is_sent_separately_from_what_was_tried():
    """The one thing this opener must not do is congratulate someone on a
    measure they rated as no help. Leaving the model to infer that from the
    counts is how that happens, so the summary states it."""
    model = StubChatModel()
    summary = _summary(
        top_measures=[("Take a day off.", 3), ("Walk for five minutes.", 1)],
        helpfulness_counts={"Take a day off.": {"didnt_help": 3}},
        best_measures=[],
    )

    build_digest_opener(summary, model=model)

    sent = str(model.received_messages[0][1].content)
    assert "what they tried: Take a day off. (3x)" in sent
    assert "what they rated as helping: nothing rated yet" in sent


def test_checkin_fire_uses_the_plain_opener():
    fire = Fire(when=None, kind="checkin")

    opener, events = opener_for(fire, SchedulerConfig())

    assert opener == OPENER_MESSAGE
    # An ordinary check-in runs no tool, so it has nothing to report.
    assert events == []


def test_digest_fire_builds_a_digest_opener(monkeypatch):
    monkeypatch.setattr(
        "scheduler.analyze_stress_patterns",
        lambda window_days, **_: _summary(enough_data=False, entry_count=0),
    )

    opener, _events = opener_for(Fire(when=None, kind="digest"), SchedulerConfig())

    assert opener != OPENER_MESSAGE
    assert "enough" in opener.lower()


def test_digest_reports_the_analysis_that_produced_it(monkeypatch):
    """The digest is the one window where the agent asserts numbers about the
    user's history. Those numbers come from an aggregation the model did not
    compute, so the aggregation reports itself."""
    monkeypatch.setattr(
        "scheduler.analyze_stress_patterns",
        lambda window_days, **_: _summary(enough_data=True, entry_count=6),
    )

    _opener, events = opener_for(Fire(when=None, kind="digest"), SchedulerConfig())

    assert [e["tool"] for e in events] == ["analyze_stress_patterns"]
    assert events[0]["exit_code"] == 0
    assert "6 entries" in events[0]["detail"]


def test_digest_says_so_when_there_was_not_enough_data(monkeypatch):
    """'Not enough data' must be visible as a fact about the aggregation, not
    only as a sentence the model chose to write."""
    monkeypatch.setattr(
        "scheduler.analyze_stress_patterns",
        lambda window_days, **_: _summary(enough_data=False, entry_count=1),
    )

    _opener, events = opener_for(Fire(when=None, kind="digest"), SchedulerConfig())

    assert "not enough data" in events[0]["detail"]


def test_the_digest_reads_the_folder_it_is_given(monkeypatch, tmp_path):
    """A --data-dir run writes its check-ins there; a digest of the default log
    would report a different week."""
    from agent.data_paths import data_paths

    seen = {}
    monkeypatch.setattr(
        "scheduler.analyze_stress_patterns",
        lambda window_days, **kwargs: seen.update(kwargs) or _summary(enough_data=False, entry_count=0),
    )

    opener_for(Fire(when=None, kind="digest"), SchedulerConfig(), data_paths(tmp_path))

    assert seen["path"] == tmp_path.resolve() / "checkin_log.csv"
    assert seen["measures_path"] == tmp_path.resolve() / "measures.csv"


def test_the_scheduler_opens_its_windows_on_the_data_dir(monkeypatch, tmp_path):
    import scheduler
    import tools.checkin_log as checkin_log

    monkeypatch.setattr(checkin_log, "PENDING_LOG_PATH", checkin_log.PENDING_LOG_PATH)
    opened = {}

    def open_once(message, events, paths=None):
        opened["paths"] = paths
        raise SystemExit

    monkeypatch.setattr("sys.argv", ["scheduler.py", "--data-dir", str(tmp_path / "data")])
    monkeypatch.setattr(scheduler.time_module, "sleep", lambda _seconds: None)
    monkeypatch.setattr(scheduler, "opener_for", lambda fire, config, paths=None: ("hi", []))
    monkeypatch.setattr(scheduler, "open_window", open_once)

    with pytest.raises(SystemExit):
        scheduler.main()

    assert opened["paths"].checkin_log == (tmp_path / "data").resolve() / "checkin_log.csv"
