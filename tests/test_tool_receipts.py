"""Tests for the machine-set tool receipts (`tool_events`).

What these are for: the user is told whether an entry was saved by *code
that watched it happen*, not by the agent's prose. The agent cannot emit a
receipt and cannot suppress one, so its claims can be checked against them.

The tests are written from that guarantee rather than from the
implementation -- the question each one asks is "could the user be misled
about what happened to their data?", not "does this function return what it
currently returns".
"""
import time
import tkinter as tk

import pytest
from langchain_core.messages import AIMessage, HumanMessage

import app as app_module
from agent.graph import build_agent
from app import ChatApp
from evals.evaluators import fabrication_evaluator
from tests.test_agent_pipeline import StubChatModel, _model_that_logs_then_replies, _stub_classifier


def _events(result) -> list[dict]:
    return result.get("tool_events") or []


def _by_tool(result, name: str) -> dict | None:
    return next((e for e in _events(result) if e.get("tool") == name), None)


# -- the receipt reflects what actually happened -----------------------------


def test_successful_write_reports_exit_0_and_the_real_row_id(tmp_path):
    log_path = tmp_path / "mood_log.csv"
    graph = build_agent(
        model=_model_that_logs_then_replies({"mood": "bad", "circumstances": "deadlines"}),
        classifier=_stub_classifier(["deadline_pressure"]),
        log_path=log_path,
    )
    result = graph.invoke(
        {"current_step": "light", "messages": [HumanMessage(content="rough day")]}
    )

    event = _by_tool(result, "log_mood")
    assert event is not None, "a write must always produce a receipt"
    assert event["exit_code"] == 0
    # The id in the receipt must be the row actually on disk, not a
    # placeholder -- otherwise the receipt is decoration, not evidence.
    assert log_path.read_text(encoding="utf-8").count(event["detail"].split()[1]) == 1


def test_failed_write_reports_exit_1_not_exit_0(tmp_path):
    """The case that matters most: a receipt that says 0 when nothing was
    written would be worse than no receipt at all."""
    blocker = tmp_path / "not-a-directory"
    blocker.write_text("blocks the write below", encoding="utf-8")

    graph = build_agent(
        model=_model_that_logs_then_replies({"mood": "bad", "circumstances": "x"}),
        classifier=_stub_classifier(["y"]),
        log_path=blocker / "mood_log.csv",
    )
    result = graph.invoke({"current_step": "light", "messages": [HumanMessage(content="hi")]})

    event = _by_tool(result, "log_mood")
    assert event["exit_code"] == 1
    assert event["status"] == "error"


def test_skipped_classification_has_no_exit_code(tmp_path):
    """A tool that never ran has no exit code. Reporting 'skipped' as exit 0
    would quietly claim work that didn't happen."""

    def skipping_classifier(**kwargs):
        return {"tags": [], "status": "skipped"}

    graph = build_agent(
        model=_model_that_logs_then_replies({"mood": "neutral", "circumstances": ""}),
        classifier=skipping_classifier,
        log_path=tmp_path / "mood_log.csv",
    )
    result = graph.invoke({"current_step": "light", "messages": [HumanMessage(content="hi")]})

    event = _by_tool(result, "classify_entry")
    assert event["exit_code"] is None
    assert event["status"] == "skipped"


def test_classification_failure_is_visible_rather_than_silent(tmp_path):
    """classify_entry swallows its own exceptions so a failed tag can't block
    the write. That is right, but it used to mean the user was never told the
    entry went in untagged."""

    def failing_classifier(**kwargs):
        return {"tags": [], "status": "error", "error": "boom"}

    graph = build_agent(
        model=_model_that_logs_then_replies({"mood": "bad", "circumstances": "x"}),
        classifier=failing_classifier,
        log_path=tmp_path / "mood_log.csv",
    )
    result = graph.invoke({"current_step": "light", "messages": [HumanMessage(content="hi")]})

    event = _by_tool(result, "classify_entry")
    assert event["exit_code"] == 1
    assert "untagged" in event["detail"]
    # ...and the row is still written, which is the behaviour being preserved.
    assert _by_tool(result, "log_mood")["exit_code"] == 0


def test_empty_history_is_reported_as_no_matches(tmp_path):
    """The honesty case: if this says 'no matches' and the reply describes a
    past entry, the contradiction is visible without parsing prose."""
    graph = build_agent(
        model=_model_that_logs_then_replies({"mood": "bad", "circumstances": "x"}),
        classifier=_stub_classifier(["tag_with_no_precedent"]),
        log_path=tmp_path / "mood_log.csv",
    )
    result = graph.invoke({"current_step": "light", "messages": [HumanMessage(content="hi")]})

    assert _by_tool(result, "find_similar_entries")["detail"] == "no matches"


def test_receipts_are_not_produced_by_the_model(tmp_path):
    """The guarantee that makes the rest worth anything: a model that says it
    logged something, without calling the tool, produces no receipt."""
    graph = build_agent(
        model=StubChatModel(responses=[AIMessage(content="I've logged that for you!")]),
        classifier=_stub_classifier(["x"]),
        log_path=tmp_path / "mood_log.csv",
    )
    result = graph.invoke({"current_step": "light", "messages": [HumanMessage(content="hi")]})

    assert _by_tool(result, "log_mood") is None


# -- the evaluator grades against the receipt, not the prose -----------------


def test_successful_search_does_not_excuse_a_false_log_claim():
    """Regression test for a real hole found in run 2's analysis: the check
    asked whether ANY tool succeeded, so a successful history lookup would
    excuse a fabricated "I logged that". Latent only because no case had yet
    reached post_log with a successful search."""
    outputs = {
        "messages": [
            {"role": "human", "content": "rough day"},
            {"role": "tool", "name": "find_similar_entries", "content": "[]", "status": "success"},
            {"role": "ai", "content": "I've logged that for you.", "tool_calls": None},
        ],
        "tool_events": [
            {"tool": "find_similar_entries", "exit_code": 0, "status": "ok", "detail": "no matches"}
        ],
    }

    assert fabrication_evaluator(outputs, {})["score"] is False


def test_a_real_write_still_passes_the_fabrication_check():
    outputs = {
        "messages": [
            {"role": "human", "content": "rough day"},
            {"role": "ai", "content": "Noted that down for you.", "tool_calls": None},
        ],
        "tool_events": [
            {"tool": "log_mood", "exit_code": 0, "status": "ok", "detail": "row 7 written"}
        ],
    }

    assert fabrication_evaluator(outputs, {})["score"] is True


def test_failed_write_is_not_treated_as_a_successful_log():
    outputs = {
        "messages": [
            {"role": "human", "content": "rough day"},
            {"role": "ai", "content": "Saved that for you.", "tool_calls": None},
        ],
        "tool_events": [
            {"tool": "log_mood", "exit_code": 1, "status": "error", "detail": "write failed"}
        ],
    }

    assert fabrication_evaluator(outputs, {})["score"] is False


# -- the UI shows the receipt, and shows it before the agent speaks ----------


class _StreamingStub:
    """Emits a tool event, then the final state -- the shape app.py consumes."""

    def __init__(self, events, reply="All set."):
        self.events = events
        self.reply = reply

    def stream(self, payload, stream_mode=None):
        for event in self.events:
            yield ("updates", {"log_write": {"tool_events": [event]}})
        yield (
            "values",
            {
                "messages": payload["messages"] + [AIMessage(content=self.reply)],
                "tool_events": list(self.events),
            },
        )




@pytest.fixture(autouse=True)
def isolated_consent_path(tmp_path, monkeypatch):
    monkeypatch.setattr(app_module, "CONSENT_PATH", tmp_path / "consent_ack.json")


def _settle(root, app, timeout=5.0):
    deadline = time.time() + timeout
    while time.time() < deadline:
        root.update()
        if str(app.send_button.cget("state")) == "normal":
            return
        time.sleep(0.01)
    raise AssertionError("turn did not finish within the timeout")


def test_receipt_is_shown_before_the_agent_reply(tk_root):
    """The reason the turn was moved onto a worker thread at all: the
    evidence has to arrive ahead of the claim, not after it."""
    graph = _StreamingStub(
        [{"tool": "log_mood", "exit_code": 0, "status": "ok", "detail": "row 7 written"}],
        reply="That sounds like a hard one.",
    )
    app = ChatApp(tk_root, skip_consent=True, agent_graph=graph)

    app.entry.insert(0, "rough day")
    app._on_send()
    _settle(tk_root, app)

    history = app.history.get("1.0", "end")
    assert "log_mood" in history and "exit 0" in history
    assert history.index("log_mood") < history.index("That sounds like a hard one.")


def test_a_turn_that_logs_nothing_says_so_explicitly(tk_root):
    """Silence is ambiguous -- nothing logged, or nothing rendered? The
    negative receipt removes the ambiguity."""
    app = ChatApp(tk_root, skip_consent=True, agent_graph=_StreamingStub([], reply="How's today?"))

    app.entry.insert(0, "hello")
    app._on_send()
    _settle(tk_root, app)

    assert "not called this turn" in app.history.get("1.0", "end")


def test_input_is_locked_while_a_turn_runs(tk_root):
    """The frozen UI used to prevent a second turn being started mid-flight.
    Now that it doesn't freeze, two concurrent turns would interleave writes
    to self.messages."""
    app = ChatApp(tk_root, skip_consent=True, agent_graph=_StreamingStub([]))

    app.entry.insert(0, "hello")
    app._on_send()
    assert str(app.send_button.cget("state")) == "disabled"

    _settle(tk_root, app)
    assert str(app.send_button.cget("state")) == "normal"
