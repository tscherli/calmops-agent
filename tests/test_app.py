"""Tests for app.py's ChatApp: crisis-gate wiring (must intercept before
the agent ever runs), graceful exception handling (no unhandled exception
may reach the Tkinter UI), and consent-screen gating. Quick-select button
wiring is tests/test_quick_select.py's job, not this file's."""
import json
import time
import tkinter as tk

import pytest
from langchain_core.messages import AIMessage

import app as app_module
from app import ChatApp
from observability.tracing import RunRecorder


def settle(root, app, timeout=5.0):
    """Pump the Tk event loop until the in-flight turn finishes.

    app.py runs the graph on a worker thread so tool receipts can be drawn
    while the turn is still going, which means _on_send() now returns before
    the reply exists. Tests drive the same event loop the user's window does
    rather than reaching into the threading, so what they exercise stays the
    real path.
    """
    deadline = time.time() + timeout
    while time.time() < deadline:
        root.update()
        if str(app.send_button.cget("state")) == "normal":
            return
        time.sleep(0.01)
    raise AssertionError("turn did not finish within the timeout")


class _PoisonGraph:
    """Fails loudly if invoked -- proves a code path never reaches the agent."""

    def invoke(self, *args, **kwargs):
        raise AssertionError("This code path must never invoke the agent graph")

    def stream(self, *args, **kwargs):
        raise AssertionError("This code path must never invoke the agent graph")


class _StubGraph:
    def __init__(self, reply_content="Stub reply", tool_events=()):
        self.reply_content = reply_content
        self.invoked_with = None
        self.tool_events = list(tool_events)

    def _final(self, payload):
        self.invoked_with = payload
        reply = AIMessage(
            content=self.reply_content,
            usage_metadata={"input_tokens": 5, "output_tokens": 3, "total_tokens": 8},
        )
        return {
            "messages": payload["messages"] + [reply],
            "tool_events": list(self.tool_events),
        }

    def invoke(self, payload):
        return self._final(payload)

    def stream(self, payload, stream_mode=None):
        """Mirror the real graph's ("updates", {node: update}) / ("values",
        state) tuple stream, so the app's consumption of it is under test."""
        final = self._final(payload)
        for event in self.tool_events:
            yield ("updates", {"some_node": {"tool_events": [event]}})
        yield ("values", final)


class _RaisingGraph:
    def invoke(self, payload):
        raise RuntimeError("simulated agent failure")

    def stream(self, payload, stream_mode=None):
        raise RuntimeError("simulated agent failure")
        yield  # pragma: no cover -- makes this a generator




@pytest.fixture(autouse=True)
def isolated_consent_path(tmp_path, monkeypatch):
    consent_path = tmp_path / "consent_ack.json"
    monkeypatch.setattr(app_module, "CONSENT_PATH", consent_path)
    return consent_path


def test_crisis_message_never_reaches_agent(tk_root):
    app = ChatApp(tk_root, skip_consent=True, agent_graph=_PoisonGraph())

    app.entry.insert(0, "I want to kill myself")
    app._on_send()  # would raise via _PoisonGraph if the crisis gate didn't intercept it

    history_text = app.history.get("1.0", "end")
    assert "Telefonseelsorge" in history_text  # part of CRISIS_RESOURCES


def test_free_text_invokes_agent_and_displays_reply(tk_root, tmp_path):
    stub_graph = _StubGraph(reply_content="Sounds like a good day.")
    recorder = RunRecorder(log_path=tmp_path / "runs.jsonl")
    app = ChatApp(tk_root, skip_consent=True, agent_graph=stub_graph, recorder=recorder)

    app.entry.insert(0, "feeling good today")
    app._on_send()
    settle(tk_root, app)

    history_text = app.history.get("1.0", "end")
    assert "feeling good today" in history_text
    assert "Sounds like a good day." in history_text
    assert stub_graph.invoked_with is not None

    lines = (tmp_path / "runs.jsonl").read_text(encoding="utf-8").splitlines()
    assert len(lines) == 1


def test_agent_exception_does_not_crash_app(tk_root):
    app = ChatApp(tk_root, skip_consent=True, agent_graph=_RaisingGraph())

    app.entry.insert(0, "hello")
    app._on_send()  # must not raise
    settle(tk_root, app)

    history_text = app.history.get("1.0", "end")
    assert "went wrong" in history_text.lower()


def test_consent_helpers(isolated_consent_path):
    assert app_module.has_consented() is False
    app_module.record_consent()
    assert app_module.has_consented() is True
    assert json.loads(isolated_consent_path.read_text(encoding="utf-8"))["acknowledged"] is True


def test_consent_screen_shown_when_not_consented(tk_root):
    app = ChatApp(tk_root, agent_graph=_PoisonGraph())  # no skip_consent, no prior consent

    assert hasattr(app, "consent_frame")
    assert not hasattr(app, "history")


def test_skip_consent_flag_bypasses_consent_screen(tk_root):
    app = ChatApp(tk_root, skip_consent=True, agent_graph=_PoisonGraph())

    assert hasattr(app, "history")
    assert not hasattr(app, "consent_frame")


def test_prior_consent_bypasses_consent_screen(tk_root, isolated_consent_path):
    app_module.record_consent()

    app = ChatApp(tk_root, agent_graph=_PoisonGraph())  # no skip_consent, but already consented

    assert hasattr(app, "history")
    assert not hasattr(app, "consent_frame")


def test_opener_message_shown_on_startup(tk_root):
    app = ChatApp(tk_root, skip_consent=True, agent_graph=_PoisonGraph(), opener_message="Hey, how are you doing?")

    history_text = app.history.get("1.0", "end")
    assert "Hey, how are you doing?" in history_text
