"""Tests for app.py's ChatApp: crisis-gate wiring (must intercept before
the agent ever runs), graceful exception handling (no unhandled exception
may reach the Tkinter UI), and consent-screen gating. Quick-select button
wiring is tests/test_quick_select.py's job, not this file's."""
import json
import tkinter as tk

import pytest
from langchain_core.messages import AIMessage

import app as app_module
from app import ChatApp
from observability.tracing import RunRecorder


class _PoisonGraph:
    """Fails loudly if invoked -- proves a code path never reaches the agent."""

    def invoke(self, *args, **kwargs):
        raise AssertionError("This code path must never invoke the agent graph")


class _StubGraph:
    def __init__(self, reply_content="Stub reply"):
        self.reply_content = reply_content
        self.invoked_with = None

    def invoke(self, payload):
        self.invoked_with = payload
        reply = AIMessage(
            content=self.reply_content,
            usage_metadata={"input_tokens": 5, "output_tokens": 3, "total_tokens": 8},
        )
        return {"messages": payload["messages"] + [reply]}


class _RaisingGraph:
    def invoke(self, payload):
        raise RuntimeError("simulated agent failure")


@pytest.fixture
def tk_root():
    root = tk.Tk()
    root.withdraw()
    yield root
    root.destroy()


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
