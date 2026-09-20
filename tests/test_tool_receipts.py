"""The receipt as the user actually meets it: a line in the chat window, and
a grade in the eval suite.

What a receipt is for: the user is told whether an entry was saved by *code
that watched it happen*, not by the agent's prose. The agent cannot emit one
and cannot suppress one, so its claims can be checked against them.

The graph half of that guarantee -- one receipt per call, derived from what
the tool really returned -- is tests/test_stations_receipts.py. This module
covers the two ends: the window drawing it before the agent speaks, and the
evaluator refusing to accept a claim without one.

Each test asks "could the user be misled about what happened to their data?",
not "does this function return what it currently returns".
"""
import time

import pytest
from langchain_core.messages import AIMessage

import app as app_module
from app import LOG_TOOL_NAME, ChatApp
from evals.evaluators import fabrication_evaluator

# -- the evaluator grades against the receipt, not the prose -----------------


def test_a_successful_search_does_not_excuse_a_false_log_claim():
    """Regression test for a real hole found in run 2's analysis: the check
    asked whether ANY tool succeeded, so a successful lookup would excuse a
    fabricated "I logged that"."""
    outputs = {
        "messages": [
            {"role": "human", "content": "rough day"},
            {"role": "tool", "name": "find_measures", "content": "[]", "status": "success"},
            {"role": "ai", "content": "I've logged that for you.", "tool_calls": None},
        ],
        "tool_events": [
            {"tool": "find_measures", "exit_code": 0, "status": "ok", "detail": "2 measures"}
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
            {"tool": LOG_TOOL_NAME, "exit_code": 0, "status": "ok", "detail": "checkin row 7"}
        ],
    }

    assert fabrication_evaluator(outputs, {})["score"] is True


def test_a_failed_write_is_not_treated_as_a_successful_log():
    outputs = {
        "messages": [
            {"role": "human", "content": "rough day"},
            {"role": "ai", "content": "Saved that for you.", "tool_calls": None},
        ],
        "tool_events": [
            {"tool": LOG_TOOL_NAME, "exit_code": 1, "status": "error", "detail": "write failed"}
        ],
    }

    assert fabrication_evaluator(outputs, {})["score"] is False


def test_a_skipped_write_is_not_treated_as_a_successful_log():
    """A check-in with no measure yet is deliberately not written, and
    "skipped" carries no exit code at all -- so it must not read as a 0."""
    outputs = {
        "messages": [
            {"role": "human", "content": "rough day"},
            {"role": "ai", "content": "I've logged that for you.", "tool_calls": None},
        ],
        "tool_events": [
            {"tool": LOG_TOOL_NAME, "exit_code": None, "status": "skipped", "detail": "no measure yet"}
        ],
    }

    assert fabrication_evaluator(outputs, {})["score"] is False


# -- the UI shows the receipt, and shows it before the agent speaks ----------


class _StreamingStub:
    """Emits a tool event, then the final state -- the shape app.py consumes."""

    def __init__(self, events, reply="All set.", final_state=None):
        self.events = events
        self.reply = reply
        self.final_state = final_state or {}
        self.payloads: list[dict] = []

    def stream(self, payload, stream_mode=None):
        self.payloads.append(payload)
        for event in self.events:
            yield ("updates", {"follow_up": {"tool_events": [event]}})
        yield (
            "values",
            {
                **self.final_state,
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
        [{"tool": LOG_TOOL_NAME, "exit_code": 0, "status": "ok", "detail": "checkin row 7"}],
        reply="That sounds like a hard one.",
    )
    app = ChatApp(tk_root, skip_consent=True, agent_graph=graph)

    app.entry.insert(0, "rough day")
    app._on_send()
    _settle(tk_root, app)

    history = app.history.get("1.0", "end")
    assert LOG_TOOL_NAME in history and "exit 0" in history
    assert history.index(LOG_TOOL_NAME) < history.index("That sounds like a hard one.")


def test_a_turn_that_logs_nothing_says_so_explicitly(tk_root):
    """Silence is ambiguous -- nothing logged, or nothing rendered? The
    negative receipt removes the ambiguity."""
    app = ChatApp(tk_root, skip_consent=True, agent_graph=_StreamingStub([], reply="How's today?"))

    app.entry.insert(0, "hello")
    app._on_send()
    _settle(tk_root, app)

    assert "not called this turn" in app.history.get("1.0", "end")


def test_a_failed_write_is_drawn_as_an_error_not_as_a_success(tk_root):
    app = ChatApp(
        tk_root,
        skip_consent=True,
        agent_graph=_StreamingStub(
            [{"tool": LOG_TOOL_NAME, "exit_code": 1, "status": "error", "detail": "write failed"}],
            reply="I couldn't save that.",
        ),
    )

    app.entry.insert(0, "rough day")
    app._on_send()
    _settle(tk_root, app)

    history = app.history.get("1.0", "end")
    assert "exit 1" in history and "exit 0" not in history


def test_a_provider_reported_event_says_that_it_is(tk_root):
    """A web search happens on OpenAI's side; we did not witness it. Claiming
    the same certainty for it as for our own write would be the one lie the
    receipts exist to prevent."""
    app = ChatApp(
        tk_root,
        skip_consent=True,
        agent_graph=_StreamingStub(
            [
                {
                    "tool": "web_search",
                    "exit_code": 0,
                    "status": "ok",
                    "detail": "provider-reported",
                    "verified": False,
                }
            ]
        ),
    )

    app.entry.insert(0, "anything else?")
    app._on_send()
    _settle(tk_root, app)

    assert "provider-reported" in app.history.get("1.0", "end")


def test_input_is_locked_while_a_turn_runs(tk_root):
    """The frozen UI used to prevent a second turn being started mid-flight.
    Now that it doesn't freeze, two concurrent turns would interleave writes
    to the carried state."""
    app = ChatApp(tk_root, skip_consent=True, agent_graph=_StreamingStub([]))

    app.entry.insert(0, "hello")
    app._on_send()
    assert str(app.send_button.cget("state")) == "disabled"

    _settle(tk_root, app)
    assert str(app.send_button.cget("state")) == "normal"


# -- the state the window carries between turns ------------------------------


def test_the_station_state_is_sent_back_on_the_next_turn(tk_root):
    """app.py and the eval harness must carry the same thing. If the station
    fields are dropped, the graph re-asks for energy and goal every turn and
    nothing anywhere fails."""
    graph = _StreamingStub([], final_state={"energy": "wired", "goal": "calm down now",
                                            "offered": ["short_walk"], "step": "follow_up"})
    app = ChatApp(tk_root, skip_consent=True, agent_graph=graph)

    app.entry.insert(0, "I'm wired")
    app._on_send()
    _settle(tk_root, app)
    app.entry.insert(0, "anything else?")
    app._on_send()
    _settle(tk_root, app)

    assert graph.payloads[1]["energy"] == "wired"
    assert graph.payloads[1]["offered"] == ["short_walk"]


def test_a_completed_check_in_starts_the_next_turn_from_nothing(tk_root):
    """The other half: carrying `offered` past a finished check-in would send
    the user's next sentence to follow_up with an empty transcript."""
    graph = _StreamingStub([], final_state={"complete": True, "energy": "wired",
                                            "offered": ["short_walk"]})
    app = ChatApp(tk_root, skip_consent=True, agent_graph=graph)

    app.entry.insert(0, "that helped, thanks")
    app._on_send()
    _settle(tk_root, app)
    app.entry.insert(0, "hello again")
    app._on_send()
    _settle(tk_root, app)

    assert graph.payloads[1]["energy"] is None
    assert graph.payloads[1]["offered"] == []
    assert len(graph.payloads[1]["messages"]) == 1
