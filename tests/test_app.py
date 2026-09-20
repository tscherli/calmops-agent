"""Tests for app.py's ChatApp: crisis-gate wiring (must intercept before
the agent ever runs), graceful exception handling (no unhandled exception
may reach the Tkinter UI), consent-screen gating, and the three button
groups. How a turn's receipts and carried state are drawn is
tests/test_tool_receipts.py's job.

The buttons carry one guarantee worth stating: **a click never asks the
model anything.** Energy and goal go straight into the payload, and a rating
does not even run the graph -- it writes its row and reports it. Several
tests below use a graph that raises if it is invoked at all, because that is
the only way to prove a cost was not paid."""
import json
import time

import pytest
from langchain_core.messages import AIMessage

import app as app_module
from app import ChatApp
from observability.tracing import RunRecorder
from tools.checkin_log import append_checkin_row, read_checkin_rows


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
    """Fails loudly if invoked -- proves a code path never reaches the agent.

    It also *counts* the attempts, and that is the part that matters. Raising
    alone is not enough: app.py runs the graph on a worker thread, and
    _stream_worker catches everything so no exception can reach the UI. An
    AssertionError raised in there becomes a polite "something went wrong"
    line and the test passes regardless. Any test about a path that runs
    after the window is up must assert on `attempts`, not on the raise.
    """

    def __init__(self):
        self.attempts = 0

    def invoke(self, *args, **kwargs):
        self.attempts += 1
        raise AssertionError("This code path must never invoke the agent graph")

    def stream(self, *args, **kwargs):
        self.attempts += 1
        raise AssertionError("This code path must never invoke the agent graph")


class _StubGraph:
    def __init__(self, reply_content="Stub reply", tool_events=(), final_state=None):
        self.reply_content = reply_content
        self.invoked_with = None
        self.tool_events = list(tool_events)
        # What the graph reports back as the turn's final state: the station
        # fields the window reads to decide which buttons now mean something.
        self.final_state = final_state or {}

    def _final(self, payload):
        self.invoked_with = payload
        reply = AIMessage(
            content=self.reply_content,
            usage_metadata={"input_tokens": 5, "output_tokens": 3, "total_tokens": 8},
        )
        # A real StateGraph returns the whole state, including what was
        # passed in. Echoing the payload back matters: the window decides
        # which buttons to show from the state it gets back, so a stub that
        # dropped the fields would test a graph nobody runs.
        carried = {k: v for k, v in payload.items() if k != "messages"}
        return {
            **carried,
            **self.final_state,
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


@pytest.fixture(autouse=True)
def isolated_log_path(tmp_path, monkeypatch):
    """No test may read or write the real check-in log.

    The rating buttons call append_checkin_row directly, so without this a
    test click would append a fabricated row to the user's own data -- and
    the window reads that file at startup to decide whether to offer a
    rating at all, which would make these tests depend on it.
    """
    path = tmp_path / "checkin_log.csv"
    monkeypatch.setattr("tools.checkin_log.CHECKIN_LOG_PATH", path)
    return path


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


# -- energy and goal: a click skips a whole station --------------------------


def _visible(frame) -> bool:
    return bool(frame.winfo_manager())


def test_the_choice_buttons_are_offered_until_both_answers_are_in(tk_root):
    app = ChatApp(tk_root, skip_consent=True, agent_graph=_PoisonGraph())

    assert _visible(app.choice_frame)


def test_one_click_records_one_value_without_asking_the_model(tk_root):
    """A click is unambiguous self-report. Reading it back out of a sentence
    would cost a model call and could get it wrong."""
    app = ChatApp(tk_root, skip_consent=True, agent_graph=_PoisonGraph())

    app._on_choice_click("energy", "wired")

    assert app.state["energy"] == "wired"
    assert _visible(app.choice_frame), "goal is still missing, so keep asking"


def test_both_answers_run_a_turn_that_already_carries_them(tk_root):
    """The point of the pair: route_start sees energy and goal set, goes
    straight to `suggest`, and the `ask` station never runs -- one model call
    for the turn instead of two."""
    stub = _StubGraph(reply_content="How about a short walk?")
    app = ChatApp(tk_root, skip_consent=True, agent_graph=stub)

    app._on_choice_click("energy", "wired")
    app._on_choice_click("goal", "calm down now")
    settle(tk_root, app)

    assert stub.invoked_with["energy"] == "wired"
    assert stub.invoked_with["goal"] == "calm down now"
    assert not _visible(app.choice_frame)


# -- the rating strip --------------------------------------------------------


def test_no_rating_is_offered_before_anything_has_been_logged(tk_root):
    """A rating is a second row about a measure that was really written.
    Offering one with nothing to attach it to would promise something the
    schema cannot hold."""
    app = ChatApp(tk_root, skip_consent=True, agent_graph=_PoisonGraph())

    assert not _visible(app.rating_frame)


def test_while_suggestions_are_on_screen_the_strip_is_only_a_rejection(tk_root):
    stub = _StubGraph(final_state={"offered": ["short_walk"], "energy": "wired",
                                   "goal": "calm down now"})
    app = ChatApp(tk_root, skip_consent=True, agent_graph=stub)

    app.entry.insert(0, "what should I do")
    app._on_send()
    settle(tk_root, app)

    assert _visible(app.rating_frame)
    assert app._rating_mode == "reject"
    assert not app._rating_buttons["helped"].winfo_manager()


def test_not_helpful_during_suggestions_sets_the_rejected_flag(tk_root):
    """It goes through the graph, but to count_rejection, which is code --
    so the rejection still costs no model call."""
    stub = _StubGraph(final_state={"offered": ["short_walk"]})
    app = ChatApp(tk_root, skip_consent=True, agent_graph=stub)

    app.entry.insert(0, "what should I do")
    app._on_send()
    settle(tk_root, app)
    app._on_rating_click("didnt_help")
    settle(tk_root, app)

    assert stub.invoked_with["rejected"] is True


def test_a_logged_check_in_turns_the_strip_into_a_rating(tk_root):
    stub = _StubGraph(final_state={"complete": True, "last_measure_id": "short_walk"})
    app = ChatApp(tk_root, skip_consent=True, agent_graph=stub)

    app.entry.insert(0, "I'll try the walk")
    app._on_send()
    settle(tk_root, app)

    assert app.last_measure_id == "short_walk"
    assert app._rating_mode == "rate"
    assert app._rating_buttons["helped"].winfo_manager()


def test_a_rating_writes_its_row_without_running_the_graph_at_all(tk_root, isolated_log_path):
    """The strongest form of "no model call": the graph is never reached.

    Asserted by counting, not by the graph raising -- see _PoisonGraph. The
    raise is swallowed by the worker thread, so a test that relied on it
    would pass even if every rating ran a full turn.
    """
    graph = _PoisonGraph()
    app = ChatApp(tk_root, skip_consent=True, agent_graph=graph, log_path=isolated_log_path)
    app.last_measure_id = "short_walk"
    app._refresh_controls()

    app._on_rating_click("helped")
    settle(tk_root, app)

    assert graph.attempts == 0, "a rating must not run the graph at all"
    rows = read_checkin_rows(isolated_log_path)
    assert len(rows) == 1
    assert rows[0]["measure_id"] == "short_walk"
    assert rows[0]["helpfulness"] == "helped"


def test_a_rating_is_a_second_row_not_an_edit(tk_root, isolated_log_path):
    """Append-only: the check-in and its verdict arrive at different times,
    so the verdict cannot be an edit of the earlier row."""
    append_checkin_row(energy="wired", circumstances="deadlines",
                       measure_id="short_walk", path=isolated_log_path)
    app = ChatApp(tk_root, skip_consent=True, agent_graph=_PoisonGraph(),
                  log_path=isolated_log_path)

    app._on_rating_click("somewhat")

    rows = read_checkin_rows(isolated_log_path)
    assert len(rows) == 2
    assert rows[0]["circumstances"] == "deadlines"
    assert rows[1]["circumstances"] == "", "a rating carries only the verdict"


def test_the_rating_is_reported_by_a_receipt(tk_root, isolated_log_path):
    app = ChatApp(tk_root, skip_consent=True, agent_graph=_PoisonGraph(),
                  log_path=isolated_log_path)
    app.last_measure_id = "short_walk"

    app._on_rating_click("helped")

    history = app.history.get("1.0", "end")
    assert "log_checkin" in history and "exit 0" in history


def test_the_same_check_in_cannot_be_rated_twice(tk_root, isolated_log_path):
    app = ChatApp(tk_root, skip_consent=True, agent_graph=_PoisonGraph(),
                  log_path=isolated_log_path)
    app.last_measure_id = "short_walk"
    app._refresh_controls()

    app._on_rating_click("helped")

    assert app.last_measure_id == ""
    assert not _visible(app.rating_frame)


def test_a_rating_still_works_after_the_window_is_reopened(tk_root, isolated_log_path):
    """"Come back later and say how it went" is the whole reason a rating is
    its own row; it would not survive a restart if the id lived only in
    memory."""
    append_checkin_row(energy="wired", circumstances="deadlines",
                       measure_id="short_walk", path=isolated_log_path)

    app = ChatApp(tk_root, skip_consent=True, agent_graph=_PoisonGraph(),
                  log_path=isolated_log_path)

    assert app.last_measure_id == "short_walk"
    assert app._rating_mode == "rate"


def test_an_already_rated_check_in_is_not_offered_again_on_reopen(tk_root, isolated_log_path):
    append_checkin_row(energy="wired", circumstances="deadlines",
                       measure_id="short_walk", path=isolated_log_path)
    append_checkin_row(measure_id="short_walk", helpfulness="helped", path=isolated_log_path)

    app = ChatApp(tk_root, skip_consent=True, agent_graph=_PoisonGraph(),
                  log_path=isolated_log_path)

    assert app.last_measure_id == ""


def test_a_measure_in_the_users_own_words_offers_no_rating(tk_root, isolated_log_path):
    """The schema ties a rating to a measure_id. With none, there is nothing
    honest for the buttons to attach to."""
    append_checkin_row(energy="wired", circumstances="deadlines",
                       measure_taken="rearranged my desk", path=isolated_log_path)

    app = ChatApp(tk_root, skip_consent=True, agent_graph=_PoisonGraph(),
                  log_path=isolated_log_path)

    assert app.last_measure_id == ""


# -- discussing an idea of the user's own ------------------------------------


def test_the_idea_button_appears_only_once_something_was_suggested(tk_root):
    stub = _StubGraph(final_state={"offered": ["short_walk"]})
    app = ChatApp(tk_root, skip_consent=True, agent_graph=stub)

    assert not _visible(app.idea_frame)

    app.entry.insert(0, "what should I do")
    app._on_send()
    settle(tk_root, app)

    assert _visible(app.idea_frame)


def test_the_next_message_travels_as_the_idea(tk_root):
    """`debate` gets the idea as a fact in the payload rather than having to
    spot it in the prose."""
    stub = _StubGraph(final_state={"offered": ["short_walk"]})
    app = ChatApp(tk_root, skip_consent=True, agent_graph=stub)
    app.entry.insert(0, "what should I do")
    app._on_send()
    settle(tk_root, app)

    app._on_idea_click()
    app.entry.insert(0, "I could just quit the project")
    app._on_send()
    settle(tk_root, app)

    assert stub.invoked_with["idea"] == "I could just quit the project"


def test_the_button_arms_one_message_not_every_later_one(tk_root):
    """The idea itself is meant to persist -- the debate runs until the user
    decides, and the graph carries `idea` until idea_decided clears it. What
    must not persist is the *button*: if it stayed armed, every following
    sentence would overwrite the idea under discussion with itself."""
    stub = _StubGraph(final_state={"offered": ["short_walk"]})
    app = ChatApp(tk_root, skip_consent=True, agent_graph=stub)
    app._on_idea_click()

    app.entry.insert(0, "I could just quit the project")
    app._on_send()
    settle(tk_root, app)
    app.entry.insert(0, "what do you think")
    app._on_send()
    settle(tk_root, app)

    assert app._idea_pending is False
    assert stub.invoked_with["idea"] == "I could just quit the project"


def test_an_idea_in_crisis_language_never_reaches_the_agent(tk_root):
    """The gate runs on the idea text like any other message. Routing it
    through a different door must not open a way around it."""
    app = ChatApp(tk_root, skip_consent=True, agent_graph=_PoisonGraph())
    app._on_idea_click()

    app.entry.insert(0, "I want to kill myself")
    app._on_send()   # would raise via _PoisonGraph if the gate let it through

    assert "Telefonseelsorge" in app.history.get("1.0", "end")


# -- the consent text describes the app that now exists ----------------------


def test_the_consent_text_no_longer_promises_a_no_ai_path(tk_root):
    """It used to point at quick-log buttons as the way to use this without
    sharing anything. Those buttons are gone, and consent text that describes
    a UI that does not exist is the one kind of stale text that matters."""
    lowered = app_module.CONSENT_TEXT.lower()

    assert "quick" not in lowered
    assert "mood" not in lowered


def test_the_consent_text_says_what_the_buttons_do_and_do_not_send(tk_root):
    lowered = app_module.CONSENT_TEXT.lower()

    assert "buttons" in lowered
    assert "everything you type is sent" in lowered
    assert "web search" in lowered
