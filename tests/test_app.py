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

    def stream(self, payload, config=None, stream_mode=None):
        """Mirror the real graph's ("updates", {node: update}) / ("values",
        state) tuple stream, so the app's consumption of it is under test."""
        final = self._final(payload)
        for event in self.tool_events:
            yield ("updates", {"some_node": {"tool_events": [event]}})
        yield ("values", final)


class _RaisingGraph:
    def invoke(self, payload):
        raise RuntimeError("simulated agent failure")

    def stream(self, payload, config=None, stream_mode=None):
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


def test_one_click_runs_a_turn_so_the_agent_can_ask_for_the_other(tk_root):
    """A click is unambiguous self-report, and it goes into the payload
    directly. But what a half-answered pair means is the ask station's job:
    waiting for both left the first click doing nothing visible at all."""
    stub = _StubGraph()
    app = ChatApp(tk_root, skip_consent=True, agent_graph=stub)

    app._on_choice_click("energy", "wired")
    settle(tk_root, app)

    assert stub.invoked_with["energy"] == "wired"
    assert not stub.invoked_with.get("goal")
    assert _visible(app.choice_frame), "goal is still missing, so keep offering it"


def test_both_answers_run_a_turn_that_already_carries_them(tk_root):
    """The point of the pair: route_start sees energy and goal set, goes
    straight to `suggest`, and the `ask` station never runs -- one model call
    for the turn instead of two."""
    stub = _StubGraph(reply_content="How about a short walk?")
    app = ChatApp(tk_root, skip_consent=True, agent_graph=stub)

    app._on_choice_click("energy", "wired")
    settle(tk_root, app)
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


def test_a_failure_after_the_reply_unlocks_the_window(tk_root):
    """_finish_turn runs on the main thread, outside the worker's try/except.
    An error there used to leave the input disabled forever with only a
    stderr traceback -- a window that answered once and then froze, which is
    indistinguishable from a hang. It has to say so and stay usable.

    A recorder that cannot write stands in for the real case: anything
    raising between the reply and _set_busy(False) has this effect.
    """
    class _BrokenRecorder:
        def record_turn(self, *args, **kwargs):
            raise OSError("disk full")

    app = ChatApp(tk_root, skip_consent=True, agent_graph=_StubGraph(),
                  recorder=_BrokenRecorder())
    app.entry.insert(0, "hello")
    app._on_send()
    settle(tk_root, app)

    assert str(app.send_button.cget("state")) == "normal", "the window must stay usable"
    assert "went wrong" in app.history.get("1.0", "end").lower()


# -- the approval card -------------------------------------------------------


ADD_REQUEST = {
    "id": "w1",
    "name": "add_measure",
    "args": {"measure_id": "foam_roll", "measure": "Roll your back."},
    "description": 'Add to your list: "Roll your back."',
    "editable": "measure",
}
RATE_REQUEST = {
    "id": "w2",
    "name": "log_checkin",
    "args": {"measure_id": "breathe", "helpfulness": "helped"},
    "description": 'Rate "breathe" as: helped',
    "editable": None,
}


class _PausingGraph:
    """Pauses once the way the real graph reports it, then finishes on the
    resume. Records every resume value and thread, so the tests assert on
    what was sent and how often -- the worker swallows exceptions, so the
    absence of an error line proves nothing."""

    def __init__(self, *requests):
        from langgraph.types import Interrupt

        self.interrupt = Interrupt(value={"action_requests": list(requests) or [ADD_REQUEST]})
        self.resumes: list[dict] = []
        self.configs: list[dict] = []

    def stream(self, graph_input, config=None, stream_mode=None):
        from langgraph.types import Command

        self.configs.append(config)
        if isinstance(graph_input, Command):
            self.resumes.append(graph_input.resume)
            yield ("values", {"messages": self.payload["messages"] + [AIMessage(content="after the card")]})
            return
        self.payload = graph_input
        yield ("updates", {"__interrupt__": (self.interrupt,)})
        yield ("values", {**graph_input, "__interrupt__": (self.interrupt,)})


def settle_until_card(root, app, timeout=5.0):
    """Pump the event loop until the approval card is on screen. The Send
    button stays disabled during a pause, so `settle` would wait forever."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        root.update()
        if app.approval_frame.winfo_manager() == "pack":
            return
        time.sleep(0.01)
    raise AssertionError("the approval card never appeared")


def paused_app(tk_root, graph):
    app = ChatApp(tk_root, skip_consent=True, agent_graph=graph)
    app.entry.insert(0, "add it")
    app._on_send()
    settle_until_card(tk_root, app)
    return app


def test_a_pause_shows_the_card_and_never_the_error_line(tk_root):
    graph = _PausingGraph()
    app = paused_app(tk_root, graph)

    history = app.history.get("1.0", "end")
    assert "went wrong" not in history.lower()
    assert "Roll your back." in app.approval_label.cget("text")


def test_while_paused_only_the_card_can_be_used(tk_root):
    """A second message mid-pause would start a turn over the paused one."""
    app = paused_app(tk_root, _PausingGraph())
    assert str(app.entry.cget("state")) == "disabled"
    assert str(app.send_button.cget("state")) == "disabled"
    assert str(app.approve_button.cget("state")) == "normal"


def test_approve_resumes_once_on_the_same_thread_and_finishes_the_turn(tk_root):
    graph = _PausingGraph()
    app = paused_app(tk_root, graph)

    app._on_decision("approve")
    settle(tk_root, app)

    assert graph.resumes == [{"decisions": [{"id": "w1", "type": "approve"}]}]
    assert graph.configs[0] == graph.configs[1]
    assert "after the card" in app.history.get("1.0", "end")
    assert app.approval_frame.winfo_manager() == ""


def test_reject_sends_a_rejection(tk_root):
    graph = _PausingGraph()
    app = paused_app(tk_root, graph)

    app._on_decision("reject")
    settle(tk_root, app)

    assert graph.resumes == [{"decisions": [{"id": "w1", "type": "reject"}]}]


def test_an_edit_sends_the_entrys_text_with_the_other_arguments_unchanged(tk_root):
    graph = _PausingGraph()
    app = paused_app(tk_root, graph)
    assert app.approval_entry.get() == "Roll your back."

    app.approval_entry.delete(0, "end")
    app.approval_entry.insert(0, "Roll gently, five minutes.")
    app._on_decision("edit")
    settle(tk_root, app)

    decision = graph.resumes[0]["decisions"][0]
    assert decision["type"] == "edit"
    assert decision["args"] == {"measure_id": "foam_roll", "measure": "Roll gently, five minutes."}


def test_an_empty_edit_is_a_rejection_not_an_approval(tk_root):
    graph = _PausingGraph()
    app = paused_app(tk_root, graph)

    app.approval_entry.delete(0, "end")
    app._on_decision("edit")
    settle(tk_root, app)

    assert graph.resumes[0]["decisions"][0]["type"] == "reject"


def test_a_rating_cannot_be_edited(tk_root):
    app = paused_app(tk_root, _PausingGraph(RATE_REQUEST))
    assert str(app.edit_button.cget("state")) == "disabled"


def test_two_writes_are_two_cards_and_one_resume(tk_root):
    graph = _PausingGraph(ADD_REQUEST, RATE_REQUEST)
    app = paused_app(tk_root, graph)

    app._on_decision("approve")
    tk_root.update()
    assert "breathe" in app.approval_label.cget("text"), "the second card follows the first"
    assert graph.resumes == [], "no resume until every write is answered"
    app._on_decision("reject")
    settle(tk_root, app)

    assert graph.resumes == [
        {"decisions": [{"id": "w1", "type": "approve"}, {"id": "w2", "type": "reject"}]}
    ]


def _real_graph_logging(tmp_path, log):
    import csv

    from langgraph.checkpoint.memory import InMemorySaver

    from agent.graph import build_agent
    from tests.conftest import StubChatModel
    from tools.measures import MEASURE_COLUMNS

    measures = tmp_path / "measures.csv"
    with measures.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(MEASURE_COLUMNS)
        writer.writerow(["breathe", "recover", "Take a few slow breaths.", "exhausted;calm;wired", "minutes", "user"])
    call = AIMessage(content="", tool_calls=[{
        "name": "log_checkin",
        "args": {"energy": "wired", "circumstances": "Deadline week.", "measure_id": "breathe"},
        "id": "w1",
    }])
    model = StubChatModel(responses=[call, AIMessage(content="Thanks.")])
    return build_agent(model=model, log_path=log, measures_path=measures, checkpointer=InMemorySaver())


@pytest.mark.parametrize("decision,rows", [("approve", 1), ("reject", 0)])
def test_through_the_real_graph_only_an_approval_reaches_the_log(tk_root, tmp_path, decision, rows):
    """End to end: stub model, real graph, real file, real window."""
    log = tmp_path / "log.csv"
    graph = _real_graph_logging(tmp_path, log)
    app = ChatApp(tk_root, skip_consent=True, agent_graph=graph, log_path=log)
    app.state.update({"energy": "wired", "goal": "calm down now", "offered": ["breathe"]})

    app.entry.insert(0, "yes, log it")
    app._on_send()
    settle_until_card(tk_root, app)
    app._on_decision(decision)
    settle(tk_root, app)

    written = read_checkin_rows(log) if log.exists() else []
    assert len(written) == rows
    assert "went wrong" not in app.history.get("1.0", "end").lower()


# -- a choice click means a fresh lookup ---------------------------------------


def test_a_choice_click_marks_the_turn_as_a_click(tk_root):
    """The graph only forces a fresh lookup when it knows the values came from
    a button; a click that did not say so would reach follow_up after the goal
    question, where the model invented options (first hand test, turn 7)."""
    stub = _StubGraph()
    app = ChatApp(tk_root, skip_consent=True, agent_graph=stub)

    app._on_choice_click("goal", "plan recovery")
    settle(tk_root, app)

    assert stub.invoked_with["choices_clicked"] is True


def test_free_text_is_never_marked_as_a_click(tk_root):
    stub = _StubGraph()
    app = ChatApp(tk_root, skip_consent=True, agent_graph=stub)

    app.entry.insert(0, "that sounds good")
    app._on_send()
    settle(tk_root, app)

    assert not stub.invoked_with.get("choices_clicked")
