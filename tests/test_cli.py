"""Tests for cli.py -- the terminal client.

It exists so the paths that never involve the model can be driven at all: the
energy/goal pair that skips a station, the rejection, the idea handoff, and
the rating that writes its row without running the graph. A client that could
only type free text would reach none of them.

The guarantees are the window's guarantees, which is the point -- both are
thin clients over the same graph, and anything true of one that is false of
the other means one of them is testing a system nobody runs.
"""

import pytest
from langchain_core.messages import AIMessage

import cli
from cli import Session
from tools.checkin_log import append_checkin_row, read_checkin_rows


@pytest.fixture(autouse=True)
def isolated_log_path(tmp_path, monkeypatch):
    """/rate writes straight to the log, so no test may see the real file."""
    path = tmp_path / "checkin_log.csv"
    monkeypatch.setattr("tools.checkin_log.CHECKIN_LOG_PATH", path)
    return path


class StubGraph:
    """Echoes the payload back the way a real StateGraph does, so the client
    sees the station fields it sent."""

    def __init__(self, reply="Stub reply", events=(), final_state=None):
        self.reply = reply
        self.events = list(events)
        self.final_state = final_state or {}
        self.payloads: list[dict] = []

    def stream(self, payload, config=None, stream_mode=None):
        self.payloads.append(payload)
        for event in self.events:
            yield ("updates", {"follow_up": {"tool_events": [event]}})
        carried = {k: v for k, v in payload.items() if k != "messages"}
        yield (
            "values",
            {
                **carried,
                **self.final_state,
                "messages": payload["messages"] + [AIMessage(content=self.reply)],
                "tool_events": list(self.events),
            },
        )


class PoisonGraph:
    """Counts attempts rather than only raising: the raise alone would be
    caught by the client's own error handling and prove nothing."""

    def __init__(self):
        self.attempts = 0

    def stream(self, *args, **kwargs):
        self.attempts += 1
        raise AssertionError("this path must not run the graph")


def session(graph=None, **kwargs):
    lines: list[str] = []
    s = Session(graph or StubGraph(), out=lines.append, **kwargs)
    s.lines = lines          # type: ignore[attr-defined]
    return s


def output(s) -> str:
    return "\n".join(s.lines)


# -- the client agrees with the window ---------------------------------------


def test_the_consent_path_matches_the_windows():
    """Both take it from agent/data_paths.py -- importing app.py would pull Tk
    into a terminal client -- and a test keeps them from drifting apart."""
    import app

    assert cli.CONSENT_PATH == app.CONSENT_PATH


def test_the_vocabularies_come_from_the_schema():
    assert cli.ENERGY_OPTIONS == ("exhausted", "calm", "wired")
    assert cli.GOAL_OPTIONS == ("calm down now", "plan recovery", "change the situation")


# -- energy and goal skip a station ------------------------------------------


def test_one_choice_runs_a_turn_so_the_agent_can_ask_for_the_other():
    """Deciding what a half-answered pair means belongs to the ask station,
    not to the client. Holding the turn back left the click looking ignored,
    and the user with no idea what was still wanted."""
    s = session()

    s.handle("/energy wired")

    assert len(s.graph.payloads) == 1
    assert s.graph.payloads[0]["energy"] == "wired"
    assert not s.graph.payloads[0].get("goal")


def test_the_second_choice_arrives_with_the_first_still_set():
    s = session()

    s.handle("/energy wired")
    s.handle("/goal calm down now")

    assert s.graph.payloads[-1]["energy"] == "wired"
    assert s.graph.payloads[-1]["goal"] == "calm down now"


def test_a_goal_can_be_given_by_prefix():
    """The full goal strings are long enough that nobody would type them."""
    s = session()

    s.handle("/energy calm")
    s.handle("/goal change")

    assert s.graph.payloads[-1]["goal"] == "change the situation"


def test_an_unknown_value_is_refused_rather_than_guessed():
    graph = PoisonGraph()
    s = session(graph)

    s.handle("/energy sleepy")

    assert "energy" not in s.state
    assert graph.attempts == 0
    assert "needs one of" in output(s)


def test_an_ambiguous_prefix_is_refused():
    """"c" matches calm and change, and picking one would be a guess about
    what the user meant."""
    s = session(PoisonGraph())

    s.handle("/goal c")

    assert "goal" not in s.state


# -- rejection ---------------------------------------------------------------


def test_the_rejection_command_is_gone_and_the_words_still_work():
    """/no and /idea were removed with the buttons they mirrored: the agent
    asks at the point where each makes sense, and the user answers in words.
    An unknown command must not be sent to the agent as a message either."""
    graph = PoisonGraph()
    s = session(graph)

    s.handle("/no")

    assert graph.attempts == 0
    assert "unknown command" in output(s).lower()


# -- an idea of the user's own -----------------------------------------------


def test_an_idea_typed_in_words_reaches_the_graph_as_a_message():
    """It used to travel in the payload from /idea. Now the model hears it and
    calls discuss_idea itself."""
    s = session()

    s.handle("I could drop the Friday meeting")

    assert "idea" not in s.graph.payloads[-1]
    assert s.graph.payloads[-1]["messages"][-1].content == "I could drop the Friday meeting"


def test_a_typed_message_is_screened_too():
    graph = PoisonGraph()
    s = session(graph)

    s.handle("I want to kill myself")

    assert graph.attempts == 0


# -- rating ------------------------------------------------------------------


def test_a_rating_writes_its_row_without_running_the_graph(isolated_log_path):
    graph = PoisonGraph()
    s = session(graph, log_path=isolated_log_path)
    s.last_measure_id = "short_walk"

    s.handle("/rate helped")

    assert graph.attempts == 0
    rows = read_checkin_rows(isolated_log_path)
    assert [(r["measure_id"], r["helpfulness"]) for r in rows] == [("short_walk", "helped")]


def test_not_helpful_is_accepted_as_a_rating_word(isolated_log_path):
    s = session(PoisonGraph(), log_path=isolated_log_path)
    s.last_measure_id = "short_walk"

    s.handle("/rate not")

    assert read_checkin_rows(isolated_log_path)[0]["helpfulness"] == "didnt_help"


def test_rating_before_anything_is_logged_is_refused(isolated_log_path):
    s = session(PoisonGraph(), log_path=isolated_log_path)

    s.handle("/rate helped")

    assert read_checkin_rows(isolated_log_path) == []
    assert "has to be logged first" in output(s)


def test_the_same_check_in_cannot_be_rated_twice(isolated_log_path):
    s = session(PoisonGraph(), log_path=isolated_log_path)
    s.last_measure_id = "short_walk"

    s.handle("/rate helped")
    s.handle("/rate somewhat")

    assert len(read_checkin_rows(isolated_log_path)) == 1


def test_a_rating_survives_a_restart(isolated_log_path):
    append_checkin_row(energy="wired", circumstances="deadlines",
                       measure_id="short_walk", path=isolated_log_path)

    s = session(PoisonGraph(), log_path=isolated_log_path)

    assert s.last_measure_id == "short_walk"
    assert "/rate" in s.hints()


# -- what the client reports -------------------------------------------------


def test_receipts_are_printed_before_the_reply():
    """Same guarantee as the window: the evidence must not trail the claim."""
    s = session(StubGraph(
        reply="I've noted that.",
        events=[{"tool": "log_checkin", "exit_code": 0, "detail": "checkin row abc"}],
    ))

    s.handle("log it please")

    text = output(s)
    assert text.index("log_checkin") < text.index("I've noted that.")


def test_a_turn_that_logs_nothing_says_so():
    s = session()

    s.handle("hello")

    assert "not called this turn" in output(s)


def test_the_station_and_completion_are_reported():
    s = session(StubGraph(final_state={"step": "suggest", "complete": False}))

    s.handle("what should I do")

    assert "station: suggest" in output(s)


def test_a_failing_graph_does_not_end_the_session():
    class Boom:
        def stream(self, *a, **k):
            raise RuntimeError("simulated outage")
            yield  # pragma: no cover

    s = session(Boom())

    assert s.handle("hello") is True
    assert "went wrong" in output(s)


def test_quit_ends_the_session():
    s = session()

    assert s.handle("/quit") is False


def test_an_unknown_command_is_not_sent_to_the_agent():
    graph = PoisonGraph()
    s = session(graph)

    s.handle("/frobnicate")

    assert graph.attempts == 0
    assert "Unknown command" in output(s)


# -- the hints mirror the window's buttons -----------------------------------


def test_choices_are_offered_until_both_are_in():
    s = session(PoisonGraph())

    assert "/energy" in s.hints()


def test_the_hints_no_longer_offer_the_removed_commands():
    s = session(StubGraph(final_state={"offered": ["short_walk"]}))
    s.handle("what should I do")

    assert "/no" not in s.hints()
    assert "/idea" not in s.hints()


def test_the_state_command_reports_what_is_carried():
    s = session(StubGraph(final_state={"offered": ["short_walk"], "step": "suggest"}))
    s.handle("what should I do")

    s.handle("/state")

    assert "short_walk" in output(s)


def test_the_click_says_only_what_was_clicked():
    """The sentence the click puts in the transcript must not claim the
    value that has not been given -- the station reads it."""
    s = session()

    s.handle("/goal calm")

    said = s.graph.payloads[0]["messages"][-1].content
    assert "calm down now" in said
    assert "feeling" not in said, "no energy was given, so none may be implied"


def test_either_order_ends_with_both_values():
    s = session()

    s.handle("/goal calm")
    s.handle("/energy exhausted")

    assert s.graph.payloads[-1]["goal"] == "calm down now"
    assert s.graph.payloads[-1]["energy"] == "exhausted"


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


class PausingGraph:
    """Pauses once, the way the real graph reports it, then finishes on the
    resume. Records every resume value so a test asserts on what was sent,
    and how often, instead of trusting the absence of an error line."""

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


def answering(*answers):
    replies = iter(answers)

    def ask(prompt):
        return next(replies)

    return ask


def test_a_pause_shows_the_card_and_never_the_error_line():
    """Read like a node update, the pause raised, and the client's own error
    handler turned it into 'something went wrong' -- a working app, visibly."""
    graph = PausingGraph()
    s = session(graph, ask=answering("a"))

    s.handle("add it")

    text = output(s)
    assert "went wrong" not in text
    assert 'Add to your list: "Roll your back."' in text
    assert graph.resumes == [{"decisions": [{"id": "w1", "type": "approve"}]}]
    assert "after the card" in text


def test_the_resume_goes_to_the_same_thread_as_the_pause():
    """A resume on another thread would find no pause to continue."""
    graph = PausingGraph()
    session(graph, ask=answering("a")).handle("add it")
    assert graph.configs[0] == graph.configs[1]
    assert graph.configs[0]["configurable"]["thread_id"]


def test_each_turn_gets_its_own_thread():
    graph = StubGraph()
    seen = []
    original = graph.stream

    def spy(payload, config=None, stream_mode=None):
        seen.append(config["configurable"]["thread_id"])
        return original(payload, config, stream_mode)

    graph.stream = spy
    s = session(graph)
    s.handle("one")
    s.handle("two")
    assert len(set(seen)) == 2


@pytest.mark.parametrize("answer", ["r", "", "maybe", "no"])
def test_anything_but_an_approval_rejects(answer):
    graph = PausingGraph()
    s = session(graph, ask=answering(answer))

    s.handle("add it")

    assert graph.resumes == [{"decisions": [{"id": "w1", "type": "reject"}]}]
    assert "nothing is written" in output(s)


def test_an_edit_sends_the_users_text_with_the_other_arguments_unchanged():
    graph = PausingGraph()
    s = session(graph, ask=answering("e", "Roll gently, five minutes."))

    s.handle("add it")

    decision = graph.resumes[0]["decisions"][0]
    assert decision["type"] == "edit"
    assert decision["args"] == {"measure_id": "foam_roll", "measure": "Roll gently, five minutes."}


def test_an_empty_edit_is_a_rejection_not_an_approval():
    graph = PausingGraph()
    session(graph, ask=answering("e", "")).handle("add it")
    assert graph.resumes[0]["decisions"][0]["type"] == "reject"


def test_a_rating_has_nothing_to_edit():
    """The card offers what the request allows; 'e' on a rating is not an
    approval of anything."""
    graph = PausingGraph(RATE_REQUEST)
    s = session(graph, ask=answering("e"))

    s.handle("it helped")

    assert "[e]dit" not in output(s)
    assert graph.resumes[0]["decisions"] == [{"id": "w2", "type": "reject"}]


def test_closed_input_rejects_instead_of_crashing():
    def closed(prompt):
        raise EOFError

    graph = PausingGraph()
    s = session(graph, ask=closed)

    s.handle("add it")

    assert graph.resumes[0]["decisions"][0]["type"] == "reject"


def test_two_writes_get_two_questions_and_one_resume():
    graph = PausingGraph(ADD_REQUEST, RATE_REQUEST)
    session(graph, ask=answering("a", "r")).handle("both")
    assert graph.resumes == [
        {"decisions": [{"id": "w1", "type": "approve"}, {"id": "w2", "type": "reject"}]}
    ]


def test_an_approved_check_in_reaches_the_real_log(tmp_path):
    """End to end through the real graph: stub model, real tools node, real
    file. The only way into the log is the card."""
    import csv

    from langchain_core.messages import AIMessage as AI
    from langgraph.checkpoint.memory import InMemorySaver

    from agent.graph import build_agent
    from tests.conftest import StubChatModel
    from tools.measures import MEASURE_COLUMNS

    measures = tmp_path / "measures.csv"
    with measures.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(MEASURE_COLUMNS)
        writer.writerow(["breathe", "recover", "Take a few slow breaths.", "exhausted;calm;wired", "minutes", "user"])
    log = tmp_path / "log.csv"
    call = AI(content="", tool_calls=[{
        "name": "log_checkin",
        "args": {"energy": "wired", "circumstances": "Deadline week.", "measure_id": "breathe"},
        "id": "w1",
    }])
    model = StubChatModel(responses=[call, AI(content="Thanks.")])
    graph = build_agent(model=model, log_path=log, measures_path=measures, checkpointer=InMemorySaver())

    s = session(graph, ask=answering("a"), log_path=log)
    s.state.update({"energy": "wired", "goal": "calm down now", "offered": ["breathe"]})
    s.handle("yes, log it")

    rows = list(csv.DictReader(log.open(encoding="utf-8", newline="")))
    assert [r["circumstances"] for r in rows] == ["Deadline week."]
    assert "went wrong" not in output(s)


# -- a choice command means a fresh lookup -------------------------------------


def test_a_choice_command_marks_the_turn_as_a_click():
    """Same rule as the window: the graph forces a fresh lookup only when it
    knows the values came from a button."""
    s = session()

    s.handle("/goal plan")

    assert s.graph.payloads[-1]["choices_clicked"] is True


def test_free_text_is_never_marked_as_a_click():
    s = session()

    s.handle("that sounds good")

    assert not s.graph.payloads[-1].get("choices_clicked")


def test_the_readers_call_is_in_the_turns_record(tmp_path):
    """The same guarantee as the window's: the reader's call is logged with the turn."""
    import json

    from observability.tracing import RunRecorder

    reader_call = AIMessage(content='{"acts": ["other"]}',
                            usage_metadata={"input_tokens": 100, "output_tokens": 10, "total_tokens": 110})
    s = session(StubGraph(final_state={"side_calls": [reader_call]}),
                recorder=RunRecorder(log_path=tmp_path / "runs.jsonl"))

    s.run_turn("what is this app for")

    record = json.loads((tmp_path / "runs.jsonl").read_text(encoding="utf-8").splitlines()[0])
    assert record["side_calls"] == 1
    assert record["input_tokens"] == 100


# -- one data folder -----------------------------------------------------------


def test_the_agent_logs_where_rate_does(tmp_path, monkeypatch):
    """--log-path used to move only /rate: the agent's own log_checkin kept
    writing to the default log."""
    built = {}
    monkeypatch.setattr(cli, "build_agent", lambda **kwargs: built.update(kwargs) or StubGraph())

    Session(log_path=tmp_path / "log.csv", measures_path=tmp_path / "measures.csv", out=lambda _: None)

    assert built["log_path"] == tmp_path / "log.csv"
    assert built["measures_path"] == tmp_path / "measures.csv"


def _run_main(monkeypatch, *argv):
    """cli.main() with these arguments, until the first prompt; the session it made."""
    import tools.checkin_log as checkin_log

    monkeypatch.setattr(checkin_log, "PENDING_LOG_PATH", checkin_log.PENDING_LOG_PATH)
    made = {}

    class FakeSession:
        def __init__(self, **kwargs):
            made.update(kwargs)

        def hints(self):
            return ""

    def end_of_input(_prompt=""):
        raise EOFError

    monkeypatch.setattr(cli, "Session", FakeSession)
    monkeypatch.setattr("sys.argv", ["cli.py", *argv])
    monkeypatch.setattr("builtins.input", end_of_input)
    cli.main()
    return made, checkin_log.PENDING_LOG_PATH


def test_main_keeps_everything_in_the_data_dir(tmp_path, monkeypatch):
    made, pending = _run_main(monkeypatch, "--data-dir", str(tmp_path / "data"))

    root = (tmp_path / "data").resolve()
    assert made["log_path"] == root / "checkin_log.csv"
    assert made["measures_path"] == root / "measures.csv"
    assert made["recorder"].log_path == root / "runs.jsonl"
    assert (root / "measures.csv").exists()
    assert pending == root / "state" / "pending_sheet_log.jsonl"


def test_log_path_still_moves_the_log_alone(tmp_path, monkeypatch):
    made, _ = _run_main(monkeypatch, "--data-dir", str(tmp_path / "data"), "--log-path", str(tmp_path / "other.csv"))

    assert made["log_path"] == tmp_path / "other.csv"
    assert made["measures_path"] == (tmp_path / "data").resolve() / "measures.csv"
