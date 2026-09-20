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

    def stream(self, payload, stream_mode=None):
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
    """Derived rather than imported, to keep Tk out of a terminal client --
    so the duplication needs a test or it drifts silently."""
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


def test_a_rejection_sets_the_flag_the_router_reads():
    s = session(StubGraph(final_state={"offered": ["short_walk"]}))
    s.handle("what should I do")

    s.handle("/no")

    assert s.graph.payloads[-1]["rejected"] is True


def test_a_rejection_before_anything_was_offered_is_refused():
    graph = PoisonGraph()
    s = session(graph)

    s.handle("/no")

    assert graph.attempts == 0
    assert "nothing has been suggested" in output(s)


# -- an idea of the user's own -----------------------------------------------


def test_an_idea_travels_in_the_payload():
    s = session()

    s.handle("/idea I could drop the Friday meeting")

    assert s.graph.payloads[-1]["idea"] == "I could drop the Friday meeting"


def test_an_idea_in_crisis_language_never_reaches_the_graph():
    """The gate runs on every path into the graph, including this one."""
    graph = PoisonGraph()
    s = session(graph)

    s.handle("/idea I want to kill myself")

    assert graph.attempts == 0
    assert "Telefonseelsorge" in output(s)


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


def test_a_rejection_is_offered_only_once_something_was_suggested():
    s = session(StubGraph(final_state={"offered": ["short_walk"]}))

    assert "/no" not in s.hints()

    s.handle("what should I do")

    assert "/no" in s.hints()
    assert "/idea" in s.hints()


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
