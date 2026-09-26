"""Tests for evals/expectations.py -- a case's own assertions about a run.

No model and no graph here: these check the grading, and the grading reads a
dict. What matters is that every field goes red for the failure it names and
stays quiet otherwise, because an expect block that cannot fail is a case
that costs a model call and asserts nothing.
"""
import csv
import json

import pytest

from evals import expectations
from evals.expectations import check_expectations, expectation_evaluator
from tools.measures import MEASURE_COLUMNS

MEASURE_ROWS = [
    ("box_breathing", "recover", "Breathe out longer than you breathe in.", "exhausted;wired", "minutes", "user"),
    ("stand_up", "recover", "Stand up and stretch once.", "exhausted", "minutes", "agent"),
    ("say_no", "act", "Say no to one request today.", "calm;wired", "minutes", "agent"),
]


@pytest.fixture
def measures_file(tmp_path):
    path = tmp_path / "measures.csv"
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(MEASURE_COLUMNS)
        writer.writerows(MEASURE_ROWS)
    return path


def _scores(results):
    return {r["key"]: r["score"] for r in results}


def _outputs(**overrides):
    outputs = {
        "messages": [{"role": "human", "content": "exhausted after a long week of meetings"}],
        "tool_events": [],
        "approvals": [],
        "log_rows": [],
        "measures_added": [],
        "reached_write": False,
    }
    outputs.update(overrides)
    return outputs


def _lookup_message(measure_ids):
    return {
        "role": "tool",
        "name": "find_measures",
        "content": json.dumps([{"measure_id": m} for m in measure_ids]),
    }


# -- the block as a whole ----------------------------------------------------


def test_no_expect_block_asserts_nothing():
    assert check_expectations(_outputs(), None) == []
    assert check_expectations(_outputs(), {}) == []


def test_one_feedback_key_per_field():
    """A block failing as one red line says nothing about which of its claims
    broke, and stops the others from being exercised at all."""
    results = check_expectations(
        _outputs(), {"reached_write": False, "measures_added": [], "tools_not_called": ["add_measure"]}
    )

    assert sorted(_scores(results)) == [
        "expect_measures_added",
        "expect_reached_write",
        "expect_tools_not_called",
    ]


def test_a_misspelled_field_fails_rather_than_being_skipped():
    results = check_expectations(_outputs(), {"tools_call": ["find_measures"]})

    assert _scores(results) == {"expect_schema": False}


def test_the_evaluator_returns_the_multi_key_shape():
    """LangSmith accepts several feedback entries from one evaluator only in
    this shape, and run_local unwraps the same one."""
    result = expectation_evaluator(
        outputs=_outputs(), reference_outputs={"expect": {"reached_write": False}}
    )

    assert _scores(result["results"]) == {"expect_reached_write": True}


# -- tools -------------------------------------------------------------------


def test_tools_called_is_a_subset_not_an_exact_list():
    """A stray lookup is not a failure. Asserting the exact set of tools makes
    every case fail on the day the agent gets one more thing right."""
    outputs = _outputs(
        tool_events=[
            {"tool": "find_measures", "exit_code": 0},
            {"tool": "search_measures", "exit_code": 0},
        ]
    )

    assert _scores(check_expectations(outputs, {"tools_called": ["find_measures"]})) == {
        "expect_tools_called": True
    }


def test_tools_called_fails_on_a_tool_that_never_ran():
    results = check_expectations(_outputs(), {"tools_called": ["find_measures"]})

    assert _scores(results) == {"expect_tools_called": False}
    assert "never called: find_measures" in results[0]["comment"]


def test_tools_not_called_catches_the_tool_running_anyway():
    outputs = _outputs(tool_events=[{"tool": "add_measure", "exit_code": 0}])

    assert _scores(check_expectations(outputs, {"tools_not_called": ["add_measure"]})) == {
        "expect_tools_not_called": False
    }


# -- the lookup --------------------------------------------------------------


def test_lookup_rows_passes_when_the_whole_slice_came_back(measures_file):
    outputs = _outputs(
        messages=[_lookup_message(["box_breathing", "stand_up"])], measures_path=str(measures_file)
    )

    results = check_expectations(outputs, {"lookup_rows": {"energy": "exhausted", "goal": "calm down now"}})

    assert _scores(results) == {"expect_lookup_rows": True}


def test_lookup_rows_catches_a_window_over_the_slice(measures_file):
    """The defect this field exists for: the rows are in the list, the model
    never sees them, and every other check stays green because whatever it
    suggested was plausible."""
    outputs = _outputs(messages=[_lookup_message(["box_breathing"])], measures_path=str(measures_file))

    results = check_expectations(outputs, {"lookup_rows": {"energy": "exhausted", "goal": "calm down now"}})

    assert _scores(results) == {"expect_lookup_rows": False}
    assert "1 of 2 rows" in results[0]["comment"]
    assert "stand_up" in results[0]["comment"]


def test_lookup_rows_checks_every_lookup_not_only_the_last(measures_file):
    outputs = _outputs(
        messages=[_lookup_message(["box_breathing"]), _lookup_message(["box_breathing", "stand_up"])],
        measures_path=str(measures_file),
    )

    results = check_expectations(outputs, {"lookup_rows": {"energy": "exhausted", "goal": "calm down now"}})

    assert _scores(results) == {"expect_lookup_rows": False}
    assert "lookup 1" in results[0]["comment"]


def test_lookup_rows_is_not_graded_when_the_case_never_looked_up(measures_file):
    outputs = _outputs(measures_path=str(measures_file))

    results = check_expectations(outputs, {"lookup_rows": {"energy": "exhausted"}})

    assert _scores(results) == {"expect_lookup_rows": None}


def test_lookup_rows_needs_the_case_to_state_the_energy(measures_file):
    """find_measures reads the energy off the state, so the call records
    nothing about which slice was asked for. Only the case can say."""
    outputs = _outputs(messages=[_lookup_message(["box_breathing"])], measures_path=str(measures_file))

    results = check_expectations(outputs, {"lookup_rows": {"goal": "calm down now"}})

    assert _scores(results) == {"expect_lookup_rows": False}
    assert "must state the energy" in results[0]["comment"]


def test_an_unparseable_lookup_result_is_an_empty_one(measures_file):
    outputs = _outputs(
        messages=[{"role": "tool", "name": "find_measures", "content": "Energy is not recorded yet."}],
        measures_path=str(measures_file),
    )

    results = check_expectations(outputs, {"lookup_rows": {"energy": "exhausted", "goal": "calm down now"}})

    assert _scores(results) == {"expect_lookup_rows": False}


# -- the rows ----------------------------------------------------------------


def test_log_rows_matches_by_shape_and_ignores_order():
    """Two measures settled in one turn are written in whatever order the
    model names them, and that order is not a guarantee."""
    outputs = _outputs(
        reached_write=True,
        log_rows=[
            {"energy": "exhausted", "measure_id": "stand_up", "circumstances": "a long week"},
            {"energy": "exhausted", "measure_id": "box_breathing", "circumstances": "a long week"},
        ],
    )

    results = check_expectations(
        outputs,
        {"log_rows": {"count": 2, "rows": [{"measure_id": "box_breathing"}, {"measure_id": "stand_up"}]}},
    )

    assert _scores(results) == {"expect_log_rows": True}


def test_log_rows_does_not_match_one_row_twice():
    outputs = _outputs(reached_write=True, log_rows=[{"measure_id": "stand_up"}])

    results = check_expectations(
        outputs, {"log_rows": {"rows": [{"measure_id": "stand_up"}, {"measure_id": "stand_up"}]}}
    )

    assert _scores(results) == {"expect_log_rows": False}


def test_log_rows_fails_on_the_wrong_count():
    outputs = _outputs(reached_write=True, log_rows=[{"measure_id": "stand_up"}])

    results = check_expectations(outputs, {"log_rows": {"count": 2}})

    assert _scores(results) == {"expect_log_rows": False}
    assert "found 1" in results[0]["comment"]


def test_row_assertions_are_not_graded_when_no_write_happened():
    """"Never got there" and "wrote the wrong thing" have different fixes, and
    one red line for both hides which one happened."""
    results = check_expectations(
        _outputs(reached_write=False), {"log_rows": {"rows": [{"measure_id": "stand_up"}]}}
    )

    assert _scores(results) == {"expect_log_rows": None}


def test_reached_write_is_its_own_key():
    assert _scores(check_expectations(_outputs(reached_write=True), {"reached_write": True})) == {
        "expect_reached_write": True
    }
    assert _scores(check_expectations(_outputs(reached_write=False), {"reached_write": True})) == {
        "expect_reached_write": False
    }


def test_measures_added_takes_a_list_or_a_count():
    outputs = _outputs(measures_added=["cold_water"])

    assert _scores(check_expectations(outputs, {"measures_added": ["cold_water"]})) == {
        "expect_measures_added": True
    }
    assert _scores(check_expectations(outputs, {"measures_added": 1})) == {"expect_measures_added": True}
    assert _scores(check_expectations(outputs, {"measures_added": []})) == {"expect_measures_added": False}


# -- the card's sentence -----------------------------------------------------


def test_circumstances_reports_a_verdict_without_failing_the_case(monkeypatch):
    """Report-only until three runs have shown how often paraphrase trips it.
    A check that fails a case on a defect of its own gets switched off, and
    then it is protecting nothing."""
    outputs = _outputs(
        approvals=[{"call_id": "c1", "args": {"circumstances": "a fortnight of appraisals"}}]
    )

    results = check_expectations(outputs, {"circumstances": "empty_or_user_words"})

    assert _scores(results) == {"expect_circumstances": None}
    assert "would have failed" in results[0]["comment"]
    assert "fortnight" in results[0]["comment"]


def test_circumstances_can_fail_once_it_is_switched_on(monkeypatch):
    monkeypatch.setattr(expectations, "CIRCUMSTANCES_REPORT_ONLY", False)
    outputs = _outputs(
        approvals=[{"call_id": "c1", "args": {"circumstances": "a fortnight of appraisals"}}]
    )

    results = check_expectations(outputs, {"circumstances": "empty_or_user_words"})

    assert _scores(results) == {"expect_circumstances": False}


def test_circumstances_passes_on_the_users_own_words(monkeypatch):
    monkeypatch.setattr(expectations, "CIRCUMSTANCES_REPORT_ONLY", False)
    outputs = _outputs(
        approvals=[{"call_id": "c1", "args": {"circumstances": "a long week of meetings"}}]
    )

    assert _scores(check_expectations(outputs, {"circumstances": "empty_or_user_words"})) == {
        "expect_circumstances": True
    }


def test_circumstances_is_checked_on_the_card_not_the_row(monkeypatch):
    """An edit turn overwrites the row with the user's own wording, so a row
    can look faultless after the agent proposed something invented."""
    monkeypatch.setattr(expectations, "CIRCUMSTANCES_REPORT_ONLY", False)
    outputs = _outputs(
        approvals=[{"call_id": "c1", "args": {"circumstances": "a fortnight of appraisals"}}],
        log_rows=[{"circumstances": "a long week of meetings"}],
        reached_write=True,
    )

    assert _scores(check_expectations(outputs, {"circumstances": "empty_or_user_words"})) == {
        "expect_circumstances": False
    }


def test_an_empty_card_sentence_is_not_an_invention(monkeypatch):
    monkeypatch.setattr(expectations, "CIRCUMSTANCES_REPORT_ONLY", False)
    outputs = _outputs(approvals=[{"call_id": "c1", "args": {"circumstances": ""}}])

    assert _scores(check_expectations(outputs, {"circumstances": "empty_or_user_words"})) == {
        "expect_circumstances": True
    }


# -- what is still waiting ---------------------------------------------------


def test_pending_ratings_counts_without_caring_which_measures():
    """Which measures the agent settled on is its own judgement and moves run
    to run; that each one gets its own rating prompt does not."""
    outputs = _outputs(
        reached_write=True,
        log_rows=[
            {"row_id": "r1", "measure_id": "stand_up", "helpfulness": ""},
            {"row_id": "r2", "measure_id": "box_breathing", "helpfulness": ""},
        ],
    )

    assert _scores(check_expectations(outputs, {"pending_ratings_at_end": 2})) == {
        "expect_pending_ratings_at_end": True
    }


def test_pending_ratings_fails_when_one_measure_closed_the_batch():
    outputs = _outputs(
        reached_write=True,
        log_rows=[
            {"row_id": "r1", "measure_id": "stand_up", "helpfulness": ""},
            {"row_id": "r2", "measure_id": "box_breathing", "helpfulness": ""},
            {"row_id": "r3", "measure_id": "stand_up", "helpfulness": "helped", "circumstances": ""},
        ],
    )

    results = check_expectations(outputs, {"pending_ratings_at_end": 2})

    assert _scores(results) == {"expect_pending_ratings_at_end": False}
    assert "found 1" in results[0]["comment"]


# -- the same assertions against a real run ----------------------------------


def _real_run(tmp_path):
    """One turn through the graph the app runs: a lookup, then a write the
    harness approves. Everything the expect block reads is produced here by
    the real code rather than by a hand-written dict -- which is the point.
    A field can be perfectly graded and still read a key the harness spells
    differently, and only a run through both halves shows that."""
    from langchain_core.messages import AIMessage

    from evals.target import build_eval_graph, run_case
    from tests.conftest import StubChatModel

    log_path = tmp_path / "log.csv"
    measures_path = tmp_path / "measures.csv"
    with measures_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(MEASURE_COLUMNS)
        writer.writerow(("short_walk", "recover", "Walk for five minutes.", "wired", "minutes", "user"))
        writer.writerow(("breathe", "recover", "Take a few slow breaths.", "wired", "minutes", "user"))
        writer.writerow(("say_no", "act", "Say no to one request today.", "wired", "minutes", "agent"))
    model = StubChatModel(
        responses=[
            AIMessage(content="", tool_calls=[{"name": "find_measures", "args": {}, "id": "look-1"}]),
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "log_checkin",
                        "args": {
                            "energy": "wired",
                            "circumstances": "a loud day",
                            "measure_id": "short_walk",
                        },
                        "id": "write-1",
                    }
                ],
            ),
            AIMessage(content="Noted."),
        ]
    )
    graph = build_eval_graph(log_path, measures_path, model=model)
    return run_case(
        {
            "turns": [
                {
                    "text": "a loud day, I will take the walk",
                    "energy": "wired",
                    "goal": "calm down now",
                    "situation": "a loud day",
                    "step": "follow_up",
                }
            ]
        },
        agent_graph=graph,
        log_path=log_path,
        measures_path=measures_path,
    )


def test_every_field_reads_what_the_harness_really_returns(tmp_path, monkeypatch):
    monkeypatch.setattr(expectations, "CIRCUMSTANCES_REPORT_ONLY", False)
    outputs = _real_run(tmp_path)

    results = check_expectations(
        outputs,
        {
            "tools_called": ["find_measures", "log_checkin"],
            "tools_not_called": ["add_measure"],
            "lookup_rows": {"energy": "wired", "goal": "calm down now"},
            "log_rows": {"count": 1, "rows": [{"measure_id": "short_walk", "energy": "wired"}]},
            "circumstances": "empty_or_user_words",
            "pending_ratings_at_end": 1,
            "measures_added": [],
            "reached_write": True,
        },
    )

    assert _scores(results) == {
        "expect_tools_called": True,
        "expect_tools_not_called": True,
        "expect_lookup_rows": True,
        "expect_log_rows": True,
        "expect_circumstances": True,
        "expect_pending_ratings_at_end": True,
        "expect_measures_added": True,
        "expect_reached_write": True,
    }


def test_the_real_lookup_is_graded_against_the_cases_own_list(tmp_path):
    """The case's list is a temporary copy, not the shipped one. Grading
    against the wrong file would pass or fail on rows the run never saw."""
    outputs = _real_run(tmp_path)

    wrong_goal = check_expectations(outputs, {"lookup_rows": {"energy": "wired", "goal": "change the situation"}})

    assert _scores(wrong_goal) == {"expect_lookup_rows": False}


def test_a_real_write_matches_the_card_it_was_shown_on(tmp_path):
    """The call id has to survive the whole way: the model's call, the card
    the harness answers, and the receipt the graph writes."""
    from evals.evaluators import write_had_card

    outputs = _real_run(tmp_path)

    assert [a["call_id"] for a in outputs["approvals"]] == ["write-1"]
    assert write_had_card(outputs=outputs, reference_outputs={})["score"] is True


def test_the_lookup_check_refuses_to_guess_which_list_was_used():
    """matching_measures with no path reads the user's own measures.csv. A run
    graded against a file it never opened would pass or fail on rows that were
    never in play, and nothing in the report would say so."""
    outputs = _outputs(messages=[_lookup_message(["box_breathing"])])

    results = check_expectations(outputs, {"lookup_rows": {"energy": "exhausted"}})

    assert _scores(results) == {"expect_lookup_rows": False}
    assert "which measure list" in results[0]["comment"]


# -- reaches: which graph nodes the case actually ran ------------------------


def test_a_case_that_reached_the_node_it_names_passes():
    result = check_expectations({"nodes_run": ["ask", "tools", "suggest"]}, {"reaches": ["suggest"]})

    assert result == [{"key": "expect_reaches", "score": True}]


def test_a_case_that_never_reached_the_node_fails():
    """Nothing in the dataset reached `debate` or `close_idea` on 2026-09-26,
    while several cases named them. A covers line is a claim; this is the
    check."""
    result = check_expectations({"nodes_run": ["ask", "tools"]}, {"reaches": ["debate"]})

    assert result[0]["score"] is False
    assert "debate" in result[0]["comment"]


def test_passing_through_other_nodes_is_not_a_failure():
    """A subset check, like tools_called. A case about `debate` should not
    fail because the turn before it went through `suggest`."""
    result = check_expectations(
        {"nodes_run": ["ask", "suggest", "debate", "close_idea"]}, {"reaches": ["debate"]}
    )

    assert result[0]["score"] is True


def test_a_harness_that_recorded_no_path_scores_nothing():
    """A stub graph with no checkpointer answers nothing, and a case must not
    fail on how it was driven."""
    result = check_expectations({}, {"reaches": ["debate"]})

    assert result[0]["score"] is None


def test_a_known_field_given_the_wrong_shape_fails_loudly():
    """It used to raise out of check_expectations, and the runner caught it a
    level up: the case reported no assertions at all -- a dash in every
    column, reading as "nothing to check" rather than "this case is broken".
    Two cases sat like that until a report was read carefully."""
    result = check_expectations({"messages": []}, {"lookup_rows": "full_slice"})

    assert result[0]["key"] == "expect_schema"
    assert result[0]["score"] is False
    assert "lookup_rows" in result[0]["comment"]


def test_a_well_formed_block_is_untouched_by_the_guard():
    result = check_expectations({"reached_write": True}, {"reached_write": True})

    assert result == [{"key": "expect_reached_write", "score": True}]
