"""Tests for tools/log_query.py -- the deterministic history lookups.

No LLM anywhere in this module, so these are plain data tests against real
temp CSV files written through the real append_mood_row.
"""
from datetime import datetime, timedelta, timezone

import pytest

from tools.csv_log import append_mood_row
from tools.log_query import (
    analyze_stress_patterns,
    find_similar_entries,
    find_similar_entries_tool,
    known_tags,
)


@pytest.fixture
def log_path(tmp_path):
    return tmp_path / "mood_log.csv"


def _add(path, mood="bad", circumstances="something", tags=None, **kwargs):
    return append_mood_row(mood=mood, circumstances=circumstances, tags=tags, path=path, **kwargs)


# -- find_similar_entries ----------------------------------------------------


def test_returns_entries_sharing_a_tag(log_path):
    _add(log_path, circumstances="vague feedback", tags=["manager_feedback", "unclear_expectations"])
    _add(log_path, circumstances="unrelated", tags=["noisy_office"])

    results = find_similar_entries(["manager_feedback"], path=log_path)

    assert len(results) == 1
    assert results[0]["circumstances"] == "vague feedback"
    assert results[0]["matched_tags"] == ["manager_feedback"]


def test_empty_log_returns_empty_list_not_an_error(log_path):
    """The 'have I dealt with this before?' case with nothing on record --
    an empty result is a real, reportable answer, and the honesty rule
    depends on this being [] rather than an exception or a fabrication."""
    assert find_similar_entries(["anything"], path=log_path) == []


def test_no_overlap_returns_empty_list(log_path):
    _add(log_path, tags=["noisy_office"])

    assert find_similar_entries(["deadline_pressure"], path=log_path) == []


def test_empty_tag_list_returns_empty_without_scanning(log_path):
    _add(log_path, tags=["deadline_pressure"])

    assert find_similar_entries([], path=log_path) == []
    assert find_similar_entries(["", "   "], path=log_path) == []


def test_ranks_more_overlapping_tags_first(log_path):
    _add(log_path, circumstances="one match", tags=["deadline_pressure"])
    _add(log_path, circumstances="two matches", tags=["deadline_pressure", "context_switching"])

    results = find_similar_entries(["deadline_pressure", "context_switching"], path=log_path)

    assert [r["circumstances"] for r in results] == ["two matches", "one match"]


def test_respects_limit(log_path):
    for i in range(5):
        _add(log_path, circumstances=f"entry {i}", tags=["deadline_pressure"])

    assert len(find_similar_entries(["deadline_pressure"], limit=2, path=log_path)) == 2


def test_matching_is_case_insensitive(log_path):
    _add(log_path, tags=["Deadline_Pressure"])

    assert len(find_similar_entries(["deadline_pressure"], path=log_path)) == 1


def test_returns_what_was_tried_and_whether_it_helped(log_path):
    _add(
        log_path,
        circumstances="overloaded",
        tags=["deadline_pressure"],
        measure_taken="asked for an extension",
        helpfulness="helped a lot",
    )

    result = find_similar_entries(["deadline_pressure"], path=log_path)[0]

    assert result["measure_taken"] == "asked for an extension"
    assert result["helpfulness"] == "helped a lot"


def test_does_not_resurface_the_most_personal_free_text(log_path):
    """thoughts/need are the most personal fields in a row; a
    similar-situations lookup doesn't need them to answer 'what did I do
    about this last time.'"""
    _add(
        log_path,
        tags=["deadline_pressure"],
        thoughts="private rumination",
        need="private need",
    )

    result = find_similar_entries(["deadline_pressure"], path=log_path)[0]

    assert "thoughts" not in result
    assert "need" not in result


def test_missing_log_file_is_not_an_error(tmp_path):
    assert find_similar_entries(["x"], path=tmp_path / "does_not_exist.csv") == []


def test_tool_wrapper_is_registered_under_the_expected_name():
    """The LLM sees this name and eval trajectories record it, so it's part
    of the contract, not an implementation detail."""
    assert find_similar_entries_tool.name == "find_similar_entries"


def test_tool_wrapper_returns_the_same_rows(log_path, monkeypatch):
    import tools.log_query as log_query_module

    monkeypatch.setattr(log_query_module, "LOG_PATH", log_path)
    _add(log_path, circumstances="via the tool", tags=["deadline_pressure"])

    results = find_similar_entries_tool.invoke({"tags": ["deadline_pressure"]})

    assert [r["circumstances"] for r in results] == ["via the tool"]


# -- known_tags --------------------------------------------------------------


def test_known_tags_returns_distinct_tags_most_frequent_first(log_path):
    _add(log_path, tags=["high_workload", "overtime"])
    _add(log_path, tags=["high_workload"])
    _add(log_path, tags=["noisy_office"])

    assert known_tags(path=log_path) == ["high_workload", "noisy_office", "overtime"]


def test_known_tags_breaks_ties_alphabetically_for_a_stable_prompt(log_path):
    """The list goes into an LLM prompt, so a nondeterministic order would
    make classification results harder to reproduce than they need to be."""
    _add(log_path, tags=["zeta", "alpha", "mid"])

    assert known_tags(path=log_path) == ["alpha", "mid", "zeta"]


def test_known_tags_is_empty_for_a_fresh_log(log_path):
    assert known_tags(path=log_path) == []


def test_known_tags_respects_limit(log_path):
    _add(log_path, tags=["a", "b", "c", "d"])

    assert len(known_tags(limit=2, path=log_path)) == 2


def test_known_tags_ignores_untagged_rows(log_path):
    _add(log_path, tags=None)
    _add(log_path, tags=["real_tag"])

    assert known_tags(path=log_path) == ["real_tag"]


# -- analyze_stress_patterns -------------------------------------------------


def test_counts_tags_moods_and_energies_in_window(log_path):
    _add(log_path, mood="bad", energy="wired", tags=["deadline_pressure"])
    _add(log_path, mood="bad", energy="exhausted", tags=["deadline_pressure", "noisy_office"])

    summary = analyze_stress_patterns(window_days=7, path=log_path)

    assert summary["entry_count"] == 2
    assert summary["enough_data"] is True
    assert summary["top_tags"][0] == ("deadline_pressure", 2)
    assert summary["mood_counts"] == {"bad": 2}
    assert summary["energy_counts"] == {"wired": 1, "exhausted": 1}


def test_excludes_entries_older_than_the_window(log_path):
    _add(log_path, tags=["recent"])
    future_now = datetime.now(timezone.utc) + timedelta(days=30)

    summary = analyze_stress_patterns(window_days=7, path=log_path, now=future_now)

    assert summary["entry_count"] == 0
    assert summary["enough_data"] is False


def test_single_entry_is_not_enough_data_for_a_pattern(log_path):
    """N=1 must not be reported as a trend -- the digest has to say 'not
    enough data' rather than extrapolate from one row."""
    _add(log_path, tags=["deadline_pressure"])

    summary = analyze_stress_patterns(path=log_path)

    assert summary["entry_count"] == 1
    assert summary["enough_data"] is False


def test_empty_log_reports_no_data_rather_than_failing(log_path):
    summary = analyze_stress_patterns(path=log_path)

    assert summary["entry_count"] == 0
    assert summary["enough_data"] is False
    assert summary["top_tags"] == []


def test_collects_measures_that_were_tried(log_path):
    _add(log_path, tags=["x"], measure_taken="short walk", helpfulness="helped")
    _add(log_path, tags=["x"])

    summary = analyze_stress_patterns(path=log_path)

    assert summary["measures_taken"] == [{"measure_taken": "short walk", "helpfulness": "helped"}]
