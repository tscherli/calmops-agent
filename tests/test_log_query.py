"""Tests for tools/log_query.py -- the deterministic digest aggregation.

No LLM anywhere in this module, so these are plain data tests against real
temp CSV files written through the real append_checkin_row.

The digest still counts the columns the previous schema had, so against a
check-in log its tag and mood tallies come back empty. That is asserted
below rather than left to be discovered: it is a known, temporary gap, and a
test that quietly stopped covering it would hide the day it stops being
temporary.
"""
from datetime import datetime, timedelta, timezone

import pytest

from tools.checkin_log import append_checkin_row
from tools.log_query import analyze_stress_patterns


@pytest.fixture
def log_path(tmp_path):
    return tmp_path / "checkin_log.csv"


def _add(path, **kwargs):
    kwargs.setdefault("energy", "wired")
    kwargs.setdefault("circumstances", "something")
    kwargs.setdefault("measure_id", "m01")
    return append_checkin_row(path=path, **kwargs)


def test_counts_entries_and_energies_in_window(log_path):
    _add(log_path, energy="wired")
    _add(log_path, energy="exhausted")

    summary = analyze_stress_patterns(window_days=7, path=log_path)

    assert summary["entry_count"] == 2
    assert summary["enough_data"] is True
    assert summary["energy_counts"] == {"wired": 1, "exhausted": 1}


def test_tag_and_mood_tallies_are_empty_on_the_check_in_schema(log_path):
    """Neither column exists any more. The digest must report nothing rather
    than invent something -- and the emptiness is what says this half of the
    summary is still waiting to be rebuilt on measures."""
    _add(log_path)
    _add(log_path)

    summary = analyze_stress_patterns(path=log_path)

    assert summary["top_tags"] == []
    assert summary["mood_counts"] == {}


def test_excludes_entries_older_than_the_window(log_path):
    _add(log_path)
    future_now = datetime.now(timezone.utc) + timedelta(days=30)

    summary = analyze_stress_patterns(window_days=7, path=log_path, now=future_now)

    assert summary["entry_count"] == 0
    assert summary["enough_data"] is False


def test_single_entry_is_not_enough_data_for_a_pattern(log_path):
    """N=1 must not be reported as a trend -- the digest has to say 'not
    enough data' rather than extrapolate from one row."""
    _add(log_path)

    summary = analyze_stress_patterns(path=log_path)

    assert summary["entry_count"] == 1
    assert summary["enough_data"] is False


def test_empty_log_reports_no_data_rather_than_failing(log_path):
    summary = analyze_stress_patterns(path=log_path)

    assert summary["entry_count"] == 0
    assert summary["enough_data"] is False
    assert summary["top_tags"] == []


def test_collects_measures_that_were_tried(log_path):
    _add(log_path, measure_id="", measure_taken="short walk")
    _add(log_path)

    summary = analyze_stress_patterns(path=log_path)

    assert {"measure_taken": "short walk", "helpfulness": ""} in summary["measures_taken"]


def test_a_missing_log_reads_as_no_data_rather_than_raising(tmp_path):
    """A digest that crashes would take the scheduler's window with it."""
    summary = analyze_stress_patterns(path=tmp_path / "does_not_exist.csv")

    assert summary["entry_count"] == 0
