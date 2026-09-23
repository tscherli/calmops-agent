"""Tests for tools/log_query.py -- the deterministic digest aggregation.

No LLM anywhere in this module, so these are plain data tests against real
temp CSV files written through the real append_checkin_row.

The guarantee that matters most here is the one the digest can violate
without anything failing: **telling the user something helped when they said
it did not.** The counts are what the opener's claims are checked against, so
a wrong count is a false statement about the user's own history, phrased
warmly.
"""
import csv
from datetime import datetime, timedelta, timezone

import pytest

from tools.checkin_log import append_checkin_row
from tools.log_query import analyze_stress_patterns
from tools.measures import MEASURE_COLUMNS

MEASURE_ROWS = [
    ("short_walk", "recover", "Walk for five minutes.", "exhausted;wired", "minutes", "user"),
    ("breathe", "recover", "Take a few slow breaths.", "exhausted;calm;wired", "minutes", "user"),
    ("day_off", "recover", "Take a day off.", "exhausted;calm;wired", "days", "user"),
]


@pytest.fixture
def log_path(tmp_path):
    return tmp_path / "checkin_log.csv"


@pytest.fixture
def measures_file(tmp_path):
    path = tmp_path / "measures.csv"
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(MEASURE_COLUMNS)
        writer.writerows(MEASURE_ROWS)
    return path


def checkin(path, measure_id="short_walk", energy="wired", **kwargs):
    return append_checkin_row(
        path=path, energy=energy, circumstances="deadlines", measure_id=measure_id, **kwargs
    )


def rating(path, measure_id="short_walk", helpfulness="helped"):
    """One occasion: the check-in it belongs to, then the verdict. The writer
    refuses the very same verdict twice with no check-in in between, which is
    how a model logged one event as two successes."""
    checkin(path, measure_id=measure_id)
    return append_checkin_row(path=path, measure_id=measure_id, helpfulness=helpfulness)


def summary(log_path, measures_file, **kwargs):
    return analyze_stress_patterns(path=log_path, measures_path=measures_file, **kwargs)


# -- the two kinds of row are counted separately -----------------------------


def test_a_rating_is_not_an_entry(log_path, measures_file):
    """Counting verdicts as entries would turn a quiet week with one measure
    and three ratings into a busy one."""
    rating(log_path)
    rating(log_path, helpfulness="somewhat")

    result = summary(log_path, measures_file)

    assert result["entry_count"] == 2
    assert result["rating_count"] == 2


def test_a_thin_week_is_reported_as_thin_even_with_many_ratings(log_path, measures_file):
    """N=1 must not become a trend because verdicts arrived about it."""
    rating(log_path)
    for _ in range(3):
        append_checkin_row(
            path=log_path, measure_id="short_walk", helpfulness="somewhat" if _ % 2 else "helped"
        )

    assert summary(log_path, measures_file)["enough_data"] is False


def test_two_check_ins_are_enough_for_a_pattern(log_path, measures_file):
    checkin(log_path)
    checkin(log_path, measure_id="breathe")

    assert summary(log_path, measures_file)["enough_data"] is True


# -- ids are resolved to something a person recognises -----------------------


def test_measures_are_reported_by_their_text_not_their_id(log_path, measures_file):
    """A digest saying "short_walk came up 3 times" is useless to read."""
    checkin(log_path)
    checkin(log_path)

    result = summary(log_path, measures_file)

    assert result["top_measures"] == [("Walk for five minutes.", 2)]


def test_an_id_missing_from_the_list_is_kept_under_its_id(log_path, measures_file):
    """Dropping it would make the digest's counts disagree with the file.
    The row is real; only its label is unavailable."""
    checkin(log_path, measure_id="")
    append_checkin_row(path=log_path, energy="wired", circumstances="x", measure_id="retired_one")

    result = summary(log_path, measures_file)

    assert ("retired_one", 1) in result["top_measures"]


def test_a_measure_in_the_users_own_words_is_named_by_that_text(log_path, measures_file):
    checkin(log_path, measure_id="", measure_taken="rearranged my desk")

    result = summary(log_path, measures_file)

    assert result["top_measures"] == [("rearranged my desk", 1)]


def test_most_frequent_first_with_ties_broken_alphabetically(log_path, measures_file):
    """The same week must not produce a differently ordered opener on a
    re-run."""
    checkin(log_path, measure_id="day_off")
    checkin(log_path, measure_id="breathe")
    checkin(log_path, measure_id="short_walk")
    checkin(log_path, measure_id="short_walk")

    result = summary(log_path, measures_file)

    assert result["top_measures"] == [
        ("Walk for five minutes.", 2),
        ("Take a day off.", 1),
        ("Take a few slow breaths.", 1),
    ]


# -- what the user said worked, and what they did not ------------------------


def test_helpfulness_is_counted_per_measure(log_path, measures_file):
    checkin(log_path)
    checkin(log_path, measure_id="day_off")
    rating(log_path, "short_walk", "helped")
    rating(log_path, "short_walk", "helped")
    rating(log_path, "day_off", "didnt_help")

    result = summary(log_path, measures_file)

    assert result["helpfulness_counts"]["Walk for five minutes."] == {"helped": 2}
    assert result["helpfulness_counts"]["Take a day off."] == {"didnt_help": 1}


def test_a_measure_rated_as_no_help_is_never_called_best(log_path, measures_file):
    """The failure this whole module exists to prevent: a warm sentence
    telling the user a thing helped, when they said it did not."""
    checkin(log_path, measure_id="day_off")
    rating(log_path, "day_off", "didnt_help")

    assert summary(log_path, measures_file)["best_measures"] == []


def test_somewhat_alone_is_not_a_success(log_path, measures_file):
    """"Somewhat" is what people say when it was not really the thing.
    Counting it as a win is how a digest starts recommending near-misses."""
    checkin(log_path)
    rating(log_path, "short_walk", "somewhat")

    assert summary(log_path, measures_file)["best_measures"] == []


def test_a_measure_that_failed_more_often_than_it_helped_is_not_best(log_path, measures_file):
    """The case a "did it ever help?" rule gets wrong. One good day out of
    four does not make a measure worth opening the week with -- and this is
    the only test that separates "helped at all" from "helped more often
    than not"."""
    checkin(log_path, measure_id="day_off")
    rating(log_path, "day_off", "helped")
    for _ in range(3):
        rating(log_path, "day_off", "didnt_help")

    assert summary(log_path, measures_file)["best_measures"] == []


def test_mixed_verdicts_go_by_the_majority(log_path, measures_file):
    checkin(log_path)
    rating(log_path, "short_walk", "helped")
    rating(log_path, "short_walk", "helped")
    rating(log_path, "short_walk", "didnt_help")

    assert summary(log_path, measures_file)["best_measures"] == ["Walk for five minutes."]


def test_best_measures_are_ordered_by_how_often_they_helped(log_path, measures_file):
    checkin(log_path)
    checkin(log_path, measure_id="breathe")
    rating(log_path, "breathe", "helped")
    for _ in range(3):
        rating(log_path, "short_walk", "helped")

    result = summary(log_path, measures_file)

    assert result["best_measures"] == ["Walk for five minutes.", "Take a few slow breaths."]


def test_an_unrated_measure_is_not_claimed_to_have_worked(log_path, measures_file):
    checkin(log_path)
    checkin(log_path, measure_id="breathe")

    assert summary(log_path, measures_file)["best_measures"] == []


# -- the window --------------------------------------------------------------


def test_energy_is_counted_from_check_ins_only(log_path, measures_file):
    checkin(log_path, energy="wired")
    checkin(log_path, energy="exhausted")
    # a rating blanks energy, so it must not show up as "" -- written directly,
    # because the helper would add a check-in of its own
    append_checkin_row(path=log_path, measure_id="short_walk", helpfulness="helped")

    result = summary(log_path, measures_file)

    assert result["energy_counts"] == {"wired": 1, "exhausted": 1}


def test_rows_older_than_the_window_are_excluded(log_path, measures_file):
    checkin(log_path)
    future_now = datetime.now(timezone.utc) + timedelta(days=30)

    result = summary(log_path, measures_file, window_days=7, now=future_now)

    assert result["entry_count"] == 0
    assert result["top_measures"] == []


def test_an_empty_log_reports_no_data_rather_than_failing(log_path, measures_file):
    result = summary(log_path, measures_file)

    assert result["entry_count"] == 0
    assert result["enough_data"] is False
    assert result["best_measures"] == []


def test_a_missing_log_reads_as_no_data_rather_than_raising(tmp_path, measures_file):
    """A digest that crashes would take the scheduler's window with it."""
    result = analyze_stress_patterns(
        path=tmp_path / "does_not_exist.csv", measures_path=measures_file
    )

    assert result["entry_count"] == 0
