"""Tests for scheduler_config.py -- next-fire-time computation.

compute_next_fire is pure and takes an injected `now`, so the whole rule set
is testable without sleeping or freezing the clock (this project's
DI-for-testability convention).
"""
import json
import random
from datetime import datetime

import pytest

from scheduler_config import (
    SchedulerConfig,
    compute_next_fire,
    load_config,
    parse_config,
)

# 2026-09-07 is a Monday; every case below is anchored to that week.
MONDAY_08_00 = datetime(2026, 9, 7, 8, 0)
MONDAY_12_00 = datetime(2026, 9, 7, 12, 0)
FRIDAY_18_00 = datetime(2026, 9, 11, 18, 0)


def _config(**overrides) -> SchedulerConfig:
    base = SchedulerConfig(digest_enabled=False)
    for key, value in overrides.items():
        setattr(base, key, value)
    return base


def test_picks_the_next_time_slot_later_the_same_day():
    fire = compute_next_fire(MONDAY_08_00, _config(checkin_times=["10:00", "16:00"]))

    assert fire.kind == "checkin"
    assert fire.when == datetime(2026, 9, 7, 10, 0)


def test_rolls_forward_to_the_next_slot_when_one_has_passed():
    fire = compute_next_fire(MONDAY_12_00, _config(checkin_times=["10:00", "16:00"]))

    assert fire.when == datetime(2026, 9, 7, 16, 0)


def test_rolls_to_the_next_configured_day_when_today_is_done():
    fire = compute_next_fire(
        datetime(2026, 9, 7, 20, 0), _config(checkin_days=["mon", "wed"], checkin_times=["10:00"])
    )

    assert fire.when == datetime(2026, 9, 9, 10, 0)  # Wednesday


def test_skips_days_not_in_the_schedule():
    """Saturday evening with a weekdays-only schedule lands on Monday."""
    fire = compute_next_fire(
        datetime(2026, 9, 12, 20, 0),
        _config(checkin_days=["mon", "tue", "wed", "thu", "fri"], checkin_times=["10:00"]),
    )

    assert fire.when == datetime(2026, 9, 14, 10, 0)  # the following Monday


def test_a_slot_exactly_now_is_treated_as_passed():
    """Otherwise the scheduler would compute a zero-second sleep and fire the
    same slot repeatedly in a tight loop."""
    fire = compute_next_fire(
        datetime(2026, 9, 7, 10, 0), _config(checkin_days=["mon"], checkin_times=["10:00"])
    )

    assert fire.when == datetime(2026, 9, 14, 10, 0)


def test_random_interval_mode_is_still_supported():
    config = _config(
        checkin_mode="random_interval", interval_min_minutes=45, interval_max_minutes=90
    )

    fire = compute_next_fire(MONDAY_08_00, config, rng=random.Random(0))

    delta_minutes = (fire.when - MONDAY_08_00).total_seconds() / 60
    assert 45 <= delta_minutes <= 90
    assert fire.kind == "checkin"


def test_empty_schedule_falls_back_instead_of_crashing():
    fire = compute_next_fire(MONDAY_08_00, _config(checkin_days=[], checkin_times=[]))

    assert fire.when == datetime(2026, 9, 8, 8, 0)


# -- digest ------------------------------------------------------------------


def test_digest_wins_when_it_comes_first():
    config = SchedulerConfig(
        checkin_days=["mon"],
        checkin_times=["10:00"],
        digest_enabled=True,
        digest_day="fri",
        digest_time="16:30",
    )

    fire = compute_next_fire(datetime(2026, 9, 11, 9, 0), config)  # Friday morning

    assert fire.kind == "digest"
    assert fire.when == datetime(2026, 9, 11, 16, 30)


def test_checkin_wins_when_it_comes_first():
    config = SchedulerConfig(
        checkin_days=["mon"], checkin_times=["10:00"], digest_enabled=True, digest_day="fri"
    )

    fire = compute_next_fire(MONDAY_08_00, config)

    assert fire.kind == "checkin"
    assert fire.when == datetime(2026, 9, 7, 10, 0)


def test_digest_wins_a_tie_so_two_windows_dont_open_at_once():
    config = SchedulerConfig(
        checkin_days=["fri"],
        checkin_times=["16:30"],
        digest_enabled=True,
        digest_day="fri",
        digest_time="16:30",
    )

    fire = compute_next_fire(datetime(2026, 9, 11, 9, 0), config)

    assert fire.kind == "digest"


def test_disabled_digest_never_fires():
    config = SchedulerConfig(
        checkin_days=["mon"], checkin_times=["10:00"], digest_enabled=False, digest_day="fri"
    )

    assert compute_next_fire(FRIDAY_18_00, config).kind == "checkin"


# -- config loading ----------------------------------------------------------


def test_missing_config_file_yields_defaults_not_an_error(tmp_path):
    config = load_config(tmp_path / "no_such_config.json")

    assert config == SchedulerConfig()


def test_config_file_is_parsed(tmp_path):
    path = tmp_path / "scheduler_config.json"
    path.write_text(
        json.dumps(
            {
                "checkins": {"mode": "times_of_day", "days": ["tue"], "times": ["09:15"]},
                "digest": {"enabled": False, "day": "wed", "time": "12:00", "window_days": 14},
            }
        ),
        encoding="utf-8",
    )

    config = load_config(path)

    assert config.checkin_days == ["tue"]
    assert config.checkin_times == ["09:15"]
    assert config.digest_enabled is False
    assert config.digest_window_days == 14


def test_partial_config_keeps_defaults_for_everything_else():
    config = parse_config({"digest": {"enabled": False}})

    assert config.digest_enabled is False
    assert config.checkin_times == SchedulerConfig().checkin_times


def test_shipped_config_file_is_valid():
    """The checked-in scheduler_config.json has to actually parse, or the app
    ships broken out of the box."""
    config = load_config()

    assert config.checkin_days
    assert config.checkin_times
    assert compute_next_fire(MONDAY_08_00, config).when > MONDAY_08_00


def test_unknown_weekday_is_rejected_clearly():
    with pytest.raises(ValueError, match="Unknown weekday"):
        compute_next_fire(MONDAY_08_00, _config(checkin_days=["funday"]))
