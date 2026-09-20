"""Schedule configuration + next-fire-time computation for scheduler.py.

Hand-rolled JSON and plain datetime arithmetic, no croniter/APScheduler
dependency -- the scheduling this app needs ("weekdays at 10:00 and 16:00,
plus a digest Friday afternoon") doesn't justify a cron parser, and adding
one would be a new dependency for a personal tool.

`compute_next_fire` is pure and takes an injected `now`, per this project's
DI-for-testability convention (tools/checkin_log.py's `path`, agent/graph.py's
`model`/`tools`) -- the whole scheduling rule set is unit-testable without
freezing the clock or sleeping.

Two check-in modes are supported, not one:
- "times_of_day": fixed clock times on chosen weekdays (the new default).
- "random_interval": the original stage-8 behavior, retained rather than
  deleted -- it's genuinely better for anyone who finds fixed times easy to
  start ignoring.
"""
from __future__ import annotations

import json
import random
from dataclasses import dataclass, field
from datetime import datetime, time, timedelta
from pathlib import Path
from typing import Any, Literal

BASE_DIR = Path(__file__).resolve().parent
DEFAULT_CONFIG_PATH = BASE_DIR / "scheduler_config.json"

WEEKDAYS = ["mon", "tue", "wed", "thu", "fri", "sat", "sun"]

FireType = Literal["checkin", "digest"]


@dataclass(frozen=True)
class Fire:
    """One scheduled event: when it happens and which kind it is."""

    when: datetime
    kind: FireType


@dataclass
class SchedulerConfig:
    checkin_mode: Literal["times_of_day", "random_interval"] = "times_of_day"
    checkin_days: list[str] = field(default_factory=lambda: ["mon", "tue", "wed", "thu", "fri"])
    checkin_times: list[str] = field(default_factory=lambda: ["10:00", "16:00"])
    interval_min_minutes: float = 45.0
    interval_max_minutes: float = 90.0
    digest_enabled: bool = True
    digest_day: str = "fri"
    digest_time: str = "16:30"
    digest_window_days: int = 7


DEFAULT_CONFIG = SchedulerConfig()


def _parse_time(raw: str) -> time:
    hour_str, _, minute_str = raw.strip().partition(":")
    return time(hour=int(hour_str), minute=int(minute_str or 0))


def _weekday_index(name: str) -> int:
    try:
        return WEEKDAYS.index(name.strip().lower()[:3])
    except ValueError as exc:
        raise ValueError(f"Unknown weekday {name!r}; expected one of {WEEKDAYS}") from exc


def load_config(path: Path | None = None) -> SchedulerConfig:
    """Read the JSON config. A missing file is not an error -- it just means
    the defaults, so the app runs out of the box without one."""
    config_path = path or DEFAULT_CONFIG_PATH
    if not config_path.exists():
        return SchedulerConfig()
    raw: dict[str, Any] = json.loads(config_path.read_text(encoding="utf-8"))
    return parse_config(raw)


def parse_config(raw: dict[str, Any]) -> SchedulerConfig:
    checkins = raw.get("checkins") or {}
    digest = raw.get("digest") or {}
    defaults = SchedulerConfig()
    return SchedulerConfig(
        checkin_mode=checkins.get("mode", defaults.checkin_mode),
        checkin_days=list(checkins.get("days", defaults.checkin_days)),
        checkin_times=list(checkins.get("times", defaults.checkin_times)),
        interval_min_minutes=float(checkins.get("interval_min_minutes", defaults.interval_min_minutes)),
        interval_max_minutes=float(checkins.get("interval_max_minutes", defaults.interval_max_minutes)),
        digest_enabled=bool(digest.get("enabled", defaults.digest_enabled)),
        digest_day=digest.get("day", defaults.digest_day),
        digest_time=digest.get("time", defaults.digest_time),
        digest_window_days=int(digest.get("window_days", defaults.digest_window_days)),
    )


def _next_weekly_occurrence(now: datetime, day: str, at: time) -> datetime:
    """Soonest datetime strictly after `now` falling on `day` at `at`."""
    target_index = _weekday_index(day)
    days_ahead = (target_index - now.weekday()) % 7
    candidate = datetime.combine(now.date() + timedelta(days=days_ahead), at, tzinfo=now.tzinfo)
    if candidate <= now:
        candidate += timedelta(days=7)
    return candidate


def _next_checkin(now: datetime, config: SchedulerConfig, rng: random.Random | None = None) -> datetime:
    if config.checkin_mode == "random_interval":
        generator = rng or random
        wait = generator.uniform(config.interval_min_minutes, config.interval_max_minutes)
        return now + timedelta(minutes=wait)

    candidates = [
        _next_weekly_occurrence(now, day, _parse_time(slot))
        for day in config.checkin_days
        for slot in config.checkin_times
    ]
    if not candidates:
        # No check-in slots configured at all -- fall back to a day out rather
        # than raising, so a half-filled config can't hard-crash the scheduler.
        return now + timedelta(days=1)
    return min(candidates)


def compute_next_fire(
    now: datetime,
    config: SchedulerConfig | None = None,
    *,
    rng: random.Random | None = None,
) -> Fire:
    """Next scheduled event after `now`, check-in or digest, whichever is sooner.

    Pure: no sleeping, no clock reads, no I/O. A digest that ties with a
    check-in wins, since the digest opener also functions as a check-in and
    firing two windows at the same minute would just be noise.
    """
    config = config or SchedulerConfig()
    next_checkin = _next_checkin(now, config, rng=rng)

    if not config.digest_enabled:
        return Fire(when=next_checkin, kind="checkin")

    next_digest = _next_weekly_occurrence(now, config.digest_day, _parse_time(config.digest_time))
    if next_digest <= next_checkin:
        return Fire(when=next_digest, kind="digest")
    return Fire(when=next_checkin, kind="checkin")
