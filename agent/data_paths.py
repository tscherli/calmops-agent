"""Where the user's own files live, so the code can sit anywhere.

The check-in log, the measure list, the consent record, the failed-write
fallback and the per-turn records. By default they stay where they have
always been, next to the code: nothing moves unless asked. CALMOPS_DATA_DIR,
or --data-dir on app.py, cli.py and scheduler.py, puts all of them into one
folder of the user's choosing -- whichever directory the code was cloned
into, and whatever the operating system.

Resolved by the entry points and handed down, the way every path in this
project is injected rather than looked up deep inside a tool.
"""
from __future__ import annotations

import os
import shutil
from dataclasses import dataclass
from pathlib import Path

import observability.tracing as tracing
import tools.checkin_log as checkin_log
import tools.measures as measures

DATA_DIR_ENV = "CALMOPS_DATA_DIR"

# Where the consent screen's answer is kept by default. app.py and cli.py
# both take theirs from here; cli.py used to derive its own copy, to avoid
# importing Tk from app.py.
DEFAULT_CONSENT_PATH = Path(__file__).resolve().parent.parent / "state" / "consent_ack.json"


@dataclass(frozen=True)
class DataPaths:
    """Every file of the user's that the app reads or writes."""

    checkin_log: Path
    measures: Path
    pending_log: Path
    consent: Path
    runs_log: Path


def data_paths(data_dir: Path | str | None = None) -> DataPaths:
    """The folder given, else CALMOPS_DATA_DIR, else where the files have always been.

    The defaults are read when this runs, not when it is imported, so a test
    that points one of them elsewhere is followed here too.
    """
    chosen = data_dir or os.environ.get(DATA_DIR_ENV, "").strip()
    if not chosen:
        return DataPaths(
            checkin_log=checkin_log.CHECKIN_LOG_PATH,
            measures=measures.MEASURES_PATH,
            pending_log=checkin_log.PENDING_LOG_PATH,
            consent=DEFAULT_CONSENT_PATH,
            runs_log=tracing.RUNS_LOG_PATH,
        )
    root = Path(chosen).expanduser().resolve()
    return DataPaths(
        checkin_log=root / "checkin_log.csv",
        measures=root / "measures.csv",
        pending_log=root / "state" / "pending_sheet_log.jsonl",
        consent=root / "state" / "consent_ack.json",
        runs_log=root / "runs.jsonl",
    )


def prepare(paths: DataPaths) -> DataPaths:
    """Make a chosen folder ready to use.

    The measure list is seeded from the one shipped with the code the first
    time, and never overwritten after: from then on it is the user's. The
    failed-write fallback is a module global that ignores the log path a
    caller passes (tools/checkin_log.py), so it is pointed here -- the same
    redirect the eval harness makes for its temp folders.

    Called by the entry points only, never from a constructor: the tests point
    that global at a temp file, and a window or session resetting it would
    send a test's failed write into the real state folder.
    """
    paths.measures.parent.mkdir(parents=True, exist_ok=True)
    if not paths.measures.exists():
        shutil.copyfile(measures.MEASURES_PATH, paths.measures)
    checkin_log.PENDING_LOG_PATH = paths.pending_log
    return paths
