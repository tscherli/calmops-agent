"""agent/data_paths.py -- the user's files, wherever the user wants them."""
from pathlib import Path

import pytest

import observability.tracing as tracing
import tools.checkin_log as checkin_log
import tools.measures as measures
from agent.data_paths import DATA_DIR_ENV, DEFAULT_CONSENT_PATH, data_paths, prepare


@pytest.fixture(autouse=True)
def no_data_dir_set(monkeypatch):
    """A developer's own setting must not decide what these tests see."""
    monkeypatch.delenv(DATA_DIR_ENV, raising=False)


def test_by_default_nothing_moves():
    paths = data_paths()

    assert paths.checkin_log == checkin_log.CHECKIN_LOG_PATH
    assert paths.measures == measures.MEASURES_PATH
    assert paths.pending_log == checkin_log.PENDING_LOG_PATH
    assert paths.consent == DEFAULT_CONSENT_PATH
    assert paths.runs_log == tracing.RUNS_LOG_PATH


def test_the_env_var_moves_every_file_into_one_folder(tmp_path, monkeypatch):
    monkeypatch.setenv(DATA_DIR_ENV, str(tmp_path / "mine"))

    paths = data_paths()

    root = (tmp_path / "mine").resolve()
    assert paths.checkin_log == root / "checkin_log.csv"
    assert paths.measures == root / "measures.csv"
    assert paths.pending_log == root / "state" / "pending_sheet_log.jsonl"
    assert paths.consent == root / "state" / "consent_ack.json"
    assert paths.runs_log == root / "runs.jsonl"


def test_the_flag_wins_over_the_env_var(tmp_path, monkeypatch):
    monkeypatch.setenv(DATA_DIR_ENV, str(tmp_path / "from_env"))

    assert data_paths(tmp_path / "from_flag").checkin_log.parent == (tmp_path / "from_flag").resolve()


def test_an_empty_env_var_is_no_setting(monkeypatch):
    """`CALMOPS_DATA_DIR=` left blank in .env must not mean "the current folder"."""
    monkeypatch.setenv(DATA_DIR_ENV, "  ")

    assert data_paths().checkin_log == checkin_log.CHECKIN_LOG_PATH


def test_home_and_relative_folders_become_absolute(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)

    assert data_paths("rel").measures == tmp_path.resolve() / "rel" / "measures.csv"
    assert data_paths("~/jla").measures == Path("~/jla").expanduser().resolve() / "measures.csv"


def test_a_new_folder_gets_the_shipped_measure_list(tmp_path, monkeypatch):
    monkeypatch.setattr(checkin_log, "PENDING_LOG_PATH", checkin_log.PENDING_LOG_PATH)
    paths = prepare(data_paths(tmp_path / "new"))

    assert paths.measures.read_bytes() == measures.MEASURES_PATH.read_bytes()


def test_the_users_own_list_is_never_overwritten(tmp_path, monkeypatch):
    monkeypatch.setattr(checkin_log, "PENDING_LOG_PATH", checkin_log.PENDING_LOG_PATH)
    paths = data_paths(tmp_path / "old")
    paths.measures.parent.mkdir(parents=True)
    paths.measures.write_text("measure_id,type\nmine,recover\n", encoding="utf-8")

    prepare(paths)

    assert paths.measures.read_text(encoding="utf-8") == "measure_id,type\nmine,recover\n"


def test_a_failed_write_falls_back_into_the_chosen_folder(tmp_path, monkeypatch):
    """The fallback is a module global that ignores the log path a caller
    passes; left alone, a failed write would land next to the code."""
    monkeypatch.setattr(checkin_log, "PENDING_LOG_PATH", checkin_log.PENDING_LOG_PATH)
    paths = prepare(data_paths(tmp_path / "data"))

    assert checkin_log.PENDING_LOG_PATH == paths.pending_log
