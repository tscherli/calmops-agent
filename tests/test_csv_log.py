import csv
import json
import uuid

import pytest

import tools.csv_log as csv_log_module
from tools.csv_log import LOG_COLUMNS, append_mood_row, log_mood


@pytest.fixture(autouse=True)
def isolated_paths(tmp_path, monkeypatch):
    """Redirect both the log file and the fallback file into a temp dir so
    tests never touch the real data/mood_log.csv or state/pending_sheet_log.jsonl."""
    log_path = tmp_path / "mood_log.csv"
    pending_path = tmp_path / "pending_sheet_log.jsonl"
    monkeypatch.setattr(csv_log_module, "LOG_PATH", log_path)
    monkeypatch.setattr(csv_log_module, "PENDING_LOG_PATH", pending_path)
    return {"log_path": log_path, "pending_path": pending_path}


def _read_rows(path):
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def test_append_mood_row_creates_file_with_header_and_row(isolated_paths):
    result = append_mood_row(valence="good", circumstances="quiet morning")

    assert result["status"] == "ok"
    assert uuid.UUID(result["row_id"])  # valid UUID

    rows = _read_rows(isolated_paths["log_path"])
    assert len(rows) == 1
    assert rows[0]["row_id"] == result["row_id"]
    assert rows[0]["valence"] == "good"
    assert rows[0]["circumstances"] == "quiet morning"


def test_append_mood_row_appends_without_duplicating_header(isolated_paths):
    append_mood_row(valence="good", circumstances="entry one")
    append_mood_row(valence="bad", circumstances="entry two")

    rows = _read_rows(isolated_paths["log_path"])
    assert len(rows) == 2
    assert rows[0]["circumstances"] == "entry one"
    assert rows[1]["circumstances"] == "entry two"

    raw_lines = isolated_paths["log_path"].read_text(encoding="utf-8").splitlines()
    assert raw_lines[0] == ",".join(LOG_COLUMNS)
    assert raw_lines.count(",".join(LOG_COLUMNS)) == 1  # header written exactly once


def test_field_with_comma_is_correctly_quoted_and_parsed_back(isolated_paths):
    append_mood_row(valence="bad", circumstances="stressed, tired, and behind on tickets")

    rows = _read_rows(isolated_paths["log_path"])
    assert rows[0]["circumstances"] == "stressed, tired, and behind on tickets"


def test_append_mood_row_write_failure_writes_fallback_and_does_not_raise(isolated_paths, monkeypatch):
    def raise_error(*args, **kwargs):
        raise OSError("simulated disk failure")

    monkeypatch.setattr(csv_log_module.csv, "DictWriter", raise_error)

    result = append_mood_row(valence="bad", circumstances="testing failure path")

    assert result["status"] == "error"
    assert "simulated disk failure" in result["error"]

    pending_path = isolated_paths["pending_path"]
    assert pending_path.exists()
    saved_row = json.loads(pending_path.read_text(encoding="utf-8").splitlines()[0])
    assert saved_row["row_id"] == result["row_id"]


def test_structuring_fields_are_packed_into_note(isolated_paths):
    append_mood_row(
        valence="bad",
        circumstances="overloaded",
        situation="six open tickets",
        thoughts="I can't prioritize",
        need="a clear next step",
        next_step="close one ticket before opening another",
    )

    rows = _read_rows(isolated_paths["log_path"])
    note = rows[0]["note"]
    assert "situation: six open tickets" in note
    assert "thoughts: I can't prioritize" in note
    assert "need: a clear next step" in note
    assert "next_step: close one ticket before opening another" in note


def test_light_check_in_has_empty_note(isolated_paths):
    append_mood_row(valence="good", circumstances="quiet day")

    rows = _read_rows(isolated_paths["log_path"])
    assert rows[0]["note"] == ""


def test_energy_level_is_written_when_provided(isolated_paths):
    append_mood_row(valence="bad", circumstances="exhausted after back-to-back meetings", energy_level="low")

    rows = _read_rows(isolated_paths["log_path"])
    assert rows[0]["energy_level"] == "low"


def test_energy_level_defaults_to_empty_when_not_mentioned(isolated_paths):
    append_mood_row(valence="good", circumstances="quiet day")

    rows = _read_rows(isolated_paths["log_path"])
    assert rows[0]["energy_level"] == ""


def test_log_mood_tool_writes_a_real_row(isolated_paths):
    result = log_mood.invoke({"valence": "good", "circumstances": "trying the tool wiring end to end"})

    assert result["status"] == "ok"
    rows = _read_rows(isolated_paths["log_path"])
    assert len(rows) == 1
    assert rows[0]["circumstances"] == "trying the tool wiring end to end"


def test_log_mood_is_a_langchain_tool_with_expected_name():
    assert log_mood.name == "log_mood"
