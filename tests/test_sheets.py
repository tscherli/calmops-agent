import json
import uuid
from unittest.mock import MagicMock

import pytest

import tools.sheets as sheets_module
from tools.sheets import SHEET_COLUMNS, append_mood_row, log_mood


@pytest.fixture(autouse=True)
def isolated_pending_log(tmp_path, monkeypatch):
    """Redirect the fallback file into a temp dir so tests never touch the
    real state/pending_sheet_log.jsonl, and never depend on a real
    MOOD_SHEET_ID being set in the environment."""
    monkeypatch.setattr(sheets_module, "PENDING_LOG_PATH", tmp_path / "pending_sheet_log.jsonl")
    monkeypatch.delenv("MOOD_SHEET_ID", raising=False)
    return tmp_path


def _mock_service():
    """A fake Google Sheets API client shaped like the real one: chained
    .spreadsheets().values().append(...).execute() calls."""
    service = MagicMock()
    execute_mock = service.spreadsheets.return_value.values.return_value.append.return_value.execute
    execute_mock.return_value = {"updates": {"updatedRows": 1}}
    return service


def test_append_mood_row_success_calls_sheets_api_with_correct_args():
    service = _mock_service()

    result = append_mood_row(
        valence="good",
        circumstances="quiet morning",
        service=service,
        sheet_id="fake-sheet-id",
    )

    assert result["status"] == "ok"
    assert uuid.UUID(result["row_id"])  # valid UUID

    append_call = service.spreadsheets.return_value.values.return_value.append
    assert append_call.call_args.kwargs["spreadsheetId"] == "fake-sheet-id"
    assert append_call.call_args.kwargs["range"] == "Log!A1"

    written_row = append_call.call_args.kwargs["body"]["values"][0]
    assert len(written_row) == len(SHEET_COLUMNS)
    # row_id column matches what was returned
    assert written_row[SHEET_COLUMNS.index("row_id")] == result["row_id"]
    assert written_row[SHEET_COLUMNS.index("valence")] == "good"
    assert written_row[SHEET_COLUMNS.index("circumstances")] == "quiet morning"


def test_append_mood_row_missing_sheet_id_writes_fallback(isolated_pending_log):
    result = append_mood_row(valence="bad", circumstances="rough call")

    assert result["status"] == "error"
    assert "MOOD_SHEET_ID" in result["error"]

    pending_file = isolated_pending_log / "pending_sheet_log.jsonl"
    assert pending_file.exists()
    saved_row = json.loads(pending_file.read_text().splitlines()[0])
    assert saved_row["row_id"] == result["row_id"]
    assert saved_row["valence"] == "bad"


def test_append_mood_row_api_failure_writes_fallback_and_does_not_raise(isolated_pending_log):
    service = MagicMock()
    service.spreadsheets.return_value.values.return_value.append.return_value.execute.side_effect = (
        Exception("simulated network failure")
    )

    result = append_mood_row(
        valence="neutral",
        circumstances="testing failure path",
        service=service,
        sheet_id="fake-sheet-id",
    )

    assert result["status"] == "error"
    assert "simulated network failure" in result["error"]

    pending_file = isolated_pending_log / "pending_sheet_log.jsonl"
    assert pending_file.exists()
    saved_row = json.loads(pending_file.read_text().splitlines()[0])
    assert saved_row["row_id"] == result["row_id"]


def test_structuring_fields_are_packed_into_note():
    service = _mock_service()

    append_mood_row(
        valence="bad",
        circumstances="overloaded",
        situation="six open tickets",
        thoughts="I can't prioritize",
        need="a clear next step",
        next_step="close one ticket before opening another",
        service=service,
        sheet_id="fake-sheet-id",
    )

    append_call = service.spreadsheets.return_value.values.return_value.append
    written_row = append_call.call_args.kwargs["body"]["values"][0]
    note = written_row[SHEET_COLUMNS.index("note")]

    assert "situation: six open tickets" in note
    assert "thoughts: I can't prioritize" in note
    assert "need: a clear next step" in note
    assert "next_step: close one ticket before opening another" in note


def test_light_check_in_has_empty_note():
    service = _mock_service()

    append_mood_row(
        valence="good",
        circumstances="quiet day",
        service=service,
        sheet_id="fake-sheet-id",
    )

    append_call = service.spreadsheets.return_value.values.return_value.append
    written_row = append_call.call_args.kwargs["body"]["values"][0]
    assert written_row[SHEET_COLUMNS.index("note")] == ""


def test_energy_level_is_written_when_provided():
    service = _mock_service()

    append_mood_row(
        valence="bad",
        circumstances="exhausted after back-to-back meetings",
        energy_level="low",
        service=service,
        sheet_id="fake-sheet-id",
    )

    append_call = service.spreadsheets.return_value.values.return_value.append
    written_row = append_call.call_args.kwargs["body"]["values"][0]
    assert written_row[SHEET_COLUMNS.index("energy_level")] == "low"


def test_energy_level_defaults_to_empty_when_not_mentioned():
    service = _mock_service()

    append_mood_row(
        valence="good",
        circumstances="quiet day",
        service=service,
        sheet_id="fake-sheet-id",
    )

    append_call = service.spreadsheets.return_value.values.return_value.append
    written_row = append_call.call_args.kwargs["body"]["values"][0]
    assert written_row[SHEET_COLUMNS.index("energy_level")] == ""


def test_log_mood_tool_falls_back_honestly_without_sheet_id(isolated_pending_log):
    # No MOOD_SHEET_ID set (isolated_pending_log fixture deletes it) and no
    # real credentials on disk -- the tool must still behave predictably:
    # report failure, save the fallback, never raise or fabricate success.
    result = log_mood.invoke(
        {"valence": "good", "circumstances": "trying the tool wiring end to end"}
    )

    assert result["status"] == "error"
    pending_file = isolated_pending_log / "pending_sheet_log.jsonl"
    assert pending_file.exists()


def test_log_mood_is_a_langchain_tool_with_expected_name():
    assert log_mood.name == "log_mood"
