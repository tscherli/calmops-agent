"""tools/checkin_log.py -- two kinds of row in one append-only file."""
import csv
import json

import pytest

from tools.checkin_log import (
    CHECKIN_COLUMNS,
    append_checkin_row,
    is_rating_row,
    make_log_checkin,
    read_checkin_rows,
)
from tools.measures import MEASURE_COLUMNS

MEASURE_ROWS = [
    ("short_walk", "recover", "Walk for five minutes.", "exhausted;wired", "minutes", "user"),
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


@pytest.fixture
def log_file(tmp_path):
    return tmp_path / "checkin_log.csv"


def _call(tool, **args):
    payload = {**args, "tool_call_id": "c1"}
    return tool.invoke({"type": "tool_call", "name": "log_checkin", "id": "c1", "args": payload})


def _receipt(command):
    return command.update["messages"][0].additional_kwargs["receipt"]


# -- the writer --------------------------------------------------------------


def test_a_checkin_row_carries_the_situation(log_file):
    append_checkin_row(
        energy="wired", circumstances="Deadlines piled up.", measure_id="short_walk", path=log_file
    )
    row = read_checkin_rows(log_file)[0]
    assert list(row) == CHECKIN_COLUMNS
    assert row["energy"] == "wired"
    assert row["circumstances"] == "Deadlines piled up."
    assert row["helpfulness"] == ""


def test_a_rating_row_carries_only_the_verdict(log_file):
    """The situation belongs to the check-in row; repeating it here would store
    the same text twice and let the two copies drift."""
    append_checkin_row(
        energy="wired", circumstances="Deadlines piled up.", measure_id="short_walk", path=log_file
    )
    append_checkin_row(measure_id="short_walk", helpfulness="helped", path=log_file)
    row = read_checkin_rows(log_file)[-1]
    assert row["helpfulness"] == "helped"
    assert row["energy"] == "" and row["circumstances"] == ""


def test_a_verdict_that_arrives_with_the_situation_writes_both_rows(log_file):
    """"they were both helpful" came with energy and the situation in one call.
    The rating row has no room for either, so the situation was dropped and the
    user asked three times why it had not been logged."""
    result = append_checkin_row(
        energy="exhausted",
        circumstances="Very tired at work.",
        measure_id="short_walk",
        helpfulness="helped",
        path=log_file,
    )
    situation, verdict = read_checkin_rows(log_file)
    assert situation["circumstances"] == "Very tired at work."
    assert situation["energy"] == "exhausted" and situation["helpfulness"] == ""
    assert verdict["helpfulness"] == "helped" and verdict["measure_id"] == "short_walk"
    assert result["checkin_row_id"] == situation["row_id"]


def test_a_verdict_for_a_waiting_check_in_adds_no_second_one(log_file):
    append_checkin_row(
        energy="wired", circumstances="Deadlines piled up.", measure_id="short_walk", path=log_file
    )
    append_checkin_row(
        energy="wired",
        circumstances="Deadlines piled up.",
        measure_id="short_walk",
        helpfulness="helped",
        path=log_file,
    )
    assert len(read_checkin_rows(log_file)) == 2


def test_the_same_verdict_twice_is_refused(log_file):
    """A model logged "instrumental_music_break helped" twice in one
    conversation, which the digest counts as two successes."""
    append_checkin_row(
        energy="wired", circumstances="Deadlines piled up.", measure_id="short_walk", path=log_file
    )
    append_checkin_row(measure_id="short_walk", helpfulness="helped", path=log_file)
    again = append_checkin_row(measure_id="short_walk", helpfulness="helped", path=log_file)
    assert again["status"] == "skipped"
    assert len(read_checkin_rows(log_file)) == 2


def test_a_changed_verdict_is_written(log_file):
    """"actually it did not help" is a correction, not a duplicate."""
    append_checkin_row(
        energy="wired", circumstances="Deadlines piled up.", measure_id="short_walk", path=log_file
    )
    append_checkin_row(measure_id="short_walk", helpfulness="helped", path=log_file)
    changed = append_checkin_row(measure_id="short_walk", helpfulness="didnt_help", path=log_file)
    assert changed["status"] == "ok"
    assert read_checkin_rows(log_file)[-1]["helpfulness"] == "didnt_help"


def test_a_checkin_without_a_measure_is_not_written(log_file):
    """The row exists to record a measure the user settled on."""
    result = append_checkin_row(energy="wired", circumstances="A rough day.", path=log_file)
    assert result["status"] == "skipped"
    assert not log_file.exists()


def test_rows_are_only_ever_appended(log_file):
    append_checkin_row(energy="calm", circumstances="One.", measure_id="short_walk", path=log_file)
    append_checkin_row(measure_id="short_walk", helpfulness="somewhat", path=log_file)
    rows = read_checkin_rows(log_file)
    assert len(rows) == 2
    assert [is_rating_row(r) for r in rows] == [False, True]


def test_commas_and_quotes_survive_the_round_trip(log_file):
    text = 'He said "later", then nothing, again.'
    append_checkin_row(energy="calm", circumstances=text, measure_id="say_no", path=log_file)
    assert read_checkin_rows(log_file)[0]["circumstances"] == text


def test_a_write_failure_falls_back_and_never_raises(tmp_path, monkeypatch):
    blocked = tmp_path / "afile"
    blocked.write_text("not a directory", encoding="utf-8")
    pending = tmp_path / "pending.jsonl"
    monkeypatch.setattr("tools.checkin_log.PENDING_LOG_PATH", pending)

    result = append_checkin_row(
        energy="wired", circumstances="x", measure_id="short_walk", path=blocked / "log.csv"
    )

    assert result["status"] == "error"
    assert json.loads(pending.read_text(encoding="utf-8").strip())["measure_id"] == "short_walk"


# -- the tool ----------------------------------------------------------------


def test_an_unknown_measure_id_is_refused(log_file, measures_file):
    command = _call(make_log_checkin(log_file, measures_file), measure_id="invented", energy="calm")
    assert _receipt(command)["status"] == "error"
    assert not log_file.exists()


def test_a_logged_checkin_signals_that_the_conversation_is_complete(log_file, measures_file):
    """The clients start the next conversation fresh on this signal."""
    command = _call(
        make_log_checkin(log_file, measures_file),
        energy="wired",
        circumstances="Deadlines piled up.",
        measure_id="short_walk",
    )
    assert command.update["checkin_logged"] is True
    assert command.update["rejections"] == 0
    assert _receipt(command)["status"] == "ok"


def test_a_verdict_that_wrote_the_situation_counts_as_a_check_in(log_file, measures_file):
    """One call carrying both rows is still a logged check-in: the clients
    start fresh on that signal, and the rating buttons need last_measure_id."""
    command = _call(
        make_log_checkin(log_file, measures_file),
        energy="wired",
        circumstances="Deadlines piled up.",
        measure_id="short_walk",
        helpfulness="helped",
    )
    assert command.update["checkin_logged"] is True
    assert command.update["last_measure_id"] == "short_walk"


def test_a_rating_does_not_end_the_conversation(log_file, measures_file):
    """Otherwise saying 'that helped' mid-conversation would wipe the context
    the user is still using."""
    command = _call(
        make_log_checkin(log_file, measures_file), measure_id="short_walk", helpfulness="helped"
    )
    assert "checkin_logged" not in command.update
    assert json.loads(command.update["messages"][0].content)["kind"] == "rating"


def test_the_tool_reports_a_skipped_write_instead_of_claiming_success(log_file, measures_file):
    command = _call(make_log_checkin(log_file, measures_file), energy="calm", circumstances="hm")
    assert _receipt(command)["status"] == "skipped"
    assert "Not logged" in command.update["messages"][0].content


def test_the_tool_writes_where_it_was_told_to(log_file, measures_file):
    """Tests and eval runs must never reach the real log."""
    _call(
        make_log_checkin(log_file, measures_file),
        energy="calm",
        circumstances="A quiet one.",
        measure_id="say_no",
    )
    assert log_file.exists()
    assert read_checkin_rows(log_file)[0]["measure_id"] == "say_no"
