"""tools/measures.py -- the curated list, the lookup, and growing the list."""
import csv

import pytest

from tools.measures import (
    FIND_MEASURES_BATCH,
    MEASURE_COLUMNS,
    MEASURES_PATH,
    fits_goal,
    make_add_measure,
    make_find_measures,
    matching_measures,
    measure_ids,
    read_measures,
)

ROWS = [
    ("quick_calm", "recover", "Take a few slow breaths.", "exhausted;calm;wired", "minutes", "stretchly:mini:aad"),
    ("short_walk", "recover", "Walk for five minutes.", "exhausted;wired", "minutes", "user"),
    ("stretch", "recover", "Stand up and stretch.", "calm", "minutes", "user"),
    ("water", "recover", "Drink a glass of water.", "exhausted;calm;wired", "minutes", "user"),
    ("day_off", "recover", "Take a day off.", "exhausted", "days", "user"),
    ("massage", "recover", "Book a massage.", "exhausted;wired", "hours", "user"),
    ("say_no", "act", "Say no to one request today.", "exhausted;calm;wired", "minutes", "agent"),
    ("focus_block", "act", "Block an hour for focused work.", "calm", "hours", "web:https://example.org/x"),
]


@pytest.fixture
def measures_file(tmp_path):
    path = tmp_path / "measures.csv"
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(MEASURE_COLUMNS)
        writer.writerows(ROWS)
    return path


def _call(tool, args, state=None):
    """Invoke a tool the way the graph will: injected args supplied by code."""
    payload = {**args, "tool_call_id": "c1"}
    if state is not None:
        payload["state"] = state
    return tool.invoke({"type": "tool_call", "name": tool.name, "id": "c1", "args": payload})


# -- the goal filter ---------------------------------------------------------


def test_calm_down_now_is_recovery_you_can_do_in_minutes():
    assert fits_goal({"type": "recover", "effort": "minutes"}, "calm down now") is True
    assert fits_goal({"type": "recover", "effort": "days"}, "calm down now") is False
    assert fits_goal({"type": "act", "effort": "minutes"}, "calm down now") is False


def test_plan_recovery_is_recovery_that_takes_hours_or_days():
    assert fits_goal({"type": "recover", "effort": "hours"}, "plan recovery") is True
    assert fits_goal({"type": "recover", "effort": "days"}, "plan recovery") is True
    assert fits_goal({"type": "recover", "effort": "minutes"}, "plan recovery") is False


def test_change_the_situation_is_every_act_measure():
    assert fits_goal({"type": "act", "effort": "days"}, "change the situation") is True
    assert fits_goal({"type": "recover", "effort": "minutes"}, "change the situation") is False


# -- matching ----------------------------------------------------------------


def test_energy_is_matched_against_the_semicolon_list(measures_file):
    ids = [m["measure_id"] for m in matching_measures("calm", "calm down now", path=measures_file)]
    assert "stretch" in ids
    assert "short_walk" not in ids  # exhausted;wired only


def test_measures_already_shown_are_left_out(measures_file):
    first = matching_measures("wired", "calm down now", path=measures_file)
    seen = {m["measure_id"] for m in first}
    again = matching_measures("wired", "calm down now", exclude=seen, path=measures_file)
    assert not seen & {m["measure_id"] for m in again}


def test_a_lookup_returns_a_batch_not_the_whole_list(measures_file):
    assert len(matching_measures("wired", "calm down now", path=measures_file)) <= FIND_MEASURES_BATCH


def test_nothing_left_returns_an_empty_list(measures_file):
    seen = measure_ids(measures_file)
    assert matching_measures("wired", "calm down now", exclude=seen, path=measures_file) == []


# -- find_measures as the model sees it --------------------------------------


def test_find_measures_takes_no_arguments_from_the_model(measures_file):
    """Energy and goal come from the state, so the model cannot pass values the
    user never gave."""
    schema = make_find_measures(measures_file).args
    assert schema == {}


def test_find_measures_records_what_it_showed(measures_file):
    tool = make_find_measures(measures_file)
    command = _call(tool, {}, state={"energy": "wired", "goal": "calm down now", "offered": []})
    shown = command.update["offered"]
    assert shown and all(isinstance(i, str) for i in shown)
    assert command.update["messages"][0].name == "find_measures"


def test_find_measures_without_recorded_choices_is_skipped_not_guessed(measures_file):
    command = _call(make_find_measures(measures_file), {}, state={"offered": []})
    message = command.update["messages"][0]
    assert message.additional_kwargs["receipt"]["status"] == "skipped"
    assert "offered" not in command.update


# -- add_measure -------------------------------------------------------------


def _add(measures_file, **overrides):
    args = {
        "measure_id": "new_idea",
        "measure": "Do the new thing.",
        "measure_type": "act",
        "fits_energy": ["wired"],
        "effort": "minutes",
        **overrides,
    }
    return _call(make_add_measure(measures_file), args)


def test_a_measure_is_appended_with_its_source(measures_file):
    _add(measures_file, from_user=True)
    row = read_measures(measures_file)[-1]
    assert row["measure_id"] == "new_idea"
    assert row["source"] == "user"


def test_a_web_idea_keeps_its_link_and_an_own_idea_is_marked_agent(measures_file):
    _add(measures_file, measure_id="from_web", source_url="https://example.org/a")
    _add(measures_file, measure_id="from_agent")
    rows = {r["measure_id"]: r["source"] for r in read_measures(measures_file)}
    assert rows["from_web"] == "web:https://example.org/a"
    assert rows["from_agent"] == "agent"


def test_a_duplicate_id_is_refused(measures_file):
    """One id must point at one measure, or a logged measure_id is ambiguous."""
    before = len(read_measures(measures_file))
    command = _add(measures_file, measure_id="water")
    assert command.update["messages"][0].additional_kwargs["receipt"]["status"] == "error"
    assert len(read_measures(measures_file)) == before


@pytest.mark.parametrize("bad_id", ["Has Spaces", "x", "id-with-dash", "", "id.with.dots"])
def test_an_unusable_id_is_refused(measures_file, bad_id):
    before = len(read_measures(measures_file))
    _add(measures_file, measure_id=bad_id)
    assert len(read_measures(measures_file)) == before


def test_an_id_is_normalised_to_lowercase(measures_file):
    _add(measures_file, measure_id="  UPPER_Case  ")
    assert read_measures(measures_file)[-1]["measure_id"] == "upper_case"


def test_measure_text_is_flattened_and_capped(measures_file):
    """The text can come from a web page: it is data, not something to trust."""
    _add(measures_file, measure_id="long_one", measure="line one\nline two\t" + "x" * 500)
    row = read_measures(measures_file)[-1]
    assert "\n" not in row["measure"]
    assert len(row["measure"]) <= 200


def test_an_added_measure_is_found_by_the_lookup(measures_file):
    _add(measures_file, measure_id="pause_first", fits_energy=["wired"], measure_type="act")
    ids = [m["measure_id"] for m in matching_measures("wired", "change the situation", path=measures_file)]
    assert "pause_first" in ids


# -- the shipped list --------------------------------------------------------


def test_the_shipped_list_is_well_formed():
    rows = read_measures(MEASURES_PATH)
    assert len(rows) >= 20
    assert len({r["measure_id"] for r in rows}) == len(rows)
    assert all(r["type"] in {"recover", "act"} for r in rows)
    assert all(r["effort"] in {"minutes", "hours", "days"} for r in rows)
    assert all(set(r["fits_energy"].split(";")) <= {"exhausted", "calm", "wired"} for r in rows)


def test_every_energy_and_goal_combination_finds_something():
    """An empty result sends the agent to the web; that should be the exception."""
    for energy in ("exhausted", "calm", "wired"):
        for goal in ("calm down now", "plan recovery", "change the situation"):
            assert matching_measures(energy, goal, path=MEASURES_PATH), f"{energy} / {goal}"
