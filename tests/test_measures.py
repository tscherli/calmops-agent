"""tools/measures.py -- the curated list, the lookup, and growing the list."""
import csv

import pytest

from tools.measures import (
    LOOKUP_ROW_LIMIT,
    GOALS,
    MEASURE_COLUMNS,
    MEASURES_PATH,
    fits_goal,
    make_add_measure,
    make_find_measures,
    make_search_measures,
    matching_measures,
    measure_ids,
    read_measures,
    search_measures_by_words,
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


def test_a_lookup_returns_every_measure_for_the_energy_and_goal(tmp_path):
    """The model picks by fit, so it has to see the candidates. A window over
    rows in file order made the selection rule "added most recently", not
    "fits best", and a second lookup could only walk further down the file.

    The list here is deliberately wider than the four-row window that used to
    cut it off: a slice smaller than the window cannot show that it is gone.
    """
    path = tmp_path / "wide.csv"
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(MEASURE_COLUMNS)
        writer.writerows(
            (f"calm_{n}", "recover", f"Measure number {n}.", "wired", "minutes", "user")
            for n in range(7)
        )

    found = [m["measure_id"] for m in matching_measures("wired", "calm down now", path=path)]

    assert found == [f"calm_{n}" for n in range(7)]


def test_the_same_lookup_twice_returns_the_same_measures(measures_file):
    """Nothing is consumed by being returned: a measure the model saw but did
    not pass on is still there next time, and one that fits best can come back."""
    first = matching_measures("wired", "calm down now", path=measures_file)
    again = matching_measures("wired", "calm down now", path=measures_file)

    assert [m["measure_id"] for m in first] == [m["measure_id"] for m in again]


def test_the_row_limit_is_a_safety_valve_not_a_selection_rule(measures_file):
    """It sits far above any real slice; if a list ever hits it, the answer is
    ranking, not a window."""
    assert LOOKUP_ROW_LIMIT > len(read_measures(measures_file))
    assert len(matching_measures("wired", "calm down now", path=measures_file)) <= LOOKUP_ROW_LIMIT


def test_an_energy_and_goal_with_no_rows_returns_an_empty_list(measures_file):
    """The only empty result left: the list itself holds nothing for this.
    The prompt turns that into an offer to search the web."""
    assert matching_measures("calm", "plan recovery", path=measures_file) == []


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


def test_find_measures_without_an_energy_is_skipped_not_guessed(measures_file):
    command = _call(make_find_measures(measures_file), {}, state={"offered": []})
    message = command.update["messages"][0]
    assert message.additional_kwargs["receipt"]["status"] == "skipped"
    assert "offered" not in command.update


def _goals_of(ids, path):
    return {g for g in GOALS if any(fits_goal(r, g) for r in read_measures(path) if r["measure_id"] in ids)}


def test_without_a_goal_one_batch_spans_every_goal(measures_file):
    """Someone who does not know what would help gets a choice across goals.
    In file order the fixture's first four cover only two of them."""
    ids = {m["measure_id"] for m in matching_measures("exhausted", None, path=measures_file)}
    assert _goals_of(ids, measures_file) == set(GOALS)
    assert all("exhausted" in r["fits_energy"] for r in read_measures(measures_file) if r["measure_id"] in ids)


def test_without_a_goal_the_shipped_list_offers_more_than_quick_recoveries():
    """The real list has fourteen quick recoveries for "exhausted" before any
    other row: a hand test got four of them and nothing else."""
    ids = {m["measure_id"] for m in matching_measures("exhausted", None, path=MEASURES_PATH)}
    assert _goals_of(ids, MEASURES_PATH) == set(GOALS)


def test_find_measures_with_energy_alone_looks_across_goals(measures_file):
    command = _call(make_find_measures(measures_file), {}, state={"energy": "exhausted", "offered": []})
    receipt = command.update["messages"][0].additional_kwargs["receipt"]
    assert receipt["status"] == "ok"
    assert "(any goal)" in receipt["detail"]
    assert command.update["offered"]


# -- search_measures ---------------------------------------------------------


def test_search_finds_a_measure_by_a_word_stem(measures_file):
    """"is walking in my list?" -- the walk has to be found."""
    assert [m["measure_id"] for m in search_measures_by_words("walking", measures_file)] == ["short_walk"]


def test_search_matches_the_id_as_well_as_the_text(measures_file):
    assert "day_off" in [m["measure_id"] for m in search_measures_by_words("day", measures_file)]


def test_search_with_only_filler_words_finds_nothing(measures_file):
    """Otherwise "is there something for this in my list" would match half the
    list: "for" alone is in "Walk for five minutes" and "Block an hour for
    focused work". (An earlier version of this test used words no row
    contained, so it passed with the filter removed -- it tested nothing.)"""
    query = "is there something for this already in the list"
    assert any("for" in r["measure"].split() for r in read_measures(measures_file))
    assert search_measures_by_words(query, measures_file) == []


def test_search_for_something_absent_says_so(measures_file):
    assert search_measures_by_words("swimming", measures_file) == []


def test_search_measures_takes_only_words_and_changes_nothing(measures_file):
    """It reads: no state, no Command, no `offered` -- asking whether a walk is
    in the list is not the same as offering one."""
    tool = make_search_measures(measures_file)
    assert list(tool.args) == ["words"]
    before = measures_file.read_bytes()
    result = tool.invoke({"words": "walk"})
    assert isinstance(result, list)
    assert measures_file.read_bytes() == before


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


def test_a_link_makes_it_a_web_idea_even_if_the_model_says_user(measures_file):
    """A hand test: two web ideas went through the debate station, whose prompt
    says from_user true, and were saved as "user" with their links dropped."""
    _add(measures_file, measure_id="digital_reset", from_user=True,
         source_url="  https://news.va.gov/147153/reclaim-your-focus-benefits-digital-reset/ ")
    row = read_measures(measures_file)[-1]
    assert row["source"] == "web:https://news.va.gov/147153/reclaim-your-focus-benefits-digital-reset/"


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


def test_web_punctuation_is_flattened_to_ascii(measures_file):
    """A measure copied from a web page arrived as "5-15 minutes" with an en
    dash, and the list is a plain ASCII file read in a terminal."""
    _add(measures_file, measure_id="legs_up",
         measure="Rest 5\u201315 minutes \u2014 it\u2019s \u201cpassive\u201d\u2026")
    row = read_measures(measures_file)[-1]
    assert row["measure"] == 'Rest 5-15 minutes - it\'s "passive"...'
    assert all(ord(c) < 128 for c in row["measure"])


def test_letters_with_accents_survive_as_plain_letters(measures_file):
    _add(measures_file, measure_id="cafe_break", measure="Take a caf\u00e9 break.")
    assert read_measures(measures_file)[-1]["measure"] == "Take a cafe break."


def test_the_shipped_list_is_plain_ascii():
    """add_measure now folds web punctuation; the file must stay readable
    in a terminal and diffable by hand."""
    assert MEASURES_PATH.read_bytes().decode("ascii")


# -- the search ranks before it cuts -----------------------------------------


def test_a_two_word_match_outranks_rows_that_match_one_word(measures_file):
    """The cut used to be made in file order, so weaker matches that happened
    to sit earlier crowded out the best one.

    Found in a real conversation: "eye palming" returned five rows that merely
    said "eyes" and not palming_eye_rest, which matches both words and sits at
    row 48. The agent searched, was told truthfully that nothing like it
    existed, and added a duplicate. Same shape as the find_measures recency
    bug -- a window over a list in insertion order, mistaken for a selection.
    """
    import csv

    from tools.measures import MEASURE_COLUMNS, search_measures_by_words

    rows = [
        ("eyes_closed", "recover", "Close your eyes for a moment.", "exhausted", "minutes", "user"),
        ("eyes_away", "recover", "Look away and rest your eyes.", "exhausted", "minutes", "user"),
        ("eyes_blink", "recover", "Blink slowly with your eyes soft.", "exhausted", "minutes", "user"),
        ("eyes_drop", "recover", "Let your eyes drop shut briefly.", "exhausted", "minutes", "user"),
        ("eyes_rest", "recover", "Give your eyes a rest from the screen.", "exhausted", "minutes", "user"),
        ("palming_eye_rest", "recover", "Cup your palms over closed eyes.", "exhausted", "minutes", "user"),
    ]
    with measures_file.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(MEASURE_COLUMNS)
        writer.writerows(rows)

    found = [r["measure_id"] for r in search_measures_by_words("eye palming", measures_file)]

    assert found[0] == "palming_eye_rest", found


def test_a_single_word_query_still_finds_its_row(measures_file):
    """Ranking must not cost the plain case anything."""
    from tools.measures import search_measures_by_words

    assert [r["measure_id"] for r in search_measures_by_words("walk", measures_file)] == ["short_walk"]
