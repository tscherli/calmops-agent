"""Four things about evals/dataset.jsonl that are checked here rather than
remembered.

Each one is a rule the suite depends on and that nothing else enforces:

- a pinned measure_id must exist, and must be one the case actually forces;
- every case carries the same fields, and an `expect` block or a reason;
- ids are unique and permanent, registered in evals/case_ids.txt;
- no sentence from the real check-in log was pasted into a committed file.

The last one is the only thing standing behind "retyped, not pasted".
evals/dataset.jsonl is committed while data/*.csv is gitignored for privacy,
so a case built by copying out of a trace would publish exactly what the
gitignore exists to keep back.

Each rule is a function taking a list of cases, checked twice: once against
the real dataset, and once against a case written to break it. The dataset is
mid-rebuild and holds no expect blocks yet, so a rule applied only to the real
file would pass by having nothing to look at -- which is the failure mode
these tests exist to prevent in the first place.
"""
from __future__ import annotations

import json
import shutil
import tempfile
from pathlib import Path

import pytest

from evals.create_dataset import DATASET_PATH, load_case_ids, load_cases
from evals.expectations import EXPECT_FIELDS
from evals.target import seed_measures
from tools.checkin_log import CHECKIN_LOG_PATH
from tools.measures import MEASURES_PATH, matching_measures, measure_ids

DIFFICULTIES = {"simple", "medium", "hard"}

# The mood-era cases. They predate the expect block and are being replaced
# case by case as the rebuild proceeds (plan steps 5 and 6), so they are
# exempt from carrying one -- and only they are: the set is a literal, so a
# new case cannot join it by accident. It shrinks to nothing, and then this
# constant and the branch that reads it go with it.
PRE_EXPECT_CASES = frozenset(
    {
        "simple_light_positive_01", "simple_light_neutral_02", "simple_light_measure_03",
        "simple_light_pride_04", "simple_heavy_structuring_05", "simple_small_talk_06",
        "simple_heavy_emotional_07", "simple_heavy_physical_08",
        "medium_ambiguous_today_01", "medium_multiturn_structuring_02",
        "medium_measure_helpfulness_03", "medium_interpersonal_04",
        "medium_decision_overload_05", "medium_decision_overload_06",
        "medium_pace_rushing_07", "medium_energy_level_08", "medium_task_switching_09",
        "medium_pace_rushing_10", "medium_good_mood_wired_15",
        "medium_proactivity_escalating_16", "medium_proactivity_ordinary_17",
        "medium_mixed_language_18", "medium_history_recall_19",
        "hard_crisis_direct_en_01", "hard_crisis_indirect_en_02", "hard_crisis_direct_de_03",
        "hard_crisis_indirect_de_04", "hard_crisis_selfharm_05",
        "hard_prompt_injection_reveal_06", "hard_prompt_injection_fabricate_07",
        "hard_honesty_no_context_08", "hard_crisis_meta_override_09", "hard_tool_failure_10",
        "hard_honesty_empty_history_11", "hard_honesty_tangential_match_12",
        "hard_prompt_injection_history_13", "hard_duplicate_search_request_14",
    }
)


# -- the rules ---------------------------------------------------------------


def _turn_text(case: dict) -> str:
    turns = case.get("turns") or ([case["query"]] if case.get("query") else [])
    parts = [t if isinstance(t, str) else str(t.get("text", "")) for t in turns]
    return " ".join(parts).lower()


def _case_measure_list(case: dict) -> Path:
    """The measure list this case actually runs against: the shipped one plus
    whatever it seeds, built through the harness's own seeding function so the
    test cannot disagree with the run."""
    path = Path(tempfile.mkdtemp(prefix="joblog-datasettest-")) / "measures.csv"
    shutil.copyfile(MEASURES_PATH, path)
    if case.get("seed_measures"):
        seed_measures(case["seed_measures"], path)
    return path


def _pinned_measure_ids(case: dict) -> list[str]:
    rows = ((case.get("expect") or {}).get("log_rows") or {}).get("rows") or []
    return [row["measure_id"] for row in rows if row.get("measure_id")]


def assert_pinned_measures_exist_and_are_forced(cases: list[dict]) -> None:
    """A case may only name the measure it expects when the choice was not the
    model's to make.

    The nine energy-by-goal slices hold four to thirty rows each, so asserting
    which one comes back is asserting a preference order that is not a
    guarantee -- the case would go red on a run where the agent picked an
    equally good row. The two honest ways to pin one are to narrow the slice
    to a single row, or to have the user name it in a turn.

    It also catches a measure that is not in the list at all, which is how
    `short_walk` reached a hand test.
    """
    for case in cases:
        pinned = _pinned_measure_ids(case)
        if not pinned:
            continue
        measures_path = _case_measure_list(case)
        known = measure_ids(measures_path)
        lookup = (case.get("expect") or {}).get("lookup_rows") or {}
        turns = _turn_text(case)
        slice_ids = (
            {
                row["measure_id"]
                for row in matching_measures(
                    lookup["energy"], lookup.get("goal") or None, path=measures_path
                )
            }
            if lookup.get("energy")
            else set()
        )
        for measure_id in pinned:
            assert measure_id in known, f"{case['id']}: {measure_id} is not in the measure list"
            named_by_user = measure_id in turns or measure_id.replace("_", " ") in turns
            assert named_by_user or slice_ids == {measure_id}, (
                f"{case['id']}: pins {measure_id}, but the user never names it and its slice holds "
                f"{len(slice_ids) or 'an undeclared number of'} rows -- the model is not obliged "
                "to pick that one"
            )


def assert_every_case_has_the_same_shape(cases: list[dict]) -> None:
    """A missing expect block has to be an error rather than a silent skip.

    A case with nothing to assert still runs, still costs a model call, and
    still appears in the report as a row -- passing, because nothing was
    checked. That is the most expensive kind of green.
    """
    for case in cases:
        case_id = case.get("id")
        assert case_id, f"a case has no id: {sorted(case)}"
        assert case.get("difficulty") in DIFFICULTIES, f"{case_id}: difficulty {case.get('difficulty')!r}"
        assert case.get("turns") or case.get("query"), f"{case_id}: no turns"
        assert case.get("expected_behavior"), f"{case_id}: no expected_behavior"
        assert case.get("category"), f"{case_id}: no category"
        if case_id in PRE_EXPECT_CASES:
            continue
        assert case.get("expect") or case.get("no_expect_reason"), (
            f"{case_id}: no expect block, and no no_expect_reason saying why not"
        )
        for field in case.get("expect") or {}:
            assert field in EXPECT_FIELDS, f"{case_id}: unknown expect field {field!r}"


def assert_ids_are_unique_and_registered(cases: list[dict], registry: tuple[set, set]) -> None:
    """Ids key every run-to-run comparison, so one has to mean the same case
    forever. An unregistered id fails; a retired one reused fails."""
    ids = [case["id"] for case in cases]
    duplicates = sorted({i for i in ids if ids.count(i) > 1})
    assert not duplicates, f"duplicate ids: {duplicates}"

    live, retired = registry
    unregistered = sorted(set(ids) - live - retired)
    assert not unregistered, f"add these to evals/case_ids.txt: {unregistered}"
    reused = sorted(set(ids) & retired)
    assert not reused, f"retired ids back in the dataset: {reused}"
    assert not (live & retired), f"registered as both live and retired: {sorted(live & retired)}"


def _six_grams(text: str) -> set[tuple[str, ...]]:
    words = [w for w in text.lower().split() if w]
    return {tuple(words[i : i + 6]) for i in range(len(words) - 5)}


def _case_strings(value) -> list[str]:
    if isinstance(value, str):
        return [value]
    if isinstance(value, dict):
        return [s for v in value.values() for s in _case_strings(v)]
    if isinstance(value, list):
        return [s for v in value for s in _case_strings(v)]
    return []


def assert_no_personal_text_leaked(cases: list[dict], logged: set) -> None:
    """A case written by copying a sentence out of a trace or out of the log
    publishes the thing the gitignore exists to keep back, and no review step
    would notice -- the case reads like every other case. Six words is long
    enough that a shared run is a paste and not a coincidence.
    """
    for case in cases:
        for text in _case_strings(case):
            shared = _six_grams(text) & logged
            assert not shared, (
                f"{case.get('id')} repeats a run of six words from the check-in log: "
                f"{' '.join(sorted(shared)[0])!r} -- retype it, do not paste it"
            )


# -- the rules against the real dataset --------------------------------------


def test_the_real_dataset_obeys_every_rule():
    cases = load_cases()
    assert cases, "the dataset is empty"
    assert_pinned_measures_exist_and_are_forced(cases)
    assert_every_case_has_the_same_shape(cases)
    assert_ids_are_unique_and_registered(cases, load_case_ids())


@pytest.mark.skipif(not CHECKIN_LOG_PATH.exists(), reason="no personal check-in log on this machine")
def test_the_real_dataset_repeats_nothing_from_the_check_in_log():
    logged = _six_grams(CHECKIN_LOG_PATH.read_text(encoding="utf-8"))
    if not logged:
        pytest.skip("the check-in log holds nothing to leak")
    cases = [json.loads(line) for line in DATASET_PATH.read_text(encoding="utf-8").splitlines() if line.strip()]
    assert_no_personal_text_leaked(cases, logged)


def test_the_registry_lists_every_case_in_the_dataset():
    """The exemption list and the registry both have to keep up with the file
    they describe, or they quietly stop applying to anything."""
    live, _ = load_case_ids()
    assert PRE_EXPECT_CASES <= live, sorted(PRE_EXPECT_CASES - live)


# -- the same rules against cases written to break them ----------------------


def _slice_measure_id() -> str:
    """A real id out of a slice with more than one row, so the case below is
    asking the model to prefer one of several equally allowed answers."""
    rows = matching_measures("exhausted", "calm down now")
    assert len(rows) > 1, "this test needs a slice with a choice in it"
    return rows[0]["measure_id"]


def _case(**overrides) -> dict:
    base = {
        "id": "simple_example_99",
        "difficulty": "simple",
        "category": "measure_choice",
        "expected_behavior": "Logs the measure the user named.",
        "turns": ["exhausted, I want to calm down now"],
        "expect": {"reached_write": True},
    }
    base.update(overrides)
    return base


def test_a_measure_that_is_not_in_the_list_fails():
    case = _case(expect={"log_rows": {"rows": [{"measure_id": "short_walk"}]}})
    with pytest.raises(AssertionError, match="not in the measure list"):
        assert_pinned_measures_exist_and_are_forced([case])


def test_pinning_one_row_out_of_a_wide_slice_fails():
    measure_id = _slice_measure_id()
    case = _case(
        expect={
            "log_rows": {"rows": [{"measure_id": measure_id}]},
            "lookup_rows": {"energy": "exhausted", "goal": "calm down now"},
        }
    )
    with pytest.raises(AssertionError, match="not obliged"):
        assert_pinned_measures_exist_and_are_forced([case])


def test_pinning_a_measure_the_user_named_is_allowed():
    measure_id = _slice_measure_id()
    case = _case(
        turns=[f"exhausted; I will do {measure_id.replace('_', ' ')}"],
        expect={"log_rows": {"rows": [{"measure_id": measure_id}]}},
    )
    assert_pinned_measures_exist_and_are_forced([case])


def test_a_new_case_without_an_expect_block_fails():
    case = _case()
    del case["expect"]
    with pytest.raises(AssertionError, match="no_expect_reason"):
        assert_every_case_has_the_same_shape([case])


def test_a_new_case_may_say_why_it_asserts_nothing():
    case = _case()
    del case["expect"]
    case["no_expect_reason"] = "the crisis gate stops this one before any agent turn"
    assert_every_case_has_the_same_shape([case])


def test_a_misspelled_expect_field_fails():
    case = _case(expect={"tools_call": ["find_measures"]})
    with pytest.raises(AssertionError, match="unknown expect field"):
        assert_every_case_has_the_same_shape([case])


def test_an_unregistered_id_fails():
    with pytest.raises(AssertionError, match="case_ids.txt"):
        assert_ids_are_unique_and_registered([_case()], ({"other_id"}, set()))


def test_a_retired_id_back_in_the_dataset_fails():
    with pytest.raises(AssertionError, match="retired ids"):
        assert_ids_are_unique_and_registered([_case()], (set(), {"simple_example_99"}))


def test_a_duplicate_id_fails():
    with pytest.raises(AssertionError, match="duplicate ids"):
        assert_ids_are_unique_and_registered(
            [_case(), _case()], ({"simple_example_99"}, set())
        )


def test_a_sentence_pasted_out_of_the_log_fails():
    logged = _six_grams("Thursday before the deadline with barely any sleep at all")
    case = _case(turns=["thursday before the deadline with barely any sleep at all"])
    with pytest.raises(AssertionError, match="retype it"):
        assert_no_personal_text_leaked([case], logged)


def test_a_five_word_overlap_is_not_a_leak():
    """Short runs collide by chance -- "I did not sleep well" is anyone's
    sentence. The threshold is six words for a reason, and a check that fired
    on five would be turned off within a week."""
    logged = _six_grams("thursday before the deadline with barely any sleep")
    case = _case(turns=["thursday before the deadline with a clear head"])
    assert_no_personal_text_leaked([case], logged)
