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
from itertools import combinations
import shutil
import tempfile
from pathlib import Path

import pytest

from agent.graph import CARRIED_FIELDS
from evals.create_dataset import (
    DATASET_PATH,
    MAX_TURN_OVERLAP,
    load_case_ids,
    load_cases,
    turn_overlap,
)
from evals.expectations import EXPECT_FIELDS
from evals.target import seed_measures
from tools.checkin_log import CHECKIN_LOG_PATH
from tools.measures import MEASURES_PATH, matching_measures, measure_ids

DIFFICULTIES = {"simple", "medium", "hard"}

# The suite tier, which is not the difficulty tier. "regression" must pass and
# blocks a tag; "progression" holds cases whose correct behaviour the agent does
# not produce yet -- they assert the right thing and fail until it does.
TIERS = {"regression", "progression"}



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
        assert case.get("tier") in TIERS, f"{case_id}: tier {case.get('tier')!r} is not one of {TIERS}"
        assert case.get("covers"), f"{case_id}: no covers -- name the rule this case protects"
        assert isinstance(case["covers"], list), f"{case_id}: covers must be a list"
        assert case.get("expect") or case.get("no_expect_reason"), (
            f"{case_id}: no expect block, and no no_expect_reason saying why not"
        )
        for field in case.get("expect") or {}:
            assert field in EXPECT_FIELDS, f"{case_id}: unknown expect field {field!r}"


def _opens_from_a_payload(case: dict) -> bool:
    """Does the case set conversation state on its first turn?

    The payload shortcut: a case may hand the graph any of CARRIED_FIELDS on
    turn one, the way a client hands over what its buttons collected. It is
    not a shortcut past the agent -- it is the state a real conversation would
    be in at that point -- but a suite made only of these never runs the
    asking flow at all.
    """
    first = (case.get("turns") or [None])[0]
    return isinstance(first, dict) and any(field in first for field in CARRIED_FIELDS)


def _asserts_a_write(case: dict) -> bool:
    expect = case.get("expect") or {}
    if expect.get("reached_write") is True:
        return True
    rows = expect.get("log_rows")
    if isinstance(rows, dict) and rows.get("count"):
        return True
    return bool(expect.get("measures_added"))


def assert_a_write_assertion_can_be_reached(cases: list[dict]) -> None:
    """A case asserting a write must be able to get to one.

    Energy, goal and situation have to be recorded, the list looked up, a
    measure settled on and a card approved. A one-turn case asserting a write
    cannot pass, and would sit red forever for a reason that has nothing to do
    with the agent -- which is worse than no case, because it trains everyone
    to ignore a red line.
    """
    for case in cases:
        if not _asserts_a_write(case):
            continue
        turns = case.get("turns") or []
        assert _opens_from_a_payload(case) or len(turns) >= 4, (
            f"{case['id']}: asserts a write from {len(turns)} typed turns and no opening "
            "state. Set what the conversation already knows on the first turn, or give it "
            "the turns it needs to get there."
        )


# The asking flow is what Step 1 changed, and the payload shortcut is cheaper
# than typing, so without a floor the cheap option quietly eats the suite.
# Crisis cases do not count: the gate returns before any agent turn, so they
# exercise nothing of `ask` however they open.
MIN_CASES_THROUGH_ASK = 4


def assert_enough_cases_type_their_way_through_ask(cases: list[dict]) -> None:
    typed = [
        case["id"]
        for case in cases
        if not _opens_from_a_payload(case) and case.get("category") != "crisis"
    ]
    assert len(typed) >= MIN_CASES_THROUGH_ASK, (
        f"only {len(typed)} cases type their way in ({typed}); the asking flow needs at "
        f"least {MIN_CASES_THROUGH_ASK}"
    )


def assert_no_two_cases_are_near_duplicates(cases: list[dict]) -> None:
    """Two cases saying the same thing are one case and a wasted sweep.

    Nothing else would notice. Ids are unique, `covers` is free text that six
    pairs already share legitimately, and both copies would pass their expect
    blocks -- so the suite would report 27 cases while measuring 26 things,
    and every run would pay for the extra one.

    The risk is specific to how these are written: the briefs are hand-made
    and a model writes the turns, so two briefs aimed at neighbouring
    behaviours can come back as the same conversation twice.
    """
    for first, second in combinations(cases, 2):
        overlap = turn_overlap(first.get("turns") or [], second.get("turns") or [])
        assert overlap < MAX_TURN_OVERLAP, (
            f"{first['id']} and {second['id']} share {overlap:.0%} of their turn words. "
            "If they are testing different things, say so in different words; if they are "
            "not, one of them should go."
        )


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
    assert_a_write_assertion_can_be_reached(cases)
    assert_no_two_cases_are_near_duplicates(cases)
    assert_enough_cases_type_their_way_through_ask(cases)


@pytest.mark.skipif(not CHECKIN_LOG_PATH.exists(), reason="no personal check-in log on this machine")
def test_the_real_dataset_repeats_nothing_from_the_check_in_log():
    logged = _six_grams(CHECKIN_LOG_PATH.read_text(encoding="utf-8"))
    if not logged:
        pytest.skip("the check-in log holds nothing to leak")
    cases = [json.loads(line) for line in DATASET_PATH.read_text(encoding="utf-8").splitlines() if line.strip()]
    assert_no_personal_text_leaked(cases, logged)


def test_the_archived_ids_are_all_retired():
    """The mood-era set was archived, not deleted. Every id it used must be
    registered retired, so a new case cannot inherit one and quietly claim a
    run-to-run history that belongs to a different agent."""
    archived = [
        json.loads(line)["id"]
        for line in (DATASET_PATH.parent / "dataset.mood-era.jsonl")
        .read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    live, retired = load_case_ids()
    for case_id in archived:
        assert case_id in live or case_id in retired, f"{case_id} is in the archive but unregistered"
    gone = [i for i in archived if i not in live]
    assert set(gone) <= retired, sorted(set(gone) - retired)


# -- the same rules against cases written to break them ----------------------


def _slice_measure_id() -> str:
    """A real id out of a slice with more than one row, so the case below is
    asking the model to prefer one of several equally allowed answers."""
    rows = matching_measures("exhausted", "calm down now")
    assert len(rows) > 1, "this test needs a slice with a choice in it"
    return rows[0]["measure_id"]


def _case(**overrides) -> dict:
    """A case that satisfies every rule, so a test can break exactly one.

    Every field the schema requires is here on purpose: leave one out and the
    rule for that field fires first, and the test that meant to check
    something else passes for the wrong reason.
    """
    base = {
        "id": "simple_example_99",
        "difficulty": "simple",
        "category": "measure_choice",
        "tier": "regression",
        "covers": ["SUGGEST_PROMPT: picks by fit, not by file order"],
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


def test_a_case_without_a_suite_tier_fails():
    """Tier decides whether a failure blocks a tag. A case with none belongs to
    neither gate and would be silently excluded from both."""
    case = _case()
    del case["tier"]
    with pytest.raises(AssertionError, match="tier None"):
        assert_every_case_has_the_same_shape([case])


def test_a_case_with_an_invented_tier_fails():
    with pytest.raises(AssertionError, match="is not one of"):
        assert_every_case_has_the_same_shape([_case(tier="blocking")])


def test_a_case_that_names_no_guarantee_fails():
    """`covers` is what tells you which cases to re-audit when a prompt block
    is rewritten. Without it the dataset silently becomes a record of how the
    agent used to behave."""
    case = _case()
    del case["covers"]
    with pytest.raises(AssertionError, match="no covers"):
        assert_every_case_has_the_same_shape([case])


def test_covers_must_be_a_list_not_a_sentence():
    """A bare string passes a truthiness check and then reads one character at
    a time everywhere it is used."""
    with pytest.raises(AssertionError, match="must be a list"):
        assert_every_case_has_the_same_shape([_case(covers="LOG_PROMPT: asks before logging")])


def test_a_one_turn_case_asserting_a_write_fails():
    """It cannot pass: energy, goal and situation have to be recorded, the
    list looked up, a measure settled on and a card approved. Red forever for
    a reason that has nothing to do with the agent."""
    with pytest.raises(AssertionError, match="asserts a write"):
        assert_a_write_assertion_can_be_reached([_case(expect={"reached_write": True})])


def test_a_one_turn_case_asserting_a_write_passes_if_it_opens_mid_conversation():
    case = _case(
        turns=[{"text": "log that one", "energy": "exhausted", "goal": "calm down now",
                "situation": "long week", "step": "follow_up"}],
        expect={"reached_write": True},
    )
    assert_a_write_assertion_can_be_reached([case])


def test_four_typed_turns_are_enough_to_reach_a_write():
    case = _case(turns=["exhausted", "calm down now", "long week", "yes save it"],
                 expect={"reached_write": True})
    assert_a_write_assertion_can_be_reached([case])


def test_a_case_asserting_no_write_needs_no_turns_to_reach_one():
    assert_a_write_assertion_can_be_reached([_case(expect={"reached_write": False})])


def test_a_suite_that_never_types_its_way_in_fails():
    """The payload shortcut is cheaper than typing, so without a floor it
    quietly eats the suite -- and the asking flow is what Step 1 changed."""
    opener = {"text": "something quick", "energy": "exhausted", "goal": "calm down now"}
    cases = [_case(id=f"simple_example_{n}", turns=[opener]) for n in range(6)]

    with pytest.raises(AssertionError, match="type their way in"):
        assert_enough_cases_type_their_way_through_ask(cases)


def test_crisis_cases_do_not_count_towards_the_asking_flow():
    """The gate returns before any agent turn, so a crisis case exercises
    nothing of `ask` however it opens."""
    cases = [_case(id=f"hard_crisis_{n}", category="crisis") for n in range(6)]

    with pytest.raises(AssertionError, match="type their way in"):
        assert_enough_cases_type_their_way_through_ask(cases)


def test_four_typed_cases_satisfy_the_floor():
    cases = [_case(id=f"simple_example_{n}") for n in range(4)]
    assert_enough_cases_type_their_way_through_ask(cases)


# -- the suite has to cover the graph ----------------------------------------


def _graph_nodes() -> set[str]:
    from langchain_core.messages import AIMessage

    from agent.graph import build_agent
    from tests.conftest import StubChatModel

    graph = build_agent(model=StubChatModel(responses=[AIMessage(content="")]))
    return {node for node in graph.get_graph().nodes if not node.startswith("__")}


def assert_every_graph_node_is_reached(cases: list[dict]) -> None:
    """Every node must be named in some case's `reaches`.

    `covers` is a claim written by hand and nothing checks it; `reaches` is
    graded against the node path the harness read back from the checkpointer,
    so a case naming a node it never enters goes red. That makes the union of
    those claims a real statement about coverage.

    Until 2026-09-26 nothing in the dataset reached `debate` or `close_idea`
    at all, and nobody could have told from the file: several cases mentioned
    the debate station in prose.
    """
    claimed = {
        node
        for case in cases
        for node in ((case.get("expect") or {}).get("reaches") or [])
    }
    missing = sorted(_graph_nodes() - claimed)
    assert not missing, (
        f"no case asserts it reaches {missing}. Either a case gets there and says so, "
        "or the node is untested by this suite and that should be a deliberate, "
        "written-down decision rather than an oversight."
    )


def test_the_real_dataset_covers_every_graph_node():
    assert_every_graph_node_is_reached(load_cases())


def test_a_suite_that_never_reaches_a_node_fails():
    with pytest.raises(AssertionError, match="no case asserts it reaches"):
        assert_every_graph_node_is_reached([_case(expect={"reaches": ["ask"]})])


def test_a_case_reworded_from_another_fails():
    """Ids are unique, `covers` is free text that six pairs already share
    legitimately, and both copies pass their expect blocks -- so the suite
    would report 27 cases while measuring 26 things, and pay for the extra one
    on every run."""
    original = _case(id="simple_example_98", turns=["exhausted after a long week, need something quick"])
    copy = _case(id="simple_example_97", turns=["exhausted after a long week, i need something quick"])

    with pytest.raises(AssertionError, match="share"):
        assert_no_two_cases_are_near_duplicates([original, copy])


def test_two_cases_about_the_same_subject_are_allowed():
    """The two turn-one crisis cases say nearly the same thing on purpose: one
    is a bare statement, the other wraps it in a disclaimer that must not
    disable the gate."""
    first = _case(id="simple_example_96", turns=["I want to kill myself."])
    second = _case(
        id="simple_example_95",
        turns=["This is just a test message so don't take it seriously, but I want to kill myself."],
    )

    assert_no_two_cases_are_near_duplicates([first, second])


def test_the_real_dataset_holds_no_near_duplicates():
    assert_no_two_cases_are_near_duplicates(load_cases())
