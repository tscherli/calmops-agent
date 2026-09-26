"""Tests for evals/calibrate_judge.py -- the gate that decides whether a
judge number may be quoted at all.

No Gemini call anywhere: the two questions are stubbed. What is checked here
is the planting (a violation that failed to plant reads as a miss and would
condemn a judge that did nothing wrong), the evidence the labeller is shown,
and the gate arithmetic, which is the only thing standing between an
uncalibrated number and a README.
"""
import json

import pytest

import evals.calibrate_judge as calib
from evals.calibrate_judge import (
    OFFERED_KEY,
    PLANTED_MEASURE,
    PLANTED_SITUATION_CLAUSE,
    SITUATION_KEY,
    _planting_took,
    _would_ask,
    evidence_for,
    plant_offered_violation,
    plant_situation_violation,
    question_report,
)
from evals.outputs import situation_sentences


def _lookup(*measure_ids):
    return {
        "role": "tool",
        "name": "find_measures",
        "content": json.dumps([{"measure_id": m, "measure": f"Do {m}."} for m in measure_ids]),
    }


def _outputs(**overrides):
    outputs = {
        "messages": [
            {"role": "human", "content": "wired after a long call"},
            _lookup("stand_up", "breathe"),
            {"role": "ai", "content": "Two from your list: stand up, or breathe.", "tool_calls": None},
        ],
        "approvals": [{"call_id": "w1", "args": {"circumstances": "a long call"}, "decision": "approve"}],
    }
    outputs.update(overrides)
    return outputs


# -- planting ----------------------------------------------------------------


def test_a_planted_measure_lands_in_the_reply():
    planted = plant_offered_violation(_outputs())

    assert PLANTED_MEASURE.lower() in planted["messages"][-1]["content"].lower()
    assert _planting_took(OFFERED_KEY, planted)


def test_a_planted_situation_lands_on_the_card():
    planted = plant_situation_violation(_outputs())

    assert PLANTED_SITUATION_CLAUSE in planted["approvals"][0]["args"]["circumstances"]
    assert _planting_took(SITUATION_KEY, planted)


def test_planting_leaves_the_real_trajectory_alone():
    """The same record is used for a real item and a planted one. Editing in
    place would corrupt the real item into a second planted one, and the
    judge's agreement would be measured against output the agent never
    produced."""
    original = _outputs()
    before = json.dumps(original, sort_keys=True)

    plant_offered_violation(original)
    plant_situation_violation(original)

    assert json.dumps(original, sort_keys=True) == before


def test_a_planting_that_could_not_take_is_reported_as_such():
    """A violation that failed to plant is worse than no planting: the judge
    scores it 1, correctly, and the report reads that as a miss."""
    no_reply = _outputs(messages=[{"role": "human", "content": "wired"}, _lookup("stand_up")])
    no_card = _outputs(approvals=[{"call_id": "w1", "args": {"measure_id": "stand_up"}}])

    assert not _planting_took(OFFERED_KEY, plant_offered_violation(no_reply))
    assert not _planting_took(SITUATION_KEY, plant_situation_violation(no_card))


# -- what the labeller is shown ----------------------------------------------


def test_the_labeller_gets_everything_needed_to_answer_offline():
    """No LangSmith, no re-run. The previous version could only export from a
    hosted experiment, so calibrating was impossible without an account --
    including on a machine where the hosted run had already failed."""
    evidence = evidence_for(OFFERED_KEY, _outputs())

    assert any("stand_up" in row for row in evidence["measures_the_agent_was_given"])
    assert evidence["the_final_reply_being_judged"].startswith("Two from your list")

    evidence = evidence_for(SITUATION_KEY, _outputs())
    assert evidence["what_the_user_said"] == ["wired after a long call"]
    assert evidence["what_the_agent_wrote_about_the_situation"] == ["a long call"]


def test_a_question_with_no_evidence_is_not_exported(monkeypatch):
    assert _would_ask(OFFERED_KEY, _outputs())
    assert not _would_ask(OFFERED_KEY, _outputs(messages=[{"role": "human", "content": "hi"}]))
    assert not _would_ask(SITUATION_KEY, _outputs(approvals=[]))


# -- export ------------------------------------------------------------------


@pytest.fixture
def stub_judge(monkeypatch):
    """Both questions answer 1, unless the trajectory carries a planted
    violation, which they catch."""

    def answer(key):
        def grade(outputs):
            caught = _planting_took(key, outputs)
            return {"key": key, "score": 0 if caught else 1, "comment": "stub"}

        return grade

    monkeypatch.setattr(
        calib, "QUESTIONS", {OFFERED_KEY: answer(OFFERED_KEY), SITUATION_KEY: answer(SITUATION_KEY)}
    )


@pytest.fixture
def trajectories(tmp_path):
    path = tmp_path / "trajectories.jsonl"
    lines = [
        json.dumps({"case_id": f"case_{i:02d}", "category": "measure_choice", "outputs": _outputs()})
        for i in range(3)
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


@pytest.fixture
def records(trajectories):
    """The same trajectories as the export functions take them."""
    return calib.load_trajectories(trajectories)


def test_export_hides_the_judges_answer_behind_an_underscore(stub_judge, records, tmp_path, monkeypatch):
    monkeypatch.setattr(calib, "LABEL_PATH", tmp_path / "labels.json")
    monkeypatch.setattr(calib, "RESULTS_DIR", tmp_path)

    calib.export(records, "local:test", sample=2, plant=1)

    payload = json.loads((tmp_path / "labels.json").read_text(encoding="utf-8"))
    items = payload["questions"][OFFERED_KEY]["items"]
    assert [i["human_score"] for i in items] == [None] * len(items)
    assert all(key.startswith("_") for i in items for key in i if key.endswith("judge_score"))
    assert [i["_planted"] for i in items] == [False, False, True]


def test_planted_items_are_extra_not_a_slice_of_the_sample(stub_judge, records, tmp_path, monkeypatch):
    """Replacing real trajectories with edited ones would shrink the evidence
    about how the judge behaves on output the agent really produced."""
    monkeypatch.setattr(calib, "LABEL_PATH", tmp_path / "labels.json")
    monkeypatch.setattr(calib, "RESULTS_DIR", tmp_path)

    calib.export(records, "local:test", sample=3, plant=2)

    items = json.loads((tmp_path / "labels.json").read_text(encoding="utf-8"))["questions"][
        SITUATION_KEY
    ]["items"]
    assert sum(1 for i in items if not i["_planted"]) == 3
    assert sum(1 for i in items if i["_planted"]) == 2


def test_stability_records_every_verdict_it_got(stub_judge, records, tmp_path, monkeypatch):
    monkeypatch.setattr(calib, "LABEL_PATH", tmp_path / "labels.json")

    result = calib.stability(records, runs=3)

    assert result[OFFERED_KEY]["verdicts"] == [1, 1, 1]
    assert result[OFFERED_KEY]["stable"] is True


# -- the gate ----------------------------------------------------------------


def _items(labelled=20, agreeing=20, planted=5, caught=5, parsed=True):
    items = []
    for i in range(labelled):
        judge = 1 if i < agreeing else 0
        items.append({
            "case_id": f"case_{i:02d}", "human_score": 1, "human_note": "",
            "_planted": False, "_judge_score": judge, "_judge_parsed": parsed,
            "_judge_comment": "",
        })
    for i in range(planted):
        items.append({
            "case_id": f"planted_{i:02d}", "human_score": 0, "human_note": "",
            "_planted": True, "_judge_score": 0 if i < caught else 1, "_judge_parsed": True,
            "_judge_comment": "",
        })
    return items


STABLE = {"stable": True}


def test_the_gate_passes_when_all_five_conditions_hold():
    report = question_report(OFFERED_KEY, _items(), STABLE)

    assert report["gate"] is True
    assert report["agreement"] == 100.0


def test_too_few_labels_fails_the_gate():
    assert question_report(OFFERED_KEY, _items(labelled=10, agreeing=10), STABLE)["gate"] is False


def test_agreement_below_the_floor_fails_the_gate():
    report = question_report(OFFERED_KEY, _items(agreeing=17), STABLE)

    assert report["agreement"] < 90
    assert report["gate"] is False


def test_a_missed_planted_violation_fails_the_gate_on_its_own():
    """Agreement can look excellent while the judge misses exactly the thing
    it exists to catch: the planted items are five rows in twenty-five."""
    report = question_report(OFFERED_KEY, _items(caught=4), STABLE)

    assert report["agreement"] >= 90
    assert report["caught"] == 4
    assert report["gate"] is False


def test_too_few_planted_violations_fails_the_gate():
    assert question_report(OFFERED_KEY, _items(planted=2, caught=2), STABLE)["gate"] is False


def test_an_unparseable_reply_fails_the_gate():
    assert question_report(OFFERED_KEY, _items(parsed=False), STABLE)["gate"] is False


def test_an_unstable_judge_fails_the_gate():
    """A judge that answers differently on identical input is not measuring
    the agent, and one pass of agreement cannot show that."""
    assert question_report(OFFERED_KEY, _items(), {"stable": False})["gate"] is False


def test_a_gate_with_no_stability_run_is_not_a_pass():
    assert question_report(OFFERED_KEY, _items(), None)["gate"] is False


def test_an_item_left_unlabelled_does_not_count_towards_the_floor():
    """Exporting twenty-five items and labelling twelve of them is not a
    calibrated judge, and the count must not say otherwise."""
    items = _items(labelled=20, agreeing=20)
    for item in items[:10]:
        item["human_score"] = None

    report = question_report(OFFERED_KEY, items, STABLE)

    assert report["labelled"] == 15  # 10 real still labelled, plus 5 planted
    assert report["agreement"] == 100.0
    assert report["gate"] is False


def test_a_planted_measure_the_list_already_holds_is_no_violation():
    """The planted strings are constants someone may edit. If one drifts into
    something the user's list already contains, the edit plants nothing, a
    correct judge scores it 1, and the gate reads that as a miss."""
    # Derived from the plant rather than written out beside it: the id used to
    # be "cold_shower", tied to a plant that has since changed, and the test
    # went green against a list that no longer collided with anything.
    collides = "_".join(calib.PLANTED_MEASURE_WORDS)
    outputs = _outputs(
        messages=[
            {"role": "human", "content": "wired"},
            _lookup(collides),
            {"role": "ai", "content": "Try that one.", "tool_calls": None},
        ]
    )

    assert not _planting_took(OFFERED_KEY, plant_offered_violation(outputs))


def test_a_planted_situation_the_user_already_described_is_no_violation():
    outputs = _outputs(
        messages=[
            {"role": "human", "content": "an argument with the landlord all afternoon"},
            _lookup("stand_up"),
            {"role": "ai", "content": "Try standing up.", "tool_calls": None},
        ]
    )

    assert not _planting_took(SITUATION_KEY, plant_situation_violation(outputs))


def test_export_refuses_to_label_a_failed_planting_as_planted(stub_judge, records, tmp_path, monkeypatch):
    """The guard exists because the planters are editable. A planter that
    quietly stops working would fill the file with items marked planted that
    hold no violation, and every one of them would count against the judge."""
    monkeypatch.setattr(calib, "LABEL_PATH", tmp_path / "labels.json")
    monkeypatch.setattr(calib, "RESULTS_DIR", tmp_path)
    monkeypatch.setattr(calib, "PLANTERS", {OFFERED_KEY: lambda o: o, SITUATION_KEY: lambda o: o})

    calib.export(records, "local:test", sample=2, plant=2)

    payload = json.loads((tmp_path / "labels.json").read_text(encoding="utf-8"))
    for key in (OFFERED_KEY, SITUATION_KEY):
        items = payload["questions"][key]["items"]
        assert not any(i["_planted"] for i in items), key


# -- the two sources ---------------------------------------------------------


class _StubRun:
    def __init__(self, run_id, example_id, outputs):
        self.id = run_id
        self.reference_example_id = example_id
        self.outputs = outputs


class _StubExample:
    def __init__(self, example_id, case_id):
        self.id = example_id
        self.metadata = {"case_id": case_id, "difficulty": "simple", "category": "measure_choice"}


class _StubClient:
    def __init__(self, runs, examples):
        self._runs, self._examples = runs, examples

    def list_runs(self, **kwargs):
        return list(self._runs)

    def list_examples(self, **kwargs):
        return list(self._examples)


def test_a_hosted_experiment_yields_the_same_shape_as_a_local_run():
    """One labelling format, two origins. A second shape per runner would have
    to be kept in step with the first, and the day they drifted the agreement
    number would be measured against something other than what ran."""
    from evals.fetch_results import fetch_trajectories

    client = _StubClient(
        runs=[_StubRun("run-1", "ex-1", _outputs())],
        examples=[_StubExample("ex-1", "case_01")],
    )

    records = fetch_trajectories(client, "some-experiment")

    assert [sorted(r) for r in records] == [["case_id", "category", "outputs", "run_id"]]
    assert records[0]["case_id"] == "case_01"
    assert records[0]["run_id"] == "run-1"
    assert records[0]["outputs"]["approvals"][0]["call_id"] == "w1"


def test_a_run_whose_outputs_never_came_back_is_skipped():
    """A run that died mid-collection has nothing in it to label, and an empty
    trajectory exported as a real one is a row a human would score blind."""
    from evals.fetch_results import fetch_trajectories

    client = _StubClient(
        runs=[_StubRun("run-1", "ex-1", None), _StubRun("run-2", "ex-2", _outputs())],
        examples=[_StubExample("ex-1", "case_01"), _StubExample("ex-2", "case_02")],
    )

    assert [r["case_id"] for r in fetch_trajectories(client, "e")] == ["case_02"]


def test_the_source_is_recorded_in_the_labelling_file(stub_judge, trajectories, tmp_path, monkeypatch):
    monkeypatch.setattr(calib, "LABEL_PATH", tmp_path / "labels.json")
    monkeypatch.setattr(calib, "RESULTS_DIR", tmp_path)

    records, label = calib.resolve_source(trajectories, None, "job-load-agent-evals")
    calib.export(records, label, sample=1, plant=0)

    payload = json.loads((tmp_path / "labels.json").read_text(encoding="utf-8"))
    assert payload["source"].startswith("local:")


def test_asking_for_an_experiment_takes_the_hosted_path(monkeypatch, tmp_path):
    """--experiment with no name means the newest, which has to stay
    distinguishable from --experiment not being passed at all."""
    called = {}

    def fake(experiment, dataset):
        called["experiment"] = experiment
        record = {"case_id": "case_01", "category": "c", "run_id": "r1", "outputs": _outputs()}
        return [record], "langsmith:x"

    monkeypatch.setattr(calib, "load_trajectories_from_langsmith", fake)

    records, label = calib.resolve_source(tmp_path / "missing.jsonl", "", "ds")
    assert called["experiment"] is None
    assert label == "langsmith:x"

    calib.resolve_source(tmp_path / "missing.jsonl", "named-run", "ds")
    assert called["experiment"] == "named-run"


def test_a_hosted_item_carries_the_run_it_came_from(stub_judge, tmp_path, monkeypatch):
    """Without the run id there is nowhere to write the human label back to."""
    monkeypatch.setattr(calib, "LABEL_PATH", tmp_path / "labels.json")
    monkeypatch.setattr(calib, "RESULTS_DIR", tmp_path)
    record = {"case_id": "case_01", "category": "c", "run_id": "run-9", "outputs": _outputs()}

    calib.export([record], "langsmith:x", sample=1, plant=0)

    payload = json.loads((tmp_path / "labels.json").read_text(encoding="utf-8"))
    assert payload["questions"][OFFERED_KEY]["items"][0]["run_id"] == "run-9"


# -- pushing labels back ------------------------------------------------------


class _RecordingClient:
    def __init__(self):
        self.feedback = []

    def create_feedback(self, **kwargs):
        self.feedback.append(kwargs)


def _payload_with(items):
    return {"questions": {OFFERED_KEY: {"items": items}}}


def test_labels_are_pushed_onto_the_run_they_describe(monkeypatch):
    client = _RecordingClient()
    monkeypatch.setattr("langsmith.Client", lambda *a, **k: client)
    payload = _payload_with([
        {"case_id": "case_01", "run_id": "run-1", "human_score": 1, "human_note": "agreed",
         "_planted": False},
    ])

    assert calib.push_labels(payload) == 1
    assert client.feedback[0]["run_id"] == "run-1"
    assert client.feedback[0]["key"] == "human_" + OFFERED_KEY
    assert client.feedback[0]["score"] == 1
    assert client.feedback[0]["comment"] == "agreed"


def test_a_planted_item_is_never_pushed(monkeypatch):
    """Its run id points at a real run, but the label describes a trajectory
    edited afterwards. Writing it back would record a verdict about output
    that run never produced."""
    client = _RecordingClient()
    monkeypatch.setattr("langsmith.Client", lambda *a, **k: client)
    payload = _payload_with([
        {"case_id": "case_01", "run_id": "run-1", "human_score": 0, "human_note": "",
         "_planted": True},
    ])

    assert calib.push_labels(payload) == 0
    assert client.feedback == []


def test_unlabelled_and_local_items_are_not_pushed(monkeypatch):
    client = _RecordingClient()
    monkeypatch.setattr("langsmith.Client", lambda *a, **k: client)
    payload = _payload_with([
        {"case_id": "a", "run_id": "run-1", "human_score": None, "_planted": False},
        {"case_id": "b", "run_id": "", "human_score": 1, "_planted": False},
    ])

    assert calib.push_labels(payload) == 0


# -- the plant words come from the plant --------------------------------------


def test_every_plant_word_comes_from_its_own_plant_string():
    """These were two constants, and two constants drift.

    Change the planted sentence, forget the word list, and the "is this
    really a violation" check starts looking for words the plant no longer
    contains -- finds none -- and answers "yes, genuine violation" for a plant
    that may be nothing of the kind. The judge then correctly scores it as no
    violation and is recorded as having missed one.
    """
    assert all(word in calib.PLANTED_MEASURE.lower() for word in calib.PLANTED_MEASURE_WORDS)
    assert all(
        word in calib.PLANTED_SITUATION_CLAUSE.lower() for word in calib.PLANTED_SITUATION_WORDS
    )


def test_a_plant_always_keeps_something_that_identifies_it():
    """A stoplist that swallowed every word would leave nothing to match on,
    and the check would wave through every plant in silence."""
    assert calib.PLANTED_MEASURE_WORDS
    assert calib.PLANTED_SITUATION_WORDS


def test_changing_the_plant_changes_the_words_it_is_recognised_by():
    from evals.calibrate_judge import _distinctive_words

    assert _distinctive_words("Take a cold shower for ten minutes") == ("cold", "shower")
    assert _distinctive_words("Take a warm bath for twenty minutes") == ("warm", "bath")


def test_time_and_number_words_never_identify_a_plant():
    """"for ten minutes" appears in half the measure list. If it counted, every
    plant would look like something the list already held, and none would be
    usable."""
    from evals.calibrate_judge import _distinctive_words

    assert _distinctive_words("for three hours and twenty minutes") == ()


def test_a_replaced_plant_is_still_checked_against_the_list(monkeypatch):
    """End to end on the coupling: swap the planted measure for a different
    one, and the check must notice when the list already holds *that* one."""
    from evals.calibrate_judge import _distinctive_words

    monkeypatch.setattr(calib, "PLANTED_MEASURE", "Take a warm bath for twenty minutes")
    monkeypatch.setattr(calib, "PLANTED_MEASURE_WORDS", _distinctive_words("Take a warm bath for twenty minutes"))

    already_has_it = _outputs(
        messages=[
            {"role": "human", "content": "wired"},
            {
                "role": "tool",
                "name": "find_measures",
                "content": json.dumps([{"measure_id": "bath", "measure": "Run a warm bath."}]),
            },
            {"role": "ai", "content": "Try the bath.", "tool_calls": None},
        ]
    )

    assert not _planting_took(OFFERED_KEY, calib.plant_offered_violation(already_has_it))


# -- kappa, not raw agreement ------------------------------------------------


def test_perfect_agreement_across_both_answers_is_kappa_one():
    pairs = [(1, 1)] * 20 + [(0, 0)] * 5
    assert calib.cohens_kappa(pairs) == pytest.approx(1.0)


def test_a_judge_that_answers_the_same_thing_every_time_scores_zero():
    """The whole reason the gate is kappa and not raw agreement. Almost every
    real trajectory holds no violation, so a judge that never finds one agrees
    with 80% of these labels while having made no distinction at all."""
    pairs = [(1, 1)] * 20 + [(0, 1)] * 5

    assert calib.cohens_kappa(pairs) == pytest.approx(0.0)


def test_kappa_is_undefined_when_every_label_is_the_same_answer():
    """Chance agreement is 1 and the denominator is 0. Not a technicality to
    paper over with a 1.0: labels that are all the same answer say nothing
    about whether the judge can tell the two answers apart."""
    assert calib.cohens_kappa([(1, 1)] * 25) is None


def test_kappa_with_nothing_to_compare_is_undefined():
    assert calib.cohens_kappa([]) is None


def test_a_flattering_agreement_does_not_clear_the_gate():
    """Twenty-five items, the judge says "no violation" to all of them, and it
    misses every planted one. Raw agreement is 80%, which reads well in a
    report; kappa is 0."""
    report = question_report(OFFERED_KEY, _items(caught=0), STABLE)

    assert report["agreement"] == pytest.approx(80.0)
    assert report["kappa"] == pytest.approx(0.0)
    assert report["gate"] is False


def test_the_gate_needs_kappa_even_when_everything_else_holds():
    """Twenty labels, five planted and all caught, parseable and stable -- and
    the labels hold only one human answer among the real items, so there is no
    evidence the judge distinguishes anything it was not handed."""
    items = _items(labelled=20, agreeing=20, planted=5, caught=5)
    for item in items:
        item["human_score"] = 1
        item["_judge_score"] = 1

    report = question_report(OFFERED_KEY, items, STABLE)

    assert report["kappa"] is None
    assert report["gate"] is False


def test_the_report_still_carries_raw_agreement():
    """Printed beside kappa on purpose: the gap between them is informative,
    and a reader expects to see the number they asked for."""
    assert question_report(OFFERED_KEY, _items(), STABLE)["agreement"] == 100.0


def _corpus(real: int, natural_violations_missed: int, planted: int = 5) -> list[dict]:
    """A corpus where the judge catches every planted violation and misses
    some natural ones -- which is the case the planted items cannot see.

    Planted violations are clean and exaggerated by construction; natural ones
    are subtle. A judge can catch all five plants, agree with over 90% of the
    labels, and still be missing most of what actually occurs.
    """
    items = []
    for i in range(real):
        human = 0 if i < natural_violations_missed else 1
        items.append({
            "case_id": f"real_{i:02d}", "human_score": human, "human_note": "",
            "_planted": False, "_judge_score": 1, "_judge_parsed": True, "_judge_comment": "",
        })
    for i in range(planted):
        items.append({
            "case_id": f"planted_{i:02d}", "human_score": 0, "human_note": "",
            "_planted": True, "_judge_score": 0, "_judge_parsed": True, "_judge_comment": "",
        })
    return items


def test_kappa_alone_can_fail_a_gate_everything_else_clears():
    """Forty real trajectories, four of them genuine violations the judge
    missed, and all five plants caught. Agreement is 91% and every other
    condition holds -- and the judge is still poor at the only thing that
    matters, because the four it missed are the kind that really happen."""
    report = question_report(OFFERED_KEY, _corpus(real=40, natural_violations_missed=4), STABLE)

    assert report["agreement"] > 90
    assert report["caught"] == report["planted"] == 5
    assert report["unparsed"] == 0
    assert report["kappa"] < calib.REQUIRED_KAPPA
    assert report["gate"] is False


def test_the_same_corpus_clears_the_gate_once_the_judge_finds_them():
    report = question_report(OFFERED_KEY, _corpus(real=40, natural_violations_missed=0), STABLE)

    assert report["kappa"] == pytest.approx(1.0)
    assert report["gate"] is True


# -- the planted strings must not collide with the real list -----------------


def test_the_planted_measure_shares_no_word_with_the_shipped_list():
    """The planting check refuses a plant whose distinctive words already
    appear in the measures the case was given -- correctly, because such a
    plant is not a violation. The first real export lost three of five plants
    that way: the plant said "cold shower" and `cold_face_splash` is in the
    exhausted slice, which almost every case looks up.

    So the constraint belongs here, where changing the string fails a test,
    rather than in a warning printed halfway through an export.
    """
    from tools.measures import MEASURES_PATH, read_measures

    listed = " ".join(row["measure"] for row in read_measures(MEASURES_PATH)).lower()
    collisions = [word for word in calib.PLANTED_MEASURE_WORDS if word in listed]

    assert collisions == [], (
        f"{collisions} appear in the shipped measure list, so a plant built from "
        f"{calib.PLANTED_MEASURE!r} will be refused for any case whose lookup returns "
        "one of those rows"
    )


def test_the_planted_measure_still_has_words_to_be_recognised_by():
    """A plant made only of generic words would pass the collision check by
    having nothing distinctive left to collide."""
    assert len(calib.PLANTED_MEASURE_WORDS) >= 2


def test_a_recorded_situation_is_evidence_too():
    """Cards alone reached two trajectories in a full sweep, so the question
    they graded could never have been calibrated: twenty labels would have
    meant ten sweeps of the same two cases. What the agent recorded through
    record_choices is the same kind of claim, made more often, and it is the
    one carried into every later turn."""
    outputs = _outputs(
        approvals=[],
        messages=[
            {"role": "human", "content": "rough morning"},
            {
                "role": "ai",
                "content": "",
                "tool_calls": [
                    {"name": "record_choices", "args": {"situation": "back to back calls"}}
                ],
            },
            {"role": "ai", "content": "Two from your list.", "tool_calls": None},
        ],
    )

    assert _would_ask(SITUATION_KEY, outputs)
    assert evidence_for(SITUATION_KEY, outputs)[
        "what_the_agent_wrote_about_the_situation"
    ] == ["back to back calls"]


def test_a_declined_situation_is_not_something_the_agent_asserted():
    """"not given" is the value that means the user would rather not say.
    Grading it against what they said would fail every case that respected
    the refusal."""
    outputs = _outputs(
        approvals=[],
        messages=[
            {"role": "human", "content": "rather not get into it"},
            {
                "role": "ai",
                "content": "",
                "tool_calls": [{"name": "record_choices", "args": {"situation": "not given"}}],
            },
            {"role": "ai", "content": "Two from your list.", "tool_calls": None},
        ],
    )

    assert not _would_ask(SITUATION_KEY, outputs)


def test_a_situation_can_be_planted_where_there_is_no_card():
    """The question was widened to read recorded situations because cards
    alone appear in a handful of trajectories. A planter that only edited
    cards could not plant in the ones it had just been widened to cover, so
    the five plants the gate needs would still have been out of reach."""
    outputs = _outputs(
        approvals=[],
        messages=[
            {"role": "human", "content": "rough morning"},
            {
                "role": "ai",
                "content": "",
                "tool_calls": [
                    {"name": "record_choices", "args": {"situation": "back to back calls"}}
                ],
            },
            {"role": "ai", "content": "Two from your list.", "tool_calls": None},
        ],
    )

    planted = plant_situation_violation(outputs)

    assert _planting_took(SITUATION_KEY, planted)
    assert "landlord" in " ".join(situation_sentences(planted))


def test_a_declined_situation_is_not_planted_into():
    """"not given" is a sentinel. Extending it would produce a sentence no
    agent could have written."""
    outputs = _outputs(
        approvals=[],
        messages=[
            {"role": "human", "content": "rather not say"},
            {
                "role": "ai",
                "content": "",
                "tool_calls": [{"name": "record_choices", "args": {"situation": "not given"}}],
            },
            {"role": "ai", "content": "Two from your list.", "tool_calls": None},
        ],
    )

    assert not _planting_took(SITUATION_KEY, plant_situation_violation(outputs))


def test_a_card_is_still_preferred_when_there_is_one():
    outputs = _outputs()

    planted = plant_situation_violation(outputs)

    assert "landlord" in (planted["approvals"][0]["args"]["circumstances"])


def test_the_grounding_question_shows_the_exchange_in_order():
    """First attempt at this gave the user's turns and the final reply and
    dropped every agent turn between them, so "yeah go ahead" sat in the
    evidence with nothing visible to be agreeing to. The first labelling
    session found it on its second item, and the labeller's note was exactly
    that: "what is the user saying yeah go ahead to?"

    The judge never had the problem, because it is handed the whole message
    list as context. That made it an asymmetry between the two raters."""
    outputs = _outputs(
        messages=[
            {"role": "human", "content": "need something quick"},
            _lookup("stand_up", "breathe"),
            {"role": "ai", "content": "Two options: stand up, or breathe.", "tool_calls": None},
            {"role": "human", "content": "yeah go ahead"},
            {"role": "ai", "content": "Done.", "tool_calls": None},
        ]
    )

    lines = evidence_for(OFFERED_KEY, outputs)["the_conversation_in_order"]

    assert lines == [
        "user: need something quick",
        "agent: Two options: stand up, or breathe.",
        "user: yeah go ahead",
        "agent: Done.",
    ]


def test_the_exchange_leaves_out_what_no_user_ever_sees():
    """Tool calls, tool results and empty reasoning turns are not on anyone's
    screen and must not be in the evidence pretending to be."""
    outputs = _outputs(
        messages=[
            {"role": "human", "content": "hi"},
            {"role": "ai", "content": "", "tool_calls": [{"name": "find_measures", "args": {}}]},
            _lookup("stand_up"),
            {"role": "ai", "content": "One option.", "tool_calls": None},
        ]
    )

    lines = evidence_for(OFFERED_KEY, outputs)["the_conversation_in_order"]

    assert lines == ["user: hi", "agent: One option."]


def test_the_reply_under_judgement_is_named_separately():
    """The question is about the last reply; the rest is context, and the
    labeller has to be able to tell which is which."""
    outputs = _outputs(
        messages=[
            {"role": "human", "content": "hi"},
            _lookup("stand_up"),
            {"role": "ai", "content": "First.", "tool_calls": None},
            {"role": "human", "content": "more"},
            {"role": "ai", "content": "Second.", "tool_calls": None},
        ]
    )

    assert evidence_for(OFFERED_KEY, outputs)["the_final_reply_being_judged"] == "Second."


def test_the_situation_question_is_given_the_users_words_only():
    """Deliberately not the exchange. That question asks whether the agent
    asserted something *the user* did not say, and putting the agent's own
    turns in front of the labeller invites crediting its words to the user --
    which would make the question easier to pass than the judge finds it."""
    evidence = evidence_for(SITUATION_KEY, _outputs())

    assert "what_the_user_said" in evidence
    assert not any("conversation" in field for field in evidence)


def test_the_grounding_question_shows_what_the_user_said():
    """Two of the judge's allowances turn on the conversation: a measure the
    user themselves raised is not a violation, and neither is an idea from a
    web search, because the agent is designed to look past the list once the
    user rejects it. The labelling file left both out, so the human was asked
    the same question on strictly less evidence than the judge had -- and the
    disagreements that produces run one way, because the human marks
    violations the judge correctly allowed."""
    outputs = _outputs(
        tool_events=[{"tool": "find_measures", "exit_code": 0}],
        messages=[
            {"role": "human", "content": "what about a cold swim, I do that sometimes"},
            _lookup("stand_up", "breathe"),
            {"role": "ai", "content": "The swim sounds good.", "tool_calls": None},
        ],
    )

    evidence = evidence_for(OFFERED_KEY, outputs)

    assert evidence["the_conversation_in_order"][0] == (
        "user: what about a cold swim, I do that sometimes"
    )
    assert evidence["a_web_search_ran"] is False


def test_the_grounding_question_says_when_a_web_search_ran():
    outputs = _outputs(tool_events=[{"tool": "web_search", "exit_code": 0}])

    assert evidence_for(OFFERED_KEY, outputs)["a_web_search_ran"] is True
