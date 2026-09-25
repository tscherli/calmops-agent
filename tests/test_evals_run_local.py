"""Tests for evals/run_local.py -- the LangSmith-free runner.

Fully stubbed: no OpenAI, no LangSmith, no network. What matters here is
that it feeds the target the same inputs a hosted run would, and that one
broken piece can't take down the whole run (the reason this module exists
is that a transient network failure destroyed a paid run's results).
"""
import evals.run_local as run_local_module
from evals.run_local import case_inputs, evaluate_case, run_local

_CASE = {
    "id": "case_01",
    "query": "Doing okay today.",
    "difficulty": "simple",
    "category": "light_checkin",
    "expected_behavior": "Logs directly.",
    "notes": "n/a",
    "reference_trajectory": [{"role": "human"}, {"role": "ai", "tool_calls": None}],
}


def _stub_outputs(*args, **kwargs):
    return {
        "messages": [
            {"role": "human", "content": "Doing okay today."},
            {"role": "ai", "content": "Nice.", "tool_calls": None},
        ],
        "crisis_intercepted": False,
    }


def test_case_inputs_match_the_hosted_examples_shape():
    """A local run and a hosted run must feed the target identical input,
    or the two reports aren't comparable."""
    from evals.create_dataset import case_to_example

    case = {**_CASE, "simulate_write_failure": True, "seed_entries": [{"mood": "bad"}]}

    assert case_inputs(case) == case_to_example(case)["inputs"]


def test_case_inputs_wraps_single_query_as_turns():
    assert case_inputs(_CASE)["turns"] == ["Doing okay today."]
    assert "seed_entries" not in case_inputs(_CASE)


def test_multiturn_case_keeps_its_turns():
    case = {**_CASE, "turns": ["one", "two"]}
    del case["query"]
    assert case_inputs(case)["turns"] == ["one", "two"]


def test_evaluate_case_collects_scores_from_every_evaluator(monkeypatch):
    monkeypatch.setattr(run_local_module, "run_case", _stub_outputs)
    monkeypatch.setattr(
        run_local_module,
        "EVALUATORS",
        (
            lambda outputs, reference_outputs: {"key": "alpha", "score": True},
            lambda outputs, reference_outputs: {"key": "beta", "score": False},
        ),
    )

    row = evaluate_case(_CASE)

    assert row["case_id"] == "case_01"
    assert row["scores"] == {"alpha": True, "beta": False}


def test_a_raising_evaluator_does_not_lose_the_case(monkeypatch):
    """The LLM judge is known to intermittently fail parsing; one bad
    evaluator must not discard an otherwise-good result."""

    def boom(outputs, reference_outputs):
        raise RuntimeError("judge parse failure")

    monkeypatch.setattr(run_local_module, "run_case", _stub_outputs)
    monkeypatch.setattr(
        run_local_module,
        "EVALUATORS",
        (lambda outputs, reference_outputs: {"key": "alpha", "score": True}, boom),
    )

    row = evaluate_case(_CASE)

    assert row["scores"]["alpha"] is True
    assert row["case_id"] == "case_01"


def test_a_failing_target_is_recorded_not_raised(monkeypatch):
    def boom(inputs, **kwargs):   # the runner also passes runs_log
        raise RuntimeError("model unavailable")

    monkeypatch.setattr(run_local_module, "run_case", boom)

    row = evaluate_case(_CASE)

    assert row["scores"] == {"target_error": False}
    assert "model unavailable" in row["error"]


def test_run_local_writes_a_report_and_preserves_dataset_order(monkeypatch, tmp_path):
    monkeypatch.setattr(run_local_module, "run_case", _stub_outputs)
    monkeypatch.setattr(
        run_local_module,
        "EVALUATORS",
        (lambda outputs, reference_outputs: {"key": "alpha", "score": True},),
    )
    monkeypatch.setattr(run_local_module, "_write_report", lambda report: tmp_path / "report.md")

    cases = [{**_CASE, "id": f"case_{i:02d}"} for i in range(5)]
    result = run_local(cases, workers=2)

    assert [r["case_id"] for r in result["rows"]] == [c["id"] for c in cases]


def test_seeded_measures_reach_the_target_the_same_way_in_both_runners():
    """A planted measure row only does anything if it travels in `inputs`:
    evaluate() hands the target function inputs and nothing else."""
    from evals.create_dataset import case_to_example

    case = {**_CASE, "seed_measures": [{"measure_id": "planted", "measure": "Do the thing."}]}

    assert case_inputs(case) == case_to_example(case)["inputs"]
    assert case_inputs(case)["seed_measures"] == case["seed_measures"]


def test_an_evaluator_that_scores_several_keys_lands_as_several_scores(monkeypatch):
    """The expect block is one evaluator answering for every field the case
    asserted. Unwrapped wrongly it arrives in the report as a single score
    called "results", and every assertion inside it stops being visible."""
    monkeypatch.setattr(run_local_module, "run_case", _stub_outputs)
    monkeypatch.setattr(
        run_local_module,
        "EVALUATORS",
        (
            lambda outputs, reference_outputs: {
                "results": [
                    {"key": "expect_log_rows", "score": True},
                    {"key": "expect_reached_write", "score": False},
                ]
            },
        ),
    )

    row = evaluate_case(_CASE)

    assert row["scores"] == {"expect_log_rows": True, "expect_reached_write": False}


def test_a_single_key_evaluator_still_lands_as_one_score(monkeypatch):
    monkeypatch.setattr(run_local_module, "run_case", _stub_outputs)
    monkeypatch.setattr(
        run_local_module,
        "EVALUATORS",
        (lambda outputs, reference_outputs: {"key": "no_fabrication", "score": True},),
    )

    assert evaluate_case(_CASE)["scores"] == {"no_fabrication": True}


def test_the_case_expect_block_reaches_the_evaluators(monkeypatch):
    """The assertions travel with the reference, not the inputs: the target
    must not be able to read what it is about to be graded on."""
    seen = {}

    def spy(outputs, reference_outputs):
        seen.update(reference_outputs)
        return {"key": "alpha", "score": True}

    monkeypatch.setattr(run_local_module, "run_case", _stub_outputs)
    monkeypatch.setattr(run_local_module, "EVALUATORS", (spy,))

    evaluate_case({**_CASE, "expect": {"reached_write": True}})

    assert seen["expect"] == {"reached_write": True}
    assert "expect" not in case_inputs({**_CASE, "expect": {"reached_write": True}})


# -- what a paid run leaves behind -------------------------------------------


def test_a_run_saves_its_trajectories_for_labelling(tmp_path):
    """Calibrating the judge means reading real trajectories by hand, and a
    run that has already been paid for should not have to be paid for again
    to get them back."""
    import json

    path = tmp_path / "trajectories.jsonl"
    rows = [
        {"case_id": "case_01", "category": "light_checkin", "scores": {}, "outputs": {"messages": [1]}},
        {"case_id": "case_02", "category": "crisis", "scores": {}, "outputs": {"messages": [2]}},
    ]

    assert run_local_module._save_trajectories(rows, path) == 2

    saved = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    assert [r["case_id"] for r in saved] == ["case_01", "case_02"]
    assert saved[0]["outputs"] == {"messages": [1]}


def test_a_case_that_never_ran_saves_no_trajectory(tmp_path):
    """A target error leaves the row with no outputs at all, and a blank line
    in the labelling source would be labelled as a real one."""
    path = tmp_path / "trajectories.jsonl"
    rows = [{"case_id": "case_01", "category": "crisis", "scores": {"target_error": False}}]

    assert run_local_module._save_trajectories(rows, path) == 0
    assert path.read_text(encoding="utf-8") == ""


def test_a_second_run_replaces_the_first_rather_than_joining_it(tmp_path):
    """Two runs in one file would be labelled as though they came from one
    agent, and the judge's agreement would be measured against a mixture."""
    path = tmp_path / "trajectories.jsonl"
    run_local_module._save_trajectories(
        [{"case_id": "old", "category": "c", "outputs": {"messages": []}}], path
    )
    run_local_module._save_trajectories(
        [{"case_id": "new", "category": "c", "outputs": {"messages": []}}], path
    )

    assert path.read_text(encoding="utf-8").count("case_id") == 1
    assert "old" not in path.read_text(encoding="utf-8")


def test_the_report_never_reads_the_trajectory_off_the_row(monkeypatch, tmp_path):
    """The outputs ride on the row so the runner can save them. If the report
    started reading them the two would be coupled, and a row without outputs
    (a target error) would break the report."""
    monkeypatch.setattr(run_local_module, "run_case", _stub_outputs)
    monkeypatch.setattr(
        run_local_module,
        "EVALUATORS",
        (lambda outputs, reference_outputs: {"key": "alpha", "score": True},),
    )
    monkeypatch.setattr(run_local_module, "_write_report", lambda report: tmp_path / "report.md")

    result = run_local_module.run_local([_CASE], workers=1)

    assert result["rows"][0]["outputs"]["crisis_intercepted"] is False
