"""Tests for evals/run_local.py -- the LangSmith-free runner.

Fully stubbed: no OpenAI, no LangSmith, no network. What matters here is
that it feeds the target the same inputs a hosted run would, and that one
broken piece can't take down the whole run (the reason this module exists
is that a transient network failure destroyed a paid run's results).
"""
import pytest

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


def test_a_run_never_overwrites_an_earlier_one(tmp_path):
    """Two runs in one file would be labelled as though they came from one
    agent -- and a second run *replacing* the first is how a paid-for
    seventy-eight trajectory sweep was lost. Same name, so it must refuse."""
    path = tmp_path / "trajectories-20260926-140315.jsonl"
    run_local_module._save_trajectories(
        [{"case_id": "old", "category": "c", "outputs": {"messages": []}}], path
    )

    with pytest.raises(FileExistsError):
        run_local_module._save_trajectories(
            [{"case_id": "new", "category": "c", "outputs": {"messages": []}}], path
        )

    assert "old" in path.read_text(encoding="utf-8")
    assert "new" not in path.read_text(encoding="utf-8")


def test_each_run_is_named_for_the_utc_moment_it_was_written(tmp_path):
    from datetime import datetime, timezone

    when = datetime(2026, 9, 26, 14, 3, 15, tzinfo=timezone.utc)

    assert run_local_module.trajectories_path(tmp_path, when).name == (
        "trajectories-20260926-140315.jsonl"
    )


def test_the_newest_run_is_found_by_its_name_not_its_file_time(tmp_path):
    """Copying a file changes its modification time and says nothing about
    when the run happened."""
    import os

    older = tmp_path / "trajectories-20260926-140315.jsonl"
    newer = tmp_path / "trajectories-20260926-161632.jsonl"
    newer.write_text("", encoding="utf-8")
    older.write_text("", encoding="utf-8")
    os.utime(older, (2_000_000_000, 2_000_000_000))  # touched long after
    (tmp_path / "trajectories.jsonl").write_text("", encoding="utf-8")  # the old fixed name

    assert run_local_module.latest_trajectories(tmp_path) == newer


def test_no_runs_means_no_newest(tmp_path):
    assert run_local_module.latest_trajectories(tmp_path) is None


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


# -- majority over several runs ----------------------------------------------


def _row(case_id, passed):
    """A row as evaluate_case returns it, scored so case_passed agrees."""
    return {
        "case_id": case_id,
        "difficulty": "simple",
        "category": "light_checkin",
        "scores": {"expect_reached_write": passed},
    }


def test_a_case_that_passes_every_run_passes():
    summary = run_local_module.majority(
        [[_row("a", True)], [_row("a", True)], [_row("a", True)]]
    )

    assert summary == [
        {"case_id": "a", "passed": 3, "runs": 3, "verdict": True, "split": False}
    ]


def test_two_of_three_is_a_pass_and_is_flagged_as_a_split():
    """The threshold is a pass in 2 of 3, and a case that does not agree with
    itself is the one to re-run -- a report showing only the majority hides
    exactly those."""
    summary = run_local_module.majority(
        [[_row("a", True)], [_row("a", False)], [_row("a", True)]]
    )

    assert summary[0]["verdict"] is True
    assert summary[0]["split"] is True


def test_one_of_three_fails():
    summary = run_local_module.majority(
        [[_row("a", True)], [_row("a", False)], [_row("a", False)]]
    )

    assert summary[0]["verdict"] is False
    assert summary[0]["split"] is True


def test_a_single_run_still_yields_a_verdict_and_no_split():
    summary = run_local_module.majority([[_row("a", True)]])

    assert summary[0] == {
        "case_id": "a", "passed": 1, "runs": 1, "verdict": True, "split": False
    }


def test_every_case_gets_its_own_row():
    summary = run_local_module.majority(
        [[_row("a", True), _row("b", False)], [_row("a", True), _row("b", True)]]
    )

    assert {r["case_id"] for r in summary} == {"a", "b"}
    assert {r["runs"] for r in summary} == {2}


def test_a_full_run_writes_one_dated_file_and_no_fixed_name(monkeypatch, tmp_path):
    """The fixed trajectories.jsonl was the file every run overwrote. Nothing
    writes it any more, so nothing can destroy a corpus by writing it."""
    monkeypatch.setattr(run_local_module, "run_case", _stub_outputs)
    monkeypatch.setattr(
        run_local_module,
        "EVALUATORS",
        (lambda outputs, reference_outputs: {"key": "alpha", "score": True},),
    )
    monkeypatch.setattr(run_local_module, "_write_report", lambda report: tmp_path / "report.md")

    run_local_module.run_local([_CASE], workers=1, trajectories_dir=tmp_path)

    written = sorted(p.name for p in tmp_path.glob("trajectories*.jsonl"))
    assert len(written) == 1
    assert written[0].startswith("trajectories-") and written[0] != "trajectories.jsonl"


def test_a_multi_run_leaves_its_majority_beside_its_trajectories(monkeypatch, tmp_path):
    """Same stamp on both, and nothing at the old fixed majority.json -- a
    three-case re-run replaced the full sweep's verdicts that way."""
    monkeypatch.setattr(run_local_module, "run_case", _stub_outputs)
    monkeypatch.setattr(
        run_local_module,
        "EVALUATORS",
        (lambda outputs, reference_outputs: {"key": "alpha", "score": True},),
    )
    monkeypatch.setattr(run_local_module, "_write_report", lambda report: tmp_path / "report.md")

    run_local_module.run_local([_CASE], workers=1, trajectories_dir=tmp_path, runs=2)

    [majority_file] = tmp_path.glob("majority-*.json")
    [trajectories_file] = tmp_path.glob("trajectories-*.jsonl")
    assert majority_file.stem.removeprefix("majority-") == trajectories_file.stem.removeprefix("trajectories-")
    assert not (tmp_path / "majority.json").exists()


def test_an_n1_run_marks_every_file_and_reaches_every_case(monkeypatch, tmp_path):
    """A reduced agent's corpus taken for a full one would calibrate the judge
    on an agent nobody ships -- so the mark is in the file names, where export
    and a person scanning the folder look, and the exclusion has to reach the
    target for every case, not stop at the runner."""
    seen = []

    def spy_run_case(*args, **kwargs):
        seen.append(tuple(kwargs.get("exclude_tools") or ()))
        return _stub_outputs()

    monkeypatch.setattr(run_local_module, "run_case", spy_run_case)
    monkeypatch.setattr(
        run_local_module,
        "EVALUATORS",
        (lambda outputs, reference_outputs: {"key": "alpha", "score": True},),
    )
    monkeypatch.setattr(run_local_module, "_write_report", lambda report: tmp_path / "report.md")

    run_local_module.run_local(
        [_CASE, {**_CASE, "id": "case_02"}], workers=2, trajectories_dir=tmp_path, runs=2,
        exclude_tools=["find_measures"],
    )

    assert seen and set(seen) == {("find_measures",)}
    [trajectories_file] = tmp_path.glob("trajectories-*.jsonl")
    [majority_file] = tmp_path.glob("majority-*.json")
    assert trajectories_file.name.endswith("-without-find_measures.jsonl")
    assert majority_file.name.endswith("-without-find_measures.json")


def test_the_newest_full_run_is_found_past_a_newer_reduced_one(tmp_path):
    """export starts a labelling round from the newest run; an N-1 run made
    afterwards must not become that round's corpus."""
    full = tmp_path / "trajectories-20260927-090000.jsonl"
    reduced = tmp_path / "trajectories-20260927-120000-without-find_measures.jsonl"
    for path in (full, reduced):
        path.write_text("", encoding="utf-8")

    assert run_local_module.latest_trajectories(tmp_path) == full


def test_a_row_with_no_outputs_is_not_written(tmp_path):
    """A case whose target raised has nothing to label."""
    path = tmp_path / "trajectories.jsonl"

    assert run_local_module._save_trajectories([{"case_id": "a", "category": "x"}], path) == 0
    assert path.read_text(encoding="utf-8") == ""


def test_the_row_carries_the_cases_usage_and_its_write_direction(monkeypatch):
    """The report reads cost from the row, never from the outputs -- so the
    runner has to lift it across, or every local report says "not recorded"."""
    usage = {"model_calls": 3, "cost_usd": 0.01, "latency_seconds": 2.0, "unpriced_models": []}
    monkeypatch.setattr(run_local_module, "run_case", lambda *a, **k: {**_stub_outputs(), "usage": usage})
    monkeypatch.setattr(run_local_module, "EVALUATORS", ())

    row = evaluate_case({**_CASE, "expect": {"reached_write": True}})

    assert row["usage"] == usage
    assert row["should_write"] is True
