"""evals/summarize_runs.py -- every run recounted, not just the last.

The reason it exists: a run_local report covers its last run only, and on
2026-09-27 that hid a zero-tolerance miss in two earlier runs behind a
reported 26/26. These tests check the recount sees every run, counts the two
task-completion directions apart, and leaves out only the checks that fail by
construction without the lookup.
"""
import json

import evals.summarize_runs as summary_module
from evals.summarize_runs import majority_file_agrees, summarize, tautological_keys

CASES = {
    "writes": {"id": "writes", "expect": {"reached_write": True, "tools_called": ["find_measures"],
                                          "lookup_rows": {"energy": "wired"}}},
    "stays_quiet": {"id": "stays_quiet", "expect": {"reached_write": False}},
}


def _run(ok=True, **extra):
    return {"crisis_gate_correctness": True, "no_fabrication": True, "expect_reached_write": ok, **extra}


def test_only_the_lookups_own_checks_are_automatic_failures():
    assert tautological_keys(CASES["writes"]) == {"expect_lookup_rows", "expect_tools_called"}
    assert tautological_keys(CASES["stays_quiet"]) == set()
    assert tautological_keys({"expect": {"tools_called": ["log_checkin"]}}) == set()


def test_a_miss_in_an_earlier_run_is_counted():
    """The report's run would have been the third; the miss is in the first."""
    scores = {"writes": [_run(no_fabrication=False), _run(), _run()], "stays_quiet": [_run()] * 3}

    result = summarize(scores, CASES)

    assert result["zero_tolerance"]["no_fabrication"] == (5, 6)


def test_majority_and_every_run_are_different_numbers():
    scores = {"writes": [_run(ok=False), _run(), _run()], "stays_quiet": [_run()] * 3}

    result = summarize(scores, CASES)

    assert result["majority"] == 2
    assert result["all_runs"] == 1


def test_task_completion_counts_both_directions_apart():
    scores = {"writes": [_run(ok=False), _run(), _run()], "stays_quiet": [_run()] * 3}

    result = summarize(scores, CASES)

    assert result["completion"][True] == (2, 3)
    assert result["completion"][False] == (3, 3)


def test_excluding_automatic_failures_rescues_only_those():
    """A case failing only its lookup checks passes once they are left out;
    one failing something else still fails."""
    lookup_only = {**_run(), "expect_lookup_rows": False, "expect_tools_called": False}
    lookup_and_more = {**lookup_only, "expect_log_rows": False}

    assert summarize({"writes": [lookup_only] * 3}, CASES, drop_tautological=True)["majority"] == 1
    assert summarize({"writes": [lookup_and_more] * 3}, CASES, drop_tautological=True)["majority"] == 0
    assert summarize({"writes": [lookup_only] * 3}, CASES)["majority"] == 0


def test_the_recount_is_checked_against_the_runs_own_majority_file(tmp_path):
    trajectories = tmp_path / "trajectories-20260927-141910.jsonl"
    trajectories.write_text("", encoding="utf-8")
    scores = {"writes": [_run(), _run(), _run(ok=False)]}

    assert majority_file_agrees(trajectories, scores) is None
    majority = tmp_path / "majority-20260927-141910.json"
    majority.write_text(json.dumps([{"case_id": "writes", "passed": 2}]), encoding="utf-8")
    assert majority_file_agrees(trajectories, scores) is True
    majority.write_text(json.dumps([{"case_id": "writes", "passed": 3}]), encoding="utf-8")
    assert majority_file_agrees(trajectories, scores) is False


def test_the_rendered_summary_names_every_zero_tolerance_miss(monkeypatch, tmp_path):
    path = tmp_path / "trajectories-20260927-141910.jsonl"
    path.write_text("", encoding="utf-8")
    monkeypatch.setattr(summary_module, "load_cases", lambda: list(CASES.values()))
    monkeypatch.setattr(summary_module, "rescore", lambda p, cases: {
        "writes": [_run(), _run(no_fabrication=False), _run()], "stays_quiet": [_run()] * 3})

    text = summary_module.render(path)

    assert "writes run 2: no_fabrication" in text
    assert "| `no_fabrication` | 5/6 |" in text


# -- a result is recounted against the dataset it was scored with ------------


def _snapshot(tmp_path, readme: str | None):
    path = tmp_path / "trajectories-20260927-141910.jsonl"
    path.write_text("", encoding="utf-8")
    if readme is not None:
        (tmp_path / "README.md").write_text(readme, encoding="utf-8")
    return path


def test_a_published_snapshot_names_the_commit_it_was_made_from(tmp_path):
    path = _snapshot(tmp_path, "# Eval runs snapshot\n\nCreated 2026-09-27 (UTC) from commit `0d81185d3086`.\n")
    assert summary_module.recorded_commit(path) == "0d81185d3086"


def test_a_file_without_a_snapshot_readme_has_no_recorded_commit(tmp_path):
    assert summary_module.recorded_commit(_snapshot(tmp_path, None)) is None
    assert summary_module.recorded_commit(_snapshot(tmp_path, "# some other readme\n")) is None


def _render_recording(monkeypatch, path, **kwargs):
    """Render with the dataset sources stubbed; return the text and which one was used."""
    used = []
    monkeypatch.setattr(summary_module, "cases_at", lambda commit: used.append(commit) or dict(CASES))
    monkeypatch.setattr(summary_module, "load_cases", lambda: used.append("current") or list(CASES.values()))
    monkeypatch.setattr(summary_module, "rescore", lambda p, cases: {"writes": [_run()] * 3})
    return summary_module.render(path, **kwargs), used


def test_a_snapshot_is_recounted_against_the_dataset_at_its_commit(monkeypatch, tmp_path):
    """v1.1 added `reaches: listen` to case 35. Recounted against the current
    dataset, the published Step 7 comparison would move from 20/26 to 19/26 for
    the agent without the lookup -- a number that was right for the dataset it
    was scored with."""
    path = _snapshot(tmp_path, "Created 2026-09-27 (UTC) from commit `0d81185d3086`.\n")

    text, used = _render_recording(monkeypatch, path)

    assert used == ["0d81185d3086"]
    assert "Scored against the dataset at commit `0d81185`." in text


def test_an_explicit_commit_wins_and_no_readme_means_the_current_dataset(monkeypatch, tmp_path):
    path = _snapshot(tmp_path, "Created 2026-09-27 (UTC) from commit `0d81185d3086`.\n")
    _, used = _render_recording(monkeypatch, path, dataset_at="8ae43f5")
    assert used == ["8ae43f5"]

    other = tmp_path / "local"
    other.mkdir()
    text, used = _render_recording(monkeypatch, _snapshot(other, None))
    assert used == ["current"]
    assert "Scored against the current dataset." in text
