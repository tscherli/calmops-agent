"""Every run of a sweep, recounted from its saved trajectories -- no model calls.

A run_local report covers only the last of its runs, and the trajectories
file does not store scores. So a zero-tolerance miss in run 1 or 2 is
invisible in the report: on 2026-09-27 the full agent's report said
no_fabrication 26/26 while the check had failed in 2 of 78 case-runs. The
deterministic checks are plain code over the outputs, so they can be run
again, offline, over all runs -- which is what this does. The judge's two
questions are left out: they need a model call.

    python -m evals.summarize_runs <trajectories.jsonl> [--n1 <trajectories-without-...jsonl>]
                                   [--dataset-at <commit>]

For an N-1 corpus it also reports the pass rate with the checks that fail by
construction once `find_measures` is gone left out (option A in
notes/BEST_PRACTICES.md): `lookup_rows`, and `tools_called` where it names
find_measures. They measure the removal, not its effect.

The recount is checked against the majority file the run wrote, when one
sits beside it: a recount that disagrees with the run is reported, not used.

A result belongs to the dataset it was scored with. A published snapshot is
recounted against the dataset at the commit its README records, so a later
change to a case -- v1.1 added `reaches: listen` to case 35 -- cannot quietly
move a published number. Anything else is scored against the current dataset,
and the summary says which.
"""
from __future__ import annotations

import argparse
import json
import re
import subprocess
from collections import defaultdict
from pathlib import Path

from evals.create_dataset import load_cases
from evals.evaluators import crisis_gate_evaluator, fabrication_evaluator, trajectory_evaluator, write_had_card
from evals.expectations import expectation_evaluator
from evals.outputs import LOOKUP_TOOL
from evals.run_experiment import ZERO_TOLERANCE_KEYS, case_passed
from evals.run_local import _feedback_items

DETERMINISTIC = (crisis_gate_evaluator, trajectory_evaluator, fabrication_evaluator, write_had_card,
                 expectation_evaluator)

REPO_ROOT = Path(__file__).resolve().parent.parent


def recorded_commit(trajectories: Path) -> str | None:
    """The commit a published snapshot says it was made from, or None.

    `publish_results` writes it into the snapshot's README and refuses to
    publish from a dirty tree, so the dataset at that commit is exactly the one
    the snapshot's numbers were scored with.
    """
    readme = trajectories.with_name("README.md")
    if not readme.exists():
        return None
    match = re.search(r"from commit `([0-9a-f]{7,40})`", readme.read_text(encoding="utf-8"))
    return match.group(1) if match else None


def cases_at(commit: str) -> dict[str, dict]:
    """The dataset as it was at a commit, read from git."""
    text = subprocess.run(
        ["git", "show", f"{commit}:evals/dataset.jsonl"],
        capture_output=True, check=True, cwd=REPO_ROOT,
    ).stdout.decode("utf-8")
    return {c["id"]: c for c in (json.loads(line) for line in text.splitlines() if line.strip())}


def tautological_keys(case: dict) -> set[str]:
    """The checks a case cannot pass once the lookup tool is gone."""
    expect = case.get("expect") or {}
    keys = set()
    if "lookup_rows" in expect:
        keys.add("expect_lookup_rows")
    if LOOKUP_TOOL in (expect.get("tools_called") or []):
        keys.add("expect_tools_called")
    return keys


def _reference(case: dict) -> dict:
    """The same reference evaluate_case builds (evals/run_local.py)."""
    reference = {"messages": case.get("reference_trajectory", []),
                 "expected_behavior": case["expected_behavior"], "category": case["category"]}
    if case.get("expect"):
        reference["expect"] = case["expect"]
    return reference


def rescore(path: Path, cases: dict[str, dict]) -> dict[str, list[dict]]:
    """Per case, one score dict per run, in run order."""
    by_case: dict[str, list[tuple[int, dict]]] = defaultdict(list)
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        scores: dict = {}
        for evaluator in DETERMINISTIC:
            result = evaluator(outputs=record["outputs"], reference_outputs=_reference(cases[record["case_id"]]))
            for item in _feedback_items(result):
                scores[item["key"]] = item.get("score")
        by_case[record["case_id"]].append((record.get("run_index") or 0, scores))
    return {case_id: [s for _, s in sorted(runs, key=lambda r: r[0])] for case_id, runs in by_case.items()}


def _passes(runs: list[dict], drop: set[str] = frozenset()) -> int:
    return sum(1 for s in runs if case_passed({k: v for k, v in s.items() if k not in drop}))


def summarize(scores: dict[str, list[dict]], cases: dict[str, dict], *, drop_tautological: bool = False) -> dict:
    drop = {c: (tautological_keys(cases[c]) if drop_tautological else set()) for c in scores}
    runs = max((len(r) for r in scores.values()), default=0)
    passes = {c: _passes(r, drop[c]) for c, r in scores.items()}
    zero = {}
    for key in ZERO_TOLERANCE_KEYS:
        values = [s.get(key) for r in scores.values() for s in r if s.get(key) is not None]
        zero[key] = (sum(1 for v in values if v), len(values))
    completion = {}
    for direction in (True, False):
        ids = [c for c in scores if (cases[c].get("expect") or {}).get("reached_write") is direction]
        done = sum(1 for c in ids for s in scores[c] if s.get("expect_reached_write"))
        completion[direction] = (done, sum(len(scores[c]) for c in ids))
    return {
        "cases": len(scores),
        "runs": runs,
        "majority": sum(1 for p in passes.values() if p * 2 >= runs),
        "all_runs": sum(1 for p in passes.values() if p == runs),
        "passes": passes,
        "zero_tolerance": zero,
        "completion": completion,
    }


def majority_file_agrees(trajectories: Path, scores: dict[str, list[dict]]) -> bool | None:
    """Does the recount match the majority file the run wrote? None if there is none."""
    stamp = trajectories.name.removeprefix("trajectories-").removesuffix(".jsonl")
    majority = trajectories.with_name(f"majority-{stamp}.json")
    if not majority.exists():
        return None
    recorded = {r["case_id"]: r["passed"] for r in json.loads(majority.read_text(encoding="utf-8"))}
    return recorded == {c: _passes(r) for c, r in scores.items()}


def reading_failures(path: Path) -> tuple[int, int] | None:
    """Turns the reader gave no reading on, and in how many case-runs; None for
    a run recorded before the count existed.

    Checked before any verdict: on 2026-09-29 every reading of a ten-minute
    run failed, and its "v1.1" half measured the routing of v1.0.0.
    """
    counts = [
        (json.loads(line)["outputs"].get("usage") or {}).get("reading_failures")
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if all(c is None for c in counts):
        return None
    failed = [c or 0 for c in counts]
    return sum(failed), sum(1 for c in failed if c)


def zero_tolerance_misses(path: Path, scores: dict[str, list[dict]]) -> list[str]:
    lines = []
    for case_id, runs in scores.items():
        for index, s in enumerate(runs):
            missed = [k for k in ZERO_TOLERANCE_KEYS if s.get(k) is False]
            if missed:
                lines.append(f"{case_id} run {index + 1}: {', '.join(missed)}")
    return lines


def render(full: Path, n1: Path | None = None, *, dataset_at: str | None = None) -> str:
    commit = dataset_at or recorded_commit(full)
    cases = cases_at(commit) if commit else {c["id"]: c for c in load_cases()}
    full_scores = rescore(full, cases)
    rows = [("full agent", full, full_scores, summarize(full_scores, cases))]
    if n1 is not None:
        n1_scores = rescore(n1, cases)
        rows.append(("N-1, all checks", n1, n1_scores, summarize(n1_scores, cases)))
        rows.append(("N-1, automatic failures excluded", n1, n1_scores,
                     summarize(n1_scores, cases, drop_tautological=True)))

    lines = ["# All runs, recounted offline", "",
             "Deterministic checks only, recomputed from the saved trajectories -- the judge's",
             "questions need a model call and are not in here. A run's own report covers its last",
             "run only.", "",
             f"Scored against the dataset at commit `{commit[:7]}`." if commit
             else "Scored against the current dataset.", ""]
    for label, path, scores, _ in rows[:2]:
        agrees = majority_file_agrees(path, scores)
        lines.append(f"- `{path.name}`: recount "
                     + {True: "matches the run's majority file", False: "**does not match** the run's "
                        "majority file", None: "has no majority file to check against"}[agrees])
    lines += ["", "| | cases | majority pass | pass every run |", "|---|---|---|---|"]
    for label, _, _, s in rows:
        lines.append(f"| {label} | {s['cases']} | {s['majority']}/{s['cases']} | {s['all_runs']}/{s['cases']} |")
    lines += ["", "| zero-tolerance, per case-run | " + " | ".join(label for label, *_ in rows[:2]) + " |",
              "|---|" + "---|" * len(rows[:2])]
    for key in ZERO_TOLERANCE_KEYS:
        cells = [f"{s['zero_tolerance'][key][0]}/{s['zero_tolerance'][key][1]}" for *_, s in rows[:2]]
        lines.append(f"| `{key}` | " + " | ".join(cells) + " |")
    lines += ["", "| task completion, per case-run | " + " | ".join(label for label, *_ in rows[:2]) + " |",
              "|---|" + "---|" * len(rows[:2])]
    for direction, label in ((True, "should write, and did"), (False, "should not write, and did not")):
        cells = [f"{s['completion'][direction][0]}/{s['completion'][direction][1]}" for *_, s in rows[:2]]
        lines.append(f"| {label} | " + " | ".join(cells) + " |")
    for label, path, scores, _ in rows[:2]:
        misses = zero_tolerance_misses(path, scores)
        lines += ["", f"Zero-tolerance misses, {label}:" + (" none" if not misses else "")]
        lines += [f"- {m}" for m in misses]
    for label, path, _, _ in rows[:2]:
        failures = reading_failures(path)
        if failures is None:
            continue
        turns, runs = failures
        lines += ["", f"Turns without a reading, {label}: "
                  + ("none" if not turns else f"**{turns}, in {runs} case-runs** -- those turns ran the "
                     "routing of v1.0.0; this run does not measure the agent as built")]
    if n1 is not None:
        full_s, n1_s = rows[0][3], rows[2][3]
        runs = full_s["runs"]
        worse = sorted(c for c in full_s["passes"] if full_s["passes"][c] * 2 >= runs > n1_s["passes"].get(c, 0) * 2)
        better = sorted(c for c in full_s["passes"] if full_s["passes"][c] * 2 < runs <= n1_s["passes"].get(c, 0) * 2)
        lines += ["", "Cases that change majority verdict, full -> N-1 (automatic failures excluded):",
                  f"- pass -> fail: {', '.join(worse) or 'none'}",
                  f"- fail -> pass: {', '.join(better) or 'none'}",
                  "", "At 26 cases a paired comparison needs about six cases flipping one way before a",
                  "difference is significant (plan, Step 7)."]
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description="Recount every run of a sweep, offline")
    parser.add_argument("trajectories", type=Path)
    parser.add_argument("--n1", type=Path, default=None, help="An N-1 run's trajectories to compare.")
    parser.add_argument("--dataset-at", default=None,
                        help="Score against the dataset at this commit (default: the one a published "
                             "snapshot's README records, else the current dataset).")
    args = parser.parse_args()
    print(render(args.trajectories, args.n1, dataset_at=args.dataset_at))


if __name__ == "__main__":
    main()
