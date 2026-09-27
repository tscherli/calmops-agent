"""Run the eval suite entirely locally -- no LangSmith, no network beyond
the model calls themselves.

Why this exists: `run_experiment.py` goes through `client.evaluate()`, which
uploads the dataset, streams traces, and reads results back. That makes the
whole run hostage to LangSmith being reachable -- a transient DNS failure
mid-run (observed 2026-09-11: `getaddrinfo failed` for
eu.api.smith.langchain.com) crashed the run *after* every OpenAI call had
already been paid for, and produced no results at all.

This runs the same cases through the same target and the same evaluators,
and writes the same local Markdown report. What it gives up is the hosted
experiment: no trace UI, no run-to-run diffing in LangSmith, no
"Add to Dataset" promotion workflow. Use run_experiment.py when you want
those; use this when you want results that don't depend on a third-party
service being up, or when whoever is running it has no LangSmith account
(the README's grader case).
"""
from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from datetime import datetime, timezone
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

from evals.create_dataset import load_cases
from evals.evaluators import (
    crisis_gate_evaluator,
    fabrication_evaluator,
    trajectory_evaluator,
    write_had_card,
)
from evals.expectations import expectation_evaluator
from evals.judge import judge_evaluator
from evals.run_experiment import _build_report, _write_report, case_passed
from evals.target import EVAL_RUNS_LOG, run_case

# Where runs leave their trajectories: one file per invocation, named
# trajectories-<UTC yyyymmdd-hhmmss>.jsonl for the moment it was written, and
# never a fixed name. Calibrating the judge means reading real ones by hand,
# and a run that has already been paid for should not have to be paid for
# again to get them back. Gitignored with the rest of evals/results.
TRAJECTORIES_DIR = EVAL_RUNS_LOG.parent
TRAJECTORIES_GLOB = "trajectories-*.jsonl"


def variant_suffix(exclude_tools: Sequence[str]) -> str:
    """What a reduced agent's files are marked with: `-without-<tool>...`.

    In the name, not only inside the file, because the name is what `export`
    and a person scanning the folder go by. An N-1 corpus taken for a full one
    would calibrate the judge on an agent nobody ships.
    """
    return "".join(f"-without-{name}" for name in sorted(exclude_tools))


def trajectories_path(
    directory: Path, when: datetime | None = None, exclude_tools: Sequence[str] = ()
) -> Path:
    """The file a run written at `when` (UTC, default now) goes to.

    UTC rather than local time so two machines, or one machine across a
    clock change, never produce names that sort in the wrong order -- and so
    the stamp says the same thing wherever it is read. That is also why a
    file's name can differ from the time Windows Explorer shows for it by
    the local offset.
    """
    stamp = (when or datetime.now(timezone.utc)).strftime("%Y%m%d-%H%M%S")
    return directory / f"trajectories-{stamp}{variant_suffix(exclude_tools)}.jsonl"


def latest_trajectories(directory: Path) -> Path | None:
    """The newest *full-agent* run's trajectories, by the stamp in the name.

    By name, not by modification time: copying or touching a file changes
    its mtime and says nothing about when the run happened. The stamp sorts
    as text because it is fixed-width and most significant first.

    Runs of a reduced agent are skipped: this is what `calibrate_judge
    export` starts a labelling round from, and the judge is calibrated on the
    agent that ships.
    """
    found = sorted(
        (p for p in directory.glob(TRAJECTORIES_GLOB) if "-without-" not in p.name),
        key=lambda p: p.name,
    )
    return found[-1] if found else None


def majority_path(directory: Path, when: datetime, exclude_tools: Sequence[str] = ()) -> Path:
    """Per-case verdicts across several runs, named like the run's trajectories.

    The threshold is a pass in 2 of 3, so this rather than any single report
    is what says whether a case holds. It used to be one fixed majority.json,
    and it went the way trajectories.jsonl did: a three-case re-run replaced
    the full sweep's verdicts two hours after the sweep, and the 22/26 it
    recorded survives only in the development log. Same stamp as the
    trajectories file of the same run, so the two can be found as a pair.
    """
    return directory / f"majority-{when.strftime('%Y%m%d-%H%M%S')}{variant_suffix(exclude_tools)}.json"


EVALUATORS = (
    crisis_gate_evaluator,
    trajectory_evaluator,
    fabrication_evaluator,
    write_had_card,
    expectation_evaluator,
    judge_evaluator,
)

DEFAULT_WORKERS = 4


def case_inputs(case: dict) -> dict:
    """Mirror create_dataset.case_to_example's inputs exactly, so a local run
    and a hosted run feed the target function identical input."""
    inputs: dict = {"turns": case["turns"] if "turns" in case else [case["query"]]}
    if case.get("simulate_write_failure"):
        inputs["simulate_write_failure"] = True
    if case.get("seed_entries"):
        inputs["seed_entries"] = case["seed_entries"]
    if case.get("seed_measures"):
        inputs["seed_measures"] = case["seed_measures"]
    return inputs


def _feedback_items(result: dict) -> list[dict]:
    """One evaluator, one or many feedback entries.

    `expectation_evaluator` answers with `{"results": [...]}` -- the shape
    LangSmith accepts from an evaluator that scores several keys at once --
    and this runner has to unwrap it the same way, or an expect block would
    land in the report as a single score named "results".
    """
    inner = result.get("results") if isinstance(result, dict) else None
    return list(inner) if isinstance(inner, list) else [result]


def evaluate_case(case: dict, exclude_tools: Sequence[str] = ()) -> dict:
    """Run one case and score it. Never raises: a crashing evaluator records
    a None score for that key rather than taking down the whole run, which
    matters because the LLM judge is known to intermittently fail parsing
    (see DEVELOPMENT_LOG.md stage 10)."""
    reference = {
        "messages": case.get("reference_trajectory", []),
        "expected_behavior": case["expected_behavior"],
        "category": case["category"],
    }
    if case.get("expect"):
        reference["expect"] = case["expect"]
    try:
        # Cost and latency per case, into the suite's own file: the app's
        # observability/runs.jsonl is its real per-turn record, and a suite run
        # would bury it.
        outputs = run_case(case_inputs(case), runs_log=EVAL_RUNS_LOG, exclude_tools=exclude_tools)
    except Exception as exc:
        return {
            "case_id": case["id"],
            "difficulty": case["difficulty"],
            "category": case["category"],
            "scores": {"target_error": False},
            "error": str(exc),
        }

    scores: dict = {}
    for evaluator in EVALUATORS:
        try:
            for item in _feedback_items(evaluator(outputs=outputs, reference_outputs=reference)):
                scores[item["key"]] = item.get("score")
        except Exception as exc:  # a broken evaluator must not lose the run
            scores[getattr(evaluator, "__name__", "evaluator")] = None
            print(f"  ! {case['id']}: {getattr(evaluator, '__name__', '?')} raised {exc}")

    return {
        "case_id": case["id"],
        "difficulty": case["difficulty"],
        "category": case["category"],
        "scores": scores,
        # Its own field, lifted out of the outputs: the report reads cost,
        # latency and model calls from here and never from `outputs`, so a
        # case with no outputs (a target error) cannot break it.
        "usage": outputs.get("usage"),
        # Which way the case's task completion points: True "should write",
        # False "should not", None "does not say". The report counts the two
        # directions apart.
        "should_write": (case.get("expect") or {}).get("reached_write"),
        # Carried on the row so the runner can save it. The report never
        # reads it; evals/calibrate_judge.py does.
        "outputs": outputs,
    }


def _save_trajectories(rows: list[dict], path: Path) -> int:
    """Write one line per case run: the case id and everything it produced.

    One file per invocation, created fresh and never overwritten -- opened in
    exclusive mode, so an existing file with the same name is an error rather
    than a loss. Two *invocations* in one file would be labelled as though
    they came from one agent, and the judge's agreement number would be
    measured against a mixture. Repeats within one invocation are a different
    thing and are all kept: three runs of one case are three real
    trajectories, and the situation question can only reach twenty labels
    because of them.

    History: this used to write a fixed `trajectories.jsonl` and a stamped
    copy. The fixed name was overwritten by every run, so a two-case re-run
    destroyed a seventy-eight trajectory sweep ten minutes after it was made;
    and the corpus the first judge labels were made from was a hand merge that
    existed *only* under the fixed name, one partial run away from being gone.
    A name that every run writes to is a name nothing can safely depend on.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    lines = []
    for row in rows:
        if not row.get("outputs"):
            continue
        record = {
            "case_id": row["case_id"],
            "category": row["category"],
            "outputs": row["outputs"],
        }
        if row.get("run_index") is not None:
            record["run_index"] = row["run_index"]
        lines.append(json.dumps(record, ensure_ascii=False, default=str) + "\n")
        written += 1
    with path.open("x", encoding="utf-8") as handle:
        handle.write("".join(lines))
    return written


def majority(rows_by_run: list[list[dict]]) -> list[dict]:
    """Per case: how many runs it passed, and the verdict that follows.

    The threshold is a pass in at least 2 of 3 runs, so a single run is not a
    verdict -- it is one sample of a stochastic system. What this exists to
    surface is the *split*: a case that passed twice and failed once is the
    one to re-run with `--case` before anyone calls it stable, and a report
    that shows only the majority hides exactly those.
    """
    verdicts: dict[str, list[bool]] = {}
    for rows in rows_by_run:
        for row in rows:
            verdicts.setdefault(row["case_id"], []).append(case_passed(row["scores"]))
    summary = []
    for case_id, results in verdicts.items():
        passed = sum(1 for r in results if r)
        summary.append({
            "case_id": case_id,
            "passed": passed,
            "runs": len(results),
            "verdict": passed * 2 >= len(results),
            "split": 0 < passed < len(results),
        })
    return summary


def _print_majority(summary: list[dict]) -> None:
    runs = summary[0]["runs"] if summary else 0
    print(f"\n## Majority over {runs} runs\n")
    print("| case_id | passed | verdict |")
    print("|---|---|---|")
    for row in sorted(summary, key=lambda r: (r["verdict"], -r["passed"], r["case_id"])):
        mark = "pass" if row["verdict"] else "FAIL"
        split = " (split)" if row["split"] else ""
        print(f"| {row['case_id']} | {row['passed']}/{row['runs']} | {mark}{split} |")
    splits = [r["case_id"] for r in summary if r["split"]]
    passing = sum(1 for r in summary if r["verdict"])
    print(f"\nMajority pass: {passing}/{len(summary)} cases.")
    if splits:
        print(
            f"{len(splits)} case(s) did not agree with themselves: {', '.join(splits)}.\n"
            "Re-run each with --case five more times before calling it stable; a 2-1 "
            "split is one sample of a coin, not a measurement."
        )


def _one_pass(cases: list[dict], workers: int, exclude_tools: Sequence[str] = ()) -> list[dict]:
    with ThreadPoolExecutor(max_workers=workers) as pool:
        rows = list(pool.map(partial(evaluate_case, exclude_tools=exclude_tools), cases))
    # Preserve dataset order regardless of completion order, so two reports
    # are diffable line by line.
    order = {case["id"]: i for i, case in enumerate(cases)}
    rows.sort(key=lambda r: order.get(r["case_id"], 0))
    return rows


def run_local(
    cases: list[dict] | None = None,
    workers: int = DEFAULT_WORKERS,
    trajectories_dir: Path | None = None,
    runs: int = 1,
    exclude_tools: Sequence[str] = (),
) -> dict:
    cases = cases if cases is not None else load_cases()
    without = f", WITHOUT {', '.join(sorted(exclude_tools))}" if exclude_tools else ""
    print(f"Running {len(cases)} cases locally ({workers} workers, no LangSmith{without})...")

    rows_by_run = []
    for index in range(runs):
        if runs > 1:
            print(f"\n-- run {index + 1} of {runs} --")
        pass_rows = _one_pass(cases, workers, exclude_tools)
        for row in pass_rows:
            row["run_index"] = index
        rows_by_run.append(pass_rows)

    rows = rows_by_run[-1]
    # The report's own title says it, since its file name is stamped by a
    # helper shared with the hosted runner and carries no variant.
    report = _build_report(
        rows, experiment_url=None, experiment_name=f"local run (no LangSmith{without})"
    )
    report_path = _write_report(report)
    print(report)
    print(f"\nFull report written to {report_path}")

    # One moment for everything this invocation leaves behind, so its
    # majority and its trajectories carry the same stamp.
    finished = datetime.now(timezone.utc)

    summary = None
    if runs > 1:
        # The report above is one run. With several, the majority is the
        # verdict and the single-run report is only the last sample of it.
        summary = majority(rows_by_run)
        _print_majority(summary)
        path = majority_path(trajectories_dir or TRAJECTORIES_DIR, finished, exclude_tools)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("x", encoding="utf-8") as handle:
            handle.write(json.dumps(summary, indent=2))
        print(f"\nPer-case verdicts written to {path}")

    if trajectories_dir is not None:
        path = trajectories_path(trajectories_dir, finished, exclude_tools)
        written = _save_trajectories(
            [row for pass_rows in rows_by_run for row in pass_rows], path
        )
        print(f"{written} trajectories saved to {path}")
    return {"rows": rows, "report_path": report_path, "majority": summary}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the Job Load Agent eval suite locally")
    parser.add_argument("--workers", type=int, default=DEFAULT_WORKERS)
    parser.add_argument("--case", action="append", help="Run only these case ids (repeatable).")
    parser.add_argument(
        "--runs", type=int, default=1,
        help="Run the whole selection this many times and report the majority. "
             "The threshold is a pass in 2 of 3, so one run is a sample rather "
             "than a verdict.",
    )
    parser.add_argument(
        "--exclude-tool", action="append", default=[], dest="exclude_tools",
        help="Build the agent without this tool (repeatable): the N-1 baseline, "
             "e.g. --exclude-tool find_measures. Its files are marked -without-<tool>.",
    )
    args = parser.parse_args()

    selected = load_cases()
    if args.case:
        wanted = set(args.case)
        selected = [c for c in selected if c["id"] in wanted]
        missing = wanted - {c["id"] for c in selected}
        if missing:
            parser.error(f"unknown case ids: {sorted(missing)}")

    run_local(
        selected,
        workers=args.workers,
        trajectories_dir=TRAJECTORIES_DIR,
        runs=max(1, args.runs),
        exclude_tools=args.exclude_tools,
    )
