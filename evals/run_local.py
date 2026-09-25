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
from concurrent.futures import ThreadPoolExecutor

from dotenv import load_dotenv

load_dotenv()

from evals.create_dataset import load_cases
from evals.evaluators import crisis_gate_evaluator, fabrication_evaluator, trajectory_evaluator
from evals.judge import judge_evaluator
from evals.run_experiment import _build_report, _write_report
from evals.target import EVAL_RUNS_LOG, run_case

EVALUATORS = (
    crisis_gate_evaluator,
    trajectory_evaluator,
    fabrication_evaluator,
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
    return inputs


def evaluate_case(case: dict) -> dict:
    """Run one case and score it. Never raises: a crashing evaluator records
    a None score for that key rather than taking down the whole run, which
    matters because the LLM judge is known to intermittently fail parsing
    (see DEVELOPMENT_LOG.md stage 10)."""
    reference = {
        "messages": case.get("reference_trajectory", []),
        "expected_behavior": case["expected_behavior"],
        "category": case["category"],
    }
    try:
        # Cost and latency per case, into the suite's own file: the app's
        # observability/runs.jsonl is its real per-turn record, and a suite run
        # would bury it.
        outputs = run_case(case_inputs(case), runs_log=EVAL_RUNS_LOG)
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
            result = evaluator(outputs=outputs, reference_outputs=reference)
            scores[result["key"]] = result.get("score")
        except Exception as exc:  # a broken evaluator must not lose the run
            scores[getattr(evaluator, "__name__", "evaluator")] = None
            print(f"  ! {case['id']}: {getattr(evaluator, '__name__', '?')} raised {exc}")

    return {
        "case_id": case["id"],
        "difficulty": case["difficulty"],
        "category": case["category"],
        "scores": scores,
    }


def run_local(cases: list[dict] | None = None, workers: int = DEFAULT_WORKERS) -> dict:
    cases = cases if cases is not None else load_cases()
    print(f"Running {len(cases)} cases locally ({workers} workers, no LangSmith)...")

    with ThreadPoolExecutor(max_workers=workers) as pool:
        rows = list(pool.map(evaluate_case, cases))

    # Preserve dataset order regardless of completion order, so two reports
    # are diffable line by line.
    order = {case["id"]: i for i, case in enumerate(cases)}
    rows.sort(key=lambda r: order.get(r["case_id"], 0))

    report = _build_report(rows, experiment_url=None, experiment_name="local run (no LangSmith)")
    report_path = _write_report(report)
    print(report)
    print(f"\nFull report written to {report_path}")
    return {"rows": rows, "report_path": report_path}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the Job Load Agent eval suite locally")
    parser.add_argument("--workers", type=int, default=DEFAULT_WORKERS)
    parser.add_argument("--case", action="append", help="Run only these case ids (repeatable).")
    args = parser.parse_args()

    selected = load_cases()
    if args.case:
        wanted = set(args.case)
        selected = [c for c in selected if c["id"] in wanted]
        missing = wanted - {c["id"] for c in selected}
        if missing:
            parser.error(f"unknown case ids: {sorted(missing)}")

    run_local(selected, workers=args.workers)
