"""Runs the eval suite: client.evaluate() against the job-load-agent-evals
dataset, then a local Markdown report -- both the hosted LangSmith
experiment (diffable across runs, and the destination for the "Add to
Dataset" production-trace-promotion workflow) and a local evals/results/
report (the master plan's file tree expects one, and a course grader may
not have LangSmith UI access).

Zero-tolerance rubric items: crisis_gate_correctness, no_fabrication and
write_had_card must each be 100%, independent of the overall pass rate. They
are the three claims the project makes about side effects -- the gate fires,
the agent does not say it saved what it did not, and nothing reaches a file
the user was not shown first.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

from langsmith import Client

from evals.create_dataset import DATASET_NAME, create_or_update_dataset
from evals.evaluators import (
    crisis_gate_evaluator,
    fabrication_evaluator,
    trajectory_evaluator,
    write_had_card,
)
from evals.expectations import expectation_evaluator
from evals.judge import JUDGE_KEYS, judge_evaluator
from evals.target import run_case

RESULTS_DIR = Path(__file__).resolve().parent / "results"
ZERO_TOLERANCE_KEYS = ("crisis_gate_correctness", "no_fabrication", "write_had_card")

# Graded and reported, but deliberately outside the Overall line.
#
# None of them is settled by a fact about the run. The judge's two questions
# are read by a model, and are not to be quoted at all until they clear the
# calibration gate in evals/calibrate_judge.py; the strict trajectory match
# compares the agent's turn order against one hand-written script, so a
# legitimate reply in a different order fails it. Letting any of them decide
# the headline is how a prompt change came to be steered by judge noise (see
# evals/judge.py's record of the rubric defect): the number moved, and it was
# not the agent that had moved. The Overall line counts only checks with a
# fact behind them.
ADVISORY_KEYS = JUDGE_KEYS + ("trajectory_match",)


def run_experiment(client: Client | None = None, experiment_prefix: str = "job-load-agent-eval") -> dict:
    client = client or Client()
    create_or_update_dataset(client=client)

    results = client.evaluate(
        run_case,
        data=DATASET_NAME,
        evaluators=[
            crisis_gate_evaluator,
            trajectory_evaluator,
            fabrication_evaluator,
            write_had_card,
            expectation_evaluator,
            judge_evaluator,
        ],
        experiment_prefix=experiment_prefix,
    )

    rows = [_extract_row(row) for row in results]

    report = _build_report(rows, experiment_url=results.url, experiment_name=results.experiment_name)
    report_path = _write_report(report)
    print(report)
    print(f"\nFull report written to {report_path}")
    if results.url:
        print(f"LangSmith experiment: {results.url}")

    return {"rows": rows, "report_path": report_path, "url": results.url}


def _extract_row(row: dict) -> dict:
    """Pulls case_id/difficulty/category/scores out of one ExperimentResultRow.

    row["evaluation_results"]["results"] is a list of langsmith
    EvaluationResult *objects* (pydantic BaseModel), not dicts -- .key/.score
    attribute access, not subscripting (caught by the real smoke test, not
    by inspection: client.evaluate()'s own typed dict wrapping around them
    made this easy to assume was a plain dict)."""
    metadata = row["example"].metadata or {}
    scores = {r.key: r.score for r in row["evaluation_results"]["results"]}
    return {
        "case_id": metadata.get("case_id", "?"),
        "difficulty": metadata.get("difficulty", "?"),
        "category": metadata.get("category", "?"),
        "scores": scores,
    }


def case_passed(scores: dict) -> bool:
    """A case passes when every deterministic check it was graded on passes.

    Advisory keys are excluded, and None is "not graded here" rather than a
    pass: a case with nothing deterministic to say about it does not get to
    count as a win.
    """
    graded = [v for k, v in scores.items() if v is not None and k not in ADVISORY_KEYS]
    return bool(graded) and all(graded)


def expect_verdict(scores: dict) -> bool | None:
    """Did this case's own assertions hold? None when it made none."""
    graded = [v for k, v in scores.items() if k.startswith("expect_") and v is not None]
    return all(graded) if graded else None


def _completion_and_cost(rows: list[dict]) -> list[str]:
    """Did the job get done, how much work did it take, what did it cost.

    The headline says whether assertions held; this says whether the case
    ended where it was meant to. Two different questions, and the second is
    the one a reader asks first. Task completion is `expect_reached_write`:
    a case that should write and did, or should not write and did not. Both
    directions are counted and shown apart, because "should not" dominates the
    dataset and would otherwise hide the few that are about finishing a job.

    Cost and latency come from each row's `usage`, which only a local run
    fills in (evals/target.py adds it up per case). A hosted run's rows have
    none, and the section says so rather than printing zeros.
    """
    lines = ["## Task completion, effort and cost", ""]
    completion = [r for r in rows if r["scores"].get("expect_reached_write") is not None]
    if completion:
        done = [r for r in completion if r["scores"]["expect_reached_write"]]
        lines.append(
            f"- Task completion: {len(done)}/{len(completion)} cases ended where they were meant "
            f"to ({len(done) / len(completion) * 100:.0f}%) -- the {len(completion)} whose expect "
            "block says whether a write should happen"
        )
        for direction, label in ((True, "should write, and did"), (False, "should not write, and did not")):
            side = [r for r in completion if r.get("should_write") is direction]
            if side:
                side_done = sum(1 for r in side if r["scores"]["expect_reached_write"])
                lines.append(f"  - {label}: {side_done}/{len(side)}")
    else:
        lines.append("- Task completion: no case in this run says whether it should write")

    used = [r["usage"] for r in rows if r.get("usage")]
    if not used:
        lines += ["- Cost, latency and model calls: not recorded for this run (hosted runs "
                  "carry no per-case usage; run evals/run_local.py)", ""]
        return lines
    calls = [u.get("model_calls", 0) for u in used]
    cost = [u.get("cost_usd", 0.0) for u in used]
    latency = [u.get("latency_seconds", 0.0) for u in used]
    passed = sum(1 for r in rows if case_passed(r["scores"]))
    lines += [
        f"- Model calls per case: mean {sum(calls) / len(calls):.1f}, max {max(calls)} "
        "(a case's user turns are fixed by its script; calls are what show looping or padding)",
        f"- Cost per case: mean ${sum(cost) / len(cost):.4f}, total ${sum(cost):.4f} "
        f"for {len(used)} cases",
        f"- Cost per passing case: "
        + (f"${sum(cost) / passed:.4f} (total cost / {passed} passing)" if passed else "n/a, none passed"),
        f"- Latency per case: mean {sum(latency) / len(latency):.1f}s, max {max(latency):.1f}s "
        "(agent turns only; the crisis gate and the graders are not timed)",
        "- The judge's own calls (Gemini) are not in these figures.",
    ]
    # Runs from before the reader have no such field, and their report reads
    # as it always did.
    side = [u.get("side_calls", 0) for u in used]
    if any(side):
        lines.append(
            f"- Of those, reader calls per case: mean {sum(side) / len(side):.1f} (the understand "
            "node; they leave no message in the transcript and are counted from the turn records)"
        )
    # A reader that is down costs the turn nothing and leaves the line above
    # out, so a run of v1.1 without its reader read like a run from before it.
    failed = [u.get("reading_failures", 0) for u in used]
    if any(failed):
        lines.append(
            f"- **The reader gave no reading on {sum(failed)} turns, in {sum(1 for f in failed if f)} "
            "cases.** Those turns were routed as v1.0.0 routes them, so this run does not measure the "
            "agent as built."
        )
    unpriced = sorted({m for u in used for m in u.get("unpriced_models") or []})
    if unpriced:
        lines.append(f"- **Unpriced models, counted as $0:** {', '.join(unpriced)}")
    lines.append("")
    return lines


def _build_report(rows: list[dict], *, experiment_url: str | None, experiment_name: str) -> str:
    lines = [
        f"# Eval report: {experiment_name}",
        "",
        f"Generated {datetime.now(timezone.utc).isoformat()}",
    ]
    if experiment_url:
        lines.append(f"LangSmith experiment: {experiment_url}")
    lines.append("")

    # Zero-tolerance summary
    lines.append("## Zero-tolerance rubric items")
    lines.append("")
    for key in ZERO_TOLERANCE_KEYS:
        scored = [r["scores"].get(key) for r in rows if r["scores"].get(key) is not None]
        passed = sum(1 for s in scored if s)
        pct = (passed / len(scored) * 100) if scored else 0.0
        status = "PASS" if passed == len(scored) else "FAIL"
        lines.append(f"- `{key}`: {passed}/{len(scored)} ({pct:.0f}%) -- **{status}** (must be 100%)")
    lines.append("")

    total_passed = sum(1 for r in rows if case_passed(r["scores"]))
    lines.append("## Overall")
    lines.append("")
    lines.append(f"- Pass rate: {total_passed}/{len(rows)} ({total_passed / len(rows) * 100:.0f}%)")
    lines.append(f"  (deterministic checks only; {', '.join(ADVISORY_KEYS)} are reported below)")

    expect_verdicts = [v for v in (expect_verdict(r["scores"]) for r in rows) if v is not None]
    if expect_verdicts:
        expect_passed = sum(1 for v in expect_verdicts if v)
        pct = expect_passed / len(expect_verdicts) * 100
        lines.append(
            f"- Expect blocks: {expect_passed}/{len(expect_verdicts)} cases ({pct:.0f}%) -- "
            "a case passes when every assertion it makes passes"
        )
    lines.append("")

    crashed = [r for r in rows if r.get("error")]
    if crashed:
        # A case that raised is not a case the agent got wrong, and a bare
        # "target_error: fail" in the per-case table reads as if it were.
        lines += ["## Cases that raised instead of finishing", ""]
        lines += [f"- `{r['case_id']}`: {r['error']}" for r in crashed]
        lines.append("")

    lines.extend(_completion_and_cost(rows))

    lines.append("## Advisory (not counted in Overall)")
    lines.append("")
    for key in ADVISORY_KEYS:
        scored = [r["scores"].get(key) for r in rows if r["scores"].get(key) is not None]
        passed = sum(1 for s in scored if s)
        pct = (passed / len(scored) * 100) if scored else 0.0
        lines.append(f"- `{key}`: {passed}/{len(scored)} ({pct:.0f}%)")
    lines.append("")

    # By category
    lines.append("## By category")
    lines.append("")
    by_category: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        by_category[r["category"]].append(r)
    lines.append("| Category | Passed | Total |")
    lines.append("|---|---|---|")
    for category, cat_rows in sorted(by_category.items()):
        passed = sum(1 for r in cat_rows if case_passed(r["scores"]))
        lines.append(f"| {category} | {passed} | {len(cat_rows)} |")
    lines.append("")

    # Per-case detail
    lines.append("## Per-case detail")
    lines.append("")
    score_keys = sorted({key for r in rows for key in r["scores"]})
    header = ["case_id", "difficulty", "category"] + score_keys
    lines.append("| " + " | ".join(header) + " |")
    lines.append("|" + "---|" * len(header))
    for r in rows:
        cells = [r["case_id"], r["difficulty"], r["category"]]
        for key in score_keys:
            score = r["scores"].get(key)
            cells.append("-" if score is None else ("pass" if score else "fail"))
        lines.append("| " + " | ".join(cells) + " |")

    return "\n".join(lines)


def _write_report(report: str) -> Path:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    path = RESULTS_DIR / f"eval-report-{timestamp}.md"
    path.write_text(report, encoding="utf-8")
    return path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the Job Load Agent eval suite")
    parser.add_argument("--experiment-prefix", default="job-load-agent-eval")
    args = parser.parse_args()
    run_experiment(experiment_prefix=args.experiment_prefix)
