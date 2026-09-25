"""Pull eval results out of LangSmith over the API, with no UI access.

Two reasons this exists:

1. The LangSmith web UI isn't always reachable or convenient (travelling,
   no session, a browser you don't want to log into). The results are all
   available over the API with the same key the run already uses.
2. The local Markdown report that `run_experiment.py` writes only carries
   pass/fail per case. The single most useful thing the UI adds is the LLM
   judge's *reasoning* for each verdict -- without it, a failing case tells
   you that something is wrong but not what. This fetches those comments
   and puts them in the report.

It also recovers results from a run whose client crashed after the work was
done: the traces and feedback are already server-side, so a run that died
on a network error during result collection is still readable here.

Usage:
    python -m evals.fetch_results --list
    python -m evals.fetch_results --experiment job-load-agent-v2-501af13c
    python -m evals.fetch_results            # newest experiment
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

from langsmith import Client

from evals.create_dataset import DATASET_NAME
from evals.judge import JUDGE_KEYS
from evals.run_experiment import RESULTS_DIR, ZERO_TOLERANCE_KEYS

# Feedback keys that are graded verdicts, in the order they read best.
SCORE_ORDER = ZERO_TOLERANCE_KEYS + ("trajectory_match",) + JUDGE_KEYS


def list_experiments(client: Client, dataset_name: str = DATASET_NAME, limit: int = 20) -> list:
    """Experiments run against the eval dataset, newest first."""
    projects = list(client.list_projects(reference_dataset_name=dataset_name, limit=limit))
    projects.sort(key=lambda p: getattr(p, "start_time", None) or datetime.min, reverse=True)
    return projects


def _case_id_by_example(client: Client, dataset_name: str = DATASET_NAME) -> dict:
    mapping = {}
    for example in client.list_examples(dataset_name=dataset_name):
        metadata = example.metadata or {}
        mapping[str(example.id)] = {
            "case_id": metadata.get("case_id", "?"),
            "difficulty": metadata.get("difficulty", "?"),
            "category": metadata.get("category", "?"),
        }
    return mapping


def fetch_rows(client: Client, experiment_name: str, dataset_name: str = DATASET_NAME) -> list[dict]:
    """One row per case: metadata, every feedback score, and its comment."""
    runs = [r for r in client.list_runs(project_name=experiment_name, is_root=True)]
    if not runs:
        return []

    example_meta = _case_id_by_example(client, dataset_name)

    feedback_by_run: dict = defaultdict(dict)
    comments_by_run: dict = defaultdict(dict)
    for fb in client.list_feedback(run_ids=[r.id for r in runs]):
        if fb.key is None:
            continue
        feedback_by_run[str(fb.run_id)][fb.key] = fb.score
        if fb.comment:
            comments_by_run[str(fb.run_id)][fb.key] = fb.comment

    rows = []
    for run in runs:
        meta = example_meta.get(str(run.reference_example_id), {})
        rows.append(
            {
                "case_id": meta.get("case_id", "?"),
                "difficulty": meta.get("difficulty", "?"),
                "category": meta.get("category", "?"),
                "scores": dict(feedback_by_run.get(str(run.id), {})),
                "comments": dict(comments_by_run.get(str(run.id), {})),
                "error": run.error,
            }
        )
    rows.sort(key=lambda r: r["case_id"])
    return rows


def fetch_trajectories(
    client: Client, experiment_name: str, dataset_name: str = DATASET_NAME
) -> list[dict]:
    """One record per case: what run_case actually returned, as saved by the
    hosted run.

    The same shape evals/run_local.py writes to trajectories.jsonl, so the
    judge calibration reads a hosted experiment and a local run through one
    code path. A run whose outputs never made it back (a crash mid-collection,
    which this project has seen) is skipped rather than exported as an empty
    trajectory -- there is nothing in it to label.

    `run_id` comes along so a human label can be written back onto the run it
    belongs to.
    """
    runs = [r for r in client.list_runs(project_name=experiment_name, is_root=True)]
    example_meta = _case_id_by_example(client, dataset_name)

    records = []
    for run in runs:
        outputs = getattr(run, "outputs", None)
        if not outputs:
            continue
        meta = example_meta.get(str(run.reference_example_id), {})
        records.append(
            {
                "case_id": meta.get("case_id", "?"),
                "category": meta.get("category", "?"),
                "run_id": str(run.id),
                "outputs": dict(outputs),
            }
        )
    records.sort(key=lambda r: r["case_id"])
    return records


def _status(score) -> str:
    if score is None:
        return "-"
    return "pass" if score else "fail"


def build_report(rows: list[dict], experiment_name: str) -> str:
    lines = [
        f"# Eval results (fetched via API): {experiment_name}",
        "",
        f"Fetched {datetime.now(timezone.utc).isoformat()} -- no UI access required.",
        f"Cases: {len(rows)}",
        "",
        "## Zero-tolerance rubric items",
        "",
    ]
    for key in ZERO_TOLERANCE_KEYS:
        scored = [r["scores"].get(key) for r in rows if r["scores"].get(key) is not None]
        passed = sum(1 for s in scored if s)
        pct = (passed / len(scored) * 100) if scored else 0.0
        status = "PASS" if scored and passed == len(scored) else "FAIL"
        lines.append(f"- `{key}`: {passed}/{len(scored)} ({pct:.0f}%) -- **{status}** (must be 100%)")

    def case_passed(row: dict) -> bool:
        graded = [v for v in row["scores"].values() if v is not None]
        return bool(graded) and all(graded)

    total_passed = sum(1 for r in rows if case_passed(r))
    lines += [
        "",
        "## Overall",
        "",
        f"- Pass rate: {total_passed}/{len(rows)}"
        + (f" ({total_passed / len(rows) * 100:.0f}%)" if rows else ""),
        "",
        "## Per-case detail",
        "",
    ]

    # Alphabetical, matching run_experiment.py's report exactly -- these two
    # reports exist to be compared, and a different column order silently
    # invites misreading one as the other (it did, immediately).
    keys = sorted({k for r in rows for k in r["scores"]})
    header = ["case_id", "difficulty", "category"] + keys
    lines.append("| " + " | ".join(header) + " |")
    lines.append("|" + "---|" * len(header))
    for row in rows:
        cells = [row["case_id"], row["difficulty"], row["category"]]
        cells += [_status(row["scores"].get(k)) for k in keys]
        lines.append("| " + " | ".join(cells) + " |")

    # The part the plain report can't give you: why each failure failed.
    failing = [r for r in rows if not case_passed(r) or r["error"]]
    lines += ["", "## Why the failures failed", ""]
    if not failing:
        lines.append("No failing cases.")
    for row in failing:
        lines.append(f"### {row['case_id']}  ({row['category']}, {row['difficulty']})")
        if row["error"]:
            lines.append(f"- **run error:** {row['error']}")
        for key in keys:
            score = row["scores"].get(key)
            if score is None or score:
                continue
            comment = row["comments"].get(key) or "(no comment recorded)"
            lines.append(f"- **{key}: fail** — {comment.strip()}")
        lines.append("")

    return "\n".join(lines)


def write_report(report: str, experiment_name: str) -> Path:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    safe = "".join(c if c.isalnum() or c in "-_" else "-" for c in experiment_name)
    path = RESULTS_DIR / f"fetched-{safe}-{stamp}.md"
    path.write_text(report, encoding="utf-8")
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description="Fetch eval results from LangSmith via API")
    parser.add_argument("--list", action="store_true", help="List experiments and exit.")
    parser.add_argument("--experiment", help="Experiment name (defaults to the newest).")
    parser.add_argument("--dataset", default=DATASET_NAME)
    args = parser.parse_args()

    client = Client()
    experiments = list_experiments(client, args.dataset)

    if args.list or not experiments:
        print(f"Experiments for dataset '{args.dataset}':")
        for project in experiments:
            print(f"  {project.name:<45} {getattr(project, 'start_time', '')}")
        if not experiments:
            print("  (none found)")
        return

    name = args.experiment or experiments[0].name
    rows = fetch_rows(client, name, args.dataset)
    if not rows:
        print(f"No runs found for experiment '{name}'. Is it still in progress?")
        return

    report = build_report(rows, name)
    path = write_report(report, name)
    print(report)
    print(f"\nWritten to {path}")


if __name__ == "__main__":
    main()
