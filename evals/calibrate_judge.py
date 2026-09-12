"""Measure whether the LLM judge agrees with a human.

Since stage 10 this project has carried a warning that the judge is
uncalibrated and that its percentage scores should not be trusted. Stage 18
showed why that warning was not paranoia: the judge was marking the agent
down for behaviour the system prompt explicitly required, and prompt tuning
was being steered by the resulting noise. An uncalibrated judge does not
reduce grading uncertainty -- it adds a new source of it.

Calibration here means one thing: **does the judge give the same verdict a
human would, on the same trajectories?** Not "is it clever", not "does it
write good comments". Agreement, measured, on real cases.

Workflow (three commands, with your own judgement in the middle):

    python -m evals.calibrate_judge export --experiment <name>
        Pull real trajectories from a LangSmith run into a labelling file,
        with the judge's own verdict HIDDEN so it cannot anchor you.

    ...open the file, read each trajectory, write 1 or 0 in "human_score"...

    python -m evals.calibrate_judge score
        Compare your labels against the judge's, and report agreement plus
        every disagreement in full so you can see which side was wrong.

A disagreement is not automatically the judge's fault. Roughly:
  - judge says 0, you say 1 -> usually the rubric is stricter than the design
  - judge says 1, you say 0 -> usually the rubric is missing a requirement
  - both wrong the same way   -> the case's expected_behavior is unclear
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

RESULTS_DIR = Path(__file__).resolve().parent / "results"
LABEL_PATH = RESULTS_DIR / "judge-calibration.json"

# Enough to detect a systematic disagreement without turning labelling into a
# chore. Below ~10 a single case swings the number; much above 20 and it stops
# getting done at all, which is worse than a rough estimate.
DEFAULT_SAMPLE = 15


def export(experiment: str | None, sample: int, dataset: str) -> Path:
    """Pull trajectories into a labelling file, judge verdict withheld."""
    from langsmith import Client

    from evals.fetch_results import fetch_rows, list_experiments

    client = Client()
    if not experiment:
        experiments = list_experiments(client, dataset)
        if not experiments:
            raise SystemExit(f"No experiments found for dataset '{dataset}'.")
        experiment = experiments[0].name
        print(f"Using newest experiment: {experiment}")

    rows = fetch_rows(client, experiment, dataset)
    graded = [r for r in rows if r["scores"].get("expected_behavior_match") is not None]
    if not graded:
        raise SystemExit("No judge-graded cases in that experiment.")

    # Deliberately include both verdicts: a sample of only-failures would
    # measure the judge's false-positive rate and nothing else.
    passing = [r for r in graded if r["scores"]["expected_behavior_match"]]
    failing = [r for r in graded if not r["scores"]["expected_behavior_match"]]
    take_fail = min(len(failing), max(1, sample // 2))
    chosen = failing[:take_fail] + passing[: sample - take_fail]

    items = []
    for row in chosen:
        items.append(
            {
                "case_id": row["case_id"],
                "category": row["category"],
                "human_score": None,       # <- you fill this in: 1 or 0
                "human_note": "",
                "_judge_score": row["scores"]["expected_behavior_match"],
                "_judge_comment": row["comments"].get("expected_behavior_match", ""),
            }
        )

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    payload = {
        "experiment": experiment,
        "instructions": (
            "For each item read the case in evals/dataset.jsonl (its expected_behavior) "
            "and the trajectory in LangSmith or via `python -m evals.fetch_results`, then "
            "set human_score to 1 (agent behaved correctly) or 0 (it did not). Ignore the "
            "_judge_* fields while labelling -- they are here only so `score` can compare, "
            "and reading them first defeats the point."
        ),
        "items": items,
    }
    LABEL_PATH.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Wrote {len(items)} cases to label ({take_fail} the judge failed, "
          f"{len(items) - take_fail} it passed):\n  {LABEL_PATH}")
    return LABEL_PATH


def score() -> dict:
    """Compare human labels against the judge's and report agreement."""
    if not LABEL_PATH.exists():
        raise SystemExit(f"No labelling file at {LABEL_PATH}. Run `export` first.")

    payload = json.loads(LABEL_PATH.read_text(encoding="utf-8"))
    items = payload["items"]
    labelled = [i for i in items if i.get("human_score") in (0, 1)]
    if not labelled:
        raise SystemExit("Nothing labelled yet -- set human_score to 1 or 0 in the file.")

    agree = [i for i in labelled if int(i["human_score"]) == int(bool(i["_judge_score"]))]
    disagree = [i for i in labelled if i not in agree]
    pct = len(agree) / len(labelled) * 100

    print(f"\nExperiment: {payload.get('experiment')}")
    print(f"Labelled:   {len(labelled)} of {len(items)}")
    print(f"Agreement:  {len(agree)}/{len(labelled)} ({pct:.0f}%)\n")

    if disagree:
        kinds = Counter(
            "judge stricter than human" if not i["_judge_score"] else "judge more lenient"
            for i in disagree
        )
        print("Disagreements:")
        for i in disagree:
            direction = "judge=0 human=1 (judge stricter)" if not i["_judge_score"] \
                else "judge=1 human=0 (judge lenient)"
            print(f"\n  {i['case_id']}  [{direction}]")
            if i.get("human_note"):
                print(f"    you:   {i['human_note']}")
            print(f"    judge: {(i.get('_judge_comment') or '').strip()[:400]}")
        print("\n  pattern:", dict(kinds))
    else:
        print("No disagreements in the labelled sample.")

    print("\n" + _verdict(pct, disagree))
    return {"agreement": pct, "labelled": len(labelled), "disagreements": len(disagree)}


def _verdict(pct: float, disagree: list) -> str:
    if pct >= 90:
        return ("Judge broadly agrees with you. Its percentage scores are usable as a "
                "trend, though still not as an absolute grade.")
    if pct >= 75:
        return ("Usable with care. Read the disagreements above for a pattern -- if they "
                "all lean one way, fix that line of the rubric in evals/judge.py and "
                "re-run, rather than accepting the noise.")
    return ("Do not trust this judge's scores yet. At this level of disagreement its "
            "numbers are close to noise: fix the rubric against the disagreements above, "
            "then re-export and re-label.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Calibrate the LLM judge against human labels")
    sub = parser.add_subparsers(dest="command", required=True)

    e = sub.add_parser("export", help="Pull trajectories into a labelling file")
    e.add_argument("--experiment", help="Experiment name (default: newest)")
    e.add_argument("--sample", type=int, default=DEFAULT_SAMPLE)
    e.add_argument("--dataset", default="job-load-agent-evals")

    sub.add_parser("score", help="Compare your labels against the judge's")

    args = parser.parse_args()
    if args.command == "export":
        export(args.experiment, args.sample, args.dataset)
    else:
        score()


if __name__ == "__main__":
    main()
