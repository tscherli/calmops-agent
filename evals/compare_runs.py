"""Compare several runs of the same suite against identical code, to separate
real results from sampling noise.

Stage 18 established the problem concretely: of seven failing cases, four
flipped to passing when re-run, two had been failing since a much earlier
release, and exactly one was a genuine regression. Every stage since has
treated single-run numbers with justified suspicion without ever measuring
the variance. This does.

A case that passes 3/3 is a result. A case that passes 2/3 is a coin flip,
and no amount of prompt tuning will move it -- the right response there is to
make the case deterministic (or accept it is judge-graded and noisy), not to
"fix" the agent.

    python -m evals.compare_runs --experiments jla-states-r1-abc jla-states-r2-def
    python -m evals.compare_runs --prefix jla-states --limit 3

Needs no labelling and no extra agent calls beyond the runs themselves,
which makes it the cheapest real signal available about an eval suite.
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
from evals.fetch_results import RESULTS_DIR, fetch_rows, list_experiments

SCORE_KEYS = (
    "crisis_gate_correctness",
    "no_fabrication",
    "trajectory_match",
    "expected_behavior_match",
)


def collect(client: Client, experiments: list[str], dataset: str) -> dict:
    """case_id -> score key -> list of per-run scores (None = not graded)."""
    per_case: dict[str, dict[str, list]] = defaultdict(lambda: defaultdict(list))
    meta: dict[str, dict] = {}
    for name in experiments:
        rows = fetch_rows(client, name, dataset)
        if not rows:
            print(f"  (no runs found for {name} -- still in progress?)")
            continue
        for row in rows:
            meta[row["case_id"]] = {"category": row["category"], "difficulty": row["difficulty"]}
            for key in SCORE_KEYS:
                per_case[row["case_id"]][key].append(row["scores"].get(key))
    return {"scores": per_case, "meta": meta}


def classify(values: list) -> str:
    """stable-pass / stable-fail / FLAKY / not-graded for one score key."""
    graded = [v for v in values if v is not None]
    if not graded:
        return "not-graded"
    if all(graded):
        return "stable-pass"
    if not any(graded):
        return "stable-fail"
    return "FLAKY"


def build_report(data: dict, experiments: list[str]) -> str:
    scores, meta = data["scores"], data["meta"]
    n = len(experiments)

    lines = [
        f"# Run-to-run stability across {n} runs of identical code",
        "",
        f"Generated {datetime.now(timezone.utc).isoformat()}",
        "",
        "Runs compared:",
        *[f"- `{e}`" for e in experiments],
        "",
        "A case passing every run is a result. A case that flips is measuring sampling",
        "noise, and tuning the agent will not move it.",
        "",
    ]

    flaky_by_key = defaultdict(list)
    for case_id, keys in scores.items():
        for key, values in keys.items():
            if classify(values) == "FLAKY":
                flaky_by_key[key].append((case_id, values))

    lines += ["## Stability by evaluator", "", "| evaluator | stable-pass | stable-fail | FLAKY |", "|---|---:|---:|---:|"]
    for key in SCORE_KEYS:
        counts = defaultdict(int)
        for case_id in scores:
            counts[classify(scores[case_id][key])] += 1
        lines.append(
            f"| `{key}` | {counts['stable-pass']} | {counts['stable-fail']} | **{counts['FLAKY']}** |"
        )

    lines += ["", "## Flaky cases — these are noise, not findings", ""]
    if not any(flaky_by_key.values()):
        lines.append("None. Every graded case returned the same verdict on every run.")
    for key, cases in flaky_by_key.items():
        lines.append(f"**`{key}`**")
        lines.append("")
        for case_id, values in sorted(cases):
            shown = "".join("1" if v else "0" if v is not None else "-" for v in values)
            passes = sum(1 for v in values if v)
            lines.append(f"- `{case_id}` — {passes}/{len([v for v in values if v is not None])} ({shown})")
        lines.append("")

    lines += ["## Stable failures — these are real and worth fixing", ""]
    stable_fails = sorted(
        {c for c in scores for k in SCORE_KEYS if classify(scores[c][k]) == "stable-fail"}
    )
    if not stable_fails:
        lines.append("None.")
    for case_id in stable_fails:
        failed = [k for k in SCORE_KEYS if classify(scores[case_id][k]) == "stable-fail"]
        info = meta.get(case_id, {})
        lines.append(f"- `{case_id}` ({info.get('category','?')}) — failed every run on: {', '.join(failed)}")

    lines += ["", "## Per-case detail", "", "| case_id | " + " | ".join(SCORE_KEYS) + " |",
              "|---|" + "---|" * len(SCORE_KEYS)]
    for case_id in sorted(scores):
        cells = []
        for key in SCORE_KEYS:
            values = scores[case_id][key]
            verdict = classify(values)
            shown = "".join("1" if v else "0" if v is not None else "-" for v in values)
            cells.append("—" if verdict == "not-graded" else f"{shown} {verdict.replace('stable-','')}")
        lines.append(f"| {case_id} | " + " | ".join(cells) + " |")

    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare repeated runs of the same eval suite")
    parser.add_argument("--experiments", nargs="*", help="Explicit experiment names.")
    parser.add_argument("--prefix", help="Take the newest experiments whose name starts with this.")
    parser.add_argument("--limit", type=int, default=3)
    parser.add_argument("--dataset", default=DATASET_NAME)
    args = parser.parse_args()

    client = Client()
    if args.experiments:
        names = args.experiments
    else:
        found = list_experiments(client, args.dataset, limit=50)
        if args.prefix:
            found = [p for p in found if p.name.startswith(args.prefix)]
        names = [p.name for p in found[: args.limit]]

    if len(names) < 2:
        raise SystemExit(f"Need at least 2 runs to compare; found {len(names)}: {names}")

    print(f"Comparing {len(names)} runs:")
    for n in names:
        print(f"  {n}")
    print()

    data = collect(client, names, args.dataset)
    report = build_report(data, names)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    path = RESULTS_DIR / f"stability-{stamp}.md"
    path.write_text(report, encoding="utf-8")
    print(report)
    print(f"\nWritten to {path}")


if __name__ == "__main__":
    main()
