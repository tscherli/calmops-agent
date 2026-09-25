"""Builds/upserts the LangSmith dataset backing the eval suite from
evals/dataset.jsonl. Idempotent -- safe to re-run whenever dataset.jsonl
changes; checks for an existing dataset/example set by name rather than
blindly re-creating (a full delete-and-recreate resync was considered and
rejected -- see the plan doc's "Tracing, cost & latency"-adjacent eval
section: it would silently wipe out any example a user promoted into this
dataset from a real trace via LangSmith's "Add to Dataset" UI action).

One Example per case, uniformly -- the one multi-turn case is not exploded
into multiple rows. inputs={"turns": [...]} always (a single-query case's
`query` gets wrapped as a 1-element list). outputs={"messages": [...],
"expected_behavior": ..., "category": ...} -- "messages" must be this exact
key, matching agentevals' trajectory-normalizer requirement.
"""
from __future__ import annotations

import json
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

from langsmith import Client

DATASET_NAME = "job-load-agent-evals"
DATASET_PATH = Path(__file__).resolve().parent / "dataset.jsonl"
CASE_IDS_PATH = Path(__file__).resolve().parent / "case_ids.txt"


def load_case_ids(path: Path = CASE_IDS_PATH) -> tuple[set[str], set[str]]:
    """The registered ids, and the ones that were retired.

    Case ids key every run-to-run comparison, so an id has to mean the same
    case forever. The registry is what makes reusing one a visible edit rather
    than a silent change of meaning -- see the file's own header.
    """
    live: set[str] = set()
    retired: set[str] = set()
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("retired "):
            retired.add(line[len("retired ") :].strip())
        else:
            live.add(line)
    return live, retired


def load_cases(path: Path = DATASET_PATH) -> list[dict]:
    cases = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                cases.append(json.loads(line))
    return cases


def case_to_example(case: dict) -> dict:
    """Convert one dataset.jsonl case into a LangSmith Example's
    inputs/outputs/metadata triple."""
    turns = case["turns"] if "turns" in case else [case["query"]]
    case_inputs: dict = {"turns": turns}
    # Harness flags travel in `inputs` rather than metadata: evaluate()'s
    # target function only receives inputs, and putting it here also makes
    # the flag visible on the example in the LangSmith UI.
    if case.get("simulate_write_failure"):
        case_inputs["simulate_write_failure"] = True
    if case.get("seed_entries"):
        case_inputs["seed_entries"] = case["seed_entries"]
    if case.get("seed_measures"):
        case_inputs["seed_measures"] = case["seed_measures"]
    case_outputs: dict = {
        "messages": case.get("reference_trajectory", []),
        "expected_behavior": case["expected_behavior"],
        "category": case["category"],
    }
    # The case's own assertions travel with the reference, not the inputs:
    # evaluators are handed `reference_outputs`, and the target must not be
    # able to read what it is about to be graded on.
    if case.get("expect"):
        case_outputs["expect"] = case["expect"]
    return {
        "inputs": case_inputs,
        "outputs": case_outputs,
        "metadata": {
            "case_id": case["id"],
            "difficulty": case["difficulty"],
            "category": case["category"],
        },
    }


def create_or_update_dataset(client: Client | None = None, cases: list[dict] | None = None) -> str:
    """Idempotent: creates the dataset if it doesn't exist, then upserts
    examples by case_id (matched via metadata) -- re-running after editing
    dataset.jsonl updates existing rows instead of duplicating them."""
    client = client or Client()
    cases = cases if cases is not None else load_cases()

    if not client.has_dataset(dataset_name=DATASET_NAME):
        dataset = client.create_dataset(
            dataset_name=DATASET_NAME,
            description=(
                "Job Load Agent eval suite -- hand-authored cases (simple/medium/hard) covering "
                "structuring depth, crisis gating, honesty/no-fabrication, prompt injection "
                "(user message, and past log entries), history use, proactivity calibration, "
                "and tool discipline."
            ),
        )
    else:
        dataset = client.read_dataset(dataset_name=DATASET_NAME)

    existing_by_case_id = {
        ex.metadata.get("case_id"): ex
        for ex in client.list_examples(dataset_id=dataset.id)
        if ex.metadata
    }

    to_create = []
    updated = 0
    for case in cases:
        example_data = case_to_example(case)
        existing = existing_by_case_id.get(case["id"])
        if existing is None:
            to_create.append(example_data)
        else:
            client.update_example(
                example_id=existing.id,
                inputs=example_data["inputs"],
                outputs=example_data["outputs"],
                metadata=example_data["metadata"],
            )
            updated += 1

    if to_create:
        client.create_examples(dataset_id=dataset.id, examples=to_create)

    print(f"Dataset '{DATASET_NAME}': {len(to_create)} created, {updated} updated, {len(cases)} total.")
    return str(dataset.id)


if __name__ == "__main__":
    create_or_update_dataset()
