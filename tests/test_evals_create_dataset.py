"""Tests for evals/create_dataset.py: case_to_example shape and the
idempotent upsert-by-case_id logic -- stubbed langsmith Client, no real
network call.

Rules *about the dataset* live in tests/test_evals_dataset.py. This module
tests the code that reads and converts it, on files it writes itself.
"""
import json

from evals.create_dataset import case_to_example, create_or_update_dataset, load_cases

_SIMPLE_CASE = {
    "id": "case_01",
    "query": "Doing okay today.",
    "difficulty": "simple",
    "category": "light_checkin",
    "expected_behavior": "Logs directly, no clarifying questions.",
    "reference_trajectory": [{"role": "human"}, {"role": "ai", "tool_calls": [{"name": "log_checkin"}]}],
}

_MULTITURN_CASE = {
    "id": "case_02",
    "turns": ["hi", "still rough"],
    "difficulty": "medium",
    "category": "structuring",
    "expected_behavior": "Asks a clarifying question before logging.",
}


def test_load_cases_reads_every_line_of_the_file_it_is_given(tmp_path):
    """The loader's own job, tested on a file the test writes.

    This used to assert facts about the live corpus -- at least 28 cases, a
    seeded case exists, a failure-flag case exists. Those are rules about the
    dataset, and they moved to tests/test_evals_dataset.py when the mood-era
    set was archived (2026-09-25). A loader test that fails because the
    dataset changed is testing the wrong thing.
    """
    path = tmp_path / "cases.jsonl"
    path.write_text(
        json.dumps(_SIMPLE_CASE) + "\n\n" + json.dumps(_MULTITURN_CASE) + "\n",
        encoding="utf-8",
    )

    cases = load_cases(path)

    assert [c["id"] for c in cases] == ["case_01", "case_02"]


def test_load_cases_ignores_blank_lines(tmp_path):
    path = tmp_path / "cases.jsonl"
    path.write_text("\n\n" + json.dumps(_SIMPLE_CASE) + "\n\n", encoding="utf-8")

    assert len(load_cases(path)) == 1


def test_the_harness_flags_reach_the_target_through_inputs():
    """These flags only do anything if case_to_example forwards them --
    evaluate() hands the target function `inputs` and nothing else."""
    seeded = {**_SIMPLE_CASE, "seed_entries": [{"energy": "wired", "circumstances": "a loud day"}]}
    assert case_to_example(seeded)["inputs"]["seed_entries"] == seeded["seed_entries"]

    planted = {**_SIMPLE_CASE, "seed_measures": [{"measure_id": "planted", "measure": "Do it."}]}
    assert case_to_example(planted)["inputs"]["seed_measures"] == planted["seed_measures"]

    failing = {**_SIMPLE_CASE, "simulate_write_failure": True}
    assert case_to_example(failing)["inputs"]["simulate_write_failure"] is True


def test_case_to_example_wraps_single_query_as_turns():
    example = case_to_example(_SIMPLE_CASE)
    assert example["inputs"] == {"turns": ["Doing okay today."]}
    assert example["outputs"]["category"] == "light_checkin"
    assert example["outputs"]["messages"] == _SIMPLE_CASE["reference_trajectory"]
    assert example["metadata"] == {
        "case_id": "case_01",
        "difficulty": "simple",
        "category": "light_checkin",
        "tier": "regression",
        "covers": [],
    }


def test_case_to_example_preserves_existing_turns_list():
    example = case_to_example(_MULTITURN_CASE)
    assert example["inputs"] == {"turns": ["hi", "still rough"]}


def test_case_to_example_defaults_missing_reference_trajectory_to_empty():
    example = case_to_example(_MULTITURN_CASE)
    assert example["outputs"]["messages"] == []


class _StubDataset:
    def __init__(self, dataset_id="ds-1"):
        self.id = dataset_id


class _StubExample:
    def __init__(self, example_id, metadata):
        self.id = example_id
        self.metadata = metadata


class _StubClient:
    def __init__(self, existing_examples=None, dataset_exists=False):
        self._dataset_exists = dataset_exists
        self._examples = existing_examples or []
        self.created_dataset = False
        self.created_examples = []
        self.updated_examples = []

    def has_dataset(self, dataset_name):
        return self._dataset_exists

    def create_dataset(self, dataset_name, description):
        self.created_dataset = True
        return _StubDataset()

    def read_dataset(self, dataset_name):
        return _StubDataset()

    def list_examples(self, dataset_id):
        return self._examples

    def create_examples(self, dataset_id, examples):
        self.created_examples.extend(examples)

    def update_example(self, example_id, inputs, outputs, metadata):
        self.updated_examples.append((example_id, inputs, outputs, metadata))


def test_create_or_update_dataset_creates_new_dataset_and_all_examples():
    client = _StubClient(dataset_exists=False, existing_examples=[])

    dataset_id = create_or_update_dataset(client=client, cases=[_SIMPLE_CASE, _MULTITURN_CASE])

    assert dataset_id == "ds-1"
    assert client.created_dataset is True
    assert len(client.created_examples) == 2
    assert client.updated_examples == []


def test_create_or_update_dataset_updates_existing_case_instead_of_duplicating():
    existing = [_StubExample("ex-1", {"case_id": "case_01"})]
    client = _StubClient(dataset_exists=True, existing_examples=existing)

    create_or_update_dataset(client=client, cases=[_SIMPLE_CASE])

    assert len(client.updated_examples) == 1
    assert client.updated_examples[0][0] == "ex-1"
    assert client.created_examples == []


def test_the_suite_tier_travels_with_the_case():
    """The release gate is applied by hand from the report, so a case that
    blocks a tag has to say so somewhere the report can read."""
    case = {**_SIMPLE_CASE, "tier": "progression", "covers": ["LOG_PROMPT: asks before logging"]}

    metadata = case_to_example(case)["metadata"]

    assert metadata["tier"] == "progression"
    assert metadata["covers"] == ["LOG_PROMPT: asks before logging"]


def test_a_case_with_no_tier_is_treated_as_blocking():
    """The safe default. A case silently downgraded to "tracked" would stop
    blocking a tag without anyone deciding that."""
    assert case_to_example(_SIMPLE_CASE)["metadata"]["tier"] == "regression"
