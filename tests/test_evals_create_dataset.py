"""Tests for evals/create_dataset.py: case_to_example shape and the
idempotent upsert-by-case_id logic -- stubbed langsmith Client, no real
network call."""
from evals.create_dataset import case_to_example, create_or_update_dataset, load_cases

_SIMPLE_CASE = {
    "id": "case_01",
    "query": "Doing okay today.",
    "difficulty": "simple",
    "category": "light_checkin",
    "expected_behavior": "Logs directly, no clarifying questions.",
    "reference_trajectory": [{"role": "human"}, {"role": "ai", "tool_calls": [{"name": "log_mood"}]}],
}

_MULTITURN_CASE = {
    "id": "case_02",
    "turns": ["hi", "still rough"],
    "difficulty": "medium",
    "category": "structuring",
    "expected_behavior": "Asks a clarifying question before logging.",
}


def test_load_cases_reads_all_lines():
    cases = load_cases()
    assert len(cases) == 28
    assert len({c["id"] for c in cases}) == 28  # unique ids


def test_case_to_example_wraps_single_query_as_turns():
    example = case_to_example(_SIMPLE_CASE)
    assert example["inputs"] == {"turns": ["Doing okay today."]}
    assert example["outputs"]["category"] == "light_checkin"
    assert example["outputs"]["messages"] == _SIMPLE_CASE["reference_trajectory"]
    assert example["metadata"] == {"case_id": "case_01", "difficulty": "simple", "category": "light_checkin"}


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
