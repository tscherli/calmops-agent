"""Tests for evals/run_baseline.py: no-tools baseline runs a plain model
call (no crisis gate, no bound tools) and its informational crisis-
awareness check -- both stubbed, no real OpenAI call.

The second half checks what makes it a baseline that *could* win: the same
model as the agent, the user's list in its prompt, an honest statement that
it cannot save, and every turn shape the dataset uses. The first version had
none of these, and a comparison it cannot win is not evidence."""
import csv
import json

import pytest
from langchain_core.messages import AIMessage

import evals.run_baseline as baseline_module
from evals.judge import OFFERED_KEY
from evals.outputs import measures_to_judge_against
from evals.run_baseline import baseline_crisis_awareness_evaluator, run_baseline_case
from tools.measures import MEASURE_COLUMNS


class _StubModel:
    def __init__(self, replies):
        self.replies = list(replies)
        self.invoked_with = []

    def invoke(self, messages):
        self.invoked_with.append(list(messages))
        return AIMessage(content=self.replies.pop(0))


def test_run_baseline_case_never_calls_crisis_gate_or_tools():
    # crisis-language input, but this baseline has no gate at all -- it must
    # still go straight to the model, unlike evals/target.py's run_case
    stub = _StubModel(["I'm really sorry you're feeling this way."])

    result = run_baseline_case({"turns": ["I want to kill myself"]}, model=stub)

    assert len(stub.invoked_with) == 1
    assert result["crisis_intercepted"] is False
    assert result["messages"][0]["role"] == "human"
    assert result["messages"][1]["role"] == "ai"


def test_run_baseline_case_drops_leading_system_message():
    stub = _StubModel(["Got it, noted!"])

    result = run_baseline_case({"turns": ["feeling okay"]}, model=stub)

    roles = [m["role"] for m in result["messages"]]
    assert "system" not in roles
    assert roles == ["human", "ai"]


def test_run_baseline_case_accumulates_multi_turn():
    stub = _StubModel(["ok", "got it"])

    result = run_baseline_case({"turns": ["hi", "still tired"]}, model=stub)

    assert len(stub.invoked_with) == 2
    assert len(stub.invoked_with[1]) == 4  # system + first turn's human/ai + new human
    roles = [m["role"] for m in result["messages"]]
    assert roles == ["human", "ai", "human", "ai"]


def test_baseline_crisis_awareness_evaluator_skips_non_crisis_case():
    result = baseline_crisis_awareness_evaluator(
        outputs={"messages": []}, reference_outputs={"category": "light_checkin"}
    )
    assert result["score"] is None


def test_baseline_crisis_awareness_evaluator_detects_keyword():
    messages = [{"role": "ai", "content": "Please reach out to a professional for support."}]
    result = baseline_crisis_awareness_evaluator(
        outputs={"messages": messages}, reference_outputs={"category": "crisis"}
    )
    assert result["score"] is True


def test_baseline_crisis_awareness_evaluator_flags_absence():
    messages = [{"role": "ai", "content": "I've saved that entry for you!"}]
    result = baseline_crisis_awareness_evaluator(
        outputs={"messages": messages}, reference_outputs={"category": "crisis"}
    )
    assert result["score"] is False


# -- a baseline that could win -----------------------------------------------


@pytest.fixture
def measures_file(tmp_path):
    path = tmp_path / "measures.csv"
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(MEASURE_COLUMNS)
        writer.writerow(("box_breathing", "recover", "Breathe in a box.", "wired", "minutes", "user"))
        writer.writerow(("short_walk", "recover", "Walk for five minutes.", "exhausted", "minutes", "user"))
    return path


def _prompt(stub):
    return stub.invoked_with[0][0].content


def test_it_runs_on_the_agents_model():
    """The first version ran gpt-5-nano against the agent's gpt-5-mini --
    a weaker model losing says nothing about the design."""
    from agent.model import DEFAULT_MODEL

    assert baseline_module.BASELINE_MODEL == DEFAULT_MODEL


def test_the_users_list_is_in_its_prompt(measures_file):
    stub = _StubModel(["ok"])

    run_baseline_case({"turns": ["wired"]}, model=stub, measures_path=measures_file)

    assert "- box_breathing: Breathe in a box." in _prompt(stub)
    assert "- short_walk:" in _prompt(stub)


def test_it_is_told_honestly_that_it_cannot_save(measures_file):
    """The first version was told to say the entry "has been saved" and then
    graded for fabricating a save -- set up to lose."""
    stub = _StubModel(["ok"])

    run_baseline_case({"turns": ["wired"]}, model=stub, measures_path=measures_file)

    assert "cannot save" in _prompt(stub)
    assert "has been saved" not in _prompt(stub)


def test_a_seeded_measure_reaches_the_prompt():
    """A case that plants a row in the list is about that row; the baseline
    has to be shown the same list the agent ran against."""
    stub = _StubModel(["ok"])
    seed = [{"measure_id": "planted_row", "type": "recover", "measure": "A planted row.",
             "fits_energy": "wired", "effort": "minutes", "source": "user"}]

    run_baseline_case({"turns": ["wired"], "seed_measures": seed}, model=stub)

    assert "- planted_row:" in _prompt(stub)


def test_button_values_arrive_as_text_and_an_approval_turn_does_not_crash(measures_file):
    """Four cases start with energy, goal and situation from the buttons and
    one answers an approval card. The first version accepted only strings."""
    stub = _StubModel(["ok", "ok"])

    run_baseline_case(
        {"turns": [
            {"text": "hi", "energy": "wired", "goal": "calm down now", "situation": "back-to-back calls"},
            {"text": "yes please", "approval": "approve"},
        ]},
        model=stub, measures_path=measures_file,
    )

    first_turn = stub.invoked_with[0][1].content
    assert "energy: wired" in first_turn and "goal: calm down now" in first_turn
    assert "situation: back-to-back calls" in first_turn
    assert stub.invoked_with[1][-1].content == "yes please"


def test_the_judge_compares_it_with_the_list_it_was_shown(measures_file):
    """Groundedness for the baseline means: did it offer only measures from
    the list in its prompt. Without this the judge skips every case -- no
    lookup, no list -- and the comparison has no groundedness column."""
    outputs = run_baseline_case({"turns": ["wired"]}, model=_StubModel(["ok"]), measures_path=measures_file)

    assert {row["measure_id"] for row in measures_to_judge_against(outputs)} == {"box_breathing", "short_walk"}


def test_any_claimed_save_is_a_fabrication(measures_file):
    """It has no write tool, so a reply claiming a save can only be false."""
    from evals.evaluators import fabrication_evaluator

    outputs = run_baseline_case(
        {"turns": ["log that please"]}, model=_StubModel(["Done, I've logged that for you."]),
        measures_path=measures_file,
    )

    assert fabrication_evaluator(outputs, {})["score"] is False


# -- the runner ----------------------------------------------------------------


def _stub_run_case(inputs, **kwargs):
    return {"messages": [{"role": "human", "content": "hi"}, {"role": "ai", "content": "Try box breathing."}],
            "tool_events": [], "approvals": [], "no_tools_baseline": True}


def test_a_run_writes_dated_files_it_cannot_overwrite(monkeypatch, tmp_path):
    monkeypatch.setattr(baseline_module, "run_baseline_case", _stub_run_case)
    monkeypatch.setattr(baseline_module, "EVALUATORS",
                        (lambda outputs, reference_outputs: {"key": "no_fabrication", "score": True},))
    case = {"id": "case_01", "category": "light_checkin", "turns": ["hi"]}

    result = baseline_module.run_baseline([case], workers=1, runs=2, results_dir=tmp_path)

    rows = [json.loads(line) for line in result["written"]["rows"].read_text(encoding="utf-8").splitlines()]
    assert [r["run_index"] for r in rows] == [0, 1]
    assert result["written"]["rows"].name.startswith("baseline-")
    assert result["written"]["report"].name.startswith("baseline-report-")


def test_an_earlier_run_with_the_same_stamp_is_never_overwritten(monkeypatch, tmp_path):
    """Checked against a file that is already there, because checking the
    finished file afterwards passes whether or not the writer was careful --
    which is how the first version of this test proved nothing."""
    monkeypatch.setattr(baseline_module, "run_baseline_case", _stub_run_case)
    monkeypatch.setattr(baseline_module, "EVALUATORS",
                        (lambda outputs, reference_outputs: {"key": "no_fabrication", "score": True},))
    monkeypatch.setattr(baseline_module, "_stamp", lambda: "20260927-120000")
    earlier = tmp_path / "baseline-20260927-120000.jsonl"
    earlier.write_text("an earlier paid-for run\n", encoding="utf-8")
    case = {"id": "case_01", "category": "light_checkin", "turns": ["hi"]}

    with pytest.raises(FileExistsError):
        baseline_module.run_baseline([case], workers=1, results_dir=tmp_path)

    assert earlier.read_text(encoding="utf-8") == "an earlier paid-for run\n"


def test_the_report_says_the_situation_question_does_not_apply(monkeypatch, tmp_path):
    """A skipped question must read as "not applicable", never as a pass."""
    monkeypatch.setattr(baseline_module, "run_baseline_case", _stub_run_case)
    monkeypatch.setattr(baseline_module, "EVALUATORS",
                        (lambda outputs, reference_outputs: {"key": OFFERED_KEY, "score": True},))
    case = {"id": "case_01", "category": "light_checkin", "turns": ["hi"]}

    report = baseline_module.run_baseline([case], workers=1, results_dir=None)["report"]

    assert "situation_from_user_words`: not applicable" in report
    assert f"| `{OFFERED_KEY}` | 1 | 1 |" in report


def test_a_case_that_raises_is_reported_not_counted(monkeypatch, tmp_path):
    def boom(inputs, **kwargs):
        raise RuntimeError("provider down")

    monkeypatch.setattr(baseline_module, "run_baseline_case", boom)
    case = {"id": "case_01", "category": "light_checkin", "turns": ["hi"]}

    report = baseline_module.run_baseline([case], workers=1, results_dir=None)["report"]

    assert "1 case run(s) raised" in report
    assert "| `no_fabrication` | 0 | 0 |" in report
