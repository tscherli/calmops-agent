"""evals/publish_results.py -- a snapshot that is safe to commit.

What matters: nothing personal leaves the machine, nothing that decides a
score is changed on the way, and the published score is the one anyone gets
by re-running it on the published file.
"""
import contextlib
import io
import json
from pathlib import Path

import pytest

import evals.publish_results as publish_module
from evals.calibrate_judge import score
from evals.judge import OFFERED_KEY, SITUATION_KEY

HOME = str(Path.home())
BLOB = "gAAAAABqt86QONAN6joesTsecret"
TOOL_CALL_CONTENT = (
    "[{'id': 'rs_1', 'summary': [], 'type': 'reasoning', 'content': [], "
    f"'encrypted_content': '{BLOB}'}}]"
)
NOTE = "is this not a duplicate? my private remark"


def _record(**extra_outputs):
    return {
        "case_id": "case_01",
        "category": "measure_choice",
        "run_index": 0,
        "outputs": {
            "messages": [
                {"role": "human", "content": "wired after a call"},
                {"role": "ai", "content": TOOL_CALL_CONTENT,
                 "tool_calls": [{"name": "find_measures", "args": {}}]},
                {"role": "ai", "content": "Try box breathing.", "tool_calls": None},
            ],
            "measures_path": f"{HOME}\\AppData\\Local\\Temp\\joblog-evalmeasures-x\\measures.csv",
            "tool_events": [],
            **extra_outputs,
        },
    }


def _items():
    items = []
    for i in range(20):
        items.append({"case_id": f"real_{i:02d}", "run": 0, "run_id": "", "human_score": 1,
                      "human_note": NOTE if i == 0 else "", "evidence": {}, "_planted": False,
                      "_judge_score": 1, "_judge_parsed": True, "_judge_comment": "fine"})
    for i in range(5):
        items.append({"case_id": f"plant_{i}", "run": 1, "run_id": "", "human_score": 0,
                      "human_note": NOTE, "evidence": {}, "_planted": True,
                      "_judge_score": 1 if i == 0 else 0, "_judge_parsed": True,
                      "_judge_comment": "missed"})
    return items


@pytest.fixture
def workspace(tmp_path):
    corpus = tmp_path / "trajectories-20260926-161714-merged.jsonl"
    corpus.write_text(json.dumps(_record()) + "\n", encoding="utf-8")
    labels = tmp_path / "judge-calibration.json"
    labels.write_text(json.dumps({
        "source": f"local:{corpus}",
        "questions": {OFFERED_KEY: {"items": _items()}, SITUATION_KEY: {"items": _items()}},
        "stability": {OFFERED_KEY: {"stable": True}, SITUATION_KEY: {"stable": True}},
    }), encoding="utf-8")
    report = tmp_path / "eval-report-20260926-140315.md"
    report.write_text("# Eval report\n\nLangSmith experiment: https://eu.smith.langchain.com/o/8c98/x\n\nrows\n",
                      encoding="utf-8")
    return {"corpus": corpus, "labels": labels, "report": report, "out": tmp_path / "published"}


def _publish(ws, **kwargs):
    return publish_module.publish(
        "grounding-calibration", [ws["report"]], label_path=ws["labels"], corpus=ws["corpus"],
        published_dir=ws["out"], commit="abc1234", dirty=False, **kwargs,
    )


def _all_text(folder):
    return {p.name: p.read_text(encoding="utf-8") for p in folder.iterdir()}


# -- what must never be published --------------------------------------------


def test_a_trajectory_loses_the_local_path_and_the_encrypted_blob():
    clean = publish_module.sanitise_trajectory(_record())

    text = json.dumps(clean)
    assert "measures_path" not in clean["outputs"]
    assert BLOB not in text
    assert HOME not in text and json.dumps(HOME)[1:-1] not in text


def test_a_path_quoted_inside_a_tool_error_loses_the_user_name():
    """The write-failure case's receipt quotes the path it failed on, repr'd
    inside the message, so the backslashes arrive doubled -- and then doubled
    again by JSON. The first real dry run was refused over exactly this."""
    home_in_repr = repr(f"{HOME}\\AppData\\Local\\Temp\\joblog-evalfail-x\\log.csv")
    record = _record()
    record["outputs"]["messages"].append(
        {"role": "tool", "name": "log_checkin", "content": f"Not logged: [WinError 183] {home_in_repr}"}
    )

    text = json.dumps(publish_module.sanitise_trajectory(record))

    assert Path.home().name not in text
    assert "<user>" in text


def test_a_spoken_reply_is_untouched():
    """The blobs sit on tool-calling messages only; a reply is what both
    raters judged and must reach the snapshot byte for byte."""
    clean = publish_module.sanitise_trajectory(_record())

    assert clean["outputs"]["messages"][2]["content"] == "Try box breathing."
    assert clean["outputs"]["messages"][0]["content"] == "wired after a call"


def test_the_published_snapshot_holds_no_note_path_or_link(workspace):
    folder = _publish(workspace)

    files = _all_text(folder)
    assert set(files) == {"trajectories.jsonl", "judge-calibration.json", "score.txt",
                          "eval-report-20260926-140315.md", "README.md"}
    everything = "\n".join(files.values())
    assert "my private remark" not in everything          # not even in score.txt
    assert BLOB not in everything
    assert "smith.langchain.com" not in everything
    assert HOME not in everything and json.dumps(HOME)[1:-1] not in everything
    assert json.loads(files["judge-calibration.json"])["source"] == (
        "local:trajectories-20260926-161714-merged.jsonl"
    )


def test_anything_personal_that_survives_stops_the_publish(workspace):
    """The sanitisers remove what is known. This catches what is not: a home
    path in a field nobody thought of -- here a judge comment in the label
    file, which no sanitiser touches. Nothing is written when it fires."""
    payload = json.loads(workspace["labels"].read_text(encoding="utf-8"))
    payload["questions"][OFFERED_KEY]["items"][3]["_judge_comment"] = f"see {HOME}\\notes.txt"
    workspace["labels"].write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(SystemExit, match="personal data"):
        _publish(workspace)

    assert not workspace["out"].exists() or not any(workspace["out"].iterdir())


# -- nothing that decides a score changes ------------------------------------


def test_the_published_score_is_what_rescoring_the_published_file_gives(workspace):
    folder = _publish(workspace)

    buffer = io.StringIO()
    with contextlib.redirect_stdout(buffer):
        score(path=folder / "judge-calibration.json")

    assert buffer.getvalue() == (folder / "score.txt").read_text(encoding="utf-8")


def test_labels_and_verdicts_arrive_unchanged(workspace):
    folder = _publish(workspace)

    before = json.loads(workspace["labels"].read_text(encoding="utf-8"))
    after = json.loads((folder / "judge-calibration.json").read_text(encoding="utf-8"))
    keep = ("case_id", "human_score", "_judge_score", "_judge_parsed", "_planted", "_judge_comment")
    for key in (OFFERED_KEY, SITUATION_KEY):
        for old, new in zip(before["questions"][key]["items"], after["questions"][key]["items"]):
            assert {k: old[k] for k in keep} == {k: new[k] for k in keep}


# -- provenance --------------------------------------------------------------


def test_the_readme_names_the_commit_and_how_to_recheck(workspace):
    readme = (_publish(workspace) / "README.md").read_text(encoding="utf-8")

    assert "abc1234" in readme
    assert "calibrate_judge score --file" in readme


def test_an_uncommitted_tree_is_refused(workspace):
    """The README records a commit; a dirty tree means that commit is not
    the code that made the snapshot."""
    with pytest.raises(SystemExit, match="uncommitted"):
        publish_module.publish(
            "x", [], label_path=workspace["labels"], corpus=workspace["corpus"],
            published_dir=workspace["out"], commit="abc1234", dirty=True,
        )
