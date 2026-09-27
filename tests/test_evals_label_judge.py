"""evals/label_judge.py -- the terminal labeller.

Two guarantees matter here and neither is about convenience:

- the judge's answer and the planted flag never reach the screen, because a
  labeller who sees either is no longer an independent rater and kappa stops
  describing anything;
- the question shown is the judge's own wording, lifted from its prompt rather
  than restated, because two copies of a rubric drift and a human and a model
  answering differently-worded questions produce agreement about nothing.
"""
import json

import pytest

from evals.judge import JUDGE_KEYS, OFFERED_KEY, OFFERED_PROMPT, SITUATION_KEY
from evals.label_judge import label, question_text, remaining, render


def _item(case_id="simple_x_01", **overrides):
    item = {
        "case_id": case_id,
        "run_id": "",
        "human_score": None,
        "human_note": "",
        "evidence": {
            "measures_the_agent_was_given": ["drink_water: Go grab a glass of water."],
            "what_the_user_said": ["exhausted, long week"],
            "a_web_search_ran": False,
            "the_agents_final_reply": "Try a glass of water.",
        },
        "_planted": False,
        "_judge_score": 0,
        "_judge_parsed": True,
        "_judge_comment": "the reply offered a cold shower, which is not in the list",
    }
    item.update(overrides)
    return item


def _payload(*items, key=OFFERED_KEY):
    return {"source": "local", "questions": {key: {"items": list(items)}}}


class _Reader:
    def __init__(self, *answers):
        self.answers = list(answers)
        self.prompts = []

    def __call__(self, prompt=""):
        self.prompts.append(prompt)
        return self.answers.pop(0) if self.answers else "q"


# -- what must never reach the screen ----------------------------------------


def test_the_judges_verdict_is_never_rendered():
    screen = render(_item(), OFFERED_KEY, 1, 10)

    assert "cold shower" not in screen
    assert "_judge" not in screen


def test_a_planted_item_does_not_announce_itself():
    """Knowing an item was edited to contain a violation is knowing the
    answer. The gate checks separately that every plant was caught, so a bad
    plant surfaces there rather than being absorbed into the agreement."""
    screen = render(_item(_planted=True), OFFERED_KEY, 1, 10)

    assert "planted" not in screen.lower()


def test_the_evidence_is_rendered_under_its_own_field_names():
    """What is on screen and what is in the JSON should be the same thing when
    someone goes looking for an item afterwards."""
    screen = render(_item(), OFFERED_KEY, 1, 10)

    assert "what the user said" in screen
    assert "exhausted, long week" in screen
    assert "Try a glass of water." in screen


def test_a_boolean_reads_as_a_word():
    assert "a web search ran:\n  no" in render(_item(), OFFERED_KEY, 1, 10)


def _planted_grounding_item():
    """Built by the real exporter's own functions, so the test follows the
    shape the file actually has rather than one written out by hand."""
    from evals.calibrate_judge import evidence_for, plant_offered_violation

    outputs = {
        "messages": [
            {"role": "human", "content": "wired, cant settle"},
            {
                "role": "tool",
                "name": "find_measures",
                "content": json.dumps([{"measure_id": "drink_water", "measure": "Go grab a glass of water."}]),
            },
            {"role": "ai", "content": "Try a glass of water.", "tool_calls": None},
        ],
        "tool_events": [],
    }
    return _item(evidence=evidence_for(OFFERED_KEY, plant_offered_violation(outputs)))


def test_the_final_reply_is_on_screen_once():
    """It is the last conversation turn and its own field as well. Shown
    twice in a row, the second copy reads as already seen, and the second
    labelling session missed the planted sentence -- the end of that second
    copy -- in all five plants."""
    screen = render(_planted_grounding_item(), OFFERED_KEY, 1, 10)

    assert screen.count("knead bread dough") == 1
    assert "shown in full below" in screen


def test_showing_it_once_leaves_the_file_alone():
    """Labels are carried across re-exports by a fingerprint of the evidence.
    A screen fix that edited the stored evidence would orphan every label."""
    item = _planted_grounding_item()
    before = json.dumps(item["evidence"], sort_keys=True)

    render(item, OFFERED_KEY, 1, 10)

    assert json.dumps(item["evidence"], sort_keys=True) == before


def test_a_conversation_that_does_not_end_on_the_reply_is_shown_whole():
    item = _planted_grounding_item()
    item["evidence"]["the_conversation_in_order"].append("user: one more thing")

    screen = render(item, OFFERED_KEY, 1, 10)

    assert "shown in full below" not in screen
    assert "one more thing" in screen


def test_an_empty_list_says_so_rather_than_rendering_nothing():
    item = _item()
    item["evidence"]["what_the_user_said"] = []

    assert "(none)" in render(item, OFFERED_KEY, 1, 10)


# -- the question is the judge's own -----------------------------------------


@pytest.mark.parametrize("key", JUDGE_KEYS)
def test_every_question_has_text_to_show(key):
    assert len(question_text(key)) > 80


def test_the_question_is_lifted_from_the_judges_prompt_not_restated():
    """A human and a model scored against two wordings of "the same" question
    produce a kappa that describes nothing."""
    text = question_text(OFFERED_KEY)

    assert text in OFFERED_PROMPT
    assert "not one of the measures above" in text


def test_the_question_carries_the_allowances_a_labeller_needs():
    """Both of these decide real items: a measure the user raised, and ideas
    after a web search."""
    text = question_text(OFFERED_KEY)

    assert "the user themselves brought up" in text
    assert "web search" in text


def test_the_situation_question_says_there_may_be_several_lines():
    """It reads a card's sentence and a recorded situation, so a labeller told
    to judge "the proposed sentence" would be answering a narrower question
    than the judge was asked."""
    assert "more than one line" in question_text(SITUATION_KEY)


# -- the loop ----------------------------------------------------------------


def test_an_answer_is_written_to_the_file_before_the_next_item(tmp_path):
    """Stopping has to be free: the labeller is an hour of work and a crash
    must not cost more than the item in hand."""
    path = tmp_path / "labels.json"
    payload = _payload(_item("a"), _item("b"))

    label(payload, path, reader=_Reader("1", "q"), printer=lambda _: None)

    written = json.loads(path.read_text(encoding="utf-8"))
    assert written["questions"][OFFERED_KEY]["items"][0]["human_score"] == 1


def test_labelling_both_answers(tmp_path):
    payload = _payload(_item("a"), _item("b"))

    assert label(payload, tmp_path / "l.json", reader=_Reader("1", "0"), printer=lambda _: None) == 2
    scores = [i["human_score"] for i in payload["questions"][OFFERED_KEY]["items"]]
    assert scores == [1, 0]


def test_skipping_leaves_the_item_unlabelled(tmp_path):
    payload = _payload(_item("a"), _item("b"))

    label(payload, tmp_path / "l.json", reader=_Reader("s", "1"), printer=lambda _: None)

    scores = [i["human_score"] for i in payload["questions"][OFFERED_KEY]["items"]]
    assert scores == [None, 1]


def test_an_unrecognised_key_asks_again_rather_than_guessing(tmp_path):
    """A stray keystroke must not silently become a score."""
    payload = _payload(_item("a"))
    lines = []

    label(payload, tmp_path / "l.json", reader=_Reader("x", "1"), printer=lines.append)

    assert payload["questions"][OFFERED_KEY]["items"][0]["human_score"] == 1
    assert any("Not one of the keys" in line for line in lines)


def test_a_note_is_kept_and_the_item_still_needs_an_answer(tmp_path):
    payload = _payload(_item("a"))

    label(payload, tmp_path / "l.json", reader=_Reader("n", "unsure about the walk", "0"),
          printer=lambda _: None)

    item = payload["questions"][OFFERED_KEY]["items"][0]
    assert item["human_note"] == "unsure about the walk"
    assert item["human_score"] == 0


def test_quitting_saves_what_was_done(tmp_path):
    path = tmp_path / "l.json"
    payload = _payload(_item("a"), _item("b"))

    assert label(payload, path, reader=_Reader("1", "q"), printer=lambda _: None) == 1
    assert json.loads(path.read_text(encoding="utf-8"))["questions"][OFFERED_KEY]["items"][1][
        "human_score"
    ] is None


def test_a_second_session_only_offers_what_is_left(tmp_path):
    payload = _payload(_item("a", human_score=1), _item("b"))
    seen = []

    label(payload, tmp_path / "l.json", reader=_Reader("0"), printer=seen.append)

    assert any("item 1 of 1" in line for line in seen)
    assert remaining(payload)[OFFERED_KEY] == 0


def test_revisiting_offers_everything_again(tmp_path):
    payload = _payload(_item("a", human_score=1), _item("b"))
    seen = []

    label(payload, tmp_path / "l.json", reader=_Reader("1", "0"), printer=seen.append,
          revisit=True)

    assert any("item 1 of 2" in line for line in seen)


def test_the_question_is_shown_again_on_demand(tmp_path):
    payload = _payload(_item("a"))
    seen = []

    label(payload, tmp_path / "l.json", reader=_Reader("?", "1"), printer=seen.append)

    # The question is wrapped, so it is counted over the whole session's
    # output with the line breaks flattened rather than line by line.
    flat = " ".join(" ".join(seen).split())
    assert flat.count("not one of the measures above") >= 2


def test_smart_punctuation_is_flattened_for_the_console():
    """A Windows console at its default codepage turns the model's em dashes
    and curly quotes into replacement characters, which is a poor thing to ask
    someone to read fifty times. The file keeps what the agent actually said."""
    item = _item()
    item["evidence"]["the_agents_final_reply"] = "I hear you — that’s heavy."

    screen = render(item, OFFERED_KEY, 1, 1)

    assert "I hear you - that's heavy." in screen
    assert "�" not in screen


def test_a_wrapped_bullet_stays_part_of_its_bullet():
    """A continuation line returning to the bullet's own indent reads as the
    next bullet, and the question is "is every one of these in the list"."""
    item = _item()
    item["evidence"]["measures_the_agent_was_given"] = [
        "plan_wellness_day: " + "Block out a whole day for gentle self-care " * 3
    ]

    lines = render(item, OFFERED_KEY, 1, 1).splitlines()
    continuation = [l for l in lines if l.startswith("    ") and "gentle" in l]

    assert continuation, lines


def test_the_screen_says_which_run_the_item_came_from():
    """A case id can appear more than once: the suite is run three times and
    two runs of one case are two different conversations. Without the run on
    screen the second reads as the file repeating itself, which is what the
    first full session concluded -- it marked real items as duplicates and a
    defect, and that took kappa to 0.01."""
    screen = render(_item(run=1), OFFERED_KEY, 3, 25)

    assert "run 2" in screen


def test_an_item_with_no_run_recorded_says_nothing_about_runs():
    """Older files and single-run exports have no run index, and inventing
    one would be worse than leaving it out."""
    item = _item()
    item.pop("run", None)

    assert "run" not in render(item, OFFERED_KEY, 1, 1).splitlines()[1]
