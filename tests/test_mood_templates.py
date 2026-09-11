import csv

import pytest

from tools.csv_log import append_mood_row
from tools.mood_templates import MOOD_TEMPLATES

VALID_MOODS = {"good", "neutral", "bad"}
VALID_ENERGY = {"exhausted", "calm", "wired"}


def test_at_least_one_template_per_mood():
    moods_present = {t["mood"] for t in MOOD_TEMPLATES.values()}
    assert VALID_MOODS.issubset(moods_present)


def test_energy_is_used_by_at_least_one_template_per_level():
    """energy is optional per-template, but every value in the vocabulary
    should actually be exercised somewhere -- otherwise part of the field is
    dead code no button ever reaches."""
    energy_present = {t["energy"] for t in MOOD_TEMPLATES.values() if "energy" in t}
    assert VALID_ENERGY.issubset(energy_present)


@pytest.mark.parametrize("key", list(MOOD_TEMPLATES.keys()))
def test_each_template_has_required_fields(key):
    template = MOOD_TEMPLATES[key]
    assert template["mood"] in VALID_MOODS
    assert isinstance(template["circumstances"], str) and template["circumstances"].strip()
    assert isinstance(template["label"], str) and template["label"].strip()
    if "energy" in template:
        assert template["energy"] in VALID_ENERGY


@pytest.mark.parametrize("key", list(MOOD_TEMPLATES.keys()))
def test_each_template_can_be_logged_without_error(key, tmp_path):
    """Simulates a quick-select button click: template fields go straight
    into append_mood_row, no LLM involved."""
    template = MOOD_TEMPLATES[key]
    log_path = tmp_path / "mood_log.csv"

    result = append_mood_row(
        mood=template["mood"],
        circumstances=template["circumstances"],
        energy=template.get("energy"),
        path=log_path,
    )

    assert result["status"] == "ok"
    with log_path.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 1
    assert rows[0]["circumstances"] == template["circumstances"]


@pytest.mark.parametrize("key", list(MOOD_TEMPLATES.keys()))
def test_no_template_carries_tags(key, tmp_path):
    """Quick-select rows are deliberately never semantically tagged --
    tools/classify.py is an LLM call, and these buttons promise "no AI,
    nothing shared." Tagging them would quietly break that guarantee."""
    template = MOOD_TEMPLATES[key]
    log_path = tmp_path / "mood_log.csv"

    append_mood_row(
        mood=template["mood"],
        circumstances=template["circumstances"],
        energy=template.get("energy"),
        path=log_path,
    )

    with log_path.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    assert rows[0]["tags"] == ""
