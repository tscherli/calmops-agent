import csv

import pytest

from tools.csv_log import append_mood_row
from tools.mood_templates import MOOD_TEMPLATES

VALID_VALENCES = {"good", "neutral", "bad"}
VALID_ENERGY_LEVELS = {"low", "medium", "high"}


def test_at_least_one_template_per_valence():
    valences_present = {t["valence"] for t in MOOD_TEMPLATES.values()}
    assert VALID_VALENCES.issubset(valences_present)


def test_energy_level_is_used_by_at_least_one_template_per_level():
    """energy_level is optional per-template, but should actually be
    exercised for low and high somewhere -- otherwise the field is dead code
    no button ever reaches."""
    energy_levels_present = {
        t["energy_level"] for t in MOOD_TEMPLATES.values() if "energy_level" in t
    }
    assert {"low", "high"}.issubset(energy_levels_present)


@pytest.mark.parametrize("key", list(MOOD_TEMPLATES.keys()))
def test_each_template_has_required_fields(key):
    template = MOOD_TEMPLATES[key]
    assert template["valence"] in VALID_VALENCES
    assert isinstance(template["circumstances"], str) and template["circumstances"].strip()
    assert isinstance(template["label"], str) and template["label"].strip()
    if "energy_level" in template:
        assert template["energy_level"] in VALID_ENERGY_LEVELS


@pytest.mark.parametrize("key", list(MOOD_TEMPLATES.keys()))
def test_each_template_can_be_logged_without_error(key, tmp_path):
    """Simulates a quick-select button click: template fields go straight
    into append_mood_row, no LLM involved."""
    template = MOOD_TEMPLATES[key]
    log_path = tmp_path / "mood_log.csv"

    result = append_mood_row(
        valence=template["valence"],
        circumstances=template["circumstances"],
        energy_level=template.get("energy_level"),
        path=log_path,
    )

    assert result["status"] == "ok"
    with log_path.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 1
    assert rows[0]["circumstances"] == template["circumstances"]
