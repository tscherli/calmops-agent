"""Verifies each quick-select button maps to the correct mood_templates.py
entry and never invokes the LLM/agent -- stage 8. Data-shape correctness of
the templates themselves is tests/test_mood_templates.py's job, not this
file's."""
import csv
import tkinter as tk

import pytest

import tools.csv_log as csv_log_module
from app import ChatApp
from tools.mood_templates import MOOD_TEMPLATES


class _PoisonGraph:
    """A stub agent graph that fails loudly if ever invoked -- proves
    quick-select buttons never touch the LLM/agent."""

    def invoke(self, *args, **kwargs):
        raise AssertionError("Quick-select buttons must never invoke the agent graph")


@pytest.fixture(autouse=True)
def isolated_log_path(tmp_path, monkeypatch):
    log_path = tmp_path / "mood_log.csv"
    monkeypatch.setattr(csv_log_module, "LOG_PATH", log_path)
    return log_path


@pytest.fixture
def tk_root():
    root = tk.Tk()
    root.withdraw()
    yield root
    root.destroy()


@pytest.fixture
def app(tk_root):
    return ChatApp(tk_root, skip_consent=True, agent_graph=_PoisonGraph())


def _read_rows(path):
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


@pytest.mark.parametrize("key", list(MOOD_TEMPLATES.keys()))
def test_quick_select_button_logs_correct_template_no_llm(app, isolated_log_path, key):
    app._on_quick_select(key)

    template = MOOD_TEMPLATES[key]
    rows = _read_rows(isolated_log_path)
    assert len(rows) == 1
    assert rows[0]["valence"] == template["valence"]
    assert rows[0]["circumstances"] == template["circumstances"]
    assert rows[0]["energy_level"] == template.get("energy_level", "")


def test_quick_select_appends_to_visible_history(app, isolated_log_path):
    app._on_quick_select("good_general")

    history_text = app.history.get("1.0", "end")
    assert MOOD_TEMPLATES["good_general"]["label"] in history_text
    assert "Logged" in history_text
