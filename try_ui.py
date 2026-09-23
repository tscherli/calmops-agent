"""Ad-hoc, non-pytest scratch script: the real window, a fake agent.

    python try_ui.py

For working on the UI itself -- layout, wording, when a button appears -- and
for seeing which handler a click reaches. The window is the real `ChatApp`; only
the agent behind it is scripted, so nothing calls OpenAI, nothing costs money,
and no real file is touched: the measure list is a copy in the temp directory
and the check-in log is a fresh file there.

Press F5 in the window to restart it with your edited code. Tkinter has no hot
reload, so that is the loop: edit, F5, look.

The scripted agent answers in a fixed cycle: it records what a click said, then
replies. Clicks, the card and the rating buttons all work; the replies are
nonsense on purpose, so nothing here can be mistaken for real behaviour.
"""
from __future__ import annotations

import os
import shutil
import sys
import tempfile
import tkinter as tk
from pathlib import Path

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage
from langchain_core.outputs import ChatGeneration, ChatResult

from agent.graph import build_agent
from app import ChatApp

PROJECT = Path(__file__).resolve().parent
SANDBOX = Path(tempfile.gettempdir()) / "job_load_agent_ui_sandbox"

# Which handler each button reaches, printed at startup: the fastest way to
# find where to put a breakpoint for the button you are looking at.
BUTTON_MAP = [
    ("I understand (consent)", "_on_consent_ack", 157),
    ("Send", "_on_send", 442),
    ("energy / goal", "_on_choice_click", 350),
    ("helped / somewhat / not helpful", "_on_rating_click", 367),
    ("discuss my idea", "_on_idea_click", 399),
    ("Approve / Save my edit / Reject", "_on_decision", 239),
]


class ScriptedModel(BaseChatModel):
    """Answers in a fixed cycle, so the window can be used without an API key."""

    turn: int = 0

    def _generate(self, messages, stop=None, run_manager=None, **kwargs) -> ChatResult:
        self.turn += 1
        if self.turn % 3 == 1:
            reply = AIMessage(
                content="",
                tool_calls=[{
                    "name": "find_measures",
                    "args": {},
                    "id": f"ui-{self.turn}",
                }],
            )
        else:
            reply = AIMessage(content=f"(scripted reply {self.turn}) What would help most right now?")
        return ChatResult(generations=[ChatGeneration(message=reply)])

    def bind_tools(self, tools, *, tool_choice=None, **kwargs):
        return self

    @property
    def _llm_type(self) -> str:
        return "scripted-ui-model"


def sandbox_paths() -> tuple[Path, Path]:
    """A copy of the measure list and an empty log, both in the temp folder."""
    SANDBOX.mkdir(parents=True, exist_ok=True)
    measures = SANDBOX / "measures.csv"
    shutil.copyfile(PROJECT / "tools" / "measures.csv", measures)
    log = SANDBOX / "checkin_log.csv"
    log.unlink(missing_ok=True)
    return measures, log


def restart() -> None:
    """Re-exec this script, so an edit shows up without touching the terminal."""
    os.execv(sys.executable, [sys.executable, *sys.argv])


def main() -> int:
    measures, log = sandbox_paths()
    print("UI sandbox -- no API calls, no real files touched")
    print(f"  measure list : {measures}")
    print(f"  check-in log : {log}")
    print("  F5 in the window restarts it with your edited code\n")
    print("  button -> handler in app.py")
    for label, handler, line in BUTTON_MAP:
        print(f"    {label:34} {handler:20} app.py:{line}")

    graph = build_agent(model=ScriptedModel(), measures_path=measures, log_path=log)
    root = tk.Tk()
    ChatApp(root, skip_consent=True, agent_graph=graph, log_path=log)
    root.bind("<F5>", lambda _event: restart())
    root.mainloop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
