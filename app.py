"""Tkinter chat window: the actual Job Load Agent UI.

Blocking first-run consent screen, a persistent row of quick-select mood
buttons (no LLM involved), and free-text chat that goes through
agent/safety.py's crisis gate first, then the real agent.

Entry point -- must load .env itself (flagged during stage 6: library
modules like agent/graph.py must not do this as an import side effect).
"""
import argparse
import json
import time
import tkinter as tk
from pathlib import Path
from tkinter import scrolledtext

from dotenv import load_dotenv

load_dotenv()

from langchain_core.messages import HumanMessage

from agent.graph import DEFAULT_MODEL, build_agent, message_text
from agent.safety import CRISIS_RESOURCES, screen_for_crisis
from observability.tracing import RunRecorder
from tools.csv_log import log_mood
from tools.mood_templates import MOOD_TEMPLATES

BASE_DIR = Path(__file__).resolve().parent
CONSENT_PATH = BASE_DIR / "state" / "consent_ack.json"

CONSENT_TEXT = (
    "Data & Privacy Disclaimer\n\n"
    "This tool shares what you type -- including personal reflections on your mood "
    "and work situations -- with OpenAI's API (to power the agent), and optionally "
    "LangSmith (for tracing). Mood entries themselves are saved to a local CSV file "
    "on your own machine, never sent to any storage service.\n\n"
    "If you are not comfortable with a large language model processing your "
    "personal information, do not use this software -- the quick-select buttons "
    "below let you log a mood without sharing anything personal.\n\n"
    "This is a self-reflection aid, not a substitute for professional mental health "
    "support; its replies are AI-generated and may be inaccurate. In a crisis, "
    "contact local emergency services or a crisis hotline directly, not this tool.\n\n"
    "Each user of this software supplies their own API key -- no data from anyone "
    "running this passes through the original developer's infrastructure."
)


def has_consented() -> bool:
    return CONSENT_PATH.exists()


def record_consent() -> None:
    CONSENT_PATH.parent.mkdir(parents=True, exist_ok=True)
    CONSENT_PATH.write_text(json.dumps({"acknowledged": True}), encoding="utf-8")


class ChatApp:
    """The chat window. model/recorder are injectable (same
    dependency-injection-for-testability pattern used throughout this
    project) so tests can stub the agent without a real OpenAI call."""

    def __init__(
        self,
        root: tk.Tk,
        *,
        skip_consent: bool = False,
        agent_graph=None,
        recorder: RunRecorder | None = None,
        opener_message: str | None = None,
    ) -> None:
        self.root = root
        self.root.title("Job Load Agent")
        self.messages: list = []
        self.graph = agent_graph if agent_graph is not None else build_agent()
        self.recorder = recorder if recorder is not None else RunRecorder()
        self._opener_message = opener_message

        if skip_consent or has_consented():
            self._build_chat_ui()
        else:
            self._build_consent_ui()

    # -- Consent screen --------------------------------------------------

    def _build_consent_ui(self) -> None:
        self.consent_frame = tk.Frame(self.root, padx=20, pady=20)
        self.consent_frame.pack(fill="both", expand=True)

        text_widget = tk.Text(self.consent_frame, wrap="word", width=64, height=18)
        text_widget.insert("1.0", CONSENT_TEXT)
        text_widget.configure(state="disabled")
        text_widget.pack(pady=(0, 15))

        button = tk.Button(self.consent_frame, text="I understand", command=self._on_consent_ack)
        button.pack()

    def _on_consent_ack(self) -> None:
        record_consent()
        self.consent_frame.destroy()
        self._build_chat_ui()

    # -- Main chat screen --------------------------------------------------

    def _build_chat_ui(self) -> None:
        tk.Label(
            self.root,
            text="Quick log — no AI, nothing shared:",
            font=("TkDefaultFont", 9, "italic"),
            fg="gray30",
        ).pack(fill="x", padx=10, pady=(10, 0), anchor="w")

        quick_frame = tk.Frame(self.root)
        quick_frame.pack(fill="x", padx=10, pady=(0, 0))
        for key, template in MOOD_TEMPLATES.items():
            tk.Button(
                quick_frame,
                text=template["label"],
                command=lambda k=key: self._on_quick_select(k),
            ).pack(side="left", padx=2, pady=2)

        self.history = scrolledtext.ScrolledText(self.root, state="disabled", width=70, height=22, wrap="word")
        self.history.pack(fill="both", expand=True, padx=10, pady=10)

        tk.Label(
            self.root,
            text="Or type freely to chat with the agent:",
            font=("TkDefaultFont", 9, "italic"),
            fg="gray30",
        ).pack(fill="x", padx=10, pady=(0, 0), anchor="w")

        input_frame = tk.Frame(self.root)
        input_frame.pack(fill="x", padx=10, pady=(0, 10))
        self.entry = tk.Entry(input_frame)
        self.entry.pack(side="left", fill="x", expand=True)
        self.entry.bind("<Return>", lambda _event: self._on_send())
        tk.Button(input_frame, text="Send", command=self._on_send).pack(side="left", padx=(5, 0))

        if self._opener_message:
            self._append_message("Agent", self._opener_message)

    def _append_message(self, sender: str, text: str) -> None:
        self.history.configure(state="normal")
        self.history.insert("end", f"{sender}: {text}\n\n")
        self.history.configure(state="disabled")
        self.history.see("end")

    # -- Quick-select: no LLM involved at all --------------------------------------------------

    def _on_quick_select(self, key: str) -> None:
        template = MOOD_TEMPLATES[key]
        try:
            # Calls the tool's own body directly, bypassing the agent graph --
            # so no LLM is involved and, deliberately, no semantic tagging
            # either (see tools/mood_templates.py's module docstring).
            result = log_mood.invoke(
                {
                    "mood": template["mood"],
                    "circumstances": template["circumstances"],
                    "energy": template.get("energy", ""),
                }
            )
        except Exception as exc:  # never let an unhandled exception reach the UI
            self._append_message("App", f"Something went wrong logging that: {exc}")
            return

        self._append_message("You", template["label"])
        if result["status"] == "ok":
            self._append_message("App", "Logged.")
        else:
            self._append_message("App", f"Couldn't save that entry: {result.get('error', 'unknown error')}")

    # -- Free text: crisis gate first, then the real agent --------------------------------------------------

    def _on_send(self) -> None:
        user_text = self.entry.get().strip()
        if not user_text:
            return
        self.entry.delete(0, "end")
        self._append_message("You", user_text)

        crisis = screen_for_crisis(user_text)
        if crisis.triggered:
            self._append_message("App", CRISIS_RESOURCES)
            return

        messages_before = self.messages + [HumanMessage(content=user_text)]
        try:
            start = time.time()
            result = self.graph.invoke({"messages": messages_before})
            latency = time.time() - start
        except Exception as exc:  # never let an unhandled exception reach the UI
            self._append_message("App", f"Something went wrong and I couldn't process that: {exc}")
            return

        self.messages = result["messages"]
        self.recorder.record_turn(messages_before, self.messages, DEFAULT_MODEL, latency)
        # message_text, not .content: with the built-in web_search bound the
        # model speaks Responses-API content blocks, and .content would put
        # raw dicts and base64 in the chat window.
        self._append_message("Agent", message_text(self.messages[-1]))


def main() -> None:
    parser = argparse.ArgumentParser(description="Job Load Agent chat window")
    parser.add_argument(
        "--skip-consent",
        action="store_true",
        help="Dev-only: skip the consent screen. Never use this as the default.",
    )
    args = parser.parse_args()

    root = tk.Tk()
    ChatApp(root, skip_consent=args.skip_consent)
    root.mainloop()


if __name__ == "__main__":
    main()
