"""Tkinter chat window: the actual Job Load Agent UI.

Blocking first-run consent screen, then free-text chat that goes through
agent/safety.py's crisis gate first and the station graph second.

Entry point -- must load .env itself (flagged during stage 6: library
modules like agent/graph.py must not do this as an import side effect).
"""
import argparse
import json
import queue
import threading
import time
import tkinter as tk
from collections.abc import Sequence
from pathlib import Path
from tkinter import scrolledtext

from dotenv import load_dotenv

load_dotenv()

from langchain_core.messages import HumanMessage

from agent.common import message_text
from agent.graph import build_agent, next_turn_state
from agent.model import DEFAULT_MODEL
from agent.safety import CRISIS_RESOURCES, screen_for_crisis
from observability.tracing import RunRecorder

# How often the main thread checks the worker's queue while a turn runs.
# Small enough that a receipt looks instant, large enough to cost nothing.
_POLL_MS = 50

LOG_TOOL_NAME = "log_checkin"


BASE_DIR = Path(__file__).resolve().parent
CONSENT_PATH = BASE_DIR / "state" / "consent_ack.json"

CONSENT_TEXT = (
    "Data & Privacy Disclaimer\n\n"
    "This tool shares what you type -- including personal reflections on your mood "
    "and work situations -- with OpenAI's API (to power the agent), and optionally "
    "LangSmith (for tracing). Mood entries themselves are saved to a local CSV file "
    "on your own machine, never sent to any storage service.\n\n"
    "If you are not comfortable with a large language model processing your "
    "personal information, do not use this software.\n\n"
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
        opener_events: "Sequence[dict]" = (),
    ) -> None:
        self.root = root
        self.root.title("Job Load Agent")
        # Everything the graph needs carried into the next turn: the
        # transcript plus the station fields. Built by next_turn_state(), the
        # one function the eval harness uses too -- the two implemented this
        # separately once and drifted apart without anything failing.
        self.state: dict = {"messages": []}
        self.graph = agent_graph if agent_graph is not None else build_agent()
        self.recorder = recorder if recorder is not None else RunRecorder()
        self._opener_message = opener_message
        # Receipts for work done before this window existed -- the scheduler's
        # digest runs analyze_stress_patterns and then opens a window to
        # report it, so the receipt has to travel with the opener text.
        self._opener_events = list(opener_events)

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
        self.history = scrolledtext.ScrolledText(self.root, state="disabled", width=70, height=22, wrap="word")
        self.history.pack(fill="both", expand=True, padx=10, pady=10)
        # Tool receipts are deliberately styled as machine output, not speech:
        # they are the one thing in this window the agent cannot write.
        self.history.tag_configure("tool", font=("TkFixedFont", 8), foreground="gray45")
        self.history.tag_configure("tool_error", font=("TkFixedFont", 8), foreground="firebrick")

        input_frame = tk.Frame(self.root)
        input_frame.pack(fill="x", padx=10, pady=(0, 10))
        self.entry = tk.Entry(input_frame)
        self.entry.pack(side="left", fill="x", expand=True)
        self.entry.bind("<Return>", lambda _event: self._on_send())
        self.send_button = tk.Button(input_frame, text="Send", command=self._on_send)
        self.send_button.pack(side="left", padx=(5, 0))

        for event in self._opener_events:
            self._append_tool_event(event)
        if self._opener_message:
            self._append_message("Agent", self._opener_message)

    def _append_message(self, sender: str, text: str) -> None:
        self.history.configure(state="normal")
        self.history.insert("end", f"{sender}: {text}\n\n")
        self.history.configure(state="disabled")
        self.history.see("end")

    def _append_tool_event(self, event: dict) -> None:
        """Render one machine-set tool receipt.

        Printed the moment the node finishes, before the agent has said
        anything about it -- so the evidence arrives ahead of the claim. The
        agent cannot produce one of these and cannot suppress one.
        """
        exit_code = event.get("exit_code")
        status = "skipped" if exit_code is None else f"exit {exit_code}"
        detail = event.get("detail") or ""
        # An unverified event is one we did not witness in our own code --
        # currently only the server-side web_search. Saying so is the point.
        source = "" if event.get("verified", True) else "  [provider-reported]"
        line = f"  > {event.get('tool', '?')} -> {status}"
        if detail:
            line += f" · {detail}"
        tag = "tool_error" if exit_code == 1 else "tool"
        self.history.configure(state="normal")
        self.history.insert("end", f"{line}{source}\n", tag)
        self.history.configure(state="disabled")
        self.history.see("end")

    def _set_busy(self, busy: bool) -> None:
        """Lock input while a turn runs.

        Until this change the UI simply froze, which accidentally prevented a
        second turn being started mid-flight. Now that the graph runs off the
        main thread that has to be done deliberately -- two concurrent turns
        would interleave writes to self.state and corrupt it.
        """
        state = "disabled" if busy else "normal"
        self.entry.configure(state=state)
        self.send_button.configure(state=state)

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

        self._run_turn(user_text)

    def _run_turn(self, user_text: str) -> None:
        """Run one turn off the main thread, reporting tool receipts live.

        The graph used to be invoked directly here, which froze the Tk event
        loop for the whole turn -- so nothing could be drawn *during* it, and
        a tool receipt could only ever appear after the agent had already
        spoken. Streaming on a worker thread is what makes "tell me the
        moment the write landed" possible at all.

        Tk is not thread-safe, so the worker only ever puts events on a
        queue; every widget call happens back on the main thread in _drain.
        """
        messages_before = list(self.state.get("messages") or []) + [HumanMessage(content=user_text)]
        payload = {**self.state, "messages": messages_before}
        events: queue.Queue = queue.Queue()
        self._set_busy(True)
        threading.Thread(target=self._stream_worker, args=(payload, events), daemon=True).start()
        self.root.after(_POLL_MS, lambda: self._drain(events, messages_before, time.time()))

    def _stream_worker(self, payload: dict, events: "queue.Queue") -> None:
        """Worker thread: no Tk calls in here, only queue puts.

        `updates` carries each node's state change as that node finishes --
        which is exactly "the tool is done". `values` carries the full state
        after each step, so the last one is the final state the old invoke()
        used to return. One state field serves both the live line and the
        eval, which still calls invoke().
        """
        try:
            final = None
            for mode, chunk in self.graph.stream(payload, stream_mode=["updates", "values"]):
                if mode == "updates":
                    for node_update in (chunk or {}).values():
                        for event in (node_update or {}).get("tool_events") or []:
                            events.put(("event", event))
                elif mode == "values":
                    final = chunk
            events.put(("done", final))
        except Exception as exc:  # never let an unhandled exception reach the UI
            events.put(("error", exc))

    def _drain(self, events: "queue.Queue", messages_before: list, start: float) -> None:
        """Main thread: drain whatever the worker has produced, then either
        finish the turn or schedule another poll."""
        # The window can be closed while a turn is still running. Without this
        # the next scheduled poll fires against destroyed widgets and dumps a
        # TclError traceback on exit. The worker is a daemon thread, so it
        # goes with the process.
        try:
            if not self.root.winfo_exists():
                return
        except tk.TclError:
            return
        try:
            while True:
                kind, data = events.get_nowait()
                if kind == "event":
                    self._append_tool_event(data)
                elif kind == "error":
                    self._append_message(
                        "App", f"Something went wrong and I couldn't process that: {data}"
                    )
                    self._set_busy(False)
                    return
                elif kind == "done":
                    self._finish_turn(data, messages_before, time.time() - start)
                    return
        except queue.Empty:
            pass
        self.root.after(_POLL_MS, lambda: self._drain(events, messages_before, start))

    def _finish_turn(self, result: dict | None, messages_before: list, latency: float) -> None:
        if result is None:
            self._append_message("App", "The agent returned nothing for that turn.")
            self._set_busy(False)
            return

        # Stated explicitly rather than left to be inferred from silence: an
        # absent receipt is ambiguous (nothing logged, or nothing rendered?),
        # a printed one is not.
        if not any(e.get("tool") == LOG_TOOL_NAME for e in result.get("tool_events") or []):
            self._append_tool_event({"tool": LOG_TOOL_NAME, "exit_code": None,
                                     "detail": "not called this turn"})

        messages = result["messages"]
        self.recorder.record_turn(messages_before, messages, DEFAULT_MODEL, latency)
        # message_text, not .content: with the built-in web_search bound the
        # model speaks Responses-API content blocks.
        self._append_message("Agent", message_text(messages[-1]))

        # One shared decision, not a second copy of it here: on a completed
        # check-in this clears the transcript *and* the station fields, and
        # otherwise compacts spent tool payloads and carries the rest.
        self.state = next_turn_state(result)

        self._set_busy(False)


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
