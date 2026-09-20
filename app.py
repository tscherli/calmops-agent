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
import typing
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
from tools.checkin_log import Helpfulness, append_checkin_row, measure_awaiting_rating
from tools.measures import Energy, Goal

# How often the main thread checks the worker's queue while a turn runs.
# Small enough that a receipt looks instant, large enough to cost nothing.
_POLL_MS = 50

LOG_TOOL_NAME = "log_checkin"

# Derived from the schema's own Literals rather than duplicated here, so a
# vocabulary change updates the buttons instead of drifting from them.
_ENERGY_OPTIONS = list(typing.get_args(Energy))
_GOAL_OPTIONS = list(typing.get_args(Goal))
_RATING_OPTIONS = list(typing.get_args(Helpfulness))

_RATING_LABELS = {"helped": "helped", "somewhat": "somewhat", "didnt_help": "not helpful"}


BASE_DIR = Path(__file__).resolve().parent
CONSENT_PATH = BASE_DIR / "state" / "consent_ack.json"

CONSENT_TEXT = (
    "Data & Privacy Disclaimer\n\n"
    "This tool shares what you type -- including personal reflections on how you "
    "are doing and what is going on at work -- with OpenAI's API (to power the "
    "agent), and optionally LangSmith (for tracing). It may also run a web search "
    "through OpenAI when it looks for ideas beyond your own list.\n\n"
    "Your check-in log and your list of measures are plain CSV files on your own "
    "machine. They are never sent to any storage service, but their contents are "
    "included in what goes to the API while you are having a conversation.\n\n"
    "The buttons in this window -- your energy and goal, and how well something "
    "worked -- are recorded without asking the model anything. Everything you type "
    "is sent.\n\n"
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
        log_path: Path | None = None,
    ) -> None:
        self.root = root
        self.root.title("Job Load Agent")
        # Everything the graph needs carried into the next turn: the
        # transcript plus the station fields. Built by next_turn_state(), the
        # one function the eval harness uses too -- the two implemented this
        # separately once and drifted apart without anything failing.
        self.state: dict = {"messages": []}
        # The rating buttons write straight to the log, so they need the same
        # path the graph's logging tool uses -- injectable for the same reason
        # everything else here is.
        self.log_path = log_path
        # Which measure a rating would be about. Restored from the log so the
        # buttons still work after the window has been closed and reopened.
        self.last_measure_id: str = measure_awaiting_rating(log_path)
        self._idea_pending = False
        self._rating_mode: str | None = None
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

        self.controls = tk.Frame(self.root)
        self.controls.pack(fill="x", padx=10, pady=(0, 2))
        self._build_control_buttons()

        input_frame = tk.Frame(self.root)
        input_frame.pack(fill="x", padx=10, pady=(0, 10))
        self.entry = tk.Entry(input_frame)
        self.entry.pack(side="left", fill="x", expand=True)
        self.entry.bind("<Return>", lambda _event: self._on_send())
        self.send_button = tk.Button(input_frame, text="Send", command=self._on_send)
        self.send_button.pack(side="left", padx=(5, 0))

        self._refresh_controls()

        for event in self._opener_events:
            self._append_tool_event(event)
        if self._opener_message:
            self._append_message("Agent", self._opener_message)

    # -- The buttons: a click is a fact, and costs no model call -------------

    def _build_control_buttons(self) -> None:
        """Three groups, none of which asks the model anything.

        A click is unambiguous self-report, so it is better data than a
        sentence the model has to interpret -- and the energy/goal pair skips
        the `ask` station entirely, which is one model call instead of two.
        """
        self._buttons: list[tk.Button] = []

        self.choice_frame = tk.Frame(self.controls)
        for field, options in (("energy", _ENERGY_OPTIONS), ("goal", _GOAL_OPTIONS)):
            row = tk.Frame(self.choice_frame)
            row.pack(fill="x")
            tk.Label(row, text=f"{field}:", width=7, anchor="w").pack(side="left")
            for value in options:
                button = tk.Button(
                    row, text=value, command=lambda f=field, v=value: self._on_choice_click(f, v)
                )
                button.pack(side="left", padx=2, pady=1)
                self._buttons.append(button)

        self.rating_frame = tk.Frame(self.controls)
        self._rating_buttons: dict[str, tk.Button] = {}
        for value in _RATING_OPTIONS:
            button = tk.Button(
                self.rating_frame,
                text=_RATING_LABELS[value],
                command=lambda v=value: self._on_rating_click(v),
            )
            self._rating_buttons[value] = button
            self._buttons.append(button)

        self.idea_frame = tk.Frame(self.controls)
        self.idea_button = tk.Button(
            self.idea_frame, text="discuss my idea", command=self._on_idea_click
        )
        self.idea_button.pack(side="left", padx=2, pady=1)
        self._buttons.append(self.idea_button)

    def _refresh_controls(self) -> None:
        """Show only the buttons that mean something right now.

        The rating pair is the one that matters: a rating is a second row
        about a measure that was actually logged, so offering it before any
        row exists would promise something the schema cannot hold. While
        suggestions are on screen the same strip shows only "not helpful",
        which is a rejection rather than a rating.
        """
        self._show(self.choice_frame, not (self.state.get("energy") and self.state.get("goal")))

        if self.last_measure_id:
            self._rating_mode = "rate"
        elif self.state.get("offered"):
            self._rating_mode = "reject"
        else:
            self._rating_mode = None
        self._show(self.rating_frame, self._rating_mode is not None)
        not_helpful = self._rating_buttons["didnt_help"]
        not_helpful.pack(side="left", padx=2, pady=1)
        for value in ("helped", "somewhat"):
            if self._rating_mode == "rate":
                self._rating_buttons[value].pack(side="left", padx=2, pady=1, before=not_helpful)
            else:
                self._rating_buttons[value].pack_forget()

        self._show(self.idea_frame, bool(self.state.get("offered")) and not self._idea_pending)

    @staticmethod
    def _show(frame: tk.Frame, visible: bool) -> None:
        if visible:
            frame.pack(fill="x", pady=1)
        else:
            frame.pack_forget()

    def _on_choice_click(self, field: str, value: str) -> None:
        """One value, straight into the payload state. Once both are in, the
        turn runs and route_start goes to `suggest` -- `ask` never happens."""
        self.state[field] = value
        self._append_message("You", f"{field}: {value}")
        self._refresh_controls()
        if self.state.get("energy") and self.state.get("goal"):
            self._run_turn(
                f"I'm feeling {self.state['energy']} and I want to {self.state['goal']}."
            )

    def _on_rating_click(self, value: str) -> None:
        """Either a rejection or a rating -- never a model call, and in the
        rating case never even a graph run."""
        if self._rating_mode == "reject":
            self._append_message("You", _RATING_LABELS[value])
            self._run_turn(_RATING_LABELS[value], rejected=True)
            return

        self._append_message("You", _RATING_LABELS[value])
        try:
            result = append_checkin_row(
                measure_id=self.last_measure_id, helpfulness=value, path=self.log_path
            )
        except Exception as exc:  # never let an unhandled exception reach the UI
            self._append_message("App", f"Something went wrong saving that: {exc}")
            return

        if result["status"] == "ok":
            self._append_tool_event(
                {"tool": LOG_TOOL_NAME, "exit_code": 0,
                 "detail": f"rating row {result['row_id'][:8]}"}
            )
            # One rating per check-in: the row is written, so there is nothing
            # left for these buttons to be about.
            self.last_measure_id = ""
        else:
            self._append_tool_event(
                {"tool": LOG_TOOL_NAME, "exit_code": 1,
                 "detail": f"write failed: {result.get('error', 'unknown error')}"}
            )
        self._refresh_controls()

    def _on_idea_click(self) -> None:
        """Arms the next Send. The idea travels in the payload, so `debate`
        gets it as a fact rather than having to spot it in the prose -- and
        the crisis gate still screens the text like any other message."""
        self._idea_pending = True
        self._append_message("App", "Go ahead -- type your idea and press Send.")
        self._refresh_controls()

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
        for button in self._buttons:
            button.configure(state=state)

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

        extra: dict = {}
        if self._idea_pending:
            extra["idea"] = user_text
            self._idea_pending = False
        self._run_turn(user_text, **extra)

    def _run_turn(self, user_text: str, **extra) -> None:
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
        payload = {**self.state, **extra, "messages": messages_before}
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

        # Read before the reset: the graph clears its state on a completed
        # check-in, but the measure that row named is exactly what the rating
        # buttons will be about afterwards.
        if result.get("last_measure_id"):
            self.last_measure_id = result["last_measure_id"]

        # One shared decision, not a second copy of it here: on a completed
        # check-in this clears the transcript *and* the station fields, and
        # otherwise compacts spent tool payloads and carries the rest.
        self.state = next_turn_state(result)

        self._refresh_controls()
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
