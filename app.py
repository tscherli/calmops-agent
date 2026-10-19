"""Tkinter chat window: the actual CalmOps UI.

Blocking first-run consent screen, then free-text chat that goes through
agent/safety.py's crisis gate first and the station graph second.

Entry point -- must load .env itself (flagged during stage 6: library
modules like agent/graph.py must not do this as an import side effect).
"""
import argparse
import json
import queue
import re
import threading
import time
import tkinter as tk
import typing
import uuid
import webbrowser
from collections.abc import Sequence
from pathlib import Path
from tkinter import font as tkfont
from tkinter import scrolledtext

from dotenv import load_dotenv

load_dotenv()

from langchain_core.messages import HumanMessage
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command

from agent.common import format_tool_event, message_text
from agent.data_paths import DATA_DIR_ENV, DEFAULT_CONSENT_PATH, DataPaths, data_paths, prepare
from agent.graph import build_agent, next_turn_state
from agent.model import DEFAULT_MODEL
from agent.prompts import choice_sentence
from agent.safety import CRISIS_RESOURCES, screen_for_crisis
from observability.tracing import RunRecorder
from tools.checkin_log import Helpfulness, append_checkin_row, measures_awaiting_rating
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

# A few calm colours, defined once: a soft sage window, a white chat panel,
# teal for the agent and the Send button, blue for links.
_WINDOW_BG = "#eef3f1"
_PANEL_BG = "#ffffff"
_BUTTON_BG = "#dcebe5"
_ACCENT = "#2f7d6d"
_LINK = "#1a5fb4"
_CARD_BG = "#fff6e0"
_SENDER_COLOURS = {"You": "#3b6ea5", "Agent": _ACCENT, "App": "#a35a2a"}

# A Markdown link shows its title, a bare address shows itself. Only web
# addresses: a click hands the URL to the browser. Punctuation right after a
# bare address ends the sentence, not the URL.
_LINK_PATTERN = re.compile(
    r"\[(?P<title>[^\]\n]+)\]\((?P<md_url>https?://[^\s)]+)\)|(?P<url>https?://[^\s<>()\[\]]+)"
)


def split_links(text: str) -> list[tuple[str, str | None]]:
    """The text as (shown, url) pieces, in order; url is None for plain text."""
    pieces: list[tuple[str, str | None]] = []
    pos = 0
    for match in _LINK_PATTERN.finditer(text):
        if match.group("url"):
            url = match.group("url").rstrip(".,;:!?'\"")
            shown, end = url, match.start() + len(url)
        else:
            url, shown, end = match.group("md_url"), match.group("title"), match.end()
        if match.start() > pos:
            pieces.append((text[pos:match.start()], None))
        pieces.append((shown, url))
        pos = end
    if pos < len(text):
        pieces.append((text[pos:], None))
    return pieces


CONSENT_PATH = DEFAULT_CONSENT_PATH

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
    "Nothing is written to your log or your list of measures without you: when "
    "the agent wants to, it shows exactly what it would write and waits for you "
    "to approve, edit or reject it.\n\n"
    "If you are not comfortable with a large language model processing your "
    "personal information, do not use this software.\n\n"
    "This is a self-reflection aid, not a substitute for professional mental health "
    "support; its replies are AI-generated and may be inaccurate. In a crisis, "
    "contact local emergency services or a crisis hotline directly, not this tool.\n\n"
    "Each user of this software supplies their own API key -- no data from anyone "
    "running this passes through the original developer's infrastructure."
)


def has_consented(path: Path | None = None) -> bool:
    return (path or CONSENT_PATH).exists()


def record_consent(path: Path | None = None) -> None:
    path = path or CONSENT_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"acknowledged": True}), encoding="utf-8")


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
        measures_path: Path | None = None,
        consent_path: Path | None = None,
    ) -> None:
        self.root = root
        self.root.title("CalmOps")
        # Through Tk's option database, so every frame, label and button built
        # after this gets the colours without repeating them.
        self.root.configure(bg=_WINDOW_BG)
        for widget_class in ("Frame", "Label", "Labelframe"):
            self.root.option_add(f"*{widget_class}.background", _WINDOW_BG)
        self.root.option_add("*Labelframe.foreground", _ACCENT)
        self.root.option_add("*Button.background", _BUTTON_BG)
        self.root.option_add("*Button.relief", "flat")
        self.root.option_add("*Button.padX", 8)
        # Everything the graph needs carried into the next turn: the
        # transcript plus the station fields. Built by next_turn_state(), the
        # one function the eval harness uses too -- the two implemented this
        # separately once and drifted apart without anything failing.
        self.state: dict = {"messages": []}
        # The rating buttons write straight to the log, so they need the same
        # path the graph's logging tool uses -- injectable for the same reason
        # everything else here is.
        self.log_path = log_path
        # Which measures a rating would be about, oldest first. Restored from
        # the log so the buttons still work after the window has been closed
        # and reopened. A turn can log more than one measure, and each of them
        # gets its own turn on the strip.
        self.pending_ratings: list[str] = measures_awaiting_rating(log_path)
        self.last_measure_id: str = self.pending_ratings[0] if self.pending_ratings else ""
        # The approval card's working state while a turn is paused: the
        # requests still to answer and the decisions given so far.
        self._pending_requests: list[dict] = []
        self._decisions: list[dict] = []
        # The checkpointer is what lets a turn pause on the approval card and
        # resume; it only ever has to hold one turn (a fresh thread per turn).
        # The same log and list the window reads, or a rating button and the
        # agent's own log_checkin would write to two different files.
        self.graph = (
            agent_graph
            if agent_graph is not None
            else build_agent(checkpointer=InMemorySaver(), log_path=log_path, measures_path=measures_path)
        )
        self.consent_path = consent_path
        self.recorder = recorder if recorder is not None else RunRecorder()
        self._opener_message = opener_message
        # Receipts for work done before this window existed -- the scheduler's
        # digest runs analyze_stress_patterns and then opens a window to
        # report it, so the receipt has to travel with the opener text.
        self._opener_events = list(opener_events)

        if skip_consent or has_consented(self.consent_path):
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
        record_consent(self.consent_path)
        self.consent_frame.destroy()
        self._build_chat_ui()

    # -- Main chat screen --------------------------------------------------

    def _build_chat_ui(self) -> None:
        self.history = scrolledtext.ScrolledText(
            self.root, state="disabled", width=70, height=22, wrap="word",
            bg=_PANEL_BG, relief="flat", padx=10, pady=8,
        )
        self.history.pack(fill="both", expand=True, padx=10, pady=10)
        # Tool receipts are deliberately styled as machine output, not speech:
        # they are the one thing in this window the agent cannot write.
        self.history.tag_configure("tool", font=("TkFixedFont", 8), foreground="gray45")
        self.history.tag_configure("tool_error", font=("TkFixedFont", 8), foreground="firebrick")
        base = tkfont.nametofont("TkDefaultFont").actual()
        self._bold = (base["family"], base["size"], "bold")
        for sender, colour in _SENDER_COLOURS.items():
            self.history.tag_configure(f"sender_{sender}", foreground=colour, font=self._bold)
        self.history.tag_configure("link", foreground=_LINK, underline=True)
        self.history.tag_bind("link", "<Enter>", lambda _event: self.history.configure(cursor="hand2"))
        self.history.tag_bind("link", "<Leave>", lambda _event: self.history.configure(cursor="xterm"))
        # Which address each link's own tag opens, so a click needs no lookup
        # in the text itself.
        self._link_urls: dict[str, str] = {}

        self.approval_frame = tk.Frame(
            self.root, relief="groove", borderwidth=2, padx=6, pady=6, bg=_CARD_BG
        )
        self._build_approval_card()

        self.controls = tk.Frame(self.root)
        self.controls.pack(fill="x", padx=10, pady=(0, 2))
        self._build_control_buttons()

        input_frame = tk.Frame(self.root)
        input_frame.pack(fill="x", padx=10, pady=(0, 10))
        self.entry = tk.Entry(input_frame)
        self.entry.pack(side="left", fill="x", expand=True)
        self.entry.bind("<Return>", lambda _event: self._on_send())
        self.send_button = tk.Button(
            input_frame, text="Send", command=self._on_send,
            bg=_ACCENT, fg="white", activebackground=_ACCENT, activeforeground="white",
        )
        self.send_button.pack(side="left", padx=(5, 0))

        self._refresh_controls()

        for event in self._opener_events:
            self._append_tool_event(event)
        if self._opener_message:
            self._append_message("Agent", self._opener_message)

    # -- The approval card: nothing is written without a click --------------

    def _build_approval_card(self) -> None:
        """Three fixed buttons; only the text shown changes.

        The model decides nothing about this card: the description is built by
        the graph from the call's own arguments, and the buttons are always the
        same three. The entry holds the one value Edit may change.
        """
        self.approval_label = tk.Label(
            self.approval_frame, text="", anchor="w", justify="left", wraplength=520, bg=_CARD_BG
        )
        self.approval_label.pack(fill="x")
        self.approval_entry = tk.Entry(self.approval_frame)
        buttons = tk.Frame(self.approval_frame, bg=_CARD_BG)
        buttons.pack(fill="x", pady=(4, 0))
        self.approve_button = tk.Button(buttons, text="Approve", command=lambda: self._on_decision("approve"))
        self.edit_button = tk.Button(buttons, text="Save my edit", command=lambda: self._on_decision("edit"))
        self.reject_button = tk.Button(buttons, text="Reject", command=lambda: self._on_decision("reject"))
        for button in (self.approve_button, self.edit_button, self.reject_button):
            button.pack(side="left", padx=2)

    def _show_approval(self, paused: dict, messages_before: list, start: float, config: dict) -> None:
        """A turn stopped for the user. Input stays locked; only the card works."""
        self._pending_requests = list(paused.get("action_requests") or [])
        self._decisions = []
        self._resume_args = (messages_before, start, config)
        if not self._pending_requests:
            self._resume_turn()
            return
        self._show_request(self._pending_requests[0])

    def _show_request(self, request: dict) -> None:
        self.approval_label.configure(text=f"Approval needed: {request.get('description', '')}")
        editable = request.get("editable")
        self.approval_entry.delete(0, "end")
        if editable:
            self.approval_entry.insert(0, str((request.get("args") or {}).get(editable, "")))
            self.approval_entry.pack(fill="x", pady=(4, 0), before=self.approve_button.master)
            self.edit_button.configure(state="normal")
        else:
            self.approval_entry.pack_forget()
            self.edit_button.configure(state="disabled")
        self.approval_frame.pack(fill="x", padx=10, pady=(0, 4), before=self.controls)

    def _on_decision(self, kind: str) -> None:
        """Record one decision; ask the next question or resume the turn.

        An edit with nothing in it is a rejection: a write must never happen
        because an answer was unclear.
        """
        if not self._pending_requests:
            return
        request = self._pending_requests.pop(0)
        call_id = request.get("id", "")
        editable = request.get("editable")
        edited = self.approval_entry.get().strip()
        if kind == "approve":
            self._decisions.append({"id": call_id, "type": "approve"})
            self._append_message("You", f"approved - {request.get('description', '')}")
        elif kind == "edit" and editable and edited:
            args = {**(request.get("args") or {}), editable: edited}
            self._decisions.append({"id": call_id, "type": "edit", "args": args})
            self._append_message("You", f'edited - "{edited}"')
        else:
            self._decisions.append({"id": call_id, "type": "reject"})
            self._append_message("You", f"rejected - {request.get('description', '')}")

        if self._pending_requests:
            self._show_request(self._pending_requests[0])
        else:
            self._resume_turn()

    def _resume_turn(self) -> None:
        """Continue the paused turn on its own thread with the decisions."""
        self.approval_frame.pack_forget()
        messages_before, start, config = self._resume_args
        decisions, self._decisions = self._decisions, []
        self._start_worker(Command(resume={"decisions": decisions}), config, messages_before, start)

    # -- The buttons: a click is a fact, and costs no model call -------------

    def _build_control_buttons(self) -> None:
        """Three groups, none of which asks the model anything.

        A click is unambiguous self-report, so it is better data than a
        sentence the model has to interpret -- and the energy/goal pair skips
        the `ask` station entirely, which is one model call instead of two.
        """
        self._buttons: list[tk.Button] = []

        self.choice_frame = tk.LabelFrame(
            self.controls, text="Quick check-in", font=self._bold, padx=6, pady=4
        )
        tk.Label(
            self.choice_frame, anchor="w", fg="#5f6f73",
            text="One click each, no typing: how is your energy, and what do you need right now?",
        ).pack(fill="x")
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
        self.rating_label = tk.Label(self.rating_frame, text="")
        self.rating_label.pack(side="left", padx=(0, 6))
        self._rating_buttons: dict[str, tk.Button] = {}
        for value in _RATING_OPTIONS:
            button = tk.Button(
                self.rating_frame,
                text=_RATING_LABELS[value],
                command=lambda v=value: self._on_rating_click(v),
            )
            self._rating_buttons[value] = button
            self._buttons.append(button)

    def _refresh_controls(self) -> None:
        """Show only the buttons that mean something right now.

        A rating is a second row about a measure that was actually logged, so
        the strip appears only once such a row exists: offering it earlier
        would promise something the schema cannot hold. Rejecting a suggestion
        and bringing up an idea of your own used to have buttons of their own
        here; both are said in words now, and the agent asks for them at the
        point where they make sense.
        """
        self._show(self.choice_frame, not (self.state.get("energy") and self.state.get("goal")))

        self._show(self.rating_frame, bool(self.last_measure_id))
        # Which one is being rated, because two can be waiting at once.
        self.rating_label.configure(text=f'Was "{self.last_measure_id}" helpful?')
        for value in _RATING_OPTIONS:
            self._rating_buttons[value].pack(side="left", padx=2, pady=1)

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
        # Every click runs a turn, including the first of the pair. What a
        # half-answered pair means is the ask station's job, not this
        # window's: with both, route_start reaches `suggest`; with one, `ask`
        # asks for the other. Waiting for both left a click doing nothing
        # visible at all.
        # The flag makes the click count like record_choices: with both values
        # set, the turn is a fresh lookup in the user's list.
        self._run_turn(
            choice_sentence(self.state.get("energy"), self.state.get("goal")), choices_clicked=True
        )

    def _set_pending_ratings(self, measure_ids: list[str]) -> None:
        """The queue, and the one the buttons are about right now."""
        self.pending_ratings = list(measure_ids)
        self.last_measure_id = self.pending_ratings[0] if self.pending_ratings else ""

    def _on_rating_click(self, value: str) -> None:
        """How a logged measure went: no model call, not even a graph run.

        The click is the approval, which is why the row is written here rather
        than proposed on a card. A rating typed in words takes the longer way
        round and is approved there."""
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
            # One rating per measure: this one is written, so the strip moves
            # on to the next measure this check-in logged, if there was one.
            self._set_pending_ratings(measures_awaiting_rating(self.log_path))
        else:
            self._append_tool_event(
                {"tool": LOG_TOOL_NAME, "exit_code": 1,
                 "detail": f"write failed: {result.get('error', 'unknown error')}"}
            )
        self._refresh_controls()

    def _append_message(self, sender: str, text: str) -> None:
        self.history.configure(state="normal")
        self.history.insert("end", f"{sender}: ", f"sender_{sender}")
        for shown, url in split_links(text):
            if url is None:
                self.history.insert("end", shown)
            else:
                self.history.insert("end", shown, ("link", self._link_tag(url)))
        self.history.insert("end", "\n\n")
        self.history.configure(state="disabled")
        self.history.see("end")

    def _link_tag(self, url: str) -> str:
        """A tag of its own per link, bound to opening that one address."""
        tag = f"link-{len(self._link_urls)}"
        self._link_urls[tag] = url
        self.history.tag_bind(tag, "<Button-1>", lambda _event: self._open_link(url))
        return tag

    def _open_link(self, url: str) -> None:
        webbrowser.open(url, new=2)

    def _append_tool_event(self, event: dict) -> None:
        """Render one machine-set tool receipt.

        Printed the moment the node finishes, before the agent has said
        anything about it -- so the evidence arrives ahead of the claim. The
        agent cannot produce one of these and cannot suppress one.
        """
        tag = "tool_error" if event.get("exit_code") == 1 else "tool"
        self.history.configure(state="normal")
        self.history.insert("end", f"{format_tool_event(event)}\n", tag)
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

        self._run_turn(user_text)

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
        # A fresh thread per turn: the carried state goes in as before, and the
        # checkpointer only has to span a pause within this one turn.
        config = {"configurable": {"thread_id": uuid.uuid4().hex}}
        self._set_busy(True)
        self._start_worker(payload, config, messages_before, time.time())

    def _start_worker(self, graph_input, config: dict, messages_before: list, start: float) -> None:
        """One stretch of a turn on the worker thread: to its end or to a pause.

        Used for the first stretch and again for the resume after the card,
        so both go through the same queue, the same polling and the same error
        handling.
        """
        events: queue.Queue = queue.Queue()
        threading.Thread(
            target=self._stream_worker, args=(graph_input, config, events), daemon=True
        ).start()
        self.root.after(_POLL_MS, lambda: self._drain(events, messages_before, start, config))

    def _stream_worker(self, graph_input, config: dict, events: "queue.Queue") -> None:
        """Worker thread: no Tk calls in here, only queue puts.

        `updates` carries each node's state change as that node finishes --
        which is exactly "the tool is done". `values` carries the full state
        after each step, so the last one is the final state the old invoke()
        used to return. One state field serves both the live line and the
        eval, which still calls invoke().

        A pause for the approval card arrives as an update under
        `__interrupt__` whose value is a tuple, not a node's dict. Read like the
        others it raised, and the handler below turned it into "something went
        wrong" -- a working app that looked broken.
        """
        try:
            final = None
            paused = None
            for mode, chunk in self.graph.stream(graph_input, config, stream_mode=["updates", "values"]):
                if mode == "updates":
                    for node, node_update in (chunk or {}).items():
                        if node == "__interrupt__":
                            paused = node_update[0].value
                            continue
                        for event in (node_update or {}).get("tool_events") or []:
                            events.put(("event", event))
                elif mode == "values":
                    final = {k: v for k, v in chunk.items() if k != "__interrupt__"}
            events.put(("interrupt", paused) if paused is not None else ("done", final))
        except Exception as exc:  # never let an unhandled exception reach the UI
            events.put(("error", exc))

    def _drain(self, events: "queue.Queue", messages_before: list, start: float, config: dict) -> None:
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
                elif kind == "interrupt":
                    # Not the end of the turn: no next_turn_state here, since
                    # the paused state is incomplete. The card resumes it.
                    self._show_approval(data, messages_before, start, config)
                    return
                elif kind == "done":
                    # _finish_turn runs here, on the main thread, outside the
                    # worker's try/except. An exception in it used to leave
                    # the input disabled forever with the traceback on stderr
                    # -- from the user's side, a window that answered once and
                    # then froze. A failure has to be visible and recoverable.
                    try:
                        self._finish_turn(data, messages_before, time.time() - start)
                    except Exception as exc:  # never let the UI lock up silently
                        self._append_message(
                            "App", f"Something went wrong finishing that turn: {exc}"
                        )
                        self._set_busy(False)
                    return
        except queue.Empty:
            pass
        self.root.after(_POLL_MS, lambda: self._drain(events, messages_before, start, config))

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
        self.recorder.record_turn(
            messages_before, messages, DEFAULT_MODEL, latency, side_calls=result.get("side_calls") or []
        )
        # message_text, not .content: with the built-in web_search bound the
        # model speaks Responses-API content blocks.
        self._append_message("Agent", message_text(messages[-1]))

        # Read before the reset: the graph clears its state on a completed
        # check-in, but the measure that row named is exactly what the rating
        # buttons will be about afterwards.
        # The log is what the strip follows, because a turn can log two
        # measures and the state field only carries the last one. The state is
        # the fallback for a graph that wrote somewhere this app cannot read.
        pending = measures_awaiting_rating(self.log_path)
        if not pending and result.get("last_measure_id"):
            pending = [result["last_measure_id"]]
        self._set_pending_ratings(pending)

        # One shared decision, not a second copy of it here: on a completed
        # check-in this clears the transcript *and* the station fields, and
        # otherwise compacts spent tool payloads and carries the rest.
        self.state = next_turn_state(result)

        self._refresh_controls()
        self._set_busy(False)


def window_paths(paths: DataPaths) -> dict[str, typing.Any]:
    """ChatApp's arguments for one data folder -- shared with the scheduler,
    which opens the same window."""
    return {
        "log_path": paths.checkin_log,
        "measures_path": paths.measures,
        "consent_path": paths.consent,
        "recorder": RunRecorder(log_path=paths.runs_log),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="CalmOps chat window")
    parser.add_argument(
        "--skip-consent",
        action="store_true",
        help="Dev-only: skip the consent screen. Never use this as the default.",
    )
    parser.add_argument("--data-dir", type=Path, default=None,
                        help=f"Keep the log, the measure list and the app's state in this folder "
                             f"(default: ${DATA_DIR_ENV}, else next to the code).")
    args = parser.parse_args()

    root = tk.Tk()
    ChatApp(root, skip_consent=args.skip_consent, **window_paths(prepare(data_paths(args.data_dir))))
    root.mainloop()


if __name__ == "__main__":
    main()
