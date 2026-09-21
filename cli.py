"""Terminal client for the station graph: the same agent app.py drives, with
the window's buttons as typed commands.

Why this exists alongside app.py: the buttons are not decoration. Three of
them write straight into the payload state and never ask the model anything,
and a conversation driven only by free text cannot reach those paths at all.
A client that could only type would be testing a different application.

It is also the honest place to watch a turn happen. After every turn it
prints which station answered, whether the check-in completed, and the
machine-set receipt for each tool call -- the receipts arrive before the
agent's reply, exactly as they do in the window, because the evidence should
never trail the claim.

Entry point -- loads .env itself (same reason as app.py).
"""
from __future__ import annotations

import argparse
import time
import typing
import uuid
from collections.abc import Callable
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

load_dotenv()

from langchain_core.messages import HumanMessage
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command

from agent.common import format_tool_event, message_text
from agent.graph import build_agent, next_turn_state
from agent.model import DEFAULT_MODEL
from agent.prompts import choice_sentence
from agent.safety import CRISIS_RESOURCES, screen_for_crisis
from observability.tracing import RunRecorder
from tools.checkin_log import Helpfulness, append_checkin_row, measure_awaiting_rating
from tools.measures import Energy, Goal

BASE_DIR = Path(__file__).resolve().parent
# Deliberately derived the same way app.py derives it rather than imported:
# importing app.py would pull Tk into a terminal client. A test asserts the
# two agree, so the duplication cannot drift unnoticed.
CONSENT_PATH = BASE_DIR / "state" / "consent_ack.json"

# From the schema's own Literals, so a vocabulary change updates the commands
# instead of drifting from them -- the same reason app.py derives its buttons.
ENERGY_OPTIONS = typing.get_args(Energy)
GOAL_OPTIONS = typing.get_args(Goal)
RATINGS = typing.get_args(Helpfulness)

HELP = """Commands (everything else is sent to the agent as a message):

  /energy <exhausted|calm|wired>     record energy, no model call
  /goal <calm|plan|change>           record goal, no model call
                                     either one alone is fine - the agent asks
                                     for whatever is still missing
  /idea <text>                       discuss an idea of your own -> debate
  /no                                "not helpful" -> counts a rejection
  /rate <helped|somewhat|not>        rate the logged measure, writes directly
  /state                             what the graph is carrying right now
  /help                              this
  /quit                              leave

Nothing is written without you: when the agent wants to add a measure or log a
check-in, it shows what it would write and waits for [a]pprove, [e]dit or
[r]eject.
"""


def _match(value: str, options: tuple[str, ...]) -> str | None:
    """Accept a prefix, so `/goal calm` means "calm down now"."""
    value = value.strip().lower()
    if not value:
        return None
    exact = [o for o in options if o == value]
    if exact:
        return exact[0]
    starts = [o for o in options if o.startswith(value)]
    return starts[0] if len(starts) == 1 else None


class Session:
    """One conversation. Everything injectable, so the tests drive the real
    command handling without OpenAI and without touching the real log."""

    def __init__(
        self,
        graph: Any = None,
        *,
        log_path: Path | None = None,
        recorder: RunRecorder | None = None,
        out: Callable[[str], None] = print,
        ask: Callable[[str], str] = input,
    ) -> None:
        # The checkpointer is what lets a turn pause on the approval card and
        # resume; it only ever has to hold one turn (a fresh thread per turn).
        self.graph = graph if graph is not None else build_agent(checkpointer=InMemorySaver())
        self.log_path = log_path
        self.recorder = recorder
        self.out = out
        self.ask = ask
        self.state: dict[str, Any] = {"messages": []}
        self.last_measure_id: str = measure_awaiting_rating(log_path)

    # -- what is currently possible -----------------------------------------

    def hints(self) -> str:
        """Mirror of the window's button visibility.

        A rating is only offered once a row exists to attach it to, and while
        suggestions are on screen the same place offers a rejection instead.
        """
        parts = []
        if not (self.state.get("energy") and self.state.get("goal")):
            parts.append("/energy /goal")
        if self.last_measure_id:
            parts.append("/rate")
        elif self.state.get("offered"):
            parts.append("/no")
        if self.state.get("offered"):
            parts.append("/idea")
        return "  available: " + " ".join(parts) if parts else ""

    # -- a turn --------------------------------------------------------------

    def run_turn(self, text: str, **extra: Any) -> None:
        """Crisis gate first, exactly as app.py does, then the graph."""
        crisis = screen_for_crisis(text)
        if crisis.triggered:
            self.out(CRISIS_RESOURCES)
            return

        messages_before = list(self.state.get("messages") or []) + [HumanMessage(content=text)]
        payload = {**self.state, **extra, "messages": messages_before}
        # A fresh thread per turn: the carried state goes in as today, and the
        # checkpointer only has to span a pause within this one turn.
        config = {"configurable": {"thread_id": uuid.uuid4().hex}}

        started = time.time()
        try:
            final, paused = self._stream(payload, config)
            while paused is not None:
                decisions = self._decide(paused)
                final, paused = self._stream(Command(resume={"decisions": decisions}), config)
        except Exception as exc:  # never let a traceback end the session
            self.out(f"Something went wrong and I couldn't process that: {exc}")
            return

        if final is None:
            self.out("The agent returned nothing for that turn.")
            return

        self._finish(final, messages_before, time.time() - started)

    def _stream(self, graph_input: Any, config: dict) -> tuple[dict | None, dict | None]:
        """Run the graph up to the end of the turn or up to the next pause.

        A pause arrives as an update under `__interrupt__` whose value is a
        tuple, not a node's dict. Read like the other updates, it raised, and
        the error handler above turned it into "something went wrong".
        """
        final: dict[str, Any] | None = None
        paused: dict | None = None
        for mode, chunk in self.graph.stream(graph_input, config, stream_mode=["updates", "values"]):
            if mode == "updates":
                for node, node_update in (chunk or {}).items():
                    if node == "__interrupt__":
                        paused = node_update[0].value
                        continue
                    for event in (node_update or {}).get("tool_events") or []:
                        self.out(format_tool_event(event))
            elif mode == "values":
                final = {k: v for k, v in chunk.items() if k != "__interrupt__"}
        return final, paused

    def _decide(self, paused: dict) -> list[dict]:
        """The approval card, one question per pending write.

        Anything but an explicit approve or a non-empty edit is a rejection:
        a write must never happen because an answer was unclear.
        """
        decisions: list[dict] = []
        for request in paused.get("action_requests") or []:
            call_id = request.get("id", "")
            editable = request.get("editable")
            self.out(f"\n  Approval needed: {request.get('description', '')}")
            choices = "[a]pprove / [e]dit / [r]eject" if editable else "[a]pprove / [r]eject"
            try:
                answer = self.ask(f"  {choices}: ").strip().lower()
                if answer.startswith("e") and editable:
                    current = (request.get("args") or {}).get(editable, "")
                    edited = self.ask(f"  new text (was: {current}): ").strip()
                else:
                    edited = ""
            except (EOFError, KeyboardInterrupt):
                answer, edited = "", ""

            if answer.startswith("a"):
                decisions.append({"id": call_id, "type": "approve"})
            elif answer.startswith("e") and editable and edited:
                args = {**(request.get("args") or {}), editable: edited}
                decisions.append({"id": call_id, "type": "edit", "args": args})
            else:
                self.out("  not approved - nothing is written")
                decisions.append({"id": call_id, "type": "reject"})
        return decisions

    def _finish(self, result: dict, messages_before: list, latency: float) -> None:
        if not any(e.get("tool") == "log_checkin" for e in result.get("tool_events") or []):
            self.out(format_tool_event(
                {"tool": "log_checkin", "exit_code": None, "detail": "not called this turn"}
            ))

        messages = result["messages"]
        if self.recorder is not None:
            self.recorder.record_turn(messages_before, messages, DEFAULT_MODEL, latency)
        self.out(f"\nAgent: {message_text(messages[-1])}")
        self.out(
            f"  [station: {result.get('step') or '-'}"
            f" | complete: {bool(result.get('complete'))}"
            f" | {latency:.1f}s]"
        )

        if result.get("last_measure_id"):
            self.last_measure_id = result["last_measure_id"]
        self.state = next_turn_state(result)

    # -- the commands --------------------------------------------------------

    def handle(self, line: str) -> bool:
        """Run one input line. Returns False when the session should end."""
        line = line.strip()
        if not line:
            return True
        if not line.startswith("/"):
            self.run_turn(line)
            return True

        command, _, argument = line[1:].partition(" ")
        command = command.lower()
        argument = argument.strip()

        if command in {"quit", "exit"}:
            return False
        if command == "help":
            self.out(HELP)
        elif command == "state":
            self.out(self._render_state())
        elif command in {"energy", "goal"}:
            self._choice(command, argument)
        elif command == "idea":
            self._idea(argument)
        elif command == "no":
            self._reject()
        elif command == "rate":
            self._rate(argument)
        else:
            self.out(f"Unknown command: /{command}. Try /help.")
        return True

    def _render_state(self) -> str:
        carried = {
            key: self.state.get(key)
            for key in ("energy", "goal", "offered", "rejections", "step", "idea")
        }
        return (
            f"  {carried}\n"
            f"  messages carried: {len(self.state.get('messages') or [])}"
            f" | measure awaiting a rating: {self.last_measure_id or '-'}"
        )

    def _choice(self, field: str, argument: str) -> None:
        options = ENERGY_OPTIONS if field == "energy" else GOAL_OPTIONS
        value = _match(argument, options)
        if value is None:
            self.out(f"  /{field} needs one of: {', '.join(options)}")
            return
        self.state[field] = value
        self.out(f"  {field}: {value}   (recorded)")
        # Every click runs a turn, including the first of the pair. Deciding
        # what a half-answered pair means is the ask station's job, not this
        # client's: with both values route_start reaches `suggest`, and with
        # one it reaches `ask`, which asks for the other. Holding the turn
        # back instead left the command looking ignored.
        # The flag makes the click count like record_choices: with both values
        # set, the turn is a fresh lookup in the user's list.
        self.run_turn(
            choice_sentence(self.state.get("energy"), self.state.get("goal")), choices_clicked=True
        )

    def _idea(self, argument: str) -> None:
        if not argument:
            self.out("  /idea needs the idea itself, e.g. /idea I could drop the Friday meeting")
            return
        self.run_turn(argument, idea=argument)

    def _reject(self) -> None:
        if not self.state.get("offered"):
            self.out("  nothing has been suggested yet, so there is nothing to reject")
            return
        self.run_turn("Not helpful.", rejected=True)

    def _rate(self, argument: str) -> None:
        """Writes its row directly. No graph, no model call -- the same
        guarantee the window's rating buttons carry."""
        value = _match(argument.replace("not helpful", "didnt_help").replace("not", "didnt_help"),
                       RATINGS)
        if value is None:
            self.out(f"  /rate needs one of: {', '.join(RATINGS)}")
            return
        if not self.last_measure_id:
            self.out("  nothing to rate yet: a check-in has to be logged first")
            return

        result = append_checkin_row(
            measure_id=self.last_measure_id, helpfulness=value, path=self.log_path
        )
        if result["status"] == "ok":
            self.out(format_tool_event(
                {"tool": "log_checkin", "exit_code": 0,
                 "detail": f"rating row {result['row_id'][:8]}"}
            ))
            self.last_measure_id = ""
        else:
            self.out(format_tool_event(
                {"tool": "log_checkin", "exit_code": 1,
                 "detail": f"write failed: {result.get('error', 'unknown error')}"}
            ))


def main() -> None:
    parser = argparse.ArgumentParser(description="Job Load Agent terminal client")
    parser.add_argument("--log-path", type=Path, default=None,
                        help="Write check-ins somewhere other than the real log.")
    parser.add_argument("--no-record", action="store_true",
                        help="Skip the per-turn cost/latency record.")
    args = parser.parse_args()

    session = Session(
        log_path=args.log_path,
        recorder=None if args.no_record else RunRecorder(),
    )
    print(f"Job Load Agent ({DEFAULT_MODEL}). /help for commands, /quit to leave.\n")

    while True:
        print(session.hints())
        try:
            line = input("You: ")
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not session.handle(line):
            break
        print()


if __name__ == "__main__":
    main()
