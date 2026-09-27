"""Ad-hoc, non-pytest scratch script: watch the agent work while you use it.

    python devtools/try_inspect.py            # the Tk window, with the inspector beside it
    python devtools/try_inspect.py cli        # the terminal client instead

Two windows open. One is the normal client; the other lights up the part of
the graph that is running right now and lists the calls underneath, with the
file and line each came from. Meant for reading the code while it runs: every
box is a function in agent/graph.py, and clicking a line tells you where to
look.

How it works, because that is half the point: the client is started as a
separate process under try_trace.py, which writes one JSON line per call into
a temporary file. This window tails that file. The two are kept apart because
the client already owns a Tk main loop and runs each turn in a worker thread,
and drawing into a window from another thread is not safe.
"""
from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import tkinter as tk
from pathlib import Path

# This file lives in devtools/; the clients it starts live in the project root.
HERE = Path(__file__).resolve().parent
PROJECT = HERE.parent
POLL_MS = 120
RECENT_CALLS = 14

# The stations and the code-only nodes, in the order a turn walks them. The
# names are the functions in agent/graph.py, so a lit box is a place to put a
# breakpoint.
NODES: list[tuple[str, str]] = [
    ("route_start", "picks the station, no model call"),
    ("ask", "energy and goal"),
    ("suggest", "two measures from your list"),
    ("follow_up", "feedback, adding, logging"),
    ("debate", "your own idea, stronger model"),
    ("web_search_ideas", "after a rejection"),
    ("count_rejection", "counts a no"),
    ("ask_goal_again", "fixed question"),
    ("tools_node", "runs the tool calls"),
    ("review", "the approval card"),
    ("close_idea", "what was added, in code"),
    ("checkin_done", "conversation ends"),
]
# Everything else worth seeing lights up the tool row instead of a node.
TOOLS = {
    "record_choices", "find_measures", "search_measures", "discuss_idea",
    "idea_decided", "suggestion_not_helpful", "add_measure", "log_checkin",
    "append_checkin_row", "append_measure",
}

IDLE = "#e8e8e8"
LIVE = "#ffd54f"
RECENT = "#fff6d5"
TEXT = "#222222"


class Inspector:
    """A window that tails the event file and colours what is running."""

    def __init__(self, root: tk.Tk, events_path: Path, client: subprocess.Popen) -> None:
        self.root = root
        self.events_path = events_path
        self.client = client
        self.handle = None
        self.boxes: dict[str, tk.Label] = {}
        self.fading: dict[str, int] = {}
        self.calls: list[str] = []

        root.title("Job Load Agent -- what is running")
        tk.Label(
            root,
            text="Each box is a function in agent/graph.py. It lights up while it runs.",
            anchor="w",
            padx=8,
            pady=6,
        ).pack(fill="x")

        grid = tk.Frame(root, padx=8)
        grid.pack(fill="x")
        for index, (name, hint) in enumerate(NODES):
            box = tk.Label(
                grid,
                text=f"{name}\n{hint}",
                bg=IDLE,
                fg=TEXT,
                width=26,
                height=3,
                relief="ridge",
                justify="left",
                anchor="w",
                padx=6,
            )
            box.grid(row=index // 3, column=index % 3, padx=3, pady=3, sticky="we")
            self.boxes[name] = box

        self.tool_label = tk.Label(
            root, text="tool: -", bg=IDLE, anchor="w", padx=8, pady=4, relief="ridge"
        )
        self.tool_label.pack(fill="x", padx=8, pady=(6, 2))

        tk.Label(root, text="calls, newest first (file:line)", anchor="w", padx=8).pack(fill="x")
        self.call_list = tk.Text(root, height=RECENT_CALLS, width=74, state="disabled", padx=8)
        self.call_list.pack(fill="both", expand=True, padx=8, pady=(0, 8))

        root.protocol("WM_DELETE_WINDOW", self.close)
        root.after(POLL_MS, self.poll)

    # -- reading the trace ---------------------------------------------------

    def poll(self) -> None:
        """Read whatever the tracer has written since the last tick."""
        if self.handle is None and self.events_path.exists():
            self.handle = self.events_path.open(encoding="utf-8")
        if self.handle is not None:
            for line in self.handle:
                line = line.strip()
                if line:
                    try:
                        self.show(json.loads(line))
                    except ValueError:
                        pass  # a half-written line: it arrives complete next tick
        self.fade()
        if self.client.poll() is not None and not self.pending():
            self.close()
            return
        self.root.after(POLL_MS, self.poll)

    def pending(self) -> bool:
        return self.handle is not None and self.events_path.stat().st_size > self.handle.tell()

    def show(self, event: dict) -> None:
        name = event.get("name", "")
        if name in self.boxes:
            self.boxes[name].configure(bg=LIVE)
            self.fading[name] = 6   # ticks before it cools down
        if name in TOOLS:
            self.tool_label.configure(text=f"tool: {name}", bg=LIVE)
            self.fading["__tool__"] = 6
        where = f"{event.get('file')}:{event.get('line')}"
        thread = event.get("thread", "")
        marker = "" if thread == "MainThread" else f"   [{thread}]"
        indent = "  " * min(int(event.get("depth", 0)), 6)
        self.calls.insert(0, f"{indent}{name:32} {where}{marker}")
        del self.calls[RECENT_CALLS:]
        self.call_list.configure(state="normal")
        self.call_list.delete("1.0", "end")
        self.call_list.insert("1.0", "\n".join(self.calls))
        self.call_list.configure(state="disabled")

    def fade(self) -> None:
        """A box stays warm for a moment, or a fast turn would only flicker."""
        for key in list(self.fading):
            self.fading[key] -= 1
            if self.fading[key] > 0:
                continue
            del self.fading[key]
            if key == "__tool__":
                self.tool_label.configure(bg=IDLE)
            else:
                self.boxes[key].configure(bg=RECENT)

    def close(self) -> None:
        if self.client.poll() is None:
            self.client.terminate()
        if self.handle is not None:
            self.handle.close()
        self.root.destroy()


def main(argv: list[str]) -> int:
    target = argv[1] if len(argv) > 1 else "app"
    if target not in ("app", "cli"):
        print("Usage: try_inspect.py [app | cli]")
        return 2
    events_path = Path(tempfile.gettempdir()) / "job_load_agent_trace.jsonl"
    events_path.unlink(missing_ok=True)
    client = subprocess.Popen(
        [sys.executable, str(HERE / "try_trace.py"), target, "--events", str(events_path)],
        cwd=str(PROJECT),
        creationflags=getattr(subprocess, "CREATE_NEW_CONSOLE", 0) if target == "cli" else 0,
    )
    root = tk.Tk()
    Inspector(root, events_path, client)
    root.mainloop()
    if client.poll() is None:
        client.terminate()
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
