"""Ad-hoc, non-pytest scratch script: run a client and print every call into
this project's own code. Not part of the app -- it starts app.py or cli.py and
watches them from the outside.

    .venv/Scripts/python.exe devtools/try_trace.py cli     # the terminal client
    .venv/Scripts/python.exe devtools/try_trace.py app     # the Tk window

Only functions defined under agent/, tools/ and observability/ are printed,
indented by depth, so a turn reads as the station path and the tools it used
without the noise of langchain and tkinter:

    -> route_start          graph.py:871
      -> says_rating          graph.py:394
      -> station_for          graph.py:410
    -> ask                  graph.py:633
    -> tools_node           graph.py:787
      -> _invoke_tool         graph.py:769
        -> record_choices       graph.py:213

The window runs each turn in a worker thread, so those lines carry the thread
name. For values rather than names, use the debugger instead: .vscode has a
launch configuration per client, and the breakpoints worth keeping are
route_start, the station functions, tools_node and review.
"""
import json
import runpy
import sys
import threading
from pathlib import Path

# The project root, one level up from devtools/. It goes on the import path
# because the client this runs (app.py, cli.py) imports `agent` from there.
PROJECT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT))
WATCHED = tuple(str(PROJECT / part) for part in ("agent", "tools", "observability"))
SKIP = {"<lambda>", "<genexpr>", "<listcomp>", "<dictcomp>", "<setcomp>", "<module>"}
CLIENTS = ("app", "cli")

_depth = 0
_events = None   # set by --events: try_inspect.py reads this file live


def _hook(frame, event, arg):
    """A profile hook: called on every call and return, so it stays cheap by
    filtering on the filename before doing anything else."""
    global _depth
    if event not in ("call", "return"):
        return
    filename = frame.f_code.co_filename
    if not filename.startswith(WATCHED):
        return
    name = frame.f_code.co_name
    # Module bodies and class bodies run once, at import: they say nothing
    # about the turn, and they would bury it before the first message.
    if name in SKIP or name.startswith("__") or name[:1].isupper():
        return
    if event == "return":
        _depth = max(0, _depth - 1)
        return
    thread = threading.current_thread().name
    marker = "" if thread == "MainThread" else f"  [{thread}]"
    if _events is None:
        print(f"    {'  ' * _depth}-> {name:34} {Path(filename).name}:{frame.f_lineno}{marker}", flush=True)
    else:
        # One JSON object per line, flushed: the viewer is a second process
        # tailing this file, so a half-written line would break it.
        _events.write(json.dumps({
            "name": name,
            "file": Path(filename).name,
            "line": frame.f_lineno,
            "depth": _depth,
            "thread": thread,
        }) + "\n")
        _events.flush()
    _depth += 1


def main(argv: list[str]) -> int:
    global _events
    target = argv[1] if len(argv) > 1 else "cli"
    if target not in CLIENTS:
        print(f"Usage: try_trace.py [{' | '.join(CLIENTS)}] [--events PATH]")
        return 2
    rest = argv[2:]
    if "--events" in rest:
        at = rest.index("--events")
        _events = open(rest[at + 1], "w", encoding="utf-8")  # noqa: SIM115 - closed below
        rest = rest[:at] + rest[at + 2:]
    sys.argv = [f"{target}.py", *rest]
    # setprofile per thread, because the window runs every turn in a worker.
    threading.setprofile(_hook)
    sys.setprofile(_hook)
    try:
        runpy.run_path(str(PROJECT / f"{target}.py"), run_name="__main__")
    finally:
        sys.setprofile(None)
        threading.setprofile(None)
        if _events is not None:
            _events.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
