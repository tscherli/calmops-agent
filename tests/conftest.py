"""Test-session setup.

**Tracing is force-disabled here, and that matters more than it looks.**
`app.py` calls `load_dotenv()` at import time (it is an entry point, so it
must). `tests/test_app.py` and `tests/test_quick_select.py` import it, which
pulls the real `.env` into the test process -- including
`LANGSMITH_TRACING=true`. Every `graph.invoke()` in the suite then tries to
POST traces to LangSmith.

Two consequences, both observed:
1. Suite runtime becomes hostage to a third-party service. One full run took
   **38 minutes instead of 6 seconds** because tracing calls were retrying
   against an unreachable endpoint.
2. Test runs pollute the real LangSmith project with fixture traffic.

`load_dotenv()` does not override variables that are already set, so setting
them here -- before any test module imports `app` -- wins.
"""
import os

os.environ["LANGSMITH_TRACING"] = "false"
os.environ["LANGCHAIN_TRACING_V2"] = "false"   # legacy name, still honoured
os.environ.setdefault("OPENAI_API_KEY", "sk-test-not-a-real-key")


import time
import tkinter as tk

import pytest


def _new_root(attempts: int = 3) -> tk.Tk:
    """Construct a Tk root, retrying a known environment flake.

    Roughly one full-suite run in three failed at `tk.Tk()` in a *random*
    test. The traceback is not about this project at all: Tk fails to
    `source` vistaTheme.tcl out of the WindowsApps store-Python directory
    ("This probably means that tk wasn't installed properly"), which is
    transient file access inside the Store app container.

    Recorded because it was misdiagnosed twice -- first as a pending
    `after` callback surviving destroy, then as a window-lifecycle bug.
    Neither was true, and both "fixes" left the flake at the same rate. The
    evidence that settled it was reading the actual exception instead of
    reasoning from when it appeared.
    """
    last: Exception | None = None
    for _ in range(attempts):
        try:
            return tk.Tk()
        except tk.TclError as exc:   # pragma: no cover -- environment-dependent
            last = exc
            time.sleep(0.2)
    raise AssertionError(f"Tk could not be constructed after {attempts} attempts: {last}")


@pytest.fixture
def tk_root():
    """A Tk root that is torn down completely.

    Shared here rather than duplicated per test module because getting it
    wrong fails at the *setup* of a later test, not the teardown of the one
    that caused it -- which reads as a random unrelated flake.

    Cancelling pending `after` callbacks before destroy is the part that
    matters. app.py polls its worker queue with `root.after(...)`, so a turn
    leaves timers registered in the Tcl interpreter; destroying the root with
    those still pending can leave the interpreter in a state where the *next*
    `tk.Tk()` fails to construct. Observed as roughly one full-suite run in
    three erroring at `tk.Tk()` in an unrelated test.
    """
    root = _new_root()
    root.withdraw()
    yield root
    try:
        for after_id in root.tk.eval("after info").split():
            try:
                root.after_cancel(after_id)
            except Exception:
                pass
        root.update()
    except tk.TclError:
        pass
    root.destroy()
