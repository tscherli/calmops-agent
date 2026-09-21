"""Test-session setup.

**Tracing is force-disabled here, and that matters more than it looks.**
`app.py` calls `load_dotenv()` at import time (it is an entry point, so it
must). `tests/test_app.py` and `tests/test_tool_receipts.py` import it, which
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

# A second isolation the suite needs for the same reason as the tracing
# switch above, found the same way -- by looking at the real file. See
# `isolated_runs_log` below.


import time
import tkinter as tk
import uuid
from collections.abc import Callable
from typing import Any

import pytest
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langgraph.types import Command
from pydantic import Field


def approve_all(request: dict) -> dict:
    return {"type": "approve"}


def invoke_approving(graph, state: dict, decide: Callable[[dict], dict] = approve_all) -> dict:
    """Run one turn the way the clients do: a fresh thread, every pause answered.

    `decide` gets each action request and returns the decision for it. The
    result carries `pauses`, how many times the turn stopped for the user, so a
    test can assert on the count rather than infer it.
    """
    config = {"configurable": {"thread_id": uuid.uuid4().hex}}
    result = graph.invoke(state, config)
    pauses = 0
    while result.get("__interrupt__"):
        pauses += 1
        requests = result["__interrupt__"][0].value["action_requests"]
        decisions = [{"id": r["id"], **decide(r)} for r in requests]
        result = graph.invoke(Command(resume={"decisions": decisions}), config)
    result["pauses"] = pauses
    return result


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


@pytest.fixture(autouse=True)
def isolated_runs_log(tmp_path, monkeypatch):
    """No test may append to the real `observability/runs.jsonl`.

    `RunRecorder` falls back to that path whenever it is constructed without
    one, and `ChatApp` constructs its own recorder when a test does not pass
    it a stub. So every UI test that ran a turn wrote a row into the file the
    project uses to measure what a real turn costs.

    The rows are not merely extra: they carry zero tokens and zero cost,
    because a stubbed model reports no usage. Measured before this fixture
    existed, 355 of 520 rows were fixture noise, and the mean cost per turn
    read three times cheaper than the truth. A budget check against that file
    would have passed while saying nothing.

    Patched here rather than in each test module because the failure is
    silent, shows up nowhere near the test that caused it, and only surfaces
    when someone finally reads the file to answer a cost question.
    """
    monkeypatch.setattr("observability.tracing.RUNS_LOG_PATH", tmp_path / "runs.jsonl")


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


class StubChatModel(BaseChatModel):
    """A scripted model that also records how it was bound.

    Lives here rather than in one test module because nearly every graph test
    needs it: a station is only testable if the model's next move can be
    dictated, and the binding record is how "this station cannot suggest" is
    checked without calling OpenAI.

    `bind_calls` keeps every bind_tools call, not just the last. A real chat
    model returns a new object per bind; this stub returns `self`, so the
    graph's several bindings would otherwise overwrite each other.
    """

    responses: list[AIMessage]
    call_count: int = 0
    received_messages: list[list[BaseMessage]] = Field(default_factory=list)
    bind_calls: list[list] = Field(default_factory=list)
    tool_choices: list[Any] = Field(default_factory=list)

    @property
    def bound_tools(self) -> list:
        """The primary binding -- what the model is actually called with."""
        return self.bind_calls[0] if self.bind_calls else []

    def bind_tools(self, tools: Any, *, tool_choice: Any = None, **kwargs: Any):
        self.bind_calls.append(list(tools))
        self.tool_choices.append(tool_choice)
        return self

    def bind(self, **kwargs: Any):
        return self

    def with_structured_output(self, schema: Any, **kwargs: Any):
        return self

    def _generate(self, messages, stop=None, run_manager=None, **kwargs) -> ChatResult:
        self.received_messages.append(messages)
        response = self.responses[min(self.call_count, len(self.responses) - 1)]
        self.call_count += 1
        return ChatResult(generations=[ChatGeneration(message=response)])

    @property
    def _llm_type(self) -> str:
        return "stub-chat-model"
