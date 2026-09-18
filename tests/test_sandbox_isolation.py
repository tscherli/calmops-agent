"""Production code must never import from `sandbox/`.

sandbox/ exists to rebuild the agent in LangGraph Studio and learn from it
(see sandbox/README.md). The value of that depends on the dependency running
one way only: code is copied *in*, findings are ported *back* deliberately.
The moment production imports an experiment, the experiment stops being
disposable and production starts inheriting whatever state it is in.

A README can state that rule; only a check the suite runs every time keeps it
true. Imports are read with `ast` rather than a text search, so a docstring
or comment that merely mentions the sandbox -- like this one -- does not trip
it, while a real `import`, `from ... import`, or `importlib.import_module`
call does.
"""
from __future__ import annotations

import ast
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent

# Everything that ships or tests what ships. tests/ is included deliberately:
# a main-suite test importing an experiment would make the suite's result
# depend on whatever state the sandbox happens to be in.
PRODUCTION_PATHS = ("agent", "tools", "evals", "observability", "tests", "app.py", "scheduler.py",
                    "scheduler_config.py")

SKIP_DIRS = {".venv", "venv", "__pycache__", ".git", "sandbox", "node_modules"}


def sandbox_imports(source: str) -> list[str]:
    """Every import of the `sandbox` package found in this source.

    Covers the three ways code can reach it: `import sandbox...`,
    `from sandbox... import ...`, and a string passed to
    `importlib.import_module` / `__import__`.
    """
    found: list[str] = []
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Import):
            found += [a.name for a in node.names if a.name.split(".")[0] == "sandbox"]
        elif isinstance(node, ast.ImportFrom):
            if node.level == 0 and (node.module or "").split(".")[0] == "sandbox":
                found.append(node.module)
        elif isinstance(node, ast.Call):
            func = node.func
            name = func.attr if isinstance(func, ast.Attribute) else getattr(func, "id", "")
            if name in {"import_module", "__import__"} and node.args:
                arg = node.args[0]
                if isinstance(arg, ast.Constant) and isinstance(arg.value, str):
                    if arg.value.split(".")[0] == "sandbox":
                        found.append(arg.value)
    return found


def _production_files() -> list[Path]:
    files: list[Path] = []
    for entry in PRODUCTION_PATHS:
        path = REPO_ROOT / entry
        if path.is_file():
            files.append(path)
        elif path.is_dir():
            files += [
                p for p in path.rglob("*.py")
                if not SKIP_DIRS.intersection(p.relative_to(REPO_ROOT).parts)
            ]
    return files


def test_no_production_code_imports_the_sandbox():
    offenders = {
        str(path.relative_to(REPO_ROOT)): hits
        for path in _production_files()
        if (hits := sandbox_imports(path.read_text(encoding="utf-8")))
    }
    assert not offenders, (
        "Production code imports from sandbox/, which is experimental and not part of the app. "
        "Port the finding into agent/ as a deliberate change instead of importing it: "
        f"{offenders}"
    )


def test_the_scan_actually_covers_production_code():
    """Guards against the check passing vacuously -- a path typo that scanned
    zero files would report "no sandbox imports" forever."""
    scanned = {p.relative_to(REPO_ROOT).parts[0] for p in _production_files()}
    assert {"agent", "tools", "tests", "app.py"} <= scanned


# -- the detector itself, checked against known inputs -----------------------
#
# Without these, the repo scan above could pass simply because the detector
# is broken and finds nothing. Each case is a way production could really
# reach the sandbox, or a way it could look like it does without doing so.


@pytest.mark.parametrize(
    "source",
    [
        "import sandbox",
        "import sandbox.react_agent",
        "import sandbox.react_agent as ra",
        "from sandbox import graph",
        "from sandbox.react_agent import build",
        "import importlib\nimportlib.import_module('sandbox.react_agent')",
        "from importlib import import_module\nimport_module('sandbox')",
        "__import__('sandbox.graph')",
        "def f():\n    from sandbox import graph\n    return graph",
    ],
)
def test_detector_catches_real_imports(source):
    assert sandbox_imports(source)


@pytest.mark.parametrize(
    "source",
    [
        '"""Nothing outside sandbox/ may import from it."""',
        "# see sandbox/README.md",
        "path = 'sandbox/langgraph.json'",
        "import sandboxed_runtime",
        "from agent.sandbox_notes import x",
        "from . import sandbox",
    ],
)
def test_detector_ignores_mentions_that_are_not_imports(source):
    assert sandbox_imports(source) == []
