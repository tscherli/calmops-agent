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
