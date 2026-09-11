"""Tests for the deterministic post-log pipeline in agent/graph.py.

This is the part of the Stage-14 redesign that exists specifically because a
prompt instruction is not a guarantee: once the model requests `log_mood`,
classification and the history lookup happen on graph edges, whether or not
the model "remembers" to want them. These tests assert that guarantee
structurally -- with a stub model that only ever emits the log_mood call and
never asks for anything else.

Everything is stubbed (scripted model, injected classifier, temp log path):
no OpenAI calls, no writes to the real log.
"""
import csv
import json
from typing import Any, Optional

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, ToolMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.tools import tool
from pydantic import Field

from agent.graph import build_agent
from tools.csv_log import parse_tags


class StubChatModel(BaseChatModel):
    """Scripted model that also records the tool list it was bound with."""

    responses: list[AIMessage]
    call_count: int = 0
    received_messages: list[list[BaseMessage]] = Field(default_factory=list)
    # Every bind_tools call, not just the last: the graph may bind a second
    # time to prepare a no-builtins fallback, and this stub returns `self`
    # rather than a new object the way a real chat model does.
    bind_calls: list[list] = Field(default_factory=list)

    @property
    def bound_tools(self) -> list:
        """The primary binding -- what the model is actually called with."""
        return self.bind_calls[0] if self.bind_calls else []

    def bind_tools(self, tools: Any, *, tool_choice: Optional[str] = None, **kwargs: Any):
        self.bind_calls.append(list(tools))
        return self

    def _generate(self, messages, stop=None, run_manager=None, **kwargs) -> ChatResult:
        self.received_messages.append(messages)
        response = self.responses[min(self.call_count, len(self.responses) - 1)]
        self.call_count += 1
        return ChatResult(generations=[ChatGeneration(message=response)])

    @property
    def _llm_type(self) -> str:
        return "stub-chat-model"


def _log_call(args: dict, call_id: str = "call_log_1") -> AIMessage:
    return AIMessage(
        content="",
        tool_calls=[{"name": "log_mood", "args": args, "id": call_id, "type": "tool_call"}],
    )


def _model_that_logs_then_replies(args: dict, reply: str = "Logged that.") -> StubChatModel:
    return StubChatModel(responses=[_log_call(args), AIMessage(content=reply)])


def _stub_classifier(tags: list[str], calls: list | None = None):
    def classifier(**kwargs):
        if calls is not None:
            calls.append(kwargs)
        return {"tags": list(tags), "status": "ok"}

    return classifier


def _read_rows(path):
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _tool_messages(result) -> list[ToolMessage]:
    return [m for m in result["messages"] if isinstance(m, ToolMessage)]


def _log_tool_payload(result) -> dict:
    message = next(m for m in _tool_messages(result) if m.name == "log_mood")
    return json.loads(message.content)


# -- the core guarantee ------------------------------------------------------


def test_entry_is_classified_and_written_with_tags_in_one_row(tmp_path):
    """The whole point of the redesign: the model asked only for log_mood,
    and the row still lands tagged, because the graph -- not the model --
    owns that ordering."""
    log_path = tmp_path / "mood_log.csv"
    model = _model_that_logs_then_replies({"mood": "bad", "circumstances": "vague feedback"})

    graph = build_agent(
        model=model,
        classifier=_stub_classifier(["manager_feedback", "unclear_expectations"]),
        log_path=log_path,
    )
    graph.invoke({"messages": [HumanMessage(content="boss said it was wrong")]})

    rows = _read_rows(log_path)
    assert len(rows) == 1, "the row must be written exactly once, never appended then rewritten"
    assert parse_tags(rows[0]["tags"]) == ["manager_feedback", "unclear_expectations"]
    assert rows[0]["mood"] == "bad"


def test_classifier_receives_the_entry_text_from_the_tool_call(tmp_path):
    calls: list = []
    model = _model_that_logs_then_replies(
        {
            "mood": "bad",
            "circumstances": "circ",
            "situation": "sit",
            "thoughts": "thought",
        }
    )

    graph = build_agent(
        model=model,
        classifier=_stub_classifier(["x"], calls),
        log_path=tmp_path / "mood_log.csv",
    )
    graph.invoke({"messages": [HumanMessage(content="hi")]})

    assert calls == [
        {
            "circumstances": "circ",
            "situation": "sit",
            "thoughts": "thought",
            "known_tags": [],  # fresh log, nothing to reuse yet
        }
    ]


def test_existing_log_vocabulary_is_passed_to_the_classifier(tmp_path):
    """Tag-drift guard, end to end: by the second entry the classifier is
    told what vocabulary already exists, so it can reuse instead of coining
    a near-duplicate."""
    log_path = tmp_path / "mood_log.csv"
    calls: list = []

    first = build_agent(
        model=_model_that_logs_then_replies({"mood": "bad", "circumstances": "crunch"}),
        classifier=_stub_classifier(["high_workload", "overtime"]),
        log_path=log_path,
    )
    first.invoke({"messages": [HumanMessage(content="crunch")]})

    second = build_agent(
        model=_model_that_logs_then_replies({"mood": "bad", "circumstances": "crunch again"}),
        classifier=_stub_classifier(["high_workload"], calls),
        log_path=log_path,
    )
    second.invoke({"messages": [HumanMessage(content="crunch again")]})

    assert calls[0]["known_tags"] == ["high_workload", "overtime"]


def test_classification_failure_still_writes_the_row_untagged(tmp_path):
    """A failed tagger must degrade to an untagged entry, not a lost one."""
    log_path = tmp_path / "mood_log.csv"

    def failing_classifier(**kwargs):
        return {"tags": [], "status": "error", "error": "simulated outage"}

    model = _model_that_logs_then_replies({"mood": "good", "circumstances": "quiet day"})
    graph = build_agent(model=model, classifier=failing_classifier, log_path=log_path)
    result = graph.invoke({"messages": [HumanMessage(content="quiet day")]})

    rows = _read_rows(log_path)
    assert len(rows) == 1
    assert rows[0]["tags"] == ""
    assert rows[0]["circumstances"] == "quiet day"
    assert _log_tool_payload(result)["status"] == "ok"


def test_history_lookup_runs_after_logging_and_reaches_the_model(tmp_path):
    log_path = tmp_path / "mood_log.csv"
    classifier = _stub_classifier(["deadline_pressure"])

    # First conversation writes a tagged entry...
    first = build_agent(
        model=_model_that_logs_then_replies(
            {"mood": "bad", "circumstances": "crunch week", "measure_taken": "asked for help"}
        ),
        classifier=classifier,
        log_path=log_path,
    )
    first.invoke({"messages": [HumanMessage(content="crunch week")]})

    # ...a later one about the same kind of thing should surface it.
    second_model = _model_that_logs_then_replies({"mood": "bad", "circumstances": "crunch again"})
    second = build_agent(model=second_model, classifier=classifier, log_path=log_path)
    result = second.invoke({"messages": [HumanMessage(content="crunch again")]})

    payload = _log_tool_payload(result)
    assert payload["tags"] == ["deadline_pressure"]
    assert len(payload["similar_past_entries"]) == 1
    assert payload["similar_past_entries"][0]["circumstances"] == "crunch week"
    assert payload["similar_past_entries"][0]["measure_taken"] == "asked for help"

    # And it actually reached the model on its next turn, not just the state.
    final_prompt = second_model.received_messages[-1]
    assert "crunch week" in str(final_prompt[-1].content)


def test_empty_history_is_reported_as_an_empty_list(tmp_path):
    """Honesty rule support: 'nothing on record' has to be representable and
    visibly empty, so the model can say so instead of inventing a pattern."""
    model = _model_that_logs_then_replies({"mood": "good", "circumstances": "first ever entry"})
    graph = build_agent(
        model=model, classifier=_stub_classifier(["brand_new_tag"]), log_path=tmp_path / "log.csv"
    )
    result = graph.invoke({"messages": [HumanMessage(content="first entry")]})

    assert _log_tool_payload(result)["similar_past_entries"] == []


def test_just_written_entry_is_not_returned_as_its_own_precedent(tmp_path):
    """Regression: the lookup runs right after the append, so without an
    explicit exclusion the new row matches its own tags and the agent gets
    handed a one-row "pattern" it would honestly-but-wrongly report."""
    log_path = tmp_path / "mood_log.csv"
    model = _model_that_logs_then_replies({"mood": "bad", "circumstances": "the only entry"})

    graph = build_agent(
        model=model, classifier=_stub_classifier(["some_tag"]), log_path=log_path
    )
    result = graph.invoke({"messages": [HumanMessage(content="hi")]})

    assert len(_read_rows(log_path)) == 1
    assert _log_tool_payload(result)["similar_past_entries"] == []


def test_failed_write_is_reported_as_an_error_tool_message(tmp_path):
    """The fabrication evaluator keys off ToolMessage.status, so a failed
    write must not come back looking like a success."""
    unwritable = tmp_path / "nope"
    unwritable.write_text("not a directory", encoding="utf-8")

    model = _model_that_logs_then_replies({"mood": "bad", "circumstances": "disk trouble"})
    graph = build_agent(
        model=model,
        classifier=_stub_classifier(["x"]),
        log_path=unwritable / "mood_log.csv",
    )
    result = graph.invoke({"messages": [HumanMessage(content="hi")]})

    log_message = next(m for m in _tool_messages(result) if m.name == "log_mood")
    assert log_message.status == "error"
    assert json.loads(log_message.content)["status"] == "error"


# -- tool surface ------------------------------------------------------------


def test_classify_entry_is_never_exposed_to_the_model(tmp_path):
    """Binding it as a tool as well as a guaranteed edge would invite the
    model to call it a second time on its own initiative."""
    model = _model_that_logs_then_replies({"mood": "good", "circumstances": "x"})
    build_agent(model=model, classifier=_stub_classifier([]), log_path=tmp_path / "log.csv")

    bound_names = {getattr(t, "name", None) for t in model.bound_tools}
    assert "classify_entry" not in bound_names


def test_default_tool_surface_is_log_mood_and_find_similar_entries(tmp_path, monkeypatch):
    monkeypatch.setenv("OPENAI_ENABLE_WEB_SEARCH", "false")
    model = _model_that_logs_then_replies({"mood": "good", "circumstances": "x"})

    build_agent(model=model, classifier=_stub_classifier([]), log_path=tmp_path / "log.csv")

    bound_names = {getattr(t, "name", None) for t in model.bound_tools}
    assert bound_names == {"log_mood", "find_similar_entries"}


def test_web_search_is_bound_by_default_and_can_be_disabled(tmp_path, monkeypatch):
    monkeypatch.delenv("OPENAI_ENABLE_WEB_SEARCH", raising=False)
    model = _model_that_logs_then_replies({"mood": "good", "circumstances": "x"})
    build_agent(model=model, classifier=_stub_classifier([]), log_path=tmp_path / "log.csv")
    assert {"type": "web_search"} in model.bound_tools

    monkeypatch.setenv("OPENAI_ENABLE_WEB_SEARCH", "false")
    off_model = _model_that_logs_then_replies({"mood": "good", "circumstances": "x"})
    build_agent(model=off_model, classifier=_stub_classifier([]), log_path=tmp_path / "log.csv")
    assert {"type": "web_search"} not in off_model.bound_tools


def test_mixed_tool_calls_are_each_answered_exactly_once(tmp_path):
    """A model can emit log_mood alongside another tool call in one message.
    The pipeline answers one and the generic node answers the other -- with
    no double execution of the log write."""
    log_path = tmp_path / "mood_log.csv"
    other_calls: list = []

    @tool
    def some_other_tool(value: str) -> str:
        """A second tool the model might call in the same message."""
        other_calls.append(value)
        return f"did {value}"

    mixed = AIMessage(
        content="",
        tool_calls=[
            {
                "name": "log_mood",
                "args": {"mood": "bad", "circumstances": "busy"},
                "id": "call_log",
                "type": "tool_call",
            },
            {
                "name": "some_other_tool",
                "args": {"value": "thing"},
                "id": "call_other",
                "type": "tool_call",
            },
        ],
    )
    model = StubChatModel(responses=[mixed, AIMessage(content="all done")])

    from tools.csv_log import log_mood

    graph = build_agent(
        model=model,
        tools=[log_mood, some_other_tool],
        classifier=_stub_classifier(["busy_tag"]),
        log_path=log_path,
    )
    result = graph.invoke({"messages": [HumanMessage(content="busy")]})

    assert len(_read_rows(log_path)) == 1, "log_mood must not be executed twice"
    assert other_calls == ["thing"]
    answered = {m.tool_call_id for m in _tool_messages(result)}
    assert answered == {"call_log", "call_other"}
    assert result["messages"][-1].content == "all done"


def test_unknown_tool_call_is_reported_as_an_error_not_a_crash(tmp_path):
    model = StubChatModel(
        responses=[
            AIMessage(
                content="",
                tool_calls=[
                    {"name": "no_such_tool", "args": {}, "id": "c1", "type": "tool_call"}
                ],
            ),
            AIMessage(content="recovered"),
        ]
    )
    from tools.csv_log import log_mood

    graph = build_agent(model=model, tools=[log_mood], log_path=tmp_path / "log.csv")
    result = graph.invoke({"messages": [HumanMessage(content="hi")]})

    error_message = _tool_messages(result)[0]
    assert error_message.status == "error"
    assert "no_such_tool" in error_message.content
    assert result["messages"][-1].content == "recovered"


def test_recalled_entries_are_trimmed_before_reaching_the_model(tmp_path):
    """Three past entries ride along on every logged turn, so their free
    text is bounded -- a real smoke test produced a 200+ char next_step."""
    log_path = tmp_path / "mood_log.csv"
    classifier = _stub_classifier(["deadline_pressure"])

    long_plan = "pick one ticket and finish it " * 20
    build_agent(
        model=_model_that_logs_then_replies(
            {
                "mood": "bad",
                "circumstances": "x" * 400,
                "next_step": long_plan,
            }
        ),
        classifier=classifier,
        log_path=log_path,
    ).invoke({"messages": [HumanMessage(content="hi")]})

    second_model = _model_that_logs_then_replies({"mood": "bad", "circumstances": "again"})
    result = build_agent(
        model=second_model, classifier=classifier, log_path=log_path
    ).invoke({"messages": [HumanMessage(content="again")]})

    recalled = _log_tool_payload(result)["similar_past_entries"][0]
    assert len(recalled["circumstances"]) <= 160
    assert len(recalled["next_step"]) <= 160
    assert recalled["circumstances"].endswith("…")
    # A date is enough to say "last Tuesday"; the full ISO timestamp and the
    # entry's complete tag list are dropped as redundant.
    assert len(recalled["date"]) == 10
    assert "timestamp" not in recalled
    assert "tags" not in recalled
    assert recalled["matched_tags"] == ["deadline_pressure"]


def test_short_recalled_fields_are_left_untouched(tmp_path):
    log_path = tmp_path / "mood_log.csv"
    classifier = _stub_classifier(["deadline_pressure"])

    build_agent(
        model=_model_that_logs_then_replies(
            {"mood": "bad", "circumstances": "crunch week", "measure_taken": "asked for help"}
        ),
        classifier=classifier,
        log_path=log_path,
    ).invoke({"messages": [HumanMessage(content="hi")]})

    result = build_agent(
        model=_model_that_logs_then_replies({"mood": "bad", "circumstances": "again"}),
        classifier=classifier,
        log_path=log_path,
    ).invoke({"messages": [HumanMessage(content="again")]})

    recalled = _log_tool_payload(result)["similar_past_entries"][0]
    assert recalled["circumstances"] == "crunch week"
    assert recalled["measure_taken"] == "asked for help"


def test_message_text_flattens_responses_api_content_blocks():
    """Regression: binding the built-in web_search switches langchain-openai
    to the Responses API, where AIMessage.content is a list of blocks
    (reasoning, encrypted payloads, text). Reading `.content` straight into
    the UI printed raw dicts and base64 at the user -- found by a real smoke
    test, invisible to every stubbed test since stubs all return str."""
    from agent.graph import message_text

    plain = AIMessage(content="just a string")
    assert message_text(plain) == "just a string"

    blocks = AIMessage(
        content=[
            {"type": "reasoning", "id": "rs_1", "encrypted_content": "gAAAAABqo4=="},
            {"type": "text", "text": "Logged that -- sounds like a rough one."},
        ]
    )
    text = message_text(blocks)
    assert text == "Logged that -- sounds like a rough one."
    assert "encrypted_content" not in text
    assert "gAAAAAB" not in text


def test_serialized_eval_messages_carry_clean_text_not_block_dicts():
    """The fabrication evaluator keyword-scans this content, so block dicts
    leaking in would make it scan reasoning metadata instead of the reply."""
    from evals.target import serialize_messages

    serialized = serialize_messages(
        [
            AIMessage(
                content=[
                    {"type": "reasoning", "id": "rs_1", "encrypted_content": "gAAAAABqo4=="},
                    {"type": "text", "text": "I haven't logged anything yet."},
                ]
            )
        ]
    )

    assert serialized[0]["content"] == "I haven't logged anything yet."


def test_recursion_limit_is_configured_on_the_compiled_graph(tmp_path):
    from agent.graph import DEFAULT_RECURSION_LIMIT

    model = _model_that_logs_then_replies({"mood": "good", "circumstances": "x"})
    graph = build_agent(model=model, log_path=tmp_path / "log.csv")

    assert graph.config.get("recursion_limit") == DEFAULT_RECURSION_LIMIT
