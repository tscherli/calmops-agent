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

from agent.graph import build_agent, message_text
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
    graph.invoke({"current_step": "light", "messages": [HumanMessage(content="boss said it was wrong")]})

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
    graph.invoke({"current_step": "light", "messages": [HumanMessage(content="hi")]})

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
    first.invoke({"current_step": "light", "messages": [HumanMessage(content="crunch")]})

    second = build_agent(
        model=_model_that_logs_then_replies({"mood": "bad", "circumstances": "crunch again"}),
        classifier=_stub_classifier(["high_workload"], calls),
        log_path=log_path,
    )
    second.invoke({"current_step": "light", "messages": [HumanMessage(content="crunch again")]})

    assert calls[0]["known_tags"] == ["high_workload", "overtime"]


def test_classification_failure_still_writes_the_row_untagged(tmp_path):
    """A failed tagger must degrade to an untagged entry, not a lost one."""
    log_path = tmp_path / "mood_log.csv"

    def failing_classifier(**kwargs):
        return {"tags": [], "status": "error", "error": "simulated outage"}

    model = _model_that_logs_then_replies({"mood": "good", "circumstances": "quiet day"})
    graph = build_agent(model=model, classifier=failing_classifier, log_path=log_path)
    result = graph.invoke({"current_step": "light", "messages": [HumanMessage(content="quiet day")]})

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
    first.invoke({"current_step": "light", "messages": [HumanMessage(content="crunch week")]})

    # ...a later one about the same kind of thing should surface it.
    second_model = _model_that_logs_then_replies({"mood": "bad", "circumstances": "crunch again"})
    second = build_agent(model=second_model, classifier=classifier, log_path=log_path)
    result = second.invoke({"current_step": "light", "messages": [HumanMessage(content="crunch again")]})

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
    result = graph.invoke({"current_step": "light", "messages": [HumanMessage(content="first entry")]})

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
    result = graph.invoke({"current_step": "light", "messages": [HumanMessage(content="hi")]})

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
    result = graph.invoke({"current_step": "light", "messages": [HumanMessage(content="hi")]})

    log_message = next(m for m in _tool_messages(result) if m.name == "log_mood")
    assert log_message.status == "error"
    assert json.loads(log_message.content)["status"] == "error"


# -- tool surface ------------------------------------------------------------


def _tools_bound_in_state(state: str, tmp_path, **build_kwargs) -> list:
    """Run one state and report what the model was bound with there.

    This is the load-bearing test helper for the whole redesign: the safety
    properties are "state X cannot do Y", and an empty tool list is what
    makes that true. It is checkable with no LLM involved.
    """
    model = StubChatModel(responses=[AIMessage(content="ok")])
    graph = build_agent(
        model=model,
        classifier=_stub_classifier([]),
        log_path=tmp_path / "mood_log.csv",
        **build_kwargs,
    )
    graph.invoke({"current_step": state, "messages": [HumanMessage(content="hi")]})
    # Structured-output states never call bind_tools at all -- that absence
    # IS the guarantee, so an empty list is the expected result there.
    return [getattr(x, "name", x) for call in model.bind_calls for x in call]


def test_triage_binds_no_tools_so_it_cannot_log(tmp_path, monkeypatch):
    """simple_small_talk_06 regressed when a prompt edit made the agent ask
    for a mood rating on a greeting. Triage now has no logging tool at all,
    so that failure mode is unreachable rather than discouraged."""
    monkeypatch.setenv("OPENAI_ENABLE_WEB_SEARCH", "false")
    assert _tools_bound_in_state("triage", tmp_path) == []


def test_structuring_binds_no_tools_so_it_cannot_log_early(tmp_path, monkeypatch):
    """medium_multiturn_structuring_02 failed by logging on turn 1. The
    structuring state cannot reach log_mood."""
    monkeypatch.setenv("OPENAI_ENABLE_WEB_SEARCH", "false")
    assert _tools_bound_in_state("structuring", tmp_path) == []


def test_light_binds_only_log_mood(tmp_path, monkeypatch):
    """medium_proactivity_ordinary_17 failed by reaching for tools on an
    ordinary bad day. A light check-in can log and nothing else -- no
    web_search, no history lookup."""
    monkeypatch.setenv("OPENAI_ENABLE_WEB_SEARCH", "false")
    assert _tools_bound_in_state("light", tmp_path) == ["log_mood"]


def test_only_post_log_can_reach_history_and_web(tmp_path, monkeypatch):
    monkeypatch.delenv("OPENAI_ENABLE_WEB_SEARCH", raising=False)
    bound = _tools_bound_in_state("post_log", tmp_path)
    assert "find_similar_entries" in bound
    assert {"type": "web_search"} in bound or "web_search" in str(bound)


def test_web_search_can_be_disabled(tmp_path, monkeypatch):
    monkeypatch.setenv("OPENAI_ENABLE_WEB_SEARCH", "false")
    bound = _tools_bound_in_state("post_log", tmp_path)
    assert bound == ["find_similar_entries"]


def test_classify_entry_is_never_exposed_to_the_model(tmp_path, monkeypatch):
    """Binding it as a tool as well as a guaranteed edge would invite the
    model to call it a second time on its own initiative."""
    monkeypatch.delenv("OPENAI_ENABLE_WEB_SEARCH", raising=False)
    for state in ("triage", "structuring", "light", "post_log"):
        assert "classify_entry" not in _tools_bound_in_state(state, tmp_path)


def test_awaiting_scale_asks_without_calling_the_model(tmp_path):
    """E1: the scale question is a constant. Generating it would cost a call
    and let the scale words drift, which is the one thing this state exists
    to prevent."""
    from agent.prompts import SCALE_QUESTION

    model = StubChatModel(responses=[AIMessage(content="SHOULD NOT BE CALLED")])
    graph = build_agent(
        model=model, classifier=_stub_classifier([]), log_path=tmp_path / "mood_log.csv"
    )
    result = graph.invoke(
        {"current_step": "light", "messages": [HumanMessage(content="feeling fine")]}
    )

    # light replied without a log_mood call -> values missing -> scale question
    assert result["current_step"] == "awaiting_scale"
    assert SCALE_QUESTION in message_text(result["messages"][-1])
    # one call for `light`, none for the question itself
    assert model.call_count == 1


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
    result = graph.invoke(
        {"current_step": "light", "messages": [HumanMessage(content="busy")]}
    )

    assert len(_read_rows(log_path)) == 1, "log_mood must not be executed twice"
    assert other_calls == ["thing"]
    answered = {m.tool_call_id for m in _tool_messages(result)}
    assert answered == {"call_log", "call_other"}


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
    result = graph.invoke(
        {"current_step": "post_log", "messages": [HumanMessage(content="hi")]}
    )

    error_message = _tool_messages(result)[0]
    assert error_message.status == "error"
    assert "no_such_tool" in error_message.content
    assert result["messages"][-1].content == "recovered"


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
