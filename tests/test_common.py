"""agent/common.py -- helpers that outlive any single graph design."""
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

import agent.common as common
import agent.graph as graph
from agent.common import (
    compact_spent_tool_messages,
    message_text,
    pending_tool_calls,
    tool_event,
    web_search_ran,
)


def test_graph_reuses_the_shared_helpers_rather_than_copying_them():
    """Two copies of a receipt helper would drift, and only one of them would
    be the one the UI and the evaluators read."""
    assert graph.message_text is common.message_text
    assert graph.tool_event is common.tool_event
    assert graph.merge_tool_events is common.merge_tool_events
    assert graph.compact_spent_tool_messages is common.compact_spent_tool_messages


def test_a_skipped_tool_has_no_exit_code():
    """A tool that never ran has no exit code; reporting 0 would be a small lie."""
    assert tool_event("x", "ok")["exit_code"] == 0
    assert tool_event("x", "error")["exit_code"] == 1
    assert tool_event("x", "skipped")["exit_code"] is None


def test_provider_reported_events_are_marked_unverified():
    assert tool_event("web_search", "ok", verified=False)["verified"] is False
    assert tool_event("log_checkin", "ok")["verified"] is True


def test_message_text_flattens_responses_api_content_blocks():
    message = AIMessage(content=[{"type": "text", "text": "hello"}])
    assert message_text(message) == "hello"


def test_web_search_is_detected_from_the_content_blocks():
    """The search runs on OpenAI's side, so a content block is the only evidence."""
    assert web_search_ran(AIMessage(content=[{"type": "web_search_call", "status": "completed"}])) is True
    assert web_search_ran(AIMessage(content="plain reply")) is False


def test_only_unanswered_tool_calls_are_pending():
    call = {"name": "t", "args": {}, "id": "c1"}
    asked = [HumanMessage(content="hi"), AIMessage(content="", tool_calls=[call])]
    assert [c["id"] for c in pending_tool_calls(asked)] == ["c1"]
    answered = asked + [ToolMessage(content="done", name="t", tool_call_id="c1")]
    assert pending_tool_calls(answered) == []


def test_compaction_only_touches_the_named_fat_tool():
    fat = "x" * 300
    messages = [
        ToolMessage(content=fat, name="log_checkin", tool_call_id="a"),
        ToolMessage(content=fat, name="find_measures", tool_call_id="b"),
    ]
    out = compact_spent_tool_messages(messages, tool_names=("log_checkin",))
    assert len(out[0].content) < 100
    assert out[1].content == fat
