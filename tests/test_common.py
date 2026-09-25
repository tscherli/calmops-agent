"""agent/common.py -- helpers that outlive any single graph design."""
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

import agent.common as common
import agent.graph as graph
from agent.common import (
    compact_spent_tool_messages,
    drop_unanswered_tool_calls,
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


# -- a call without its result poisons the whole conversation ----------------


def _call_message(name="suggestion_not_helpful", call_id="c1", content=""):
    return AIMessage(content=content, tool_calls=[{"name": name, "args": {}, "id": call_id}])


def test_a_call_that_was_never_answered_is_left_out():
    """A station whose only edge is to END called a tool in a hand test. The
    unanswered call made OpenAI reject the transcript ("No tool output found
    for function call", 400) on that turn and on every turn after it."""
    messages = [HumanMessage(content="not helpful"), _call_message()]
    assert drop_unanswered_tool_calls(messages) == messages[:1]


def test_an_answered_call_is_kept():
    messages = [
        HumanMessage(content="ideas?"),
        _call_message("find_measures", "c1"),
        ToolMessage(content="[]", tool_call_id="c1", name="find_measures"),
        AIMessage(content="here you go"),
    ]
    assert drop_unanswered_tool_calls(messages) == messages


def test_the_results_of_a_dropped_message_go_with_it():
    """Two calls in one message, one answered: keeping the answer alone would
    leave a result pointing at a call the model never sees."""
    both = AIMessage(
        content="",
        tool_calls=[
            {"name": "find_measures", "args": {}, "id": "c1"},
            {"name": "suggestion_not_helpful", "args": {}, "id": "c2"},
        ],
    )
    messages = [
        HumanMessage(content="hi"),
        both,
        ToolMessage(content="[]", tool_call_id="c1", name="find_measures"),
    ]
    assert drop_unanswered_tool_calls(messages) == messages[:1]


def test_nothing_is_copied_when_every_call_was_answered():
    """The common case must not rebuild the list or reorder anything."""
    messages = [HumanMessage(content="hi"), AIMessage(content="hello")]
    assert drop_unanswered_tool_calls(messages) == messages


def test_the_next_turn_carries_no_unanswered_call():
    """next_turn_state is what both clients hand to the next turn."""
    carried = graph.next_turn_state(
        {"messages": [HumanMessage(content="not helpful"), _call_message()], "energy": "wired"}
    )
    assert carried["messages"] == carried["messages"][:1]
    assert not any(getattr(m, "tool_calls", None) for m in carried["messages"])


def test_a_receipt_carries_the_call_it_answers():
    """The id is what ties a write to the card the user was shown; a receipt
    without one can only say that some tool of that name ran."""
    assert tool_event("log_checkin", "ok", call_id="c1")["call_id"] == "c1"


def test_a_receipt_with_no_call_behind_it_says_so():
    """The provider-run web search is not a call this code made, so there is
    no id to report and inventing one would be a small lie."""
    assert tool_event("web_search", "ok", verified=False)["call_id"] == ""
