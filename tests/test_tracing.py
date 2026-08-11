import json

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from observability.tracing import RunRecorder, summarize_turn

PRICING = {"test-model": {"input": 1.0, "cached_input": 0.1, "output": 2.0}}


def test_summarize_turn_light_reply_no_tool_call():
    messages_before = [HumanMessage(content="feeling good today")]
    messages_after = messages_before + [
        AIMessage(content="Glad to hear it!", usage_metadata={"input_tokens": 10, "output_tokens": 5, "total_tokens": 15})
    ]

    record = summarize_turn(messages_before, messages_after, "test-model", 0.5, pricing=PRICING)

    assert record.input_tokens == 10
    assert record.output_tokens == 5
    assert record.total_tokens == 15
    assert record.tool_calls == []
    assert record.latency_seconds == 0.5


def test_summarize_turn_sums_across_multiple_ai_messages_in_tool_calling_turn():
    """The critical case: a tool-calling turn produces two separate
    AIMessages, each with its own usage_metadata -- both must be counted."""
    messages_before = [HumanMessage(content="exhausted after back-to-back meetings")]
    tool_call_message = AIMessage(
        content="",
        tool_calls=[{"name": "log_mood", "args": {"valence": "bad"}, "id": "call_1", "type": "tool_call"}],
        usage_metadata={"input_tokens": 20, "output_tokens": 5, "total_tokens": 25},
    )
    tool_result_message = ToolMessage(
        content='{"status": "ok"}', name="log_mood", tool_call_id="call_1", status="success"
    )
    final_reply = AIMessage(
        content="Logged it. That sounds draining.",
        usage_metadata={"input_tokens": 30, "output_tokens": 8, "total_tokens": 38},
    )
    messages_after = messages_before + [tool_call_message, tool_result_message, final_reply]

    record = summarize_turn(messages_before, messages_after, "test-model", 1.2, pricing=PRICING)

    assert record.input_tokens == 50  # 20 + 30, NOT just the last message's 30
    assert record.output_tokens == 13  # 5 + 8
    assert record.total_tokens == 63
    assert len(record.tool_calls) == 1
    assert record.tool_calls[0].name == "log_mood"
    assert record.tool_calls[0].status == "success"


def test_summarize_turn_captures_tool_error_status():
    messages_before = [HumanMessage(content="log this")]
    messages_after = messages_before + [
        AIMessage(content="", tool_calls=[{"name": "log_mood", "args": {}, "id": "call_1", "type": "tool_call"}]),
        ToolMessage(content="disk error", name="log_mood", tool_call_id="call_1", status="error"),
        AIMessage(content="That didn't save -- disk error."),
    ]

    record = summarize_turn(messages_before, messages_after, "test-model", 0.8, pricing=PRICING)

    assert len(record.tool_calls) == 1
    assert record.tool_calls[0].status == "error"


def test_summarize_turn_computes_cost_using_pricing_dict():
    messages_before = []
    messages_after = [
        AIMessage(content="hi", usage_metadata={"input_tokens": 1_000_000, "output_tokens": 500_000, "total_tokens": 1_500_000})
    ]

    record = summarize_turn(messages_before, messages_after, "test-model", 0.1, pricing=PRICING)

    # 1M input tokens @ $1.0/1M + 0.5M output tokens @ $2.0/1M = 1.0 + 1.0 = 2.0
    assert record.estimated_cost_usd == 2.0


def test_summarize_turn_unknown_model_returns_zero_cost_without_raising():
    messages_before = []
    messages_after = [
        AIMessage(content="hi", usage_metadata={"input_tokens": 100, "output_tokens": 50, "total_tokens": 150})
    ]

    record = summarize_turn(messages_before, messages_after, "some-unpriced-model", 0.1, pricing=PRICING)

    assert record.estimated_cost_usd == 0.0
    assert record.input_tokens == 100  # tokens still tracked even without pricing data


def test_run_recorder_appends_jsonl_line(tmp_path):
    log_path = tmp_path / "runs.jsonl"
    recorder = RunRecorder(log_path=log_path, pricing=PRICING)
    messages_before = [HumanMessage(content="hi")]
    messages_after = messages_before + [
        AIMessage(content="hello", usage_metadata={"input_tokens": 10, "output_tokens": 5, "total_tokens": 15})
    ]

    recorder.record_turn(messages_before, messages_after, "test-model", 0.3)

    lines = log_path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 1
    entry = json.loads(lines[0])
    assert entry["model"] == "test-model"
    assert entry["input_tokens"] == 10
    assert entry["output_tokens"] == 5
    assert entry["latency_seconds"] == 0.3
    assert entry["tool_calls"] == []


def test_run_recorder_appends_multiple_lines_across_calls(tmp_path):
    log_path = tmp_path / "runs.jsonl"
    recorder = RunRecorder(log_path=log_path, pricing=PRICING)
    messages = [HumanMessage(content="hi"), AIMessage(content="hello", usage_metadata={"input_tokens": 1, "output_tokens": 1, "total_tokens": 2})]

    recorder.record_turn([], messages, "test-model", 0.1)
    recorder.record_turn([], messages, "test-model", 0.2)

    lines = log_path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 2
