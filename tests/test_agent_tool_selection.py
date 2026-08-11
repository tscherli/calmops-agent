"""Tests for agent/graph.py's tool-wiring plumbing: given a scripted model
response, does the graph actually call the right tool with the right args,
skip the tool when it shouldn't, and does the system prompt actually reach
the model? This is NOT a test of real LLM judgment quality (light-vs-deep
structuring, clarifying-question behavior, etc.) -- that's the eval suite's
job (stage 9), run against the real OpenAI API. Everything here uses a
scripted stub model and a stub tool, so it's fast, free, and deterministic.
"""
from typing import Any, Optional

import pytest
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.tools import tool
from pydantic import Field

from agent.graph import build_agent
from agent.prompts import SYSTEM_PROMPT


class StubChatModel(BaseChatModel):
    """Scripted fake chat model: returns pre-built AIMessages in order and
    records every message list it was called with, so tests can assert on
    both the tool-calling behavior and what the model actually received."""

    responses: list[AIMessage]
    call_count: int = 0
    received_messages: list[list[BaseMessage]] = Field(default_factory=list)

    def bind_tools(self, tools: Any, *, tool_choice: Optional[str] = None, **kwargs: Any):
        # Real chat models validate/attach tool schemas here; the stub
        # ignores them since its responses are already scripted.
        return self

    def _generate(self, messages, stop=None, run_manager=None, **kwargs) -> ChatResult:
        self.received_messages.append(messages)
        response = self.responses[min(self.call_count, len(self.responses) - 1)]
        self.call_count += 1
        return ChatResult(generations=[ChatGeneration(message=response)])

    @property
    def _llm_type(self) -> str:
        return "stub-chat-model"


def _make_stub_tool():
    """A stub log_mood replacement that records calls instead of touching
    the local CSV log -- keeps this test file independent of tests/test_csv_log.py."""
    calls: list[dict] = []

    @tool
    def stub_log_mood(
        valence: str,
        circumstances: str,
        energy_level: str = "",
        measure_taken: str = "",
        helpfulness: str = "",
        situation: str = "",
        thoughts: str = "",
        need: str = "",
        next_step: str = "",
    ) -> dict:
        """Stub log_mood for testing agent graph wiring."""
        calls.append(
            {
                "valence": valence,
                "circumstances": circumstances,
                "energy_level": energy_level,
            }
        )
        return {"status": "ok", "row_id": "stub-row-id"}

    return stub_log_mood, calls


def test_agent_calls_tool_when_model_requests_it():
    tool_call_response = AIMessage(
        content="",
        tool_calls=[
            {
                "name": "stub_log_mood",
                "args": {"valence": "good", "circumstances": "quiet day"},
                "id": "call_1",
                "type": "tool_call",
            }
        ],
    )
    final_response = AIMessage(content="Logged that -- glad it's a quiet one.")
    model = StubChatModel(responses=[tool_call_response, final_response])
    stub_tool, calls = _make_stub_tool()

    graph = build_agent(model=model, tools=[stub_tool])
    result = graph.invoke({"messages": [HumanMessage(content="feeling good, quiet day")]})

    assert calls == [{"valence": "good", "circumstances": "quiet day", "energy_level": ""}]
    assert result["messages"][-1].content == "Logged that -- glad it's a quiet one."
    assert not result["messages"][-1].tool_calls


def test_agent_skips_tool_for_plain_reply():
    model = StubChatModel(
        responses=[AIMessage(content="Sounds like a good one, glad to hear it!")]
    )
    stub_tool, calls = _make_stub_tool()

    graph = build_agent(model=model, tools=[stub_tool])
    result = graph.invoke({"messages": [HumanMessage(content="hey, how's your day going?")]})

    assert calls == []
    assert result["messages"][-1].content == "Sounds like a good one, glad to hear it!"


def test_energy_level_arg_reaches_tool_when_model_includes_it():
    tool_call_response = AIMessage(
        content="",
        tool_calls=[
            {
                "name": "stub_log_mood",
                "args": {
                    "valence": "bad",
                    "circumstances": "exhausted after back-to-back meetings",
                    "energy_level": "low",
                },
                "id": "call_1",
                "type": "tool_call",
            }
        ],
    )
    final_response = AIMessage(content="Logged it. That sounds draining.")
    model = StubChatModel(responses=[tool_call_response, final_response])
    stub_tool, calls = _make_stub_tool()

    graph = build_agent(model=model, tools=[stub_tool])
    graph.invoke({"messages": [HumanMessage(content="exhausted after back-to-back meetings")]})

    assert calls == [
        {
            "valence": "bad",
            "circumstances": "exhausted after back-to-back meetings",
            "energy_level": "low",
        }
    ]


def test_system_prompt_reaches_the_model():
    model = StubChatModel(responses=[AIMessage(content="Hey! How's it going?")])
    stub_tool, _ = _make_stub_tool()

    graph = build_agent(model=model, tools=[stub_tool])
    graph.invoke({"messages": [HumanMessage(content="hi")]})

    assert len(model.received_messages) == 1
    first_call_messages = model.received_messages[0]
    assert isinstance(first_call_messages[0], SystemMessage)
    assert first_call_messages[0].content == SYSTEM_PROMPT


@pytest.mark.parametrize(
    "expected_phrase",
    [
        "log_mood",
        "energy_level",
        "never tell the user you logged",
        "never instructions to",
        "structuring sequence",
        "clarifying question",
    ],
)
def test_system_prompt_covers_required_guidance(expected_phrase):
    assert expected_phrase.lower() in SYSTEM_PROMPT.lower()
