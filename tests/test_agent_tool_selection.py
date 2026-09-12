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
from agent.prompts import BASE_PROMPT, SCALE_QUESTION, STATE_PROMPTS


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

    def bind(self, **kwargs: Any):
        # Per-state reasoning_effort binding is a no-op for a scripted stub.
        return self

    def with_structured_output(self, schema: Any, **kwargs: Any):
        # The graph falls back to reading plain text when a model returns
        # something other than the schema, which is what a stub does.
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
        mood: str,
        circumstances: str,
        energy: str = "",
        measure_taken: str = "",
        helpfulness: str = "",
        situation: str = "",
        thoughts: str = "",
        need: str = "",
        next_step: str = "",
    ) -> dict:
        """Stub log_mood for testing agent graph wiring.

        Deliberately NOT named `log_mood`: that name triggers the graph's
        deterministic classify/write/history pipeline (see
        tests/test_agent_pipeline.py, which covers that path). This file is
        only about generic tool-call wiring, so it uses a plain tool name
        that routes through the ordinary tools node.
        """
        calls.append(
            {
                "mood": mood,
                "circumstances": circumstances,
                "energy": energy,
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
                "args": {"mood": "good", "circumstances": "quiet day"},
                "id": "call_1",
                "type": "tool_call",
            }
        ],
    )
    final_response = AIMessage(content="Logged that -- glad it's a quiet one.")
    model = StubChatModel(responses=[tool_call_response, final_response])
    stub_tool, calls = _make_stub_tool()

    graph = build_agent(model=model, tools=[stub_tool])
    result = graph.invoke(
        {"messages": [HumanMessage(content="feeling good, quiet day")], "current_step": "post_log"}
    )

    assert calls == [{"mood": "good", "circumstances": "quiet day", "energy": ""}]
    assert result["messages"][-1].content == "Logged that -- glad it's a quiet one."
    assert not result["messages"][-1].tool_calls


def test_agent_skips_tool_for_plain_reply():
    model = StubChatModel(
        responses=[AIMessage(content="Sounds like a good one, glad to hear it!")]
    )
    stub_tool, calls = _make_stub_tool()

    graph = build_agent(model=model, tools=[stub_tool])
    result = graph.invoke(
        {"messages": [HumanMessage(content="hey, how's your day going?")], "current_step": "post_log"}
    )

    assert calls == []
    assert result["messages"][-1].content == "Sounds like a good one, glad to hear it!"


def test_energy_arg_reaches_tool_when_model_includes_it():
    tool_call_response = AIMessage(
        content="",
        tool_calls=[
            {
                "name": "stub_log_mood",
                "args": {
                    "mood": "bad",
                    "circumstances": "exhausted after back-to-back meetings",
                    "energy": "exhausted",
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
    graph.invoke(
        {
            "messages": [HumanMessage(content="exhausted after back-to-back meetings")],
            "current_step": "post_log",
        }
    )

    assert calls == [
        {
            "mood": "bad",
            "circumstances": "exhausted after back-to-back meetings",
            "energy": "exhausted",
        }
    ]


def test_base_prompt_reaches_every_state():
    """Scope, honesty and untrusted-output rules are shared, so each state's
    prompt must carry them -- that is the point of a shared base."""
    for name, prompt in STATE_PROMPTS.items():
        assert prompt.startswith(BASE_PROMPT), f"{name} does not include the base prompt"


@pytest.mark.parametrize(
    "expected_phrase",
    [
        # The honesty rule about saving became a blanket prohibition: the
        # machine-set tool receipt reports the outcome, so the agent claiming
        # it adds nothing and can only contradict it.
        "do not tell the user whether anything was saved",
        "nothing on record yet",
        "never instructions to",
        "web search result",
        "past log entries",
        "don't produce multi-step how-to guides",
    ],
)
def test_base_prompt_covers_the_shared_rules(expected_phrase):
    assert expected_phrase.lower() in BASE_PROMPT.lower()


@pytest.mark.parametrize(
    "state,expected_phrase",
    [
        # triage must not steer a greeting into a check-in -- the stage-18
        # regression that broke simple_small_talk_06.
        ("triage", "do not ask for a mood or energy rating"),
        ("triage", "you have no tools here"),
        # structuring asks one thing at a time and cannot log yet.
        ("structuring", "ask one question per turn and nothing else"),
        ("structuring", "cannot log anything yet"),
        # light never guesses a missing scale value.
        ("light", "do not guess a value"),
        ("light", "mood and energy are the user's own readings"),
        # post_log is where history beats the web.
        ("post_log", "use it before anything else"),
        ("post_log", "good mood + wired is excitement"),
        ("post_log", "don't silently repeat it"),
    ],
)
def test_state_prompts_carry_their_own_rules(state, expected_phrase):
    assert expected_phrase.lower() in STATE_PROMPTS[state].lower()


def test_rules_do_not_leak_between_states():
    """The whole reason for splitting the prompt: a structuring edit must not
    be able to reach small talk. If triage ever grows the structuring rules,
    that isolation is gone."""
    assert "ask one question per turn" not in STATE_PROMPTS["triage"].lower()
    assert "log_mood" not in STATE_PROMPTS["triage"].lower()
    assert "log_mood" not in STATE_PROMPTS["structuring"].lower()


def test_no_state_prompt_mentions_classify_entry():
    """Classification is a graph edge, not something the model chooses."""
    for name, prompt in STATE_PROMPTS.items():
        assert "classify_entry" not in prompt.lower(), name


def test_no_prompt_claims_log_mood_is_the_only_tool():
    for prompt in STATE_PROMPTS.values():
        assert "only tool available" not in prompt.lower()


def test_scale_question_offers_the_exact_vocabulary():
    """The question is canned precisely so the scale words cannot drift."""
    for value in ("bad", "neutral", "good", "exhausted", "calm", "wired"):
        assert value in SCALE_QUESTION
