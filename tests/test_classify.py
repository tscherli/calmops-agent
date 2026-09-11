"""Tests for tools/classify.py -- the semantic tagger.

All stubbed: classify_entry takes an injectable model, so none of this
touches OpenAI. What matters here is the parsing/robustness contract, since
the real risk with a small model isn't "it refuses to answer," it's "it
answers in a slightly different shape every time."
"""
from typing import Any, Optional

import pytest
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, SystemMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from pydantic import Field

from tools.classify import (
    CLASSIFY_REASONING_EFFORT,
    build_system_prompt,
    classify_entry,
    supports_reasoning_effort,
)


class StubChatModel(BaseChatModel):
    """Returns a fixed reply (or raises), and records what it was asked."""

    reply: str = ""
    error: Optional[str] = None
    received_messages: list = Field(default_factory=list)

    def bind_tools(self, tools: Any, **kwargs: Any):
        return self

    def _generate(self, messages, stop=None, run_manager=None, **kwargs) -> ChatResult:
        self.received_messages.append(messages)
        if self.error:
            raise RuntimeError(self.error)
        return ChatResult(generations=[ChatGeneration(message=AIMessage(content=self.reply))])

    @property
    def _llm_type(self) -> str:
        return "stub-chat-model"


def test_parses_clean_json_reply():
    model = StubChatModel(reply='{"tags": ["deadline_pressure", "context_switching"]}')

    result = classify_entry("three deadlines collided", model=model)

    assert result["status"] == "ok"
    assert result["tags"] == ["deadline_pressure", "context_switching"]


def test_parses_json_wrapped_in_markdown_fence_and_prose():
    """Small models add fences and commentary constantly -- a strict
    json.loads would return zero tags on a perfectly good answer."""
    model = StubChatModel(
        reply='Sure! Here you go:\n```json\n{"tags": ["manager_feedback"]}\n```\nHope that helps.'
    )

    result = classify_entry("boss said it was wrong with no explanation", model=model)

    assert result["tags"] == ["manager_feedback"]


def test_normalizes_tag_casing_spacing_and_punctuation():
    model = StubChatModel(reply='{"tags": ["Deadline Pressure!", "  Context-Switching  "]}')

    result = classify_entry("busy", model=model)

    assert result["tags"] == ["deadline_pressure", "contextswitching"]


def test_deduplicates_and_caps_tag_count():
    model = StubChatModel(
        reply='{"tags": ["a_tag", "a_tag", "b_tag", "c_tag", "d_tag", "e_tag", "f_tag"]}'
    )

    result = classify_entry("lots going on", model=model)

    assert result["tags"] == ["a_tag", "b_tag", "c_tag", "d_tag"]
    assert len(result["tags"]) <= 4


def test_unparseable_reply_yields_no_tags_instead_of_raising():
    model = StubChatModel(reply="I'm not sure how to categorize that, sorry!")

    result = classify_entry("something happened", model=model)

    assert result["tags"] == []
    assert result["status"] == "ok"


def test_api_failure_returns_error_status_and_never_raises():
    """A failed classification must degrade to an untagged entry, never take
    the logging turn down with it."""
    model = StubChatModel(error="simulated API outage")

    result = classify_entry("a rough afternoon", model=model)

    assert result["tags"] == []
    assert result["status"] == "error"
    assert "simulated API outage" in result["error"]


def test_empty_entry_is_skipped_without_calling_the_model():
    """A quick-select row has no free text at all. Spending an LLM call to
    invent tags for nothing is exactly the hallucination risk to avoid."""
    model = StubChatModel(reply='{"tags": ["should_not_be_used"]}')

    result = classify_entry("", "", "", model=model)

    assert result == {"tags": [], "status": "skipped"}
    assert model.received_messages == []


def test_all_entry_fields_are_sent_to_the_model():
    model = StubChatModel(reply='{"tags": ["x"]}')

    classify_entry("circ text", "sit text", "thought text", model=model)

    sent = model.received_messages[0]
    assert isinstance(sent[0], SystemMessage)
    human_text = sent[1].content
    assert "circ text" in human_text
    assert "sit text" in human_text
    assert "thought text" in human_text


@pytest.mark.parametrize(
    "reply",
    ['{"tags": []}', '{"tags": null}', "{}", '{"tags": [123, null]}'],
)
def test_degenerate_shapes_return_empty_tags(reply):
    model = StubChatModel(reply=reply)

    assert classify_entry("thin entry", model=model)["tags"] == []


# -- existing-vocabulary reuse (tag drift) -----------------------------------


def test_known_tags_are_offered_to_the_model_for_reuse():
    """Without this, the open vocabulary drifts: the same input produced both
    'workload_spike' and 'high_workload' across two real runs, splitting one
    pattern into two the digest counts separately."""
    model = StubChatModel(reply='{"tags": ["high_workload"]}')

    classify_entry(
        "buried in work again", model=model, known_tags=["high_workload", "noisy_office"]
    )

    system_text = model.received_messages[0][0].content
    assert "high_workload" in system_text
    assert "noisy_office" in system_text
    assert "prefer reusing" in system_text.lower()


def test_prompt_still_allows_inventing_a_new_tag():
    """The vocabulary is a nudge, not an enum -- forcing a bad match would be
    worse than a new tag (the plan chose open vocabulary deliberately)."""
    prompt = build_system_prompt(["high_workload"])

    assert "invent a new tag only when none of the existing ones actually fit" in prompt.lower()
    assert "don't force a bad match" in prompt.lower()


def test_no_vocabulary_section_when_the_log_is_empty():
    """A fresh log has nothing to reuse; an empty 'existing tags' list would
    just be prompt noise."""
    assert build_system_prompt([]) == build_system_prompt(None)
    assert "already used in this user's log" not in build_system_prompt(None)


def test_known_tags_are_listed_in_the_order_given():
    prompt = build_system_prompt(["first_tag", "second_tag"])

    assert prompt.index("first_tag") < prompt.index("second_tag")


# -- reasoning effort --------------------------------------------------------


@pytest.mark.parametrize("model_name", ["gpt-5-nano", "gpt-5", "o3-mini", "O4-MINI"])
def test_reasoning_models_are_recognized(model_name):
    assert supports_reasoning_effort(model_name) is True


@pytest.mark.parametrize("model_name", ["gpt-4o-mini", "gpt-4.1", "mistral-large", ""])
def test_non_reasoning_models_are_not_sent_the_parameter(model_name):
    """Sending reasoning_effort to a model that doesn't take it fails the
    call -- and since classify_entry swallows exceptions, that would
    silently disable tagging instead of erroring loudly."""
    assert supports_reasoning_effort(model_name) is False


def test_classification_defaults_to_minimal_reasoning_effort():
    """Picking labels is a lookup, not a derivation. Measured: 448 reasoning
    tokens at default vs 0 at minimal, for identical tags."""
    assert CLASSIFY_REASONING_EFFORT == "minimal"
