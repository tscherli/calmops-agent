"""agent/model.py -- the one place that builds a chat model."""
import pytest

from agent.model import (
    DEBATE_MODEL,
    DEFAULT_MODEL,
    build_chat_model,
    model_name_of,
    resolve_debate_model,
    resolve_model,
    supports_reasoning_effort,
)


# -- reasoning effort --------------------------------------------------------


@pytest.mark.parametrize("model_name", ["gpt-5-nano", "gpt-5", "o3-mini", "O4-MINI"])
def test_reasoning_models_are_recognized(model_name):
    assert supports_reasoning_effort(model_name) is True


@pytest.mark.parametrize("model_name", ["gpt-4o-mini", "gpt-4.1", "mistral-large", ""])
def test_non_reasoning_models_are_not_sent_the_parameter(model_name):
    """Sending reasoning_effort to a model that doesn't take it fails the call."""
    assert supports_reasoning_effort(model_name) is False


# -- how the model is built --------------------------------------------------


def test_every_model_uses_the_responses_api():
    """A station that binds no built-in tool still reads a history that may
    contain web-search results. Without the Responses API that request is
    rejected with 400 Invalid value: 'web_search_call'."""
    assert build_chat_model("gpt-5-nano").use_responses_api is True


def test_reasoning_effort_is_low_not_minimal_by_default():
    """OpenAI refuses web_search together with reasoning_effort=minimal."""
    assert build_chat_model("gpt-5-nano").reasoning_effort == "low"


def test_effort_is_omitted_for_models_that_reject_it():
    assert build_chat_model("gpt-4o-mini").reasoning_effort is None


def test_effort_can_be_overridden_per_node():
    assert build_chat_model("gpt-5-mini", reasoning_effort="medium").reasoning_effort == "medium"


# -- resolving ---------------------------------------------------------------


def test_none_resolves_to_the_default_model():
    assert model_name_of(resolve_model(None)) == DEFAULT_MODEL


def test_a_name_is_built_and_an_instance_is_passed_through():
    built = resolve_model("gpt-5-mini")
    assert model_name_of(built) == "gpt-5-mini"
    assert resolve_model(built) is built


def test_debate_defaults_to_the_stronger_model():
    """Honest pushback instead of agreement is where small models are weakest."""
    assert model_name_of(resolve_debate_model(None)) == DEBATE_MODEL
    assert DEBATE_MODEL != DEFAULT_MODEL
