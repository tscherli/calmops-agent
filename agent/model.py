"""One place that decides which model a node talks to, and how it is built.

**Why the agent is on OpenAI, and why the project is not on one provider.**
Asked often enough to write down. Four reasons, in order of how hard they are
to undo:

1. **An LLM judge must not share a family with what it grades.** A model finds
   its own reasoning persuasive, so grading GPT output with GPT invites
   correlated blind spots (`evals/judge.py`). That means *something* here has
   to be a second provider -- a single-provider setup does not remove the
   problem, it just decides which side is compromised. The judge is the
   cheapest place to put the boundary: short trajectories, and its whole
   output is one bit.
2. **The web search is OpenAI's own server-side tool**, bound as
   `{"type": "web_search"}` (`agent/graph.py`). It is executed by OpenAI
   rather than by this code, so there is nothing here to point at another
   provider -- moving the agent means rebuilding that path, not reconfiguring
   it.
3. **Cost tracking only prices OpenAI.** `observability/tracing.py` holds
   rates for `gpt-5-nano` and `gpt-5-mini`; anything else lands in
   `unpriced_models` and reports nothing. A silent zero is worse than no
   figure, and cost tracking is a stated requirement of this project.
4. **Every recorded finding is on `gpt-5-mini`.** The hand tests, the
   stage-54 measurements, the cost per turn the eval budget is planned
   against. Switching the agent does not just change a setting; it invalidates
   the baseline everything else is compared to.

None of this argues OpenAI is better. It argues that the split is structural
rather than incidental: reason 1 means two providers are required, and reasons
2 to 4 decide which one the agent sits on today.

Two settings are project-wide and easy to get wrong in only one of several
call sites:

- `use_responses_api=True`. Binding OpenAI's built-in web search switches
  langchain-openai to the Responses API, and its results are stored in the
  conversation as Responses-API content blocks. A later call that is *not*
  on that API rejects the history with `400 Invalid value: 'web_search_call'`.
  Since one station searches and the others read the same history, every model
  in the app uses the Responses API.
- `reasoning_effort`. Only reasoning-model families accept the parameter, so it
  is sent through `supports_reasoning_effort` rather than unconditionally.
  `minimal` is refused together with web search (400), which is why the default
  here is `low` rather than `minimal`.
"""
from __future__ import annotations

import os

from langchain_core.language_models import BaseChatModel

# Nano kept ignoring prompt rules in hand tests -- it invented a measure when
# asked to pick one, claimed it had added measures it never wrote, and showed
# four options where the prompt says two. Mini is the default while that is
# being watched; OPENAI_MODEL=gpt-5-nano puts the cheap one back.
DEFAULT_MODEL = os.environ.get("OPENAI_MODEL", "gpt-5-mini")

# The debate station argues with the user's own idea instead of agreeing with
# it, which is where small models are weakest (see DEVELOPMENT_LOG, stage 47).
# It is its own setting so it can be raised above the default on its own.
DEBATE_MODEL = os.environ.get("OPENAI_DEBATE_MODEL", "gpt-5-mini")

DEFAULT_REASONING_EFFORT = os.environ.get("OPENAI_REASONING_EFFORT", "low")

# Only reasoning-model families accept `reasoning_effort` at all; sending it to
# e.g. gpt-4o-mini fails the call.
REASONING_MODEL_PREFIXES = ("gpt-5", "o1", "o3", "o4")


def supports_reasoning_effort(model_name: str) -> bool:
    return model_name.strip().lower().startswith(REASONING_MODEL_PREFIXES)


def model_name_of(model: BaseChatModel) -> str:
    return getattr(model, "model_name", None) or getattr(model, "model", "") or ""


def build_chat_model(model_name: str, *, reasoning_effort: str | None = None) -> BaseChatModel:
    """A ChatOpenAI configured the way every node in this app needs it."""
    from langchain_openai import ChatOpenAI

    effort = DEFAULT_REASONING_EFFORT if reasoning_effort is None else reasoning_effort
    kwargs: dict[str, object] = {"model": model_name, "use_responses_api": True}
    if effort and supports_reasoning_effort(model_name):
        kwargs["reasoning_effort"] = effort
    return ChatOpenAI(**kwargs)


def resolve_model(model: str | BaseChatModel | None) -> BaseChatModel:
    """A model instance is used as given; a name or None is built here."""
    if model is None:
        return build_chat_model(DEFAULT_MODEL)
    if isinstance(model, str):
        return build_chat_model(model)
    return model


def resolve_debate_model(model: str | BaseChatModel | None) -> BaseChatModel:
    if model is None:
        return build_chat_model(DEBATE_MODEL)
    if isinstance(model, str):
        return build_chat_model(model)
    return model
