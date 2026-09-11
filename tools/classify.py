"""Semantic classification of a log entry into free-text tags.

One dedicated, narrow-purpose LLM call -- not a general agentic loop, and
deliberately NOT a `@tool`. It is never bound into agent/graph.py's tools
list, so the model can never choose to call it (or call it twice). Instead
the graph runs it on a deterministic edge, immediately after the model
requests `log_mood` and before the row is actually written to disk. See the
redesign plan's "Agent architecture: hybrid LangGraph design" for why the
ordering is a code guarantee rather than a prompt instruction.

Open vocabulary on purpose: the model invents tags as needed
("deadline_pressure", "context_switching", ...) rather than picking from a
fixed enum, because the useful categories for one person's job load aren't
knowable up front. The downstream consumers (tools/log_query.py's
`find_similar_entries` / `analyze_stress_patterns`) are plain deterministic
lookups over whatever strings end up stored -- no query language, no
LLM-generated code touching the file.

Never raises. A failed or unparseable classification returns empty tags, and
the caller still writes the row -- logging must not depend on tagging
succeeding.
"""
from __future__ import annotations

import json
import os
import re
from typing import Any

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage

DEFAULT_CLASSIFY_MODEL = os.environ.get("OPENAI_CLASSIFY_MODEL", os.environ.get("OPENAI_MODEL", "gpt-5-nano"))

MAX_TAGS = 4

# Picking labels is a lookup, not a derivation -- it needs no chain of
# thought, and left at the default a reasoning model spends one anyway.
# Measured on this exact call (see DEVELOPMENT_LOG.md, stage 17):
#   default:  471 output tokens (448 reasoning), 4.7s, $0.00020845
#   minimal:   23 output tokens (0 reasoning),   0.8s, $0.00002925
# ...for byte-identical tags. "low" was actually *worse* than "minimal",
# inventing a tag the entry didn't support, so this isn't a cost/quality
# tradeoff -- minimal is simply the right setting for this job.
CLASSIFY_REASONING_EFFORT = os.environ.get("OPENAI_CLASSIFY_REASONING_EFFORT", "minimal")

# Only reasoning-model families accept the parameter at all; sending it to
# e.g. gpt-4o-mini would fail the call, and since classify_entry swallows
# exceptions that would silently disable tagging rather than erroring loudly.
REASONING_MODEL_PREFIXES = ("gpt-5", "o1", "o3", "o4")


def supports_reasoning_effort(model_name: str) -> bool:
    return model_name.strip().lower().startswith(REASONING_MODEL_PREFIXES)

CLASSIFY_PROMPT = """You label short work-stress journal entries with semantic tags.

Return ONLY a JSON object of the form {"tags": ["tag_one", "tag_two"]} -- no prose, no \
markdown fences, no explanation.

Rules:
- 1 to 4 tags. Fewer is better than padding with vague ones.
- lower_snake_case, 1-3 words each (e.g. "deadline_pressure", "context_switching", \
"unclear_expectations", "manager_feedback", "interrupted_focus").
- Tag the *situation and its cause*, not the mood itself. "bad_day" or "feeling_sad" are \
useless tags; "no_ticket_system" or "conflicting_priorities" are useful ones.
- Reuse obvious, conventional phrasings so that separate entries about the same kind of \
problem end up sharing a tag. That matching is the entire point.
- If the entry is too thin to say anything specific (e.g. a one-word check-in with no \
context), return {"tags": []} rather than inventing detail that isn't there."""

EXISTING_TAGS_PROMPT = """
Tags already used in this user's log, most frequent first:
{tag_list}

Strongly prefer reusing one of these when it genuinely describes this entry -- entries only \
connect to each other when they share a tag, so a near-duplicate new tag ("workload_spike" \
alongside an existing "high_workload") splits one real pattern into two invisible ones. Invent \
a new tag only when none of the existing ones actually fit; don't force a bad match just to \
reuse something."""


def _extract_tags(raw_text: str) -> list[str]:
    """Pull a tag list out of the model's reply, tolerating stray prose or
    markdown fences around the JSON (small models add both fairly often)."""
    if not raw_text:
        return []

    candidates: list[Any] = []
    try:
        candidates.append(json.loads(raw_text))
    except (ValueError, TypeError):
        match = re.search(r"\{.*\}", raw_text, re.DOTALL)
        if match:
            try:
                candidates.append(json.loads(match.group(0)))
            except (ValueError, TypeError):
                pass

    for candidate in candidates:
        if isinstance(candidate, dict):
            tags = candidate.get("tags")
        elif isinstance(candidate, list):
            tags = candidate
        else:
            tags = None
        if isinstance(tags, list):
            cleaned = []
            for tag in tags:
                if not isinstance(tag, str):
                    continue
                normalized = re.sub(r"\s+", "_", tag.strip().lower())
                normalized = re.sub(r"[^a-z0-9_]", "", normalized).strip("_")
                if normalized and normalized not in cleaned:
                    cleaned.append(normalized)
            return cleaned[:MAX_TAGS]
    return []


def _build_entry_text(circumstances: str, situation: str, thoughts: str) -> str:
    parts = []
    if circumstances:
        parts.append(f"circumstances: {circumstances}")
    if situation:
        parts.append(f"situation: {situation}")
    if thoughts:
        parts.append(f"thoughts: {thoughts}")
    return "\n".join(parts)


def build_system_prompt(known_tags: list[str] | None = None) -> str:
    """The classifier's instructions, plus the log's existing vocabulary when
    there is any. Split out so tests can assert on it directly."""
    if not known_tags:
        return CLASSIFY_PROMPT
    tag_list = "\n".join(f"- {tag}" for tag in known_tags)
    return CLASSIFY_PROMPT + "\n" + EXISTING_TAGS_PROMPT.format(tag_list=tag_list)


def classify_entry(
    circumstances: str,
    situation: str = "",
    thoughts: str = "",
    *,
    model: BaseChatModel | None = None,
    known_tags: list[str] | None = None,
) -> dict[str, Any]:
    """Classify one log entry into semantic tags.

    Returns {"tags": [...], "status": "ok"|"skipped"|"error"}. Never raises:
    on any failure (API error, unparseable reply) it returns empty tags with
    status="error" so the caller can still write the row untagged.

    `known_tags` is the vocabulary already present in the log (see
    tools/log_query.py:known_tags). Passing it keeps the open vocabulary from
    drifting into near-duplicate tags that fragment the history lookup and
    the weekly digest's counts. Omitting it is still valid -- the first entry
    in a fresh log has nothing to reuse.

    `model` is injectable, following this project's DI-for-testability
    convention (tools/csv_log.py's `path`, agent/graph.py's `model`/`tools`).
    """
    entry_text = _build_entry_text(circumstances.strip(), situation.strip(), thoughts.strip())
    if not entry_text:
        # Nothing to classify (e.g. a quick-select row). Don't spend an LLM
        # call inventing tags for an empty entry.
        return {"tags": [], "status": "skipped"}

    try:
        chat_model = model if model is not None else _default_model()
        reply = chat_model.invoke(
            [
                SystemMessage(content=build_system_prompt(known_tags)),
                HumanMessage(content=entry_text),
            ]
        )
        content = reply.content if isinstance(reply.content, str) else str(reply.content)
        return {"tags": _extract_tags(content), "status": "ok"}
    except Exception as exc:
        return {"tags": [], "status": "error", "error": str(exc)}


def _default_model() -> BaseChatModel:
    # Imported lazily so importing this module never requires an API key --
    # same reason tools/sheets.py built its client lazily back in stage 3.
    from langchain_openai import ChatOpenAI

    kwargs: dict[str, Any] = {}
    if CLASSIFY_REASONING_EFFORT and supports_reasoning_effort(DEFAULT_CLASSIFY_MODEL):
        kwargs["reasoning_effort"] = CLASSIFY_REASONING_EFFORT
    return ChatOpenAI(model=DEFAULT_CLASSIFY_MODEL, **kwargs)
