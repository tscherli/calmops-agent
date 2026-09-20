"""RunRecorder: per-turn token/cost/latency/tool-call logging, independent
of LangSmith. LangSmith tracing itself needs no code here -- it's fully
automatic once LANGCHAIN_TRACING_V2/LANGCHAIN_API_KEY/LANGCHAIN_PROJECT (or
their LANGSMITH_* equivalents) are set in the environment; this module is
the separate, structured JSONL record of tokens/cost/latency/tool status.

Critical correctness note (verified against installed langchain/langgraph
source before writing this): langchain.agents.create_agent's model node
runs once per LLM call, not once per user turn -- a turn where the agent
calls a tool and then replies produces multiple AIMessages, each with its
own independent usage_metadata. Reading only the last message would
undercount tokens/cost on every tool-calling turn. summarize_turn sums
usage_metadata across every new AIMessage produced during one
graph.invoke() call instead.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from langchain_core.messages import AIMessage, BaseMessage, ToolMessage

BASE_DIR = Path(__file__).resolve().parent.parent
RUNS_LOG_PATH = BASE_DIR / "observability" / "runs.jsonl"

# USD per 1M tokens, keyed by exact model-id string so an OPENAI_MODEL
# override doesn't get silently mispriced under a neighboring model's rate
# (e.g. gpt-5-nano vs the separate, pricier gpt-5.4-nano). Verified against
# OpenAI's own pricing docs at the time this was written -- re-verify before
# relying on this for real budgeting, since pricing changes.
PRICING_PER_MILLION_TOKENS: dict[str, dict[str, float]] = {
    "gpt-5-nano": {"input": 0.05, "cached_input": 0.005, "output": 0.40},
    "gpt-5-mini": {"input": 0.25, "cached_input": 0.025, "output": 2.00},
}

# OpenAI's built-in web search is billed per call on top of the tokens
# ($10 / 1k calls, checked 2026-09-20). At roughly one search per check-in it
# is the largest single item in the running cost, so it is counted here rather
# than left out of the record.
WEB_SEARCH_USD_PER_CALL = 0.01


@dataclass
class ToolCallSummary:
    name: str
    status: str


@dataclass
class TurnRecord:
    model: str
    input_tokens: int
    output_tokens: int
    total_tokens: int
    estimated_cost_usd: float
    latency_seconds: float
    tool_calls: list[ToolCallSummary] = field(default_factory=list)
    models_used: list[str] = field(default_factory=list)
    unpriced_models: list[str] = field(default_factory=list)
    web_searches: int = 0

    def to_json_dict(self) -> dict[str, Any]:
        return {
            "model": self.model,
            "models_used": self.models_used,
            "unpriced_models": self.unpriced_models,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "total_tokens": self.total_tokens,
            "web_searches": self.web_searches,
            "estimated_cost_usd": round(self.estimated_cost_usd, 6),
            "latency_seconds": round(self.latency_seconds, 3),
            "tool_calls": [{"name": t.name, "status": t.status} for t in self.tool_calls],
        }


def _estimate_cost(
    model: str, input_tokens: int, output_tokens: int, pricing: dict[str, dict[str, float]]
) -> float:
    rates = pricing.get(model)
    if rates is None:
        return 0.0
    return (input_tokens / 1_000_000) * rates["input"] + (output_tokens / 1_000_000) * rates["output"]


def _model_of(message: AIMessage, fallback: str) -> str:
    """Which model produced this message.

    Stations run on different models (the debate station is deliberately a
    stronger one), so a turn can mix them. Pricing the whole turn at the
    caller's default model would misprice exactly the expensive part.
    """
    return (message.response_metadata or {}).get("model_name") or fallback


def _web_search_calls(message: AIMessage) -> int:
    """Built-in searches run inside the model call, so they leave no ToolMessage."""
    content = message.content
    if not isinstance(content, list):
        return 0
    return sum(
        1
        for block in content
        if isinstance(block, dict) and "web_search" in str(block.get("type", ""))
    )


def summarize_turn(
    messages_before: list[BaseMessage],
    messages_after: list[BaseMessage],
    model: str,
    latency_seconds: float,
    *,
    pricing: dict[str, dict[str, float]] | None = None,
    web_search_usd_per_call: float = WEB_SEARCH_USD_PER_CALL,
) -> TurnRecord:
    """Summarize one agent turn's token usage, cost, and tool calls.

    messages_before/messages_after are the graph state's message list
    immediately before and immediately after one graph.invoke() call --
    capture messages_before BEFORE calling invoke(). Sums usage_metadata
    across every new AIMessage produced during that call, since a
    tool-calling turn involves more than one LLM call (see module
    docstring); reading only the last message would undercount.

    Tokens are attributed per message to the model that produced them, and a
    model with no price entry is listed in `unpriced_models` rather than
    quietly costing nothing.
    """
    pricing = pricing if pricing is not None else PRICING_PER_MILLION_TOKENS
    new_messages = messages_after[len(messages_before):]

    input_tokens = 0
    output_tokens = 0
    web_searches = 0
    cost = 0.0
    per_model: dict[str, list[int]] = {}
    tool_calls: list[ToolCallSummary] = []

    for message in new_messages:
        if isinstance(message, AIMessage):
            web_searches += _web_search_calls(message)
            if message.usage_metadata:
                name = _model_of(message, model)
                tokens_in = message.usage_metadata.get("input_tokens", 0)
                tokens_out = message.usage_metadata.get("output_tokens", 0)
                totals = per_model.setdefault(name, [0, 0])
                totals[0] += tokens_in
                totals[1] += tokens_out
                input_tokens += tokens_in
                output_tokens += tokens_out
        if isinstance(message, ToolMessage):
            tool_calls.append(ToolCallSummary(name=message.name or "", status=message.status))

    for name, (tokens_in, tokens_out) in per_model.items():
        cost += _estimate_cost(name, tokens_in, tokens_out, pricing)
    cost += web_searches * web_search_usd_per_call

    return TurnRecord(
        model=model,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        total_tokens=input_tokens + output_tokens,
        estimated_cost_usd=cost,
        latency_seconds=latency_seconds,
        tool_calls=tool_calls,
        models_used=sorted(per_model),
        unpriced_models=sorted(name for name in per_model if name not in pricing),
        web_searches=web_searches,
    )


class RunRecorder:
    """Appends one JSONL line per agent turn to observability/runs.jsonl,
    independent of LangSmith tracing."""

    def __init__(
        self,
        log_path: Path | None = None,
        pricing: dict[str, dict[str, float]] | None = None,
    ) -> None:
        self.log_path = log_path or RUNS_LOG_PATH
        self.pricing = pricing if pricing is not None else PRICING_PER_MILLION_TOKENS

    def record_turn(
        self,
        messages_before: list[BaseMessage],
        messages_after: list[BaseMessage],
        model: str,
        latency_seconds: float,
    ) -> TurnRecord:
        record = summarize_turn(
            messages_before, messages_after, model, latency_seconds, pricing=self.pricing
        )
        self._append(record)
        return record

    def _append(self, record: TurnRecord) -> None:
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        with self.log_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record.to_json_dict()) + "\n")
