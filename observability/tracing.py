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
}


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

    def to_json_dict(self) -> dict[str, Any]:
        return {
            "model": self.model,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "total_tokens": self.total_tokens,
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


def summarize_turn(
    messages_before: list[BaseMessage],
    messages_after: list[BaseMessage],
    model: str,
    latency_seconds: float,
    *,
    pricing: dict[str, dict[str, float]] | None = None,
) -> TurnRecord:
    """Summarize one agent turn's token usage, cost, and tool calls.

    messages_before/messages_after are the graph state's message list
    immediately before and immediately after one graph.invoke() call --
    capture messages_before BEFORE calling invoke(). Sums usage_metadata
    across every new AIMessage produced during that call, since a
    tool-calling turn involves more than one LLM call (see module
    docstring); reading only the last message would undercount.
    """
    pricing = pricing if pricing is not None else PRICING_PER_MILLION_TOKENS
    new_messages = messages_after[len(messages_before):]

    input_tokens = 0
    output_tokens = 0
    tool_calls: list[ToolCallSummary] = []

    for message in new_messages:
        if isinstance(message, AIMessage) and message.usage_metadata:
            input_tokens += message.usage_metadata.get("input_tokens", 0)
            output_tokens += message.usage_metadata.get("output_tokens", 0)
        if isinstance(message, ToolMessage):
            tool_calls.append(ToolCallSummary(name=message.name or "", status=message.status))

    return TurnRecord(
        model=model,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        total_tokens=input_tokens + output_tokens,
        estimated_cost_usd=_estimate_cost(model, input_tokens, output_tokens, pricing),
        latency_seconds=latency_seconds,
        tool_calls=tool_calls,
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
