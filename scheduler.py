"""Background script: sleeps until the next scheduled event, then opens the
chat window -- either a normal check-in or a proactive weekly pattern digest.
Independent of app.py -- either can be used alone.

Stage-14 redesign: the loop is config-driven (scheduler_config.py /
scheduler_config.json) instead of a bare random sleep, and gained the digest
path. The original randomized-interval behavior is retained as a mode, not
deleted -- `--interval-min`/`--interval-max` still work and still override
whatever the config says.

The digest is where this app stops being only a logger: analyze_stress_patterns
does a plain, deterministic group-by over the tags column (no LLM, no code
execution), and exactly one LLM call turns that summary dict into a sentence.
The aggregation the user sees is therefore always real, even if the phrasing
around it is generated.

Entry point -- must load .env itself (same reason as app.py).
"""
from __future__ import annotations

import argparse
import time as time_module
import tkinter as tk
from collections.abc import Sequence
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

from langchain_core.messages import HumanMessage, SystemMessage

from agent.graph import tool_event
from scheduler_config import Fire, SchedulerConfig, compute_next_fire, load_config
from tools.log_query import analyze_stress_patterns

OPENER_MESSAGE = "Hey, how are you doing?"

DIGEST_PROMPT = """You write a single short, warm opening message for a personal work-stress \
check-in app. You are given a factual summary of the user's own log entries from the past week.

Rules:
- 1-3 sentences, conversational, no bullet points, no headings.
- Only state what the summary actually says. Do not invent causes, diagnoses, advice, or \
numbers that aren't there.
- Mention the most frequent tag(s) and, if it's notable, whether rough days skewed exhausted \
or wired.
- End by inviting them to check in now. Warm, not clinical, not alarmed."""


def build_digest_opener(summary: dict, *, model=None) -> str:
    """Turn the aggregation into one conversational line. Falls back to a
    plain, factual sentence if the LLM call fails -- the digest should still
    open a window rather than silently not firing."""
    if not summary.get("enough_data"):
        return (
            "Hey -- I don't have enough logged this past week to spot any patterns yet. "
            "How are you doing right now?"
        )

    try:
        chat_model = model if model is not None else _default_model()
        reply = chat_model.invoke(
            [
                SystemMessage(content=DIGEST_PROMPT),
                HumanMessage(content=_render_summary(summary)),
            ]
        )
        text = reply.content if isinstance(reply.content, str) else str(reply.content)
        if text.strip():
            return text.strip()
    except Exception:
        pass
    return _fallback_opener(summary)


def _render_summary(summary: dict) -> str:
    top_tags = ", ".join(f"{tag} ({count}x)" for tag, count in summary.get("top_tags", [])[:5])
    return (
        f"entries in the last {summary.get('window_days')} days: {summary.get('entry_count')}\n"
        f"most frequent tags: {top_tags or 'none recorded'}\n"
        f"mood counts: {summary.get('mood_counts')}\n"
        f"energy counts: {summary.get('energy_counts')}\n"
        f"things they tried: {summary.get('measures_taken')}"
    )


def _fallback_opener(summary: dict) -> str:
    top_tags = summary.get("top_tags") or []
    if not top_tags:
        return "Hey -- here's your weekly check-in. How are things going?"
    tag, count = top_tags[0]
    return (
        f"Hey -- looking back at this week, '{tag}' came up {count} times in your log. "
        "How are things going right now?"
    )


def _default_model():
    from langchain_openai import ChatOpenAI

    from agent.graph import DEFAULT_MODEL

    return ChatOpenAI(model=DEFAULT_MODEL)


def open_window(opener_message: str, opener_events: Sequence[dict] = ()) -> None:
    from app import ChatApp

    root = tk.Tk()
    ChatApp(root, opener_message=opener_message, opener_events=opener_events)
    root.mainloop()


def opener_for(fire: Fire, config: SchedulerConfig) -> tuple[str, list[dict]]:
    """The opener text, plus any tool receipts that produced it.

    The digest is the one place the agent makes *quantitative* claims about
    the user's history ("you logged deadline_pressure 4 times"), phrased by
    an LLM from an aggregation it did not compute. So it needs the same
    receipt as any other tool call -- otherwise the one window where numbers
    are asserted is the one window with nothing to check them against.
    """
    if fire.kind == "digest":
        summary = analyze_stress_patterns(window_days=config.digest_window_days)
        detail = (
            f"{summary.get('entry_count', 0)} entries over {summary.get('window_days')} days"
            if summary.get("enough_data")
            else f"not enough data ({summary.get('entry_count', 0)} entries)"
        )
        return build_digest_opener(summary), [
            tool_event("analyze_stress_patterns", "ok", detail)
        ]
    return OPENER_MESSAGE, []


def main() -> None:
    parser = argparse.ArgumentParser(description="Scheduled check-ins and weekly digest for Job Load Agent")
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="Path to scheduler_config.json (defaults to the one next to this script).",
    )
    parser.add_argument(
        "--interval-min",
        type=float,
        default=None,
        help="Dev override: minimum minutes between nudges. Forces random_interval mode.",
    )
    parser.add_argument(
        "--interval-max",
        type=float,
        default=None,
        help="Dev override: maximum minutes between nudges. Forces random_interval mode.",
    )
    args = parser.parse_args()

    config = load_config(args.config)
    if args.interval_min is not None or args.interval_max is not None:
        config.checkin_mode = "random_interval"
        if args.interval_min is not None:
            config.interval_min_minutes = args.interval_min
        if args.interval_max is not None:
            config.interval_max_minutes = args.interval_max

    while True:
        now = datetime.now()
        fire = compute_next_fire(now, config)
        wait_seconds = max(0.0, (fire.when - now).total_seconds())
        time_module.sleep(wait_seconds)
        opener_message, opener_events = opener_for(fire, config)
        open_window(opener_message, opener_events)


if __name__ == "__main__":
    main()
