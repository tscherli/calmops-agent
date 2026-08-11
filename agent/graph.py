"""Agent graph: binds the log_mood tool to GPT via langchain.agents.create_agent.

Only reached after agent/safety.py's crisis gate has already cleared the
message -- this module has no crisis-handling logic of its own.

model/tools are injectable, mirroring tools/csv_log.py's path
dependency-injection pattern -- lets tests stub both without hitting OpenAI
or the local mood log.
"""
from __future__ import annotations

import os
from collections.abc import Sequence

from langchain.agents import create_agent
from langchain_core.language_models import BaseChatModel
from langchain_core.tools import BaseTool
from langchain_openai import ChatOpenAI

from agent.prompts import SYSTEM_PROMPT
from tools.csv_log import log_mood

DEFAULT_MODEL = os.environ.get("OPENAI_MODEL", "gpt-5-nano")


def build_agent(
    model: str | BaseChatModel | None = None,
    tools: Sequence[BaseTool] | None = None,
):
    """Build the mood check-in agent graph.

    model/tools default to a real ChatOpenAI client and the real log_mood
    tool. Inject a fake model and/or a stub tool to test graph wiring
    without touching OpenAI or the local mood log.
    """
    if model is None:
        model = ChatOpenAI(model=DEFAULT_MODEL)
    if tools is None:
        tools = [log_mood]
    return create_agent(model, tools=tools, system_prompt=SYSTEM_PROMPT)
