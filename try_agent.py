"""Ad-hoc, non-pytest scratch script: a terminal REPL for talking to the
real agent (real OpenAI calls, real log_mood writes to data/mood_log.csv).
Not part of the app -- app.py (the actual Tkinter UI) is stage 8, not yet
built. This is for manually exercising the agent's judgment in the
meantime: light-vs-deep structuring, clarifying questions, honesty about
tool status, etc.

Type a message and press Enter. Type "quit" or "exit" to stop.
"""
import time

from dotenv import load_dotenv

load_dotenv()

from langchain_core.messages import HumanMessage

from agent.graph import DEFAULT_MODEL, build_agent
from observability.tracing import RunRecorder

graph = build_agent()
recorder = RunRecorder()
messages = []

print(f"Talking to the agent (model: {DEFAULT_MODEL}). Type 'quit' to stop.\n")

while True:
    try:
        user_input = input("You: ").strip()
    except (EOFError, KeyboardInterrupt):
        print()
        break

    if user_input.lower() in {"quit", "exit"}:
        break
    if not user_input:
        continue

    messages_before = messages + [HumanMessage(content=user_input)]

    start = time.time()
    result = graph.invoke({"messages": messages_before})
    latency = time.time() - start

    messages = result["messages"]
    record = recorder.record_turn(messages_before, messages, DEFAULT_MODEL, latency)

    print(f"Agent: {messages[-1].content}\n")
    tool_summary = ", ".join(f"{t.name} ({t.status})" for t in record.tool_calls) or "none"
    print(
        f"  [tokens: {record.input_tokens} in / {record.output_tokens} out"
        f" | cost: ${record.estimated_cost_usd:.6f} | latency: {record.latency_seconds:.2f}s"
        f" | tools called: {tool_summary}]\n"
    )
