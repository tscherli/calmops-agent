"""Ad-hoc, non-pytest scratch script: a terminal REPL for talking to the
real agent (real OpenAI calls, real writes to data/checkin_log.csv and
tools/measures.csv). Not part of the app -- app.py is the actual UI. This is
for walking a conversation through the stations by hand: which station a
sentence lands in, whether a rejection is counted, whether the log row is
written only after the user confirms.

Type a message and press Enter. Type "quit" or "exit" to stop.
"""
import time

from dotenv import load_dotenv

load_dotenv()

from langchain_core.messages import HumanMessage

from agent.common import message_text
from agent.graph import build_agent, next_turn_state
from agent.model import DEFAULT_MODEL
from observability.tracing import RunRecorder

graph = build_agent()
recorder = RunRecorder()
state = {"messages": []}

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

    messages_before = list(state.get("messages") or []) + [HumanMessage(content=user_input)]

    start = time.time()
    result = graph.invoke({**state, "messages": messages_before})
    latency = time.time() - start

    messages = result["messages"]
    record = recorder.record_turn(messages_before, messages, DEFAULT_MODEL, latency)
    state = next_turn_state(result)

    print(f"Agent: {message_text(messages[-1])}")
    print(f"  [station: {result.get('step') or '-'} | complete: {bool(result.get('complete'))}]\n")
    tool_summary = ", ".join(f"{t.name} ({t.status})" for t in record.tool_calls) or "none"
    print(
        f"  [tokens: {record.input_tokens} in / {record.output_tokens} out"
        f" | cost: ${record.estimated_cost_usd:.6f} | latency: {record.latency_seconds:.2f}s"
        f" | tools called: {tool_summary}]\n"
    )
