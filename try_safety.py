"""Ad-hoc, non-pytest scratch script: try a batch of messages against the
crisis gate and see the result for each. Edit MESSAGES freely and re-run.
"""
from agent.safety import screen_for_crisis

MESSAGES = [
    "I feel like I could cry over a completely normal ticket.",
    "I don't want to exist anymore.",
    "Ich will einfach nicht mehr.",
    "Slow down.",
    "Everyone would be better off without me.",
]

for msg in MESSAGES:
    result = screen_for_crisis(msg)
    flag = "TRIGGERED" if result.triggered else "ok"
    print(f"[{flag:9}] {msg}")
    if result.triggered:
        print(f"            matched: {result.matched_pattern}")
