"""Fixed, developer-authored mood categories backing the quick-select
buttons in app.py. Not agent-editable -- the LLM never sees or writes these.

Clicking a button calls log_mood directly with one of these entries, no LLM
involvement and no personal free text beyond the fixed label -- the option
for anyone who read the consent disclaimer and doesn't want to share detail
with an LLM at all.
"""
from __future__ import annotations

from typing import Literal, NotRequired, TypedDict


class MoodTemplate(TypedDict):
    valence: Literal["good", "neutral", "bad"]
    circumstances: str
    label: str
    # Optional and orthogonal to valence -- only set where fatigue/arousal is
    # clearly implied by the template itself (see README "Why energy_level
    # exists" for the non-clinical, burnout-research-informed rationale).
    energy_level: NotRequired[Literal["low", "medium", "high"]]


MOOD_TEMPLATES: dict[str, MoodTemplate] = {
    "good_general": {
        "valence": "good",
        "circumstances": "Feeling good, no particular reason needed.",
        "label": "🙂 Feeling good",
    },
    "proud": {
        "valence": "good",
        "circumstances": "Proud of how I handled something today.",
        "label": "💪 Proud of myself",
    },
    "quiet_day": {
        "valence": "neutral",
        "circumstances": "Quiet, uneventful day.",
        "label": "😐 Quiet day",
    },
    "okay": {
        "valence": "neutral",
        "circumstances": "Just okay, nothing notable either way.",
        "label": "🤷 Just okay",
    },
    "stressed": {
        "valence": "bad",
        "circumstances": "Feeling stressed or overwhelmed.",
        "label": "😣 Stressed",
    },
    "tired": {
        "valence": "bad",
        "circumstances": "Feeling tired or drained.",
        "label": "😴 Tired",
        "energy_level": "low",
    },
    "exhausted_from_work": {
        "valence": "bad",
        "circumstances": "Exhausted from work; feeling depleted, not just sleepy.",
        "label": "🔋 Exhausted (work)",
        "energy_level": "low",
    },
    "frustrated": {
        "valence": "bad",
        "circumstances": "Feeling frustrated.",
        "label": "😤 Frustrated",
    },
    "anxious": {
        "valence": "bad",
        "circumstances": "Feeling anxious or on edge.",
        "label": "😰 Anxious",
        "energy_level": "high",
    },
}
