"""Fixed, developer-authored mood categories backing the quick-select
buttons in app.py. Not agent-editable -- the LLM never sees or writes these.

Clicking a button calls log_mood directly with one of these entries, no LLM
involvement and no personal free text beyond the fixed label -- the option
for anyone who read the consent disclaimer and doesn't want to share detail
with an LLM at all.

That LLM-free promise is also why quick-select rows are never semantically
tagged: tools/classify.py is an LLM call, so running it here would quietly
break the "no AI, nothing shared" guarantee the UI makes right above these
buttons. Quick-select rows land with an empty `tags` column, exactly the way
`energy` is already left empty when nothing implied it.
"""
from __future__ import annotations

from typing import Literal, NotRequired, TypedDict


class MoodTemplate(TypedDict):
    mood: Literal["bad", "neutral", "good"]
    circumstances: str
    label: str
    # Optional and orthogonal to mood -- only set where activation level is
    # clearly implied by the template itself (see README "Why mood and energy
    # are separate" for the non-clinical, research-informed rationale).
    energy: NotRequired[Literal["exhausted", "calm", "wired"]]


MOOD_TEMPLATES: dict[str, MoodTemplate] = {
    "good_general": {
        "mood": "good",
        "circumstances": "Feeling good, no particular reason needed.",
        "label": "🙂 Feeling good",
    },
    "proud": {
        "mood": "good",
        "circumstances": "Proud of how I handled something today.",
        "label": "💪 Proud of myself",
    },
    "quiet_day": {
        "mood": "neutral",
        "circumstances": "Quiet, uneventful day.",
        "label": "😐 Quiet day",
        "energy": "calm",
    },
    "okay": {
        "mood": "neutral",
        "circumstances": "Just okay, nothing notable either way.",
        "label": "🤷 Just okay",
    },
    "stressed": {
        "mood": "bad",
        "circumstances": "Feeling stressed or overwhelmed.",
        "label": "😣 Stressed",
    },
    "tired": {
        "mood": "bad",
        "circumstances": "Feeling tired or drained.",
        "label": "😴 Tired",
        "energy": "exhausted",
    },
    "exhausted_from_work": {
        "mood": "bad",
        "circumstances": "Exhausted from work; feeling depleted, not just sleepy.",
        "label": "🔋 Exhausted (work)",
        "energy": "exhausted",
    },
    "frustrated": {
        "mood": "bad",
        "circumstances": "Feeling frustrated.",
        "label": "😤 Frustrated",
    },
    "anxious": {
        "mood": "bad",
        "circumstances": "Feeling anxious or on edge.",
        "label": "😰 Anxious",
        "energy": "wired",
    },
}
