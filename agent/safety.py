"""Deterministic, pre-LLM crisis-language gate.

Plain Python, no model calls. Biased toward recall: a false positive
(an innocuous message triggering crisis resources) is an acceptable
annoyance. A false negative is not acceptable at all.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

CRISIS_RESOURCES = (
    "It sounds like things might be really hard right now. I'm not able to help "
    "with this myself, but please reach out to people who can:\n\n"
    "- Telefonseelsorge (Germany, free, 24/7): 0800 111 0 111 or 0800 111 0 222\n"
    "- Emergency services: 112\n\n"
    "If you're in immediate danger, please contact emergency services now."
)

# Word/phrase fragments, not full sentences, so they match regardless of
# surrounding phrasing. Case-insensitive; \b keeps "die" from matching "died"
# in an unrelated sense is NOT guaranteed here on purpose -- recall over precision.
_CRISIS_PATTERNS: list[str] = [
    # English -- direct
    r"\bsuicid\w*",
    r"\bkill(ing)?\s+(myself|me)\b",
    r"\bend(ing)?\s+(my\s+life|it\s+all)\b",
    r"\b(want|wish)\s+(to\s+be\s+dead|i\s+w(as|ere)\s+dead|to\s+die)\b",
    r"\bworth\s+living\b",
    r"\bno\s+reason\s+to\s+(live|go\s+on)\b",
    r"\bself[\s-]?harm\w*",
    r"\bhurt\w*\s+myself\b",
    r"\b(cutting|cut)\s+myself\b",
    r"\boverdos\w*",
    r"\bbetter\s+off\s+(without\s+me|dead)\b",
    # English -- indirect / softer signals
    r"\bdon'?t\s+want\s+to\s+(be\s+here|exist)\s+anymore\b",
    r"\bcan'?t\s+see\s+(a\s+)?(future|point)\s+in\s+anything\b",
    r"\bwhat'?s\s+the\s+point\s+of\s+(anything|going\s+on|living)\b",
    r"\beveryone\s+would\s+be\s+better\s+off\s+without\s+me\b",
    # German -- direct
    r"\bsuizid\w*",
    r"\bselbstmord\w*",
    r"\bmich\s+(\w+\s+){0,2}(um)?bringen\b",
    r"\bmich\s+t[öo]ten\b",
    r"\b(nicht\s+mehr\s+leben\s+(will|m[öo]chte)|(will|m[öo]chte)\s+nicht\s+mehr\s+leben)\b",
    r"\bmich\s+(selbst\s+)?(zu\s+)?verletzen\b",
    r"\bmir\s+selbst\s+weh\s?tun\b",
    # German -- indirect
    r"\bkeinen\s+sinn\s+mehr\b",
    r"\bbesser\s+dran\s+ohne\s+mich\b",
    r"\bwill\s+(\w+\s+)?nicht\s+mehr\b",
]

_COMPILED = [re.compile(p, re.IGNORECASE | re.UNICODE) for p in _CRISIS_PATTERNS]


@dataclass(frozen=True)
class CrisisCheckResult:
    triggered: bool
    matched_pattern: str | None = None


def screen_for_crisis(text: str) -> CrisisCheckResult:
    """Check user input against the deterministic crisis-keyword gate.

    Runs before any LLM call. Returns which pattern matched (if any) so
    callers can log/test the specific trigger, not just a bare bool.
    """
    if not text:
        return CrisisCheckResult(triggered=False)

    for pattern, compiled in zip(_CRISIS_PATTERNS, _COMPILED):
        if compiled.search(text):
            return CrisisCheckResult(triggered=True, matched_pattern=pattern)

    return CrisisCheckResult(triggered=False)
