"""Tests for evals/draft_cases.py -- the play-then-freeze drafting aid.

Both models are stubbed: no Gemini call, no OpenAI call, no network. What
matters here is what the drafter is shown, what it is refused, and that the
turns it produces are the user's side only.

The guarantee this module exists to protect is narrow and absolute: the
drafter writes turns and never writes an assertion. A generated assertion that
is wrong does not fail the suite -- it passes it, while checking something
nobody intended.
"""
import json

import pytest
from langchain_core.messages import AIMessage, HumanMessage

import evals.draft_cases as draft_module
from evals.draft_cases import ASSERTION_KEYS, _format_transcript, _reject_assertions, draft, load_brief

BRIEF = {
    "persona": "Wired after four hours of back-to-back calls, nothing eaten since breakfast.",
    "goal": "Wants something short that will help in the next ten minutes.",
}


class _StubAuthor:
    """The drafter. Says whatever it was told to, in order."""

    def __init__(self, *lines):
        self.lines = list(lines)
        self.prompts = []

    def invoke(self, prompt):
        self.prompts.append(prompt)
        return AIMessage(content=self.lines.pop(0) if self.lines else "ok")


class _StubGraph:
    """The agent. Replies once per turn and records what it was handed."""

    def __init__(self, reply="What's going on today?", complete_after=None):
        self.reply = reply
        self.complete_after = complete_after
        self.calls = []

    def invoke(self, payload, config=None):
        self.calls.append(payload)
        messages = list(payload["messages"]) + [AIMessage(content=self.reply)]
        done = self.complete_after is not None and len(self.calls) >= self.complete_after
        return {"messages": messages, "complete": done}


@pytest.fixture
def paths(tmp_path):
    return {"log_path": tmp_path / "log.csv", "measures_path": tmp_path / "measures.csv"}


def _run(author, graph, paths, **kwargs):
    return draft(BRIEF, author=author, agent_graph=graph, **paths, **kwargs)


# -- what comes back ---------------------------------------------------------


def test_only_the_users_turns_come_back_as_turns(paths):
    author = _StubAuthor("wired, back to back calls", "nothing yet", "the breathing one")
    result = _run(author, _StubGraph(), paths, turns=3)

    assert result["turns"] == ["wired, back to back calls", "nothing yet", "the breathing one"]
    assert all("What's going on" not in t for t in result["turns"])


def test_the_conversation_keeps_both_sides_for_reading(paths):
    """The turns go into the dataset; the conversation is what you read while
    writing the expect block, which is the half that makes this worth doing."""
    result = _run(_StubAuthor("wired", "nothing yet"), _StubGraph(), paths, turns=2)

    who = [entry["who"] for entry in result["conversation"]]
    assert who == ["user", "agent", "user", "agent"]


def test_the_drafter_sees_the_conversation_so_far(paths):
    """Turns written blind are written against an imagined agent. The whole
    point is that turn three answers what the agent actually asked in turn
    two."""
    author = _StubAuthor("wired", "nothing yet", "ok")
    _run(author, _StubGraph(reply="How long has that been going on?"), paths, turns=3)

    assert "(nothing yet -- write the opening message)" in author.prompts[0]
    assert "How long has that been going on?" in author.prompts[2]
    assert "You: wired" in author.prompts[2]


def test_the_drafter_is_shown_the_measure_list(paths):
    """Without it the drafter invents measures that do not exist -- which is
    how `short_walk` reached a hand test."""
    brief = {**BRIEF, "seed_measures": [{"measure_id": "box_breathing", "measure": "Breathe out longer."}]}
    author = _StubAuthor("wired")

    draft(brief, author=author, agent_graph=_StubGraph(), turns=1, **paths)

    assert "box_breathing" in author.prompts[0]


def test_a_finished_conversation_stops_the_pass(paths):
    """A logged check-in ends the conversation and resets the state. Carrying
    on past it drafts turns for a second, different conversation."""
    author = _StubAuthor("wired", "the breathing one", "should not be reached")
    result = _run(author, _StubGraph(complete_after=2), paths, turns=5)

    assert result["turns"] == ["wired", "the breathing one"]


def test_crisis_language_ends_the_pass(paths):
    """The gate ends the conversation in the app, so it ends it here too.
    Drafting past it would produce turns the harness could never replay."""
    author = _StubAuthor("rough week", "I want to kill myself.", "should not be reached")
    result = _run(author, _StubGraph(), paths, turns=5)

    assert result["turns"] == ["rough week", "I want to kill myself."]


# -- what the drafter is refused ---------------------------------------------


@pytest.mark.parametrize("bad", ['{"turns": ["hi"]}', '[{"id": "x"}]'])
def test_structured_output_is_refused(bad):
    with pytest.raises(SystemExit, match="structured data"):
        _reject_assertions(bad)


@pytest.mark.parametrize("key", sorted(ASSERTION_KEYS))
def test_every_assertion_key_is_refused(key):
    """Loud rather than quiet. A returned assertion that got silently dropped
    would look identical to one that was never produced, and nobody would know
    the prompt had stopped working."""
    with pytest.raises(SystemExit, match="assertion key"):
        _reject_assertions(f'wired today, "{key}": something')


def test_an_ordinary_message_passes_through():
    assert _reject_assertions('  wired, back to back calls  ') == "wired, back to back calls"


def test_a_brief_that_writes_the_test_is_refused(tmp_path):
    """A brief describes a person. The moment it carries an expect block, the
    assertions are being written before the run rather than after reading it,
    which is the habit this whole split exists to prevent."""
    path = tmp_path / "brief.json"
    path.write_text(json.dumps({**BRIEF, "expect": {"reached_write": True}}), encoding="utf-8")

    with pytest.raises(SystemExit, match="describes a person, not a test"):
        load_brief(path)


def test_a_brief_needs_a_person_and_a_goal(tmp_path):
    path = tmp_path / "brief.json"
    path.write_text(json.dumps({"persona": "wired"}), encoding="utf-8")

    with pytest.raises(SystemExit, match="non-empty 'goal'"):
        load_brief(path)


def test_a_good_brief_loads(tmp_path):
    path = tmp_path / "brief.json"
    path.write_text(json.dumps(BRIEF), encoding="utf-8")

    assert load_brief(path)["persona"] == BRIEF["persona"]


# -- the prompt itself -------------------------------------------------------


@pytest.mark.parametrize(
    "instruction",
    [
        "under fifteen words",          # real messages are short
        "You are allowed to be unhelpful",
        "changing your mind",
        "typos are normal",
        "never narrate",
        "Never mention that this is a test",
    ],
)
def test_the_prompt_demands_the_traits_a_model_will_not_volunteer(instruction):
    """Models write articulate, cooperative, well-organised people. Every one
    of these has to be asked for by name or the suite only ever tests the easy
    user."""
    assert instruction in draft_module.PERSONA_PROMPT


def test_the_transcript_reads_as_a_conversation():
    rendered = _format_transcript([HumanMessage(content="wired"), AIMessage(content="Since when?")])

    assert rendered == "You: wired\nThe app: Since when?"


# -- the drafter sees what a user sees, and nothing else ----------------------


def test_tool_results_are_never_shown_to_the_drafter():
    """The first real pass handed the drafter all 31 rows of the measure list
    rendered as JSON, under "The app:". That wastes most of the prompt, and it
    contradicts the short list the drafter is given on purpose -- a person
    does not see their whole list quoted back at them mid-conversation."""
    from langchain_core.messages import ToolMessage

    messages = [
        HumanMessage(content="wired"),
        AIMessage(content="", tool_calls=[{"name": "find_measures", "args": {}, "id": "c1"}]),
        ToolMessage(content='[{"measure_id": "box_breathing"}]', name="find_measures", tool_call_id="c1"),
        AIMessage(content="Two things from your list."),
    ]

    rendered = _format_transcript(messages)

    assert "box_breathing" not in rendered
    assert "find_measures" not in rendered
    assert rendered == "You: wired\nThe app: Two things from your list."


def test_an_empty_reply_is_not_rendered_as_a_blank_turn():
    """The tool-calling AIMessage has no text. Rendered anyway it becomes
    "The app: " and reads as the agent having said nothing back."""
    messages = [
        HumanMessage(content="wired"),
        AIMessage(content="", tool_calls=[{"name": "find_measures", "args": {}, "id": "c1"}]),
    ]

    assert _format_transcript(messages) == "You: wired"


def test_what_the_tools_did_still_comes_back_for_the_author(paths):
    """Hidden from the drafter, kept for whoever writes the expect block --
    which tools ran is half of what makes that block honest."""

    class _ToolingGraph(_StubGraph):
        def invoke(self, payload, config=None):
            self.calls.append(payload)
            messages = list(payload["messages"]) + [
                AIMessage(content="", tool_calls=[{"name": "find_measures", "args": {}, "id": "c1"}]),
                AIMessage(content="Two things from your list."),
            ]
            return {"messages": messages, "complete": False}

    result = _run(_StubAuthor("wired"), _ToolingGraph(), paths, turns=1)

    assert [t["name"] for t in result["tools"]] == ["find_measures"]
    assert [e["who"] for e in result["conversation"]] == ["user", "agent"]
