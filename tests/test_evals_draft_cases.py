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


# -- opening mid-conversation ------------------------------------------------


def test_a_brief_may_open_from_a_payload_set_state(paths):
    """Most Step 6 cases start with energy and goal already set, the way a
    client sets them from its buttons. A pass that always started cold would
    draft turns answering questions the case never asks."""
    graph = _StubGraph()
    brief = {**BRIEF, "state": {"energy": "exhausted", "goal": "calm down now"}}

    draft(brief, author=_StubAuthor("need something quick"), agent_graph=graph, turns=1, **paths)

    assert graph.calls[0]["energy"] == "exhausted"
    assert graph.calls[0]["goal"] == "calm down now"


def test_the_opening_state_is_not_re_sent_on_later_turns(paths):
    """A case sets the payload on its first turn only; everything after that
    is carried by next_turn_state. Re-applying the brief every turn would hide
    a graph that had dropped the field, and the drafted turns would make sense
    against a state the app never produces."""
    graph = _StubGraph()
    brief = {**BRIEF, "state": {"energy": "exhausted"}}

    draft(brief, author=_StubAuthor("one", "two"), agent_graph=graph, turns=2, **paths)

    assert graph.calls[0]["energy"] == "exhausted"
    assert graph.calls[1].get("energy") != "exhausted"


def test_the_drafter_is_told_what_the_app_already_knows(paths):
    """Otherwise it opens by announcing an energy the app already has, and the
    frozen turns read as a non-sequitur to whoever finds the case later."""
    author = _StubAuthor("need something quick")
    brief = {**BRIEF, "state": {"energy": "exhausted", "situation": "slept badly"}}

    draft(brief, author=author, agent_graph=_StubGraph(), turns=1, **paths)

    assert "exhausted" in author.prompts[0]
    assert "slept badly" in author.prompts[0]


def test_a_cold_brief_says_so_rather_than_leaving_the_block_empty(paths):
    author = _StubAuthor("hi")

    draft(BRIEF, author=author, agent_graph=_StubGraph(), turns=1, **paths)

    assert "Nothing yet. This is the first thing you have said today." in author.prompts[0]


def test_a_brief_cannot_open_from_a_field_no_client_carries(tmp_path):
    """`mood` was a real field once. A brief setting one the graph does not
    carry would play from a state the case can never reproduce, and the turns
    would be drafted against a conversation that never happened."""
    path = tmp_path / "brief.json"
    path.write_text(json.dumps({**BRIEF, "state": {"mood": "grim"}}), encoding="utf-8")

    with pytest.raises(SystemExit, match="mood"):
        load_brief(path)


def test_a_brief_may_open_from_any_field_a_client_carries(tmp_path):
    path = tmp_path / "brief.json"
    path.write_text(
        json.dumps({**BRIEF, "state": {"energy": "wired", "step": "follow_up"}}), encoding="utf-8"
    )

    assert load_brief(path)["state"]["step"] == "follow_up"


# -- the approval card, which the person sees and the transcript does not ----


class _CardGraph(_StubGraph):
    """Pauses on a card the first time it is called, like a real write."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.resumed = []

    def invoke(self, payload, config=None):
        from langgraph.types import Command

        if isinstance(payload, Command):
            self.resumed.append(payload.resume)
            return {"messages": [AIMessage(content="Done.")], "complete": False}
        self.calls.append(payload)
        return {
            "messages": list(payload["messages"]),
            "__interrupt__": [
                type("I", (), {"value": {"action_requests": [
                    {"id": "call_1", "tool": "log_checkin", "description": "Log: slept badly."}
                ]}})()
            ],
        }


def test_the_drafter_is_shown_the_card_the_person_saw(paths):
    """A card stops the conversation and asks a yes/no question. A drafter
    that cannot see it writes the next turn as though nothing was proposed."""
    author = _StubAuthor("save that", "ok")

    draft(BRIEF, author=author, agent_graph=_CardGraph(), turns=2, **paths)

    assert "Log: slept badly." in author.prompts[1]
    assert "You confirmed it." in author.prompts[1]


def test_a_brief_may_decline_the_card(paths):
    """A case about what the agent says AFTER a rejected card cannot be
    drafted from a pass that approved it: the turns would answer a reply the
    case never produces."""
    graph = _CardGraph()
    author = _StubAuthor("save that", "did you save it?")

    draft({**BRIEF, "approval": "reject"}, author=author, agent_graph=graph, turns=2, **paths)

    assert graph.resumed[0]["decisions"][0]["type"] == "reject"
    assert "You declined it." in author.prompts[1]


def test_a_misspelled_approval_is_refused(tmp_path):
    """An unrecognised value falls through to approve, which is the one
    outcome a brief about a declined card must not silently get."""
    path = tmp_path / "brief.json"
    path.write_text(json.dumps({**BRIEF, "approval": "rejected"}), encoding="utf-8")

    with pytest.raises(SystemExit, match="rejected"):
        load_brief(path)


def test_a_brief_may_play_against_a_write_that_really_fails(tmp_path, monkeypatch):
    """Drafted over a working log, every turn after the write answers a
    success the case will never produce."""
    seen = {}

    def _fake_unwritable():
        seen["used"] = True
        return tmp_path / "blocked" / "checkin_log.csv"

    monkeypatch.setattr(draft_module, "_unwritable_log_path", _fake_unwritable)

    draft(
        {**BRIEF, "simulate_write_failure": True},
        author=_StubAuthor("log it"),
        agent_graph=_StubGraph(),
        turns=1,
        measures_path=tmp_path / "measures.csv",
    )

    assert seen.get("used")


def test_a_brief_may_seed_the_log(tmp_path):
    """A rating case needs a check-in already waiting to be rated."""
    log_path = tmp_path / "log.csv"

    draft(
        {**BRIEF, "seed_entries": [{"energy": "wired", "measure_id": "deep_breaths"}]},
        author=_StubAuthor("that one helped"),
        agent_graph=_StubGraph(),
        turns=1,
        log_path=log_path,
        measures_path=tmp_path / "measures.csv",
    )

    assert "deep_breaths" in log_path.read_text(encoding="utf-8")


# -- turns that will not survive being frozen --------------------------------


def test_a_turn_pointing_at_the_agents_own_suggestion_is_flagged():
    """The agent suggests box breathing, the drafter answers "ill try the
    breathing one", and the turn is perfectly sensible -- in this pass. On a
    replay the agent suggests something else and the turn refers to nothing.

    Three cases in the suite had exactly this and none went red: the expect
    block cannot see whether a conversation coheres, so they passed every run
    while making no sense. Found by a person reading a transcript."""
    pairs = [
        ("user", "something quick please"),
        ("agent", "Two options: box breathing, or palming eye rest."),
        ("user", "ill try the eye palming thing"),
    ]

    flagged = draft_module.turns_that_depend_on_the_reply(pairs)

    assert len(flagged) == 1
    assert flagged[0]["text"] == "ill try the eye palming thing"
    assert "palming" in flagged[0]["echoes"]


def test_pointing_at_a_position_is_not_flagged():
    """"the first one" survives any suggestion, which is the rewrite."""
    pairs = [
        ("user", "something quick please"),
        ("agent", "Two options: box breathing, or palming eye rest."),
        ("user", "ill try the first one"),
    ]

    assert draft_module.turns_that_depend_on_the_reply(pairs) == []


def test_proposing_something_outright_is_not_flagged():
    pairs = [
        ("user", "something quick please"),
        ("agent", "Two options: box breathing, or palming eye rest."),
        ("user", "can i add a wind-down checklist for tonight"),
    ]

    assert draft_module.turns_that_depend_on_the_reply(pairs) == []


def test_a_word_the_agent_never_said_is_not_flagged():
    """The check is about echoing this pass, not about vocabulary."""
    pairs = [
        ("user", "something quick please"),
        ("agent", "Two options: box breathing, or a glass of water."),
        ("user", "ill try the swimming thing"),
    ]

    assert draft_module.turns_that_depend_on_the_reply(pairs) == []


def test_the_drafter_reports_fragile_turns_with_its_output(paths):
    author = _StubAuthor("something quick", "ill try the breathing one")
    graph = _StubGraph(reply="Two options: box breathing, or a glass of water.")

    result = _run(author, graph, paths, turns=2)

    assert [entry["text"] for entry in result["depends_on_the_reply"]] == [
        "ill try the breathing one"
    ]


# -- a case the dataset already holds ----------------------------------------


def test_a_draft_that_repeats_an_existing_case_is_named(paths, monkeypatch):
    """The briefs are hand-written and a model writes the turns, so two briefs
    aimed at neighbouring behaviours can come back as the same conversation.
    Catching it while the turns are on screen costs nothing; catching it later
    means a labeller reading the same conversation twice and wondering what is
    broken."""
    existing = [{"id": "simple_existing_01", "turns": ["exhausted after a long week, need something quick"]}]
    monkeypatch.setattr(
        draft_module, "cases_resembling",
        lambda turns, cases=None: __import__("evals.create_dataset", fromlist=["x"]).cases_resembling(
            turns, existing
        ),
    )
    author = _StubAuthor("exhausted after a long week, i need something quick")

    result = _run(author, _StubGraph(), paths, turns=1)

    assert [case_id for case_id, _ in result["resembles"]] == ["simple_existing_01"]


def test_a_draft_about_something_else_is_not_named(paths, monkeypatch):
    existing = [{"id": "simple_existing_01", "turns": ["exhausted after a long week, need something quick"]}]
    monkeypatch.setattr(
        draft_module, "cases_resembling",
        lambda turns, cases=None: __import__("evals.create_dataset", fromlist=["x"]).cases_resembling(
            turns, existing
        ),
    )
    author = _StubAuthor("do you know any jokes")

    assert _run(author, _StubGraph(), paths, turns=1)["resembles"] == []


def test_a_missing_dataset_does_not_end_a_draft(paths, monkeypatch):
    """A drafting pass costs model calls. Losing one because the comparison
    could not be made would be a poor trade for a warning."""
    def _boom(turns, cases=None):
        raise FileNotFoundError("no dataset here")

    monkeypatch.setattr(draft_module, "cases_resembling", _boom)

    assert _run(_StubAuthor("hello"), _StubGraph(), paths, turns=1)["resembles"] == []
