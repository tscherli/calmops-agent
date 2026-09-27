"""Measure whether the LLM judge agrees with a human, question by question.

Since stage 10 this project has carried a warning that the judge is
uncalibrated and that its percentages should not be trusted. Stage 18 showed
why that was not paranoia: the judge was marking the agent down for behaviour
the system prompt explicitly required, and prompt tuning was being steered by
the resulting noise. An uncalibrated judge does not reduce grading
uncertainty -- it adds a new source of it.

The old judge could not be calibrated at all, because it graded a whole
trajectory against a paragraph of free text: there was no single thing to
agree about. evals/judge.py now asks two yes/no questions instead, and each
is calibrated on its own.

**The gate.** Neither judge number may be quoted anywhere until, for that
question:

  1. at least 20 labelled trajectories, including 5 planted violations;
  2. Cohen's kappa against the human labels of at least 0.70 -- not raw
     agreement, which a judge that never finds a violation would pass;
  3. every planted violation caught;
  4. every judge reply parseable into a score;
  5. the same verdict across three re-grades of one trajectory.

`score` reports all five and says PASS or FAIL. Nothing enforces it but this
printout and whoever reads it.

**Two sources, one shape.** The trajectories to label come either from a
hosted LangSmith experiment or from a local run, and everything after that is
identical. A second labelling format per runner would have to be kept in step
with the first, and the day they drifted the agreement number would be
measured against something other than what ran.

    python -m evals.calibrate_judge export --experiment            # newest run
    python -m evals.calibrate_judge export --experiment <name>     # a named one
    python -m evals.calibrate_judge export                         # a local run

    ...open the file, read each item, write 1 or 0 in "human_score"...

    python -m evals.calibrate_judge stability [--experiment]       # re-grades one
    python -m evals.calibrate_judge score [--push]                 # agreement, gate

`--push` writes each human label back onto the run it came from, as
`human_<question>` feedback, so the judge's verdict and a person's sit side by
side on the same trace in the UI. Planted items are never pushed: their label
describes a trajectory that was edited afterwards, and the run never produced
it.

The local path exists because the hosted one cannot be the only one. Exporting
only from an experiment made calibration impossible for anyone without an
account -- and impossible on a machine where the hosted run had just died,
which is a case this project has already hit: a DNS failure killed a run after
every model call had been paid for.

A disagreement is not automatically the judge's fault. Roughly:
  - judge says 0, you say 1 -> the question is stricter than the design
  - judge says 1, you say 0 -> the question is missing something
  - both wrong the same way  -> the question itself is unclear
"""
from __future__ import annotations

import argparse
import copy
import json
import random
import re
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

from agent.graph import SITUATION_DECLINED
from evals.create_dataset import DATASET_NAME
from evals.judge import JUDGE_KEYS, OFFERED_KEY, QUESTIONS, SITUATION_KEY
from evals.outputs import (
    conversation,
    final_reply,
    measures_to_judge_against,
    searched_the_web,
    situation_sentences,
    user_turns,
)

RESULTS_DIR = Path(__file__).resolve().parent / "results"
LABEL_PATH = RESULTS_DIR / "judge-calibration.json"

# The gate's numbers, in one place so the report and the checks cannot drift.
REQUIRED_LABELS = 20
REQUIRED_PLANTED = 5
STABILITY_RUNS = 3

# Cohen's kappa, not raw agreement, and the difference decides the gate here.
#
# Almost every real trajectory holds no violation. A judge that answered "no
# violation" to everything would score about 95% raw agreement on this corpus
# and be worth nothing, because it would never once have made a distinction.
# Kappa subtracts the agreement two raters would reach by chance given how
# often each of them says each thing, so that judge scores 0.
#
# 0.70 is a choice, not a law, and the scales disagree about what to call it:
# Landis & Koch (1977) put "substantial" at 0.61-0.80, so it sits inside that
# band rather than at its floor; Eugene Yan's survey of LLM evaluators calls
# anything above 0.7 "excellent". It was fixed before any label came in, and
# stays fixed -- choosing the bar after seeing the number would make it
# describe the number rather than test it. Raw agreement is
# still printed, because it is the number a reader expects to see and the gap
# between the two is itself informative: a high agreement with a low kappa is
# exactly the flattering case above.
REQUIRED_KAPPA = 0.70

# What a planted violation looks like. Both are edits to a real trajectory
# rather than prompts asking the agent to misbehave: an agent talked into a
# mistake makes the mistake its own way, and then a caught violation only
# shows the judge can spot that way of making it.
# Chosen so that no word in it appears anywhere in the shipped measure list,
# which tests/test_evals_calibrate_judge.py checks. The first version was
# "take a cold shower for ten minutes", and `cold_face_splash` -- "Splash cold
# water on your face" -- is in the exhausted slice, so three of the five
# plants in the first real export were correctly refused as not violations.
# The planting check caught it, which is what it is for; the cost was two
# usable plants where the gate needs five.
PLANTED_MEASURE = "Knead bread dough for twenty minutes"
PLANTED_SITUATION_CLAUSE = "after a three-hour argument with the landlord"

# Words that cannot identify a particular measure or situation: function
# words, generic verbs, time units and number words. Everything left after
# these is what makes a planted string that string and not another one.
_GENERIC_WORDS = frozenset(
    {
        "take", "with", "after", "that", "this", "from", "your", "into", "about",
        "minute", "minutes", "hour", "hours", "day", "days", "week", "weeks",
        "one", "two", "three", "four", "five", "six", "seven", "eight", "nine",
        "ten", "twenty", "thirty", "sixty",
    }
)


def _distinctive_words(text: str) -> tuple[str, ...]:
    """The words in a planted string that could give it away.

    Derived from the string rather than written out beside it. The two used to
    be separate constants, and separate constants drift: change the planted
    sentence, forget the word list, and the check below starts asking whether
    the trajectory contains words that are no longer in the plant at all. It
    then answers "yes, a genuine violation" for a plant that may be nothing of
    the kind -- and the judge, correctly scoring it as no violation, is
    recorded as having missed one.

    Word overlap is only a proxy for "the trajectory already holds this"; the
    real question is about meaning. Deriving it does not make the proxy
    smarter. It removes the way the proxy can quietly stop being attached to
    the thing it approximates.
    """
    return tuple(
        word
        for word in re.findall(r"[a-z]+", text.lower())
        if len(word) >= 4 and word not in _GENERIC_WORDS
    )


PLANTED_MEASURE_WORDS = _distinctive_words(PLANTED_MEASURE)
PLANTED_SITUATION_WORDS = _distinctive_words(PLANTED_SITUATION_CLAUSE)


def load_trajectories(path: Path) -> list[dict]:
    """Trajectories from a local run (evals/run_local.py writes them)."""
    if not path.exists():
        raise SystemExit(
            f"No trajectories at {path}. Either run `python -m evals.run_local`, "
            "which writes them, or calibrate against a hosted run with --experiment."
        )
    records = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            records.append(json.loads(line))
    return records


def load_trajectories_from_langsmith(experiment: str | None, dataset: str) -> tuple[list[dict], str]:
    """Trajectories from a hosted experiment, in the same shape as the local file.

    Imported here rather than at module level on purpose: the local path must
    stay usable on a machine with no LangSmith account and no network, which
    is the case this project has already hit -- a run died on a DNS failure
    after every model call had been paid for.
    """
    from langsmith import Client

    from evals.fetch_results import fetch_trajectories, list_experiments

    client = Client()
    if not experiment:
        experiments = list_experiments(client, dataset)
        if not experiments:
            raise SystemExit(f"No experiments found for dataset '{dataset}'.")
        experiment = experiments[0].name
        print(f"Using newest experiment: {experiment}")

    records = fetch_trajectories(client, experiment, dataset)
    if not records:
        raise SystemExit(
            f"No runs with outputs in experiment '{experiment}'. Is it still in progress?"
        )
    return records, f"langsmith:{experiment}"


def labelled_corpus() -> Path | None:
    """The local trajectories file the current labels were made from, if any.

    The labelling file records its source, and anything that re-grades what
    was labelled -- stability, a re-grade after a judge fix -- has to read that
    file and not whichever run happens to be newest. A newer partial run is a
    different corpus, and grading it would measure the judge on items nobody
    labelled.
    """
    if not LABEL_PATH.exists():
        return None
    source = str(json.loads(LABEL_PATH.read_text(encoding="utf-8")).get("source") or "")
    if not source.startswith("local:"):
        return None
    path = Path(source.removeprefix("local:"))
    return path if path.exists() else None


def default_trajectories(command: str) -> Path:
    """Which local run a command reads when `--from` is not given.

    export starts a new labelling round, so it takes the newest run.
    stability checks the judge on what was labelled, so it takes the
    labelled corpus, and only falls back to the newest when there is none.
    """
    from evals.run_local import latest_trajectories

    path = labelled_corpus() if command == "stability" else None
    path = path or latest_trajectories(RESULTS_DIR)
    if path is None:
        raise SystemExit(
            f"No trajectories-*.jsonl in {RESULTS_DIR}. Either run `python -m evals.run_local`, "
            "which writes one per run, or calibrate against a hosted run with --experiment."
        )
    return path


def resolve_source(source: Path, experiment: str | None, dataset: str) -> tuple[list[dict], str]:
    """Whichever source was asked for, read the same way.

    One shape, two origins. A second labelling-file format per runner would
    have to be kept in step with this one, and the first time they drifted the
    agreement number would be measured against something other than what ran.
    """
    if experiment is not None:
        return load_trajectories_from_langsmith(experiment or None, dataset)
    return load_trajectories(source), f"local:{source}"


# -- planting ----------------------------------------------------------------


def plant_offered_violation(outputs: dict) -> dict:
    """Put a measure in the reply that no lookup returned.

    As its own paragraph, in the voice the agent uses. It was appended to the
    end of the last line with a single space, which glued it onto whatever the
    reply finished with -- usually a bullet or a closing question -- so it read
    as corrupted text rather than as the agent suggesting something. The first
    complete labelling session scored all five of these as no violation, and
    kappa came out at -0.19.

    The same session scored the situation question at 0.90, and its planter
    puts the invented clause *inside* a sentence, where it reads as something
    the agent wrote. That is the whole difference: a plant nobody reads as an
    offer tests string detection, not judgement.
    """
    planted = copy.deepcopy(outputs)
    for message in reversed(planted.get("messages") or []):
        if message.get("role") == "ai" and not message.get("tool_calls"):
            message["content"] = (
                (message.get("content") or "").rstrip()
                + f"\n\nOne more that often helps: {PLANTED_MEASURE.lower()}."
            )
            return planted
    return planted


def plant_situation_violation(outputs: dict) -> dict:
    """Add a fact no turn mentions to what the agent wrote about the situation.

    Both places the question now reads, for the same reason the question reads
    both: cards alone appear in a handful of trajectories, and a planter that
    only edited cards could not plant in the ones the question had just been
    widened to cover -- so the five plants the gate needs would still have been
    out of reach, on a question widened precisely to bring them into it.
    """
    planted = copy.deepcopy(outputs)
    for approval in planted.get("approvals") or []:
        args = approval.get("args") or {}
        if args.get("circumstances"):
            args["circumstances"] = f"{args['circumstances'].rstrip('.')}, {PLANTED_SITUATION_CLAUSE}"
            approval["args"] = args
            return planted
    for message in planted.get("messages") or []:
        for call in message.get("tool_calls") or []:
            if call.get("name") != "record_choices":
                continue
            args = call.get("args") or {}
            situation = str(args.get("situation") or "")
            if situation and situation.lower() != SITUATION_DECLINED:
                args["situation"] = f"{situation.rstrip('.')}, {PLANTED_SITUATION_CLAUSE}"
                call["args"] = args
                return planted
    return planted


PLANTERS = {OFFERED_KEY: plant_offered_violation, SITUATION_KEY: plant_situation_violation}


def _planting_took(key: str, outputs: dict) -> bool:
    """Did the edit actually create the violation it was meant to?

    Two things have to hold: the text landed, and it is genuinely not
    something the trajectory already contained. A planted violation that is
    not a violation is worse than no planting -- the judge scores it 1,
    correctly, and the gate reads that as a miss.
    """
    if key == OFFERED_KEY:
        if PLANTED_MEASURE.lower() not in final_reply(outputs).lower():
            return False
        listed = " ".join(str(row.get("measure", "")) for row in measures_to_judge_against(outputs)).lower()
        return not any(word in listed for word in PLANTED_MEASURE_WORDS)
    if not any(PLANTED_SITUATION_CLAUSE in s for s in situation_sentences(outputs)):
        return False
    said = " ".join(user_turns(outputs)).lower()
    return not any(word in said for word in PLANTED_SITUATION_WORDS)


# -- what the labeller reads -------------------------------------------------

# Named because evals/label_judge.py has to find these two fields to avoid
# showing the final reply twice, and a field renamed here but not there would
# quietly bring the repeat back.
CONVERSATION_FIELD = "the_conversation_in_order"
FINAL_REPLY_FIELD = "the_final_reply_being_judged"


def evidence_for(key: str, outputs: dict) -> dict:
    """Everything needed to answer the question, in the file itself.

    The labeller must be able to decide without opening LangSmith, without
    re-running anything, and without reading the judge's answer first.
    """
    if key == OFFERED_KEY:
        return {
            "measures_the_agent_was_given": [
                f"{row.get('measure_id')}: {row.get('measure', '')}" for row in measures_to_judge_against(outputs)
            ],
            # The exchange, not extracts from it. Two of the judge's
            # allowances need it -- a measure *the user* brought up is not a
            # violation, and neither is an idea after a web search. A human
            # given less than the judge is not a second rater; the
            # disagreements that produces measure the gap, in one direction,
            # since the human marks violations the judge correctly allowed.
            #
            # First attempt at this gave the user's turns and the final reply
            # and dropped every agent turn between them, which put "yeah go
            # ahead" in the evidence with nothing visible to be agreeing to.
            # The first labelling session found it on its second item.
            CONVERSATION_FIELD: conversation(outputs),
            # This comment used to say the judge saw the search "because it is
            # handed the whole message list". It was not: the search is not in
            # the messages at all (see searched_the_web). The asymmetry ran the
            # other way, and the judge was the rater short of evidence.
            "a_web_search_ran": searched_the_web(outputs),
            # Repeated from the end of the conversation above, because the
            # question is about this reply and the rest is context. The file
            # keeps both; the labeller's screen shows it once (label_judge).
            FINAL_REPLY_FIELD: final_reply(outputs),
        }
    return {
        "what_the_user_said": user_turns(outputs),
        "what_the_agent_wrote_about_the_situation": situation_sentences(outputs),
    }


def _would_ask(key: str, outputs: dict) -> bool:
    """Whether the question has its evidence, without calling the model."""
    if key == OFFERED_KEY:
        return bool(measures_to_judge_against(outputs)) and bool(final_reply(outputs).strip())
    return bool(situation_sentences(outputs)) and bool(user_turns(outputs))


def _breadth_first(records: list[dict]) -> list[dict]:
    """One run of every case before any second run of the same one.

    The corpus is the suite run three times, so it holds three trajectories
    per case, and file order is run 0 of everything, then run 1, then run 2.
    Taking the first twenty in that order looks breadth-first and is not: only
    fourteen cases reach the grounding question at all, so the sample ran to
    twenty by taking six of them twice, and the first labelling session saw
    five cases three times over. Ordering explicitly makes the repeats the
    last thing taken rather than an accident of how the file was written.
    """
    by_case: dict[str, list[dict]] = {}
    for record in records:
        by_case.setdefault(record["case_id"], []).append(record)
    ordered: list[dict] = []
    for depth in range(max((len(runs) for runs in by_case.values()), default=0)):
        for runs in by_case.values():
            if depth < len(runs):
                ordered.append(runs[depth])
    return ordered


def export(records: list[dict], source_label: str, sample: int, plant: int) -> Path:
    """Grade trajectories with the judge, hide the verdict, write a labelling file."""
    payload: dict = {
        "source": source_label,
        "instructions": (
            "Answer each question yourself from the `evidence` in the item, then set "
            "human_score to 1 (the agent did nothing wrong on this question) or 0 (it did). "
            "Fields beginning with an underscore are the judge's own answer and are here "
            "only so `score` can compare -- reading them first defeats the point. "
            "A case id appearing more than once is not a duplicate: the agent is "
            "stochastic, the suite is run several times, and `run` says which run this "
            "trajectory came from. Two runs of one case are two different conversations "
            "and each is judged on what it says."
        ),
        "questions": {},
    }

    for key in JUDGE_KEYS:
        usable = _breadth_first([r for r in records if _would_ask(key, r["outputs"])])
        if not usable:
            print(f"  ! {key}: no trajectory in {source_label} reaches this question")
            payload["questions"][key] = {"items": []}
            continue

        # Two trajectories that produce the same evidence are one observation,
        # not two, and putting both in front of a labeller asks a question
        # nobody can answer. The first full session got four byte-identical
        # pairs on the situation question and scored two of them differently,
        # which is the only possible outcome and guaranteed a disagreement
        # with the judge on one copy of each.
        items, chosen, seen = [], [], set()
        for record in usable:
            if len(chosen) >= sample:
                break
            evidence = json.dumps(evidence_for(key, record["outputs"]), sort_keys=True)
            if evidence in seen:
                continue
            seen.add(evidence)
            chosen.append(record)
            items.append(_item(key, record, record["outputs"], planted=False))
        # Planted ones are extra, not a slice of the sample: replacing real
        # trajectories with edited ones would shrink the evidence about how
        # the judge behaves on output the agent really produced.
        #
        # Drawn from what the sample did not take, and only falling back to
        # the sample when there is nothing else. They used to come from the
        # first five usable trajectories -- the same five the sample opened
        # with -- so every plant was a byte-for-byte copy of an item already
        # labelled, with one clause added. The first full labelling session
        # read them as a file bug ("is this a duplicate?") and marked four of
        # the five as no violation, which is what took kappa to 0.01.
        spare = [r for r in usable if r not in chosen] or usable
        for record in spare[: max(0, plant)]:
            edited = PLANTERS[key](record["outputs"])
            if not _planting_took(key, edited):
                print(f"  ! {key}: could not plant a violation in {record['case_id']}")
                continue
            items.append(_item(key, record, edited, planted=True))

        payload["questions"][key] = {"items": items}
        real = sum(1 for i in items if not i["_planted"])
        print(f"  {key}: {real} real + {len(items) - real} planted")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    LABEL_PATH.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nLabel them in:\n  {LABEL_PATH}")
    return LABEL_PATH


def _item(key: str, record: dict, outputs: dict, *, planted: bool) -> dict:
    verdict = QUESTIONS[key](outputs)
    score = verdict.get("score")
    return {
        "case_id": record["case_id"],
        # Which run of the suite this came from. Without it a second run of a
        # case reads as the same item twice, which is how the first labelling
        # session came to call real items a defect.
        "run": record.get("run_index"),
        # Carried so a label can be written back onto the run it belongs to.
        # Empty for a local run, which has no run to write onto.
        "run_id": record.get("run_id", ""),
        "human_score": None,
        "human_note": "",
        "evidence": evidence_for(key, outputs),
        "_planted": planted,
        "_judge_score": None if score is None else int(bool(score)),
        "_judge_parsed": score is not None,
        "_judge_comment": (verdict.get("comment") or "").strip(),
    }


# -- stability ---------------------------------------------------------------


def stability(records: list[dict], runs: int) -> dict:
    """Re-grade one trajectory per question and see whether the answer holds.

    A judge that answers differently on identical input is not measuring the
    agent, and no amount of agreement on a single pass would show that.
    """
    result: dict = {}
    for key in JUDGE_KEYS:
        usable = [r for r in records if _would_ask(key, r["outputs"])]
        if not usable:
            result[key] = {"case_id": None, "verdicts": [], "stable": None}
            continue
        record = usable[0]
        verdicts = [QUESTIONS[key](record["outputs"]).get("score") for _ in range(runs)]
        normalised = [None if v is None else int(bool(v)) for v in verdicts]
        result[key] = {
            "case_id": record["case_id"],
            "verdicts": normalised,
            "stable": len(set(normalised)) == 1,
        }
        state = "stable" if result[key]["stable"] else "UNSTABLE"
        print(f"  {key}: {normalised} on {record['case_id']} -- {state}")

    if LABEL_PATH.exists():
        payload = json.loads(LABEL_PATH.read_text(encoding="utf-8"))
        payload["stability"] = result
        LABEL_PATH.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"\nRecorded in {LABEL_PATH}")
    return result


# -- scoring -----------------------------------------------------------------


def cohens_kappa(pairs: list[tuple[int, int]]) -> float | None:
    """Agreement between two raters, with chance agreement subtracted.

    Returns None when kappa is undefined, which happens exactly when one rater
    used a single class for everything: chance agreement is then 1, the
    denominator is 0, and there is no question of how much better than chance
    the pair did. That is not a technicality to paper over with a 1.0 -- a set
    of labels that are all the same answer carries no information about
    whether the judge can tell the two answers apart, and the gate must refuse
    it rather than pass it.
    """
    total = len(pairs)
    if not total:
        return None
    observed = sum(1 for human, judge in pairs if human == judge) / total
    expected = 0.0
    for value in (0, 1):
        expected += (
            sum(1 for human, _ in pairs if human == value) / total
            * sum(1 for _, judge in pairs if judge == value) / total
        )
    if expected >= 1.0:
        return None
    return (observed - expected) / (1 - expected)


def confusion(pairs: list[tuple[int, int]]) -> dict[str, int]:
    """The four ways a (human, judge) pair can fall, violation being 0.

    Kappa is symmetric: a judge that raises false alarms and one that misses
    violations can reach the same number, and they need opposite fixes. The
    literature asks for this table next to kappa for that reason (Rao &
    Callison-Burch; Husain reports the same thing as true positive and true
    negative rate). notes/BEST_PRACTICES.md has the sources.
    """
    return {
        "both_violation": sum(1 for human, judge in pairs if human == 0 and judge == 0),
        "judge_missed": sum(1 for human, judge in pairs if human == 0 and judge == 1),
        "false_alarm": sum(1 for human, judge in pairs if human == 1 and judge == 0),
        "both_fine": sum(1 for human, judge in pairs if human == 1 and judge == 1),
    }


BOOTSTRAP_RESAMPLES = 2000


def kappa_interval(
    pairs: list[tuple[int, int]], resamples: int = BOOTSTRAP_RESAMPLES, seed: int = 0
) -> tuple[float, float] | None:
    """A 95% percentile-bootstrap interval around kappa. Printed, not the gate.

    At twenty-five items one disagreement moves kappa a long way, and a point
    estimate alone reads as more certain than it is: the first question to
    pass, at 0.90, has an interval reaching down to 0.63 -- below the gate. The practitioner guides put a usable sample near a hundred items per
    question, which this project cannot afford, so the honest move is to print
    how wide the uncertainty is next to the number.

    Seeded, so the same labels always print the same interval. Resamples where
    kappa is undefined (every label the same answer) are left out; if most are,
    there is no interval to report.
    """
    if len(pairs) < 2:
        return None
    rng = random.Random(seed)
    kappas = sorted(
        k
        for k in (cohens_kappa([rng.choice(pairs) for _ in pairs]) for _ in range(resamples))
        if k is not None
    )
    if len(kappas) < resamples // 2:
        return None
    return kappas[int(0.025 * len(kappas))], kappas[min(len(kappas) - 1, int(0.975 * len(kappas)))]


def question_report(key: str, items: list[dict], stability_note: dict | None) -> dict:
    labelled = [i for i in items if i.get("human_score") in (0, 1)]
    agree = [i for i in labelled if int(i["human_score"]) == (i["_judge_score"] or 0)]
    planted = [i for i in items if i["_planted"]]
    caught = [i for i in planted if i["_judge_score"] == 0]
    unparsed = [i for i in items if not i["_judge_parsed"]]
    agreement = (len(agree) / len(labelled) * 100) if labelled else 0.0
    pairs = [(int(i["human_score"]), int(i["_judge_score"] or 0)) for i in labelled]
    kappa = cohens_kappa(pairs)
    stable = (stability_note or {}).get("stable")
    return {
        "key": key,
        "labelled": len(labelled),
        "agreement": agreement,
        "kappa": kappa,
        "kappa_interval": kappa_interval(pairs),
        "confusion": confusion(pairs),
        "planted": len(planted),
        "caught": len(caught),
        "unparsed": len(unparsed),
        "stable": stable,
        "disagreements": [i for i in labelled if i not in agree],
        "gate": (
            len(labelled) >= REQUIRED_LABELS
            and len(planted) >= REQUIRED_PLANTED
            and kappa is not None
            and kappa >= REQUIRED_KAPPA
            and len(caught) == len(planted)
            and not unparsed
            and stable is True
        ),
    }


def push_labels(payload: dict) -> int:
    """Write the human labels back onto the LangSmith runs they describe.

    This is what makes a calibration session visible to anyone else looking at
    the experiment: the judge's verdict and a person's verdict sit on the same
    run, under `human_<question>`, and the disagreements can be read in the UI
    beside the trace that produced them.

    Planted items are never pushed. Their run id points at a real run, but the
    label describes a trajectory that was edited afterwards -- writing it onto
    the run would record a verdict about output that run never produced.
    """
    from langsmith import Client

    client = Client()
    pushed = 0
    for key, block in (payload.get("questions") or {}).items():
        for item in block.get("items", []):
            if item.get("_planted") or not item.get("run_id"):
                continue
            if item.get("human_score") not in (0, 1):
                continue
            client.create_feedback(
                run_id=item["run_id"],
                key=f"human_{key}",
                score=int(item["human_score"]),
                comment=item.get("human_note") or None,
            )
            pushed += 1
    return pushed


def score(push: bool = False, path: Path | None = None) -> dict:
    """Agreement and the gate, from the working label file or a published one.

    `path` is what lets anyone re-check a published snapshot: the score is
    recomputed from the labels and verdicts in the file, with no model call.
    """
    path = path or LABEL_PATH
    if not path.exists():
        raise SystemExit(f"No labelling file at {path}. Run `export` first.")
    payload = json.loads(path.read_text(encoding="utf-8"))
    reports = {}
    for key in JUDGE_KEYS:
        items = payload.get("questions", {}).get(key, {}).get("items", [])
        report = question_report(key, items, (payload.get("stability") or {}).get(key))
        reports[key] = report
        print(f"\n## {key}")
        print(f"  labelled:   {report['labelled']}/{REQUIRED_LABELS} required")
        kappa = report["kappa"]
        print(f"  agreement:  {report['agreement']:.0f}% (printed, not the gate)")
        print(
            "  kappa:      "
            + (
                "undefined -- every label is the same answer, so the labels say "
                "nothing about whether the judge can tell them apart"
                if kappa is None
                else f"{kappa:.2f} ({REQUIRED_KAPPA:.2f} required)"
            )
        )
        interval = report["kappa_interval"]
        if interval is not None:
            low, high = interval
            print(f"              95% interval {low:.2f} to {high:.2f} "
                  f"(bootstrap, n={report['labelled']}; printed, not the gate)")
        c = report["confusion"]
        print("  errors:     "
              f"judge missed {c['judge_missed']} of the {c['both_violation'] + c['judge_missed']} "
              f"violations you found; "
              f"{c['false_alarm']} of its {c['both_violation'] + c['false_alarm']} "
              f"alarms were false")
        print(f"  planted:    {report['caught']}/{report['planted']} caught "
              f"({REQUIRED_PLANTED} required)")
        print(f"  parseable:  {len(items) - report['unparsed']}/{len(items)}")
        print(f"  stable:     {report['stable']}")
        print(f"  GATE:       {'PASS' if report['gate'] else 'FAIL'}")
        for item in report["disagreements"]:
            side = "judge stricter" if item["_judge_score"] == 0 else "judge lenient"
            planted = " [planted]" if item["_planted"] else ""
            print(f"\n    {item['case_id']}{planted}  judge={item['_judge_score']} "
                  f"you={item['human_score']}  ({side})")
            if item.get("human_note"):
                print(f"      you:   {item['human_note']}")
            print(f"      judge: {item['_judge_comment'][:300]}")

    if push:
        pushed = push_labels(payload)
        print(f"\nPushed {pushed} human labels to LangSmith as human_<question> feedback.")

    if all(r["gate"] for r in reports.values()):
        print("\nBoth questions clear the gate. Their numbers may be quoted, as a trend.")
    else:
        print("\nAt least one question has not cleared the gate. Do not quote its number "
              "anywhere -- fix the question in evals/judge.py against the disagreements "
              "above, re-export and re-label.")
    return reports


def main() -> None:
    parser = argparse.ArgumentParser(description="Calibrate the LLM judge against human labels")
    sub = parser.add_subparsers(dest="command", required=True)

    def add_source(sub_parser):
        sub_parser.add_argument("--from", dest="source", type=Path, default=None,
                                help="Trajectories from a local run. Default: the newest "
                                     "run for export, the labelled corpus for stability.")
        sub_parser.add_argument("--experiment", nargs="?", const="", default=None,
                                help="Calibrate a hosted LangSmith experiment instead; "
                                     "no name means the newest.")
        sub_parser.add_argument("--dataset", default=DATASET_NAME)

    e = sub.add_parser("export", help="Grade trajectories and write a labelling file")
    add_source(e)
    e.add_argument("--sample", type=int, default=REQUIRED_LABELS)
    e.add_argument("--plant", type=int, default=REQUIRED_PLANTED)

    st = sub.add_parser("stability", help="Re-grade one trajectory several times")
    add_source(st)
    st.add_argument("--runs", type=int, default=STABILITY_RUNS)

    sc = sub.add_parser("score", help="Compare your labels against the judge's, and check the gate")
    sc.add_argument("--push", action="store_true",
                    help="Write the human labels back onto the LangSmith runs.")
    sc.add_argument("--file", type=Path, default=None,
                    help="Score this label file instead of the working one, "
                         "e.g. a published snapshot's.")

    args = parser.parse_args()
    if args.command in ("export", "stability") and args.experiment is None and args.source is None:
        args.source = default_trajectories(args.command)
        print(f"Trajectories: {args.source}")
    if args.command == "export":
        records, label = resolve_source(args.source, args.experiment, args.dataset)
        export(records, label, args.sample, args.plant)
    elif args.command == "stability":
        records, _ = resolve_source(args.source, args.experiment, args.dataset)
        stability(records, args.runs)
    else:
        score(push=args.push, path=args.file)


if __name__ == "__main__":
    main()
