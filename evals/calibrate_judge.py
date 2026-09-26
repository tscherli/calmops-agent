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
import re
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

from evals.create_dataset import DATASET_NAME
from evals.judge import JUDGE_KEYS, OFFERED_KEY, QUESTIONS, SITUATION_KEY
from evals.outputs import final_reply, offered_measures, situation_sentences, user_turns

RESULTS_DIR = Path(__file__).resolve().parent / "results"
LABEL_PATH = RESULTS_DIR / "judge-calibration.json"
TRAJECTORIES_PATH = RESULTS_DIR / "trajectories.jsonl"

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
# 0.70 is the conventional floor for "substantial" agreement. Raw agreement is
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
    """Put a measure in the reply that no lookup returned."""
    planted = copy.deepcopy(outputs)
    for message in reversed(planted.get("messages") or []):
        if message.get("role") == "ai" and not message.get("tool_calls"):
            message["content"] = (
                (message.get("content") or "").rstrip()
                + f" One more idea: {PLANTED_MEASURE.lower()}."
            )
            return planted
    return planted


def plant_situation_violation(outputs: dict) -> dict:
    """Add a fact to the card's sentence that no turn mentions."""
    planted = copy.deepcopy(outputs)
    for approval in planted.get("approvals") or []:
        args = approval.get("args") or {}
        if args.get("circumstances"):
            args["circumstances"] = f"{args['circumstances'].rstrip('.')}, {PLANTED_SITUATION_CLAUSE}"
            approval["args"] = args
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
        listed = " ".join(str(row.get("measure", "")) for row in offered_measures(outputs)).lower()
        return not any(word in listed for word in PLANTED_MEASURE_WORDS)
    if not any(PLANTED_SITUATION_CLAUSE in s for s in situation_sentences(outputs)):
        return False
    said = " ".join(user_turns(outputs)).lower()
    return not any(word in said for word in PLANTED_SITUATION_WORDS)


# -- what the labeller reads -------------------------------------------------


def evidence_for(key: str, outputs: dict) -> dict:
    """Everything needed to answer the question, in the file itself.

    The labeller must be able to decide without opening LangSmith, without
    re-running anything, and without reading the judge's answer first.
    """
    if key == OFFERED_KEY:
        return {
            "measures_the_agent_was_given": [
                f"{row.get('measure_id')}: {row.get('measure', '')}" for row in offered_measures(outputs)
            ],
            "the_agents_final_reply": final_reply(outputs),
        }
    return {
        "what_the_user_said": user_turns(outputs),
        "what_the_agent_wrote_about_the_situation": situation_sentences(outputs),
    }


def _would_ask(key: str, outputs: dict) -> bool:
    """Whether the question has its evidence, without calling the model."""
    if key == OFFERED_KEY:
        return bool(offered_measures(outputs)) and bool(final_reply(outputs).strip())
    return bool(situation_sentences(outputs)) and bool(user_turns(outputs))


def export(records: list[dict], source_label: str, sample: int, plant: int) -> Path:
    """Grade trajectories with the judge, hide the verdict, write a labelling file."""
    payload: dict = {
        "source": source_label,
        "instructions": (
            "Answer each question yourself from the `evidence` in the item, then set "
            "human_score to 1 (the agent did nothing wrong on this question) or 0 (it did). "
            "Fields beginning with an underscore are the judge's own answer and are here "
            "only so `score` can compare -- reading them first defeats the point."
        ),
        "questions": {},
    }

    for key in JUDGE_KEYS:
        usable = [r for r in records if _would_ask(key, r["outputs"])]
        if not usable:
            print(f"  ! {key}: no trajectory in {source_label} reaches this question")
            payload["questions"][key] = {"items": []}
            continue

        items = []
        chosen = usable[:sample]
        for record in chosen:
            items.append(_item(key, record, record["outputs"], planted=False))
        # Planted ones are extra, not a slice of the sample: replacing real
        # trajectories with edited ones would shrink the evidence about how
        # the judge behaves on output the agent really produced.
        for record in usable[: max(0, plant)]:
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


def question_report(key: str, items: list[dict], stability_note: dict | None) -> dict:
    labelled = [i for i in items if i.get("human_score") in (0, 1)]
    agree = [i for i in labelled if int(i["human_score"]) == (i["_judge_score"] or 0)]
    planted = [i for i in items if i["_planted"]]
    caught = [i for i in planted if i["_judge_score"] == 0]
    unparsed = [i for i in items if not i["_judge_parsed"]]
    agreement = (len(agree) / len(labelled) * 100) if labelled else 0.0
    kappa = cohens_kappa(
        [(int(i["human_score"]), int(i["_judge_score"] or 0)) for i in labelled]
    )
    stable = (stability_note or {}).get("stable")
    return {
        "key": key,
        "labelled": len(labelled),
        "agreement": agreement,
        "kappa": kappa,
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


def score(push: bool = False) -> dict:
    if not LABEL_PATH.exists():
        raise SystemExit(f"No labelling file at {LABEL_PATH}. Run `export` first.")
    payload = json.loads(LABEL_PATH.read_text(encoding="utf-8"))
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
        sub_parser.add_argument("--from", dest="source", type=Path, default=TRAJECTORIES_PATH,
                                help="Trajectories from a local run (the default).")
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

    args = parser.parse_args()
    if args.command == "export":
        records, label = resolve_source(args.source, args.experiment, args.dataset)
        export(records, label, args.sample, args.plant)
    elif args.command == "stability":
        records, _ = resolve_source(args.source, args.experiment, args.dataset)
        stability(records, args.runs)
    else:
        score(push=args.push)


if __name__ == "__main__":
    main()
