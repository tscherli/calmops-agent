"""Label the judge's trajectories one at a time, in the terminal.

`calibrate_judge export` writes a JSON file with fifty items in it. Reading
that file in an editor and scrolling between an array of measure strings and a
wall of reply text is the wrong shape for the job: the labeller is answering
the same yes/no question fifty times, and what they need on screen is one
item, wrapped, with the question above it.

This is the whole of the tool. It shows one item, takes a key, writes the
answer straight back to the file, and moves on. Nothing here computes
anything -- `calibrate_judge score` still does that.

**It never shows the judge's answer, and it never says which items are
planted.** Both are in the file, and both would anchor a labeller who saw
them: knowing an item was edited to contain a violation is knowing the answer.
Blinding is not a nicety here, it is the measurement. If a planted item does
not read as a violation, label it as you see it -- the gate checks separately
that every plant was caught, so a bad plant shows up there as a failure rather
than being quietly absorbed into the agreement number.

**The question shown is the judge's own.** It is pulled out of the prompt in
evals/judge.py rather than restated here, because a human and a model scored
against two different wordings of "the same" question produce a kappa that
means nothing, and two copies of a rubric drift the first time one is edited.

Usage:

    python -m evals.label_judge                 # the newest exported file
    python -m evals.label_judge --file <path>
    python -m evals.label_judge --all           # revisit items already done

Every answer is written before the next item is shown, so it is safe to stop
with `q` or Ctrl-C and pick it up later.
"""
from __future__ import annotations

import argparse
import json
import re
import textwrap
from pathlib import Path
from typing import Callable, Iterable

from evals.calibrate_judge import CONVERSATION_FIELD, FINAL_REPLY_FIELD, LABEL_PATH
from tools.measures import _PUNCTUATION
from evals.judge import JUDGE_KEYS, OFFERED_KEY, OFFERED_PROMPT, SITUATION_KEY, SITUATION_PROMPT

WIDTH = 96

PROMPTS = {OFFERED_KEY: OFFERED_PROMPT, SITUATION_KEY: SITUATION_PROMPT}

HELP = """  1  no violation -- the agent did nothing wrong on this question
  0  violation
  s  skip for now
  n  add a note to this item (then answer it)
  ?  show the question again
  q  save and stop
"""


def question_text(key: str) -> str:
    """The judge's own wording, lifted out of its prompt.

    Not restated here on purpose. A human and a model scored against two
    different wordings of "the same" question produce a kappa that describes
    nothing, and the second copy of a rubric is wrong the first time anyone
    edits the first.
    """
    match = re.search(r"<Question>(.*?)</Question>", PROMPTS[key], re.S)
    if not match:  # pragma: no cover - the prompts are tested for this
        raise SystemExit(f"{key}: no <Question> block in the judge's prompt")
    return textwrap.dedent(match.group(1)).strip()


def _readable(text: str) -> str:
    """Smart punctuation flattened to ASCII, for the console.

    The model writes em dashes and curly quotes, and a Windows console at its
    default codepage renders them as replacement characters -- a poor thing to
    ask someone to read fifty times. The same table the measure list uses,
    applied to the screen only: nothing is written back, so the file keeps
    exactly what the agent said.
    """
    return str(text).translate(_PUNCTUATION).encode("ascii", "replace").decode("ascii")


def _wrap(text: str, indent: str = "  ", hanging: str | None = None) -> str:
    lines = []
    for paragraph in _readable(text).splitlines() or [""]:
        if not paragraph.strip():
            lines.append("")
            continue
        lines.extend(
            textwrap.wrap(
                paragraph,
                width=WIDTH,
                initial_indent=indent,
                # A wrapped bullet returning to the bullet's own indent reads
                # as the next bullet. Two more spaces keep the list countable,
                # which matters when the question is "is every one of these in
                # the list above".
                subsequent_indent=indent if hanging is None else hanging,
            )
        )
    return "\n".join(lines)


FINAL_REPLY_STANDIN = "agent: (the final reply -- shown in full below)"


def _without_the_repeated_reply(evidence: dict) -> dict:
    """The same evidence, with the final reply on screen once instead of twice.

    The grounding item carries the whole conversation and then the final reply
    again as its own field. Rendered as stored, the reply being judged appeared
    twice in a row, and the second copy read as something already seen. The
    second labelling session skimmed past the planted sentence in all five
    plants -- it sat at the very end of that second copy -- and asked of two of
    them whether they were duplicates.

    Screen only. The file is untouched, so labels keyed on a fingerprint of the
    evidence still match, and the judge is graded on exactly what it was.
    """
    lines = evidence.get(CONVERSATION_FIELD)
    reply = str(evidence.get(FINAL_REPLY_FIELD) or "").strip()
    if not isinstance(lines, list) or not lines or not reply:
        return evidence
    if str(lines[-1]).strip() != f"agent: {reply}":
        return evidence
    return {**evidence, CONVERSATION_FIELD: [*lines[:-1], FINAL_REPLY_STANDIN]}


def _render_evidence(evidence: dict) -> str:
    """The item's own evidence, in the order it was written.

    Lists become one indented line each; everything else is wrapped. The field
    names come straight from the file rather than being mapped to prettier
    labels, so that what is on screen and what is in the JSON are the same
    thing when someone goes looking. One exception: the final reply is shown
    once, not twice (_without_the_repeated_reply).
    """
    blocks = []
    for name, value in _without_the_repeated_reply(evidence).items():
        heading = name.replace("_", " ")
        if isinstance(value, list):
            body = (
                "\n".join(_wrap(f"- {entry}", indent="  ", hanging="    ") for entry in value)
                if value
                else "  (none)"
            )
        elif isinstance(value, bool):
            body = f"  {'yes' if value else 'no'}"
        else:
            body = _wrap(value) if str(value).strip() else "  (none)"
        blocks.append(f"{heading}:\n{body}")
    return "\n\n".join(blocks)


def render(item: dict, key: str, position: int, total: int) -> str:
    """One screen. Never includes the judge's verdict or the planted flag."""
    note = item.get("human_note") or ""
    # The run number is on the header because a case id can appear more than
    # once: the agent is stochastic, the suite is run three times, and two
    # runs of one case are two different conversations. Without it the second
    # one reads as the file repeating itself, which is what the first full
    # session concluded -- it marked real items as duplicates and a defect.
    run = item.get("run")
    where = f"[{item.get('case_id', '?')}]" + ("" if run is None else f" run {run + 1}")
    return "\n".join(
        [
            "=" * WIDTH,
            f"{key}   item {position} of {total}   {where}",
            "=" * WIDTH,
            "",
            _render_evidence(item.get("evidence") or {}),
            "",
            f"  note: {note}" if note else "",
            "-" * WIDTH,
        ]
    )


def _items_to_do(payload: dict, key: str, revisit: bool) -> list[dict]:
    items = payload.get("questions", {}).get(key, {}).get("items", [])
    if revisit:
        return list(items)
    return [item for item in items if item.get("human_score") not in (0, 1)]


def label(
    payload: dict,
    path: Path,
    *,
    reader: Callable[[str], str] = input,
    printer: Callable[[str], None] = print,
    keys: Iterable[str] = JUDGE_KEYS,
    revisit: bool = False,
) -> int:
    """The loop. Writes the file after every answer, so stopping is free."""
    answered = 0
    for key in keys:
        todo = _items_to_do(payload, key, revisit)
        if not todo:
            printer(f"\n{key}: nothing left to label.")
            continue
        printer("\n" + "#" * WIDTH)
        printer(f"# {key} -- {len(todo)} to go")
        printer("#" * WIDTH)
        printer("\n" + _wrap(question_text(key)))
        printer("\n" + HELP)

        for position, item in enumerate(todo, 1):
            printer("\n" + render(item, key, position, len(todo)))
            while True:
                answer = (reader("  1 / 0 / s / n / ? / q > ") or "").strip().lower()
                if answer == "?":
                    printer("\n" + _wrap(question_text(key)) + "\n" + HELP)
                    continue
                if answer == "n":
                    item["human_note"] = (reader("  note > ") or "").strip()
                    _save(payload, path)
                    continue
                if answer == "q":
                    _save(payload, path)
                    printer(f"\nStopped. {answered} labelled this session, saved to {path}.")
                    return answered
                if answer == "s":
                    break
                if answer in ("0", "1"):
                    item["human_score"] = int(answer)
                    answered += 1
                    _save(payload, path)
                    break
                printer("  Not one of the keys. ? for the question, q to stop.")
    printer(f"\nDone. {answered} labelled this session, saved to {path}.")
    return answered


def _save(payload: dict, path: Path) -> None:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def remaining(payload: dict) -> dict[str, int]:
    return {key: len(_items_to_do(payload, key, False)) for key in JUDGE_KEYS}


def main() -> None:
    parser = argparse.ArgumentParser(description="Label the judge's trajectories by hand")
    parser.add_argument("--file", type=Path, default=LABEL_PATH)
    parser.add_argument(
        "--all", action="store_true", dest="revisit",
        help="Go through every item again, including ones already labelled.",
    )
    args = parser.parse_args()

    if not args.file.exists():
        raise SystemExit(f"No labelling file at {args.file}. Run `calibrate_judge export` first.")
    payload = json.loads(args.file.read_text(encoding="utf-8"))

    left = remaining(payload)
    print(f"Labelling {args.file}")
    print("  " + ",  ".join(f"{key}: {count} left" for key, count in left.items()))
    label(payload, args.file, revisit=args.revisit)
    print("\nNext: python -m evals.calibrate_judge stability, then score.")


if __name__ == "__main__":
    main()
