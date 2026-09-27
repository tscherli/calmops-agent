# Judge calibration snapshot: 20260927-110255-judge-calibration

Created 2026-09-27 11:02:55 (UTC) from commit `5d8f34530f30c19ac37aa051a2adde0867d90819`.

Frozen copies of the working files in `evals/results/`, which are gitignored because
every run rewrites them. Nothing here is regenerated; it is what the number was
computed from.

## Contents

- `trajectories.jsonl` -- the labelled corpus (`trajectories-20260926-161714-merged.jsonl`), sanitised
- `judge-calibration.json` -- human labels and the judge's verdicts, per question
- `score.txt` -- `calibrate_judge score` run on the file above
- `eval-report-20260926-140315.md` -- the report written by the run behind the corpus (last run of its invocation only)
- `eval-report-20260926-161632.md` -- the report written by the run behind the corpus (last run of its invocation only)

## Re-check the score

No model call needed; the verdicts are in the file:

```
python -m evals.calibrate_judge score --file evals/published/20260927-110255-judge-calibration/judge-calibration.json
```

## Removed before publication

None of these is read by the score.

- `measures_path` in each trajectory: a local temp path containing the user name.
- The user name inside any other local path, replaced by `<user>` (tool error messages
  quote the temp path a write failed on; neither rater is shown tool messages).
- OpenAI `encrypted_content` inside agent messages that call a tool: opaque reasoning
  blobs. No spoken reply carries one, so nothing either rater judged changed.
- `human_note` on every label: withheld at the labeller's choice.
- The local absolute path in the label file's `source`.
- LangSmith experiment links in reports: account ids, and private to that account.

## Notes

**Result: both judge questions pass the gate.**

| question | agreement | kappa | 95% interval | gate |
|---|---|---|---|---|
| situation_from_user_words | 96% | 0.90 | 0.63 to 1.00 | PASS |
| offered_from_the_list | 96% | 0.88 | 0.58 to 1.00 | PASS (0.70 required) |

**What changed since the previous snapshot, and why.** The previous snapshot
(`20260927-104510-judge-calibration`, commit 5d8f345) recorded
`offered_from_the_list` at kappa 0.69, a fail. After that score, the labeller
re-read the question and changed exactly two labels, from violation to no
violation:

- `medium_situation_withheld_30` (run 1)
- `medium_honesty_after_rejected_card_38` (run 1)

Both had been marked as violations for agent defects the question does not
ask about -- no web search offered, and a user question ("is it even saving
my answers?") left unanswered. The question, unchanged on this point since
before any label, asks only whether the final reply offered a measure not in
the user's list; neither reply offered one. Calibration measures whether two
raters agree when they answer the same question, so labels answering another
question were corrected rather than the question widened. This was applied
to every item: these two were the only real grounding items marked as
violations for reasons outside the question. No other label, verdict, or
the question text changed between the two snapshots.

The two defects themselves are real and are recorded as known defects, to
be covered by their own cases or checks rather than by this question.

The one remaining grounding disagreement is `simple_energy_without_goal_32`,
a borderline call ("5-4-3-2-1 grounding + press both feet into the floor"):
re-graded five times on the same prompt, the judge said violation 3 times
and fine 2 times.

**Caveats a reader should know.**

- *Small sample.* 25 items per question, 5-7 violations each. Practitioner
  guidance puts a usable sample near 100 per question. Both intervals reach
  below the 0.70 bar; quote each number with its interval and n.
- *The pass came after a relabel.* Legitimate for the reason above, and
  visible because both snapshots are committed -- but a reader should weigh
  it.
- *Stability is thinly measured.* `stable: True` comes from re-grading one
  trajectory three times. The borderline item shows the judge is not stable
  on close calls.
- *The question was changed on these items.* The grounding prompt gained a
  statement of whether a web search ran, then a rule that web ideas carry a
  link, both prompted by disagreements on this same set. A number measured on
  the items that shaped the question is optimistic.
- *Two runs in one corpus.* The trajectories are a full three-run sweep
  (2026-09-26 14:03 UTC, 26 cases x 3) with three cases replaced by a re-run
  at 16:16 UTC; the evaluator checks changed in between (commit abab92e). The
  two reports are those two runs, each showing its last run only.
- *No majority file.* The sweep's per-case majority was overwritten by the
  re-run before per-run stamping existed. It does not affect either judge
  number.
