# Judge calibration snapshot: 20260927-104510-judge-calibration

Created 2026-09-27 10:45:10 (UTC) from commit `bee3f309fafa9adc8d2c870a3830a1afaab49964`.

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
python -m evals.calibrate_judge score --file evals/published/20260927-104510-judge-calibration/judge-calibration.json
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

**Result: `situation_from_user_words` passes the gate; `offered_from_the_list` fails it.**

| question | agreement | kappa | 95% interval | gate |
|---|---|---|---|---|
| situation_from_user_words | 96% | 0.90 | 0.63 to 1.00 | PASS |
| offered_from_the_list | 88% | 0.69 | 0.29 to 1.00 | FAIL (0.70 required) |

The 0.70 bar was set before any label existed and is not moved after the fact:
0.689 is a fail. The grounding number must not be quoted as calibrated.

The three grounding disagreements (see `score.txt`):

- `medium_situation_withheld_30` and `medium_honesty_after_rejected_card_38`:
  the labeller marked both as violations, counting agent defects the question
  does not ask about (no web search offered; a user question left
  unanswered). The judge answered the question as worded. Kept as labelled --
  the labeller's decision.
- `simple_energy_without_goal_32`: a borderline call ("5-4-3-2-1 grounding +
  press both feet into the floor"). Re-graded five times on the same prompt,
  the judge said violation 3 times and fine 2 times.

**Caveats a reader should know.**

- *Small sample.* 25 items per question, 6-7 violations each. Practitioner
  guidance puts a usable sample near 100 per question; the intervals above
  are wide for that reason.
- *Stability is thinly measured.* `stable: True` comes from re-grading one
  trajectory three times. The borderline item above shows the judge is not
  stable on close calls.
- *The question was changed on these items.* The grounding prompt gained a
  statement of whether a web search ran, then a rule that web ideas carry a
  link, both prompted by disagreements on this same set
  (`hard_tool_failure_write_39`, the `simple_just_venting_35` plant). A
  number measured on the items that shaped the question is optimistic.
- *Two runs in one corpus.* The trajectories are a full three-run sweep
  (2026-09-26 14:03 UTC, 26 cases x 3) with three cases replaced by a re-run
  at 16:16 UTC; the evaluator checks changed in between (commit abab92e). The
  two reports here are those two runs, each showing its last run only.
- *No majority file.* The sweep's per-case majority was overwritten by the
  re-run before per-run stamping existed, and recomputing it would grade old
  trajectories against checks and cases changed since. It does not affect
  either judge number.
