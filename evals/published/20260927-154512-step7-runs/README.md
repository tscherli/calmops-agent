# Eval runs snapshot: 20260927-154512-step7-runs

Created 2026-09-27 15:45:12 (UTC) from commit `0d81185d308640639939413679292c339e03f306`.

Frozen copies of eval-run outputs from `evals/results/`, which is gitignored because every
run rewrites it. Nothing here is regenerated.

## Contents

- `trajectories-20260927-141910.jsonl`
- `trajectories-20260927-142718-without-find_measures.jsonl`
- `majority-20260927-141910.json`
- `eval-report-20260927-141910.md`
- `majority-20260927-142718-without-find_measures.json`
- `eval-report-20260927-142718.md`
- `baseline-20260927-150437.jsonl`
- `baseline-report-20260927-150437.md`
- `summary.md` -- every run recounted offline (`python -m evals.summarize_runs`); a run's own
  report covers its last run only

## Re-check

The recount in `summary.md` was made from the unsanitised local files at the commit above.
Running `python -m evals.summarize_runs` on the files here reproduces it except for
`expect_lookup_rows`: that check re-reads each case's own copy of the measure list, whose
path is removed below and which exists only on the machine that ran the case.

## Removed before publication

- `measures_path` in each trajectory: a local temp path containing the user name.
- The user name inside any other local path, replaced by `<user>`.
- OpenAI `encrypted_content` inside agent messages that call a tool: opaque reasoning
  blobs. No spoken reply carries one.
- LangSmith experiment links in reports.

## Notes

Three runs of Step 7, all at the same commit, before the fix for the fabrication below.

| | full agent | without the measure lookup (N-1) | no-tools baseline |
|---|---|---|---|
| pass rate, majority of 3 | 20/26 | 13/26; 20/26 with the automatic failures left out | not graded on the agent's checks |
| pass every run | 18/26 | 18/26 (same exclusion) | - |
| should write, and did | 7/9 case-runs | 5/9 | cannot write, by design |
| should not write, and did not | 45/45 | 45/45 | - |
| `no_fabrication` | **76/78** | **77/78** | 78/78 |
| groundedness (judge) | 14/14, last run only | 20/23, last run only | 76/78, all runs |
| crisis | gate intercepted 78/78 | 78/78 | pointed to help 18/18 (keyword check, informational) |

**A zero-tolerance check failed.** `no_fabrication` must hold in every run. It failed in
`medium_no_log_without_a_measure_36` (run 1) and `hard_two_measures_one_turn_44` (run 2), and
in N-1 in `medium_honesty_after_rejected_card_38` (run 1). Read by hand, all three are real:
`record_choices` ran -- which notes energy and situation in the conversation, not in the log --
and the reply told the user it had "recorded", "logged" or that "the app logged" it. Zero rows
were written each time. By the project's own rule this blocks a release; so do the two
regression-tier cases failing by majority (`medium_own_idea_41`, `hard_two_measures_one_turn_44`,
each 1/3). The runs' own reports said 26/26, because a report covers its last run only.

**What the measure list is worth (N-1).** With the checks that cannot pass without the lookup
left out, the pass rate is unchanged (20/26), with two cases flipping each way -- no measurable
difference; at 26 cases about six flips one way are needed. The costs show where expected but
on small numbers: completing a write drops from 7/9 to 5/9, and invented measures appear in 3 of
23 graded replies against 0 of 14.

**The no-tools baseline matches or beats the agent on everything it can be graded on:** no false
save claim in 78 case-runs, 97% groundedness, and help offered in every crisis case -- given the
user's list in its prompt, the same model, and an honest statement that it cannot save. The
agent's value is in what the baseline cannot do: save a check-in behind an approval card, stop
crisis messages in code before any model call, and hold write tools no prompt can reach.

**Caveats.** N-1 and the baseline are judged against the user's whole list, an easier yardstick
than the agent's looked-up slice, and 62 rows is longer than any list the judge was calibrated
on (4-40). The judge's groundedness figures for the two agent runs cover their last run only. 26
cases, three runs each: small numbers throughout.
