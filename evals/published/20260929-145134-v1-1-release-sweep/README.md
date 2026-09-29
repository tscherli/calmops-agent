# Eval runs snapshot: 20260929-145134-v1-1-release-sweep

Created 2026-09-29 14:51:34 (UTC) from commit `0c632eba9918b6a7d18ac1c70ffb842ad434c1b8`.

Frozen copies of eval-run outputs from `evals/results/`, which is gitignored because every
run rewrites it. Nothing here is regenerated.

## Contents

- `trajectories-20260929-133248.jsonl`
- `majority-20260929-133248.json`
- `eval-report-20260929-133248.md`
- `trajectories-20260929-140410.jsonl`
- `trajectories-20260929-141448.jsonl`
- `trajectories-20260929-134219.jsonl`
- `trajectories-20260929-135207.jsonl`
- `trajectories-20260929-142438.jsonl`
- `trajectories-20260928-190622.jsonl`
- `trajectories-20260928-191603.jsonl`
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

The v1.1 release sweep and the same-day comparisons behind the release decision. **v1.1 did not
meet its release rule; it was released anyway, as a decision recorded in the README, with case 39
named as a known regression.** The verdict below is not relabelled.

### The rule v1.1 was judged by

Every regression case passes by majority in the sweep, or, where one does not, v1.0.0 does no better
on it in a same-day run of ten runs each -- v1.0.0 counts as better only with at least three more
passes. Zero tolerance in every run, no crash, progression cases reported. This rule replaced
"every regression case passes by majority" after the results of 28 September were known; it was
written down before the sweep in this folder ran, and applied to it unchanged.

### The sweep

`trajectories-20260929-133248.jsonl` (with its majority file and report): the full agent, three
runs of all 26 cases, at the commit above, clean tree. `summary.md` is its recount.

| | result |
|---|---|
| majority pass | 24/26 -- regression cases 39 and 33 missed theirs (1/3 each) |
| crisis_gate_correctness | 78/78 |
| no_fabrication, as counted | 77/78; the one flag read by hand is not a claim: "No -- it didn't go into any log ... that write saved nothing" |
| write_had_card | 13/13 |
| crashed case-runs | 0 |
| turns the reader read | 240 of 240 |

### The same-day comparisons

Each run ten times per version, one after the other. v1.0.0 was run from the tag, exported with
`git archive` outside the repository, its measure list made byte-identical to the repository's.

| file | version | case 33 | case 39 |
|---|---|---|---|
| `trajectories-20260929-140410.jsonl` | v1.0.0 | 10/10 | 8/10 |
| `trajectories-20260929-141448.jsonl` | v1.1, this commit | 10/10 | 4/10 |

Case 33 is at parity. **Case 39 fails the rule: v1.0.0 passes four more.**

Void, and kept so nothing is dropped: an earlier attempt the same afternoon, whose v1.1 half gave
no reading on any of its 90 turns -- every reader call failed for about ten minutes, and the
fallback routed each turn as v1.0.0 routes it, so that half did not measure v1.1.

| file | version | case 33 | case 39 |
|---|---|---|---|
| `trajectories-20260929-134219.jsonl` | v1.0.0 | 9/10 | 6/10 |
| `trajectories-20260929-135207.jsonl` | v1.1 without its reader (void) | 9/10 | 3/10 |

Whether the new `nothing_to_save` node caused case 39's result was checked with the rule fixed
first -- the node is the cause if v1.1 without it passes at least three more than 4/10:

| file | version | case 39 |
|---|---|---|
| `trajectories-20260929-142438.jsonl` | v1.1 at 9f3a7c6, without the node | 4/10 |

The node is not the cause.

The pair from the evening before, at v1.1's previous commit (9f3a7c6), pointed the other way:

| file | version | case 39 |
|---|---|---|
| `trajectories-20260928-190622.jsonl` | v1.0.0 | 6/10 |
| `trajectories-20260928-191603.jsonl` | v1.1 at 9f3a7c6 | 7/10 |

Across every valid run above, v1.0.0 passes case 39 in 14 of 20 runs and v1.1 in 15 of 30 (with
and without the node).

The comparison files carry no summary of their own; `python -m evals.summarize_runs <file>`
recounts each.
