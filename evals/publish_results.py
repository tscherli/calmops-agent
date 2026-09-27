"""Freeze the evidence behind a judge-calibration number into a committed folder.

The working files in evals/results are gitignored for a reason: every run,
export and labelling session rewrites them, so a commit would catch them
halfway through a change. A number quoted in the README needs the opposite --
the exact labels, verdicts and trajectories it was computed from, fixed, so
anyone can re-check it. This writes that as a snapshot:

    evals/published/<UTC yyyymmdd-hhmmss>-<name>/
        trajectories.jsonl       the labelled corpus, sanitised
        judge-calibration.json   labels and judge verdicts, notes removed
        score.txt                `calibrate_judge score` on the file above
        <report>.md ...          the report(s) of the run(s) behind the corpus
        README.md                what this is, which commit, what was removed

    python -m evals.publish_results --name grounding-calibration \\
        --report evals/results/eval-report-20260926-140315.md

Re-check a snapshot without any model call:

    python -m evals.calibrate_judge score --file evals/published/<dir>/judge-calibration.json

**What is removed, and why.** Nothing that decides a score.
  - `measures_path` in each trajectory: a temp path that carries the local
    Windows user name, 82 times in the first corpus.
  - The user name inside any other path, replaced by `<user>`: the
    write-failure case's tool receipts quote the temp path they failed on.
    Tool messages are not shown to either rater.
  - OpenAI's `encrypted_content` inside agent messages that call a tool: an
    opaque reasoning blob nobody outside OpenAI can read, and most of the
    file's size. Only on tool-calling messages -- no spoken reply, and so
    nothing either rater judged, carries one.
  - `human_note` on every label: the labeller's own remarks, withheld by
    their choice. The score does not read them.
  - The local absolute path in the label file's `source`.
  - LangSmith experiment links in reports: they carry account ids and open
    only for the account that made them.

Then a last check refuses to write anything if the user's home directory or
user name still appears anywhere in what would be published.

**The working tree must be clean.** The README records the commit, and a
commit that is not the code that produced the snapshot records nothing.
"""
from __future__ import annotations

import argparse
import contextlib
import io
import json
import re
import subprocess
import tempfile
from datetime import datetime, timezone
from pathlib import Path

from evals.calibrate_judge import LABEL_PATH, labelled_corpus, load_trajectories, score

PUBLISHED_DIR = Path(__file__).resolve().parent / "published"
REPO_ROOT = Path(__file__).resolve().parent.parent

_ENCRYPTED = re.compile(r"""(['"]encrypted_content['"]\s*:\s*)(['"])(?:(?!\2).)*\2""")
_LANGSMITH_LINE = re.compile(r"^LangSmith experiment:.*$\n?", re.MULTILINE)
REMOVED = "[removed for publication]"


def _without_user_name(text: str) -> str:
    """Replace the user-name segment of any home path with <user>.

    Matches the segment after `Users` however many backslashes escaping has
    stacked in front of it: a Windows error quoted inside a tool message,
    repr'd, then JSON-encoded arrives with four. Found by the last check on the
    first real dry run -- the write-failure case's tool receipts quote the temp
    path they could not write to.
    """
    name = re.escape(Path.home().name)
    return re.sub(rf"(?i)(users(?:\\+|/+)){name}(?=\\|/)", r"\1<user>", text)


def sanitise_trajectory(record: dict) -> dict:
    """One trajectory, without local paths and the encrypted blobs."""
    record = json.loads(json.dumps(record))
    outputs = record.get("outputs") or {}
    outputs.pop("measures_path", None)
    for message in outputs.get("messages") or []:
        content = message.get("content")
        if isinstance(content, str) and "encrypted_content" in content:
            message["content"] = _ENCRYPTED.sub(rf"\1\2{REMOVED}\2", content)
    return json.loads(_without_user_name(json.dumps(record)))


def sanitise_labels(payload: dict, corpus_name: str) -> dict:
    """The label file, without notes and without a local path."""
    payload = json.loads(json.dumps(payload))
    payload["source"] = f"local:{corpus_name}"
    for block in (payload.get("questions") or {}).values():
        for item in block.get("items", []):
            item.pop("human_note", None)
    return payload


def sanitise_report(text: str) -> str:
    return _LANGSMITH_LINE.sub("", text)


def personal_markers() -> list[str]:
    """Strings that must not appear in anything published."""
    home = Path.home()
    return [str(home), home.as_posix(), json.dumps(str(home))[1:-1], home.name]


def find_personal(files: dict[str, str], markers: list[str]) -> list[str]:
    return [f"{name}: contains {marker!r}" for name, text in files.items() for marker in markers
            if marker and marker in text]


def git_commit() -> str:
    return subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True,
                          check=True, cwd=REPO_ROOT).stdout.strip()


def git_dirty() -> bool:
    out = subprocess.run(["git", "status", "--porcelain", "--", ".", ":!evals/published"],
                         capture_output=True, text=True, check=True, cwd=REPO_ROOT).stdout
    return bool(out.strip())


def _readme(name: str, created: str, commit: str, corpus: Path, reports: list[Path], note: str) -> str:
    lines = [
        f"# Judge calibration snapshot: {name}",
        "",
        f"Created {created} (UTC) from commit `{commit}`.",
        "",
        "Frozen copies of the working files in `evals/results/`, which are gitignored because",
        "every run rewrites them. Nothing here is regenerated; it is what the number was",
        "computed from.",
        "",
        "## Contents",
        "",
        f"- `trajectories.jsonl` -- the labelled corpus (`{corpus.name}`), sanitised",
        "- `judge-calibration.json` -- human labels and the judge's verdicts, per question",
        "- `score.txt` -- `calibrate_judge score` run on the file above",
        *[f"- `{r.name}` -- the report written by the run behind the corpus (last run of its "
          "invocation only)" for r in reports],
        "",
        "## Re-check the score",
        "",
        "No model call needed; the verdicts are in the file:",
        "",
        "```",
        f"python -m evals.calibrate_judge score --file evals/published/{name}/judge-calibration.json",
        "```",
        "",
        "## Removed before publication",
        "",
        "None of these is read by the score.",
        "",
        "- `measures_path` in each trajectory: a local temp path containing the user name.",
        "- The user name inside any other local path, replaced by `<user>` (tool error messages",
        "  quote the temp path a write failed on; neither rater is shown tool messages).",
        "- OpenAI `encrypted_content` inside agent messages that call a tool: opaque reasoning",
        "  blobs. No spoken reply carries one, so nothing either rater judged changed.",
        "- `human_note` on every label: withheld at the labeller's choice.",
        "- The local absolute path in the label file's `source`.",
        "- LangSmith experiment links in reports: account ids, and private to that account.",
    ]
    if note:
        lines += ["", "## Notes", "", note]
    return "\n".join(lines) + "\n"


def publish(
    name: str,
    reports: list[Path],
    note: str = "",
    *,
    label_path: Path = LABEL_PATH,
    corpus: Path | None = None,
    published_dir: Path = PUBLISHED_DIR,
    commit: str | None = None,
    dirty: bool | None = None,
) -> Path:
    if not re.fullmatch(r"[a-z0-9][a-z0-9-]*", name):
        raise SystemExit("--name: lower-case letters, digits and hyphens only")
    if (git_dirty() if dirty is None else dirty):
        raise SystemExit("The working tree has uncommitted changes. Commit first: the README "
                         "records the commit, and it has to be the code that made this.")
    commit = commit or git_commit()
    corpus = corpus or labelled_corpus()
    if corpus is None or not corpus.exists():
        raise SystemExit("No labelled corpus: the label file's source does not point at a local run.")

    created = datetime.now(timezone.utc)
    folder = published_dir / f"{created:%Y%m%d-%H%M%S}-{name}"

    labels = json.loads(label_path.read_text(encoding="utf-8"))
    files: dict[str, str] = {
        "trajectories.jsonl": "".join(
            json.dumps(sanitise_trajectory(r), ensure_ascii=False) + "\n" for r in load_trajectories(corpus)
        ),
        "judge-calibration.json": json.dumps(sanitise_labels(labels, corpus.name), indent=2, ensure_ascii=False),
    }
    for report in reports:
        files[report.name] = sanitise_report(report.read_text(encoding="utf-8"))

    # The score is computed from the sanitised label file, so what score.txt
    # says is exactly what anyone re-running it on the published file gets.
    with tempfile.TemporaryDirectory() as scratch_dir:
        scratch = Path(scratch_dir) / "judge-calibration.json"
        scratch.write_text(files["judge-calibration.json"], encoding="utf-8")
        buffer = io.StringIO()
        with contextlib.redirect_stdout(buffer):
            score(path=scratch)
        files["score.txt"] = buffer.getvalue()

    files["README.md"] = _readme(folder.name, f"{created:%Y-%m-%d %H:%M:%S}", commit, corpus, reports, note)

    leaks = find_personal(files, personal_markers())
    if leaks:
        raise SystemExit("Refusing to publish -- personal data would be included:\n  " + "\n  ".join(leaks))

    folder.mkdir(parents=True, exist_ok=False)
    for filename, text in files.items():
        with (folder / filename).open("x", encoding="utf-8", newline="\n") as handle:
            handle.write(text)
    return folder


def sanitise_result_file(path: Path) -> str:
    """One eval-run result file, cleaned by kind.

    Trajectories and baseline rows both carry `outputs` per line, so they go
    through the same trajectory sanitiser; reports lose their LangSmith line;
    anything else (a majority file) only has user names scrubbed. The final
    personal-data check still runs over all of it.
    """
    text = path.read_text(encoding="utf-8")
    if path.suffix == ".jsonl":
        return "".join(
            json.dumps(sanitise_trajectory(json.loads(line)), ensure_ascii=False) + "\n"
            for line in text.splitlines() if line.strip()
        )
    if path.suffix == ".md":
        return _without_user_name(sanitise_report(text))
    return _without_user_name(text)


def _runs_readme(name: str, created: str, commit: str, files: list[Path], note: str) -> str:
    lines = [
        f"# Eval runs snapshot: {name}",
        "",
        f"Created {created} (UTC) from commit `{commit}`.",
        "",
        "Frozen copies of eval-run outputs from `evals/results/`, which is gitignored because every",
        "run rewrites it. Nothing here is regenerated.",
        "",
        "## Contents",
        "",
        *[f"- `{f.name}`" for f in files],
        "- `summary.md` -- every run recounted offline (`python -m evals.summarize_runs`); a run's own",
        "  report covers its last run only",
        "",
        "## Re-check",
        "",
        "The recount in `summary.md` was made from the unsanitised local files at the commit above.",
        "Running `python -m evals.summarize_runs` on the files here reproduces it except for",
        "`expect_lookup_rows`: that check re-reads each case's own copy of the measure list, whose",
        "path is removed below and which exists only on the machine that ran the case.",
        "",
        "## Removed before publication",
        "",
        "- `measures_path` in each trajectory: a local temp path containing the user name.",
        "- The user name inside any other local path, replaced by `<user>`.",
        "- OpenAI `encrypted_content` inside agent messages that call a tool: opaque reasoning",
        "  blobs. No spoken reply carries one.",
        "- LangSmith experiment links in reports.",
    ]
    if note:
        lines += ["", "## Notes", "", note]
    return "\n".join(lines) + "\n"


def publish_runs(
    name: str,
    files: list[Path],
    summary: str,
    note: str = "",
    *,
    published_dir: Path = PUBLISHED_DIR,
    commit: str | None = None,
    dirty: bool | None = None,
) -> Path:
    """Freeze a set of eval runs -- trajectories, baseline rows, majority files, reports."""
    if not re.fullmatch(r"[a-z0-9][a-z0-9-]*", name):
        raise SystemExit("--name: lower-case letters, digits and hyphens only")
    if (git_dirty() if dirty is None else dirty):
        raise SystemExit("The working tree has uncommitted changes. Commit first: the README "
                         "records the commit, and it has to be the code that made this.")
    commit = commit or git_commit()
    created = datetime.now(timezone.utc)
    folder = published_dir / f"{created:%Y%m%d-%H%M%S}-{name}"

    out: dict[str, str] = {path.name: sanitise_result_file(path) for path in files}
    out["summary.md"] = summary
    out["README.md"] = _runs_readme(folder.name, f"{created:%Y-%m-%d %H:%M:%S}", commit, files, note)

    leaks = find_personal(out, personal_markers())
    if leaks:
        raise SystemExit("Refusing to publish -- personal data would be included:\n  " + "\n  ".join(leaks))

    folder.mkdir(parents=True, exist_ok=False)
    for filename, text in out.items():
        with (folder / filename).open("x", encoding="utf-8", newline="\n") as handle:
            handle.write(text)
    return folder


def main() -> None:
    parser = argparse.ArgumentParser(description="Publish a judge-calibration or eval-runs snapshot")
    parser.add_argument("--name", required=True, help="Short slug, e.g. grounding-calibration.")
    parser.add_argument("--report", type=Path, action="append", default=[],
                        help="A report of the run(s) behind the corpus. Repeatable.")
    parser.add_argument("--note", default="", help="Text for the README's Notes section.")
    parser.add_argument("--note-file", type=Path, default=None,
                        help="Read the Notes section from a file (PowerShell mangles multi-line "
                             "arguments that contain quotes).")
    parser.add_argument("--runs", action="store_true",
                        help="Publish eval runs instead of a calibration: pass the files with --file, "
                             "and --full/--n1 for the offline recount.")
    parser.add_argument("--file", type=Path, action="append", default=[],
                        help="An eval-run result file to publish (with --runs). Repeatable.")
    parser.add_argument("--full", type=Path, default=None, help="The full agent's trajectories (with --runs).")
    parser.add_argument("--n1", type=Path, default=None, help="An N-1 run's trajectories (with --runs).")
    args = parser.parse_args()
    note = args.note_file.read_text(encoding="utf-8").strip() if args.note_file else args.note
    if args.runs:
        if args.full is None:
            parser.error("--runs needs --full, the trajectories the recount is made from")
        from evals.summarize_runs import render

        # The recount's own inputs are always part of the snapshot.
        files = list(dict.fromkeys([args.full, *([args.n1] if args.n1 else []), *args.file]))
        folder = publish_runs(args.name, files, render(args.full, args.n1), note)
    else:
        folder = publish(args.name, args.report, note)
    print(f"Published {folder}")
    for path in sorted(folder.iterdir()):
        print(f"  {path.name}  ({path.stat().st_size} bytes)")


if __name__ == "__main__":
    main()
