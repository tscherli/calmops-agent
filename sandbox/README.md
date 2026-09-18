# sandbox/

**Not production code.** A place to build the agent node by node in LangGraph Studio and see what
each node does. Nothing outside `sandbox/` imports from it; findings come back as deliberate changes
to `agent/`. `tests/test_sandbox_isolation.py` fails the main suite if any production module ever
imports `sandbox`.

## Start Studio

One-time setup, already done for this checkout:

```powershell
cd C:\job-load-agent\sandbox
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

Start the server in your own PowerShell window and leave it open (`Ctrl+C` when done):

```powershell
cd C:\job-load-agent\sandbox
.\.venv\Scripts\langgraph.exe dev --studio-url https://eu.smith.langchain.com
```

`--studio-url` is required because this LangSmith account is in the EU region and the CLI defaults to
the US site. The command normally opens Studio already connected. The server reloads when you save
`react_agent.py`; start a **new thread** after a change, because an old thread still carries replies
from the previous version.

**If Studio shows "Configure Studio connection":** Base URL `http://127.0.0.1:2024` (not `localhost`,
which can resolve to IPv6 first), Custom Headers empty, Advanced Settings unchanged, then **Connect**
and pick the **`react_agent`** graph. The dialog only works while the server is running.

**If it says "Failed to initialize Studio / Failed to fetch"** while `http://127.0.0.1:2024/docs`
opens fine: Chrome 142+ blocks an HTTPS page from reaching your machine. On `eu.smith.langchain.com`,
click the site-info icon, set **Local network access** to **Allow**, reload. If it still fails, try
with browser extensions disabled.

## What the agent does now

```
START --(keyword "not helpful")--> count_rejection --+
  |                                                  |
  v                                                  v
companion <--> tools --(model: not helpful)--> 1st: web_search_ideas --> END
  |                                             2nd: ask_goal_again ---> END
  v
 END
```

- **`companion`** — one model call per turn (`gpt-5-nano`, `reasoning_effort="low"`; `minimal` is not
  possible because OpenAI refuses `web_search` at that setting).
- **Before suggesting, it needs two things:** the user's energy (`exhausted` / `calm` / `wired`) and
  their goal — calm down now, plan something to recover, or change the situation. It asks for what is
  missing.
- **`find_measures(energy, goal)`** filters `measures.csv` in code: the goal maps to `type` and
  `effort`. At most two measures are suggested, each with one sentence on why it fits.
- **Rejections are counted in code.** A keyword check (`NOT_HELPFUL_PHRASES`) runs before the model;
  the model can also call `suggestion_not_helpful` for phrasings not on the list. The first rejection
  routes to **`web_search_ideas`**, a model call forced to search (`tool_choice`); the second routes
  to **`ask_goal_again`**, a fixed question with no model call that resets the counter.
- **`add_measure`** appends a new measure to `measures.csv`, from the web (`source` = `web:<url>`) or
  the agent's own idea (`source` = `agent`). **No approval step while testing in Studio**: the row
  lands as soon as the tool is called. Put an approval back (buttons, not typed words) before this
  reaches the app. Rows are never edited or deleted.
- **`log_entry`** appends to `data/sandbox_mood_log.csv`, never to the real log. Each rating is a new
  row with `measure_id` and `helpfulness` (`helped` / `somewhat` / `didnt_help`); nothing is
  overwritten. A `measure_id` that is not in the list is refused by code.
- **No crisis screening in here.** In production `screen_for_crisis` runs before the graph, so Studio,
  which invokes the graph directly, skips it. Don't read anything into how this graph handles crisis
  language.
- **Studio makes real API calls** with the same key as the app, cheap on `gpt-5-nano` but not free.

## `measures.csv`

The list the agent may suggest from. Curated by hand; the agent appends through `add_measure`.

- **`type`:** `recover` measures help you recover (rest, calm down); `act` measures change the cause
  of the stress.
- **`fits_energy`:** the energy values a measure suits, `;`-separated. The fit is a judgement call;
  change it when your experience says otherwise.
- **`effort`:** `minutes`, `hours` or `days`. It keeps a weekend away from being suggested as a break
  for right now.
- **The goal filter:** calm down now = `recover` + `minutes`; plan recovery = `recover` +
  `hours`/`days`; change the situation = `act`.
- **`source`:** `stretchly:…` rows come from Stretchly's break ideas under the BSD 2-Clause licence.
  Keep [third_party/stretchly/LICENSE](third_party/stretchly/LICENSE) and
  [NOTICE.md](third_party/stretchly/NOTICE.md) next to the list and record further changes to those
  rows in the notice. Your own measures use `user`.

## Environment

- **Own venv, `sandbox/.venv`**: installing `langgraph-cli[inmem]` into the app's `.venv` would
  downgrade `protobuf` under the production code. Core libraries are pinned to the production
  versions in `requirements.txt`.
- **`langgraph.json` lives here**, so `langgraph dev` points at the experiment, never at the
  production graph. It reads the project's `../.env`, so there are no copied secrets — and a plain
  `python -c "import react_agent"` fails on missing credentials, which is expected.
- **Traces go to the `job-load-agent-sandbox` LangSmith project**, out of the way of the eval runs.
- **The main test suite ignores this folder** (`testpaths = ["tests"]`), so a broken experiment can't
  turn the suite red.
- **No checkpointer in the code.** `langgraph dev` runs an Agent Server that provides one, which is
  what makes threads, forking and "re-run from here" work.

## What stays out of git

- **`.langgraph_api/`** is the dev server's own state directory. Studio stores conversation threads
  there, so anything you type into Studio ends up on disk. The root `.gitignore` excludes it.
- **`.venv/`** and **`__pycache__/`** are covered by existing rules.

Run `git status --porcelain -uall sandbox` before committing and look at what is actually listed.
