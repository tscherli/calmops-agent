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

- `--studio-url` is required: this LangSmith account is in the EU region and the CLI defaults to the
  US site. The command normally opens Studio already connected.
- The server reloads when you save `react_agent.py`. Start a **new thread** after a change: an old
  thread still carries replies, and state fields, from the previous version.

**If Studio shows "Configure Studio connection":** Base URL `http://127.0.0.1:2024` (not `localhost`,
which can resolve to IPv6 first), Custom Headers empty, Advanced Settings unchanged, then **Connect**
and pick the **`react_agent`** graph. The dialog only works while the server is running.

**If it says "Failed to initialize Studio / Failed to fetch"** while `http://127.0.0.1:2024/docs`
opens fine: Chrome 142+ blocks an HTTPS page from reaching your machine. On `eu.smith.langchain.com`,
click the site-info icon, set **Local network access** to **Allow**, reload. If it still fails, try
with browser extensions disabled.

## What the agent does now

```
START --(short "not helpful")--> count_rejection --> 1st: web_search_ideas --> END
  |                                                  2nd: ask_goal_again ----> END
  |--(user's own idea open)----> debate -----+
  |--(energy or goal unknown)--> ask --------+
  |--(nothing suggested yet)---> suggest ----+--> tools --> back to the station
  +--(suggestions made)--------> follow_up --+
```

Every station gets `BASE_PROMPT` (who it is, tone, honesty) plus its own short prompt, and binds only
the tools it needs. What is not bound cannot happen: `ask` cannot suggest anything.

### Stations

| Station | Job | Prompt | Tools | Model |
|---|---|---|---|---|
| `ask` | find out energy and goal; a tentative answer ("maybe changing the situation") counts | `ASK_PROMPT` | `record_choices`, `discuss_idea` | nano |
| `suggest` | look up measures and present at most two, one sentence why each | `SUGGEST_PROMPT` | `find_measures` (forced until there is a fresh result), `record_choices`, `discuss_idea`, web search | nano |
| `follow_up` | rejection, more ideas, adding a measure, the confirmed check-in row, a later rating | `FEEDBACK_PROMPT` + `LOG_PROMPT` | all except `idea_decided`, plus web search | nano |
| `debate` | an honest sparring partner for an idea the user brought: one strength, one limit, a comparison, a small test; stays until the user decides | `DEBATE_PROMPT` | `add_measure`, `record_choices`, `idea_decided` | **mini** |
| `web_search_ideas` | after the first rejection: a model call forced to search, links required | `SEARCH_INSTRUCTION` | web search (forced) | nano |
| `ask_goal_again` | after the second rejection: a fixed question; clears the goal, resets the counter | none | none | no model call |

### State

The graph remembers these, so the model never has to re-derive them from the conversation:

| Field | Meaning |
|---|---|
| `energy`, `goal` | the user's own answers, written by `record_choices` |
| `offered` | measure ids `find_measures` has already returned, so nothing is shown twice |
| `rejections` | rejections in a row; reset by the second rejection and by logging a measure |
| `step` | the active station, so a tool result returns to the station that asked for it |
| `idea` | the user's own idea while it is being discussed; cleared by `idea_decided` |

### Tools

| Tool | What it does |
|---|---|
| `record_choices(energy, goal)` | writes both into the state; also how the user changes either later |
| `find_measures()` | reads energy and goal from the state, filters `measures.csv` in code, returns up to four measures not shown yet; empty means nothing is left, so the agent searches the web |
| `discuss_idea(idea)` / `idea_decided(outcome)` | open and close the `debate` station (`add`, `try` or `drop`) |
| `add_measure(...)` | appends to `measures.csv`; `source` is `user`, `agent` or `web:<url>`. **No approval step and no duplicate-id check while testing in Studio**; both return in the production port |
| `log_entry(...)` | appends to `data/sandbox_checkin_log.csv`, never to the real log: a check-in row (energy, a situation sentence the user confirmed, `measure_id`) once a measure is settled, and later a rating row with only `measure_id` and `helpfulness`, enforced in code. Unknown ids are refused |
| `suggestion_not_helpful()` | the model's way to count a rejection |
| web search | OpenAI's built-in tool; runs inside a model call, so it never shows in the `tools` node |

### Rejections

- **Code first:** a keyword check (`NOT_HELPFUL_PHRASES`) runs before any model call, but only for
  messages of at most 60 characters, only once something was suggested, and not while an idea is
  being debated. A long message that also says something new goes to the model.
- **Model second:** `suggestion_not_helpful` for anything the list misses.
- **Code decides the consequence:** the first rejection forces a web search, the second asks for the
  goal again.

### Models

- `gpt-5-nano` for every station except `debate`, at `reasoning_effort="low"`: `minimal` is refused by
  OpenAI together with web search.
- `gpt-5-mini` for `debate` (override with `OPENAI_DEBATE_MODEL`): honest pushback instead of automatic
  agreement is where small models are weakest.
- `use_responses_api=True` on both, so every station can read a history that contains web-search
  results.

### Know before you test

- **No crisis screening in here.** In production `screen_for_crisis` runs before the graph; Studio
  invokes the graph directly and skips it. Don't read anything into how this graph handles crisis
  language.
- **Studio makes real API calls** with the same key as the app. Cheap, not free.

## `measures.csv`

The list the agent may suggest from. Curated by hand; the agent appends through `add_measure`.

| Column | Meaning |
|---|---|
| `measure_id` | short lowercase id; the log refers to it, so never rename one that has been logged |
| `type` | `recover` helps you recover (rest, calm down); `act` changes the cause of the stress |
| `measure` | one sentence, plain ASCII, in neutral words: no roles, no specific situations |
| `fits_energy` | the energy values it suits, in the order `exhausted;calm;wired`; a judgement call, change it when your experience says otherwise |
| `effort` | `minutes`, `hours` or `days`; keeps a weekend away from being suggested as a break for right now |
| `source` | `stretchly:...`, `user`, `agent` or `web:<url>` |

**The goal filter:** calm down now = `recover` + `minutes`; plan recovery = `recover` +
`hours`/`days`; change the situation = `act`.

**Stretchly rows** come from Stretchly's break ideas under the BSD 2-Clause licence. Keep
[third_party/stretchly/LICENSE](third_party/stretchly/LICENSE) and
[NOTICE.md](third_party/stretchly/NOTICE.md) next to the list, and record further changes to those
rows in the notice.

## Environment

- **Own venv, `sandbox/.venv`**: installing `langgraph-cli[inmem]` into the app's `.venv` would
  downgrade `protobuf` under the production code. Core libraries are pinned to the production
  versions in `requirements.txt`.
- **`langgraph.json` lives here**, so `langgraph dev` points at the experiment, never at the
  production graph. It reads the project's `../.env`, so there are no copied secrets; a plain
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
