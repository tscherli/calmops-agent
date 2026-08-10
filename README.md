# Job Load Companion

> ⚠️ **Data & Privacy Disclaimer**: This tool shares what you type — including personal reflections on your mood and work situations — with third-party services: OpenAI's API (to power the agent) and Google Sheets (to log entries), and optionally LangSmith (for tracing). **If you are not comfortable with a large language model and these services processing your personal information, do not use this software** — the quick-select buttons let you log a mood without sharing anything personal. This is a self-reflection aid, not a substitute for professional mental health support; its replies are AI-generated and may be inaccurate. In a crisis, contact local emergency services or a crisis hotline directly, not this tool. Each user of this software supplies their own API keys/credentials — no data from anyone running this passes through the original developer's infrastructure.

## What it does

_TODO: fill in once the agent flow is built._

## Target user & scope

_TODO_

## Architecture

_TODO: component summary._

### Agent decision flow

This is the decision tree the agent follows for every incoming message. It's the spec `agent/graph.py` is built against, not just a post-hoc diagram — implemented so far: the crisis gate (`agent/safety.py`) and the agent itself (`agent/graph.py` + `agent/prompts.py`, GPT + `log_mood` bound via `langchain.agents.create_agent`). What's still pending is everything that actually drives a real conversation with it: `app.py` (the Tkinter UI) and `scheduler.py`.

```mermaid
flowchart TD
    A[User message] --> B{Quick-select button<br/>or free text?}
    B -- button --> C[log_mood called directly<br/>with fixed template values]
    C --> Z1[No LLM involved at all]

    B -- free text --> D{Crisis gate<br/>agent/safety.py<br/>pre-LLM, deterministic}
    D -- triggered --> E[Hardcoded crisis resources<br/>END - no LLM, no tool call]
    D -- clear --> F[langchain.agents.create_agent<br/>GPT + log_mood tool]

    F --> G{What kind of message?}
    G -- small talk / off-topic --> H[Plain conversational reply<br/>no tool call]
    G -- heavy / overwhelming signal --> I[Structuring sequence:<br/>situation -> thoughts -> need -> next step<br/>across as many turns as needed]
    G -- light check-in --> J[Log directly:<br/>valence / energy_level / circumstances /<br/>measure_taken / helpfulness]
    G -- ambiguous --> K[Ask a clarifying question<br/>instead of guessing]

    I --> L[log_mood tool call]
    J --> L
    K --> G

    L --> M{Sheets write<br/>succeeded?}
    M -- yes --> N[Honest confirmation reply]
    M -- no --> O[state/pending_sheet_log.jsonl fallback<br/>+ honest failure message]
```

**Key rule embedded in this diagram:** the agent must never claim `log_mood` succeeded unless it actually did — that's why both the success and failure branches at the bottom end in an *honest* reply, never a guessed one.

### Why `energy_level` exists

The Sheet schema includes an optional `energy_level` field (`low`/`medium`/`high`), separate from `valence` (`good`/`neutral`/`bad`). These are deliberately two different axes, not duplicates: you can be low-energy-but-content (a relaxed evening) or high-energy-but-stressed (anxious, wired), so collapsing them into one field would lose information.

The motivation is emotional exhaustion — one of three dimensions identified in burnout research (alongside cynicism/detachment and reduced sense of accomplishment) as distinct from day-to-day mood, and specifically relevant to this app's target user (someone in a stressful, high-interrupt job). **This app is not a validated burnout assessment and does not diagnose anything.** It borrows the *concept* that fatigue and mood are related but separate signals worth tracking separately, purely as a self-reflection aid — consistent with the rest of this project's non-clinical framing (see the disclaimer at the top of this document, and "Results & limitations" below).

For an example of how to reference this kind of research responsibly (cite the concept, disclaim clinical validity), see [psychology.com's burnout self-assessment](https://psychology.com/tests/burnout), which states plainly that it's "an honest educational adaptation... for reflection, not a validated diagnostic score" — the same framing this project follows.

**Deliberately excluded: the third burnout dimension.** Burnout research identifies three dimensions: emotional exhaustion (what `energy_level` tracks), depersonalization/cynicism (a shift in how you relate to *other people* at work — colleagues, clients), and reduced sense of accomplishment. This app intentionally logs only the relationship-to-self dimension (exhaustion, via `valence`/`energy_level`) — not the relationship-to-others dimension, and not accomplishment. That's a deliberate scope boundary, not an oversight: cynicism/detachment isn't a clean single-scale value the way energy is, and a nearby but distinct concept — general apathy/loss of interest (anhedonia) — is a core depression symptom, not a burnout-dimension proxy; tracking it would push this tool toward symptom screening, which is out of scope for a workplace mood diary. See "Limitations" below.

### Architectural decisions

**Log backend: local CSV, not Google Sheets/OAuth.** Originally planned as a Google Sheets integration via OAuth. Real-world setup turned out to be significantly more friction than a personal/solo-developer tool should require (Google's OAuth consent-screen UI was recently restructured, several steps deep to locate the right screens). Alternatives considered: a Google service account (still needs Cloud Console setup), `gcloud` CLI credentials (needs installing a new tool), local CSV synced via Google Drive Desktop (real sync-race risk, no official Linux client), and a GitHub PAT via Gist/repo API (simplest auth, but a flat file with no live-spreadsheet UI, and a bigger pivot from what's disclosed as the data recipient). **Decision: a local CSV file, no external service at all.** Real tradeoff: no live multi-device/cloud access or native Sheets UI — in exchange, zero setup friction and zero external credentials for the app to hold, actually a *stronger* security posture than every cloud-backed option considered. Full reasoning and rejected alternatives: `DEVELOPMENT_LOG.md`.

**Rejected: `create_csv_agent`-style tools for any future CSV analysis feature.** Considered for a hypothetical (not-yet-built) "analyze my mood log" feature. LangChain's `create_csv_agent`/`create_pandas_dataframe_agent` work by giving the LLM a *local* Python REPL to execute arbitrary generated code — gated behind an explicit `allow_dangerous_code=True` flag for good reason: there's a published security advisory ([GHSA-3645-fxcv-hqr4](https://github.com/langflow-ai/langflow/security/advisories/GHSA-3645-fxcv-hqr4)) describing prompt-injection-triggered Remote Code Execution via exactly this pattern. That directly conflicts with this project's "bounded tool surface" safety rule. If that feature is ever built, OpenAI's Code Interpreter is the safer analog — it runs LLM-generated code in an isolated, ephemeral, remote sandbox instead of the user's own machine.

**Eval suite (stages 9-10, not yet built): LangSmith's `evaluate()` over hand-rolled runner.** Researched rather than assumed — confirmed `Client.create_dataset()`/`create_examples()` and `client.evaluate()` are real, current APIs in the installed `langsmith` package, and that structural tool-call verification (this project's 0%-tolerance fabricated-success-claim check) is natively achievable via a plain evaluator over `run.child_runs`, no extra tooling needed for that part.

## Setup prerequisites

You'll need a personal (not employer/work) Google account and an OpenAI API key. Each user of this software sets up their own — nothing here passes through the original developer's infrastructure.

### Google Sheets access (OAuth)

1. **Create/select a Google Cloud project** at [console.cloud.google.com/cloud-resource-manager](https://console.cloud.google.com/cloud-resource-manager) (that link reflects one specific project/account session — if it doesn't land you in the right place, just go to [console.cloud.google.com](https://console.cloud.google.com) and create or pick a project from there instead). Use a personal account, not a work one.
2. **Enable two APIs** at [console.cloud.google.com](https://console.cloud.google.com/welcome) by making sure your current project is selected. Go to "APIs & Services" → "Library" → enable both the **Google Sheets API** and the **Google Drive API** (the `drive.file` scope this app uses is governed by the Drive API, even though we're only writing to Sheets).
3. **Configure the Google Auth Platform** (Google's current name for what used to be called the "OAuth consent screen" — if the left sidebar shows Overview / Branding / Audience / Clients / Data Access / Verification Center / Settings, you're in the right place):
   - If you see "Google Auth Platform not configured yet," click **Get started**. It's a single wizard with a few steps: under **App Information**, enter any app name + your email as support email → Next. Under **Audience**, choose **External** (this is what a personal, non-Workspace Google account needs — not "Internal") → Next. Under **Contact Information**, enter your email again → Next → agree to the terms → **Create**.
   - Once that's done, go to the **Audience** section in the left sidebar and add your own Google account email under **Test users** (required since this app won't go through Google's verification process — only accounts on this list can authorize it).
   - Then go to the **Data Access** section in the left sidebar → **Add or remove scopes** → add `https://www.googleapis.com/auth/drive.file` → Save.
4. **Create OAuth Desktop credentials** — left sidebar → **Clients** → **Create Client** → Application type **Desktop app** → name it anything → **Create** → download the JSON.
5. **Place the file** — rename the downloaded file to exactly `credentials.json` and put it in the project root (already gitignored — never commit the real file, only `credentials.json.example`).
6. **Create the actual Sheet** — a new, private (unshared) Google Sheet. Rename the first tab to `Log`. Add this exact header row (columns A–H): `timestamp | row_id | valence | energy_level | circumstances | measure_taken | helpfulness | note`. Copy the sheet ID from the URL (the string between `/d/` and `/edit`) into `MOOD_SHEET_ID` in your `.env` file (copy `.env.example` to `.env` first).

The first real Sheets write triggers a one-time browser consent screen (`flow.run_local_server`) — click "Allow," and a `token.json` gets cached locally (also gitignored) so you won't see that prompt again.

### OpenAI API key

Create a key at [platform.openai.com](https://platform.openai.com/api-keys) and set `OPENAI_API_KEY` in `.env`.

## How to run

_TODO: `app.py` on demand, `scheduler.py` for occasional nudges._

## How the tool works

_TODO_

## How to run evals

_TODO_

## Results & limitations

_TODO — filled in after the eval suite runs._

**Limitations (stated explicitly):**
- Mood/measure detection is entirely self-report, not sensed.
- The "irregular" popup is a simple randomized local timer, not adaptive.
- Single-user personal tool.
- Helpfulness scores are self-reported and unverified.
- This is a wellness aid, not a clinical tool; the crisis gate is a basic keyword list, not a validated risk-assessment instrument (deliberately biased toward false positives over false negatives).
- `energy_level` is inspired by a concept from burnout research (emotional exhaustion) but is not a validated burnout assessment or diagnostic tool — see "Why `energy_level` exists" above.
- Only the relationship-to-self burnout dimension is tracked (exhaustion). The relationship-to-others dimension (depersonalization/cynicism) and the reduced-accomplishment dimension are deliberately not logged — this tool doesn't attempt to capture how you're relating to colleagues/clients, only your own state.

**Data handling:** this tool records genuinely personal reflections; that content is sent to LangSmith (if tracing is enabled) and to Google Sheets, both third-party services, and cached in local plaintext files that must never be committed to a public repo.

## Phase 2 / not implemented in this version

A break-companion feature (agent-suggested, human-confirmed real break via Stretchly) was fully designed alongside this MVP but deliberately deferred as a scope decision — see the project plan for the full design, including its human-in-the-loop safety model for the one subprocess call it would make.
