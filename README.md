# Stanford AI Experiment

A locally runnable research MVP for observing how language-model agents interact under controlled authority structures. Python 3.12+, the official OpenAI SDK, Pydantic, SQLite, Streamlit, Pandas and Plotly. No multi-agent framework.

**Status: engineering MVP, not experimental findings.** The mock backend is a seeded fixture sampler for testing software. No real LLM study has been conducted or claimed. This project is not affiliated with Stanford University and does not involve human participants.

## Start here (Windows PowerShell)

```powershell
cd C:\Users\Admin\Documents\Projects\Stanford-Prision-AI
py -3.12 -m venv .venv
.venv\Scripts\python.exe -m pip install -e ".[dev]"
.venv\Scripts\python.exe -m streamlit run src/stanford_ai/dashboard/app.py --server.address 127.0.0.1
```

Open <http://localhost:8501>. Leave **mock**, four residents, two days and two rounds selected. Choose A/B/C/D, press **Start experiment**, and open **Live simulation**. The background worker records events as they happen; Stop takes effect after the current API call. Agent profiles, analytics and saved runs refresh using **Refresh notebook**. Live events poll automatically. One dashboard experiment can run at a time per server process.

On macOS/Linux use `python3.12` to create the environment and `.venv/bin/python` in place of `.venv\Scripts\python.exe`. Run commands from the repository root. The existing local folder has the spelling `Stanford-Prision-AI`; the public repository uses `Stanford-Prison-AI`.

## First real LLM simulation

Copy `.env.example` to `.env` and set `OPENAI_API_KEY` locally. Alternatively set `$env:OPENAI_API_KEY` in the terminal. Keys are never stored in experiment records or committed. Restart Streamlit after editing `.env`.

Verify the model's **current** standard uncached input/output prices in the [official model documentation](https://developers.openai.com/api/docs/models/gpt-4.1-mini). The example below uses the documented $0.40/$1.60 per million token rates; change both flags if your model/pricing differs. The pinned model's availability depends on your account. The SDK uses the [Responses API with strict structured outputs](https://developers.openai.com/api/docs/guides/structured-outputs); API errors never trigger a model substitution.

```powershell
.venv\Scripts\python.exe -m stanford_ai run --config configs/neutral_unequal.yaml --backend openai --model gpt-4.1-mini-2025-04-14 --n-agents 4 --days 1 --rounds-per-day 1 --seed 42 --max-requests 4 --max-tokens 100000 --budget-usd 0.25 --input-usd-per-million 0.40 --output-usd-per-million 1.60 --confirm-pricing
```

This requests four decisions, one per agent. Budget checks can end it earlier. In the dashboard, select `openai`, supply the same controls, and check the rate confirmation before Start. For models without temperature support, use the dashboard's **Omit temperature** option or set `temperature: null` in a YAML config. No API key is needed for tests, mocks or replay.

## Experimental conditions

| Condition | Agent-visible labels | Administrative authority |
| --- | --- | --- |
| A | Guards / prisoners | Group A only |
| B | Group A / Group B | Group A only |
| C | Guards / prisoners | Both groups |
| D | Group A / Group B | Both groups |

All other initial mechanics, goals, randomized traits, information rules and resource supplies are paired at matching seeds. Labels are a distinct observation field, not a permission source. B and D agent prompts do not mention prison, Stanford, the hypothesis or authoritarianism. Versioned YAML configs are in `configs/`; the complete resolved configuration is saved per run.

Four residents is the cheap default; `--n-agents 20` creates ten residents in each randomized group. Any even size from 2–100 is supported. `--days 10`, `30` or `100` changes simulated time, not elapsed time. Each day contains configurable rounds and every resident gets one opportunity per round in a seeded shuffled order. Planned requests = agents × days × rounds, plus any retries. Increase budgets explicitly for larger studies.

## Facility and agent decisions

Agents choose their own structured action through a separate LLM call on each turn. Six randomized personality tendencies and individual activity preferences are visible to that agent; they do not select its actions in code. There is no aggression reward, no prescribed rebellion and no historical script. The mock backend intentionally samples synthetic actions and must never be used as behavioural evidence.

The deterministic engine supports communication (public/private), rules and advisory votes, resource requests/allocation, task assignment/volunteering/completion, exception requests and appeals, negotiation and bilateral agreements, challenges, activity order changes, limited amenity restrictions, amenity use and no-op. Capabilities are controlled by `environment/permissions.py`.

Every day adds two discretionary leisure credits per resident to a common pool. Allocations have a base daily cap of three per resident; approved typed rules can reduce that cap for a group or everyone. One credit buys one amenity use. A restriction blocks one discretionary amenity until the next day. No food, shelter, medical care, physical force or extreme abuse is simulated. Daily tasks create a shared coordination opportunity. Schedule order is recorded and changeable but does not gate access in this MVP. Agreements record terms and consent; they cannot transfer resources or silently change policies. Appeals can reverse active rules, current restrictions or unfinished current assignments and require an independent administrator. In a two-agent unequal run no independent reviewer exists; use at least four agents for appeal experiments.

## Data, budgets and replay

The default database is `outputs/experiments.sqlite`; override with `STANFORD_AI_DB` or the CLI's global `--db` option **before** the subcommand. `outputs/` and `.env` are ignored by Git. The local observer dashboard can inspect private conversations; agents can only inspect their own permitted views.

SQLite records resolved config, initial state, traits, prompts/observations, strict response schema, raw and parsed responses, provider model/usage, action validation, state hashes, changed fields, replay snapshots, timestamps, metrics, source-file hashes, package versions and Git revision. A request reservation is persisted **before** contacting the API. Raw events/requests and processed metrics use separate tables/files. The event stream and snapshots are committed atomically. Failed actions remain recorded and cannot mutate facility mechanics.

```powershell
# No spending: short engineering smoke run
.venv\Scripts\python.exe -m stanford_ai run --backend mock --condition B --n-agents 4 --days 2
.venv\Scripts\python.exe -m stanford_ai list
# Replace RUN_ID with the printed UUID
.venv\Scripts\python.exe -m stanford_ai export RUN_ID --out outputs/exports
.venv\Scripts\python.exe -m stanford_ai replay RUN_ID --seq 3
.venv\Scripts\python.exe -m stanford_ai compare RUN_ID_1 RUN_ID_2
# Explicit repeated trials, incrementing seeds. Each trial has its own full budget.
.venv\Scripts\python.exe -m stanford_ai run --backend mock --condition B --seed 42 --repeats 3
```

Export creates `raw.json` (including all snapshots), `actions.csv`, `requests.csv`, `metrics.json` and `metrics.csv`. Replay restores the saved event position without API calls. The dashboard also prepares a downloadable JSON/CSV zip. Raw exports contain all private messages and exact prompts; treat them as privileged observer data. CSV text is untrusted agent output; use a data import workflow if opening it in a spreadsheet application.

Hard request limits include retries. Before each paid call, token and dollar limits must cover a conservative UTF-8 byte envelope for the prompt/schema plus 4096 tokens of overhead, and the full output allowance. Reported usage releases unused capacity; unknown billing keeps the full reservation. The uncached estimate deliberately overestimates cache discounts. Unsupported models, timeouts, missing usage and exhausted budgets stop safely; they do not generate substitute agent actions. Only rate limits and schema-invalid/refused/incomplete outputs receive bounded retries with identical prompts. SDK automatic retries are disabled. Exact actual costs still depend on provider billing and the supplied prices; an application estimate is not a provider-enforced account spending cap. Input token envelopes are conservative engineering bounds rather than a provider token-count guarantee; an unexpected excess is logged and immediately stops the run.

Memory compression deterministically folds older visible outcomes into action/success counts while retaining the configured number of recent detailed memories. Full raw history remains in SQLite. This is lossy compression, not an LLM-generated autobiographical summary. A seed reproduces initialization, order and mock decisions, **not real LLM outputs**.

If the process is killed, its run may remain `running`. After confirming the original process is gone, use `stanford-ai recover RUN_ID` to mark it interrupted and retain unknown reservations. It does not resume or replay API calls. Automatic continuation after a crash is intentionally unsupported.

## Analysis and interpretation

Analytics show restrictive actions, cooperation/conflict, rules, daily allocation inequality, group and individual activity, and initial/later periods. Comparisons show individual run points, box distributions and variability. For complete matched A/B/C/D seed blocks, the dashboard reports exploratory authority/label/interaction contrasts with a run-block bootstrap interval. It rejects mixed generation, memory, budget, size, duration, backend or model settings for these contrasts. Single-block intervals are omitted. Rows/turns are not independent experimental replications.

The [behaviour rubric](docs/BEHAVIOUR_RUBRIC.md) defines every proxy. “Authoritarian” is a repeated restrictive-action descriptor, not evidence of motives or consciousness. The MVP uses no LLM judge and makes no language-based abuse claims. See [RESEARCH_DESIGN.md](RESEARCH_DESIGN.md) for confounders, preregistration guidance and limitations, and [FUTURE_RESEARCH.md](FUTURE_RESEARCH.md) for publication-oriented work.

## Verification and development

```powershell
.venv\Scripts\python.exe -m pytest --basetemp outputs/pytest-tmp -p no:cacheprovider
.venv\Scripts\ruff.exe check src tests
.venv\Scripts\python.exe -m build
```

Tests exercise permission matrices, label controls, privacy, resources, rules, consent, appeals, bounded memory, deterministic mocks, persistence/hash chains, exports/replay, budgets/retries/unknown billing, factorial analysis, real-SDK transport with stubbed HTTP, and Streamlit setup/viewer/profiles/analytics/replay. They require no network credentials. CI runs on Windows and Linux with Python 3.12 and 3.14. `requirements-lock.txt` records the validated Windows/Python 3.12 dependency set; `pyproject.toml` remains the portable dependency definition. [VALIDATION.md](docs/VALIDATION.md) records the delivered engineering checks and their limits.

```text
configs/                       Paired factorial condition presets
src/stanford_ai/
  models.py                    Pydantic configuration, actions, agents, world
  agents/                      Neutral prompts, permitted views, OpenAI/mock backends
  environment/                 Permission matrix and deterministic engine
  simulation/                  Scheduler, budget reservations, provenance
  experiments/                 Initialization, exports, paired factorial analysis
  metrics/                     Action-based measurements and rubric implementation
  storage/                     SQLite event/usage/snapshot persistence
  dashboard/                   Streamlit UI and local background jobs
tests/                         Unit, integration and dashboard acceptance tests
outputs/                       Local data only; ignored by Git
```

The dashboard binds to localhost in the documented command. It has no authentication and is intended for one trusted local researcher. Large runs can generate substantial SQLite/snapshot/export data; they do not run automatically.
