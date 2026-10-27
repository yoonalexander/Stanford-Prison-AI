# Delivered engineering validation

## v0.2 model compatibility validation · 2026-10-26

The expanded suite passes 66 tests, including local-schema transport, cloud/missing-model/digest rejection, independent model routing, per-agent rates with shared budgets, explicit compatible JSON modes/token parameters, model-identity prompt isolation, zero-dollar paid-route blocking, context-envelope enforcement and dashboard roster controls. Linting remains clean. Earlier v0.1 checks below describe that initial delivery.

Actual locally installed LLMs were also exercised with no hosted API calls or API charges:

| Engineering smoke run | Decisions | Reported input / output tokens | API cost |
| --- | --- | --- | --- |
| Gemma 3 4B, 2 agents, 1 day/round | 2 | 3950 / 173 | USD 0 |
| Gemma 3 4B + Qwen 3.5 4B, 4 agents, 1 day/round | 4 | 9382 / 407 | USD 0 |
| Gemma 3 4B, 4 agents, 2 days × 2 rounds | 16 | 54897 / 1599 | USD 0 |

All three completed with provider token counts, raw structured responses, validation results and local model digests/version recorded. These real local decisions are **integration smoke traces, not confirmatory experimental findings**. Machine-specific raw records remain under ignored `outputs/`. Repeating them can produce different actions/token counts. The OpenAI-compatible hosted adapter is verified with stubbed HTTP; no external hosted provider call was made.

## Initial v0.1 delivery

Validated locally on Windows with Python 3.12 on 2026-10-26. No remote OpenAI calls or paid behavioural trials were made. All simulation outcomes below are explicitly synthetic software checks, not findings about authoritarianism, cooperation or the research hypotheses.

## Automated acceptance

The Pytest suite covers all permission cells, unauthorized administrative attempts with no facility mutation, exact label-independent initial states and capabilities, neutral prompts, private conversation/goal isolation, conserved resources and enforced caps, rules/votes, expiring restrictions and independent appeals, bilateral negotiation, task eligibility, bounded persistent memory, JSON validation, request/token/dollar limits, retries and retained unknown billing, exports, deterministic mock repeats, complete replay/hash chains, paired factorial analysis, the official OpenAI SDK through stubbed HTTP, and Streamlit setup/background execution, profiles, charts, comparison and sequential replay.

Run the suite with:

```powershell
.venv\Scripts\python.exe -m pytest --basetemp outputs/pytest-tmp -p no:cacheprovider
.venv\Scripts\ruff.exe check src tests
```

The delivered suite contains 45 passing tests. Streamlit emits a third-party NumPy/Pandas timedelta deprecation warning during tests; no application exceptions or test failures occurred. CI is configured for Windows/Linux and Python 3.12/3.14; those remote jobs are separate from the local validation.

## Synthetic end-to-end checks

- A/B/C/D, each at seeds 42, 43 and 44: 12 completed runs, four agents, two days, two rounds, 16 decisions each. Verified every saved snapshot against its state hash and preceding hash, JSON/CSV exports, and complete three-block factorial input handling.
- Twenty-agent support: completed one day with two rounds, 40 decisions and balanced 10/10 membership.
- Hundred-day support: completed 100 simulated days with two agents and one round/day, 200 decisions. Request cap increased explicitly to 300.
- The default local notebook contains a clearly labeled four-agent neutral/unequal mock sample; no raw data is committed to GitHub.
- Wheel and source distribution build successfully. Source distribution includes configs, tests, conftest and research documentation; installation and CLI do not require API credentials.
- Inspected the running Streamlit dashboard at `http://127.0.0.1:8501` using a browser, including condition controls and the live observer view.

Local generated evidence is under ignored `outputs/`, including `acceptance-summary.json`, `acceptance.sqlite` and `acceptance-exports/`. These files are engineering traces only. No key or private experiment log is included in the public repository.

## Limits of this validation

The real SDK transport test verifies request/response integration against a mocked HTTP response; it does not prove live account/model availability, current provider prices, network resilience at production scale or scientific external validity. No live model study was run. Token/dollar reservations are conservative local controls rather than provider invoice guarantees. No human annotation, psychometric validation, emergence identification or statistical power claim is made. The dashboard is intended for a trusted local researcher.
