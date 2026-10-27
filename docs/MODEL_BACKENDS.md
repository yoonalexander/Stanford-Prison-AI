# Models and per-agent assignments · v0.2

Every resident can use its own model. Named profiles specify connections and generation settings; `agent_models` maps stable resident IDs to profiles. Unassigned residents use the experiment default. Profiles never change permissions, goals or private-message visibility.

## Free real LLM option: local Ollama

Select **ollama** in Experiment setup and enter an installed model ID, such as `gemma3:4b`. Set the dollar budget to zero. This runs an actual local LLM for every decision, without a paid API or API key. Hardware, electricity, disk space and loading time still have costs. It is distinct from the synthetic mock backend.

Ollama and local models `gemma3:4b` and `qwen3.5:4b` were already installed on the delivery machine when checked on 2026-10-26. Other machines need [Ollama](https://ollama.com/download) and a locally installed model. This project never downloads models automatically.

```powershell
ollama list
# Only if not already installed; this user-initiated command downloads several GB:
ollama pull gemma3:4b
# Start the Ollama desktop application, or `ollama serve` if needed.
.venv\Scripts\python.exe -m stanford_ai run --config configs/local_free.yaml
```

The preset uses four residents, two days and two rounds with zero API budget. For a tiny test:

```powershell
.venv\Scripts\python.exe -m stanford_ai run --backend ollama --model gemma3:4b --n-agents 2 --days 1 --rounds-per-day 1 --max-requests 4 --max-tokens 100000 --budget-usd 0
```

The native adapter uses Ollama's [schema-constrained output format](https://docs.ollama.com/capabilities/structured-outputs) and [chat endpoint](https://docs.ollama.com/api/chat). It records the installed digest, quantization/details, Ollama version, explicit context size and token counts. It requests `think=false` and records that setting. An optional `model_digest` profile field rejects a changed installed tag before generation.

Free mode accepts a loopback server only and verifies the model is installed locally. It refuses cloud-routed models. Missing models, unsupported parameters and connection errors fail visibly; there is no paid fallback or automatic substitution.

Request/token limits still apply. `local_context_tokens` is sent as `num_ctx`. A conservative input/schema/output envelope must fit before calling the model; otherwise the run stops instead of silently truncating the observation. Explicitly increase context within model/hardware limits or reduce the memory/history controls. The byte envelope can be much larger than the actual token count. Keep context and memory settings fixed across experimental cells.

## Mix local models

```powershell
.venv\Scripts\python.exe -m stanford_ai run --config configs/mixed_local.yaml
```

This assigns Gemma to agents 01/03 and Qwen to 02/04, with four decisions in one round and zero API budget. Both tags must be installed. Calls are sequential within a run, so twenty residents do not require twenty simultaneous model instances. Switching models can increase loading time and memory pressure.

In the dashboard, enable **Assign different models to individual agents**. Define profiles in the first table, then select a profile for each resident in the assignment table. **default** uses the experiment's top-level controls. Profile settings include backend, model, endpoint, credential variable name, temperature, output limit, local context, optional digest and API rates. Blank profile temperature omits it. Role memberships are shown but cannot be edited in this table.

```yaml
backend: ollama
model: gemma3:4b
budget_usd: 0
n_agents: 4
model_profiles:
  another_local_model:
    backend: ollama
    model: qwen3.5:4b
    temperature: 0.7
    max_output_tokens: 400
agent_models:
  agent_02: another_local_model
  agent_04: another_local_model
```

CLI assignment overrides are supported with `--agent-models '{"agent_02":"another_local_model"}'` when that profile is defined in the selected YAML. `--repeats` preserves model-to-ID assignments while incrementing the initialization seed.

## OpenAI and compatible endpoints

`openai` retains the official Responses API and OpenAI SDK. `openai_compatible` uses the SDK's Chat Completions client with an explicit endpoint, model and credential variable. This covers providers and local servers implementing that protocol. It does not provide native support for every vendor's distinct API.

`configs/mixed_providers.example.yaml` shows official OpenAI, another endpoint and local residents in one experiment. Replace its illustrative endpoint/model/rates and confirm pricing before use. Secret values belong in `.env`; profiles store only variable names. Remote compatible endpoints require HTTPS. Local compatible servers can omit a key; hosted endpoints require the selected credential variable.

Prefer `output_mode: json_schema`. If a server only supports JSON mode, explicitly choose `json_object`; all outputs still undergo strict Pydantic/environment validation. There is no automatic format downgrade. Set `token_limit_field` to `max_tokens` or `max_completion_tokens` according to that server. Unsupported parameters fail visibly rather than silently changing conditions.

Hosted profiles require confirmed current rates, including explicitly confirmed zero rates for any provider free tier. Hosted free tiers and quotas are provider-dependent and can change. A zero local budget blocks profiles with nonzero API rates from making their next request. It is not an account-side spending cap. Rates/output limits are resolved per resident, while one shared experiment ledger enforces requests, tokens and dollars. Unknown usage retains its reservation.

## Audit and research controls

Provenance saves the resolved roster and runtime local-model identities. Every request records the acting profile, endpoint, requested model, rates and generation settings alongside its raw response and any returned provider model identifier. Action CSVs and the live observer table identify the acting model. Configuration model identities are not added to agent prompts.

Comparisons distinguish **synthetic**, **llm**, and **mixed_synthetic** populations. Experiments containing mock fixtures cannot be treated as fully autonomous LLM populations. Factorial contrasts require matching rosters/settings, prompt versions and local runtime identities. Counterbalance model assignments across authority groups and seeds; otherwise model capability can confound authority effects. Analyse a homogeneous free-model study separately from a heterogeneous population unless roster composition is an explicit factor.

Prompt version 1.1 includes the action schema in the common system prompt for every backend. Do not pool earlier prompt-1.0 experiments as if the protocol were unchanged. Local smoke runs verify integration and make no substantive behavioural claims.
