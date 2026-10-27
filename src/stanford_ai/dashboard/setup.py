"""Model roster editing is observer-only and independent of randomized role assignment."""

import os

import httpx
import pandas as pd
import streamlit as st
from pydantic import ValidationError

from stanford_ai.agents.backends import local_models
from stanford_ai.dashboard.jobs import Jobs
from stanford_ai.environment.permissions import matrix
from stanford_ai.experiments.design import CONDITIONS, initial_world
from stanford_ai.models import ExperimentConfig, ModelProfile

BACKENDS = ["mock", "ollama", "openai", "openai_compatible"]


def profile_rows(rows) -> dict[str, ModelProfile]:
    result = {}
    for row in rows:
        name = str(row.get("name", "")).strip()
        if not name or name in result or name == "default":
            raise ValueError("Profiles need distinct non-empty names; 'default' is reserved")
        values = {k: v for k, v in row.items() if k != "name"}
        for key in ("base_url", "temperature", "model_digest"):
            if key in values and (values[key] == "" or pd.isna(values[key])):
                values[key] = None
        result[name] = ModelProfile.model_validate(values)
    return result


def render_setup(path, manager: Jobs):
    st.subheader("Configure a controlled run")
    left, right = st.columns(2)
    with left:
        condition = st.selectbox("Condition", list(CONDITIONS), index=1,
                                 format_func=lambda x: f"{x} · {CONDITIONS[x]}")
        backend = st.selectbox("Decision backend", BACKENDS, help=(
            "Ollama runs installed local models without API charges. Mock is synthetic test data."))
        defaults = {"mock": "gpt-4.1-mini-2025-04-14", "openai": "gpt-4.1-mini-2025-04-14",
                    "ollama": "gemma3:4b", "openai_compatible": "your-provider-model-id"}
        model = st.text_input("Default model ID", defaults[backend], key=f"default_model_{backend}")
        base_url = None
        if backend in {"ollama", "openai_compatible"}:
            base_url = st.text_input("Default server URL", "http://127.0.0.1:11434" if backend == "ollama"
                                    else "https://your-provider.example/v1", key=f"url_{backend}")
        api_key_env = "OPENAI_API_KEY"
        output_mode, token_limit_field = "json_schema", "max_tokens"
        if backend == "openai_compatible":
            api_key_env = st.text_input("API key environment variable name", "PROVIDER_API_KEY")
            output_mode = st.selectbox("Compatible output format", ["json_schema", "json_object"])
            token_limit_field = st.selectbox("Compatible output limit field",
                                             ["max_tokens", "max_completion_tokens"])
        if backend == "ollama":
            st.caption("Local inference only. No API key, paid fallback, cloud routing or automatic downloads.")
            if st.button("Discover installed local models"):
                try:
                    st.session_state["installed_models"] = local_models(base_url)
                except (httpx.HTTPError, ValueError) as exc:
                    st.error(f"Could not list local models ({type(exc).__name__}). Start Ollama first.")
            if st.session_state.get("installed_models"):
                st.dataframe(pd.DataFrame(st.session_state["installed_models"])[["name", "digest", "size"]],
                             hide_index=True, width="stretch")
        n_agents = st.number_input("Residents (even; split equally)", 2, 100, 4, 2)
        days = st.number_input("Simulated days", 1, 100, 2)
        rounds = st.number_input("Rounds per day", 1, 20, 2)
        seed = st.number_input("Random seed", 0, 2**32 - 1, 42)
    with right:
        free = backend in {"mock", "ollama"}
        budget = st.number_input("Maximum estimated spend (USD)", 0.0, 1000.0,
                                 0.0 if free else 1.0, .25, key=f"budget_{backend}")
        requests = st.number_input("Maximum requests (including retries)", 1, 100000, 100)
        tokens = st.number_input("Maximum accounted tokens", 1, 100000000, 250000)
        output = st.number_input("Maximum output tokens per call", 128, 4096, 400)
        input_rate = st.number_input("Default input USD per million tokens", 0.0, 1000.0,
                                     0.0 if free else .4, key=f"input_rate_{backend}", disabled=free)
        output_rate = st.number_input("Default output USD per million tokens", 0.0, 1000.0,
                                      0.0 if free else 1.6, key=f"output_rate_{backend}", disabled=free)
        pricing = st.checkbox("I verified the default hosted model rates", disabled=free)
    with st.expander("Generation and memory controls"):
        temperature = st.number_input("Default temperature", 0.0, 2.0, 0.7, .1)
        omit_temp = st.checkbox("Omit default temperature for models that do not support it")
        local_context = st.number_input("Local context tokens (Ollama)", 4096, 262144, 32768, 4096)
        timeout = st.number_input("Default request timeout (seconds)", 1, 600, 120)
        memory_window = st.number_input("Detailed memories retained per resident", 1, 100, 12)
        history_window = st.number_input("Recent visible events per observation", 1, 100, 12)
    preview = ExperimentConfig(condition=condition, n_agents=int(n_agents), seed=int(seed))
    world = initial_world(preview)
    use_roster = st.checkbox("Assign different models to individual agents")
    profiles, assignments = {}, {}
    if use_roster:
        st.write("Model profiles")
        st.caption("Add a profile per model/connection, then choose it for each resident. "
                   "Environment variable names belong here; secret values belong in .env. "
                   "Blank profile temperature omits it. Local models use zero API rates.")
        initial = [dict(name="local_gemma", backend="ollama", model="gemma3:4b",
            base_url="http://127.0.0.1:11434", api_key_env="OPENAI_API_KEY", output_mode="json_schema",
            token_limit_field="max_tokens", temperature=.7, max_output_tokens=400,
            local_context_tokens=32768, timeout_seconds=120, model_digest="",
            input_usd_per_million=0.0, output_usd_per_million=0.0, pricing_confirmed=False),
            dict(name="local_qwen", backend="ollama", model="qwen3.5:4b",
            base_url="http://127.0.0.1:11434", api_key_env="OPENAI_API_KEY", output_mode="json_schema",
            token_limit_field="max_tokens", temperature=.7, max_output_tokens=400,
            local_context_tokens=32768, timeout_seconds=120, model_digest="",
            input_usd_per_million=0.0, output_usd_per_million=0.0, pricing_confirmed=False)]
        rows = st.data_editor(pd.DataFrame(initial), num_rows="dynamic", hide_index=True,
            key="model_profile_editor", width="stretch", column_config={
                "backend": st.column_config.SelectboxColumn(options=BACKENDS, required=True),
                "output_mode": st.column_config.SelectboxColumn(options=["json_schema", "json_object"]),
                "token_limit_field": st.column_config.SelectboxColumn(
                    options=["max_tokens", "max_completion_tokens"])})
        try:
            profiles = profile_rows(rows.to_dict("records"))
        except (ValueError, ValidationError) as exc:
            st.error(str(exc))
        st.write("Agent model assignments")
        roster = st.data_editor(pd.DataFrame([{"agent": a.id, "group": a.group, "profile": "default"}
            for a in world.agents.values()]), hide_index=True, width="stretch",
            key=f"agent_roster_{int(n_agents)}_{int(seed)}", disabled=["agent", "group"],
            column_config={"profile": st.column_config.SelectboxColumn(
                options=["default"] + list(profiles), required=True)})
        assignments = {r["agent"]: r["profile"] for r in roster.to_dict("records")
                       if r["profile"] != "default"}
        st.caption("Role assignment is randomized by seed. Hold model-to-agent assignments constant "
                   "across conditions; vary/counterbalance them across replications to study model effects.")
    st.caption(f"Planned action opportunities: {int(n_agents * days * rounds):,}. "
               "Request/token budgets still apply to local inference. No model fallback is automatic.")
    if st.button("Start experiment", type="primary"):
        try:
            if use_roster:
                profiles = profile_rows(rows.to_dict("records"))
            cfg = ExperimentConfig(condition=condition, backend=backend, model=model, base_url=base_url,
                api_key_env=api_key_env, output_mode=output_mode, token_limit_field=token_limit_field,
                model_profiles=profiles, agent_models=assignments, n_agents=int(n_agents), days=int(days),
                rounds_per_day=int(rounds), seed=int(seed), budget_usd=budget, max_requests=int(requests),
                max_tokens=int(tokens), max_output_tokens=int(output), input_usd_per_million=input_rate,
                output_usd_per_million=output_rate, pricing_confirmed=pricing,
                temperature=None if omit_temp else temperature, local_context_tokens=int(local_context),
                timeout_seconds=int(timeout), memory_window=int(memory_window), history_window=int(history_window))
            missing = {p["api_key_env"] for p in cfg.resolved_models().values()
                       if p["backend"] in {"openai", "openai_compatible"} and not os.getenv(p["api_key_env"])
                       and not (p["base_url"] or "").startswith(("http://127.0.0.1", "http://localhost"))}
            if missing:
                st.error("Set these environment variables in .env and restart: " + ", ".join(sorted(missing)))
                return
            job = manager.start(cfg, path)
            st.session_state["selected_run"] = job.run_id
            st.rerun()
        except (ValidationError, ValueError) as exc:
            st.error(str(exc))
    st.caption("Permission matrix for this condition (independent of models and role labels)")
    st.dataframe(pd.DataFrame({g: {a: a in actions for a in sorted(set(sum(matrix(preview).values(), [])))}
                               for g, actions in matrix(preview).items()}), width="stretch")
