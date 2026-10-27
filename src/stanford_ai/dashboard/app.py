"""Launch: python -m streamlit run src/stanford_ai/dashboard/app.py"""

import io
import json
import os
import zipfile

import pandas as pd
import plotly.express as px
import streamlit as st
from dotenv import load_dotenv
from pydantic import ValidationError

from stanford_ai.dashboard.jobs import Jobs
from stanford_ai.environment.permissions import matrix
from stanford_ai.experiments.analysis import factorial_contrasts
from stanford_ai.experiments.design import CONDITIONS
from stanford_ai.metrics.compute import action_frame, compare_runs
from stanford_ai.models import ExperimentConfig
from stanford_ai.storage.sqlite import Store

load_dotenv()
st.set_page_config(page_title="Stanford AI · Authority Lab", page_icon="◈", layout="wide")


@st.cache_resource
def jobs():
    return Jobs()


def style():
    # Laboratory notebook: cool paper, marine ink, copper annotations, group-coded plots.
    st.markdown("""<style>
    .stApp { background: #F2F6FA; color: #17324D; }
    h1,h2,h3 { font-family: 'Trebuchet MS',sans-serif; letter-spacing: -.025em; color:#17324D; }
    h1 { font-size: 2.7rem !important; }
    [data-testid='stSidebar'] { background: #E3ECF4; }
    .lab-tag { font: 12px Consolas,monospace; letter-spacing:.13em; color:#8A491F; }
    .condition-strip { display:grid; grid-template-columns:repeat(4,1fr); gap:8px; margin:20px 0; }
    .condition-cell { border-top:4px solid #237C83; background:#E3ECF4; padding:12px; }
    .condition-cell b { font:20px Consolas,monospace; }
    [data-testid='stMetric'] { background:#E3ECF4; border-radius:8px; padding:12px; }
    .stButton button:focus-visible { outline:3px solid #8A491F; }
    @media(max-width:640px) { .condition-strip { grid-template-columns:repeat(2,1fr); } }
    @media(prefers-reduced-motion:reduce) { * { animation:none !important; transition:none !important; } }
    </style>""", unsafe_allow_html=True)


def facility(world):
    a, b, c, d = st.columns(4)
    a.metric("Simulated day / round", f"{world.day} / {world.round}")
    b.metric("Available credits", world.pool)
    c.metric("Active rules", sum(r["status"] == "active" for r in world.rules.values()))
    d.metric("Open disputes", sum(r["status"] == "open" for r in world.disputes.values()))
    st.caption("Activity order: " + " → ".join(world.schedule))
    st.dataframe(pd.DataFrame([{"resident": agent.id, "group": agent.group,
        "credits": world.holdings[agent.id],
        "access_restrictions": ", ".join(world.restrictions.get(agent.id, {})) or "None"}
        for agent in world.agents.values()]), hide_index=True, width="stretch")
    with st.expander("Facility rules, tasks, pending requests and reviews"):
        st.json({"rules": world.rules, "tasks": world.tasks, "requests": world.requests,
                 "appeals": world.appeals, "agreements": world.agreements})


@st.fragment(run_every="1s")
def live(path, selected):
    manager = jobs()
    job = manager.current
    if job and not job.done.is_set():
        st.info(f"Running {job.run_id}. Stop takes effect after the current API call finishes.")
        if st.button("Stop experiment", key="stop_live"):
            job.stop.set()
    if job and job.error:
        st.error(job.error)
    with Store(path) as store:
        try:
            run = store.run(selected)
        except KeyError:
            st.info("The experiment is starting. Its first event will appear here.")
            return
        world = store.replay(selected)
        st.write(f"**{run['status']}** · {CONDITIONS[run['config']['condition']]} · "
                 f"{run['config']['backend']}")
        if run["reason"]:
            st.warning(run["reason"])
        usage = run["usage"]
        st.caption(f"{usage['requests']} requests · {usage['input_tokens'] + usage['output_tokens']:,} "
                   f"reported tokens · USD {usage['estimated_cost_usd']:.4f} estimated reported cost · "
                   f"USD {usage['accounted_cost_usd']:.4f} including outstanding/unknown reservations")
        facility(world)
        st.subheader("Latest interactions")
        st.caption("Observer view includes private conversations. Agents receive only their permitted view.")
        frame = action_frame(world)
        st.dataframe(frame.tail(30).iloc[::-1], hide_index=True, width="stretch")
        with st.expander("Conflicts and negotiations"):
            st.json({"disputes": world.disputes, "agreements": world.agreements})


def setup(path):
    st.subheader("Configure a controlled run")
    with st.form("setup"):
        left, right = st.columns(2)
        with left:
            condition = st.selectbox("Condition", list(CONDITIONS), index=1,
                                     format_func=lambda x: f"{x} · {CONDITIONS[x]}")
            backend = st.selectbox("Decision backend", ["mock", "openai"],
                                   help="Mock choices are synthetic engineering fixtures.")
            model = st.text_input("OpenAI model ID", "gpt-4.1-mini-2025-04-14")
            n_agents = st.number_input("Residents (even; split equally)", 2, 100, 4, 2)
            days = st.number_input("Simulated days", 1, 100, 2)
            rounds = st.number_input("Rounds per day", 1, 20, 2)
            seed = st.number_input("Random seed", 0, 2**32 - 1, 42)
        with right:
            budget = st.number_input("Maximum estimated spend (USD)", 0.01, 1000.0, 1.0, 0.25)
            requests = st.number_input("Maximum requests (including retries)", 1, 100000, 100)
            tokens = st.number_input("Maximum accounted tokens", 1, 100000000, 250000)
            output = st.number_input("Maximum output tokens per call", 128, 4096, 400)
            input_rate = st.number_input("Input USD per million tokens", 0.0, 1000.0, 0.4)
            output_rate = st.number_input("Output USD per million tokens", 0.0, 1000.0, 1.6)
            pricing = st.checkbox("I verified these rates for the selected model (required for OpenAI)")
        with st.expander("Generation and memory controls"):
            temperature = st.number_input("Temperature", 0.0, 2.0, 0.7, 0.1)
            omit_temp = st.checkbox("Omit temperature for models that do not support it")
            memory_window = st.number_input("Detailed memories retained per resident", 1, 100, 12)
            history_window = st.number_input("Recent visible events per observation", 1, 100, 12)
        st.caption(f"Planned action opportunities: {int(n_agents * days * rounds):,}. "
                   "Budgets may stop the run early. Daily time advances immediately.")
        start = st.form_submit_button("Start experiment", type="primary")
    preview = ExperimentConfig(condition=condition)
    st.caption("Permission matrix for this condition (independent of role labels)")
    st.dataframe(pd.DataFrame({g: {a: a in actions for a in sorted(set(sum(matrix(preview).values(), [])))}
                               for g, actions in matrix(preview).items()}), width="stretch")
    if start:
        try:
            cfg = ExperimentConfig(condition=condition, backend=backend, model=model,
                n_agents=int(n_agents), days=int(days), rounds_per_day=int(rounds), seed=int(seed),
                budget_usd=budget, max_requests=int(requests), max_tokens=int(tokens),
                max_output_tokens=int(output), input_usd_per_million=input_rate,
                output_usd_per_million=output_rate, pricing_confirmed=pricing,
                temperature=None if omit_temp else temperature,
                memory_window=int(memory_window), history_window=int(history_window))
            if backend == "openai" and not os.getenv("OPENAI_API_KEY"):
                st.error("Set OPENAI_API_KEY in .env or your environment, then restart the dashboard.")
                return
            job = jobs().start(cfg, path)
            st.session_state["selected_run"] = job.run_id
            st.success(f"Started {job.run_id}. Open Live simulation to observe the run.")
            st.rerun()
        except (ValidationError, ValueError) as exc:
            st.error(str(exc))


def profiles(world, config):
    agent_id = st.selectbox("Resident", list(world.agents))
    agent = world.agents[agent_id]
    st.write(f"**Group {agent.group}** · " + ("Guard" if agent.group == "A" else "Prisoner")
             if config.explicit_labels else f"**Group {agent.group}**")
    st.write("Goals", agent.goals)
    col1, col2 = st.columns(2)
    with col1:
        traits = pd.DataFrame({"trait": list(agent.personality.model_dump()),
                               "value": list(agent.personality.model_dump().values())})
        st.plotly_chart(px.bar(traits, x="value", y="trait", range_x=[0, 1],
                               color_discrete_sequence=["#237C83"]), width="stretch")
    with col2:
        st.write("Available actions", matrix(config)[agent.group])
        st.write("Relationship interaction counts (no inferred affinity)")
        st.json(agent.relationships)
    with st.expander("Resident-visible memory and compressed counts"):
        st.json({"summary": agent.memory_summary, "recent": agent.memory})
    frame = action_frame(world)
    own = frame[frame.actor == agent_id]
    st.dataframe(own, hide_index=True, width="stretch")
    if not own.empty:
        trend = own.groupby("day")[["restrictive", "cooperative", "resistant"]].sum().reset_index()
        st.plotly_chart(px.line(trend, x="day", y=["restrictive", "cooperative", "resistant"],
                                markers=True), width="stretch")


def analytics(store, run_id, world):
    metrics = pd.DataFrame(store.metrics(run_id))
    if metrics.empty:
        st.info("Metrics appear after the first recorded event.")
        return
    st.caption("Action-based proxies. Classifications do not establish motives or internal states.")
    st.plotly_chart(px.line(metrics, x="seq", y=["restrictive_decisions", "cooperation", "conflict"],
                            labels={"seq": "Event", "value": "Cumulative actions"}), width="stretch")
    a, b = st.columns(2)
    with a:
        st.plotly_chart(px.line(metrics, x="seq", y=["rules_proposed", "rules_approved",
                                                    "sanctions"]), width="stretch")
    with b:
        distribution = pd.DataFrame([{"resident": x, "group": world.agents[x].group, "credits": v}
                                      for x, v in world.allocated_today.items()])
        st.plotly_chart(px.bar(distribution, x="resident", y="credits", color="group",
                                color_discrete_map={"A": "#237C83", "B": "#B86632"},
                                title="Today's cumulative credit allocations"), width="stretch")
    frame = action_frame(world)
    if not frame.empty:
        group = frame.groupby(["day", "group"])[["restrictive", "cooperative", "resistant"]].sum()
        st.plotly_chart(px.line(group.reset_index(), x="day", y="cooperative", color="group",
                                markers=True, title="Daily cooperation by group"), width="stretch")
        midpoint = max(1, world.day // 2)
        frame["period"] = frame.day.map(lambda d: "initial" if d <= midpoint else "later")
        summary = frame.groupby("period").agg(turns=("id", "count"),
            restrictive=("restrictive", "mean"), cooperation=("cooperative", "mean"),
            resistance=("resistant", "mean"), messages=("action", lambda s: sum(s == "communicate")))
        st.write("Initial versus later observed days (descriptive; requires multiple days)")
        st.dataframe(summary, width="stretch")
    st.dataframe(metrics, hide_index=True, width="stretch")


def comparison(store):
    completed = [r for r in store.runs() if r["status"] == "completed"]
    selected = st.multiselect("Completed experiments", [r["id"] for r in completed],
        format_func=lambda x: next(f"{r['config']['condition']} · {r['config']['backend']} · "
                                   f"seed {r['config']['seed']} · {x[:8]}" for r in completed if r["id"] == x))
    data = compare_runs(store, selected)
    if data.empty:
        st.info("Select completed runs. Use several seeds per condition for variability.")
        return
    st.warning("Mock and OpenAI runs are separate evidence classes. Model, duration, agent count, "
               "generation settings and budgets must also match for a controlled comparison.")
    evidence = st.selectbox("Backend to compare", sorted(data.backend.unique()))
    data = data[data.backend == evidence]
    metric = st.selectbox("Observable outcome", ["restrictive_per_100_turns",
        "cooperation_per_100_turns", "conflict_per_100_turns", "resource_gini"])
    st.plotly_chart(px.box(data, x="condition", y=metric, color="authority", points="all",
                            hover_data=["seed", "run_id", "model"]), width="stretch")
    st.dataframe(data, hide_index=True, width="stretch")
    st.dataframe(data.groupby(["condition", "authority", "labels"])[metric].agg(
        ["count", "mean", "std", "min", "median", "max"]), width="stretch")
    try:
        contrasts = factorial_contrasts(data, metric)
        if contrasts.empty:
            st.info("Select A/B/C/D runs at matching seeds for paired factorial contrasts.")
        else:
            st.write("Exploratory paired factorial contrasts · run-block bootstrap 95% intervals")
            st.dataframe(contrasts, hide_index=True, width="stretch")
            st.caption("Intervals resample complete seed blocks. Small samples and exploratory "
                       "proxies do not support confirmatory causal or human-behaviour claims.")
    except ValueError as exc:
        st.info(str(exc))
    st.download_button("Download run comparison CSV", data.to_csv(index=False), "comparison.csv")


def download_bundle(store, run_id):
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as archive:
        raw = {"run": store.run(run_id), "events": store.events(run_id),
               "requests": store.requests(run_id),
               "snapshots": [{"seq": r[0], "state": json.loads(r[1])} for r in store.db.execute(
                   "SELECT seq,state FROM snapshots WHERE run_id=? ORDER BY seq", (run_id,))]}
        archive.writestr("raw.json", json.dumps(raw, ensure_ascii=False, indent=2))
        archive.writestr("metrics.csv", pd.DataFrame(store.metrics(run_id)).to_csv(index=False))
        archive.writestr("actions.csv", action_frame(store.replay(run_id)).to_csv(index=False))
    return buffer.getvalue()


def main():
    style()
    st.markdown('<div class="lab-tag">COMPUTATIONAL BEHAVIOUR / OBSERVATION WORKBENCH</div>',
                unsafe_allow_html=True)
    st.title("Authority, under observation.")
    st.write("A controlled facility for studying how language-model agents use unequal authority.")
    st.markdown('<div class="condition-strip">' + "".join(
        f'<div class="condition-cell"><b>{c}</b><br>{text}</div>' for c, text in CONDITIONS.items())
        + '</div>', unsafe_allow_html=True)
    path = os.getenv("STANFORD_AI_DB", "outputs/experiments.sqlite")
    with st.sidebar:
        st.header("Experiment notebook")
        st.caption("Local observer dashboard · no university affiliation")
        with Store(path) as store:
            runs = store.runs()
        ids = [r["id"] for r in runs]
        active = jobs().current
        if active and active.run_id not in ids:
            ids.insert(0, active.run_id)
        if ids:
            if st.session_state.get("selected_run") not in ids:
                st.session_state["selected_run"] = ids[0]
            st.selectbox("Saved / active experiment", ids, key="selected_run",
                format_func=lambda x: next((f"{r['config']['condition']} · {r['config']['backend']} · "
                    f"{r['status']} · {x[:8]}" for r in runs if r["id"] == x), x[:8]))
        st.button("Refresh notebook")
        st.caption("Mock data tests the software. It cannot test the scientific hypothesis.")
    tabs = st.tabs(["Experiment setup", "Live simulation", "Agent profiles", "Analytics",
                    "Compare experiments", "Replay & export"])
    with tabs[0]:
        setup(path)
    selected = st.session_state.get("selected_run")
    with tabs[1]:
        if selected:
            live(path, selected)
        else:
            st.info("Start a small mock experiment from Experiment setup.")
    with Store(path) as store:
        try:
            run = store.run(selected) if selected else None
        except KeyError:
            run = None
        if run:
            world = store.replay(selected)
            if run["config"]["backend"] == "mock":
                st.caption("Selected experiment: SYNTHETIC MOCK DATA — engineering validation only.")
            with tabs[2]:
                profiles(world, ExperimentConfig.model_validate(run["config"]))
            with tabs[3]:
                analytics(store, selected, world)
            with tabs[5]:
                st.subheader("Replay the recorded world")
                st.caption("Replay uses saved events and snapshots only. It makes no API requests.")
                maximum = len(world.history)
                position_key = f"replay_position_{selected}"
                seq = st.slider("Event position (0 = initial state)", 0, max(1, maximum),
                                key=position_key)
                seq = min(seq, maximum)
                def step(delta):
                    st.session_state[position_key] = max(0, min(maximum,
                        st.session_state[position_key] + delta))
                col1, col2 = st.columns(2)
                with col1:
                    st.button("Previous event", on_click=step, args=(-1,))
                with col2:
                    st.button("Next event", on_click=step, args=(1,))
                position = seq
                replayed = store.replay(selected, position)
                st.write(f"Recorded event {position} of {maximum}")
                facility(replayed)
                if position:
                    st.json(replayed.history[-1])
                with st.expander("Configuration, usage and provenance"):
                    st.json({k: run[k] for k in ("config", "usage", "provenance")})
                if st.button("Prepare export bundle"):
                    st.session_state[f"export_{selected}"] = download_bundle(store, selected)
                bundle = st.session_state.get(f"export_{selected}")
                if bundle:
                    st.download_button("Download raw JSON and processed CSV bundle", bundle,
                                       f"{selected}.zip")
        with tabs[4]:
            comparison(store)


if __name__ == "__main__":
    main()
