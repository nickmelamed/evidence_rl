import streamlit as st
import pandas as pd
import os
import json
import time
import plotly.express as px
import plotly.graph_objects as go

st.set_page_config(layout="wide")
st.title("Claims RL Dashboard")

BASE_DIR = "artifacts/experiments"

# sidebar
auto_refresh = st.sidebar.checkbox("Live Monitoring", value=False)
refresh_rate = st.sidebar.slider("Refresh (sec)", 1, 10, 3)

# helpers
def list_experiments():
    if not os.path.exists(BASE_DIR):
        return []
    return [
        os.path.join(BASE_DIR, d)
        for d in os.listdir(BASE_DIR)
        if os.path.isdir(os.path.join(BASE_DIR, d))
    ]

def load_metrics(exp_path):
    path = os.path.join(exp_path, "metrics.csv")
    if os.path.exists(path):
        df = pd.read_csv(path)
        df["experiment"] = os.path.basename(exp_path)
        return df
    return None

def load_config(exp_path):
    path = os.path.join(exp_path, "config.json")
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return {}

def load_trajectory(exp_path, episode):
    path = os.path.join(exp_path, "trajectories", f"episode_{episode}.json")
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return []

# load experiments
experiments = list_experiments()

selected = st.sidebar.multiselect(
    "Experiments",
    experiments,
    default=experiments[:1]
)

if not selected:
    st.warning("Select at least one experiment")
    st.stop()

dfs = []
configs = {}

for exp in selected:
    df = load_metrics(exp)
    if df is not None:
        dfs.append(df)
        configs[os.path.basename(exp)] = load_config(exp)

data = pd.concat(dfs)

# tabs
tab1, tab2, tab3, tab4 = st.tabs([
    "📊 Single Run",
    "📈 Compare",
    "🔍 Episode Drilldown",
    "⚙️ Config"
])

# single run + best experiment
with tab1:
    st.header("Single Experiment")

    exp_name = st.selectbox(
        "Experiment",
        [os.path.basename(e) for e in selected]
    )

    cfg = configs.get(exp_name, {})

    st.subheader("Experiment Setup")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("Algorithm", cfg.get("algo", "N/A"))
        st.metric("Policy", cfg.get("policy_type", "N/A"))

    with col2:
        st.metric("LR", cfg.get("lr", "N/A"))
        st.metric("Gamma", cfg.get("gamma", "N/A"))

    with col3:
        st.metric("Clip", cfg.get("clip", "N/A"))
        st.metric("Entropy Coef", cfg.get("entropy_coef", "N/A"))

    exp_path = [e for e in selected if os.path.basename(e) == exp_name][0]
    df = data[data["experiment"] == exp_name]

    df["reward_smooth"] = df["reward"].rolling(window=5).mean()

    st.subheader("LLM Evaluation Metrics")

    traj_all = []
    for ep in df["episode"].unique():
        traj = load_trajectory(exp_path, int(ep))
        traj_all.extend(traj)

    traj_df = pd.DataFrame(traj_all)

    if "llm_scores" in traj_df.columns:
        traj_df["LCS"] = traj_df["llm_scores"].apply(lambda x: x.get("LCS", 0) if isinstance(x, dict) else 0)
        traj_df["ESS"] = traj_df["llm_scores"].apply(lambda x: x.get("ESS", 0) if isinstance(x, dict) else 0)
        traj_df["HRS"] = traj_df["llm_scores"].apply(lambda x: x.get("HRS", 0) if isinstance(x, dict) else 0)

        fig = px.line(traj_df, x="step", y=["LCS", "ESS", "HRS"], title="LLM Scores Over Time")
        st.plotly_chart(fig, width='stretch')

    # value estimate for actor critic
    if "value_estimate" in traj_df:
        fig = px.line(traj_df, x="step", y="value_estimate", title="Value Estimates")
        st.plotly_chart(fig)

    if "advantage" in traj_df:
        fig = px.line(traj_df, x="step", y="advantage", title="Advantage Signal")
        st.plotly_chart(fig)

    # token usage
    fig = px.line(df, x="episode", y="tokens", title="Token Usage")
    st.plotly_chart(fig, width='stretch')

    # BEST EPISODE
    if "reward" not in df.columns or df["reward"].dropna().empty:
        st.warning("No reward data yet (training may still be initializing)")
        best_ep = None
    else:
        best_idx = df["reward"].dropna().idxmax()
        best_ep = df.loc[best_idx, "episode"]

    if best_ep is not None:

        colA, colB = st.columns(2)

        with colA:
            fig = px.line(df, x="episode", y="reward", title="Reward")
            st.plotly_chart(fig, width='stretch')

        with colB:
            fig = px.line(df, x="episode", y="reward_smooth", title="Smoothed")
            st.plotly_chart(fig, width='stretch')

        st.success(f"🏆 Best Episode: {int(best_ep)}")

    # Policy behavior
    st.subheader("Policy Behavior")

    fig = px.line(df, x="episode", y="entropy", title="Entropy (Exploration)")
    st.plotly_chart(fig, width='stretch')

    if "num_steps" in df.columns:
        fig = px.line(df, x="episode", y="num_steps", title="Steps per episode")
        st.plotly_chart(fig, use_container_width=True)

    if "llm_reward" in traj_df.columns:
        traj_df["base_reward"] = traj_df["reward"] - traj_df.get("llm_reward", 0)
        fig = px.line(traj_df, x="step", y=["reward", "llm_reward"], title="Base vs LLM reward per step")
        st.plotly_chart(fig, use_container_width=True)

    if "curriculum_level" in df.columns:
        fig = px.line(df, x="episode", y="curriculum_level", title="Curriculum level")
        st.plotly_chart(fig, use_container_width=True)

    if "reward_raw" in df.columns and "reward" in df.columns:
        fig = px.line(df, x="episode", y=["reward", "reward_raw"], title="Normalised vs raw reward")
        st.plotly_chart(fig, use_container_width=True)

    # Action mix
    action_dist_cols = [c for c in df.columns if c.startswith("action_dist.")]
    if action_dist_cols:
        fig = px.area(df, x="episode", y=action_dist_cols, title="Action distribution over time")
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Action distribution data not yet available (requires commit 5 trainer changes)")

# comparison
with tab2:
    st.header("Compare Experiments")

    fig = go.Figure()

    for exp in selected:
        name = os.path.basename(exp)
        df = data[data["experiment"] == name]
        df["reward_smooth"] = df["reward"].rolling(10).mean()

        fig.add_trace(go.Scatter(
            x=df["episode"],
            y=df["reward_smooth"],
            mode='lines',
            name=f"{name} ({configs[name].get('algo', 'unknown')})"
        ))

    st.plotly_chart(fig, width='stretch')

    eval_cols = [c for c in data.columns if c.startswith("eval/")]
    if eval_cols:
        st.subheader("Eval metrics")
        fig_eval = go.Figure()
        for exp in selected:
            name = os.path.basename(exp)
            df_e = data[data["experiment"] == name]
            if "eval/mean_reward" in df_e.columns:
                df_e = df_e.dropna(subset=["eval/mean_reward"])
                fig_eval.add_trace(go.Scatter(
                    x=df_e["episode"], y=df_e["eval/mean_reward"],
                    mode="lines", name=f"{name} eval",
                    error_y=dict(type="data", array=df_e.get("eval/std_reward", pd.Series()).fillna(0).tolist(), visible=True)
                ))
        st.plotly_chart(fig_eval, use_container_width=True)

# drilldown + claim
with tab3:
    st.header("Episode Drilldown")

    exp_name = st.selectbox(
        "Experiment",
        [os.path.basename(e) for e in selected],
        key="drill"
    )

    exp_path = [e for e in selected if os.path.basename(e) == exp_name][0]
    df = data[data["experiment"] == exp_name]

    # Clean episode column
    df = df.copy()

    df["episode"] = pd.to_numeric(df["episode"], errors="coerce")
    df["reward"] = pd.to_numeric(df["reward"], errors="coerce")
    df = df.dropna(subset=["episode", "reward"])

    # GUARD
    if df.empty:
        st.warning("No episode data yet (training still initializing)")
    else:
        ep_min = int(df["episode"].min())
        ep_max = int(df["episode"].max())

        # GUARD (edge case: single episode)
        if ep_min == ep_max:
            ep = ep_min
            st.info(f"Only one episode available: {ep}")
        else:
            ep = st.slider("Episode", ep_min, ep_max)

        traj = load_trajectory(exp_path, ep)

        if not traj:
            st.warning("Trajectory not available yet for this episode")
        else:
            # CLAIM
            st.subheader("Claim")
            st.write(traj[0].get("claim", "N/A"))

            # EVIDENCE POOL
            st.subheader("Evidence Pool")

            evidence = traj[0].get("evidence_pool", [])

            for e in evidence:
                st.markdown(f"**[{e['id']}]** {e['text']}")

            # REPLAY
            st.subheader("Step Replay")

            if len(traj) == 0:
                st.warning("No steps recorded for this episode")
            else:
                step_min = 1
                step_max = len(traj)

                if step_min == step_max:
                    step_idx = step_min
                    st.info(f"Only one step available: {step_idx}")
                else:
                    step_idx = st.slider("Step", step_min, step_max)

                step = traj[step_idx - 1]

                st.subheader("LLM Evaluation")

                llm_scores = step.get("llm_scores", {})
                if llm_scores:
                    score_df = pd.DataFrame([
                        {"metric": k, "score": v}
                        for k, v in llm_scores.items()
                        if k in ["LCS", "ESS", "HRS", "COMP", "confidence"]
                    ])
                    if not score_df.empty:
                        fig = px.bar(score_df, x="metric", y="score", title="LLM judge scores",
                                     range_y=[0, 1], color="metric",
                                     color_discrete_map={"HRS": "#E24B4A", "LCS": "#378ADD",
                                                         "ESS": "#1D9E75", "COMP": "#7F77DD",
                                                         "confidence": "#888780"})
                        st.plotly_chart(fig, use_container_width=True)

                st.metric("Tokens Used", step.get("tokens", 0))

                if step.get("query_count") is not None:
                    st.metric("Query count", step["query_count"])

                if step.get("value_estimate") is not None:
                    st.metric("Value Estimate", round(step["value_estimate"], 3))

                if step.get("advantage") is not None:
                    st.metric("Advantage", round(step["advantage"], 3))

                if step.get("claim_evidence_sim") is not None:
                    st.metric("Claim-evidence similarity", round(step["claim_evidence_sim"], 3))

                st.subheader("Generated Argument")

                arg = step.get("argument") or (step.get("action_payload", {}) or {}).get("argument")
                if arg:
                    st.info(arg)
                else:
                    st.write("No argument generated this step")

                st.subheader("Evidence Used")
                evidence_ids = step.get("evidence_used") or step.get("selected_ids", [])
                evidence_pool = step.get("evidence_pool", [])

                for e in evidence_pool:
                    if e["id"] in (evidence_ids or []):
                        st.success(f"[{e['id']}] {e['text']}")

                st.json(step)

                st.subheader("Policy Distribution")

                probs = step.get("action_probs")
                names = step.get("action_names")
                chosen_idx = step.get("action_idx")
                entropy = step.get("entropy")

                if probs and names:
                    df_probs = pd.DataFrame({
                        "action": names,
                        "probability": probs
                    })

                    fig = px.bar(
                        df_probs,
                        x='action',
                        y='probability',
                        title='Action Probabilities'
                    )

                    st.plotly_chart(fig, width='stretch')

                    if chosen_idx is not None:
                        st.success(f"Chosen Action: {names[chosen_idx]}")

                st.subheader("Policy Evolution")

                traj_df = pd.DataFrame(traj)

                if "entropy" in traj_df:
                    fig = px.line(traj_df, x='step', y='entropy', title='Entropy Over Steps')
                    st.plotly_chart(fig, width='stretch')

                # Highlight selected
                st.subheader("Selected Evidence")

                selected_ids = step.get("selected_ids", [])

                for e in evidence:
                    if e["id"] in selected_ids:
                        st.success(f"[{e['id']}] {e['text']}")

# config
with tab4:
    st.header("Configs")

    for name, cfg in configs.items():
        st.subheader(name)
        st.subheader("Algorithm")
        st.write(cfg.get("algo"))

        st.subheader("Policy")
        st.write(cfg.get("policy_type"))

        st.subheader("Hyperparameters")
        df_cfg = pd.DataFrame(cfg.items(), columns=["param", "value"])
        df_cfg["value"] = df_cfg["value"].astype(str)
        st.table(df_cfg)

# live refresh
if auto_refresh:
    time.sleep(refresh_rate)
    st.rerun()
