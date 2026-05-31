import streamlit as st
import pandas as pd
import numpy as np
import os
import json
import math
import time
import plotly.express as px
import plotly.graph_objects as go
from evid_rl_env.environment.actions import ACTIONS

st.set_page_config(layout="wide")
st.title("EvidenceRL Dashboard")

BASE_DIR = "artifacts/experiments"

# Full action name → short display label
ACTION_SHORT = {
    "generate_support_argument":    "SUPPORT",
    "generate_contradict_argument": "CONTRADICT",
    "concede_point":                "CONCEDE",
    "select_evidence":              "SELECT",
    "remove_evidence":              "REMOVE",
    "query_evidence":               "QUERY",
    "rerank_evidence":              "RERANK",
    "summarize_evidence":           "SUMMARIZE",
    "finalize":                     "FINALIZE",
}
ACTION_COLORS = {
    "SUPPORT":    "#1D9E75",
    "CONTRADICT": "#E24B4A",
    "CONCEDE":    "#F5A623",
    "SELECT":     "#378ADD",
    "REMOVE":     "#888780",
    "QUERY":      "#7F77DD",
    "RERANK":     "#AAAAAA",
    "SUMMARIZE":  "#D85A30",
    "FINALIZE":   "#FFD700",
}

# sidebar
auto_refresh = st.sidebar.checkbox("Live Monitoring", value=False)
refresh_rate = st.sidebar.slider("Refresh (sec)", 1, 10, 3)
demo_mode = st.sidebar.toggle(
    "🎬 Demo mode", value=False,
    help="Hides internal training diagnostics; shows only demo-relevant charts."
)

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

def load_eval_results(exp_path):
    path = os.path.join(exp_path, "eval_results.json")
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return None

def compute_baselines(data, selected):
    results = {}
    for exp in selected:
        name = os.path.basename(exp)
        df = data[data["experiment"] == name].copy()
        if df.empty or "reward" not in df.columns:
            continue
        first_ep = df.sort_values("episode").iloc[0]["reward"] if not df.empty else 0
        trained = df.sort_values("episode").tail(3)["reward"].mean()
        results[name] = {
            "random (ep 0)":        round(first_ep, 3),
            "early finalize":       -0.5,
            "trained (last 3 eps)": round(trained, 3),
        }
    return results

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

if not dfs:
    st.warning("No experiment data found. Check that your experiments have a metrics.csv file.")
    st.stop()
data = pd.concat(dfs, ignore_index=True)

# sidebar health summary (ADD 8)
st.sidebar.divider()
st.sidebar.subheader("Training health")
for exp in selected[:1]:
    name = os.path.basename(exp)
    df_s = data[data["experiment"] == name].copy() if not data.empty else pd.DataFrame()
    if not df_s.empty and "reward" in df_s.columns:
        recent = df_s.sort_values("episode").tail(5)["reward"]
        early  = df_s.sort_values("episode").head(5)["reward"]
        trend  = "📈" if recent.mean() > early.mean() else "📉"
        st.sidebar.metric("Reward trend", trend, f"{recent.mean():.2f} vs {early.mean():.2f}")
        if "entropy" in df_s.columns:
            ent     = df_s.sort_values("episode")["entropy"].iloc[-1]
            max_ent = math.log(len(ACTIONS))
            st.sidebar.metric("Entropy", f"{ent:.3f}", f"{ent / max_ent * 100:.0f}% of max")
        if "num_steps" in df_s.columns:
            avg_steps = df_s.tail(3)["num_steps"].mean()
            st.sidebar.metric("Avg steps (last 3)", f"{avg_steps:.1f}")

# tabs
tab0, tab1, tab2, tab3, tab4 = st.tabs([
    "📊 Eval Results",
    "📈 Single Run",
    "🔀 Compare",
    "🔍 Episode Drilldown",
    "⚙️ Config"
])

# ── tab0: Eval Results (ADD 1) ────────────────────────────────────────────────
with tab0:
    st.header("Eval Results — RL vs Baselines")

    exp_name = st.selectbox(
        "Experiment", [os.path.basename(e) for e in selected], key="eval_exp"
    )
    exp_path = [e for e in selected if os.path.basename(e) == exp_name][0]
    eval_data = load_eval_results(exp_path)

    if eval_data is None:
        st.info(
            "No eval_results.json found for this experiment. "
            "Run `evid-eval` with `--output-json` to generate it, "
            "or see the README for how to wire the JSON output from the CLI."
        )
    else:
        baselines_eval = eval_data.get("baselines", {})
        n_episodes     = eval_data.get("n_episodes", "?")
        checkpoint     = eval_data.get("checkpoint", "?")

        st.caption(f"Checkpoint: `{checkpoint}` | Episodes: {n_episodes}")

        greedy_mean = baselines_eval.get("greedy_llm", {}).get("mean")
        rl_mean     = baselines_eval.get("rl", {}).get("mean")
        rl_std      = baselines_eval.get("rl", {}).get("std")

        kpi1, kpi2, kpi3 = st.columns(3)
        if greedy_mean is not None and rl_mean is not None:
            delta   = rl_mean - greedy_mean
            win_pct = eval_data.get("rl_win_rate_vs_greedy")
            with kpi1:
                st.metric("RL mean reward", f"{rl_mean:.3f}", f"Δ {delta:+.3f} vs greedy_llm")
            with kpi2:
                st.metric("greedy_llm mean reward", f"{greedy_mean:.3f}")
            with kpi3:
                if win_pct is not None:
                    st.metric("RL win rate vs greedy_llm", f"{win_pct:.0%}")

        METHOD_ORDER = ["random", "majority", "greedy_llm", "fewshot_k3",
                        "fewshot_k5", "best_of_5", "rl"]
        COLOR_MAP = {
            "random":     "#888780",
            "majority":   "#AAAAAA",
            "greedy_llm": "#D85A30",
            "fewshot_k3": "#7F77DD",
            "fewshot_k5": "#9B91E8",
            "best_of_5":  "#378ADD",
            "rl":         "#1D9E75",
        }

        rows = [
            {
                "method":      method,
                "mean_reward": baselines_eval[method].get("mean", 0),
                "std":         baselines_eval[method].get("std", 0),
            }
            for method in METHOD_ORDER if method in baselines_eval
        ]

        if rows:
            bar_df  = pd.DataFrame(rows)
            fig_bar = go.Figure()
            for _, row in bar_df.iterrows():
                fig_bar.add_trace(go.Bar(
                    x=[row["method"]],
                    y=[row["mean_reward"]],
                    error_y=dict(type="data", array=[row["std"]], visible=True),
                    name=row["method"],
                    marker_color=COLOR_MAP.get(row["method"], "#999"),
                    marker_line_width=3 if row["method"] == "rl" else 0,
                    marker_line_color="#fff" if row["method"] == "rl" else "rgba(0,0,0,0)",
                ))
            if greedy_mean is not None:
                fig_bar.add_hline(
                    y=greedy_mean, line_dash="dash", line_color="#D85A30",
                    annotation_text="greedy_llm", annotation_position="top left",
                )
            fig_bar.update_layout(
                title="Mean reward ± std by method",
                showlegend=False,
                xaxis_title="Method",
                yaxis_title="Mean reward",
            )
            st.plotly_chart(fig_bar, width="stretch")

        if greedy_mean is not None and rows:
            st.subheader("Δ vs greedy_llm")
            delta_rows = [
                {
                    "method":         r["method"],
                    "mean reward":    round(r["mean_reward"], 3),
                    "± std":          round(r["std"], 3),
                    "Δ vs greedy_llm": round(r["mean_reward"] - greedy_mean, 3),
                }
                for r in rows
            ]
            delta_df = pd.DataFrame(delta_rows)

            def color_delta(val):
                if val > 0:
                    return "color: #1D9E75; font-weight: bold"
                elif val < 0:
                    return "color: #E24B4A"
                return ""

            st.dataframe(
                delta_df.style.map(color_delta, subset=["Δ vs greedy_llm"]),
                width="stretch",
                hide_index=True,
            )

        judge_metrics = eval_data.get("judge_metrics", {})
        if judge_metrics:
            st.subheader("LLM judge score breakdown (radar)")
            DIMS      = ["LCS", "ESS", "COMP", "GRS", "BIAS"]
            radar_fig = go.Figure()
            for method, scores in judge_metrics.items():
                vals = [scores.get(d, 0) for d in DIMS]
                vals += vals[:1]
                radar_fig.add_trace(go.Scatterpolar(
                    r=vals,
                    theta=DIMS + [DIMS[0]],
                    fill="toself",
                    name=method,
                    line_color=COLOR_MAP.get(method, "#999"),
                    opacity=0.7,
                ))
            radar_fig.update_layout(
                polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
                title="LLM judge dimensions — RL vs baselines",
            )
            st.plotly_chart(radar_fig, width="stretch")
            st.caption(
                "GRS (grounding risk) and BIAS are ↓ lower-is-better; all others ↑ higher-is-better. "
                "RL's debate loop should produce structurally better LCS, ESS, and COMP "
                "even when raw reward differences are small."
            )

# ── tab1: Single Run ──────────────────────────────────────────────────────────
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

    # Load all trajectories once; traj_by_episode avoids duplicate file reads
    traj_all       = []
    traj_by_episode = {}
    for ep in df["episode"].unique():
        traj_ep = load_trajectory(exp_path, int(ep))
        traj_all.extend(traj_ep)
        traj_by_episode[int(ep)] = traj_ep

    traj_df = pd.DataFrame(traj_all)

    if "llm_scores" in traj_df.columns:
        # Use 0.5 as the fallback to match the judge's actual fallback value
        traj_df["LCS"] = traj_df["llm_scores"].apply(
            lambda x: x.get("LCS", 0.5) if isinstance(x, dict) else 0.5
        )
        traj_df["ESS"] = traj_df["llm_scores"].apply(
            lambda x: x.get("ESS", 0.5) if isinstance(x, dict) else 0.5
        )
        traj_df["HRS"] = traj_df["llm_scores"].apply(
            lambda x: x.get("HRS", 0.5) if isinstance(x, dict) else 0.5
        )

        # FIX 3: warn when scores are stuck at the fallback value
        if abs(traj_df["LCS"].mean() - 0.5) < 0.01:
            st.warning(
                "⚠️ LLM judge scores are all 0.5 — this is the fallback value. "
                "The judge JSON parse is likely failing. "
                "Check src/evid_rl_env/judge/llm_judge.py for parsing errors."
            )

        fig = px.line(traj_df, x="step", y=["LCS", "ESS", "HRS"], title="LLM Scores Over Time")
        st.plotly_chart(fig, width="stretch")

    if not demo_mode:
        if "value_estimate" in traj_df:
            fig = px.line(traj_df, x="step", y="value_estimate", title="Value Estimates")
            st.plotly_chart(fig, width="stretch")

        if "advantage" in traj_df:
            fig = px.line(traj_df, x="step", y="advantage", title="Advantage Signal")
            st.plotly_chart(fig, width="stretch")

        fig = px.line(df, x="episode", y="tokens", title="Token Usage")
        st.plotly_chart(fig, width="stretch")

    # BEST EPISODE
    if "reward" not in df.columns or df["reward"].dropna().empty:
        st.warning("No reward data yet (training may still be initializing)")
        best_ep = None
    else:
        best_idx = df["reward"].dropna().idxmax()
        best_ep  = df.loc[best_idx, "episode"]

    if best_ep is not None:
        colA, colB = st.columns(2)
        with colA:
            fig = px.line(df, x="episode", y="reward", title="Reward")
            st.plotly_chart(fig, width="stretch")
        with colB:
            fig = px.line(df, x="episode", y="reward_smooth", title="Smoothed")
            st.plotly_chart(fig, width="stretch")
        st.success(f"🏆 Best Episode: {int(best_ep)}")

    # ADD 5: Reward decomposition (uses already-loaded traj_by_episode)
    st.subheader("Reward decomposition")
    decomp_rows = []
    for ep, traj_ep in traj_by_episode.items():
        if not traj_ep:
            continue
        base_r = sum(s.get("reward", 0) - s.get("llm_reward", 0) for s in traj_ep)
        llm_r  = sum(s.get("llm_reward", 0) for s in traj_ep)
        decomp_rows.append({"episode": ep, "base reward": base_r, "llm reward": llm_r})
    if decomp_rows:
        decomp_df = pd.DataFrame(decomp_rows)
        fig_decomp = px.bar(
            decomp_df, x="episode", y=["base reward", "llm reward"],
            title="Reward sources per episode",
            barmode="relative",
            color_discrete_map={"base reward": "#378ADD", "llm reward": "#1D9E75"},
        )
        st.plotly_chart(fig_decomp, width="stretch")

    if not demo_mode:
        st.subheader("Train vs eval reward")
        eval_rows = (
            df.dropna(subset=["eval/mean_reward"])
            if "eval/mean_reward" in df.columns
            else pd.DataFrame()
        )
        if not eval_rows.empty:
            fig_gap = go.Figure()
            fig_gap.add_trace(go.Scatter(
                x=df["episode"], y=df["reward_smooth"],
                mode="lines", name="Train (smoothed)", line=dict(color="#378ADD")
            ))
            fig_gap.add_trace(go.Scatter(
                x=eval_rows["episode"], y=eval_rows["eval/mean_reward"],
                mode="lines+markers", name="Eval mean",
                error_y=dict(
                    type="data",
                    array=(
                        eval_rows["eval/std_reward"].fillna(0).tolist()
                        if "eval/std_reward" in eval_rows.columns else []
                    ),
                    visible=True,
                ),
                line=dict(color="#D85A30", dash="dash"),
            ))
            fig_gap.update_layout(title="Train vs eval reward — gap indicates overfitting")
            st.plotly_chart(fig_gap, width="stretch")
        else:
            st.info("Eval data not yet available. Set eval_every in Trainer to enable.")

    # Policy behavior
    st.subheader("Policy Behavior")

    # FIX 2: Entropy with full 0→ln(n_actions) y-axis so chart is honest
    fig = px.line(df, x="episode", y="entropy", title="Entropy (Exploration)")
    n_actions_ent = len(ACTIONS)
    if "action_probs" in traj_df.columns and len(traj_df) > 0:
        first_probs = traj_df["action_probs"].dropna()
        if not first_probs.empty and hasattr(first_probs.iloc[0], "__len__"):
            n_actions_ent = len(first_probs.iloc[0])
    max_entropy = math.log(n_actions_ent)
    fig.update_layout(yaxis=dict(range=[0, max_entropy * 1.05]))
    st.plotly_chart(fig, width="stretch")

    # ADD 3: Policy collapse detector
    if "entropy" in df.columns:
        latest_entropy = df.sort_values("episode")["entropy"].iloc[-1]
        max_entropy_val = math.log(len(ACTIONS))
        entropy_pct = latest_entropy / max_entropy_val * 100
        col_e1, col_e2, col_e3 = st.columns(3)
        with col_e1:
            st.metric("Current entropy", f"{latest_entropy:.4f}")
        with col_e2:
            st.metric("Max possible entropy", f"{max_entropy_val:.4f}")
        with col_e3:
            status = (
                "✅ Exploring"  if entropy_pct > 70
                else "⚠️ Narrowing" if entropy_pct > 40
                else "🚨 Collapsing"
            )
            st.metric("Policy status", status, f"{entropy_pct:.1f}% of max")

    if "num_steps" in df.columns:
        fig = px.line(df, x="episode", y="num_steps", title="Steps per episode")
        st.plotly_chart(fig, width="stretch")

    if "llm_reward" in traj_df.columns:
        traj_df["base_reward"] = traj_df["reward"] - traj_df.get("llm_reward", 0)
        fig = px.line(traj_df, x="step", y=["reward", "llm_reward"], title="Base vs LLM reward per step")
        st.plotly_chart(fig, width="stretch")

    if "curriculum_level" in df.columns:
        fig = px.line(df, x="episode", y="curriculum_level", title="Curriculum level")
        st.plotly_chart(fig, width="stretch")

    if "reward_raw" in df.columns and "reward" in df.columns:
        fig = px.line(df, x="episode", y=["reward", "reward_raw"], title="Normalised vs raw reward")
        st.plotly_chart(fig, width="stretch")

    action_dist_cols = [c for c in df.columns if c.startswith("action_dist.")]
    if action_dist_cols:
        fig = px.area(df, x="episode", y=action_dist_cols, title="Action distribution over time")
        st.plotly_chart(fig, width="stretch")
    else:
        st.info("Action distribution data not yet available (requires commit 5 trainer changes)")

    # ADD 4: Action frequency heatmap (uses traj_by_episode — no extra file reads)
    st.subheader("Action preference heatmap")
    if traj_by_episode:
        ep_action_rows = []
        for ep, traj_ep in traj_by_episode.items():
            if not traj_ep:
                continue
            action_counts = {}
            for step in traj_ep:
                a = step.get("action", "unknown")
                action_counts[a] = action_counts.get(a, 0) + 1
            row = {"episode": ep}
            row.update(action_counts)
            ep_action_rows.append(row)
        if ep_action_rows:
            heatmap_df   = pd.DataFrame(ep_action_rows).fillna(0).set_index("episode")
            action_cols_hm = [c for c in heatmap_df.columns]
            fig_hm = px.imshow(
                heatmap_df[action_cols_hm].T,
                title="Action frequency per episode (darker = more frequent)",
                labels=dict(x="Episode", y="Action", color="Count"),
                color_continuous_scale="Blues",
                aspect="auto",
            )
            st.plotly_chart(fig_hm, width="stretch")

# ── tab2: Compare ─────────────────────────────────────────────────────────────
with tab2:
    st.header("Compare Experiments")

    # ADD 2: Baseline comparison bar chart
    st.subheader("Baseline comparison")
    baselines_cmp = compute_baselines(data, selected)
    if baselines_cmp:
        baseline_rows = []
        for exp_name_cmp, vals in baselines_cmp.items():
            algo = configs.get(exp_name_cmp, {}).get("algo", "unknown")
            for baseline_name, val in vals.items():
                baseline_rows.append({
                    "experiment":  exp_name_cmp,
                    "algo":        algo,
                    "baseline":    baseline_name,
                    "mean_reward": val,
                })
        bl_df  = pd.DataFrame(baseline_rows)
        fig_bl = px.bar(
            bl_df, x="experiment", y="mean_reward", color="baseline",
            barmode="group",
            title="Trained agent vs baselines",
            color_discrete_map={
                "random (ep 0)":        "#888780",
                "early finalize":       "#E24B4A",
                "trained (last 3 eps)": "#1D9E75",
            },
        )
        fig_bl.add_hline(y=0, line_dash="dot", line_color="#888780",
                         annotation_text="zero reward")
        st.plotly_chart(fig_bl, width="stretch")

    # Smoothed training curves
    fig = go.Figure()
    for exp in selected:
        name = os.path.basename(exp)
        df   = data[data["experiment"] == name].copy()
        df["reward_smooth"] = df["reward"].rolling(10, min_periods=1).mean()

        if df.empty or "reward" not in df.columns or df["reward"].dropna().empty:
            continue

        algo_label  = configs.get(name, {}).get("algo", "unknown")
        trace_label = f"{name} ({algo_label})"
        fig.add_trace(go.Scatter(
            x=df["episode"], y=df["reward_smooth"],
            mode="lines", name=trace_label,
        ))

    # ADD 2: annotated baseline bands from eval_results.json
    first_exp       = selected[0]
    eval_data_cmp   = load_eval_results(first_exp)
    if eval_data_cmp:
        greedy_mean = eval_data_cmp["baselines"].get("greedy_llm", {}).get("mean")
        greedy_std  = eval_data_cmp["baselines"].get("greedy_llm", {}).get("std")
        best5_mean  = eval_data_cmp["baselines"].get("best_of_5",  {}).get("mean")
        best5_std   = eval_data_cmp["baselines"].get("best_of_5",  {}).get("std")

        if greedy_mean is not None and greedy_std is not None:
            fig.add_hrect(
                y0=greedy_mean - greedy_std, y1=greedy_mean + greedy_std,
                fillcolor="#D85A30", opacity=0.10, line_width=0,
                annotation_text="greedy_llm ± 1σ", annotation_position="top left",
            )
            fig.add_hline(y=greedy_mean, line_dash="dash", line_color="#D85A30", line_width=1.5)

        if best5_mean is not None and best5_std is not None:
            fig.add_hrect(
                y0=best5_mean - best5_std, y1=best5_mean + best5_std,
                fillcolor="#378ADD", opacity=0.10, line_width=0,
                annotation_text="best_of_5 ± 1σ", annotation_position="top right",
            )
            fig.add_hline(y=best5_mean, line_dash="dot", line_color="#378ADD", line_width=1.5)

        # Annotate first episode where smoothed reward holds above greedy for 3+ consecutive eps
        if greedy_mean is not None:
            for exp in selected:
                name     = os.path.basename(exp)
                df_conv  = data[data["experiment"] == name].copy()
                df_conv["reward_smooth"] = df_conv["reward"].rolling(10, min_periods=1).mean()
                df_conv  = df_conv.sort_values("episode").reset_index(drop=True)
                episodes = df_conv["episode"].tolist()
                smooths  = df_conv["reward_smooth"].tolist()

                conv_ep = None
                for i in range(len(episodes) - 2):
                    if (smooths[i]   > greedy_mean and
                        smooths[i+1] > greedy_mean and
                        smooths[i+2] > greedy_mean):
                        conv_ep = int(episodes[i])
                        break
                # Fallback: first single episode above greedy_mean
                if conv_ep is None:
                    above = df_conv[df_conv["reward_smooth"] > greedy_mean]
                    if not above.empty:
                        conv_ep = int(above.iloc[0]["episode"])

                if conv_ep is not None:
                    fig.add_vline(
                        x=conv_ep, line_dash="dash", line_color="#1D9E75",
                        annotation_text=f"beats greedy @ ep {conv_ep}",
                        annotation_position="top right",
                    )

    fig.update_layout(title="Training reward vs baselines")

    if not fig.data:
        st.warning("No reward data available for the selected experiments yet.")
    else:
        st.plotly_chart(fig, width="stretch")

    # Only show periodic eval chart if at least one experiment actually has eval rows.
    # eval/mean_reward is always a column (it's in FIXED_FIELDS) but is NaN for every
    # training row when eval_every hasn't fired yet — that produces a blank chart.
    has_eval_rows = (
        "eval/mean_reward" in data.columns
        and data["eval/mean_reward"].notna().any()
    )
    if has_eval_rows:
        st.subheader("Eval metrics")
        fig_eval = go.Figure()
        for exp in selected:
            name = os.path.basename(exp)
            df_e = data[data["experiment"] == name].dropna(subset=["eval/mean_reward"])
            if not df_e.empty:
                fig_eval.add_trace(go.Scatter(
                    x=df_e["episode"], y=df_e["eval/mean_reward"],
                    mode="lines", name=f"{name} eval",
                    error_y=dict(
                        type="data",
                        array=df_e.get("eval/std_reward", pd.Series()).fillna(0).tolist(),
                        visible=True,
                    ),
                ))
        if fig_eval.data:
            st.plotly_chart(fig_eval, width="stretch")
    else:
        st.info(
            "No periodic eval rounds recorded yet — increase training episodes past "
            "`eval_every` (currently 10) to see this chart."
        )

    st.subheader("LLM judge scores by experiment")
    fig_llm = go.Figure()
    for exp in selected:
        name = os.path.basename(exp)
        df   = data[data["experiment"] == name].copy()
        lcs_per_ep = []
        for ep in df["episode"].dropna().unique():
            traj = load_trajectory(exp, int(ep))
            if not traj:
                continue
            scores = [
                s.get("llm_scores", {}).get("LCS")
                for s in traj
                if isinstance(s.get("llm_scores"), dict)
            ]
            scores = [s for s in scores if s is not None]
            lcs_per_ep.append({
                "episode":  ep,
                "mean_LCS": float(np.mean(scores)) if scores else None,
            })
        if lcs_per_ep:
            lcs_df = pd.DataFrame(lcs_per_ep).dropna()
            fig_llm.add_trace(go.Scatter(
                x=lcs_df["episode"], y=lcs_df["mean_LCS"],
                mode="lines", name=f"{name} LCS",
            ))
    if fig_llm.data:
        st.plotly_chart(fig_llm, width="stretch")

# ── tab3: Episode Drilldown ───────────────────────────────────────────────────
with tab3:
    st.header("Episode Drilldown")

    exp_name = st.selectbox(
        "Experiment",
        [os.path.basename(e) for e in selected],
        key="drill"
    )

    exp_path = [e for e in selected if os.path.basename(e) == exp_name][0]
    df = data[data["experiment"] == exp_name].copy()

    df["episode"] = pd.to_numeric(df["episode"], errors="coerce")
    df["reward"]  = pd.to_numeric(df["reward"],  errors="coerce")
    df = df.dropna(subset=["episode", "reward"])

    if df.empty:
        st.warning("No episode data yet (training still initializing)")
    else:
        ep_min = int(df["episode"].min())
        ep_max = int(df["episode"].max())

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

            # ADD 7: Debate timeline — ACTION_SHORT maps full names to color keys
            st.subheader("Debate timeline")
            debate_html = "<div style='display:flex;flex-wrap:wrap;gap:8px;margin-bottom:16px;'>"
            for s_idx, s in enumerate(traj):
                action_raw   = s.get("action", "?")
                action_short = ACTION_SHORT.get(action_raw, action_raw.upper())
                reward_val   = s.get("reward", 0)
                color        = ACTION_COLORS.get(action_short, "#999")
                debate_html += (
                    f"<div style='background:{color};color:#fff;padding:4px 10px;"
                    f"border-radius:4px;font-size:0.82rem;font-family:monospace;'>"
                    f"<b>{s_idx + 1}.</b> {action_short} "
                    f"<span style='opacity:0.8'>({reward_val:+.2f})</span></div>"
                )
            debate_html += "</div>"
            st.markdown(debate_html, unsafe_allow_html=True)

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
                        fig = px.bar(
                            score_df, x="metric", y="score",
                            title="LLM judge scores",
                            range_y=[0, 1], color="metric",
                            color_discrete_map={
                                "HRS": "#E24B4A", "LCS": "#378ADD",
                                "ESS": "#1D9E75", "COMP": "#7F77DD",
                                "confidence": "#888780",
                            },
                        )
                        st.plotly_chart(fig, width="stretch")

                if not demo_mode:
                    st.metric("Tokens Used", step.get("tokens", 0))

                if step.get("query_count") is not None:
                    st.metric("Query count", step["query_count"])

                if not demo_mode:
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
                evidence_ids  = step.get("evidence_used") or step.get("selected_ids", [])
                evidence_pool = step.get("evidence_pool", [])
                for e in evidence_pool:
                    if e["id"] in (evidence_ids or []):
                        st.success(f"[{e['id']}] {e['text']}")

                if not demo_mode:
                    with st.expander("Raw step JSON", expanded=False):
                        st.json(step)

                st.subheader("Policy Distribution")
                probs      = step.get("action_probs")
                names      = step.get("action_names")
                chosen_idx = step.get("action_idx")

                # FIX 1: fall back to ACTIONS when action_names is absent or mismatched
                if probs:
                    if not names:
                        names = ACTIONS
                    if len(names) != len(probs):
                        names = [f"action_{i}" for i in range(len(probs))]

                    df_probs = pd.DataFrame({"action": names, "probability": probs})
                    fig = px.bar(df_probs, x="action", y="probability", title="Action Probabilities")
                    st.plotly_chart(fig, width="stretch")

                    if chosen_idx is not None:
                        st.success(f"Chosen Action: {names[chosen_idx]}")

                st.subheader("Policy Evolution")
                traj_df = pd.DataFrame(traj)

                if "entropy" in traj_df:
                    fig = px.line(traj_df, x="step", y="entropy", title="Entropy Over Steps")
                    # FIX 2: fixed y-axis shows true exploration range
                    max_ent = math.log(len(ACTIONS))
                    fig.update_layout(yaxis=dict(range=[0, max_ent * 1.05]))
                    st.plotly_chart(fig, width="stretch")

                st.subheader("Selected Evidence")
                selected_ids = step.get("selected_ids") or step.get("evidence_used", [])
                for e in evidence:
                    if e["id"] in selected_ids:
                        st.success(f"[{e['id']}] {e['text']}")

                # ADD 6: Evidence selection timeline
                # Infer evidence IDs from selected_ids delta between steps
                st.subheader("Evidence selection timeline")
                timeline_rows = []
                prev_selected = set()
                for s in traj:
                    action       = s.get("action", "")
                    curr_selected = set(s.get("selected_ids") or [])
                    if action == "select_evidence":
                        added = curr_selected - prev_selected
                        ev_id = next(iter(added), "?")
                        timeline_rows.append({
                            "step":        s.get("step", 0),
                            "action":      action,
                            "evidence_id": ev_id,
                        })
                    elif action == "remove_evidence":
                        removed = prev_selected - curr_selected
                        ev_id   = next(iter(removed), "?")
                        timeline_rows.append({
                            "step":        s.get("step", 0),
                            "action":      action,
                            "evidence_id": ev_id,
                        })
                    prev_selected = curr_selected

                if timeline_rows:
                    tl_df  = pd.DataFrame(timeline_rows)
                    fig_tl = px.scatter(
                        tl_df, x="step", y="evidence_id", color="action",
                        title="Evidence selection and removal over episode",
                        color_discrete_map={
                            "select_evidence": "#1D9E75",
                            "remove_evidence": "#E24B4A",
                        },
                        symbol="action",
                    )
                    st.plotly_chart(fig_tl, width="stretch")
                else:
                    st.info("No evidence select/remove actions in this episode")

# ── tab4: Config ──────────────────────────────────────────────────────────────
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
