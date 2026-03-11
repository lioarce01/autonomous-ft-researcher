"""Streamlit live dashboard for IFEval fine-tuning experiments."""
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import os
import sys

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)

from db import get_all_experiments

BASELINE = 0.612

st.set_page_config(
    page_title="IFEval Fine-Tuning Researcher",
    page_icon="🧪",
    layout="wide",
)

st.title("🧪 IFEval Fine-Tuning Researcher")
st.caption("Qwen3.5-2B · Baseline: 0.612 · Target: beat Qwen3-4B (0.834)")

# Auto-refresh every 30 seconds
st.markdown(
    """<meta http-equiv="refresh" content="30">""",
    unsafe_allow_html=True,
)

experiments = get_all_experiments()

if not experiments:
    st.info("No experiments logged yet. Run your first experiment!")
    st.stop()

df = pd.DataFrame(experiments)
df["delta"] = df["accuracy"] - BASELINE
df["delta_str"] = df["delta"].apply(lambda x: f"+{x:.4f}" if x >= 0 else f"{x:.4f}")
df["kept_str"] = df["kept"].apply(lambda x: "✓" if x == 1 else "")

# ── KPI row ──────────────────────────────────────────────────────────────────
best = df[df["kept"] == 1]["accuracy"].max() if (df["kept"] == 1).any() else None
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("Experiments Run", len(df))
with col2:
    st.metric("Baseline (raw)", f"{BASELINE:.3f}")
with col3:
    if best:
        delta = best - BASELINE
        st.metric("Best Accuracy", f"{best:.4f}", delta=f"{delta:+.4f}")
    else:
        st.metric("Best Accuracy", "—")
with col4:
    target = 0.834
    if best:
        gap = target - best
        st.metric("Gap to Qwen3-4B (0.834)", f"{gap:.4f}")
    else:
        st.metric("Gap to Qwen3-4B (0.834)", f"{target - BASELINE:.4f}")

st.divider()

# ── Accuracy over experiments chart ─────────────────────────────────────────
st.subheader("Accuracy by Experiment")

df_sorted = df.sort_values("timestamp")
fig = go.Figure()

# All experiments
fig.add_trace(go.Bar(
    x=df_sorted["name"],
    y=df_sorted["accuracy"],
    marker_color=["#2ecc71" if k == 1 else "#3498db" for k in df_sorted["kept"]],
    name="accuracy",
    text=df_sorted["accuracy"].apply(lambda x: f"{x:.4f}"),
    textposition="outside",
))

# Baseline line
fig.add_hline(y=BASELINE, line_dash="dash", line_color="orange",
              annotation_text=f"Baseline {BASELINE}", annotation_position="bottom right")

# Target line
fig.add_hline(y=0.834, line_dash="dot", line_color="red",
              annotation_text="Qwen3-4B 0.834", annotation_position="top right")

fig.update_layout(
    xaxis_title="Experiment",
    yaxis_title="prompt_level_strict_acc",
    yaxis_range=[max(0, df_sorted["accuracy"].min() - 0.05), min(1.0, df_sorted["accuracy"].max() + 0.05)],
    showlegend=False,
    height=400,
)
st.plotly_chart(fig, use_container_width=True)

# ── Leaderboard table ────────────────────────────────────────────────────────
st.subheader("Leaderboard")
display_cols = ["name", "accuracy", "delta_str", "kept_str", "notes", "hypothesis", "timestamp"]
st.dataframe(
    df[display_cols].rename(columns={
        "name": "Name", "accuracy": "Accuracy", "delta_str": "Delta",
        "kept_str": "Kept", "notes": "Notes", "hypothesis": "Hypothesis",
        "timestamp": "Timestamp",
    }),
    use_container_width=True,
    hide_index=True,
)

# ── Notes sidebar ────────────────────────────────────────────────────────────
notes_path = os.path.join(ROOT, "NOTES.md")
if os.path.exists(notes_path):
    with st.expander("Research Notes (NOTES.md)"):
        with open(notes_path, "r", encoding="utf-8") as f:
            st.markdown(f.read())
