from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

st.set_page_config(page_title="Prompt Optimisation Lab", page_icon="🧪", layout="wide")
st.title("Prompt Optimisation Lab")
st.caption("Cross-model prompt architecture, ablation, transfer and efficiency research")

root = Path("artifacts/demo")
if not (root / "summary.json").exists():
    st.info("Run `python scripts/run_research_demo.py` first.")
    st.stop()

summary = json.loads((root / "summary.json").read_text(encoding="utf-8"))
effects = pd.read_csv(root / "component_effects.csv")
frontier = pd.read_csv(root / "pareto_frontier.csv")
transfer = pd.read_csv(root / "cross_model_transfer.csv")
robustness = pd.read_csv(root / "robustness_retention.csv")

c1, c2, c3, c4 = st.columns(4)
c1.metric("Core trials", f"{summary['trials']:,}")
c2.metric("Stress trials", f"{summary['robustness_trials']:,}")
c3.metric("Prompt architectures", summary["variants"])
c4.metric("Model profiles", summary["models"])

st.subheader("Different profiles, different winning prompts")
best = pd.DataFrame(summary["best_by_model"].values())
st.dataframe(
    best[["provider", "variant", "mean_quality", "pass_rate", "mean_prompt_tokens"]],
    hide_index=True,
    use_container_width=True,
)

st.subheader("Prompt-component effects")
effect_chart = px.bar(
    effects,
    x="component",
    y="effect",
    color="model",
    barmode="group",
    error_y=effects["ci95_high"] - effects["effect"],
    error_y_minus=effects["effect"] - effects["ci95_low"],
    labels={"effect": "Estimated quality effect"},
)
st.plotly_chart(effect_chart, use_container_width=True)

left, right = st.columns(2)
with left:
    st.subheader("Quality–prompt length frontier")
    st.plotly_chart(
        px.scatter(
            frontier,
            x="mean_prompt_tokens",
            y="mean_quality",
            color="provider",
            hover_name="variant",
            size="mean_latency_ms",
        ),
        use_container_width=True,
    )
with right:
    st.subheader("Cross-model transfer regret")
    pivot = transfer.pivot(index="source_model", columns="target_model", values="transfer_regret")
    st.plotly_chart(
        px.imshow(pivot, text_auto=".3f", color_continuous_scale="Reds"),
        use_container_width=True,
    )

st.subheader("Robustness retention")
st.plotly_chart(
    px.box(robustness, x="model", y="retention", color="model", points="all"),
    use_container_width=True,
)

with st.expander("Evidence boundary"):
    st.write(summary["evidence_boundary"])
