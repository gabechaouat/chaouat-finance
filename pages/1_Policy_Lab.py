import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

st.set_page_config(page_title="Policy Lab — Chaouat Economics Lab", page_icon="🧪", layout="wide")

st.title("Policy Lab")
st.caption("Interactive policy experiments with explicit assumptions. Educational use only.")

st.subheader("Module 1 — Monetary Policy & Inflation (Taylor Rule Toy Model)")

with st.expander("What this is (and what it is not)", expanded=True):
    st.write(
        "This lab is a simplified **teaching model**: it illustrates how a rule-based central bank might adjust a policy rate "
        "in response to inflation and the output gap. It does **not** forecast real outcomes and it does **not** represent a full macro model."
    )

colA, colB = st.columns([1.1, 1.4], gap="large")

with colA:
    st.markdown("### Assumptions & Parameters")

    r_star = st.slider("Neutral real rate r*", 0.0, 4.0, 2.0, 0.25)
    pi_target = st.slider("Inflation target π*", 0.0, 5.0, 2.0, 0.25)
    phi_pi = st.slider("Response to inflation gap (φπ)", 0.0, 3.0, 1.5, 0.1)
    phi_y = st.slider("Response to output gap (φy)", 0.0, 2.0, 0.5, 0.1)

    st.markdown("### Scenario inputs")
    pi = st.slider("Current inflation π (%)", 0.0, 12.0, 3.5, 0.1)
    y_gap = st.slider("Output gap ỹ (%)", -8.0, 8.0, -0.5, 0.1)

    smoothing = st.slider("Rate smoothing (0 = none)", 0.0, 0.9, 0.6, 0.05)
    horizon = st.slider("Scenario horizon (months)", 6, 36, 18, 1)

with colB:
    st.markdown("### Resulting policy path (illustrative)")

    # Taylor rule (nominal rate i)
    i_star = r_star + pi_target + phi_pi * (pi - pi_target) + phi_y * (y_gap)

    # Build an illustrative path with smoothing
    months = np.arange(horizon + 1)
    i_path = np.zeros_like(months, dtype=float)
    i_path[0] = i_star  # initial recommendation

    # For illustration, we let inflation/output gap gradually drift toward targets
    pi_path = np.linspace(pi, pi_target, horizon + 1)
    y_path = np.linspace(y_gap, 0.0, horizon + 1)

    for t in range(1, horizon + 1):
        i_t = r_star + pi_target + phi_pi * (pi_path[t] - pi_target) + phi_y * (y_path[t])
        i_path[t] = smoothing * i_path[t - 1] + (1 - smoothing) * i_t

    df = pd.DataFrame({
        "Month": months,
        "Policy rate (illustrative, %)": i_path,
        "Inflation path (assumed, %)": pi_path,
        "Output gap path (assumed, %)": y_path,
    })

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df["Month"], y=df["Policy rate (illustrative, %)"], mode="lines", name="Policy rate"))
    fig.add_trace(go.Scatter(x=df["Month"], y=df["Inflation path (assumed, %)"], mode="lines", name="Inflation (assumed)"))
    fig.update_layout(height=430, margin=dict(l=10, r=10, t=30, b=10), legend=dict(orientation="h", y=1.02, x=0))
    fig.update_xaxes(title_text="Months (scenario)")
    fig.update_yaxes(title_text="Percent (%)")
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("### Interpretation (teaching-oriented)")
    st.write(
        f"- Under these parameters, the rule implies an initial nominal policy rate around **{i_star:.2f}%**.\n"
        "- Rate smoothing makes the path adjust gradually rather than jump.\n"
        "- Because this is a toy model, treat the chart as a way to reason about *mechanisms*, not outcomes."
    )

st.divider()
st.markdown("### Export (for tutors)")
csv = df.to_csv(index=False).encode("utf-8")
st.download_button("Download scenario CSV", data=csv, file_name="policy_lab_monetary_scenario.csv", mime="text/csv")
