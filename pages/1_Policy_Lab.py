import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

# --- Policy Lab utilities (additive) ---
def _shock_series(horizon: int, kind: str, start: int, size: float, duration: int, rho: float) -> np.ndarray:
    """
    Returns an additive shock series of length horizon+1 (Month 0..horizon).
    kind: "None", "Step", "Pulse", "AR(1) decay"
    start: month shock starts (0..horizon)
    size: shock magnitude in percentage points
    duration: months (for Pulse)
    rho: persistence/decay parameter (0..0.99)
    """
    s = np.zeros(horizon + 1, dtype=float)
    start = int(np.clip(start, 0, horizon))

    if kind == "None":
        return s

    if kind == "Step":
        s[start:] = size
        return s

    if kind == "Pulse":
        end = int(np.clip(start + max(int(duration), 1), 0, horizon + 1))
        s[start:end] = size
        return s

    if kind == "AR(1) decay":
        s[start] = size
        for t in range(start + 1, horizon + 1):
            s[t] = rho * s[t - 1]
        return s

    return s


def build_assumed_paths(
    pi0: float,
    y0: float,
    pi_target: float,
    horizon: int,
    pi_shock_kind: str,
    pi_shock_start: int,
    pi_shock_size: float,
    pi_shock_duration: int,
    pi_shock_rho: float,
    y_shock_kind: str,
    y_shock_start: int,
    y_shock_size: float,
    y_shock_duration: int,
    y_shock_rho: float,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Baseline assumption: linear drift toward target/zero.
    Additive shocks are layered on top.
    Returns: (pi_path, y_path), each length horizon+1
    """
    pi_base = np.linspace(pi0, pi_target, horizon + 1)
    y_base = np.linspace(y0, 0.0, horizon + 1)

    pi_shock = _shock_series(horizon, pi_shock_kind, pi_shock_start, pi_shock_size, pi_shock_duration, pi_shock_rho)
    y_shock = _shock_series(horizon, y_shock_kind, y_shock_start, y_shock_size, y_shock_duration, y_shock_rho)

    return (pi_base + pi_shock, y_base + y_shock)


# --- Session state for saved runs (additive) ---
if "policy_lab_runs" not in st.session_state:
    st.session_state.policy_lab_runs = []  # list[dict], each dict has keys: label, params, df
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
    st.markdown("### Presets & shocks (assumed paths)")

    preset = st.selectbox(
        "Preset (adds structured shocks; does not change your inputs)",
        ["Custom", "Demand boom (hot economy)", "Supply inflation (cost push)", "Recession (demand slump)", "Disinflation + recovery"],
        index=0,
    )

    # Default shock suggestions by preset (users can override via sliders below)
    preset_defaults = {
        "Custom": dict(
            pi_kind="None", pi_start=0, pi_size=0.0, pi_duration=6, pi_rho=0.7,
            y_kind="None", y_start=0, y_size=0.0, y_duration=6, y_rho=0.7
        ),
        "Demand boom (hot economy)": dict(
            pi_kind="AR(1) decay", pi_start=0, pi_size=0.6, pi_duration=6, pi_rho=0.85,
            y_kind="Step", y_start=0, y_size=1.5, y_duration=6, y_rho=0.7
        ),
        "Supply inflation (cost push)": dict(
            pi_kind="Step", pi_start=0, pi_size=1.2, pi_duration=6, pi_rho=0.7,
            y_kind="Pulse", y_start=1, y_size=-1.0, y_duration=6, y_rho=0.7
        ),
        "Recession (demand slump)": dict(
            pi_kind="AR(1) decay", pi_start=1, pi_size=-0.8, pi_duration=6, pi_rho=0.85,
            y_kind="Step", y_start=0, y_size=-2.5, y_duration=6, y_rho=0.7
        ),
        "Disinflation + recovery": dict(
            pi_kind="Step", pi_start=0, pi_size=-0.6, pi_duration=6, pi_rho=0.7,
            y_kind="AR(1) decay", y_start=0, y_size=1.0, y_duration=6, y_rho=0.8
        ),
    }

    d = preset_defaults[preset]

    with st.expander("Shock builder (additive on top of linear drift)", expanded=False):
        st.markdown("**Inflation shock (Δπ, percentage points)**")
        pi_shock_kind = st.selectbox("Inflation shock type", ["None", "Step", "Pulse", "AR(1) decay"], index=["None","Step","Pulse","AR(1) decay"].index(d["pi_kind"]))
        pi_shock_start = st.slider("π shock start month", 0, horizon, int(d["pi_start"]), 1)
        pi_shock_size = st.slider("π shock size (pp)", -3.0, 3.0, float(d["pi_size"]), 0.1)
        pi_shock_duration = st.slider("π pulse duration (months)", 1, max(2, horizon), int(d["pi_duration"]), 1)
        pi_shock_rho = st.slider("π AR(1) persistence (ρ)", 0.0, 0.99, float(d["pi_rho"]), 0.01)

        st.markdown("**Output gap shock (Δỹ, percentage points)**")
        y_shock_kind = st.selectbox("Output gap shock type", ["None", "Step", "Pulse", "AR(1) decay"], index=["None","Step","Pulse","AR(1) decay"].index(d["y_kind"]))
        y_shock_start = st.slider("ỹ shock start month", 0, horizon, int(d["y_start"]), 1)
        y_shock_size = st.slider("ỹ shock size (pp)", -5.0, 5.0, float(d["y_size"]), 0.1)
        y_shock_duration = st.slider("ỹ pulse duration (months)", 1, max(2, horizon), int(d["y_duration"]), 1)
        y_shock_rho = st.slider("ỹ AR(1) persistence (ρ)", 0.0, 0.99, float(d["y_rho"]), 0.01)

with colB:
    st.markdown("### Resulting policy path (illustrative)")

    # Taylor rule (nominal rate i)
    i_star = r_star + pi_target + phi_pi * (pi - pi_target) + phi_y * (y_gap)

    # Build an illustrative path with smoothing
    months = np.arange(horizon + 1)
    i_path = np.zeros_like(months, dtype=float)
    i_path[0] = i_star  # initial recommendation

    # For illustration, we let inflation/output gap drift toward targets, with optional additive shocks
    # (Presets/shocks are additive and do not modify the policy rule.)
    try:
        pi_path, y_path = build_assumed_paths(
            pi0=pi,
            y0=y_gap,
            pi_target=pi_target,
            horizon=horizon,
            pi_shock_kind=pi_shock_kind,
            pi_shock_start=pi_shock_start,
            pi_shock_size=pi_shock_size,
            pi_shock_duration=pi_shock_duration,
            pi_shock_rho=pi_shock_rho,
            y_shock_kind=y_shock_kind,
            y_shock_start=y_shock_start,
            y_shock_size=y_shock_size,
            y_shock_duration=y_shock_duration,
            y_shock_rho=y_shock_rho,
        )
    except NameError:
        # If shock builder expander was never opened (variables not defined), fall back gracefully
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
    # Optional: overlay previously saved runs for comparison (policy rate only)
    overlay_saved = st.checkbox("Overlay saved runs (policy rate)", value=(len(st.session_state.policy_lab_runs) > 0))
    if overlay_saved and len(st.session_state.policy_lab_runs) > 0:
        for run in st.session_state.policy_lab_runs[-8:]:  # keep it readable; last 8 runs
            run_df = run["df"]
            fig.add_trace(
                go.Scatter(
                    x=run_df["Month"],
                    y=run_df["Policy rate (illustrative, %)"],
                    mode="lines",
                    name=f"Saved: {run['label']}",
                    line=dict(dash="dot"),
                )
            )
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
