import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import streamlit as st
from datetime import datetime

st.set_page_config(page_title="Policy Lab — Chaouat Economics Lab", page_icon="🧪", layout="wide")

# =====================================================================
# SHARED STYLE — matches app.py editorial aesthetic
# =====================================================================
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Serif+Display:ital@0;1&family=DM+Sans:wght@300;400;500;600&display=swap');

:root {
  --ink: #1a1814;
  --ink-muted: #6b6760;
  --ink-faint: #b0ada8;
  --cream: #faf8f4;
  --warm: #f2ede4;
  --rule: #e0dbd2;
  --accent: #c9622a;
  --accent-light: #f7ece3;
  --teal: #2a7a6f;
  --teal-light: #e3f2ef;
  --gold: #b08735;
  --gold-light: #f5edda;
  --blue: #2a5fa5;
  --blue-light: #e8eef8;
}

html, body, * { font-family: 'DM Sans', sans-serif !important; }
.block-container { padding-top: 0 !important; padding-bottom: 3rem; max-width: 1240px; }

/* ---- masthead ---- */
.pl-masthead {
  border-bottom: 3px double var(--rule);
  padding: 28px 0 18px 0;
  margin-bottom: 0;
}
.pl-eyebrow {
  font-size: 10px; letter-spacing: 3px; text-transform: uppercase;
  color: var(--ink-muted); margin-bottom: 8px;
}
.pl-title {
  font-family: 'DM Serif Display', serif !important;
  font-size: 52px; line-height: 1.0; color: var(--ink);
  margin: 0 0 8px 0; letter-spacing: -0.5px;
}
.pl-sub { font-size: 15px; color: var(--ink-muted); font-weight: 300; }

/* ---- section labels ---- */
.section-label {
  font-size: 10px; letter-spacing: 3px; text-transform: uppercase;
  color: var(--ink-muted); border-top: 1px solid var(--rule);
  padding-top: 10px; margin: 28px 0 16px 0;
}
.section-label-inline {
  font-size: 10px; letter-spacing: 3px; text-transform: uppercase;
  color: var(--ink-muted); margin-bottom: 10px;
}

/* ---- module tabs ---- */
.module-tab-row {
  display: flex; gap: 0; border-bottom: 2px solid var(--rule);
  margin-bottom: 24px;
}

/* ---- param panel ---- */
.param-panel {
  background: var(--warm); border: 1px solid var(--rule);
  border-radius: 4px; padding: 20px 18px; height: 100%;
}
.param-title {
  font-family: 'DM Serif Display', serif !important;
  font-size: 18px; color: var(--ink); margin: 0 0 4px 0;
}
.param-desc {
  font-size: 12.5px; color: var(--ink-muted);
  line-height: 1.55; margin: 0 0 16px 0;
}
.param-group {
  font-size: 10px; letter-spacing: 2px; text-transform: uppercase;
  color: var(--ink-faint); margin: 14px 0 6px 0; border-top: 1px solid var(--rule);
  padding-top: 10px;
}

/* ---- result cards ---- */
.result-card {
  background: var(--ink); color: #fff;
  border-radius: 4px; padding: 16px 18px; text-align: center;
}
.result-card-accent { background: var(--accent); }
.result-card-teal   { background: var(--teal); }
.result-card-gold   { background: var(--gold); }
.result-card-blue   { background: var(--blue); }
.result-num {
  font-family: 'DM Serif Display', serif !important;
  font-size: 28px; line-height: 1; color: #fff; margin: 0;
}
.result-label { font-size: 11px; opacity: 0.7; text-transform: uppercase; letter-spacing: 1.5px; margin-top: 4px; }

/* ---- interpretation box ---- */
.interp-box {
  border-left: 3px solid var(--accent);
  background: var(--accent-light);
  padding: 14px 18px; border-radius: 0 4px 4px 0;
  margin-top: 16px;
}
.interp-box-teal {
  border-left: 3px solid var(--teal);
  background: var(--teal-light);
}
.interp-box-gold {
  border-left: 3px solid var(--gold);
  background: var(--gold-light);
}
.interp-title { font-weight: 500; font-size: 13px; color: var(--ink); margin: 0 0 4px 0; }
.interp-text  { font-size: 13px; color: var(--ink-muted); line-height: 1.6; margin: 0; }

/* ---- buttons ---- */
div.stButton > button {
  background: var(--ink) !important; color: #fff !important;
  border: none !important; border-radius: 3px !important;
  font-size: 11px !important; letter-spacing: 1.5px !important;
  text-transform: uppercase !important; font-weight: 500 !important;
  padding: 9px 16px !important; width: 100%;
}
div.stButton > button:hover { background: var(--accent) !important; }

/* ---- download button ---- */
div.stDownloadButton > button {
  background: transparent !important; color: var(--ink) !important;
  border: 1px solid var(--rule) !important; border-radius: 3px !important;
  font-size: 11px !important; letter-spacing: 1.5px !important;
  text-transform: uppercase !important;
}

/* ---- tabs override ---- */
[data-baseweb="tab-list"] { border-bottom: 2px solid var(--rule) !important; gap: 0 !important; }
[data-baseweb="tab"] {
  font-size: 11px !important; letter-spacing: 2px !important;
  text-transform: uppercase !important; color: var(--ink-muted) !important;
  padding: 10px 20px !important; background: transparent !important;
  border-bottom: 2px solid transparent !important; margin-bottom: -2px !important;
}
[aria-selected="true"][data-baseweb="tab"] {
  color: var(--ink) !important; border-bottom: 2px solid var(--accent) !important;
  font-weight: 500 !important;
}
[data-baseweb="tab-highlight"] { display: none !important; }
[data-baseweb="tab-border"] { display: none !important; }

/* ---- expander ---- */
[data-testid="stExpander"] summary {
  font-size: 12px !important; letter-spacing: 1.5px !important;
  text-transform: uppercase !important; color: var(--ink-muted) !important;
}

/* ---- slider labels ---- */
[data-testid="stSlider"] label { font-size: 12px !important; color: var(--ink-muted) !important; }

/* ---- metric ---- */
[data-testid="stMetricValue"] { font-size: 22px !important; font-family: 'DM Serif Display', serif !important; }
[data-testid="stMetricLabel"] { font-size: 11px !important; letter-spacing: 1px !important; text-transform: uppercase !important; }

/* ---- saved run chip ---- */
.run-chip {
  display: inline-block; background: var(--warm); border: 1px solid var(--rule);
  border-radius: 3px; padding: 4px 10px; font-size: 12px; color: var(--ink);
  margin: 3px 3px 3px 0;
}

/* ---- footer ---- */
.site-footer {
  border-top: 3px double var(--rule); padding-top: 16px; margin-top: 40px;
  font-size: 12px; color: var(--ink-muted); text-align: center;
}
</style>
""", unsafe_allow_html=True)

# =====================================================================
# PLOTLY THEME — matches editorial palette
# =====================================================================
COLORS = {
    "accent":  "#c9622a",
    "teal":    "#2a7a6f",
    "gold":    "#b08735",
    "blue":    "#2a5fa5",
    "ink":     "#1a1814",
    "muted":   "#b0ada8",
    "rule":    "#e0dbd2",
    "cream":   "#faf8f4",
}

PALETTE = [COLORS["accent"], COLORS["teal"], COLORS["gold"], COLORS["blue"], COLORS["ink"]]

def editorial_layout(height=400, title=""):
    return dict(
        height=height,
        paper_bgcolor=COLORS["cream"],
        plot_bgcolor=COLORS["cream"],
        font=dict(family="DM Sans", color=COLORS["ink"], size=12),
        title=dict(text=title, font=dict(family="DM Serif Display", size=16, color=COLORS["ink"]), x=0, xanchor="left"),
        margin=dict(l=12, r=12, t=40 if title else 20, b=12),
        legend=dict(orientation="h", y=1.08, x=0, font=dict(size=11)),
        xaxis=dict(gridcolor=COLORS["rule"], linecolor=COLORS["rule"], tickfont=dict(size=11)),
        yaxis=dict(gridcolor=COLORS["rule"], linecolor=COLORS["rule"], tickfont=dict(size=11)),
        colorway=PALETTE,
    )

# =====================================================================
# SESSION STATE
# =====================================================================
if "policy_lab_runs" not in st.session_state:
    st.session_state.policy_lab_runs = []
if "debt_runs" not in st.session_state:
    st.session_state.debt_runs = []
if "trade_runs" not in st.session_state:
    st.session_state.trade_runs = []

# =====================================================================
# MASTHEAD
# =====================================================================
st.markdown("""
<div class="pl-masthead">
  <div class="pl-eyebrow">Chaouat Economics Lab · Interactive Modules</div>
  <div class="pl-title">Policy Lab</div>
  <div class="pl-sub">Run economic experiments. Adjust parameters. Observe mechanisms — not forecasts.</div>
</div>
""", unsafe_allow_html=True)

# =====================================================================
# MODULE TABS
# =====================================================================
tab1, tab2, tab3, tab4 = st.tabs([
    "01 · Monetary Policy",
    "02 · Fiscal & Multipliers",
    "03 · Debt Dynamics",
    "04 · Trade & Incidence",
])

# =====================================================================
# ── MODULE 1: MONETARY POLICY / TAYLOR RULE ──────────────────────────
# =====================================================================
with tab1:
    st.markdown('<div class="section-label">Taylor Rule Toy Model — Monetary Policy</div>', unsafe_allow_html=True)

    col_params, col_charts = st.columns([1, 2.2], gap="large")

    with col_params:
        st.markdown("""
        <div class="param-panel">
          <div class="param-title">Parameters</div>
          <div class="param-desc">Adjust the rule-based central bank response to inflation and output conditions.</div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown('<div class="param-group">Rule parameters</div>', unsafe_allow_html=True)
        r_star   = st.slider("Neutral real rate r* (%)", 0.0, 4.0, 2.0, 0.25, key="t1_rstar")
        pi_target = st.slider("Inflation target π* (%)", 0.0, 5.0, 2.0, 0.25, key="t1_pitarget")
        phi_pi   = st.slider("Inflation response φπ", 0.0, 3.0, 1.5, 0.1, key="t1_phipi")
        phi_y    = st.slider("Output gap response φy", 0.0, 2.0, 0.5, 0.1, key="t1_phiy")
        smoothing = st.slider("Rate smoothing ρ", 0.0, 0.9, 0.6, 0.05, key="t1_smooth")

        st.markdown('<div class="param-group">Initial conditions</div>', unsafe_allow_html=True)
        pi_now   = st.slider("Current inflation π (%)", 0.0, 12.0, 3.5, 0.1, key="t1_pi")
        y_gap    = st.slider("Output gap ỹ (%)", -8.0, 8.0, -0.5, 0.1, key="t1_ygap")
        horizon  = st.slider("Horizon (months)", 6, 48, 24, 1, key="t1_horizon")

        st.markdown('<div class="param-group">Shock scenario</div>', unsafe_allow_html=True)
        preset = st.selectbox("Preset", [
            "None",
            "Demand boom",
            "Supply shock (cost push)",
            "Recession",
            "Disinflation + recovery",
        ], key="t1_preset")

        preset_map = {
            "None":                      dict(pk="None", ps=0.0, pρ=0.7, yk="None", ys=0.0, yρ=0.7),
            "Demand boom":               dict(pk="AR(1)", ps=0.6, pρ=0.85, yk="Step", ys=1.5, yρ=0.7),
            "Supply shock (cost push)":  dict(pk="Step",  ps=1.2, pρ=0.7,  yk="Pulse", ys=-1.0, yρ=0.7),
            "Recession":                 dict(pk="AR(1)", ps=-0.8, pρ=0.85, yk="Step", ys=-2.5, yρ=0.7),
            "Disinflation + recovery":   dict(pk="Step",  ps=-0.6, pρ=0.7,  yk="AR(1)", ys=1.0, yρ=0.8),
        }
        d = preset_map[preset]

    # ---- compute paths ----
    def shock_series(H, kind, start, size, rho):
        s = np.zeros(H+1)
        if kind == "None":  return s
        if kind == "Step":  s[start:] = size; return s
        if kind == "Pulse": s[max(0,start):min(H+1,start+6)] = size; return s
        s[start] = size
        for t in range(start+1, H+1): s[t] = rho * s[t-1]
        return s

    H = horizon
    months = np.arange(H+1)
    pi_base = np.linspace(pi_now, pi_target, H+1)
    y_base  = np.linspace(y_gap, 0.0, H+1)
    pi_path = pi_base + shock_series(H, d["pk"], 0, d["ps"], d["pρ"])
    y_path  = y_base  + shock_series(H, d["yk"], 0, d["ys"], d["yρ"])

    # Taylor implied rate (no smoothing)
    i_implied = r_star + pi_target + phi_pi*(pi_path - pi_target) + phi_y*y_path

    # Smoothed path
    i_smooth = np.zeros(H+1)
    i_smooth[0] = i_implied[0]
    for t in range(1, H+1):
        i_smooth[t] = smoothing*i_smooth[t-1] + (1-smoothing)*i_implied[t]

    # Real rate implied
    r_real = i_smooth - pi_path

    # Decomposition
    base_comp  = r_star + pi_target
    pi_contrib = phi_pi*(pi_path - pi_target)
    y_contrib  = phi_y*y_path

    with col_charts:

        # ── Chart 1: Taylor Rate Fan (implied + smoothed + real) ──
        fig1 = go.Figure()
        fig1.add_trace(go.Scatter(
            x=months, y=i_implied, name="Implied (no smoothing)",
            line=dict(color=COLORS["muted"], width=1.5, dash="dot"), mode="lines"
        ))
        fig1.add_trace(go.Scatter(
            x=months, y=i_smooth, name="Policy rate (smoothed)",
            line=dict(color=COLORS["accent"], width=2.5), mode="lines"
        ))
        fig1.add_trace(go.Scatter(
            x=months, y=r_real, name="Real rate (smoothed − π)",
            line=dict(color=COLORS["teal"], width=2), mode="lines"
        ))
        fig1.add_hline(y=0, line_color=COLORS["rule"], line_width=1)
        fig1.add_hline(y=pi_target, line_color=COLORS["gold"],
                       line_width=1, line_dash="dot",
                       annotation_text=f"π* = {pi_target}%",
                       annotation_font_color=COLORS["gold"],
                       annotation_position="right")
        fig1.update_layout(**editorial_layout(340, "Policy rate path"))
        fig1.update_yaxes(title_text="Rate (%)")
        fig1.update_xaxes(title_text="Months")
        st.plotly_chart(fig1, use_container_width=True)

        # ── Chart 2: Stacked area decomposition ──
        fig2 = make_subplots(rows=1, cols=2,
            subplot_titles=["Rate decomposition (Month 0)", "Assumed paths: π and ỹ"],
            horizontal_spacing=0.08)

        # Waterfall-style bar for decomposition at t=0
        cats   = ["Base (r*+π*)", "Infl. gap", "Output gap", "Implied rate"]
        vals   = [base_comp, pi_contrib[0], y_contrib[0], i_implied[0]]
        colors_bar = [COLORS["ink"], COLORS["accent"], COLORS["teal"], COLORS["gold"]]
        fig2.add_trace(go.Bar(
            x=cats, y=vals, marker_color=colors_bar,
            text=[f"{v:.2f}%" for v in vals], textposition="outside",
            textfont=dict(size=11), showlegend=False
        ), row=1, col=1)

        # Assumed paths
        fig2.add_trace(go.Scatter(
            x=months, y=pi_path, name="Inflation (assumed)",
            line=dict(color=COLORS["accent"], width=2), mode="lines"
        ), row=1, col=2)
        fig2.add_trace(go.Scatter(
            x=months, y=y_path, name="Output gap (assumed)",
            line=dict(color=COLORS["teal"], width=2), mode="lines"
        ), row=1, col=2)
        fig2.add_hline(y=0, row=1, col=2, line_color=COLORS["rule"], line_width=1)
        fig2.add_hline(y=pi_target, row=1, col=2, line_color=COLORS["gold"],
                       line_width=1, line_dash="dot")

        fig2.update_layout(
            height=300,
            paper_bgcolor=COLORS["cream"], plot_bgcolor=COLORS["cream"],
            font=dict(family="DM Sans", color=COLORS["ink"], size=11),
            margin=dict(l=12, r=12, t=40, b=12),
            legend=dict(orientation="h", y=1.12, x=0.5, font=dict(size=10)),
            colorway=PALETTE,
        )
        for ax in ["xaxis", "xaxis2", "yaxis", "yaxis2"]:
            fig2.update_layout(**{ax: dict(gridcolor=COLORS["rule"], linecolor=COLORS["rule"])})
        st.plotly_chart(fig2, use_container_width=True)

        # ── Chart 3: Phase diagram — inflation gap vs output gap ──
        st.markdown('<div class="section-label-inline" style="margin-top:8px;">Phase diagram — central bank reaction surface</div>', unsafe_allow_html=True)

        # Grid
        pi_grid = np.linspace(-2, 6, 60)
        y_grid  = np.linspace(-5, 5, 60)
        PG, YG = np.meshgrid(pi_grid, y_grid)
        IG = r_star + pi_target + phi_pi*(PG - pi_target) + phi_y*YG

        fig3 = go.Figure()
        fig3.add_trace(go.Contour(
            x=pi_grid, y=y_grid, z=IG,
            colorscale=[[0, COLORS["teal"]], [0.5, COLORS["cream"]], [1, COLORS["accent"]]],
            contours=dict(showlabels=True, labelfont=dict(size=10, color=COLORS["ink"])),
            colorbar=dict(title="Rate (%)", tickfont=dict(size=10)),
            line=dict(width=0.5),
        ))
        # Trace the scenario path on the phase diagram
        fig3.add_trace(go.Scatter(
            x=pi_path - pi_target, y=y_path,
            mode="lines+markers",
            line=dict(color=COLORS["ink"], width=2),
            marker=dict(size=5, color=COLORS["accent"],
                        symbol=["circle"]*H + ["star"]),
            name="Scenario path",
        ))
        fig3.add_vline(x=0, line_color=COLORS["ink"], line_width=1, line_dash="dot")
        fig3.add_hline(y=0, line_color=COLORS["ink"], line_width=1, line_dash="dot")
        fig3.update_layout(**editorial_layout(360, "Rate implied by Taylor rule (contour map)"))
        fig3.update_xaxes(title_text="Inflation gap (π − π*)")
        fig3.update_yaxes(title_text="Output gap (ỹ)")
        st.plotly_chart(fig3, use_container_width=True)

    # ── Result cards ──
    i0 = i_smooth[0]
    i_end = i_smooth[-1]
    r_real0 = r_real[0]
    pi_gap0 = pi_now - pi_target

    st.markdown('<div class="section-label">Key results</div>', unsafe_allow_html=True)
    c1, c2, c3, c4 = st.columns(4, gap="medium")
    for col, num, lbl, style in [
        (c1, f"{i0:.2f}%",      "Initial policy rate",      "result-card result-card-accent"),
        (c2, f"{i_end:.2f}%",   f"Rate at month {horizon}", "result-card result-card-teal"),
        (c3, f"{r_real0:.2f}%", "Real rate (month 0)",      "result-card result-card-gold"),
        (c4, f"{pi_gap0:+.2f}%","Inflation gap",             "result-card"),
    ]:
        with col:
            st.markdown(f'<div class="{style}"><div class="result-num">{num}</div><div class="result-label">{lbl}</div></div>', unsafe_allow_html=True)

    # ── Interpretation ──
    stance = "restrictive" if i0 > (r_star + pi_target) else "accommodative"
    real_stance = "positive" if r_real0 > 0 else "negative"
    st.markdown(f"""
    <div class="interp-box" style="margin-top:16px;">
      <div class="interp-title">Reading the model</div>
      <div class="interp-text">
        Under these parameters, the Taylor rule prescribes an initial policy rate of <strong>{i0:.2f}%</strong> — a <strong>{stance}</strong> stance.
        The real rate is <strong>{real_stance}</strong> ({r_real0:.2f}%), which matters for investment and borrowing costs.
        Rate smoothing (ρ = {smoothing}) means the central bank adjusts gradually rather than jumping to the implied rate immediately.
        The phase diagram shows the full reaction surface: darker orange = tighter policy; darker teal = looser.
        The scenario path traces how the economy moves through this surface over {horizon} months.
      </div>
    </div>
    """, unsafe_allow_html=True)

    # ── Save / Compare ──
    st.markdown('<div class="section-label">Save & compare runs</div>', unsafe_allow_html=True)
    sv1, sv2, sv3 = st.columns([1.8, 0.8, 1.4])
    with sv1:
        run_label = st.text_input("Run label",
            value=f"π={pi_now:.1f}%, ỹ={y_gap:.1f}%, φπ={phi_pi}, φy={phi_y}, {preset}",
            key="t1_label")
    with sv2:
        if st.button("Save run", key="t1_save"):
            st.session_state.policy_lab_runs.append({
                "label": run_label,
                "months": months.tolist(),
                "i_smooth": i_smooth.tolist(),
                "pi_path": pi_path.tolist(),
                "i_implied": i_implied.tolist(),
                "r_real": r_real.tolist(),
            })
            st.success("Saved.")
    with sv3:
        if st.button("Clear all saved runs", key="t1_clear"):
            st.session_state.policy_lab_runs = []

    if st.session_state.policy_lab_runs:
        fig_cmp = go.Figure()
        for run in st.session_state.policy_lab_runs[-6:]:
            fig_cmp.add_trace(go.Scatter(
                x=run["months"], y=run["i_smooth"],
                name=run["label"][:40], mode="lines",
                line=dict(width=1.8)
            ))
        fig_cmp.add_trace(go.Scatter(
            x=months.tolist(), y=i_smooth.tolist(),
            name="Current", line=dict(color=COLORS["accent"], width=2.5, dash="dot"),
            mode="lines"
        ))
        fig_cmp.update_layout(**editorial_layout(280, "Policy rate comparison"))
        fig_cmp.update_yaxes(title_text="Rate (%)")
        st.plotly_chart(fig_cmp, use_container_width=True)

    # ── Export ──
    df_export = pd.DataFrame({
        "Month": months,
        "Policy rate (smoothed, %)": i_smooth,
        "Implied rate (no smoothing, %)": i_implied,
        "Real rate (%)": r_real,
        "Inflation path (%)": pi_path,
        "Output gap path (%)": y_path,
        "Base contrib (%)": base_comp,
        "Inflation gap contrib (%)": pi_contrib,
        "Output gap contrib (%)": y_contrib,
    })
    st.download_button("Export scenario CSV",
        data=df_export.to_csv(index=False).encode(),
        file_name="policy_lab_monetary.csv", mime="text/csv", key="t1_dl")


# =====================================================================
# ── MODULE 2: FISCAL POLICY & MULTIPLIERS ────────────────────────────
# =====================================================================
with tab2:
    st.markdown('<div class="section-label">Fiscal Policy — Keynesian Multiplier Model</div>', unsafe_allow_html=True)

    col_fp, col_fc = st.columns([1, 2.2], gap="large")

    with col_fp:
        st.markdown("""
        <div class="param-panel">
          <div class="param-title">Parameters</div>
          <div class="param-desc">Model a government spending or tax shock and trace its path through the economy via the multiplier mechanism.</div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown('<div class="param-group">Economy parameters</div>', unsafe_allow_html=True)
        mpc       = st.slider("Marginal propensity to consume (MPC)", 0.3, 0.95, 0.75, 0.01, key="fp_mpc")
        tax_rate  = st.slider("Tax rate (t)", 0.0, 0.5, 0.2, 0.01, key="fp_tax")
        import_rate = st.slider("Import propensity (m)", 0.0, 0.5, 0.15, 0.01, key="fp_import")
        crowding  = st.slider("Crowding-out factor", 0.0, 1.0, 0.2, 0.05, key="fp_crowding",
                              help="0 = no crowding out; 1 = full crowding out (fiscal policy ineffective)")

        st.markdown('<div class="param-group">Shock</div>', unsafe_allow_html=True)
        shock_type = st.radio("Shock type", ["Government spending (ΔG)", "Tax cut (−ΔT)"], key="fp_shock")
        shock_size = st.slider("Shock size (% of GDP)", 0.5, 10.0, 2.0, 0.5, key="fp_size")
        shock_persistence = st.selectbox("Persistence", ["Temporary (1 period)", "Permanent", "Phased out over 5 periods"], key="fp_persist")
        fp_horizon = st.slider("Horizon (periods)", 4, 20, 10, 1, key="fp_horizon")
        economy_state = st.selectbox("Economy state", ["Below potential (slack)", "At potential", "Above potential"], key="fp_state")

    # ── Multiplier calculation ──
    state_adj = {"Below potential (slack)": 1.0, "At potential": 0.7, "Above potential": 0.4}[economy_state]
    base_mult = 1 / (1 - mpc*(1-tax_rate) + import_rate)
    effective_mult = base_mult * (1 - crowding) * state_adj
    tax_mult = -mpc / (1 - mpc*(1-tax_rate) + import_rate) * (1 - crowding) * state_adj

    if shock_type == "Tax cut (−ΔT)":
        impact_mult = tax_mult
    else:
        impact_mult = effective_mult

    # Build GDP path
    periods = np.arange(fp_horizon + 1)
    if shock_persistence == "Temporary (1 period)":
        shock_path = np.zeros(fp_horizon+1); shock_path[0] = shock_size
    elif shock_persistence == "Permanent":
        shock_path = np.full(fp_horizon+1, shock_size)
    else:
        shock_path = np.array([shock_size * max(0, 1 - t/5) for t in range(fp_horizon+1)])

    # Simplified dynamic response: each period's output gap responds to cumulative shock
    # with a lag structure (first-order adjustment)
    gdp_impact = np.zeros(fp_horizon+1)
    gdp_impact[0] = shock_path[0] * impact_mult
    for t in range(1, fp_horizon+1):
        adj = 0.6  # partial adjustment speed
        gdp_impact[t] = adj * shock_path[t] * impact_mult + (1-adj) * gdp_impact[t-1] * 0.85

    # Crowded-out investment (negative counterpart)
    invest_crowd = -shock_path * crowding * 0.5

    # Debt accumulation (simplified)
    deficit_path = shock_path - gdp_impact * tax_rate  # spending minus revenue gain
    debt_cumul   = np.cumsum(deficit_path)

    # Consumption path
    cons_path = gdp_impact * mpc * (1 - tax_rate)

    with col_fc:
        # ── Chart 1: Multiplied GDP impact stacked ──
        fig_fp1 = go.Figure()
        fig_fp1.add_trace(go.Bar(x=periods, y=gdp_impact,
            name="GDP impact (ΔY)", marker_color=COLORS["teal"], opacity=0.85))
        fig_fp1.add_trace(go.Bar(x=periods, y=invest_crowd,
            name="Investment crowded out", marker_color=COLORS["accent"], opacity=0.7))
        fig_fp1.add_trace(go.Scatter(x=periods, y=shock_path,
            name="Initial shock (ΔG or −ΔT)", line=dict(color=COLORS["ink"], dash="dot", width=2),
            mode="lines"))
        fig_fp1.update_layout(**editorial_layout(320, "GDP impact & crowding-out"))
        fig_fp1.update_layout(barmode="relative")
        fig_fp1.update_yaxes(title_text="% of GDP")
        fig_fp1.update_xaxes(title_text="Periods")
        st.plotly_chart(fig_fp1, use_container_width=True)

        # ── Chart 2: Leakage waterfall (single period) ──
        total = shock_size * base_mult
        saving_leak = shock_size * base_mult * (1 - mpc)
        tax_leak    = shock_size * base_mult * mpc * tax_rate
        import_leak = shock_size * base_mult * mpc * (1 - tax_rate) * import_rate
        crowd_drag  = shock_size * crowding * 0.5
        net_gdp     = total - saving_leak - tax_leak - import_leak - crowd_drag

        fig_fp2 = make_subplots(rows=1, cols=2,
            subplot_titles=["Multiplier leakages (period 0)", "Debt accumulation path"],
            horizontal_spacing=0.10)

        leak_labels = ["Gross impact", "Saving leak", "Tax leak", "Import leak", "Crowding drag", "Net ΔY"]
        leak_vals   = [total, -saving_leak, -tax_leak, -import_leak, -crowd_drag, net_gdp]
        leak_colors = [COLORS["teal"], COLORS["accent"], COLORS["accent"],
                       COLORS["accent"], COLORS["gold"], COLORS["ink"]]
        fig_fp2.add_trace(go.Bar(
            x=leak_labels, y=leak_vals, marker_color=leak_colors,
            text=[f"{v:.2f}" for v in leak_vals], textposition="outside",
            textfont=dict(size=10), showlegend=False
        ), row=1, col=1)

        fig_fp2.add_trace(go.Scatter(
            x=periods, y=debt_cumul,
            line=dict(color=COLORS["accent"], width=2.5), mode="lines+markers",
            marker=dict(size=5), name="Cumulative deficit", showlegend=False
        ), row=1, col=2)
        fig_fp2.add_trace(go.Scatter(
            x=periods, y=deficit_path,
            line=dict(color=COLORS["teal"], width=1.5, dash="dot"),
            mode="lines", name="Period deficit", showlegend=False
        ), row=1, col=2)

        fig_fp2.update_layout(
            height=300, paper_bgcolor=COLORS["cream"], plot_bgcolor=COLORS["cream"],
            font=dict(family="DM Sans", color=COLORS["ink"], size=11),
            margin=dict(l=12, r=12, t=40, b=12),
        )
        for ax in ["xaxis", "xaxis2", "yaxis", "yaxis2"]:
            fig_fp2.update_layout(**{ax: dict(gridcolor=COLORS["rule"], linecolor=COLORS["rule"])})
        st.plotly_chart(fig_fp2, use_container_width=True)

        # ── Chart 3: Multiplier sensitivity surface ──
        st.markdown('<div class="section-label-inline">Multiplier sensitivity — MPC × import propensity</div>', unsafe_allow_html=True)

        mpc_grid = np.linspace(0.3, 0.95, 40)
        m_grid   = np.linspace(0.0, 0.4, 40)
        MG, ImG  = np.meshgrid(mpc_grid, m_grid)
        MULT_SRF = 1 / (1 - MG*(1-tax_rate) + ImG) * (1-crowding) * state_adj

        fig_fp3 = go.Figure(go.Surface(
            x=mpc_grid, y=m_grid, z=MULT_SRF,
            colorscale=[[0, COLORS["teal"]], [0.5, COLORS["cream"]], [1, COLORS["accent"]]],
            showscale=True,
            contours=dict(z=dict(show=True, usecolormap=True, highlightcolor=COLORS["gold"], project_z=True)),
        ))
        # Current position marker
        fig_fp3.add_trace(go.Scatter3d(
            x=[mpc], y=[import_rate], z=[effective_mult],
            mode="markers",
            marker=dict(size=8, color=COLORS["ink"], symbol="diamond"),
            name="Current params",
        ))
        fig_fp3.update_layout(
            height=380,
            paper_bgcolor=COLORS["cream"],
            font=dict(family="DM Sans", color=COLORS["ink"], size=11),
            margin=dict(l=0, r=0, t=30, b=0),
            scene=dict(
                xaxis_title="MPC",
                yaxis_title="Import propensity",
                zaxis_title="Multiplier",
                xaxis=dict(gridcolor=COLORS["rule"]),
                yaxis=dict(gridcolor=COLORS["rule"]),
                zaxis=dict(gridcolor=COLORS["rule"]),
                bgcolor=COLORS["cream"],
            ),
            title=dict(text="Multiplier surface", font=dict(family="DM Serif Display", size=15)),
        )
        st.plotly_chart(fig_fp3, use_container_width=True)

    # ── Result cards ──
    st.markdown('<div class="section-label">Key results</div>', unsafe_allow_html=True)
    fc1, fc2, fc3, fc4 = st.columns(4)
    for col, num, lbl, style in [
        (fc1, f"{effective_mult:.2f}×", "Effective multiplier",  "result-card result-card-teal"),
        (fc2, f"{gdp_impact[0]:.2f}%",  "Impact on GDP (t=0)",   "result-card result-card-accent"),
        (fc3, f"{debt_cumul[-1]:.2f}%", "Cumulative deficit",    "result-card result-card-gold"),
        (fc4, f"{tax_mult:.2f}×",       "Tax cut multiplier",    "result-card"),
    ]:
        with col:
            st.markdown(f'<div class="{style}"><div class="result-num">{num}</div><div class="result-label">{lbl}</div></div>', unsafe_allow_html=True)

    st.markdown(f"""
    <div class="interp-box interp-box-teal" style="margin-top:16px;">
      <div class="interp-title">Reading the model</div>
      <div class="interp-text">
        With MPC = {mpc}, tax rate = {tax_rate}, and import propensity = {import_rate}, the base multiplier is <strong>{base_mult:.2f}</strong>.
        After crowding-out ({int(crowding*100)}%) and the state-of-the-economy adjustment
        ({'full slack — multiplier at maximum' if economy_state=='Below potential (slack)' else 'partial slack' if economy_state=='At potential' else 'near capacity — multiplier suppressed'}),
        the effective multiplier is <strong>{effective_mult:.2f}</strong>.
        The 3D surface shows how the multiplier varies across the MPC–import space under current crowding-out assumptions.
        The diamond marks your current parameter combination.
      </div>
    </div>
    """, unsafe_allow_html=True)

    df_fp_export = pd.DataFrame({
        "Period": periods, "Shock (% GDP)": shock_path,
        "GDP impact (% GDP)": gdp_impact, "Crowded-out investment": invest_crowd,
        "Period deficit": deficit_path, "Cumulative deficit": debt_cumul,
    })
    st.download_button("Export fiscal scenario CSV",
        data=df_fp_export.to_csv(index=False).encode(),
        file_name="policy_lab_fiscal.csv", mime="text/csv", key="fp_dl")


# =====================================================================
# ── MODULE 3: DEBT DYNAMICS ──────────────────────────────────────────
# =====================================================================
with tab3:
    st.markdown('<div class="section-label">Debt Sustainability — r−g Dynamics</div>', unsafe_allow_html=True)

    col_dp, col_dc = st.columns([1, 2.2], gap="large")

    with col_dp:
        st.markdown("""
        <div class="param-panel">
          <div class="param-title">Parameters</div>
          <div class="param-desc">Explore debt sustainability through the fundamental identity: Δd = (r−g)d − pb.</div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown('<div class="param-group">Initial conditions</div>', unsafe_allow_html=True)
        d0      = st.slider("Initial debt (% of GDP)", 0.0, 200.0, 80.0, 1.0, key="dd_d0")
        pb0     = st.slider("Primary balance (% of GDP, + = surplus)", -8.0, 8.0, 1.0, 0.1, key="dd_pb0")
        r_nom   = st.slider("Nominal interest rate r (%)", 0.0, 10.0, 4.0, 0.1, key="dd_r")
        g_nom   = st.slider("Nominal GDP growth g (%)", -3.0, 8.0, 3.0, 0.1, key="dd_g")
        d_horiz = st.slider("Horizon (years)", 5, 30, 15, 1, key="dd_horiz")

        st.markdown('<div class="param-group">Shocks</div>', unsafe_allow_html=True)
        shock_yr    = st.slider("Shock in year", 1, d_horiz, 3, 1, key="dd_sy")
        shock_pb    = st.slider("Primary balance shock (pp)", -5.0, 5.0, -2.0, 0.1, key="dd_spb",
                                help="Negative = fiscal expansion / crisis")
        shock_r     = st.slider("Interest rate shock (pp)", -2.0, 4.0, 1.0, 0.1, key="dd_sr")
        shock_g     = st.slider("Growth shock (pp)", -5.0, 3.0, -1.5, 0.1, key="dd_sg")
        shock_dur   = st.slider("Shock duration (years)", 1, 10, 3, 1, key="dd_sdur")

    # ── Simulate baseline + shocked paths ──
    years_d = np.arange(d_horiz+1)

    def sim_debt(d0, pb, r, g, H, shock_yr=None, shock_pb=0, shock_r=0, shock_g=0, shock_dur=0):
        d = np.zeros(H+1); d[0] = d0
        rg_arr = np.zeros(H+1); pb_arr = np.zeros(H+1)
        for t in range(1, H+1):
            in_shock = shock_yr is not None and shock_yr <= t < shock_yr + shock_dur
            r_t = r + (shock_r if in_shock else 0)
            g_t = g + (shock_g if in_shock else 0)
            pb_t = pb + (shock_pb if in_shock else 0)
            rg = (r_t - g_t) / 100
            d[t] = (1 + rg) * d[t-1] - pb_t
            rg_arr[t] = rg * 100
            pb_arr[t] = pb_t
        return d, rg_arr, pb_arr

    d_base, rg_base, pb_base = sim_debt(d0, pb0, r_nom, g_nom, d_horiz)
    d_shock, rg_shock, pb_shock = sim_debt(d0, pb0, r_nom, g_nom, d_horiz,
        shock_yr, shock_pb, shock_r, shock_g, shock_dur)

    # Stabilizing primary balance
    stab_pb = (r_nom - g_nom) / 100 * d0

    with col_dc:
        # ── Chart 1: Debt trajectories ──
        fig_dd1 = go.Figure()
        fig_dd1.add_trace(go.Scatter(
            x=years_d, y=d_base, name="Baseline",
            line=dict(color=COLORS["teal"], width=2.5), mode="lines"
        ))
        fig_dd1.add_trace(go.Scatter(
            x=years_d, y=d_shock, name="With shock",
            line=dict(color=COLORS["accent"], width=2.5), mode="lines"
        ))
        # Uncertainty fan (simple ±1σ band around baseline)
        g_std = 1.5  # assumed growth std dev
        upper = d_base + np.arange(d_horiz+1) * g_std * 0.6
        lower = d_base - np.arange(d_horiz+1) * g_std * 0.6
        fig_dd1.add_trace(go.Scatter(
            x=np.concatenate([years_d, years_d[::-1]]),
            y=np.concatenate([upper, lower[::-1]]),
            fill="toself", fillcolor=f"rgba(42,122,111,0.12)",
            line=dict(width=0), name="Uncertainty band (±1σ growth)", showlegend=True
        ))
        fig_dd1.add_hline(y=d0, line_color=COLORS["rule"], line_width=1, line_dash="dot",
                          annotation_text="Initial debt", annotation_font_color=COLORS["muted"])
        fig_dd1.add_vline(x=shock_yr, line_color=COLORS["gold"], line_width=1.5, line_dash="dash",
                          annotation_text="Shock", annotation_font_color=COLORS["gold"])
        fig_dd1.update_layout(**editorial_layout(340, "Debt-to-GDP trajectory"))
        fig_dd1.update_yaxes(title_text="Debt (% of GDP)")
        fig_dd1.update_xaxes(title_text="Years")
        st.plotly_chart(fig_dd1, use_container_width=True)

        # ── Chart 2: r−g and primary balance ──
        fig_dd2 = make_subplots(rows=1, cols=2,
            subplot_titles=["r − g differential", "Primary balance required vs actual"],
            horizontal_spacing=0.10)

        fig_dd2.add_trace(go.Scatter(
            x=years_d, y=rg_base, name="Baseline r−g",
            line=dict(color=COLORS["teal"], width=2), mode="lines"
        ), row=1, col=1)
        fig_dd2.add_trace(go.Scatter(
            x=years_d, y=rg_shock, name="With shock r−g",
            line=dict(color=COLORS["accent"], width=2, dash="dot"), mode="lines"
        ), row=1, col=1)
        fig_dd2.add_hline(y=0, row=1, col=1, line_color=COLORS["ink"], line_width=1)

        # Stabilizing PB needed over time (changes with debt level)
        stab_path = rg_base[1:] * d_base[:-1] / 100
        fig_dd2.add_trace(go.Scatter(
            x=years_d[1:], y=stab_path, name="Stabilizing PB needed",
            line=dict(color=COLORS["gold"], width=2), mode="lines"
        ), row=1, col=2)
        fig_dd2.add_trace(go.Scatter(
            x=years_d[1:], y=pb_base[1:], name="Actual PB (baseline)",
            line=dict(color=COLORS["teal"], width=2, dash="dot"), mode="lines"
        ), row=1, col=2)
        fig_dd2.add_trace(go.Scatter(
            x=years_d[1:], y=pb_shock[1:], name="Actual PB (shocked)",
            line=dict(color=COLORS["accent"], width=1.5, dash="dot"), mode="lines"
        ), row=1, col=2)
        fig_dd2.add_hline(y=0, row=1, col=2, line_color=COLORS["rule"], line_width=1)

        fig_dd2.update_layout(
            height=290, paper_bgcolor=COLORS["cream"], plot_bgcolor=COLORS["cream"],
            font=dict(family="DM Sans", color=COLORS["ink"], size=11),
            margin=dict(l=12, r=12, t=40, b=12),
            legend=dict(orientation="h", y=-0.2, x=0, font=dict(size=9)),
        )
        for ax in ["xaxis","xaxis2","yaxis","yaxis2"]:
            fig_dd2.update_layout(**{ax: dict(gridcolor=COLORS["rule"], linecolor=COLORS["rule"])})
        st.plotly_chart(fig_dd2, use_container_width=True)

        # ── Chart 3: Phase portrait — debt vs primary balance ──
        st.markdown('<div class="section-label-inline">Debt sustainability map</div>', unsafe_allow_html=True)

        pb_range = np.linspace(-6, 8, 60)
        d_range  = np.linspace(0, 200, 60)
        PB_G, D_G = np.meshgrid(pb_range, d_range)
        rg_val = (r_nom - g_nom) / 100
        Delta_D = rg_val * D_G - PB_G  # >0 = debt rising

        fig_dd3 = go.Figure()
        fig_dd3.add_trace(go.Contour(
            x=pb_range, y=d_range, z=Delta_D,
            colorscale=[[0, COLORS["teal"]], [0.5, COLORS["cream"]], [1, COLORS["accent"]]],
            contours=dict(start=-8, end=8, size=1,
                          showlabels=True, labelfont=dict(size=9)),
            colorbar=dict(title="Δdebt/yr", tickfont=dict(size=9)),
            line=dict(width=0.5),
        ))
        # Zero contour (stability boundary)
        fig_dd3.add_shape(type="line",
            x0=pb_range[0], x1=pb_range[-1],
            y0=pb_range[0]/rg_val if rg_val != 0 else 0,
            y1=pb_range[-1]/rg_val if rg_val != 0 else 200,
            line=dict(color=COLORS["ink"], width=2, dash="dot"))
        # Trace path
        fig_dd3.add_trace(go.Scatter(
            x=pb_base[1:], y=d_base[:-1],
            mode="lines+markers", name="Baseline path",
            line=dict(color=COLORS["ink"], width=2),
            marker=dict(size=5, color=COLORS["teal"])
        ))
        fig_dd3.add_trace(go.Scatter(
            x=pb_shock[1:], y=d_shock[:-1],
            mode="lines+markers", name="Shocked path",
            line=dict(color=COLORS["accent"], width=2, dash="dot"),
            marker=dict(size=5, color=COLORS["accent"])
        ))
        fig_dd3.update_layout(**editorial_layout(360, "Debt sustainability map (teal = debt falling, orange = rising)"))
        fig_dd3.update_xaxes(title_text="Primary balance (% GDP)")
        fig_dd3.update_yaxes(title_text="Debt (% GDP)")
        st.plotly_chart(fig_dd3, use_container_width=True)

    # ── Results ──
    rg_val_pct = r_nom - g_nom
    st.markdown('<div class="section-label">Key results</div>', unsafe_allow_html=True)
    dc1, dc2, dc3, dc4 = st.columns(4)
    for col, num, lbl, style in [
        (dc1, f"{rg_val_pct:+.1f}pp",  "r − g differential",            "result-card result-card-accent" if rg_val_pct > 0 else "result-card result-card-teal"),
        (dc2, f"{stab_pb:.1f}%",        "Stabilizing primary balance",   "result-card result-card-gold"),
        (dc3, f"{d_base[-1]:.1f}%",     f"Baseline debt (yr {d_horiz})", "result-card result-card-teal"),
        (dc4, f"{d_shock[-1]:.1f}%",    f"Shocked debt (yr {d_horiz})",  "result-card result-card-accent"),
    ]:
        with col:
            st.markdown(f'<div class="{style}"><div class="result-num">{num}</div><div class="result-label">{lbl}</div></div>', unsafe_allow_html=True)

    sustainability = "sustainable" if pb0 >= stab_pb and rg_val_pct <= 0 else \
                     "on a potentially explosive path" if rg_val_pct > 2 and pb0 < stab_pb else \
                     "requires a primary surplus to stabilize"
    st.markdown(f"""
    <div class="interp-box interp-box-gold" style="margin-top:16px;">
      <div class="interp-title">Reading the model</div>
      <div class="interp-text">
        With r − g = <strong>{rg_val_pct:+.1f}pp</strong>, the debt trajectory is <strong>{sustainability}</strong>.
        To stabilize debt at {d0:.0f}% of GDP, the government needs a primary balance of at least <strong>{stab_pb:.2f}% of GDP</strong>.
        The sustainability map shows the full debt-primary-balance space: the dashed line is the zero-Δ boundary —
        above it debt rises, below it debt falls. The shock scenario shifts the path
        by {shock_r:+.1f}pp on r, {shock_g:+.1f}pp on g, and {shock_pb:+.1f}pp on the primary balance for {shock_dur} year(s) from year {shock_yr}.
      </div>
    </div>
    """, unsafe_allow_html=True)

    df_dd_export = pd.DataFrame({
        "Year": years_d, "Debt baseline (%)": d_base, "Debt shocked (%)": d_shock,
        "r−g baseline (pp)": rg_base, "r−g shocked (pp)": rg_shock,
        "Primary balance baseline (%)": pb_base, "Primary balance shocked (%)": pb_shock,
    })
    st.download_button("Export debt dynamics CSV",
        data=df_dd_export.to_csv(index=False).encode(),
        file_name="policy_lab_debt.csv", mime="text/csv", key="dd_dl")


# =====================================================================
# ── MODULE 4: TRADE & TARIFF INCIDENCE ───────────────────────────────
# =====================================================================
with tab4:
    st.markdown('<div class="section-label">Trade Policy — Tariff Incidence & Welfare</div>', unsafe_allow_html=True)

    col_tp, col_tc = st.columns([1, 2.2], gap="large")

    with col_tp:
        st.markdown("""
        <div class="param-panel">
          <div class="param-title">Parameters</div>
          <div class="param-desc">Analyse how a tariff is split between consumers and foreign exporters, and trace welfare effects.</div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown('<div class="param-group">Market structure</div>', unsafe_allow_html=True)
        p_world   = st.slider("World price (Pw)", 50, 200, 100, 5, key="tr_pw")
        tariff_pct = st.slider("Ad valorem tariff (%)", 0, 80, 20, 1, key="tr_tariff")
        ed_demand = st.slider("Price elasticity of demand (|εD|)", 0.1, 5.0, 1.5, 0.1, key="tr_ed")
        es_supply = st.slider("Price elasticity of domestic supply (εS)", 0.1, 5.0, 1.0, 0.1, key="tr_es")
        es_foreign = st.slider("Foreign export supply elasticity (εX*)", 0.1, 10.0, 3.0, 0.1, key="tr_ex",
                               help="High = small country; low = large country with terms of trade power")

        st.markdown('<div class="param-group">Baseline quantities</div>', unsafe_allow_html=True)
        Q_d0 = st.slider("Domestic demand at Pw (units)", 50, 500, 200, 10, key="tr_qd")
        Q_s0 = st.slider("Domestic supply at Pw (units)", 10, 300, 80, 10, key="tr_qs")
        Q_m0 = Q_d0 - Q_s0  # imports

        st.markdown('<div class="param-group">Scenario</div>', unsafe_allow_html=True)
        country_size = st.radio("Country type", ["Small (price-taker)", "Large (ToT power)"], key="tr_size")
        show_dw = st.checkbox("Show deadweight loss triangles", value=True, key="tr_dw")

    # ── Incidence calculation ──
    # Tariff per unit
    t_abs = p_world * tariff_pct / 100

    # Incidence split (pass-through to domestic consumers vs borne by exporters)
    # Small country: full pass-through
    if country_size == "Small (price-taker)":
        pass_through = 1.0
    else:
        # Large country: partial pass-through based on elasticities
        pass_through = es_foreign / (es_foreign + ed_demand)

    consumer_burden = t_abs * pass_through
    exporter_burden = t_abs * (1 - pass_through)

    p_domestic = p_world + consumer_burden   # price domestic consumers pay
    p_foreign  = p_world - exporter_burden   # price foreign exporters receive

    # New quantities
    Q_d1 = Q_d0 * (1 - ed_demand * consumer_burden / p_world)
    Q_s1 = Q_s0 * (1 + es_supply * consumer_burden / p_world)
    Q_m1 = max(0, Q_d1 - Q_s1)

    # Welfare triangles
    CS_loss   = -0.5 * (Q_d0 + Q_d1) * consumer_burden  # consumer surplus loss
    PS_gain   =  0.5 * (Q_s0 + Q_s1) * consumer_burden  # producer surplus gain
    TR_gain   = Q_m1 * t_abs                             # tariff revenue
    ToT_gain  = Q_m1 * exporter_burden                   # terms of trade gain (large country only)
    DWL       = CS_loss - PS_gain - TR_gain + ToT_gain   # net welfare (should be ≤ 0 for small)
    net_welfare = CS_loss + PS_gain + TR_gain + ToT_gain  # keeping signs

    with col_tc:
        # ── Chart 1: Supply-demand diagram ──
        p_axis  = np.linspace(max(1, p_world*0.3), p_world*1.8, 200)
        # Demand: Q = Q_d0 * (Pw/P)^ed  (constant elasticity)
        Qd_curve = Q_d0 * (p_world / p_axis) ** ed_demand
        Qs_curve = Q_s0 * (p_axis / p_world) ** es_supply

        fig_tr1 = go.Figure()
        fig_tr1.add_trace(go.Scatter(x=Qd_curve, y=p_axis, name="Demand",
            line=dict(color=COLORS["accent"], width=2.5), mode="lines"))
        fig_tr1.add_trace(go.Scatter(x=Qs_curve, y=p_axis, name="Domestic Supply",
            line=dict(color=COLORS["teal"], width=2.5), mode="lines"))

        # World price
        fig_tr1.add_hline(y=p_world, line_color=COLORS["ink"],
            line_width=1.5, line_dash="dot",
            annotation_text=f"Pw = {p_world}", annotation_font_color=COLORS["ink"])
        # Domestic price after tariff
        fig_tr1.add_hline(y=p_domestic, line_color=COLORS["accent"],
            line_width=2, line_dash="dash",
            annotation_text=f"Pd = {p_domestic:.1f}",
            annotation_font_color=COLORS["accent"])
        # Foreign price (if large country)
        if country_size == "Large (ToT power)":
            fig_tr1.add_hline(y=p_foreign, line_color=COLORS["gold"],
                line_width=1.5, line_dash="dot",
                annotation_text=f"Pf = {p_foreign:.1f}",
                annotation_font_color=COLORS["gold"])

        if show_dw:
            # Consumer surplus loss shading (between Pw and Pd, from 0 to Q_d1)
            fig_tr1.add_shape(type="rect",
                x0=0, x1=Q_d1, y0=p_world, y1=p_domestic,
                fillcolor=f"rgba(201,98,42,0.15)", line_width=0)
            # Producer surplus gain
            fig_tr1.add_shape(type="rect",
                x0=0, x1=Q_s1, y0=p_world, y1=p_domestic,
                fillcolor=f"rgba(42,122,111,0.2)", line_width=0)

        fig_tr1.update_layout(**editorial_layout(380, "Tariff incidence — supply & demand"))
        fig_tr1.update_xaxes(title_text="Quantity", range=[0, Q_d0*1.2])
        fig_tr1.update_yaxes(title_text="Price", range=[p_world*0.5, p_world*1.5])
        st.plotly_chart(fig_tr1, use_container_width=True)

        # ── Chart 2: Welfare decomposition + incidence ──
        fig_tr2 = make_subplots(rows=1, cols=2,
            subplot_titles=["Welfare decomposition", "Tariff incidence split"],
            horizontal_spacing=0.12)

        welfare_labels = ["CS change", "PS change", "Tariff revenue", "ToT gain", "Net welfare"]
        welfare_vals   = [CS_loss, PS_gain, TR_gain, ToT_gain, net_welfare]
        w_colors = [COLORS["accent"] if v < 0 else COLORS["teal"] for v in welfare_vals[:-1]] + [COLORS["gold"]]
        fig_tr2.add_trace(go.Bar(
            x=welfare_labels, y=welfare_vals, marker_color=w_colors,
            text=[f"{v:.1f}" for v in welfare_vals], textposition="outside",
            textfont=dict(size=10), showlegend=False
        ), row=1, col=1)

        # Pie: incidence split
        fig_tr2.add_trace(go.Pie(
            labels=["Consumer burden", "Exporter burden"],
            values=[consumer_burden, max(0.001, exporter_burden)],
            marker=dict(colors=[COLORS["accent"], COLORS["teal"]]),
            textinfo="label+percent",
            textfont=dict(size=11),
            hole=0.4,
            showlegend=False,
        ), row=1, col=2)

        fig_tr2.update_layout(
            height=300, paper_bgcolor=COLORS["cream"], plot_bgcolor=COLORS["cream"],
            font=dict(family="DM Sans", color=COLORS["ink"], size=11),
            margin=dict(l=12, r=12, t=40, b=12),
        )
        for ax in ["xaxis", "yaxis"]:
            fig_tr2.update_layout(**{ax: dict(gridcolor=COLORS["rule"], linecolor=COLORS["rule"])})
        st.plotly_chart(fig_tr2, use_container_width=True)

        # ── Chart 3: Sensitivity — tariff rate vs net welfare ──
        st.markdown('<div class="section-label-inline">Welfare sensitivity — tariff rate × foreign elasticity</div>', unsafe_allow_html=True)

        tariff_range = np.linspace(0, 80, 50)
        fes_range    = np.linspace(0.5, 10, 40)
        TR_grid, FES_grid = np.meshgrid(tariff_range, fes_range)

        def net_welfare_surface(t_pct, fes):
            t = p_world * t_pct / 100
            pt = fes / (fes + ed_demand) if country_size == "Large (ToT power)" else 1.0
            pd_l = p_world + t * pt
            cb   = t * pt
            pf_l = p_world - t * (1-pt)
            qd1_l = Q_d0 * (p_world / pd_l) ** ed_demand if pd_l > 0 else Q_d0
            qs1_l = Q_s0 * (pd_l / p_world) ** es_supply
            qm1_l = max(0, qd1_l - qs1_l)
            cs_l = -0.5*(Q_d0+qd1_l)*cb
            ps_l =  0.5*(Q_s0+qs1_l)*cb
            tr_l = qm1_l * t
            tot_l = qm1_l * t*(1-pt)
            return cs_l + ps_l + tr_l + tot_l

        NW_surf = np.vectorize(net_welfare_surface)(TR_grid, FES_grid)

        fig_tr3 = go.Figure(go.Contour(
            x=tariff_range, y=fes_range, z=NW_surf,
            colorscale=[[0, COLORS["accent"]], [0.5, COLORS["cream"]], [1, COLORS["teal"]]],
            contours=dict(showlabels=True, labelfont=dict(size=9)),
            colorbar=dict(title="Net welfare", tickfont=dict(size=9)),
            line=dict(width=0.5),
        ))
        fig_tr3.add_trace(go.Scatter(
            x=[tariff_pct], y=[es_foreign],
            mode="markers", marker=dict(size=12, color=COLORS["ink"], symbol="star"),
            name="Current params"
        ))
        fig_tr3.add_contour(
            x=tariff_range, y=fes_range, z=NW_surf,
            contours=dict(start=0, end=0, coloring="none",
                         showlabels=False, size=1),
            line=dict(color=COLORS["ink"], width=2, dash="dot"),
            showscale=False, name="Zero welfare line"
        )
        fig_tr3.update_layout(**editorial_layout(340, "Net welfare map — tariff rate × foreign supply elasticity"))
        fig_tr3.update_xaxes(title_text="Tariff rate (%)")
        fig_tr3.update_yaxes(title_text="Foreign export supply elasticity")
        st.plotly_chart(fig_tr3, use_container_width=True)

    # ── Results ──
    st.markdown('<div class="section-label">Key results</div>', unsafe_allow_html=True)
    tc1, tc2, tc3, tc4 = st.columns(4)
    for col, num, lbl, style in [
        (tc1, f"{pass_through*100:.0f}%",    "Pass-through to consumers", "result-card result-card-accent"),
        (tc2, f"{Q_m1:.0f}",                 "Imports after tariff",      "result-card result-card-teal"),
        (tc3, f"{TR_gain:.1f}",              "Tariff revenue",            "result-card result-card-gold"),
        (tc4, f"{net_welfare:.1f}",          "Net welfare change",        "result-card" if net_welfare >= 0 else "result-card result-card-accent"),
    ]:
        with col:
            st.markdown(f'<div class="{style}"><div class="result-num">{num}</div><div class="result-label">{lbl}</div></div>', unsafe_allow_html=True)

    small_or_large = "small country" if country_size == "Small (price-taker)" else "large country"
    st.markdown(f"""
    <div class="interp-box" style="margin-top:16px;">
      <div class="interp-title">Reading the model</div>
      <div class="interp-text">
        As a <strong>{small_or_large}</strong>, a {tariff_pct}% tariff raises the domestic price from {p_world} to <strong>{p_domestic:.1f}</strong>.
        <strong>{pass_through*100:.0f}%</strong> of the tariff is borne by domestic consumers;
        <strong>{(1-pass_through)*100:.0f}%</strong> is absorbed by foreign exporters (terms of trade effect).
        Imports fall from {Q_m0:.0f} to {Q_m1:.0f} units. Consumer surplus falls by {abs(CS_loss):.1f},
        producer surplus rises by {PS_gain:.1f}, and tariff revenue is {TR_gain:.1f}.
        Net welfare is <strong>{net_welfare:.1f}</strong> ({('positive — optimal tariff range' if net_welfare > 0 else 'negative — net loss')}).
        The welfare map shows the tariff rate and foreign elasticity combinations that generate positive vs negative net welfare.
      </div>
    </div>
    """, unsafe_allow_html=True)

    df_tr_export = pd.DataFrame({
        "Scenario": ["Baseline", "With tariff"],
        "Price": [p_world, p_domestic],
        "Demand": [Q_d0, Q_d1],
        "Domestic supply": [Q_s0, Q_s1],
        "Imports": [Q_m0, Q_m1],
        "CS change": [0, CS_loss],
        "PS change": [0, PS_gain],
        "Tariff revenue": [0, TR_gain],
        "Net welfare": [0, net_welfare],
    })
    st.download_button("Export trade scenario CSV",
        data=df_tr_export.to_csv(index=False).encode(),
        file_name="policy_lab_trade.csv", mime="text/csv", key="tr_dl")


# =====================================================================
# FOOTER
# =====================================================================
st.markdown("""
<div class="site-footer">
  Chaouat Economics Lab — Policy Lab · All models are teaching tools. No real forecasting intended.<br/>
  <span style="font-size:11px;">© Chaouat Economics Lab · Educational use only</span>
</div>
""", unsafe_allow_html=True)
