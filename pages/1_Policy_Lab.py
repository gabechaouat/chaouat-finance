import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

st.set_page_config(page_title="Policy Lab — Chaouat Economics Lab", page_icon="🧪", layout="wide")

# =====================================================================
# STYLE
# =====================================================================
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Serif+Display:ital@0;1&family=DM+Sans:wght@300;400;500;600&display=swap');

:root {
  --ink:          #1a1814;
  --ink-muted:    #6b6760;
  --ink-faint:    #b0ada8;
  --cream:        #faf8f4;
  --warm:         #f2ede4;
  --rule:         #e0dbd2;
  --terra:        #c9622a;
  --terra-mid:    #a84e20;
  --terra-light:  #f7ece3;
  --sienna:       #8b3a1a;
  --sand:         #c4a882;
  --sand-dark:    #9e8060;
  --stone:        #7a6f62;
  --stone-light:  #ece8e2;
}

html, body, * { font-family: 'DM Sans', sans-serif !important; }
.block-container { padding-top: 0 !important; padding-bottom: 3rem; max-width: 1240px; }

.pl-masthead {
  border-bottom: 3px double var(--rule);
  padding: 28px 0 18px 0; margin-bottom: 0;
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

.section-label {
  font-size: 10px; letter-spacing: 3px; text-transform: uppercase;
  color: var(--ink-muted); border-top: 1px solid var(--rule);
  padding-top: 10px; margin: 28px 0 16px 0;
}
.section-label-inline {
  font-size: 10px; letter-spacing: 3px; text-transform: uppercase;
  color: var(--ink-muted); margin-bottom: 10px;
}

.param-panel {
  background: var(--warm); border: 1px solid var(--rule);
  border-radius: 4px; padding: 20px 18px;
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
  color: var(--ink-faint); margin: 14px 0 6px 0;
  border-top: 1px solid var(--rule); padding-top: 10px;
}

.result-card {
  border-radius: 4px; padding: 16px 18px; text-align: center; color: #fff;
}
.rc-terra  { background: var(--terra); }
.rc-sienna { background: var(--sienna); }
.rc-sand   { background: var(--sand); color: var(--ink); }
.rc-stone  { background: var(--stone); }
.rc-ink    { background: var(--ink); }
.result-num {
  font-family: 'DM Serif Display', serif !important;
  font-size: 28px; line-height: 1; margin: 0;
}
.result-label {
  font-size: 11px; opacity: 0.75; text-transform: uppercase;
  letter-spacing: 1.5px; margin-top: 4px;
}

.interp-box {
  border-left: 3px solid var(--terra);
  background: var(--terra-light);
  padding: 14px 18px; border-radius: 0 4px 4px 0; margin-top: 16px;
}
.interp-box-sand  { border-left-color: var(--sand-dark); background: #f5eedd; }
.interp-box-stone { border-left-color: var(--stone);     background: var(--stone-light); }
.interp-title { font-weight: 500; font-size: 13px; color: var(--ink); margin: 0 0 4px 0; }
.interp-text  { font-size: 13px; color: var(--ink-muted); line-height: 1.6; margin: 0; }

div.stButton > button {
  background: var(--ink) !important; color: #fff !important;
  border: none !important; border-radius: 3px !important;
  font-size: 11px !important; letter-spacing: 1.5px !important;
  text-transform: uppercase !important; font-weight: 500 !important;
  padding: 9px 16px !important; width: 100%;
}
div.stButton > button:hover { background: var(--terra) !important; }

div.stDownloadButton > button {
  background: transparent !important; color: var(--ink) !important;
  border: 1px solid var(--rule) !important; border-radius: 3px !important;
  font-size: 11px !important; letter-spacing: 1.5px !important;
  text-transform: uppercase !important;
}
div.stDownloadButton > button:hover {
  border-color: var(--terra) !important; color: var(--terra) !important;
}

[data-baseweb="tab-list"] { border-bottom: 2px solid var(--rule) !important; gap: 0 !important; }
[data-baseweb="tab"] {
  font-size: 11px !important; letter-spacing: 2px !important;
  text-transform: uppercase !important; color: var(--ink-muted) !important;
  padding: 10px 20px !important; background: transparent !important;
  border-bottom: 2px solid transparent !important; margin-bottom: -2px !important;
}
[aria-selected="true"][data-baseweb="tab"] {
  color: var(--ink) !important; border-bottom: 2px solid var(--terra) !important;
  font-weight: 500 !important;
}
[data-baseweb="tab-highlight"] { display: none !important; }
[data-baseweb="tab-border"]    { display: none !important; }

[data-testid="stExpander"] summary {
  font-size: 12px !important; letter-spacing: 1.5px !important;
  text-transform: uppercase !important; color: var(--ink-muted) !important;
}
[data-testid="stSlider"] label { font-size: 12px !important; color: var(--ink-muted) !important; }
[data-testid="stMetricValue"]  { font-size: 22px !important; font-family: 'DM Serif Display', serif !important; }
[data-testid="stMetricLabel"]  { font-size: 11px !important; letter-spacing: 1px !important; text-transform: uppercase !important; }

.site-footer {
  border-top: 3px double var(--rule); padding-top: 16px; margin-top: 40px;
  font-size: 12px; color: var(--ink-muted); text-align: center;
}
</style>
""", unsafe_allow_html=True)

# =====================================================================
# EARTH-TONE PALETTE  (terracotta family only — no blue, no gold, no teal)
# =====================================================================
C = {
    "terra":      "#c9622a",
    "terra_mid":  "#a84e20",
    "sienna":     "#8b3a1a",
    "sand":       "#c4a882",
    "sand_dark":  "#9e8060",
    "stone":      "#7a6f62",
    "ink":        "#1a1814",
    "muted":      "#b0ada8",
    "rule":       "#e0dbd2",
    "cream":      "#faf8f4",
    "warm":       "#f2ede4",
}

PALETTE = [C["terra"], C["sienna"], C["sand_dark"], C["stone"], C["ink"], C["sand"]]

def base_layout(height=400, title=""):
    return dict(
        height=height,
        paper_bgcolor=C["cream"],
        plot_bgcolor=C["cream"],
        font=dict(family="DM Sans", color=C["ink"], size=12),
        title=dict(
            text=title,
            font=dict(family="DM Serif Display", size=15, color=C["ink"]),
            x=0, xanchor="left",
        ),
        margin=dict(l=12, r=12, t=44 if title else 20, b=12),
        legend=dict(orientation="h", y=1.10, x=0, font=dict(size=11)),
        xaxis=dict(gridcolor=C["rule"], linecolor=C["rule"], tickfont=dict(size=11)),
        yaxis=dict(gridcolor=C["rule"], linecolor=C["rule"], tickfont=dict(size=11)),
        colorway=PALETTE,
    )

# =====================================================================
# SESSION STATE
# =====================================================================
for k in ["policy_lab_runs", "debt_runs", "trade_runs"]:
    if k not in st.session_state:
        st.session_state[k] = []

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

tab1, tab2, tab3, tab4 = st.tabs([
    "01 · Monetary Policy",
    "02 · Fiscal & Multipliers",
    "03 · Debt Dynamics",
    "04 · Trade & Incidence",
])

# ─────────────────────────────────────────────────────────────────────
# MODULE 1 — MONETARY POLICY
# ─────────────────────────────────────────────────────────────────────
with tab1:
    st.markdown('<div class="section-label">Taylor Rule Toy Model — Monetary Policy</div>', unsafe_allow_html=True)
    col_p, col_c = st.columns([1, 2.2], gap="large")

    with col_p:
        st.markdown('<div class="param-panel"><div class="param-title">Parameters</div><div class="param-desc">Adjust the rule-based central bank response to inflation and output conditions.</div></div>', unsafe_allow_html=True)
        st.markdown('<div class="param-group">Rule parameters</div>', unsafe_allow_html=True)
        r_star    = st.slider("Neutral real rate r* (%)", 0.0, 4.0,  2.0, 0.25, key="t1_rstar")
        pi_target = st.slider("Inflation target π* (%)", 0.0, 5.0,  2.0, 0.25, key="t1_pit")
        phi_pi    = st.slider("Inflation response φπ",   0.0, 3.0,  1.5, 0.10, key="t1_phipi")
        phi_y     = st.slider("Output gap response φy",  0.0, 2.0,  0.5, 0.10, key="t1_phiy")
        smoothing = st.slider("Rate smoothing ρ",        0.0, 0.9,  0.6, 0.05, key="t1_smooth")
        st.markdown('<div class="param-group">Initial conditions</div>', unsafe_allow_html=True)
        pi_now  = st.slider("Current inflation π (%)", 0.0, 12.0, 3.5, 0.1, key="t1_pi")
        y_gap   = st.slider("Output gap ỹ (%)",       -8.0,  8.0,-0.5, 0.1, key="t1_ygap")
        horizon = st.slider("Horizon (months)",         6,   48,  24,  1,   key="t1_hz")
        st.markdown('<div class="param-group">Shock scenario</div>', unsafe_allow_html=True)
        preset = st.selectbox("Preset", [
            "None","Demand boom","Supply shock (cost push)","Recession","Disinflation + recovery",
        ], key="t1_preset")

    pmap = {
        "None":                     dict(pk="None",  ps=0.0,  pr=0.70, yk="None",  ys=0.0,  yr=0.70),
        "Demand boom":              dict(pk="AR(1)", ps=0.6,  pr=0.85, yk="Step",  ys=1.5,  yr=0.70),
        "Supply shock (cost push)": dict(pk="Step",  ps=1.2,  pr=0.70, yk="Pulse", ys=-1.0, yr=0.70),
        "Recession":                dict(pk="AR(1)", ps=-0.8, pr=0.85, yk="Step",  ys=-2.5, yr=0.70),
        "Disinflation + recovery":  dict(pk="Step",  ps=-0.6, pr=0.70, yk="AR(1)", ys=1.0,  yr=0.80),
    }
    d = pmap[preset]

    def shock_s(H, kind, size, rho):
        s = np.zeros(H+1)
        if kind == "None":  return s
        if kind == "Step":  s[:] = size; return s
        if kind == "Pulse": s[:min(H+1,6)] = size; return s
        s[0] = size
        for t in range(1, H+1): s[t] = rho*s[t-1]
        return s

    H = horizon
    months   = np.arange(H+1)
    pi_path  = np.linspace(pi_now, pi_target, H+1) + shock_s(H, d["pk"], d["ps"], d["pr"])
    y_path   = np.linspace(y_gap,  0.0,       H+1) + shock_s(H, d["yk"], d["ys"], d["yr"])
    i_impl   = r_star + pi_target + phi_pi*(pi_path-pi_target) + phi_y*y_path
    i_smooth = np.zeros(H+1); i_smooth[0] = i_impl[0]
    for t in range(1, H+1):
        i_smooth[t] = smoothing*i_smooth[t-1] + (1-smoothing)*i_impl[t]
    r_real     = i_smooth - pi_path
    base_comp  = r_star + pi_target
    pi_contrib = phi_pi*(pi_path-pi_target)
    y_contrib  = phi_y*y_path

    with col_c:
        # Chart 1 — rate fan
        fig1 = go.Figure()
        fig1.add_trace(go.Scatter(x=months, y=i_impl,   name="Implied (no smoothing)",
            line=dict(color=C["sand_dark"], width=1.5, dash="dot"), mode="lines"))
        fig1.add_trace(go.Scatter(x=months, y=i_smooth, name="Policy rate (smoothed)",
            line=dict(color=C["terra"],     width=2.5), mode="lines"))
        fig1.add_trace(go.Scatter(x=months, y=r_real,   name="Real rate (smoothed − π)",
            line=dict(color=C["sienna"],    width=2.0), mode="lines"))
        fig1.add_hline(y=0,         line_color=C["rule"], line_width=1)
        fig1.add_hline(y=pi_target, line_color=C["sand"], line_width=1, line_dash="dot",
            annotation_text=f"π* = {pi_target}%",
            annotation_font_color=C["sand_dark"], annotation_position="right")
        fig1.update_layout(**base_layout(330, "Policy rate path"))
        fig1.update_yaxes(title_text="Rate (%)"); fig1.update_xaxes(title_text="Months")
        st.plotly_chart(fig1, use_container_width=True)

        # Charts 2a + 2b side by side (standalone figures, no subplots)
        c2a, c2b = st.columns(2, gap="medium")
        with c2a:
            cats  = ["Base (r*+π*)", "Inflation gap", "Output gap", "Implied rate"]
            vals  = [base_comp, pi_contrib[0], y_contrib[0], i_impl[0]]
            bcols = [C["stone"], C["terra"], C["sienna"], C["sand_dark"]]
            fig2a = go.Figure(go.Bar(
                x=cats, y=vals, marker_color=bcols,
                text=[f"{v:.2f}%" for v in vals], textposition="outside",
                textfont=dict(size=10),
            ))
            fig2a.update_layout(**base_layout(270, "Decomposition (month 0)"))
            fig2a.update_yaxes(title_text="Rate (%)")
            st.plotly_chart(fig2a, use_container_width=True)

        with c2b:
            fig2b = go.Figure()
            fig2b.add_trace(go.Scatter(x=months, y=pi_path, name="Inflation (assumed)",
                line=dict(color=C["terra"],    width=2), mode="lines"))
            fig2b.add_trace(go.Scatter(x=months, y=y_path,  name="Output gap (assumed)",
                line=dict(color=C["sienna"],   width=2), mode="lines"))
            fig2b.add_hline(y=0,         line_color=C["rule"], line_width=1)
            fig2b.add_hline(y=pi_target, line_color=C["sand"], line_width=1, line_dash="dot")
            fig2b.update_layout(**base_layout(270, "Assumed economic paths"))
            fig2b.update_yaxes(title_text="(%)")
            st.plotly_chart(fig2b, use_container_width=True)

        # Chart 3 — phase diagram / contour
        st.markdown('<div class="section-label-inline">Phase diagram — central bank reaction surface</div>', unsafe_allow_html=True)
        pi_g   = np.linspace(-2, 6, 60)
        y_g    = np.linspace(-5, 5, 60)
        PG, YG = np.meshgrid(pi_g, y_g)
        IG     = r_star + pi_target + phi_pi*(PG-pi_target) + phi_y*YG
        fig3   = go.Figure()
        fig3.add_trace(go.Contour(
            x=pi_g, y=y_g, z=IG,
            colorscale=[[0, C["cream"]], [0.4, C["sand"]], [1, C["terra_mid"]]],
            contours=dict(showlabels=True, labelfont=dict(size=10, color=C["ink"])),
            colorbar=dict(title="Rate (%)", tickfont=dict(size=10)),
            line=dict(width=0.5),
        ))
        fig3.add_trace(go.Scatter(
            x=pi_path-pi_target, y=y_path, mode="lines+markers",
            line=dict(color=C["ink"], width=2),
            marker=dict(size=5, color=C["sienna"]),
            name="Scenario path",
        ))
        fig3.add_vline(x=0, line_color=C["ink"], line_width=1, line_dash="dot")
        fig3.add_hline(y=0, line_color=C["ink"], line_width=1, line_dash="dot")
        fig3.update_layout(**base_layout(360, "Rate implied by Taylor rule (contour map)"))
        fig3.update_xaxes(title_text="Inflation gap (π − π*)")
        fig3.update_yaxes(title_text="Output gap (ỹ)")
        st.plotly_chart(fig3, use_container_width=True)

    i0     = i_smooth[0]; i_end = i_smooth[-1]
    r0     = r_real[0];   pig   = pi_now - pi_target
    stance = "restrictive" if i0 > base_comp else "accommodative"
    st.markdown('<div class="section-label">Key results</div>', unsafe_allow_html=True)
    for col, num, lbl, cls in zip(
        st.columns(4, gap="medium"),
        [f"{i0:.2f}%", f"{i_end:.2f}%", f"{r0:.2f}%", f"{pig:+.2f}%"],
        ["Initial policy rate", f"Rate at month {horizon}", "Real rate (month 0)", "Inflation gap"],
        ["rc-terra", "rc-sienna", "rc-stone", "rc-sand"],
    ):
        with col:
            st.markdown(f'<div class="result-card {cls}"><div class="result-num">{num}</div><div class="result-label">{lbl}</div></div>', unsafe_allow_html=True)

    st.markdown(f"""
    <div class="interp-box" style="margin-top:16px;">
      <div class="interp-title">Reading the model</div>
      <div class="interp-text">
        The Taylor rule prescribes an initial rate of <strong>{i0:.2f}%</strong> — a <strong>{stance}</strong> stance.
        The real rate is <strong>{r0:.2f}%</strong>. Rate smoothing (ρ = {smoothing}) means the central bank adjusts
        gradually. The contour map shows the full reaction surface: darker terracotta = tighter policy.
        The path traces the economy's movement through this surface over {horizon} months.
      </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="section-label">Save & compare runs</div>', unsafe_allow_html=True)
    sv1, sv2, sv3 = st.columns([1.8, 0.8, 1.4])
    with sv1:
        run_label = st.text_input("Run label",
            value=f"π={pi_now:.1f}%, ỹ={y_gap:.1f}%, φπ={phi_pi}, φy={phi_y}, {preset}",
            key="t1_lbl")
    with sv2:
        if st.button("Save run", key="t1_save"):
            st.session_state.policy_lab_runs.append({
                "label": run_label, "months": months.tolist(),
                "i_smooth": i_smooth.tolist(), "r_real": r_real.tolist(),
            })
            st.success("Saved.")
    with sv3:
        if st.button("Clear saved runs", key="t1_clear"):
            st.session_state.policy_lab_runs = []

    if st.session_state.policy_lab_runs:
        fig_cmp = go.Figure()
        for i, run in enumerate(st.session_state.policy_lab_runs[-6:]):
            fig_cmp.add_trace(go.Scatter(
                x=run["months"], y=run["i_smooth"],
                name=run["label"][:40], mode="lines",
                line=dict(width=1.8, color=PALETTE[i % len(PALETTE)])
            ))
        fig_cmp.add_trace(go.Scatter(
            x=months.tolist(), y=i_smooth.tolist(), name="Current",
            line=dict(color=C["ink"], width=2.5, dash="dot"), mode="lines",
        ))
        fig_cmp.update_layout(**base_layout(260, "Policy rate — saved runs comparison"))
        st.plotly_chart(fig_cmp, use_container_width=True)

    df_t1 = pd.DataFrame({
        "Month": months, "Policy rate (smoothed, %)": i_smooth,
        "Implied rate (no smoothing, %)": i_impl, "Real rate (%)": r_real,
        "Inflation path (%)": pi_path, "Output gap path (%)": y_path,
    })
    st.download_button("Export scenario CSV",
        data=df_t1.to_csv(index=False).encode(),
        file_name="policy_lab_monetary.csv", mime="text/csv", key="t1_dl")


# ─────────────────────────────────────────────────────────────────────
# MODULE 2 — FISCAL POLICY & MULTIPLIERS
# ─────────────────────────────────────────────────────────────────────
with tab2:
    st.markdown('<div class="section-label">Fiscal Policy — Keynesian Multiplier Model</div>', unsafe_allow_html=True)
    col_fp, col_fc = st.columns([1, 2.2], gap="large")

    with col_fp:
        st.markdown('<div class="param-panel"><div class="param-title">Parameters</div><div class="param-desc">Model a government spending or tax shock and trace its path through the economy.</div></div>', unsafe_allow_html=True)
        st.markdown('<div class="param-group">Economy parameters</div>', unsafe_allow_html=True)
        mpc      = st.slider("MPC",                          0.3, 0.95, 0.75, 0.01, key="fp_mpc")
        tax_rate = st.slider("Tax rate (t)",                 0.0, 0.50, 0.20, 0.01, key="fp_tax")
        import_r = st.slider("Import propensity (m)",        0.0, 0.50, 0.15, 0.01, key="fp_imp")
        crowding = st.slider("Crowding-out factor",          0.0, 1.0,  0.20, 0.05, key="fp_crowd")
        st.markdown('<div class="param-group">Shock</div>', unsafe_allow_html=True)
        shock_type = st.radio("Shock type",
            ["Government spending (ΔG)", "Tax cut (−ΔT)"], key="fp_stype")
        shock_size = st.slider("Shock size (% of GDP)",      0.5, 10.0, 2.0, 0.5,  key="fp_sz")
        shock_pers = st.selectbox("Persistence",
            ["Temporary (1 period)", "Permanent", "Phased out over 5 periods"], key="fp_pers")
        fp_horiz   = st.slider("Horizon (periods)",          4,   20,   10,  1,    key="fp_hz")
        econ_state = st.selectbox("Economy state",
            ["Below potential (slack)", "At potential", "Above potential"], key="fp_state")

    state_adj = {"Below potential (slack)": 1.0, "At potential": 0.7, "Above potential": 0.4}[econ_state]
    base_mult = 1 / (1 - mpc*(1-tax_rate) + import_r)
    eff_mult  = base_mult * (1-crowding) * state_adj
    tax_mult  = -mpc / (1 - mpc*(1-tax_rate) + import_r) * (1-crowding) * state_adj
    imp_mult  = tax_mult if shock_type == "Tax cut (−ΔT)" else eff_mult

    periods = np.arange(fp_horiz+1)
    if shock_pers == "Temporary (1 period)":
        sp = np.zeros(fp_horiz+1); sp[0] = shock_size
    elif shock_pers == "Permanent":
        sp = np.full(fp_horiz+1, shock_size)
    else:
        sp = np.array([shock_size*max(0, 1-t/5) for t in range(fp_horiz+1)])

    gdp_imp = np.zeros(fp_horiz+1); gdp_imp[0] = sp[0]*imp_mult
    for t in range(1, fp_horiz+1):
        gdp_imp[t] = 0.6*sp[t]*imp_mult + 0.4*gdp_imp[t-1]*0.85
    inv_crowd  = -sp*crowding*0.5
    deficit_p  = sp - gdp_imp*tax_rate
    debt_cumul = np.cumsum(deficit_p)

    with col_fc:
        # Chart 1
        fig_f1 = go.Figure()
        fig_f1.add_trace(go.Bar(x=periods, y=gdp_imp,   name="GDP impact (ΔY)",
            marker_color=C["terra"], opacity=0.85))
        fig_f1.add_trace(go.Bar(x=periods, y=inv_crowd, name="Investment crowded out",
            marker_color=C["sienna"], opacity=0.75))
        fig_f1.add_trace(go.Scatter(x=periods, y=sp, name="Initial shock",
            line=dict(color=C["ink"], dash="dot", width=2), mode="lines"))
        fig_f1.update_layout(**base_layout(300, "GDP impact & crowding-out"), barmode="relative")
        fig_f1.update_yaxes(title_text="% of GDP"); fig_f1.update_xaxes(title_text="Periods")
        st.plotly_chart(fig_f1, use_container_width=True)

        # Charts 2a + 2b side by side
        c2a, c2b = st.columns(2, gap="medium")
        with c2a:
            gross   = shock_size*base_mult
            sv_leak = gross*(1-mpc)
            tx_leak = gross*mpc*tax_rate
            im_leak = gross*mpc*(1-tax_rate)*import_r
            cr_drag = shock_size*crowding*0.5
            net_gdp = gross - sv_leak - tx_leak - im_leak - cr_drag
            lk_l = ["Gross", "−Saving", "−Tax", "−Import", "−Crowding", "Net ΔY"]
            lk_v = [gross, -sv_leak, -tx_leak, -im_leak, -cr_drag, net_gdp]
            lk_c = [C["stone"], C["sand_dark"], C["sand_dark"], C["sand_dark"], C["sienna"], C["terra"]]
            fig_f2a = go.Figure(go.Bar(
                x=lk_l, y=lk_v, marker_color=lk_c,
                text=[f"{v:.2f}" for v in lk_v], textposition="outside",
                textfont=dict(size=10),
            ))
            fig_f2a.update_layout(**base_layout(270, "Multiplier leakages (period 0)"))
            fig_f2a.update_yaxes(title_text="% of GDP")
            st.plotly_chart(fig_f2a, use_container_width=True)

        with c2b:
            fig_f2b = go.Figure()
            fig_f2b.add_trace(go.Scatter(x=periods, y=debt_cumul, name="Cumulative deficit",
                line=dict(color=C["terra"], width=2.5), mode="lines+markers",
                marker=dict(size=5)))
            fig_f2b.add_trace(go.Scatter(x=periods, y=deficit_p, name="Period deficit",
                line=dict(color=C["sand_dark"], width=1.5, dash="dot"), mode="lines"))
            fig_f2b.update_layout(**base_layout(270, "Debt accumulation path"))
            fig_f2b.update_yaxes(title_text="% of GDP")
            st.plotly_chart(fig_f2b, use_container_width=True)

        # Chart 3 — 3D multiplier surface
        st.markdown('<div class="section-label-inline">Multiplier sensitivity — MPC × import propensity</div>', unsafe_allow_html=True)
        mpc_g = np.linspace(0.3, 0.95, 40)
        m_g   = np.linspace(0.0, 0.40, 40)
        MG, ImG = np.meshgrid(mpc_g, m_g)
        MULT_S  = 1 / (1 - MG*(1-tax_rate) + ImG) * (1-crowding) * state_adj
        fig_f3 = go.Figure(go.Surface(
            x=mpc_g, y=m_g, z=MULT_S,
            colorscale=[[0, C["cream"]], [0.5, C["sand"]], [1, C["terra"]]],
            showscale=True,
            contours=dict(z=dict(show=True, usecolormap=True, project_z=True)),
        ))
        fig_f3.add_trace(go.Scatter3d(
            x=[mpc], y=[import_r], z=[eff_mult], mode="markers",
            marker=dict(size=8, color=C["ink"], symbol="diamond"),
            name="Current params",
        ))
        fig_f3.update_layout(
            height=380, paper_bgcolor=C["cream"],
            font=dict(family="DM Sans", color=C["ink"], size=11),
            margin=dict(l=0, r=0, t=36, b=0),
            scene=dict(
                xaxis_title="MPC", yaxis_title="Import propensity", zaxis_title="Multiplier",
                xaxis=dict(gridcolor=C["rule"]), yaxis=dict(gridcolor=C["rule"]),
                zaxis=dict(gridcolor=C["rule"]), bgcolor=C["cream"],
            ),
            title=dict(text="Multiplier surface",
                font=dict(family="DM Serif Display", size=15, color=C["ink"])),
        )
        st.plotly_chart(fig_f3, use_container_width=True)

    st.markdown('<div class="section-label">Key results</div>', unsafe_allow_html=True)
    for col, num, lbl, cls in zip(
        st.columns(4, gap="medium"),
        [f"{eff_mult:.2f}×", f"{gdp_imp[0]:.2f}%", f"{debt_cumul[-1]:.2f}%", f"{tax_mult:.2f}×"],
        ["Effective multiplier", "GDP impact (t=0)", "Cumulative deficit", "Tax cut multiplier"],
        ["rc-terra", "rc-sienna", "rc-stone", "rc-sand"],
    ):
        with col:
            st.markdown(f'<div class="result-card {cls}"><div class="result-num">{num}</div><div class="result-label">{lbl}</div></div>', unsafe_allow_html=True)

    st.markdown(f"""
    <div class="interp-box interp-box-sand" style="margin-top:16px;">
      <div class="interp-title">Reading the model</div>
      <div class="interp-text">
        With MPC = {mpc}, tax rate = {tax_rate}, import propensity = {import_r}, the base multiplier is
        <strong>{base_mult:.2f}</strong>. After crowding-out ({int(crowding*100)}%) and state adjustment,
        the effective multiplier is <strong>{eff_mult:.2f}</strong>.
        The 3D surface shows how the multiplier varies across the MPC–import space.
        The diamond marks your current parameter combination.
      </div>
    </div>
    """, unsafe_allow_html=True)

    df_f2 = pd.DataFrame({
        "Period": periods, "Shock (% GDP)": sp,
        "GDP impact": gdp_imp, "Crowded-out investment": inv_crowd,
        "Period deficit": deficit_p, "Cumulative deficit": debt_cumul,
    })
    st.download_button("Export fiscal scenario CSV",
        data=df_f2.to_csv(index=False).encode(),
        file_name="policy_lab_fiscal.csv", mime="text/csv", key="fp_dl")


# ─────────────────────────────────────────────────────────────────────
# MODULE 3 — DEBT DYNAMICS
# ─────────────────────────────────────────────────────────────────────
with tab3:
    st.markdown('<div class="section-label">Debt Sustainability — r−g Dynamics</div>', unsafe_allow_html=True)
    col_dp, col_dc = st.columns([1, 2.2], gap="large")

    with col_dp:
        st.markdown('<div class="param-panel"><div class="param-title">Parameters</div><div class="param-desc">Explore debt sustainability through Δd = (r−g)d − pb.</div></div>', unsafe_allow_html=True)
        st.markdown('<div class="param-group">Initial conditions</div>', unsafe_allow_html=True)
        d0      = st.slider("Initial debt (% of GDP)",     0.0, 200.0, 80.0, 1.0, key="dd_d0")
        pb0     = st.slider("Primary balance (% of GDP)", -8.0,   8.0,  1.0, 0.1, key="dd_pb")
        r_nom   = st.slider("Nominal interest rate r (%)", 0.0,  10.0,  4.0, 0.1, key="dd_r")
        g_nom   = st.slider("Nominal GDP growth g (%)",   -3.0,   8.0,  3.0, 0.1, key="dd_g")
        d_horiz = st.slider("Horizon (years)",              5,    30,   15,  1,   key="dd_hz")
        st.markdown('<div class="param-group">Shock</div>', unsafe_allow_html=True)
        shock_yr  = st.slider("Shock in year",               1, d_horiz,  3,   1,  key="dd_sy")
        shock_pb  = st.slider("Primary balance shock (pp)", -5.0,   5.0, -2.0, 0.1, key="dd_spb")
        shock_r   = st.slider("Interest rate shock (pp)",  -2.0,   4.0,  1.0, 0.1, key="dd_sr")
        shock_g   = st.slider("Growth shock (pp)",         -5.0,   3.0, -1.5, 0.1, key="dd_sg")
        shock_dur = st.slider("Shock duration (years)",      1,    10,    3,  1,   key="dd_sdur")

    def sim_debt(d0, pb, r, g, H, sy=None, spb=0, sr=0, sg=0, sdur=0):
        d = np.zeros(H+1); d[0] = d0
        rg_a, pb_a = np.zeros(H+1), np.zeros(H+1)
        for t in range(1, H+1):
            in_s = sy is not None and sy <= t < sy+sdur
            r_t  = r + (sr if in_s else 0)
            g_t  = g + (sg if in_s else 0)
            pb_t = pb + (spb if in_s else 0)
            rg   = (r_t - g_t)/100
            d[t] = (1+rg)*d[t-1] - pb_t
            rg_a[t] = rg*100; pb_a[t] = pb_t
        return d, rg_a, pb_a

    yrs   = np.arange(d_horiz+1)
    d_b,  rg_b,  pb_b  = sim_debt(d0, pb0, r_nom, g_nom, d_horiz)
    d_sh, rg_sh, pb_sh = sim_debt(d0, pb0, r_nom, g_nom, d_horiz,
                                   shock_yr, shock_pb, shock_r, shock_g, shock_dur)
    stab_pb = (r_nom-g_nom)/100 * d0

    with col_dc:
        # Chart 1 — trajectories with fan
        upper = d_b + yrs*1.5*0.6
        lower = d_b - yrs*1.5*0.6
        fig_d1 = go.Figure()
        fig_d1.add_trace(go.Scatter(
            x=np.concatenate([yrs, yrs[::-1]]),
            y=np.concatenate([upper, lower[::-1]]),
            fill="toself", fillcolor="rgba(196,168,130,0.18)",
            line=dict(width=0), name="Uncertainty band (±1σ)",
        ))
        fig_d1.add_trace(go.Scatter(x=yrs, y=d_b,  name="Baseline",
            line=dict(color=C["stone"], width=2.5), mode="lines"))
        fig_d1.add_trace(go.Scatter(x=yrs, y=d_sh, name="With shock",
            line=dict(color=C["terra"], width=2.5), mode="lines"))
        fig_d1.add_hline(y=d0, line_color=C["rule"], line_width=1, line_dash="dot",
            annotation_text="Initial debt", annotation_font_color=C["muted"])
        fig_d1.add_vline(x=shock_yr, line_color=C["sienna"], line_width=1.5, line_dash="dash",
            annotation_text="Shock", annotation_font_color=C["sienna"])
        fig_d1.update_layout(**base_layout(320, "Debt-to-GDP trajectory"))
        fig_d1.update_yaxes(title_text="Debt (% of GDP)"); fig_d1.update_xaxes(title_text="Years")
        st.plotly_chart(fig_d1, use_container_width=True)

        # Charts 2a + 2b side by side
        c2a, c2b = st.columns(2, gap="medium")
        stab_path = rg_b[1:]*d_b[:-1]/100
        with c2a:
            fig_d2a = go.Figure()
            fig_d2a.add_trace(go.Scatter(x=yrs, y=rg_b,  name="Baseline r−g",
                line=dict(color=C["stone"], width=2), mode="lines"))
            fig_d2a.add_trace(go.Scatter(x=yrs, y=rg_sh, name="With shock r−g",
                line=dict(color=C["terra"], width=2, dash="dot"), mode="lines"))
            fig_d2a.add_hline(y=0, line_color=C["ink"], line_width=1)
            fig_d2a.update_layout(**base_layout(260, "r − g differential"))
            fig_d2a.update_yaxes(title_text="pp")
            st.plotly_chart(fig_d2a, use_container_width=True)

        with c2b:
            fig_d2b = go.Figure()
            fig_d2b.add_trace(go.Scatter(x=yrs[1:], y=stab_path, name="Stabilising PB needed",
                line=dict(color=C["sienna"], width=2), mode="lines"))
            fig_d2b.add_trace(go.Scatter(x=yrs[1:], y=pb_b[1:],  name="Actual PB (baseline)",
                line=dict(color=C["stone"],  width=2, dash="dot"), mode="lines"))
            fig_d2b.add_trace(go.Scatter(x=yrs[1:], y=pb_sh[1:], name="Actual PB (shocked)",
                line=dict(color=C["terra"],  width=1.5, dash="dot"), mode="lines"))
            fig_d2b.add_hline(y=0, line_color=C["rule"], line_width=1)
            fig_d2b.update_layout(**base_layout(260, "Primary balance: required vs actual"))
            fig_d2b.update_yaxes(title_text="% of GDP")
            st.plotly_chart(fig_d2b, use_container_width=True)

        # Chart 3 — sustainability contour map
        st.markdown('<div class="section-label-inline">Debt sustainability map</div>', unsafe_allow_html=True)
        pb_rng = np.linspace(-6, 8, 60)
        d_rng  = np.linspace(0, 200, 60)
        PB_G, D_G = np.meshgrid(pb_rng, d_rng)
        rg_val  = (r_nom-g_nom)/100
        Delta_D = rg_val*D_G - PB_G
        fig_d3  = go.Figure()
        fig_d3.add_trace(go.Contour(
            x=pb_rng, y=d_rng, z=Delta_D,
            colorscale=[[0, C["stone"]], [0.5, C["cream"]], [1, C["terra"]]],
            contours=dict(start=-8, end=8, size=1,
                showlabels=True, labelfont=dict(size=9)),
            colorbar=dict(title="Δdebt/yr", tickfont=dict(size=9)),
            line=dict(width=0.5),
        ))
        fig_d3.add_trace(go.Scatter(x=pb_b[1:], y=d_b[:-1],
            mode="lines+markers", name="Baseline path",
            line=dict(color=C["ink"], width=2),
            marker=dict(size=5, color=C["stone"])))
        fig_d3.add_trace(go.Scatter(x=pb_sh[1:], y=d_sh[:-1],
            mode="lines+markers", name="Shocked path",
            line=dict(color=C["terra"], width=2, dash="dot"),
            marker=dict(size=5, color=C["terra"])))
        fig_d3.update_layout(**base_layout(350, "Sustainability map (stone = debt falling, terracotta = rising)"))
        fig_d3.update_xaxes(title_text="Primary balance (% GDP)")
        fig_d3.update_yaxes(title_text="Debt (% GDP)")
        st.plotly_chart(fig_d3, use_container_width=True)

    rg_pct = r_nom - g_nom
    sust   = ("sustainable" if pb0 >= stab_pb and rg_pct <= 0
              else "potentially explosive" if rg_pct > 2 and pb0 < stab_pb
              else "requires a primary surplus to stabilise")
    st.markdown('<div class="section-label">Key results</div>', unsafe_allow_html=True)
    for col, num, lbl, cls in zip(
        st.columns(4, gap="medium"),
        [f"{rg_pct:+.1f}pp", f"{stab_pb:.1f}%", f"{d_b[-1]:.1f}%", f"{d_sh[-1]:.1f}%"],
        ["r − g differential", "Stabilising primary balance",
         f"Baseline debt yr {d_horiz}", f"Shocked debt yr {d_horiz}"],
        ["rc-terra" if rg_pct > 0 else "rc-stone", "rc-sienna", "rc-stone", "rc-terra"],
    ):
        with col:
            st.markdown(f'<div class="result-card {cls}"><div class="result-num">{num}</div><div class="result-label">{lbl}</div></div>', unsafe_allow_html=True)

    st.markdown(f"""
    <div class="interp-box interp-box-stone" style="margin-top:16px;">
      <div class="interp-title">Reading the model</div>
      <div class="interp-text">
        With r − g = <strong>{rg_pct:+.1f}pp</strong>, the debt trajectory is <strong>{sust}</strong>.
        To stabilise debt at {d0:.0f}% of GDP, the government needs a primary balance of at least
        <strong>{stab_pb:.2f}%</strong>. The sustainability map shows where debt rises vs falls.
        The shock shifts the trajectory by {shock_r:+.1f}pp on r, {shock_g:+.1f}pp on g,
        and {shock_pb:+.1f}pp on the primary balance for {shock_dur} year(s).
      </div>
    </div>
    """, unsafe_allow_html=True)

    df_d3 = pd.DataFrame({
        "Year": yrs, "Debt baseline (%)": d_b, "Debt shocked (%)": d_sh,
        "r−g baseline (pp)": rg_b, "r−g shocked (pp)": rg_sh,
        "PB baseline (%)": pb_b, "PB shocked (%)": pb_sh,
    })
    st.download_button("Export debt dynamics CSV",
        data=df_d3.to_csv(index=False).encode(),
        file_name="policy_lab_debt.csv", mime="text/csv", key="dd_dl")


# ─────────────────────────────────────────────────────────────────────
# MODULE 4 — TRADE & TARIFF INCIDENCE
# ─────────────────────────────────────────────────────────────────────
with tab4:
    st.markdown('<div class="section-label">Trade Policy — Tariff Incidence & Welfare</div>', unsafe_allow_html=True)
    col_tp, col_tc = st.columns([1, 2.2], gap="large")

    with col_tp:
        st.markdown('<div class="param-panel"><div class="param-title">Parameters</div><div class="param-desc">Analyse how a tariff is split between consumers and foreign exporters, and trace welfare effects.</div></div>', unsafe_allow_html=True)
        st.markdown('<div class="param-group">Market structure</div>', unsafe_allow_html=True)
        p_world    = st.slider("World price (Pw)",               50, 200, 100,  5,   key="tr_pw")
        tariff_pct = st.slider("Ad valorem tariff (%)",           0,  80,  20,  1,   key="tr_t")
        ed_demand  = st.slider("Price elasticity of demand |εD|", 0.1, 5.0, 1.5, 0.1, key="tr_ed")
        es_supply  = st.slider("Domestic supply elasticity εS",   0.1, 5.0, 1.0, 0.1, key="tr_es")
        es_foreign = st.slider("Foreign export elasticity εX*",   0.1,10.0, 3.0, 0.1, key="tr_ex")
        st.markdown('<div class="param-group">Baseline quantities</div>', unsafe_allow_html=True)
        Q_d0 = st.slider("Domestic demand at Pw", 50, 500, 200, 10, key="tr_qd")
        Q_s0 = st.slider("Domestic supply at Pw", 10, 300,  80, 10, key="tr_qs")
        Q_m0 = Q_d0 - Q_s0
        st.markdown('<div class="param-group">Scenario</div>', unsafe_allow_html=True)
        csize   = st.radio("Country type",
            ["Small (price-taker)", "Large (ToT power)"], key="tr_cs")
        show_dw = st.checkbox("Show welfare shading", value=True, key="tr_dw")

    t_abs           = p_world * tariff_pct / 100
    pass_through    = 1.0 if csize == "Small (price-taker)" else es_foreign/(es_foreign+ed_demand)
    consumer_burden = t_abs * pass_through
    exporter_burden = t_abs * (1-pass_through)
    p_domestic      = p_world + consumer_burden
    p_foreign       = p_world - exporter_burden
    Q_d1 = Q_d0*(p_world/p_domestic)**ed_demand if p_domestic > 0 else Q_d0
    Q_s1 = Q_s0*(p_domestic/p_world)**es_supply
    Q_m1 = max(0, Q_d1-Q_s1)
    CS_loss  = -0.5*(Q_d0+Q_d1)*consumer_burden
    PS_gain  =  0.5*(Q_s0+Q_s1)*consumer_burden
    TR_gain  = Q_m1*t_abs
    ToT_gain = Q_m1*exporter_burden
    net_welf = CS_loss + PS_gain + TR_gain + ToT_gain

    with col_tc:
        # Chart 1 — supply-demand diagram
        p_ax  = np.linspace(max(1, p_world*0.3), p_world*1.8, 200)
        Qd_cv = Q_d0*(p_world/p_ax)**ed_demand
        Qs_cv = Q_s0*(p_ax/p_world)**es_supply
        fig_t1 = go.Figure()
        fig_t1.add_trace(go.Scatter(x=Qd_cv, y=p_ax, name="Demand",
            line=dict(color=C["terra"], width=2.5), mode="lines"))
        fig_t1.add_trace(go.Scatter(x=Qs_cv, y=p_ax, name="Domestic supply",
            line=dict(color=C["stone"], width=2.5), mode="lines"))
        fig_t1.add_hline(y=p_world,    line_color=C["ink"],      line_width=1.5, line_dash="dot",
            annotation_text=f"Pw = {p_world}", annotation_font_color=C["ink"])
        fig_t1.add_hline(y=p_domestic, line_color=C["terra"],    line_width=2,   line_dash="dash",
            annotation_text=f"Pd = {p_domestic:.1f}", annotation_font_color=C["terra"])
        if csize == "Large (ToT power)":
            fig_t1.add_hline(y=p_foreign, line_color=C["sand_dark"], line_width=1.5, line_dash="dot",
                annotation_text=f"Pf = {p_foreign:.1f}", annotation_font_color=C["sand_dark"])
        if show_dw:
            fig_t1.add_shape(type="rect", x0=0, x1=Q_d1, y0=p_world, y1=p_domestic,
                fillcolor="rgba(201,98,42,0.13)", line_width=0)
            fig_t1.add_shape(type="rect", x0=0, x1=Q_s1, y0=p_world, y1=p_domestic,
                fillcolor="rgba(122,111,98,0.18)", line_width=0)
        fig_t1.update_layout(**base_layout(360, "Tariff incidence — supply & demand"))
        fig_t1.update_xaxes(title_text="Quantity", range=[0, Q_d0*1.2])
        fig_t1.update_yaxes(title_text="Price",    range=[p_world*0.5, p_world*1.5])
        st.plotly_chart(fig_t1, use_container_width=True)

        # Charts 2a (welfare bars) + 2b (incidence horizontal bar) — side by side, no subplots
        c2a, c2b = st.columns(2, gap="medium")
        with c2a:
            wl = ["CS change", "PS change", "Tariff revenue", "ToT gain", "Net welfare"]
            wv = [CS_loss, PS_gain, TR_gain, ToT_gain, net_welf]
            wc = [C["sienna"] if v < 0 else C["stone"] for v in wv[:-1]] + \
                 [C["terra"] if net_welf >= 0 else C["sienna"]]
            fig_t2a = go.Figure(go.Bar(
                x=wl, y=wv, marker_color=wc,
                text=[f"{v:.1f}" for v in wv], textposition="outside",
                textfont=dict(size=10),
            ))
            fig_t2a.update_layout(**base_layout(270, "Welfare decomposition"))
            fig_t2a.update_yaxes(title_text="Units")
            st.plotly_chart(fig_t2a, use_container_width=True)

        with c2b:
            # Horizontal stacked bar for incidence split — replaces go.Pie
            fig_t2b = go.Figure()
            fig_t2b.add_trace(go.Bar(
                x=[consumer_burden], y=["Tariff split"],
                name=f"Consumer ({pass_through*100:.0f}%)",
                orientation="h", marker_color=C["terra"],
                text=f"Consumer  {pass_through*100:.0f}%",
                textposition="inside", insidetextanchor="middle",
                textfont=dict(color="#fff", size=13),
            ))
            fig_t2b.add_trace(go.Bar(
                x=[max(0.001, exporter_burden)], y=["Tariff split"],
                name=f"Exporter ({(1-pass_through)*100:.0f}%)",
                orientation="h", marker_color=C["stone"],
                text=f"Exporter  {(1-pass_through)*100:.0f}%",
                textposition="inside", insidetextanchor="middle",
                textfont=dict(color="#fff", size=13),
            ))
            fig_t2b.update_layout(
                **base_layout(270, "Tariff incidence split"),
                barmode="stack",
            )
            fig_t2b.update_xaxes(title_text="Tariff per unit absorbed")
            fig_t2b.update_yaxes(showticklabels=False)
            st.plotly_chart(fig_t2b, use_container_width=True)

        # Chart 3 — welfare contour map
        st.markdown('<div class="section-label-inline">Net welfare sensitivity — tariff rate × foreign elasticity</div>', unsafe_allow_html=True)
        t_rng  = np.linspace(0, 80, 50)
        fe_rng = np.linspace(0.5, 10, 40)
        T_G, FE_G = np.meshgrid(t_rng, fe_rng)

        def nw_surf(t_pct, fes):
            t_u  = p_world*t_pct/100
            pt   = fes/(fes+ed_demand) if csize == "Large (ToT power)" else 1.0
            pd_l = p_world + t_u*pt
            if pd_l <= 0: return 0.0
            qd_l = Q_d0*(p_world/pd_l)**ed_demand
            qs_l = Q_s0*(pd_l/p_world)**es_supply
            qm_l = max(0, qd_l-qs_l)
            return -0.5*(Q_d0+qd_l)*t_u*pt + 0.5*(Q_s0+qs_l)*t_u*pt + qm_l*t_u + qm_l*t_u*(1-pt)

        NW_S  = np.vectorize(nw_surf)(T_G, FE_G)
        fig_t3 = go.Figure()
        fig_t3.add_trace(go.Contour(
            x=t_rng, y=fe_rng, z=NW_S,
            colorscale=[[0, C["sienna"]], [0.5, C["cream"]], [1, C["stone"]]],
            contours=dict(showlabels=True, labelfont=dict(size=9)),
            colorbar=dict(title="Net welfare", tickfont=dict(size=9)),
            line=dict(width=0.5),
        ))
        fig_t3.add_trace(go.Scatter(
            x=[tariff_pct], y=[es_foreign], mode="markers",
            marker=dict(size=12, color=C["ink"], symbol="star"),
            name="Current params",
        ))
        fig_t3.update_layout(**base_layout(330, "Net welfare map — tariff rate × foreign supply elasticity"))
        fig_t3.update_xaxes(title_text="Tariff rate (%)")
        fig_t3.update_yaxes(title_text="Foreign export supply elasticity")
        st.plotly_chart(fig_t3, use_container_width=True)

    sl = "small country" if csize == "Small (price-taker)" else "large country"
    st.markdown('<div class="section-label">Key results</div>', unsafe_allow_html=True)
    for col, num, lbl, cls in zip(
        st.columns(4, gap="medium"),
        [f"{pass_through*100:.0f}%", f"{Q_m1:.0f}", f"{TR_gain:.1f}", f"{net_welf:.1f}"],
        ["Pass-through to consumers", "Imports after tariff", "Tariff revenue", "Net welfare change"],
        ["rc-terra", "rc-stone", "rc-sienna", "rc-sand"],
    ):
        with col:
            st.markdown(f'<div class="result-card {cls}"><div class="result-num">{num}</div><div class="result-label">{lbl}</div></div>', unsafe_allow_html=True)

    st.markdown(f"""
    <div class="interp-box" style="margin-top:16px;">
      <div class="interp-title">Reading the model</div>
      <div class="interp-text">
        As a <strong>{sl}</strong>, a {tariff_pct}% tariff raises the domestic price from {p_world}
        to <strong>{p_domestic:.1f}</strong>. <strong>{pass_through*100:.0f}%</strong> falls on consumers;
        <strong>{(1-pass_through)*100:.0f}%</strong> is absorbed by foreign exporters.
        Imports fall from {Q_m0:.0f} to {Q_m1:.0f} units. Consumer surplus changes by {CS_loss:.1f},
        producer surplus by {PS_gain:.1f}, tariff revenue is {TR_gain:.1f}.
        Net welfare: <strong>{net_welf:.1f}</strong>
        ({'net gain — optimal tariff territory' if net_welf > 0 else 'net loss'}).
      </div>
    </div>
    """, unsafe_allow_html=True)

    df_t4 = pd.DataFrame({
        "Scenario": ["Baseline", "With tariff"],
        "Price": [p_world, p_domestic], "Demand": [Q_d0, Q_d1],
        "Domestic supply": [Q_s0, Q_s1], "Imports": [Q_m0, Q_m1],
        "CS change": [0, CS_loss], "PS change": [0, PS_gain],
        "Tariff revenue": [0, TR_gain], "Net welfare": [0, net_welf],
    })
    st.download_button("Export trade scenario CSV",
        data=df_t4.to_csv(index=False).encode(),
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
