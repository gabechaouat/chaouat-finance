import streamlit as st

# =========================
# EDIT THESE PAGE PATHS ONLY
# =========================
# Use your real multipage file names inside the /pages folder.
# Examples:
# "pages/Policy_Lab.py"
# "pages/Teaching_Material.py"
PAGES = {
    "Policy Lab": "pages/1_Policy_Lab.py",
    "Teaching Material": "pages/3_Teaching_Material.py",
}

st.set_page_config(page_title="Chaouat Economics Lab", page_icon="📘", layout="wide")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Montserrat:wght@400;600;700&display=swap');
:root{
  --primary:#007BA7;
  --accent:#00B0FF;
  --bg:#F7FAFC;
  --card:#FFFFFF;
  --text:#0F172A;
  --muted:#64748B;
  --border:#E2E8F0;
}
html, body, * { font-family: 'Montserrat', sans-serif !important; }

.block-container { padding-top: 1.2rem; padding-bottom: 2rem; }
h1, h2, h3 { color: var(--text); }
small, .muted { color: var(--muted); }

.cf-hero{
  background: linear-gradient(135deg, var(--primary) 0%, var(--accent) 100%);
  color: #fff;
  padding: 30px 34px;
  border-radius: 20px;
  box-shadow: 0 10px 30px rgba(0,123,167,.22);
  margin: 8px 0 14px 0;
}
.cf-brand{ font-weight: 800; font-size: 42px; letter-spacing: .2px; line-height: 1.1; }
.cf-sub{ margin-top: 10px; opacity: .95; font-size: 15.5px; line-height: 1.55; max-width: 980px; }

.cf-panel{
  background: var(--card);
  border: 1px solid var(--border);
  border-radius: 18px;
  padding: 18px 18px;
  box-shadow: 0 8px 30px rgba(15,23,42,.06);
}

.cf-tile{
  background: var(--card);
  border: 1px solid var(--border);
  border-radius: 18px;
  padding: 16px 16px;
  box-shadow: 0 8px 30px rgba(15,23,42,.06);
  height: 100%;
}

.cf-kpi{
  background: #FFFFFF;
  border: 1px solid var(--border);
  border-radius: 16px;
  padding: 14px 14px;
  box-shadow: 0 8px 30px rgba(15,23,42,.05);
}

.hr {
  border-top: 1px solid var(--border);
  margin: 14px 0;
}
</style>
""", unsafe_allow_html=True)

# ---------- HERO ----------
st.markdown("""
<div class="cf-hero">
  <div class="cf-brand">Chaouat Economics Lab</div>
  <div class="cf-sub">
    Build a tutoring session in minutes: pick a topic, set assumptions, run a lab, export a clean handout.
    Everything here is teaching-first and explicit about inputs and mechanisms.
  </div>
</div>
""", unsafe_allow_html=True)

# ---------- QUICK LAUNCH ----------
left, right = st.columns([1.15, 0.85], gap="large")

def _switch_to(page_path: str):
    """Robust navigation: uses st.switch_page if available; otherwise shows a helpful message."""
    try:
        st.switch_page(page_path)
    except Exception:
        st.error(
            "Navigation failed. Update the PAGES paths at the top of this file to match your /pages filenames."
        )

with left:
    st.markdown("## Start a session")
    st.caption("Choose a configuration; the page will generate a suggested lesson flow and recommended next clicks.")

    topic = st.selectbox(
        "Topic",
        [
            "Monetary policy (Taylor rule)",
            "Fiscal policy (multipliers)",
            "Debt dynamics (snowball vs primary balance)",
            "Growth & development (institutions, poverty traps)",
            "Trade & tariffs (welfare, incidence)",
        ],
        index=0,
    )

    level = st.selectbox("Audience level", ["High school", "Undergraduate", "Advanced"], index=1)
    duration = st.select_slider("Session duration", options=[30, 45, 60, 75, 90], value=60)
    format_ = st.selectbox("Format", ["1:1 tutoring", "Small group", "Classroom"], index=0)

    st.markdown('<div class="hr"></div>', unsafe_allow_html=True)

    # A simple, explicit “session generator” (no AI, no mission copy, just structure)
    templates = {
        "Monetary policy (Taylor rule)": {
            "Warm-up": "Define inflation vs inflation target; interpret output gap; why a rule exists.",
            "Core": "Run the Taylor-rule toy model under 2–3 shocks (demand vs supply).",
            "Extension": "Compare φπ and smoothing. Ask: when does the rule become destabilizing?",
            "Takeaway": "Policy rate decomposes into base + inflation gap response + output gap response.",
        },
        "Fiscal policy (multipliers)": {
            "Warm-up": "What is a multiplier? Marginal propensity to consume, slack vs capacity.",
            "Core": "Simulate a temporary spending shock under different MPC / leakages.",
            "Extension": "Debt implications; compare short-run boost vs medium-run financing.",
            "Takeaway": "Multipliers are state-dependent and assumption-dependent.",
        },
        "Debt dynamics (snowball vs primary balance)": {
            "Warm-up": "Debt/GDP identity; r-g intuition; primary balance definition.",
            "Core": "Simulate debt paths under r>g vs r<g and primary surpluses/deficits.",
            "Extension": "Stress test with growth shocks; interpret sustainability vs liquidity.",
            "Takeaway": "Sustainability depends on r-g, starting debt, and policy reaction.",
        },
        "Growth & development (institutions, poverty traps)": {
            "Warm-up": "Define development dimensions; institutions and state capacity.",
            "Core": "Walk through a poverty-trap mechanism and a credible intervention design.",
            "Extension": "Discuss evidence: identification, external validity, measurement.",
            "Takeaway": "Mechanisms + evidence discipline are complements, not substitutes.",
        },
        "Trade & tariffs (welfare, incidence)": {
            "Warm-up": "Consumer surplus, producer surplus, deadweight loss.",
            "Core": "Tariff incidence: elasticities and who pays. Compare small vs large country.",
            "Extension": "Retaliation and supply chains: second-round effects.",
            "Takeaway": "Tariffs redistribute and create efficiency losses; incidence is empirical.",
        },
    }

    plan = templates[topic]

    st.markdown("### Suggested flow")
    st.write(
        f"**{duration} minutes · {format_} · {level}**\n\n"
        f"- **Warm-up (10 min):** {plan['Warm-up']}\n"
        f"- **Core (25–35 min):** {plan['Core']}\n"
        f"- **Extension (10–20 min):** {plan['Extension']}\n"
        f"- **Takeaway (5 min):** {plan['Takeaway']}"
    )

    st.markdown("### Next clicks")
    cta1, cta2 = st.columns(2, gap="medium")
    with cta1:
        if st.button("Open Policy Lab", use_container_width=True):
            _switch_to(PAGES["Policy Lab"])
    with cta2:
        if st.button("Open Teaching Material", use_container_width=True):
            _switch_to(PAGES["Teaching Material"])

with right:
    st.markdown("## Home dashboard")

    r1, r2 = st.columns(2, gap="medium")
    with r1:
        st.markdown('<div class="cf-kpi"><b>Default posture</b><div class="muted">Explicit assumptions</div></div>', unsafe_allow_html=True)
    with r2:
        st.markdown('<div class="cf-kpi"><b>Outputs</b><div class="muted">Charts + CSV exports</div></div>', unsafe_allow_html=True)

    st.markdown("")

    st.markdown('<div class="cf-panel">', unsafe_allow_html=True)
    st.markdown("### Featured starting points")
    st.caption("Small set of high-signal entries (avoid clutter).")

    f1, f2 = st.columns(2, gap="medium")
    with f1:
        st.markdown(
            '<div class="cf-tile"><h4>Monetary Policy Lab</h4>'
            '<div class="muted">Rule mechanics, shocks, and decomposition.</div></div>',
            unsafe_allow_html=True,
        )
        if st.button("Launch: Taylor rule lab", use_container_width=True, key="launch_taylor"):
            _switch_to(PAGES["Policy Lab"])

    with f2:
        st.markdown(
            '<div class="cf-tile"><h4>Printable teaching decks</h4>'
            '<div class="muted">Structured slides and worksheets for sessions.</div></div>',
            unsafe_allow_html=True,
        )
        if st.button("Browse teaching material", use_container_width=True, key="launch_decks"):
            _switch_to(PAGES["Teaching Material"])

    st.markdown('</div>', unsafe_allow_html=True)

# ---------- FOOTER ----------
st.markdown("")
st.caption("© Chaouat Economics Lab · Built with Python/Streamlit · Educational use only")
