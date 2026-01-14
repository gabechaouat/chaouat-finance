import streamlit as st
from datetime import datetime

# =========================
# EDIT THESE PAGE PATHS ONLY
# =========================
PAGES = {
    "Policy Lab": "pages/Policy_Lab.py",
    "Teaching Material": "pages/Teaching_Material.py",
}

st.set_page_config(page_title="Chaouat Economics Lab", page_icon="📘", layout="wide")

# ---- State: recent launches (home-only, additive) ----
if "home_recent" not in st.session_state:
    st.session_state.home_recent = []  # [{"label":..., "page":..., "ts":...}]

def _switch_to(page_path: str, label_for_recent: str, page_name: str):
    st.session_state.home_recent.insert(
        0,
        {"label": label_for_recent, "page": page_name, "ts": datetime.now().strftime("%Y-%m-%d %H:%M")},
    )
    st.session_state.home_recent = st.session_state.home_recent[:8]
    try:
        st.switch_page(page_path)
    except Exception:
        st.error("Navigation failed. Verify that the PAGES paths match your /pages filenames.")

# ---- Style: stronger brand banner + blue CTAs + less 'blank white' ----
st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=Montserrat:wght@400;600;700;800&display=swap');

html, body, * { font-family: 'Montserrat', sans-serif !important; }

:root{
  --primary:#007BA7;
  --accent:#00B0FF;
  --navy:#0B2B3A;
  --bg1:#F3F8FE;     /* soft blue-gray */
  --bg2:#FFFFFF;
  --card:#FFFFFF;
  --text:#0F172A;
  --muted:#64748B;
  --border:#DCE6F2;
}

body {
  background: radial-gradient(1200px 600px at 20% -10%, rgba(0,176,255,.18) 0%, rgba(0,176,255,0) 60%),
              radial-gradient(900px 500px at 95% 10%, rgba(0,123,167,.16) 0%, rgba(0,123,167,0) 55%),
              linear-gradient(180deg, var(--bg1) 0%, var(--bg2) 70%);
}

.block-container { padding-top: 0.8rem; padding-bottom: 2rem; max-width: 1220px; }

.cf-hero{
  background: linear-gradient(135deg, var(--primary) 0%, var(--accent) 100%);
  color: #fff;
  padding: 28px 30px;
  border-radius: 22px;
  box-shadow: 0 14px 36px rgba(0,123,167,.26);
  margin: 6px 0 14px 0;
  position: relative;
  overflow: hidden;
}
.cf-hero:before{
  content:"";
  position:absolute;
  inset:-40px -80px auto auto;
  width: 340px;
  height: 340px;
  background: rgba(255,255,255,.12);
  border-radius: 999px;
  transform: rotate(10deg);
}
.cf-brand{
  font-weight: 800;
  font-size: 46px;
  letter-spacing: .2px;
  line-height: 1.05;
}
.cf-sub{
  margin-top: 10px;
  opacity: .95;
  font-size: 15.5px;
  line-height: 1.55;
  max-width: 980px;
}
.cf-sub b { font-weight: 800; }

.section{
  background: var(--card);
  border: 1px solid var(--border);
  border-radius: 18px;
  padding: 16px 16px;
  box-shadow: 0 10px 30px rgba(15,23,42,.06);
}

.badge{
  display:inline-block;
  padding: 6px 10px;
  border-radius: 999px;
  font-size: 12px;
  font-weight: 700;
  color: #073042;
  background: rgba(0,176,255,.18);
  border: 1px solid rgba(0,176,255,.25);
}

.smallmuted{ color: var(--muted); font-size: 12px; }

.hr { border-top: 1px solid var(--border); margin: 12px 0; }

/* Make primary buttons visibly blue */
div.stButton > button {
  width: 100%;
  border-radius: 14px;
  border: 1px solid rgba(0,123,167,.35);
  background: linear-gradient(135deg, var(--primary) 0%, var(--accent) 100%);
  color: white;
  font-weight: 800;
  padding: 0.7rem 1rem;
  box-shadow: 0 10px 24px rgba(0,123,167,.22);
}
div.stButton > button:hover {
  filter: brightness(1.03);
  transform: translateY(-1px);
  transition: 120ms ease;
}

/* Secondary button look (we'll use st.button but wrap in container class via markdown hint text) */
.secondary-note {
  border: 1px dashed rgba(100,116,139,.45);
  background: rgba(248,250,252,.85);
  border-radius: 14px;
  padding: 12px 12px;
}

/* Tighten default Streamlit element spacing a bit */
[data-testid="stCaptionContainer"] { margin-top: -4px; }
</style>
""",
    unsafe_allow_html=True,
)

# =========================
# HERO / BRAND BANNER
# =========================
st.markdown(
    """
<div class="cf-hero">
  <div class="badge">HOME</div>
  <div class="cf-brand">Chaouat Economics Lab</div>
  <div class="cf-sub">
    Two core areas:
    <b>Policy Lab</b> (interactive models with explicit assumptions) and
    <b>Teaching Material</b> (decks/handouts for tutoring & classroom use).
  </div>
</div>
""",
    unsafe_allow_html=True,
)

# =========================
# PRIMARY NAVIGATION (BIG CTAs)
# =========================
cta_left, cta_right = st.columns([1, 1], gap="large")

with cta_left:
    st.markdown('<div class="section">', unsafe_allow_html=True)
    st.subheader("Policy Lab")
    st.caption("Run interactive policy experiments. Export charts + CSV scenarios.")
    if st.button("Open Policy Lab", use_container_width=True, key="open_policy"):
        _switch_to(PAGES["Policy Lab"], "Opened Policy Lab", "Policy Lab")
    st.markdown('</div>', unsafe_allow_html=True)

with cta_right:
    st.markdown('<div class="section">', unsafe_allow_html=True)
    st.subheader("Teaching Material")
    st.caption("Browse slide decks, worksheets, and tutor-ready session structures.")
    if st.button("Open Teaching Material", use_container_width=True, key="open_teaching"):
        _switch_to(PAGES["Teaching Material"], "Opened Teaching Material", "Teaching Material")
    st.markdown('</div>', unsafe_allow_html=True)

# =========================
# WHAT TO DO NEXT (CLEARER THAN “SESSION PLAN / LIBRARY”)
# =========================
left, right = st.columns([1.25, 0.75], gap="large")

with left:
    st.markdown('<div class="section">', unsafe_allow_html=True)
    st.subheader("Quick start (recommended)")
    st.write(
        "Use this if you want a simple, predictable tutoring flow:\n\n"
        "1) Open **Policy Lab** and run one experiment (baseline + one shock).\n"
        "2) Export the CSV for notes or follow-up exercises.\n"
        "3) Open **Teaching Material** to assign a short deck/worksheet for reinforcement."
    )
    st.markdown('<div class="hr"></div>', unsafe_allow_html=True)

    st.markdown("### Choose a topic (optional)")
    topic = st.selectbox(
        "Topic focus",
        [
            "Monetary policy (Taylor rule)",
            "Fiscal policy (multipliers)",
            "Debt dynamics (sustainability)",
            "Growth & development (institutions)",
            "Trade (tariffs & incidence)",
        ],
        index=0,
        label_visibility="visible",
    )

    level = st.selectbox("Audience level", ["High school", "Undergraduate", "Advanced"], index=1)
    minutes = st.select_slider("Time available", options=[30, 45, 60, 75, 90], value=60)

    # Ultra-plain explanation: what this does
    st.caption("This simply pre-fills a suggested flow so you can teach consistently. It does not change any models.")

    flows = {
        "Monetary policy (Taylor rule)": [
            ("Warm-up", "Define inflation vs target; interpret output gap; why a rule exists."),
            ("Experiment", "Run baseline, then supply vs demand shock; compare φπ and smoothing."),
            ("Wrap", "Explain decomposition: base + inflation gap response + output gap response."),
        ],
        "Fiscal policy (multipliers)": [
            ("Warm-up", "Multiplier intuition: MPC, leakages, slack vs capacity."),
            ("Experiment", "Temporary spending shock under different MPC / openness assumptions."),
            ("Wrap", "State dependence: why multipliers differ across cycles."),
        ],
        "Debt dynamics (sustainability)": [
            ("Warm-up", "Debt identity; r−g; primary balance."),
            ("Experiment", "Simulate r>g vs r<g; add a growth shock; interpret path."),
            ("Wrap", "Sustainability vs liquidity; what levers matter."),
        ],
        "Growth & development (institutions)": [
            ("Warm-up", "Development beyond GDP; institutions; state capacity."),
            ("Experiment", "Mechanism walkthrough: poverty trap + measurement pitfalls."),
            ("Wrap", "Evidence discipline: identification + external validity."),
        ],
        "Trade (tariffs & incidence)": [
            ("Warm-up", "Surplus, incidence, deadweight loss."),
            ("Experiment", "Tariff incidence with elasticities; small vs large country."),
            ("Wrap", "Distributional effects and second-round impacts."),
        ],
    }

    st.markdown("### Suggested flow")
    plan = flows[topic]
    for i, (t, desc) in enumerate(plan, start=1):
        st.markdown(f"**{i}. {t}** — {desc}")

    st.markdown('</div>', unsafe_allow_html=True)

with right:
    st.markdown('<div class="section">', unsafe_allow_html=True)
    st.subheader("Recent")
    st.caption("Shows what you opened from the home page (local session only).")

    if len(st.session_state.home_recent) == 0:
        st.markdown('<div class="secondary-note">Nothing yet. Use the blue buttons above.</div>', unsafe_allow_html=True)
    else:
        for r in st.session_state.home_recent:
            st.markdown(f"**{r['label']}**  \n<span class='smallmuted'>{r['page']} · {r['ts']}</span>", unsafe_allow_html=True)
            st.markdown('<div class="hr"></div>', unsafe_allow_html=True)

        if st.button("Clear recent", use_container_width=True, key="clear_recent"):
            st.session_state.home_recent = []
            st.rerun()

    st.markdown('</div>', unsafe_allow_html=True)

# =========================
# FOOTER
# =========================
st.markdown("")
st.caption("© Chaouat Economics Lab · Built with Python/Streamlit · Educational use only")
