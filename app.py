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

# ----- Minimal, non-card dashboard styling -----
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Montserrat:wght@400;600;700&display=swap');

html, body, * { font-family: 'Montserrat', sans-serif !important; }

.block-container { padding-top: 0.8rem; padding-bottom: 2rem; max-width: 1200px; }

:root{
  --primary:#007BA7;
  --accent:#00B0FF;
  --text:#0F172A;
  --muted:#64748B;
  --border:#E2E8F0;
  --bg:#FFFFFF;
}

.topbar{
  border: 1px solid var(--border);
  border-radius: 16px;
  padding: 14px 14px;
  background: var(--bg);
}

.titleline{
  display:flex;
  align-items:baseline;
  justify-content:space-between;
  gap: 16px;
  margin-bottom: 10px;
}

.brand{
  font-size: 26px;
  font-weight: 800;
  color: var(--text);
}

.tag{
  font-size: 12px;
  color: var(--muted);
}

.section{
  margin-top: 14px;
  border: 1px solid var(--border);
  border-radius: 16px;
  padding: 14px 14px;
  background: var(--bg);
}

.step{
  border-left: 4px solid var(--primary);
  padding: 10px 12px;
  margin: 10px 0;
  background: #FBFDFF;
  border-radius: 10px;
}

.smallmuted{ color: var(--muted); font-size: 12px; }
hr { border: none; border-top: 1px solid var(--border); margin: 12px 0; }

kbd{
  border: 1px solid var(--border);
  border-bottom-width: 2px;
  padding: 2px 6px;
  border-radius: 8px;
  font-size: 12px;
  color: var(--text);
  background: #F8FAFC;
}
</style>
""", unsafe_allow_html=True)

# ---- Session-state for recent runs / bookmarks (home-only; additive) ----
if "home_recent" not in st.session_state:
    st.session_state.home_recent = []  # list of dicts: {"label":..., "page":..., "ts":...}

def _switch_to(page_path: str, label_for_recent: str, page_name: str):
    # Record "recent"
    st.session_state.home_recent.insert(0, {
        "label": label_for_recent,
        "page": page_name,
        "ts": datetime.now().strftime("%Y-%m-%d %H:%M"),
    })
    st.session_state.home_recent = st.session_state.home_recent[:8]

    # Navigate
    try:
        st.switch_page(page_path)
    except Exception:
        st.error("Navigation failed. Verify PAGES paths match your /pages filenames.")

# =========================
# TOP COMMAND BAR
# =========================
st.markdown("""
<div class="topbar">
  <div class="titleline">
    <div class="brand">Chaouat Economics Lab</div>
    <div class="tag">Press <kbd>Ctrl</kbd> + <kbd>R</kbd> to refresh after edits · Educational use only</div>
  </div>
</div>
""", unsafe_allow_html=True)

# Controls row (feels like a command bar, not a hero)
c1, c2, c3, c4, c5 = st.columns([1.4, 1.0, 0.8, 1.0, 0.9], gap="medium")
with c1:
    topic = st.selectbox(
        "Topic",
        [
            "Monetary policy — Taylor rule",
            "Fiscal policy — multipliers",
            "Debt dynamics — sustainability",
            "Growth & development — institutions",
            "Trade — tariffs & incidence",
        ],
        index=0,
        label_visibility="visible",
    )
with c2:
    level = st.selectbox("Level", ["High school", "Undergraduate", "Advanced"], index=1)
with c3:
    minutes = st.selectbox("Minutes", [30, 45, 60, 75, 90], index=2)
with c4:
    format_ = st.selectbox("Format", ["1:1 tutoring", "Small group", "Classroom"], index=0)
with c5:
    output = st.selectbox("Output", ["Session plan", "Plan + handout checklist"], index=0)

# =========================
# MAIN BODY: two columns
# =========================
left, right = st.columns([1.55, 1.0], gap="large")

TEMPLATES = {
    "Monetary policy — Taylor rule": [
        ("Warm-up (10 min)", "Define inflation vs target; output gap; why a rule exists."),
        ("Lab (25–35 min)", "Run 2 shocks (demand vs supply). Compare φπ and smoothing."),
        ("Synthesis (10 min)", "Decompose the implied rate into base + inflation gap + output gap."),
        ("Exit ticket (5 min)", "One question: when can the rule become destabilizing?"),
    ],
    "Fiscal policy — multipliers": [
        ("Warm-up (10 min)", "Multiplier intuition: MPC, leakages, slack vs capacity."),
        ("Lab (25–35 min)", "Temporary spending shock under different MPC / openness."),
        ("Synthesis (10 min)", "State-dependence: why multipliers differ across cycles."),
        ("Exit ticket (5 min)", "Who benefits first, and who pays later?"),
    ],
    "Debt dynamics — sustainability": [
        ("Warm-up (10 min)", "Debt/GDP identity; r−g; primary balance."),
        ("Lab (25–35 min)", "Simulate debt paths under r>g vs r<g; add a growth shock."),
        ("Synthesis (10 min)", "Interpret sustainability vs liquidity; what policy levers matter."),
        ("Exit ticket (5 min)", "What variable is most dangerous to mis-estimate?"),
    ],
    "Growth & development — institutions": [
        ("Warm-up (10 min)", "Development beyond GDP; institutions; state capacity."),
        ("Lab (25–35 min)", "Mechanism: poverty trap; intervention logic; measurement risks."),
        ("Synthesis (10 min)", "Evidence discipline: identification and external validity."),
        ("Exit ticket (5 min)", "What would falsify the intervention story?"),
    ],
    "Trade — tariffs & incidence": [
        ("Warm-up (10 min)", "Surplus, incidence, deadweight loss."),
        ("Lab (25–35 min)", "Tariff incidence with elasticities; small vs large country."),
        ("Synthesis (10 min)", "Distributional effects and second-round supply-chain impacts."),
        ("Exit ticket (5 min)", "Who pays: consumers, producers, or foreigners?"),
    ],
}

with left:
    st.markdown('<div class="section">', unsafe_allow_html=True)
    st.subheader("Session plan")
    st.caption(f"{minutes} minutes · {format_} · {level}")

    for title, body in TEMPLATES[topic]:
        st.markdown(f"""
        <div class="step">
          <div style="font-weight:700; color: var(--text);">{title}</div>
          <div class="smallmuted">{body}</div>
        </div>
        """, unsafe_allow_html=True)

    if output == "Plan + handout checklist":
        st.markdown("<hr/>", unsafe_allow_html=True)
        st.markdown("**Handout checklist**")
        st.write(
            "- Assumptions stated (units, baseline values)\n"
            "- One chart showing the mechanism\n"
            "- One sensitivity check\n"
            "- One takeaway sentence (student writes it)\n"
            "- Optional: CSV export for tutor notes"
        )

    st.markdown("<hr/>", unsafe_allow_html=True)

    b1, b2 = st.columns(2, gap="medium")
    with b1:
        if st.button("Open Policy Lab", use_container_width=True):
            _switch_to(PAGES["Policy Lab"], f"{topic} · {minutes}m", "Policy Lab")
    with b2:
        if st.button("Open Teaching Material", use_container_width=True):
            _switch_to(PAGES["Teaching Material"], f"{topic} · {minutes}m", "Teaching Material")

    st.markdown('</div>', unsafe_allow_html=True)

with right:
    # Library search (simple but useful)
    st.markdown('<div class="section">', unsafe_allow_html=True)
    st.subheader("Library")
    q = st.text_input("Search (e.g., “Taylor”, “debt”, “tariff”)", value="")
    filter_ = st.selectbox("Filter", ["All", "Policy Lab", "Teaching Material"], index=0)

    items = [
        {"title": "Taylor Rule Toy Model", "type": "Policy Lab", "hint": "Monetary policy mechanics; shocks; decomposition"},
        {"title": "Fiscal Multiplier Lab", "type": "Policy Lab", "hint": "MPC, leakages, state dependence"},
        {"title": "Debt Dynamics Lab", "type": "Policy Lab", "hint": "r−g, primary balance, stress tests"},
        {"title": "Development Econ Deck", "type": "Teaching Material", "hint": "Poverty, institutions, evidence"},
        {"title": "Trade & Tariffs Deck", "type": "Teaching Material", "hint": "Incidence, welfare, distribution"},
    ]

    q_low = q.strip().lower()
    shown = []
    for it in items:
        if filter_ != "All" and it["type"] != filter_:
            continue
        if q_low and (q_low not in it["title"].lower()) and (q_low not in it["hint"].lower()):
            continue
        shown.append(it)

    if not shown:
        st.caption("No results. Try a broader query.")
    else:
        for it in shown:
            st.markdown(f"**{it['title']}**  \n<span class='smallmuted'>{it['type']} · {it['hint']}</span>", unsafe_allow_html=True)
            st.markdown("---")

    st.markdown('</div>', unsafe_allow_html=True)

    # Recent activity
    st.markdown('<div class="section">', unsafe_allow_html=True)
    st.subheader("Recent")
    if len(st.session_state.home_recent) == 0:
        st.caption("Nothing yet. Launch a lab or deck from the left.")
    else:
        for r in st.session_state.home_recent:
            st.markdown(f"**{r['label']}**  \n<span class='smallmuted'>{r['page']} · {r['ts']}</span>", unsafe_allow_html=True)
            st.markdown("---")
        if st.button("Clear recent", use_container_width=True):
            st.session_state.home_recent = []
            st.rerun()
    st.markdown('</div>', unsafe_allow_html=True)

st.caption("© Chaouat Economics Lab · Built with Python/Streamlit · Educational use only")
