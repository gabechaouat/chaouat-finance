import streamlit as st
import os
import glob
from datetime import datetime

# =========================
# PAGE PATHS
# =========================
PAGES = {
    "Policy Lab":       "pages/Policy_Lab.py",
    "Teaching Material":"pages/Teaching_Material.py",
}

st.set_page_config(page_title="Chaouat Economics Lab", page_icon="📘", layout="wide")

if "home_recent" not in st.session_state:
    st.session_state.home_recent = []

def _switch_to(page_path, label, page_name):
    st.session_state.home_recent.insert(0, {
        "label": label, "page": page_name,
        "ts": datetime.now().strftime("%Y-%m-%d %H:%M"),
    })
    st.session_state.home_recent = st.session_state.home_recent[:8]
    try:
        st.switch_page(page_path)
    except Exception:
        st.error("Navigation failed — check PAGES paths.")

# =====================================================================
# STYLE  (same earth palette as Policy Lab)
# =====================================================================
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Serif+Display:ital@0;1&family=DM+Sans:wght@300;400;500;600&display=swap');

:root {
  --ink:         #1a1814;
  --ink-muted:   #6b6760;
  --ink-faint:   #b0ada8;
  --cream:       #faf8f4;
  --warm:        #f2ede4;
  --rule:        #e0dbd2;
  --terra:       #c9622a;
  --terra-light: #f7ece3;
  --terra-mid:   #a84e20;
  --sienna:      #8b3a1a;
  --sand:        #c4a882;
  --sand-dark:   #9e8060;
  --stone:       #7a6f62;
  --stone-light: #ece8e2;
}

html, body, * { font-family: 'DM Sans', sans-serif !important; }
.block-container { padding-top: 0 !important; padding-bottom: 3rem; max-width: 1200px; }

/* ── Masthead ── */
.masthead {
  border-bottom: 3px double var(--rule);
  padding: 30px 0 20px 0;
  text-align: center;
  margin-bottom: 0;
}
.masthead-eyebrow {
  font-size: 10px; letter-spacing: 3.5px; text-transform: uppercase;
  color: var(--ink-muted); margin-bottom: 10px;
}
.masthead-title {
  font-family: 'DM Serif Display', serif !important;
  font-size: 58px; line-height: 1.0; color: var(--ink);
  margin: 0 0 10px 0; letter-spacing: -0.5px;
}
.masthead-sub {
  font-size: 15px; color: var(--ink-muted); font-weight: 300;
  letter-spacing: 0.2px; margin-bottom: 16px;
}
.masthead-rule {
  display: flex; align-items: center; gap: 12px;
  justify-content: center; margin-top: 16px;
}
.masthead-rule-line { flex:1; max-width:110px; height:1px; background:var(--rule); }
.masthead-rule-diamond {
  width:7px; height:7px; background:var(--terra);
  transform:rotate(45deg); flex-shrink:0;
}

/* ── Section labels ── */
.section-label {
  font-size: 10px; letter-spacing: 3px; text-transform: uppercase;
  color: var(--ink-muted); border-top: 1px solid var(--rule);
  padding-top: 10px; margin: 30px 0 16px 0;
}

/* ── Nav cards ── */
.nav-card {
  border-radius: 4px; padding: 22px 20px 18px 20px;
  color: #fff; position: relative; overflow: hidden;
}
.nc-terra  { background: var(--terra); }
.nc-sienna { background: var(--sienna); }
.nc-stone  { background: var(--stone); }
.nc-ink    { background: var(--ink); }
.nav-card-tag {
  font-size: 10px; letter-spacing: 2.5px; text-transform: uppercase;
  opacity: 0.65; margin-bottom: 8px;
}
.nav-card-title {
  font-family: 'DM Serif Display', serif !important;
  font-size: 22px; margin: 0 0 6px 0; color: #fff;
}
.nav-card-desc { font-size: 13px; opacity: 0.75; line-height: 1.5; margin: 0; }

/* ── Story section ── */
.lede {
  font-family: 'DM Serif Display', serif !important;
  font-size: 25px; line-height: 1.45; color: var(--ink);
  border-left: 3px solid var(--terra); padding-left: 20px;
  margin: 0 0 22px 0;
}
.body-text {
  font-size: 15px; line-height: 1.78; color: var(--ink);
  font-weight: 300; margin: 0 0 13px 0;
}
.pull-quote {
  font-family: 'DM Serif Display', serif !important;
  font-style: italic; font-size: 19px; line-height: 1.5;
  color: var(--ink-muted);
  border-top: 1px solid var(--rule); border-bottom: 1px solid var(--rule);
  padding: 16px 0; margin: 24px 0;
}

/* ── Partner cards ── */
.partner-card {
  border: 1px solid var(--rule); border-radius: 4px;
  padding: 20px; background: var(--cream); height: 100%;
}
.partner-tag {
  font-size: 10px; letter-spacing: 2px; text-transform: uppercase;
  color: var(--ink-muted); margin-bottom: 7px;
}
.partner-name {
  font-family: 'DM Serif Display', serif !important;
  font-size: 19px; color: var(--ink); margin: 0 0 8px 0;
}
.partner-desc {
  font-size: 13.5px; color: var(--ink-muted);
  line-height: 1.6; margin: 0 0 14px 0;
}

/* ── Link buttons ── */
.link-btn {
  display: inline-block; border-radius: 3px;
  padding: 8px 16px; font-size: 11px; letter-spacing: 1.5px;
  text-transform: uppercase; font-weight: 500; text-decoration: none;
  border: 1px solid var(--rule); color: var(--ink);
  transition: border-color 120ms, color 120ms;
}
.link-btn:hover { border-color: var(--terra); color: var(--terra); text-decoration: none; }
.lb-terra { border-color: var(--terra); color: var(--terra); background: var(--terra-light); }
.lb-terra:hover { background: #f0ddd1; }
.lb-stone { border-color: var(--stone); color: var(--stone); background: var(--stone-light); }
.lb-stone:hover { background: #ddd9d4; }

/* ── Quick start ── */
.qs-step {
  border-left: 2px solid var(--terra); padding-left: 14px;
  margin-bottom: 14px;
}
.qs-step-title { font-weight: 500; font-size: 13.5px; color: var(--ink); margin: 0 0 3px 0; }
.qs-step-desc  { font-size: 13px; color: var(--ink-muted); line-height: 1.55; margin: 0; }
.qs-meta-box {
  background: var(--warm); border: 1px solid var(--rule);
  border-radius: 4px; padding: 20px;
}
.qs-meta-label {
  font-size: 10px; letter-spacing: 2.5px; text-transform: uppercase;
  color: var(--ink-muted); margin-bottom: 10px;
}
.qs-meta-row {
  display: flex; justify-content: space-between; align-items: baseline;
  padding: 7px 0; border-bottom: 1px solid var(--rule); font-size: 13.5px;
}
.qs-meta-row:last-child { border-bottom: none; }
.qs-meta-key   { color: var(--ink-muted); }
.qs-meta-value { font-weight: 500; color: var(--ink); }
.qs-tip {
  border-left: 3px solid var(--sand-dark);
  background: #f5eedd; padding: 12px 16px;
  border-radius: 0 4px 4px 0; margin-top: 14px;
  font-size: 13px; color: var(--ink-muted); line-height: 1.6;
}

/* ── Photo ── */
.photo-label {
  font-size: 10px; letter-spacing: 2.5px; text-transform: uppercase;
  color: var(--ink-muted); margin-bottom: 10px;
}
.photo-placeholder {
  background: var(--warm); border: 1px dashed var(--rule);
  border-radius: 4px; padding: 44px 16px;
  text-align: center; color: var(--ink-faint);
  font-size: 12px; line-height: 1.6;
}

/* ── Recent ── */
.recent-item {
  border-bottom: 1px solid var(--rule); padding: 9px 0;
  font-size: 13px; color: var(--ink);
}
.recent-ts { font-size: 11px; color: var(--ink-faint); margin-top: 2px; }

/* ── Buttons ── */
div.stButton > button {
  background: var(--ink) !important; color: #fff !important;
  border: none !important; border-radius: 3px !important;
  font-size: 11px !important; letter-spacing: 1.5px !important;
  text-transform: uppercase !important; font-weight: 500 !important;
  padding: 10px 16px !important; width: 100%;
}
div.stButton > button:hover { background: var(--terra) !important; }

/* ── Selectbox / slider labels ── */
[data-testid="stSelectbox"] label,
[data-testid="stSlider"] label { font-size: 12px !important; color: var(--ink-muted) !important; }

/* ── Footer ── */
.site-footer {
  border-top: 3px double var(--rule); padding-top: 16px; margin-top: 40px;
  font-size: 12px; color: var(--ink-muted); text-align: center; letter-spacing: 0.2px;
}
</style>
""", unsafe_allow_html=True)

# =====================================================================
# MASTHEAD
# =====================================================================
st.markdown("""
<div class="masthead">
  <div class="masthead-eyebrow">Open educational platform · Economics &amp; Policy</div>
  <div class="masthead-title">Chaouat Economics Lab</div>
  <div class="masthead-sub">Simulations, visual tools, and teaching materials made by educators, for educators.</div>
  <div class="masthead-rule">
    <div class="masthead-rule-line"></div>
    <div class="masthead-rule-diamond"></div>
    <div class="masthead-rule-line"></div>
  </div>
</div>
""", unsafe_allow_html=True)

# =====================================================================
# NAVIGATION CARDS
# =====================================================================
st.markdown('<div class="section-label">Explore the platform</div>', unsafe_allow_html=True)

nav1, nav2, nav3 = st.columns(3, gap="medium")

with nav1:
    st.markdown("""
    <div class="nav-card nc-terra">
      <div class="nav-card-tag">Module 01</div>
      <div class="nav-card-title">Policy Lab</div>
      <div class="nav-card-desc">Interactive monetary &amp; fiscal experiments. Adjust parameters, observe mechanisms, export charts and CSV scenarios.</div>
    </div>
    """, unsafe_allow_html=True)
    st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)
    if st.button("Open Policy Lab", key="open_policy"):
        _switch_to(PAGES["Policy Lab"], "Opened Policy Lab", "Policy Lab")

with nav2:
    st.markdown("""
    <div class="nav-card nc-sienna">
      <div class="nav-card-tag">Module 02</div>
      <div class="nav-card-title">Teaching Material</div>
      <div class="nav-card-desc">Slide decks, worksheets, and structured session guides — adapted for low-resource and low-bandwidth settings.</div>
    </div>
    """, unsafe_allow_html=True)
    st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)
    if st.button("Open Teaching Material", key="open_teach"):
        _switch_to(PAGES["Teaching Material"], "Opened Teaching Material", "Teaching Material")

with nav3:
    st.markdown("""
    <div class="nav-card nc-stone">
      <div class="nav-card-tag">Module 03</div>
      <div class="nav-card-title">Finance Tools</div>
      <div class="nav-card-desc">Live market data, investment simulator, financial news, and stock analysis dashboard.</div>
    </div>
    """, unsafe_allow_html=True)
    st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)
    if st.button("Open Finance Tools", key="open_finance"):
        st.info("Navigate to Finance Tools from the sidebar.")

# =====================================================================
# STORY  +  PHOTOS / RECENT
# =====================================================================
st.markdown('<div class="section-label">About the lab</div>', unsafe_allow_html=True)

left_col, right_col = st.columns([1.55, 1], gap="large")

with left_col:
    st.markdown("""
    <p class="lede">Built by educators, tested in classrooms, and shaped by feedback across three continents.</p>

    <p class="body-text">
      The Chaouat Economics Lab started in 2025 as a set of economics simulations and teaching tools,
      originally built while volunteering with tutors at
      <a href="https://bloglebilingue.wordpress.com/2020/04/28/koh-ed-a-detailed-view-of-the-organization/" target="_blank" style="color:#c9622a; text-decoration:none; font-weight:500;">Koh-Ed</a>.
      The goal was to make interactive economics education available beyond a single organization.
    </p>

    <p class="body-text">
      What started as a solo coding project expanded rather quickly. Koh-Ed tutors tested every module,
      flagged what didn't work, and helped shape the platform into something actually useful in a lesson.
      When we reached out to teachers in South Asia, they pushed us to rethink what "accessible" really means;
      the result was a set of activities and presentations that work with little or no internet access.
    </p>

    <div class="pull-quote">Collaboration is what made our work meaningful.</div>

    <p class="body-text">
      Alongside the teaching side, the lab has become a space to raise awareness for educational projects we believe in.
      In partnership with
      <a href="https://www.idealist.org/en/nonprofit/30eaaf27a8564a40a71faa66b6a8c02c-omelora-missouri-city" target="_blank" style="color:#c9622a; text-decoration:none; font-weight:500;">The Omelora Project</a>
      and Kyra Ezikeuzor, we helped fund books, blankets, and supplies for a Nigerian orphanage library.
      We are now fundraising for school resources for a 270-child orphanage in Uganda —
      led by Rabira Dosho, who heads our outreach.
    </p>
    """, unsafe_allow_html=True)

with right_col:
    # ── Pakistan photos ──
    st.markdown('<div class="photo-label">Impact in Pakistan</div>', unsafe_allow_html=True)

    PAKISTAN_PHOTOS = "images/pakistan"
    found = []
    for ext in ("jpg", "jpeg", "png", "webp"):
        found += glob.glob(f"{PAKISTAN_PHOTOS}/*.{ext}")
    found = sorted(found)

    if found:
        for p in found[:4]:
            st.image(p, use_container_width=True)
            st.markdown("<div style='height:6px'></div>", unsafe_allow_html=True)
    else:
        for label in ["Photo 1", "Photo 2"]:
            st.markdown(f"""
            <div class="photo-placeholder">
              {label}<br/><span style="font-size:11px;">Add images to <code>images/pakistan/</code></span>
            </div>
            """, unsafe_allow_html=True)
            st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)

    # ── Recent activity ──
    st.markdown('<div class="section-label" style="margin-top:20px;">Recent activity</div>', unsafe_allow_html=True)
    if not st.session_state.home_recent:
        st.markdown('<p style="font-size:13px; color:#b0ada8;">Nothing yet — use the buttons above.</p>', unsafe_allow_html=True)
    else:
        for r in st.session_state.home_recent:
            st.markdown(f"""
            <div class="recent-item">
              {r['label']}
              <div class="recent-ts">{r['page']} · {r['ts']}</div>
            </div>
            """, unsafe_allow_html=True)
        if st.button("Clear history", key="clear_recent"):
            st.session_state.home_recent = []
            st.rerun()

# =====================================================================
# PARTNERS
# =====================================================================
st.markdown('<div class="section-label">Partners &amp; collaborators</div>', unsafe_allow_html=True)

p1, p2, p3 = st.columns(3, gap="medium")

with p1:
    st.markdown("""
    <div class="partner-card">
      <div class="partner-tag">Education partner</div>
      <div class="partner-name">Koh-Ed</div>
      <div class="partner-desc">
        The tutoring organization where every module was first tested. Koh-Ed tutors gave the feedback
        that turned a prototype into a real teaching tool.
      </div>
    </div>
    """, unsafe_allow_html=True)
    st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)
    st.markdown("""
    <a class="link-btn lb-stone"
       href="https://bloglebilingue.wordpress.com/2020/04/28/koh-ed-a-detailed-view-of-the-organization/"
       target="_blank">Learn about Koh-Ed →</a>
    """, unsafe_allow_html=True)

with p2:
    st.markdown("""
    <div class="partner-card">
      <div class="partner-tag">Fundraising partner</div>
      <div class="partner-name">The Omelora Project</div>
      <div class="partner-desc">
        With Kyra Ezikeuzor and the Omelora Project, we've raised funds for orphanages in Nigeria and Uganda —
        books, blankets, and school supplies for hundreds of children.
      </div>
    </div>
    """, unsafe_allow_html=True)
    st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)
    st.markdown("""
    <a class="link-btn lb-terra"
       href="https://www.idealist.org/en/nonprofit/30eaaf27a8564a40a71faa66b6a8c02c-omelora-missouri-city"
       target="_blank">Visit Omelora →</a>
    """, unsafe_allow_html=True)

with p3:
    st.markdown("""
    <div class="partner-card">
      <div class="partner-tag">Active campaign</div>
      <div class="partner-name">Uganda — 270 children</div>
      <div class="partner-desc">
        We are currently fundraising for school resources for a 270-child orphanage in Uganda,
        led by Rabira Dosho, who heads the platform's outreach.
      </div>
    </div>
    """, unsafe_allow_html=True)
    st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)
    st.markdown("""
    <a class="link-btn lb-terra"
       href="https://www.idealist.org/en/nonprofit/30eaaf27a8564a40a71faa66b6a8c02c-omelora-missouri-city"
       target="_blank">Support the campaign →</a>
    """, unsafe_allow_html=True)

# =====================================================================
# QUICK START FOR TUTORS  (redesigned)
# =====================================================================
st.markdown('<div class="section-label">Quick start for tutors</div>', unsafe_allow_html=True)

qs_left, qs_right = st.columns([1.5, 1], gap="large")

# ── Data ──
FLOWS = {
    "Monetary policy (Taylor rule)": {
        "module": "Policy Lab → Module 01",
        "prereqs": "Inflation, interest rates, output gap",
        "steps": [
            ("Concept check",  "Ask students: what should a central bank do when inflation is above target? Establish intuition before opening the model."),
            ("Baseline run",   "Set parameters to the current macro environment. Read off the implied rate. Ask: does this match what the real central bank is doing?"),
            ("Shock scenario", "Apply a supply shock. Watch how the policy path changes. Compare φπ = 1.5 vs φπ = 0.5 — which central bank reacts more aggressively?"),
            ("Rate smoothing", "Toggle smoothing between 0 and 0.8. Discuss why central banks prefer gradual adjustment over jumping to the implied rate."),
            ("Wrap-up",        "Use the decomposition chart: break the rate into base + inflation gap + output gap. Each component maps to a lecture concept."),
        ],
        "tip": "Export the CSV and ask students to replicate the chart in a spreadsheet — good for reinforcing the formula.",
    },
    "Fiscal policy (multipliers)": {
        "module": "Policy Lab → Module 02",
        "prereqs": "MPC, aggregate demand, Keynesian cross",
        "steps": [
            ("Concept check",  "What happens to GDP when the government spends €1 more? Collect answers before touching the model."),
            ("Base multiplier","Set MPC = 0.8, no openness, no crowding. Compute the multiplier mentally first, then verify with the model."),
            ("Add leakages",   "Increase import propensity from 0 to 0.3. Watch the multiplier surface shift. Which leakage matters most?"),
            ("State dependence","Switch between 'below potential' and 'above potential'. Discuss why the same spending has very different effects."),
            ("Wrap-up",        "The leakage waterfall chart is ideal for showing where each pound of stimulus goes. Walk through it step by step."),
        ],
        "tip": "The 3D surface is best used after students understand the basic formula — it shows all parameter combinations at once.",
    },
    "Debt dynamics (r − g)": {
        "module": "Policy Lab → Module 03",
        "prereqs": "Government budget identity, compound growth",
        "steps": [
            ("Concept check",  "When does government debt spiral? Ask students to guess before running any numbers."),
            ("r vs g",         "Set r = 4%, g = 3%. Observe baseline trajectory. Then flip to r = 2%, g = 4%. The debt path looks very different."),
            ("Fiscal shock",   "Add a primary balance deterioration in year 3. How many years until debt stabilises again?"),
            ("Sustainability", "Use the contour map to find the primary balance that keeps debt flat. Compare to current real-world figures."),
            ("Wrap-up",        "The fan chart around the baseline is a natural way to introduce uncertainty — small changes in g compound over 15 years."),
        ],
        "tip": "Ask students to find the 'point of no return' on the sustainability map — the combination where debt explodes regardless of adjustment.",
    },
    "Trade (tariffs & incidence)": {
        "module": "Policy Lab → Module 04",
        "prereqs": "Consumer/producer surplus, elasticity",
        "steps": [
            ("Concept check",  "Who really pays a tariff — the importer or the consumer? Take a poll first."),
            ("Small country",  "Set foreign elasticity high. Show that domestic consumers bear nearly all the cost. Walk through the welfare rectangles."),
            ("Large country",  "Reduce foreign elasticity. Watch the exporter burden grow. Introduce the optimal tariff argument."),
            ("Welfare map",    "Explore the contour map. Under what conditions is a tariff welfare-positive? When is it always negative?"),
            ("Wrap-up",        "The incidence bar chart directly answers the opening poll. Useful for making the maths concrete."),
        ],
        "tip": "Pair this with a real example — US steel tariffs 2018 or EU carbon border adjustments — and ask students to estimate elasticities.",
    },
}

level_notes = {
    "High school":    "Keep the scenario simple. Skip the 3D surface. Focus on the main chart and the result cards.",
    "Undergraduate":  "All charts are appropriate. Use the decomposition and sensitivity maps for discussion.",
    "Advanced":       "Push on the model's assumptions. What is missing? Where would a real model diverge?",
}

time_notes = {
    30: "Run one scenario only. Concept check + one experiment + result cards.",
    45: "Concept check + two scenarios + one sensitivity chart.",
    60: "Full flow: all five steps. Leave 10 minutes for the wrap-up discussion.",
    75: "Full flow + export CSV + one extension question.",
    90: "Full flow + independent exploration + short written response.",
}

with qs_left:
    st.markdown('<p style="font-size:14px; color:#6b6760; margin-bottom:18px;">Select a topic and session parameters to get a step-by-step teaching plan.</p>', unsafe_allow_html=True)

    topic   = st.selectbox("Topic", list(FLOWS.keys()), label_visibility="collapsed", key="qs_topic")
    col_lv, col_tm = st.columns(2, gap="medium")
    with col_lv:
        level   = st.selectbox("Audience level", ["High school", "Undergraduate", "Advanced"], index=1, key="qs_level")
    with col_tm:
        minutes = st.select_slider("Session length", options=[30, 45, 60, 75, 90], value=60, key="qs_mins")

    flow = FLOWS[topic]
    st.markdown("<div style='height:6px'></div>", unsafe_allow_html=True)

    for i, (title, desc) in enumerate(flow["steps"], start=1):
        st.markdown(f"""
        <div class="qs-step">
          <div class="qs-step-title">{i}. {title}</div>
          <div class="qs-step-desc">{desc}</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown(f"""
    <div class="qs-tip">
      <strong style="color:#9e8060;">Tutor tip —</strong> {flow['tip']}
    </div>
    """, unsafe_allow_html=True)

with qs_right:
    st.markdown(f"""
    <div class="qs-meta-box">
      <div class="qs-meta-label">Session overview</div>
      <div class="qs-meta-row">
        <span class="qs-meta-key">Module</span>
        <span class="qs-meta-value">{flow['module']}</span>
      </div>
      <div class="qs-meta-row">
        <span class="qs-meta-key">Level</span>
        <span class="qs-meta-value">{level}</span>
      </div>
      <div class="qs-meta-row">
        <span class="qs-meta-key">Duration</span>
        <span class="qs-meta-value">{minutes} min</span>
      </div>
      <div class="qs-meta-row">
        <span class="qs-meta-key">Prerequisites</span>
        <span class="qs-meta-value" style="text-align:right; max-width:60%;">{flow['prereqs']}</span>
      </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown(f"""
    <div style="margin-top:14px; background:var(--warm); border:1px solid var(--rule);
                border-radius:4px; padding:18px;">
      <div class="qs-meta-label">For {level.lower()} students</div>
      <p style="font-size:13.5px; color:#1a1814; line-height:1.65; margin:0;">
        {level_notes[level]}
      </p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown(f"""
    <div style="margin-top:14px; background:var(--terra-light); border-left:3px solid var(--terra);
                border-radius:0 4px 4px 0; padding:14px 16px;">
      <div class="qs-meta-label">In {minutes} minutes</div>
      <p style="font-size:13.5px; color:#6b6760; line-height:1.65; margin:0;">
        {time_notes[minutes]}
      </p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div style="margin-top:14px; background:var(--stone-light); border:1px solid var(--rule);
                border-radius:4px; padding:18px;">
      <div class="qs-meta-label">Platform design principle</div>
      <p style="font-size:13px; color:#6b6760; line-height:1.65; margin:0;">
        Every module was tested by a real tutor before being made available.
        If it didn't work in a classroom, it didn't ship.
        Core activities function with limited internet access.
      </p>
    </div>
    """, unsafe_allow_html=True)

# =====================================================================
# FOOTER
# =====================================================================
st.markdown("""
<div class="site-footer">
  © Chaouat Economics Lab · Built with Python / Streamlit · Educational use only<br/>
  <span style="font-size:11px;">
    In partnership with
    <a href="https://bloglebilingue.wordpress.com/2020/04/28/koh-ed-a-detailed-view-of-the-organization/"
       target="_blank" style="color:#9e8060; text-decoration:none;">Koh-Ed</a>
    &amp;
    <a href="https://www.idealist.org/en/nonprofit/30eaaf27a8564a40a71faa66b6a8c02c-omelora-missouri-city"
       target="_blank" style="color:#c9622a; text-decoration:none;">The Omelora Project</a>
  </span>
</div>
""", unsafe_allow_html=True)
