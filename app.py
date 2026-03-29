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

if "home_recent" not in st.session_state:
    st.session_state.home_recent = []

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
}

html, body, * { font-family: 'DM Sans', sans-serif !important; }

.block-container { padding-top: 0 !important; padding-bottom: 3rem; max-width: 1180px; }

/* ---- masthead ---- */
.masthead {
  border-bottom: 3px double var(--rule);
  padding: 28px 0 18px 0;
  margin-bottom: 0;
  text-align: center;
}
.masthead-eyebrow {
  font-size: 11px;
  letter-spacing: 3px;
  text-transform: uppercase;
  color: var(--ink-muted);
  margin-bottom: 10px;
}
.masthead-title {
  font-family: 'DM Serif Display', serif !important;
  font-size: 56px;
  line-height: 1.0;
  color: var(--ink);
  margin: 0 0 10px 0;
  letter-spacing: -0.5px;
}
.masthead-sub {
  font-size: 15px;
  color: var(--ink-muted);
  font-weight: 300;
  letter-spacing: 0.3px;
  margin-bottom: 14px;
}
.masthead-rule {
  display: flex;
  align-items: center;
  gap: 12px;
  justify-content: center;
  margin-top: 14px;
}
.masthead-rule-line { flex: 1; max-width: 120px; height: 1px; background: var(--rule); }
.masthead-rule-diamond { width: 8px; height: 8px; background: var(--accent); transform: rotate(45deg); flex-shrink: 0; }

/* ---- section rule ---- */
.section-label {
  font-size: 10px;
  letter-spacing: 3px;
  text-transform: uppercase;
  color: var(--ink-muted);
  border-top: 1px solid var(--rule);
  padding-top: 10px;
  margin: 32px 0 18px 0;
}

/* ---- lede / mission block ---- */
.lede {
  font-family: 'DM Serif Display', serif !important;
  font-size: 26px;
  line-height: 1.45;
  color: var(--ink);
  border-left: 3px solid var(--accent);
  padding-left: 22px;
  margin: 0 0 24px 0;
}
.body-text {
  font-size: 15.5px;
  line-height: 1.75;
  color: var(--ink);
  font-weight: 300;
  margin: 0 0 14px 0;
}

/* ---- nav cards ---- */
.nav-card {
  background: var(--ink);
  color: #fff;
  border-radius: 4px;
  padding: 22px 22px 18px 22px;
  position: relative;
  overflow: hidden;
}
.nav-card-accent { background: var(--accent); }
.nav-card-teal { background: var(--teal); }
.nav-card-tag {
  font-size: 10px;
  letter-spacing: 2.5px;
  text-transform: uppercase;
  opacity: 0.6;
  margin-bottom: 8px;
}
.nav-card-title {
  font-family: 'DM Serif Display', serif !important;
  font-size: 22px;
  margin: 0 0 6px 0;
  color: #fff;
}
.nav-card-desc {
  font-size: 13px;
  opacity: 0.75;
  line-height: 1.5;
  margin: 0;
}

/* ---- partner cards ---- */
.partner-card {
  border: 1px solid var(--rule);
  border-radius: 4px;
  padding: 20px;
  background: var(--cream);
  height: 100%;
}
.partner-tag {
  font-size: 10px;
  letter-spacing: 2px;
  text-transform: uppercase;
  color: var(--ink-muted);
  margin-bottom: 8px;
}
.partner-name {
  font-family: 'DM Serif Display', serif !important;
  font-size: 20px;
  color: var(--ink);
  margin: 0 0 8px 0;
}
.partner-desc {
  font-size: 13.5px;
  color: var(--ink-muted);
  line-height: 1.6;
  margin: 0 0 14px 0;
}

/* ---- stat row ---- */
.stat-block {
  border-top: 1px solid var(--rule);
  padding-top: 14px;
  text-align: center;
}
.stat-number {
  font-family: 'DM Serif Display', serif !important;
  font-size: 36px;
  color: var(--accent);
  line-height: 1;
  margin: 0;
}
.stat-label {
  font-size: 12px;
  color: var(--ink-muted);
  text-transform: uppercase;
  letter-spacing: 1.5px;
  margin-top: 4px;
}

/* ---- photo strip ---- */
.photo-strip-label {
  font-size: 11px;
  letter-spacing: 2px;
  text-transform: uppercase;
  color: var(--ink-muted);
  margin-bottom: 10px;
}
.photo-placeholder {
  background: var(--warm);
  border: 1px dashed var(--rule);
  border-radius: 4px;
  padding: 48px 16px;
  text-align: center;
  color: var(--ink-faint);
  font-size: 12px;
  line-height: 1.6;
}

/* ---- blockquote ---- */
.pull-quote {
  font-family: 'DM Serif Display', serif !important;
  font-style: italic;
  font-size: 21px;
  line-height: 1.5;
  color: var(--ink-muted);
  border-top: 1px solid var(--rule);
  border-bottom: 1px solid var(--rule);
  padding: 18px 0;
  margin: 28px 0;
}

/* ---- buttons ---- */
div.stButton > button {
  background: var(--ink) !important;
  color: #fff !important;
  border: none !important;
  border-radius: 3px !important;
  font-size: 12px !important;
  letter-spacing: 1.5px !important;
  text-transform: uppercase !important;
  font-weight: 500 !important;
  padding: 10px 18px !important;
  width: 100%;
}
div.stButton > button:hover {
  background: var(--accent) !important;
  transition: background 150ms ease;
}

/* ---- link button ---- */
.link-btn {
  display: inline-block;
  background: transparent;
  border: 1px solid var(--rule);
  border-radius: 3px;
  padding: 8px 16px;
  font-size: 12px;
  letter-spacing: 1.5px;
  text-transform: uppercase;
  color: var(--ink);
  text-decoration: none;
  font-weight: 500;
  transition: border-color 150ms, color 150ms;
}
.link-btn:hover { border-color: var(--accent); color: var(--accent); text-decoration: none; }
.link-btn-accent {
  background: var(--accent-light);
  border-color: var(--accent);
  color: var(--accent);
}
.link-btn-teal {
  background: var(--teal-light);
  border-color: var(--teal);
  color: var(--teal);
}

/* ---- footer ---- */
.site-footer {
  border-top: 3px double var(--rule);
  padding-top: 16px;
  margin-top: 40px;
  font-size: 12px;
  color: var(--ink-muted);
  text-align: center;
  letter-spacing: 0.3px;
}

/* ---- recent item ---- */
.recent-item {
  border-bottom: 1px solid var(--rule);
  padding: 10px 0;
  font-size: 13px;
  color: var(--ink);
}
.recent-ts { font-size: 11px; color: var(--ink-faint); margin-top: 2px; }
</style>
""", unsafe_allow_html=True)


# =====================
# MASTHEAD
# =====================
st.markdown("""
<div class="masthead">
  <div class="masthead-eyebrow">An open educational platform</div>
  <div class="masthead-title">Chaouat Economics Lab</div>
  <div class="masthead-sub">Lessons, simulations, and visual tools — built with educators, for educators.</div>
  <div class="masthead-rule">
    <div class="masthead-rule-line"></div>
    <div class="masthead-rule-diamond"></div>
    <div class="masthead-rule-line"></div>
  </div>
</div>
""", unsafe_allow_html=True)


# =====================
# NAVIGATION CARDS
# =====================
st.markdown('<div class="section-label">Explore the platform</div>', unsafe_allow_html=True)

col_pol, col_teach, col_fin = st.columns([1, 1, 1], gap="medium")

with col_pol:
    st.markdown("""
    <div class="nav-card nav-card-accent">
      <div class="nav-card-tag">Module 01</div>
      <div class="nav-card-title">Policy Lab</div>
      <div class="nav-card-desc">Run interactive monetary & fiscal experiments. Export charts and CSV scenarios for classroom use.</div>
    </div>
    """, unsafe_allow_html=True)
    st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)
    if st.button("Open Policy Lab", key="open_policy"):
        _switch_to(PAGES["Policy Lab"], "Opened Policy Lab", "Policy Lab")

with col_teach:
    st.markdown("""
    <div class="nav-card nav-card-teal">
      <div class="nav-card-tag">Module 02</div>
      <div class="nav-card-title">Teaching Material</div>
      <div class="nav-card-desc">Slide decks, worksheets, and tutor-ready session structures adapted for low-resource settings.</div>
    </div>
    """, unsafe_allow_html=True)
    st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)
    if st.button("Open Teaching Material", key="open_teaching"):
        _switch_to(PAGES["Teaching Material"], "Opened Teaching Material", "Teaching Material")

with col_fin:
    st.markdown("""
    <div class="nav-card" style="background:#2c2c2a;">
      <div class="nav-card-tag">Module 03</div>
      <div class="nav-card-title">Finance Tools</div>
      <div class="nav-card-desc">Live market data, investment simulator, financial news, and stock analysis dashboard.</div>
    </div>
    """, unsafe_allow_html=True)
    st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)
    if st.button("Open Finance Tools", key="open_finance"):
        st.info("Navigate to Finance Tools from the sidebar.")


# =====================
# IMPACT STATS
# =====================
st.markdown('<div class="section-label">Our reach</div>', unsafe_allow_html=True)

s1, s2, s3, s4 = st.columns(4, gap="medium")
for col, number, label in [
    (s1, "2+", "years building"),
    (s2, "Dozens", "educators using the platform"),
    (s3, "$1000s", "raised for education"),
    (s4, "3", "countries reached"),
]:
    with col:
        st.markdown(f"""
        <div class="stat-block">
          <div class="stat-number">{number}</div>
          <div class="stat-label">{label}</div>
        </div>
        """, unsafe_allow_html=True)


# =====================
# ORIGIN STORY
# =====================
st.markdown('<div class="section-label">Our story</div>', unsafe_allow_html=True)

left_story, right_sidebar = st.columns([1.5, 1], gap="large")

with left_story:
    st.markdown("""
    <p class="lede">A coding project that became a shared teaching resource — built through collaboration across three continents.</p>

    <p class="body-text">In October 2025, the Chaouat Economics Lab began as a platform of lessons, simulations, and visual tools in economics — designed while volunteering with tutors at <strong>Koh-Ed</strong>. The goal was simple: make those resources accessible beyond one organization.</p>

    <p class="body-text">The project quickly became collaborative. Koh-Ed tutors tested modules, gave feedback, and once the issues were fixed, dozens used the platform in their lessons. To reach classrooms where economics is not usually taught, we then contacted teachers in South Asia, who helped us adapt the platform for <strong>low-resource settings</strong> — mainly activities that work with limited internet access.</p>

    <div class="pull-quote">"Without these educators, the platform would have remained a coding project, rather than a shared teaching resource."</div>

    <p class="body-text">As the platform grew, we sought to fundraise for educational projects by advertising them directly on the platform. We worked with <strong>Kyra Ezikeuzor</strong> through the Omelora Project to raise funds for books, blankets, and supplies for an orphanage in Nigeria to help launch a library. We are currently fundraising for school resources for a <strong>270-child orphanage in Uganda</strong>.</p>

    <p class="body-text">Alongside <strong>Rabira Dosho</strong>, who leads the platform's outreach, we have raised thousands of dollars toward this effort. The Chaouat Economics Lab couldn't have had a meaningful impact without the help of these wonderful people.</p>
    """, unsafe_allow_html=True)

with right_sidebar:

    # ---- Pakistan impact photos ----
    st.markdown('<div class="photo-strip-label">📍 Impact in Pakistan</div>', unsafe_allow_html=True)

    PAKISTAN_PHOTOS = "images/pakistan"
    import os, glob

    found = []
    for ext in ("jpg", "jpeg", "png", "webp"):
        found += glob.glob(f"{PAKISTAN_PHOTOS}/*.{ext}")
    found = sorted(found)

    if found:
        for p in found[:4]:
            st.image(p, use_container_width=True)
    else:
        st.markdown("""
        <div class="photo-placeholder">
          Add images to <code>images/pakistan/</code><br/>to display impact photos here.<br/><br/>
          Supported: jpg, jpeg, png, webp
        </div>
        """, unsafe_allow_html=True)
        st.markdown("<div style='height:10px'></div>", unsafe_allow_html=True)
        st.markdown("""
        <div class="photo-placeholder">
          Photo 2
        </div>
        """, unsafe_allow_html=True)
        st.markdown("<div style='height:10px'></div>", unsafe_allow_html=True)
        st.markdown("""
        <div class="photo-placeholder">
          Photo 3
        </div>
        """, unsafe_allow_html=True)

    # ---- Recent activity ----
    st.markdown('<div class="section-label" style="margin-top:28px;">Recent activity</div>', unsafe_allow_html=True)
    if not st.session_state.home_recent:
        st.markdown('<p style="font-size:13px;color:#b0ada8;">Nothing yet — use the buttons above to navigate.</p>', unsafe_allow_html=True)
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


# =====================
# PARTNER ORGANISATIONS
# =====================
st.markdown('<div class="section-label">Partners & collaborators</div>', unsafe_allow_html=True)

p1, p2, p3 = st.columns(3, gap="medium")

with p1:
    st.markdown("""
    <div class="partner-card">
      <div class="partner-tag">Education partner</div>
      <div class="partner-name">Koh-Ed</div>
      <div class="partner-desc">The tutoring organization where the platform was first built and tested. Koh-Ed tutors shaped every iteration of the teaching modules through direct classroom feedback.</div>
    </div>
    """, unsafe_allow_html=True)
    st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)
    st.markdown('<a class="link-btn link-btn-teal" href="https://www.koh-ed.org" target="_blank">Visit Koh-Ed →</a>', unsafe_allow_html=True)

with p2:
    st.markdown("""
    <div class="partner-card">
      <div class="partner-tag">Fundraising partner</div>
      <div class="partner-name">The Omelora Project</div>
      <div class="partner-desc">Working with Kyra Ezikeuzor and the Omelora Project, we've raised funds for orphanages in Nigeria and Uganda — books, blankets, and school supplies for hundreds of children.</div>
    </div>
    """, unsafe_allow_html=True)
    st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)
    st.markdown('<a class="link-btn link-btn-accent" href="https://www.omelora.org" target="_blank">Visit Omelora →</a>', unsafe_allow_html=True)

with p3:
    st.markdown("""
    <div class="partner-card">
      <div class="partner-tag">Current campaign</div>
      <div class="partner-name">Uganda Orphanage</div>
      <div class="partner-desc">We are currently fundraising for school resources for a 270-child orphanage in Uganda. Led by Rabira Dosho, who heads the platform's outreach efforts.</div>
    </div>
    """, unsafe_allow_html=True)
    st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)
    st.markdown('<a class="link-btn link-btn-accent" href="https://www.omelora.org" target="_blank">Support the campaign →</a>', unsafe_allow_html=True)


# =====================
# QUICK START
# =====================
st.markdown('<div class="section-label">Quick start for tutors</div>', unsafe_allow_html=True)

qs_left, qs_right = st.columns([1.5, 1], gap="large")

with qs_left:
    st.markdown("""
    <p class="body-text">Use this flow for a consistent tutoring session:</p>
    """, unsafe_allow_html=True)

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

    topic = st.selectbox("Topic", list(flows.keys()), label_visibility="collapsed")
    level = st.selectbox("Level", ["High school", "Undergraduate", "Advanced"], index=1)
    minutes = st.select_slider("Time", options=[30, 45, 60, 75, 90], value=60)

    plan = flows[topic]
    for i, (t, desc) in enumerate(plan, start=1):
        st.markdown(f"""
        <div style="border-left:2px solid #c9622a; padding-left:14px; margin-bottom:12px;">
          <div style="font-weight:500; font-size:14px; color:#1a1814;">{i}. {t}</div>
          <div style="font-size:13px; color:#6b6760; margin-top:3px;">{desc}</div>
        </div>
        """, unsafe_allow_html=True)

with qs_right:
    st.markdown("""
    <div style="background:#f2ede4; border:1px solid #e0dbd2; border-radius:4px; padding:22px; margin-top:22px;">
      <div style="font-size:10px; letter-spacing:2.5px; text-transform:uppercase; color:#6b6760; margin-bottom:12px;">Platform philosophy</div>
      <p style="font-size:14px; line-height:1.7; color:#1a1814; margin:0 0 12px 0;">
        Every module was tested by a real tutor before being made public. If something doesn't work in a classroom, it doesn't ship.
      </p>
      <p style="font-size:14px; line-height:1.7; color:#1a1814; margin:0;">
        The platform is designed to work in low-bandwidth environments. Core teaching tools require no live internet connection.
      </p>
    </div>
    """, unsafe_allow_html=True)


# =====================
# FOOTER
# =====================
st.markdown("""
<div class="site-footer">
  © Chaouat Economics Lab · Built with Python / Streamlit · Educational use only<br/>
  <span style="font-size:11px;">In partnership with Koh-Ed &amp; The Omelora Project</span>
</div>
""", unsafe_allow_html=True)
