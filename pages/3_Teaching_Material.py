import os
import glob
import base64
import streamlit as st

st.set_page_config(
    page_title="Teaching Material — Chaouat Economics Lab",
    page_icon="📑",
    layout="wide",
)

# =====================================================================
# HELPERS
# =====================================================================
MATERIALS_DIR = "materials"
COVERS_DIR    = os.path.join(MATERIALS_DIR, "covers")

@st.cache_data(ttl=12*60*60, max_entries=60, show_spinner=False)
def read_file_bytes(path: str) -> bytes:
    with open(path, "rb") as f:
        return f.read()

@st.cache_data(ttl=12*60*60, show_spinner=False)
def read_cover_b64(path: str) -> str:
    """Return base64-encoded image string for a cover, or '' if missing."""
    for ext in ("png", "jpg", "jpeg", "webp"):
        for candidate in [
            os.path.join(COVERS_DIR, os.path.splitext(os.path.basename(path))[0] + f".{ext}"),
        ]:
            if os.path.exists(candidate):
                with open(candidate, "rb") as f:
                    raw = f.read()
                mime = "image/jpeg" if ext in ("jpg","jpeg") else f"image/{ext}"
                return f"data:{mime};base64,{base64.b64encode(raw).decode()}"
    return ""

def list_pdfs() -> list[str]:
    if not os.path.exists(MATERIALS_DIR):
        return []
    return sorted(glob.glob(os.path.join(MATERIALS_DIR, "*.pdf")))

def pretty_title(path: str) -> str:
    return os.path.basename(path).replace(".pdf","").replace("_"," ").replace("-"," ").title()

def file_size_str(path: str) -> str:
    kb = os.path.getsize(path) / 1024
    return f"{kb:.0f} KB" if kb < 1024 else f"{kb/1024:.1f} MB"

# Topic tag inference (simple keyword match on filename)
TOPIC_TAGS = {
    "monetary":  ("Monetary Policy",  "#c9622a"),
    "taylor":    ("Monetary Policy",  "#c9622a"),
    "inflation": ("Monetary Policy",  "#c9622a"),
    "fiscal":    ("Fiscal Policy",    "#8b3a1a"),
    "multiplier":("Fiscal Policy",    "#8b3a1a"),
    "deficit":   ("Fiscal Policy",    "#8b3a1a"),
    "debt":      ("Debt Dynamics",    "#9e8060"),
    "sustainab": ("Debt Dynamics",    "#9e8060"),
    "trade":     ("Trade",            "#7a6f62"),
    "tariff":    ("Trade",            "#7a6f62"),
    "growth":    ("Development",      "#5a4a3a"),
    "develop":   ("Development",      "#5a4a3a"),
    "micro":     ("Microeconomics",   "#b08050"),
    "macro":     ("Macroeconomics",   "#8b3a1a"),
    "statistic": ("Statistics",       "#7a6f62"),
    "econometr": ("Econometrics",     "#7a6f62"),
    "finance":   ("Finance",          "#c9622a"),
}

def infer_tag(filename: str):
    fn = filename.lower()
    for kw, (label, color) in TOPIC_TAGS.items():
        if kw in fn:
            return label, color
    return "Economics", "#b0ada8"

# =====================================================================
# STYLE
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
.block-container { padding-top: 0 !important; padding-bottom: 3rem; max-width: 1220px; }

/* ── Masthead ── */
.tm-masthead {
  border-bottom: 3px double var(--rule);
  padding: 28px 0 18px 0; margin-bottom: 0;
}
.tm-eyebrow {
  font-size: 10px; letter-spacing: 3.5px; text-transform: uppercase;
  color: var(--ink-muted); margin-bottom: 8px;
}
.tm-title {
  font-family: 'DM Serif Display', serif !important;
  font-size: 52px; line-height: 1.0; color: var(--ink);
  margin: 0 0 8px 0; letter-spacing: -0.5px;
}
.tm-sub { font-size: 15px; color: var(--ink-muted); font-weight: 300; }

/* ── Section labels ── */
.section-label {
  font-size: 10px; letter-spacing: 3px; text-transform: uppercase;
  color: var(--ink-muted); border-top: 1px solid var(--rule);
  padding-top: 10px; margin: 28px 0 16px 0;
}

/* ── Search bar ── */
.search-note {
  font-size: 12px; color: var(--ink-faint); margin-top: -10px; margin-bottom: 16px;
}

/* ── Deck card ── */
.deck-card {
  background: var(--cream);
  border: 1px solid var(--rule);
  border-radius: 4px;
  overflow: hidden;
  display: flex;
  flex-direction: column;
  height: 100%;
  transition: border-color 150ms;
}
.deck-card:hover { border-color: var(--sand); }

/* Cover image area */
.deck-cover {
  width: 100%;
  aspect-ratio: 4/3;
  background: var(--warm);
  display: flex;
  align-items: center;
  justify-content: center;
  overflow: hidden;
  position: relative;
}
.deck-cover img {
  width: 100%; height: 100%; object-fit: cover;
  display: block;
}
.deck-cover-placeholder {
  width: 100%; height: 100%;
  display: flex; flex-direction: column;
  align-items: center; justify-content: center;
  background: var(--warm);
  gap: 8px;
}
.deck-cover-icon {
  width: 40px; height: 52px;
  border: 2px solid var(--rule); border-radius: 3px;
  background: var(--cream); position: relative;
}
.deck-cover-icon::after {
  content: '';
  position: absolute; top: 0; right: 0;
  width: 0; height: 0;
  border-left: 12px solid var(--warm);
  border-bottom: 12px solid var(--rule);
}
.deck-cover-lines {
  display: flex; flex-direction: column; gap: 5px; width: 26px;
  position: absolute;
  top: 50%; left: 50%; transform: translate(-50%, -50%);
}
.deck-cover-line {
  height: 2px; background: var(--rule); border-radius: 1px;
}

/* Body */
.deck-body {
  padding: 14px 16px 16px 16px;
  display: flex; flex-direction: column; flex: 1;
}
.deck-tag {
  display: inline-block; font-size: 10px; font-weight: 600;
  letter-spacing: 1.5px; text-transform: uppercase;
  padding: 3px 8px; border-radius: 2px;
  margin-bottom: 8px;
  border: 1px solid currentColor;
}
.deck-title {
  font-family: 'DM Serif Display', serif !important;
  font-size: 16px; line-height: 1.3; color: var(--ink);
  margin: 0 0 6px 0; flex: 1;
}
.deck-meta {
  font-size: 11px; color: var(--ink-faint);
  margin-bottom: 12px; letter-spacing: 0.3px;
}
.deck-actions {
  display: flex; gap: 8px; margin-top: auto;
}
.deck-btn {
  flex: 1; text-align: center;
  padding: 8px 0; border-radius: 3px;
  font-size: 11px; font-weight: 500; letter-spacing: 1.5px;
  text-transform: uppercase; cursor: pointer; border: none;
  text-decoration: none; display: inline-block;
}
.deck-btn-primary {
  background: var(--ink); color: #fff !important;
  text-decoration: none !important;
}
.deck-btn-primary:hover { background: var(--terra); text-decoration: none !important; }
.deck-btn-secondary {
  background: transparent; color: var(--ink-muted) !important;
  border: 1px solid var(--rule) !important;
  text-decoration: none !important;
}
.deck-btn-secondary:hover { border-color: var(--terra) !important; color: var(--terra) !important; }

/* ── Viewer panel ── */
.viewer-header {
  display: flex; align-items: baseline; gap: 12px; margin-bottom: 12px;
}
.viewer-title {
  font-family: 'DM Serif Display', serif !important;
  font-size: 20px; color: var(--ink); margin: 0;
}
.viewer-close {
  font-size: 11px; color: var(--ink-muted); letter-spacing: 1px;
  text-transform: uppercase; cursor: pointer; margin-left: auto;
}

/* ── Filter chips ── */
.filter-row {
  display: flex; flex-wrap: wrap; gap: 8px; margin-bottom: 20px;
}
.filter-chip {
  display: inline-block; padding: 5px 14px; border-radius: 2px;
  font-size: 11px; font-weight: 500; letter-spacing: 1.5px;
  text-transform: uppercase; cursor: pointer;
  border: 1px solid var(--rule); color: var(--ink-muted);
  background: var(--cream);
}
.filter-chip-active {
  background: var(--ink); color: #fff; border-color: var(--ink);
}

/* ── Empty state ── */
.empty-state {
  background: var(--warm); border: 1px dashed var(--rule);
  border-radius: 4px; padding: 60px 24px; text-align: center;
  color: var(--ink-faint);
}
.empty-state-title {
  font-family: 'DM Serif Display', serif !important;
  font-size: 22px; color: var(--ink-muted); margin: 0 0 8px 0;
}

/* ── Resources section ── */
.resource-card {
  border: 1px solid var(--rule); border-radius: 4px;
  padding: 16px 18px; background: var(--cream);
  margin-bottom: 10px;
}
.resource-card:hover { border-color: var(--sand); }
.resource-label {
  font-size: 10px; letter-spacing: 2px; text-transform: uppercase;
  color: var(--ink-muted); margin-bottom: 4px;
}
.resource-title {
  font-size: 14px; font-weight: 500; color: var(--ink); margin: 0;
}

/* ── Stats strip ── */
.stats-strip {
  display: flex; gap: 0; border: 1px solid var(--rule);
  border-radius: 4px; overflow: hidden; margin-bottom: 24px;
}
.stat-cell {
  flex: 1; padding: 14px 16px; text-align: center;
  border-right: 1px solid var(--rule);
}
.stat-cell:last-child { border-right: none; }
.stat-num {
  font-family: 'DM Serif Display', serif !important;
  font-size: 26px; color: var(--terra); line-height: 1; margin: 0;
}
.stat-lbl {
  font-size: 10px; letter-spacing: 2px; text-transform: uppercase;
  color: var(--ink-muted); margin-top: 3px;
}

/* ── Streamlit overrides ── */
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
  text-transform: uppercase !important; width: 100%;
}
div.stDownloadButton > button:hover {
  border-color: var(--terra) !important; color: var(--terra) !important;
}

[data-testid="stTextInput"] input {
  border: 1px solid var(--rule) !important;
  border-radius: 3px !important;
  background: var(--cream) !important;
  font-size: 14px !important;
  padding: 10px 14px !important;
}
[data-testid="stTextInput"] input:focus {
  border-color: var(--terra) !important;
  box-shadow: none !important;
}

[data-baseweb="tab-list"] { border-bottom: 2px solid var(--rule) !important; gap: 0 !important; }
[data-baseweb="tab"] {
  font-size: 11px !important; letter-spacing: 2px !important;
  text-transform: uppercase !important; color: var(--ink-muted) !important;
  padding: 10px 20px !important; background: transparent !important;
  border-bottom: 2px solid transparent !important; margin-bottom: -2px !important;
}
[aria-selected="true"][data-baseweb="tab"] {
  color: var(--ink) !important;
  border-bottom: 2px solid var(--terra) !important;
  font-weight: 500 !important;
}
[data-baseweb="tab-highlight"] { display: none !important; }
[data-baseweb="tab-border"]    { display: none !important; }

.site-footer {
  border-top: 3px double var(--rule); padding-top: 16px; margin-top: 40px;
  font-size: 12px; color: var(--ink-muted); text-align: center;
}
</style>
""", unsafe_allow_html=True)

# =====================================================================
# SESSION STATE
# =====================================================================
if "tm_viewing" not in st.session_state:
    st.session_state.tm_viewing = None   # path of PDF being viewed
if "tm_filter" not in st.session_state:
    st.session_state.tm_filter = "All"

# =====================================================================
# MASTHEAD
# =====================================================================
st.markdown("""
<div class="tm-masthead">
  <div class="tm-eyebrow">Chaouat Economics Lab · Teaching Resources</div>
  <div class="tm-title">Teaching Material</div>
  <div class="tm-sub">Slide decks and handouts — view in browser or download for classroom use.</div>
</div>
""", unsafe_allow_html=True)

# =====================================================================
# LOAD PDF LIST
# =====================================================================
all_pdfs = list_pdfs()

# =====================================================================
# PDF VIEWER (shown at top when a deck is open)
# =====================================================================
if st.session_state.tm_viewing and os.path.exists(st.session_state.tm_viewing):
    vpath = st.session_state.tm_viewing
    vtitle = pretty_title(vpath)
    vdata  = read_file_bytes(vpath)
    vb64   = base64.b64encode(vdata).decode()

    st.markdown(f"""
    <div style="margin-top:20px; margin-bottom:6px;">
      <div class="viewer-header">
        <div class="viewer-title">{vtitle}</div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    vcol1, vcol2 = st.columns([1, 6])
    with vcol1:
        if st.button("← Close viewer", key="close_viewer"):
            st.session_state.tm_viewing = None
            st.rerun()
    with vcol2:
        st.download_button(
            f"Download {os.path.basename(vpath)}",
            data=vdata,
            file_name=os.path.basename(vpath),
            mime="application/pdf",
            key="dl_viewer",
        )

    st.markdown(
        f"""<iframe src="data:application/pdf;base64,{vb64}"
            width="100%" height="680"
            style="border:1px solid #e0dbd2; border-radius:4px; margin-top:10px; display:block;">
        </iframe>""",
        unsafe_allow_html=True,
    )
    st.markdown('<div class="section-label">All decks</div>', unsafe_allow_html=True)

# =====================================================================
# STATS STRIP  (only when PDFs exist)
# =====================================================================
if all_pdfs:
    tags_all = [infer_tag(p)[0] for p in all_pdfs]
    n_topics = len(set(tags_all))
    total_mb = sum(os.path.getsize(p) for p in all_pdfs) / (1024*1024)
    n_covers = sum(1 for p in all_pdfs if read_cover_b64(p))

    st.markdown(f"""
    <div class="stats-strip" style="margin-top:20px;">
      <div class="stat-cell">
        <div class="stat-num">{len(all_pdfs)}</div>
        <div class="stat-lbl">Decks available</div>
      </div>
      <div class="stat-cell">
        <div class="stat-num">{n_topics}</div>
        <div class="stat-lbl">Topics covered</div>
      </div>
      <div class="stat-cell">
        <div class="stat-num">{total_mb:.1f} MB</div>
        <div class="stat-lbl">Total size</div>
      </div>
      <div class="stat-cell">
        <div class="stat-num">Free</div>
        <div class="stat-lbl">Always</div>
      </div>
    </div>
    """, unsafe_allow_html=True)

# =====================================================================
# SEARCH + FILTER
# =====================================================================
search_col, _ = st.columns([2, 3])
with search_col:
    query = st.text_input("", placeholder="Search decks by title or topic…",
                          label_visibility="collapsed", key="tm_search")

# Collect unique topics for filter chips
unique_topics = sorted(set(infer_tag(p)[0] for p in all_pdfs)) if all_pdfs else []

if unique_topics:
    # Render filter chips using buttons
    chip_cols = st.columns(len(unique_topics) + 1)
    labels = ["All"] + unique_topics

    for i, lbl in enumerate(labels):
        with chip_cols[i]:
            is_active = st.session_state.tm_filter == lbl
            btn_style = "filter_active" if is_active else "filter_inactive"
            if st.button(lbl, key=f"chip_{lbl}",
                         type="primary" if is_active else "secondary"):
                st.session_state.tm_filter = lbl
                st.rerun()

# =====================================================================
# FILTER PDF LIST
# =====================================================================
filtered_pdfs = all_pdfs

if query:
    q = query.lower()
    filtered_pdfs = [p for p in filtered_pdfs if q in pretty_title(p).lower() or q in infer_tag(p)[0].lower()]

if st.session_state.tm_filter != "All":
    filtered_pdfs = [p for p in filtered_pdfs if infer_tag(p)[0] == st.session_state.tm_filter]

# =====================================================================
# EMPTY STATES
# =====================================================================
if not all_pdfs:
    st.markdown("""
    <div class="empty-state" style="margin-top:24px;">
      <div class="empty-state-title">No decks yet</div>
      <p style="font-size:14px; color:#b0ada8; margin:0;">
        Add PDF files to the <code>materials/</code> folder in your repository.<br/>
        Cover images go in <code>materials/covers/</code> with the same filename (png, jpg, or webp).
      </p>
    </div>
    """, unsafe_allow_html=True)
    st.stop()

if not filtered_pdfs:
    st.markdown(f"""
    <div class="empty-state" style="margin-top:16px;">
      <div class="empty-state-title">No results</div>
      <p style="font-size:14px; color:#b0ada8; margin:0;">
        No decks match <em>"{query or st.session_state.tm_filter}"</em>.
        Try a different search or clear the filter.
      </p>
    </div>
    """, unsafe_allow_html=True)
    st.stop()

# =====================================================================
# DECK GRID
# =====================================================================
st.markdown(
    f'<p style="font-size:12px; color:#b0ada8; margin-bottom:18px;">'
    f'Showing {len(filtered_pdfs)} of {len(all_pdfs)} deck{"s" if len(all_pdfs)!=1 else ""}'
    f'</p>',
    unsafe_allow_html=True
)

COLS = 3
rows = [filtered_pdfs[i:i+COLS] for i in range(0, len(filtered_pdfs), COLS)]

for row in rows:
    cols = st.columns(COLS, gap="medium")
    for col, path in zip(cols, row):
        title      = pretty_title(path)
        size_str   = file_size_str(path)
        tag_label, tag_color = infer_tag(path)
        cover_b64  = read_cover_b64(path)   # empty string = no cover

        with col:
            # ── Cover image (or SVG placeholder) ──
            if cover_b64:
                st.markdown(
                    f'<div class="deck-cover">'
                    f'<img src="{cover_b64}" loading="lazy" alt="{title} cover"/>'
                    f'</div>',
                    unsafe_allow_html=True,
                )
            else:
                # Lightweight SVG placeholder — zero network requests, renders instantly
                initials = "".join(w[0] for w in title.split()[:2]).upper()
                st.markdown(f"""
                <div class="deck-cover" style="background:var(--warm);">
                  <svg width="100%" height="100%" viewBox="0 0 280 210"
                       xmlns="http://www.w3.org/2000/svg" style="display:block;">
                    <rect width="280" height="210" fill="#f2ede4"/>
                    <!-- paper lines -->
                    <rect x="80" y="40"  width="120" height="130" rx="3"
                          fill="#faf8f4" stroke="#e0dbd2" stroke-width="1.5"/>
                    <line x1="96"  y1="72"  x2="184" y2="72"  stroke="#e0dbd2" stroke-width="1.5"/>
                    <line x1="96"  y1="88"  x2="184" y2="88"  stroke="#e0dbd2" stroke-width="1.5"/>
                    <line x1="96"  y1="104" x2="168" y2="104" stroke="#e0dbd2" stroke-width="1.5"/>
                    <line x1="96"  y1="120" x2="176" y2="120" stroke="#e0dbd2" stroke-width="1.5"/>
                    <line x1="96"  y1="136" x2="160" y2="136" stroke="#e0dbd2" stroke-width="1.5"/>
                    <!-- dog-ear -->
                    <polygon points="164,40 200,40 200,76" fill="#e0dbd2"/>
                    <polygon points="164,40 200,76 164,76" fill="#faf8f4" stroke="#e0dbd2" stroke-width="1"/>
                    <!-- initials badge -->
                    <circle cx="140" cy="105" r="28" fill="#c9622a" opacity="0.12"/>
                    <text x="140" y="111" text-anchor="middle"
                          font-family="DM Serif Display, serif"
                          font-size="18" fill="#c9622a" font-weight="400">{initials}</text>
                  </svg>
                </div>
                """, unsafe_allow_html=True)

            # ── Card body ──
            st.markdown(f"""
            <div style="padding:14px 0 4px 0;">
              <span class="deck-tag" style="color:{tag_color}; border-color:{tag_color}22;
                    background:{tag_color}10;">{tag_label}</span>
              <div class="deck-title">{title}</div>
              <div class="deck-meta">{size_str} · PDF</div>
            </div>
            """, unsafe_allow_html=True)

            # ── Action buttons (Streamlit native for reliability) ──
            btn_col1, btn_col2 = st.columns(2, gap="small")

            with btn_col1:
                if st.button("View", key=f"view_{path}", use_container_width=True):
                    if st.session_state.tm_viewing == path:
                        st.session_state.tm_viewing = None
                    else:
                        st.session_state.tm_viewing = path
                    st.rerun()

            with btn_col2:
                pdf_bytes = read_file_bytes(path)
                st.download_button(
                    "Download",
                    data=pdf_bytes,
                    file_name=os.path.basename(path),
                    mime="application/pdf",
                    key=f"dl_{path}",
                    use_container_width=True,
                )

        # Spacer between rows
    st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)

# =====================================================================
# RESOURCES SECTION
# =====================================================================
st.markdown('<div class="section-label">Suggested resources by topic</div>', unsafe_allow_html=True)

res_left, res_right = st.columns([1.2, 1], gap="large")

RESOURCES = {
    "Macroeconomics": [
        ("Macroeconomics course", "https://www.khanacademy.org/economics-finance-domain/macroeconomics", "Khan Academy · Full course"),
        ("Inflation and price indices", "https://www.khanacademy.org/economics-finance-domain/macroeconomics/inflation-topic", "Khan Academy · Module"),
        ("Monetary policy and central banking", "https://www.khanacademy.org/economics-finance-domain/macroeconomics/monetary-system-topic", "Khan Academy · Module"),
        ("Fiscal policy", "https://www.khanacademy.org/economics-finance-domain/macroeconomics/fiscal-policy-topic", "Khan Academy · Module"),
    ],
    "Microeconomics": [
        ("Microeconomics course", "https://www.khanacademy.org/economics-finance-domain/microeconomics", "Khan Academy · Full course"),
        ("Supply, demand and equilibrium", "https://www.khanacademy.org/economics-finance-domain/microeconomics/supply-demand-equilibrium", "Khan Academy · Module"),
        ("Consumer and producer surplus", "https://www.khanacademy.org/economics-finance-domain/microeconomics/consumer-producer-surplus", "Khan Academy · Module"),
        ("Elasticity", "https://www.khanacademy.org/economics-finance-domain/microeconomics/elasticity-topic", "Khan Academy · Module"),
    ],
    "Finance & Capital Markets": [
        ("Finance & capital markets", "https://www.khanacademy.org/economics-finance-domain/core-finance", "Khan Academy · Full course"),
        ("Time value of money", "https://www.khanacademy.org/economics-finance-domain/core-finance/interest-tutorial", "Khan Academy · Module"),
        ("Bonds and interest rates", "https://www.khanacademy.org/economics-finance-domain/core-finance/stock-and-bonds", "Khan Academy · Module"),
        ("Risk and return", "https://www.khanacademy.org/economics-finance-domain/core-finance/risk-tutorial", "Khan Academy · Module"),
    ],
    "Statistics & Econometrics": [
        ("Statistics and probability", "https://www.khanacademy.org/math/statistics-probability", "Khan Academy · Full course"),
        ("Regression — intro", "https://www.khanacademy.org/math/statistics-probability/describing-relationships-quantitative-data", "Khan Academy · Module"),
        ("Sampling distributions & CLT", "https://www.khanacademy.org/math/statistics-probability/sampling-distributions-library", "Khan Academy · Module"),
        ("Confidence intervals & hypothesis testing", "https://www.khanacademy.org/math/statistics-probability/confidence-intervals-one-sample", "Khan Academy · Module"),
    ],
}

WORKFLOW_TIPS = {
    "Macroeconomics": "Open the relevant Policy Lab module first (Taylor rule for monetary policy, Module 02 for fiscal). Run one scenario, then assign the Khan Academy reading to solidify the mechanism.",
    "Microeconomics": "Pair the surplus module with the Trade & Incidence simulator in Policy Lab — students can verify deadweight loss calculations against the model's output.",
    "Finance & Capital Markets": "The Finance Tools pages (Investment Simulator, Financial News) provide live data to ground the concepts in real market behaviour.",
    "Statistics & Econometrics": "Use the regression module before introducing any empirical results from the teaching decks. Identification and external validity are the key wrap-up concepts.",
}

with res_left:
    topic_res = st.selectbox(
        "Topic",
        list(RESOURCES.keys()),
        label_visibility="collapsed",
        key="res_topic",
    )
    st.markdown("<div style='height:6px'></div>", unsafe_allow_html=True)

    for title, url, source in RESOURCES[topic_res]:
        st.markdown(f"""
        <a href="{url}" target="_blank" style="text-decoration:none; display:block; margin-bottom:8px;">
          <div class="resource-card">
            <div class="resource-label">{source}</div>
            <div class="resource-title">{title}</div>
          </div>
        </a>
        """, unsafe_allow_html=True)

with res_right:
    st.markdown(f"""
    <div style="background:var(--warm); border:1px solid var(--rule); border-radius:4px; padding:20px; margin-top:2px;">
      <div style="font-size:10px; letter-spacing:2.5px; text-transform:uppercase; color:var(--ink-muted); margin-bottom:10px;">
        Suggested workflow — {topic_res}
      </div>
      <p style="font-size:14px; line-height:1.7; color:#1a1814; margin:0 0 16px 0;">
        {WORKFLOW_TIPS[topic_res]}
      </p>
      <div style="border-top:1px solid var(--rule); padding-top:14px;">
        <div style="font-size:10px; letter-spacing:2px; text-transform:uppercase; color:var(--ink-muted); margin-bottom:8px;">
          Session pattern
        </div>
        <div style="font-size:13.5px; color:#6b6760; line-height:1.7;">
          <strong style="color:#1a1814;">1 · Deck</strong> — conceptual framing and key definitions.<br/>
          <strong style="color:#1a1814;">2 · Lab</strong> — run one experiment in the Policy Lab module.<br/>
          <strong style="color:#1a1814;">3 · Practice</strong> — 20–40 min Khan Academy on the weakest sub-topic.<br/>
          <strong style="color:#1a1814;">4 · Recap</strong> — definition, mechanism, one numerical intuition.
        </div>
      </div>
    </div>
    """, unsafe_allow_html=True)

# =====================================================================
# FOOTER
# =====================================================================
st.markdown("""
<div class="site-footer">
  Chaouat Economics Lab — Teaching Material · Free for educational use<br/>
  <span style="font-size:11px;">© Chaouat Economics Lab</span>
</div>
""", unsafe_allow_html=True)
