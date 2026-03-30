import datetime as dt
import re
from urllib.parse import urlparse, urlunparse, parse_qsl, urlencode

import feedparser
import numpy as np
import pandas as pd
import streamlit as st
import yfinance as yf
from dateutil import parser as dtparse, tz

st.set_page_config(
    page_title="Financial News — Chaouat Economics Lab",
    page_icon="📰",
    layout="wide",
)

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
.block-container { padding-top: 0 !important; padding-bottom: 3rem; max-width: 1240px; }

/* ── Masthead ── */
.fn-masthead {
  border-bottom: 3px double var(--rule);
  padding: 28px 0 18px 0; margin-bottom: 0;
}
.fn-eyebrow {
  font-size: 10px; letter-spacing: 3.5px; text-transform: uppercase;
  color: var(--ink-muted); margin-bottom: 8px;
}
.fn-title {
  font-family: 'DM Serif Display', serif !important;
  font-size: 52px; line-height: 1.0; color: var(--ink);
  margin: 0 0 8px 0; letter-spacing: -0.5px;
}
.fn-sub { font-size: 15px; color: var(--ink-muted); font-weight: 300; }

/* ── Section labels ── */
.section-label {
  font-size: 10px; letter-spacing: 3px; text-transform: uppercase;
  color: var(--ink-muted); border-top: 1px solid var(--rule);
  padding-top: 10px; margin: 28px 0 16px 0;
}
.section-label-flush {
  font-size: 10px; letter-spacing: 3px; text-transform: uppercase;
  color: var(--ink-muted); margin-bottom: 14px;
}

/* ── News card ── */
.news-card {
  border-bottom: 1px solid var(--rule);
  padding: 14px 0; display: flex; gap: 14px; align-items: flex-start;
}
.news-card:last-child { border-bottom: none; }
.news-favicon {
  width: 20px; height: 20px; border-radius: 3px;
  flex-shrink: 0; margin-top: 2px; opacity: 0.85;
}
.news-body { flex: 1; min-width: 0; }
.news-title {
  font-size: 15px; font-weight: 500; color: var(--ink);
  line-height: 1.4; margin: 0 0 5px 0;
  text-decoration: none; display: block;
}
.news-title:hover { color: var(--terra); text-decoration: none; }
.news-meta {
  font-size: 11.5px; color: var(--ink-faint);
  display: flex; gap: 8px; align-items: center; flex-wrap: wrap;
}
.news-source { color: var(--sand-dark); font-weight: 500; }
.news-dot { color: var(--rule); }

/* ── Source badge ── */
.source-pill {
  display: inline-block; padding: 3px 8px; border-radius: 2px;
  font-size: 10px; font-weight: 600; letter-spacing: 1px;
  text-transform: uppercase; border: 1px solid var(--rule);
  color: var(--stone); background: var(--warm); margin: 2px 2px 2px 0;
}
.source-pill-active {
  background: var(--ink); color: #fff; border-color: var(--ink);
}

/* ── Pager ── */
.pager-row {
  display: flex; align-items: center; justify-content: center;
  gap: 16px; padding: 16px 0; margin-top: 4px;
}
.pager-info { font-size: 12px; color: var(--ink-muted); }

/* ── Earnings table ── */
.earn-badge-beat {
  background: #eef5e8; color: #3a6b1a;
  border: 1px solid #c2dba8;
  padding: 2px 8px; border-radius: 2px;
  font-size: 11px; font-weight: 600; letter-spacing: 0.5px;
}
.earn-badge-miss {
  background: var(--terra-light); color: var(--sienna);
  border: 1px solid #f0c4aa;
  padding: 2px 8px; border-radius: 2px;
  font-size: 11px; font-weight: 600; letter-spacing: 0.5px;
}
.earn-badge-na {
  background: var(--warm); color: var(--ink-faint);
  border: 1px solid var(--rule);
  padding: 2px 8px; border-radius: 2px;
  font-size: 11px; letter-spacing: 0.5px;
}

/* ── Sidebar overrides ── */
[data-testid="stSidebar"] {
  background: var(--warm) !important;
  border-right: 1px solid var(--rule) !important;
}
[data-testid="stSidebar"] * { font-size: 13px !important; }
[data-testid="stSidebar"] h2,
[data-testid="stSidebar"] h3 {
  font-family: 'DM Serif Display', serif !important;
  font-size: 16px !important; color: var(--ink) !important;
}

/* ── Buttons ── */
div.stButton > button {
  background: var(--ink) !important; color: #fff !important;
  border: none !important; border-radius: 3px !important;
  font-size: 11px !important; letter-spacing: 1.5px !important;
  text-transform: uppercase !important; font-weight: 500 !important;
  padding: 8px 14px !important;
}
div.stButton > button:hover { background: var(--terra) !important; }

/* ── Tabs ── */
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
[data-baseweb="tab-highlight"],
[data-baseweb="tab-border"] { display: none !important; }

/* ── Text input ── */
[data-testid="stTextInput"] input {
  border: 1px solid var(--rule) !important; border-radius: 3px !important;
  background: var(--cream) !important; font-size: 13px !important;
}
[data-testid="stTextInput"] input:focus {
  border-color: var(--terra) !important; box-shadow: none !important;
}

/* ── Metric ── */
[data-testid="stMetricValue"] {
  font-family: 'DM Serif Display', serif !important;
  font-size: 24px !important;
}
[data-testid="stMetricLabel"] {
  font-size: 10px !important; letter-spacing: 1.5px !important;
  text-transform: uppercase !important; color: var(--ink-muted) !important;
}

/* ── Footer ── */
.site-footer {
  border-top: 3px double var(--rule); padding-top: 16px; margin-top: 40px;
  font-size: 12px; color: var(--ink-muted); text-align: center;
}
</style>
""", unsafe_allow_html=True)

# =====================================================================
# CONSTANTS & HELPERS
# =====================================================================
SOURCES = {
    "Reuters — Business":    "https://feeds.reuters.com/reuters/businessNews",
    "Reuters — M&A":         "https://feeds.reuters.com/reuters/mergersNews",
    "NYT — DealBook":        "https://rss.nytimes.com/services/xml/rss/nyt/DealBook.xml",
    "Crunchbase News":        "https://news.crunchbase.com/feed/",
    "Google News — M&A (7d)":"https://news.google.com/rss/search?q=acquisition+OR+acquires+OR+merger+OR+to+buy+when:7d&hl=en-US&gl=US&ceid=US:en",
    "CNBC — Top News":       "https://www.cnbc.com/id/100003114/device/rss/rss.html",
}

SP500_DEFAULT = [
    "AAPL","MSFT","NVDA","AMZN","GOOGL","META","TSLA","JPM","XOM","UNH",
    "LLY","AVGO","V","JNJ","PG","HD","MA","BAC","CVX","COST",
    "MRK","PEP","WMT","ABBV","KO","ADBE","NFLX","ORCL","AMD","TMO",
]

TRACKING = ("utm_", "mc_", "gclid", "fbclid")
_ws    = re.compile(r"\s+")
_punct = re.compile(r"[^\w\s]")

def normalize_url(u):
    pu = urlparse(u)
    q  = [(k,v) for k,v in parse_qsl(pu.query) if not k.lower().startswith(TRACKING)]
    pu = pu._replace(netloc=pu.netloc.lower().replace("www.",""), query=urlencode(q), fragment="")
    return urlunparse(pu)

def normalize_title(t):
    return _ws.sub(" ", _punct.sub(" ", t.lower())).strip()

def domain_from(url):
    try:    return urlparse(url).netloc.replace("www.","")
    except: return ""

def favicon(url):
    d = domain_from(url)
    return f"https://www.google.com/s2/favicons?domain={d}&sz=32" if d else ""

def time_ago(ts):
    if not ts: return ""
    delta = dt.datetime.now(tz=tz.tzlocal()) - ts
    s = int(delta.total_seconds())
    if s < 60:   return f"{s}s ago"
    if s < 3600: return f"{s//60}m ago"
    if s < 86400:return f"{s//3600}h ago"
    return f"{s//86400}d ago"

# =====================================================================
# DATA FETCHING (cached)
# =====================================================================
@st.cache_data(ttl=15*60, show_spinner=False)
def fetch_feeds(selected: list[str]) -> pd.DataFrame:
    rows = []
    for name in selected:
        url = SOURCES.get(name)
        if not url: continue
        try:
            fp = feedparser.parse(url)
            for e in fp.entries:
                link  = e.get("link","")
                title = (e.get("title","") or "").strip()
                pub   = e.get("published") or e.get("updated","")
                try:   ts = dtparse.parse(pub).astimezone(tz.tzlocal()) if pub else None
                except: ts = None
                rows.append({"source":name,"title":title,"link":link,
                             "published":ts,"domain":domain_from(link)})
        except: continue

    if not rows:
        return pd.DataFrame(columns=["source","title","link","published","domain"])

    df = pd.DataFrame(rows)
    df = df.drop_duplicates(subset=["title","link"])
    df["t_sig"] = df["title"].apply(normalize_title)
    df["u_sig"] = df["link"].apply(normalize_url)
    df = df.drop_duplicates(subset=["t_sig","u_sig"]).drop(columns=["t_sig","u_sig"])
    return df.sort_values("published", ascending=False, na_position="last").reset_index(drop=True)

@st.cache_data(ttl=20*60, show_spinner=False)
def fetch_earnings(tickers: list[str]) -> pd.DataFrame:
    """
    Fetch latest EPS estimate vs actual for each ticker.
    Returns DataFrame with: Ticker, Date, Estimate, Actual, Surprise$, Surprise%, Beat
    """
    rows = []
    for sym in tickers:
        try:
            t  = yf.Ticker(sym)
            ed = None
            try:   ed = t.get_earnings_dates(limit=8)
            except: pass

            if isinstance(ed, pd.DataFrame) and not ed.empty:
                df = ed.reset_index()
                # find date col
                date_col = next((c for c in df.columns if "date" in c.lower()), df.columns[0])
                est_col  = next((c for c in df.columns if "estimate" in c.lower()), None)
                act_col  = next((c for c in df.columns if "reported" in c.lower() or "actual" in c.lower()), None)

                df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
                # Only rows in the past (actual earnings)
                past = df[df[date_col] <= pd.Timestamp.now(tz="UTC")].sort_values(date_col, ascending=False)
                if past.empty: continue

                r   = past.iloc[0]
                d   = pd.to_datetime(r[date_col]).date()
                est = float(pd.to_numeric(r.get(est_col, np.nan), errors="coerce")) if est_col else np.nan
                act = float(pd.to_numeric(r.get(act_col, np.nan), errors="coerce")) if act_col else np.nan

                if np.isfinite(est) and np.isfinite(act) and est != 0:
                    surp_abs = act - est
                    surp_pct = 100.0 * surp_abs / abs(est)
                    beat     = "Beat" if surp_abs > 0 else "Miss"
                else:
                    surp_abs = np.nan; surp_pct = np.nan; beat = "N/A"

                rows.append({
                    "Ticker": sym, "Date": d,
                    "EPS estimate": round(est,2) if np.isfinite(est) else None,
                    "EPS actual":   round(act,2) if np.isfinite(act) else None,
                    "Surprise ($)": round(surp_abs,2) if np.isfinite(surp_abs) else None,
                    "Surprise (%)": round(surp_pct,1) if np.isfinite(surp_pct) else None,
                    "Result": beat,
                })
        except: continue

    if not rows:
        return pd.DataFrame()

    out = pd.DataFrame(rows).sort_values("Date", ascending=False)
    return out.reset_index(drop=True)

# =====================================================================
# SESSION STATE
# =====================================================================
for k,v in [("fn_page",1),("fn_earn_page",1)]:
    if k not in st.session_state: st.session_state[k] = v

# =====================================================================
# MASTHEAD
# =====================================================================
st.markdown("""
<div class="fn-masthead">
  <div class="fn-eyebrow">Chaouat Economics Lab · Markets &amp; Data</div>
  <div class="fn-title">Financial News</div>
  <div class="fn-sub">Live headlines from top finance outlets, plus S&amp;P 500 earnings vs expectations.</div>
</div>
""", unsafe_allow_html=True)

# =====================================================================
# SIDEBAR
# =====================================================================
with st.sidebar:
    st.markdown("### Sources")
    picked = st.multiselect(
        "Active feeds",
        list(SOURCES.keys()),
        default=list(SOURCES.keys()),
        label_visibility="collapsed",
    )

    st.markdown("### Filters")
    q         = st.text_input("Search headlines", placeholder="company, merger, acquisition…")
    days_back = st.slider("Max age (days)", 1, 30, 7)

    st.markdown("### Earnings universe")
    tickers_raw = st.text_input(
        "Tickers (comma-separated)",
        value="AAPL, MSFT, NVDA, AMZN, META, TSLA, JPM, GOOGL",
        help="Leave empty to scan the default S&P 500 list.",
    )
    earn_filter = st.radio("Show", ["All","Beats only","Misses only"], horizontal=True)
    earn_sort   = st.selectbox("Sort earnings by", [
        "Latest first","Biggest beat (%)","Biggest miss (%)","Biggest beat ($)","Biggest miss ($)",
    ])

    st.divider()
    if st.button("Refresh all data", use_container_width=True):
        fetch_feeds.clear()
        fetch_earnings.clear()
        st.session_state.fn_page = 1
        st.session_state.fn_earn_page = 1
        st.rerun()

# =====================================================================
# FETCH NEWS
# =====================================================================
with st.spinner("Loading headlines…"):
    df_raw = fetch_feeds(picked)

cutoff = dt.datetime.now(tz=tz.tzlocal()) - dt.timedelta(days=days_back)
df     = df_raw.copy()
df     = df[(df["published"].isna()) | (df["published"] >= cutoff)]
if q:
    df = df[df["title"].str.lower().str.contains(q.lower(), na=False)]
df = df.reset_index(drop=True)

# =====================================================================
# TABS
# =====================================================================
tab_news, tab_earn = st.tabs(["01 · Headlines", "02 · Earnings vs Expectations"])

# ─────────────────────────────────────────────────────────────────────
# TAB 1 — HEADLINES
# ─────────────────────────────────────────────────────────────────────
with tab_news:

    if df.empty:
        st.markdown("""
        <div style="background:var(--warm); border:1px dashed var(--rule); border-radius:4px;
                    padding:48px 24px; text-align:center; margin-top:20px;">
          <p style="font-size:22px; font-family:'DM Serif Display',serif; color:var(--ink-muted); margin:0 0 8px 0;">No headlines found</p>
          <p style="font-size:13px; color:var(--ink-faint); margin:0;">
            Try selecting more sources, widening the date range, or clearing the search filter.
          </p>
        </div>
        """, unsafe_allow_html=True)
    else:
        # ── Layout: news list left, sidebar right ──
        left, right = st.columns([2.2, 1], gap="large")

        with left:
            st.markdown(f'<div class="section-label-flush">{len(df)} headlines · refreshed every 15 min</div>', unsafe_allow_html=True)

            PAGE_SIZE = 8
            total_pages = max(1, (len(df) + PAGE_SIZE - 1) // PAGE_SIZE)
            st.session_state.fn_page = max(1, min(st.session_state.fn_page, total_pages))
            page_df = df.iloc[(st.session_state.fn_page-1)*PAGE_SIZE : st.session_state.fn_page*PAGE_SIZE]

            # News cards
            for _, row in page_df.iterrows():
                fav  = favicon(row["link"])
                when = time_ago(row["published"])
                src  = row["source"].split("—")[-1].strip() if "—" in row["source"] else row["source"]
                dom  = row["domain"]

                fav_html = f'<img src="{fav}" class="news-favicon"/>' if fav else \
                           '<div style="width:20px;height:20px;border-radius:3px;background:var(--warm);flex-shrink:0;margin-top:2px;"></div>'

                st.markdown(f"""
                <div class="news-card">
                  {fav_html}
                  <div class="news-body">
                    <a class="news-title" href="{row['link']}" target="_blank" rel="noopener">{row['title']}</a>
                    <div class="news-meta">
                      <span class="news-source">{src}</span>
                      <span class="news-dot">·</span>
                      <span>{dom}</span>
                      {'<span class="news-dot">·</span><span>' + when + '</span>' if when else ''}
                    </div>
                  </div>
                </div>
                """, unsafe_allow_html=True)

            # Pager
            p_prev, p_mid, p_next = st.columns([1, 3, 1])
            with p_prev:
                if st.button("← Prev", disabled=st.session_state.fn_page <= 1, key="fn_prev"):
                    st.session_state.fn_page -= 1; st.rerun()
            with p_mid:
                st.markdown(f'<div style="text-align:center;font-size:12px;color:var(--ink-muted);padding-top:10px;">Page {st.session_state.fn_page} of {total_pages}</div>', unsafe_allow_html=True)
            with p_next:
                if st.button("Next →", disabled=st.session_state.fn_page >= total_pages, key="fn_next"):
                    st.session_state.fn_page += 1; st.rerun()

        with right:
            # ── Source breakdown ──
            st.markdown('<div class="section-label-flush">By source</div>', unsafe_allow_html=True)
            counts = df.groupby("source").size().sort_values(ascending=False)
            for src, cnt in counts.items():
                label = src.split("—")[-1].strip() if "—" in src else src
                pct   = cnt / len(df)
                st.markdown(f"""
                <div style="margin-bottom:10px;">
                  <div style="display:flex;justify-content:space-between;font-size:12px;
                              color:var(--ink-muted);margin-bottom:4px;">
                    <span>{label}</span><span style="color:var(--ink);font-weight:500;">{cnt}</span>
                  </div>
                  <div style="background:var(--rule);border-radius:2px;height:3px;">
                    <div style="background:var(--terra);width:{pct*100:.0f}%;height:3px;border-radius:2px;"></div>
                  </div>
                </div>
                """, unsafe_allow_html=True)

            # ── Active filters summary ──
            st.markdown('<div class="section-label-flush" style="margin-top:20px;">Active filters</div>', unsafe_allow_html=True)
            st.markdown(f"""
            <div style="background:var(--warm);border:1px solid var(--rule);border-radius:4px;padding:14px 16px;font-size:13px;color:var(--ink-muted);line-height:1.8;">
              <strong style="color:var(--ink);">Search:</strong> {q if q else "—"}<br/>
              <strong style="color:var(--ink);">Max age:</strong> {days_back} day{'s' if days_back!=1 else ''}<br/>
              <strong style="color:var(--ink);">Sources:</strong> {len(picked)} active<br/>
              <strong style="color:var(--ink);">Results:</strong> {len(df)} headline{'s' if len(df)!=1 else ''}
            </div>
            """, unsafe_allow_html=True)

            # ── Freshest domains ──
            if not df.empty and df["published"].notna().any():
                freshest = df.dropna(subset=["published"]).iloc[0]
                when_f   = time_ago(freshest["published"])
                st.markdown(f"""
                <div style="margin-top:14px;border-left:3px solid var(--terra);background:var(--terra-light);
                            padding:12px 14px;border-radius:0 4px 4px 0;font-size:13px;color:var(--ink-muted);">
                  <strong style="color:var(--ink);font-size:11px;letter-spacing:1.5px;text-transform:uppercase;">
                    Latest headline
                  </strong><br/>
                  <span style="color:var(--ink);">{freshest['title'][:80]}{'…' if len(freshest['title'])>80 else ''}</span><br/>
                  <span style="font-size:11px;">{when_f}</span>
                </div>
                """, unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────
# TAB 2 — EARNINGS VS EXPECTATIONS
# ─────────────────────────────────────────────────────────────────────
with tab_earn:

    st.markdown('<div class="section-label">S&P 500 Earnings — Actual vs Analyst Estimate</div>', unsafe_allow_html=True)

    # Parse tickers
    if tickers_raw.strip():
        tickers = [t.strip().upper() for t in tickers_raw.split(",") if t.strip()]
    else:
        tickers = SP500_DEFAULT

    st.markdown(f"""
    <p style="font-size:13px;color:var(--ink-muted);margin-bottom:20px;">
      Showing most recent reported earnings for <strong style="color:var(--ink);">{len(tickers)}</strong> ticker{'s' if len(tickers)!=1 else ''}.
      A <em>beat</em> means EPS actual exceeded the analyst consensus estimate.
      Data via Yahoo Finance · cached 20 min.
    </p>
    """, unsafe_allow_html=True)

    with st.spinner("Fetching earnings data…"):
        earn_df = fetch_earnings(tickers)

    if earn_df.empty:
        st.markdown("""
        <div style="background:var(--warm);border:1px dashed var(--rule);border-radius:4px;
                    padding:48px 24px;text-align:center;">
          <p style="font-family:'DM Serif Display',serif;font-size:20px;color:var(--ink-muted);margin:0 0 8px 0;">
            No earnings data available
          </p>
          <p style="font-size:13px;color:var(--ink-faint);margin:0;">
            Yahoo Finance may be rate-limiting. Try refreshing, or reduce the number of tickers.
          </p>
        </div>
        """, unsafe_allow_html=True)
    else:
        # Apply filter
        if earn_filter == "Beats only":
            earn_df = earn_df[earn_df["Result"] == "Beat"]
        elif earn_filter == "Misses only":
            earn_df = earn_df[earn_df["Result"] == "Miss"]

        # Apply sort
        sort_map = {
            "Latest first":       ("Date",         False),
            "Biggest beat (%)":   ("Surprise (%)", False),
            "Biggest miss (%)":   ("Surprise (%)", True),
            "Biggest beat ($)":   ("Surprise ($)", False),
            "Biggest miss ($)":   ("Surprise ($)", True),
        }
        sort_col, sort_asc = sort_map[earn_sort]
        if sort_col in earn_df.columns:
            earn_df = earn_df.sort_values(sort_col, ascending=sort_asc, na_position="last")
        earn_df = earn_df.reset_index(drop=True)

        # ── Summary metrics ──
        n_total = len(earn_df)
        n_beat  = (earn_df["Result"]=="Beat").sum()
        n_miss  = (earn_df["Result"]=="Miss").sum()
        n_na    = (earn_df["Result"]=="N/A").sum()
        beat_rt = n_beat/max(1,n_beat+n_miss)*100
        avg_surp = earn_df["Surprise (%)"].dropna().mean()

        m1,m2,m3,m4,m5 = st.columns(5, gap="medium")
        for col, val, lbl in [
            (m1, str(n_total),          "Tickers shown"),
            (m2, str(n_beat),           "Beat estimates"),
            (m3, str(n_miss),           "Missed estimates"),
            (m4, f"{beat_rt:.0f}%",     "Beat rate"),
            (m5, f"{avg_surp:+.1f}%" if not np.isnan(avg_surp) else "—", "Avg surprise"),
        ]:
            with col:
                st.markdown(f"""
                <div style="background:var(--warm);border:1px solid var(--rule);border-radius:4px;
                            padding:14px 16px;text-align:center;margin-bottom:4px;">
                  <div style="font-family:'DM Serif Display',serif;font-size:26px;
                              color:var(--terra);line-height:1;">{val}</div>
                  <div style="font-size:10px;letter-spacing:1.5px;text-transform:uppercase;
                              color:var(--ink-muted);margin-top:4px;">{lbl}</div>
                </div>
                """, unsafe_allow_html=True)

        st.markdown("<div style='height:4px'></div>", unsafe_allow_html=True)

        # ── Earnings table ──
        el, er = st.columns([2.5, 1], gap="large")

        with el:
            # Render rows as styled HTML table for full visual control
            E_PAGE = 12
            e_total_pages = max(1, (len(earn_df) + E_PAGE - 1) // E_PAGE)
            st.session_state.fn_earn_page = max(1, min(st.session_state.fn_earn_page, e_total_pages))
            earn_page = earn_df.iloc[(st.session_state.fn_earn_page-1)*E_PAGE : st.session_state.fn_earn_page*E_PAGE]

            table_rows = ""
            for _, r in earn_page.iterrows():
                result = r.get("Result","N/A")
                if result == "Beat":
                    badge = f'<span class="earn-badge-beat">Beat</span>'
                    row_bg = ""
                elif result == "Miss":
                    badge = f'<span class="earn-badge-miss">Miss</span>'
                    row_bg = ""
                else:
                    badge = f'<span class="earn-badge-na">N/A</span>'
                    row_bg = ""

                est  = f"{r['EPS estimate']:.2f}"  if r["EPS estimate"] is not None else "—"
                act  = f"{r['EPS actual']:.2f}"    if r["EPS actual"]   is not None else "—"
                sabs = (f"+{r['Surprise ($)']:.2f}" if r["Surprise ($)"] and r["Surprise ($)"] > 0
                        else f"{r['Surprise ($)']:.2f}" if r["Surprise ($)"] is not None else "—")
                spct = (f"+{r['Surprise (%)']:.1f}%" if r["Surprise (%)"] and r["Surprise (%)"] > 0
                        else f"{r['Surprise (%)']:.1f}%" if r["Surprise (%)"] is not None else "—")

                s_color = "#3a6b1a" if result=="Beat" else ("#8b3a1a" if result=="Miss" else "#b0ada8")

                table_rows += f"""
                <tr style="border-bottom:1px solid #e0dbd2;">
                  <td style="padding:10px 12px;font-weight:500;color:#1a1814;font-size:14px;">{r['Ticker']}</td>
                  <td style="padding:10px 12px;font-size:13px;color:#6b6760;">{r['Date']}</td>
                  <td style="padding:10px 12px;font-size:13px;text-align:right;">{est}</td>
                  <td style="padding:10px 12px;font-size:13px;text-align:right;font-weight:500;">{act}</td>
                  <td style="padding:10px 12px;font-size:13px;text-align:right;color:{s_color};font-weight:500;">{sabs}</td>
                  <td style="padding:10px 12px;font-size:13px;text-align:right;color:{s_color};font-weight:500;">{spct}</td>
                  <td style="padding:10px 12px;text-align:center;">{badge}</td>
                </tr>
                """

            st.markdown(f"""
            <div style="overflow-x:auto;border:1px solid #e0dbd2;border-radius:4px;">
              <table style="width:100%;border-collapse:collapse;background:#faf8f4;">
                <thead>
                  <tr style="border-bottom:2px solid #e0dbd2;background:#f2ede4;">
                    <th style="padding:10px 12px;text-align:left;font-size:10px;letter-spacing:2px;
                               text-transform:uppercase;color:#6b6760;font-weight:500;">Ticker</th>
                    <th style="padding:10px 12px;text-align:left;font-size:10px;letter-spacing:2px;
                               text-transform:uppercase;color:#6b6760;font-weight:500;">Date</th>
                    <th style="padding:10px 12px;text-align:right;font-size:10px;letter-spacing:2px;
                               text-transform:uppercase;color:#6b6760;font-weight:500;">Estimate</th>
                    <th style="padding:10px 12px;text-align:right;font-size:10px;letter-spacing:2px;
                               text-transform:uppercase;color:#6b6760;font-weight:500;">Actual</th>
                    <th style="padding:10px 12px;text-align:right;font-size:10px;letter-spacing:2px;
                               text-transform:uppercase;color:#6b6760;font-weight:500;">Surprise $</th>
                    <th style="padding:10px 12px;text-align:right;font-size:10px;letter-spacing:2px;
                               text-transform:uppercase;color:#6b6760;font-weight:500;">Surprise %</th>
                    <th style="padding:10px 12px;text-align:center;font-size:10px;letter-spacing:2px;
                               text-transform:uppercase;color:#6b6760;font-weight:500;">Result</th>
                  </tr>
                </thead>
                <tbody>
                  {table_rows}
                </tbody>
              </table>
            </div>
            """, unsafe_allow_html=True)

            # Pager
            ep, ei, en = st.columns([1, 3, 1])
            with ep:
                if st.button("← Prev", disabled=st.session_state.fn_earn_page<=1, key="ep_prev"):
                    st.session_state.fn_earn_page -= 1; st.rerun()
            with ei:
                st.markdown(f'<div style="text-align:center;font-size:12px;color:var(--ink-muted);padding-top:10px;">Page {st.session_state.fn_earn_page} of {e_total_pages}</div>', unsafe_allow_html=True)
            with en:
                if st.button("Next →", disabled=st.session_state.fn_earn_page>=e_total_pages, key="ep_next"):
                    st.session_state.fn_earn_page += 1; st.rerun()

        with er:
            # ── Beat/Miss visual breakdown ──
            st.markdown("""
            <div style="font-size:10px;letter-spacing:2.5px;text-transform:uppercase;
                        color:var(--ink-muted);margin-bottom:12px;">Beat vs miss breakdown</div>
            """, unsafe_allow_html=True)

            # Simple bar chart using CSS
            if n_beat + n_miss > 0:
                beat_w = int(n_beat / (n_beat+n_miss) * 100)
                miss_w = 100 - beat_w
                st.markdown(f"""
                <div style="margin-bottom:20px;">
                  <div style="display:flex;border-radius:3px;overflow:hidden;height:28px;margin-bottom:8px;">
                    <div style="width:{beat_w}%;background:#5a8a30;display:flex;align-items:center;
                                justify-content:center;font-size:12px;color:#fff;font-weight:500;">
                      {beat_w}%
                    </div>
                    <div style="width:{miss_w}%;background:var(--sienna);display:flex;align-items:center;
                                justify-content:center;font-size:12px;color:#fff;font-weight:500;">
                      {miss_w}%
                    </div>
                  </div>
                  <div style="display:flex;gap:16px;font-size:12px;color:var(--ink-muted);">
                    <span><span style="color:#5a8a30;font-weight:500;">■</span> Beat ({n_beat})</span>
                    <span><span style="color:var(--sienna);font-weight:500;">■</span> Miss ({n_miss})</span>
                  </div>
                </div>
                """, unsafe_allow_html=True)

            # Top beats
            top_beats = earn_df[earn_df["Result"]=="Beat"].nlargest(5,"Surprise (%)")
            if not top_beats.empty:
                st.markdown("""
                <div style="font-size:10px;letter-spacing:2px;text-transform:uppercase;
                            color:var(--ink-muted);margin-bottom:10px;margin-top:4px;">Biggest beats</div>
                """, unsafe_allow_html=True)
                for _, r in top_beats.iterrows():
                    spct = r["Surprise (%)"]
                    bar_w = min(100, abs(spct)/max(1, top_beats["Surprise (%)"].abs().max())*100)
                    st.markdown(f"""
                    <div style="margin-bottom:10px;">
                      <div style="display:flex;justify-content:space-between;font-size:13px;margin-bottom:3px;">
                        <span style="font-weight:500;color:var(--ink);">{r['Ticker']}</span>
                        <span style="color:#3a6b1a;font-weight:500;">+{spct:.1f}%</span>
                      </div>
                      <div style="background:var(--rule);border-radius:2px;height:3px;">
                        <div style="background:#5a8a30;width:{bar_w:.0f}%;height:3px;border-radius:2px;"></div>
                      </div>
                    </div>
                    """, unsafe_allow_html=True)

            # Top misses
            top_misses = earn_df[earn_df["Result"]=="Miss"].nsmallest(5,"Surprise (%)")
            if not top_misses.empty:
                st.markdown("""
                <div style="font-size:10px;letter-spacing:2px;text-transform:uppercase;
                            color:var(--ink-muted);margin-bottom:10px;margin-top:16px;">Biggest misses</div>
                """, unsafe_allow_html=True)
                for _, r in top_misses.iterrows():
                    spct = r["Surprise (%)"]
                    bar_w = min(100, abs(spct)/max(1, top_misses["Surprise (%)"].abs().max())*100)
                    st.markdown(f"""
                    <div style="margin-bottom:10px;">
                      <div style="display:flex;justify-content:space-between;font-size:13px;margin-bottom:3px;">
                        <span style="font-weight:500;color:var(--ink);">{r['Ticker']}</span>
                        <span style="color:var(--sienna);font-weight:500;">{spct:.1f}%</span>
                      </div>
                      <div style="background:var(--rule);border-radius:2px;height:3px;">
                        <div style="background:var(--sienna);width:{bar_w:.0f}%;height:3px;border-radius:2px;"></div>
                      </div>
                    </div>
                    """, unsafe_allow_html=True)

            # Reading note
            st.markdown("""
            <div style="margin-top:20px;border-left:3px solid var(--sand-dark);background:#f5eedd;
                        padding:12px 14px;border-radius:0 4px 4px 0;font-size:12.5px;
                        color:var(--ink-muted);line-height:1.6;">
              <strong style="color:var(--ink);">How to read this.</strong>
              Surprise = Actual EPS − Analyst estimate. A positive surprise means
              the company earned more than analysts expected — this often moves the
              stock price on the day of release.
            </div>
            """, unsafe_allow_html=True)

# =====================================================================
# FOOTER
# =====================================================================
st.markdown("""
<div class="site-footer">
  Chaouat Economics Lab — Financial News · Data via RSS feeds &amp; Yahoo Finance · Educational use only<br/>
  <span style="font-size:11px;">© Chaouat Economics Lab</span>
</div>
""", unsafe_allow_html=True)
