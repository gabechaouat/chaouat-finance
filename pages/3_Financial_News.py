# pages/3_Financial_News.py
import datetime as dt
from urllib.parse import urlparse
import feedparser
import streamlit as st
import pandas as pd
from dateutil import parser as dtparse, tz
# --- Deduplication helpers ---
from urllib.parse import urlparse, urlunparse, parse_qsl, urlencode
import re
import numpy as np
import yfinance as yf

TRACKING_PREFIXES = ("utm_", "mc_", "gclid", "fbclid")

def normalize_url(u: str) -> str:
    """Remove tracking params & normalize domain for duplicate detection."""
    pu = urlparse(u)
    q = [(k, v) for k, v in parse_qsl(pu.query, keep_blank_values=True)
         if not k.lower().startswith(TRACKING_PREFIXES)]
    pu = pu._replace(netloc=pu.netloc.lower().replace("www.", ""),
                     query=urlencode(q, doseq=True),
                     fragment="")
    return urlunparse(pu)

_ws = re.compile(r"\s+")
_punct = re.compile(r"[^\w\s]")

def normalize_title(t: str) -> str:
    """Lowercase, remove punctuation & compress whitespace."""
    t = t.lower()
    t = _punct.sub(" ", t)
    return _ws.sub(" ", t).strip()


# ---------- Page config ----------
st.set_page_config(page_title="Financial News", page_icon="📰", layout="wide")

# ---------- Styles + header (identique au style de la page principale) ----------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Montserrat:wght@400;600;700&display=swap');
:root{
  --primary:#007BA7; --primary-dark:#005F7D; --accent:#00B0FF;
  --bg:#F7FAFC; --card:#FFFFFF; --text:#0F172A; --muted:#64748B;
}
html, body, * { font-family:'Montserrat', sans-serif !important; }
[data-testid="stAppViewContainer"] > .main { padding-top: 54px; }
.cf-sticky {
  position: fixed; top:0; left:0; right:0; z-index:10000;
  width:100%; height:44px; color:#fff;
  background: linear-gradient(135deg, var(--primary) 0%, var(--accent) 100%);
  display:flex; align-items:center; padding:0 14px;
  border-bottom:1px solid rgba(255,255,255,.15);
  box-shadow:0 6px 14px rgba(0,0,0,.08);
  font-weight:700; letter-spacing:.2px; font-size:18px;
}
.cf-hero{
  background: linear-gradient(135deg, var(--primary) 0%, var(--accent) 100%);
  color:#fff; padding:28px 32px; border-radius:20px;
  box-shadow:0 8px 24px rgba(0,123,167,.25); margin:8px 0 24px 0;
}
.cf-brand{ font-weight:700; font-size:40px; letter-spacing:.4px; }
.cf-section{
  background: var(--card); border:1px solid #E2E8F0; border-radius:18px;
  padding:22px; box-shadow:0 8px 30px rgba(15,23,42,.06); margin:6px 0 24px 0;
}
.news-card{
  border:1px solid #E2E8F0; border-radius:14px; padding:14px 14px;
  margin-bottom:10px; background:#fff;
}
.news-title{ font-size:16px; font-weight:700; color:#0F172A; margin:0; }
.news-meta{ color:var(--muted); font-size:13px; margin-top:4px; }
.source-chip{
  display:inline-flex; align-items:center; gap:8px; padding:4px 8px;
  border:1px solid #E2E8F0; border-radius:999px; font-size:12px; color:#0F172A;
  background:#F8FAFC;
}
.source-chip img{ width:16px; height:16px; border-radius:3px; }
.search-row { margin-bottom: 10px; }
</style>
<div class="cf-sticky">Chaouat Finance</div>
""", unsafe_allow_html=True)

st.markdown("""
<div class="cf-hero">
  <div class="cf-brand">Financial News</div>
  <div>Fresh headlines from top finance outlets. Click to read the full story.</div>
</div>
""", unsafe_allow_html=True)

# ---------- Sources RSS ----------
SOURCES = {
    "Reuters – Business": "https://feeds.reuters.com/reuters/businessNews",
    "Reuters – Mergers & Acquisitions": "https://feeds.reuters.com/reuters/mergersNews",
    "NYT – DealBook": "https://rss.nytimes.com/services/xml/rss/nyt/DealBook.xml",
    "Crunchbase News": "https://news.crunchbase.com/feed/",
    # Google News requête M&A (7 derniers jours)
    "Google News – M&A (7d)": "https://news.google.com/rss/search?q=acquisition+OR+acquires+OR+merger+OR+to+buy+when:7d&hl=en-US&gl=US&ceid=US:en",
    # CNBC Top News (flux principal)
    "CNBC – Top News": "https://www.cnbc.com/id/100003114/device/rss/rss.html",
}
# ---------- Default tickers for earnings (broad, so we rarely get an empty table) ----------
DEFAULT_EARN_TICKERS = [
    "AAPL","MSFT","NVDA","AMZN","GOOGL","META","TSLA","JPM","XOM","UNH",
    "LLY","AVGO","V","JNJ","PG","HD","MA","BAC","CVX","COST",
    "MRK","PEP","WMT","ABBV","KO","ADBE","NFLX","ORCL","AMD","TMO"
]

# ---------- Helpers ----------
def domain_from_url(url: str) -> str:
    try:
        return urlparse(url).netloc.replace("www.", "")
    except Exception:
        return ""

def favicon_for(url: str) -> str:
    dom = domain_from_url(url)
    return f"https://www.google.com/s2/favicons?domain={dom}&sz=64" if dom else ""

@st.cache_data(ttl=15*60, show_spinner=False)
def fetch_feeds(selected_sources: list[str]) -> pd.DataFrame:
    """Fetch and normalize items from multiple RSS/Atom feeds."""
    rows = []
    for name in selected_sources:
        feed_url = SOURCES.get(name)
        if not feed_url:
            continue
        try:
            fp = feedparser.parse(feed_url)
            for e in fp.entries:
                link = e.get("link") or ""
                title = e.get("title") or "(no title)"
                published = e.get("published") or e.get("updated") or ""
                try:
                    # Best-effort parse
                    ts = dtparse.parse(published).astimezone(tz.tzlocal()) if published else None
                except Exception:
                    ts = None
                rows.append({
                    "source": name,
                    "title": title.strip(),
                    "link": link,
                    "published": ts,
                    "domain": domain_from_url(link),
                })
        except Exception:
            continue
    if not rows:
        return pd.DataFrame(columns=["source","title","link","published","domain"])
    df = pd.DataFrame(rows).drop_duplicates(subset=["title","link"])
    # Most recent first
    df = df.sort_values("published", ascending=False, na_position="last").reset_index(drop=True)
    return df

def time_ago(ts: dt.datetime | None) -> str:
    if not ts:
        return ""
    now = dt.datetime.now(tz=tz.tzlocal())
    delta = now - ts
    s = int(delta.total_seconds())
    if s < 60: return f"{s}s ago"
    m = s // 60
    if m < 60: return f"{m}m ago"
    h = m // 60
    if h < 24: return f"{h}h ago"
    d = h // 24
    return f"{d}d ago"
@st.cache_data(ttl=15*60)
def get_latest_earnings_df(tickers: list[str]) -> pd.DataFrame:
    """
    For each ticker, fetch the most recent earnings row (date, estimate, actual),
    then compute the beat/miss in $ and % if needed.
    Works against multiple yfinance schemas.
    """
    rows = []
    for sym in tickers:
        try:
            t = yf.Ticker(sym)

            # Preferred: get_earnings_dates (newer yfinance)
            ed = None
            try:
                ed = t.get_earnings_dates(limit=12)
            except Exception:
                ed = None

            df = None
            if isinstance(ed, pd.DataFrame) and not ed.empty:
                df = ed.reset_index()
                # Normalize likely column names
                # date
                date_col = next(
                    (c for c in df.columns if "date" in c.lower()),
                    df.columns[0]
                )
                # estimate & actual
                est_col = next((c for c in df.columns if "estimate" in c.lower() and "eps" in c.lower()), None)
                act_col = next((c for c in df.columns if ("reported" in c.lower() or "actual" in c.lower()) and "eps" in c.lower()), None)
                spct_col = next((c for c in df.columns if "surprise" in c.lower() and "%" in c), None)
                sabs_col = next((c for c in df.columns if "surprise" in c.lower() and "%" not in c.lower()), None)

                df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
                df = df.dropna(subset=[date_col]).sort_values(date_col, ascending=False)

                if not df.empty:
                    r = df.iloc[0]
                    d = pd.to_datetime(r[date_col]).date()
                    est = pd.to_numeric(r.get(est_col, np.nan), errors="coerce") if est_col else np.nan
                    act = pd.to_numeric(r.get(act_col, np.nan), errors="coerce") if act_col else np.nan

                    # Surprise
                    if np.isfinite(est) and np.isfinite(act) and est != 0:
                        beat_abs = act - est
                        beat_pct = 100.0 * (act - est) / abs(est)
                    else:
                        beat_abs = pd.to_numeric(r.get(sabs_col, np.nan), errors="coerce") if sabs_col else np.nan
                        beat_pct = pd.to_numeric(r.get(spct_col, np.nan), errors="coerce") if spct_col else np.nan

                    rows.append({
                        "Ticker": sym,
                        "Earnings date": d,
                        "EPS estimate": est,
                        "EPS actual": act,
                        "Beat ($)": beat_abs,
                        "Beat (%)": beat_pct
                    })
                    continue  # go next symbol

            # Fallback: legacy .calendar (may or may not exist)
            cal = None
            try:
                cal = t.calendar
            except Exception:
                cal = None

            if isinstance(cal, pd.DataFrame) and not cal.empty:
                # calendar uses index rows as fields
                idx = {str(i).lower(): i for i in cal.index}
                d = pd.to_datetime(cal.loc[idx.get("earnings date")].iloc[0], errors="coerce").date() if "earnings date" in idx else None

                # Try various label variants
                est_keys = [k for k in idx.keys() if "estimate" in k and "eps" in k]
                act_keys = [k for k in idx.keys() if ("actual" in k or "reported" in k) and "eps" in k]

                est = pd.to_numeric(cal.loc[idx[est_keys[0]]].iloc[0], errors="coerce") if est_keys else np.nan
                act = pd.to_numeric(cal.loc[idx[act_keys[0]]].iloc[0], errors="coerce") if act_keys else np.nan

                if np.isfinite(est) and np.isfinite(act) and est != 0:
                    beat_abs = act - est
                    beat_pct = 100.0 * (act - est) / abs(est)
                else:
                    beat_abs = np.nan
                    beat_pct = np.nan

                rows.append({
                    "Ticker": sym,
                    "Earnings date": d,
                    "EPS estimate": est,
                    "EPS actual": act,
                    "Beat ($)": beat_abs,
                    "Beat (%)": beat_pct
                })
        except Exception:
            # swallow per-ticker errors
            continue

    out = pd.DataFrame(rows)
    if not out.empty:
        out = out.sort_values("Earnings date", ascending=False)
    return out

# ---------- Sidebar controls ----------
with st.sidebar:
    st.header("Sources")
    default_sel = list(SOURCES.keys())  # tout sélectionné par défaut
    picked_sources = st.multiselect("Pick sources", options=list(SOURCES.keys()), default=default_sel)
    st.divider()
    st.header("Filters")
    q = st.text_input("Search headline (optional)", placeholder="acquires, merger, company name…")
    days_limit = st.slider("Max age (days)", 1, 30, 7)
    st.caption("Headlines older than this window are hidden.")
    refresh = st.button("Refresh now")
    st.divider()
    st.header("Earnings")

    tickers_in = st.text_input(
        "Tickers (comma-separated)",
        value="AAPL, MSFT, NVDA",
        help="Leave empty to use a broad default list."
    ).strip()

    earn_show = st.checkbox("Show earnings", value=True)

    earn_sort = st.selectbox(
        "Sort by",
        ["Latest first", "Biggest beat ($)", "Biggest beat (%)", "Biggest miss ($)", "Biggest miss (%)"],
        index=0
    )

    earn_filter = st.radio(
        "Filter",
        ["All", "Beats only", "Misses only"],
        index=0,
        horizontal=True
    )



if refresh:
    fetch_feeds.clear()
    st.session_state.news_page = 1
    st.session_state.earn_page = 1

# ---------- Fetch ----------
df = fetch_feeds(picked_sources)
if df.empty:
    st.info("No news retrieved. Try different sources or refresh.")
    st.stop()

# Age filter
cutoff = dt.datetime.now(tz=tz.tzlocal()) - dt.timedelta(days=days_limit)
df = df[(df["published"].isna()) | (df["published"] >= cutoff)]

# Text filter
if q:
    q_low = q.lower()
    df = df[df["title"].str.lower().str.contains(q_low, na=False)]
# ---------- Global de-duplication (title + URL signatures) ----------
if not df.empty:
    df = df.copy()
    df["t_sig"] = df["title"].astype(str).apply(normalize_title)
    df["u_sig"] = df["link"].astype(str).apply(normalize_url)
    df = df.drop_duplicates(subset=["t_sig", "u_sig"]).drop(columns=["t_sig", "u_sig"])

# ---------- Pagination (5 headlines max on screen) ----------
PAGE_SIZE = 5
if "news_page" not in st.session_state:
    st.session_state.news_page = 1

total_items = len(df)
total_pages = max(1, (total_items + PAGE_SIZE - 1) // PAGE_SIZE)

# Clamp current page into valid range (handles filter changes gracefully)
st.session_state.news_page = max(1, min(st.session_state.news_page, total_pages))

start_idx = (st.session_state.news_page - 1) * PAGE_SIZE
end_idx = start_idx + PAGE_SIZE
df_page = df.iloc[start_idx:end_idx]


# ---------- Render ----------
st.markdown('<div class="cf-section">', unsafe_allow_html=True)
left, right = st.columns([2.2, 1], gap="large")

with left:
    st.subheader("Latest headlines")
    for _, r in df_page.iterrows():
        # --- RENDER ---
        fav = favicon_for(r["link"])
        meta = f'{r["source"]} · {r["domain"]}'
        when = time_ago(r["published"])
        st.markdown(
        f"""
            <div class="news-card">
              <div class="news-title"><a href="{r['link']}" target="_blank" rel="noopener noreferrer">{r['title']}</a></div>
              <div class="news-meta">
                <span class="source-chip">
                  {'<img src="'+fav+'" />' if fav else ''}{meta}
                </span>
                {' · ' + when if when else ''}
              </div>
            </div>
            """,
            unsafe_allow_html=True
        )
    # Pager controls (never show more than 5 at a time)
    cp, ci, cn = st.columns([0.18, 0.64, 0.18])
    with cp:
        if st.button("◀ Previous", disabled=(st.session_state.news_page <= 1), key="news_prev"):
            st.session_state.news_page -= 1
            st.rerun()
    with ci:
        st.markdown(
            f"<div style='text-align:center;color:#64748B;'>Page {st.session_state.news_page} / {total_pages}</div>",
            unsafe_allow_html=True
        )
    with cn:
        if st.button("Next ▶", disabled=(st.session_state.news_page >= total_pages), key="news_next"):
            st.session_state.news_page += 1
            st.rerun()


with right:
    st.subheader("By source")
    counts = df.groupby("source", as_index=False).size().sort_values("size", ascending=False)
    st.dataframe(counts.rename(columns={"size":"Headlines"}), use_container_width=True, hide_index=True, height=250)
    st.caption("Click a headline to open the original article in a new tab.")

st.markdown('</div>', unsafe_allow_html=True)

# ---------- Earnings section (own 5-item pager) ----------
if earn_show:
    # Resolve universe
    if tickers_in:
        syms = [s.strip().upper() for s in tickers_in.split(",") if s.strip()]
    else:
        syms = DEFAULT_EARN_TICKERS

    earn = get_latest_earnings_df(syms)

    # Optional filter: beats/misses
    if not earn.empty and earn_filter != "All":
        if earn_filter == "Beats only":
            earn = earn[pd.to_numeric(earn["Beat ($)"], errors="coerce") > 0]
        elif earn_filter == "Misses only":
            earn = earn[pd.to_numeric(earn["Beat ($)"], errors="coerce") < 0]

    # Sorting
    if not earn.empty:
        if earn_sort == "Latest first":
            earn = earn.sort_values("Earnings date", ascending=False, na_position="last")
        elif earn_sort == "Biggest beat ($)":
            earn = earn.sort_values("Beat ($)", ascending=False, na_position="last")
        elif earn_sort == "Biggest beat (%)":
            earn = earn.sort_values("Beat (%)", ascending=False, na_position="last")
        elif earn_sort == "Biggest miss ($)":
            earn = earn.sort_values("Beat ($)", ascending=True, na_position="last")
        elif earn_sort == "Biggest miss (%)":
            earn = earn.sort_values("Beat (%)", ascending=True, na_position="last")

    # Render panel
    st.markdown('<div class="cf-section">', unsafe_allow_html=True)
    st.subheader("Earnings")

    if earn.empty:
        st.info("No earnings data found for the current settings.")
        st.markdown('</div>', unsafe_allow_html=True)
    else:
        # Pagination: 5 rows per screen
        E_PAGE_SIZE = 5
        if "earn_page" not in st.session_state:
            st.session_state.earn_page = 1

        total_e = len(earn)
        total_e_pages = max(1, (total_e + E_PAGE_SIZE - 1) // E_PAGE_SIZE)
        st.session_state.earn_page = max(1, min(st.session_state.earn_page, total_e_pages))

        e_start = (st.session_state.earn_page - 1) * E_PAGE_SIZE
        e_end = e_start + E_PAGE_SIZE
        earn_page = earn.iloc[e_start:e_end].copy()

        # Round neat display
        for c in ["EPS estimate", "EPS actual", "Beat ($)", "Beat (%)"]:
            if c in earn_page.columns:
                earn_page[c] = pd.to_numeric(earn_page[c], errors="coerce").round(2)

        # Show as a compact table (max 5 rows on screen)
        st.dataframe(
            earn_page,
            use_container_width=True,
            hide_index=True,
            height=220,  # fits ~5 rows; you can adjust
            column_config={
                "Earnings date": st.column_config.DateColumn(format="YYYY-MM-DD"),
                "EPS estimate": st.column_config.NumberColumn(format="%.2f"),
                "EPS actual":   st.column_config.NumberColumn(format="%.2f"),
                "Beat ($)":     st.column_config.NumberColumn(format="%.2f"),
                "Beat (%)":     st.column_config.NumberColumn(format="%.2f%%"),
            }
        )

        # Pager controls centered
        ep, ei, en = st.columns([0.18, 0.64, 0.18])
        with ep:
            if st.button("◀ Previous", disabled=(st.session_state.earn_page <= 1), key="earn_prev"):
                st.session_state.earn_page -= 1
                st.rerun()
        with ei:
            st.markdown(
                f"<div style='text-align:center;color:#64748B;'>Page {st.session_state.earn_page} / {total_e_pages}</div>",
                unsafe_allow_html=True
            )
        with en:
            if st.button("Next ▶", disabled=(st.session_state.earn_page >= total_e_pages), key="earn_next"):
                st.session_state.earn_page += 1
                st.rerun()

        st.caption("Beat = Actual EPS − Estimate EPS. Positive = beat; negative = miss.")
        st.markdown('</div>', unsafe_allow_html=True)

