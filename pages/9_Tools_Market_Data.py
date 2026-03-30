import datetime as dt
import streamlit.components.v1 as components
import math
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import streamlit as st
import yfinance as yf

st.set_page_config(
    page_title="Market Data — Chaouat Economics Lab",
    page_icon="💹",
    layout="wide",
)

# ─────────────────────────────────────────────────────────────────────
# STYLE
# ─────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Serif+Display:ital@0;1&family=DM+Sans:wght@300;400;500;600&family=DM+Mono:wght@400;500&display=swap');

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
  --green:       #3a6b1a;
  --green-light: #eef5e8;
}

html, body, * { font-family: 'DM Sans', sans-serif !important; }
.block-container { padding-top: 0 !important; padding-bottom: 3rem; max-width: 1300px; }

/* ── Masthead ── */
.md-masthead {
  border-bottom: 3px double var(--rule);
  padding: 28px 0 18px 0; margin-bottom: 0;
}
.md-eyebrow {
  font-size: 10px; letter-spacing: 3.5px; text-transform: uppercase;
  color: var(--ink-muted); margin-bottom: 8px;
}
.md-title {
  font-family: 'DM Serif Display', serif !important;
  font-size: 52px; line-height: 1.0; color: var(--ink);
  margin: 0 0 8px 0; letter-spacing: -0.5px;
}
.md-sub { font-size: 15px; color: var(--ink-muted); font-weight: 300; }

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

/* ── Ticker tape ── */
.ticker-tape-wrap {
  overflow: hidden; border-top: 1px solid var(--rule);
  border-bottom: 1px solid var(--rule);
  background: var(--warm); padding: 8px 0; margin: 18px 0;
  position: relative;
}
.ticker-tape {
  display: flex; gap: 40px; white-space: nowrap;
  animation: scroll-tape 30s linear infinite;
}
@keyframes scroll-tape {
  0%   { transform: translateX(0); }
  100% { transform: translateX(-50%); }
}
.tape-item {
  display: inline-flex; align-items: baseline; gap: 8px;
  font-size: 12px;
}
.tape-sym  { font-family:'DM Mono',monospace; font-weight:500; color:var(--ink); letter-spacing:1px; }
.tape-val  { font-family:'DM Mono',monospace; color:var(--ink-muted); }
.tape-chg-pos { font-family:'DM Mono',monospace; color:var(--green); font-size:11px; }
.tape-chg-neg { font-family:'DM Mono',monospace; color:var(--sienna); font-size:11px; }

/* ── KPI cards ── */
.kpi-card {
  background: var(--warm); border: 1px solid var(--rule);
  border-radius: 4px; padding: 14px 16px; text-align: center;
}
.kpi-num {
  font-family: 'DM Serif Display', serif !important;
  font-size: 26px; line-height: 1; color: var(--ink); margin: 0;
}
.kpi-num-pos   { color: var(--green); }
.kpi-num-neg   { color: var(--sienna); }
.kpi-num-terra { color: var(--terra); }
.kpi-label {
  font-size: 10px; letter-spacing: 2px; text-transform: uppercase;
  color: var(--ink-muted); margin-top: 4px;
}

/* ── Metric chip ── */
.chip {
  display: inline-block; padding: 3px 10px; border-radius: 2px;
  font-size: 11px; font-weight: 500; letter-spacing: 1px;
  font-family: 'DM Mono', monospace;
}
.chip-pos  { background: var(--green-light); color: var(--green); border: 1px solid #c2dba8; }
.chip-neg  { background: var(--terra-light); color: var(--sienna); border: 1px solid #f0c4aa; }
.chip-neu  { background: var(--warm); color: var(--stone); border: 1px solid var(--rule); }

/* ── Candlestick info ── */
.ohlc-row {
  display: flex; gap: 20px; align-items: center;
  padding: 10px 0; border-bottom: 1px solid var(--rule);
  font-size: 13px;
}
.ohlc-label { color: var(--ink-muted); width: 40px; flex-shrink: 0; }
.ohlc-val   { font-family: 'DM Mono', monospace; color: var(--ink); font-weight: 500; }

/* ── Radar legend ── */
.radar-legend { display: flex; flex-wrap: wrap; gap: 8px; margin-top: 10px; }
.radar-dot { width:10px; height:10px; border-radius:50%; display:inline-block; margin-right:4px; }

/* ── Regime badge ── */
.regime-badge {
  display: inline-block; padding: 6px 16px; border-radius: 3px;
  font-size: 12px; font-weight: 600; letter-spacing: 2px; text-transform: uppercase;
}
.regime-low  { background: var(--green-light); color: var(--green); border: 1px solid #c2dba8; }
.regime-mid  { background: var(--warm); color: var(--sand-dark); border: 1px solid var(--sand); }
.regime-high { background: var(--terra-light); color: var(--sienna); border: 1px solid #f0c4aa; }

/* ── Buttons ── */
div.stButton > button {
  background: var(--ink) !important; color: #fff !important;
  border: none !important; border-radius: 3px !important;
  font-size: 11px !important; letter-spacing: 1.5px !important;
  text-transform: uppercase !important; font-weight: 500 !important;
  padding: 9px 16px !important; width: 100%;
}
div.stButton > button:hover { background: var(--terra) !important; }

/* ── Tabs ── */
[data-baseweb="tab-list"] { border-bottom: 2px solid var(--rule) !important; gap: 0 !important; }
[data-baseweb="tab"] {
  font-size: 11px !important; letter-spacing: 2px !important;
  text-transform: uppercase !important; color: var(--ink-muted) !important;
  padding: 10px 18px !important; background: transparent !important;
  border-bottom: 2px solid transparent !important; margin-bottom: -2px !important;
}
[aria-selected="true"][data-baseweb="tab"] {
  color: var(--ink) !important; border-bottom: 2px solid var(--terra) !important;
  font-weight: 500 !important;
}
[data-baseweb="tab-highlight"], [data-baseweb="tab-border"] { display: none !important; }

/* ── Sidebar ── */
[data-testid="stSidebar"] { background: var(--warm) !important; border-right: 1px solid var(--rule) !important; }
[data-testid="stSlider"] label,
[data-testid="stSelectbox"] label { font-size: 12px !important; color: var(--ink-muted) !important; }

/* ── Interp box ── */
.interp-box {
  border-left: 3px solid var(--terra); background: var(--terra-light);
  padding: 12px 16px; border-radius: 0 4px 4px 0; margin-top: 12px;
  font-size: 13px; color: var(--ink-muted); line-height: 1.65;
}
.interp-box-sand  { border-left-color: var(--sand-dark); background: #f5eedd; }
.interp-box-stone { border-left-color: var(--stone); background: var(--stone-light); }
.interp-box-green { border-left-color: var(--green); background: var(--green-light); }

/* ── Mono values ── */
.mono { font-family: 'DM Mono', monospace !important; }

/* ── Footer ── */
.site-footer {
  border-top: 3px double var(--rule); padding-top: 16px; margin-top: 40px;
  font-size: 12px; color: var(--ink-muted); text-align: center;
}
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────
# PALETTE
# ─────────────────────────────────────────────────────────────────────
C = {
    "terra":     "#c9622a", "terra_mid":"#a84e20",
    "sienna":    "#8b3a1a", "sand":     "#c4a882",
    "sand_dark": "#9e8060", "stone":    "#7a6f62",
    "ink":       "#1a1814", "muted":    "#b0ada8",
    "rule":      "#e0dbd2", "cream":    "#faf8f4",
    "warm":      "#f2ede4", "green":    "#3a6b1a",
}
PALETTE = [C["terra"],C["sienna"],C["sand_dark"],C["stone"],C["ink"],C["sand"],C["terra_mid"],C["muted"]]

def plot_base(height=400, title="", legend=True):
    return dict(
        height=height, paper_bgcolor=C["cream"], plot_bgcolor=C["cream"],
        font=dict(family="DM Sans", color=C["ink"], size=12),
        title=dict(text=title, font=dict(family="DM Serif Display",size=15,color=C["ink"]),
                   x=0, xanchor="left") if title else dict(text=""),
        margin=dict(l=16,r=16,t=44 if title else 20,b=16),
        legend=dict(orientation="h",y=1.08,x=0,font=dict(size=11)) if legend else dict(visible=False),
        xaxis=dict(gridcolor=C["rule"],linecolor=C["rule"],tickfont=dict(size=11),zeroline=False),
        yaxis=dict(gridcolor=C["rule"],linecolor=C["rule"],tickfont=dict(size=11),zeroline=False),
        colorway=PALETTE, hovermode="x unified",
    )

# ─────────────────────────────────────────────────────────────────────
# CACHED DATA FETCHERS
# ─────────────────────────────────────────────────────────────────────
@st.cache_data(ttl=60*60, show_spinner=False)
def load_sp500() -> pd.DataFrame:
    url = "https://raw.githubusercontent.com/datasets/s-and-p-500-companies/main/data/constituents.csv"
    df  = pd.read_csv(url)
    df.columns = [c.strip() for c in df.columns]
    for alias in ["GICS Sector","Sector"]:
        if alias in df.columns: df = df.rename(columns={alias:"Sector"}); break
    for alias in ["Security","Name","Company"]:
        if alias in df.columns: df = df.rename(columns={alias:"Name"}); break
    df["Symbol"] = df["Symbol"].astype(str).str.upper().str.strip()
    return df[["Symbol","Name","Sector"]].dropna(subset=["Symbol"])

@st.cache_data(ttl=30*60, show_spinner=False)
def get_ohlcv(sym: str, period: str = "1y", interval: str = "1d") -> pd.DataFrame:
    t = yf.Ticker(sym)
    df = t.history(period=period, interval=interval, auto_adjust=True)
    if df.empty: return df
    # Strip timezone robustly — handle both tz-aware and tz-naive indexes
    idx = df.index
    if hasattr(idx, "tz") and idx.tz is not None:
        idx = idx.tz_convert("UTC").tz_localize(None)
    df.index = pd.to_datetime(idx)
    df = df[["Open","High","Low","Close","Volume"]].dropna()
    # Convert index to plain dates so Plotly Candlestick never sees tz info
    df.index = df.index.normalize()
    return df

@st.cache_data(ttl=30*60, show_spinner=False)
def get_multi_close(tickers: tuple, period: str = "1y") -> pd.DataFrame:
    if not tickers: return pd.DataFrame()
    raw = yf.download(list(tickers), period=period, auto_adjust=True, progress=False, group_by="ticker")
    out = {}
    for sym in tickers:
        try:
            col = raw[(sym,"Close")] if isinstance(raw.columns, pd.MultiIndex) else raw["Close"]
            out[sym] = col.dropna()
        except: pass
    return pd.DataFrame(out).dropna(how="all")

@st.cache_data(ttl=60*60, show_spinner=False)
def get_sector_perf() -> pd.DataFrame:
    sector_etfs = {
        "Technology":        "XLK", "Health Care":      "XLV",
        "Financials":        "XLF", "Consumer Discr.":  "XLY",
        "Industrials":       "XLI", "Communication":    "XLC",
        "Consumer Staples":  "XLP", "Energy":           "XLE",
        "Utilities":         "XLU", "Real Estate":      "XLRE",
        "Materials":         "XLB",
    }
    rows = []
    for sector, etf in sector_etfs.items():
        df = get_ohlcv(etf, period="1y")
        if df.empty or len(df) < 22: continue
        r1m  = (df["Close"].iloc[-1]/df["Close"].iloc[-22]-1)*100
        r3m  = (df["Close"].iloc[-1]/df["Close"].iloc[min(-63,-(len(df)-1))]-1)*100
        r1y  = (df["Close"].iloc[-1]/df["Close"].iloc[0]-1)*100
        vol  = df["Close"].pct_change().std()*np.sqrt(252)*100
        rows.append({"Sector":sector,"ETF":etf,"1M%":r1m,"3M%":r3m,"1Y%":r1y,"Vol%":vol,
                     "Price":df["Close"].iloc[-1]})
    return pd.DataFrame(rows)

@st.cache_data(ttl=30*60, show_spinner=False)
def tape_data() -> list:
    symbols = ["^GSPC","^IXIC","^DJI","^VIX","GC=F","CL=F","BTC-USD","^TNX"]
    labels  = ["S&P 500","Nasdaq","Dow Jones","VIX","Gold","Oil","Bitcoin","10Y Treasury"]
    rows = []
    for sym, lbl in zip(symbols, labels):
        try:
            df = get_ohlcv(sym, period="5d", interval="1d")
            if len(df) >= 2:
                chg = (df["Close"].iloc[-1]/df["Close"].iloc[-2]-1)*100
                rows.append({"label":lbl,"price":df["Close"].iloc[-1],"chg":chg})
        except: pass
    return rows

# ─────────────────────────────────────────────────────────────────────
# TECHNICAL INDICATORS
# ─────────────────────────────────────────────────────────────────────
def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["SMA20"]  = df["Close"].rolling(20).mean()
    df["SMA50"]  = df["Close"].rolling(50).mean()
    df["SMA200"] = df["Close"].rolling(200).mean()
    df["EMA12"]  = df["Close"].ewm(span=12).mean()
    df["EMA26"]  = df["Close"].ewm(span=26).mean()
    df["MACD"]   = df["EMA12"] - df["EMA26"]
    df["Signal"] = df["MACD"].ewm(span=9).mean()
    df["Hist"]   = df["MACD"] - df["Signal"]
    roll = df["Close"].rolling(20)
    df["BB_mid"] = roll.mean()
    df["BB_std"] = roll.std()
    df["BB_up"]  = df["BB_mid"] + 2*df["BB_std"]
    df["BB_dn"]  = df["BB_mid"] - 2*df["BB_std"]
    delta = df["Close"].diff()
    gain  = delta.clip(lower=0).rolling(14).mean()
    loss  = (-delta.clip(upper=0)).rolling(14).mean()
    rs    = gain / loss.replace(0, np.nan)
    df["RSI"] = 100 - (100/(1+rs))
    df["RollMax"] = df["Close"].cummax()
    df["Drawdown"] = (df["Close"]/df["RollMax"]-1)*100
    df["VolAnn"]   = df["Close"].pct_change().rolling(20).std()*np.sqrt(252)*100
    return df

# ─────────────────────────────────────────────────────────────────────
# MASTHEAD
# ─────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="md-masthead">
  <div class="md-eyebrow">Chaouat Economics Lab · Finance Tools</div>
  <div class="md-title">Market Data</div>
  <div class="md-sub">Deep analysis tools — price, volatility, sector structure, cross-sectional scanning, and correlation anatomy.</div>
</div>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────
# LIVE TICKER TAPE
# ─────────────────────────────────────────────────────────────────────
with st.spinner("Loading market pulse…"):
    tape = tape_data()

if tape:
    items_html = ""
    for item in tape * 2:   # duplicate for seamless loop
        chg_cls = "tape-chg-pos" if item["chg"] >= 0 else "tape-chg-neg"
        chg_color = "#3a6b1a" if item["chg"] >= 0 else "#8b3a1a"
        sign = "▲" if item["chg"] >= 0 else "▼"
        price_str = f"{item['price']:,.2f}"
        chg_str   = f"{sign} {abs(item['chg']):.2f}%"
        items_html += f"""<div style="display:inline-flex;align-items:baseline;gap:8px;font-size:12px;margin-right:40px;">
          <span style="font-family:'DM Mono',monospace;font-weight:500;color:#1a1814;letter-spacing:1px;">{item['label']}</span>
          <span style="font-family:'DM Mono',monospace;color:#6b6760;">{price_str}</span>
          <span style="font-family:'DM Mono',monospace;color:{chg_color};font-size:11px;">{chg_str}</span>
        </div>"""

    tape_html = f"""<!DOCTYPE html>
<html><head>
<link href="https://fonts.googleapis.com/css2?family=DM+Mono:wght@400;500&display=swap" rel="stylesheet">
<style>
  body {{ margin:0; background:#f2ede4; overflow:hidden; }}
  .tape-wrap {{
    overflow:hidden; border-top:1px solid #e0dbd2; border-bottom:1px solid #e0dbd2;
    background:#f2ede4; padding:7px 0; width:100%;
  }}
  .tape-inner {{
    display:inline-flex; white-space:nowrap;
    animation: scroll-tape 35s linear infinite;
  }}
  @keyframes scroll-tape {{
    0%   {{ transform: translateX(0); }}
    100% {{ transform: translateX(-50%); }}
  }}
</style>
</head><body>
<div class="tape-wrap">
  <div class="tape-inner">{items_html}</div>
</div>
</body></html>"""
    components.html(tape_html, height=38, scrolling=False)

# ─────────────────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### Universe")
    try:
        sp500_df = load_sp500()
        all_syms = sorted(sp500_df["Symbol"].unique().tolist())
    except:
        all_syms = ["AAPL","MSFT","NVDA","AMZN","GOOGL","META","TSLA","JPM","XOM"]

    sectors  = ["All"] + sorted(sp500_df["Sector"].dropna().unique()) if "sp500_df" in dir() else ["All"]
    sector_f = st.selectbox("Sector filter", sectors, key="sec_f")

    if sector_f != "All" and "sp500_df" in dir():
        syms_filtered = sp500_df[sp500_df["Sector"]==sector_f]["Symbol"].tolist()
    else:
        syms_filtered = all_syms

    st.markdown("### Primary ticker")
    primary = st.selectbox("Select ticker", options=syms_filtered,
                           index=syms_filtered.index("AAPL") if "AAPL" in syms_filtered else 0,
                           key="primary_sym")

    st.markdown("### Comparison tickers")
    compare_raw = st.text_input("Add tickers (comma-separated)", value="MSFT, NVDA, META", key="compare_t")
    compare_syms = [t.strip().upper() for t in compare_raw.split(",") if t.strip()]

    st.markdown("### Time range")
    period_map = {"1 month":"1mo","3 months":"3mo","6 months":"6mo",
                  "1 year":"1y","2 years":"2y","5 years":"5y"}
    period_lbl = st.selectbox("Period", list(period_map.keys()), index=3, key="period_lbl")
    period     = period_map[period_lbl]

    st.markdown("### Scanner")
    scan_metric = st.selectbox("Metric", [
        "1Y return","Volatility","RSI","Sharpe ratio","Max drawdown","Price vs 200-day SMA"
    ], key="scan_metric")
    scan_n = st.slider("Top/bottom N", 3, 20, 8, key="scan_n")

    st.divider()
    if st.button("Refresh data", key="refresh_all"):
        st.cache_data.clear()
        st.rerun()

# ─────────────────────────────────────────────────────────────────────
# FETCH PRIMARY DATA
# ─────────────────────────────────────────────────────────────────────
with st.spinner(f"Loading {primary}…"):
    ohlcv = get_ohlcv(primary, period=period)

if ohlcv.empty:
    st.error(f"No data for {primary}. Try a different ticker.")
    st.stop()

ohlcv = add_indicators(ohlcv)

# Comparison data
all_tickers = tuple(sorted(set([primary] + compare_syms)))
with st.spinner("Loading comparison data…"):
    closes = get_multi_close(all_tickers, period=period)

# ─────────────────────────────────────────────────────────────────────
# TABS
# ─────────────────────────────────────────────────────────────────────
t1,t2,t3,t4,t5,t6 = st.tabs([
    "01 · Deep Dive",
    "02 · Sector X-Ray",
    "03 · Volatility Lab",
    "04 · Scanner",
    "05 · Correlation Anatomy",
    "06 · Radar Comparison",
])

# ═════════════════════════════════════════════════════════════════════
# TAB 1 — DEEP DIVE  (candlestick + BB + MACD + RSI gauge)
# ═════════════════════════════════════════════════════════════════════
with t1:
    st.markdown(f'<div class="section-label">Stock deep dive — {primary}</div>', unsafe_allow_html=True)

    last  = ohlcv.iloc[-1]
    prev  = ohlcv.iloc[-2]
    chg   = (last["Close"]/prev["Close"]-1)*100
    chg_c = C["green"] if chg >= 0 else C["sienna"]

    # ── KPI strip ──
    k1,k2,k3,k4,k5,k6 = st.columns(6, gap="medium")
    for col, val, lbl, cls in [
        (k1, f"${last['Close']:.2f}",       "Last price",         "kpi-num"),
        (k2, f"{chg:+.2f}%",                "Day change",         "kpi-num-pos" if chg>=0 else "kpi-num-neg"),
        (k3, f"${last['High']:.2f}",         "Today's high",       "kpi-num"),
        (k4, f"${last['Low']:.2f}",          "Today's low",        "kpi-num"),
        (k5, f"{last['RSI']:.1f}",           "RSI (14)",           "kpi-num-terra"),
        (k6, f"{last['VolAnn']:.1f}%",       "20d vol (ann.)",     "kpi-num"),
    ]:
        with col:
            st.markdown(f'<div class="kpi-card"><div class="kpi-num {cls}">{val}</div><div class="kpi-label">{lbl}</div></div>', unsafe_allow_html=True)

    st.markdown("<div style='height:12px'></div>", unsafe_allow_html=True)
    dl, dr = st.columns([2.2, 1], gap="large")

    with dl:
        # ── Candlestick + BB + overlays ──
        fig_c = make_subplots(rows=3, cols=1, shared_xaxes=True,
                              row_heights=[0.58, 0.22, 0.20],
                              vertical_spacing=0.02,
                              subplot_titles=["","MACD","RSI"])

        # Candlestick — use explicit list to guarantee no tz-aware objects reach Plotly
        x_dates = ohlcv.index.strftime("%Y-%m-%d").tolist()
        fig_c.add_trace(go.Candlestick(
            x=x_dates,
            open=ohlcv["Open"].tolist(),
            high=ohlcv["High"].tolist(),
            low=ohlcv["Low"].tolist(),
            close=ohlcv["Close"].tolist(),
            increasing=dict(line=dict(color=C["green"], width=1), fillcolor="rgba(58,107,26,0.33)"),
            decreasing=dict(line=dict(color=C["sienna"], width=1), fillcolor="rgba(139,58,26,0.33)"),
            name="OHLC", showlegend=False,
        ), row=1, col=1)

        # Bollinger bands
        fig_c.add_trace(go.Scatter(x=x_dates, y=ohlcv["BB_up"].tolist(),
            name="BB upper", line=dict(color=C["sand"],width=1,dash="dot"), showlegend=True), row=1, col=1)
        fig_c.add_trace(go.Scatter(x=x_dates, y=ohlcv["BB_dn"].tolist(),
            name="BB lower", line=dict(color=C["sand"],width=1,dash="dot"),
            fill="tonexty", fillcolor="rgba(196,168,130,0.08)", showlegend=False), row=1, col=1)
        fig_c.add_trace(go.Scatter(x=x_dates, y=ohlcv["BB_mid"].tolist(),
            name="BB mid", line=dict(color=C["sand_dark"],width=1), showlegend=True), row=1, col=1)

        # SMAs
        for ma, color, w in [("SMA20",C["terra"],1.5),("SMA50",C["stone"],1.5),("SMA200",C["ink"],1.8)]:
            fig_c.add_trace(go.Scatter(x=x_dates, y=ohlcv[ma].tolist(),
                name=ma, line=dict(color=color,width=w)), row=1, col=1)

        # MACD
        macd_colors = [C["green"] if v>=0 else C["sienna"] for v in ohlcv["Hist"]]
        fig_c.add_trace(go.Bar(x=x_dates, y=ohlcv["Hist"].tolist(),
            name="MACD hist", marker_color=macd_colors, showlegend=False), row=2, col=1)
        fig_c.add_trace(go.Scatter(x=x_dates, y=ohlcv["MACD"].tolist(),
            name="MACD", line=dict(color=C["terra"],width=1.5)), row=2, col=1)
        fig_c.add_trace(go.Scatter(x=x_dates, y=ohlcv["Signal"].tolist(),
            name="Signal", line=dict(color=C["stone"],width=1.5,dash="dot")), row=2, col=1)

        # RSI
        fig_c.add_trace(go.Scatter(x=x_dates, y=ohlcv["RSI"].tolist(),
            name="RSI", line=dict(color=C["sienna"],width=2), showlegend=False), row=3, col=1)
        fig_c.add_hline(y=70, row=3, col=1, line_color=C["sienna"], line_width=1, line_dash="dot")
        fig_c.add_hline(y=30, row=3, col=1, line_color=C["green"],  line_width=1, line_dash="dot")
        fig_c.add_hrect(y0=70, y1=100, row=3, col=1, fillcolor="rgba(139,58,26,0.09)", line_width=0)
        fig_c.add_hrect(y0=0,  y1=30,  row=3, col=1, fillcolor="rgba(58,107,26,0.09)",  line_width=0)

        fig_c.update_layout(
            height=640, paper_bgcolor=C["cream"], plot_bgcolor=C["cream"],
            font=dict(family="DM Sans", color=C["ink"], size=11),
            margin=dict(l=16,r=16,t=20,b=16),
            legend=dict(orientation="h",y=1.06,x=0,font=dict(size=10)),
            xaxis_rangeslider_visible=False,
            colorway=PALETTE,
        )
        for ax in ["xaxis","xaxis2","xaxis3","yaxis","yaxis2","yaxis3"]:
            fig_c.update_layout(**{ax:dict(gridcolor=C["rule"],linecolor=C["rule"])})
        fig_c.update_yaxes(title_text="Price ($)", row=1, col=1)
        fig_c.update_yaxes(title_text="MACD",      row=2, col=1)
        fig_c.update_yaxes(title_text="RSI",        row=3, col=1, range=[0,100])

        st.plotly_chart(fig_c, use_container_width=True)

    with dr:
        # ── RSI gauge (custom SVG) ──
        rsi_val  = float(last["RSI"]) if not np.isnan(last["RSI"]) else 50
        rsi_angle = (rsi_val/100)*180 - 90   # -90 to +90
        rsi_rad   = math.radians(rsi_angle)
        needle_x  = 100 + 65*math.cos(rsi_rad)
        needle_y  = 90  - 65*math.sin(rsi_rad)
        rsi_status = ("Overbought" if rsi_val > 70 else
                      "Oversold"   if rsi_val < 30 else "Neutral")
        rsi_col    = C["sienna"] if rsi_val>70 else (C["green"] if rsi_val<30 else C["sand_dark"])

        st.markdown(f"""
        <div style="background:var(--warm);border:1px solid var(--rule);border-radius:4px;padding:18px;margin-bottom:14px;text-align:center;">
          <div class="section-label-flush" style="text-align:center;">RSI Gauge</div>
          <svg viewBox="0 0 200 110" xmlns="http://www.w3.org/2000/svg" style="width:100%;max-width:240px;">
            <!-- Background arc zones -->
            <path d="M 25 90 A 75 75 0 0 1 65 28" stroke="{C['green']}" stroke-width="12" fill="none" stroke-linecap="round" opacity="0.35"/>
            <path d="M 65 28 A 75 75 0 0 1 135 28" stroke="{C['sand']}" stroke-width="12" fill="none" stroke-linecap="round" opacity="0.35"/>
            <path d="M 135 28 A 75 75 0 0 1 175 90" stroke="{C['sienna']}" stroke-width="12" fill="none" stroke-linecap="round" opacity="0.35"/>
            <!-- Needle -->
            <line x1="100" y1="90" x2="{needle_x:.1f}" y2="{needle_y:.1f}"
                  stroke="{rsi_col}" stroke-width="3" stroke-linecap="round"/>
            <circle cx="100" cy="90" r="5" fill="{rsi_col}"/>
            <!-- Labels -->
            <text x="18"  y="100" font-size="9" fill="{C['green']}"   font-family="DM Sans">30</text>
            <text x="90"  y="20"  font-size="9" fill="{C['sand_dark']}" font-family="DM Sans" text-anchor="middle">50</text>
            <text x="174" y="100" font-size="9" fill="{C['sienna']}"  font-family="DM Sans">70</text>
            <!-- Value -->
            <text x="100" y="78" font-size="18" font-weight="500"
                  fill="{rsi_col}" font-family="DM Serif Display,serif"
                  text-anchor="middle">{rsi_val:.0f}</text>
          </svg>
          <div style="font-size:12px;font-weight:500;color:{rsi_col};letter-spacing:1.5px;text-transform:uppercase;margin-top:4px;">{rsi_status}</div>
        </div>
        """, unsafe_allow_html=True)

        # ── Key stats panel ──
        st.markdown('<div class="section-label-flush">Key statistics</div>', unsafe_allow_html=True)
        high52 = ohlcv["Close"].rolling(min(252,len(ohlcv))).max().iloc[-1]
        low52  = ohlcv["Close"].rolling(min(252,len(ohlcv))).min().iloc[-1]
        dist_hi = (last["Close"]/high52-1)*100
        dist_lo = (last["Close"]/low52-1)*100
        beta_raw = ohlcv["Close"].pct_change().dropna()

        stats = [
            ("52W High",    f"${high52:.2f}"),
            ("52W Low",     f"${low52:.2f}"),
            ("Dist. to high",f"{dist_hi:+.1f}%"),
            ("SMA 20",      f"${last['SMA20']:.2f}" if not np.isnan(last['SMA20']) else "—"),
            ("SMA 50",      f"${last['SMA50']:.2f}" if not np.isnan(last['SMA50']) else "—"),
            ("SMA 200",     f"${last['SMA200']:.2f}" if not np.isnan(last['SMA200']) else "—"),
            ("Drawdown",    f"{last['Drawdown']:.1f}%"),
            ("Vol 20d ann.", f"{last['VolAnn']:.1f}%"),
        ]
        rows_html = ""
        for lbl, val in stats:
            rows_html += f"""
            <tr style="border-bottom:1px solid #e0dbd2;">
              <td style="padding:7px 10px;font-size:12px;color:#6b6760;">{lbl}</td>
              <td style="padding:7px 10px;font-size:12px;font-family:'DM Mono',monospace;
                         text-align:right;color:#1a1814;font-weight:500;">{val}</td>
            </tr>"""
        st.markdown(f"""
        <div style="border:1px solid #e0dbd2;border-radius:4px;overflow:hidden;">
          <table style="width:100%;border-collapse:collapse;background:#faf8f4;font-family:'DM Sans',sans-serif;">
            <tbody>{rows_html}</tbody>
          </table>
        </div>
        """, unsafe_allow_html=True)

        # Signal interpretation
        above_200 = last["Close"] > last["SMA200"] if not np.isnan(last["SMA200"]) else None
        macd_bull  = last["MACD"] > last["Signal"]
        st.markdown(f"""
        <div class="interp-box" style="margin-top:14px;">
          <strong style="color:var(--ink);">{primary}</strong> is trading
          {'<strong>above</strong>' if above_200 else '<strong>below</strong>'} its 200-day SMA —
          {'a bullish long-term signal.' if above_200 else 'a bearish long-term signal.'}<br/>
          MACD is {'<strong>bullish</strong> (above signal line)' if macd_bull else '<strong>bearish</strong> (below signal line)'}.
          RSI at {rsi_val:.0f} signals {rsi_status.lower()} conditions.
        </div>
        """, unsafe_allow_html=True)


# ═════════════════════════════════════════════════════════════════════
# TAB 2 — SECTOR X-RAY
# ═════════════════════════════════════════════════════════════════════
with t2:
    st.markdown('<div class="section-label">Sector X-Ray — performance & risk across all S&amp;P 500 sectors</div>', unsafe_allow_html=True)

    with st.spinner("Loading sector data…"):
        sec_df = get_sector_perf()

    if sec_df.empty:
        st.info("Sector data unavailable.")
    else:
        sl, sr = st.columns([1.5, 1], gap="large")

        with sl:
            # ── Horizontal bar chart — 1Y, 3M, 1M returns ──
            period_choice = st.radio("Return period", ["1Y%","3M%","1M%"], horizontal=True, key="sec_period")
            sec_sorted = sec_df.sort_values(period_choice, ascending=True)

            bar_colors = [C["green"] if v>=0 else C["sienna"] for v in sec_sorted[period_choice]]
            fig_sec = go.Figure(go.Bar(
                x=sec_sorted[period_choice], y=sec_sorted["Sector"],
                orientation="h",
                marker_color=bar_colors,
                text=[f"{v:+.1f}%" for v in sec_sorted[period_choice]],
                textposition="outside",
                textfont=dict(size=11, family="DM Mono"),
                hovertemplate="%{y}: %{x:.2f}%<extra></extra>",
            ))
            fig_sec.add_vline(x=0, line_color=C["rule"], line_width=1.5)
            fig_sec.update_layout(**plot_base(400, f"Sector returns ({period_choice.replace('%','').replace('Y',' Year').replace('M',' Month')})", legend=False))
            fig_sec.update_xaxes(ticksuffix="%")
            st.plotly_chart(fig_sec, use_container_width=True)

            # ── Bubble: return vs volatility ──
            fig_bubble = go.Figure()
            for i, row in sec_df.iterrows():
                r   = row["1Y%"]
                v   = row["Vol%"]
                color = PALETTE[i % len(PALETTE)]
                fig_bubble.add_trace(go.Scatter(
                    x=[v], y=[r], mode="markers+text",
                    name=row["Sector"],
                    marker=dict(size=22, color=color,
                                line=dict(color=C["cream"],width=2)),
                    text=[row["ETF"]], textposition="top center",
                    textfont=dict(size=10),
                    hovertemplate=(f"<b>{row['Sector']}</b><br>"
                                   f"1Y: {r:.1f}%<br>Vol: {v:.1f}%<extra></extra>"),
                ))
            fig_bubble.add_hline(y=0, line_color=C["rule"], line_width=1)
            fig_bubble.update_layout(**plot_base(380, "Return vs volatility (each bubble = sector ETF)"))
            fig_bubble.update_xaxes(title_text="Annualised volatility (%)", ticksuffix="%")
            fig_bubble.update_yaxes(title_text="1-year return (%)",         ticksuffix="%")
            st.plotly_chart(fig_bubble, use_container_width=True)

        with sr:
            st.markdown('<div class="section-label-flush">Sector heat table</div>', unsafe_allow_html=True)
            sec_sorted_v = sec_df.sort_values("1Y%", ascending=False)
            rows_h = ""
            for _, row in sec_sorted_v.iterrows():
                def clr(v):
                    if v >= 10:  return f"background:#eef5e8;color:#3a6b1a"
                    if v >= 0:   return f"background:#f5f9f0;color:#3a6b1a"
                    if v >= -10: return f"background:#fdf4ef;color:#8b3a1a"
                    return f"background:#f7ece3;color:#8b3a1a"
                rows_h += f"""
                <tr style="border-bottom:1px solid #e0dbd2;">
                  <td style="padding:8px 10px;font-size:13px;font-weight:500;">{row['Sector']}</td>
                  <td style="padding:8px 10px;font-size:11px;color:#9e8060;font-family:'DM Mono',monospace;">{row['ETF']}</td>
                  <td style="padding:8px 10px;text-align:right;font-size:12px;{clr(row['1M%'])};font-family:'DM Mono',monospace;">{row['1M%']:+.1f}%</td>
                  <td style="padding:8px 10px;text-align:right;font-size:12px;{clr(row['3M%'])};font-family:'DM Mono',monospace;">{row['3M%']:+.1f}%</td>
                  <td style="padding:8px 10px;text-align:right;font-size:12px;{clr(row['1Y%'])};font-family:'DM Mono',monospace;">{row['1Y%']:+.1f}%</td>
                  <td style="padding:8px 10px;text-align:right;font-size:12px;color:#6b6760;font-family:'DM Mono',monospace;">{row['Vol%']:.1f}%</td>
                </tr>"""
            st.markdown(f"""
            <div style="overflow-x:auto;border:1px solid #e0dbd2;border-radius:4px;">
              <table style="width:100%;border-collapse:collapse;background:#faf8f4;font-family:'DM Sans',sans-serif;">
                <thead>
                  <tr style="background:#f2ede4;border-bottom:2px solid #e0dbd2;">
                    <th style="padding:8px 10px;text-align:left;font-size:10px;letter-spacing:2px;text-transform:uppercase;color:#6b6760;font-weight:500;">Sector</th>
                    <th style="padding:8px 10px;font-size:10px;letter-spacing:2px;text-transform:uppercase;color:#6b6760;font-weight:500;">ETF</th>
                    <th style="padding:8px 10px;text-align:right;font-size:10px;letter-spacing:2px;text-transform:uppercase;color:#6b6760;font-weight:500;">1M</th>
                    <th style="padding:8px 10px;text-align:right;font-size:10px;letter-spacing:2px;text-transform:uppercase;color:#6b6760;font-weight:500;">3M</th>
                    <th style="padding:8px 10px;text-align:right;font-size:10px;letter-spacing:2px;text-transform:uppercase;color:#6b6760;font-weight:500;">1Y</th>
                    <th style="padding:8px 10px;text-align:right;font-size:10px;letter-spacing:2px;text-transform:uppercase;color:#6b6760;font-weight:500;">Vol</th>
                  </tr>
                </thead>
                <tbody>{rows_h}</tbody>
              </table>
            </div>
            """, unsafe_allow_html=True)
            st.markdown("""
            <div class="interp-box interp-box-stone" style="margin-top:14px;">
              Green cells = positive return; orange = negative.
              The bubble chart reveals which sectors offer the best
              <em>return per unit of risk</em> — look for bubbles in the top-left quadrant.
            </div>
            """, unsafe_allow_html=True)


# ═════════════════════════════════════════════════════════════════════
# TAB 3 — VOLATILITY LAB
# ═════════════════════════════════════════════════════════════════════
with t3:
    st.markdown(f'<div class="section-label">Volatility Lab — {primary}</div>', unsafe_allow_html=True)

    vl, vr = st.columns([1.6, 1], gap="large")
    with vl:
        # ── Rolling vol surface: window × time ──
        windows   = [5, 10, 20, 40, 63, 126]
        rets      = ohlcv["Close"].pct_change().dropna()
        surf_data = []
        for w in windows:
            rv = rets.rolling(w).std() * np.sqrt(252) * 100
            surf_data.append(rv.dropna().values[-min(200,len(rv.dropna())):])

        # Align lengths
        min_len = min(len(x) for x in surf_data)
        surf_mat = np.array([x[-min_len:] for x in surf_data])
        t_axis   = list(range(min_len))

        fig_surf = go.Figure(go.Surface(
            x=t_axis, y=windows, z=surf_mat,
            colorscale=[[0,C["cream"]],[0.4,C["sand"]],[0.7,C["terra"]],[1,C["sienna"]]],
            showscale=True,
            contours=dict(z=dict(show=True, usecolormap=True, project_z=True)),
            colorbar=dict(title="Vol %", tickfont=dict(size=10), len=0.7),
            hovertemplate="Window %{y}d, T-%{x}: %{z:.1f}%<extra></extra>",
        ))
        fig_surf.update_layout(
            height=440, paper_bgcolor=C["cream"],
            font=dict(family="DM Sans", color=C["ink"], size=11),
            margin=dict(l=0,r=0,t=36,b=0),
            scene=dict(
                xaxis_title="Days ago (0 = today)",
                yaxis_title="Rolling window (days)",
                zaxis_title="Ann. volatility (%)",
                xaxis=dict(gridcolor=C["rule"],autorange="reversed"),
                yaxis=dict(gridcolor=C["rule"]),
                zaxis=dict(gridcolor=C["rule"]),
                bgcolor=C["cream"],
            ),
            title=dict(text="Volatility surface — rolling window × time",
                       font=dict(family="DM Serif Display",size=15,color=C["ink"])),
        )
        st.plotly_chart(fig_surf, use_container_width=True)

        st.markdown("""
        <div class="interp-box interp-box-sand">
          The 3D surface shows how volatility estimates change depending on the lookback window you choose.
          Short windows (5-day) are noisy but react fast to shocks; longer windows (63-day, 126-day) are
          smoother but lag. A spike across all windows simultaneously signals a true volatility regime shift.
        </div>
        """, unsafe_allow_html=True)

    with vr:
        # ── Volatility regime detection ──
        st.markdown('<div class="section-label-flush">Volatility regime</div>', unsafe_allow_html=True)

        vol20  = float(ohlcv["VolAnn"].iloc[-1]) if not np.isnan(ohlcv["VolAnn"].iloc[-1]) else 20
        vol_hist_mean = float(ohlcv["VolAnn"].dropna().mean())
        vol_hist_std  = float(ohlcv["VolAnn"].dropna().std())
        z_score = (vol20 - vol_hist_mean) / vol_hist_std if vol_hist_std > 0 else 0

        if z_score > 1.0:
            regime, reg_cls, reg_desc = "High volatility", "regime-high", "Current vol is significantly above historical average — elevated uncertainty or recent shock."
        elif z_score < -0.5:
            regime, reg_cls, reg_desc = "Low volatility",  "regime-low",  "Vol is below its historical mean — calm market conditions, but reversals can be sharp."
        else:
            regime, reg_cls, reg_desc = "Normal volatility","regime-mid", "Vol is near historical average. Monitor for breakouts in either direction."

        st.markdown(f"""
        <div style="text-align:center;padding:20px 0;">
          <span class="regime-badge {reg_cls}">{regime}</span>
          <div style="margin-top:14px;font-size:13px;color:var(--ink-muted);line-height:1.6;">
            {reg_desc}
          </div>
        </div>
        """, unsafe_allow_html=True)

        # Volatility time series comparison (multiple windows)
        fig_vt = go.Figure()
        for w, color in [(20,C["terra"]),(63,C["stone"]),(126,C["sand_dark"])]:
            rv = rets.rolling(w).std()*np.sqrt(252)*100
            fig_vt.add_trace(go.Scatter(
                x=rv.index, y=rv.values, name=f"{w}d vol",
                line=dict(color=color, width=1.8), mode="lines",
            ))
        fig_vt.add_hline(y=vol_hist_mean, line_color=C["rule"], line_dash="dot",
                         annotation_text="Hist. avg", annotation_font_color=C["muted"])
        fig_vt.update_layout(**plot_base(280, "Rolling volatility — 3 windows"))
        fig_vt.update_yaxes(title_text="Ann. vol (%)", ticksuffix="%")
        st.plotly_chart(fig_vt, use_container_width=True)

        # Vol distribution histogram
        vol_vals = ohlcv["VolAnn"].dropna().values
        fig_vh = go.Figure()
        fig_vh.add_trace(go.Histogram(
            x=vol_vals, nbinsx=30,
            marker_color=C["terra"], opacity=0.75, name="Vol distribution",
            hovertemplate="Vol: %{x:.1f}%  Count: %{y}<extra></extra>",
        ))
        fig_vh.add_vline(x=vol20, line_color=C["ink"], line_width=2,
                         annotation_text=f"Now: {vol20:.1f}%",
                         annotation_font_color=C["ink"])
        fig_vh.add_vline(x=vol_hist_mean, line_color=C["stone"], line_width=1.5, line_dash="dot",
                         annotation_text=f"Avg: {vol_hist_mean:.1f}%",
                         annotation_font_color=C["stone"])
        fig_vh.update_layout(**plot_base(220, "Historical vol distribution", legend=False))
        fig_vh.update_xaxes(title_text="Ann. vol (%)", ticksuffix="%")
        fig_vh.update_yaxes(title_text="Count")
        st.plotly_chart(fig_vh, use_container_width=True)


# ═════════════════════════════════════════════════════════════════════
# TAB 4 — CROSS-SECTION SCANNER
# ═════════════════════════════════════════════════════════════════════
with t4:
    st.markdown('<div class="section-label">Cross-section scanner — leaders &amp; laggards</div>', unsafe_allow_html=True)

    if sector_f != "All":
        universe = sp500_df[sp500_df["Sector"]==sector_f]["Symbol"].tolist()[:80]
    else:
        universe = all_syms[:60]

    st.markdown(f'<p style="font-size:13px;color:var(--ink-muted);margin-bottom:16px;">Scanning <strong>{len(universe)}</strong> tickers. This may take a moment — results are cached for 30 minutes.</p>', unsafe_allow_html=True)

    with st.spinner("Running cross-sectional scan…"):
        scan_rows = []
        raw_scan = yf.download(universe[:50], period="1y", auto_adjust=True, progress=False, group_by="ticker")
        for sym in universe[:50]:
            try:
                close = raw_scan[(sym,"Close")] if isinstance(raw_scan.columns,pd.MultiIndex) else raw_scan["Close"]
                close = close.dropna()
                if len(close) < 30: continue
                rets_s = close.pct_change().dropna()
                r1y    = (close.iloc[-1]/close.iloc[0]-1)*100
                vol_s  = rets_s.std()*np.sqrt(252)*100
                sharpe = (rets_s.mean()*252)/(rets_s.std()*np.sqrt(252)) if rets_s.std()>0 else 0
                sma200 = close.rolling(200).mean().iloc[-1]
                vs200  = (close.iloc[-1]/sma200-1)*100 if not np.isnan(sma200) else 0
                rm     = close.cummax(); dd = (close/rm-1).min()*100
                delta  = close.diff(); g=delta.clip(lower=0); l=-delta.clip(upper=0)
                rs_s   = g.rolling(14).mean()/l.rolling(14).mean().replace(0,np.nan)
                rsi_s  = float((100-100/(1+rs_s)).iloc[-1])
                scan_rows.append({"Symbol":sym,"1Y%":r1y,"Vol%":vol_s,"Sharpe":sharpe,
                                  "RSI":rsi_s,"MaxDD%":dd,"vs200%":vs200})
            except: continue

    if not scan_rows:
        st.info("Scanner returned no results. Try refreshing.")
    else:
        scan_df = pd.DataFrame(scan_rows)
        metric_col = {
            "1Y return":"1Y%","Volatility":"Vol%","RSI":"RSI",
            "Sharpe ratio":"Sharpe","Max drawdown":"MaxDD%",
            "Price vs 200-day SMA":"vs200%",
        }[scan_metric]

        desc = scan_metric not in ["Max drawdown"]
        top_n = scan_df.nlargest(scan_n, metric_col) if desc else scan_df.nsmallest(scan_n, metric_col)
        bot_n = scan_df.nsmallest(scan_n, metric_col) if desc else scan_df.nlargest(scan_n, metric_col)

        s4l, s4r = st.columns(2, gap="large")

        with s4l:
            # Dot-plot style ranking chart — TOP
            top_sorted = top_n.sort_values(metric_col, ascending=True)
            colors_top = [C["green"] if v>=0 else C["sienna"] for v in top_sorted[metric_col]]
            fig_top = go.Figure()
            fig_top.add_trace(go.Scatter(
                x=top_sorted[metric_col], y=top_sorted["Symbol"],
                mode="markers+text",
                marker=dict(size=14, color=colors_top,
                            line=dict(color=C["cream"],width=2)),
                text=[f"{v:.1f}" for v in top_sorted[metric_col]],
                textposition="middle right", textfont=dict(size=10,family="DM Mono"),
                hovertemplate="%{y}: %{x:.2f}<extra></extra>",
            ))
            # Connector lines to axis
            for i, (_, row) in enumerate(top_sorted.iterrows()):
                fig_top.add_shape(type="line",
                    x0=0, x1=row[metric_col], y0=i, y1=i,
                    line=dict(color=C["rule"],width=1.5))
            fig_top.add_vline(x=0, line_color=C["rule"], line_width=1)
            fig_top.update_layout(**plot_base(360, f"Top {scan_n} — {scan_metric}", legend=False))
            fig_top.update_xaxes(ticksuffix="%" if "%" in metric_col else "")
            st.plotly_chart(fig_top, use_container_width=True)

        with s4r:
            bot_sorted = bot_n.sort_values(metric_col, ascending=False)
            colors_bot = [C["green"] if v>=0 else C["sienna"] for v in bot_sorted[metric_col]]
            fig_bot = go.Figure()
            fig_bot.add_trace(go.Scatter(
                x=bot_sorted[metric_col], y=bot_sorted["Symbol"],
                mode="markers+text",
                marker=dict(size=14, color=colors_bot,
                            line=dict(color=C["cream"],width=2)),
                text=[f"{v:.1f}" for v in bot_sorted[metric_col]],
                textposition="middle right", textfont=dict(size=10,family="DM Mono"),
                hovertemplate="%{y}: %{x:.2f}<extra></extra>",
            ))
            for i, (_, row) in enumerate(bot_sorted.iterrows()):
                fig_bot.add_shape(type="line",
                    x0=0, x1=row[metric_col], y0=i, y1=i,
                    line=dict(color=C["rule"],width=1.5))
            fig_bot.add_vline(x=0, line_color=C["rule"], line_width=1)
            fig_bot.update_layout(**plot_base(360, f"Bottom {scan_n} — {scan_metric}", legend=False))
            fig_bot.update_xaxes(ticksuffix="%" if "%" in metric_col else "")
            st.plotly_chart(fig_bot, use_container_width=True)

        # Full scan table
        with st.expander("Full scan results"):
            st.dataframe(
                scan_df.sort_values(metric_col, ascending=False).reset_index(drop=True),
                use_container_width=True,
                column_config={
                    "1Y%":    st.column_config.NumberColumn("1Y Return %",   format="%.1f%%"),
                    "Vol%":   st.column_config.NumberColumn("Volatility %",  format="%.1f%%"),
                    "Sharpe": st.column_config.NumberColumn("Sharpe",        format="%.2f"),
                    "RSI":    st.column_config.NumberColumn("RSI",           format="%.1f"),
                    "MaxDD%": st.column_config.NumberColumn("Max Drawdown %",format="%.1f%%"),
                    "vs200%": st.column_config.NumberColumn("vs 200D SMA %", format="%.1f%%"),
                },
                height=300,
            )


# ═════════════════════════════════════════════════════════════════════
# TAB 5 — CORRELATION ANATOMY
# ═════════════════════════════════════════════════════════════════════
with t5:
    st.markdown('<div class="section-label">Correlation anatomy — how your tickers move together</div>', unsafe_allow_html=True)

    if closes.empty or closes.shape[1] < 2:
        st.info("Add at least 2 tickers in the sidebar to see correlations.")
    else:
        rets_multi = closes.pct_change().dropna()
        corr       = rets_multi.corr()
        syms_c     = corr.columns.tolist()

        ca_l, ca_r = st.columns([1.3, 1], gap="large")

        with ca_l:
            # ── Annotated heatmap ──
            cscale = [[0,C["stone"]],[0.25,C["sand"]],[0.5,C["cream"]],
                      [0.75,C["sand"]],[1,C["terra"]]]
            fig_hm = go.Figure(go.Heatmap(
                z=corr.values, x=syms_c, y=syms_c,
                colorscale=cscale, zmin=-1, zmax=1,
                text=[[f"{v:.2f}" for v in row] for row in corr.values],
                texttemplate="%{text}",
                textfont=dict(size=13, family="DM Mono", color=C["ink"]),
                colorbar=dict(title="ρ", tickfont=dict(size=10), len=0.8,
                              tickvals=[-1,-0.5,0,0.5,1]),
                hovertemplate="%{y} × %{x}: ρ = %{z:.3f}<extra></extra>",
            ))
            fig_hm.update_layout(
                height=420, paper_bgcolor=C["cream"], plot_bgcolor=C["cream"],
                font=dict(family="DM Sans", color=C["ink"], size=12),
                margin=dict(l=16,r=16,t=44,b=16),
                title=dict(text="Pairwise return correlation",
                           font=dict(family="DM Serif Display",size=15,color=C["ink"])),
                xaxis=dict(side="bottom", tickfont=dict(size=12,family="DM Mono")),
                yaxis=dict(tickfont=dict(size=12,family="DM Mono")),
            )
            st.plotly_chart(fig_hm, use_container_width=True)

            # ── Rolling correlation between primary and each other ──
            if primary in rets_multi.columns and len(syms_c) > 1:
                st.markdown('<div class="section-label-flush">Rolling 60-day correlation vs primary</div>', unsafe_allow_html=True)
                fig_rc = go.Figure()
                for i, sym in enumerate(syms_c):
                    if sym == primary: continue
                    rc = rets_multi[primary].rolling(60).corr(rets_multi[sym])
                    fig_rc.add_trace(go.Scatter(
                        x=rc.index, y=rc.values, name=sym, mode="lines",
                        line=dict(color=PALETTE[i%len(PALETTE)], width=1.8),
                        hovertemplate=f"{sym}: ρ = %{{y:.2f}}<extra></extra>",
                    ))
                fig_rc.add_hline(y=0,    line_color=C["rule"], line_width=1)
                fig_rc.add_hline(y=0.7,  line_color=C["sienna"], line_width=1, line_dash="dot",
                                 annotation_text="High correlation",annotation_font_color=C["sienna"])
                fig_rc.add_hline(y=-0.3, line_color=C["green"],  line_width=1, line_dash="dot",
                                 annotation_text="Negative",annotation_font_color=C["green"])
                fig_rc.update_layout(**plot_base(280, f"Rolling 60d correlation vs {primary}"))
                fig_rc.update_yaxes(title_text="ρ", range=[-1,1])
                st.plotly_chart(fig_rc, use_container_width=True)

        with ca_r:
            st.markdown('<div class="section-label-flush">Correlation insights</div>', unsafe_allow_html=True)

            # Extract upper triangle pairs
            pairs = []
            for i in range(len(syms_c)):
                for j in range(i+1, len(syms_c)):
                    pairs.append((syms_c[i], syms_c[j], corr.iloc[i,j]))
            pairs.sort(key=lambda x: x[2])

            if pairs:
                st.markdown(f"""
                <div style="font-size:10px;letter-spacing:2px;text-transform:uppercase;
                            color:var(--ink-muted);margin-bottom:10px;">Least correlated pairs</div>
                """, unsafe_allow_html=True)
                for a,b,r in pairs[:3]:
                    chip_cls = "chip-neg" if r < -0.2 else ("chip-neu" if r < 0.5 else "chip-pos")
                    st.markdown(f"""
                    <div style="display:flex;justify-content:space-between;align-items:center;
                                padding:8px 0;border-bottom:1px solid var(--rule);font-size:13px;">
                      <span style="font-family:'DM Mono',monospace;font-weight:500;">{a} × {b}</span>
                      <span class="chip chip-neu">ρ = {r:.2f}</span>
                    </div>
                    """, unsafe_allow_html=True)

                st.markdown(f"""
                <div style="font-size:10px;letter-spacing:2px;text-transform:uppercase;
                            color:var(--ink-muted);margin-bottom:10px;margin-top:18px;">Most correlated pairs</div>
                """, unsafe_allow_html=True)
                for a,b,r in pairs[-3:][::-1]:
                    st.markdown(f"""
                    <div style="display:flex;justify-content:space-between;align-items:center;
                                padding:8px 0;border-bottom:1px solid var(--rule);font-size:13px;">
                      <span style="font-family:'DM Mono',monospace;font-weight:500;">{a} × {b}</span>
                      <span class="chip chip-pos">ρ = {r:.2f}</span>
                    </div>
                    """, unsafe_allow_html=True)

            avg_corr = np.array([p[2] for p in pairs]).mean() if pairs else 0
            div_word = ("well diversified" if avg_corr < 0.4
                        else "moderately correlated" if avg_corr < 0.7
                        else "highly concentrated")
            st.markdown(f"""
            <div class="interp-box interp-box-sand" style="margin-top:20px;">
              Average pairwise correlation: <strong>ρ = {avg_corr:.2f}</strong>.<br/>
              Your selection is <strong>{div_word}</strong>.
              Low or negative correlations between holdings reduce portfolio volatility —
              when one falls, another may rise. High correlations offer less protection
              during broad market drawdowns.
            </div>
            """, unsafe_allow_html=True)

            # Scatter matrix for 3–5 tickers
            if 2 <= len(syms_c) <= 5 and len(rets_multi) > 20:
                st.markdown('<div class="section-label-flush" style="margin-top:20px;">Return scatter</div>', unsafe_allow_html=True)
                if len(syms_c) == 2:
                    s1, s2 = syms_c[0], syms_c[1]
                    fig_sc = go.Figure(go.Scatter(
                        x=rets_multi[s1]*100, y=rets_multi[s2]*100,
                        mode="markers",
                        marker=dict(size=5, color=C["terra"], opacity=0.5,
                                    line=dict(color=C["cream"],width=0.5)),
                        hovertemplate=f"{s1}: %{{x:.2f}}%<br>{s2}: %{{y:.2f}}%<extra></extra>",
                    ))
                    fig_sc.update_layout(**plot_base(260, f"{s1} vs {s2} daily returns", legend=False))
                    fig_sc.update_xaxes(title_text=f"{s1} daily return (%)", ticksuffix="%")
                    fig_sc.update_yaxes(title_text=f"{s2} daily return (%)", ticksuffix="%")
                    st.plotly_chart(fig_sc, use_container_width=True)


# ═════════════════════════════════════════════════════════════════════
# TAB 6 — RADAR MULTI-METRIC COMPARISON
# ═════════════════════════════════════════════════════════════════════
with t6:
    st.markdown('<div class="section-label">Radar comparison — multi-metric profile of each ticker</div>', unsafe_allow_html=True)

    st.markdown("""
    <p style="font-size:13px;color:var(--ink-muted);margin-bottom:16px;">
      Each axis of the radar shows a normalised percentile score (0–100) relative to the comparison group.
      Higher = better on every axis. Sharpe, return, and momentum are inverted where higher is better;
      volatility and drawdown are inverted so that <em>lower risk = higher score</em>.
    </p>
    """, unsafe_allow_html=True)

    radar_metrics = ["1Y Return", "Volatility ↓", "Sharpe", "RSI Momentum", "Max Drawdown ↓", "vs 200D SMA"]
    all_radar_tickers = [primary] + compare_syms[:5]

    radar_rows = {}
    for sym in all_radar_tickers:
        try:
            df_r = get_ohlcv(sym, period="1y")
            if len(df_r) < 50: continue
            rets_r = df_r["Close"].pct_change().dropna()
            r1y    = (df_r["Close"].iloc[-1]/df_r["Close"].iloc[0]-1)*100
            vol_r  = rets_r.std()*np.sqrt(252)*100
            sharpe_r = rets_r.mean()*252/(rets_r.std()*np.sqrt(252)) if rets_r.std()>0 else 0
            sma200_r = df_r["Close"].rolling(200).mean().iloc[-1]
            vs200_r  = (df_r["Close"].iloc[-1]/sma200_r-1)*100 if not np.isnan(sma200_r) else 0
            rm_r     = df_r["Close"].cummax()
            dd_r     = (df_r["Close"]/rm_r-1).min()*100
            delta_r  = df_r["Close"].diff()
            g_r = delta_r.clip(lower=0).rolling(14).mean()
            l_r = (-delta_r.clip(upper=0)).rolling(14).mean()
            rsi_r = float((100-100/(1+g_r/l_r.replace(0,np.nan))).iloc[-1])
            radar_rows[sym] = [r1y, vol_r, sharpe_r, rsi_r, dd_r, vs200_r]
        except: continue

    if len(radar_rows) < 2:
        st.info("Add comparison tickers in the sidebar to build the radar chart.")
    else:
        radar_df = pd.DataFrame(radar_rows, index=radar_metrics).T

        # Normalise: convert each metric to 0–100 percentile
        normed = radar_df.copy()
        for col in radar_df.columns:
            vals = radar_df[col].dropna()
            if len(vals) < 2: continue
            mn, mx = vals.min(), vals.max()
            if mx == mn: normed[col] = 50; continue
            pct = (radar_df[col]-mn)/(mx-mn)*100
            # Invert risk metrics so higher = better
            if col in ["Volatility ↓","Max Drawdown ↓"]:
                pct = 100 - pct
            normed[col] = pct

        rl, rr = st.columns([1.5, 1], gap="large")

        with rl:
            fig_rad = go.Figure()
            categories = radar_metrics + [radar_metrics[0]]  # close the polygon

            for i, (sym, row) in enumerate(normed.iterrows()):
                vals = row[radar_metrics].tolist() + [row[radar_metrics[0]]]
                hex_c = PALETTE[i % len(PALETTE)].lstrip("#")
                r_c, g_c, b_c = int(hex_c[0:2],16), int(hex_c[2:4],16), int(hex_c[4:6],16)
                fill_rgba = f"rgba({r_c},{g_c},{b_c},0.16)"
                fig_rad.add_trace(go.Scatterpolar(
                    r=vals, theta=categories,
                    name=sym, mode="lines+markers",
                    line=dict(color=PALETTE[i%len(PALETTE)], width=2.5),
                    marker=dict(size=6, color=PALETTE[i%len(PALETTE)],
                                line=dict(color=C["cream"],width=1.5)),
                    fill="toself",
                    fillcolor=fill_rgba,
                    hovertemplate=f"<b>{sym}</b><br>%{{theta}}: %{{r:.0f}}/100<extra></extra>",
                ))

            fig_rad.update_layout(
                height=480, paper_bgcolor=C["cream"],
                font=dict(family="DM Sans", color=C["ink"], size=12),
                margin=dict(l=40,r=40,t=60,b=40),
                polar=dict(
                    bgcolor=C["cream"],
                    radialaxis=dict(
                        visible=True, range=[0,100],
                        tickfont=dict(size=9,color=C["muted"]),
                        gridcolor=C["rule"], linecolor=C["rule"],
                        tickvals=[0,25,50,75,100],
                        ticktext=["0","25","50","75","100"],
                    ),
                    angularaxis=dict(
                        tickfont=dict(size=12, family="DM Sans", color=C["ink"]),
                        gridcolor=C["rule"], linecolor=C["rule"],
                    ),
                    gridshape="linear",
                ),
                legend=dict(orientation="h", y=-0.08, x=0.5, xanchor="center",
                            font=dict(size=12)),
                title=dict(text="Multi-metric percentile radar",
                           font=dict(family="DM Serif Display",size=15,color=C["ink"])),
            )
            st.plotly_chart(fig_rad, use_container_width=True)

        with rr:
            st.markdown('<div class="section-label-flush">Percentile scores (0–100)</div>', unsafe_allow_html=True)

            # Table of raw values + percentiles
            for metric in radar_metrics:
                st.markdown(f"""
                <div style="margin-bottom:14px;">
                  <div style="font-size:10px;letter-spacing:2px;text-transform:uppercase;
                              color:var(--ink-muted);margin-bottom:6px;">{metric}</div>
                """, unsafe_allow_html=True)
                row_vals = [(sym, float(normed.loc[sym,metric]),
                             float(radar_df.loc[sym,metric]))
                            for sym in normed.index if sym in normed.index]
                row_vals.sort(key=lambda x:-x[1])
                for sym, pct, raw in row_vals:
                    color = PALETTE[list(normed.index).index(sym) % len(PALETTE)]
                    raw_str = f"{raw:.1f}%" if "%" in metric or metric in ["1Y Return","Volatility ↓","Max Drawdown ↓","vs 200D SMA"] else f"{raw:.2f}"
                    st.markdown(f"""
                    <div style="display:flex;align-items:center;gap:8px;margin-bottom:5px;">
                      <span style="font-family:'DM Mono',monospace;font-size:11px;
                                   font-weight:600;color:{color};min-width:44px;">{sym}</span>
                      <div style="flex:1;background:var(--rule);border-radius:2px;height:6px;">
                        <div style="width:{pct:.0f}%;background:{color};height:6px;border-radius:2px;"></div>
                      </div>
                      <span style="font-size:11px;font-family:'DM Mono',monospace;
                                   color:var(--ink-muted);min-width:42px;text-align:right;">{raw_str}</span>
                    </div>
                    """, unsafe_allow_html=True)
                st.markdown("</div>", unsafe_allow_html=True)

            st.markdown("""
            <div class="interp-box interp-box-stone">
              Scores are percentile ranks within the comparison group — not absolute.
              A score of 80 on <em>Volatility ↓</em> means this ticker is calmer than
              80% of the others being compared. The radar makes asymmetries immediately visible:
              a ticker that looks good on return but collapses on the drawdown axis deserves scrutiny.
            </div>
            """, unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────
# FOOTER
# ─────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="site-footer">
  Chaouat Economics Lab — Market Data · Data via Yahoo Finance · Educational use only<br/>
  <span style="font-size:11px;">© Chaouat Economics Lab · All analysis is illustrative and not financial advice</span>
</div>
""", unsafe_allow_html=True)
