import math
import datetime as dt
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.io as pio
import streamlit as st
import yfinance as yf
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import time

def get_query_params():
    try:
        return st.query_params
    except Exception:
        return st.experimental_get_query_params()

@st.cache_data(ttl=24*60*60)
def load_sp500_df() -> pd.DataFrame:
    url = "https://raw.githubusercontent.com/datasets/s-and-p-500-companies/main/data/constituents.csv"
    df = pd.read_csv(url)

    # Normalise les noms de colonnes (trim)
    df = df.rename(columns={c: c.strip() for c in df.columns})

    # --- Sector (garde "Sector" si présent, sinon crée-le) ---
    # Accepte aussi "GICS Sector" si c'est ce qu'on trouve
    sector_col = None
    for c in df.columns:
        cl = c.strip().lower()
        if cl in {"sector", "gics sector"}:
            sector_col = c
            break
    if sector_col is None:
        df["Sector"] = "Unknown"
    else:
        if sector_col != "Sector":
            df = df.rename(columns={sector_col: "Sector"})
        df["Sector"] = df["Sector"].fillna("Unknown").astype(str)

    # --- Symbol / Ticker ---
    sym_col = None
    for c in df.columns:
        if c.strip().lower() in {"symbol", "ticker"}:
            sym_col = c
            break
    if sym_col is None:
        raise ValueError("Le fichier S&P 500 ne contient pas de colonne Symbol/Ticker.")
    if sym_col != "Symbol":
        df = df.rename(columns={sym_col: "Symbol"})
    df["Symbol"] = df["Symbol"].astype(str).str.upper().str.strip()

    # --- Name (accepte plusieurs alias, sinon fabrique à partir de Symbol) ---
    name_aliases = ["Name", "Security", "Company", "Company Name"]
    name_col = next((c for c in name_aliases if c in df.columns), None)
    if name_col is None:
        df["Name"] = df["Symbol"]
    elif name_col != "Name":
        df = df.rename(columns={name_col: "Name"})
    df["Name"] = df["Name"].astype(str).str.strip()

    # Ne garde que les colonnes disponibles parmi Symbol/Name/Sector
    keep = [c for c in ["Symbol", "Name", "Sector"] if c in df.columns]
    return df[keep]


@st.cache_data(ttl=6*60*60)
def get_company_meta(sym: str):
    """Return (company_name, logo_url) using S&P list/yfinance + Clearbit fallback."""
    name, logo = None, None

    # Try S&P 500 list for name
    try:
        df = load_sp500_df()
        hit = df.loc[df["Symbol"].str.upper() == sym.upper()]
        if not hit.empty:
            name = hit.iloc[0]["Name"]
    except Exception:
        pass

    # Try yfinance for better name + website
    website = None
    try:
        info = yf.Ticker(sym).get_info()
        name = info.get("longName") or info.get("shortName") or name
        website = info.get("website") or info.get("website_url")
        # yfinance sometimes DOES provide logo_url; use it if present
        logo = info.get("logo_url") or logo
    except Exception:
        pass

    # Build a robust logo URL from the website domain using Clearbit
    if not logo and website:
        try:
            from urllib.parse import urlparse
            u = urlparse(website if website.startswith("http") else f"https://{website}")
            host = (u.netloc or u.path).split("/")[0].replace("www.", "")
            if host:
                logo = f"https://logo.clearbit.com/{host}"
        except Exception:
            pass

    if not name:
        name = sym.upper()

    return name, logo

@st.cache_data(ttl=60*60)
def scan_volatility(tickers: list[str], lookback_days: int, use_log: bool) -> pd.DataFrame:
    """
    Returns a DataFrame with columns: ['Ticker','AnnVol'] for the given tickers.
    Volatility is std of daily returns over the last `lookback_days`, annualized by sqrt(252).
    """
    # Convert trading days to calendar days (roughly)
    cal_days = int(lookback_days * 1.6) + 10  # buffer
    start = dt.date.today() - dt.timedelta(days=cal_days)

    # Batch download to keep it fast; auto_adjust gives split/div adjusted close
    data = yf.download(
        tickers,
        start=start,
        end=dt.date.today(),
        interval="1d",
        auto_adjust=True,
        progress=False,
        group_by="ticker",
        threads=True
    )

    out = []
    TRD = 252

    # If a single ticker sneaks in, yfinance returns a single-level frame. Normalize access.
    for sym in tickers:
        try:
            if isinstance(data.columns, pd.MultiIndex):
                close = data[(sym, "Close")].dropna()
            else:
                # Single-level fallback; happens if tickers has length 1
                close = data["Close"].dropna()

            if close.size < lookback_days + 5:
                continue

            ret = np.log(close).diff() if use_log else close.pct_change()
            ret = ret.dropna().tail(lookback_days)
            if ret.empty:
                continue

            vol_daily = float(ret.std())
            ann_vol = vol_daily * math.sqrt(TRD)
            out.append((sym, ann_vol))
        except Exception:
            # Skip problematic symbols gracefully
            continue

    df = pd.DataFrame(out, columns=["Ticker", "AnnVol"]).dropna()
    df = df.sort_values("AnnVol", ascending=False).reset_index(drop=True)
    return df
    
@st.cache_data(ttl=30*60, show_spinner=True)
def scan_cross_section(metric: str, universe: list[str], lookback_days: int, use_log: bool=True) -> pd.DataFrame:
    """
    Return a DataFrame with columns ['Ticker','Value'] for the chosen metric,
    computed cross-sectionally across `universe`.

    Supported metrics:
      - 'Volatility' (annualized over lookback_days)
      - 'Price change' (over lookback_days)
      - 'YTD return'
      - '52W change' (last 252 trading days)
      - 'Max drawdown' (min drawdown over last 252 trading days)
      - 'RSI (14)' (latest)
      - 'Distance to 52W high' (last price / 52W high - 1)
      - 'Sharpe (1y)' (mean/sd over last 252 returns)
      - 'Beta vs S&P 500' (cov/var vs ^GSPC over last 252 returns)
    """
    if not universe:
        return pd.DataFrame(columns=["Ticker", "Value"])

    # Enough data for all metrics without being huge
    cal_days = int(max(lookback_days, 60) * 1.6) + 20
    start = dt.date.today() - dt.timedelta(days=max(420, cal_days))
    end = dt.date.today()

    data = yf.download(
        universe, start=start, end=end, interval="1d",
        auto_adjust=True, progress=False, group_by="ticker", threads=True
    )

    # Market for beta
    mkt_close = yf.download("^GSPC", start=start, end=end, auto_adjust=True, progress=False)["Close"]
    mkt_ret = mkt_close.pct_change()

    rows = []
    for sym in universe:
        try:
            close = data[(sym, "Close")] if isinstance(data.columns, pd.MultiIndex) else data["Close"]
            close = close.dropna()
            if close.size < 30:
                continue

            if metric == "Volatility":
                r = np.log(close).diff() if use_log else close.pct_change()
                val = float(r.dropna().tail(lookback_days).std() * np.sqrt(252))
            elif metric == "Price change":
                c = close.tail(lookback_days + 1)
                if c.size < 2: 
                    continue
                val = float(c.iloc[-1] / c.iloc[0] - 1.0)
            elif metric == "YTD return":
                y0 = dt.date(dt.date.today().year, 1, 1)
                c = close[close.index >= pd.to_datetime(y0)]
                if c.size < 2: 
                    continue
                val = float(c.iloc[-1] / c.iloc[0] - 1.0)
            elif metric == "52W change":
                c = close.tail(252)
                if c.size < 2:
                    continue
                val = float(c.iloc[-1] / c.iloc[0] - 1.0)
            elif metric == "Max drawdown":
                c = close.tail(252)
                roll_max = c.rolling(252, min_periods=1).max()
                dd = c / roll_max - 1.0
                val = float(dd.min())   # negative number
            elif metric == "RSI (14)":
                delta = close.diff()
                gain = delta.clip(lower=0.0)
                loss = -delta.clip(upper=0.0)
                rs = gain.rolling(14).mean() / loss.rolling(14).mean().replace(0, np.nan)
                rsi = 100 - (100 / (1 + rs))
                val = float(rsi.dropna().iloc[-1])
            elif metric == "Distance to 52W high":
                c = close.tail(252)
                high = c.max()
                val = float(c.iloc[-1] / high - 1.0)   # negative if below high
            elif metric == "Sharpe (1y)":
                r = close.pct_change().dropna().tail(252)
                if r.std() == 0:
                    continue
                val = float((r.mean() * np.sqrt(252)) / r.std())
            elif metric == "Beta vs S&P 500":
                r = close.pct_change().dropna().tail(252)
                common = r.index.intersection(mkt_ret.index)
                v = mkt_ret.loc[common].var()
                if v == 0:
                    continue
                val = float(r.loc[common].cov(mkt_ret.loc[common]) / v)
            else:
                continue

            rows.append((sym, val))
        except Exception:
            continue

    return pd.DataFrame(rows, columns=["Ticker", "Value"]).dropna()



@st.cache_data(show_spinner=True)
def fetch_history_multi(syms: list[str], start_d, end_d):
    """Batch download and return a tidy DataFrame: Date, Ticker, Close."""
    if not syms:
        return pd.DataFrame(columns=["Date", "Ticker", "Close"])
    data = yf.download(
        syms, start=start_d, end=end_d, interval="1d",
        auto_adjust=True, progress=False, group_by="ticker", threads=True
    )
    rows = []
    if isinstance(data.columns, pd.MultiIndex):
        for s in syms:
            try:
                close = data[(s, "Close")].dropna()
                df_s = close.rename("Close").reset_index().rename(columns={"index": "Date"})
                df_s["Ticker"] = s
                rows.append(df_s[["Date", "Ticker", "Close"]])
            except Exception:
                pass
    else:
        # Single-level: only one ticker
        close = data["Close"].dropna()
        df_s = close.rename("Close").reset_index().rename(columns={"index": "Date"})
        df_s["Ticker"] = syms[0]
        rows.append(df_s[["Date", "Ticker", "Close"]])
    if not rows:
        return pd.DataFrame(columns=["Date", "Ticker", "Close"])
    out = pd.concat(rows, ignore_index=True)
    out["Date"] = pd.to_datetime(out["Date"], errors="coerce")
    out = out.dropna(subset=["Date", "Close"])
    return out


def compute_rolling_vol(df_prices: pd.DataFrame, window: int, use_log: bool) -> pd.DataFrame:
    """
    Return tidy rolling annualized volatility with columns: Date, Ticker, AnnVol.
    Uses transform() to keep index alignment stable.
    """
    if df_prices.empty:
        return pd.DataFrame(columns=["Date", "Ticker", "AnnVol"])

    df = df_prices.sort_values(["Ticker", "Date"]).copy()

    # returns
    if use_log:
        df["ret"] = df.groupby("Ticker")["Close"].transform(lambda s: np.log(s).diff())
    else:
        df["ret"] = df.groupby("Ticker")["Close"].pct_change()

    # rolling daily vol, keep alignment with transform
    df["vol_daily"] = df.groupby("Ticker")["ret"].transform(lambda s: s.rolling(window).std())
    df["AnnVol"] = df["vol_daily"] * math.sqrt(252)

    # keep only what we need, drop rows where we don't have vol yet
    out = df.dropna(subset=["AnnVol"])[["Date", "Ticker", "AnnVol"]]
    return out
    
def compute_indicators(df_prices: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """
    Return a dict[ticker] -> DataFrame(Date index) with:
    Close, Return, SMA50, SMA200, Drawdown, RSI14.
    """
    out: dict[str, pd.DataFrame] = {}
    if df_prices.empty:
        return out

    df = df_prices.sort_values(["Ticker", "Date"]).copy()
    for tkr, g in df.groupby("Ticker"):
        s = g.set_index("Date")["Close"].astype(float)
        ret = s.pct_change()

        sma50 = s.rolling(50).mean()
        sma200 = s.rolling(200).mean()

        # 52w rolling max for drawdown
        roll_max = s.rolling(252, min_periods=1).max()
        dd = (s / roll_max) - 1.0

        # RSI(14)
        delta = s.diff()
        gain = delta.clip(lower=0.0)
        loss = -delta.clip(upper=0.0)
        avg_gain = gain.rolling(14).mean()
        avg_loss = loss.rolling(14).mean()
        rs = avg_gain / avg_loss.replace(0, np.nan)
        rsi = 100.0 - (100.0 / (1.0 + rs))

        out[tkr] = pd.DataFrame(
            {"Close": s, "Return": ret, "SMA50": sma50, "SMA200": sma200,
             "Drawdown": dd, "RSI14": rsi}
        )
    return out


def load_tickers():
    # S&P 500 symbols (fast and lightweight). You can swap this URL later.
    url = "https://raw.githubusercontent.com/datasets/s-and-p-500-companies/main/data/constituents.csv"
    df = pd.read_csv(url)
    syms = sorted(df["Symbol"].dropna().unique().tolist())
    # Guarantee AAPL exists as a default
    if "AAPL" not in syms:
        syms.insert(0, "AAPL")
    return syms

# --- Page and style setup ---
st.set_page_config(page_title="Chaouat Finance", page_icon="💹", layout="wide")
st.markdown('<div class="cf-sticky">Chaouat Finance</div>', unsafe_allow_html=True)
pio.templates.default = "plotly_white"

# --- Custom CSS: Montserrat + cerulean design + header ---
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Montserrat:wght@400;600;700&display=swap');

:root{
  --primary:#007BA7;        /* cerulean */
  --primary-dark:#005F7D;
  --accent:#00B0FF;
  --bg:#F7FAFC;
  --card:#FFFFFF;
  --text:#0F172A;
  --muted:#64748B;
}

html, body, * {
  font-family: 'Montserrat', sans-serif !important;
}
[data-testid="stAppViewContainer"],
[data-testid="stSidebar"],
[data-testid="stMarkdownContainer"] * {
  font-family: 'Montserrat', sans-serif !important;
}

.cf-hero{
  background: linear-gradient(135deg, var(--primary) 0%, var(--accent) 100%);
  color: #fff;
  padding: 28px 32px;
  border-radius: 20px;
  box-shadow: 0 8px 24px rgba(0,123,167,.25);
  margin: 8px 0 24px 0;
}
.cf-brand{
  font-weight: 700;
  font-size: 40px;
  letter-spacing: .4px;
}
.cf-sub{
  margin-top: 6px;
  opacity: .95;
}
/* Wrap + position for description panel and its arrows */
.cf-info-wrap { position: relative; }


.cf-info-wrap .cf-left  { left: 12px;  }
.cf-info-wrap .cf-right { right: 12px; }


/* Row that is visually pulled onto the description panel */
.cf-arrowwrap{
  position: relative;
  margin-top: -54px;            /* pulls the buttons up over the panel */
  z-index: 6;                   /* above the panel */
}

/* Make the arrow buttons blue and compact */
.cf-arrowwrap [data-testid="stButton"] > button{
  background: var(--primary) !important;
  color:#fff !important;
  border:0 !important;
  width:42px; height:42px;
  border-radius:10px;
  box-shadow:0 2px 6px rgba(0,123,167,.30);
  font-weight:700;
  line-height: 1;
  padding: 0 !important;
}
/* Style the Streamlit buttons inside those containers */
.cf-info-wrap .cf-arrow [data-testid="stButton"] > button{
  background: var(--primary) !important;
  color:#fff !important;
  border:0 !important;
  width:42px; height:42px;
  border-radius:10px;
  box-shadow:0 2px 6px rgba(0,123,167,.30);
  font-weight:700;
  line-height: 1;
  padding: 0 !important;
}

/* Buttons */
.stButton>button, .stDownloadButton>button{
  background: var(--primary) !important;
  color: #fff !important;
  border: 0 !important;
  border-radius: 12px !important;
  padding: 10px 16px !important;
}
.stButton>button:hover, .stDownloadButton>button:hover{
  background: var(--primary-dark) !important;
}

/* Card styling for metric box */
.metric-card{
  background: var(--card);
  border: 1px solid #E2E8F0;
  border-radius: 16px;
  padding: 16px;
  box-shadow: 0 2px 8px rgba(15,23,42,.06);
}
/* Stock line header (logo + text, perfectly aligned) */
.cf-stockline{
  display:flex; align-items:center; gap:12px;
  margin: 0 0 8px 4px;
}
.cf-logo{
  width:32px; height:32px; object-fit:contain;
}
/* Volatility info box */
.cf-info{
  background: #E0F2FE; /* light cerulean tint */
  border-left: 5px solid #007BA7;
  padding: 14px 18px;
  border-radius: 10px;
  color: #0F172A;
  font-size: 15px;
  line-height: 1.5;
  box-shadow: 0 2px 6px rgba(0,0,0,0.05);
  margin: 10px 0 25px 0;
}
.cf-info strong{
  color: #005F7D;
}

.cf-stocktext{
  font-size:22px; font-weight:700; color:#0F172A;
}
/* Tame metric sizes in the single-ticker “Stock statistics” panel */
[data-testid="stMetricValue"]{
  font-size: 18px !important;   /* was 28px */
  line-height: 1.15;
  white-space: nowrap;
  overflow: visible !important;
  text-overflow: clip !important;
}
[data-testid="stMetricLabel"]{
  font-size: 12px !important;
  color: var(--muted);
}
[data-testid="stMetricDelta"]{
  font-size: 12px !important;
}

/* Footer */
small, .cf-foot{
  color: var(--muted);
  font-size: 12px;
  margin-top: 24px;
  display: block;
  text-align: center;
}
/* Small fixed header that always stays on top */
.cf-sticky {
  position: fixed;
  top: 0;
  left: 0;
  right: 0;
  z-index: 10000;
  width: 100%;
  background: linear-gradient(135deg, var(--primary) 0%, var(--accent) 100%);
  color: #fff;
  height: 44px;
  display: flex;
  align-items: center;
  padding: 0 14px;
  border-bottom: 1px solid rgba(255,255,255,.15);
  box-shadow: 0 6px 14px rgba(0,0,0,.08);
  font-weight: 700;
  letter-spacing: .2px;
  font-size: 18px; /* small */
}

/* Push main content down so it doesn't sit under the fixed bar */
[data-testid="stAppViewContainer"] > .main {
  padding-top: 54px; /* header 44px + 10px breathing room */
}
/* Pretty section wrapper for the dashboard */
.cf-section{
  background: var(--card);
  border: 1px solid #E2E8F0;
  border-radius: 18px;
  padding: 22px 22px;
  box-shadow: 0 8px 30px rgba(15,23,42,.06);
  margin: 6px 0 24px 0;
}
/* Make any bordered container that starts with our big header look like a card */
div[aria-label="stContainerBorder"]{
  background: var(--card);
  border: 1px solid #E2E8F0;
  border-radius: 18px;
  padding: 20px 20px 16px 20px;
  box-shadow: 0 8px 30px rgba(15,23,42,.06);
}

/* Streamlit gives the bordered container an inner wrapper; tighten spacing */
div[aria-label="stContainerBorder"] > div:first-child{
  padding: 0 !important;
}


/* Big headline for the section */
.cf-h1{
  font-size: 44px;
  font-weight: 800;
  letter-spacing: .2px;
  color: #0F172A;
  margin: 0 0 4px 0;
}
.cf-caption{
  color: var(--muted);
  font-size: 14px;
  margin-bottom: 18px;
}

/* Panel with built-in arrows */
.cf-info.cf-info-has-arrows{
  position: relative;
  padding-left: 64px;   /* leave room for left arrow */
  padding-right: 64px;  /* leave room for right arrow */
}

/* Arrow buttons sitting on the panel */
.cf-arrow-btn{
  position: absolute;
  top: 50%;
  transform: translateY(-50%);
  width: 42px; height: 42px;
  border-radius: 10px;
  background: var(--primary);
  color: #fff !important;
  text-decoration: none !important;
  display: flex; align-items: center; justify-content: center;
  box-shadow: 0 2px 6px rgba(0,123,167,.30);
  font-weight: 700;
  line-height: 1;
}
.cf-arrow-btn.left  { left: 12px; }
.cf-arrow-btn.right { right: 12px; }

/* Gentle card styling for the right-side metrics panel (you already had .metric-card; keeping it consistent) */
</style>
""", unsafe_allow_html=True)

# --- Branded header ---
st.markdown("""
<div class="cf-hero">
  <div class="cf-brand">Chaouat Finance</div>
</div>
""", unsafe_allow_html=True)
# ---- Rotating single info panel (auto every 10s; manual arrows) ----
try:
    from streamlit_autorefresh import st_autorefresh
except Exception:
    st_autorefresh = None

PANELS = [
    ("Volatility",
     "measures how much a stock’s price moves over time. A highly volatile stock fluctuates widely, offering both opportunity and risk, while a low-volatility stock tends to move more steadily. Understanding volatility helps investors gauge uncertainty, assess portfolio risk, and balance stability with potential returns."),
    ("Price Change",
     "measures how much a stock has moved over a given period, showing whether it has been appreciating or declining in value. It helps traders quickly identify momentum leaders and laggards in the market."),
    ("YTD Return",
     "tracks how much a stock has gained or lost since the start of the calendar year. It is widely used by portfolio managers to benchmark performance against market indices over the current year."),
    ("52-Week Change",
     "compares today’s price with its level one year ago, showing the long-term trend beyond short-term noise. It is useful for spotting stocks that are outperforming or underperforming over a full market cycle."),
    ("Max Drawdown",
     "measures the largest peak-to-trough loss over a period, showing how much an investor could have lost at worst. It is a key risk metric used to evaluate downside exposure and portfolio resilience."),
    ("RSI (14)",
     "tracks recent price momentum on a scale from 0 to 100, identifying overbought and oversold conditions. Traders use it to anticipate trend reversals and time entries or exits."),
    ("Distance to 52-Week High",
     "measures how far a stock currently trades from its highest price in the last year. It helps investors see whether a stock is breaking out to new highs or still recovering from past losses."),
    ("Sharpe Ratio (1y)",
     "compares a stock’s return to its volatility, measuring how much return an investor earns per unit of risk. A higher Sharpe indicates a more efficient, risk-adjusted performer."),
    ("Beta vs S&P 500",
     "measures how sensitive a stock is to movements in the overall market: values above 1 move more than the index, below 1 move less. It is essential for portfolio construction, hedging, and estimating systematic risk.")
]

if "panel_idx" not in st.session_state:
    st.session_state.panel_idx = 0
# Handle HTML-arrow clicks via query param ?p=prev|next
_qp = get_query_params()
_p = _qp.get("p")
if isinstance(_p, list):  # experimental_get_query_params() returns lists
    _p = _p[0]
if _p == "prev":
    st.session_state.panel_idx = (st.session_state.panel_idx - 1) % len(PANELS)
elif _p == "next":
    st.session_state.panel_idx = (st.session_state.panel_idx + 1) % len(PANELS)
# clear the param so the page doesn't keep advancing on refresh
try:
    st.query_params.clear()                         # new API
except Exception:
    st.experimental_set_query_params()             # fallback for older Streamlit
if "panel_refresh_count" not in st.session_state:
    st.session_state.panel_refresh_count = -1

# Auto-advance every 10 seconds (but pause during heavy actions like scans)
pause_until = st.session_state.get("pause_autorefresh_until", 0.0)

if st_autorefresh is not None and time.time() >= pause_until:
    count = st_autorefresh(interval=10_000, key="rotating_info_panel")
    if count != st.session_state.panel_refresh_count:
        st.session_state.panel_idx = (st.session_state.panel_idx + 1) % len(PANELS)
        st.session_state.panel_refresh_count = count

title, text = PANELS[st.session_state.panel_idx]

# One HTML panel that includes the arrows as links
st.markdown(
    f"""
    <div class="cf-info cf-info-has-arrows">
      <a class="cf-arrow-btn left"  href="?p=prev" aria-label="Previous">◀</a>
      <a class="cf-arrow-btn right" href="?p=next" aria-label="Next">▶</a>
      <div><strong>{title}</strong> {text}</div>
    </div>
    """,
    unsafe_allow_html=True
)



# ---- end rotating panel ----

with st.container(border=True):
    st.markdown('<div class="cf-h1">Stock Analysis Dashboard</div>', unsafe_allow_html=True)
    st.markdown('<div class="cf-caption">Data source: Yahoo Finance via yfinance.</div>', unsafe_allow_html=True)

    # >>> keep ALL the dashboard code (query params, sidebar filters,
    # date range, fetch, left/right columns, plots, metrics) INSIDE this block


    # Optional: deep-link into a ticker with ?sym=XXXX
    qp = get_query_params()

    if "sym" in qp:
        _sym_qs = qp["sym"]
        if isinstance(_sym_qs, str):
            default_pick = [_sym_qs.upper()]
        else:  # list-like
            default_pick = [s.upper() for s in _sym_qs][:4]
    else:
        default_pick = ["AAPL"]
    # Broad category presets -> underlying S&P sectors
    CATEGORY_PRESETS = {
        "All": [],
        "Tech": ["Information Technology", "Communication Services"],
        "Finance": ["Financials"],
        "Health": ["Health Care"],
        "Energy": ["Energy"],
        "Consumer": ["Consumer Discretionary", "Consumer Staples"],
        "Industrials": ["Industrials"],
        "Utilities": ["Utilities"],
        "Materials": ["Materials"],
        "Real Estate": ["Real Estate"],
        # "Sports" is not a GICS sector; this preset approximates sports-related names
        # via brands/media (Discretionary) + media rights/broadcasters (Comm Services).
        "Sports (brands & media)": ["Consumer Discretionary", "Communication Services"],
    }
    with st.sidebar:
        st.header("Settings")

        sp500 = load_sp500_df()
        sp500["Sector"] = sp500["Sector"].fillna("Unknown")
        all_symbols = sp500["Symbol"].dropna().unique().tolist()
        # --- Filters ---
        st.subheader("Filter universe")

        preset = st.selectbox(
            "Category preset",
            list(CATEGORY_PRESETS.keys()),
            help="Choose a broad category. You can still refine sectors below."
        )

        sector_options = sorted(sp500["Sector"].unique())
        pre_selected = CATEGORY_PRESETS[preset] if preset != "All" else []
        picked_sectors = st.multiselect(
            "Sectors",
            sector_options,
            default=pre_selected,
            help="Filter the ticker list by S&P 500 sector(s). Leave empty for all."
        )

        # Filtered symbol universe
        if picked_sectors:
            f_syms = (
                sp500.loc[sp500["Sector"].isin(picked_sectors), "Symbol"]
                .dropna().unique().tolist()
            )
        else:
            f_syms = all_symbols

        st.caption(f"Universe size: {len(f_syms)} tickers")

        # Keep default selection if still valid
        default_safe = [s for s in default_pick if s in f_syms] or (
            ["AAPL"] if "AAPL" in f_syms else f_syms[:1]
        )

        pick = st.multiselect(
            "Select up to 4 tickers (type to search)",
            options=sorted(f_syms),
            default=default_safe,
            max_selections=4,
            help="Start typing (e.g., NVDA, MSFT)."
        )

        # Optional free-text extra symbols
        custom = st.text_input("Optional extra symbols (comma-separated)").strip()
        if custom:
            extra = [x.strip().upper() for x in custom.split(",") if x.strip()]
            pick = (pick + extra)[:4]  # enforce max 4 total

        # Core chart controls — always defined (not under `if custom`)
        years = st.slider("History (years)", 1, 10, 5)
        vol_window = st.slider("Rolling volatility window (trading days)", 5, 252, 20)
        use_log_returns = st.toggle("Use log returns (recommended)", True)

        st.divider()
        st.subheader("Chart overlays")
        chart_overlays = st.multiselect(
            "Select overlays",
            ["Price", "Rolling Volatility", "SMA 50", "SMA 200", "RSI (14)", "Drawdown"],
            default=["Price"],
            help="RSI and Drawdown are plotted on a secondary axis."
        )


        st.divider()
        st.subheader("Cross-section scan")
        lookback_days = st.slider(
            "Lookback (trading days) for scan",
            20, 252, 60,
            key="scan_lookback",
            help="Window used to rank highest/lowest volatility across the S&P 500."
        )
        run_scan = st.checkbox(
            "Show Top/Bottom lists",
            value=True,
            key="scan_toggle"
        )

        st.divider()
        ttl_minutes = st.number_input("Data cache TTL (minutes)", min_value=1, value=60)
        refresh = st.button("Force refresh now")



    # date range AFTER sidebar variables exist
    start = dt.date.today() - dt.timedelta(days=365 * years)
    end = dt.date.today()

    # STEP 3 — fetch data for all selected tickers
    if refresh:
        fetch_history_multi.clear()

    prices_all = fetch_history_multi(pick, start, end)
    if prices_all.empty:
        st.error("No price data returned. Try different symbols.")
        st.stop()

    # === CHART + COMPARISON STATS IN ONE ROW ===
    left, right = st.columns([2.5, 1], gap="large")

    with left:
        # Header: show logo + full name ONLY when there is a single ticker.
        # For multiple tickers, show only the tickers string (no logo, no long name).
        if len(pick) == 1:
            sym = pick[0]
            company_name, company_logo = get_company_meta(sym)

            # Build one HTML block to avoid nested triple-quote pitfalls
            logo_html = f'<img src="{company_logo}" alt="{sym} logo" class="cf-logo"/>' if company_logo else ""
            name_html = f' — {company_name}' if company_name else ""

            st.markdown(
                f"""
                <div class="cf-stockline">
                  {logo_html}
                  <div class="cf-stocktext"><b>{sym}</b>{name_html}</div>
                </div>
                """,
                unsafe_allow_html=True
            )
        else:
            st.markdown(
                f"""
                <div class="cf-stockline">
                  <div class="cf-stocktext"><b>{' / '.join(pick)}</b></div>
                </div>
                """,
                unsafe_allow_html=True
            )


        # ---- build overlays figure (uses the sidebar multiselect `chart_overlays`)
        ind_map = compute_indicators(prices_all)

        vol_df = None
        if "Rolling Volatility" in chart_overlays:
            vol_df = compute_rolling_vol(prices_all, window=vol_window, use_log=use_log_returns)

        fig = make_subplots(specs=[[{"secondary_y": True}]])

        # Price
        if "Price" in chart_overlays:
            for tkr, g in prices_all.sort_values("Date").groupby("Ticker"):
                fig.add_trace(
                    go.Scatter(x=g["Date"], y=g["Close"], name=f"{tkr} Price", mode="lines"),
                    secondary_y=False
                    )

        # SMA overlays
        if "SMA 50" in chart_overlays:
            for tkr, ind in ind_map.items():
                fig.add_trace(
                    go.Scatter(x=ind.index, y=ind["SMA50"], name=f"{tkr} SMA50", mode="lines", line=dict(dash="dot")),
                    secondary_y=False
                )
        if "SMA 200" in chart_overlays:
            for tkr, ind in ind_map.items():
                fig.add_trace(
                    go.Scatter(x=ind.index, y=ind["SMA200"], name=f"{tkr} SMA200", mode="lines", line=dict(dash="dash")),
                    secondary_y=False
                )

        # Rolling Vol (annualized)
        if vol_df is not None and not vol_df.empty:
            for tkr, g in vol_df.sort_values("Date").groupby("Ticker"):
                fig.add_trace(
                    go.Scatter(x=g["Date"], y=g["AnnVol"], name=f"{tkr} Ann. Vol", mode="lines"),
                    secondary_y=False
                )

        # RSI (secondary axis)
        if "RSI (14)" in chart_overlays:
            for tkr, ind in ind_map.items():
                fig.add_trace(
                    go.Scatter(x=ind.index, y=ind["RSI14"], name=f"{tkr} RSI(14)", mode="lines"),
                    secondary_y=True
                )

        # Drawdown (secondary axis)
        if "Drawdown" in chart_overlays:
            for tkr, ind in ind_map.items():
                fig.add_trace(
                    go.Scatter(x=ind.index, y=ind["Drawdown"], name=f"{tkr} Drawdown", mode="lines", fill="tozeroy"),
                    secondary_y=True
                )

        fig.update_layout(
            height=460,
            margin=dict(l=10, r=10, t=30, b=10),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0)
        )
        fig.update_xaxes(title_text="Date")
        fig.update_yaxes(title_text="Price / Volatility", secondary_y=False)
        fig.update_yaxes(title_text="RSI / Drawdown", secondary_y=True)

        st.plotly_chart(fig, use_container_width=True)

    with right:
        # ---------- build the data used by either view ----------
        # Latest price per ticker
        last_prices = (
            prices_all.sort_values("Date")
            .groupby("Ticker")
            .tail(1)[["Ticker", "Close"]]
            .set_index("Ticker")["Close"]
        )

        # YTD return
        ytd_start = dt.date(dt.date.today().year, 1, 1)
        ytd_prices = prices_all[prices_all["Date"] >= pd.to_datetime(ytd_start)].copy()
        ytd_first = (
            ytd_prices.sort_values("Date")
            .groupby("Ticker")
            .head(1)
            .set_index("Ticker")["Close"]
        )
        ytd_ret = ((last_prices / ytd_first) - 1).replace([np.inf, -np.inf], np.nan)

        # Latest annualized vol using the chosen window
        vol_df = compute_rolling_vol(prices_all, window=vol_window, use_log=use_log_returns)
        last_vol = (
            vol_df.sort_values("Date")
            .groupby("Ticker")
            .tail(1)
            .set_index("Ticker")["AnnVol"]
        )

        st.markdown('<div class="metric-card">', unsafe_allow_html=True)

    if len(pick) >= 2:
        # ---------- COMPARISON (2+ tickers) ----------
        st.subheader("Comparison")

        comp = pd.DataFrame({
            "Last Price ($)": last_prices.round(2),
            f"Vol {vol_window}d (ann)": (last_vol * 100).round(2).astype(float),
            "YTD Return (%)": (ytd_ret * 100).round(2).astype(float),
        })
        comp = comp.reindex(pick)  # preserve user selection order
        # ---- extras: 52W High/Low and Max DD from indicators
        ind_map = compute_indicators(prices_all)
        extras = []
        for t in pick:
            ind = ind_map.get(t)
            if ind is None or ind.empty:
                extras.append((t, np.nan, np.nan, np.nan))
                continue
            tail = ind.tail(252)
            high_52w = tail["Close"].max()
            low_52w  = tail["Close"].min()
            max_dd   = ind["Drawdown"].min() * 100  # %
            extras.append((t, high_52w, low_52w, max_dd))

        extra_df = pd.DataFrame(extras, columns=["Ticker","52W High","52W Low","Max DD (%)"]).set_index("Ticker")
        comp = comp.join(extra_df, how="left")

        st.dataframe(
            comp,
            use_container_width=True,
            height=180,
            column_config={
                "Last Price ($)": st.column_config.NumberColumn(format="%.2f"),
                f"Vol {vol_window}d (ann)": st.column_config.NumberColumn(format="%.2f%%"),
                "YTD Return (%)": st.column_config.NumberColumn(format="%.2f%%"),
                "52W High": st.column_config.NumberColumn(format="%.2f"),
                "52W Low": st.column_config.NumberColumn(format="%.2f"),
                "Max DD (%)": st.column_config.NumberColumn(format="%.2f%%"),
            }
        )
    else:
        # ---------- SINGLE TICKER STATS ----------
        st.subheader("Stock statistics")

        sym = pick[0]
        lp = float(last_prices.get(sym, np.nan)) if sym in last_prices.index else float("nan")
        lv = float(last_vol.get(sym, np.nan) * 100) if sym in last_vol.index else float("nan")
        yr = float(ytd_ret.get(sym, np.nan) * 100) if sym in ytd_ret.index else float("nan")

        # Build indicators for single symbol (needed in this scope)
        ind_map = compute_indicators(prices_all)
        ind = ind_map.get(sym)

        if ind is not None and not ind.empty:
            last_252 = ind.tail(252)
            high_52w = last_252["Close"].max()
            low_52w  = last_252["Close"].min()
            max_dd   = float(ind["Drawdown"].min())  # fraction, e.g. -0.34

            # Sharpe over last 1y (simple)
            ret_1y = last_252["Return"].dropna()
            sharpe_1y = float((ret_1y.mean() * np.sqrt(252)) / ret_1y.std()) if ret_1y.std() > 0 else np.nan

            # Beta vs S&P 500 over last 1y
            try:
                spy = yf.download("^GSPC", start=ret_1y.index.min(), end=ret_1y.index.max(),
                                  auto_adjust=True, progress=False)["Close"].pct_change()
                common = ret_1y.index.intersection(spy.index)
                beta = float(ret_1y.loc[common].cov(spy.loc[common]) / spy.loc[common].var()) if spy.loc[common].var() > 0 else np.nan
            except Exception:
                beta = np.nan

            rsi_last   = ind["RSI14"].iloc[-1]
            sma50_last = ind["SMA50"].iloc[-1]
            sma200_last= ind["SMA200"].iloc[-1]
        else:
            high_52w = low_52w = max_dd = sharpe_1y = beta = np.nan
            rsi_last = sma50_last = sma200_last = np.nan

        # Row 1
        row1 = st.columns(3)
        if not np.isnan(lp):
            row1[0].metric("Last Price ($)", f"{lp:.2f}")
        if not np.isnan(lv):
            row1[1].metric(f"Vol {vol_window}d (ann)", f"{lv:.2f}%")
        if not np.isnan(yr):
            row1[2].metric("YTD Return", f"{yr:.2f}%")

        # Row 2
        row2 = st.columns(3)
        if not np.isnan(high_52w):
            row2[0].metric("52-Week High", f"{high_52w:.2f}")
        if not np.isnan(low_52w):
            row2[1].metric("52-Week Low", f"{low_52w:.2f}")
        if not np.isnan(max_dd):
            row2[2].metric("Max Drawdown", f"{max_dd*100:.2f}%")

        # Row 3
        row3 = st.columns(3)
        if not np.isnan(rsi_last):
            row3[0].metric("RSI (14)", f"{rsi_last:.1f}")
        if not np.isnan(sharpe_1y):
            row3[1].metric("Sharpe (1y)", f"{sharpe_1y:.2f}")
        if not np.isnan(beta):
            row3[2].metric("Beta vs S&P 500", f"{beta:.2f}")



    st.markdown('</div>', unsafe_allow_html=True)



# === TOP / BOTTOM VOLATILITY (FAST MODE: run only on click) ===
# === TOP / BOTTOM (stable: does NOT disappear on reruns) ===
st.subheader("Leaders and laggards")
# Unique id for each scan run (prevents Streamlit duplicate element errors)
if "scan_run_id" not in st.session_state:
    st.session_state.scan_run_id = 0

# --- Remember the last successful scan so charts stay visible on reruns ---
if "last_scan_df" not in st.session_state:
    st.session_state.last_scan_df = None
if "last_scan_meta" not in st.session_state:
    st.session_state.last_scan_meta = None

# Put the controls in a form so moving sliders does not trigger a new scan
with st.form("scan_form", clear_on_submit=False):
    metrics_list = [
        "Volatility",
        "Price change",
        "YTD return",
        "52W change",
        "Max drawdown",
        "RSI (14)",
        "Distance to 52W high",
        "Sharpe (1y)",
        "Beta vs S&P 500",
    ]

    c1, c2, c3 = st.columns([1.1, 1.0, 1.2])
    with c1:
        metric_mode = st.selectbox("Metric", metrics_list)
    with c2:
        top_n = st.slider("Top-N", 1, 50, 5, step=1)
    with c3:
        sector_choice = st.selectbox(
            "Sector",
            ["All"] + sorted(sp500["Sector"].dropna().unique().tolist()),
            index=0
        )

    # Only ask for lookback when it matters
    if metric_mode in ["Volatility", "Price change"]:
        lookback_days = st.slider("Lookback (trading days)", 20, 252, 60)
    else:
        lookback_days = 60

    run_scan_now = st.form_submit_button("Run scan")

# If user clicked Run scan, compute and SAVE results
if run_scan_now:
    # Pause auto-refresh for 90 seconds so the scan is not interrupted
    st.session_state.pause_autorefresh_until = time.time() + 90

    if sector_choice == "All":
        universe = sp500["Symbol"].dropna().unique().tolist()
    else:
        universe = (
            sp500.loc[sp500["Sector"] == sector_choice, "Symbol"]
            .dropna().unique().tolist()
        )

    scope = "S&P 500" if sector_choice == "All" else f"{sector_choice} sector"

    with st.spinner(f"Scanning {metric_mode.lower()} across the {scope}..."):
        scan_df = scan_cross_section(metric_mode, universe, lookback_days, use_log_returns)

    # IMPORTANT: only overwrite the saved results if we actually got data
    if scan_df is not None and not scan_df.empty:
        st.session_state.last_scan_df = scan_df
        st.session_state.last_scan_meta = {
            "metric_mode": metric_mode,
            "top_n": top_n,
            "sector_choice": sector_choice,
            "lookback_days": lookback_days,
            "scope": scope,
        }
    else:
        st.warning("The scan returned no data (Yahoo sometimes rate-limits). Try again, or select a smaller sector.")

# If we still have no saved scan, show instructions and DO NOT stop the page
if st.session_state.last_scan_df is None or st.session_state.last_scan_df.empty:
    st.info("Adjust settings above, then click **Run scan**.")
else:
    # Use saved results
    scan_df = st.session_state.last_scan_df
    meta = st.session_state.last_scan_meta or {}
    metric_mode = meta.get("metric_mode", "Volatility")
    top_n = meta.get("top_n", 5)

    # Ranking rules
    desc_top = metric_mode not in ["Max drawdown", "Distance to 52W high"]

    pct_metrics = {
        "Volatility", "Price change", "YTD return", "52W change",
        "Max drawdown", "Distance to 52W high"
    }
    fmt = (lambda x: f"{x*100:.2f}%") if metric_mode in pct_metrics else (lambda x: f"{x:.2f}")

    topN = scan_df.sort_values("Value", ascending=not desc_top).head(top_n).copy()
    botN = scan_df.sort_values("Value", ascending=desc_top).head(top_n).copy()

    topN["Label"] = topN["Value"].apply(fmt)
    botN["Label"] = botN["Value"].apply(fmt)

    title_top = {
        "Volatility": "Highest volatility",
        "Price change": "Top gainers",
        "YTD return": "Top YTD",
        "52W change": "Top 52-week change",
        "Max drawdown": "Smallest drawdown",
        "RSI (14)": "Highest RSI (14)",
        "Distance to 52W high": "Closest to 52W high",
        "Sharpe (1y)": "Highest Sharpe (1y)",
        "Beta vs S&P 500": "Highest Beta",
    }[metric_mode]
    title_bot = {
        "Volatility": "Lowest volatility",
        "Price change": "Top losers",
        "YTD return": "Lowest YTD",
        "52W change": "Lowest 52-week change",
        "Max drawdown": "Largest drawdown",
        "RSI (14)": "Lowest RSI (14)",
        "Distance to 52W high": "Furthest from 52W high",
        "Sharpe (1y)": "Lowest Sharpe (1y)",
        "Beta vs S&P 500": "Lowest Beta",
    }[metric_mode]

    leftc, rightc = st.columns(2, gap="large")

    with leftc:
        st.markdown(f"**{title_top}**")
        fig_top = px.bar(topN, x="Ticker", y="Value", text="Label")
        fig_top.update_traces(textposition="outside")
        fig_top.update_layout(height=360, margin=dict(l=60, r=10, t=30, b=60), showlegend=False)
        fig_top.update_xaxes(showticklabels=True)
        st.plotly_chart(fig_top, use_container_width=True)

    with rightc:
        st.markdown(f"**{title_bot}**")
        fig_bot = px.bar(botN, x="Ticker", y="Value", text="Label")
        fig_bot.update_traces(textposition="outside")
        fig_bot.update_layout(height=360, margin=dict(l=60, r=10, t=30, b=60), showlegend=False)
        fig_bot.update_xaxes(showticklabels=True)
        st.plotly_chart(fig_bot, use_container_width=True)




st.subheader("Data")
st.dataframe(
    prices_all.sort_values(["Ticker","Date"]).tail(400),
    use_container_width=True,
    height=180  # ≈ 4 rows visible; scroll for more
)
st.download_button(
    "Download CSV",
    data=prices_all.to_csv(index=False).encode("utf-8"),
    file_name=f"{'-'.join(pick)}_prices.csv",
    mime="text/csv",
)

st.caption("Volatility should be computed on returns, not raw prices. 252 trading days used for annualization.")
st.markdown('<div class="cf-foot">© Chaouat Finance · Built with Python</div>', unsafe_allow_html=True)





















































































