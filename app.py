import math
import datetime as dt
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.io as pio
import streamlit as st
import yfinance as yf

@st.cache_data(ttl=24*60*60)
def load_sp500_df():
    url = "https://raw.githubusercontent.com/datasets/s-and-p-500-companies/main/data/constituents.csv"
    return pd.read_csv(url)  # columns: Symbol, Name, Sector...

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
    """Input tidy prices (Date, Ticker, Close). Return tidy rolling vol with AnnVol column."""
    if df_prices.empty:
        return pd.DataFrame(columns=["Date", "Ticker", "AnnVol"])
    df = df_prices.sort_values(["Ticker", "Date"]).copy()
    df["ret"] = df.groupby("Ticker")["Close"].apply(
        lambda s: np.log(s).diff() if use_log else s.pct_change()
    ).reset_index(level=0, drop=True)
    df["vol_daily"] = df.groupby("Ticker")["ret"].rolling(window).std().reset_index(level=0, drop=True)
    df["AnnVol"] = df["vol_daily"] * math.sqrt(252)
    return df.dropna(subset=["AnnVol"])

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
/* Fix st.metric values getting truncated with ellipsis */
[data-testid="stMetricValue"]{
  font-size: 24px;
  line-height: 1.1;
  white-space: nowrap;
  overflow: visible !important;
  text-overflow: clip !important;
}

/* Footer */
small, .cf-foot{
  color: var(--muted);
  font-size: 12px;
  margin-top: 24px;
  display: block;
  text-align: center;
}
</style>
""", unsafe_allow_html=True)

# --- Branded header ---
st.markdown("""
<div class="cf-hero">
  <div class="cf-brand">Chaouat Finance</div>
  <div class="cf-sub">Clean, fast analytics for price trends and risk.</div>
</div>
""", unsafe_allow_html=True)
st.markdown("""
<div class="cf-info">
  <strong>Volatility</strong> measures how much a stock’s price moves over time. 
  A highly volatile stock fluctuates widely, offering both opportunity and risk, 
  while a low-volatility stock tends to move more steadily. 
  Understanding volatility helps investors gauge uncertainty, 
  assess portfolio risk, and balance stability with potential returns.
</div>
""", unsafe_allow_html=True)

st.title("Stock Volatility Dashboard")
st.caption("Data source: Yahoo Finance via yfinance (unofficial).")

# Optional: deep-link into a ticker with ?sym=XXXX
qp = st.query_params
if "sym" in qp:
    _sym_qs = qp["sym"]
    if isinstance(_sym_qs, str):
        default_pick = [_sym_qs.upper()]
    else:
        default_pick = [s.upper() for s in _sym_qs][:4]
else:
    default_pick = ["AAPL"]

with st.sidebar:
    st.header("Settings")

    sp500 = load_sp500_df()
    sp_syms = sp500["Symbol"].dropna().unique().tolist()

    pick = st.multiselect(
        "Select up to 4 tickers (type to search)",
        options=sp_syms,
        default=default_pick,
        max_selections=4,
        help="Start typing (e.g., NVDA, MSFT)."
    )
    custom = st.text_input("Optional extra symbols (comma-separated)").strip()
    if custom:
        extra = [x.strip().upper() for x in custom.split(",") if x.strip()]
        pick = (pick + extra)[:4]  # enforce max 4 total

    years = st.slider("History (years)", 1, 10, 5)
    vol_window = st.slider("Rolling volatility window (trading days)", 5, 252, 20)
    use_log_returns = st.toggle("Use log returns (recommended)", True)

    st.divider()
    st.subheader("Chart mode")
    chart_mode = st.radio("What to overlay?", ["Price", "Rolling Volatility"], horizontal=True)

    # >>> Add these four lines (this defines lookback_days and run_scan) <<<
    st.divider()
    st.subheader("Cross-section scan")
    lookback_days = st.slider(
        "Lookback (trading days) for scan", 20, 252, 60, key="scan_lookback",
        help="Window used to rank highest/lowest volatility across the S&P 500."
    )
    run_scan = st.checkbox("Show Top 5 / Bottom 5 volatility lists", value=True, key="scan_toggle")
    # <<< end inserted lines >>>

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
    # For the header, show first selected company name + logo (optional)
    first_ticker = pick[0]
    company_name, company_logo = get_company_meta(first_ticker)
    if company_logo:
        st.markdown(
            f"""
            <div class="cf-stockline">
              <img src="{company_logo}" alt="{first_ticker} logo" class="cf-logo"/>
              <div class="cf-stocktext"><b>{' / '.join(pick)}</b> — {company_name}</div>
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

    if chart_mode == "Price":
        fig = px.line(
            prices_all,
            x="Date", y="Close", color="Ticker",
            labels={"Close": "Adj. Close ($)", "Date": "Date"},
        )
        fig.update_layout(height=460, margin=dict(l=10, r=10, t=30, b=10))
        st.plotly_chart(fig, use_container_width=True)

    else:  # Rolling Volatility
        vol_df = compute_rolling_vol(prices_all, window=vol_window, use_log=use_log_returns)
        fig = px.line(
            vol_df,
            x="Date", y="AnnVol", color="Ticker",
            labels={"AnnVol": "Annualized Volatility", "Date": "Date"},
        )
        fig.update_layout(height=460, margin=dict(l=10, r=10, t=30, b=10))
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

        st.dataframe(
            comp,
            use_container_width=True,
            column_config={
                "Last Price ($)": st.column_config.NumberColumn(format="%.2f"),
                f"Vol {vol_window}d (ann)": st.column_config.NumberColumn(format="%.2f%%"),
                "YTD Return (%)": st.column_config.NumberColumn(format="%.2f%%"),
            }
        )
    else:
        # ---------- SINGLE TICKER STATS ----------
        st.subheader("Stock statistics")

        sym = pick[0]
        lp = float(last_prices.get(sym, np.nan)) if sym in last_prices.index else float("nan")
        lv = float(last_vol.get(sym, np.nan) * 100) if sym in last_vol.index else float("nan")
        yr = float(ytd_ret.get(sym, np.nan) * 100) if sym in ytd_ret.index else float("nan")

        c1, c2, c3 = st.columns(3)
        if not np.isnan(lp):
            c1.metric("Last Price ($)", f"{lp:.2f}")
        if not np.isnan(lv):
            c2.metric(f"Vol {vol_window}d (ann)", f"{lv:.2f}%")
        if not np.isnan(yr):
            c3.metric("YTD Return", f"{yr:.2f}%")

    st.markdown('</div>', unsafe_allow_html=True)


# === TOP / BOTTOM VOLATILITY ===
if run_scan:
    st.subheader("Volatility leaders and laggards")

    # Use the S&P 500 universe you already load
    universe = load_sp500_df()["Symbol"].dropna().unique().tolist()

    with st.spinner("Scanning volatility across the S&P 500..."):
        scan_df = scan_volatility(universe, lookback_days=lookback_days, use_log=use_log_returns)

    if scan_df.empty or len(scan_df) < 5:
        st.info("Not enough data to compute the cross-section right now.")
    else:
        top5 = scan_df.head(5).copy()
        bot5 = scan_df.sort_values("AnnVol", ascending=True).head(5).copy()

        # Format percent for labels (as strings) and keep numeric for axis
        top5["VolPct"] = (top5["AnnVol"] * 100).round(2)
        top5["VolPctStr"] = top5["VolPct"].astype(str) + "%"
        bot5["VolPct"] = (bot5["AnnVol"] * 100).round(2)
        bot5["VolPctStr"] = bot5["VolPct"].astype(str) + "%"

        # Nice blue palettes
        blues_top = px.colors.sequential.Blues[-5:]       # darker blues for Top 5
        blues_bot = px.colors.sequential.Blues[2:7]       # lighter blues for Bottom 5

        leftc, rightc = st.columns(2, gap="large")

        with leftc:
            st.markdown("**Top 5 highest volatility**")
            fig_top = px.bar(
                top5, x="Ticker", y="AnnVol", text="VolPctStr",
            )
            fig_top.update_traces(
                marker_color=blues_top,
                textposition="outside",
                textfont=dict(size=12),
                hovertemplate="<b>%{x}</b><br>Ann. Vol: %{customdata:.2%}<extra></extra>",
                customdata=top5["AnnVol"],
            )
            fig_top.update_layout(
                yaxis_title="Annualized Volatility",
                xaxis_title="",
                height=340,
                margin=dict(l=10, r=10, t=30, b=10),
                showlegend=False,
                fig_top.update_xaxes(showticklabels=False)
        
            st.plotly_chart(fig_top, use_container_width=True)

            # Clickable ticker links under the chart
            links = " · ".join([f'<a href="?sym={s}">{s}</a>' for s in top5["Ticker"]])
            st.markdown(f'<div style="margin-top:-6px;color:#005F7D">{links}</div>', unsafe_allow_html=True)

        with rightc:
            st.markdown("**Top 5 lowest volatility**")
            fig_bot = px.bar(
                bot5, x="Ticker", y="AnnVol", text="VolPctStr",
            )
            fig_bot.update_traces(
                marker_color=blues_bot,
                textposition="outside",
                textfont=dict(size=12),
                hovertemplate="<b>%{x}</b><br>Ann. Vol: %{customdata:.2%}<extra></extra>",
                customdata=bot5["AnnVol"],
            )
            fig_bot.update_layout(
                yaxis_title="Annualized Volatility",
                xaxis_title="",
                height=340,
                margin=dict(l=10, r=10, t=30, b=10),
                showlegend=False,
                fig_bot.update_xaxes(showticklabels=False)
            )
            st.plotly_chart(fig_bot, use_container_width=True)

            links = " · ".join([f'<a href=\"?sym={s}\">{s}</a>' for s in bot5["Ticker"]])
            st.markdown(f'<div style="margin-top:-6px;color:#005F7D">{links}</div>', unsafe_allow_html=True)

st.subheader("Data")
st.dataframe(
    prices_all.sort_values(["Ticker","Date"]).tail(400),
    use_container_width=True
)
st.download_button(
    "Download CSV",
    data=prices_all.to_csv(index=False).encode("utf-8"),
    file_name=f"{'-'.join(pick)}_prices.csv",
    mime="text/csv",
)

st.caption("Volatility should be computed on returns, not raw prices. 252 trading days used for annualization.")
st.markdown('<div class="cf-foot">© Chaouat Finance · Built with Python</div>', unsafe_allow_html=True)


































