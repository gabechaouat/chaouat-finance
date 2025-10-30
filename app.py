import math
import datetime as dt
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.io as pio
import streamlit as st
import yfinance as yf
@st.cache_data(ttl=24*60*60)

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


st.title("Stock Volatility Dashboard")
st.caption("Data source: Yahoo Finance via yfinance (unofficial).")

with st.sidebar:
    st.header("Settings")

    tickers = load_tickers()
    ticker = st.selectbox(
        "Ticker (type to search)",
        options=tickers,
        index=tickers.index("AAPL") if "AAPL" in tickers else 0,
        help="Start typing letters to filter (e.g., AA → AAPL)."
    )

    custom = st.text_input("Or enter any symbol (optional)").strip().upper()
    if custom:
        ticker = custom

    years = st.slider("History (years)", 1, 10, 5)
    vol_window = st.slider("Rolling volatility window (trading days)", 5, 252, 20)
    use_log_returns = st.toggle("Use log returns (recommended)", True)
    ttl_minutes = st.number_input("Auto refresh cache TTL (minutes)", min_value=1, value=60)
    refresh = st.button("Force refresh now")
company_name, company_logo = get_company_meta(ticker)

start = dt.date.today() - dt.timedelta(days=365 * years)
end = dt.date.today()

@st.cache_data(show_spinner=True)  # simpler, avoids using ttl_minutes at import time
def fetch_history(sym: str, start_d, end_d) -> pd.DataFrame:
    df = yf.download(sym, start=start_d, end=end_d, interval="1d", auto_adjust=True)
    df = df.rename_axis("Date").reset_index()
    return df

if refresh:
    fetch_history.clear()

try:
    raw = fetch_history(ticker, start, end)
    if raw.empty:
        st.warning("No data returned. Check the ticker symbol.")
        st.stop()
except Exception as e:
    st.error(f"Error fetching data: {e}")
    st.stop()

# --- Normalize raw data into a simple (Date, Close) DataFrame ---
df = raw.copy()

# If index is the date, move it to a column
if "Date" not in df.columns:
    df = df.reset_index().rename(columns={"index": "Date"})

close_series = None

if isinstance(raw.columns, pd.MultiIndex):
    # Try ('Close', <ticker>) first
    try:
        close_series = raw[("Close", ticker)]
    except Exception:
        # Fallback: first column under top-level 'Close'
        if "Close" in raw.columns.get_level_values(0):
            close_series = raw["Close"].iloc[:, 0]
        elif "Adj Close" in raw.columns.get_level_values(0):
            close_series = raw["Adj Close"].iloc[:, 0]
else:
    # Single-level columns
    if "Close" in raw.columns:
        close_series = raw["Close"]
    elif "Adj Close" in raw.columns:
        close_series = raw["Adj Close"]
    else:
        # Last resort: first numeric column
        close_series = raw.select_dtypes(include="number").iloc[:, 0]

prices = pd.DataFrame({"Date": df["Date"], "Close": close_series})

# Clean types
prices["Date"] = pd.to_datetime(prices["Date"], errors="coerce")
prices["Close"] = pd.to_numeric(prices["Close"], errors="coerce")
prices = prices.dropna(subset=["Date", "Close"]).reset_index(drop=True)

# Guard: stop if still empty
if prices.empty:
    st.warning("No price data found. Try a single ticker like AAPL.")
    st.stop()

# Guard: stop if columns missing or frame empty
if prices.empty or not {"Date", "Close"}.issubset(prices.columns):
    st.error(f"Unexpected data shape. Columns: {list(raw.columns)}")
    st.stop()
prices["ret"] = np.log(prices["Close"]).diff() if use_log_returns else prices["Close"].pct_change()
prices["vol_daily"] = prices["ret"].rolling(vol_window).std()
prices["vol_annualized"] = prices["vol_daily"] * math.sqrt(252)

latest = prices.dropna().iloc[-1] if not prices.dropna().empty else None
latest_vol = float(latest["vol_annualized"]) if latest is not None else None

# === PRICE CHART COLUMN ===
col1, col2 = st.columns([2, 1])

with col1:
    st.subheader(f"Price: {ticker}")

    fig_price = px.line(
        x=prices["Date"],
        y=prices["Close"],
        labels={"x": "Date", "y": "Adj. Close ($)"},
        color_discrete_sequence=["#007BA7"]
    )

    # leave room at the top for annotation + logo
    fig_price.update_layout(height=420, margin=dict(l=10, r=10, t=90, b=10))

    # --- “TICKER — Company name” + logo side by side ---

# First, text annotation (shifted right to leave space for the logo)
fig_price.add_annotation(
    xref="paper", yref="paper",
    x=0.13, y=1.12,  # move slightly right and down
    showarrow=False,
    text=f"<b>{ticker}</b> — {company_name}",
    font=dict(size=22, color="#0F172A")
)

# Then, the logo on the left
if company_logo:
    fig_price.add_layout_image(dict(
        source=company_logo,
        xref="paper", yref="paper",
        x=0.06, y=1.11,           # fine-tuned horizontal alignment
        sizex=0.07, sizey=0.07,   # good scale to match text height
        xanchor="right", yanchor="middle",
        layer="above"
    ))

# Increase top margin to make space
fig_price.update_layout(margin=dict(l=10, r=10, t=100, b=10))



st.plotly_chart(fig_price, use_container_width=True)

with col2:
    st.markdown('<div class="metric-card">', unsafe_allow_html=True)
    st.subheader("Latest stats")
    if latest is not None:
        st.metric("Annualized Volatility", f"{latest_vol:.2%}",
                  help="Std dev of daily returns over the selected window, annualized by √252.")
        st.write(f"Window: **{vol_window}** trading days")
        st.write(f"Returns: **{'log' if use_log_returns else 'simple'}**")
        st.write(f"Data through: **{latest['Date'].date()}**")
    else:
        st.info("Insufficient data for the selected window.")
    st.markdown('</div>', unsafe_allow_html=True)

    
st.subheader(f"Rolling Annualized Volatility ({vol_window}d)")
fig_vol = px.line(
    x=prices["Date"], y=prices["vol_annualized"],
    labels={"x": "Date", "y": "Ann. Volatility"},
    color_discrete_sequence=["#005F7D"]
)
fig_vol.update_layout(height=360, margin=dict(l=10, r=10, t=30, b=10))
st.plotly_chart(fig_vol, use_container_width=True)

st.subheader("Data")
dl_cols = ["Date", "Close", "ret", "vol_daily", "vol_annualized"]
st.dataframe(prices[dl_cols].tail(400), use_container_width=True)
st.download_button("Download CSV",
                   data=prices[dl_cols].to_csv(index=False).encode("utf-8"),
                   file_name=f"{ticker}_volatility.csv",
                   mime="text/csv")

st.caption("Volatility should be computed on returns, not raw prices. 252 trading days used for annualization.")
st.markdown('<div class="cf-foot">© Chaouat Finance · Built with Python</div>', unsafe_allow_html=True)


















