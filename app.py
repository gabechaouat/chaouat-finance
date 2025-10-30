import math
from datetime import date, timedelta

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
import yfinance as yf

st.set_page_config(page_title="Stock Volatility Dashboard", layout="wide")
st.title("Stock Volatility Dashboard")
st.caption("Data source: Yahoo Finance via yfinance (unofficial).")

with st.sidebar:
    st.header("Settings")
    ticker = st.text_input("Ticker (Yahoo symbol)", value="AAPL").strip().upper()
    years = st.slider("History (years)", 1, 10, 5)
    vol_window = st.slider("Rolling volatility window (trading days)", 5, 252, 20)
    use_log_returns = st.toggle("Use log returns (recommended)", True)
    ttl_minutes = st.number_input("Auto refresh cache TTL (minutes)", min_value=1, value=60)
    refresh = st.button("Force refresh now")

start = date.today() - timedelta(days=365 * years)
end = date.today()

@st.cache_data(ttl=int(ttl_minutes) * 60, show_spinner=True)
def fetch_history(sym: str, start_d: date, end_d: date) -> pd.DataFrame:
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

prices = raw[["Date", "Close"]].copy()
prices["Date"] = pd.to_datetime(prices["Date"], errors="coerce")
prices["Close"] = pd.to_numeric(prices["Close"], errors="coerce")
prices = prices.dropna(subset=["Date", "Close"]).reset_index(drop=True)

# Guard: stop if columns missing or frame empty
if prices.empty or not {"Date", "Close"}.issubset(prices.columns):
    st.error(f"Unexpected data shape. Columns: {list(raw.columns)}")
    st.stop()
prices["ret"] = np.log(prices["Close"]).diff() if use_log_returns else prices["Close"].pct_change()
prices["vol_daily"] = prices["ret"].rolling(vol_window).std()
prices["vol_annualized"] = prices["vol_daily"] * math.sqrt(252)

latest = prices.dropna().iloc[-1] if not prices.dropna().empty else None
latest_vol = float(latest["vol_annualized"]) if latest is not None else None

col1, col2 = st.columns([2, 1])
with col1:
    st.subheader(f"Price: {ticker}")
    fig_price = px.line(x=prices["Date"], y=prices["Close"],
                    labels={"x": "Date", "y": "Adj. Close ($)"})
    fig_price.update_layout(height=420, margin=dict(l=10, r=10, t=30, b=10))
    st.plotly_chart(fig_price, use_container_width=True)
with col2:
    st.subheader("Latest stats")
    if latest is not None:
        st.metric("Annualized Volatility", f"{latest_vol:.2%}",
                  help="Std dev of daily returns over the selected window, annualized by √252.")
        st.write(f"Window: **{vol_window}** trading days")
        st.write(f"Returns: **{'log' if use_log_returns else 'simple'}**")
        st.write(f"Data through: **{latest['Date'].date()}**")
    else:
        st.info("Insufficient data for the selected window.")

st.subheader(f"Rolling Annualized Volatility ({vol_window}d)")
fig_vol = px.line(x=prices["Date"], y=prices["vol_annualized"],
                  labels={"x": "Date", "y": "Ann. Volatility"})
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


