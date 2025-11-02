# pages/2_Investment_Simulator.py
import math
import datetime as dt
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
import yfinance as yf

st.set_page_config(page_title="Investment Simulator · Chaouat Finance", page_icon="💹", layout="wide")

# -------- helpers --------
@st.cache_data(ttl=24*60*60)
def load_sp500_df():
    url = "https://raw.githubusercontent.com/datasets/s-and-p-500-companies/main/data/constituents.csv"
    return pd.read_csv(url)  # Symbol, Name, Sector

def _nearest_prices(sym: str, start_date: dt.date, end_date: dt.date) -> pd.Series:
    """
    Download adjusted daily prices and return the sub-series from start..end.
    """
    data = yf.download(
        sym,
        start=start_date - dt.timedelta(days=7),  # small buffer for non-trading start
        end=end_date + dt.timedelta(days=1),      # yfinance end is exclusive
        interval="1d",
        auto_adjust=True,
        progress=False,
        threads=True,
    )
    if data.empty:
        return pd.Series([], dtype="float64", name="Close")

    # Ensure we have a plain Series with datetime index
    close = data["Close"].dropna().copy()
    close.index = pd.to_datetime(close.index).tz_localize(None)

    # Clip to the requested window
    close = close[(close.index.date >= start_date) & (close.index.date <= end_date)]
    return close

def _first_on_or_after(idx: pd.DatetimeIndex, d: dt.date):
    vals = idx[idx.date >= d]
    return vals[0] if len(vals) else None

def _last_on_or_before(idx: pd.DatetimeIndex, d: dt.date):
    vals = idx[idx.date <= d]
    return vals[-1] if len(vals) else None

# -------- UI --------
st.title("Investment Simulator")

left, right = st.columns([2, 1])

with left:
    sp500 = load_sp500_df()
    syms = sp500["Symbol"].dropna().unique().tolist()
    if "AAPL" not in syms:
        syms.insert(0, "AAPL")

    ticker = st.selectbox("Ticker", options=syms, index=syms.index("AAPL") if "AAPL" in syms else 0)
    amount = st.number_input("Investment amount ($)", min_value=1.0, value=1_000.0, step=100.0)

    min_day = dt.date(1990, 1, 1)
    today = dt.date.today()

    c1, c2 = st.columns(2)
    with c1:
        buy_date = st.date_input("Buy date", value=today - dt.timedelta(days=365), min_value=min_day, max_value=today)
    with c2:
        sell_date = st.date_input("Sell date", value=today, min_value=min_day, max_value=today)

    if buy_date >= sell_date:
        st.warning("Buy date must be before sell date.")
        st.stop()

    # --- fetch data ---
    close = _nearest_prices(ticker, buy_date, sell_date)

    if close.empty:
        st.error("No price data available for the selected window.")
        st.stop()

    # find first and last trading sessions relative to the chosen dates
    first_dt = _first_on_or_after(close.index, buy_date)
    last_dt  = _last_on_or_before(close.index, sell_date)

    if not first_dt or not last_dt:
        st.error("No trading days found between your dates.")
        st.stop()

    px_buy  = float(close.loc[first_dt])
    px_sell = float(close.loc[last_dt])

    shares = amount / px_buy
    final_value = shares * px_sell
    ret = (final_value / amount) - 1.0

    # CAGR
    days = (last_dt.date() - first_dt.date()).days
    cagr = (final_value / amount) ** (365.0 / days) - 1.0 if days > 0 else np.nan

    # --- chart ---
    seg = close.loc[first_dt:last_dt].rename("Adj Close ($)").reset_index(names="Date")
    fig = px.line(seg, x="Date", y="Adj Close ($)", title=f"{ticker} price")
    fig.update_layout(height=420, margin=dict(l=10, r=10, t=40, b=10))

    st.plotly_chart(fig, use_container_width=True)

with right:
    st.subheader("Results")
    st.metric("Buy price", f"${px_buy:,.2f}", help=str(first_dt.date()))
    st.metric("Sell price", f"${px_sell:,.2f}", help=str(last_dt.date()))
    st.metric("Shares purchased", f"{shares:,.4f}")
    st.metric("Final value", f"${final_value:,.2f}")
    st.metric("Total return", f"{ret*100:,.2f}%")
    if not math.isnan(cagr):
        st.metric("CAGR (annualized)", f"{cagr*100:,.2f}%")

st.caption("Prices are split/dividend adjusted (yfinance auto_adjust=True). If your dates fall on non-trading days, "
           "the simulator uses the first trading day on/after the buy date and the last trading day on/before the sell date.")
