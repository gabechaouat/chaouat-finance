# pages/2_Investment_Simulator.py
import math
import datetime as dt
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
import yfinance as yf

# ---------- shared helpers (same behavior as in app) ----------
@st.cache_data(ttl=24*60*60)
def load_sp500_df():
    url = "https://raw.githubusercontent.com/datasets/s-and-p-500-companies/main/data/constituents.csv"
    return pd.read_csv(url)  # Symbol, Name, Sector...

@st.cache_data(ttl=6*60*60)
def get_company_meta(sym: str):
    """Return (company_name, logo_url) using S&P list/yfinance + Clearbit fallback."""
    name, logo = None, None

    # S&P list for a quick name
    try:
        df = load_sp500_df()
        hit = df.loc[df["Symbol"].str.upper() == sym.upper()]
        if not hit.empty:
            name = hit.iloc[0]["Name"]
    except Exception:
        pass

    # yfinance enrichment
    website = None
    try:
        info = yf.Ticker(sym).get_info()
        name = info.get("longName") or info.get("shortName") or name
        website = info.get("website") or info.get("website_url")
        logo = info.get("logo_url") or logo
    except Exception:
        pass

    # Clearbit logo from domain
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

# Robust close fetch that always returns a tidy Series of adj close
def _adj_close_series(df_like):
    """
    Accepts either a Series or a 1+ column DataFrame and returns a Series of adjusted close.
    """
    if isinstance(df_like, pd.Series):
        return df_like.dropna()

    # DataFrame path
    df = df_like.copy()
    # common yfinance adjusted-close column names
    for cand in ["Adj Close", "Adj Close ($)", "Adj Close*", "Close"]:
        if cand in df.columns:
            s = df[cand].dropna()
            if not s.empty:
                return s
    # last resort: first numeric column
    num = df.select_dtypes(include="number")
    if not num.empty:
        return num.iloc[:, 0].dropna()
    return pd.Series(dtype=float)

# ---------- page ----------
st.set_page_config(page_title="Investment Simulator", page_icon="🧮", layout="wide")

# small sticky header like your app
st.markdown(
    """
    <style>
      .cf-sticky {
        position: sticky; top: 0; z-index: 9999;
        background: #F7FAFC; border-bottom: 1px solid #E2E8F0;
        padding: 10px 6px; margin: 0 0 8px 0;
      }
      .cf-sticky .brand { font-weight: 700; letter-spacing: .2px; }
      .cf-stockline{ display:flex; align-items:center; gap:10px; margin: 10px 0 0 2px;}
      .cf-logo{ width:28px; height:28px; object-fit:contain; }
      .cf-stocktext{ font-size:18px; font-weight:700; color:#0F172A; }
      /* tighter table default height */
      div[data-testid="stDataFrame"] div[role="grid"] {
        height: 260px !important;
      }
    </style>
    <div class="cf-sticky">
      <span class="brand">Chaouat Finance</span>
      <span style="color:#64748B; font-size:12px; margin-left:6px;">Investment simulator</span>
    </div>
    """,
    unsafe_allow_html=True
)

st.title("Investment Simulator")
st.caption("Enter multiple purchase lots across different tickers and dates. Use a manual price if you bought at a specific price; otherwise the adjusted close for that day is used. The value is computed at the sell date (default today).")

with st.sidebar:
    st.header("Sell date")
    sell_date = st.date_input("Sell date", dt.date.today(), key="sell_date_sim")

# ---- editable lots table ----
st.subheader("Lots")
init_rows = pd.DataFrame(
    [
        {"Ticker": "AAPL", "Buy date": dt.date.today() - dt.timedelta(days=365), "Cash ($)": 1000.0, "Shares": None, "Manual price ($)": None},
    ]
)
lots_df = st.data_editor(
    init_rows,
    num_rows="dynamic",
    use_container_width=True,
    hide_index=True,
    column_config={
        "Ticker": st.column_config.TextColumn(width="small", help="e.g., AAPL, MSFT"),
        "Buy date": st.column_config.DateColumn(format="YYYY/MM/DD"),
        "Cash ($)": st.column_config.NumberColumn(min_value=0.0, step=10.0),
        "Shares": st.column_config.NumberColumn(min_value=0.0, step=0.01, help="If filled, it is used directly."),
        "Manual price ($)": st.column_config.NumberColumn(min_value=0.0, step=0.01, help="Optional: overrides historical price for that lot."),
    }
)

# Clean lots
lots = lots_df.copy()
lots["Ticker"] = lots["Ticker"].astype(str).str.upper().str.strip()
lots = lots.dropna(subset=["Ticker", "Buy date"])
lots = lots[lots["Ticker"] != ""]
if lots.empty:
    st.info("Add at least one lot above.")
    st.stop()

# Figure out data download range
first_buy = min(pd.to_datetime(lots["Buy date"]).dt.date)
start_dl = first_buy - dt.timedelta(days=14)  # small buffer
end_dl = sell_date

# Download all needed tickers once
tickers = sorted(lots["Ticker"].unique().tolist())
data = yf.download(
    tickers,
    start=start_dl,
    end=end_dl + dt.timedelta(days=1),  # yfinance end is exclusive
    interval="1d",
    auto_adjust=True,
    progress=False,
    group_by="ticker",
    threads=True,
)

# Helper to get a single date's adjusted close (or nearest previous trading day)
def price_on_or_before(sym: str, when: dt.date) -> float | None:
    try:
        if isinstance(data.columns, pd.MultiIndex):
            # Multi-index (many tickers)
            s = _adj_close_series(data[sym])
        else:
            # Single ticker frame
            s = _adj_close_series(data)
        if s.empty:
            return None
        s = s.copy()
        s.index = pd.to_datetime(s.index).date  # date index
        # exact or earlier trading date
        idx = [d for d in s.index if d <= when]
        if not idx:
            return None
        val = float(s[max(idx)])
        return val
    except Exception:
        return None

# Compute positions from lots
computed_rows = []
for _, row in lots.iterrows():
    sym = row["Ticker"]
    buy_dt = pd.to_datetime(row["Buy date"]).date()
    cash = float(row["Cash ($)"]) if pd.notna(row["Cash ($)"]) else 0.0
    shares_in = float(row["Shares"]) if pd.notna(row["Shares"]) else None
    manual_px = float(row["Manual price ($)"]) if pd.notna(row["Manual price ($)"]) and row["Manual price ($)"] > 0 else None

    # price to use
    if shares_in is None:
        # need a price to derive shares
        px = manual_px if manual_px is not None else price_on_or_before(sym, buy_dt)
        if px is None or px <= 0:
            continue
        shares = cash / px if cash > 0 else 0.0
        used_price = px
    else:
        shares = shares_in
        used_price = manual_px if manual_px is not None else price_on_or_before(sym, buy_dt)

    last_px = price_on_or_before(sym, sell_date)
    value = shares * last_px if last_px is not None else 0.0

    computed_rows.append(
        {
            "Ticker": sym,
            "Buy date": buy_dt,
            "Used buy price ($)": used_price,
            "Cash ($)": cash,
            "Shares": shares,
            "Last price ($)": last_px,
            "Value ($)": value,
        }
    )

positions = pd.DataFrame(computed_rows)
if positions.empty:
    st.warning("Could not compute any positions. Check tickers and dates.")
    st.stop()

# Aggregate per ticker
agg = (
    positions.groupby("Ticker", as_index=False)
    .agg(
        Invested=("Cash ($)", "sum"),
        Shares=("Shares", "sum"),
        LastPrice=("Last price ($)", "last"),
        Value=("Value ($)", "sum"),
    )
)
agg["P/L ($)"] = agg["Value"] - agg["Invested"]
agg["P/L (%)"] = np.where(agg["Invested"] > 0, 100.0 * agg["P/L ($)"] / agg["Invested"], np.nan)

# Order by value desc
agg = agg.sort_values("Value", ascending=False).reset_index(drop=True)

left, right = st.columns([2.2, 1], gap="large")

with left:
    # If exactly one ticker in the lots, show logo + name + price chart
    uniq = agg["Ticker"].unique().tolist()
    if len(uniq) == 1:
        sym = uniq[0]
        company_name, company_logo = get_company_meta(sym)

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

        # Build a tidy segment for chart: from first buy of that sym to sell date
        # Collect close series for the selected ticker
        if isinstance(data.columns, pd.MultiIndex):
            df_sym = data[sym]
        else:
            df_sym = data
        close = _adj_close_series(df_sym)

        first_sym_buy = positions.loc[positions["Ticker"] == sym, "Buy date"].min()
        # segment and normalize columns
        seg = close.loc[pd.to_datetime(first_sym_buy):pd.to_datetime(sell_date)]
        if isinstance(seg, pd.Series):
            seg = seg.to_frame(name="Adj Close ($)")
        else:
            seg = seg.copy()
            if seg.shape[1] == 1:
                seg.columns = ["Adj Close ($)"]
            elif "Adj Close" in seg.columns:
                seg = seg.rename(columns={"Adj Close": "Adj Close ($)"})
            elif "Adj Close ($)" not in seg.columns and "Close" in seg.columns:
                seg = seg.rename(columns={"Close": "Adj Close ($)"})
        seg = seg.reset_index().rename(columns={"index": "Date", seg.columns[0]: "Date"} if "Date" not in seg.columns else {})
        fig = px.line(seg, x="Date", y="Adj Close ($)", title=f"{sym} price")
        fig.update_layout(height=420, margin=dict(l=10, r=10, t=40, b=10))
        st.plotly_chart(fig, use_container_width=True)

with right:
    st.subheader("Summary")
    st.dataframe(
        agg.rename(columns={
            "Invested": "Invested ($)",
            "Shares": "Shares",
            "LastPrice": "Last price ($)",
            "Value": "Value ($)",
        }),
        use_container_width=True,
    )

    totals = agg[["Invested ($)", "Value ($)"] if "Invested ($)" in agg.columns else ["Invested", "Value"]].sum()
    invested_total = float(totals.iloc[0])
    value_total = float(totals.iloc[1])
    pl_total = value_total - invested_total
    pl_pct_total = (pl_total / invested_total * 100.0) if invested_total > 0 else np.nan

    st.metric("Portfolio value", f"${value_total:,.2f}", delta=f"{pl_total:,.2f}  ({pl_pct_total:.2f}%)")

st.subheader("Lots (expanded)")
st.dataframe(
    positions,
    use_container_width=True,
    hide_index=True,
)
