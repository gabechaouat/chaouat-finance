# pages/2_Investment_Simulator.py
import datetime as dt
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
import yfinance as yf

# ---------- helpers (self-contained; same behavior as your app) ----------
@st.cache_data(ttl=24*60*60)
def load_sp500_df():
    url = "https://raw.githubusercontent.com/datasets/s-and-p-500-companies/main/data/constituents.csv"
    return pd.read_csv(url)  # Symbol, Name, Sector...

@st.cache_data(ttl=6*60*60)
def get_company_meta(sym: str):
    """Return (company_name, logo_url) using S&P list/yfinance + Clearbit fallback."""
    name, logo = None, None

    try:
        df = load_sp500_df()
        hit = df.loc[df["Symbol"].str.upper() == sym.upper()]
        if not hit.empty:
            name = hit.iloc[0]["Name"]
    except Exception:
        pass

    website = None
    try:
        info = yf.Ticker(sym).get_info()
        name = info.get("longName") or info.get("shortName") or name
        website = info.get("website") or info.get("website_url")
        logo = info.get("logo_url") or logo
    except Exception:
        pass

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

def _adj_close_series(df_like: pd.DataFrame | pd.Series) -> pd.Series:
    """Return a clean adjusted-close Series from any yfinance frame/series."""
    if isinstance(df_like, pd.Series):
        s = df_like
    else:
        df = df_like.copy()
        for cand in ["Adj Close", "Adj Close ($)", "Adj Close*", "Close"]:
            if cand in df.columns:
                s = df[cand]
                break
        else:
            num = df.select_dtypes(include="number")
            s = num.iloc[:, 0] if not num.empty else pd.Series(dtype=float)
    s = s.dropna()
    if not s.empty and not isinstance(s.index, pd.DatetimeIndex):
        s.index = pd.to_datetime(s.index)
    return s

@st.cache_data(ttl=30*60)
def download_prices(tickers: list[str], start: dt.date, end: dt.date):
    """Batch download daily prices; returns the raw yfinance frame."""
    return yf.download(
        tickers,
        start=start,
        end=end + dt.timedelta(days=1),  # yfinance end is exclusive
        interval="1d",
        auto_adjust=True,
        progress=False,
        group_by="ticker",
        threads=True,
    )

def px_on_or_before(raw, sym: str, when: dt.date) -> float | None:
    """Adjusted close on or before given date, or None."""
    try:
        if isinstance(raw.columns, pd.MultiIndex):
            s = _adj_close_series(raw[sym])
        else:
            s = _adj_close_series(raw)
        if s.empty:
            return None
        s = s.copy()
        s.index = pd.to_datetime(s.index).date
        candidates = [d for d in s.index if d <= when]
        if not candidates:
            return None
        return float(s[max(candidates)])
    except Exception:
        return None

# ---------- page ----------
st.set_page_config(page_title="Investment Simulator", page_icon="🧮", layout="wide")

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

html, body, * { font-family: 'Montserrat', sans-serif !important; }
[data-testid="stAppViewContainer"], [data-testid="stSidebar"],
[data-testid="stMarkdownContainer"] * { font-family: 'Montserrat', sans-serif !important; }

/* Small fixed header bar */
.cf-sticky {
  position: fixed;
  top: 0; left: 0; right: 0;
  z-index: 10000;
  width: 100%;
  background: linear-gradient(135deg, var(--primary) 0%, var(--accent) 100%);
  color: #fff;
  height: 44px;
  display: flex; align-items: center;
  padding: 0 14px;
  border-bottom: 1px solid rgba(255,255,255,.15);
  box-shadow: 0 6px 14px rgba(0,0,0,.08);
  font-weight: 700; letter-spacing: .2px; font-size: 18px;
}

/* Push content down so it doesn't sit under the fixed bar */
[data-testid="stAppViewContainer"] > .main {
  padding-top: 54px; /* 44px header + 10px breathing room */
}

/* Brand hero block (same as app page, without subtitle) */
.cf-hero{
  background: linear-gradient(135deg, var(--primary) 0%, var(--accent) 100%);
  color: #fff;
  padding: 28px 32px;
  border-radius: 20px;
  box-shadow: 0 8px 24px rgba(0,123,167,.25);
  margin: 8px 0 24px 0;
}
.cf-brand{
  font-weight: 700; font-size: 40px; letter-spacing: .4px;
}

/* Buttons (consistent) */
.stButton>button, .stDownloadButton>button{
  background: var(--primary) !important; color: #fff !important; border: 0 !important;
  border-radius: 12px !important; padding: 10px 16px !important;
}
.stButton>button:hover{ background: var(--primary-dark) !important; }

/* Metric card container */
.metric-card{
  background: var(--card);
  border: 1px solid #E2E8F0;
  border-radius: 16px;
  padding: 16px;
  box-shadow: 0 2px 8px rgba(15,23,42,.06);
}

/* Stock line header (logo + name) */
.cf-stockline{ display:flex; align-items:center; gap:12px; margin: 0 0 8px 4px; }
.cf-logo{ width:32px; height:32px; object-fit:contain; }
.cf-stocktext{ font-size:22px; font-weight:700; color:#0F172A; }

/* Small note text */
.small-note{ color: var(--muted); font-size: 12px; }
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="cf-sticky">Chaouat Finance</div>', unsafe_allow_html=True)
st.markdown("""
<div class="cf-hero">
  <div class="cf-brand">Chaouat Finance</div>
</div>
""", unsafe_allow_html=True)

st.title("Investment Simulator")
st.caption("Model multiple purchases across tickers and dates. Manual price overrides historical price; otherwise the adjusted close is used.")

with st.sidebar:
    st.header("Valuation date")
    sell_date = st.date_input("Sell date", dt.date.today(), key="sell_date_sim")

# ---- editable lots table ----
st.subheader("Lots")
init_rows = pd.DataFrame(
    [
        {"Ticker": "AAPL", "Buy date": dt.date.today() - dt.timedelta(days=365), "Cash ($)": 1000.0, "Shares": None, "Manual price ($)": None},
        {"Ticker": "",     "Buy date": dt.date.today() - dt.timedelta(days=100), "Cash ($)": None,  "Shares": None, "Manual price ($)": None},
    ]
)
sp_syms = load_sp500_df()["Symbol"].dropna().unique().tolist()
lots_df = st.data_editor(
    init_rows,
    num_rows="dynamic",
    use_container_width=True,
    hide_index=True,
    column_config={
        "Ticker": st.column_config.SelectboxColumn(
            options=sp_syms,            # type-ahead suggestions
            width="small",
            help="Start typing (e.g., AAPL, MSFT).",
            required=False,
        ),

        "Buy date": st.column_config.DateColumn(format="YYYY/MM/DD"),
        "Cash ($)": st.column_config.NumberColumn(min_value=0.0, step=10.0),
        "Shares": st.column_config.NumberColumn(min_value=0.0, step=0.01, help="If provided, used directly."),
        "Manual price ($)": st.column_config.NumberColumn(min_value=0.0, step=0.01, help="Optional: overrides historical price."),
    },
)

# Clean lots
lots = lots_df.copy()
lots["Ticker"] = lots["Ticker"].astype(str).str.upper().str.strip()
lots = lots.dropna(subset=["Ticker", "Buy date"])
lots = lots[lots["Ticker"] != ""]
if lots.empty:
    st.info("Add at least one lot above.")
    st.stop()

# Download once for the whole set
first_buy_dt = min(pd.to_datetime(lots["Buy date"]).dt.date)
raw_prices = download_prices(
    tickers=sorted(lots["Ticker"].unique().tolist()),
    start=first_buy_dt - dt.timedelta(days=14),
    end=sell_date,
)

# Build positions
records = []
for _, r in lots.iterrows():
    sym = r["Ticker"]
    buy_dt = pd.to_datetime(r["Buy date"]).date()
    cash = float(r["Cash ($)"]) if pd.notna(r["Cash ($)"]) else 0.0
    shares_in = float(r["Shares"]) if pd.notna(r["Shares"]) else None
    manual_px = float(r["Manual price ($)"]) if pd.notna(r["Manual price ($)"]) and r["Manual price ($)"] > 0 else None

    # price to use
    if shares_in is None:
        use_buy_px = manual_px if manual_px is not None else px_on_or_before(raw_prices, sym, buy_dt)
        if not use_buy_px or use_buy_px <= 0:
            continue
        shares = cash / use_buy_px if cash > 0 else 0.0
    else:
        shares = shares_in
        use_buy_px = manual_px if manual_px is not None else px_on_or_before(raw_prices, sym, buy_dt)

    last_px = px_on_or_before(raw_prices, sym, sell_date)
    value = shares * last_px if last_px else 0.0

    records.append(
        {
            "Ticker": sym,
            "Buy date": buy_dt,
            "Used buy price ($)": use_buy_px,
            "Cash ($)": cash,
            "Shares": shares,
            "Last price ($)": last_px,
            "Value ($)": value,
        }
    )

positions = pd.DataFrame(records)
if positions.empty:
    st.warning("No positions could be computed. Check tickers/dates/prices.")
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
agg = agg.sort_values("Value", ascending=False).reset_index(drop=True)

# ----- layout
left, right = st.columns([2.2, 1], gap="large")

with left:
    # Chart option for multiple tickers
    norm_to_100 = st.toggle(
        "Normalize to 100 at each ticker's first buy date",
        value=True,
        help="Helps compare tickers on the same scale."
    )
    # ---------- header + chart ----------
    uniq = positions["Ticker"].unique().tolist()

    if len(uniq) == 1:
        # Single-ticker header with logo + name + price chart
        sym = uniq[0]
        name, logo = get_company_meta(sym)
        logo_html = f'<img src="{logo}" alt="{sym} logo" class="cf-logo"/>' if logo else ""
        name_html = f' — {name}' if name else ""
        st.markdown(
            f"""<div class="cf-stockline">{logo_html}<div class="cf-stocktext"><b>{sym}</b>{name_html}</div></div>""",
            unsafe_allow_html=True,
        )

        # price chart: first buy of that sym to sell date
        if isinstance(raw_prices.columns, pd.MultiIndex):
            df_sym = raw_prices[sym]
        else:
            df_sym = raw_prices
        s = _adj_close_series(df_sym)
        first_sym_buy = positions.loc[positions["Ticker"] == sym, "Buy date"].min()
        seg = s.loc[pd.to_datetime(first_sym_buy):pd.to_datetime(sell_date)]
        chart_df = seg.reset_index().rename(columns={"index": "Date", seg.name: "Adj Close ($)"})
        fig = px.line(chart_df, x="Date", y="Adj Close ($)", title=f"{sym} price")
        fig.update_layout(height=420, margin=dict(l=10, r=10, t=40, b=10))
        st.plotly_chart(fig, use_container_width=True)

    else:
        # Multiple tickers: overlay in one chart
        rows = []
        multi_cols = isinstance(raw_prices.columns, pd.MultiIndex)

        for sym in uniq:
            try:
                df_sym = raw_prices[sym] if multi_cols else raw_prices
                s = _adj_close_series(df_sym)
                if s.empty:
                    continue

                first_buy = positions.loc[positions["Ticker"] == sym, "Buy date"].min()
                seg = s.loc[pd.to_datetime(first_buy):pd.to_datetime(sell_date)].rename("Price")
                if seg.empty:
                    continue

                df = seg.reset_index().rename(columns={"index": "Date"})
                if norm_to_100:
                    base = float(seg.iloc[0])
                    if base > 0:
                        df["Price"] = 100.0 * df["Price"] / base
                df["Ticker"] = sym
                rows.append(df)
            except Exception:
                continue

        if not rows:
            st.info("Not enough data to draw the comparison chart.")
        else:
            overlay = pd.concat(rows, ignore_index=True)
            y_label = "Index (100 = first buy)" if norm_to_100 else "Adj Close ($)"
            fig = px.line(
                overlay,
                x="Date", y="Price", color="Ticker",
                labels={"Price": y_label, "Date": "Date"}
            )
            fig.update_layout(height=420, margin=dict(l=10, r=10, t=30, b=10))
            st.plotly_chart(fig, use_container_width=True)

with right:
    st.markdown('<div class="metric-card">', unsafe_allow_html=True)
    st.subheader("Summary")
    st.dataframe(
        agg.rename(columns={
            "Invested": "Invested ($)",
            "LastPrice": "Last price ($)",
            "Value": "Value ($)",
        }),
        use_container_width=True,
        height=220,   # compact, scrollable if long
        hide_index=True,
    )

    invested_total = float(agg["Invested"].sum())
    value_total = float(agg["Value"].sum())
    pl_total = value_total - invested_total
    pl_pct_total = (pl_total / invested_total * 100.0) if invested_total > 0 else np.nan
    st.metric("Portfolio value", f"${value_total:,.2f}", delta=f"{pl_total:,.2f}  ({pl_pct_total:.2f}%)")
st.markdown('</div>', unsafe_allow_html=True)

st.subheader("Lots (expanded)")
st.dataframe(
    positions,
    use_container_width=True,
    height=260,
    hide_index=True,
)
