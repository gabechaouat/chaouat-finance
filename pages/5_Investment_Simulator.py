import datetime as dt
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import streamlit as st
import yfinance as yf

st.set_page_config(
    page_title="Investment Simulator — Chaouat Economics Lab",
    page_icon="🧮",
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
.block-container { padding-top: 0 !important; padding-bottom: 3rem; max-width: 1280px; }

/* Masthead */
.is-masthead {
  border-bottom: 3px double var(--rule);
  padding: 28px 0 18px 0; margin-bottom: 0;
}
.is-eyebrow {
  font-size: 10px; letter-spacing: 3.5px; text-transform: uppercase;
  color: var(--ink-muted); margin-bottom: 8px;
}
.is-title {
  font-family: 'DM Serif Display', serif !important;
  font-size: 52px; line-height: 1.0; color: var(--ink);
  margin: 0 0 8px 0; letter-spacing: -0.5px;
}
.is-sub { font-size: 15px; color: var(--ink-muted); font-weight: 300; }

/* Section labels */
.section-label {
  font-size: 10px; letter-spacing: 3px; text-transform: uppercase;
  color: var(--ink-muted); border-top: 1px solid var(--rule);
  padding-top: 10px; margin: 28px 0 16px 0;
}
.section-label-flush {
  font-size: 10px; letter-spacing: 3px; text-transform: uppercase;
  color: var(--ink-muted); margin-bottom: 14px;
}

/* KPI cards */
.kpi-card {
  background: var(--warm); border: 1px solid var(--rule);
  border-radius: 4px; padding: 16px 18px; text-align: center;
}
.kpi-num {
  font-family: 'DM Serif Display', serif !important;
  font-size: 28px; line-height: 1; color: var(--ink); margin: 0;
}
.kpi-num-pos { color: #3a6b1a; }
.kpi-num-neg { color: var(--sienna); }
.kpi-num-terra { color: var(--terra); }
.kpi-label {
  font-size: 10px; letter-spacing: 2px; text-transform: uppercase;
  color: var(--ink-muted); margin-top: 5px;
}

/* Ticker badge */
.ticker-badge {
  display: inline-block; padding: 2px 8px; border-radius: 2px;
  font-size: 11px; font-weight: 600; letter-spacing: 1px;
  background: var(--ink); color: #fff; margin-right: 4px;
}

/* Holding row */
.holding-row {
  display: flex; align-items: center; gap: 12px;
  padding: 10px 0; border-bottom: 1px solid var(--rule);
}
.holding-row:last-child { border-bottom: none; }
.holding-bar-bg {
  flex: 1; height: 4px; background: var(--rule); border-radius: 2px;
}
.holding-bar-fill {
  height: 4px; border-radius: 2px;
}

/* Interp box */
.interp-box {
  border-left: 3px solid var(--terra); background: var(--terra-light);
  padding: 12px 16px; border-radius: 0 4px 4px 0; margin-top: 12px;
  font-size: 13px; color: var(--ink-muted); line-height: 1.65;
}
.interp-box-sand {
  border-left-color: var(--sand-dark); background: #f5eedd;
}
.interp-box-stone {
  border-left-color: var(--stone); background: var(--stone-light);
}

/* Buttons */
div.stButton > button {
  background: var(--ink) !important; color: #fff !important;
  border: none !important; border-radius: 3px !important;
  font-size: 11px !important; letter-spacing: 1.5px !important;
  text-transform: uppercase !important; font-weight: 500 !important;
  padding: 9px 16px !important; width: 100%;
}
div.stButton > button:hover { background: var(--terra) !important; }

div.stDownloadButton > button {
  background: transparent !important; color: var(--ink) !important;
  border: 1px solid var(--rule) !important; border-radius: 3px !important;
  font-size: 11px !important; letter-spacing: 1.5px !important;
  text-transform: uppercase !important; width: 100%;
}
div.stDownloadButton > button:hover {
  border-color: var(--terra) !important; color: var(--terra) !important;
}

/* Tabs */
[data-baseweb="tab-list"] { border-bottom: 2px solid var(--rule) !important; gap: 0 !important; }
[data-baseweb="tab"] {
  font-size: 11px !important; letter-spacing: 2px !important;
  text-transform: uppercase !important; color: var(--ink-muted) !important;
  padding: 10px 20px !important; background: transparent !important;
  border-bottom: 2px solid transparent !important; margin-bottom: -2px !important;
}
[aria-selected="true"][data-baseweb="tab"] {
  color: var(--ink) !important; border-bottom: 2px solid var(--terra) !important;
  font-weight: 500 !important;
}
[data-baseweb="tab-highlight"], [data-baseweb="tab-border"] { display: none !important; }

/* Slider / select labels */
[data-testid="stSlider"] label,
[data-testid="stSelectbox"] label,
[data-testid="stNumberInput"] label { font-size: 12px !important; color: var(--ink-muted) !important; }

/* Sidebar */
[data-testid="stSidebar"] { background: var(--warm) !important; border-right: 1px solid var(--rule) !important; }

/* Footer */
.site-footer {
  border-top: 3px double var(--rule); padding-top: 16px; margin-top: 40px;
  font-size: 12px; color: var(--ink-muted); text-align: center;
}
</style>
""", unsafe_allow_html=True)

# =====================================================================
# EARTH PALETTE for Plotly
# =====================================================================
C = {
    "terra":     "#c9622a",
    "terra_mid": "#a84e20",
    "sienna":    "#8b3a1a",
    "sand":      "#c4a882",
    "sand_dark": "#9e8060",
    "stone":     "#7a6f62",
    "ink":       "#1a1814",
    "muted":     "#b0ada8",
    "rule":      "#e0dbd2",
    "cream":     "#faf8f4",
    "warm":      "#f2ede4",
    "green":     "#3a6b1a",
}
PALETTE = [C["terra"], C["sienna"], C["sand_dark"], C["stone"], C["ink"],
           C["sand"], C["terra_mid"], C["muted"]]

def plot_layout(height=400, title="", show_legend=True):
    return dict(
        height=height,
        paper_bgcolor=C["cream"], plot_bgcolor=C["cream"],
        font=dict(family="DM Sans", color=C["ink"], size=12),
        title=dict(text=title, font=dict(family="DM Serif Display", size=15, color=C["ink"]),
                   x=0, xanchor="left") if title else dict(text=""),
        margin=dict(l=16, r=16, t=44 if title else 20, b=16),
        legend=dict(orientation="h", y=1.08, x=0, font=dict(size=11)) if show_legend else dict(visible=False),
        xaxis=dict(gridcolor=C["rule"], linecolor=C["rule"], tickfont=dict(size=11), zeroline=False),
        yaxis=dict(gridcolor=C["rule"], linecolor=C["rule"], tickfont=dict(size=11), zeroline=False),
        colorway=PALETTE,
        hovermode="x unified",
    )

# =====================================================================
# HELPERS
# =====================================================================
@st.cache_data(ttl=30*60, show_spinner=False)
def fetch_prices(tickers: tuple, start: dt.date, end: dt.date) -> pd.DataFrame:
    """Returns tidy DataFrame: Date, Ticker, Close"""
    if not tickers:
        return pd.DataFrame(columns=["Date","Ticker","Close"])
    raw = yf.download(
        list(tickers), start=start, end=end + dt.timedelta(days=1),
        interval="1d", auto_adjust=True, progress=False,
        group_by="ticker", threads=True,
    )
    rows = []
    for sym in tickers:
        try:
            if isinstance(raw.columns, pd.MultiIndex):
                close = raw[(sym, "Close")].dropna()
            else:
                close = raw["Close"].dropna()
            df_s = close.rename("Close").reset_index()
            df_s.columns = ["Date", "Close"]
            df_s["Ticker"] = sym
            rows.append(df_s[["Date","Ticker","Close"]])
        except: continue
    if not rows:
        return pd.DataFrame(columns=["Date","Ticker","Close"])
    out = pd.concat(rows, ignore_index=True)
    out["Date"] = pd.to_datetime(out["Date"])
    return out.dropna()

@st.cache_data(ttl=30*60, show_spinner=False)
def fetch_spy(start: dt.date, end: dt.date) -> pd.Series:
    raw = yf.download("SPY", start=start, end=end+dt.timedelta(days=1),
                      auto_adjust=True, progress=False)
    if raw.empty: return pd.Series(dtype=float)
    close = raw["Close"].dropna()
    close.index = pd.to_datetime(close.index)
    return close

def colour_for(i): return PALETTE[i % len(PALETTE)]

def fmt_money(v):
    if abs(v) >= 1e6: return f"${v/1e6:.2f}M"
    if abs(v) >= 1e3: return f"${v/1e3:.1f}K"
    return f"${v:.2f}"

def fmt_pct(v): return f"{v:+.2f}%"

# =====================================================================
# MASTHEAD
# =====================================================================
st.markdown("""
<div class="is-masthead">
  <div class="is-eyebrow">Chaouat Economics Lab · Finance Tools</div>
  <div class="is-title">Investment Simulator</div>
  <div class="is-sub">Build a portfolio, analyse performance, stress-test with Monte Carlo, and compare against the market.</div>
</div>
""", unsafe_allow_html=True)

# =====================================================================
# SIDEBAR — Portfolio builder
# =====================================================================
with st.sidebar:
    st.markdown("### Portfolio")

    # Valuation date
    val_date = st.date_input("Valuation date", dt.date.today(), key="val_date")

    st.markdown("---")
    st.markdown("### Benchmark")
    benchmark = st.selectbox("Compare against", ["SPY (S&P 500)", "QQQ (Nasdaq 100)", "GLD (Gold)", "None"], key="bench")
    bench_ticker = benchmark.split(" ")[0] if benchmark != "None" else None

    st.markdown("---")
    st.markdown("### Monte Carlo")
    mc_years   = st.slider("Projection horizon (years)", 1, 10, 3, key="mc_yr")
    mc_sims    = st.slider("Simulations", 100, 2000, 500, 100, key="mc_n")
    mc_enabled = st.checkbox("Run Monte Carlo", value=True, key="mc_on")

    st.markdown("---")
    if st.button("Clear price cache", key="clear_cache"):
        fetch_prices.clear()
        fetch_spy.clear()
        st.rerun()

# =====================================================================
# LOT ENTRY TABLE
# =====================================================================
st.markdown('<div class="section-label">Portfolio — enter your lots</div>', unsafe_allow_html=True)

init_df = pd.DataFrame([
    {"Ticker": "AAPL", "Buy date": dt.date.today()-dt.timedelta(days=730), "Cash ($)": 5000.0, "Shares": None, "Manual price ($)": None},
    {"Ticker": "MSFT", "Buy date": dt.date.today()-dt.timedelta(days=365), "Cash ($)": 4000.0, "Shares": None, "Manual price ($)": None},
    {"Ticker": "NVDA", "Buy date": dt.date.today()-dt.timedelta(days=200), "Cash ($)": 3000.0, "Shares": None, "Manual price ($)": None},
    {"Ticker": "",     "Buy date": dt.date.today()-dt.timedelta(days=100), "Cash ($)": None,   "Shares": None, "Manual price ($)": None},
])

lots_raw = st.data_editor(
    init_df, num_rows="dynamic", use_container_width=True, hide_index=True,
    column_config={
        "Ticker":          st.column_config.TextColumn("Ticker", help="e.g. AAPL, TSLA", width="small"),
        "Buy date":        st.column_config.DateColumn("Buy date", format="YYYY-MM-DD"),
        "Cash ($)":        st.column_config.NumberColumn("Cash invested ($)", min_value=0, step=100.0),
        "Shares":          st.column_config.NumberColumn("Shares (optional)", min_value=0, step=0.01),
        "Manual price ($)":st.column_config.NumberColumn("Manual price ($)", min_value=0, step=0.01,
                           help="Override historical price"),
    },
)

# ── Clean lots ──
lots = lots_raw.copy()
lots["Ticker"] = lots["Ticker"].astype(str).str.upper().str.strip()
lots = lots[lots["Ticker"].str.len() > 0].dropna(subset=["Buy date"])
lots = lots[lots["Ticker"] != "NAN"].reset_index(drop=True)

if lots.empty:
    st.info("Add at least one lot above to begin.")
    st.stop()

# ── Date range ──
all_buy_dates = pd.to_datetime(lots["Buy date"]).dt.date
first_buy     = min(all_buy_dates) - dt.timedelta(days=5)
tickers_tuple = tuple(sorted(lots["Ticker"].unique()))

# ── Fetch prices ──
with st.spinner("Fetching price data…"):
    prices_all = fetch_prices(tickers_tuple, first_buy, val_date)

if prices_all.empty:
    st.error("Could not fetch price data. Check tickers and try again.")
    st.stop()

# ── Resolve positions ──
def px_on_or_before(sym, when):
    sub = prices_all[prices_all["Ticker"]==sym].copy()
    if sub.empty: return None
    sub["_d"] = sub["Date"].dt.date
    cands = sub[sub["_d"] <= when]
    if cands.empty: return None
    return float(cands.sort_values("Date").iloc[-1]["Close"])

records = []
for _, r in lots.iterrows():
    sym      = r["Ticker"]
    buy_d    = pd.to_datetime(r["Buy date"]).date()
    cash     = float(r["Cash ($)"]) if pd.notna(r.get("Cash ($)")) else 0.0
    shares_in= float(r["Shares"])   if pd.notna(r.get("Shares"))   else None
    man_px   = float(r["Manual price ($)"]) if pd.notna(r.get("Manual price ($)")) and r.get("Manual price ($)", 0) > 0 else None

    buy_px = man_px if man_px else px_on_or_before(sym, buy_d)
    if buy_px is None or buy_px <= 0: continue

    shares = shares_in if shares_in else (cash / buy_px if cash > 0 else 0)
    last_px = px_on_or_before(sym, val_date) or buy_px
    value   = shares * last_px
    cost    = shares * buy_px

    records.append({
        "Ticker":     sym,
        "Buy date":   buy_d,
        "Buy price":  buy_px,
        "Shares":     shares,
        "Cost basis": cost,
        "Value":      value,
        "P/L ($)":    value - cost,
        "P/L (%)":    100*(value-cost)/cost if cost > 0 else 0,
    })

if not records:
    st.error("Could not compute positions — check tickers and dates.")
    st.stop()

positions = pd.DataFrame(records)

# ── Aggregate per ticker ──
agg = positions.groupby("Ticker", as_index=False).agg(
    Cost=("Cost basis","sum"),
    Value=("Value","sum"),
    Shares=("Shares","sum"),
)
agg["P/L ($)"]  = agg["Value"] - agg["Cost"]
agg["P/L (%)"]  = 100 * agg["P/L ($)"] / agg["Cost"].replace(0, np.nan)
agg = agg.sort_values("Value", ascending=False).reset_index(drop=True)

total_cost  = agg["Cost"].sum()
total_value = agg["Value"].sum()
total_pl    = total_value - total_cost
total_pl_pct= 100*total_pl/total_cost if total_cost > 0 else 0
best  = agg.loc[agg["P/L (%)"].idxmax()]
worst = agg.loc[agg["P/L (%)"].idxmin()]

# =====================================================================
# KPI STRIP
# =====================================================================
st.markdown('<div class="section-label">Portfolio summary</div>', unsafe_allow_html=True)

k1,k2,k3,k4,k5,k6 = st.columns(6, gap="medium")
pl_cls = "kpi-num-pos" if total_pl >= 0 else "kpi-num-neg"

for col, num, lbl, cls in [
    (k1, fmt_money(total_value),    "Portfolio value",     "kpi-num-terra"),
    (k2, fmt_money(total_cost),     "Total invested",      "kpi-num"),
    (k3, fmt_money(total_pl),       "Total P/L",           pl_cls),
    (k4, fmt_pct(total_pl_pct),     "Return",              pl_cls),
    (k5, f"{best['Ticker']} {best['P/L (%)']:+.1f}%",  "Best performer",  "kpi-num-pos"),
    (k6, f"{worst['Ticker']} {worst['P/L (%)']:+.1f}%","Worst performer", "kpi-num-neg"),
]:
    with col:
        st.markdown(f'<div class="kpi-card"><div class="kpi-num {cls}">{num}</div><div class="kpi-label">{lbl}</div></div>', unsafe_allow_html=True)

# =====================================================================
# FETCH BENCHMARK
# =====================================================================
spy_series = pd.Series(dtype=float)
if bench_ticker:
    with st.spinner(f"Fetching {bench_ticker}…"):
        spy_series = fetch_spy(first_buy, val_date)
        if not isinstance(spy_series, pd.Series):
            spy_series = pd.Series(dtype=float)
        # If it's a DataFrame column, squeeze
        if isinstance(spy_series, pd.DataFrame):
            spy_series = spy_series.squeeze()

# =====================================================================
# TABS
# =====================================================================
t1, t2, t3, t4, t5 = st.tabs([
    "01 · Portfolio value",
    "02 · Performance breakdown",
    "03 · Risk & correlation",
    "04 · Drawdown",
    "05 · Monte Carlo",
])

# ─────────────────────────────────────────────────────────────────────
# TAB 1 — PORTFOLIO VALUE OVER TIME + ALLOCATION
# ─────────────────────────────────────────────────────────────────────
with t1:
    left, right = st.columns([2.2, 1], gap="large")

    with left:
        st.markdown('<div class="section-label-flush">Portfolio value over time</div>', unsafe_allow_html=True)

        # Build daily portfolio value
        date_range = pd.date_range(first_buy, val_date, freq="B")
        port_val   = pd.Series(0.0, index=date_range)
        cost_val   = pd.Series(0.0, index=date_range)

        for _, r in positions.iterrows():
            sym    = r["Ticker"]
            buy_d  = pd.Timestamp(r["Buy date"])
            shares = r["Shares"]
            cost_u = r["Buy price"]

            sub = prices_all[prices_all["Ticker"]==sym][["Date","Close"]].set_index("Date")["Close"]
            sub = sub.reindex(date_range, method="ffill")
            active = sub.copy(); active[date_range < buy_d] = np.nan

            port_val += (active * shares).fillna(0)
            cost_val += np.where(date_range >= buy_d, shares * cost_u, 0)

        port_val = port_val[port_val > 0]
        cost_val_s = pd.Series(cost_val.values, index=date_range)[port_val.index]

        fig1 = go.Figure()
        # Cost basis area
        fig1.add_trace(go.Scatter(
            x=cost_val_s.index, y=cost_val_s.values,
            name="Cost basis", mode="lines",
            line=dict(color=C["rule"], width=1.5, dash="dot"),
            fill=None,
        ))
        # Portfolio value area
        fig1.add_trace(go.Scatter(
            x=port_val.index, y=port_val.values,
            name="Portfolio value", mode="lines",
            line=dict(color=C["terra"], width=2.5),
            fill="tonexty",
            fillcolor="rgba(201,98,42,0.10)",
        ))

        # Benchmark overlay (normalised to starting portfolio value)
        if not spy_series.empty and len(port_val) > 0:
            spy_aligned = spy_series.reindex(port_val.index, method="ffill").dropna()
            if not spy_aligned.empty:
                spy_norm = spy_aligned / spy_aligned.iloc[0] * port_val.iloc[0]
                fig1.add_trace(go.Scatter(
                    x=spy_norm.index, y=spy_norm.values,
                    name=bench_ticker, mode="lines",
                    line=dict(color=C["stone"], width=1.8, dash="dash"),
                ))

        # Buy markers
        for i, pos_r in positions.iterrows():
            sym   = pos_r["Ticker"]
            buy_d = pd.Timestamp(pos_r["Buy date"])
            if buy_d in port_val.index:
                fig1.add_trace(go.Scatter(
                    x=[buy_d], y=[port_val.get(buy_d, port_val.iloc[0])],
                    mode="markers+text",
                    marker=dict(size=9, color=PALETTE[i % len(PALETTE)],
                                symbol="circle", line=dict(color=C["cream"], width=2)),
                    text=[sym], textposition="top center",
                    textfont=dict(size=10, color=C["ink"]),
                    name=f"Buy {sym}", showlegend=False,
                ))

        fig1.update_layout(**plot_layout(420, "Portfolio value vs cost basis"))
        fig1.update_yaxes(title_text="USD ($)", tickprefix="$")
        fig1.update_xaxes(title_text="Date")
        st.plotly_chart(fig1, use_container_width=True)

        # Interpretation
        if not spy_series.empty and not spy_aligned.empty:
            spy_ret = (spy_aligned.iloc[-1]/spy_aligned.iloc[0]-1)*100
            outperf = total_pl_pct - spy_ret
            perf_word = "outperformed" if outperf > 0 else "underperformed"
            st.markdown(f"""
            <div class="interp-box">
              Your portfolio returned <strong>{total_pl_pct:+.2f}%</strong> vs
              <strong>{spy_ret:+.2f}%</strong> for {bench_ticker} over the same period —
              <strong>{abs(outperf):.2f}pp {perf_word}</strong>.
              Entry points are marked on the chart. The shaded area shows the gap between
              your current value and what you put in.
            </div>
            """, unsafe_allow_html=True)

    with right:
        st.markdown('<div class="section-label-flush">Allocation by value</div>', unsafe_allow_html=True)

        total_v = agg["Value"].sum()
        for i, row in agg.iterrows():
            pct   = row["Value"]/total_v*100
            color = PALETTE[i % len(PALETTE)]
            pl_c  = "#3a6b1a" if row["P/L ($)"] >= 0 else C["sienna"]
            st.markdown(f"""
            <div class="holding-row">
              <span class="ticker-badge" style="background:{color};">{row['Ticker']}</span>
              <div style="flex:1;">
                <div class="holding-bar-bg">
                  <div class="holding-bar-fill" style="width:{pct:.0f}%;background:{color};"></div>
                </div>
              </div>
              <div style="text-align:right;min-width:90px;">
                <div style="font-size:13px;font-weight:500;color:var(--ink);">{fmt_money(row['Value'])}</div>
                <div style="font-size:11px;color:{pl_c};">{row['P/L (%)']:+.1f}%</div>
              </div>
            </div>
            """, unsafe_allow_html=True)

        # Donut (plotly)
        fig_don = go.Figure(go.Pie(
            labels=agg["Ticker"].tolist(),
            values=agg["Value"].tolist(),
            hole=0.6,
            marker=dict(colors=PALETTE[:len(agg)],
                        line=dict(color=C["cream"], width=3)),
            textinfo="label+percent",
            textfont=dict(size=11, family="DM Sans"),
            hovertemplate="%{label}: $%{value:,.0f}<extra></extra>",
        ))
        fig_don.update_layout(
            height=260, paper_bgcolor=C["cream"],
            margin=dict(l=8,r=8,t=8,b=8),
            showlegend=False,
            annotations=[dict(text=fmt_money(total_v), x=0.5, y=0.5, showarrow=False,
                              font=dict(size=15, family="DM Serif Display", color=C["ink"]))]
        )
        st.plotly_chart(fig_don, use_container_width=True)

# ─────────────────────────────────────────────────────────────────────
# TAB 2 — PERFORMANCE BREAKDOWN
# ─────────────────────────────────────────────────────────────────────
with t2:
    st.markdown('<div class="section-label-flush">Individual holding performance</div>', unsafe_allow_html=True)

    c2l, c2r = st.columns([1.6, 1], gap="large")

    with c2l:
        # Waterfall: P/L per holding then total
        syms_w  = agg["Ticker"].tolist() + ["Total"]
        vals_w  = agg["P/L ($)"].tolist() + [total_pl]
        meas_w  = ["relative"]*len(agg) + ["total"]
        cols_w  = [C["green"] if v>=0 else C["sienna"] for v in agg["P/L ($)"].tolist()] + [C["terra"]]

        fig2a = go.Figure(go.Waterfall(
            x=syms_w, y=vals_w, measure=meas_w,
            connector=dict(line=dict(color=C["rule"], width=1)),
            increasing=dict(marker=dict(color=C["green"])),
            decreasing=dict(marker=dict(color=C["sienna"])),
            totals=dict(marker=dict(color=C["terra"])),
            text=[f"${v:,.0f}" for v in vals_w],
            textposition="outside",
            textfont=dict(size=11),
            hovertemplate="%{x}: %{y:$,.2f}<extra></extra>",
        ))
        fig2a.update_layout(**plot_layout(360, "P/L waterfall — contribution per holding", show_legend=False))
        fig2a.update_yaxes(title_text="P/L ($)", tickprefix="$")
        st.plotly_chart(fig2a, use_container_width=True)

        # Normalised price history (100 at first buy)
        fig2b = go.Figure()
        for i, sym in enumerate(tickers_tuple):
            sub = prices_all[prices_all["Ticker"]==sym][["Date","Close"]].set_index("Date")["Close"]
            # Find earliest buy for this sym
            sym_buys = positions[positions["Ticker"]==sym]["Buy date"]
            if sym_buys.empty: continue
            first_sym_buy = pd.Timestamp(sym_buys.min())
            seg = sub[sub.index >= first_sym_buy]
            if seg.empty: continue
            norm = 100 * seg / seg.iloc[0]
            fig2b.add_trace(go.Scatter(
                x=norm.index, y=norm.values, name=sym, mode="lines",
                line=dict(color=PALETTE[i % len(PALETTE)], width=2),
                hovertemplate=f"{sym}: %{{y:.1f}}<extra></extra>",
            ))

        if bench_ticker and not spy_series.empty:
            spy_seg = spy_series[spy_series.index >= pd.Timestamp(first_buy)]
            if not spy_seg.empty:
                fig2b.add_trace(go.Scatter(
                    x=spy_seg.index, y=100*spy_seg/spy_seg.iloc[0],
                    name=bench_ticker, mode="lines",
                    line=dict(color=C["stone"], width=1.5, dash="dash"),
                ))

        fig2b.add_hline(y=100, line_color=C["rule"], line_width=1, line_dash="dot")
        fig2b.update_layout(**plot_layout(360, "Normalised price (100 = purchase date)"))
        fig2b.update_yaxes(title_text="Index (100 = buy date)")
        st.plotly_chart(fig2b, use_container_width=True)

    with c2r:
        st.markdown('<div class="section-label-flush">Holding detail</div>', unsafe_allow_html=True)

        for i, row in agg.iterrows():
            color  = PALETTE[i % len(PALETTE)]
            pl_pos = row["P/L ($)"] >= 0
            pl_col = "#3a6b1a" if pl_pos else C["sienna"]
            st.markdown(f"""
            <div style="border:1px solid var(--rule);border-radius:4px;padding:14px 16px;
                        margin-bottom:10px;background:var(--cream);">
              <div style="display:flex;align-items:center;gap:8px;margin-bottom:10px;">
                <span class="ticker-badge" style="background:{color};">{row['Ticker']}</span>
                <span style="font-size:13px;font-weight:500;color:{pl_col};">{row['P/L (%)']:+.1f}%</span>
              </div>
              <div style="display:grid;grid-template-columns:1fr 1fr;gap:4px;font-size:12.5px;">
                <span style="color:var(--ink-muted);">Invested</span>
                <span style="text-align:right;color:var(--ink);">{fmt_money(row['Cost'])}</span>
                <span style="color:var(--ink-muted);">Current value</span>
                <span style="text-align:right;color:var(--ink);">{fmt_money(row['Value'])}</span>
                <span style="color:var(--ink-muted);">P/L</span>
                <span style="text-align:right;color:{pl_col};font-weight:500;">{fmt_money(row['P/L ($)'])}</span>
                <span style="color:var(--ink-muted);">Shares</span>
                <span style="text-align:right;color:var(--ink);">{row['Shares']:.4f}</span>
              </div>
            </div>
            """, unsafe_allow_html=True)

        # Download
        st.download_button(
            "Export positions CSV",
            data=positions.to_csv(index=False).encode(),
            file_name="portfolio_positions.csv",
            mime="text/csv", key="dl_pos",
        )

# ─────────────────────────────────────────────────────────────────────
# TAB 3 — RISK & CORRELATION
# ─────────────────────────────────────────────────────────────────────
with t3:
    st.markdown('<div class="section-label-flush">Risk & return analysis</div>', unsafe_allow_html=True)

    # Compute daily returns per ticker
    ret_dict = {}
    for sym in tickers_tuple:
        sub = prices_all[prices_all["Ticker"]==sym][["Date","Close"]].set_index("Date")["Close"]
        sub = sub.sort_index()
        ret_dict[sym] = sub.pct_change().dropna()

    if not ret_dict:
        st.info("Not enough data for risk analysis.")
    else:
        ret_df = pd.DataFrame(ret_dict).dropna(how="all")

        c3l, c3r = st.columns([1.3, 1], gap="large")

        with c3l:
            # Risk/return scatter
            ann_ret  = ret_df.mean()  * 252
            ann_vol  = ret_df.std()   * np.sqrt(252)
            sharpe   = ann_ret / ann_vol.replace(0, np.nan)

            fig3a = go.Figure()
            for i, sym in enumerate(ann_ret.index):
                r_val = ann_ret[sym]*100
                v_val = ann_vol[sym]*100
                s_val = sharpe[sym]
                size  = max(12, min(40, abs(agg.loc[agg["Ticker"]==sym,"Value"].sum() / total_value * 120)))
                color = PALETTE[i % len(PALETTE)]
                fig3a.add_trace(go.Scatter(
                    x=[v_val], y=[r_val],
                    mode="markers+text",
                    name=sym,
                    marker=dict(size=size, color=color,
                                line=dict(color=C["cream"], width=2)),
                    text=[sym], textposition="top center",
                    textfont=dict(size=11, color=C["ink"]),
                    hovertemplate=(f"<b>{sym}</b><br>Ann. return: %{{y:.1f}}%"
                                   f"<br>Volatility: %{{x:.1f}}%"
                                   f"<br>Sharpe: {s_val:.2f}<extra></extra>"),
                ))

            # Benchmark dot
            if bench_ticker and not spy_series.empty:
                spy_r = spy_series.pct_change().dropna()
                spy_ann_r = spy_r.mean()*252*100
                spy_ann_v = spy_r.std()*np.sqrt(252)*100
                fig3a.add_trace(go.Scatter(
                    x=[spy_ann_v], y=[spy_ann_r],
                    mode="markers+text", name=bench_ticker,
                    marker=dict(size=16, color=C["stone"], symbol="diamond",
                                line=dict(color=C["cream"], width=2)),
                    text=[bench_ticker], textposition="top center",
                    textfont=dict(size=11),
                ))

            fig3a.add_hline(y=0, line_color=C["rule"], line_width=1)
            fig3a.update_layout(**plot_layout(400, "Risk / return scatter (bubble size = portfolio weight)"))
            fig3a.update_xaxes(title_text="Annualised volatility (%)", ticksuffix="%")
            fig3a.update_yaxes(title_text="Annualised return (%)",     ticksuffix="%")
            st.plotly_chart(fig3a, use_container_width=True)

            st.markdown("""
            <div class="interp-box interp-box-stone">
              Holdings in the <strong>top-left</strong> have high returns and low volatility — ideal.
              <strong>Bottom-right</strong> means high risk, poor return.
              Bubble size reflects portfolio weight. Use this to spot whether your riskiest
              positions are actually delivering commensurate returns.
            </div>
            """, unsafe_allow_html=True)

        with c3r:
            # Correlation heatmap
            st.markdown('<div class="section-label-flush">Correlation matrix</div>', unsafe_allow_html=True)

            if ret_df.shape[1] >= 2:
                corr = ret_df.corr()
                syms_c = corr.columns.tolist()

                # Custom terracotta-cream-stone colorscale
                cscale = [[0, C["stone"]], [0.5, C["cream"]], [1, C["terra"]]]

                fig3b = go.Figure(go.Heatmap(
                    z=corr.values,
                    x=syms_c, y=syms_c,
                    colorscale=cscale,
                    zmin=-1, zmax=1,
                    text=[[f"{v:.2f}" for v in row] for row in corr.values],
                    texttemplate="%{text}",
                    textfont=dict(size=12, family="DM Sans"),
                    colorbar=dict(title="ρ", tickfont=dict(size=10), len=0.8),
                    hovertemplate="%{y} vs %{x}: %{z:.3f}<extra></extra>",
                ))
                fig3b.update_layout(
                    height=360, paper_bgcolor=C["cream"], plot_bgcolor=C["cream"],
                    font=dict(family="DM Sans", color=C["ink"], size=11),
                    margin=dict(l=16,r=16,t=40,b=16),
                    title=dict(text="Daily return correlation",
                               font=dict(family="DM Serif Display", size=14)),
                    xaxis=dict(side="bottom"),
                )
                st.plotly_chart(fig3b, use_container_width=True)

                avg_corr = corr.values[np.triu_indices_from(corr.values, k=1)].mean()
                diversification = "well diversified" if avg_corr < 0.5 else \
                                  "moderately correlated" if avg_corr < 0.75 else \
                                  "highly concentrated — consider diversifying"
                st.markdown(f"""
                <div class="interp-box interp-box-sand">
                  Average pairwise correlation: <strong>{avg_corr:.2f}</strong>.
                  Your portfolio appears <strong>{diversification}</strong>.
                  Terracotta = high positive correlation (holdings move together);
                  stone = negative (natural hedge).
                </div>
                """, unsafe_allow_html=True)
            else:
                st.info("Add at least 2 tickers to see the correlation matrix.")

            # Sharpe table
            st.markdown('<div class="section-label-flush" style="margin-top:20px;">Risk metrics</div>', unsafe_allow_html=True)
            risk_rows = ""
            for sym in ann_ret.index:
                s = sharpe[sym]
                s_col = "#3a6b1a" if s >= 1 else (C["sienna"] if s < 0 else C["sand_dark"])
                risk_rows += f"""
                <tr style="border-bottom:1px solid #e0dbd2;">
                  <td style="padding:8px 10px;font-weight:500;font-size:13px;">{sym}</td>
                  <td style="padding:8px 10px;text-align:right;font-size:13px;">{ann_ret[sym]*100:+.1f}%</td>
                  <td style="padding:8px 10px;text-align:right;font-size:13px;">{ann_vol[sym]*100:.1f}%</td>
                  <td style="padding:8px 10px;text-align:right;font-size:13px;color:{s_col};font-weight:500;">{s:.2f}</td>
                </tr>
                """
            st.markdown(f"""
            <div style="overflow-x:auto;border:1px solid #e0dbd2;border-radius:4px;">
              <table style="width:100%;border-collapse:collapse;background:#faf8f4;font-family:'DM Sans',sans-serif;">
                <thead>
                  <tr style="background:#f2ede4;border-bottom:2px solid #e0dbd2;">
                    <th style="padding:8px 10px;text-align:left;font-size:10px;letter-spacing:2px;text-transform:uppercase;color:#6b6760;font-weight:500;">Ticker</th>
                    <th style="padding:8px 10px;text-align:right;font-size:10px;letter-spacing:2px;text-transform:uppercase;color:#6b6760;font-weight:500;">Ann. Return</th>
                    <th style="padding:8px 10px;text-align:right;font-size:10px;letter-spacing:2px;text-transform:uppercase;color:#6b6760;font-weight:500;">Volatility</th>
                    <th style="padding:8px 10px;text-align:right;font-size:10px;letter-spacing:2px;text-transform:uppercase;color:#6b6760;font-weight:500;">Sharpe</th>
                  </tr>
                </thead>
                <tbody>{risk_rows}</tbody>
              </table>
            </div>
            """, unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────
# TAB 4 — DRAWDOWN
# ─────────────────────────────────────────────────────────────────────
with t4:
    st.markdown('<div class="section-label-flush">Drawdown — how far each holding fell from its peak</div>', unsafe_allow_html=True)

    fig4 = go.Figure()
    dd_summary = []

    for i, sym in enumerate(tickers_tuple):
        sub = prices_all[prices_all["Ticker"]==sym][["Date","Close"]].set_index("Date")["Close"]
        sub = sub.sort_index()
        rolling_max = sub.cummax()
        drawdown    = (sub / rolling_max - 1) * 100

        fig4.add_trace(go.Scatter(
            x=drawdown.index, y=drawdown.values,
            name=sym, mode="lines",
            line=dict(color=PALETTE[i % len(PALETTE)], width=1.8),
            fill="tozeroy",
            fillcolor=f"rgba{tuple(int(PALETTE[i%len(PALETTE)].lstrip('#')[j:j+2],16) for j in (0,2,4))+(0.08,)}",
            hovertemplate=f"{sym}: %{{y:.1f}}%<extra></extra>",
        ))
        dd_summary.append({
            "Ticker":       sym,
            "Max drawdown": f"{drawdown.min():.1f}%",
            "Current DD":   f"{drawdown.iloc[-1]:.1f}%",
            "Recovery":     "Recovered" if drawdown.iloc[-1] > -2 else "In drawdown",
        })

    # Portfolio drawdown
    if len(port_val) > 0:
        port_s = pd.Series(port_val.values, index=port_val.index)
        port_dd = (port_s / port_s.cummax() - 1) * 100
        fig4.add_trace(go.Scatter(
            x=port_dd.index, y=port_dd.values,
            name="Portfolio", mode="lines",
            line=dict(color=C["ink"], width=2.5, dash="dot"),
            hovertemplate="Portfolio: %{y:.1f}%<extra></extra>",
        ))

    fig4.add_hline(y=0, line_color=C["rule"], line_width=1)
    fig4.add_hline(y=-10, line_color=C["sand"], line_width=1, line_dash="dot",
                   annotation_text="-10%", annotation_font_color=C["sand_dark"])
    fig4.add_hline(y=-20, line_color=C["sienna"], line_width=1, line_dash="dot",
                   annotation_text="-20% (bear)", annotation_font_color=C["sienna"])

    fig4.update_layout(**plot_layout(440, "Drawdown from all-time high (%)"))
    fig4.update_yaxes(title_text="Drawdown (%)", ticksuffix="%")
    fig4.update_xaxes(title_text="Date")
    st.plotly_chart(fig4, use_container_width=True)

    # Summary table
    dd_rows = ""
    for r in dd_summary:
        rec_col = "#3a6b1a" if r["Recovery"]=="Recovered" else C["sienna"]
        dd_rows += f"""
        <tr style="border-bottom:1px solid #e0dbd2;">
          <td style="padding:9px 12px;font-weight:500;font-size:13px;">{r['Ticker']}</td>
          <td style="padding:9px 12px;text-align:right;font-size:13px;color:var(--sienna);font-weight:500;">{r['Max drawdown']}</td>
          <td style="padding:9px 12px;text-align:right;font-size:13px;">{r['Current DD']}</td>
          <td style="padding:9px 12px;text-align:center;font-size:12px;color:{rec_col};font-weight:500;">{r['Recovery']}</td>
        </tr>
        """
    st.markdown(f"""
    <div style="overflow-x:auto;border:1px solid #e0dbd2;border-radius:4px;margin-top:8px;">
      <table style="width:100%;border-collapse:collapse;background:#faf8f4;font-family:'DM Sans',sans-serif;">
        <thead>
          <tr style="background:#f2ede4;border-bottom:2px solid #e0dbd2;">
            <th style="padding:9px 12px;text-align:left;font-size:10px;letter-spacing:2px;text-transform:uppercase;color:#6b6760;font-weight:500;">Ticker</th>
            <th style="padding:9px 12px;text-align:right;font-size:10px;letter-spacing:2px;text-transform:uppercase;color:#6b6760;font-weight:500;">Max drawdown</th>
            <th style="padding:9px 12px;text-align:right;font-size:10px;letter-spacing:2px;text-transform:uppercase;color:#6b6760;font-weight:500;">Current</th>
            <th style="padding:9px 12px;text-align:center;font-size:10px;letter-spacing:2px;text-transform:uppercase;color:#6b6760;font-weight:500;">Status</th>
          </tr>
        </thead>
        <tbody>{dd_rows}</tbody>
      </table>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="interp-box" style="margin-top:12px;">
      The drawdown chart shows how far each holding (and the overall portfolio) fell from its rolling peak.
      The <strong>-10%</strong> line marks a standard correction; <strong>-20%</strong> marks a bear market.
      Holdings that are "In drawdown" have not yet recovered to their previous high.
    </div>
    """, unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────
# TAB 5 — MONTE CARLO PROJECTION
# ─────────────────────────────────────────────────────────────────────
with t5:
    st.markdown('<div class="section-label-flush">Monte Carlo portfolio projection</div>', unsafe_allow_html=True)

    if not mc_enabled:
        st.info("Enable Monte Carlo in the sidebar to run the projection.")
    elif len(port_val) < 20:
        st.info("Not enough price history to run Monte Carlo. Add older buy dates.")
    else:
        # Use portfolio daily returns
        port_series = pd.Series(port_val.values, index=port_val.index)
        port_ret = port_series.pct_change().dropna()

        mu    = float(port_ret.mean())
        sigma = float(port_ret.std())
        T     = mc_years * 252
        S0    = float(port_series.iloc[-1])

        np.random.seed(42)
        sim_mat = np.zeros((mc_sims, T+1))
        sim_mat[:,0] = S0
        shocks = np.random.normal(mu, sigma, (mc_sims, T))
        for t in range(1, T+1):
            sim_mat[:,t] = sim_mat[:,t-1] * (1 + shocks[:,t-1])

        # Percentile bands
        p5   = np.percentile(sim_mat, 5,  axis=0)
        p25  = np.percentile(sim_mat, 25, axis=0)
        p50  = np.percentile(sim_mat, 50, axis=0)
        p75  = np.percentile(sim_mat, 75, axis=0)
        p95  = np.percentile(sim_mat, 95, axis=0)

        future_dates = pd.date_range(port_series.index[-1], periods=T+1, freq="B")

        fig5 = go.Figure()

        # Draw a sample of individual paths (faint)
        sample_n = min(80, mc_sims)
        idx_sample = np.random.choice(mc_sims, sample_n, replace=False)
        for i in idx_sample:
            fig5.add_trace(go.Scatter(
                x=future_dates, y=sim_mat[i],
                mode="lines", showlegend=False,
                line=dict(color="rgba(201,98,42,0.06)", width=1),
                hoverinfo="skip",
            ))

        # Historical portfolio
        fig5.add_trace(go.Scatter(
            x=port_series.index, y=port_series.values,
            name="Historical", mode="lines",
            line=dict(color=C["ink"], width=2.5),
        ))

        # Outer band (5–95)
        fig5.add_trace(go.Scatter(
            x=np.concatenate([future_dates, future_dates[::-1]]),
            y=np.concatenate([p95, p5[::-1]]),
            fill="toself", fillcolor="rgba(201,98,42,0.08)",
            line=dict(width=0), name="5–95th pct",
        ))
        # Inner band (25–75)
        fig5.add_trace(go.Scatter(
            x=np.concatenate([future_dates, future_dates[::-1]]),
            y=np.concatenate([p75, p25[::-1]]),
            fill="toself", fillcolor="rgba(201,98,42,0.18)",
            line=dict(width=0), name="25–75th pct",
        ))
        # Median
        fig5.add_trace(go.Scatter(
            x=future_dates, y=p50, name="Median projection",
            mode="lines", line=dict(color=C["terra"], width=2.5),
        ))
        # p5 / p95 lines
        fig5.add_trace(go.Scatter(
            x=future_dates, y=p95, name="95th pct",
            mode="lines", line=dict(color=C["sand_dark"], width=1, dash="dot"),
        ))
        fig5.add_trace(go.Scatter(
            x=future_dates, y=p5, name="5th pct",
            mode="lines", line=dict(color=C["sienna"], width=1, dash="dot"),
        ))

        # Current value line
        fig5.add_hline(y=S0, line_color=C["rule"], line_width=1, line_dash="dot",
                       annotation_text="Today", annotation_font_color=C["muted"])

        fig5.update_layout(**plot_layout(460, f"Monte Carlo projection — {mc_years}yr, {mc_sims:,} simulations"))
        fig5.update_yaxes(title_text="Portfolio value ($)", tickprefix="$")
        st.plotly_chart(fig5, use_container_width=True)

        # Outcome summary
        final_vals = sim_mat[:,-1]
        prob_profit = (final_vals > S0).mean()*100
        prob_double = (final_vals > S0*2).mean()*100
        prob_halve  = (final_vals < S0*0.5).mean()*100
        median_final= np.median(final_vals)
        p5_final    = np.percentile(final_vals, 5)
        p95_final   = np.percentile(final_vals, 95)

        mo1,mo2,mo3,mo4,mo5 = st.columns(5, gap="medium")
        for col, val, lbl, cls in [
            (mo1, fmt_money(median_final), "Median outcome",    "kpi-num-terra"),
            (mo2, fmt_money(p5_final),     "5th pct (bad)",     "kpi-num-neg"),
            (mo3, fmt_money(p95_final),    "95th pct (good)",   "kpi-num-pos"),
            (mo4, f"{prob_profit:.0f}%",   "Prob. of profit",   "kpi-num"),
            (mo5, f"{prob_double:.0f}%",   "Prob. of 2× return","kpi-num"),
        ]:
            with col:
                st.markdown(f'<div class="kpi-card"><div class="kpi-num {cls}">{val}</div><div class="kpi-label">{lbl}</div></div>', unsafe_allow_html=True)

        st.markdown(f"""
        <div class="interp-box" style="margin-top:16px;">
          Based on <strong>{mc_sims:,} simulations</strong> using the portfolio's historical daily return
          (μ = {mu*252*100:.1f}%/yr, σ = {sigma*np.sqrt(252)*100:.1f}%/yr), over <strong>{mc_years} year{'s' if mc_years!=1 else ''}</strong>:
          there is a <strong>{prob_profit:.0f}% probability of profit</strong>,
          a <strong>{prob_double:.0f}% chance of doubling</strong>,
          and a <strong>{prob_halve:.0f}% chance of losing more than half</strong>.
          The shaded bands show the 25–75th and 5–95th percentile ranges.
          Individual simulation paths are shown faintly in the background.
          <br/><br/>
          <strong>Important:</strong> This model assumes returns follow a normal distribution
          with constant mean and variance — a simplification. Real markets exhibit fat tails,
          volatility clustering, and regime changes that this model does not capture.
        </div>
        """, unsafe_allow_html=True)

# =====================================================================
# FOOTER
# =====================================================================
st.markdown("""
<div class="site-footer">
  Chaouat Economics Lab — Investment Simulator · Data via Yahoo Finance · Educational use only<br/>
  <span style="font-size:11px;">Past performance does not predict future results. © Chaouat Economics Lab</span>
</div>
""", unsafe_allow_html=True)
