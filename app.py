import datetime as dt

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import pytz
import streamlit as st
import yfinance as yf

st.set_page_config(
    page_title="F&O Mobile Signal",
    page_icon="📈",
    layout="centered",
    initial_sidebar_state="collapsed",
)

IST = pytz.timezone("Asia/Kolkata")
OPEN_MIN = 9 * 60 + 15      # 09:15
SQUARE_OFF_MIN = 15 * 60 + 15
CLOSE_MIN = 15 * 60 + 30
MIN_CANDLES = 25            # EMA21 and RSI14 need history before they mean anything

INSTRUMENTS = {
    "NIFTY 50": {"ticker": "^NSEI", "step": 50},
    "BANK NIFTY": {"ticker": "^NSEBANK", "step": 100},
    "RELIANCE": {"ticker": "RELIANCE.NS", "step": 20},
    "HDFC BANK": {"ticker": "HDFCBANK.NS", "step": 10},
    "ICICI BANK": {"ticker": "ICICIBANK.NS", "step": 10},
    "SBIN": {"ticker": "SBIN.NS", "step": 10},
}


# ------------------------------------------------------------------ market clock
def market_state(now=None):
    """'open', 'square_off' (15:15-15:30) or 'closed'. Holidays are not checked."""
    now = now or dt.datetime.now(IST)
    mins = now.hour * 60 + now.minute
    if now.weekday() >= 5 or mins < OPEN_MIN or mins >= CLOSE_MIN:
        return "closed"
    return "square_off" if mins >= SQUARE_OFF_MIN else "open"


# ------------------------------------------------------------------ data
@st.cache_data(ttl=20, show_spinner=False)
def load_data(ticker, interval):
    df = yf.download(ticker, period="1d", interval=interval, progress=False, auto_adjust=True)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    if not df.empty:
        idx = pd.to_datetime(df.index)
        df.index = idx.tz_localize("UTC").tz_convert(IST) if idx.tz is None else idx.tz_convert(IST)
    return df


def compute_indicators(df):
    df = df.copy()
    tp = (df["High"] + df["Low"] + df["Close"]) / 3
    has_volume = float(df["Volume"].sum()) > 0
    if has_volume:
        vol = df["Volume"].replace(0, 1)
        df["VWAP"] = (tp * vol).cumsum() / vol.cumsum()
    else:
        # Yahoo gives no volume for indices, so a true VWAP is impossible.
        df["VWAP"] = tp.expanding().mean()
    df.attrs["vwap_label"] = "VWAP" if has_volume else "Session avg"

    df["EMA9"] = df["Close"].ewm(span=9, adjust=False).mean()
    df["EMA21"] = df["Close"].ewm(span=21, adjust=False).mean()

    delta = df["Close"].diff()                      # Wilder's RSI
    gain = delta.clip(lower=0).ewm(alpha=1 / 14, adjust=False).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1 / 14, adjust=False).mean()
    rs = gain / loss.replace(0, np.nan)
    df["RSI"] = 100 - (100 / (1 + rs))
    df.loc[loss == 0, "RSI"] = 100.0

    high_low = df["High"] - df["Low"]
    high_close = (df["High"] - df["Close"].shift()).abs()
    low_close = (df["Low"] - df["Close"].shift()).abs()
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    df["ATR"] = tr.ewm(alpha=1 / 14, adjust=False).mean()
    return df


# ------------------------------------------------------------------ signal
def generate_trade_signal(df, step_size, state, spot_now):
    """df must hold only COMPLETED candles. Levels are on the underlying, not the option premium."""
    latest, prev = df.iloc[-1], df.iloc[-2]
    spot = float(spot_now)
    vwap, ema9, ema21 = float(latest["VWAP"]), float(latest["EMA9"]), float(latest["EMA21"])
    rsi = float(latest["RSI"])
    atr = float(latest["ATR"]) if not np.isnan(latest["ATR"]) else spot * 0.002
    atm = int(round(spot / step_size) * step_size)
    base = {"sl": None, "t1": None, "t2": None, "spot": spot}

    if state == "square_off":
        return {**base, "action": "🔴 PULL OUT / SQUARE OFF", "option": "NONE",
                "reason": "Past 3:15 PM IST: intraday square-off window.",
                "pull_out": "Close all intraday F&O positions now."}

    if spot > vwap and ema9 > ema21 and 55 <= rsi <= 72:
        return {**base, "action": "🟢 BUY CALL (CE)", "option": f"{atm} CE",
                "reason": "Above VWAP/avg, 9 EMA over 21 EMA, RSI strong",
                "sl": round(spot - 1.5 * atr, 2), "t1": round(spot + 2.0 * atr, 2), "t2": round(spot + 3.5 * atr, 2),
                "pull_out": f"Exit if a candle closes below {ema21:,.2f} (21 EMA) or RSI drops under 50."}

    if spot < vwap and ema9 < ema21 and 28 <= rsi <= 45:
        return {**base, "action": "🟢 BUY PUT (PE)", "option": f"{atm} PE",
                "reason": "Below VWAP/avg, 9 EMA under 21 EMA, RSI weak",
                "sl": round(spot + 1.5 * atr, 2), "t1": round(spot - 2.0 * atr, 2), "t2": round(spot - 3.5 * atr, 2),
                "pull_out": f"Exit if a candle closes above {ema21:,.2f} (21 EMA) or RSI rises over 50."}

    note = "Hold your own rules or wait for a breakout."
    if rsi > 75:
        note = "🚨 RSI overbought: consider exiting open CALLS."
    elif rsi < 25:
        note = "🚨 RSI oversold: consider exiting open PUTS."
    elif prev["EMA9"] > prev["EMA21"] and ema9 < ema21:
        note = "🚨 Bearish EMA cross: consider exiting open CALLS."
    elif prev["EMA9"] < prev["EMA21"] and ema9 > ema21:
        note = "🚨 Bullish EMA cross: consider exiting open PUTS."
    return {**base, "action": "⚪ WAIT / NO TRADE", "option": f"Watch {atm}",
            "reason": "Mixed signals or sideways. Option premium decays with time.", "pull_out": note}


# ------------------------------------------------------------------ UI
st.markdown("### 📱 NSE F&O Live Signal")

with st.expander("⚙️ Instrument & Auto-Refresh Settings", expanded=True):
    c_inst, c_tf = st.columns(2)
    selected_name = c_inst.selectbox("Instrument", list(INSTRUMENTS.keys()))
    interval = c_tf.selectbox("Timeframe", ["1m", "5m", "15m"], index=1)
    auto_refresh = st.toggle("Auto-refresh every 30 seconds", value=True)

if st.button("🔄 Refresh Live Price Now", use_container_width=True):
    load_data.clear()

meta = INSTRUMENTS[selected_name]


# Only this block re-runs every 30s, so the settings above never reset or freeze the page.
@st.fragment(run_every=30 if auto_refresh else None)
def live_view():
    try:
        raw = load_data(meta["ticker"], interval)
    except Exception as e:  # network, rate limit
        st.error(f"Could not fetch data from Yahoo Finance: {e}")
        return
    if raw.empty:
        st.error("No data. The market may be closed, or Yahoo is rate-limiting this server.")
        return

    state = market_state()
    df = compute_indicators(raw)
    label = df.attrs.get("vwap_label", "VWAP")
    # The last candle is still forming while the market is open; signal on closed candles only.
    closed = df.iloc[:-1] if state != "closed" else df
    last_time = df.index[-1].strftime("%d %b %H:%M IST")

    if state == "closed":
        st.caption(f"Market closed. Showing the last session, last candle {last_time}.")
    else:
        st.caption(f"Last candle {last_time}. Data can lag; Yahoo is unofficial.")

    if len(closed) < MIN_CANDLES:
        st.metric("Spot Price", f"₹{float(df['Close'].iloc[-1]):,.2f}")
        st.info(f"Waiting for history: {len(closed)} of {MIN_CANDLES} candles. "
                "Signals need the 21 EMA and RSI to settle.")
    else:
        sig = generate_trade_signal(closed, meta["step"], state, float(df["Close"].iloc[-1]))
        r1c1, r1c2 = st.columns(2)
        r1c1.metric("Spot Price", f"₹{sig['spot']:,.2f}")
        r1c2.metric("Strike (ATM)", sig["option"])
        st.info(f"**Action:** {sig['action']}\n\n**Trigger:** {sig['reason']}")
        if sig["sl"]:
            s1, s2, s3 = st.columns(3)
            s1.error(f"🛑 **SL**\n₹{sig['sl']:,.1f}")
            s2.success(f"🎯 **T1**\n₹{sig['t1']:,.1f}")
            s3.success(f"🚀 **T2**\n₹{sig['t2']:,.1f}")
            st.caption("SL and targets are levels on the underlying price, not your option's premium.")
        st.warning(f"⚠️ **When to pull out:** {sig['pull_out']}")

    tail = df.tail(40)
    fig = go.Figure()
    fig.add_trace(go.Candlestick(x=tail.index, open=tail["Open"], high=tail["High"],
                                 low=tail["Low"], close=tail["Close"], name="Price"))
    fig.add_trace(go.Scatter(x=tail.index, y=tail["VWAP"], line=dict(color="orange", width=2), name=label))
    fig.add_trace(go.Scatter(x=tail.index, y=tail["EMA9"], line=dict(color="cyan", width=1.5), name="9 EMA"))
    fig.add_trace(go.Scatter(x=tail.index, y=tail["EMA21"], line=dict(color="magenta", width=1.5), name="21 EMA"))
    fig.update_layout(height=340, margin=dict(l=5, r=5, t=20, b=5), xaxis_rangeslider_visible=False,
                      template="plotly_dark",
                      legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
    st.plotly_chart(fig, use_container_width=True)
    if label != "VWAP":
        st.caption("Indices have no volume on Yahoo, so the orange line is the session's running average price, not a true VWAP.")


live_view()
st.caption("Indicator signals are an experiment, not advice. Most individual F&O traders lose money; options can expire worthless.")
