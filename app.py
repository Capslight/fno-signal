import datetime as dt
import math

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import pytz
import streamlit as st
import yfinance as yf

st.set_page_config(page_title="F&O Signal + Budget", page_icon="📈", layout="centered",
                   initial_sidebar_state="collapsed")

IST = pytz.timezone("Asia/Kolkata")
OPEN_MIN, SQUARE_OFF_MIN, CLOSE_MIN = 9 * 60 + 15, 15 * 60 + 15, 15 * 60 + 30
MIN_CANDLES = 25
DELTA = 0.5  # an at-the-money option moves roughly half as much as the underlying

# Lot sizes are set by NSE and change from time to time. These are only starting
# values; the app lets you type the lot size your broker shows.
INSTRUMENTS = {
    "NIFTY 50": {"ticker": "^NSEI", "step": 50, "lot": 65, "kind": "index"},
    "BANK NIFTY": {"ticker": "^NSEBANK", "step": 100, "lot": 30, "kind": "index"},
    "RELIANCE": {"ticker": "RELIANCE.NS", "step": 20, "lot": 500, "kind": "stock"},
    "HDFC BANK": {"ticker": "HDFCBANK.NS", "step": 10, "lot": 700, "kind": "stock"},
    "ICICI BANK": {"ticker": "ICICIBANK.NS", "step": 10, "lot": 700, "kind": "stock"},
    "SBIN": {"ticker": "SBIN.NS", "step": 10, "lot": 750, "kind": "stock"},
}
DEFAULT_IV = {"index": 14, "stock": 25}


def inr(x):
    """12345678 -> ₹1,23,45,678 (lakh/crore grouping)."""
    x = float(x)
    whole, frac = f"{abs(x):.2f}".split(".")
    if len(whole) > 3:
        head, tail, groups = whole[:-3], whole[-3:], []
        while len(head) > 2:
            groups.insert(0, head[-2:])
            head = head[:-2]
        if head:
            groups.insert(0, head)
        whole = ",".join(groups + [tail])
    return ("-" if x < 0 else "") + "₹" + whole + ("" if frac == "00" else "." + frac)


# ------------------------------------------------------------------ market clock
def market_state(now=None):
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


@st.cache_data(ttl=120, show_spinner=False)
def last_close(ticker):
    try:
        d = yf.download(ticker, period="5d", interval="1d", progress=False, auto_adjust=True)
        if isinstance(d.columns, pd.MultiIndex):
            d.columns = d.columns.get_level_values(0)
        return float(d["Close"].dropna().iloc[-1])
    except Exception:
        return None


def compute_indicators(df):
    df = df.copy()
    tp = (df["High"] + df["Low"] + df["Close"]) / 3
    has_volume = float(df["Volume"].sum()) > 0
    if has_volume:
        vol = df["Volume"].replace(0, 1)
        df["VWAP"] = (tp * vol).cumsum() / vol.cumsum()
    else:
        df["VWAP"] = tp.expanding().mean()
    df.attrs["vwap_label"] = "VWAP" if has_volume else "Session avg"
    df["EMA9"] = df["Close"].ewm(span=9, adjust=False).mean()
    df["EMA21"] = df["Close"].ewm(span=21, adjust=False).mean()
    delta = df["Close"].diff()
    gain = delta.clip(lower=0).ewm(alpha=1 / 14, adjust=False).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1 / 14, adjust=False).mean()
    rs = gain / loss.replace(0, np.nan)
    df["RSI"] = 100 - (100 / (1 + rs))
    df.loc[loss == 0, "RSI"] = 100.0
    high_low = df["High"] - df["Low"]
    high_close = (df["High"] - df["Close"].shift()).abs()
    low_close = (df["Low"] - df["Close"].shift()).abs()
    df["ATR"] = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1).ewm(alpha=1 / 14, adjust=False).mean()
    return df


# ------------------------------------------------------------------ signal
def generate_trade_signal(df, step_size, state, spot_now):
    """df holds only completed candles. SL/targets are levels on the underlying, not the premium."""
    latest, prev = df.iloc[-1], df.iloc[-2]
    spot = float(spot_now)
    vwap, ema9, ema21, rsi = (float(latest[c]) for c in ("VWAP", "EMA9", "EMA21", "RSI"))
    atr = float(latest["ATR"]) if not np.isnan(latest["ATR"]) else spot * 0.002
    atm = int(round(spot / step_size) * step_size)
    base = {"sl": None, "t1": None, "t2": None, "spot": spot, "side": None, "atm": atm}

    if state == "square_off":
        return {**base, "action": "SQUARE OFF", "tone": "stop", "option": "NONE",
                "reason": "It is past 3:15 PM IST, the intraday square-off window.",
                "pull_out": "Close all intraday F&O positions now."}
    if spot > vwap and ema9 > ema21 and 55 <= rsi <= 72:
        return {**base, "action": "BUY CALL (CE)", "tone": "buy", "side": "CE", "option": f"{atm} CE",
                "reason": "Price is above its average, the 9 EMA is over the 21 EMA, and momentum (RSI) is strong.",
                "sl": round(spot - 1.5 * atr, 2), "t1": round(spot + 2.0 * atr, 2), "t2": round(spot + 3.5 * atr, 2),
                "pull_out": f"Exit if a candle closes below {ema21:,.2f} (21 EMA) or RSI drops under 50."}
    if spot < vwap and ema9 < ema21 and 28 <= rsi <= 45:
        return {**base, "action": "BUY PUT (PE)", "tone": "buy", "side": "PE", "option": f"{atm} PE",
                "reason": "Price is below its average, the 9 EMA is under the 21 EMA, and momentum (RSI) is weak.",
                "sl": round(spot + 1.5 * atr, 2), "t1": round(spot - 2.0 * atr, 2), "t2": round(spot - 3.5 * atr, 2),
                "pull_out": f"Exit if a candle closes above {ema21:,.2f} (21 EMA) or RSI rises over 50."}

    note = "If you already hold a position, follow your own exit rules."
    if rsi > 75:
        note = "Momentum is stretched (RSI above 75). Consider exiting open CALLS."
    elif rsi < 25:
        note = "Momentum is stretched (RSI below 25). Consider exiting open PUTS."
    elif prev["EMA9"] > prev["EMA21"] and ema9 < ema21:
        note = "Bearish EMA cross just happened. Consider exiting open CALLS."
    elif prev["EMA9"] < prev["EMA21"] and ema9 > ema21:
        note = "Bullish EMA cross just happened. Consider exiting open PUTS."
    return {**base, "action": "WAIT, NO TRADE", "tone": "wait", "option": f"Watch {atm}",
            "reason": "The indicators disagree or the market is sideways. Option premiums lose value with time.",
            "pull_out": note}


# ------------------------------------------------------------------ budget maths
def estimate_premium(spot, iv_pct, days):
    """Rough at-the-money option price: 0.4 x spot x volatility x sqrt(time). Rounded to the Rs 0.05 tick."""
    p = 0.4 * spot * (iv_pct / 100.0) * math.sqrt(max(days, 0.5) / 365.0)
    return max(round(round(p / 0.05) * 0.05, 2), 0.05)


def plan_position(budget, risk_pct, premium, lot_size, spot, sl, t1):
    """How many lots, what they cost, and the rough loss/gain if the underlying hits SL/T1."""
    cost_lot = premium * lot_size
    afford = int(budget // cost_lot) if cost_lot > 0 else 0
    loss_lot = min(DELTA * abs(spot - sl) * lot_size, cost_lot)  # you cannot lose more than the premium paid
    gain_lot = DELTA * abs(t1 - spot) * lot_size
    allowed = budget * risk_pct / 100.0
    by_risk = int(allowed // loss_lot) if loss_lot > 0 else 0
    lots = min(afford, by_risk)
    return {"cost_lot": cost_lot, "afford": afford, "by_risk": by_risk, "lots": lots,
            "loss_lot": loss_lot, "gain_lot": gain_lot, "allowed_loss": allowed,
            "cost": lots * cost_lot, "max_loss": lots * loss_lot, "target_gain": lots * gain_lot}


# ------------------------------------------------------------------ UI: inputs
st.markdown("## 📱 NSE F&O Signal + Budget")
st.caption("Pick an instrument, enter your budget, and the app shows the signal and how many lots fit.")

c1, c2 = st.columns(2)
name = c1.selectbox("Instrument", list(INSTRUMENTS.keys()))
interval = c2.selectbox("Candle size", ["1m", "5m", "15m"], index=1)
meta = INSTRUMENTS[name]

st.markdown("#### 💰 Your budget")
b1, b2 = st.columns(2)
budget = b1.number_input("Money I can invest (₹)", min_value=1000, value=50000, step=5000)
risk_pct = b2.slider("Max loss I accept on one trade (% of budget)", 1, 10, 2,
                     help="The app sizes the trade so that, if the stop-loss level is hit, the estimated loss stays under this.")

with st.expander("⚙️ Advanced settings"):
    a1, a2 = st.columns(2)
    lot_size = a1.number_input("Lot size (check your broker)", min_value=1, value=meta["lot"], step=1, key=f"lot_{name}",
                               help="NSE changes lot sizes from time to time. Type the one your broker shows.")
    days = a2.number_input("Days to expiry", min_value=1, max_value=60, value=7, step=1)
    iv = st.slider("Expected volatility (IV, %)", 5, 80, DEFAULT_IV[meta["kind"]], key=f"iv_{name}",
                   help="Used only to estimate the option price. Higher means a pricier option.")
    own_premium = st.number_input("Option price you see in your broker app (₹, optional)", min_value=0.0, value=0.0, step=0.5,
                                  help="Leave 0 to use the app's rough estimate. Typing the real price makes the plan more accurate.")
    auto_refresh = st.toggle("Auto-refresh every 30 seconds", value=True)
    if st.button("🔄 Refresh now", use_container_width=True):
        load_data.clear()
        last_close.clear()


# ------------------------------------------------------------------ UI: live view
@st.fragment(run_every=30 if auto_refresh else None)
def live_view():
    try:
        raw = load_data(meta["ticker"], interval)
    except Exception as e:
        st.error(f"Could not fetch data from Yahoo Finance: {e}")
        return
    if raw.empty:
        st.error("No data. The market may be closed, or Yahoo is rate-limiting this server. Try again in a minute.")
        return

    state = market_state()
    df = compute_indicators(raw)
    label = df.attrs.get("vwap_label", "VWAP")
    closed = df.iloc[:-1] if state != "closed" else df
    last_time = df.index[-1].strftime("%d %b %H:%M IST")
    spot_now = float(df["Close"].iloc[-1])

    badge = {"open": "🟢 Market open", "square_off": "🟠 Square-off time", "closed": "⚪ Market closed"}[state]
    st.caption(f"{badge} · last candle {last_time} · data can lag (Yahoo is unofficial)")
    if state == "closed":
        st.info("The market is closed, so this shows the last session. Live signals run Monday to Friday, 9:15 to 15:30 IST.")

    if len(closed) < MIN_CANDLES:
        st.metric(f"{name} price", inr(spot_now))
        st.info(f"Waiting for history: {len(closed)} of {MIN_CANDLES} candles. Signals need the 21 EMA and RSI to settle, "
                "so they appear about two hours after the open on 5-minute candles.")
        sig = None
    else:
        sig = generate_trade_signal(closed, meta["step"], state, spot_now)
        m1, m2 = st.columns(2)
        m1.metric(f"{name} price", inr(sig["spot"]))
        m2.metric("Strike (nearest)", sig["option"])
        box = {"buy": st.success, "stop": st.error, "wait": st.info}[sig["tone"]]
        box(f"### {sig['action']}\n{sig['reason']}")

        if sig["sl"]:
            s1, s2, s3 = st.columns(3)
            s1.error(f"🛑 **Stop-loss**\n\n{inr(sig['sl'])}")
            s2.success(f"🎯 **Target 1**\n\n{inr(sig['t1'])}")
            s3.success(f"🚀 **Target 2**\n\n{inr(sig['t2'])}")
            st.caption(f"These are levels on the {name} price, not on the option's own price.")
        st.warning(f"⚠️ **When to get out:** {sig['pull_out']}")

    # ---------- budget plan
    st.markdown("### 🧮 What your budget buys")
    premium = float(own_premium) if own_premium > 0 else estimate_premium(spot_now, iv, days)
    source = "your price" if own_premium > 0 else "rough estimate"
    cost_one = premium * lot_size
    st.caption(f"Option price used: {inr(premium)} per share ({source}) × lot of {lot_size} = {inr(cost_one)} for one lot.")

    if sig and sig["sl"]:
        p = plan_position(budget, risk_pct, premium, lot_size, sig["spot"], sig["sl"], sig["t1"])
        if p["lots"] >= 1:
            st.success(
                f"**Plan: buy {p['lots']} lot{'s' if p['lots'] > 1 else ''} of {sig['option']}**\n\n"
                f"- Cost: about {inr(p['cost'])} ({p['cost'] / budget * 100:.0f}% of your budget)\n"
                f"- If the stop-loss level is hit: lose about {inr(p['max_loss'])} ({p['max_loss'] / budget * 100:.1f}% of budget)\n"
                f"- If Target 1 is hit: gain about {inr(p['target_gain'])}\n"
                f"- Keep the other {inr(budget - p['cost'])} aside.")
            st.caption("Loss and gain use a rule of thumb that an at-the-money option moves about half as much as the price. "
                       "Real results differ, and the option can lose its full cost.")
        elif p["afford"] < 1:
            st.error(f"Your budget does not cover one lot. One lot costs about {inr(cost_one)}; you have {inr(budget)}. "
                     "Raise the budget or check the table below for something cheaper.")
        else:
            st.warning(f"One lot could lose about {inr(p['loss_lot'])}, which is more than the {inr(p['allowed_loss'])} "
                       f"({risk_pct}% of budget) you chose. The app will not size a trade above your own limit.")
    elif sig and sig["tone"] == "wait":
        st.info("No trade signal right now, so no position is suggested. Keeping your money is a valid choice.")

    # ---------- chart
    tail = df.tail(40)
    fig = go.Figure()
    fig.add_trace(go.Candlestick(x=tail.index, open=tail["Open"], high=tail["High"], low=tail["Low"],
                                 close=tail["Close"], name="Price"))
    fig.add_trace(go.Scatter(x=tail.index, y=tail["VWAP"], line=dict(color="orange", width=2), name=label))
    fig.add_trace(go.Scatter(x=tail.index, y=tail["EMA9"], line=dict(color="cyan", width=1.5), name="9 EMA"))
    fig.add_trace(go.Scatter(x=tail.index, y=tail["EMA21"], line=dict(color="magenta", width=1.5), name="21 EMA"))
    fig.update_layout(height=340, margin=dict(l=5, r=5, t=20, b=5), xaxis_rangeslider_visible=False,
                      template="plotly_dark", legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
    st.plotly_chart(fig, use_container_width=True)
    if label != "VWAP":
        st.caption("Yahoo has no volume for indices, so the orange line is the session's running average price, not a true VWAP.")

    # ---------- what fits
    with st.expander("📋 Which instruments fit my budget?"):
        rows = []
        for n, m in INSTRUMENTS.items():
            s = last_close(m["ticker"])
            if not s:
                continue
            pr = estimate_premium(s, DEFAULT_IV[m["kind"]], days)
            c = pr * m["lot"]
            rows.append({"Instrument": n, "Price": inr(s), "1 lot costs (est.)": inr(c),
                         "Lots I can afford": int(budget // c), "Fits": "✅" if c <= budget else "❌"})
        if rows:
            st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)
            st.caption("Uses default lot sizes and typical volatility, so treat costs as ballpark. "
                       "Affording a lot does not mean it is a good trade.")
        else:
            st.write("Prices unavailable right now.")


live_view()

with st.expander("ℹ️ How to read this"):
    st.markdown(
        "- **Signal:** buy a call if the price looks strong, a put if it looks weak, otherwise wait.\n"
        "- **Stop-loss and targets:** price levels on the index or stock, worked out from recent price swings (ATR).\n"
        "- **Plan:** lots are limited by two things: what you can afford, and the loss you said you accept.\n"
        "- **Square-off:** intraday positions should be closed by 3:15 PM IST.")
st.caption("Indicator signals are an experiment, not advice. Most individual F&O traders lose money, and options can "
           "expire worthless. Do not trade money you cannot afford to lose.")
