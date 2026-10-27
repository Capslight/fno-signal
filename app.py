import copy
import datetime as dt
import json
import math
import urllib.request
import http.cookiejar

import numpy as np
import pandas as pd
import pytz
import streamlit as st
import streamlit.components.v1 as components
import yfinance as yf

st.set_page_config(page_title="NIFTY Paper Trader", page_icon="📈", layout="centered",
                   initial_sidebar_state="collapsed")

IST = pytz.timezone("Asia/Kolkata")
OPEN_T, SQUARE_T, CLOSE_T = dt.time(9, 15), dt.time(15, 15), dt.time(15, 30)
ENTRY_FROM, ENTRY_UNTIL = dt.time(9, 30), dt.time(14, 45)
TICKER, VIX_TICKER = "^NSEI", "^INDIAVIX"
STEP = 50                 # NIFTY strikes are 50 points apart
MAX_COST = 5000.0         # the most one lot may cost (your rule)
DEFAULT_LOT = 65          # NIFTY lot size since Jan 2026; NSE can change it, so it is editable in Settings
MIN_DELTA = 0.20          # skip strikes so far out of the money that they barely react
RATE, FALLBACK_IV = 0.065, 0.14
MIN_ADX, MAX_VIX, MIN_DAYS_TO_EXPIRY = 20, 25, 2
TICK = 0.05
DAYS = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"]

# Historical AI & Slippage parameters
MIN_HIST_WIN_RATE = 60.0  # Minimum % win rate of the K-nearest historical twin setups
KNN_K = 25                # Number of closest historical market setups to compare against
SLIPPAGE_SL = 0.60        # Adverse slippage (₹) when a stop-loss is triggered
SLIPPAGE_EXIT = 0.25      # Bid-ask spread cost (₹) on regular exits


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


def num(x, d=0):
    return f"{x:,.{d}f}"


def round_tick(p):
    return round(round(p / TICK) * TICK, 2)


# ------------------------------------------------------------------ market clock
def market_state(now=None):
    """'open', 'square_off' (15:15 to 15:30) or 'closed'. Holidays are not checked."""
    now = now or dt.datetime.now(IST)
    t = now.time()
    if now.weekday() >= 5 or t < OPEN_T or t >= CLOSE_T:
        return "closed"
    return "square_off" if t >= SQUARE_T else "open"


# ------------------------------------------------------------------ Path 1: Live NSE Option Chain Scraper (Zero-Config)
@st.cache_data(ttl=25, show_spinner=False)
def fetch_nse_chain():
    """Fetches real-time NIFTY Option Chain from NSE India using Python standard library (zero extra dependencies).
    Automatically falls back to None if NSE blocks cloud IPs or times out."""
    try:
        cj = http.cookiejar.CookieJar()
        opener = urllib.request.build_opener(urllib.request.HTTPCookieProcessor(cj))
        headers = [
            ("User-Agent", "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                           "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"),
            ("Accept", "application/json, text/plain, */*"),
            ("Accept-Language", "en-US,en;q=0.9"),
            ("Referer", "https://www.nseindia.com/option-chain"),
        ]
        opener.addheaders = headers
        # Step 1: Initialize NSE session cookies
        opener.open("https://www.nseindia.com/option-chain", timeout=4)
        # Step 2: Fetch NIFTY option chain JSON
        resp = opener.open("https://www.nseindia.com/api/option-chain-indices?symbol=NIFTY", timeout=5)
        raw = json.loads(resp.read().decode("utf-8"))
        records = raw.get("records", {})
        data = records.get("data", [])
        exp_dates = records.get("expiryDates", [])
        if not data or not exp_dates:
            return None

        tot_ce_oi = sum(row.get("CE", {}).get("openInterest", 0) for row in data if "CE" in row)
        tot_pe_oi = sum(row.get("PE", {}).get("openInterest", 0) for row in data if "PE" in row)
        pcr = round(tot_pe_oi / tot_ce_oi, 2) if tot_ce_oi > 0 else None

        by_expiry = {}
        for row in data:
            exp_str = row.get("expiryDate")
            K = row.get("strikePrice")
            if not exp_str or K is None:
                continue
            by_expiry.setdefault(exp_str, {})[int(K)] = {
                "CE": row.get("CE"),
                "PE": row.get("PE")
            }

        return {
            "spot": float(records.get("underlyingValue", 0.0)),
            "timestamp": records.get("timestamp", ""),
            "expiry_dates": exp_dates,
            "pcr": pcr,
            "tot_ce_oi": tot_ce_oi,
            "tot_pe_oi": tot_pe_oi,
            "by_expiry": by_expiry
        }
    except Exception:
        return None


# ---- LOGIC START (pure functions, no Streamlit; covered by tests)
# ------------------------------------------------------------------ option pricing with skew & NSE live chain
def _ncdf(x):
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def strike_iv(base_iv, spot, K, side):
    """Approximate NSE volatility skew: OTM puts carry heavier IV skew than OTM calls."""
    moneyness = (K - spot) / spot
    if side == "PE" and moneyness < 0:
        return base_iv * (1.0 + abs(moneyness) * 2.2)  # Put crash-fear skew
    if side == "CE" and moneyness > 0:
        return base_iv * (1.0 + abs(moneyness) * 1.1)  # Call wing skew
    return base_iv


def _d1d2(S, K, T, iv, r=RATE):
    T = max(T, 1e-6)
    d1 = (math.log(S / K) + (r + 0.5 * iv * iv) * T) / (iv * math.sqrt(T))
    return d1, d1 - iv * math.sqrt(T), T


def bs_price(S, K, T, iv, side, r=RATE):
    """Black-Scholes price of a NIFTY option (European, no dividend). An estimate, not a quote."""
    d1, d2, T = _d1d2(S, K, T, iv, r)
    if side == "CE":
        return S * _ncdf(d1) - K * math.exp(-r * T) * _ncdf(d2)
    return K * math.exp(-r * T) * _ncdf(-d2) - S * _ncdf(-d1)


def bs_delta(S, K, T, iv, side, r=RATE):
    d1, _, _ = _d1d2(S, K, T, iv, r)
    return _ncdf(d1) if side == "CE" else 1.0 - _ncdf(d1)   # size of the move, always positive


def years(expiry, ts):
    return max((expiry - ts).total_seconds(), 60.0) / (365.0 * 24 * 3600)


def next_expiry(now, weekday):
    """Nearest weekly expiry that is at least 2 days away, so the app never buys into expiry-day decay."""
    d = now.date()
    cand = d + dt.timedelta(days=(weekday - d.weekday()) % 7)
    if (cand - d).days < MIN_DAYS_TO_EXPIRY:
        cand += dt.timedelta(days=7)
    return IST.localize(dt.datetime.combine(cand, CLOSE_T))


def _match_nse_expiry(nse_chain, now):
    """Pick the nearest real NSE expiry date that is at least MIN_DAYS_TO_EXPIRY away."""
    if not nse_chain:
        return None, None
    d = now.date()
    for exp_str in nse_chain.get("expiry_dates", []):
        try:
            exp_d = dt.datetime.strptime(exp_str, "%d-%b-%Y").date()
            if (exp_d - d).days >= MIN_DAYS_TO_EXPIRY:
                return exp_str, IST.localize(dt.datetime.combine(exp_d, CLOSE_T))
        except Exception:
            continue
    return None, None


def pick_option(spot, side, now, iv, lot, weekday, max_cost=MAX_COST, nse_chain=None):
    """Closest-to-the-money strike whose one-lot cost fits the cap.
    Uses real NSE Option Chain LTP/Ask/IV if available; automatically falls back to skew-adjusted Black-Scholes."""
    exp_str, nse_exp_dt = _match_nse_expiry(nse_chain, now)
    expiry = nse_exp_dt if nse_exp_dt else next_expiry(now, weekday)
    T = years(expiry, now)
    atm = int(round(spot / STEP) * STEP)
    chain_strikes = nse_chain["by_expiry"].get(exp_str, {}) if (nse_chain and exp_str) else {}

    for k in range(0, 60):
        K = atm + k * STEP if side == "CE" else atm - k * STEP
        live_leg = chain_strikes.get(int(K), {}).get(side) if chain_strikes else None

        if live_leg and float(live_leg.get("lastPrice", 0) or 0) > 0:
            ask = float(live_leg.get("askPrice", 0) or 0)
            ltp = float(live_leg.get("lastPrice", 0))
            prem = round_tick(ask if (ask > 0 and abs(ask - ltp) / ltp < 0.05) else ltp)
            live_iv = float(live_leg.get("impliedVolatility", 0) or 0) / 100.0
            iv_k = live_iv if 0.05 <= live_iv <= 0.80 else strike_iv(iv, spot, K, side)
            model = bs_price(spot, K, T, iv_k, side)
            quote_source = "NSE Live Quote"
            oi = int(live_leg.get("openInterest", 0) or 0)
            chg_oi = int(live_leg.get("changeinOpenInterest", 0) or 0)
            spread = round(float(live_leg.get("askPrice", 0) - live_leg.get("bidprice", 0)), 2) if live_leg.get("bidprice") else None
        else:
            iv_k = strike_iv(iv, spot, K, side)
            model = bs_price(spot, K, T, iv_k, side)
            prem = round_tick(model)
            quote_source = "BS Skew Model"
            oi, chg_oi, spread = None, None, None

        if prem * lot <= max_cost:
            delta = bs_delta(spot, K, T, iv_k, side)
            if prem < 0.5 or delta < MIN_DELTA:
                return None       # anything cheaper is even less responsive
            return {
                "strike": int(K), "side": side, "expiry": expiry.isoformat(), "exp_str": exp_str,
                "iv": iv_k, "prem": prem, "model": model, "cost": prem * lot, "delta": delta,
                "steps_otm": k, "quote_source": quote_source, "oi": oi, "chg_oi": chg_oi, "spread": spread,
                "label": f"NIFTY {expiry:%d %b} {int(K)} {side}"
            }
    return None


def live_option_ltp(pos, nse_chain):
    """Look up the live NSE market bid/lastPrice for an open paper position if available."""
    if not nse_chain:
        return None
    exp_dt = dt.datetime.fromisoformat(pos["expiry"])
    exp_str = pos.get("exp_str") or exp_dt.strftime("%d-%b-%Y")
    strike_map = nse_chain.get("by_expiry", {}).get(exp_str, {})
    leg = strike_map.get(int(pos["strike"]), {}).get(pos["side"])
    if leg and float(leg.get("lastPrice", 0) or 0) > 0:
        bid = float(leg.get("bidprice", 0) or 0)
        ltp = float(leg.get("lastPrice", 0))
        return round_tick(bid if (bid > 0 and abs(ltp - bid) / ltp < 0.05) else ltp)
    return None


def prem_at(pos, spot, ts, slippage=0.0, nse_chain=None, is_now=False):
    """Estimated option price if NIFTY is at `spot` at time `ts`.
    If checking current live value (`is_now=True`) and NSE live chain is connected, uses real NSE bid/LTP directly."""
    if is_now and nse_chain:
        live_p = live_option_ltp(pos, nse_chain)
        if live_p is not None:
            return max(live_p, 0.05)
    T = years(dt.datetime.fromisoformat(pos["expiry"]), ts)
    p = pos["entry_prem"] + bs_price(spot, pos["strike"], T, pos["iv"], pos["side"]) - pos["model_entry"] - slippage
    return max(round_tick(p), 0.05)


def build_levels(opt, spot, sl, t1, t2, entry_prem=None):
    """Option price at the stop-loss and targets, including expected holding-time theta decay."""
    exp_dt = dt.datetime.fromisoformat(opt["expiry"])
    now_dt = dt.datetime.fromisoformat(opt["now"])
    entry = float(entry_prem) if entry_prem else opt["prem"]
    off = entry - opt["model"]

    def at(level, est_mins=0, slip=0.0):
        T_proj = years(exp_dt, min(now_dt + dt.timedelta(minutes=est_mins), exp_dt))
        return max(round_tick(bs_price(level, opt["strike"], T_proj, opt["iv"], opt["side"]) + off - slip), 0.05)

    return {
        "entry": entry,
        "p_sl": at(sl, est_mins=20, slip=SLIPPAGE_SL),
        "p_t1": at(t1, est_mins=45, slip=SLIPPAGE_EXIT),
        "p_t2": at(t2, est_mins=90, slip=SLIPPAGE_EXIT)
    }


# ------------------------------------------------------------------ indicators
def add_indicators(df):
    df = df.copy()
    h, l, c = df["High"], df["Low"], df["Close"]
    tp = (h + l + c) / 3
    # Range-weighted intraday VWAP proxy (since Yahoo has no volume for ^NSEI, high-activity candles get higher weight)
    weight = (h - l).clip(lower=0.5)
    df["DAYAVG"] = (
        (tp * weight).groupby(df.index.date).cumsum() /
        weight.groupby(df.index.date).cumsum()
    )
    df["EMA9"] = c.ewm(span=9, adjust=False).mean()
    df["EMA21"] = c.ewm(span=21, adjust=False).mean()
    d = c.diff()
    gain = d.clip(lower=0).ewm(alpha=1 / 14, adjust=False).mean()
    loss = (-d.clip(upper=0)).ewm(alpha=1 / 14, adjust=False).mean()
    rs = gain / loss.replace(0, np.nan)
    df["RSI"] = 100 - 100 / (1 + rs)
    df.loc[loss == 0, "RSI"] = 100.0
    tr = pd.concat([h - l, (h - c.shift()).abs(), (l - c.shift()).abs()], axis=1).max(axis=1)
    atr = tr.ewm(alpha=1 / 14, adjust=False).mean()
    df["ATR"] = atr
    up, down = h.diff(), -l.diff()
    pdm = pd.Series(np.where((up > down) & (up > 0), up, 0.0), index=df.index)
    mdm = pd.Series(np.where((down > up) & (down > 0), down, 0.0), index=df.index)
    pdi = 100 * pdm.ewm(alpha=1 / 14, adjust=False).mean() / atr
    mdi = 100 * mdm.ewm(alpha=1 / 14, adjust=False).mean() / atr
    dx = 100 * (pdi - mdi).abs() / (pdi + mdi).replace(0, np.nan)
    df["ADX"] = dx.ewm(alpha=1 / 14, adjust=False).mean()
    return df


# ------------------------------------------------------------------ Part 2: Historical Learning Engine
def _feature_matrix(d5, d15):
    """Build normalized, scale-free feature matrix for every 5-minute bar."""
    atr = d5["ATR"].replace(0, np.nan).bfill().fillna(d5["Close"] * 0.0015)
    rng = (d5["High"] - d5["Low"]).replace(0, np.nan)

    if d15 is not None and not d15.empty:
        d15_shifted = d15[["EMA9", "EMA21"]].shift(1).reindex(d5.index, method="ffill")
        ema15_spread = (d15_shifted["EMA9"] - d15_shifted["EMA21"]) / atr
    else:
        ema15_spread = pd.Series(0.0, index=d5.index)

    feats = pd.DataFrame({
        "ema_spread": (d5["EMA9"] - d5["EMA21"]) / atr,
        "vwap_dist":  (d5["Close"] - d5["DAYAVG"]) / atr,
        "rsi_norm":   (d5["RSI"] - 50.0) / 50.0,
        "adx_norm":   d5["ADX"].fillna(0) / 50.0,
        "body_ratio": ((d5["Close"] - d5["Open"]) / rng).fillna(0.0),
        "htf_spread": ema15_spread.fillna(0.0),
        "hour_norm":  (d5.index.hour + d5.index.minute / 60.0 - 9.25) / 6.25,
    }, index=d5.index)
    return feats.replace([np.inf, -np.inf], 0.0).fillna(0.0)


def _simulate_outcome(df_day, entry_idx, side, spot, atr):
    """Forward-test a single historical signal on the rest of that day's 5m candles."""
    sgn = 1 if side == "CE" else -1
    sl = spot - sgn * 1.5 * atr
    t1 = spot + sgn * 2.0 * atr
    t2 = spot + sgn * 3.5 * atr

    future = df_day.iloc[entry_idx + 1:]
    hit_t1 = False
    stop_u = sl

    for ts, row in future.iterrows():
        if ts.time() >= SQUARE_T:
            r_mult = sgn * (float(row["Close"]) - spot) / (1.5 * atr)
            return (r_mult > 0), r_mult

        lo, hi = float(row["Low"]), float(row["High"])
        if (side == "CE" and lo <= stop_u) or (side == "PE" and hi >= stop_u):
            return (True, 0.25) if hit_t1 else (False, -1.0)

        if not hit_t1 and ((side == "CE" and hi >= t1) or (side == "PE" and lo <= t1)):
            hit_t1 = True
            stop_u = spot

        if hit_t1 and ((side == "CE" and hi >= t2) or (side == "PE" and lo <= t2)):
            return True, 3.5 / 1.5

    return hit_t1, (0.25 if hit_t1 else 0.0)


@st.cache_data(ttl=300, show_spinner=False)
def learn_from_history(d5, d15, side, now_iso):
    """Scan up to 60 days of past 5-minute data, simulate every matching signal,
    and score today's live candle against its closest historical twins."""
    now = dt.datetime.fromisoformat(now_iso)
    forming = (d5.index[-1] + pd.Timedelta(minutes=5)) > now
    closed = d5.iloc[:-1] if forming else d5
    if len(closed) < 100 or side is None:
        return None

    feats = _feature_matrix(closed, d15)
    x_now = feats.iloc[-1].values
    today = closed.index[-1].date()

    hist = closed[closed.index.date < today]
    if hist.empty:
        return None

    records = []
    for day, df_day in hist.groupby(hist.index.date):
        for i in range(len(df_day) - 3):
            ts = df_day.index[i]
            if not (ENTRY_FROM <= ts.time() <= ENTRY_UNTIL):
                continue
            r = df_day.iloc[i]
            if side == "CE" and not (r["Close"] > r["DAYAVG"] and r["EMA9"] > r["EMA21"] and r["ADX"] >= MIN_ADX):
                continue
            if side == "PE" and not (r["Close"] < r["DAYAVG"] and r["EMA9"] < r["EMA21"] and r["ADX"] >= MIN_ADX):
                continue

            won, r_mult = _simulate_outcome(df_day, i, side, float(r["Close"]), float(r["ATR"]))
            f_vec = feats.loc[ts].values
            dist = float(np.linalg.norm(x_now - f_vec))
            records.append({"ts": ts, "won": won, "r_mult": r_mult, "dist": dist})

    if len(records) < 10:
        return None

    res_df = pd.DataFrame(records).sort_values("dist")
    overall_wr = float(res_df["won"].mean() * 100.0)

    knn = res_df.head(min(KNN_K, len(res_df))).copy()
    weights = np.exp(-0.5 * (knn["dist"].values ** 2))
    knn_wr = float(np.average(knn["won"].values, weights=weights) * 100.0)
    expectancy_r = float(np.average(knn["r_mult"].values, weights=weights))

    return {
        "samples": len(res_df),
        "overall_wr": overall_wr,
        "knn_wr": knn_wr,
        "expectancy_r": expectancy_r,
        "passed": knn_wr >= MIN_HIST_WIN_RATE and expectancy_r > 0.15
    }


# ------------------------------------------------------------------ the analysis
def analyse(d5, d15, vix, now, lot, weekday, nse_chain=None):
    """Judge NIFTY from completed 5m/15m candles, 60-day historical KNN memory, and live NSE Option Chain OI/PCR."""
    out = {"decision": "WAIT", "side": None, "checks": [], "reason": "", "opt": None, "plan": None, "spot": None,
           "core_pass": 0, "core_total": 0, "bonus_pass": 0, "bonus_total": 0, "bonus_need": 0}
    if d5 is None or d5.empty:
        out["reason"] = "No price data."
        return out
    span = pd.Timedelta(minutes=5)
    last_ts = d5.index[-1]
    forming = (last_ts + span) > now
    closed = d5.iloc[:-1] if forming else d5
    spot = float(d5["Close"].iloc[-1])
    out["spot"] = spot
    if len(closed) < 3:
        out["reason"] = "Not enough candles yet."
        return out
    c = closed.iloc[-1]
    today = d5.index[-1].date()

    def add(label, ok, detail, core=True):
        out["checks"].append({"label": label, "ok": ok, "detail": detail, "core": core})

    # direction from the 5-minute picture
    up = c["Close"] > c["DAYAVG"] and c["EMA9"] > c["EMA21"]
    down_ = c["Close"] < c["DAYAVG"] and c["EMA9"] < c["EMA21"]
    side = "CE" if up else "PE" if down_ else None
    out["side"] = side
    lean = {"CE": "up", "PE": "down", None: "unclear"}[side]
    add("5-min direction clear", side is not None,
        f"close {num(c['Close'], 1)}, range-VWAP {num(c['DAYAVG'], 1)}, 9 EMA {num(c['EMA9'], 1)}, 21 EMA {num(c['EMA21'], 1)}"
        + ("" if side else " (they disagree)"))
    sgn = 1 if side == "CE" else -1

    in_window = now.weekday() < 5 and ENTRY_FROM <= now.time() <= ENTRY_UNTIL
    add("Entry window (9:30 to 2:45 PM IST)", in_window, now.strftime("%H:%M IST") +
        ("" if in_window else ": the first 15 minutes are too wild and after 2:45 there is too little time"))
    lag = (now - last_ts).total_seconds() / 60
    fresh = lag <= 5 + 10 and last_ts.date() == now.date()
    add("Data is fresh", fresh, f"last candle {last_ts:%H:%M}, {lag:.0f} min ago" + ("" if fresh else " (Yahoo looks delayed)"))

    # Fixed 15-min lookahead bias: only judge completed 15-minute candles
    if d15 is not None and len(d15) >= 25:
        forming15 = (d15.index[-1] + pd.Timedelta(minutes=15)) > now
        closed15 = d15.iloc[:-1] if forming15 else d15
        if len(closed15) > 0:
            h = closed15.iloc[-1]
            ok15 = (side is not None) and ((h["EMA9"] > h["EMA21"]) if side == "CE" else (h["EMA9"] < h["EMA21"]))
            add("15-min trend agrees (completed candle)", bool(ok15),
                f"15-min 9 EMA {num(h['EMA9'], 1)} vs 21 EMA {num(h['EMA21'], 1)}")
        else:
            add("15-min trend agrees", False, "no completed 15-minute candle yet")
    else:
        add("15-min trend agrees", False, "15-minute data unavailable")

    if side:
        rsi = float(c["RSI"])
        ok_rsi = (55 <= rsi <= 72) if side == "CE" else (28 <= rsi <= 45)
        add("Momentum (RSI) in range", ok_rsi, f"RSI {rsi:.0f}; needs " + ("55 to 72" if side == "CE" else "28 to 45"))
    else:
        add("Momentum (RSI) in range", False, "no direction yet")
    adx = float(c["ADX"]) if not np.isnan(c["ADX"]) else 0.0
    add("Trending, not sideways (ADX)", adx >= MIN_ADX, f"ADX {adx:.0f}; needs at least {MIN_ADX}")
    if vix is None:
        add("India VIX not extreme", True, "VIX unavailable; assumed normal (14)")
    else:
        add("India VIX not extreme", vix <= MAX_VIX, f"VIX {vix:.1f}; options get very expensive above {MAX_VIX}")
    iv = min(max((vix / 100.0) if vix else FALLBACK_IV, 0.08), 0.40)

    opt = pick_option(spot, side, now, iv, lot, weekday, nse_chain=nse_chain) if side else None
    if side:
        add("A strike under ₹5,000 per lot exists", opt is not None,
            (f"{opt['label']} costs {inr(opt['cost'])} via {opt['quote_source']} "
             f"(delta {opt['delta']:.2f}, IV {opt['iv']*100:.1f}%)" if opt
             else f"nothing under {inr(MAX_COST)} with enough response to NIFTY moves"))
    else:
        add("A strike under ₹5,000 per lot exists", False, "no direction yet")

    # Part 2: Historical KNN Win Probability Check
    hist_ml = learn_from_history(d5, d15, side, now.isoformat()) if side else None
    if side and hist_ml:
        add(f"Historical AI Win Rate (≥{MIN_HIST_WIN_RATE:.0f}%)", hist_ml["passed"],
            f"Top {min(KNN_K, hist_ml['samples'])} twin setups (of {hist_ml['samples']} in 60d): "
            f"{hist_ml['knn_wr']:.1f}% win rate, {hist_ml['expectancy_r']:+.2f}R expectancy "
            f"(base rate {hist_ml['overall_wr']:.1f}%)", core=True)
    elif side:
        add(f"Historical AI Win Rate (≥{MIN_HIST_WIN_RATE:.0f}%)", False,
            "Not enough past signals in 60-day history to verify edge", core=True)
    else:
        add(f"Historical AI Win Rate (≥{MIN_HIST_WIN_RATE:.0f}%)", False, "no direction yet", core=True)

    atr = float(c["ATR"]) if not np.isnan(c["ATR"]) else spot * 0.0015
    # bonus checks
    today_rows = d5[d5.index.date == today]
    orw = today_rows[today_rows.index.time < ENTRY_FROM]
    if side and len(orw) >= 3:
        orh, orl = float(orw["High"].max()), float(orw["Low"].min())
        ok = spot > orh if side == "CE" else spot < orl
        add("Breaks the opening range", bool(ok), f"opening range {num(orl, 1)} to {num(orh, 1)}", core=False)
    else:
        add("Breaks the opening range", None, "not measurable yet", core=False)
    if side:
        body, rng = abs(float(c["Close"] - c["Open"])), float(c["High"] - c["Low"])
        dir_ok = (c["Close"] > c["Open"]) if side == "CE" else (c["Close"] < c["Open"])
        ok = bool(dir_ok and rng > 0 and body / rng >= 0.5)
        add("Last candle confirms", ok,
            f"closed {'up' if c['Close'] > c['Open'] else 'down'}, body {body / rng * 100 if rng else 0:.0f}% of its range", core=False)
    else:
        add("Last candle confirms", None, "no direction yet", core=False)
    prior = d5[d5.index.date < today]
    if side and len(prior):
        pday = prior[prior.index.date == prior.index.date.max()]
        ph, pl = float(pday["High"].max()), float(pday["Low"].min())
        if side == "CE":
            ok = spot > ph or (ph - spot) >= 2 * atr
            det = f"yesterday's high {num(ph, 1)}"
        else:
            ok = spot < pl or (spot - pl) >= 2 * atr
            det = f"yesterday's low {num(pl, 1)}"
        add("Room before yesterday's high/low", bool(ok), det + ("" if ok else " is within reach of Target 1"), core=False)
    else:
        add("Room before yesterday's high/low", None, "no earlier day in the data", core=False)

    # Path 1 Bonus Check: Live NSE Open Interest / Put-Call Ratio (PCR) Alignment
    if side and nse_chain and nse_chain.get("pcr") is not None:
        pcr = nse_chain["pcr"]
        ok_pcr = (pcr >= 0.95) if side == "CE" else (pcr <= 1.05)
        add("NSE Option Chain PCR supports direction", bool(ok_pcr),
            f"Live NIFTY PCR is {pcr:.2f} (Put OI {num(nse_chain['tot_pe_oi'])} vs Call OI {num(nse_chain['tot_ce_oi'])})",
            core=False)
    else:
        add("NSE Option Chain PCR supports direction", None,
            "NSE Option Chain feed not reachable from this IP (using fallback)", core=False)

    core = [k for k in out["checks"] if k["core"]]
    bonus = [k for k in out["checks"] if not k["core"] and k["ok"] is not None]
    bonus_pass = sum(1 for k in bonus if k["ok"])
    need_bonus = min(2, len(bonus))
    out["core_pass"], out["core_total"] = sum(1 for k in core if k["ok"]), len(core)
    out["bonus_pass"], out["bonus_total"], out["bonus_need"] = bonus_pass, len(bonus), need_bonus
    if side and all(k["ok"] for k in core) and bonus_pass >= need_bonus and opt:
        sl, t1, t2 = (round(spot - sgn * 1.5 * atr, 2), round(spot + sgn * 2.0 * atr, 2), round(spot + sgn * 3.5 * atr, 2))
        opt = {**opt, "now": now.isoformat()}
        out.update(decision="BUY CALL" if side == "CE" else "BUY PUT", opt=opt,
                   plan={"sl": sl, "t1": t1, "t2": t2, "atr": atr, "lot": lot, "levels": build_levels(opt, spot, sl, t1, t2),
                         "asof": last_ts.isoformat()})
        out["reason"] = "All core + historical AI checks passed" + (f" and {bonus_pass} of {len(bonus)} bonus checks." if bonus else ".")
    else:
        miss = [k["label"] for k in core if not k["ok"]]
        if bonus_pass < need_bonus and not miss:
            miss.append(f"at least {need_bonus} of the {len(bonus)} extra confirmations (only {bonus_pass})")
        out["reason"] = ("NIFTY is leaning " + lean + ". Waiting for: " if side else "Direction unclear. Waiting for: ") + "; ".join(miss) + "."
    return out


# ------------------------------------------------------------------ paper account
INTERVAL_MIN = 5


def new_account(charges=100.0):
    return {"charges": float(charges), "positions": [], "closed": [], "next_id": 1, "use_pullout": True,
            "t1_mode": "trail", "created": dt.datetime.now(IST).isoformat()}


def open_position(acct, res, entry_prem, source, now):
    if acct["positions"]:
        return False, "You already have an open paper trade. Finish it before starting another."
    o, p = res["opt"], res["plan"]
    lv = build_levels(o, res["spot"], p["sl"], p["t1"], p["t2"], entry_prem)
    pos = {"id": acct["next_id"], "label": o["label"], "side": o["side"], "strike": o["strike"],
           "expiry": o["expiry"], "exp_str": o.get("exp_str"), "iv": o["iv"], "lot": int(p["lot"]),
           "lots_open": 1, "entry_time": now.isoformat(), "entry_spot": float(res["spot"]),
           "entry_prem": float(entry_prem), "model_entry": float(o["model"]), "source": source,
           "sl": p["sl"], "t1": p["t1"], "t2": p["t2"], "p_sl": lv["p_sl"], "p_t1": lv["p_t1"], "p_t2": lv["p_t2"],
           "stop_u": p["sl"], "t1_done": False, "legs": [], "events": [], "interval_min": INTERVAL_MIN}
    acct["next_id"] += 1
    acct["positions"].append(pos)
    return True, f"Paper trade started: 1 lot of {o['label']} at {inr(entry_prem)} (cost {inr(entry_prem * p['lot'])})."


def _stop_hit(pos, low, high):
    return low <= pos["stop_u"] if pos["side"] == "CE" else high >= pos["stop_u"]


def _target_hit(pos, level, low, high):
    return high >= level if pos["side"] == "CE" else low <= level


def evaluate_position(pos, df5, now, use_pullout=True, t1_mode="trail", df1=None, nse_chain=None):
    """Replay the app's own rules on the candles since entry.
    Uses 1-minute candles (`df1`) when available for 5x sharper stop/target hit detection,
    plus 5-minute completed candles (`df5`) for the 21 EMA / RSI indicator pullout rule."""
    p = copy.deepcopy(pos)
    entry_t = dt.datetime.fromisoformat(p["entry_time"])
    if df5 is None or df5.empty or df5.index[-1].date() != entry_t.date():
        return {"pos": p, "closed": False, "stale": True, "new_legs": [], "new_events": []}

    sq = IST.localize(dt.datetime.combine(entry_t.date(), SQUARE_T))
    n0, e0 = len(p["legs"]), len(p["events"])

    # Use 1-minute bars for high-resolution stop/target checks if available for today
    use_1m = df1 is not None and not df1.empty and df1.index[-1].date() == entry_t.date()
    bars = df1 if use_1m else df5
    bar_span = pd.Timedelta(minutes=1 if use_1m else 5)
    mid_span = pd.Timedelta(seconds=30 if use_1m else 150)

    # Pre-index completed 5m bars for the 21 EMA / RSI pullout check
    closed_5m = df5[(df5.index > entry_t) & ((df5.index + pd.Timedelta(minutes=5)) <= min(now, sq))]

    def leg(ts, spot_level, reason, slip=SLIPPAGE_EXIT, is_now=False):
        p["legs"].append({"time": ts.isoformat(), "lots": p["lots_open"],
                          "prem": float(prem_at(p, spot_level, ts, slippage=slip, nse_chain=nse_chain, is_now=is_now)),
                          "reason": reason})
        p["lots_open"] = 0

    rows = bars[(bars.index > entry_t) & (bars.index < sq)]
    for ts, r in rows.iterrows():
        if p["lots_open"] <= 0:
            break
        low, high, close = float(r["Low"]), float(r["High"]), float(r["Close"])
        completed = (ts + bar_span) <= now
        hit_ts = min(ts + mid_span, now) if completed else now

        if _stop_hit(p, low, high):
            leg(hit_ts, p["stop_u"], "Stop-loss (1m precision)" if use_1m and not p["t1_done"] else
                ("Stop-loss" if not p["t1_done"] else "Stop moved to entry price after Target 1"),
                slip=SLIPPAGE_SL)
            break

        if not p["t1_done"] and _target_hit(p, p["t1"], low, high):
            if t1_mode == "exit":
                leg(hit_ts, p["t1"], "Target 1", slip=SLIPPAGE_EXIT)
                break
            p["t1_done"], p["stop_u"] = True, p["entry_spot"]
            p["events"].append({"time": hit_ts.isoformat(), "text": "Target 1 reached: stop-loss moved to your entry level"})

        if p["t1_done"] and _target_hit(p, p["t2"], low, high):
            leg(hit_ts, p["t2"], "Target 2", slip=SLIPPAGE_EXIT)
            break

        # Check 5m indicator pullout at the exact close of each 5-minute boundary
        if use_pullout and completed:
            bar_end = ts + bar_span
            matching_5m = closed_5m[(closed_5m.index + pd.Timedelta(minutes=5)) == bar_end]
            if len(matching_5m):
                r5 = matching_5m.iloc[-1]
                ema21, rsi = float(r5["EMA21"]), float(r5["RSI"])
                c5 = float(r5["Close"])
                if not (np.isnan(ema21) or np.isnan(rsi)):
                    bad = (c5 < ema21 or rsi < 50) if p["side"] == "CE" else (c5 > ema21 or rsi > 50)
                    if bad:
                        leg(bar_end, c5, "App exit rule (5m candle crossed 21 EMA or RSI flipped)", slip=SLIPPAGE_EXIT)
                        break

    # Live tick check against current spot & real NSE option price
    if p["lots_open"] > 0 and now < sq:
        spot = float(bars["Close"].iloc[-1])
        if _stop_hit(p, spot, spot):
            leg(now, p["stop_u"], "Stop-loss" if not p["t1_done"] else "Stop moved to entry price after Target 1",
                slip=SLIPPAGE_SL, is_now=True)
        elif not p["t1_done"] and _target_hit(p, p["t1"], spot, spot):
            if t1_mode == "exit":
                leg(now, p["t1"], "Target 1", slip=SLIPPAGE_EXIT, is_now=True)
            else:
                p["t1_done"], p["stop_u"] = True, p["entry_spot"]
                p["events"].append({"time": now.isoformat(), "text": "Target 1 reached: stop-loss moved to your entry level"})
        elif p["t1_done"] and _target_hit(p, p["t2"], spot, spot):
            leg(now, p["t2"], "Target 2", slip=SLIPPAGE_EXIT, is_now=True)

    if p["lots_open"] > 0 and now >= sq:
        sub = bars[bars.index <= sq]
        leg(sq, float((sub if len(sub) else bars)["Close"].iloc[-1]), "Square-off at 3:15 PM", slip=SLIPPAGE_EXIT)

    return {"pos": p, "closed": p["lots_open"] <= 0, "stale": False, "new_legs": p["legs"][n0:], "new_events": p["events"][e0:]}


def closed_record(pos, charges):
    gross = sum((g["prem"] - pos["entry_prem"]) * g["lots"] * pos["lot"] for g in pos["legs"])
    return {"id": pos["id"], "label": pos["label"], "side": pos["side"], "lot": pos["lot"], "entry_time": pos["entry_time"],
            "exit_time": pos["legs"][-1]["time"], "entry_prem": pos["entry_prem"], "exit_prem": pos["legs"][-1]["prem"],
            "gross": round(gross, 2), "charges": float(charges), "pnl": round(gross - charges, 2),
            "reason": " → ".join(dict.fromkeys(g["reason"] for g in pos["legs"])), "legs": pos["legs"]}


def settle(acct, pos_id, res):
    idx = next(i for i, x in enumerate(acct["positions"]) if x["id"] == pos_id)
    if res["closed"]:
        rec = closed_record(res["pos"], acct["charges"])
        acct["closed"].append(rec)
        acct["positions"].pop(idx)
        return rec
    acct["positions"][idx] = res["pos"]
    return None


def paper_stats(closed):
    pn = [t["pnl"] for t in closed]
    wins, losses = [x for x in pn if x > 0], [x for x in pn if x <= 0]
    gp, gl = sum(wins), -sum(losses)
    return {"n": len(pn), "total": sum(pn), "win_rate": (len(wins) / len(pn) * 100) if pn else 0.0,
            "avg_win": (gp / len(wins)) if wins else 0.0, "avg_loss": (sum(losses) / len(losses)) if losses else 0.0,
            "best": max(pn) if pn else 0.0, "worst": min(pn) if pn else 0.0, "pf": (gp / gl) if gl > 0 else None}
# ---- LOGIC END


# ------------------------------------------------------------------ data
def _clean(df):
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    if df.empty:
        return df
    idx = pd.to_datetime(df.index)
    df.index = idx.tz_localize("UTC").tz_convert(IST) if idx.tz is None else idx.tz_convert(IST)
    return df.dropna(subset=["Close"])


@st.cache_data(ttl=12, show_spinner=False)
def load_prices(ticker, interval, period):
    return _clean(yf.download(ticker, period=period, interval=interval, progress=False, auto_adjust=True))


@st.cache_data(ttl=120, show_spinner=False)
def load_vix():
    try:
        d = _clean(yf.download(VIX_TICKER, period="5d", interval="1d", progress=False, auto_adjust=True))
        return float(d["Close"].iloc[-1]) if len(d) else None
    except Exception:
        return None


# ------------------------------------------------------------------ Live Heartbeat Monitor
def render_heartbeat(now, run_count, auto_refresh, interval_sec=15):
    """Client-side JS live timer that detects if the browser tab went to sleep or missed a Streamlit cycle."""
    sync_str = now.strftime("%H:%M:%S IST")
    if not auto_refresh:
        st.caption(f"⏸️ **Auto-refresh paused** · Last manual sync: `{sync_str}` (Cycle #{run_count})")
        return

    warn_sec = interval_sec + 15
    html_code = f"""
    <div id="hb-box" style="font-family: -apple-system, BlinkMacSystemFont, sans-serif; font-size: 13px;
         padding: 8px 12px; border-radius: 8px; background: rgba(16, 185, 129, 0.12);
         border: 1px solid rgba(16, 185, 129, 0.35); color: #065f46; display: flex;
         justify-content: space-between; align-items: center;">
        <div>
            <span id="hb-dot" style="display:inline-block; width:9px; height:9px; border-radius:50%;
                  background:#10b981; margin-right:6px; box-shadow: 0 0 6px #10b981;"></span>
            <strong id="hb-status">Live Sync Active ({interval_sec}s)</strong> · Cycle <b>#{run_count}</b> at <code>{sync_str}</code>
        </div>
        <div id="hb-timer" style="font-variant-numeric: tabular-nums; font-weight: 600;">
            Next refresh in {interval_sec}s
        </div>
    </div>
    <script>
        const startMs = Date.now();
        const intervalSec = {interval_sec};
        const warnSec = {warn_sec};
        const box = document.getElementById("hb-box");
        const dot = document.getElementById("hb-dot");
        const status = document.getElementById("hb-status");
        const timer = document.getElementById("hb-timer");

        setInterval(() => {{
            const elapsed = Math.floor((Date.now() - startMs) / 1000);
            const rem = intervalSec - elapsed;
            if (rem >= 0) {{
                timer.textContent = "Next refresh in " + rem + "s (" + elapsed + "s ago)";
            }} else if (elapsed <= warnSec) {{
                timer.textContent = "Syncing with server... (" + elapsed + "s ago)";
                box.style.background = "rgba(245, 158, 11, 0.14)";
                box.style.borderColor = "rgba(245, 158, 11, 0.45)";
                box.style.color = "#92400e";
                dot.style.background = "#f59e0b";
                dot.style.boxShadow = "0 0 6px #f59e0b";
                status.textContent = "Waiting for Streamlit";
            }} else {{
                timer.textContent = "Last sync " + elapsed + "s ago — click Refresh!";
                box.style.background = "rgba(239, 68, 68, 0.15)";
                box.style.borderColor = "rgba(239, 68, 68, 0.5)";
                box.style.color = "#991b1b";
                dot.style.background = "#ef4444";
                dot.style.boxShadow = "none";
                status.textContent = "⚠️ Tab Asleep / Stale";
            }}
        }}, 1000);
    </script>
    """
    components.html(html_code, height=45)


# ------------------------------------------------------------------ small UI helpers
def level_table(rows):
    md = "| | NIFTY level | Option price (est.) | Per lot |\n|---|---:|---:|---|\n"
    for lbl, u, o, money in rows:
        md += f"| **{lbl}** | {num(u, 1)} | {inr(o)} | {money} |\n"
    return md


def plan_rows(spot, plan, lv, lot):
    return [("Entry (buy)", spot, lv["entry"], f"pay {inr(lv['entry'] * lot)}"),
            ("Stop-loss (incl. 20m decay + slip)", plan["sl"], lv["p_sl"], f"lose about {inr((lv['entry'] - lv['p_sl']) * lot)}"),
            ("Target 1 (incl. 45m decay + slip)", plan["t1"], lv["p_t1"], f"gain about {inr((lv['p_t1'] - lv['entry']) * lot)}"),
            ("Target 2 (incl. 90m decay + slip)", plan["t2"], lv["p_t2"], f"gain about {inr((lv['p_t2'] - lv['entry']) * lot)}")]


def show_checks(res):
    for k in res["checks"]:
        icon = "➖" if k["ok"] is None else ("✅" if k["ok"] else "❌")
        tag = "" if k["core"] else " *(bonus)*"
        st.markdown(f"{icon} **{k['label']}**{tag}: {k['detail']}")


def render_suggestion(res, lot, acct, now):
    o, p = res["opt"], res["plan"]
    lv, spot = p["levels"], res["spot"]
    call = o["side"] == "CE"
    with st.container(border=True):
        st.markdown(f"### {'🟢' if call else '🔴'} {res['decision']}: {o['label']}")
        oi_str = f" · Strike OI {num(o['oi'])} (ΔOI {o['chg_oi']:+,})" if o.get("oi") else ""
        st.caption(f"Signal from 5m candle at {dt.datetime.fromisoformat(p['asof']).strftime('%H:%M')} IST · "
                   f"Source: **{o.get('quote_source', 'Model')}**{oi_str} · {res['reason']}")
        a, b, c = st.columns(3)
        a.metric("1 lot costs", inr(lv["entry"] * lot))
        b.metric("Loss at stop-loss", inr((lv["entry"] - lv["p_sl"]) * lot))
        c.metric("Gain at Target 1", inr((lv["p_t1"] - lv["entry"]) * lot))
        st.markdown(level_table(plan_rows(spot, p, lv, lot)))
        st.warning(
            f"⚠️ **How to exit:** at the stop-loss; at Target 2 (once NIFTY reaches Target 1 the stop moves up to your entry level); "
            f"or earlier if a 5-minute candle closes {'below' if call else 'above'} the 21 EMA or RSI goes {'under' if call else 'over'} 50. "
            f"Close everything by 3:15 PM IST.")
        with st.expander("Why this trade: all checks"):
            show_checks(res)
        st.caption(f"Option price source: **{o.get('quote_source')}** (IV {o['iv'] * 100:.1f}%). "
                   f"Always verify {o['label']} in your broker app before trading.")
        real = st.number_input("Override option price from your broker app (₹, optional)", min_value=0.0, value=0.0, step=0.5,
                               key="real_price", help="Leave 0 to use the price shown above.")
        entry = float(real) if real > 0 else o["prem"]
        if real > 0 and real * lot > MAX_COST:
            st.error(f"At the real price one lot costs {inr(real * lot)}, above your {inr(MAX_COST)} limit. "
                     "In real trading you would skip this one.")
        if st.button(f"📝 Start paper trade (1 lot at {inr(entry)})", key="start_paper", use_container_width=True, type="primary"):
            ok, msg = open_position(acct, res, entry, "your price" if real > 0 else o.get("quote_source", "estimate"), now)
            st.session_state["flash"] = msg
            st.rerun()


# ------------------------------------------------------------------ page
if "paper" not in st.session_state:
    st.session_state["paper"] = new_account()
if "hb_count" not in st.session_state:
    st.session_state["hb_count"] = 0
if "flash" in st.session_state:
    st.success(st.session_state.pop("flash"))

st.markdown("## 📈 NIFTY Paper Trader (v3 · Live NSE Chain + 1m Tracker)")
st.caption("One NIFTY option trade at a time, 1 lot, under ₹5,000. Combines live NSE Option Chain quotes/PCR, "
           "60-day historical AI twin scoring, and 1-minute intrabar stop/target tracking.")

with st.expander("⚙️ Settings"):
    lot_size = st.number_input("NIFTY lot size", min_value=1, max_value=500, value=DEFAULT_LOT, step=1, key="lot_size",
                               help="65 since January 2026. NSE changes this from time to time; type what your broker shows.")
    exp_name = st.selectbox("Weekly expiry day", DAYS, index=1, key="exp_day",
                            help="Tuesday since September 2025. Used as fallback if NSE live chain is unreachable.")
    auto_refresh = st.toggle("Auto-refresh (15s when trade open, 30s when scanning)", value=True)
    if st.button("🔄 Refresh now", use_container_width=True):
        load_prices.clear()
        load_vix.clear()
        fetch_nse_chain.clear()
        learn_from_history.clear()
weekday = DAYS.index(exp_name)

# Dynamic polling speed: 15s when a trade is active for sharp exits, 30s when scanning
_has_open_pos = bool(st.session_state["paper"]["positions"])
_poll_sec = (15 if _has_open_pos else 30) if auto_refresh else None


@st.fragment(run_every=_poll_sec)
def trade_view(lot, weekday):
    st.session_state["hb_count"] += 1
    acct = st.session_state["paper"]
    now = dt.datetime.now(IST)
    active_interval = 15 if acct["positions"] else 30
    render_heartbeat(now, st.session_state["hb_count"], auto_refresh, interval_sec=active_interval)

    state = market_state(now)
    try:
        raw5 = load_prices(TICKER, "5m", "60d")
        raw15 = load_prices(TICKER, "15m", "60d")
        # Fetch 1-minute candles when a position is open for 5x sharper stop/target resolution
        raw1 = load_prices(TICKER, "1m", "5d") if acct["positions"] else pd.DataFrame()
    except Exception as e:
        st.error(f"Could not fetch NIFTY data from Yahoo Finance: {e}")
        return
    if raw5.empty:
        st.error("No NIFTY data. Yahoo may be rate-limiting this server. Try again in a minute.")
        return

    d5 = add_indicators(raw5)
    d15 = add_indicators(raw15) if len(raw15) else None
    vix = load_vix()
    nse_chain = fetch_nse_chain()

    spot = float(d5["Close"].iloc[-1])
    badge = {"open": "🟢 Market open", "square_off": "🟠 Square-off time", "closed": "⚪ Market closed"}[state]
    nse_badge = (f"🟢 NSE Chain Live (PCR {nse_chain['pcr']:.2f})"
                 if (nse_chain and nse_chain.get("pcr")) else "🟡 Option Model Fallback")
    st.caption(f"{badge} · {nse_badge} · NIFTY {num(spot, 2)} · last 5m bar {d5.index[-1]:%d %b %H:%M} IST · "
               f"India VIX {('%.1f' % vix) if vix else 'n/a'}")

    # ---------- a paper trade is open: show only that trade
    if acct["positions"]:
        pos = acct["positions"][0]
        res = evaluate_position(pos, d5, now, acct.get("use_pullout", True), acct.get("t1_mode", "trail"),
                                df1=raw1, nse_chain=nse_chain)
        if not res["stale"]:
            for e in res["new_events"]:
                st.toast(e["text"], icon="📝")
            if res["new_legs"] or res["new_events"]:
                rec = settle(acct, pos["id"], res)
                if rec:
                    st.session_state["flash"] = (f"Paper trade closed: {rec['reason']}. "
                                                 f"{'Profit' if rec['pnl'] > 0 else 'Loss'} {inr(abs(rec['pnl']))} after charges.")
                    st.rerun()
                pos = acct["positions"][0]
        cur = None if res["stale"] else prem_at(pos, spot, now, slippage=SLIPPAGE_EXIT, nse_chain=nse_chain, is_now=True)
        live_ltp = live_option_ltp(pos, nse_chain)
        call = pos["side"] == "CE"
        opened = dt.datetime.fromisoformat(pos["entry_time"])
        with st.container(border=True):
            st.markdown(f"### {'🟢' if call else '🔴'} Your paper trade: {pos['label']}")
            track_mode = "1m high-res candles + NSE live quote" if live_ltp is not None else "1m high-res candles + skew model"
            st.caption(f"1 lot · opened {opened:%d %b %H:%M} · bought at {inr(pos['entry_prem'])} ({pos['source']}) · "
                       f"Tracking via **{track_mode}**")
            m1, m2, m3 = st.columns(3)
            m1.metric("Bought at", inr(pos["entry_prem"]))
            m2.metric("Now (" + ("NSE Live" if live_ltp is not None else "est.") + ")",
                      inr(cur) if cur is not None else "n/a")
            if cur is not None:
                m3.metric("P&L so far", inr((cur - pos["entry_prem"]) * pos["lot"]))
            rows = [("Stop-loss" if not pos["t1_done"] else "Stop (moved to entry)", pos["stop_u"],
                     pos["p_sl"] if not pos["t1_done"] else pos["entry_prem"]),
                    ("Target 1" + (" ✅ reached" if pos["t1_done"] else ""), pos["t1"], pos["p_t1"]),
                    ("Target 2", pos["t2"], pos["p_t2"])]
            md = "| | NIFTY level | Option price (est.) |\n|---|---:|---:|\n"
            for lbl, u, o in rows:
                md += f"| **{lbl}** | {num(u, 1)} | {inr(o)} |\n"
            st.markdown(md)
            st.caption(f"NIFTY is {num(spot, 1)}; you entered at {num(pos['entry_spot'], 1)}.")
            closed5 = d5.iloc[:-1] if (d5.index[-1] + pd.Timedelta(minutes=5)) > now else d5
            if len(closed5) and not res["stale"]:
                c = closed5.iloc[-1]
                ok_ema = c["Close"] > c["EMA21"] if call else c["Close"] < c["EMA21"]
                ok_rsi = c["RSI"] >= 50 if call else c["RSI"] <= 50
                st.markdown(
                    f"**Exit rule right now:** {'✅' if ok_ema else '❌'} last 5m candle closed {'above' if call else 'below'} the 21 EMA "
                    f"({num(c['Close'], 1)} vs {num(c['EMA21'], 1)}) · {'✅' if ok_rsi else '❌'} RSI {c['RSI']:.0f} "
                    f"(must stay {'above' if call else 'below'} 50)")
            for e in pos["events"]:
                st.caption(f"✔ {dt.datetime.fromisoformat(e['time']):%H:%M}: {e['text']}")
            if res["stale"]:
                st.error("This trade is from an earlier day (or data is missing), so it cannot be checked automatically. "
                         "Type your own exit price and close it.")
            elif state == "closed":
                st.info("The market is closed. The app will settle this trade at the 3:15 PM square-off price.")
            ex = st.number_input("Exit option price (₹, optional)", min_value=0.0, value=0.0, step=0.5, key="exit_price",
                                 help="Leave 0 to use the current price shown above.")
            if st.button("Close this paper trade now", key="close_now", use_container_width=True):
                price = float(ex) if ex > 0 else cur
                if price is None:
                    st.error("No price available. Type the exit price above.")
                else:
                    p2 = copy.deepcopy(pos)
                    p2["legs"].append({"time": now.isoformat(), "lots": 1, "prem": float(price), "reason": "Closed manually"})
                    p2["lots_open"] = 0
                    rec = settle(acct, pos["id"], {"pos": p2, "closed": True, "new_legs": [], "new_events": []})
                    st.session_state["flash"] = f"Paper trade closed. {'Profit' if rec['pnl'] > 0 else 'Loss'} {inr(abs(rec['pnl']))} after charges."
                    st.rerun()
        st.caption("While this trade is open the app polls every 15 seconds using 1-minute candles.")
        return

    # ---------- no open trade: suggest one, or say wait
    if state == "closed":
        st.info("The market is closed. Suggestions run Monday to Friday, 9:30 AM to 2:45 PM IST.")
        return
    if state == "square_off":
        st.warning("It is past 3:15 PM IST. No new trades today.")
        return
    res = analyse(d5, d15, vix, now, lot, weekday, nse_chain=nse_chain)
    if acct["closed"]:
        t = acct["closed"][-1]
        st.caption(f"Last paper trade: {t['label']}, {t['reason']}, {inr(t['pnl'])}.")
    if res["decision"] != "WAIT":
        render_suggestion(res, lot, acct, now)
    else:
        with st.container(border=True):
            st.markdown("### ⚪ No trade right now")
            st.write(res["reason"])
            st.caption(f"Core checks {res['core_pass']} of {res['core_total']} passed"
                       + (f" · extras {res['bonus_pass']} of {res['bonus_total']} (need {res['bonus_need']})" if res["bonus_total"] else "")
                       + ". The app re-checks every 30 seconds and judges only completed 5-minute & 15-minute candles.")
            with st.expander("See every check"):
                show_checks(res)


@st.fragment(run_every=30 if auto_refresh else None)
def results_view():
    acct = st.session_state["paper"]
    done = paper_stats(acct["closed"])
    st.markdown("### 📝 Paper results")
    if not acct["closed"]:
        st.info("No finished paper trades yet. Start one from a suggestion on the Trade tab.")
    else:
        a, b, c = st.columns(3)
        a.metric("Trades", done["n"])
        b.metric("Win rate", f"{done['win_rate']:.0f}%")
        c.metric("Net P&L", inr(done["total"]), f"{done['total'] / MAX_COST * 100:+.1f}% of ₹5,000")
        d, e, f = st.columns(3)
        d.metric("Avg win", inr(done["avg_win"]))
        e.metric("Avg loss", inr(done["avg_loss"]))
        f.metric("Profit factor", f"{done['pf']:.2f}" if done["pf"] else "n/a")
        if done["n"] < 30:
            st.caption(f"Only {done['n']} trade{'s' if done['n'] != 1 else ''} so far. Win rate and profit factor mean little "
                       "until you have at least 30. Keep going before trusting them.")
        hist = pd.DataFrame([{
            "Closed": dt.datetime.fromisoformat(t["exit_time"]).strftime("%d %b %H:%M"), "Trade": t["label"],
            "Bought": inr(t["entry_prem"]), "Sold": inr(t["exit_prem"]), "P&L": inr(t["pnl"]), "How it ended": t["reason"]}
            for t in reversed(acct["closed"])])
        st.dataframe(hist, hide_index=True, use_container_width=True)
        csv = pd.DataFrame([{k: v for k, v in t.items() if k != "legs"} for t in acct["closed"]]).to_csv(index=False)
        st.download_button("⬇️ Download trade history (CSV)", csv, "paper_trades.csv", "text/csv", use_container_width=True)

    with st.expander("⚙️ Paper settings, backup and reset"):
        st.warning("Your paper account lives only in this browser session. If you refresh the page, close it, or the app "
                   "restarts, it resets. Download a backup after each session and restore it next time.")
        st.download_button("⬇️ Download account backup", json.dumps(acct, indent=1), "paper_account_backup.json",
                           "application/json", use_container_width=True)
        up = st.file_uploader("Restore from a backup file", type="json", key="restore_file")
        if up is not None and st.button("Restore this backup", use_container_width=True):
            try:
                data = json.loads(up.getvalue().decode("utf-8"))
                if not {"charges", "positions", "closed", "next_id"}.issubset(data):
                    raise ValueError("missing fields")
                if any("label" not in t for t in data["closed"]) or any("strike" not in x for x in data["positions"]):
                    raise ValueError("old format")
                data.setdefault("use_pullout", True)
                data.setdefault("t1_mode", "trail")
                st.session_state["paper"] = data
                st.session_state["flash"] = "Backup restored."
                st.rerun()
            except Exception:
                st.error("That file is not a valid backup from this version of the app.")
        acct["use_pullout"] = st.toggle("Close trades automatically on the exit rule", value=acct.get("use_pullout", True),
                                        help="Exit when a completed 5-minute candle closes against the 21 EMA or RSI flips. "
                                             "Stop-loss, targets and the 3:15 PM square-off are always automatic.")
        mode = st.selectbox("At Target 1", ["Move the stop to my entry level and aim for Target 2", "Sell the lot at Target 1"],
                            index=0 if acct.get("t1_mode", "trail") == "trail" else 1,
                            help="With one lot you cannot sell half, so you must pick one of these.")
        acct["t1_mode"] = "trail" if mode.startswith("Move") else "exit"
        acct["charges"] = float(st.number_input("Charges per trade (₹, estimate)", min_value=0.0, value=float(acct["charges"]),
                                                step=10.0, help="Brokerage and taxes for the round trip. Your broker's contract note has the real figure."))
        sure = st.checkbox("Yes, erase my paper trades and start again")
        if st.button("Reset paper account", disabled=not sure):
            st.session_state["paper"] = new_account(acct["charges"])
            st.session_state["flash"] = "Paper account reset."
            st.rerun()
    st.caption("Exits use 1-minute candles for stop-loss and targets, 5-minute completed candles for the 21 EMA/RSI exit rule, "
               "and live NSE option quotes when connected.")


tab_trade, tab_results = st.tabs(["📍 Trade", "📝 Paper results"])
with tab_trade:
    trade_view(int(lot_size), weekday)
with tab_results:
    results_view()

with st.expander("ℹ️ How the suggestion is made"):
    st.markdown(
        "**Every check in the first group must pass before a trade is suggested;** at least two of the extras must too.\n\n"
        "- **Clear direction:** the latest completed 5-minute candle is above (or below) the day's range-weighted average price, and the 9 EMA is above "
        "(or below) the 21 EMA.\n"
        "- **15-minute trend agrees (completed candle),** so you are not fighting the bigger move.\n"
        "- **RSI** shows momentum without being stretched (55 to 72 for a call, 28 to 45 for a put).\n"
        "- **ADX** is at least 20: NIFTY is trending, not chopping sideways.\n"
        "- **India VIX** is not above 25 (options get very expensive).\n"
        "- **Entry window** 9:30 AM to 2:45 PM IST, and Yahoo's data is fresh.\n"
        "- **A strike under ₹5,000 per lot exists** using live NSE Option Chain ask/LTP and implied volatility (or skew-adjusted Black-Scholes fallback).\n"
        "- **Historical AI Win Rate (≥60%):** scans the last 60 days of 5-minute data, finds the 25 closest historical twin setups "
        "using 7 normalized market features, and verifies they won at least 60% of the time with positive R-expectancy.\n\n"
        "**Extras (at least two must pass):** price has broken the opening range; the last candle closed firmly in the trade's direction; "
        "yesterday's high/low is not blocking Target 1; **live NSE Put-Call Ratio (PCR)** supports the direction.\n\n"
        "**Levels & Tracking:** stop-loss 1.5x ATR, Target 1 2.0x ATR, Target 2 3.5x ATR. Open trades are monitored every 15 seconds on 1-minute candles.")
st.caption("An experiment, not advice. Most individual F&O traders lose money, and options can expire worthless.")
