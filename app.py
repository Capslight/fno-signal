import copy
import datetime as dt
import json
import math

import numpy as np
import pandas as pd
import pytz
import streamlit as st
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


# ---- LOGIC START (pure functions, no Streamlit; covered by tests)
# ------------------------------------------------------------------ option pricing
def _ncdf(x):
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


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


def pick_option(spot, side, now, iv, lot, weekday, max_cost=MAX_COST):
    """Closest-to-the-money strike whose one-lot cost fits the cap. None if nothing decent fits."""
    expiry = next_expiry(now, weekday)
    T = years(expiry, now)
    atm = int(round(spot / STEP) * STEP)
    for k in range(0, 60):
        K = atm + k * STEP if side == "CE" else atm - k * STEP
        model = bs_price(spot, K, T, iv, side)
        prem = round_tick(model)
        if prem * lot <= max_cost:
            delta = bs_delta(spot, K, T, iv, side)
            if prem < 0.5 or delta < MIN_DELTA:
                return None       # anything cheaper is even less responsive
            return {"strike": int(K), "side": side, "expiry": expiry.isoformat(), "iv": iv, "prem": prem,
                    "model": model, "cost": prem * lot, "delta": delta, "steps_otm": k,
                    "label": f"NIFTY {expiry:%d %b} {int(K)} {side}"}
    return None


def prem_at(pos, spot, ts):
    """Estimated option price if NIFTY is at `spot` at time `ts`. Re-priced with Black-Scholes (so time decay and
    curvature are included) and anchored to the price you actually 'paid'."""
    T = years(dt.datetime.fromisoformat(pos["expiry"]), ts)
    p = pos["entry_prem"] + bs_price(spot, pos["strike"], T, pos["iv"], pos["side"]) - pos["model_entry"]
    return max(round_tick(p), 0.0)


def build_levels(opt, spot, sl, t1, t2, entry_prem=None):
    """Option price at the stop-loss and targets (no time decay; it is a plan, not a forecast)."""
    T = years(dt.datetime.fromisoformat(opt["expiry"]), dt.datetime.fromisoformat(opt["now"]))
    entry = float(entry_prem) if entry_prem else opt["prem"]
    off = entry - opt["model"]

    def at(level):
        return max(round_tick(bs_price(level, opt["strike"], T, opt["iv"], opt["side"]) + off), 0.0)
    return {"entry": entry, "p_sl": at(sl), "p_t1": at(t1), "p_t2": at(t2)}


# ------------------------------------------------------------------ indicators
def add_indicators(df):
    df = df.copy()
    h, l, c = df["High"], df["Low"], df["Close"]
    tp = (h + l + c) / 3
    # Yahoo has no volume for the NIFTY index, so this is the day's running average price, not a true VWAP.
    df["DAYAVG"] = tp.groupby(df.index.date).transform(lambda s: s.expanding().mean())
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


# ------------------------------------------------------------------ the analysis
def analyse(d5, d15, vix, now, lot, weekday):
    """Judge NIFTY from completed 5-minute candles. Returns the decision and every check with its numbers.

    Core checks must ALL pass. Of the three bonus checks, at least two must pass (or all that can be measured)."""
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
        f"close {num(c['Close'], 1)}, day average {num(c['DAYAVG'], 1)}, 9 EMA {num(c['EMA9'], 1)}, 21 EMA {num(c['EMA21'], 1)}"
        + ("" if side else " (they disagree)"))
    sgn = 1 if side == "CE" else -1

    in_window = now.weekday() < 5 and ENTRY_FROM <= now.time() <= ENTRY_UNTIL
    add("Entry window (9:30 to 2:45 PM IST)", in_window, now.strftime("%H:%M IST") +
        ("" if in_window else ": the first 15 minutes are too wild and after 2:45 there is too little time"))
    lag = (now - last_ts).total_seconds() / 60
    fresh = lag <= 5 + 10 and last_ts.date() == now.date()
    add("Data is fresh", fresh, f"last candle {last_ts:%H:%M}, {lag:.0f} min ago" + ("" if fresh else " (Yahoo looks delayed)"))

    if d15 is not None and len(d15) >= 25:
        h = d15.iloc[-1]
        ok15 = (side is not None) and ((h["EMA9"] > h["EMA21"]) if side == "CE" else (h["EMA9"] < h["EMA21"]))
        add("15-min trend agrees", bool(ok15), f"15-min 9 EMA {num(h['EMA9'], 1)} vs 21 EMA {num(h['EMA21'], 1)}")
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

    opt = pick_option(spot, side, now, iv, lot, weekday) if side else None
    if side:
        add("A strike under ₹5,000 per lot exists", opt is not None,
            (f"{opt['label']} costs about {inr(opt['cost'])} (delta {opt['delta']:.2f})" if opt
             else f"nothing under {inr(MAX_COST)} with enough response to NIFTY moves"))
    else:
        add("A strike under ₹5,000 per lot exists", False, "no direction yet")

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
        out["reason"] = "All core checks passed" + (f" and {bonus_pass} of {len(bonus)} bonus checks." if bonus else ".")
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
    pos = {"id": acct["next_id"], "label": o["label"], "side": o["side"], "strike": o["strike"], "expiry": o["expiry"],
           "iv": o["iv"], "lot": int(p["lot"]), "lots_open": 1, "entry_time": now.isoformat(), "entry_spot": float(res["spot"]),
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


def evaluate_position(pos, df, now, use_pullout=True, t1_mode="trail"):
    """Replay the app's own rules on the candles since entry. Inside one candle the cautious order is used:
    stop-loss first, then Target 1, then Target 2, then the exit rule on completed candles. With one lot nothing can be
    split, so at Target 1 the stop moves to the entry price (t1_mode 'trail') or the lot is sold ('exit').
    Anything still open at 3:15 PM IST is squared off. Option prices are re-priced at the moment of each exit."""
    p = copy.deepcopy(pos)
    entry_t = dt.datetime.fromisoformat(p["entry_time"])
    if df is None or df.empty or df.index[-1].date() != entry_t.date():
        return {"pos": p, "closed": False, "stale": True, "new_legs": [], "new_events": []}
    span = pd.Timedelta(minutes=int(p.get("interval_min", 5)))
    sq = IST.localize(dt.datetime.combine(entry_t.date(), SQUARE_T))
    n0, e0 = len(p["legs"]), len(p["events"])

    def leg(ts, spot_level, reason):
        p["legs"].append({"time": ts.isoformat(), "lots": p["lots_open"], "prem": float(prem_at(p, spot_level, ts)),
                          "reason": reason})
        p["lots_open"] = 0

    def check(ts, low, high, close, ema21, rsi, completed):
        if p["lots_open"] <= 0:
            return
        if _stop_hit(p, low, high):
            leg(ts, p["stop_u"], "Stop-loss" if not p["t1_done"] else "Stop moved to entry price after Target 1")
            return
        if not p["t1_done"] and _target_hit(p, p["t1"], low, high):
            if t1_mode == "exit":
                leg(ts, p["t1"], "Target 1")
                return
            p["t1_done"], p["stop_u"] = True, p["entry_spot"]
            p["events"].append({"time": ts.isoformat(), "text": "Target 1 reached: stop-loss moved to your entry level"})
        if p["t1_done"] and _target_hit(p, p["t2"], low, high):
            leg(ts, p["t2"], "Target 2")
            return
        if use_pullout and completed and not (np.isnan(ema21) or np.isnan(rsi)):
            bad = (close < ema21 or rsi < 50) if p["side"] == "CE" else (close > ema21 or rsi > 50)
            if bad:
                leg(ts + span, close, "App exit rule (price crossed the 21 EMA or RSI flipped)")

    rows = df[(df.index > entry_t) & (df.index < sq)]
    for ts, r in rows.iterrows():
        if p["lots_open"] <= 0:
            break
        check(ts, float(r["Low"]), float(r["High"]), float(r["Close"]), float(r["EMA21"]), float(r["RSI"]), (ts + span) <= now)
    if p["lots_open"] > 0 and now < sq:
        spot = float(df["Close"].iloc[-1])
        check(now, spot, spot, spot, float("nan"), float("nan"), False)
    if p["lots_open"] > 0 and now >= sq:
        sub = df[df.index <= sq]
        leg(sq, float((sub if len(sub) else df)["Close"].iloc[-1]), "Square-off at 3:15 PM")
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


@st.cache_data(ttl=20, show_spinner=False)
def load_prices(ticker, interval, period):
    return _clean(yf.download(ticker, period=period, interval=interval, progress=False, auto_adjust=True))


@st.cache_data(ttl=120, show_spinner=False)
def load_vix():
    try:
        d = _clean(yf.download(VIX_TICKER, period="5d", interval="1d", progress=False, auto_adjust=True))
        return float(d["Close"].iloc[-1]) if len(d) else None
    except Exception:
        return None


# ------------------------------------------------------------------ small UI helpers
def level_table(rows):
    md = "| | NIFTY level | Option price (est.) | Per lot |\n|---|---:|---:|---|\n"
    for lbl, u, o, money in rows:
        md += f"| **{lbl}** | {num(u, 1)} | {inr(o)} | {money} |\n"
    return md


def plan_rows(spot, plan, lv, lot):
    return [("Entry (buy)", spot, lv["entry"], f"pay {inr(lv['entry'] * lot)}"),
            ("Stop-loss", plan["sl"], lv["p_sl"], f"lose about {inr((lv['entry'] - lv['p_sl']) * lot)}"),
            ("Target 1", plan["t1"], lv["p_t1"], f"gain about {inr((lv['p_t1'] - lv['entry']) * lot)}"),
            ("Target 2", plan["t2"], lv["p_t2"], f"gain about {inr((lv['p_t2'] - lv['entry']) * lot)}")]


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
        st.caption(f"Signal from the 5-minute candle that closed at {dt.datetime.fromisoformat(p['asof']).strftime('%H:%M')} IST · "
                   f"{res['reason']}")
        a, b, c = st.columns(3)
        a.metric("1 lot costs (est.)", inr(lv["entry"] * lot))
        b.metric("Loss at stop-loss", inr((lv["entry"] - lv["p_sl"]) * lot))
        c.metric("Gain at Target 1", inr((lv["p_t1"] - lv["entry"]) * lot))
        st.markdown(level_table(plan_rows(spot, p, lv, lot)))
        st.warning(
            f"⚠️ **How to exit:** at the stop-loss; at Target 2 (once NIFTY reaches Target 1 the stop moves up to your entry level); "
            f"or earlier if a 5-minute candle closes {'below' if call else 'above'} the 21 EMA or RSI goes {'under' if call else 'over'} 50. "
            f"Close everything by 3:15 PM IST.")
        with st.expander("Why this trade: all checks"):
            show_checks(res)
        st.caption(f"Option prices are estimates from a pricing model (volatility {o['iv'] * 100:.0f}%), so the real price can differ, "
                   f"especially for strikes away from the money. Check the real price of {o['label']} in your broker app first. "
                   f"If one lot costs more than {inr(MAX_COST)} there, skip it.")
        real = st.number_input("Real option price from your broker app (₹, optional)", min_value=0.0, value=0.0, step=0.5,
                               key="real_price", help="Makes the paper trade use the real price instead of the estimate.")
        entry = float(real) if real > 0 else o["prem"]
        if real > 0 and real * lot > MAX_COST:
            st.error(f"At the real price one lot costs {inr(real * lot)}, above your {inr(MAX_COST)} limit. "
                     "In real trading you would skip this one.")
        if st.button(f"📝 Start paper trade (1 lot at {inr(entry)})", key="start_paper", use_container_width=True, type="primary"):
            ok, msg = open_position(acct, res, entry, "your price" if real > 0 else "estimate", now)
            st.session_state["flash"] = msg
            st.rerun()


# ------------------------------------------------------------------ page
if "paper" not in st.session_state:
    st.session_state["paper"] = new_account()
if "flash" in st.session_state:
    st.success(st.session_state.pop("flash"))

st.markdown("## 📈 NIFTY Paper Trader")
st.caption("One NIFTY option trade at a time, 1 lot, under ₹5,000. The app suggests a trade when its checks line up, "
           "then follows it for you.")

with st.expander("⚙️ Settings"):
    lot_size = st.number_input("NIFTY lot size", min_value=1, max_value=500, value=DEFAULT_LOT, step=1, key="lot_size",
                               help="65 since January 2026. NSE changes this from time to time; type what your broker shows.")
    exp_name = st.selectbox("Weekly expiry day", DAYS, index=1, key="exp_day",
                            help="Tuesday since September 2025. If it is a holiday, expiry moves to the previous day.")
    auto_refresh = st.toggle("Auto-refresh every 30 seconds", value=True)
    if st.button("🔄 Refresh now", use_container_width=True):
        load_prices.clear()
        load_vix.clear()
weekday = DAYS.index(exp_name)


@st.fragment(run_every=30 if auto_refresh else None)
def trade_view(lot, weekday):
    acct = st.session_state["paper"]
    now = dt.datetime.now(IST)
    state = market_state(now)
    try:
        raw5 = load_prices(TICKER, "5m", "5d")
        raw15 = load_prices(TICKER, "15m", "5d")
    except Exception as e:
        st.error(f"Could not fetch NIFTY data from Yahoo Finance: {e}")
        return
    if raw5.empty:
        st.error("No NIFTY data. Yahoo may be rate-limiting this server. Try again in a minute.")
        return
    d5 = add_indicators(raw5)
    d15 = add_indicators(raw15) if len(raw15) else None
    vix = load_vix()
    spot = float(d5["Close"].iloc[-1])
    badge = {"open": "🟢 Market open", "square_off": "🟠 Square-off time", "closed": "⚪ Market closed"}[state]
    st.caption(f"{badge} · NIFTY {num(spot, 2)} · last candle {d5.index[-1]:%d %b %H:%M} IST · "
               f"India VIX {('%.1f' % vix) if vix else 'n/a'} · data can lag (Yahoo is unofficial)")

    # ---------- a paper trade is open: show only that trade
    if acct["positions"]:
        pos = acct["positions"][0]
        res = evaluate_position(pos, d5, now, acct.get("use_pullout", True), acct.get("t1_mode", "trail"))
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
        cur = None if res["stale"] else prem_at(pos, spot, now)
        call = pos["side"] == "CE"
        opened = dt.datetime.fromisoformat(pos["entry_time"])
        with st.container(border=True):
            st.markdown(f"### {'🟢' if call else '🔴'} Your paper trade: {pos['label']}")
            st.caption(f"1 lot · opened {opened:%d %b %H:%M} · bought at {inr(pos['entry_prem'])} ({pos['source']})")
            m1, m2, m3 = st.columns(3)
            m1.metric("Bought at", inr(pos["entry_prem"]))
            m2.metric("Now (est.)", inr(cur) if cur is not None else "n/a")
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
                    f"**Exit rule right now:** {'✅' if ok_ema else '❌'} last candle closed {'above' if call else 'below'} the 21 EMA "
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
                                 help="Leave 0 to use the app's estimate of the current price.")
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
        st.caption("While this trade is open the app gives no other suggestions.")
        return

    # ---------- no open trade: suggest one, or say wait
    if state == "closed":
        st.info("The market is closed. Suggestions run Monday to Friday, 9:30 AM to 2:45 PM IST.")
        return
    if state == "square_off":
        st.warning("It is past 3:15 PM IST. No new trades today.")
        return
    res = analyse(d5, d15, vix, now, lot, weekday)
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
                       + ". The app re-checks every 30 seconds and judges only completed 5-minute candles. "
                       "Doing nothing is a valid position.")
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
    st.caption("Exits use the same stop-loss, targets, exit rule and 3:15 PM square-off shown on the suggestion. Prices are model "
               "estimates and fills are perfect, so real trading will usually do worse than paper (bid-ask gap, gaps past your stop, "
               "slippage).")


tab_trade, tab_results = st.tabs(["📍 Trade", "📝 Paper results"])
with tab_trade:
    trade_view(int(lot_size), weekday)
with tab_results:
    results_view()

with st.expander("ℹ️ How the suggestion is made"):
    st.markdown(
        "**Every check in the first group must pass before a trade is suggested;** at least two of the three extras must too.\n\n"
        "- **Clear direction:** the latest completed 5-minute candle is above (or below) the day's average price, and the 9 EMA is above "
        "(or below) the 21 EMA.\n"
        "- **15-minute trend agrees,** so you are not fighting the bigger move.\n"
        "- **RSI** shows momentum without being stretched (55 to 72 for a call, 28 to 45 for a put).\n"
        "- **ADX** is at least 20: NIFTY is trending, not chopping sideways.\n"
        "- **India VIX** is not above 25 (options get very expensive).\n"
        "- **Entry window** 9:30 AM to 2:45 PM IST, and Yahoo's data is fresh.\n"
        "- **A strike under ₹5,000 per lot exists** with enough response to NIFTY moves. The app uses the nearest weekly expiry "
        "that is at least two days away.\n\n"
        "**Extras (two of three):** price has broken the opening range; the last candle closed firmly in the trade's direction; "
        "yesterday's high (for calls) or low (for puts) is not in the way of Target 1.\n\n"
        "**Levels:** stop-loss 1.5 times, Target 1 two times and Target 2 3.5 times the recent average candle range (ATR) from the entry.")
st.caption("An experiment, not advice. These checks have not been tested on past data, so there is no evidence they make money. "
           "Most individual F&O traders lose money, and options can expire worthless.")
