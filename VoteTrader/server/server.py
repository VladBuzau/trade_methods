"""
VoteTrader Server
12 metode de analiza independente — vot majoritar → semnal
Fiecare metoda returneaza BUY / SELL / NEUTRAL
Daca >= min_votes (default 7/12) sunt de acord → pozitie
"""

import json, logging, threading
from datetime import datetime, timezone
from flask import Flask, request, Response
import yfinance as yf
import pandas as pd
import numpy as np

# ── Config ───────────────────────────────────────────────────────────────────
with open("config.json") as f:
    CFG = json.load(f)

HOST       = CFG.get("host", "localhost")
PORT       = CFG.get("port", 5003)
MIN_VOTES  = CFG.get("min_votes", 7)
SYMBOLS    = CFG.get("symbols", ["EURUSD", "GBPUSD"])

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

app = Flask(__name__)

# ── Signal log ────────────────────────────────────────────────────────────────
_signal_log = []
_log_lock   = threading.Lock()

def push_log(entry):
    with _log_lock:
        _signal_log.append(entry)
        if len(_signal_log) > 300:
            _signal_log.pop(0)

# ── JSON encoder numpy-safe ───────────────────────────────────────────────────
class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer): return int(obj)
        if isinstance(obj, np.floating): return float(obj)
        if isinstance(obj, np.ndarray): return obj.tolist()
        return super().default(obj)

def jresp(data):
    return Response(json.dumps(data, cls=NpEncoder, separators=(',', ':')),
                    mimetype="application/json")

# ── Yahoo Finance symbol map ──────────────────────────────────────────────────
YF_MAP = {
    "EURUSD": "EURUSD=X", "GBPUSD": "GBPUSD=X", "USDJPY": "JPY=X",
    "USDCHF": "CHF=X",    "AUDUSD": "AUDUSD=X", "USDCAD": "CAD=X",
    "NZDUSD": "NZDUSD=X", "EURGBP": "EURGBP=X", "EURJPY": "EURJPY=X",
    "GBPJPY": "GBPJPY=X", "AUDJPY": "AUDJPY=X", "EURAUD": "EURAUD=X",
    "GBPAUD": "GBPAUD=X", "XAUUSD": "GC=F",     "XAGUSD": "SI=F",
}

# ── Data fetch ────────────────────────────────────────────────────────────────
def fetch(symbol, interval, period):
    yf_sym = YF_MAP.get(symbol, symbol + "=X")
    try:
        df = yf.download(yf_sym, interval=interval, period=period,
                         progress=False, auto_adjust=True)
        if df is None or len(df) < 30:
            return None
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        df.columns = [c[0].lower() if isinstance(c, tuple) else str(c).lower()
                      for c in df.columns]
        return df.dropna(subset=["open","high","low","close"])
    except Exception as e:
        log.warning(f"fetch {symbol}/{interval}: {e}")
        return None

# ── Indicatori de baza ────────────────────────────────────────────────────────
def ema(s, n):
    return s.ewm(span=n, adjust=False).mean()

def rsi_series(s, n=14):
    d = s.diff()
    g = d.clip(lower=0).rolling(n).mean()
    l = (-d.clip(upper=0)).rolling(n).mean()
    return 100 - 100 / (1 + g / l.replace(0, np.nan))

def atr_val(df, n=14):
    h, l, cp = df["high"], df["low"], df["close"].shift(1)
    tr = pd.concat([h-l, (h-cp).abs(), (l-cp).abs()], axis=1).max(axis=1)
    return float(tr.rolling(n).mean().iloc[-1])

def find_pivots(highs, lows, lb=4):
    ph, pl = [], []
    n = len(highs)
    for i in range(lb, n - lb):
        if all(highs[i] >= highs[i-j] for j in range(1, lb+1)) and \
           all(highs[i] >= highs[i+j] for j in range(1, lb+1)):
            ph.append((i, highs[i]))
        if all(lows[i] <= lows[i-j] for j in range(1, lb+1)) and \
           all(lows[i] <= lows[i+j] for j in range(1, lb+1)):
            pl.append((i, lows[i]))
    return ph, pl

# ════════════════════════════════════════════════════════════════════════════════
# CELE 12 METODE
# Fiecare primeste df_m5 si df_h1, returneaza 'BUY' / 'SELL' / 'NEUTRAL'
# ════════════════════════════════════════════════════════════════════════════════

# 1. Trend Structure — HH+HL = BUY, LH+LL = SELL
def m_trend_structure(df_m5, df_h1):
    for df in [df_h1, df_m5]:
        if df is None or len(df) < 20: continue
        ph, pl = find_pivots(df["high"].values, df["low"].values, lb=4)
        if len(ph) >= 2 and len(pl) >= 2:
            hh = ph[-1][1] > ph[-2][1]
            hl = pl[-1][1] > pl[-2][1]
            lh = ph[-1][1] < ph[-2][1]
            ll = pl[-1][1] < pl[-2][1]
            if hh and hl: return "BUY"
            if lh and ll: return "SELL"
    return "NEUTRAL"

# 2. EMA 20/50 crossover
def m_ema_cross(df_m5, df_h1):
    df = df_h1 if df_h1 is not None and len(df_h1) >= 55 else df_m5
    if df is None or len(df) < 55: return "NEUTRAL"
    e20 = float(ema(df["close"], 20).iloc[-1])
    e50 = float(ema(df["close"], 50).iloc[-1])
    if e20 > e50: return "BUY"
    if e20 < e50: return "SELL"
    return "NEUTRAL"

# 3. MACD crossover (12,26,9)
def m_macd(df_m5, df_h1):
    df = df_m5 if df_m5 is not None and len(df_m5) >= 35 else df_h1
    if df is None or len(df) < 35: return "NEUTRAL"
    macd_line   = ema(df["close"], 12) - ema(df["close"], 26)
    signal_line = ema(macd_line, 9)
    prev_diff   = float(macd_line.iloc[-2]) - float(signal_line.iloc[-2])
    curr_diff   = float(macd_line.iloc[-1]) - float(signal_line.iloc[-1])
    if prev_diff < 0 and curr_diff > 0: return "BUY"
    if prev_diff > 0 and curr_diff < 0: return "SELL"
    # Directie curenta daca nu e crossover recent
    if curr_diff > 0: return "BUY"
    if curr_diff < 0: return "SELL"
    return "NEUTRAL"

# 4. RSI extremes (<35 = BUY, >65 = SELL)
def m_rsi(df_m5, df_h1):
    df = df_m5 if df_m5 is not None and len(df_m5) >= 20 else df_h1
    if df is None or len(df) < 20: return "NEUTRAL"
    r = float(rsi_series(df["close"]).iloc[-1])
    if r < 35: return "BUY"
    if r > 65: return "SELL"
    return "NEUTRAL"

# 5. Bollinger Bands (20, 2σ)
def m_bollinger(df_m5, df_h1):
    df = df_m5 if df_m5 is not None and len(df_m5) >= 25 else df_h1
    if df is None or len(df) < 25: return "NEUTRAL"
    mid  = df["close"].rolling(20).mean()
    std  = df["close"].rolling(20).std()
    upper = float((mid + 2 * std).iloc[-1])
    lower = float((mid - 2 * std).iloc[-1])
    price = float(df["close"].iloc[-1])
    if price < lower: return "BUY"
    if price > upper: return "SELL"
    return "NEUTRAL"

# 6. Pin Bar / Hammer (coada lunga)
def m_pin_bar(df_m5, df_h1):
    df = df_m5 if df_m5 is not None and len(df_m5) >= 5 else df_h1
    if df is None or len(df) < 5: return "NEUTRAL"
    bar  = df.iloc[-1]
    o, h, l, c = float(bar["open"]), float(bar["high"]), float(bar["low"]), float(bar["close"])
    rng  = h - l
    if rng == 0: return "NEUTRAL"
    body      = abs(c - o)
    lower_wick = min(o, c) - l
    upper_wick = h - max(o, c)
    # Hammer: coada jos >= 2x corp, corp in treimea superioara
    if lower_wick >= 2 * body and lower_wick / rng >= 0.55: return "BUY"
    # Shooting star: coada sus >= 2x corp, corp in treimea inferioara
    if upper_wick >= 2 * body and upper_wick / rng >= 0.55: return "SELL"
    return "NEUTRAL"

# 7. Engulfing candle
def m_engulfing(df_m5, df_h1):
    df = df_m5 if df_m5 is not None and len(df_m5) >= 5 else df_h1
    if df is None or len(df) < 3: return "NEUTRAL"
    prev = df.iloc[-2]
    curr = df.iloc[-1]
    po, pc = float(prev["open"]), float(prev["close"])
    co, cc = float(curr["open"]), float(curr["close"])
    # Bullish engulfing: bara anterioara bearish, bara curenta bullish si o inghite
    if pc < po and cc > co and co < pc and cc > po: return "BUY"
    # Bearish engulfing
    if pc > po and cc < co and co > pc and cc < po: return "SELL"
    return "NEUTRAL"

# 8. Inside Bar breakout
def m_inside_bar(df_m5, df_h1):
    df = df_m5 if df_m5 is not None and len(df_m5) >= 5 else df_h1
    if df is None or len(df) < 4: return "NEUTRAL"
    mother = df.iloc[-3]
    inside = df.iloc[-2]
    curr   = df.iloc[-1]
    mh, ml = float(mother["high"]), float(mother["low"])
    ih, il = float(inside["high"]), float(inside["low"])
    cc     = float(curr["close"])
    # Inside bar: bara 2 este inchisa in interiorul barei 1
    if ih <= mh and il >= ml:
        if cc > mh: return "BUY"   # breakout sus
        if cc < ml: return "SELL"  # breakout jos
    return "NEUTRAL"

# 9. Fair Value Gap (FVG)
def m_fvg(df_m5, df_h1):
    df = df_m5 if df_m5 is not None and len(df_m5) >= 10 else df_h1
    if df is None or len(df) < 5: return "NEUTRAL"
    # Cauta FVG in ultimele 10 bare
    for i in range(-10, -2):
        b1 = df.iloc[i]
        b2 = df.iloc[i+1]
        b3 = df.iloc[i+2]
        price = float(df["close"].iloc[-1])
        # Bullish FVG: gap intre high[i] si low[i+2]
        if float(b3["low"]) > float(b1["high"]):
            gap_low  = float(b1["high"])
            gap_high = float(b3["low"])
            if gap_low <= price <= gap_high:
                return "BUY"
        # Bearish FVG: gap intre low[i] si high[i+2]
        if float(b3["high"]) < float(b1["low"]):
            gap_high = float(b1["low"])
            gap_low  = float(b3["high"])
            if gap_low <= price <= gap_high:
                return "SELL"
    return "NEUTRAL"

# 10. Order Block
def m_order_block(df_m5, df_h1):
    df = df_h1 if df_h1 is not None and len(df_h1) >= 20 else df_m5
    if df is None or len(df) < 15: return "NEUTRAL"
    price = float(df["close"].iloc[-1])
    # Cauta ultimul order block bearish/bullish in ultimele 20 bare
    for i in range(-20, -3):
        bar  = df.iloc[i]
        next_bars = df.iloc[i+1:i+4]
        bo, bc = float(bar["open"]), float(bar["close"])
        # Bullish OB: bara bearish urmata de miscare bullish puternica
        if bc < bo:  # bara bearish
            if float(next_bars["close"].max()) > bo * 1.001:  # miscare bullish
                ob_high = bo
                ob_low  = bc
                if ob_low <= price <= ob_high:
                    return "BUY"
        # Bearish OB: bara bullish urmata de miscare bearish puternica
        if bc > bo:  # bara bullish
            if float(next_bars["close"].min()) < bo * 0.999:
                ob_high = bc
                ob_low  = bo
                if ob_low <= price <= ob_high:
                    return "SELL"
    return "NEUTRAL"

# 11. Breakout din range (consolidare)
def m_breakout(df_m5, df_h1):
    df = df_m5 if df_m5 is not None and len(df_m5) >= 25 else df_h1
    if df is None or len(df) < 25: return "NEUTRAL"
    # Range = ultimele 20 bare (exclude ultima)
    window  = df.iloc[-21:-1]
    rng_hi  = float(window["high"].max())
    rng_lo  = float(window["low"].min())
    rng_sz  = rng_hi - rng_lo
    price   = float(df["close"].iloc[-1])
    # Breakout valid daca range-ul e suficient de strans (ATR-based)
    atr     = atr_val(df)
    if rng_sz == 0 or atr == 0: return "NEUTRAL"
    if rng_sz > atr * 3: return "NEUTRAL"  # prea larg, nu e consolidare
    last_bar_body = abs(float(df.iloc[-1]["close"]) - float(df.iloc[-1]["open"]))
    if price > rng_hi and last_bar_body > atr * 0.5: return "BUY"
    if price < rng_lo and last_bar_body > atr * 0.5: return "SELL"
    return "NEUTRAL"

# 12. Fibonacci 61.8% retracement
def m_fibonacci(df_m5, df_h1):
    df = df_h1 if df_h1 is not None and len(df_h1) >= 30 else df_m5
    if df is None or len(df) < 30: return "NEUTRAL"
    ph, pl = find_pivots(df["high"].values, df["low"].values, lb=5)
    if len(ph) < 2 or len(pl) < 2: return "NEUTRAL"
    price = float(df["close"].iloc[-1])
    # Uptrend: ultimul swing low → swing high, pret la 61.8% retracere → BUY
    if pl[-1][0] < ph[-1][0]:  # swing low inainte de swing high = uptrend
        swing_lo = pl[-1][1]
        swing_hi = ph[-1][1]
        fib618   = swing_hi - (swing_hi - swing_lo) * 0.618
        fib50    = swing_hi - (swing_hi - swing_lo) * 0.50
        if fib618 <= price <= fib50: return "BUY"
    # Downtrend: ultimul swing high → swing low, pret la 61.8% retracere → SELL
    if ph[-1][0] < pl[-1][0]:
        swing_hi = ph[-1][1]
        swing_lo = pl[-1][1]
        fib618   = swing_lo + (swing_hi - swing_lo) * 0.618
        fib50    = swing_lo + (swing_hi - swing_lo) * 0.50
        if fib50 <= price <= fib618: return "SELL"
    return "NEUTRAL"

# ── Lista completa metode ─────────────────────────────────────────────────────
METHODS = [
    ("TrendStructure",  m_trend_structure),
    ("EMACross",        m_ema_cross),
    ("MACD",            m_macd),
    ("RSI",             m_rsi),
    ("BollingerBands",  m_bollinger),
    ("PinBar",          m_pin_bar),
    ("Engulfing",       m_engulfing),
    ("InsideBar",       m_inside_bar),
    ("FVG",             m_fvg),
    ("OrderBlock",      m_order_block),
    ("Breakout",        m_breakout),
    ("Fibonacci",       m_fibonacci),
]

# ── ATR pentru SL/TP ──────────────────────────────────────────────────────────
def calc_levels(df_h1, direction, price):
    if df_h1 is None or len(df_h1) < 20 or direction == "HOLD":
        return 0.0, 0.0
    a = atr_val(df_h1)
    ph, pl = find_pivots(df_h1["high"].values, df_h1["low"].values, lb=4)
    buf = a * 0.1
    if direction == "BUY":
        lows_below = [p for _, p in pl if p < price]
        sl  = round(lows_below[-1] - buf, 5) if lows_below else round(price - a * 1.5, 5)
        highs_above = [p for _, p in ph if p > price]
        tp  = round(highs_above[0], 5) if highs_above else round(price + a * 2.0, 5)
        sl_d = abs(price - sl)
        if sl_d > 0 and abs(price - tp) / sl_d < 1.5:
            tp = round(price + sl_d * 2.0, 5)
    else:
        highs_above = [p for _, p in ph if p > price]
        sl  = round(highs_above[-1] + buf, 5) if highs_above else round(price + a * 1.5, 5)
        lows_below = [p for _, p in pl if p < price]
        tp  = round(lows_below[-1], 5) if lows_below else round(price - a * 2.0, 5)
        sl_d = abs(price - sl)
        if sl_d > 0 and abs(price - tp) / sl_d < 1.5:
            tp = round(price - sl_d * 2.0, 5)
    return sl, tp

# ── Analiza simbol ────────────────────────────────────────────────────────────
def analyze(symbol):
    df_m5 = fetch(symbol, "5m",  "2d")
    df_h1 = fetch(symbol, "1h", "10d")

    votes_buy  = []
    votes_sell = []
    votes_all  = {}

    for name, fn in METHODS:
        try:
            result = fn(df_m5, df_h1)
        except Exception as e:
            result = "NEUTRAL"
            log.warning(f"{symbol} {name}: {e}")
        votes_all[name] = result
        if result == "BUY":
            votes_buy.append(name)
        elif result == "SELL":
            votes_sell.append(name)

    n_buy  = len(votes_buy)
    n_sell = len(votes_sell)
    total  = len(METHODS)

    if n_buy >= MIN_VOTES and n_buy > n_sell:
        direction  = "BUY"
        confidence = round(n_buy / total * 100, 1)
    elif n_sell >= MIN_VOTES and n_sell > n_buy:
        direction  = "SELL"
        confidence = round(n_sell / total * 100, 1)
    else:
        direction  = "HOLD"
        confidence = round(max(n_buy, n_sell) / total * 100, 1)

    price = float(df_h1["close"].iloc[-1]) if df_h1 is not None and len(df_h1) > 0 else 0.0
    sl, tp = calc_levels(df_h1, direction, price)

    log.info(f"{symbol} → {direction} | BUY={n_buy} SELL={n_sell} | conf={confidence}%")

    result = {
        "symbol":     symbol,
        "direction":  direction,
        "confidence": confidence,
        "votes_buy":  n_buy,
        "votes_sell": n_sell,
        "votes":      votes_all,
        "price":      round(price, 5),
        "sl":         sl,
        "tp":         tp,
        "timestamp":  datetime.now(timezone.utc).isoformat(),
    }

    push_log({
        "time":      datetime.now().strftime("%H:%M:%S"),
        "symbol":    symbol,
        "direction": direction,
        "buy":       n_buy,
        "sell":      n_sell,
        "conf":      confidence,
    })
    return result

# ── Routes ────────────────────────────────────────────────────────────────────
@app.route("/signal")
def signal():
    symbol = request.args.get("symbol", "EURUSD").upper()
    if symbol not in SYMBOLS:
        return jresp({"error": f"{symbol} not in list"}), 400
    return jresp(analyze(symbol))

@app.route("/health")
def health():
    return jresp({"status": "ok", "port": PORT})

@app.route("/signals")
def dashboard():
    with _log_lock:
        rows = list(reversed(_signal_log))

    def color(d):
        return {"BUY": "#1a7a1a", "SELL": "#8b0000", "HOLD": "#555"}.get(d, "#555")

    def bg(d):
        return {"BUY": "#0d2b0d", "SELL": "#2b0d0d", "HOLD": "#1a1a1a"}.get(d, "#1a1a1a")

    rows_html = "".join(f"""
    <tr style="background:{bg(r['direction'])}">
      <td>{r['time']}</td><td>{r['symbol']}</td>
      <td style="color:{color(r['direction'])};font-weight:bold">{r['direction']}</td>
      <td>{r['conf']}%</td>
      <td style="color:#1a7a1a">{r['buy']}</td>
      <td style="color:#8b0000">{r['sell']}</td>
    </tr>""" for r in rows)

    return f"""<!DOCTYPE html><html><head>
<meta charset="utf-8"><title>VoteTrader</title>
<meta http-equiv="refresh" content="15">
<style>body{{background:#111;color:#eee;font-family:monospace;padding:20px}}
h2{{color:#aaa}}table{{border-collapse:collapse;width:100%}}
th{{background:#222;padding:8px 12px;text-align:left;color:#888;border-bottom:1px solid #333}}
td{{padding:7px 12px;border-bottom:1px solid #222}}</style></head><body>
<h2>VoteTrader — {len(METHODS)} metode, min {MIN_VOTES} voturi (auto-refresh 15s)</h2>
<table><tr><th>Time</th><th>Symbol</th><th>Direction</th><th>Conf</th><th>BUY votes</th><th>SELL votes</th></tr>
{rows_html}</table></body></html>"""

# ── Main ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    log.info(f"VoteTrader server pornit pe {HOST}:{PORT} — {len(METHODS)} metode, min {MIN_VOTES} voturi")
    app.run(host=HOST, port=PORT, debug=False)
