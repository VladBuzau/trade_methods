"""
TrendDetector Server
Detecteaza noi trenduri ascendente/descendente pe M1, M5, M30, H1
bazat pe structura de piata (Higher Highs/Higher Lows sau Lower Highs/Lower Lows)
"""

import json, logging, threading, time
from datetime import datetime, timezone
from flask import Flask, jsonify, request, render_template_string, Response
import yfinance as yf
import pandas as pd
import numpy as np

# Rezolva numpy int64/float64 care nu sunt JSON serializable
class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, (np.integer,)): return int(obj)
        if isinstance(obj, (np.floating,)): return float(obj)
        if isinstance(obj, np.ndarray): return obj.tolist()
        return super().default(obj)

# ── Config ──────────────────────────────────────────────────────────────────
with open("config.json") as f:
    CFG = json.load(f)

HOST = CFG.get("host", "localhost")
PORT = CFG.get("port", 5002)
MIN_CONFIDENCE   = CFG.get("min_confidence", 60)
MIN_STRUCT_SCORE = CFG.get("min_structure_score", 3)
SYMBOLS          = CFG.get("symbols", ["EURUSD", "GBPUSD"])

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

app = Flask(__name__)

def np_jsonify(data):
    return Response(json.dumps(data, cls=NpEncoder, separators=(',', ':')), mimetype="application/json")

# ── Signal log ───────────────────────────────────────────────────────────────
_signal_log = []
MAX_LOG = 300
_log_lock = threading.Lock()

def push_log(entry: dict):
    with _log_lock:
        _signal_log.append(entry)
        if len(_signal_log) > MAX_LOG:
            _signal_log.pop(0)

# ── Yahoo Finance symbol map ──────────────────────────────────────────────────
YF_MAP = {
    "EURUSD": "EURUSD=X", "GBPUSD": "GBPUSD=X", "USDJPY": "JPY=X",
    "USDCHF": "CHF=X",    "AUDUSD": "AUDUSD=X", "USDCAD": "CAD=X",
    "NZDUSD": "NZDUSD=X", "EURGBP": "EURGBP=X", "EURJPY": "EURJPY=X",
    "GBPJPY": "GBPJPY=X", "AUDJPY": "AUDJPY=X", "EURAUD": "EURAUD=X",
    "GBPAUD": "GBPAUD=X", "XAUUSD": "GC=F",     "XAGUSD": "SI=F",
}

TF_CFG = {
    # tf_name: (yf_interval, yf_period, pivot_lookback)
    "M1":  ("1m",  "1d",  2),
    "M5":  ("5m",  "2d",  3),
    "M30": ("30m", "5d",  4),
    "H1":  ("1h",  "10d", 5),
}

# ── Data fetch ────────────────────────────────────────────────────────────────
def fetch_ohlcv(symbol: str, interval: str, period: str):
    yf_sym = YF_MAP.get(symbol, symbol + "=X")
    try:
        df = yf.download(yf_sym, interval=interval, period=period,
                         progress=False, auto_adjust=True)
        if df is None or len(df) < 30:
            log.warning(f"fetch {symbol}/{interval}: prea putine date ({len(df) if df is not None else 0})")
            return None
        # yfinance nou returneaza coloane ca tupluri ('Close', 'EURUSD=X')
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        df.columns = [c[0].lower() if isinstance(c, tuple) else str(c).lower()
                      for c in df.columns]
        df = df.dropna(subset=["open", "high", "low", "close"])
        log.info(f"fetch {symbol}/{interval}: {len(df)} bare OK")
        return df
    except Exception as e:
        log.warning(f"fetch {symbol}/{interval}: {e}")
        return None

# ── Pivot detection ───────────────────────────────────────────────────────────
def find_pivots(df: pd.DataFrame, lookback: int = 5):
    """
    Returneaza lista de pivot highs si pivot lows.
    Un pivot high = bara i are high mai mare decat lookback bare de ambele parti.
    """
    highs = df["high"].values
    lows  = df["low"].values
    n = len(highs)
    pivot_highs = []  # (index, price)
    pivot_lows  = []

    for i in range(lookback, n - lookback):
        if all(highs[i] >= highs[i-j] for j in range(1, lookback+1)) and \
           all(highs[i] >= highs[i+j] for j in range(1, lookback+1)):
            pivot_highs.append((i, highs[i]))
        if all(lows[i] <= lows[i-j] for j in range(1, lookback+1)) and \
           all(lows[i] <= lows[i+j] for j in range(1, lookback+1)):
            pivot_lows.append((i, lows[i]))

    return pivot_highs, pivot_lows

# ── Trend structure detection ─────────────────────────────────────────────────
def detect_trend_structure(pivot_highs, pivot_lows, min_pivots: int = 2):
    """
    Detecteaza:
      - ASCENDING:  HH (Higher High) + HL (Higher Low) — cel putin min_pivots consecutive
      - DESCENDING: LH (Lower High)  + LL (Lower Low)  — cel putin min_pivots consecutive
      - RANGING:    nu e clar
    Returneaza ('ASCENDING'|'DESCENDING'|'RANGING', score 0-5, detalii)
    """
    score_bull = 0
    score_bear = 0
    notes = []

    # Analiza pivot highs
    if len(pivot_highs) >= 2:
        ph_prices = [p for _, p in pivot_highs[-4:]]  # ultimele 4
        hh_count = sum(ph_prices[i] > ph_prices[i-1] for i in range(1, len(ph_prices)))
        lh_count = sum(ph_prices[i] < ph_prices[i-1] for i in range(1, len(ph_prices)))
        if hh_count >= min_pivots:
            score_bull += hh_count
            notes.append(f"HH x{hh_count}")
        if lh_count >= min_pivots:
            score_bear += lh_count
            notes.append(f"LH x{lh_count}")

    # Analiza pivot lows
    if len(pivot_lows) >= 2:
        pl_prices = [p for _, p in pivot_lows[-4:]]
        hl_count = sum(pl_prices[i] > pl_prices[i-1] for i in range(1, len(pl_prices)))
        ll_count = sum(pl_prices[i] < pl_prices[i-1] for i in range(1, len(pl_prices)))
        if hl_count >= min_pivots:
            score_bull += hl_count
            notes.append(f"HL x{hl_count}")
        if ll_count >= min_pivots:
            score_bear += ll_count
            notes.append(f"LL x{ll_count}")

    if score_bull > score_bear and score_bull >= MIN_STRUCT_SCORE:
        return "ASCENDING", score_bull, notes
    elif score_bear > score_bull and score_bear >= MIN_STRUCT_SCORE:
        return "DESCENDING", score_bear, notes
    else:
        return "RANGING", max(score_bull, score_bear), notes

# ── New trend break detection ─────────────────────────────────────────────────
def detect_new_trend_break(df: pd.DataFrame, pivot_highs, pivot_lows):
    """
    Detecteaza daca s-a rupt recent structura veche si inceput un nou trend:
    - Bullish break: ultimul close > cel mai inalt pivot high din ultimele 10 bare
    - Bearish break: ultimul close < cel mai mic pivot low din ultimele 10 bare
    """
    if df is None or len(df) < 15:
        return None

    last_close = float(df["close"].iloc[-1])
    recent_bars = df.iloc[-15:-1]

    recent_high = float(recent_bars["high"].max())
    recent_low  = float(recent_bars["low"].min())

    # O bara mare de breakout (corp >= 60% din range)
    last_bar = df.iloc[-1]
    body = abs(float(last_bar["close"]) - float(last_bar["open"]))
    rng  = float(last_bar["high"]) - float(last_bar["low"])
    strong_candle = (rng > 0 and body / rng >= 0.55)

    if last_close > recent_high and strong_candle:
        return "BULLISH_BREAK"
    if last_close < recent_low and strong_candle:
        return "BEARISH_BREAK"
    return None

# ── EMA helper ────────────────────────────────────────────────────────────────
def ema(series: pd.Series, period: int) -> float:
    if len(series) < period:
        return float("nan")
    return float(series.ewm(span=period, adjust=False).mean().iloc[-1])

def get_ema_bias(df: pd.DataFrame) -> str:
    """EMA20 vs EMA50 bias"""
    if df is None or len(df) < 55:
        return "NEUTRAL"
    e20 = ema(df["close"], 20)
    e50 = ema(df["close"], 50)
    if e20 > e50:
        return "BULL"
    elif e20 < e50:
        return "BEAR"
    return "NEUTRAL"

# ── RSI ───────────────────────────────────────────────────────────────────────
def rsi(series: pd.Series, period: int = 14) -> float:
    delta = series.diff()
    gain = delta.clip(lower=0).rolling(period).mean()
    loss = (-delta.clip(upper=0)).rolling(period).mean()
    rs = gain / loss.replace(0, np.nan)
    r = 100 - 100 / (1 + rs)
    return float(r.iloc[-1]) if not r.empty else 50.0

# ── ATR (for SL/TP) ───────────────────────────────────────────────────────────
def atr(df: pd.DataFrame, period: int = 14) -> float:
    high = df["high"]
    low  = df["low"]
    close_prev = df["close"].shift(1)
    tr = pd.concat([high - low,
                    (high - close_prev).abs(),
                    (low  - close_prev).abs()], axis=1).max(axis=1)
    return float(tr.rolling(period).mean().iloc[-1])

# ── Per-timeframe analysis ────────────────────────────────────────────────────
def analyze_tf(symbol: str, tf: str) -> dict:
    interval, period, lookback = TF_CFG[tf]
    df = fetch_ohlcv(symbol, interval, period)
    if df is None:
        return {"tf": tf, "trend": "UNKNOWN", "score": 0, "notes": []}

    pivot_highs, pivot_lows = find_pivots(df, lookback)
    trend, score, notes = detect_trend_structure(pivot_highs, pivot_lows)
    new_break = detect_new_trend_break(df, pivot_highs, pivot_lows)
    ema_bias  = get_ema_bias(df)
    rsi_val   = rsi(df["close"])
    atr_val   = atr(df)
    last_close = float(df["close"].iloc[-1])

    return {
        "tf":        tf,
        "trend":     trend,
        "score":     score,
        "notes":     notes,
        "new_break": new_break,
        "ema_bias":  ema_bias,
        "rsi":       round(rsi_val, 1),
        "atr":       round(atr_val, 6),
        "last_close": round(last_close, 5),
    }

# ── SL/TP bazat pe pivoti (logica trader profesionist) ────────────────────────
def calc_pivot_levels(symbol: str, direction: str, price: float, atr_val: float):
    """
    BUY:
      SL  = ultimul pivot low H1 sub price (cu buffer 10% ATR)
      TP1 = urmatorul pivot high H1 deasupra price
      TP2 = al doilea pivot high sau 2x distanta TP1

    SELL:
      SL  = ultimul pivot high H1 deasupra price
      TP1 = urmatorul pivot low H1 sub price
      TP2 = al doilea pivot low sau 2x distanta TP1

    Fallback: ATR daca nu exista pivoti suficienti.
    """
    if direction == "HOLD" or price == 0:
        return 0.0, 0.0, 0.0

    df = fetch_ohlcv(symbol, "1h", "10d")
    if df is None or len(df) < 30:
        # fallback ATR
        sl_d  = atr_val * 1.5
        tp1_d = atr_val * 2.0
        tp2_d = atr_val * 3.5
        if direction == "BUY":
            return round(price - sl_d, 5), round(price + tp1_d, 5), round(price + tp2_d, 5)
        else:
            return round(price + sl_d, 5), round(price - tp1_d, 5), round(price - tp2_d, 5)

    pivot_highs, pivot_lows = find_pivots(df, lookback=4)
    buf = atr_val * 0.1  # buffer mic sub/deasupra pivotului

    if direction == "BUY":
        # SL = cel mai recent pivot low SUB price
        lows_below = [(i, p) for i, p in pivot_lows if p < price]
        sl = round(lows_below[-1][1] - buf, 5) if lows_below else round(price - atr_val * 1.5, 5)

        # TP1 = primul pivot high DEASUPRA price
        highs_above = [(i, p) for i, p in pivot_highs if p > price]
        if highs_above:
            tp1 = round(highs_above[0][1], 5)
            tp2 = round(highs_above[1][1], 5) if len(highs_above) > 1 else round(price + (tp1 - price) * 2, 5)
        else:
            tp1 = round(price + atr_val * 2.0, 5)
            tp2 = round(price + atr_val * 3.5, 5)

    else:  # SELL
        # SL = cel mai recent pivot high DEASUPRA price
        highs_above = [(i, p) for i, p in pivot_highs if p > price]
        sl = round(highs_above[-1][1] + buf, 5) if highs_above else round(price + atr_val * 1.5, 5)

        # TP1 = primul pivot low SUB price
        lows_below = [(i, p) for i, p in pivot_lows if p < price]
        if lows_below:
            tp1 = round(lows_below[-1][1], 5)
            tp2 = round(lows_below[-2][1], 5) if len(lows_below) > 1 else round(price - (price - tp1) * 2, 5)
        else:
            tp1 = round(price - atr_val * 2.0, 5)
            tp2 = round(price - atr_val * 3.5, 5)

    # Verifica R:R minim 1.5
    sl_dist  = abs(price - sl)
    tp1_dist = abs(price - tp1)
    if sl_dist > 0 and tp1_dist / sl_dist < 1.5:
        # TP1 prea aproape — extinde la minim 1.5R
        if direction == "BUY":
            tp1 = round(price + sl_dist * 1.5, 5)
            tp2 = round(price + sl_dist * 2.5, 5)
        else:
            tp1 = round(price - sl_dist * 1.5, 5)
            tp2 = round(price - sl_dist * 2.5, 5)

    return sl, tp1, tp2

# ── Multi-TF confluence ───────────────────────────────────────────────────────
def analyze_symbol(symbol: str) -> dict:
    tf_results = {}
    for tf in ["M1", "M5", "M30", "H1"]:
        tf_results[tf] = analyze_tf(symbol, tf)

    # Count ascending / descending per TF (weighted)
    weights = {"M1": 1, "M5": 2, "M30": 3, "H1": 4}
    bull_total = 0
    bear_total = 0
    max_total  = 0

    for tf, w in weights.items():
        res = tf_results[tf]
        t = res["trend"]
        s = min(res["score"], 4)  # cap at 4
        if t == "ASCENDING":
            bull_total += w * s
        elif t == "DESCENDING":
            bear_total += w * s
        max_total += w * 4

        # New break bonus
        nb = res.get("new_break")
        if nb == "BULLISH_BREAK":
            bull_total += w * 2
        elif nb == "BEARISH_BREAK":
            bear_total += w * 2

        # EMA bias bonus
        eb = res.get("ema_bias")
        if eb == "BULL":
            bull_total += w
        elif eb == "BEAR":
            bear_total += w
        max_total += w * 3  # break + ema bonus

    bull_total = int(bull_total)
    bear_total = int(bear_total)
    max_total  = int(max_total)

    if max_total == 0:
        confidence = 0.0
        direction  = "HOLD"
    elif bull_total > bear_total:
        confidence = round(bull_total / max_total * 100, 1)
        direction  = "BUY" if confidence >= MIN_CONFIDENCE else "HOLD"
    elif bear_total > bull_total:
        confidence = round(bear_total / max_total * 100, 1)
        direction  = "SELL" if confidence >= MIN_CONFIDENCE else "HOLD"
    else:
        confidence = 0.0
        direction  = "HOLD"

    log.info(f"{symbol} dir={direction} conf={confidence}% bull={bull_total} bear={bear_total} max={max_total} min_conf={MIN_CONFIDENCE}")

    # SL/TP bazat pe pivoti H1 (ca un trader profesionist)
    h1_atr   = tf_results["H1"].get("atr", 0) or 0
    h1_close = tf_results["H1"].get("last_close", 0) or 0
    sl, tp1, tp2 = calc_pivot_levels(symbol, direction, h1_close, h1_atr)

    result = {
        "symbol":     symbol,
        "direction":  direction,
        "confidence": confidence,
        "bull_score": bull_total,
        "bear_score": bear_total,
        "sl":         sl,
        "tp1":        tp1,
        "tp2":        tp2,
        "price":      h1_close,
        "tf_detail":  tf_results,
        "timestamp":  datetime.now(timezone.utc).isoformat(),
    }

    push_log({
        "time":       datetime.now().strftime("%H:%M:%S"),
        "symbol":     symbol,
        "direction":  direction,
        "confidence": confidence,
        "bull":       bull_total,
        "bear":       bear_total,
    })

    return result

# ── Flask routes ──────────────────────────────────────────────────────────────
@app.route("/signal", methods=["GET"])
def signal():
    symbol = request.args.get("symbol", "EURUSD").upper()
    if symbol not in SYMBOLS:
        return jsonify({"error": f"Symbol {symbol} not in list"}), 400
    data = analyze_symbol(symbol)
    log.info(f"{symbol} → {data['direction']} {data['confidence']}%")
    return np_jsonify(data)

@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok", "port": PORT})

@app.route("/signals", methods=["GET"])
def signals_dashboard():
    with _log_lock:
        rows = list(reversed(_signal_log))

    def color(d):
        return {"BUY": "#1a7a1a", "SELL": "#8b0000", "HOLD": "#333"}.get(d, "#333")

    rows_html = ""
    for r in rows:
        bg = {"BUY": "#0d3b0d", "SELL": "#3b0d0d", "HOLD": "#1a1a1a"}.get(r["direction"], "#1a1a1a")
        rows_html += f"""
        <tr style="background:{bg}">
          <td>{r['time']}</td>
          <td>{r['symbol']}</td>
          <td style="color:{color(r['direction'])};font-weight:bold">{r['direction']}</td>
          <td>{r['confidence']}%</td>
          <td>{r['bull']}</td>
          <td>{r['bear']}</td>
        </tr>"""

    html = f"""<!DOCTYPE html>
<html><head>
<meta charset="utf-8">
<title>TrendDetector Signals</title>
<meta http-equiv="refresh" content="15">
<style>
  body{{background:#111;color:#eee;font-family:monospace;padding:20px}}
  h2{{color:#aaa}}
  table{{border-collapse:collapse;width:100%}}
  th{{background:#222;padding:8px 12px;text-align:left;color:#888;border-bottom:1px solid #333}}
  td{{padding:7px 12px;border-bottom:1px solid #222}}
</style>
</head><body>
<h2>TrendDetector — Live Signals (auto-refresh 15s)</h2>
<table>
  <tr><th>Time</th><th>Symbol</th><th>Direction</th><th>Confidence</th><th>Bull</th><th>Bear</th></tr>
  {rows_html}
</table>
</body></html>"""
    return html

# ── Main ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    log.info(f"TrendDetector server pornit pe {HOST}:{PORT}")
    app.run(host=HOST, port=PORT, debug=False)
