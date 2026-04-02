"""
ChartVisualizer — grafic + auto-trader integrat
"""

from flask import Flask, render_template_string, request, Response, send_file
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import json, logging, os

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

# ── MetaTrader5 ───────────────────────────────────────────────────────────────
try:
    import MetaTrader5 as mt5
    MT5_AVAILABLE = mt5.initialize()
    log.info("MT5 conectat" if MT5_AVAILABLE else "MT5 pornit dar nu conectat")
except ImportError:
    mt5 = None
    MT5_AVAILABLE = False
    log.info("MT5 nu e instalat")

app = Flask(__name__)

# AutoTrader blueprint
try:
    from autotrader import autotrader_bp
    app.register_blueprint(autotrader_bp)
    log.info("AutoTrader blueprint inregistrat")
except Exception as e:
    log.warning(f"AutoTrader blueprint nu s-a putut incarca: {e}")

MT5_TF = {
    "M1": 1, "M5": 5, "M15": 15,
    "M30": 30, "H1": 16385, "H4": 16388, "D1": 16408,
}
ALL_TFS    = ["M1", "M5", "M15", "M30", "H1", "H4", "D1"]
MULTI_BARS = {t: 500 for t in ALL_TFS}

SYMBOLS = [
    "EURUSD","GBPUSD","USDJPY","AUDUSD","USDCAD",
    "EURGBP","EURJPY","GBPJPY","AUDJPY","XAUUSD",
    "USDCHF",
]

RISK_DOLLARS    = 50.0
TRADE_MAGIC     = 202800
MIN_TF_VOTES    = 2         # voturi minime absolute
MIN_CONFIDENCE  = 60.0      # % minim confidence pentru a intra (ex: 3/4 = 75%, 3/5 = 60%)
MAX_OPEN_TRADES = 5

# Sesiuni active (UTC) — in afara acestor ferestre botul nu deschide trades
TRADING_SESSIONS = [
    (7, 0, 12, 0),   # London
    (13, 0, 17, 0),  # New York / Overlap
]

ADX_MIN = 25  # forta minima a trendului pentru a intra

# ── Reguli FTMO ───────────────────────────────────────────────────────────────
FTMO_DAILY_LOSS_PCT   = 0.05   # 5% din balanta initiala
FTMO_MAX_LOSS_PCT     = 0.10   # 10% drawdown total
FTMO_NEWS_BLOCK_MIN   = 2      # minute blocate inainte/dupa stiri majore
FTMO_ENABLED          = True   # activeaza verificarile FTMO

# Cache stiri ForexFactory
_news_cache = {"events": [], "fetched_at": None}
_news_lock  = __import__("threading").Lock()

def fetch_red_news():
    """Descarca stirile rosii (High impact) din ForexFactory XML. Cache 30 min."""
    import urllib.request, xml.etree.ElementTree as ET
    from datetime import datetime, timezone, timedelta

    with _news_lock:
        now = datetime.now(timezone.utc)
        if (_news_cache["fetched_at"] and
                (now - _news_cache["fetched_at"]).total_seconds() < 14400):  # 4 ore
            return _news_cache["events"]

    try:
        urls = [
            "https://nfs.faireconomy.media/ff_calendar_thisweek.xml",
            "https://cdn-nfs.faireconomy.media/ff_calendar_thisweek.xml",
        ]
        data = None
        for url in urls:
            try:
                req = urllib.request.Request(url, headers={
                    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
                })
                with urllib.request.urlopen(req, timeout=10) as resp:
                    data = resp.read()
                break
            except:
                continue
        if not data:
            return _news_cache["events"]
        root = ET.fromstring(data)
        events = []
        for ev in root.findall("event"):
            impact = ev.findtext("impact", "").strip().lower()
            if impact != "high":
                continue
            title    = ev.findtext("title", "")
            country  = ev.findtext("country", "")
            date_str = ev.findtext("date", "")
            time_str = ev.findtext("time", "")
            try:
                # format: "04-01-2026" si "8:30am" — ForexFactory e in ET (America/New_York)
                dt_str = f"{date_str} {time_str}"
                dt_naive = datetime.strptime(dt_str, "%m-%d-%Y %I:%M%p")
                try:
                    from zoneinfo import ZoneInfo
                    _ET = ZoneInfo("America/New_York")
                    dt = dt_naive.replace(tzinfo=_ET).astimezone(timezone.utc)
                except Exception:
                    # fallback: EDT = UTC-4
                    dt = (dt_naive - timedelta(hours=-4)).replace(tzinfo=timezone.utc)
                events.append({"title": title, "country": country, "dt": dt})
            except:
                pass
        with _news_lock:
            _news_cache["events"]     = events
            _news_cache["fetched_at"] = datetime.now(timezone.utc)
        log.info(f"ForexFactory: {len(events)} stiri rosii incarcate")
        return events
    except Exception as e:
        log.warning(f"ForexFactory fetch eroare: {e}")
        return _news_cache["events"]  # returneaza cache vechi daca exista


def get_upcoming_red_news(minutes_ahead=5):
    """Returneaza stirile rosii din urmatoarele N minute."""
    from datetime import datetime, timezone, timedelta
    now    = datetime.now(timezone.utc)
    events = fetch_red_news()
    upcoming = []
    for ev in events:
        diff = (ev["dt"] - now).total_seconds() / 60
        if -FTMO_NEWS_BLOCK_MIN <= diff <= minutes_ahead:
            upcoming.append({**ev, "in_minutes": round(diff, 1),
                             "dt": ev["dt"].strftime("%H:%M UTC")})
    return upcoming


def close_all_positions_for_news():
    """Inchide toate pozitiile deschise inainte de stire rosie."""
    if not MT5_AVAILABLE or mt5 is None:
        return []
    positions = mt5.positions_get()
    if not positions:
        return []
    closed = []
    for pos in positions:
        tick  = mt5.symbol_info_tick(pos.symbol)
        info  = mt5.symbol_info(pos.symbol)
        if not tick or not info:
            continue
        close_price = tick.bid if pos.type == 0 else tick.ask
        fm = info.filling_mode
        if fm & 2:    filling = mt5.ORDER_FILLING_IOC
        elif fm & 1:  filling = mt5.ORDER_FILLING_FOK
        else:         filling = mt5.ORDER_FILLING_RETURN
        req = {
            "action":       mt5.TRADE_ACTION_DEAL,
            "symbol":       pos.symbol,
            "volume":       pos.volume,
            "type":         mt5.ORDER_TYPE_SELL if pos.type == 0 else mt5.ORDER_TYPE_BUY,
            "position":     pos.ticket,
            "price":        close_price,
            "deviation":    30,
            "magic":        pos.magic,
            "comment":      "news_close",
            "type_time":    mt5.ORDER_TIME_GTC,
            "type_filling": filling,
        }
        result = mt5.order_send(req)
        if result and result.retcode == mt5.TRADE_RETCODE_DONE:
            closed.append(pos.symbol)
            log.info(f"Pozitie inchisa inainte de stire: {pos.symbol} #{pos.ticket}")
        else:
            log.warning(f"Eroare inchidere {pos.symbol}: {result.retcode if result else -1}")
    return closed

class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, (np.integer,)): return int(obj)
        if isinstance(obj, (np.floating,)): return float(obj)
        if isinstance(obj, np.ndarray): return obj.tolist()
        return super().default(obj)

# ── Fetch MT5 ─────────────────────────────────────────────────────────────────
def fetch_mt5(symbol, tf, bars):
    if not MT5_AVAILABLE or mt5 is None:
        return None
    tf_const = MT5_TF.get(tf)
    if tf_const is None:
        return None
    rates = mt5.copy_rates_from_pos(symbol, tf_const, 0, int(bars) + 10)
    if rates is None or len(rates) == 0:
        return None
    df = pd.DataFrame(rates)
    df["time"] = pd.to_datetime(df["time"], unit="s")
    df = df.set_index("time")
    df = df.rename(columns={"tick_volume": "volume"})
    return df[["open","high","low","close","volume"]].tail(int(bars))

def fetch(symbol, tf, bars):
    df = fetch_mt5(symbol, tf, bars)
    if df is not None and len(df) >= 10:
        return df, "MT5"
    return None, "MT5 indisponibil"

# ── Pivot detection ───────────────────────────────────────────────────────────
def find_pivots(df, lookback=5):
    highs = df["high"].values
    lows  = df["low"].values
    n     = len(highs)
    ph, pl = [], []
    for i in range(lookback, n - lookback):
        if all(highs[i] >= highs[i-j] for j in range(1, lookback+1)) and \
           all(highs[i] >= highs[i+j] for j in range(1, lookback+1)):
            ph.append(i)
        if all(lows[i] <= lows[i-j] for j in range(1, lookback+1)) and \
           all(lows[i] <= lows[i+j] for j in range(1, lookback+1)):
            pl.append(i)
    return ph, pl

# ── Trend detection (pe ultimele recent_bars) ─────────────────────────────────
def detect_trend(ph_idx, pl_idx, highs, lows, recent_bars=100):
    n = len(highs)
    cutoff = n - recent_bars
    ph_r = [i for i in ph_idx if i >= cutoff]
    pl_r = [i for i in pl_idx if i >= cutoff]
    if len(ph_r) < 2 or len(pl_r) < 2:
        return "RANGING"
    ph_p = [highs[i] for i in ph_r[-6:]]
    pl_p = [lows[i]  for i in pl_r[-6:]]
    hh = sum(ph_p[i] > ph_p[i-1] for i in range(1, len(ph_p)))
    lh = sum(ph_p[i] < ph_p[i-1] for i in range(1, len(ph_p)))
    hl = sum(pl_p[i] > pl_p[i-1] for i in range(1, len(pl_p)))
    ll = sum(pl_p[i] < pl_p[i-1] for i in range(1, len(pl_p)))
    bull = hh + hl
    bear = lh + ll
    if bull > bear and bull >= 2:
        return "ASCENDING"
    if bear > bull and bear >= 2:
        return "DESCENDING"
    return "RANGING"

# ── ADX calculation ───────────────────────────────────────────────────────────
def calc_adx(df, period=14):
    """Calculeaza ADX. Returneaza seria ADX."""
    high  = df["high"]
    low   = df["low"]
    close = df["close"]
    tr    = pd.concat([high - low,
                       (high - close.shift()).abs(),
                       (low  - close.shift()).abs()], axis=1).max(axis=1)
    dm_plus  = (high - high.shift()).clip(lower=0)
    dm_minus = (low.shift() - low).clip(lower=0)
    # anuleaza unde nu e clar
    dm_plus  = dm_plus.where(dm_plus > dm_minus, 0)
    dm_minus = dm_minus.where(dm_minus > dm_plus, 0)

    atr14    = tr.rolling(period).mean()
    di_plus  = 100 * dm_plus.rolling(period).mean()  / atr14.replace(0, np.nan)
    di_minus = 100 * dm_minus.rolling(period).mean() / atr14.replace(0, np.nan)
    dx       = 100 * (di_plus - di_minus).abs() / (di_plus + di_minus).replace(0, np.nan)
    adx      = dx.rolling(period).mean()
    return adx


# ── H4 trend filter ───────────────────────────────────────────────────────────
def get_h4_direction(symbol, bars=100):
    """Returneaza directia trendului pe H4: 'BUY', 'SELL' sau None (lateral)."""
    try:
        df, _ = fetch(symbol, "H4", bars)
        if df is None or len(df) < 60:
            return None
        ema20 = float(df["close"].ewm(span=20, adjust=False).mean().iloc[-1])
        ema50 = float(df["close"].ewm(span=50, adjust=False).mean().iloc[-1])
        adx   = calc_adx(df)
        adx_now = float(adx.iloc[-1]) if not adx.empty else 0
        if adx_now < ADX_MIN:
            return None   # piata laterala pe H4 — nu tranzactiona
        if ema20 > ema50 * 1.0001:
            return "BUY"
        if ema20 < ema50 * 0.9999:
            return "SELL"
        return None
    except Exception as e:
        log.warning(f"get_h4_direction {symbol}: {e}")
        return None


# ── Session filter ────────────────────────────────────────────────────────────
def in_trading_session():
    """Returneaza True daca suntem in fereastra London sau New York."""
    from datetime import datetime, timezone
    now = datetime.now(timezone.utc)
    h, m = now.hour, now.minute
    cur = h * 60 + m
    for sh, sm, eh, em in TRADING_SESSIONS:
        if sh * 60 + sm <= cur <= eh * 60 + em:
            return True
    return False



# ── Fibonacci retracement ────────────────────────────────────────────────────
FIB_LEVELS = [0.0, 0.236, 0.382, 0.5, 0.618, 0.786, 1.0]

def calc_fib_levels(df, ph_idx, pl_idx, trend):
    """
    Calculeaza nivelele Fibonacci de retracement din ultimul swing major.
    Returneaza dict cu nivelele si swing_high/swing_low, sau None.
    """
    highs  = df["high"].values
    lows   = df["low"].values
    n      = len(highs)
    cutoff = max(0, n - 200)
    ph_r   = [i for i in ph_idx if i >= cutoff]
    pl_r   = [i for i in pl_idx if i >= cutoff]

    if not ph_r or not pl_r:
        return None

    last_ph = highs[ph_r[-1]]
    last_pl = lows[pl_r[-1]]

    # Swing direction: in trend ascendent, swing e de la ultimul PL la ultimul PH
    if trend == "ASCENDING":
        # masurim de la swing_low (pl) la swing_high (ph)
        # dar PH trebuie sa fie DUPA PL pentru a fi un swing valid
        valid_pls = [i for i in pl_r if i < ph_r[-1]]
        if not valid_pls:
            return None
        swing_low  = lows[valid_pls[-1]]
        swing_high = last_ph
    elif trend == "DESCENDING":
        # masurim de la swing_high (ph) la swing_low (pl)
        valid_phs = [i for i in ph_r if i < pl_r[-1]]
        if not valid_phs:
            return None
        swing_high = highs[valid_phs[-1]]
        swing_low  = last_pl
    else:
        return None

    diff = swing_high - swing_low
    if diff <= 0:
        return None

    levels = {}
    for f in FIB_LEVELS:
        if trend == "ASCENDING":
            levels[f] = round(swing_high - diff * f, 5)  # retracement de la high in jos
        else:
            levels[f] = round(swing_low + diff * f, 5)   # retracement de la low in sus

    return {
        "swing_high": round(swing_high, 5),
        "swing_low":  round(swing_low, 5),
        "levels":     levels,  # {0.0: price, 0.382: price, ...}
        "trend":      trend,
    }


def price_near_fib(price, fib_data, tolerance=0.002):
    """
    Verifica daca pretul e in 'golden zone' Fibonacci (38.2% sau 61.8%).
    Returneaza (True, nivel_cel_mai_apropiat) sau (False, None).
    """
    if fib_data is None:
        return False, None
    golden = [0.382, 0.5, 0.618]
    best_level = None
    best_dist  = float("inf")
    for f in golden:
        fib_price = fib_data["levels"].get(f)
        if fib_price is None:
            continue
        dist = abs(price - fib_price) / fib_price
        if dist < best_dist:
            best_dist  = dist
            best_level = f
    if best_dist <= tolerance:
        return True, best_level
    return False, None


# ── Entry conditions (noua strategie: EMA50 pullback + Fibonacci + ADX) ──────
def calc_entry(df, ph_idx, pl_idx, trend, ema20, ema50, rsi):
    """
    Intrare pe pullback la EMA50 SAU nivel Fibonacci 38.2%/61.8%,
    confirmat de ADX > 25 si RSI in zona neutra.
    """
    highs     = df["high"].values
    lows      = df["low"].values
    price_now = float(df["close"].iloc[-1])
    rsi_now   = float(rsi.iloc[-1])
    ema20_now = float(ema20.iloc[-1])
    ema50_now = float(ema50.iloc[-1])
    entry_signal = "HOLD"
    entry_reason = []

    # ADX pe acest TF
    adx     = calc_adx(df)
    adx_now = float(adx.iloc[-1]) if not adx.empty else 0

    # Fibonacci levels
    fib_data = calc_fib_levels(df, ph_idx, pl_idx, trend)
    at_fib, fib_level = price_near_fib(price_now, fib_data)

    # Conditie ADX
    if adx_now < ADX_MIN:
        entry_reason.append(f"ADX {round(adx_now,1)} < {ADX_MIN} — piata laterala")
        return "HOLD", entry_reason, price_now

    rsi_ok = 38 <= rsi_now <= 62

    if trend == "ASCENDING":
        ema_aligned = ema20_now > ema50_now
        near_ema50  = ema50_now * 0.998 <= price_now <= ema50_now * 1.002
        # intrare valida: la EMA50 SAU la nivel fib golden zone
        at_support  = near_ema50 or at_fib

        if ema_aligned:  entry_reason.append("EMA20>EMA50 ✓")
        if near_ema50:   entry_reason.append(f"pullback EMA50 ({round(ema50_now,5)}) ✓")
        if at_fib:       entry_reason.append(f"Fibonacci {round(fib_level*100,1)}% ({round(fib_data['levels'][fib_level],5)}) ✓")
        if rsi_ok:       entry_reason.append(f"RSI {round(rsi_now,1)} zona neutra ✓")
        entry_reason.append(f"ADX {round(adx_now,1)} ✓")

        if ema_aligned and at_support and rsi_ok:
            entry_signal = "BUY"

    elif trend == "DESCENDING":
        ema_aligned = ema20_now < ema50_now
        near_ema50  = ema50_now * 0.998 <= price_now <= ema50_now * 1.002
        at_resist   = near_ema50 or at_fib

        if ema_aligned:  entry_reason.append("EMA20<EMA50 ✓")
        if near_ema50:   entry_reason.append(f"pullback EMA50 ({round(ema50_now,5)}) ✓")
        if at_fib:       entry_reason.append(f"Fibonacci {round(fib_level*100,1)}% ({round(fib_data['levels'][fib_level],5)}) ✓")
        if rsi_ok:       entry_reason.append(f"RSI {round(rsi_now,1)} zona neutra ✓")
        entry_reason.append(f"ADX {round(adx_now,1)} ✓")

        if ema_aligned and at_resist and rsi_ok:
            entry_signal = "SELL"

    return entry_signal, entry_reason, price_now

# ── SL/TP din pivoti ──────────────────────────────────────────────────────────
def calc_sl_tp(df, ph_idx, pl_idx, signal, price):
    highs = df["high"].values
    lows  = df["low"].values
    n     = len(highs)
    cutoff = n - 100
    atr   = float(df["high"].sub(df["low"]).rolling(14).mean().iloc[-1])

    sl = tp = 0.0
    ph_r = [i for i in ph_idx if i >= cutoff]
    pl_r = [i for i in pl_idx if i >= cutoff]

    if signal == "BUY":
        sl = lows[pl_r[-1]] - atr * 0.3 if pl_r else price - atr * 2
        risk = price - sl
        tp   = price + risk * 2.0
    elif signal == "SELL":
        sl = highs[ph_r[-1]] + atr * 0.3 if ph_r else price + atr * 2
        risk = sl - price
        tp   = price - risk * 2.0

    return round(sl, 5), round(tp, 5)

# ── Analiza per TF ────────────────────────────────────────────────────────────
def get_signal_data(symbol, tf, bars=500):
    df, _ = fetch(symbol, tf, bars)
    if df is None:
        return None
    highs = df["high"].values
    lows  = df["low"].values
    ph_idx, pl_idx = find_pivots(df, lookback=5)
    trend = detect_trend(ph_idx, pl_idx, highs, lows, recent_bars=100)
    ema20 = df["close"].ewm(span=20, adjust=False).mean()
    ema50 = df["close"].ewm(span=50, adjust=False).mean()
    delta = df["close"].diff()
    gain  = delta.clip(lower=0).rolling(14).mean()
    loss  = (-delta.clip(upper=0)).rolling(14).mean()
    rsi   = 100 - 100 / (1 + gain / loss.replace(0, np.nan))
    signal, reasons, price = calc_entry(df, ph_idx, pl_idx, trend, ema20, ema50, rsi)
    sl, tp = calc_sl_tp(df, ph_idx, pl_idx, signal, price)
    return {
        "tf": tf,
        "signal": signal,
        "trend": trend,
        "conviction": len(reasons),
        "reasons": reasons,
        "price": round(price, 5),
        "sl": sl,
        "tp": tp,
    }

# ── Agregate multi-TF ─────────────────────────────────────────────────────────
def analyze_symbol(symbol, tfs=None, bars=500):
    if tfs is None:
        tfs = ["M1", "M5", "M15", "H1", "H4"]
    results = [r for tf in tfs if (r := get_signal_data(symbol, tf, bars)) is not None]
    buy_v  = [r for r in results if r["signal"] == "BUY"]
    sell_v = [r for r in results if r["signal"] == "SELL"]
    n_buy, n_sell = len(buy_v), len(sell_v)
    final = "HOLD"
    best  = None
    if n_buy >= MIN_TF_VOTES and n_buy > n_sell:
        final = "BUY"
        best  = max(buy_v,  key=lambda x: x["conviction"])
    elif n_sell >= MIN_TF_VOTES and n_sell > n_buy:
        final = "SELL"
        best  = max(sell_v, key=lambda x: x["conviction"])
    return {
        "symbol": symbol,
        "signal": final,
        "n_buy": n_buy, "n_sell": n_sell,
        "n_total": len(results),
        "best": best,
        "all": results,
    }

# ── Verificari FTMO ──────────────────────────────────────────────────────────
def check_ftmo_rules():
    """Returneaza (ok, motiv) — ok=False inseamna trade blocat."""
    if not FTMO_ENABLED or not MT5_AVAILABLE or mt5 is None:
        return True, ""

    from datetime import datetime, timezone
    now_utc = datetime.now(timezone.utc)

    # 1. Verificare news — folosim cache-ul real ForexFactory
    upcoming = get_upcoming_red_news(minutes_ahead=FTMO_NEWS_BLOCK_MIN)
    if upcoming:
        ev = upcoming[0]
        return False, f"Blocat: stire {ev['title']} ({ev['country']}) in {ev['in_minutes']} min"

    # 2. Verificare drawdown zilnic 5%
    acc = mt5.account_info()
    if acc:
        balance    = acc.balance
        equity     = acc.equity
        # drawdown zilnic: equity sub 95% din balance
        daily_floor = balance * (1 - FTMO_DAILY_LOSS_PCT)
        if equity <= daily_floor:
            return False, f"Blocat: drawdown zilnic atins (equity {equity:.2f} <= {daily_floor:.2f})"

        # 3. Verificare drawdown total 10%
        # folosim balance ca proxy pentru initial balance (nu avem istoricul exact)
        # daca equity < 90% din balance curent, blocam
        total_floor = balance * (1 - FTMO_MAX_LOSS_PCT)
        if equity <= total_floor:
            return False, f"Blocat: drawdown total 10% atins (equity {equity:.2f})"

    # 4. Weekend — nu tranzactiona vineri dupa 21:00 UTC si sambata/duminica
    weekday = now_utc.weekday()  # 4=vineri, 5=sambata, 6=duminica
    if weekday == 6:
        return False, "Blocat: duminica — piata inchisa"
    if weekday == 5:
        return False, "Blocat: sambata — piata inchisa"
    if weekday == 4 and now_utc.hour >= 21:
        return False, "Blocat: vineri dupa 21:00 UTC — inchidere weekend"

    return True, ""


# ── Executa trade MT5 ─────────────────────────────────────────────────────────
def get_signal_file_path():
    """Returneaza calea catre fisierul de semnal in directorul MT5 Files"""
    if MT5_AVAILABLE and mt5 is not None:
        info = mt5.terminal_info()
        if info:
            return os.path.join(info.data_path, "MQL5", "Files", "cv_signal.json")
    # fallback langa app.py
    return os.path.join(os.path.dirname(__file__), "cv_signal.json")

def place_trade(symbol, signal, sl, tp, risk_dollars=50.0):
    import math, os
    if not MT5_AVAILABLE or mt5 is None:
        return False, "MT5 indisponibil"

    if not mt5.initialize():
        return False, "MT5 initialize() esuat"

    # Verificare sesiune activa
    if not in_trading_session():
        from datetime import datetime, timezone
        now_utc = datetime.now(timezone.utc)
        return False, f"In afara sesiunii de tranzactionare ({now_utc.strftime('%H:%M')} UTC) — activ 07-12 si 13-17"

    # Verificare H4 direction — tranzactionam doar cu trendul mare
    h4_dir = get_h4_direction(symbol)
    if h4_dir is None:
        return False, f"H4 lateral sau ADX slab pe {symbol} — nu intra"
    if h4_dir != signal:
        return False, f"H4 direction={h4_dir} dar semnal={signal} — contra-trend blocat"

    # Verificare reguli FTMO
    ftmo_ok, ftmo_msg = check_ftmo_rules()
    if not ftmo_ok:
        log.warning(f"FTMO block: {ftmo_msg}")
        return False, ftmo_msg

    # Verificare numar maxim pozitii deschise
    open_count = mt5.positions_total()
    if open_count >= MAX_OPEN_TRADES:
        return False, f"Limita atinsa: {open_count}/{MAX_OPEN_TRADES} pozitii deschise — asteapta sa se inchida una"

    # Verificare: nu deschide al doilea trade pe acelasi simbol
    existing = mt5.positions_get(symbol=symbol)
    if existing and len(existing) > 0:
        return False, f"Deja exista {len(existing)} pozitie deschisa pe {symbol} — skip"

    # Verificare corelatie USD — max 1 trade in aceeasi directie USD simultan
    # Ex: EURUSD BUY + GBPUSD BUY + AUDUSD BUY = toate "vand USD" → risc corelat
    USD_BASE   = {"USDJPY", "USDCHF", "USDCAD"}   # USD e prima — BUY = cumperi USD
    USD_QUOTE  = {"EURUSD", "GBPUSD", "AUDUSD", "NZDUSD", "XAUUSD"}  # USD e a doua — BUY = vinzi USD
    # directia fata de USD
    if symbol in USD_BASE:
        usd_direction = "BUY_USD" if signal == "BUY" else "SELL_USD"
    elif symbol in USD_QUOTE:
        usd_direction = "SELL_USD" if signal == "BUY" else "BUY_USD"
    else:
        usd_direction = None  # pereche cross (EURGBP, EURJPY etc.) — nu se aplica

    if usd_direction:
        all_positions = mt5.positions_get() or []
        usd_dir_count = 0
        for pos in all_positions:
            ps = pos.symbol
            pt = "BUY" if pos.type == 0 else "SELL"
            if ps in USD_BASE:
                pd = "BUY_USD" if pt == "BUY" else "SELL_USD"
            elif ps in USD_QUOTE:
                pd = "SELL_USD" if pt == "BUY" else "BUY_USD"
            else:
                continue
            if pd == usd_direction:
                usd_dir_count += 1
        if usd_dir_count >= 1:
            return False, f"Corelatie USD: deja exista {usd_dir_count} trade(s) in directia {usd_direction} — skip {symbol}"

    info = mt5.symbol_info(symbol)
    if info is None:
        return False, f"Symbol {symbol} negasit in MT5"

    if not info.visible:
        mt5.symbol_select(symbol, True)

    tick = mt5.symbol_info_tick(symbol)
    if tick is None:
        return False, "Nu s-a putut obtine pretul"

    exec_price = tick.ask if signal == "BUY" else tick.bid
    tick_val   = info.trade_tick_value
    tick_size  = info.trade_tick_size
    sl_dist    = abs(exec_price - sl)

    if tick_size <= 0 or tick_val <= 0:
        return False, f"Date simbol invalide (tick_size={tick_size})"

    # Distanta minima SL: broker minimum SAU 20 pips, oricare e mai mare
    try:
        stops_level = getattr(info, "trade_stops_level", None) or getattr(info, "stops_level", 0)
        broker_min  = stops_level * info.point
    except:
        broker_min  = 0
    pip_size = info.point * (10 if info.digits in (5, 3) else 1)  # 1 pip in price units
    min_dist = max(broker_min, 20 * pip_size, exec_price * 0.0015)  # minim 20 pips SAU 0.15% din pret

    # Ajusteaza SL daca e prea aproape (impinge mai departe, nu mai aproape)
    if signal == "BUY":
        sl = min(sl, exec_price - min_dist)
    else:
        sl = max(sl, exec_price + min_dist)

    sl_dist = abs(exec_price - sl)
    if sl_dist <= 0:
        return False, f"SL invalid dupa ajustare"

    # TP fix 1:1 (risca 1, castiga 1)
    if signal == "BUY":
        tp = exec_price + sl_dist * 1.0
    else:
        tp = exec_price - sl_dist * 1.0

    lot_step = info.volume_step
    min_lot  = info.volume_min
    max_lot  = min(info.volume_max, 1.0)  # cap la 1 lot maxim pentru siguranta
    lots = math.floor(risk_dollars / (sl_dist / tick_size * tick_val) / lot_step) * lot_step
    lots = max(min_lot, min(max_lot, lots))
    lots = round(lots, 2)
    log.info(f"place_trade {symbol} {signal}: price={exec_price} sl={round(sl,info.digits)} sl_dist={round(sl_dist,info.digits)} ({round(sl_dist/pip_size,1)} pips) lots={lots}")

    order_type = mt5.ORDER_TYPE_BUY if signal == "BUY" else mt5.ORDER_TYPE_SELL

    # Detecteaza filling mode suportat de broker (bitmask)
    fm = info.filling_mode
    if fm & 2:    filling = mt5.ORDER_FILLING_IOC
    elif fm & 1:  filling = mt5.ORDER_FILLING_FOK
    else:         filling = mt5.ORDER_FILLING_RETURN

    req = {
        "action":       mt5.TRADE_ACTION_DEAL,
        "symbol":       symbol,
        "volume":       lots,
        "type":         order_type,
        "price":        exec_price,
        "sl":           round(sl, info.digits),
        "tp":           round(tp, info.digits),
        "deviation":    30,
        "magic":        TRADE_MAGIC,
        "comment":      f"CV_{signal}",
        "type_time":    mt5.ORDER_TIME_GTC,
        "type_filling": filling,
    }
    log.info(f"order_send: {req}")

    # Incearca toate filling modes + fara filling mode
    attempts = [
        {**req, "type_filling": mt5.ORDER_FILLING_IOC},
        {**req, "type_filling": mt5.ORDER_FILLING_FOK},
        {**req, "type_filling": mt5.ORDER_FILLING_RETURN},
        {k: v for k, v in req.items() if k != "type_filling"},  # fara filling
    ]
    last_code = -1
    for attempt in attempts:
        result = mt5.order_send(attempt)
        if result is not None and result.retcode == mt5.TRADE_RETCODE_DONE:
            ticket = result.order
            import threading as _thr
            _thr.Thread(target=save_trade_snapshot, args=(ticket, symbol, signal, exec_price,
                round(sl, info.digits), round(tp, info.digits)), kwargs={"tf": "M5"}, daemon=True).start()
            return True, f"OK — {lots} loturi {signal} {symbol} @ {exec_price}  SL={round(sl,info.digits)}  TP={round(tp,info.digits)}"
        last_code = result.retcode if result else -1
        log.warning(f"attempt filling={attempt.get('type_filling','none')} retcode={last_code}")

    msgs = {
        10027: "AutoTrading dezactivat — Tools→Options→Expert Advisors→Allow algorithmic trading",
        10030: "Filling mode incompatibil cu brokerul (10030)",
        10018: "Piata inchisa",
        10019: "Fonduri insuficiente",
        10016: f"SL/TP invalid ({info.digits} zecimale)",
        10014: f"Volum invalid (lots={lots})",
        10006: "Ordin respins de broker",
        10013: "Parametri invalizi",
    }
    return False, msgs.get(last_code, f"Eroare MT5: {last_code}")

# ── Build chart ───────────────────────────────────────────────────────────────
def build_chart(symbol, tf, lookback, compact=False):
    df, source = fetch(symbol, tf, lookback)
    if df is None:
        return f"<div style='color:#ef5350;padding:12px'>Nu s-au putut incarca datele MT5 pentru {symbol}/{tf}.</div>"
    highs = df["high"].values
    lows  = df["low"].values
    dates = df.index
    ph_idx, pl_idx = find_pivots(df, lookback=5)
    trend = detect_trend(ph_idx, pl_idx, highs, lows, recent_bars=100)
    ema20 = df["close"].ewm(span=20, adjust=False).mean()
    ema50 = df["close"].ewm(span=50, adjust=False).mean()
    delta = df["close"].diff()
    gain  = delta.clip(lower=0).rolling(14).mean()
    loss  = (-delta.clip(upper=0)).rolling(14).mean()
    rsi   = 100 - 100 / (1 + gain / loss.replace(0, np.nan))
    entry_signal, entry_reason, price_now = calc_entry(df, ph_idx, pl_idx, trend, ema20, ema50, rsi)
    fib_data = calc_fib_levels(df, ph_idx, pl_idx, trend)
    height      = 460 if compact else 680
    row_heights = [0.72, 0.28] if compact else [0.78, 0.22]
    fig = make_subplots(rows=2, cols=1, row_heights=row_heights,
                        shared_xaxes=True, vertical_spacing=0.03)
    fig.add_trace(go.Candlestick(
        x=dates, open=df["open"], high=df["high"], low=df["low"], close=df["close"],
        name="OHLC", increasing_line_color="#26a69a", decreasing_line_color="#ef5350",
        increasing_fillcolor="#26a69a", decreasing_fillcolor="#ef5350",
        showlegend=not compact), row=1, col=1)
    fig.add_trace(go.Scatter(x=dates, y=ema20, line=dict(color="#ffeb3b", width=1.2),
        name="EMA20", opacity=0.9, showlegend=not compact), row=1, col=1)
    fig.add_trace(go.Scatter(x=dates, y=ema50, line=dict(color="#ff9800", width=1.2),
        name="EMA50", opacity=0.9, showlegend=not compact), row=1, col=1)
    if ph_idx:
        fig.add_trace(go.Scatter(x=[dates[i] for i in ph_idx], y=[highs[i]*1.0002 for i in ph_idx],
            mode="markers+text", marker=dict(symbol="triangle-down", size=8 if compact else 10, color="#ef5350"),
            text=["H"]*len(ph_idx), textposition="top center", textfont=dict(size=8, color="#ef5350"),
            name="Pivot High", showlegend=not compact), row=1, col=1)
    if pl_idx:
        fig.add_trace(go.Scatter(x=[dates[i] for i in pl_idx], y=[lows[i]*0.9998 for i in pl_idx],
            mode="markers+text", marker=dict(symbol="triangle-up", size=8 if compact else 10, color="#26a69a"),
            text=["L"]*len(pl_idx), textposition="bottom center", textfont=dict(size=8, color="#26a69a"),
            name="Pivot Low", showlegend=not compact), row=1, col=1)
    if len(ph_idx) >= 2:
        lc = "#ef5350" if trend == "DESCENDING" else "#ff9800"
        fig.add_trace(go.Scatter(x=[dates[i] for i in ph_idx[-6:]], y=[highs[i] for i in ph_idx[-6:]],
            mode="lines", line=dict(color=lc, width=1, dash="dot"), showlegend=False, opacity=0.7), row=1, col=1)
    if len(pl_idx) >= 2:
        lc = "#26a69a" if trend == "ASCENDING" else "#ef5350"
        fig.add_trace(go.Scatter(x=[dates[i] for i in pl_idx[-6:]], y=[lows[i] for i in pl_idx[-6:]],
            mode="lines", line=dict(color=lc, width=1, dash="dot"), showlegend=False, opacity=0.7), row=1, col=1)
    # ── Fibonacci lines ───────────────────────────────────────────────────────
    FIB_COLORS = {
        0.0:   ("#888",   "0%"),
        0.236: ("#b0bec5","23.6%"),
        0.382: ("#64b5f6","38.2%"),
        0.5:   ("#fff176","50%"),
        0.618: ("#ffb74d","61.8%"),  # golden ratio — cel mai important
        0.786: ("#ef9a9a","78.6%"),
        1.0:   ("#888",   "100%"),
    }
    if fib_data and not compact:
        for f, (color, label) in FIB_COLORS.items():
            fv = fib_data["levels"].get(f)
            if fv is None:
                continue
            is_golden = f in (0.382, 0.5, 0.618)
            fig.add_hline(y=fv,
                line=dict(color=color, width=1.5 if is_golden else 0.8,
                          dash="solid" if is_golden else "dot"),
                annotation_text=f"  Fib {label} ({fv})",
                annotation_font=dict(color=color, size=9),
                annotation_position="right",
                row=1, col=1)
        # zona golden zone (38.2% - 61.8%) colorata subtil
        fib_382 = fib_data["levels"].get(0.382)
        fib_618 = fib_data["levels"].get(0.618)
        if fib_382 and fib_618:
            y0, y1 = min(fib_382, fib_618), max(fib_382, fib_618)
            fig.add_hrect(y0=y0, y1=y1,
                fillcolor="rgba(255,183,77,0.07)",
                line=dict(color="rgba(255,183,77,0.3)", width=1),
                row=1, col=1)

    if entry_signal in ("BUY", "SELL"):
        zone_dates = list(dates[-10:])
        entry_col  = "rgba(38,166,154,0.15)" if entry_signal == "BUY" else "rgba(239,83,80,0.15)"
        border_col = "#26a69a" if entry_signal == "BUY" else "#ef5350"
        fig.add_vrect(x0=zone_dates[0], x1=zone_dates[-1], fillcolor=entry_col,
            line=dict(color=border_col, width=1.5, dash="dash"),
            annotation_text=f"ENTRY {entry_signal}: {', '.join(entry_reason)}",
            annotation_position="top left",
            annotation_font=dict(size=9 if compact else 10, color=border_col), row=1, col=1)
        fig.add_hline(y=price_now, line=dict(color=border_col, width=1, dash="dash"),
            annotation_text=f"  {entry_signal} @ {round(price_now,5)}",
            annotation_font=dict(color=border_col, size=9 if compact else 10), row=1, col=1)
    fig.add_trace(go.Scatter(x=dates, y=rsi, line=dict(color="#ab47bc", width=1.2),
        name="RSI(14)", showlegend=not compact), row=2, col=1)
    fig.add_hline(y=70, line=dict(color="#ef5350", width=0.8, dash="dash"), row=2, col=1)
    fig.add_hline(y=30, line=dict(color="#26a69a", width=0.8, dash="dash"), row=2, col=1)
    fig.add_hline(y=50, line=dict(color="#555",    width=0.6, dash="dot"),  row=2, col=1)
    trend_label = {"ASCENDING":"▲ ASCENDING","DESCENDING":"▼ DESCENDING","RANGING":"— RANGING"}[trend]
    trend_col   = {"ASCENDING":"#26a69a","DESCENDING":"#ef5350","RANGING":"#aaa"}[trend]
    entry_badge = ""
    if entry_signal == "BUY":   entry_badge = "  <span style='color:#26a69a'>● BUY</span>"
    elif entry_signal == "SELL": entry_badge = "  <span style='color:#ef5350'>● SELL</span>"
    fig.update_layout(
        template="plotly_dark", paper_bgcolor="#111", plot_bgcolor="#111",
        title=dict(text=(f"<b>{symbol}</b> — {tf}  |  Trend: <span style='color:{trend_col}'>{trend_label}</span>"
                         f"  |  Pivoti: {len(ph_idx)} H, {len(pl_idx)} L{entry_badge}"
                         f"  |  <span style='color:#888'>sursa: {source}</span>"),
                   font=dict(size=13 if compact else 14, color="#ddd")),
        xaxis_rangeslider_visible=False, height=height,
        margin=dict(l=50, r=30, t=50, b=10 if compact else 20),
        legend=dict(orientation="h", yanchor="bottom", y=1.01, xanchor="left", x=0,
                    font=dict(size=9 if compact else 10), bgcolor="rgba(0,0,0,0)"),
        yaxis=dict(gridcolor="#222", zerolinecolor="#333"),
        yaxis2=dict(gridcolor="#222", zerolinecolor="#333", range=[0,100]),
        xaxis2=dict(gridcolor="#222"),
    )
    return fig.to_html(full_html=False, include_plotlyjs="cdn")

SNAPSHOTS_DIR = os.path.join(os.path.dirname(__file__), "snapshots")
os.makedirs(SNAPSHOTS_DIR, exist_ok=True)

def save_trade_snapshot(ticket, symbol, signal, entry, sl, tp, tf="M5", analysis=None):
    """Genereaza si salveaza un grafic HTML cu Entry/SL/TP la momentul executiei."""
    try:
        from datetime import datetime
        df, source = fetch(symbol, tf, 200)
        if df is None:
            log.warning(f"snapshot {ticket}: nu s-au putut incarca datele")
            return

        highs  = df["high"].values
        lows   = df["low"].values
        dates  = df.index
        ph_idx, pl_idx = find_pivots(df, lookback=5)
        trend  = detect_trend(ph_idx, pl_idx, highs, lows, recent_bars=100)
        ema20  = df["close"].ewm(span=20, adjust=False).mean()
        ema50  = df["close"].ewm(span=50, adjust=False).mean()
        delta  = df["close"].diff()
        gain   = delta.clip(lower=0).rolling(14).mean()
        loss   = (-delta.clip(upper=0)).rolling(14).mean()
        rsi    = 100 - 100 / (1 + gain / loss.replace(0, np.nan))

        fig = make_subplots(rows=2, cols=1, row_heights=[0.78, 0.22],
                            shared_xaxes=True, vertical_spacing=0.03)

        # Candlestick
        fig.add_trace(go.Candlestick(
            x=dates, open=df["open"], high=df["high"], low=df["low"], close=df["close"],
            name="OHLC", increasing_line_color="#26a69a", decreasing_line_color="#ef5350",
            increasing_fillcolor="#26a69a", decreasing_fillcolor="#ef5350"), row=1, col=1)

        # EMA
        fig.add_trace(go.Scatter(x=dates, y=ema20, line=dict(color="#ffeb3b", width=1.2), name="EMA20"), row=1, col=1)
        fig.add_trace(go.Scatter(x=dates, y=ema50, line=dict(color="#ff9800", width=1.2), name="EMA50"), row=1, col=1)

        # Pivoti
        if ph_idx:
            fig.add_trace(go.Scatter(x=[dates[i] for i in ph_idx], y=[highs[i]*1.0002 for i in ph_idx],
                mode="markers", marker=dict(symbol="triangle-down", size=8, color="#ef5350"),
                name="Pivot H"), row=1, col=1)
        if pl_idx:
            fig.add_trace(go.Scatter(x=[dates[i] for i in pl_idx], y=[lows[i]*0.9998 for i in pl_idx],
                mode="markers", marker=dict(symbol="triangle-up", size=8, color="#26a69a"),
                name="Pivot L"), row=1, col=1)

        # ── Linii Entry / SL / TP ──
        entry_col = "#26a69a" if signal == "BUY" else "#ef5350"
        fig.add_hline(y=entry, line=dict(color=entry_col, width=2, dash="solid"),
            annotation_text=f"  ENTRY {signal} @ {round(entry,5)}",
            annotation_font=dict(color=entry_col, size=11), row=1, col=1)
        fig.add_hline(y=sl, line=dict(color="#ef5350", width=1.5, dash="dash"),
            annotation_text=f"  SL @ {round(sl,5)}",
            annotation_font=dict(color="#ef5350", size=10), row=1, col=1)
        fig.add_hline(y=tp, line=dict(color="#26a69a", width=1.5, dash="dash"),
            annotation_text=f"  TP @ {round(tp,5)}",
            annotation_font=dict(color="#26a69a", size=10), row=1, col=1)

        # Zona colorata intre SL si TP
        sl_color = "rgba(239,83,80,0.07)"
        tp_color = "rgba(38,166,154,0.07)"
        fig.add_hrect(y0=sl, y1=entry, fillcolor=sl_color, line_width=0, row=1, col=1)
        fig.add_hrect(y0=entry, y1=tp, fillcolor=tp_color, line_width=0, row=1, col=1)

        # RSI
        fig.add_trace(go.Scatter(x=dates, y=rsi, line=dict(color="#ab47bc", width=1.2), name="RSI(14)"), row=2, col=1)
        fig.add_hline(y=70, line=dict(color="#ef5350", width=0.8, dash="dash"), row=2, col=1)
        fig.add_hline(y=30, line=dict(color="#26a69a", width=0.8, dash="dash"), row=2, col=1)

        trend_label = {"ASCENDING":"▲ ASCENDING","DESCENDING":"▼ DESCENDING","RANGING":"— RANGING"}.get(trend, trend)
        trend_col   = {"ASCENDING":"#26a69a","DESCENDING":"#ef5350","RANGING":"#aaa"}.get(trend, "#aaa")
        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        rr = abs(tp - entry) / abs(entry - sl) if abs(entry - sl) > 0 else 0
        title_text = (
            f"<b>#{ticket} — {symbol} {signal}</b>  |  "
            f"Entry: {round(entry,5)}  SL: {round(sl,5)}  TP: {round(tp,5)}  R:R 1:{round(rr,1)}<br>"
            f"<span style='font-size:11px;color:#888'>Trend: <span style='color:{trend_col}'>{trend_label}</span>  |  "
            f"TF: {tf}  |  {ts}</span>"
        )
        if analysis:
            just = analysis.get("justification", [])
            if just:
                title_text += f"<br><span style='font-size:10px;color:#666'>{' · '.join(just[:3])}</span>"

        fig.update_layout(
            template="plotly_dark", paper_bgcolor="#111", plot_bgcolor="#111",
            title=dict(text=title_text, font=dict(size=13, color="#ddd")),
            xaxis_rangeslider_visible=False, height=680,
            margin=dict(l=60, r=30, t=90, b=20),
            legend=dict(orientation="h", yanchor="bottom", y=1.01, xanchor="left", x=0,
                        font=dict(size=9), bgcolor="rgba(0,0,0,0)"),
            yaxis=dict(gridcolor="#222", zerolinecolor="#333"),
            yaxis2=dict(gridcolor="#222", zerolinecolor="#333", range=[0,100]),
            xaxis2=dict(gridcolor="#222"),
        )

        html_content = f"""<!DOCTYPE html><html><head><meta charset="utf-8">
<title>Snapshot #{ticket} {symbol}</title>
<style>body{{background:#111;margin:0;padding:0}}</style></head><body>
{fig.to_html(full_html=False, include_plotlyjs="cdn")}
</body></html>"""

        path = os.path.join(SNAPSHOTS_DIR, f"{ticket}.html")
        with open(path, "w", encoding="utf-8") as f:
            f.write(html_content)

        # salveaza si metadata JSON
        meta_path = os.path.join(SNAPSHOTS_DIR, f"{ticket}.json")
        meta = {
            "ticket":  ticket,
            "symbol":  symbol,
            "signal":  signal,
            "entry":   entry,
            "sl":      sl,
            "tp":      tp,
            "tf":      tf,
            "ts":      ts,
            "rr":      round(rr, 2),
        }
        with open(meta_path, "w") as f:
            json.dump(meta, f)

        log.info(f"Snapshot salvat: snapshots/{ticket}.html")
    except Exception as e:
        log.error(f"save_trade_snapshot eroare: {e}")


def build_multi(symbol, selected_tfs, bars_per_tf):
    parts = []
    for tf in ALL_TFS:
        if tf not in selected_tfs:
            continue
        bars = bars_per_tf.get(tf, 500)
        html = build_chart(symbol, tf, bars, compact=True)
        parts.append(f'<div style="margin-bottom:6px">{html}</div>')
    return "\n".join(parts)

# ── HTML ──────────────────────────────────────────────────────────────────────
HTML = """<!DOCTYPE html>
<html><head>
<meta charset="utf-8">
<title>ChartVisualizer</title>
<style>
* { box-sizing:border-box; margin:0; padding:0; }
body { background:#111; color:#eee; font-family:'Segoe UI',sans-serif; }

/* ── Navbar ── */
.navbar {
    background:#161616; border-bottom:1px solid #2a2a2a;
    padding:0 20px; height:44px;
    display:flex; align-items:center; justify-content:space-between;
}
.navbar-brand {
    font-size:1rem; font-weight:600; color:#eee; letter-spacing:0.3px;
    display:flex; align-items:center; gap:8px;
}
.navbar-brand .brand-dot { width:8px; height:8px; border-radius:50%; background:#1976d2; display:inline-block; }
.navbar-links { display:flex; gap:8px; align-items:center; }

/* ── Controls bar ── */
.controls-bar {
    background:#1a1a1a; border-bottom:1px solid #2a2a2a;
    padding:8px 20px; display:flex; align-items:flex-end; gap:20px; flex-wrap:wrap;
}
.ctrl-group { display:flex; gap:10px; align-items:flex-end; }
.ctrl-group + .ctrl-group { border-left:1px solid #2a2a2a; padding-left:20px; }
.ctrl-sep { width:1px; background:#2a2a2a; align-self:stretch; }
.ctrl-divider { border-left:1px solid #2a2a2a; height:28px; align-self:center; }
.ctrl-item { display:flex; flex-direction:column; gap:3px; }

/* ── Status strip ── */
.status-strip {
    background:#141414; border-bottom:1px solid #222;
    padding:3px 20px; font-size:0.72rem; color:#555;
    display:flex; align-items:center; gap:16px;
}
#chart-refresh-status { color:#555; }

select, input[type=number] {
    background:#242424; color:#eee; border:1px solid #383838;
    padding:5px 9px; border-radius:4px; font-size:0.84rem;
    transition: border-color 0.15s;
}
select:focus, input[type=number]:focus { outline:none; border-color:#1976d2; }
.btn {
    background:#1976d2; color:#fff; border:none;
    padding:6px 14px; border-radius:4px; cursor:pointer;
    font-size:0.83rem; text-decoration:none; display:inline-flex;
    align-items:center; gap:5px; transition:background 0.15s;
    white-space:nowrap;
}
.btn:hover { background:#1565c0; }
.btn-multi { background:#6a1b9a; }
.btn-multi:hover { background:#4a148c; }
.btn-active { outline:2px solid #ce93d8; }
.btn-decide { background:#37474f; }
.btn-decide:hover { background:#455a64; }
.btn-autotrader { background:#1565c0; }
.btn-autotrader:hover { background:#0d47a1; }
.btn-analyze { background:#00838f; }
.btn-analyze:hover { background:#006064; }
.btn-trade-buy  { background:#1b5e20; color:#a5d6a7; font-weight:bold; }
.btn-trade-sell { background:#b71c1c; color:#ef9a9a; font-weight:bold; }
label { font-size:0.75rem; color:#666; display:block; margin-bottom:2px; letter-spacing:0.2px; text-transform:uppercase; }

.tf-checks { display:flex; gap:4px; flex-wrap:wrap; align-items:center; }
.tf-check-item {
    display:flex; align-items:center; gap:4px;
    background:#242424; border:1px solid #383838; border-radius:4px;
    padding:4px 8px; cursor:pointer; font-size:0.8rem; color:#bbb; user-select:none;
    transition:background 0.15s, border-color 0.15s;
}
.tf-check-item input { display:none; }
.tf-check-item.checked { background:#4a148c; border-color:#9c27b0; color:#fff; }
.tf-bars-row { display:flex; gap:5px; flex-wrap:wrap; }
.tf-bars-item { display:flex; flex-direction:column; align-items:center; gap:1px; }
.tf-bars-item span { font-size:0.7rem; color:#666; text-transform:uppercase; }
.tf-bars-item input { width:58px; text-align:center; padding:4px; }
.chart-container { padding:10px 16px; }

/* Panoul de analiza automata */
#analyze-panel {
    background:#1a1a1a; border:1px solid #333; border-radius:6px;
    margin:10px 16px; padding:14px 18px; display:none;
}
#analyze-panel h3 { font-size:0.95rem; color:#ccc; margin-bottom:10px; }
.tf-vote-table { width:100%; border-collapse:collapse; font-size:0.83rem; margin-bottom:12px; }
.tf-vote-table th { color:#888; font-weight:400; padding:4px 8px; border-bottom:1px solid #333; text-align:left; }
.tf-vote-table td { padding:5px 8px; border-bottom:1px solid #222; }
.sig-buy  { color:#26a69a; font-weight:bold; }
.sig-sell { color:#ef5350; font-weight:bold; }
.sig-hold { color:#888; }
.verdict-box {
    background:#222; border-radius:6px; padding:12px 16px;
    display:flex; align-items:center; gap:20px; flex-wrap:wrap;
}
.verdict-big { font-size:1.4rem; font-weight:bold; }
.verdict-buy  { color:#26a69a; }
.verdict-sell { color:#ef5350; }
.verdict-hold { color:#888; }
.verdict-detail { font-size:0.82rem; color:#aaa; line-height:1.6; }
.verdict-detail b { color:#ccc; }
#trade-result { margin-top:10px; font-size:0.85rem; padding:8px 12px; border-radius:4px; display:none; }
.trade-ok   { background:#1b5e20; color:#a5d6a7; }
.trade-err  { background:#b71c1c; color:#ef9a9a; }
/* Manual trade */
#manual-panel {
    margin-top:12px; padding:12px 14px; background:#1e1e2e;
    border:1px solid #444; border-radius:6px; display:none;
}
#manual-panel h4 { font-size:0.85rem; color:#888; margin-bottom:10px; font-weight:400; }
.manual-row { display:flex; gap:10px; align-items:flex-end; flex-wrap:wrap; }
.manual-field { display:flex; flex-direction:column; gap:3px; }
.manual-field label { font-size:0.78rem; color:#888; }
.manual-field input { background:#2a2a2a; color:#eee; border:1px solid #555; padding:5px 8px; border-radius:4px; width:110px; font-size:0.85rem; }
.manual-field input:focus { outline:1px solid #9c27b0; }
</style>
<script>
async function decideNow(symbol) {
    window.open(`/autotrader?symbol=${symbol}&decide=1`, '_blank');
}

function toggleTf(el) {
    el.classList.toggle('checked');
    el.querySelector('input').checked = el.classList.contains('checked');
}

async function runAnalysis() {
    const symbol  = document.querySelector('[name=symbol]').value;

    // multi mode: ia checkboxurile bifate; single mode: foloseste M1,M5,M15,H1,H4 implicit
    let checked = [...document.querySelectorAll('.tf-check-item.checked input')].map(i=>i.value);
    if(!checked.length) checked = ['M1','M5','M15','H1','H4'];

    // bars: din inputurile per-TF (multi) sau din inputul bars (single)
    const barsInputs = {};
    const singleBars = document.querySelector('[name=bars]');
    const singleVal  = singleBars ? singleBars.value : '500';
    document.querySelectorAll('.tf-bars-item input').forEach(inp => {
        barsInputs[inp.dataset.tf] = inp.value;
    });
    const barsParam = checked.map(tf => `bars_${tf}=${barsInputs[tf]||singleVal}`).join('&');
    const url = `/analyze?symbol=${symbol}&tfs=${checked.join(',')}&${barsParam}`;

    document.getElementById('analyze-btn').textContent = 'Se analizeaza...';
    document.getElementById('analyze-btn').disabled = true;

    try {
        const resp = await fetch(url);
        const data = await resp.json();
        showResult(data);
        // Reîncarcă graficele în fundal pentru a fi sincronizate cu analiza
        document.getElementById('chart-age').textContent = 'Grafic: sinc cu analiza ✓';
        document.getElementById('chart-age').style.color = '#26a69a';
    } catch(e) {
        alert('Eroare la analiza: ' + e);
    } finally {
        document.getElementById('analyze-btn').textContent = 'Analizeaza';
        document.getElementById('analyze-btn').disabled = false;
    }
}

function showResult(data) {
    const panel = document.getElementById('analyze-panel');
    panel.style.display = 'block';

    // tabel TF
    let rows = '';
    for(const r of data.all) {
        const cls = r.signal==='BUY'?'sig-buy':r.signal==='SELL'?'sig-sell':'sig-hold';
        const trend_icon = r.trend==='ASCENDING'?'▲':r.trend==='DESCENDING'?'▼':'—';
        rows += `<tr>
            <td><b>${r.tf}</b></td>
            <td>${trend_icon} ${r.trend}</td>
            <td class="${cls}">${r.signal}</td>
            <td>${'★'.repeat(r.conviction)}${'☆'.repeat(Math.max(0,4-r.conviction))}</td>
            <td style="color:#888;font-size:0.78rem">${r.reasons.join(', ')}</td>
        </tr>`;
    }
    document.getElementById('tf-rows').innerHTML = rows;

    // verdict
    const sig  = data.signal;
    const cls  = sig==='BUY'?'verdict-buy':sig==='SELL'?'verdict-sell':'verdict-hold';
    const icon = sig==='BUY'?'▲ BUY':sig==='SELL'?'▼ SELL':'— HOLD';
    document.getElementById('verdict-sig').className = 'verdict-big ' + cls;
    document.getElementById('verdict-sig').textContent = icon;
    document.getElementById('verdict-votes').textContent =
        `${data.n_buy} BUY  /  ${data.n_sell} SELL  /  ${data.n_total - data.n_buy - data.n_sell} HOLD`;

    const tradeBtn = document.getElementById('trade-btn');
    document.getElementById('trade-result').style.display = 'none';

    // pre-fill SL/TP manual din cel mai bun TF (daca exista)
    const best = data.best || (data.all && data.all.length ? data.all.reduce((a,b)=>a.conviction>b.conviction?a:b) : null);
    if(best) {
        document.getElementById('manual-sl').value = best.sl;
        document.getElementById('manual-tp').value = best.tp;
    }
    document.getElementById('manual-symbol').value = data.symbol;
    document.getElementById('manual-panel').style.display = 'block';

    if(data.best && sig !== 'HOLD') {
        const b = data.best;
        document.getElementById('verdict-detail').innerHTML =
            `<b>Cel mai convingator TF:</b> ${b.tf}  (${b.conviction} conditii)<br>
             <b>Pret:</b> ${b.price}  &nbsp; <b>SL:</b> ${b.sl}  &nbsp; <b>TP:</b> ${b.tp}<br>
             <b>Risc:</b> $${{{ RISK_DOLLARS }}} &nbsp; <b>Lotaj:</b> calculat automat la executie`;
        tradeBtn.style.display = 'inline-block';
        tradeBtn.className = sig==='BUY'?'btn btn-trade-buy':'btn btn-trade-sell';
        tradeBtn.textContent = sig==='BUY' ? '▲ Executa BUY' : '▼ Executa SELL';
        tradeBtn.onclick = () => executeTrade(data.symbol, sig, b.sl, b.tp);
    } else {
        document.getElementById('verdict-detail').innerHTML =
            `<span style='color:#888'>Conditii insuficiente pentru trade automat. Minim ${{{ MIN_TF_VOTES }}} TF-uri trebuie sa fie de acord.</span>`;
        tradeBtn.style.display = 'none';
    }
}

async function executeTrade(symbol, signal, sl, tp) {
    if(!confirm(`Executa ${signal} pe ${symbol}?  SL=${sl}  TP=${tp}  Risc=$${{{ RISK_DOLLARS }}}`)) return;
    const url = `/trade?symbol=${symbol}&signal=${signal}&sl=${sl}&tp=${tp}`;
    const resp = await fetch(url, {method:'POST'});
    const data = await resp.json();
    const box = document.getElementById('trade-result');
    box.style.display = 'block';
    box.className = data.ok ? 'trade-ok' : 'trade-err';
    box.innerHTML = data.message;
    if(!data.ok) {
        // arata butoanele de fortat langa eroare
        document.getElementById('force-trade-bar').style.display = 'flex';
        window._lastSymbol = symbol;
        window._lastSl = sl;
        window._lastTp = tp;
    }
}

// ── Live updates ──────────────────────────────────────────────────────────────
const CHART_REFRESH_S    = 5;   // grafic complet
const ANALYSIS_REFRESH_S = 10;  // analiza multi-TF

let _chartCountdown    = CHART_REFRESH_S;
let _analysisCountdown = ANALYSIS_REFRESH_S;
let _prevBid = null;

function currentSymbol() {
    return document.querySelector('[name=symbol]')?.value || 'EURUSD';
}

function buildChartUrl() {
    const params = new URLSearchParams(window.location.search);
    params.set('symbol', currentSymbol());
    return '/chart_html?' + params.toString();
}

// Ticker preț live (1s)
async function updateTicker() {
    try {
        const r = await fetch(`/tick?symbol=${currentSymbol()}`);
        const d = await r.json();
        if (!d.bid) return;
        document.getElementById('tick-symbol').textContent = d.symbol;
        const bidEl = document.getElementById('tick-bid');
        const askEl = document.getElementById('tick-ask');
        bidEl.textContent = 'BID ' + d.bid;
        askEl.textContent = 'ASK ' + d.ask;
        document.getElementById('tick-spread').textContent = d.spread + ' pips spread';
        // flash la schimbare pret
        if (_prevBid !== null && _prevBid !== d.bid) {
            const col = d.bid > _prevBid ? '#26a69a' : '#ef5350';
            bidEl.style.color = col; askEl.style.color = col;
            setTimeout(() => { bidEl.style.color='#ef5350'; askEl.style.color='#26a69a'; }, 400);
        }
        _prevBid = d.bid;
    } catch(e) {}
}

// Refresh grafic
async function refreshChart() {
    const container = document.querySelector('.chart-container');
    if (!container) return;
    const el = document.getElementById('chart-refresh-status');
    if (el) { el.textContent = '↻ se incarca...'; el.style.color = '#ff9800'; }
    try {
        const resp = await fetch(buildChartUrl());
        if (resp.ok) {
            container.innerHTML = await resp.text();
            if (el) { el.textContent = '↻ grafic live'; el.style.color = '#26a69a'; }
        }
    } catch(e) {
        if (el) { el.textContent = '↻ eroare'; el.style.color = '#ef5350'; }
    }
}

// Analiza automata
async function autoAnalysis() {
    const el = document.getElementById('analysis-status');
    if (el) { el.textContent = '◎ analizez...'; el.style.color = '#ff9800'; }
    try {
        await runAnalysis();
        if (el) { el.textContent = '◎ analiza live'; el.style.color = '#26a69a'; }
    } catch(e) {
        if (el) { el.style.color = '#555'; }
    }
}

// Loop principal 1s
function liveLoop() {
    updateTicker();
    _chartCountdown--;
    _analysisCountdown--;

    const chartEl    = document.getElementById('chart-refresh-status');
    const analysisEl = document.getElementById('analysis-status');

    if (chartEl && _chartCountdown > 0)
        chartEl.textContent = `↻ grafic ${_chartCountdown}s`;
    if (analysisEl && _analysisCountdown > 0)
        analysisEl.textContent = `◎ analiza ${_analysisCountdown}s`;

    if (_chartCountdown <= 0) {
        _chartCountdown = CHART_REFRESH_S;
        refreshChart();
    }
    if (_analysisCountdown <= 0) {
        _analysisCountdown = ANALYSIS_REFRESH_S;
        autoAnalysis();
    }
}

// Porneste live updates daca exista grafic incarcat
window.addEventListener('load', () => {
    if (document.querySelector('.chart-container > div')) {
        setInterval(liveLoop, 1000);
        liveLoop();
    } else {
        // Doar ticker, fara grafic
        setInterval(updateTicker, 1000);
        updateTicker();
    }
});

async function forceManual(signal) {
    const symbol = document.getElementById('manual-symbol').value ||
                   document.querySelector('[name=symbol]').value;
    const sl = parseFloat(document.getElementById('manual-sl').value);
    const tp = parseFloat(document.getElementById('manual-tp').value);
    if(!sl || !tp) { alert('Completeaza SL si TP in campurile de jos'); return; }
    const url = `/trade?symbol=${symbol}&signal=${signal}&sl=${sl}&tp=${tp}`;
    const resp = await fetch(url, {method:'POST'});
    const data = await resp.json();
    const box = document.getElementById('trade-result');
    box.style.display = 'block';
    box.className = data.ok ? 'trade-ok' : 'trade-err';
    box.innerHTML = data.message;
}

async function executeManual(signal) {
    const symbol = document.getElementById('manual-symbol').value;
    const sl     = parseFloat(document.getElementById('manual-sl').value);
    const tp     = parseFloat(document.getElementById('manual-tp').value);
    if(!symbol || isNaN(sl) || isNaN(tp) || sl===0 || tp===0) {
        alert('Completeaza SL si TP inainte de executie.');
        return;
    }
    if(!confirm(`TRADE MANUAL: ${signal} ${symbol}\nSL=${sl}  TP=${tp}  Risc=$${{{ RISK_DOLLARS }}}\n\nConfirmi?`)) return;
    const url = `/trade?symbol=${symbol}&signal=${signal}&sl=${sl}&tp=${tp}`;
    const resp = await fetch(url, {method:'POST'});
    const data = await resp.json();
    const box = document.getElementById('trade-result');
    box.style.display = 'block';
    box.className = data.ok ? 'trade-ok' : 'trade-err';
    box.textContent = data.message;
}
</script>
</head><body>

<!-- ── Navbar ── -->
<div class="navbar">
    <div class="navbar-brand">
        <span class="brand-dot"></span>
        ChartVisualizer
    </div>
    <div style="display:flex;align-items:center;gap:20px">
        <div id="live-ticker" style="font-size:0.82rem;color:#aaa;display:flex;gap:14px;align-items:center">
            <span id="tick-symbol" style="color:#666;font-size:0.75rem"></span>
            <span id="tick-bid" style="color:#ef5350;font-weight:600"></span>
            <span id="tick-ask" style="color:#26a69a;font-weight:600"></span>
            <span id="tick-spread" style="color:#555;font-size:0.72rem"></span>
        </div>
        <div class="navbar-links">
            <a href="/autotrader" class="btn btn-autotrader" style="font-size:0.8rem;padding:5px 12px">⚡ AutoTrader</a>
            <a href="/trades" class="btn" style="background:#00695c;font-size:0.8rem;padding:5px 12px">📊 Trades</a>
            <a href="/news" class="btn" style="background:#6a1b9a;font-size:0.8rem;padding:5px 12px">📰 Stiri</a>
            <a href="/account" class="btn" style="background:#333;color:#bbb;font-size:0.8rem;padding:5px 12px">Cont MT5</a>
        </div>
    </div>
</div>

<!-- ── Controls bar ── -->
<form method="get">
<input type="hidden" name="mode" value="{{ mode }}">
<div class="controls-bar">

    <!-- Grup 1: Simbol + TF + Bare -->
    <div class="ctrl-group">
        <div class="ctrl-item">
            <label>Simbol</label>
            <select name="symbol">
                {% for s in symbols %}
                <option value="{{ s }}" {% if s == symbol %}selected{% endif %}>{{ s }}</option>
                {% endfor %}
            </select>
        </div>

        {% if mode == "single" %}
        <div class="ctrl-item">
            <label>Timeframe</label>
            <select name="tf">
                {% for t in all_tfs %}
                <option value="{{ t }}" {% if t == tf %}selected{% endif %}>{{ t }}</option>
                {% endfor %}
            </select>
        </div>
        <div class="ctrl-item">
            <label>Lumânări</label>
            <input type="number" name="bars" value="{{ bars }}" min="50" max="5000" step="50" style="width:78px">
        </div>
        {% else %}
        <div class="ctrl-item">
            <label>TF-uri afisate</label>
            <div class="tf-checks">
                {% for t in all_tfs %}
                <label class="tf-check-item {% if t in selected_tfs %}checked{% endif %}" onclick="toggleTf(this)">
                    <input type="checkbox" name="mtf" value="{{ t }}" {% if t in selected_tfs %}checked{% endif %}>
                    {{ t }}
                </label>
                {% endfor %}
            </div>
        </div>
        <div class="ctrl-item">
            <label>Lumânări / TF</label>
            <div class="tf-bars-row">
                {% for t in all_tfs %}
                <div class="tf-bars-item">
                    <span>{{ t }}</span>
                    <input type="number" name="bars_{{ t }}" data-tf="{{ t }}" value="{{ bars_map[t] }}" min="50" max="5000" step="50">
                </div>
                {% endfor %}
            </div>
        </div>
        {% endif %}
    </div>

    <!-- Grup 2: Actiuni grafic -->
    <div class="ctrl-group">
        <button type="submit" class="btn">↻ Incarca</button>
        {% if mode == "single" %}
        <a href="?mode=multi&symbol={{ symbol }}" class="btn btn-multi">▼ Multi-TF</a>
        {% else %}
        <a href="?mode=single&symbol={{ symbol }}&tf=M5&bars=500" class="btn btn-multi btn-active">▲ Single-TF</a>
        {% endif %}
        <button type="button" class="btn btn-decide" onclick="decideNow('{{ symbol }}')">⚡ Decide</button>
    </div>

    <!-- Grup 3: Analiza + status live -->
    <div class="ctrl-group">
        <div class="ctrl-item">
            <label>Analiza auto</label>
            <button id="analyze-btn" type="button" class="btn btn-analyze" onclick="runAnalysis()">Analizeaza</button>
        </div>
        <div class="ctrl-item" style="justify-content:flex-end;gap:4px">
            <span id="chart-refresh-status" style="font-size:0.72rem;color:#555;white-space:nowrap">↻ grafic</span>
            <span id="analysis-status"      style="font-size:0.72rem;color:#555;white-space:nowrap">◎ analiza</span>
        </div>
    </div>

</div>
</form>

<!-- Panou rezultate analiza -->
<div id="analyze-panel">
    <h3>Analiza Multi-TF</h3>
    <table class="tf-vote-table">
        <thead><tr>
            <th>TF</th><th>Trend</th><th>Semnal</th><th>Convingere</th><th>Motive</th>
        </tr></thead>
        <tbody id="tf-rows"></tbody>
    </table>
    <div class="verdict-box">
        <span id="verdict-sig" class="verdict-big verdict-hold">—</span>
        <div>
            <div id="verdict-votes" style="color:#888;font-size:0.85rem;margin-bottom:4px"></div>
            <div id="verdict-detail" class="verdict-detail"></div>
        </div>
        <button id="trade-btn" class="btn" style="display:none">Executa</button>
    </div>
    <div id="trade-result"></div>

    <!-- Butoane fortat dupa eroare -->
    <div id="force-trade-bar" style="display:none;gap:8px;margin-top:8px;align-items:center;flex-wrap:wrap">
        <span style="font-size:0.8rem;color:#aaa">Incearca oricum:</span>
        <button class="btn btn-trade-buy"  onclick="forceManual('BUY')">▲ Forteaza BUY</button>
        <button class="btn btn-trade-sell" onclick="forceManual('SELL')">▼ Forteaza SELL</button>
        <span style="font-size:0.75rem;color:#666">SL/TP din campurile de jos</span>
    </div>

    <!-- Trade manual (mereu disponibil dupa analiza) -->
    <div id="manual-panel">
        <input type="hidden" id="manual-symbol">
        <h4>Trade manual — ignora semnalul automat</h4>
        <div class="manual-row">
            <div class="manual-field">
                <label>Stop Loss</label>
                <input type="number" id="manual-sl" placeholder="ex: 1.14200" step="0.00001">
            </div>
            <div class="manual-field">
                <label>Take Profit</label>
                <input type="number" id="manual-tp" placeholder="ex: 1.15000" step="0.00001">
            </div>
            <button class="btn btn-trade-buy"  onclick="executeManual('BUY')">▲ BUY manual</button>
            <button class="btn btn-trade-sell" onclick="executeManual('SELL')">▼ SELL manual</button>
        </div>
        <div style="font-size:0.75rem;color:#666;margin-top:6px">
            SL/TP pre-completat din analiza — poti modifica liber. Risc fix: $RISK_DOLLARS_VAL
        </div>
    </div>
</div>

<div class="chart-container">
    {% if chart %}
        {{ chart | safe }}
    {% else %}
        <p style="color:#888;font-size:0.85rem;padding:4px 0">Selecteaza un simbol si apasa Actualizeaza.</p>
    {% endif %}
</div>

</body></html>"""

# ── Routes ────────────────────────────────────────────────────────────────────
@app.route("/")
def index():
    symbol = request.args.get("symbol", "EURUSD").upper()
    mode   = request.args.get("mode", "single")
    tf     = request.args.get("tf", "M5")
    bars   = int(request.args.get("bars", 500))
    selected_tfs = request.args.getlist("mtf") or ["M1","M5","M15","H1","H4"]
    bars_map = {t: int(request.args.get(f"bars_{t}", 500)) for t in ALL_TFS}
    chart = None
    if request.args:
        if mode == "multi":
            chart = build_multi(symbol, selected_tfs, bars_map)
        else:
            chart = build_chart(symbol, tf, bars, compact=False)
    return render_template_string(
        HTML.replace("{{ RISK_DOLLARS }}", str(RISK_DOLLARS))
            .replace("{{ MIN_TF_VOTES }}", str(MIN_TF_VOTES))
            .replace("RISK_DOLLARS_VAL", str(int(RISK_DOLLARS))),
        symbols=SYMBOLS, all_tfs=ALL_TFS,
        symbol=symbol, mode=mode, tf=tf, bars=bars,
        selected_tfs=selected_tfs, bars_map=bars_map, chart=chart,
    )

@app.route("/ftmo_status")
def ftmo_status():
    from datetime import datetime, timezone
    ok, msg = check_ftmo_rules()
    acc = mt5.account_info() if MT5_AVAILABLE and mt5 else None
    now_utc = datetime.now(timezone.utc)
    # urmatoarea stire rosie din cache ForexFactory
    upcoming_news = get_upcoming_red_news(minutes_ahead=120)
    next_news = None
    if upcoming_news:
        ev = upcoming_news[0]
        next_news = {"time": ev.get("dt", ""), "in_minutes": round(ev["in_minutes"])}
    elif True:
        # cauta urmatoarea stire (chiar si peste 2 min)
        all_red = fetch_red_news()
        from datetime import datetime, timezone as _tz
        _now = datetime.now(_tz.utc)
        for ev in sorted(all_red, key=lambda x: x["dt"]):
            diff_m = (ev["dt"] - _now).total_seconds() / 60
            if diff_m > 0:
                next_news = {"time": ev["dt"].strftime("%H:%M UTC"), "in_minutes": round(diff_m)}
                break
    return Response(json.dumps({
        "ok":           ok,
        "message":      msg,
        "ftmo_enabled": FTMO_ENABLED,
        "balance":      round(acc.balance, 2)  if acc else 0,
        "equity":       round(acc.equity,  2)  if acc else 0,
        "daily_floor":  round(acc.balance * (1 - FTMO_DAILY_LOSS_PCT), 2) if acc else 0,
        "daily_used_pct": round((1 - acc.equity / acc.balance) * 100, 2) if acc and acc.balance > 0 else 0,
        "next_news":    next_news,
        "time_utc":     now_utc.strftime("%H:%M:%S"),
        "weekday":      now_utc.strftime("%A"),
    }), mimetype="application/json")

@app.route("/tick")
def tick_route():
    symbol = request.args.get("symbol", "EURUSD").upper()
    if not MT5_AVAILABLE or mt5 is None:
        return Response(json.dumps({}), mimetype="application/json")
    tick = mt5.symbol_info_tick(symbol)
    if tick is None:
        return Response(json.dumps({}), mimetype="application/json")
    info = mt5.symbol_info(symbol)
    digits = info.digits if info else 5
    spread = round((tick.ask - tick.bid) / (10 ** -digits)) if info else 0
    return Response(json.dumps({
        "symbol": symbol,
        "bid": round(tick.bid, digits),
        "ask": round(tick.ask, digits),
        "spread": int(spread),
    }), mimetype="application/json")

@app.route("/chart_html")
def chart_html_route():
    symbol = request.args.get("symbol", "EURUSD").upper()
    mode   = request.args.get("mode", "single")
    tf     = request.args.get("tf", "M5")
    bars   = int(request.args.get("bars", 500))
    selected_tfs = request.args.getlist("mtf") or ["M1","M5","M15","H1","H4"]
    bars_map = {t: int(request.args.get(f"bars_{t}", 500)) for t in ALL_TFS}
    if mode == "multi":
        html = build_multi(symbol, selected_tfs, bars_map)
    else:
        html = build_chart(symbol, tf, bars, compact=False)
    return Response(html, mimetype="text/html")

@app.route("/analyze")
def analyze_route():
    symbol  = request.args.get("symbol", "EURUSD").upper()
    tfs_str = request.args.get("tfs", "M1,M5,M15,H1,H4")
    tfs     = [t.strip() for t in tfs_str.split(",") if t.strip()]
    bars_map = {t: int(request.args.get(f"bars_{t}", 500)) for t in tfs}
    results = []
    for tf in tfs:
        r = get_signal_data(symbol, tf, bars_map.get(tf, 500))
        if r:
            results.append(r)
    buy_v  = [r for r in results if r["signal"] == "BUY"]
    sell_v = [r for r in results if r["signal"] == "SELL"]
    n_buy, n_sell = len(buy_v), len(sell_v)
    final = "HOLD"
    best  = None
    if n_buy >= MIN_TF_VOTES and n_buy > n_sell:
        final = "BUY"
        best  = max(buy_v,  key=lambda x: x["conviction"])
    elif n_sell >= MIN_TF_VOTES and n_sell > n_buy:
        final = "SELL"
        best  = max(sell_v, key=lambda x: x["conviction"])
    data = {"symbol": symbol, "signal": final,
            "n_buy": n_buy, "n_sell": n_sell, "n_total": len(results),
            "best": best, "all": results}
    return Response(json.dumps(data, cls=NpEncoder), mimetype="application/json")

@app.route("/debug_trade")
def debug_trade():
    symbol = request.args.get("symbol", "EURUSD").upper()
    if not MT5_AVAILABLE or mt5 is None:
        return Response(json.dumps({"error": "MT5 indisponibil"}), mimetype="application/json")
    try:
        info = mt5.symbol_info(symbol)
        tick = mt5.symbol_info_tick(symbol)
        term = mt5.terminal_info()
        acc  = mt5.account_info()
        data = {
            "symbol":        symbol,
            "filling_mode":  int(info.filling_mode) if info else None,
            "trade_mode":    int(info.trade_mode)   if info else None,
            "digits":        int(info.digits)        if info else None,
            "ask":           float(tick.ask)         if tick else None,
            "bid":           float(tick.bid)         if tick else None,
            "term_trade_allowed":    bool(term.trade_allowed)    if term else None,
            "acc_trade_allowed":     bool(acc.trade_allowed)     if acc else None,
            "acc_trade_expert":      bool(acc.trade_expert)      if acc else None,
            "ORDER_FILLING_FOK":     int(mt5.ORDER_FILLING_FOK),
            "ORDER_FILLING_IOC":     int(mt5.ORDER_FILLING_IOC),
            "ORDER_FILLING_RETURN":  int(mt5.ORDER_FILLING_RETURN),
        }
    except Exception as e:
        data = {"error": str(e)}
    return Response(json.dumps(data), mimetype="application/json")

@app.route("/mt5_status")
def mt5_status():
    if not MT5_AVAILABLE or mt5 is None:
        return Response(json.dumps({"ok": False, "msg": "MT5 nu e disponibil"}), mimetype="application/json")
    info = mt5.terminal_info()
    acc  = mt5.account_info()
    data = {
        "ok": True,
        "connected": info.connected if info else False,
        "trade_allowed": info.trade_allowed if info else False,
        "balance": acc.balance if acc else 0,
        "equity": acc.equity if acc else 0,
        "server": acc.server if acc else "",
        "login": acc.login if acc else 0,
    }
    return Response(json.dumps(data), mimetype="application/json")

@app.route("/trade", methods=["POST"])
def trade_route():
    symbol = request.args.get("symbol", "EURUSD").upper()
    signal = request.args.get("signal", "")
    sl     = float(request.args.get("sl", 0))
    tp     = float(request.args.get("tp", 0))
    if signal not in ("BUY", "SELL") or sl == 0 or tp == 0:
        return Response(json.dumps({"ok": False, "message": "Parametri invalizi"}),
                        mimetype="application/json")
    ok, msg = place_trade(symbol, signal, sl, tp, RISK_DOLLARS)
    return Response(json.dumps({"ok": ok, "message": msg}), mimetype="application/json")

ACCOUNT_HTML = """<!DOCTYPE html>
<html><head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Cont MT5</title>
<style>
* { box-sizing:border-box; margin:0; padding:0; }
body { background:#111; color:#eee; font-family:'Segoe UI',monospace; padding:16px; }
h2 { font-size:1rem; color:#888; font-weight:400; margin-bottom:14px; }
.grid { display:grid; grid-template-columns:repeat(auto-fit,minmax(160px,1fr)); gap:10px; margin-bottom:20px; }
.card { background:#1a1a1a; border:1px solid #333; border-radius:8px; padding:14px 16px; }
.card .label { font-size:0.75rem; color:#888; margin-bottom:4px; }
.card .value { font-size:1.3rem; font-weight:bold; color:#eee; }
.card .value.green { color:#26a69a; }
.card .value.red   { color:#ef5350; }
.card .value.yellow{ color:#ffeb3b; }
table { width:100%; border-collapse:collapse; font-size:0.83rem; }
th { color:#888; font-weight:400; padding:6px 10px; border-bottom:1px solid #333; text-align:left; }
td { padding:7px 10px; border-bottom:1px solid #1e1e1e; }
.buy  { color:#26a69a; font-weight:bold; }
.sell { color:#ef5350; font-weight:bold; }
.section { background:#1a1a1a; border:1px solid #333; border-radius:8px; padding:14px 16px; margin-bottom:14px; }
.section h3 { font-size:0.85rem; color:#888; font-weight:400; margin-bottom:10px; }
.close-btn { background:#b71c1c; color:#ef9a9a; border:none; padding:4px 10px; border-radius:4px; cursor:pointer; font-size:0.78rem; }
.close-btn:hover { background:#c62828; }
#msg { margin-top:10px; padding:8px 12px; border-radius:4px; display:none; font-size:0.85rem; }
.ok  { background:#1b5e20; color:#a5d6a7; }
.err { background:#b71c1c; color:#ef9a9a; }
.refresh { color:#1976d2; font-size:0.8rem; cursor:pointer; text-decoration:underline; margin-left:10px; }
a.back { color:#888; font-size:0.82rem; text-decoration:none; display:inline-block; margin-bottom:14px; }
a.back:hover { color:#ccc; }
.login-section { background:#1a1a1a; border:1px solid #444; border-radius:8px; padding:16px 20px; margin-bottom:16px; }
.login-section h3 { font-size:0.9rem; color:#aaa; margin-bottom:12px; font-weight:400; }
.login-row { display:flex; gap:10px; flex-wrap:wrap; align-items:flex-end; }
.login-field { display:flex; flex-direction:column; gap:4px; }
.login-field label { font-size:0.74rem; color:#777; text-transform:uppercase; }
.login-field input { background:#242424; color:#eee; border:1px solid #444; padding:6px 10px; border-radius:4px; font-size:0.85rem; width:160px; }
.login-field input:focus { outline:none; border-color:#1976d2; }
.btn-connect { background:#1976d2; color:#fff; border:none; padding:7px 18px; border-radius:4px; cursor:pointer; font-size:0.85rem; }
.btn-connect:hover { background:#1565c0; }
#login-msg { margin-top:10px; padding:7px 12px; border-radius:4px; display:none; font-size:0.84rem; }
</style>
<script>
let autoRefresh = null;

function startAutoRefresh() {
    if(autoRefresh) clearInterval(autoRefresh);
    autoRefresh = setInterval(loadData, 1000);
}

async function loadData() {
    try {
        const resp = await fetch('/account_data');
        const d = await resp.json();
        if(!d.ok) {
            // arata formularul de login daca MT5 nu e conectat
            document.getElementById('login-section').style.display = 'block';
            document.getElementById('content').style.display = 'none';
            document.getElementById('account-title').textContent = 'neconectat';
            return;
        }
        // ascunde login daca e conectat
        document.getElementById('login-section').style.display = 'none';
        document.getElementById('content').style.display = 'block';

        // cards
        const pl_color = d.profit >= 0 ? 'green' : 'red';
        const eq_color = d.equity >= d.balance ? 'green' : 'red';
        document.getElementById('card-balance').textContent  = d.balance.toFixed(2) + ' ' + d.currency;
        document.getElementById('card-equity').textContent   = d.equity.toFixed(2)  + ' ' + d.currency;
        document.getElementById('card-equity').className     = 'value ' + eq_color;
        document.getElementById('card-margin').textContent   = d.margin.toFixed(2)   + ' ' + d.currency;
        document.getElementById('card-free').textContent     = d.free_margin.toFixed(2) + ' ' + d.currency;
        document.getElementById('card-profit').textContent   = (d.profit >= 0 ? '+' : '') + d.profit.toFixed(2) + ' ' + d.currency;
        document.getElementById('card-profit').className     = 'value ' + pl_color;
        document.getElementById('card-positions').textContent = d.positions.length;
        document.getElementById('card-level').textContent    = d.margin_level > 0 ? d.margin_level.toFixed(0) + '%' : '—';
        document.getElementById('account-title').textContent = d.login + ' · ' + d.server;

        // pozitii
        let rows = '';
        if(d.positions.length === 0) {
            rows = '<tr><td colspan="8" style="color:#666;text-align:center">Nicio pozitie deschisa</td></tr>';
        } else {
            for(const p of d.positions) {
                const pc = p.type==='BUY' ? 'buy' : 'sell';
                const plc = p.profit >= 0 ? 'green' : 'red';
                rows += `<tr>
                    <td><b>${p.symbol}</b></td>
                    <td class="${pc}">${p.type}</td>
                    <td>${p.volume}</td>
                    <td>${p.price_open}</td>
                    <td>${p.price_current}</td>
                    <td>${p.sl || '—'}</td>
                    <td>${p.tp || '—'}</td>
                    <td style="color:${p.profit>=0?'#26a69a':'#ef5350'};font-weight:bold">
                        ${p.profit>=0?'+':''}${p.profit.toFixed(2)}
                    </td>
                    <td><button class="close-btn" onclick="closePos(${p.ticket})">✕</button></td>
                </tr>`;
            }
        }
        document.getElementById('pos-rows').innerHTML = rows;
        document.getElementById('last-update').textContent = 'Actualizat: ' + new Date().toLocaleTimeString();
    } catch(e) {
        console.error(e);
    }
}

async function mt5Login() {
    const login    = document.getElementById('inp-login').value.trim();
    const password = document.getElementById('inp-pass').value.trim();
    const server   = document.getElementById('inp-server').value.trim();
    const path     = document.getElementById('inp-path').value.trim();
    if (!login || !password || !server) { alert('Completeaza Login, Parola si Server'); return; }
    const btn = document.getElementById('btn-login');
    btn.textContent = 'Se conecteaza...'; btn.disabled = true;
    try {
        const resp = await fetch('/mt5_login', {
            method: 'POST',
            headers: {'Content-Type':'application/json'},
            body: JSON.stringify({login: parseInt(login), password, server, path})
        });
        const d = await resp.json();
        const msg = document.getElementById('login-msg');
        msg.style.display = 'block';
        msg.className = d.ok ? 'ok' : 'err';
        msg.textContent = d.message;
        if (d.ok) { loadData(); document.getElementById('login-section').style.display='none'; }
    } catch(e) { alert('Eroare: ' + e); }
    finally { btn.textContent = 'Conecteaza'; btn.disabled = false; }
}

async function closePos(ticket) {
    if(!confirm('Inchizi pozitia ' + ticket + '?')) return;
    const resp = await fetch('/close_position?ticket=' + ticket, {method:'POST'});
    const d = await resp.json();
    const msg = document.getElementById('msg');
    msg.style.display = 'block';
    msg.className = d.ok ? 'ok' : 'err';
    msg.textContent = d.message;
    loadData();
}

window.onload = () => { loadData(); startAutoRefresh(); };
</script>
</head><body>
<a class="back" href="/">← ChartVisualizer</a>
<h2>Cont MT5 — <span id="account-title">...</span>
    <span class="refresh" onclick="loadData()">↻ refresh</span>
    <span id="last-update" style="color:#555;font-size:0.75rem;margin-left:10px"></span>
</h2>

<!-- Login MT5 -->
<div class="login-section" id="login-section">
    <h3>🔑 Conectare cont MT5</h3>
    <div class="login-row">
        <div class="login-field">
            <label>Login (nr. cont)</label>
            <input type="number" id="inp-login" placeholder="ex: 12345678">
        </div>
        <div class="login-field">
            <label>Parola</label>
            <input type="password" id="inp-pass" placeholder="parola cont">
        </div>
        <div class="login-field">
            <label>Server broker</label>
            <input type="text" id="inp-server" placeholder="ex: ICMarkets-Demo">
        </div>
        <div class="login-field">
            <label>Cale terminal (optional)</label>
            <input type="text" id="inp-path" placeholder="C:\\...\\terminal64.exe" style="width:220px">
        </div>
        <button class="btn-connect" id="btn-login" onclick="mt5Login()">Conecteaza</button>
    </div>
    <div id="login-msg"></div>
</div>

<div id="content">
    <div class="grid">
        <div class="card"><div class="label">Balance</div><div class="value" id="card-balance">...</div></div>
        <div class="card"><div class="label">Equity</div><div class="value" id="card-equity">...</div></div>
        <div class="card"><div class="label">Profit flotant</div><div class="value" id="card-profit">...</div></div>
        <div class="card"><div class="label">Margin</div><div class="value yellow" id="card-margin">...</div></div>
        <div class="card"><div class="label">Free Margin</div><div class="value" id="card-free">...</div></div>
        <div class="card"><div class="label">Margin Level</div><div class="value" id="card-level">...</div></div>
        <div class="card"><div class="label">Pozitii deschise</div><div class="value" id="card-positions">...</div></div>
    </div>

    <div class="section">
        <h3>Pozitii deschise</h3>
        <table>
            <thead><tr>
                <th>Symbol</th><th>Tip</th><th>Volum</th>
                <th>Deschis la</th><th>Pret curent</th>
                <th>SL</th><th>TP</th><th>Profit</th><th></th>
            </tr></thead>
            <tbody id="pos-rows"></tbody>
        </table>
        <div id="msg"></div>
    </div>
</div>
</body></html>"""

@app.route("/account")
def account_page():
    return ACCOUNT_HTML

@app.route("/mt5_login", methods=["POST"])
def mt5_login():
    global MT5_AVAILABLE
    import os
    body = request.get_json(silent=True) or {}
    login    = body.get("login")
    password = body.get("password", "")
    server   = body.get("server", "")
    path     = body.get("path", "").strip() or None

    if not login or not password or not server:
        return Response(json.dumps({"ok": False, "message": "Login, parola si server sunt obligatorii"}),
                        mimetype="application/json")
    if mt5 is None:
        return Response(json.dumps({"ok": False, "message": "Modulul MetaTrader5 nu e instalat"}),
                        mimetype="application/json")
    try:
        mt5.shutdown()
    except:
        pass
    try:
        kwargs = dict(login=int(login), password=str(password), server=str(server), timeout=30000)
        if path and os.path.exists(path):
            kwargs["path"] = path
        ok = mt5.initialize(**kwargs)
        if ok:
            MT5_AVAILABLE = True
            acc = mt5.account_info()
            name = acc.name if acc else ""
            bal  = acc.balance if acc else 0
            return Response(json.dumps({
                "ok": True,
                "message": f"Conectat: {name} | Cont #{login} | Balance: {bal:.2f}"
            }), mimetype="application/json")
        else:
            err = mt5.last_error()
            MT5_AVAILABLE = False
            return Response(json.dumps({
                "ok": False,
                "message": f"Eroare MT5: {err}"
            }), mimetype="application/json")
    except Exception as e:
        return Response(json.dumps({"ok": False, "message": str(e)}), mimetype="application/json")

@app.route("/account_data")
def account_data():
    if not MT5_AVAILABLE or mt5 is None:
        return Response(json.dumps({"ok": False, "msg": "MT5 indisponibil"}), mimetype="application/json")
    acc = mt5.account_info()
    if acc is None:
        return Response(json.dumps({"ok": False, "msg": "Nu s-a putut obtine info cont"}), mimetype="application/json")

    positions = []
    for i in range(mt5.positions_total()):
        p = mt5.positions_get()[i] if mt5.positions_total() > 0 else None
        if p is None: continue
        positions.append({
            "ticket":        int(p.ticket),
            "symbol":        p.symbol,
            "type":          "BUY" if p.type == 0 else "SELL",
            "volume":        p.volume,
            "price_open":    round(p.price_open, 5),
            "price_current": round(p.price_current, 5),
            "sl":            round(p.sl, 5) if p.sl else 0,
            "tp":            round(p.tp, 5) if p.tp else 0,
            "profit":        round(p.profit, 2),
            "swap":          round(p.swap, 2),
        })

    # re-fetch all positions properly
    all_pos = mt5.positions_get()
    positions = []
    if all_pos:
        for p in all_pos:
            positions.append({
                "ticket":        int(p.ticket),
                "symbol":        p.symbol,
                "type":          "BUY" if p.type == 0 else "SELL",
                "volume":        p.volume,
                "price_open":    round(p.price_open, 5),
                "price_current": round(p.price_current, 5),
                "sl":            round(p.sl, 5) if p.sl else 0,
                "tp":            round(p.tp, 5) if p.tp else 0,
                "profit":        round(p.profit, 2),
            })

    total_profit = sum(p["profit"] for p in positions)

    data = {
        "ok":           True,
        "login":        acc.login,
        "server":       acc.server,
        "currency":     acc.currency,
        "balance":      round(acc.balance, 2),
        "equity":       round(acc.equity, 2),
        "margin":       round(acc.margin, 2),
        "free_margin":  round(acc.margin_free, 2),
        "margin_level": round(acc.margin_level, 2) if acc.margin_level else 0,
        "profit":       round(total_profit, 2),
        "positions":    positions,
    }
    return Response(json.dumps(data, cls=NpEncoder), mimetype="application/json")

@app.route("/close_position", methods=["POST"])
def close_position():
    if not MT5_AVAILABLE or mt5 is None:
        return Response(json.dumps({"ok": False, "message": "MT5 indisponibil"}), mimetype="application/json")
    ticket = int(request.args.get("ticket", 0))
    if not mt5.positions_get(ticket=ticket):
        return Response(json.dumps({"ok": False, "message": "Pozitia nu exista"}), mimetype="application/json")
    pos = mt5.positions_get(ticket=ticket)[0]
    tick = mt5.symbol_info_tick(pos.symbol)
    close_price = tick.bid if pos.type == 0 else tick.ask
    info = mt5.symbol_info(pos.symbol)
    fm = info.filling_mode if info else 2
    if fm & 2:    filling = mt5.ORDER_FILLING_IOC
    elif fm & 1:  filling = mt5.ORDER_FILLING_FOK
    else:         filling = mt5.ORDER_FILLING_RETURN
    req = {
        "action":       mt5.TRADE_ACTION_DEAL,
        "symbol":       pos.symbol,
        "volume":       pos.volume,
        "type":         mt5.ORDER_TYPE_SELL if pos.type == 0 else mt5.ORDER_TYPE_BUY,
        "position":     ticket,
        "price":        close_price,
        "deviation":    30,
        "magic":        pos.magic,
        "comment":      "close",
        "type_time":    mt5.ORDER_TIME_GTC,
        "type_filling": filling,
    }
    result = mt5.order_send(req)
    if result and result.retcode == mt5.TRADE_RETCODE_DONE:
        return Response(json.dumps({"ok": True, "message": f"Pozitia {ticket} inchisa"}), mimetype="application/json")
    code = result.retcode if result else -1
    return Response(json.dumps({"ok": False, "message": f"Eroare inchidere: {code}"}), mimetype="application/json")

@app.route("/modify_trade", methods=["POST"])
def modify_trade():
    if not MT5_AVAILABLE or mt5 is None:
        return Response(json.dumps({"ok": False, "message": "MT5 indisponibil"}), mimetype="application/json")
    body   = request.get_json(silent=True) or {}
    ticket = int(body.get("ticket", 0))
    new_sl = float(body.get("sl", 0))
    new_tp = float(body.get("tp", 0))
    pos_list = mt5.positions_get(ticket=ticket)
    if not pos_list:
        return Response(json.dumps({"ok": False, "message": "Pozitia nu exista"}), mimetype="application/json")
    pos  = pos_list[0]
    info = mt5.symbol_info(pos.symbol)
    digits = info.digits if info else 5
    req = {
        "action":   mt5.TRADE_ACTION_SLTP,
        "position": ticket,
        "symbol":   pos.symbol,
        "sl":       round(new_sl, digits),
        "tp":       round(new_tp, digits),
    }
    result = mt5.order_send(req)
    if result and result.retcode == mt5.TRADE_RETCODE_DONE:
        return Response(json.dumps({"ok": True, "message": f"SL/TP modificat pe {pos.symbol}"}), mimetype="application/json")
    code = result.retcode if result else -1
    return Response(json.dumps({"ok": False, "message": f"Eroare modificare: {code}"}), mimetype="application/json")


@app.route("/trade_chart/<int:ticket>")
def trade_chart(ticket):
    if not MT5_AVAILABLE or mt5 is None:
        return Response("<p style='color:#ef5350'>MT5 indisponibil</p>", mimetype="text/html")
    pos_list = mt5.positions_get(ticket=ticket)
    if not pos_list:
        return Response("<p style='color:#ef5350'>Pozitia nu exista</p>", mimetype="text/html")
    pos    = pos_list[0]
    symbol = pos.symbol
    tf     = request.args.get("tf", "M5")
    bars   = int(request.args.get("bars", 200))

    df, source = fetch(symbol, tf, bars)
    if df is None:
        return Response("<p style='color:#ef5350'>Date indisponibile</p>", mimetype="text/html")

    tick = mt5.symbol_info_tick(symbol)
    price_now = float(tick.bid if pos.type == 0 else tick.ask) if tick else pos.price_current
    is_buy    = (pos.type == 0)
    col_entry = "#90caf9"
    col_sl    = "#ef5350"
    col_tp    = "#26a69a"
    col_price = "#ffeb3b"
    pl        = round(price_now - pos.price_open, 5) if is_buy else round(pos.price_open - price_now, 5)
    pl_pct    = round(pl / pos.price_open * 100, 3)

    # Calculeaza range Y bazat pe lumânari + linii trade (ignora valori aberante)
    candle_low  = float(df["low"].min())
    candle_high = float(df["high"].max())
    atr = float(df["high"].sub(df["low"]).rolling(14).mean().iloc[-1])
    # include SL si TP in range doar daca sunt rezonabile (in raza de 20xATR)
    y_points = [pos.price_open, price_now]
    if pos.sl and abs(pos.sl - price_now) < atr * 20: y_points.append(pos.sl)
    if pos.tp and abs(pos.tp - price_now) < atr * 20: y_points.append(pos.tp)
    y_min = min(candle_low,  min(y_points)) - atr * 2
    y_max = max(candle_high, max(y_points)) + atr * 2

    fig = make_subplots(rows=1, cols=1)
    fig.add_trace(go.Candlestick(
        x=df.index, open=df["open"], high=df["high"], low=df["low"], close=df["close"],
        name="OHLC", increasing_line_color="#26a69a", decreasing_line_color="#ef5350",
        increasing_fillcolor="#26a69a", decreasing_fillcolor="#ef5350",
        showlegend=False))

    info = mt5.symbol_info(symbol)
    digits = info.digits if info else 5

    def hline(y, color, dash, label):
        if abs(y - price_now) > atr * 25: return  # skip linii aberante
        fig.add_hline(y=y,
            line=dict(color=color, width=1.5, dash=dash),
            annotation_text=f"  {label}: {round(y, digits)}",
            annotation_font=dict(color=color, size=10))

    hline(pos.price_open, col_entry, "dot",   "Entry")
    if pos.sl: hline(pos.sl, col_sl, "dash",  "SL")
    if pos.tp: hline(pos.tp, col_tp, "dash",  "TP")
    hline(price_now, col_price, "solid", f"{'▲' if is_buy else '▼'} {round(price_now, digits)}")

    # zona profit/loss colorata (doar daca SL/TP sunt rezonabile)
    if pos.sl and pos.tp and abs(pos.tp - price_now) < atr * 20:
        fig.add_hrect(y0=pos.price_open, y1=pos.tp if is_buy else pos.sl,
            fillcolor="rgba(38,166,154,0.07)", line_width=0)
        fig.add_hrect(y0=pos.sl if is_buy else pos.tp, y1=pos.price_open,
            fillcolor="rgba(239,83,80,0.07)", line_width=0)

    pl_color = "#26a69a" if pl >= 0 else "#ef5350"
    title = (f"<b>{symbol}</b> {'BUY' if is_buy else 'SELL'} {pos.volume}L"
             f"  |  Entry: {round(pos.price_open, digits)}"
             f"  |  <span style='color:{pl_color}'>P&L: {'+' if pl>=0 else ''}{round(pos.profit,2)}$</span>"
             f"  |  {tf}")
    fig.update_layout(
        template="plotly_dark", paper_bgcolor="#111", plot_bgcolor="#111",
        title=dict(text=title, font=dict(size=12, color="#ddd")),
        xaxis_rangeslider_visible=False, height=380,
        margin=dict(l=50, r=120, t=45, b=20),
        yaxis=dict(gridcolor="#1e1e1e", range=[y_min, y_max]),
        xaxis=dict(gridcolor="#1e1e1e"),
    )
    return Response(fig.to_html(full_html=False, include_plotlyjs="cdn"), mimetype="text/html")


TRADES_HTML = """<!DOCTYPE html>
<html><head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Trades Active</title>
<style>
* { box-sizing:border-box; margin:0; padding:0; }
body { background:#111; color:#eee; font-family:'Segoe UI',sans-serif; }
.navbar { background:#161616; border-bottom:1px solid #2a2a2a; padding:0 20px; height:44px; display:flex; align-items:center; justify-content:space-between; }
.navbar-brand { font-size:1rem; font-weight:600; color:#eee; }
.nav-links { display:flex; gap:8px; }
.btn { background:#1976d2; color:#fff; border:none; padding:6px 14px; border-radius:4px; cursor:pointer; font-size:0.82rem; text-decoration:none; display:inline-flex; align-items:center; gap:5px; }
.btn:hover { background:#1565c0; }
.btn-red  { background:#b71c1c; } .btn-red:hover  { background:#c62828; }
.btn-teal { background:#00695c; } .btn-teal:hover { background:#004d40; }
.btn-grey { background:#333; color:#bbb; } .btn-grey:hover { background:#444; }
.content { padding:16px 20px; }
.summary { display:grid; grid-template-columns:repeat(auto-fit,minmax(140px,1fr)); gap:10px; margin-bottom:20px; }
.sum-card { background:#1a1a1a; border:1px solid #2a2a2a; border-radius:6px; padding:12px 14px; }
.sum-card .lbl { font-size:0.72rem; color:#666; text-transform:uppercase; margin-bottom:4px; }
.sum-card .val { font-size:1.2rem; font-weight:bold; color:#eee; }
.val.green { color:#26a69a; } .val.red { color:#ef5350; } .val.yellow { color:#ffeb3b; }
.trade-card { background:#1a1a1a; border:1px solid #2a2a2a; border-radius:8px; margin-bottom:14px; overflow:hidden; }
.trade-header { display:flex; align-items:center; gap:14px; padding:12px 16px; border-bottom:1px solid #222; flex-wrap:wrap; cursor:pointer; user-select:none; }
.trade-header:hover { background:#1e1e1e; }
.th-sym  { font-size:1rem; font-weight:bold; color:#eee; min-width:80px; }
.th-type { font-size:0.9rem; font-weight:bold; padding:3px 10px; border-radius:4px; }
.th-type.buy  { background:#1b5e20; color:#a5d6a7; }
.th-type.sell { background:#b71c1c; color:#ef9a9a; }
.th-stat { display:flex; flex-direction:column; gap:1px; }
.th-stat .lbl { font-size:0.7rem; color:#666; }
.th-stat .val { font-size:0.85rem; font-weight:600; }
.th-pl { font-size:1.1rem; font-weight:bold; margin-left:auto; }
.th-toggle { color:#555; font-size:0.8rem; margin-left:8px; }
.trade-body { padding:14px 16px; display:none; }
.trade-body.open { display:block; }
.pbar-wrap { margin-bottom:14px; }
.pbar-label { display:flex; justify-content:space-between; font-size:0.72rem; color:#888; margin-bottom:4px; }
.pbar { height:8px; background:#2a2a2a; border-radius:4px; position:relative; overflow:visible; }
.pbar-fill-loss { position:absolute; left:0; top:0; height:100%; background:#ef5350; border-radius:4px; }
.pbar-fill-profit { position:absolute; top:0; height:100%; background:#26a69a; border-radius:4px; }
.pbar-cursor { position:absolute; top:-4px; width:3px; height:16px; background:#ffeb3b; border-radius:2px; transform:translateX(-50%); }
.modify-row { display:flex; gap:10px; align-items:flex-end; flex-wrap:wrap; margin-bottom:12px; }
.mf { display:flex; flex-direction:column; gap:3px; }
.mf label { font-size:0.72rem; color:#666; text-transform:uppercase; }
.mf input { background:#242424; color:#eee; border:1px solid #383838; padding:5px 8px; border-radius:4px; font-size:0.84rem; width:120px; }
.mf input:focus { outline:none; border-color:#1976d2; }
.modify-result { font-size:0.82rem; padding:5px 10px; border-radius:4px; display:none; margin-top:6px; }
.ok-msg  { background:#1b5e20; color:#a5d6a7; }
.err-msg { background:#b71c1c; color:#ef9a9a; }
.chart-section { margin-top:10px; }
.tf-tabs { display:flex; gap:6px; margin-bottom:6px; }
.tf-tab { background:#2a2a2a; color:#aaa; border:1px solid #383838; padding:3px 10px; border-radius:4px; cursor:pointer; font-size:0.78rem; }
.tf-tab.active { background:#37474f; color:#fff; border-color:#607d8b; }
.chart-wrap { border:1px solid #222; border-radius:4px; overflow:hidden; background:#111; }
.chart-wrap iframe { width:100%; height:390px; border:none; }
.empty-state { text-align:center; padding:60px 20px; color:#555; font-size:0.9rem; }
.empty-state .icon { font-size:2.5rem; margin-bottom:10px; }
.section-title { font-size:0.85rem; font-weight:600; color:#888; text-transform:uppercase;
    letter-spacing:1px; margin:28px 0 12px; padding-bottom:6px; border-bottom:1px solid #222;
    display:flex; align-items:center; gap:10px; }
.section-title span { font-size:0.75rem; color:#555; font-weight:400; text-transform:none; letter-spacing:0; }
.hist-stats { display:flex; gap:16px; flex-wrap:wrap; margin-bottom:14px; }
.hist-stat { background:#1a1a1a; border:1px solid #2a2a2a; border-radius:5px; padding:8px 14px; font-size:0.82rem; }
.hist-stat .lbl { color:#555; font-size:0.7rem; margin-bottom:2px; text-transform:uppercase; }
.hist-stat .val { font-weight:bold; }
.hist-table { width:100%; border-collapse:collapse; font-size:0.8rem; }
.hist-table th { color:#555; font-weight:500; text-transform:uppercase; font-size:0.7rem;
    padding:6px 10px; border-bottom:1px solid #222; text-align:left; }
.hist-table td { padding:7px 10px; border-bottom:1px solid #1a1a1a; vertical-align:middle; }
.hist-table tr:hover td { background:#1a1a1a; }
.hist-table .type-buy  { color:#26a69a; font-weight:bold; }
.hist-table .type-sell { color:#ef5350; font-weight:bold; }
.hist-table .profit-pos { color:#26a69a; font-weight:bold; }
.hist-table .profit-neg { color:#ef5350; font-weight:bold; }
.hist-table .profit-zero { color:#888; }
.hist-load-more { background:#242424; border:1px solid #333; color:#888; padding:8px 20px;
    border-radius:4px; cursor:pointer; font-size:0.8rem; width:100%; margin-top:10px; }
.hist-load-more:hover { background:#2a2a2a; color:#aaa; }
</style>
</head>
<body>
<div class="navbar">
    <div class="navbar-brand">📊 Trades Active</div>
    <div class="nav-links">
        <a href="/" class="btn btn-grey">ChartVisualizer</a>
        <a href="/autotrader" class="btn btn-grey">AutoTrader</a>
        <a href="/account" class="btn btn-grey">Cont</a>
    </div>
</div>

<div class="content">
    <div class="summary">
        <div class="sum-card"><div class="lbl">Pozitii deschise</div><div class="val" id="s-count">—</div></div>
        <div class="sum-card"><div class="lbl">Profit total</div><div class="val" id="s-profit">—</div></div>
        <div class="sum-card"><div class="lbl">Balance</div><div class="val" id="s-balance">—</div></div>
        <div class="sum-card"><div class="lbl">Equity</div><div class="val" id="s-equity">—</div></div>
        <div class="sum-card"><div class="lbl">Actualizat</div><div class="val yellow" id="s-time" style="font-size:0.85rem">—</div></div>
    </div>
    <div id="trades-container"></div>

    <div class="section-title">📋 Istoric Trades <span id="hist-subtitle"></span></div>
    <div class="hist-stats" id="hist-stats"></div>
    <div id="hist-container"><div style="color:#555;font-size:0.82rem;padding:10px">Se incarca istoricul...</div></div>
</div>

<script>
const TFS = ["M1","M5","M15","H1","H4"];
let tradeState = {};  // ticket -> {open, tf}

async function loadTrades() {
    try {
        const r = await fetch('/account_data');
        const d = await r.json();
        if (!d.ok) {
            document.getElementById('trades-container').innerHTML =
                '<div class="empty-state"><div class="icon">⚠️</div>MT5 neconectat — <a href="/account" style="color:#1976d2">conecteaza-te</a></div>';
            return;
        }
        updateSummary(d);
        renderTrades(d);
    } catch(e) { console.warn(e); }
}

function updateSummary(d) {
    const profit = d.positions.reduce((s, p) => s + p.profit, 0);
    const pc = profit >= 0 ? 'green' : 'red';
    document.getElementById('s-count').textContent   = d.positions.length;
    document.getElementById('s-profit').textContent  = (profit >= 0 ? '+' : '') + profit.toFixed(2) + ' ' + d.currency;
    document.getElementById('s-profit').className    = 'val ' + pc;
    document.getElementById('s-balance').textContent = d.balance.toFixed(2) + ' ' + d.currency;
    document.getElementById('s-equity').textContent  = d.equity.toFixed(2)  + ' ' + d.currency;
    document.getElementById('s-equity').className    = 'val ' + (d.equity >= d.balance ? 'green' : 'red');
    document.getElementById('s-time').textContent    = new Date().toLocaleTimeString();
}

function renderTrades(d) {
    const container = document.getElementById('trades-container');
    if (d.positions.length === 0) {
        container.innerHTML = '<div class="empty-state"><div class="icon">💤</div>Nicio pozitie deschisa momentan</div>';
        return;
    }
    const existing = {};
    container.querySelectorAll('.trade-card').forEach(c => { existing[c.dataset.ticket] = c; });

    // adauga/actualizeaza carduri
    d.positions.forEach(p => {
        const ticket = String(p.ticket);
        const state  = tradeState[ticket] || {open: false, tf: "M5"};
        const isBuy  = p.type === 'BUY';
        const plColor= p.profit >= 0 ? '#26a69a' : '#ef5350';
        const plSign = p.profit >= 0 ? '+' : '';

        // bara progres SL-TP
        let pbarHtml = '';
        if (p.sl && p.tp) {
            const total  = Math.abs(p.tp - p.sl);
            const moved  = isBuy ? (p.price_current - p.sl) : (p.sl - p.price_current);
            const pct    = Math.max(0, Math.min(100, (moved / total) * 100));
            const inProfit = isBuy ? p.price_current > p.price_open : p.price_current < p.price_open;
            const entryPct = Math.max(0, Math.min(100,
                isBuy ? ((p.price_open - p.sl) / total * 100) : ((p.sl - p.price_open) / total * 100)));
            pbarHtml = `
            <div class="pbar-wrap">
                <div class="pbar-label">
                    <span style="color:#ef5350">SL ${p.sl}</span>
                    <span style="color:#ffeb3b">Entry ${p.price_open}</span>
                    <span style="color:#26a69a">TP ${p.tp}</span>
                </div>
                <div class="pbar">
                    <div class="pbar-fill-loss"  style="width:${entryPct}%"></div>
                    <div class="pbar-fill-profit" style="left:${entryPct}%;width:${Math.max(0,pct-entryPct)}%"></div>
                    <div class="pbar-cursor" style="left:${pct}%"></div>
                </div>
            </div>`;
        }

        const cardHtml = `
        <div class="trade-header" onclick="toggleCard('${ticket}')">
            <span class="th-sym">${p.symbol}</span>
            <span class="th-type ${isBuy ? 'buy' : 'sell'}">${p.type}</span>
            <div class="th-stat"><span class="lbl">Volume</span><span class="val">${p.volume}</span></div>
            <div class="th-stat"><span class="lbl">Entry</span><span class="val">${p.price_open}</span></div>
            <div class="th-stat"><span class="lbl">Curent</span><span class="val" style="color:#ffeb3b">${p.price_current}</span></div>
            <div class="th-stat"><span class="lbl">SL</span><span class="val" style="color:#ef5350">${p.sl || '—'}</span></div>
            <div class="th-stat"><span class="lbl">TP</span><span class="val" style="color:#26a69a">${p.tp || '—'}</span></div>
            <span class="th-pl" style="color:${plColor}">${plSign}${p.profit.toFixed(2)}$</span>
            <span class="th-toggle">${state.open ? '▲' : '▼'}</span>
        </div>
        <div class="trade-body ${state.open ? 'open' : ''}" id="body-${ticket}">
            ${pbarHtml}
            <div class="modify-row">
                <div class="mf"><label>Nou SL</label><input type="number" id="sl-${ticket}" value="${p.sl || ''}" step="0.00001" placeholder="Stop Loss"></div>
                <div class="mf"><label>Nou TP</label><input type="number" id="tp-${ticket}" value="${p.tp || ''}" step="0.00001" placeholder="Take Profit"></div>
                <button class="btn btn-teal" onclick="modifyTrade(${p.ticket})">✎ Modifica SL/TP</button>
                <button class="btn btn-red"  onclick="closeTrade(${p.ticket}, '${p.symbol}')">✕ Inchide</button>
            </div>
            <div class="modify-result" id="res-${ticket}"></div>
            <div class="chart-section">
                <div class="tf-tabs" id="tabs-${ticket}">
                    ${TFS.map(tf => `<button class="tf-tab ${tf === state.tf ? 'active' : ''}" onclick="setTf('${ticket}','${p.symbol}',this,'${tf}')">${tf}</button>`).join('')}
                </div>
                <div class="chart-wrap">
                    <iframe id="chart-${ticket}" src="${state.open ? '/trade_chart/'+p.ticket+'?tf='+state.tf : 'about:blank'}"></iframe>
                </div>
            </div>
        </div>`;

        if (existing[ticket]) {
            // actualizeaza doar header (fara sa distruga iframe)
            const card = existing[ticket];
            const oldHeader = card.querySelector('.trade-header');
            const newDiv = document.createElement('div');
            newDiv.innerHTML = cardHtml;
            card.replaceChild(newDiv.querySelector('.trade-header'), oldHeader);
            // actualizeaza bara progres daca panoul e deschis
            const pbarEl = card.querySelector('.pbar-wrap');
            const newPbar = newDiv.querySelector('.pbar-wrap');
            if (pbarEl && newPbar) pbarEl.outerHTML = newPbar.outerHTML;
        } else {
            const card = document.createElement('div');
            card.className = 'trade-card';
            card.dataset.ticket = ticket;
            card.innerHTML = cardHtml;
            container.appendChild(card);
        }
    });

    // sterge carduri pentru pozitii inchise
    const activeTickets = new Set(d.positions.map(p => String(p.ticket)));
    Object.keys(existing).forEach(t => {
        if (!activeTickets.has(t)) existing[t].remove();
    });
}

function toggleCard(ticket) {
    if (!tradeState[ticket]) tradeState[ticket] = {open: false, tf: "M5"};
    tradeState[ticket].open = !tradeState[ticket].open;
    const body = document.getElementById('body-' + ticket);
    body.classList.toggle('open', tradeState[ticket].open);
    // incarca graficul la prima deschidere
    const iframe = document.getElementById('chart-' + ticket);
    if (tradeState[ticket].open && iframe.src.includes('about:blank')) {
        iframe.src = '/trade_chart/' + ticket + '?tf=' + tradeState[ticket].tf;
    }
    // update toggle arrow in header
    const card = body.closest('.trade-card');
    const arrow = card.querySelector('.th-toggle');
    if (arrow) arrow.textContent = tradeState[ticket].open ? '▲' : '▼';
}

function setTf(ticket, symbol, btn, tf) {
    if (!tradeState[ticket]) tradeState[ticket] = {open: true, tf};
    tradeState[ticket].tf = tf;
    document.querySelectorAll(`#tabs-${ticket} .tf-tab`).forEach(b => b.classList.remove('active'));
    btn.classList.add('active');
    document.getElementById('chart-' + ticket).src = '/trade_chart/' + ticket + '?tf=' + tf;
}

async function modifyTrade(ticket) {
    const sl = parseFloat(document.getElementById('sl-' + ticket).value);
    const tp = parseFloat(document.getElementById('tp-' + ticket).value);
    if (isNaN(sl) || isNaN(tp)) { alert('Completeaza SL si TP'); return; }
    const r = await fetch('/modify_trade', {
        method: 'POST',
        headers: {'Content-Type':'application/json'},
        body: JSON.stringify({ticket, sl, tp})
    });
    const d = await r.json();
    const el = document.getElementById('res-' + ticket);
    el.style.display = 'block';
    el.className = 'modify-result ' + (d.ok ? 'ok-msg' : 'err-msg');
    el.textContent = d.message;
    setTimeout(() => el.style.display = 'none', 3000);
    if (d.ok) loadTrades();
}

async function closeTrade(ticket, symbol) {
    if (!confirm('Inchizi ' + symbol + ' #' + ticket + '?')) return;
    const r = await fetch('/close_position?ticket=' + ticket, {method:'POST'});
    const d = await r.json();
    if (d.ok) { delete tradeState[String(ticket)]; loadTrades(); }
    else alert('Eroare: ' + d.message);
}

// ── Istoric ───────────────────────────────────────────────────────────────────
let histPage = 0;
const HIST_PAGE_SIZE = 30;
let allHistory = [];

async function loadHistory() {
    try {
        const r = await fetch('/history_data');
        const d = await r.json();
        if (!d.ok) {
            document.getElementById('hist-container').innerHTML =
                '<div style="color:#555;font-size:0.82rem;padding:10px">Nu s-a putut incarca istoricul: ' + (d.msg||'') + '</div>';
            return;
        }
        allHistory = d.history || [];
        histPage = 0;
        // stats
        const net = d.total_net;
        const wr  = d.wins + d.losses > 0 ? Math.round(d.wins / (d.wins + d.losses) * 100) : 0;
        document.getElementById('hist-subtitle').textContent = `ultimele 90 zile · ${d.count} trades`;
        document.getElementById('hist-stats').innerHTML = `
            <div class="hist-stat"><div class="lbl">Net Total</div><div class="val ${net>=0?'profit-pos':'profit-neg'}">${net>=0?'+':''}${net.toFixed(2)} $</div></div>
            <div class="hist-stat"><div class="lbl">Castigate</div><div class="val profit-pos">${d.wins}</div></div>
            <div class="hist-stat"><div class="lbl">Pierdute</div><div class="val profit-neg">${d.losses}</div></div>
            <div class="hist-stat"><div class="lbl">Win Rate</div><div class="val ${wr>=50?'profit-pos':'profit-neg'}">${wr}%</div></div>
            <div class="hist-stat"><div class="lbl">Total</div><div class="val">${d.count}</div></div>
        `;
        renderHistory();
    } catch(e) { console.warn('history err', e); }
}

function renderHistory() {
    const slice = allHistory.slice(0, (histPage + 1) * HIST_PAGE_SIZE);
    const hasMore = allHistory.length > slice.length;
    let html = '<table class="hist-table"><thead><tr>';
    ['Data inchidere','Simbol','Tip','Volum','Entry','Close','Profit','Comision','Net','Comment','📸'].forEach(h => {
        html += `<th>${h}</th>`;
    });
    html += '</tr></thead><tbody>';
    slice.forEach(h => {
        const netCls  = h.net > 0 ? 'profit-pos' : h.net < 0 ? 'profit-neg' : 'profit-zero';
        const profCls = h.profit > 0 ? 'profit-pos' : h.profit < 0 ? 'profit-neg' : 'profit-zero';
        const hasSnap = snapshotTickets.has(h.ticket);
        const snapBtn = hasSnap
            ? `<a href="/snapshot/${h.ticket}" target="_blank" style="background:#1a2a1a;border:1px solid #2e4a2e;color:#66bb6a;padding:3px 8px;border-radius:3px;font-size:0.75rem;text-decoration:none;white-space:nowrap">📸 Vezi</a>`
            : `<span style="color:#333;font-size:0.75rem">—</span>`;
        html += `<tr>
            <td style="color:#666;white-space:nowrap">${h.close_time || h.open_time}</td>
            <td style="font-weight:bold">${h.symbol}</td>
            <td class="type-${h.type.toLowerCase()}">${h.type}</td>
            <td style="color:#aaa">${h.volume}</td>
            <td style="color:#aaa">${h.price_open}</td>
            <td style="color:#aaa">${h.price_close ?? '—'}</td>
            <td class="${profCls}">${h.profit>=0?'+':''}${h.profit.toFixed(2)}</td>
            <td style="color:#666">${h.commission.toFixed(2)}</td>
            <td class="${netCls}" style="font-size:0.85rem">${h.net>=0?'+':''}${h.net.toFixed(2)}</td>
            <td style="color:#555;font-size:0.75rem">${h.comment||'—'}</td>
            <td>${snapBtn}</td>
        </tr>`;
    });
    html += '</tbody></table>';
    if (hasMore) {
        html += `<button class="hist-load-more" onclick="loadMoreHistory()">↓ Arata mai multe (${allHistory.length - slice.length} ramase)</button>`;
    }
    document.getElementById('hist-container').innerHTML = html;
}

function loadMoreHistory() {
    histPage++;
    renderHistory();
}

// ── Snapshots ─────────────────────────────────────────────────────────────────
let snapshotTickets = new Set();

async function loadSnapshots() {
    try {
        const r = await fetch('/snapshots_list');
        const d = await r.json();
        snapshotTickets = new Set(d.tickets || []);
    } catch(e) {}
}

// reload history la 30s (nu are nevoie de 1s)
loadSnapshots();
loadHistory();
setInterval(loadHistory, 30000);
setInterval(loadSnapshots, 30000);

// ── Live refresh ─────────────────────────────────────────────────────────────
// refresh date la 1s, graficele la 10s
setInterval(loadTrades, 1000);
setInterval(() => {
    Object.keys(tradeState).forEach(ticket => {
        if (tradeState[ticket].open) {
            const iframe = document.getElementById('chart-' + ticket);
            if (iframe) {
                const tf = tradeState[ticket].tf || 'M5';
                iframe.src = '/trade_chart/' + ticket + '?tf=' + tf + '&t=' + Date.now();
            }
        }
    });
}, 10000);

loadTrades();
</script>
</body></html>"""


def _fetch_all_news():
    """Descarca TOATE stirile (nu doar High) din FF XML. Cache comun 4 ore."""
    import urllib.request, xml.etree.ElementTree as ET
    from datetime import datetime, timezone

    with _news_lock:
        now = datetime.now(timezone.utc)
        cached = _news_cache.get("all_events")
        fetched_at = _news_cache.get("fetched_at")
        if cached is not None and fetched_at and (now - fetched_at).total_seconds() < 14400:
            return cached, None

    urls = [
        "https://nfs.faireconomy.media/ff_calendar_thisweek.xml",
        "https://cdn-nfs.faireconomy.media/ff_calendar_thisweek.xml",
    ]
    data = None
    last_err = None
    for url in urls:
        try:
            req = urllib.request.Request(url, headers={
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
                "Accept": "application/xml,text/xml,*/*",
                "Cache-Control": "no-cache",
            })
            with urllib.request.urlopen(req, timeout=12) as resp:
                data = resp.read()
            break
        except Exception as e:
            last_err = str(e)
            log.warning(f"News URL {url} eroare: {e}")

    if not data:
        return _news_cache.get("all_events", []), last_err

    now = datetime.now(timezone.utc)
    events = []
    try:
        root = ET.fromstring(data)
        for ev in root.findall("event"):
            impact   = ev.findtext("impact",   "").strip().lower()
            title    = ev.findtext("title",    "")
            country  = ev.findtext("country",  "")
            date_str = ev.findtext("date",     "").strip()
            time_str = ev.findtext("time",     "").strip()
            forecast = ev.findtext("forecast", "")
            previous = ev.findtext("previous", "")
            actual   = ev.findtext("actual",   "")
            dt = None
            # ForexFactory e in ET (America/New_York) — convertim la UTC
            try:
                from zoneinfo import ZoneInfo as _ZI
                _ET = _ZI("America/New_York")
                def _to_utc(dt_naive): return dt_naive.replace(tzinfo=_ET).astimezone(timezone.utc)
            except Exception:
                from datetime import timedelta as _td
                def _to_utc(dt_naive): return (dt_naive + _td(hours=4)).replace(tzinfo=timezone.utc)
            for fmt in ["%m-%d-%Y %I:%M%p", "%m-%d-%Y %H:%M", "%Y-%m-%d %H:%M", "%m/%d/%Y %I:%M%p"]:
                try:
                    dt = _to_utc(datetime.strptime(f"{date_str} {time_str}", fmt))
                    break
                except:
                    pass
            if dt is None:
                try:
                    dt = _to_utc(datetime.strptime(date_str, "%m-%d-%Y"))
                except:
                    continue
            diff_min = (dt - now).total_seconds() / 60
            # time_et = ora FF (ET), time_ro = ora Romaniei (Europe/Bucharest)
            try:
                from zoneinfo import ZoneInfo as _ZI2
                dt_et = dt.astimezone(_ZI2("America/New_York"))
                dt_ro = dt.astimezone(_ZI2("Europe/Bucharest"))
                time_et_str  = dt_et.strftime("%H:%M")
                date_et_str  = dt_et.strftime("%Y-%m-%d")
                time_ro_str  = dt_ro.strftime("%H:%M")
            except Exception:
                from datetime import timedelta as _td2
                dt_et = dt - _td2(hours=4)
                dt_ro = dt + _td2(hours=3)
                time_et_str  = dt_et.strftime("%H:%M")
                date_et_str  = dt_et.strftime("%Y-%m-%d")
                time_ro_str  = dt_ro.strftime("%H:%M")
            events.append({
                "title":    title,
                "country":  country,
                "impact":   impact,
                "time_utc": dt.strftime("%Y-%m-%d %H:%M"),
                "time_et":  f"{date_et_str} {time_et_str}",
                "time_ro":  time_ro_str,
                "forecast": forecast,
                "previous": previous,
                "actual":   actual,
                "in_min":   round(diff_min),
                "past":     diff_min < 0,
            })
    except Exception as e:
        log.error(f"News XML parse error: {e}")
        return _news_cache.get("all_events", []), str(e)

    events.sort(key=lambda x: x["in_min"])
    with _news_lock:
        _news_cache["all_events"] = events
        _news_cache["fetched_at"] = datetime.now(timezone.utc)
    log.info(f"News cache actualizat: {len(events)} evenimente")
    return events, None


@app.route("/news_data")
def news_data():
    events, err = _fetch_all_news()
    if not events and err:
        return Response(json.dumps({"ok": False, "error": f"ForexFactory indisponibil: {err}. Incearca din nou in 5 minute."}), mimetype="application/json")
    return Response(json.dumps({"ok": True, "events": events, "count": len(events)}), mimetype="application/json")


NEWS_HTML = """<!DOCTYPE html>
<html><head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Calendar Stiri Forex</title>
<style>
* { box-sizing:border-box; margin:0; padding:0; }
body { background:#111; color:#eee; font-family:'Segoe UI',sans-serif; }
.navbar { background:#161616; border-bottom:1px solid #2a2a2a; padding:0 20px; height:44px; display:flex; align-items:center; justify-content:space-between; }
.navbar-brand { font-size:1rem; font-weight:600; color:#eee; }
.nav-links { display:flex; gap:8px; }
.btn { background:#1976d2; color:#fff; border:none; padding:6px 14px; border-radius:4px; cursor:pointer; font-size:0.82rem; text-decoration:none; display:inline-flex; align-items:center; }
.btn:hover { background:#1565c0; }
.btn-grey { background:#333; color:#bbb; } .btn-grey:hover { background:#444; }
.content { padding:16px 20px; max-width:1100px; }
.filter-bar { display:flex; gap:8px; margin-bottom:16px; flex-wrap:wrap; align-items:center; }
.filter-btn { background:#2a2a2a; border:1px solid #383838; color:#aaa; padding:5px 14px; border-radius:4px; cursor:pointer; font-size:0.82rem; }
.filter-btn.active { color:#fff; }
.filter-btn.high.active  { background:#b71c1c; border-color:#ef5350; }
.filter-btn.medium.active { background:#e65100; border-color:#ff9800; }
.filter-btn.low.active   { background:#1b5e20; border-color:#26a69a; }
.filter-btn.all.active   { background:#1976d2; border-color:#42a5f5; }
.section-day { margin-bottom:20px; }
.day-header { font-size:0.8rem; color:#666; text-transform:uppercase; letter-spacing:1px; padding:6px 0; border-bottom:1px solid #222; margin-bottom:8px; }
.news-row {
    display:grid;
    grid-template-columns: 60px 60px 40px 1fr 90px 90px 90px 100px;
    gap:8px; align-items:center;
    padding:8px 10px; border-radius:4px;
    margin-bottom:4px; font-size:0.83rem;
    border-left:3px solid transparent;
    transition: background 0.15s;
}
.news-row:hover { background:#1a1a1a; }
.news-row.high   { border-left-color:#ef5350; background:#1a1212; }
.news-row.medium { border-left-color:#ff9800; background:#1a1510; }
.news-row.low    { border-left-color:#26a69a; background:#111a18; }
.news-row.past   { opacity:0.4; }
.news-row.upcoming { animation: glow 1.5s infinite alternate; }
@keyframes glow { from { background:#1a1212; } to { background:#2a1515; } }
.impact-dot { width:10px; height:10px; border-radius:50%; display:inline-block; }
.impact-dot.high   { background:#ef5350; box-shadow:0 0 6px #ef5350; }
.impact-dot.medium { background:#ff9800; }
.impact-dot.low    { background:#26a69a; }
.col-time    { color:#888; font-size:0.78rem; }
.col-country { font-size:0.75rem; background:#2a2a2a; padding:2px 6px; border-radius:3px; text-align:center; }
.col-title   { font-weight:500; color:#ddd; }
.col-val     { text-align:right; font-size:0.8rem; color:#aaa; }
.col-actual.positive { color:#26a69a; font-weight:bold; }
.col-actual.negative { color:#ef5350; font-weight:bold; }
.badge-soon { background:#b71c1c; color:#ef9a9a; font-size:0.7rem; padding:2px 7px; border-radius:10px; margin-left:6px; animation:pulse 1s infinite; }
@keyframes pulse { 0%,100%{opacity:1} 50%{opacity:0.5} }
.col-header { color:#555; font-size:0.72rem; text-transform:uppercase; }
.headers { display:grid; grid-template-columns:60px 60px 40px 1fr 90px 90px 90px 100px; gap:8px; padding:4px 10px; margin-bottom:6px; }
.col-time-ro { color:#4fc3f7; font-size:0.78rem; font-weight:500; }
.update-time { font-size:0.72rem; color:#555; margin-left:auto; }
.empty { color:#555; padding:40px; text-align:center; }
</style>
</head><body>
<div class="navbar">
    <div class="navbar-brand">📰 Calendar Stiri Forex</div>
    <div class="nav-links">
        <a href="/" class="btn btn-grey">ChartVisualizer</a>
        <a href="/autotrader" class="btn btn-grey">AutoTrader</a>
        <a href="/trades" class="btn btn-grey">Trades</a>
    </div>
</div>
<div class="content">
    <div class="filter-bar">
        <button class="filter-btn high active"   onclick="setFilter('high')"  >🔴 Impact Mare</button>
        <button class="filter-btn medium active" onclick="setFilter('medium')">🟡 Impact Mediu</button>
        <button class="filter-btn low"           onclick="setFilter('low')"   >🟢 Impact Mic</button>
        <button class="filter-btn all active"    onclick="setFilter('all')"   >Toate</button>
        <span class="update-time" id="update-time"></span>
    </div>
    <div class="headers">
        <span class="col-header" title="US Eastern Time (New York)">Ora (ET)</span>
        <span class="col-header" style="color:#4fc3f7" title="Ora Romaniei (Europe/Bucharest)">Ora (RO)</span>
        <span class="col-header">Tara</span>
        <span class="col-header">Eveniment</span>
        <span class="col-header" style="text-align:right">Prognoza</span>
        <span class="col-header" style="text-align:right">Anterior</span>
        <span class="col-header" style="text-align:right">Actual</span>
        <span class="col-header" style="text-align:right">Status</span>
    </div>
    <div id="news-container"><div class="empty">Se incarca...</div></div>
</div>
<script>
let activeFilters = new Set(["high","medium","low"]);
let allEvents = [];

function setFilter(f) {
    if (f === "all") {
        if (activeFilters.size === 3) activeFilters.clear();
        else { activeFilters = new Set(["high","medium","low"]); }
    } else {
        if (activeFilters.has(f)) activeFilters.delete(f);
        else activeFilters.add(f);
    }
    document.querySelectorAll(".filter-btn").forEach(b => {
        const bf = b.className.match(/high|medium|low|all/)?.[0];
        if (bf === "all") b.classList.toggle("active", activeFilters.size === 3);
        else if (bf) b.classList.toggle("active", activeFilters.has(bf));
    });
    renderNews(allEvents);
}

async function loadNews() {
    try {
        const r = await fetch("/news_data");
        const d = await r.json();
        if (!d.ok) {
            document.getElementById("news-container").innerHTML =
                `<div class="empty">⚠ ${d.error}<br><br><button onclick="loadNews()" style="background:#1976d2;color:#fff;border:none;padding:8px 16px;border-radius:4px;cursor:pointer">↻ Incearca din nou</button></div>`;
            return;
        }
        allEvents = d.events;
        renderNews(allEvents);
        document.getElementById("update-time").textContent = `Actualizat: ${new Date().toLocaleTimeString()} · ${d.count} evenimente`;
    } catch(e) { console.warn(e); }
}

function renderNews(events) {
    const filtered = events.filter(e => activeFilters.has(e.impact));
    if (!filtered.length) {
        document.getElementById("news-container").innerHTML = '<div class="empty">Niciun eveniment pentru filtrele selectate</div>';
        return;
    }
    // grupeaza pe zi (dupa data ET — asa cum apare pe ForexFactory)
    const byDay = {};
    filtered.forEach(e => {
        const day = (e.time_et || e.time_utc).split(" ")[0];
        if (!byDay[day]) byDay[day] = [];
        byDay[day].push(e);
    });
    let html = "";
    Object.keys(byDay).sort().forEach(day => {
        const d = new Date(day + "T00:00:00Z");
        const dayLabel = d.toLocaleDateString("ro-RO", {weekday:"long", day:"numeric", month:"long", timeZone:"UTC"});
        html += `<div class="section-day"><div class="day-header">${dayLabel}</div>`;
        byDay[day].forEach(e => {
            const soon   = !e.past && e.in_min >= 0 && e.in_min <= 5;
            const actualClass = e.actual
                ? (parseFloat(e.actual) >= parseFloat(e.forecast || e.previous || 0) ? "positive" : "negative")
                : "";
            const statusBadge = e.past
                ? `<span style="color:#555;font-size:0.75rem">Trecut</span>`
                : soon
                ? `<span class="badge-soon">⚠ in ${e.in_min}min</span>`
                : `<span style="color:#666;font-size:0.75rem">in ${e.in_min}min</span>`;
            html += `
            <div class="news-row ${e.impact} ${e.past ? "past" : ""} ${soon ? "upcoming" : ""}">
                <span class="col-time">${(e.time_et || e.time_utc).split(" ")[1]}</span>
                <span class="col-time-ro">${e.time_ro || "—"}</span>
                <span class="col-country">${e.country}</span>
                <span class="col-title">
                    <span class="impact-dot ${e.impact}"></span>
                    &nbsp;${e.title}
                </span>
                <span class="col-val">${e.forecast || "—"}</span>
                <span class="col-val">${e.previous || "—"}</span>
                <span class="col-val col-actual ${actualClass}">${e.actual || "—"}</span>
                <span class="col-val">${statusBadge}</span>
            </div>`;
        });
        html += "</div>";
    });
    document.getElementById("news-container").innerHTML = html;
}

loadNews();
setInterval(loadNews, 60000);  // refresh la 1 minut
</script>
</body></html>"""


@app.route("/news")
def news_page():
    return NEWS_HTML


@app.route("/snapshot/<int:ticket>")
def snapshot_view(ticket):
    path = os.path.join(SNAPSHOTS_DIR, f"{ticket}.html")
    if not os.path.exists(path):
        return "<html><body style='background:#111;color:#555;font-family:sans-serif;padding:40px;text-align:center'><h2>Snapshot negasit pentru #{}</h2><p>Trade-ul a fost executat inainte de implementarea snapshot-urilor.</p></body></html>".format(ticket), 404
    return send_file(path, mimetype="text/html")


@app.route("/snapshots_list")
def snapshots_list():
    """Returneaza lista de ticket-uri care au snapshot."""
    try:
        tickets = set()
        for f in os.listdir(SNAPSHOTS_DIR):
            if f.endswith(".json"):
                try:
                    tickets.add(int(f.replace(".json", "")))
                except:
                    pass
        return Response(json.dumps({"tickets": list(tickets)}), mimetype="application/json")
    except Exception as e:
        return Response(json.dumps({"tickets": []}), mimetype="application/json")


@app.route("/history_data")
def history_data():
    if not MT5_AVAILABLE or mt5 is None:
        return Response(json.dumps({"ok": False, "msg": "MT5 indisponibil"}), mimetype="application/json")
    try:
        from datetime import datetime, timedelta, timezone
        date_to   = datetime.now(timezone.utc)
        date_from = date_to - timedelta(days=90)
        deals = mt5.history_deals_get(date_from, date_to)
        if deals is None:
            deals = []
        # grupeaza deals pe position_id — fiecare pozitie are open+close deal
        positions = {}
        for d in deals:
            pid = d.position_id
            if pid not in positions:
                positions[pid] = []
            positions[pid].append(d)
        history = []
        for pid, dlist in positions.items():
            # deal tip IN = deschidere (entry=1), OUT = inchidere (entry=2)
            entry_deal = next((d for d in dlist if d.entry == 0), None)  # DEAL_ENTRY_IN=0
            exit_deal  = next((d for d in dlist if d.entry == 1), None)  # DEAL_ENTRY_OUT=1
            if entry_deal is None:
                continue
            profit = sum(d.profit for d in dlist)
            commission = sum(d.commission for d in dlist)
            swap = sum(d.swap for d in dlist)
            deal_type = "BUY" if entry_deal.type == 0 else "SELL"  # DEAL_TYPE_BUY=0
            open_time  = datetime.fromtimestamp(entry_deal.time, tz=timezone.utc).strftime("%Y-%m-%d %H:%M")
            close_time = datetime.fromtimestamp(exit_deal.time, tz=timezone.utc).strftime("%Y-%m-%d %H:%M") if exit_deal else None
            history.append({
                "ticket":      pid,
                "symbol":      entry_deal.symbol,
                "type":        deal_type,
                "volume":      round(entry_deal.volume, 2),
                "price_open":  round(entry_deal.price, 5),
                "price_close": round(exit_deal.price, 5) if exit_deal else None,
                "profit":      round(profit, 2),
                "commission":  round(commission, 2),
                "swap":        round(swap, 2),
                "net":         round(profit + commission + swap, 2),
                "open_time":   open_time,
                "close_time":  close_time,
                "comment":     entry_deal.comment,
            })
        # sortat descrescator dupa data deschidere
        history.sort(key=lambda x: x["open_time"], reverse=True)
        # pastreaza doar pozitiile inchise (au close_time)
        closed = [h for h in history if h["close_time"]]
        total_net = round(sum(h["net"] for h in closed), 2)
        wins  = len([h for h in closed if h["net"] > 0])
        losses= len([h for h in closed if h["net"] < 0])
        return Response(json.dumps({
            "ok": True,
            "history": closed[:200],
            "total_net": total_net,
            "wins": wins,
            "losses": losses,
            "count": len(closed),
        }), mimetype="application/json")
    except Exception as e:
        log.error(f"history_data error: {e}")
        return Response(json.dumps({"ok": False, "msg": str(e)}), mimetype="application/json")


@app.route("/trades")
def trades_page():
    return TRADES_HTML


if __name__ == "__main__":
    print("ChartVisualizer pornit pe http://localhost:5004")
    app.run(host="0.0.0.0", port=5004, debug=False)
