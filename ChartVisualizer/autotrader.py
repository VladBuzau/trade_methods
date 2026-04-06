"""
AutoTrader Blueprint — dashboard de auto-trading integrat cu ChartVisualizer
"""

import threading
import time
import json
import logging
from datetime import datetime

import numpy as np
import pandas as pd
from flask import Blueprint, Response, request

from app import (
    SYMBOLS, SYMBOLS_CRYPTO, ALL_TFS, RISK_DOLLARS, MIN_TF_VOTES, MIN_CONFIDENCE,
    fetch, find_pivots, detect_trend, calc_entry, calc_sl_tp, place_trade,
    NpEncoder, MT5_AVAILABLE, mt5, build_chart,
    get_upcoming_red_news, close_all_positions_for_news, FTMO_NEWS_BLOCK_MIN,
    get_h4_direction, in_trading_session, calc_adx, ADX_MIN,
    calc_entry_smc, find_order_blocks, find_fvg, detect_bos,
)
import app as _app
from app import login_required
from app import login_required

log = logging.getLogger(__name__)

autotrader_bp = Blueprint("autotrader", __name__)

# ── Scanner state ─────────────────────────────────────────────────────────────
scanner = {
    "running":        False,
    "interval":       60,
    "auto_execute":   False,
    "use_h4_filter":  False,
    "use_session_filter": False,
    "symbols":        list(SYMBOLS),
    "last_scan":      None,
    "scan_count":     0,

    # ── Sectiunea Clasica ──
    "classic": {
        "enabled":  True,
        "tfs":      ["M5", "M15", "H1"],
        "tf_bars":  {"M1":500,"M5":500,"M15":500,"M30":500,"H1":500,"H4":500,"D1":500},
        "elements": {"ema": True, "fib": True, "adx": True, "rsi": True},
        "min_confidence": 66.0,
    },

    # ── Sectiunea SMC ──
    "smc": {
        "enabled":  True,
        "tfs":      ["M15", "H1", "H4"],
        "tf_bars":  {"M1":500,"M5":500,"M15":500,"M30":500,"H1":500,"H4":500,"D1":500},
        "elements": {"bos": True, "ob": True, "fvg": True, "structure": True},
        "min_confidence": 66.0,
    },
    # ── Strategii noi (dezactivate by default) ──
    "macd": {
        "enabled":  False,
        "tfs":      ["M15", "H1", "H4"],
        "tf_bars":  {"M1":500,"M5":500,"M15":500,"M30":500,"H1":500,"H4":500,"D1":500},
        "elements": {"macd_cross": True, "ema200": True, "histogram": True},
        "min_confidence": 66.0,
    },
    "bollinger": {
        "enabled":  False,
        "tfs":      ["M15", "H1", "H4"],
        "tf_bars":  {"M1":300,"M5":300,"M15":300,"M30":300,"H1":300,"H4":300,"D1":300},
        "elements": {"band_touch": True, "rsi_confirm": True, "squeeze": True},
        "min_confidence": 66.0,
    },
    "supertrend": {
        "enabled":  False,
        "tfs":      ["H1", "H4"],
        "tf_bars":  {"M1":300,"M5":300,"M15":300,"M30":300,"H1":300,"H4":300,"D1":300},
        "elements": {"supertrend": True, "ema50": True, "adx": True},
        "min_confidence": 66.0,
    },
    "london_breakout": {
        "enabled":  False,
        "tfs":      ["M15", "H1"],
        "tf_bars":  {"M1":200,"M5":200,"M15":200,"M30":200,"H1":200,"H4":200,"D1":200},
        "elements": {"asian_range": True, "breakout": True, "session_gate": True},
        "min_confidence": 66.0,
    },
    "ny_breakout": {
        "enabled":  False,
        "tfs":      ["M15", "H1"],
        "tf_bars":  {"M1":200,"M5":200,"M15":200,"M30":200,"H1":200,"H4":200,"D1":200},
        "elements": {"pre_ny_range": True, "breakout": True, "session_gate": True},
        "min_confidence": 66.0,
    },
    "rsi_divergence": {
        "enabled":  False,
        "tfs":      ["M15", "H1", "H4"],
        "tf_bars":  {"M1":300,"M5":300,"M15":300,"M30":300,"H1":300,"H4":300,"D1":300},
        "elements": {"bullish_div": True, "bearish_div": True, "rsi_zone": True},
        "min_confidence": 66.0,
    },
    "engulfing": {
        "enabled":  False,
        "tfs":      ["H1", "H4", "D1"],
        "tf_bars":  {"M1":200,"M5":200,"M15":200,"M30":200,"H1":200,"H4":200,"D1":200},
        "elements": {"engulfing": True, "pin_bar": True, "key_level": True},
        "min_confidence": 66.0,
    },
    "ichimoku": {
        "enabled":  False,
        "tfs":      ["H1", "H4"],
        "tf_bars":  {"M1":300,"M5":300,"M15":300,"M30":300,"H1":300,"H4":300,"D1":300},
        "elements": {"tk_cross": True, "kumo": True, "chikou": True},
        "min_confidence": 66.0,
    },
    "ema_cross": {
        "enabled":  False,
        "tfs":      ["M5", "M15", "H1"],
        "tf_bars":  {"M1":200,"M5":200,"M15":200,"M30":200,"H1":200,"H4":200,"D1":200},
        "elements": {"ema_cross": True, "ema50": True, "momentum": True},
        "min_confidence": 66.0,
    },
}
results   = {}   # {symbol: {"classic": ..., "smc": ...}}
decisions = []
_scanner_thread = None
_scanner_lock = threading.Lock()

# ── Log persistent pe disc ────────────────────────────────────────────────────
import os as _os
_LOG_FILE = _os.path.join(_os.path.dirname(__file__), "review_log.json")

def _log_action(entry: dict):
    """Salveaza o actiune (early exit, auto-execute, news close) in review_log.json."""
    try:
        if _os.path.exists(_LOG_FILE):
            with open(_LOG_FILE, "r", encoding="utf-8") as f:
                data = json.load(f)
        else:
            data = []
        data.insert(0, entry)
        data = data[:500]  # pastreaza ultimele 500
        with open(_LOG_FILE, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
    except Exception as e:
        log.warning(f"_log_action error: {e}")


# ── Price target estimation ───────────────────────────────────────────────────
def estimate_target(df, ph_idx, pl_idx, signal, price):
    """Estimeaza un target de pret pe baza extensiei Fibonacci 1.618 a ultimelor swing-uri."""
    if df is None or len(df) < 10:
        return None

    highs = df["high"].values
    lows  = df["low"].values
    n     = len(highs)
    cutoff = max(0, n - 100)

    ph_r = [i for i in ph_idx if i >= cutoff]
    pl_r = [i for i in pl_idx if i >= cutoff]

    if len(ph_r) < 2 or len(pl_r) < 2:
        return None

    # Calculeaza amplitudinile ultimelor 3 swing-uri
    amplitudes = []
    ph_vals = [highs[i] for i in ph_r[-5:]]
    pl_vals = [lows[i]  for i in pl_r[-5:]]
    min_len = min(len(ph_vals), len(pl_vals))
    for k in range(min_len):
        amp = abs(ph_vals[k] - pl_vals[k])
        if amp > 0:
            amplitudes.append(amp)

    amplitudes = amplitudes[-3:]  # ultimele 3
    if not amplitudes:
        return None

    avg_swing = sum(amplitudes) / len(amplitudes)

    if signal == "BUY":
        last_pl = lows[pl_r[-1]]
        target = last_pl + avg_swing * 1.618
    elif signal == "SELL":
        last_ph = highs[ph_r[-1]]
        target = last_ph - avg_swing * 1.618
    else:
        return None

    return round(float(target), 5)


# ── Full analysis function ────────────────────────────────────────────────────
def analyze_symbol_full(symbol, tfs, bars=500, tf_bars=None):
    """Analizeaza un simbol pe mai multe timeframe-uri si returneaza un dict complet."""
    tf_results = []

    for tf in tfs:
        try:
            n_bars = (tf_bars or {}).get(tf, bars)
            df, _ = fetch(symbol, tf, n_bars)
            if df is None or len(df) < 50:
                continue

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
            target = estimate_target(df, ph_idx, pl_idx, signal, price)

            tf_results.append({
                "tf":         tf,
                "signal":     signal,
                "trend":      trend,
                "conviction": len(reasons),
                "reasons":    reasons,
                "price":      round(float(price), 5),
                "sl":         sl,
                "tp":         tp,
                "target":     target,
            })
        except Exception as exc:
            log.warning(f"analyze_symbol_full {symbol}/{tf}: {exc}")
            continue

    # H4 direction filter (poate fi dezactivat din UI)
    use_h4  = scanner.get("use_h4_filter", True)
    use_ses = scanner.get("use_session_filter", True)
    h4_dir = get_h4_direction(symbol) if use_h4 else "ANY"
    session_ok = in_trading_session() if use_ses else True

    # Vot majoritar — filtrat de H4
    buy_v  = [r for r in tf_results if r["signal"] == "BUY"]
    sell_v = [r for r in tf_results if r["signal"] == "SELL"]
    n_buy, n_sell = len(buy_v), len(sell_v)
    n_total = len(tf_results)

    confidence = 0.0
    if n_total > 0:
        max_votes = max(n_buy, n_sell)
        confidence = round((max_votes / n_total) * 100, 1)

    final_signal = "HOLD"
    best_tf = None
    if n_buy >= MIN_TF_VOTES and n_buy > n_sell and confidence >= MIN_CONFIDENCE:
        if h4_dir in ("BUY", "ANY"):
            final_signal = "BUY"
            best_tf = max(buy_v, key=lambda x: x["conviction"])
    elif n_sell >= MIN_TF_VOTES and n_sell > n_buy and confidence >= MIN_CONFIDENCE:
        if h4_dir in ("SELL", "ANY"):
            final_signal = "SELL"
            best_tf = max(sell_v, key=lambda x: x["conviction"])

    # Target si SL/TP din best_tf
    target = best_tf["target"] if best_tf else None
    sl_best = best_tf["sl"] if best_tf else None
    tp_best = best_tf["tp"] if best_tf else None
    price_best = best_tf["price"] if best_tf else None

    # Calcul R:R
    rr_str = ""
    if best_tf and price_best and sl_best and tp_best:
        risk = abs(price_best - sl_best)
        reward = abs(tp_best - price_best)
        if risk > 0:
            rr = round(reward / risk, 1)
            rr_str = f"1:{rr}"

    # Justificare in romana
    justification = []
    trend_counts = {}
    for r in tf_results:
        trend_counts[r["trend"]] = trend_counts.get(r["trend"], 0) + 1

    dominant_trend = max(trend_counts, key=trend_counts.get) if trend_counts else "RANGING"
    trend_ro = {"ASCENDING": "ASCENDENT", "DESCENDING": "DESCENDENT", "RANGING": "LATERAL"}.get(dominant_trend, dominant_trend)
    dominant_count = trend_counts.get(dominant_trend, 0)
    justification.append(
        f"Trend {trend_ro} confirmat pe {dominant_count} din {n_total} timeframe-uri"
    )

    for r in tf_results:
        if r["signal"] != "HOLD" and r["reasons"]:
            trend_ro_tf = {"ASCENDING": "ASCENDENT", "DESCENDING": "DESCENDENT", "RANGING": "LATERAL"}.get(r["trend"], r["trend"])
            reasons_clean = [re.replace(" ✓", "") for re in r["reasons"]]
            justification.append(
                f"{r['tf']}: Trend {trend_ro_tf}, {', '.join(reasons_clean)}"
            )

    if final_signal != "HOLD":
        if target:
            justification.append(
                f"Target estimat: {target} (extensie Fibonacci 1.618 din ultimul swing)"
            )
        if best_tf:
            if final_signal == "BUY":
                justification.append(
                    f"SL: {sl_best} (sub ultimul pivot Low)"
                )
            else:
                justification.append(
                    f"SL: {sl_best} (deasupra ultimului pivot High)"
                )
        if rr_str:
            justification.append(f"Risc/Recompensa: {rr_str}")

    if final_signal == "HOLD":
        if h4_dir is None:
            justification.append(f"H4 lateral / ADX slab — asteapta trend clar pe H4")
        elif n_buy >= MIN_TF_VOTES and confidence >= MIN_CONFIDENCE and h4_dir == "SELL":
            justification.append(f"BUY blocat — H4 e BEARISH (contra-trend)")
        elif n_sell >= MIN_TF_VOTES and confidence >= MIN_CONFIDENCE and h4_dir == "BUY":
            justification.append(f"SELL blocat — H4 e BULLISH (contra-trend)")
        elif n_buy >= MIN_TF_VOTES and h4_dir == "BUY" and confidence < MIN_CONFIDENCE:
            justification.append(f"Confidence {confidence}% sub minimul {MIN_CONFIDENCE}% — semnal slab ({n_buy}/{n_total} TF-uri)")
        elif n_sell >= MIN_TF_VOTES and h4_dir == "SELL" and confidence < MIN_CONFIDENCE:
            justification.append(f"Confidence {confidence}% sub minimul {MIN_CONFIDENCE}% — semnal slab ({n_sell}/{n_total} TF-uri)")
        else:
            justification.append(
                f"Semnale insuficiente: {n_buy} BUY / {n_sell} SELL pe {n_total} TF-uri "
                f"(minim {MIN_TF_VOTES} voturi + {MIN_CONFIDENCE}% confidence)"
            )

    if not session_ok:
        from datetime import datetime as _dt, timezone as _tz
        _now = _dt.now(_tz.utc).strftime("%H:%M")
        justification.append(f"⚠ In afara sesiunii ({_now} UTC) — executia automata blocata")

    return {
        "symbol":        symbol,
        "timestamp":     datetime.now().isoformat(),
        "signal":        final_signal,
        "h4_dir":        h4_dir,
        "session_ok":    session_ok,
        "n_buy":         n_buy,
        "n_sell":        n_sell,
        "n_total":       n_total,
        "confidence":    confidence,
        "best_tf":       best_tf,
        "target":        target,
        "tfs":           tf_results,
        "justification": justification,
        "auto_executed": False,
    }


# ── re import for reason cleaning ─────────────────────────────────────────────
import re


# ── SMC analysis function ─────────────────────────────────────────────────────
def analyze_symbol_smc(symbol, tfs, bars=500, tf_bars=None, elements=None, min_confidence=50.0):
    """Analizeaza un simbol cu strategia SMC pe mai multe TF-uri."""
    if elements is None:
        elements = {"bos": True, "ob": True, "fvg": True, "structure": True}

    tf_results = []

    for tf in tfs:
        try:
            n_bars = (tf_bars or {}).get(tf, bars)
            df, _  = fetch(symbol, tf, n_bars)
            if df is None or len(df) < 50:
                continue

            highs  = df["high"].values
            lows   = df["low"].values
            ph_idx, pl_idx = find_pivots(df, lookback=5)

            signal, reasons, price, conviction = calc_entry_smc(df, ph_idx, pl_idx, elements)
            sl, tp = calc_sl_tp(df, ph_idx, pl_idx, signal, price)

            tf_results.append({
                "tf":         tf,
                "signal":     signal,
                "conviction": conviction,
                "reasons":    reasons,
                "price":      round(float(price), 5),
                "sl":         sl,
                "tp":         tp,
            })
        except Exception as exc:
            log.warning(f"analyze_symbol_smc {symbol}/{tf}: {exc}")

    if not tf_results:
        return {"symbol": symbol, "signal": "HOLD", "confidence": 0,
                "tfs": [], "best_tf": None, "justification": ["Fara date"], "timestamp": datetime.now().isoformat()}

    buy_v  = [r for r in tf_results if r["signal"] == "BUY"]
    sell_v = [r for r in tf_results if r["signal"] == "SELL"]
    n_buy, n_sell, n_total = len(buy_v), len(sell_v), len(tf_results)

    confidence = round((max(n_buy, n_sell) / n_total) * 100, 1) if n_total > 0 else 0.0

    final_signal = "HOLD"
    best_tf = None
    if n_buy > n_sell and n_buy >= 1 and confidence >= min_confidence:
        final_signal = "BUY"
        best_tf = max(buy_v, key=lambda x: x["conviction"])
    elif n_sell > n_buy and n_sell >= 1 and confidence >= min_confidence:
        final_signal = "SELL"
        best_tf = max(sell_v, key=lambda x: x["conviction"])

    justification = []
    if best_tf:
        justification += [r.replace(" ✓","") for r in best_tf["reasons"]]
        justification.append(f"Confidence {confidence}% ({max(n_buy,n_sell)}/{n_total} TF-uri)")
    else:
        justification.append(f"Semnal insuficient: {n_buy} BUY / {n_sell} SELL / {n_total} TF-uri")

    return {
        "symbol":        symbol,
        "timestamp":     datetime.now().isoformat(),
        "signal":        final_signal,
        "confidence":    confidence,
        "n_buy":         n_buy,
        "n_sell":        n_sell,
        "n_total":       n_total,
        "best_tf":       best_tf,
        "tfs":           tf_results,
        "justification": justification,
        "auto_executed": False,
        "strategy":      "smc",
    }


# ── Trade review — early exit daca trendul s-a inversat ──────────────────────
def review_open_trades(tfs, bars, auto_ex, cls_tfs=None, smc_tfs=None):
    """
    Verifica fiecare pozitie deschisa:
    - Daca semnalul s-a inversat (BUY deschis dar acum SELL) → inchide early
    - Daca trendul pe TF principal s-a schimbat → inchide early
    Inchide doar daca e in pierdere sau la break-even (nu taie profiturile).
    Foloseste TF-urile configurate din UI pentru Classic si SMC.
    """
    if not MT5_AVAILABLE or mt5 is None or not auto_ex:
        return []

    positions = mt5.positions_get()
    if not positions:
        return []

    closed_early = []
    # Foloseste TF-urile configurate, nu hardcodat M15/H1
    all_review_tfs = list(dict.fromkeys((cls_tfs or []) + (smc_tfs or []) + list(tfs)))
    review_tfs = all_review_tfs if all_review_tfs else tfs
    # Alege cel mai lent TF disponibil pentru review
    tf_order = ["M1","M5","M15","M30","H1","H4","D1"]
    review_tfs_sorted = sorted(review_tfs, key=lambda t: tf_order.index(t) if t in tf_order else 99)
    if not review_tfs_sorted:
        return []

    for pos in positions:
        try:
            symbol    = pos.symbol
            pos_type  = "BUY" if pos.type == 0 else "SELL"
            profit    = pos.profit
            price_now = pos.price_current
            price_open= pos.price_open

            # Analizeaza pe cel mai lent TF configurat
            tf_check = review_tfs_sorted[-1]
            df, _    = fetch(symbol, tf_check, bars)
            if df is None or len(df) < 50:
                continue

            highs  = df["high"].values
            lows   = df["low"].values
            ph_idx, pl_idx = find_pivots(df, lookback=5)
            trend  = detect_trend(ph_idx, pl_idx, highs, lows, recent_bars=100)
            ema20  = df["close"].ewm(span=20, adjust=False).mean()
            ema50  = df["close"].ewm(span=50, adjust=False).mean()
            delta  = df["close"].diff()
            gain   = delta.clip(lower=0).rolling(14).mean()
            loss   = (-delta.clip(upper=0)).rolling(14).mean()
            rsi    = 100 - 100 / (1 + gain / loss.replace(0, np.nan))

            current_signal, _, _ = calc_entry(df, ph_idx, pl_idx, trend, ema20, ema50, rsi)
            ema20_now = float(ema20.iloc[-1])
            ema50_now = float(ema50.iloc[-1])

            # Conditii de iesire anticipata:
            reason = None

            # 1. Semnal opus confirmat (cel mai puternic)
            if pos_type == "BUY" and current_signal == "SELL":
                reason = f"Semnal inversat: era BUY, acum SELL pe {tf_check}"
            elif pos_type == "SELL" and current_signal == "BUY":
                reason = f"Semnal inversat: era SELL, acum BUY pe {tf_check}"

            # 2. EMA cross opus (trend schimbat)
            elif pos_type == "BUY" and ema20_now < ema50_now * 0.9998:
                reason = f"EMA cross bearish pe {tf_check} — trend schimbat"
            elif pos_type == "SELL" and ema20_now > ema50_now * 1.0002:
                reason = f"EMA cross bullish pe {tf_check} — trend schimbat"

            # 3. Pret a rupt EMA50 in directia opusa
            elif pos_type == "BUY" and price_now < ema50_now * 0.9985:
                reason = f"Pret sub EMA50 pe {tf_check} — suport rupt"
            elif pos_type == "SELL" and price_now > ema50_now * 1.0015:
                reason = f"Pret deasupra EMA50 pe {tf_check} — rezistenta rupta"

            if reason is None:
                continue

            # Inchide DOAR daca e in pierdere sau profit mic (<50% din SL)
            sl_dist = abs(price_open - pos.sl) if pos.sl else 0
            loss_threshold = sl_dist * 0.5  # inchide daca a pierdut mai putin de 50% din SL

            if profit >= 0:
                # In profit — nu taia, lasa sa ruleze
                continue
            if sl_dist > 0 and abs(profit) > loss_threshold * 2:
                # Pierdere deja mare (>50% SL atins) — nu mai are rost, SL se va activa oricum
                continue

            # Executa inchiderea
            tick = mt5.symbol_info_tick(symbol)
            if tick is None:
                continue
            info     = mt5.symbol_info(symbol)
            close_price = tick.bid if pos_type == "BUY" else tick.ask
            order_type  = mt5.ORDER_TYPE_SELL if pos_type == "BUY" else mt5.ORDER_TYPE_BUY
            fm = info.filling_mode if info else 0
            if fm & 2:    filling = mt5.ORDER_FILLING_IOC
            elif fm & 1:  filling = mt5.ORDER_FILLING_FOK
            else:         filling = mt5.ORDER_FILLING_RETURN

            req = {
                "action":       mt5.TRADE_ACTION_DEAL,
                "symbol":       symbol,
                "volume":       pos.volume,
                "type":         order_type,
                "price":        close_price,
                "position":     pos.ticket,
                "deviation":    30,
                "magic":        pos.magic,
                "comment":      "early_exit",
                "type_time":    mt5.ORDER_TIME_GTC,
                "type_filling": filling,
            }
            result = mt5.order_send(req)
            if result and result.retcode == mt5.TRADE_RETCODE_DONE:
                msg = f"Early exit #{pos.ticket} {symbol} {pos_type} profit={round(profit,2)}$ — {reason}"
                log.info(msg)
                closed_early.append({
                    "ticket": pos.ticket,
                    "symbol": symbol,
                    "type":   pos_type,
                    "profit": round(profit, 2),
                    "reason": reason,
                })
            else:
                code = result.retcode if result else -1
                log.warning(f"Early exit {symbol} #{pos.ticket} esuat: {code}")

        except Exception as e:
            log.warning(f"review_open_trades {pos.symbol}: {e}")

    return closed_early


# ── Weekend close — inchide cu 30 min inainte de inchiderea pietei vineri ────
def _is_market_closing_soon(minutes_before=30):
    """
    Piata Forex se inchide vineri la 22:00 UTC (17:00 EST).
    Returneaza True daca suntem in fereastra [21:30, 22:05] UTC vineri.
    """
    from datetime import datetime, timezone
    now = datetime.now(timezone.utc)
    if now.weekday() != 4:  # 4 = vineri
        return False
    # Fereastra: 21:30 - 22:05 UTC (5 minute dupa inchidere ca safety)
    close_hour, close_min = 22, 0
    warn_min_total  = close_hour * 60 + close_min - minutes_before  # 21:30
    safety_min_total = close_hour * 60 + close_min + 5              # 22:05
    now_min_total   = now.hour * 60 + now.minute
    return warn_min_total <= now_min_total <= safety_min_total


# ── Background scanner ────────────────────────────────────────────────────────
def _scanner_loop():
    log.info("AutoTrader scanner pornit.")
    _last_news_close  = None  # evita inchideri repetate pentru aceeasi stire
    _weekend_closed   = False  # evita inchideri repetate in aceeasi fereastra
    while scanner["running"]:
        try:
            symbols   = list(scanner["symbols"])
            auto_ex   = scanner["auto_execute"]
            cls_cfg   = scanner["classic"]
            smc_cfg   = scanner["smc"]

            # ── Verificare inchidere weekend — vineri 21:30 UTC ───────────
            if _is_market_closing_soon(30):
                if not _weekend_closed:
                    _weekend_closed = True
                    # Inchide doar pozitiile Forex — crypto ramane deschis
                    CRYPTO_KEYWORDS = {"BTC","ETH","XRP","LTC","ADA","SOL","BNB","DOT","DOGE","MATIC","XLM","LINK","UNI","AVAX"}
                    positions = mt5.positions_get() or [] if MT5_AVAILABLE and mt5 else []
                    closed = []
                    for pos in positions:
                        sym_upper = pos.symbol.upper()
                        is_crypto = any(kw in sym_upper for kw in CRYPTO_KEYWORDS)
                        if is_crypto:
                            log.info(f"Weekend close: skip crypto {pos.symbol}")
                            continue
                        tick = mt5.symbol_info_tick(pos.symbol)
                        info = mt5.symbol_info(pos.symbol)
                        if not tick: continue
                        close_price = tick.bid if pos.type == 0 else tick.ask
                        order_type  = mt5.ORDER_TYPE_SELL if pos.type == 0 else mt5.ORDER_TYPE_BUY
                        fm = info.filling_mode if info else 0
                        if fm & 2: filling = mt5.ORDER_FILLING_IOC
                        elif fm & 1: filling = mt5.ORDER_FILLING_FOK
                        else: filling = mt5.ORDER_FILLING_RETURN
                        req = {
                            "action": mt5.TRADE_ACTION_DEAL, "symbol": pos.symbol,
                            "volume": pos.volume, "type": order_type, "price": close_price,
                            "position": pos.ticket, "deviation": 30, "magic": pos.magic,
                            "comment": "weekend_close", "type_time": mt5.ORDER_TIME_GTC,
                            "type_filling": filling,
                        }
                        result = mt5.order_send(req)
                        if result and result.retcode == mt5.TRADE_RETCODE_DONE:
                            closed.append(pos.symbol)
                    msg = f"Weekend close: inchis {len(closed)} pozitii Forex cu 30min inainte de inchiderea pietei"
                    log.warning(msg)
                    entry = {
                        "timestamp": datetime.now().isoformat(),
                        "symbol":    ", ".join(closed) if closed else "—",
                        "signal":    "WEEKEND CLOSE",
                        "confidence": 100,
                        "executed":  True,
                        "result":    msg,
                        "strategy":  "—",
                    }
                    _log_action(entry)
                    with _scanner_lock:
                        decisions.insert(0, entry)
                        while len(decisions) > 50: decisions.pop()
                    scanner["news_block"] = "⛔ Piata se inchide — weekend"
            else:
                _weekend_closed = False  # reseteaza pentru saptamana urmatoare
                if scanner.get("news_block") == "⛔ Piata se inchide — weekend":
                    scanner["news_block"] = None

            # ── Verificare stiri rosii — inchide toate pozitiile ──────────
            upcoming = get_upcoming_red_news(minutes_ahead=FTMO_NEWS_BLOCK_MIN)
            if upcoming:
                news_key = upcoming[0]["dt"]  # cheia primei stiri
                if news_key != _last_news_close:
                    _last_news_close = news_key
                    closed = close_all_positions_for_news()
                    if closed:
                        msg = f"Inchis {len(closed)} pozitii inainte de stire rosie: {upcoming[0]['title']} ({upcoming[0]['dt']})"
                        log.warning(msg)
                        entry = {
                            "timestamp":  datetime.now().isoformat(),
                            "symbol":     ", ".join(closed),
                            "signal":     "NEWS CLOSE",
                            "confidence": 100,
                            "executed":   True,
                            "result":     msg,
                            "strategy":   "—",
                        }
                        _log_action(entry)
                        with _scanner_lock:
                            decisions.insert(0, entry)
                scanner["news_block"] = f"⛔ Stire rosie: {upcoming[0]['title']} ({upcoming[0]['dt']})"
            else:
                scanner["news_block"] = None

            # ── Review trade-uri deschise ──────────────────────────────────
            if auto_ex:
                early_closed = review_open_trades(
                    [], 500, auto_ex,
                    cls_tfs=cls_cfg["tfs"] if cls_cfg["enabled"] else [],
                    smc_tfs=smc_cfg["tfs"] if smc_cfg["enabled"] else [],
                )
                for ec in early_closed:
                    entry = {
                        "timestamp":  datetime.now().isoformat(),
                        "symbol":     ec["symbol"],
                        "signal":     "EARLY EXIT",
                        "confidence": 0,
                        "executed":   True,
                        "result":     f"profit={ec['profit']}$ — {ec['reason']}",
                        "strategy":   "—",
                    }
                    _log_action(entry)
                    with _scanner_lock:
                        decisions.insert(0, entry)
                        while len(decisions) > 50:
                            decisions.pop()

            # ── Incarcare strategii active din registry ────────────────────
            import strategies as _strat_pkg
            enabled_strategies = _strat_pkg.get_enabled(scanner)

            for sym in symbols:
                if not scanner["running"]:
                    break
                try:
                    sym_results = results.get(sym, {})

                    for strat_key, strat_obj in enabled_strategies:
                        cfg = scanner.get(strat_key, {})
                        tfs_s   = cfg.get("tfs", strat_obj.default_tfs)
                        tf_bars = cfg.get("tf_bars", {})
                        elems   = cfg.get("elements", {k: True for k in strat_obj.elements})
                        min_conf = cfg.get("min_confidence", 66.0)

                        res = strat_obj.analyze(
                            sym, tfs_s, 500, tf_bars, elems, min_conf,
                            use_h4_filter=scanner.get("use_h4_filter", False),
                            use_session_filter=scanner.get("use_session_filter", False),
                        )
                        res["strategy"] = strat_key
                        sym_results[strat_key] = res

                        if res["signal"] != "HOLD" and auto_ex:
                            bf = res.get("best_tf")
                            if bf:
                                ok, msg = place_trade(sym, res["signal"], bf["sl"], bf["tp"], RISK_DOLLARS, strategy=strat_key)
                                res["auto_executed"] = ok
                                entry = {
                                    "timestamp":  datetime.now().isoformat(),
                                    "symbol":     sym,
                                    "signal":     res["signal"],
                                    "confidence": res["confidence"],
                                    "executed":   ok,
                                    "result":     msg,
                                    "strategy":   strat_key,
                                }
                                _log_action(entry)
                                with _scanner_lock:
                                    decisions.insert(0, entry)
                                    while len(decisions) > 50: decisions.pop()

                    with _scanner_lock:
                        results[sym] = sym_results

                except Exception as exc:
                    log.warning(f"Scanner error {sym}: {exc}")

            scanner["last_scan"] = datetime.now().isoformat()
            scanner["scan_count"] += 1
        except Exception as exc:
            log.error(f"Scanner loop error: {exc}")

        # Asteapta intervalul configurat, cu posibilitate de oprire rapida
        interval = int(scanner.get("interval", 60))
        for _ in range(interval * 2):
            if not scanner["running"]:
                break
            time.sleep(0.5)

    log.info("AutoTrader scanner oprit.")


def start_scanner():
    global _scanner_thread
    if scanner["running"]:
        return
    scanner["running"] = True
    _scanner_thread = threading.Thread(target=_scanner_loop, daemon=True, name="autotrader-scanner")
    _scanner_thread.start()


def stop_scanner():
    scanner["running"] = False


# ── HTML page ─────────────────────────────────────────────────────────────────
AUTOTRADER_HTML = """<!DOCTYPE html>
<html lang="ro"><head>
<meta charset="utf-8">
<title>AutoTrader — ChartVisualizer</title>
<style>
* { box-sizing:border-box; margin:0; padding:0; }
body { background:#111; color:#eee; font-family:'Segoe UI',sans-serif; font-size:15px; }

/* ── HEADER ── */
.header {
    background:#1a1a1a; border-bottom:2px solid #333;
    padding:0 20px; height:54px;
    display:flex; align-items:center; gap:10px;
    position:sticky; top:0; z-index:100;
    transition:border-color 0.4s, background 0.4s;
}
body.scanning .header { background:#0d1f1a; border-bottom-color:#26a69a; }
.header-title { font-size:1.15rem; font-weight:600; color:#ccc; margin-right:auto; }
body.scanning .header-title { color:#26a69a; }
.status-pill {
    display:flex; align-items:center; gap:7px;
    background:#1e1e1e; border:1px solid #333; border-radius:20px;
    padding:5px 14px; font-size:0.85rem; color:#888;
}
.status-dot { width:9px; height:9px; border-radius:50%; background:#555; flex-shrink:0; }
.status-dot.running { background:#26a69a; animation:pulse 1.2s infinite; box-shadow:0 0 6px #26a69a; }
@keyframes pulse { 0%,100%{opacity:1} 50%{opacity:0.4} }

/* ── SETTINGS PANEL ── */
.settings-panel {
    background:#161616; border-bottom:1px solid #2a2a2a;
    padding:16px 20px;
    display:grid; grid-template-columns:1fr 1fr 1fr; gap:20px;
}
.settings-panel.collapsed { display:none; }
.settings-section { display:flex; flex-direction:column; gap:8px; }
.settings-section h4 { font-size:0.8rem; color:#888; text-transform:uppercase; letter-spacing:0.5px; border-bottom:1px solid #2a2a2a; padding-bottom:6px; margin-bottom:2px; }

/* Symbol chips */
.sym-chips { display:flex; flex-wrap:wrap; gap:5px; margin-bottom:6px; }
.sym-chip { display:inline-flex; align-items:center; gap:4px; background:#2a2a2a; border:1px solid #444; border-radius:4px; padding:4px 8px; font-size:0.84rem; color:#ccc; user-select:none; }
.sym-chip.active { background:#4a148c; border-color:#9c27b0; color:#fff; }
.sym-chip .rem-x { color:#666; font-size:0.78rem; cursor:pointer; margin-left:2px; }
.sym-chip .rem-x:hover { color:#ef5350; }
.sym-add-wrap { position:relative; }
.sym-add-wrap input { width:100%; background:#2a2a2a; color:#eee; border:1px solid #444; padding:7px 10px; border-radius:4px; font-size:0.86rem; }
.sym-add-wrap input:focus { outline:none; border-color:#9c27b0; }
.sym-autocomplete { position:absolute; top:100%; left:0; right:0; z-index:300; background:#1a1a1a; border:1px solid #555; border-radius:4px; max-height:240px; overflow-y:auto; box-shadow:0 4px 16px #000a; }
.sym-ac-item { display:flex; align-items:center; justify-content:space-between; padding:7px 10px; cursor:pointer; border-bottom:1px solid #222; }
.sym-ac-item:hover { background:#2a2a2a; }
.sym-ac-item .ac-name { font-size:0.86rem; font-weight:600; color:#eee; }
.sym-ac-item .ac-cat { font-size:0.72rem; padding:1px 6px; border-radius:3px; margin-left:6px; }
.ac-cat-forex   { background:#0d2d1a; color:#66bb6a; }
.ac-cat-crypto  { background:#2d1a0d; color:#ff9800; }
.ac-cat-index   { background:#0d1a2d; color:#42a5f5; }
.ac-cat-metal   { background:#2d2a0d; color:#ffd54f; }
.ac-cat-other   { background:#222; color:#888; }
.sym-ac-item .ac-add { font-size:0.75rem; color:#555; margin-left:auto; }
.sym-ac-item:hover .ac-add { color:#9c27b0; }

/* TF bars table */
.tf-bars-table { width:100%; border-collapse:collapse; font-size:0.86rem; }
.tf-bars-table th { color:#666; font-weight:400; font-size:0.78rem; padding:3px 6px; text-align:left; border-bottom:1px solid #2a2a2a; }
.tf-bars-table td { padding:5px 6px; border-bottom:1px solid #1e1e1e; }
.tf-bars-table input[type=number] { width:85px; background:#2a2a2a; color:#eee; border:1px solid #383838; padding:4px 6px; border-radius:3px; font-size:0.83rem; }
.tf-cb { width:16px; height:16px; accent-color:#9c27b0; cursor:pointer; }

/* Config rows */
.config-row { display:flex; align-items:center; justify-content:space-between; padding:7px 0; border-bottom:1px solid #1e1e1e; gap:10px; }
.config-row label { font-size:0.9rem; color:#ccc; }
.config-row .sub { font-size:0.76rem; color:#666; display:block; margin-top:1px; }
input[type=range] { accent-color:#9c27b0; cursor:pointer; }

/* inputs */
input[type=number], input[type=text], select { background:#2a2a2a; color:#eee; border:1px solid #444; padding:5px 8px; border-radius:4px; font-size:0.85rem; }

/* buttons */
.btn {
    background:#1976d2; color:#fff; border:none;
    padding:8px 16px; border-radius:5px; cursor:pointer;
    font-size:0.88rem; text-decoration:none; display:inline-flex; align-items:center; gap:5px;
    transition:background 0.15s; white-space:nowrap;
}
.btn:hover { background:#1565c0; }
.btn-sm { padding:5px 12px; font-size:0.82rem; }
.btn-green { background:#00695c; }
.btn-green:hover { background:#004d40; }
.btn-red { background:#c62828; }
.btn-red:hover { background:#b71c1c; }
.btn-back { background:#333; color:#bbb; }
.btn-back:hover { background:#444; }
.btn-execute-buy  { background:#1b5e20; color:#a5d6a7; font-weight:bold; padding:7px 18px; font-size:0.88rem; }
.btn-execute-buy:hover  { background:#2e7d32; }
.btn-execute-sell { background:#b71c1c; color:#ef9a9a; font-weight:bold; padding:7px 18px; font-size:0.88rem; }
.btn-execute-sell:hover { background:#c62828; }

/* auto-execute toggle */
.toggle-wrap { display:flex; align-items:center; gap:8px; }
.toggle { position:relative; width:42px; height:22px; }
.toggle input { opacity:0; width:0; height:0; }
.slider {
    position:absolute; cursor:pointer; inset:0;
    background:#333; border-radius:22px; transition:background 0.2s;
}
.slider:before {
    content:""; position:absolute; width:16px; height:16px;
    left:3px; bottom:3px; background:#aaa; border-radius:50%;
    transition:transform 0.2s, background 0.2s;
}
input:checked + .slider { background:#9c27b0; }
input:checked + .slider:before { transform:translateX(20px); background:#fff; }
#auto-exec-toggle:checked + .slider { background:#c62828; }
.auto-ex-warn { color:#ef5350; font-size:0.78rem; font-weight:bold; display:none; }

/* MAIN CONTENT */
.main-content { padding:14px 20px; }
.grid-header { display:flex; justify-content:space-between; align-items:center; margin-bottom:12px; flex-wrap:wrap; gap:8px; }
.grid-header h2 { font-size:0.92rem; color:#888; font-weight:400; }
.scan-progress-bar { height:3px; background:#333; border-radius:2px; overflow:hidden; width:200px; }
.scan-progress-fill { height:100%; background:#26a69a; border-radius:2px; transition:width 1s linear; }
.next-scan-label { font-size:0.8rem; color:#888; }
body.scanning .next-scan-label { color:#4db6ac; }

.symbol-grid {
    display:grid;
    grid-template-columns:repeat(auto-fill, minmax(178px, 1fr));
    gap:10px;
    margin-bottom:16px;
}
.sym-card {
    background:#1a1a1a; border:1px solid #2a2a2a; border-radius:6px;
    padding:12px 14px; cursor:pointer; transition:border-color 0.15s, background 0.15s;
    user-select:none;
}
.sym-card:hover { background:#202020; border-color:#444; }
.sym-card.selected { border-color:#9c27b0; background:#1e1028; }
.sym-card.sig-buy  { border-left:3px solid #26a69a; }
.sym-card.sig-sell { border-left:3px solid #ef5350; }
.sym-card.sig-hold { border-left:3px solid #444; }
.card-name   { font-size:0.96rem; font-weight:bold; color:#ddd; margin-bottom:4px; }
.card-signal { font-size:1.15rem; font-weight:bold; margin-bottom:3px; }
.card-signal.buy  { color:#26a69a; }
.card-signal.sell { color:#ef5350; }
.card-signal.hold { color:#666; }
.card-conf   { font-size:0.78rem; color:#888; margin-bottom:2px; }
.card-trend  { font-size:0.76rem; color:#777; margin-bottom:2px; }
.card-time   { font-size:0.72rem; color:#555; }
.scanning-card { border-left:3px solid #37474f !important; opacity:0.7; }
.card-scanning { font-size:0.84rem; color:#607d8b; margin-top:4px; }
.scan-spin { display:inline-block; animation:spin 1s linear infinite; }
@keyframes spin { to { transform:rotate(360deg); } }

/* Banner activ */
.scanner-banner {
    display:none; background:#0a2218; border-bottom:1px solid #1b5e20;
    color:#a5d6a7; padding:8px 20px; font-size:0.84rem;
    align-items:center; gap:12px;
}
.scanner-banner.visible { display:flex; }
.banner-dot { width:8px; height:8px; border-radius:50%; background:#26a69a; animation:pulse 1.2s infinite; flex-shrink:0; }
.banner-text { flex:1; }
.banner-text b { color:#66bb6a; }

/* Toast */
.toast {
    position:fixed; bottom:24px; right:24px; z-index:9999;
    background:#1b5e20; color:#a5d6a7; padding:12px 20px; border-radius:6px;
    font-size:0.88rem; box-shadow:0 4px 16px rgba(0,0,0,0.5);
    transform:translateY(80px); opacity:0;
    transition:transform 0.3s, opacity 0.3s; pointer-events:none;
}
.toast.show { transform:translateY(0); opacity:1; }

/* DETAIL PANEL */
#detail-panel {
    background:#1a1a1a; border:1px solid #333; border-radius:6px;
    padding:16px 18px; margin-bottom:16px; display:none;
}
.detail-header { display:flex; align-items:center; gap:14px; margin-bottom:12px; flex-wrap:wrap; }
.detail-sym  { font-size:1.2rem; font-weight:bold; color:#ddd; }
.badge { font-size:0.92rem; font-weight:bold; padding:4px 14px; border-radius:4px; }
.badge.buy  { background:#1b5e20; color:#a5d6a7; }
.badge.sell { background:#b71c1c; color:#ef9a9a; }
.badge.hold { background:#333; color:#888; }
.conf-text  { font-size:0.85rem; color:#aaa; }

.detail-body  { display:flex; gap:14px; flex-wrap:wrap; }
.detail-left  { flex:1; min-width:260px; }
.detail-right { flex:2; min-width:300px; }

/* TF vote table */
.tf-table { width:100%; border-collapse:collapse; font-size:0.85rem; margin-bottom:12px; }
.tf-table th { color:#888; font-weight:400; padding:4px 8px; border-bottom:1px solid #333; text-align:left; }
.tf-table td { padding:5px 8px; border-bottom:1px solid #222; }
.sig-buy   { color:#26a69a; font-weight:bold; }
.sig-sell  { color:#ef5350; font-weight:bold; }
.sig-hold  { color:#666; }
.sig-early { color:#ffb74d; font-weight:bold; }
.sig-close { color:#ab47bc; font-weight:bold; }

/* Justification */
.justif-box {
    background:#161616; border:1px solid #2a2a2a; border-radius:4px;
    padding:10px 12px; font-size:0.83rem; color:#aaa; line-height:1.8; margin-bottom:12px;
}
.justif-box li { list-style:none; padding-left:14px; position:relative; }
.justif-box li::before { content:"•"; position:absolute; left:0; color:#9c27b0; }

/* SL/TP/Target row */
.price-row { display:flex; gap:14px; flex-wrap:wrap; margin-bottom:12px; }
.price-item { display:flex; flex-direction:column; gap:2px; }
.price-item .plabel { font-size:0.74rem; color:#888; }
.price-item .pvalue { font-size:0.92rem; font-weight:bold; color:#ddd; }
.price-item .pvalue.sl    { color:#ef5350; }
.price-item .pvalue.tp    { color:#26a69a; }
.price-item .pvalue.tgt   { color:#ffc107; }
.price-item .pvalue.rr    { color:#ab47bc; }
.price-item .pvalue.entry { color:#ccc; }

.execute-row { display:flex; gap:10px; align-items:center; flex-wrap:wrap; margin-bottom:8px; }
#execute-result { font-size:0.85rem; padding:6px 10px; border-radius:4px; display:none; margin-top:6px; }
.exec-ok  { background:#1b5e20; color:#a5d6a7; }
.exec-err { background:#b71c1c; color:#ef9a9a; }

/* Chart iframe */
.chart-frame-wrap { width:100%; min-height:440px; background:#111; border-radius:4px; overflow:hidden; border:1px solid #222; }
.chart-frame-wrap iframe { width:100%; height:460px; border:none; background:#111; }
.tf-tab { background:#2a2a2a; color:#aaa; border:1px solid #444; padding:4px 10px; border-radius:4px; cursor:pointer; font-size:0.82rem; }
.tf-tab:hover { background:#333; }
.tf-tab.active { background:#37474f; color:#fff; border-color:#607d8b; }

/* DECISIONS LOG */
.decisions-section { margin-top:4px; }
.decisions-section h3 { font-size:0.9rem; color:#888; font-weight:400; margin-bottom:8px; }
.dec-table { width:100%; border-collapse:collapse; font-size:0.84rem; }
.dec-table th { color:#777; font-weight:400; padding:4px 8px; border-bottom:1px solid #2a2a2a; text-align:left; }
.dec-table td { padding:6px 8px; border-bottom:1px solid #1e1e1e; }
.dec-table tr:hover td { background:#1c1c1c; }
.dec-yes { color:#26a69a; }
.dec-no  { color:#666; }

::-webkit-scrollbar { width:6px; height:6px; }
::-webkit-scrollbar-track { background:#111; }
::-webkit-scrollbar-thumb { background:#333; border-radius:3px; }

.btn-red   { background:#c62828; } .btn-red:hover   { background:#b71c1c; }
.btn-grey  { background:#333; color:#bbb; } .btn-grey:hover { background:#444; }
.btn-teal  { background:#00695c; } .btn-teal:hover  { background:#004d40; }
.btn-execute-buy  { background:#1b5e20; color:#a5d6a7; font-weight:bold; padding:8px 18px; font-size:0.9rem; }
.btn-execute-buy:hover  { background:#2e7d32; }
.btn-execute-sell { background:#b71c1c; color:#ef9a9a; font-weight:bold; padding:8px 18px; font-size:0.9rem; }
.btn-execute-sell:hover { background:#c62828; }
</style>
</head>
<body>

<!-- HEADER -->
<div class="header">
    <span class="header-title">⚡ AutoTrader</span>
    <div class="status-pill">
        <div class="status-dot" id="status-dot"></div>
        <span id="status-text">Oprit</span>
        &nbsp;·&nbsp; #<span id="scan-count">0</span>
        &nbsp;&nbsp;<span class="next-scan-label" id="next-scan-label"></span>
    </div>
    <div id="ftmo-indicator" style="font-size:0.82rem;padding:4px 12px;border-radius:4px;background:#1b5e20;color:#a5d6a7">✓ FTMO OK</div>
    <button class="btn btn-sm btn-grey" onclick="toggleSettings()">⚙ Setari</button>
    <button class="btn btn-sm btn-green" id="btn-start" onclick="startScanner()">▶ Start</button>
    <button class="btn btn-sm btn-red"   id="btn-stop"  onclick="stopScanner()" disabled>■ Stop</button>
    <a href="/trades" class="btn btn-sm btn-teal">📊 Trades</a>
    <a href="/" class="btn btn-sm btn-grey">← Chart</a>
    <a href="/logout" class="btn btn-sm btn-grey" style="margin-left:4px;color:#ef9a9a">⏻ Logout</a>
</div>

<!-- SETTINGS PANEL -->
<div class="settings-panel" id="settings-panel" style="grid-template-columns:280px 1fr 180px">

    <!-- COL 1: Simboluri + Config globala -->
    <div class="settings-section">
        <h4>Simboluri monitorizate</h4>
        <div class="sym-chips" id="sym-chips"></div>
        <div class="sym-add-wrap">
            <input type="text" id="sym-add-input" placeholder="Cauta simbol (ex: BTC, EUR, NAS...)" maxlength="20"
                   oninput="onSymInput(this)" onkeydown="onSymKey(event)">
            <div class="sym-autocomplete" id="sym-autocomplete" style="display:none"></div>
        </div>
        <div id="sym-ac-count" style="font-size:0.72rem;color:#555;margin-top:3px;min-height:14px"></div>
        <div style="margin-top:12px">
        <div class="config-row">
            <label>Interval scanare<span class="sub" id="interval-label">60s</span></label>
            <input type="range" id="interval-range" min="0" max="3" step="1" value="1" style="width:90px" oninput="onIntervalChange(this)">
        </div>
        <div class="config-row">
            <label>Trades maxime<span class="sub">Pozitii simultane</span></label>
            <input type="number" id="max-trades-input" value="5" min="1" max="50" step="1" style="width:60px" onchange="sendGlobal('max_open_trades',parseInt(this.value))">
        </div>
        <div class="config-row">
            <label>TP/SL Ratio<span class="sub">1.0=1:1 &nbsp; 2.0=1:2</span></label>
            <input type="number" id="tp-ratio-input" value="1.0" min="0.1" max="10" step="0.1" style="width:60px" onchange="sendGlobal('tp_ratio',parseFloat(this.value))">
        </div>
        <div class="config-row">
            <label>Auto Execute<span class="sub">Plaseaza automat</span></label>
            <div style="display:flex;align-items:center;gap:6px">
                <label class="toggle"><input type="checkbox" id="auto-exec-toggle" onchange="toggleAutoExec(this)"><span class="slider"></span></label>
                <span class="auto-ex-warn" id="auto-ex-warn">⚠ ACTIV</span>
            </div>
        </div>
        <div class="config-row">
            <label>Crypto<span class="sub">Adauga simboluri crypto</span></label>
            <label class="toggle"><input type="checkbox" id="crypto-toggle" onchange="toggleCrypto(this.checked)"><span class="slider"></span></label>
        </div>
        </div>
    </div>

    <!-- Selector + detalii strategie -->
    <div class="settings-section" id="settings-strategies">
        <h4>Strategii</h4>
        <!-- Lista strategii -->
        <table id="strat-list-table" style="width:100%;border-collapse:collapse;margin-bottom:10px">
            <thead>
                <tr style="border-bottom:1px solid #2a2a2a">
                    <th style="text-align:left;padding:4px 8px;color:#555;font-weight:400;font-size:0.74rem">Strategie</th>
                    <th style="text-align:center;padding:4px 8px;color:#555;font-weight:400;font-size:0.74rem">Status</th>
                    <th style="width:24px"></th>
                </tr>
            </thead>
            <tbody id="strat-list-body">
                <!-- Generat de buildStratList() -->
            </tbody>
        </table>
        <!-- Detalii strategie selectata -->
        <div id="strat-detail-panel"></div>
    </div>

    <!-- Legenda -->
    <div class="settings-section" id="settings-legend">
        <h4>Legenda</h4>
        <div style="font-size:0.78rem;color:#aaa;line-height:2">
            <b style="color:#26a69a">EMA</b> — trend aliniat<br>
            <b style="color:#26a69a">FIB</b> — Fibonacci 38-62%<br>
            <b style="color:#26a69a">ADX</b> — trend puternic<br>
            <b style="color:#26a69a">RSI</b> — zona neutra 38-62<br>
            <div style="border-top:1px solid #2a2a2a;margin:6px 0"></div>
            <b style="color:#ff9800">BOS</b> — Break of Structure<br>
            <b style="color:#ff9800">OB</b> — Order Block<br>
            <b style="color:#ff9800">FVG</b> — Fair Value Gap<br>
            <b style="color:#ff9800">STR</b> — Market Structure<br>
        </div>
    </div>

</div>

<!-- Banner scanner activ -->
<div class="scanner-banner" id="scanner-banner">
    <div class="banner-dot"></div>
    <div class="banner-text">
        <b>AutoTrader ACTIV</b> — scanez simbolurile ·
        urm. scan: <span id="banner-countdown">—</span> ·
        scanari: <span id="banner-scans">0</span>
    </div>
</div>

<!-- Toast -->
<div class="toast" id="toast"></div>

<!-- MAIN CONTENT -->
<div class="main-content">

    <div class="grid-header">
        <h2>Ultima scanare: <span id="last-scan">—</span></h2>
        <div class="scan-progress-bar" id="scan-progress-bar" style="display:none">
            <div class="scan-progress-fill" id="scan-progress-fill" style="width:0%"></div>
        </div>
    </div>

    <!-- TABS STRATEGII -->
    <div style="display:flex;gap:0;margin-bottom:0;border-bottom:2px solid #2a2a2a">
        <button id="tab-classic" onclick="switchTab('classic')"
            style="padding:8px 22px;border:none;border-bottom:2px solid #26a69a;margin-bottom:-2px;background:#111;color:#26a69a;font-weight:600;font-size:0.85rem;cursor:pointer">
            🔵 Classic
        </button>
        <button id="tab-smc" onclick="switchTab('smc')"
            style="padding:8px 22px;border:none;border-bottom:2px solid transparent;margin-bottom:-2px;background:#111;color:#555;font-weight:600;font-size:0.85rem;cursor:pointer">
            🟠 SMC
        </button>
    </div>

    <!-- Sectiune Clasica -->
    <div id="section-classic" style="margin-bottom:16px">
        <div class="symbol-grid" id="symbol-grid-classic"></div>
    </div>

    <!-- Sectiune SMC -->
    <div id="section-smc" style="display:none;margin-bottom:16px">
        <div class="symbol-grid" id="symbol-grid-smc"></div>
    </div>

    <!-- Detail panel -->
    <div id="detail-panel">
        <div class="detail-header">
            <span class="detail-sym" id="dp-symbol">—</span>
            <span class="badge hold" id="dp-badge">HOLD</span>
            <span class="conf-text" id="dp-conf"></span>
        </div>
        <div class="detail-body">
            <div class="detail-left">
                <table class="tf-table">
                    <thead><tr>
                        <th>TF</th><th>Signal</th><th>Trend</th><th>Convingere</th>
                    </tr></thead>
                    <tbody id="dp-tf-table"></tbody>
                </table>

                <div class="price-row">
                    <div class="price-item">
                        <span class="plabel">Entry</span>
                        <span class="pvalue entry" id="dp-price">—</span>
                    </div>
                    <div class="price-item">
                        <span class="plabel">SL</span>
                        <span class="pvalue sl" id="dp-sl">—</span>
                    </div>
                    <div class="price-item">
                        <span class="plabel">TP</span>
                        <span class="pvalue tp" id="dp-tp">—</span>
                    </div>
                    <div class="price-item">
                        <span class="plabel">Target</span>
                        <span class="pvalue tgt" id="dp-target">—</span>
                    </div>
                    <div class="price-item">
                        <span class="plabel">R:R</span>
                        <span class="pvalue rr" id="dp-rr">—</span>
                    </div>
                </div>

                <div class="justif-box">
                    <ul id="dp-justif"></ul>
                </div>

                <div class="execute-row">
                    <button class="btn btn-execute-buy"  id="btn-exec-buy"  onclick="executeManual('BUY')">▲ Executa BUY</button>
                    <button class="btn btn-execute-sell" id="btn-exec-sell" onclick="executeManual('SELL')">▼ Executa SELL</button>
                </div>
                <div id="execute-result"></div>
            </div>

            <div class="detail-right">
                <div style="display:flex;gap:6px;margin-bottom:6px;flex-wrap:wrap;align-items:center">
                    <span style="font-size:0.75rem;color:#666">Grafic TF:</span>
                    <div id="chart-tf-btns" style="display:flex;gap:4px;flex-wrap:wrap"></div>
                </div>
                <div class="chart-frame-wrap">
                    <iframe id="dp-chart-frame" src="about:blank"></iframe>
                </div>
            </div>
        </div>
    </div>

    <!-- Decisions log -->
    <div class="decisions-section">
        <h3>Istoric decizii (ultimele 20)</h3>
        <table class="dec-table">
            <thead><tr>
                <th>Timestamp</th><th>Simbol</th><th>Signal</th><th>Strategie</th>
                <th>Incredere</th><th>Executat</th><th>Rezultat</th>
            </tr></thead>
            <tbody id="decisions-body"></tbody>
        </table>
    </div>

</div><!-- /main-content -->

<script>
const SYMBOLS_ALL    = {{ symbols_json }};
const SYMBOLS_CRYPTO = {{ crypto_json }};
const TFS_ALL        = ["M1","M5","M15","M30","H1","H4","D1"];
const INTERVALS     = [30, 60, 120, 300];
const DEFAULT_BARS  = 500;

const CLASSIC_ELEMENTS = {ema:"EMA (trend aliniat)", fib:"FIB (Fibonacci zone)", adx:"ADX (forta trend)", rsi:"RSI (zona neutra)"};
const SMC_ELEMENTS     = {bos:"BOS (Break of Structure)", ob:"OB (Order Block)", fvg:"FVG (Fair Value Gap)", structure:"STR (Market Structure)"};

let selectedSymbols = new Set(SYMBOLS_ALL);
// State per strategie
let stratState = {
    classic: { tfs: new Set(["M5","M15","H1"]), tfBars: {}, elements: {ema:true,fib:true,adx:true,rsi:true}, enabled: true, minConfidence: 66 },
    smc:     { tfs: new Set(["M15","H1","H4"]), tfBars: {}, elements: {bos:true,ob:true,fvg:true,structure:true}, enabled: true, minConfidence: 66 },
};
TFS_ALL.forEach(tf => { stratState.classic.tfBars[tf]=DEFAULT_BARS; stratState.smc.tfBars[tf]=DEFAULT_BARS; });

let currentSymbol   = null;
let currentStrategy = "classic";
let currentSignal   = null;
let pollTimer       = null;
let lastDecisionTs  = null;
let settingsOpen    = true;
let mt5SymbolsAll   = [];

// ── Settings toggle ────────────────────────────────────────────────────────
function toggleSettings() {
    settingsOpen = !settingsOpen;
    document.getElementById("settings-panel").classList.toggle("collapsed", !settingsOpen);
}

// ── Master-detail strategii ──────────────────────────────────────────────
// Incarcat dinamic din /autotrader/strategies
let STRAT_DEFS = [
    {key:"classic", label:"Clasica", color:"#26a69a", icon:"🔵"},
    {key:"smc",     label:"SMC",     color:"#ff9800", icon:"🟠"},
];

async function loadStratDefs() {
    try {
        const r = await fetch("/autotrader/strategies");
        const defs = await r.json();
        STRAT_DEFS = defs.map(d => ({
            key:   d.key,
            label: d.name,
            color: d.color,
            icon:  d.icon,
        }));
        // Adauga in stratState strategiile noi (daca nu exista deja)
        defs.forEach(d => {
            if (!stratState[d.key]) {
                const tfsSet = new Set(d.tfs || []);
                const tfBars = {};
                TFS_ALL.forEach(tf => tfBars[tf] = (d.tf_bars || {})[tf] || DEFAULT_BARS);
                stratState[d.key] = {
                    tfs: tfsSet,
                    tfBars: tfBars,
                    elements: d.elements || {},
                    enabled: d.enabled || false,
                    minConfidence: d.min_confidence || 66,
                    elementLabels: d.element_labels || {},
                };
            }
        });
        buildStratList();
        buildStratDetail(_selectedStrat);
        syncTabsWithState();
    } catch(e) { console.warn("loadStratDefs:", e); }
}
let _selectedStrat = "classic";

function buildStratList() {
    const tbody = document.getElementById("strat-list-body");
    if (!tbody) return;
    tbody.innerHTML = "";
    STRAT_DEFS.forEach(({key, label, color, icon}) => {
        const enabled  = stratState[key] ? stratState[key].enabled : true;
        const selected = key === _selectedStrat;
        const tr = document.createElement("tr");
        tr.id = `strat-row-${key}`;
        tr.style.cssText = `cursor:pointer;border-bottom:1px solid #1e1e1e;background:${selected?"#1a2420":"transparent"};border-left:3px solid ${selected?color:"transparent"};transition:background .15s`;
        tr.onmouseover = () => { if (!selected) tr.style.background="#1c1c1c"; };
        tr.onmouseout  = () => { if (!selected) tr.style.background="transparent"; };
        tr.onclick = () => selectStrategy(key);
        tr.innerHTML = `
            <td style="padding:9px 10px;color:${color};font-weight:600;font-size:0.86rem">${icon} ${label}</td>
            <td style="text-align:center;padding:9px 8px">
                <span id="strat-badge-${key}" style="font-size:0.73rem;padding:2px 9px;border-radius:10px;background:${enabled?"#1b3a2a":"#2a2020"};color:${enabled?"#66bb6a":"#ef5350"};font-weight:600">
                    ${enabled ? "● Activa" : "● Oprita"}
                </span>
            </td>
            <td style="padding:9px 6px;color:#444;font-size:0.8rem">${selected?"▶":""}</td>
        `;
        tbody.appendChild(tr);
    });
}

function buildStratDetail(strat) {
    const panel = document.getElementById("strat-detail-panel");
    if (!panel) return;
    const def = STRAT_DEFS.find(d => d.key === strat);
    if (!def) return;
    const {color, label} = def;
    const st = stratState[strat];

    panel.innerHTML = `
        <div style="border-top:2px solid ${color};padding-top:10px">
            <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:10px">
                <span style="color:${color};font-weight:700;font-size:0.88rem">${label}</span>
                <div style="display:flex;align-items:center;gap:8px">
                    <span style="font-size:0.76rem;color:#666">Activa</span>
                    <label class="toggle"><input type="checkbox" id="${strat}-enabled" ${st.enabled?"checked":""}
                        onchange="stratState['${strat}'].enabled=this.checked;sendStrategySet('${strat}','enabled',this.checked);syncTabsWithState()"><span class="slider"></span></label>
                </div>
            </div>
            <div class="config-row" style="margin-bottom:10px">
                <label>Confidence min<span class="sub">% minim semnal</span></label>
                <input type="number" id="${strat}-confidence" value="${st.minConfidence||66}" min="0" max="100" step="1" style="width:55px"
                       onchange="stratState['${strat}'].minConfidence=parseFloat(this.value);sendStrategySet('${strat}','min_confidence',parseFloat(this.value))">
            </div>
            <div style="font-size:0.72rem;color:#555;font-weight:600;letter-spacing:0.5px;margin-bottom:5px">TIMEFRAME-URI &amp; CANDELE</div>
            <table class="tf-bars-table" style="width:100%">
                <thead><tr><th>Activ</th><th>TF</th><th>Candele</th></tr></thead>
                <tbody id="${strat}-tf-body"></tbody>
            </table>
            <div style="font-size:0.72rem;color:#555;font-weight:600;letter-spacing:0.5px;margin-top:10px;margin-bottom:6px">ELEMENTE ACTIVE</div>
            <div id="${strat}-elements" style="display:flex;flex-direction:column;gap:5px"></div>
        </div>`;

    // TF rows
    const tbody = document.getElementById(`${strat}-tf-body`);
    TFS_ALL.forEach(tf => {
        const active = st.tfs.has(tf);
        const tr = document.createElement("tr");
        tr.innerHTML = `
            <td><input type="checkbox" ${active?"checked":""} style="accent-color:${color}"
                onchange="onTfToggle('${strat}','${tf}',this.checked)"></td>
            <td style="font-weight:bold;color:${active?"#ddd":"#555"}" id="${strat}-tf-lbl-${tf}">${tf}</td>
            <td><input type="number" id="${strat}-bars-${tf}" value="${st.tfBars[tf]||DEFAULT_BARS}"
                min="100" max="10000" step="100" ${active?"":"disabled"} style="opacity:${active?1:0.4}"
                onchange="stratState['${strat}'].tfBars['${tf}']=parseInt(this.value)||DEFAULT_BARS;sendStratUpdate('${strat}')"></td>`;
        tbody.appendChild(tr);
    });

    // Elements
    const wrap = document.getElementById(`${strat}-elements`);
    const elDefs = st.elementLabels && Object.keys(st.elementLabels).length
        ? st.elementLabels
        : (strat === "classic" ? CLASSIC_ELEMENTS : SMC_ELEMENTS);
    Object.entries(elDefs).forEach(([key, lbl]) => {
        const checked = st.elements[key] !== false;
        const row = document.createElement("label");
        row.style = "display:flex;align-items:center;gap:8px;cursor:pointer;font-size:0.84rem;color:#bbb";
        row.innerHTML = `<input type="checkbox" ${checked?"checked":""} style="accent-color:${color};width:14px;height:14px"
            onchange="onElementToggle('${strat}','${key}',this.checked)"> ${lbl}`;
        wrap.appendChild(row);
    });
}

function selectStrategy(strat) {
    _selectedStrat = strat;
    switchTab(strat);
    buildStratList();
    buildStratDetail(strat);
}

function buildStrategyTable() {
    buildStratList();
    buildStratDetail(_selectedStrat);
}

// Compatibilitate
function buildTfBarsTable(strat) { if (strat===_selectedStrat) buildStratDetail(strat); }
function buildElementsToggles(strat) { /* inclus in buildStratDetail */ }

function onTfToggle(strat, tf, checked) {
    if (checked) stratState[strat].tfs.add(tf); else stratState[strat].tfs.delete(tf);
    const lbl = document.getElementById(`${strat}-tf-lbl-${tf}`);
    if (lbl) lbl.style.color = checked ? "#ddd" : "#555";
    const inp = document.getElementById(`${strat}-bars-${tf}`);
    if (inp) { inp.disabled = !checked; inp.style.opacity = checked ? 1 : 0.4; }
    sendStratUpdate(strat);
}

function onElementToggle(strat, key, checked) {
    stratState[strat].elements[key] = checked;
    sendStratUpdate(strat);
}

function sendStrategySet(strat, key, value) {
    const body = {[strat]: {[key]: value}};
    fetch("/autotrader/set", {method:"POST", headers:{"Content-Type":"application/json"}, body:JSON.stringify(body)});
}

function sendStratUpdate(strat) {
    const st = stratState[strat];
    const body = {[strat]: {tfs:[...st.tfs], tf_bars:st.tfBars, elements:st.elements}};
    fetch("/autotrader/set", {method:"POST", headers:{"Content-Type":"application/json"}, body:JSON.stringify(body)});
}

// ── Symbol management ──────────────────────────────────────────────────────
async function loadMt5Symbols() {
    try { const r=await fetch("/autotrader/mt5_symbols"); mt5SymbolsAll=await r.json(); } catch(e){}
}

function buildSymChips() {
    const wrap = document.getElementById("sym-chips"); wrap.innerHTML="";
    [...selectedSymbols].forEach(s => {
        const chip=document.createElement("span");
        chip.className="sym-chip active"; chip.dataset.sym=s;
        chip.innerHTML=`${s} <span class="rem-x" onclick="removeSymbol('${s}',event)">✕</span>`;
        wrap.appendChild(chip);
    });
}

function removeSymbol(sym, ev) {
    ev.stopPropagation();
    selectedSymbols.delete(sym);
    buildSymChips();
    const card=document.getElementById("symbol-grid").querySelector(`[data-sym="${sym}"]`);
    if (card) card.remove();
    if (currentSymbol===sym) { currentSymbol=null; document.getElementById("detail-panel").style.display="none"; }
    if (document.body.classList.contains("scanning"))
        fetch("/autotrader/set",{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify({symbols:[...selectedSymbols]})});
}

const SYM_LS_KEY = "cv_watchlist_v1";

function saveWatchlist() {
    try { localStorage.setItem(SYM_LS_KEY, JSON.stringify([...selectedSymbols])); } catch(e){}
}

function loadWatchlist() {
    try {
        const saved = localStorage.getItem(SYM_LS_KEY);
        if (saved) {
            const arr = JSON.parse(saved);
            if (Array.isArray(arr) && arr.length) { selectedSymbols = new Set(arr); return true; }
        }
    } catch(e){}
    return false;
}

function symCategory(name) {
    const n = name.toUpperCase();
    if (/BTC|ETH|XRP|LTC|ADA|SOL|BNB|DOT|DOGE|MATIC|XLM|LINK|UNI|AVAX/.test(n)) return ["crypto","Crypto"];
    if (/XAU|XAG|OIL|BRENT|WTI|GAS|COPPER/.test(n)) return ["metal","Metal/Marfa"];
    if (/NAS|SPX|DOW|DAX|FTSE|CAC|NIK|US30|US500|US100|GER|UK|JP|AUS200/.test(n)) return ["index","Index"];
    if (/EUR|GBP|USD|JPY|CHF|AUD|NZD|CAD/.test(n)) return ["forex","Forex"];
    return ["other","Altele"];
}

function addSymbol(sym) {
    if (!sym) {
        const inp = document.getElementById("sym-add-input");
        sym = inp.value.trim().toUpperCase();
    }
    if (!sym) return;
    document.getElementById("sym-add-input").value = "";
    document.getElementById("sym-autocomplete").style.display = "none";
    document.getElementById("sym-ac-count").textContent = "";
    if (selectedSymbols.has(sym)) return;
    selectedSymbols.add(sym);
    saveWatchlist();
    buildSymChips();
    ["symbol-grid-classic","symbol-grid-smc"].forEach(gid => {
        const grid = document.getElementById(gid);
        if (grid && !grid.querySelector(`[data-sym="${sym}"]`)) {
            const card = document.createElement("div");
            card.className = "sym-card sig-hold"; card.dataset.sym = sym;
            card.innerHTML = `<div class="card-name">${sym}</div><div class="card-scanning">astept scanare...</div>`;
            card.onclick = () => selectCard(sym, null, gid.includes("smc")?"smc":"classic");
            grid.appendChild(card);
        }
    });
    if (document.body.classList.contains("scanning"))
        fetch("/autotrader/set",{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify({symbols:[...selectedSymbols]})});
}

function removeSymbol(sym, ev) {
    if (ev) ev.stopPropagation();
    selectedSymbols.delete(sym);
    saveWatchlist();
    buildSymChips();
    document.querySelectorAll(`[data-sym="${sym}"]`).forEach(c => c.remove());
    if (currentSymbol === sym) { currentSymbol = null; document.getElementById("detail-panel").style.display="none"; }
    if (document.body.classList.contains("scanning"))
        fetch("/autotrader/set",{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify({symbols:[...selectedSymbols]})});
}

let _acIdx = -1;
function onSymInput(inp) {
    _acIdx = -1;
    const val = inp.value.trim().toUpperCase();
    const ac = document.getElementById("sym-autocomplete");
    const cnt = document.getElementById("sym-ac-count");
    if (!val || !mt5SymbolsAll.length) { ac.style.display="none"; cnt.textContent=""; return; }
    const matches = mt5SymbolsAll.filter(s => s.toUpperCase().includes(val));
    cnt.textContent = matches.length ? `${matches.length} rezultate` : "Niciun simbol gasit";
    if (!matches.length) { ac.style.display="none"; return; }
    ac.innerHTML = "";
    matches.slice(0, 20).forEach(s => {
        const [catKey, catLabel] = symCategory(s);
        const already = selectedSymbols.has(s);
        const item = document.createElement("div");
        item.className = "sym-ac-item";
        item.innerHTML = `<span class="ac-name">${s}</span><span class="ac-cat ac-cat-${catKey}">${catLabel}</span><span class="ac-add">${already ? "✓ adaugat" : "+ adauga"}</span>`;
        item.onclick = () => { if (!already) addSymbol(s); };
        ac.appendChild(item);
    });
    if (matches.length > 20) {
        const more = document.createElement("div");
        more.style = "padding:5px 10px;color:#555;font-size:0.75rem";
        more.textContent = `... si inca ${matches.length - 20} rezultate — continua sa scrii`;
        ac.appendChild(more);
    }
    ac.style.display = "block";
}

function onSymKey(e) {
    const ac = document.getElementById("sym-autocomplete");
    const items = ac.querySelectorAll(".sym-ac-item");
    if (e.key === "ArrowDown") { _acIdx = Math.min(_acIdx+1, items.length-1); items.forEach((el,i)=>el.style.background=i===_acIdx?"#2a2a2a":""); e.preventDefault(); }
    else if (e.key === "ArrowUp") { _acIdx = Math.max(_acIdx-1, 0); items.forEach((el,i)=>el.style.background=i===_acIdx?"#2a2a2a":""); e.preventDefault(); }
    else if (e.key === "Enter") { if (_acIdx >= 0 && items[_acIdx]) items[_acIdx].click(); else addSymbol(); }
    else if (e.key === "Escape") { ac.style.display="none"; }
}

document.addEventListener("click", e => {
    if (!e.target.closest(".sym-add-wrap")) document.getElementById("sym-autocomplete").style.display = "none";
});

// ── Interval slider ───────────────────────────────────────────────────────
function onIntervalChange(el) {
    document.getElementById("interval-label").textContent = INTERVALS[parseInt(el.value)]+"s";
}

function sendGlobal(key, value) {
    fetch("/autotrader/set", {method:"POST", headers:{"Content-Type":"application/json"},
          body: JSON.stringify({[key]: value})});
}

function toggleAutoExec(el) {
    const warn = document.getElementById("auto-ex-warn");
    warn.style.display = el.checked ? "inline" : "none";
    fetch("/autotrader/set", {
        method: "POST",
        headers: {"Content-Type": "application/json"},
        body: JSON.stringify({auto_execute: el.checked})
    });
}

function toggleFilter(key, value) {
    fetch("/autotrader/set", {
        method: "POST",
        headers: {"Content-Type": "application/json"},
        body: JSON.stringify({[key]: value})
    });
}

function showToast(msg, color) {
    const t = document.getElementById("toast");
    t.textContent = msg;
    t.style.background = color || "#1b5e20";
    t.classList.add("show");
    setTimeout(() => t.classList.remove("show"), 3000);
}

// ── Scanner control ───────────────────────────────────────────────────────
function startScanner() {
    const intervalIdx = parseInt(document.getElementById("interval-range").value);
    const body = {
        interval:     INTERVALS[intervalIdx],
        symbols:      [...selectedSymbols],
        auto_execute: document.getElementById("auto-exec-toggle").checked,
        classic: { enabled: stratState.classic.enabled, tfs:[...stratState.classic.tfs], tf_bars:stratState.classic.tfBars, elements:stratState.classic.elements },
        smc:     { enabled: stratState.smc.enabled,     tfs:[...stratState.smc.tfs],     tf_bars:stratState.smc.tfBars,     elements:stratState.smc.elements     },
    };
    fetch("/autotrader/start", {method:"POST", headers:{"Content-Type":"application/json"}, body:JSON.stringify(body)})
    .then(() => {
        document.getElementById("btn-start").disabled = true;
        document.getElementById("btn-stop").disabled  = false;
        showToast("▶ Scanner pornit", "#00695c");
        document.body.classList.add("scanning");
        document.getElementById("scanner-banner").classList.add("visible");
        updateGrid({});
    });
}

// ── Strategy tabs ─────────────────────────────────────────────────────────
let _activeTab = "classic";

function switchTab(strat) {
    _activeTab    = strat;
    currentStrategy = strat;
    ["classic","smc"].forEach(s => {
        const active = s === strat;
        const col    = s === "classic" ? "#26a69a" : "#ff9800";
        const tab    = document.getElementById(`tab-${s}`);
        const sec    = document.getElementById(`section-${s}`);
        if (tab) {
            tab.style.color             = active ? col : "#555";
            tab.style.borderBottomColor = active ? col : "transparent";
            tab.style.background        = active ? "#161616" : "#111";
        }
        if (sec) sec.style.display = active ? "block" : "none";
    });
}

function syncTabsWithState() {
    const clsOn = stratState.classic.enabled;
    const smcOn = stratState.smc.enabled;
    // Taburile sunt mereu vizibile (altfel nu poti activa strategia dezactivata)
    // Doar stilul se schimba pentru a indica starea disabled
    const clsTab = document.getElementById("tab-classic");
    const smcTab = document.getElementById("tab-smc");
    if (clsTab) { clsTab.style.display = "inline-block"; clsTab.style.opacity = clsOn ? "1" : "0.5"; }
    if (smcTab) { smcTab.style.display = "inline-block"; smcTab.style.opacity = smcOn ? "1" : "0.5"; }
    // Nu forta schimbarea tab-ului — userul alege manual
    switchTab(_activeTab);
    // Actualizeaza badge-urile din lista de strategii
    STRAT_DEFS.forEach(({key, color}) => {
        const badge = document.getElementById(`strat-badge-${key}`);
        if (!badge) return;
        const on = stratState[key].enabled;
        badge.textContent = on ? "● Activa" : "● Oprita";
        badge.style.background = on ? "#1b3a2a" : "#2a2020";
        badge.style.color = on ? "#66bb6a" : "#ef5350";
    });
}

function toggleCrypto(enabled) {
    if (enabled) {
        SYMBOLS_CRYPTO.forEach(s => {
            if (!selectedSymbols.has(s)) addSymbol(s);
        });
        showToast("₿ Crypto adaugat", "#e65100");
    } else {
        SYMBOLS_CRYPTO.forEach(s => {
            selectedSymbols.delete(s);
            document.querySelectorAll(`[data-sym="${s}"]`).forEach(c => c.remove());
            if (currentSymbol === s) { currentSymbol = null; const dp=document.getElementById("detail-panel"); if(dp) dp.style.display="none"; }
        });
        saveWatchlist();
        buildSymChips();
        if (document.body.classList.contains("scanning"))
            fetch("/autotrader/set",{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify({symbols:[...selectedSymbols]})});
        showToast("Crypto eliminat", "#555");
    }
}

function stopScanner() {
    fetch("/autotrader/stop", {method:"POST"}).then(() => {
        document.getElementById("btn-start").disabled = false;
        document.getElementById("btn-stop").disabled  = true;
        document.body.classList.remove("scanning");
        document.getElementById("scanner-banner").classList.remove("visible");
        showToast("■ Scanner oprit", "#b71c1c");
        stopCountdown();
    });
}

// ── Poll status ───────────────────────────────────────────────────────────
function pollStatus() {
    fetch("/autotrader/status")
        .then(r => r.json())
        .then(data => updateUI(data))
        .catch(e => console.warn("poll error", e));
}

let _countdownTimer = null;
let _countdownVal   = 0;
let _scanInterval   = 60;

function startCountdown(intervalSec) {
    _scanInterval = intervalSec;
    _countdownVal = intervalSec;
    if (_countdownTimer) clearInterval(_countdownTimer);
    const bar   = document.getElementById("scan-progress-bar");
    const fill  = document.getElementById("scan-progress-fill");
    const label = document.getElementById("next-scan-label");
    if (bar) bar.style.display = "block";
    _countdownTimer = setInterval(() => {
        _countdownVal--;
        if (_countdownVal < 0) _countdownVal = _scanInterval;
        const pct = Math.round((1 - _countdownVal / _scanInterval) * 100);
        if (fill)  fill.style.width = pct + "%";
        if (label) label.textContent = `urm. scan in ${_countdownVal}s`;
    }, 1000);
}

function stopCountdown() {
    if (_countdownTimer) { clearInterval(_countdownTimer); _countdownTimer = null; }
    const bar   = document.getElementById("scan-progress-bar");
    const fill  = document.getElementById("scan-progress-fill");
    const label = document.getElementById("next-scan-label");
    if (bar)   bar.style.display = "none";
    if (fill)  fill.style.width = "0%";
    if (label) label.textContent = "";
}

function updateUI(data) {
    const sc = data.scanner;

    // Status bar
    const dot  = document.getElementById("status-dot");
    const stxt = document.getElementById("status-text");
    if (sc.running) {
        dot.className  = "status-dot running";
        stxt.textContent = "Scanner activ — ruleaza pe server (independent de browser)";
        document.getElementById("btn-start").disabled = true;
        document.getElementById("btn-stop").disabled  = false;
        document.body.classList.add("scanning");
        if (!_countdownTimer) startCountdown(sc.interval || 60);
    } else {
        dot.className  = "status-dot";
        stxt.textContent = "Scanner oprit";
        document.getElementById("btn-start").disabled = false;
        document.getElementById("btn-stop").disabled  = true;
        document.body.classList.remove("scanning");
        stopCountdown();
    }
    document.getElementById("scan-count").textContent = sc.scan_count || 0;
    document.getElementById("last-scan").textContent  = sc.last_scan
        ? sc.last_scan.substring(11,19)
        : "—";

    // Banner
    const banner = document.getElementById("scanner-banner");
    if (sc.running) {
        banner.classList.add("visible");
        const bs = document.getElementById("banner-scans");
        if (bs) bs.textContent = sc.scan_count || 0;
    } else {
        banner.classList.remove("visible");
    }
    const bc = document.getElementById("banner-countdown");
    if (bc && _countdownVal > 0) bc.textContent = _countdownVal + "s";

    // Symbol grid
    updateGrid(data.results);

    // Decisions
    updateDecisions(data.decisions);

    // Auto-select first non-HOLD DOAR daca utilizatorul nu a selectat nimic inca
    const decs = data.decisions || [];
    if (decs.length > 0) {
        const newest = decs[0].timestamp;
        if (newest !== lastDecisionTs) {
            lastDecisionTs = newest;
            if (!currentSymbol) {
                const first = decs[0];
                if (first.signal !== "HOLD" && data.results[first.symbol]) {
                    selectCard(first.symbol, data.results[first.symbol]);
                }
            }
        }
    }
}

function buildCard(sym, res, gridId, strat) {
    const grid = document.getElementById(gridId);
    let card = grid.querySelector(`[data-sym="${sym}"]`);
    if (!card) {
        card = document.createElement("div");
        card.className = "sym-card sig-hold";
        card.dataset.sym = sym;
        grid.appendChild(card);
    }

    const sig  = res ? res.signal : "HOLD";
    const conf = res ? res.confidence : 0;
    const ts   = res ? (res.timestamp||"").substring(11,19) : "";
    const isScanning = document.body.classList.contains("scanning");

    card.className = `sym-card sig-${sig.toLowerCase()}` +
        (currentSymbol===sym && currentStrategy===strat ? " selected" : "") +
        (isScanning && !res ? " scanning-card" : "");

    const stratColor = strat === "classic" ? "#26a69a" : "#ff9800";
    card.innerHTML = isScanning && !res
        ? `<div class="card-name">${sym}</div><div class="card-scanning"><span class="scan-spin">⟳</span> scanez...</div>`
        : `<div class="card-name">${sym}</div>
           <div class="card-signal ${sig.toLowerCase()}">${sig}</div>
           <div class="card-conf">${conf>0?conf.toFixed(1)+"% incredere":"—"}</div>
           <div class="card-time" style="color:${stratColor}">${ts}</div>`;
    card.onclick = () => selectCard(sym, res, strat);
}

function updateGrid(results) {
    const syms = [...selectedSymbols];
    syncTabsWithState();

    // Construieste doar grila tab-ului activ (performanta)
    ["classic","smc"].forEach(strat => {
        const grid = document.getElementById(`symbol-grid-${strat}`);
        if (!grid) return;
        // Sterge simboluri disparute
        grid.querySelectorAll(".sym-card").forEach(c => {
            if (!selectedSymbols.has(c.dataset.sym)) c.remove();
        });
        if (!stratState[strat].enabled) { grid.innerHTML = ""; return; }
        syms.forEach(sym => {
            const symData = results[sym] || {};
            buildCard(sym, symData[strat] || null, `symbol-grid-${strat}`, strat);
        });
    });

    if (currentSymbol && results[currentSymbol]) {
        const res = results[currentSymbol][currentStrategy];
        if (res) refreshDetailIfSelected(res);
    }
}

function trendRo(t) {
    return {ASCENDING:"▲ Ascendent", DESCENDING:"▼ Descendent", RANGING:"— Lateral"}[t] || t;
}

function selectCard(sym, res, strat) {
    currentSymbol   = sym;
    currentStrategy = strat || "classic";
    document.querySelectorAll(".sym-card").forEach(c => {
        c.classList.toggle("selected", c.dataset.sym === sym && c.closest(`#symbol-grid-${currentStrategy}`) !== null);
    });
    showDetailPanel(sym, res);
}

function refreshDetailIfSelected(res) {
    if (!res || res.symbol !== currentSymbol) return;
    currentSignal = res.signal;
    // Only refresh non-chart parts to avoid iframe reload
    updateDetailContent(res, false);
}

function showDetailPanel(sym, res) {
    const panel = document.getElementById("detail-panel");
    panel.style.display = "block";
    panel.scrollIntoView({behavior:"smooth", block:"nearest"});

    if (!res) {
        document.getElementById("dp-symbol").textContent = sym;
        document.getElementById("dp-badge").className = "badge hold";
        document.getElementById("dp-badge").textContent = "SEM NONE";
        return;
    }

    currentSignal = res.signal;
    updateDetailContent(res, true);
}

function updateDetailContent(res, updateChart) {
    document.getElementById("dp-symbol").textContent = res.symbol;

    const badge = document.getElementById("dp-badge");
    badge.textContent = res.signal;
    badge.className   = "badge " + res.signal.toLowerCase();

    document.getElementById("dp-conf").textContent =
        res.confidence > 0 ? res.confidence.toFixed(1) + "% incredere" : "";

    // TF table
    const tbody = document.getElementById("dp-tf-table");
    tbody.innerHTML = "";
    (res.tfs || []).forEach(r => {
        const cls = r.signal === "BUY" ? "sig-buy" : r.signal === "SELL" ? "sig-sell" : "sig-hold";
        const tr = document.createElement("tr");
        tr.innerHTML = `
            <td>${r.tf}</td>
            <td class="${cls}">${r.signal}</td>
            <td>${trendRo(r.trend)}</td>
            <td>${"★".repeat(r.conviction)}</td>
        `;
        tbody.appendChild(tr);
    });

    // Price info
    const bf = res.best_tf || {};
    document.getElementById("dp-price").textContent  = bf.price  || "—";
    document.getElementById("dp-sl").textContent     = bf.sl     || "—";
    document.getElementById("dp-tp").textContent     = bf.tp     || "—";
    document.getElementById("dp-target").textContent = res.target || "—";

    // R:R
    let rrText = "—";
    if (bf.price && bf.sl && bf.tp) {
        const risk   = Math.abs(bf.price - bf.sl);
        const reward = Math.abs(bf.tp   - bf.price);
        if (risk > 0) rrText = "1:" + (reward/risk).toFixed(1);
    }
    document.getElementById("dp-rr").textContent = rrText;

    // Justification
    const ul = document.getElementById("dp-justif");
    ul.innerHTML = "";
    (res.justification || []).forEach(line => {
        const li = document.createElement("li");
        li.textContent = line;
        ul.appendChild(li);
    });

    // Execute buttons — mereu vizibile, highlight pe cel recomandat
    const btnBuy  = document.getElementById("btn-exec-buy");
    const btnSell = document.getElementById("btn-exec-sell");
    btnBuy.style.opacity  = res.signal === "BUY"  ? "1" : "0.4";
    btnSell.style.opacity = res.signal === "SELL" ? "1" : "0.4";

    // Chart iframe + TF selector tabs
    const availTfs = (res.tfs || []).map(r => r.tf);
    const chartTf  = bf.tf || (availTfs.length > 0 ? availTfs[0] : "M5");
    buildChartTfTabs(res.symbol, availTfs, chartTf);
    if (updateChart) {
        loadChartTf(res.symbol, chartTf);
    }

    // Clear old execute result
    const exRes = document.getElementById("execute-result");
    exRes.style.display = "none";
}

// ── Chart TF tabs ─────────────────────────────────────────────────────────
function buildChartTfTabs(symbol, tfs, activeTf) {
    const wrap = document.getElementById("chart-tf-btns");
    wrap.innerHTML = "";
    tfs.forEach(tf => {
        const btn = document.createElement("button");
        btn.className = "tf-tab" + (tf === activeTf ? " active" : "");
        btn.textContent = tf;
        btn.onclick = () => {
            wrap.querySelectorAll(".tf-tab").forEach(b => b.classList.remove("active"));
            btn.classList.add("active");
            loadChartTf(symbol, tf);
        };
        wrap.appendChild(btn);
    });
}

function loadChartTf(symbol, tf) {
    const bars = (stratState[currentStrategy]||stratState.classic).tfBars[tf] || DEFAULT_BARS;
    document.getElementById("dp-chart-frame").src = `/autotrader/chart/${symbol}/${tf}?bars=${bars}`;
}

// ── Manual execute ────────────────────────────────────────────────────────
function executeManual(signal) {
    if (!currentSymbol) return;
    const exRes = document.getElementById("execute-result");
    exRes.style.display = "none";

    fetch("/autotrader/status")
        .then(r => r.json())
        .then(data => {
            const res = data.results[currentSymbol];
            const bf  = res && res.best_tf ? res.best_tf : {};
            return fetch("/autotrader/execute", {
                method: "POST",
                headers: {"Content-Type":"application/json"},
                body: JSON.stringify({
                    symbol: currentSymbol,
                    signal: signal,
                    sl: bf.sl || 0,
                    tp: bf.tp || 0,
                })
            });
        })
        .then(r => r.json())
        .then(d => {
            exRes.style.display = "block";
            exRes.className = d.ok ? "exec-ok" : "exec-err";
            exRes.textContent = d.message;
        })
        .catch(e => {
            exRes.style.display = "block";
            exRes.className = "exec-err";
            exRes.textContent = "Eroare: " + e;
        });
}

// ── Decisions log ─────────────────────────────────────────────────────────
function updateDecisions(decs) {
    const tbody = document.getElementById("decisions-body");
    tbody.innerHTML = "";
    (decs || []).slice(0, 20).forEach(d => {
        const cls = d.signal === "BUY" ? "sig-buy"
                  : d.signal === "SELL" ? "sig-sell"
                  : d.signal === "EARLY EXIT" ? "sig-early"
                  : d.signal === "CLOSE" ? "sig-close"
                  : "sig-hold";
        const tr = document.createElement("tr");
        const stratLabel = d.strategy === "smc" ? `<span style="color:#ff9800">SMC</span>` : `<span style="color:#26a69a">Classic</span>`;
        tr.innerHTML = `
            <td>${d.timestamp.substring(0,19).replace("T"," ")}</td>
            <td>${d.symbol}</td>
            <td class="${cls}">${d.signal}</td>
            <td>${stratLabel}</td>
            <td>${d.confidence ? d.confidence.toFixed(1)+"%" : "—"}</td>
            <td class="${d.executed ? "dec-yes" : "dec-no"}">${d.executed ? "DA" : "NU"}</td>
            <td style="color:#666;font-size:0.76rem">${(d.result||"").substring(0,80)}</td>
        `;
        tbody.appendChild(tr);
    });
}

// ── FTMO poll ─────────────────────────────────────────────────────────────
async function pollFtmo() {
    try {
        const r=await fetch("/ftmo_status"), d=await r.json();
        const el=document.getElementById("ftmo-indicator"); if(!el) return;
        if (!d.ftmo_enabled) { el.style.display="none"; return; }
        el.style.display="block";
        if (d.ok) {
            let txt="✓ FTMO OK";
            if (d.next_news) txt+=` · stiri in ${d.next_news.in_minutes}min`;
            if (d.daily_used_pct>0) txt+=` · DD: ${d.daily_used_pct}%`;
            el.textContent=txt;
            el.style.background=d.daily_used_pct>3?"#b71c1c":"#1b5e20";
            el.style.color=d.daily_used_pct>3?"#ef9a9a":"#a5d6a7";
        } else { el.textContent=`⛔ ${d.message}`; el.style.background="#b71c1c"; el.style.color="#ef9a9a"; }
    } catch(e){}
}
pollFtmo(); setInterval(pollFtmo, 10000);

// ── Init ─────────────────────────────────────────────────────────────────
loadStratDefs();  // incarca toate strategiile din backend
buildStrategyTable();
// Restaureaza watchlist din localStorage (daca exista), altfel foloseste lista default
if (!loadWatchlist()) { selectedSymbols = new Set(SYMBOLS_ALL); }
buildSymChips();
loadMt5Symbols();
// Seteaza checkbox crypto daca watchlist-ul contine simboluri crypto
{ const hasCrypto = SYMBOLS_CRYPTO.some(s => selectedSymbols.has(s));
  const ct = document.getElementById("crypto-toggle"); if(ct) ct.checked = hasCrypto; }
switchTab("classic"); // aplica tab-ul initial corect

fetch("/autotrader/status").then(r=>r.json()).then(data=>{
    if (data.max_open_trades != null) document.getElementById("max-trades-input").value = data.max_open_trades;
    if (data.tp_ratio != null) document.getElementById("tp-ratio-input").value = data.tp_ratio;
    const sc=data.scanner;
    const toggle=document.getElementById("auto-exec-toggle");
    if (toggle) { toggle.checked=sc.auto_execute||false; document.getElementById("auto-ex-warn").style.display=sc.auto_execute?"inline":"none"; }
    if (sc.symbols) { selectedSymbols=new Set(sc.symbols); buildSymChips(); }
    if (sc.classic) {
        if (sc.classic.tfs) stratState.classic.tfs = new Set(sc.classic.tfs);
        if (sc.classic.tf_bars) Object.assign(stratState.classic.tfBars, sc.classic.tf_bars);
        if (sc.classic.elements) Object.assign(stratState.classic.elements, sc.classic.elements);
        if (sc.classic.enabled != null) stratState.classic.enabled = sc.classic.enabled;
        if (sc.classic.min_confidence != null) stratState.classic.minConfidence = sc.classic.min_confidence;
    }
    if (sc.smc) {
        if (sc.smc.tfs) stratState.smc.tfs = new Set(sc.smc.tfs);
        if (sc.smc.tf_bars) Object.assign(stratState.smc.tfBars, sc.smc.tf_bars);
        if (sc.smc.elements) Object.assign(stratState.smc.elements, sc.smc.elements);
        if (sc.smc.enabled != null) stratState.smc.enabled = sc.smc.enabled;
        if (sc.smc.min_confidence != null) stratState.smc.minConfidence = sc.smc.min_confidence;
    }
    buildStrategyTable();
    if (sc.running) {
        document.getElementById("btn-start").disabled=true;
        document.getElementById("btn-stop").disabled=false;
        document.body.classList.add("scanning");
        document.getElementById("scanner-banner").classList.add("visible");
    }
    updateGrid(data.results||{});
    updateDecisions(data.decisions||[]);
});
pollStatus();
pollTimer = setInterval(pollStatus, 1000);

// Preselect symbol din URL
const _preselect = "{{ preselect_symbol }}";
const _decideNow = "{{ decide_now }}" === "1";
if (_preselect) {
    // Porneste scanner direct si analizeaza simbolul selectat
    setTimeout(() => {
        if (_decideNow) {
            fetch("/autotrader/start", {
                method: "POST",
                headers: {"Content-Type":"application/json"},
                body: JSON.stringify({
                    interval: 60,
                    symbols: [_preselect],
                    auto_execute: false,
                })
            }).then(() => {
                // Dupa 3s, afiseaza rezultatul
                setTimeout(() => {
                    fetch("/autotrader/status")
                        .then(r => r.json())
                        .then(data => {
                            updateUI(data);
                            if (data.results[_preselect]) {
                                selectCard(_preselect, data.results[_preselect]);
                            }
                        });
                }, 3000);
            });
        }
    }, 500);
}
</script>
</body></html>
"""


# ── Blueprint routes ───────────────────────────────────────────────────────────
@autotrader_bp.route("/autotrader")
@login_required
def autotrader_page():
    import json
    symbols_json = json.dumps(SYMBOLS)
    crypto_json  = json.dumps(SYMBOLS_CRYPTO)
    preselect    = request.args.get("symbol", "")
    decide_now   = request.args.get("decide", "0")
    html = AUTOTRADER_HTML.replace("{{ symbols_json }}", symbols_json) \
                          .replace("{{ crypto_json }}", crypto_json) \
                          .replace("{{ preselect_symbol }}", preselect) \
                          .replace("{{ decide_now }}", decide_now)
    return Response(html, content_type="text/html; charset=utf-8")


@autotrader_bp.route("/autotrader/strategies")
@login_required
def autotrader_strategies():
    """Returneaza lista tuturor strategiilor disponibile + config din scanner."""
    import strategies as _sp
    import json as _json
    defs = []
    for s in _sp.list_all():
        cfg = scanner.get(s.key, {})
        defs.append({
            "key":            s.key,
            "name":           s.name,
            "icon":           s.icon,
            "color":          s.color,
            "enabled":        cfg.get("enabled", False),
            "tfs":            cfg.get("tfs", s.default_tfs),
            "tf_bars":        cfg.get("tf_bars", {}),
            "elements":       cfg.get("elements", {k: True for k in s.elements}),
            "element_labels": s.elements,
            "min_confidence": cfg.get("min_confidence", 66.0),
        })
    return Response(_json.dumps(defs), mimetype="application/json")


@autotrader_bp.route("/autotrader/status")
@login_required
def autotrader_status():
    with _scanner_lock:
        res_copy  = dict(results)
        dec_copy  = list(decisions)
        scan_copy = dict(scanner)
    payload = {
        "scanner":        scan_copy,
        "results":        res_copy,
        "decisions":      dec_copy[:20],
        "max_open_trades": _app.MAX_OPEN_TRADES,
        "tp_ratio":        _app.TP_RATIO,
    }
    return Response(json.dumps(payload, cls=NpEncoder), content_type="application/json")


def _apply_strategy_config(body):
    """Aplica configuratia oricarei strategii din body in scanner."""
    import strategies as _sp
    all_keys = {s.key for s in _sp.list_all()}
    for key in all_keys:
        if key not in body:
            continue
        cfg = body[key]
        if key not in scanner:
            scanner[key] = {"enabled": False, "tfs": [], "tf_bars": {}, "elements": {}, "min_confidence": 66.0}
        if "enabled" in cfg:
            scanner[key]["enabled"] = bool(cfg["enabled"])
        if "tfs" in cfg:
            scanner[key]["tfs"] = [t for t in cfg["tfs"] if t in ALL_TFS]
        if "tf_bars" in cfg:
            scanner[key].setdefault("tf_bars", {}).update(
                {k: int(v) for k, v in cfg["tf_bars"].items() if k in ALL_TFS}
            )
        if "elements" in cfg:
            scanner[key].setdefault("elements", {}).update(
                {k: bool(v) for k, v in cfg["elements"].items()}
            )
        if "min_confidence" in cfg:
            scanner[key]["min_confidence"] = max(0.0, float(cfg["min_confidence"]))


@autotrader_bp.route("/autotrader/start", methods=["POST"])
@login_required
def autotrader_start():
    body = request.get_json(silent=True) or {}
    if "interval" in body:
        scanner["interval"] = int(body["interval"])
    if "symbols" in body:
        scanner["symbols"] = [s for s in body["symbols"] if s]
    if "auto_execute" in body:
        scanner["auto_execute"] = bool(body["auto_execute"])
    _apply_strategy_config(body)
    start_scanner()
    return Response(json.dumps({"ok": True}), content_type="application/json")


@autotrader_bp.route("/autotrader/stop", methods=["POST"])
@login_required
def autotrader_stop():
    stop_scanner()
    return Response(json.dumps({"ok": True}), content_type="application/json")


@autotrader_bp.route("/autotrader/set", methods=["POST"])
@login_required
def autotrader_set():
    """Modifica setari fara sa reporneasca scannerul."""
    body = request.get_json(silent=True) or {}
    if "auto_execute" in body:
        scanner["auto_execute"] = bool(body["auto_execute"])
    if "interval" in body:
        scanner["interval"] = int(body["interval"])
    if "symbols" in body:
        scanner["symbols"] = [s for s in body["symbols"] if s]
    if "max_open_trades" in body:
        _app.MAX_OPEN_TRADES = max(1, int(body["max_open_trades"]))
    if "tp_ratio" in body:
        _app.TP_RATIO = max(0.1, float(body["tp_ratio"]))
    _apply_strategy_config(body)
    return Response(json.dumps({"ok": True}), content_type="application/json")


@autotrader_bp.route("/autotrader/execute", methods=["POST"])
@login_required
def autotrader_execute():
    try:
        body = request.get_json(silent=True) or {}
        symbol = body.get("symbol", "").upper()
        signal = body.get("signal", "")
        sl     = float(body.get("sl", 0) or 0)
        tp     = float(body.get("tp", 0) or 0)

        if not symbol or signal not in ("BUY", "SELL"):
            return Response(
                json.dumps({"ok": False, "message": f"Parametri invalizi: symbol={symbol} signal={signal}"}),
                content_type="application/json"
            )

        ok, msg = place_trade(symbol, signal, sl, tp, RISK_DOLLARS)
        decision = {
            "timestamp":  datetime.now().isoformat(),
            "symbol":     symbol,
            "signal":     signal,
            "confidence": results.get(symbol, {}).get("confidence", 0),
            "executed":   ok,
            "result":     msg,
        }
        with _scanner_lock:
            decisions.insert(0, decision)
            while len(decisions) > 50:
                decisions.pop()

        return Response(
            json.dumps({"ok": ok, "message": msg}, cls=NpEncoder),
            content_type="application/json"
        )
    except Exception as e:
        log.error(f"autotrader_execute error: {e}")
        return Response(
            json.dumps({"ok": False, "message": str(e)}),
            content_type="application/json"
        )


@autotrader_bp.route("/autotrader/switch_market", methods=["POST"])
@login_required
def switch_market():
    """Comuta intre piata Forex si Crypto."""
    body = request.get_json(silent=True) or {}
    market = body.get("market", "forex")
    if market == "crypto":
        scanner["symbols"] = list(SYMBOLS_CRYPTO)
    else:
        scanner["symbols"] = list(SYMBOLS)
    scanner["market_mode"] = market
    return Response(json.dumps({"ok": True, "market": market, "symbols": scanner["symbols"]}),
                    content_type="application/json")


@autotrader_bp.route("/autotrader/mt5_symbols")
@login_required
def mt5_symbols_list():
    """Returneaza lista simbolurilor disponibile in MT5."""
    if MT5_AVAILABLE and mt5 is not None:
        try:
            syms = mt5.symbols_get()
            if syms:
                names = sorted([s.name for s in syms])
                return Response(json.dumps(names), content_type="application/json")
        except Exception:
            pass
    return Response(json.dumps(list(SYMBOLS)), content_type="application/json")


@autotrader_bp.route("/autotrader/review_trades", methods=["POST"])
@login_required
def autotrader_review_trades():
    """Declanseaza manual un review al trade-urilor deschise."""
    try:
        closed = review_open_trades(
            [], 500, True,
            cls_tfs=scanner["classic"]["tfs"] if scanner["classic"]["enabled"] else [],
            smc_tfs=scanner["smc"]["tfs"] if scanner["smc"]["enabled"] else [],
        )
        for ec in closed:
            with _scanner_lock:
                decisions.insert(0, {
                    "timestamp":  datetime.now().isoformat(),
                    "symbol":     ec["symbol"],
                    "signal":     "EARLY EXIT",
                    "confidence": 0,
                    "executed":   True,
                    "result":     f"profit={ec['profit']}$ — {ec['reason']}",
                })
        return Response(json.dumps({"ok": True, "closed": len(closed), "details": closed}, cls=NpEncoder),
                        content_type="application/json")
    except Exception as e:
        return Response(json.dumps({"ok": False, "message": str(e)}), content_type="application/json")


@autotrader_bp.route("/autotrader/chart/<symbol>/<tf>")
@login_required
def autotrader_chart(symbol, tf):
    if tf not in ALL_TFS:
        return Response("<div style='color:#ef5350;padding:12px'>TF invalid.</div>",
                        content_type="text/html")
    bars = int(request.args.get("bars", 300))
    chart_html = build_chart(symbol, tf, bars, compact=True)
    full_html = f"""<!DOCTYPE html>
<html><head>
<meta charset="utf-8">
<style>
* {{ margin:0; padding:0; box-sizing:border-box; }}
body {{ background:#111; }}
</style>
</head><body>
{chart_html}
</body></html>"""
    return Response(full_html, content_type="text/html; charset=utf-8")
