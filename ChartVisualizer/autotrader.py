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
    SYMBOLS, ALL_TFS, RISK_DOLLARS, MIN_TF_VOTES, MIN_CONFIDENCE,
    fetch, find_pivots, detect_trend, calc_entry, calc_sl_tp, place_trade,
    NpEncoder, MT5_AVAILABLE, mt5, build_chart,
    get_upcoming_red_news, close_all_positions_for_news, FTMO_NEWS_BLOCK_MIN,
    get_h4_direction, in_trading_session, calc_adx, ADX_MIN,
)

log = logging.getLogger(__name__)

autotrader_bp = Blueprint("autotrader", __name__)

# ── Scanner state ─────────────────────────────────────────────────────────────
scanner = {
    "running": False,
    "interval": 60,
    "auto_execute": False,
    "use_h4_filter": True,   # filtru H4 direction activ implicit
    "use_session_filter": True,  # filtru sesiuni London/NY activ implicit
    "tfs": ["M1", "M5", "M15", "H1"],
    "bars": 500,
    "symbols": list(SYMBOLS),
    "last_scan": None,
    "scan_count": 0,
}
results = {}    # {symbol: analyze_symbol_full result}
decisions = []  # list of taken decisions (max 50)
_scanner_thread = None
_scanner_lock = threading.Lock()


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
def analyze_symbol_full(symbol, tfs, bars=500):
    """Analizeaza un simbol pe mai multe timeframe-uri si returneaza un dict complet."""
    tf_results = []

    for tf in tfs:
        try:
            df, _ = fetch(symbol, tf, bars)
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


# ── Trade review — early exit daca trendul s-a inversat ──────────────────────
def review_open_trades(tfs, bars, auto_ex):
    """
    Verifica fiecare pozitie deschisa:
    - Daca semnalul s-a inversat (BUY deschis dar acum SELL) → inchide early
    - Daca trendul pe TF principal s-a schimbat → inchide early
    Inchide doar daca e in pierdere sau la break-even (nu taie profiturile).
    """
    if not MT5_AVAILABLE or mt5 is None or not auto_ex:
        return []

    positions = mt5.positions_get()
    if not positions:
        return []

    closed_early = []
    review_tfs   = [t for t in tfs if t in ("M15", "H1")]  # TF-uri relevante pentru review
    if not review_tfs:
        review_tfs = tfs[:2]

    for pos in positions:
        try:
            symbol    = pos.symbol
            pos_type  = "BUY" if pos.type == 0 else "SELL"
            profit    = pos.profit
            price_now = pos.price_current
            price_open= pos.price_open

            # Analizeaza pe TF principal (M15 sau H1)
            tf_check = review_tfs[-1]  # cel mai lent TF din lista
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


# ── Background scanner ────────────────────────────────────────────────────────
def _scanner_loop():
    log.info("AutoTrader scanner pornit.")
    _last_news_close = None  # evita inchideri repetate pentru aceeasi stire
    while scanner["running"]:
        try:
            symbols  = list(scanner["symbols"])
            tfs      = list(scanner["tfs"])
            bars     = int(scanner["bars"])
            auto_ex  = scanner["auto_execute"]

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
                        with _scanner_lock:
                            decisions.insert(0, {
                                "timestamp": datetime.now().isoformat(),
                                "symbol":    ", ".join(closed),
                                "signal":    "CLOSE",
                                "confidence": 100,
                                "executed":  True,
                                "result":    msg,
                            })
                scanner["news_block"] = f"⛔ Stire rosie: {upcoming[0]['title']} ({upcoming[0]['dt']})"
            else:
                scanner["news_block"] = None

            # ── Review trade-uri deschise — early exit daca trend inversat ──
            if auto_ex:
                early_closed = review_open_trades(tfs, bars, auto_ex)
                for ec in early_closed:
                    with _scanner_lock:
                        decisions.insert(0, {
                            "timestamp":  datetime.now().isoformat(),
                            "symbol":     ec["symbol"],
                            "signal":     "EARLY EXIT",
                            "confidence": 0,
                            "executed":   True,
                            "result":     f"profit={ec['profit']}$ — {ec['reason']}",
                        })
                        while len(decisions) > 50:
                            decisions.pop()

            for sym in symbols:
                if not scanner["running"]:
                    break
                try:
                    res = analyze_symbol_full(sym, tfs, bars)
                    with _scanner_lock:
                        results[sym] = res

                    if res["signal"] != "HOLD" and auto_ex and res.get("session_ok", True):
                        bf = res["best_tf"]
                        if bf:
                            ok, msg = place_trade(sym, res["signal"], bf["sl"], bf["tp"], RISK_DOLLARS)
                            res["auto_executed"] = True
                            decision = {
                                "timestamp":  datetime.now().isoformat(),
                                "symbol":     sym,
                                "signal":     res["signal"],
                                "confidence": res["confidence"],
                                "executed":   ok,
                                "result":     msg,
                            }
                            with _scanner_lock:
                                decisions.insert(0, decision)
                                while len(decisions) > 50:
                                    decisions.pop()
                            log.info(f"AutoTrade {sym} {res['signal']}: {msg}")
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
body { background:#111; color:#eee; font-family:'Segoe UI',monospace; font-size:14px; }

/* TOP BAR */
.topbar {
    background:#1a1a1a; border-bottom:2px solid #333;
    padding:10px 16px; display:flex; flex-wrap:wrap; gap:14px; align-items:flex-end;
    transition: background 0.4s, border-color 0.4s;
}
body.scanning .topbar {
    background:#0d1f1a;
    border-bottom-color:#26a69a;
    box-shadow: 0 2px 12px rgba(38,166,154,0.15);
}
.topbar-title { font-size:1.05rem; color:#aaa; font-weight:400; align-self:center; margin-right:6px; transition:color 0.3s; }
body.scanning .topbar-title { color:#26a69a; }
.topbar-section { display:flex; flex-direction:column; gap:4px; }
.topbar-section label { font-size:0.74rem; color:#888; }
.topbar-row { display:flex; gap:6px; align-items:center; flex-wrap:wrap; }

/* checkboxes */
.cb-group { display:flex; gap:4px; flex-wrap:wrap; }
.cb-item {
    display:flex; align-items:center; gap:3px;
    background:#2a2a2a; border:1px solid #444; border-radius:4px;
    padding:3px 8px; cursor:pointer; font-size:0.78rem; color:#ccc;
    user-select:none; transition:background 0.15s,border-color 0.15s;
}
.cb-item input { display:none; }
.cb-item.checked { background:#4a148c; border-color:#9c27b0; color:#fff; }

/* inputs */
input[type=number], input[type=text], select {
    background:#2a2a2a; color:#eee; border:1px solid #444;
    padding:5px 8px; border-radius:4px; font-size:0.83rem;
}
input[type=range] { accent-color:#9c27b0; width:120px; cursor:pointer; }

/* buttons */
.btn {
    background:#1976d2; color:#fff; border:none;
    padding:6px 14px; border-radius:4px; cursor:pointer;
    font-size:0.83rem; text-decoration:none; display:inline-block;
    transition:background 0.15s;
}
.btn:hover { background:#1565c0; }
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
input:checked + .slider { background:#c62828; }
input:checked + .slider:before { transform:translateX(20px); background:#fff; }
.auto-ex-warn { color:#ef5350; font-size:0.75rem; font-weight:bold; display:none; }

/* MAIN GRID */
.main-content { padding:12px 16px; }
.grid-header { display:flex; justify-content:space-between; align-items:center; margin-bottom:10px; }
.grid-header h2 { font-size:0.9rem; color:#888; font-weight:400; }
.status-dot { display:inline-block; width:9px; height:9px; border-radius:50%; margin-right:5px; background:#555; }
.status-dot.running { background:#26a69a; animation:pulse 1.2s infinite; box-shadow:0 0 6px #26a69a; }
@keyframes pulse { 0%,100%{opacity:1;transform:scale(1)} 50%{opacity:0.5;transform:scale(1.3)} }
.scan-progress-bar {
    height:3px; background:#333; border-radius:2px; margin-top:6px; overflow:hidden;
}
.scan-progress-fill {
    height:100%; background:#26a69a; border-radius:2px;
    transition: width 1s linear;
}
.next-scan-label { font-size:0.75rem; color:#888; }
body.scanning .next-scan-label { color:#4db6ac; }

.symbol-grid {
    display:grid;
    grid-template-columns:repeat(auto-fill, minmax(170px, 1fr));
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
.card-name { font-size:0.92rem; font-weight:bold; color:#ddd; margin-bottom:4px; }
.card-signal { font-size:1.1rem; font-weight:bold; margin-bottom:3px; }
.card-signal.buy  { color:#26a69a; }
.card-signal.sell { color:#ef5350; }
.card-signal.hold { color:#666; }
.card-conf { font-size:0.75rem; color:#888; margin-bottom:2px; }
.card-trend { font-size:0.74rem; color:#777; margin-bottom:2px; }
.card-time  { font-size:0.7rem; color:#555; }
.scanning-card { border-left:3px solid #37474f !important; opacity:0.7; }
.card-scanning { font-size:0.82rem; color:#607d8b; margin-top:4px; }
.scan-spin { display:inline-block; animation:spin 1s linear infinite; }
@keyframes spin { to { transform:rotate(360deg); } }

/* Banner activ */
.scanner-banner {
    display:none; background:#0a2218; border:1px solid #1b5e20;
    color:#a5d6a7; padding:8px 20px; font-size:0.82rem;
    align-items:center; gap:12px;
}
.scanner-banner.visible { display:flex; }
.banner-dot { width:8px; height:8px; border-radius:50%; background:#26a69a; animation:pulse 1.2s infinite; flex-shrink:0; }
.banner-text { flex:1; }
.banner-text b { color:#66bb6a; }

/* Toast notificare */
.toast {
    position:fixed; bottom:24px; right:24px; z-index:9999;
    background:#1b5e20; color:#a5d6a7; padding:12px 20px; border-radius:6px;
    font-size:0.85rem; box-shadow:0 4px 16px rgba(0,0,0,0.5);
    transform:translateY(80px); opacity:0;
    transition:transform 0.3s, opacity 0.3s;
    pointer-events:none;
}
.toast.show { transform:translateY(0); opacity:1; }

/* DETAIL PANEL */
#detail-panel {
    background:#1a1a1a; border:1px solid #333; border-radius:6px;
    padding:16px 18px; margin-bottom:16px; display:none;
}
.detail-header { display:flex; align-items:center; gap:14px; margin-bottom:12px; flex-wrap:wrap; }
.detail-sym { font-size:1.15rem; font-weight:bold; color:#ddd; }
.badge {
    font-size:0.9rem; font-weight:bold; padding:4px 14px; border-radius:4px;
}
.badge.buy  { background:#1b5e20; color:#a5d6a7; }
.badge.sell { background:#b71c1c; color:#ef9a9a; }
.badge.hold { background:#333; color:#888; }
.conf-text { font-size:0.82rem; color:#aaa; }

.detail-body { display:flex; gap:14px; flex-wrap:wrap; }
.detail-left { flex:1; min-width:260px; }
.detail-right { flex:2; min-width:300px; }

/* TF vote table */
.tf-table { width:100%; border-collapse:collapse; font-size:0.82rem; margin-bottom:12px; }
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
    padding:10px 12px; font-size:0.8rem; color:#aaa; line-height:1.8;
    margin-bottom:12px;
}
.justif-box li { list-style:none; padding-left:12px; position:relative; }
.justif-box li::before { content:"•"; position:absolute; left:0; color:#9c27b0; }

/* SL/TP/Target row */
.price-row { display:flex; gap:14px; flex-wrap:wrap; margin-bottom:12px; }
.price-item { display:flex; flex-direction:column; gap:2px; }
.price-item .plabel { font-size:0.72rem; color:#888; }
.price-item .pvalue { font-size:0.9rem; font-weight:bold; color:#ddd; }
.price-item .pvalue.sl    { color:#ef5350; }
.price-item .pvalue.tp    { color:#26a69a; }
.price-item .pvalue.tgt   { color:#ffc107; }
.price-item .pvalue.rr    { color:#ab47bc; }
.price-item .pvalue.entry { color:#ccc; }

.execute-row { display:flex; gap:10px; align-items:center; flex-wrap:wrap; margin-bottom:8px; }
#execute-result { font-size:0.82rem; padding:6px 10px; border-radius:4px; display:none; margin-top:6px; }
.exec-ok  { background:#1b5e20; color:#a5d6a7; }
.exec-err { background:#b71c1c; color:#ef9a9a; }

/* Chart iframe */
.chart-frame-wrap { width:100%; min-height:440px; background:#111; border-radius:4px; overflow:hidden; border:1px solid #222; }
.chart-frame-wrap iframe { width:100%; height:460px; border:none; background:#111; }
.tf-tab { background:#2a2a2a; color:#aaa; border:1px solid #444; padding:3px 10px; border-radius:4px; cursor:pointer; font-size:0.78rem; }
.tf-tab:hover { background:#333; }
.tf-tab.active { background:#37474f; color:#fff; border-color:#607d8b; }

/* DECISIONS LOG */
.decisions-section { margin-top:4px; }
.decisions-section h3 { font-size:0.85rem; color:#888; font-weight:400; margin-bottom:8px; }
.dec-table { width:100%; border-collapse:collapse; font-size:0.8rem; }
.dec-table th { color:#777; font-weight:400; padding:4px 8px; border-bottom:1px solid #2a2a2a; text-align:left; }
.dec-table td { padding:5px 8px; border-bottom:1px solid #1e1e1e; }
.dec-table tr:hover td { background:#1c1c1c; }
.dec-yes { color:#26a69a; }
.dec-no  { color:#666; }

/* scrollbar */
::-webkit-scrollbar { width:6px; height:6px; }
::-webkit-scrollbar-track { background:#111; }
::-webkit-scrollbar-thumb { background:#333; border-radius:3px; }
</style>
</head>
<body>

<!-- TOP BAR -->
<div class="topbar">
    <span class="topbar-title">⚡ AutoTrader</span>

    <div class="topbar-section">
        <label>Simboluri</label>
        <div class="cb-group" id="sym-checks"></div>
    </div>

    <div class="topbar-section">
        <label>Timeframe-uri</label>
        <div class="cb-group" id="tf-checks"></div>
    </div>

    <div class="topbar-section">
        <label>Bare</label>
        <input type="number" id="bars-input" value="500" min="100" max="2000" step="100" style="width:80px">
    </div>

    <div class="topbar-section">
        <label>Scanare la fiecare: <span id="interval-val">60</span>s</label>
        <input type="range" id="interval-range" min="0" max="3" step="1" value="1">
    </div>

    <div class="topbar-section">
        <label>Auto Execute</label>
        <div class="toggle-wrap">
            <label class="toggle">
                <input type="checkbox" id="auto-exec-toggle" onchange="toggleAutoExec(this)">
                <span class="slider"></span>
            </label>
            <span class="auto-ex-warn" id="auto-ex-warn">⚠ ACTIV</span>
        </div>
    </div>

    <div class="topbar-section">
        <label style="font-size:0.72rem;color:#888">Filtre</label>
        <div style="display:flex;flex-direction:column;gap:4px">
            <label style="display:flex;align-items:center;gap:6px;cursor:pointer;font-size:0.78rem;color:#bbb">
                <label class="toggle" style="width:32px;height:16px">
                    <input type="checkbox" id="h4-filter-toggle" checked onchange="toggleFilter('use_h4_filter', this.checked)">
                    <span class="slider"></span>
                </label>
                H4 Direction
            </label>
            <label style="display:flex;align-items:center;gap:6px;cursor:pointer;font-size:0.78rem;color:#bbb">
                <label class="toggle" style="width:32px;height:16px">
                    <input type="checkbox" id="session-filter-toggle" checked onchange="toggleFilter('use_session_filter', this.checked)">
                    <span class="slider"></span>
                </label>
                Sesiuni (LN/NY)
            </label>
        </div>
    </div>

    <div class="topbar-row" style="align-self:flex-end; gap:8px;">
        <button class="btn btn-green" id="btn-start" onclick="startScanner()">▶ Start Scanner</button>
        <button class="btn btn-red"   id="btn-stop"  onclick="stopScanner()" disabled>■ Stop</button>
        <a href="/" class="btn btn-back">← ChartVisualizer</a>
    </div>
</div>

<!-- Banner scanner activ -->
<div class="scanner-banner" id="scanner-banner">
    <div class="banner-dot"></div>
    <div class="banner-text">
        <b>AutoTrader ACTIV</b> — scanez toate simbolurile ·
        urm. scan: <span id="banner-countdown">—</span> ·
        scanari: <span id="banner-scans">0</span>
    </div>
    <div id="ftmo-indicator" style="margin-left:auto;font-size:0.78rem;padding:3px 10px;border-radius:4px;background:#1b5e20;color:#a5d6a7">
        ✓ FTMO OK
    </div>
</div>

<!-- Toast -->
<div class="toast" id="toast"></div>

<!-- MAIN CONTENT -->
<div class="main-content">

    <div class="grid-header">
        <h2>
            <span class="status-dot" id="status-dot"></span>
            <span id="status-text">Scanner oprit</span>
            &nbsp;·&nbsp; Scan #<span id="scan-count">0</span>
            &nbsp;·&nbsp; Ultima scanare: <span id="last-scan">—</span>
            &nbsp;·&nbsp; <span class="next-scan-label" id="next-scan-label"></span>
        </h2>
        <div class="scan-progress-bar" id="scan-progress-bar" style="display:none">
            <div class="scan-progress-fill" id="scan-progress-fill" style="width:0%"></div>
        </div>
    </div>

    <!-- Symbol cards grid -->
    <div class="symbol-grid" id="symbol-grid">
        <!-- populated by JS -->
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
                <th>Timestamp</th><th>Simbol</th><th>Signal</th>
                <th>Incredere</th><th>Executat</th><th>Rezultat</th>
            </tr></thead>
            <tbody id="decisions-body"></tbody>
        </table>
    </div>

</div><!-- /main-content -->

<script>
const SYMBOLS_ALL = {{ symbols_json }};
const TFS_ALL     = ["M1","M5","M15","H1","H4"];
const INTERVALS   = [30, 60, 120, 300];

let selectedSymbols = new Set(SYMBOLS_ALL);
let selectedTFs     = new Set(["M1","M5","M15","H1"]);
let currentSymbol   = null;
let currentSignal   = null;
let pollTimer       = null;
let lastDecisionTs  = null;

// ── Build controls ────────────────────────────────────────────────────────
function buildControls() {
    const symWrap = document.getElementById("sym-checks");
    SYMBOLS_ALL.forEach(s => {
        const item = document.createElement("div");
        item.className = "cb-item checked";
        item.dataset.val = s;
        item.innerHTML = `<input type="checkbox" checked><span>${s}</span>`;
        item.onclick = () => {
            const checked = item.classList.toggle("checked");
            if (checked) {
                selectedSymbols.add(s);
                // re-adauga cardul daca lipseste
                const grid = document.getElementById("symbol-grid");
                if (!grid.querySelector(`[data-sym="${s}"]`)) {
                    const card = document.createElement("div");
                    card.className = "sym-card sig-hold scanning-card";
                    card.dataset.sym = s;
                    card.innerHTML = `<div class="card-name">${s}</div><div class="card-scanning"><span class="scan-spin">⟳</span> scanez...</div>`;
                    card.onclick = () => selectCard(s, null);
                    grid.appendChild(card);
                }
            } else {
                selectedSymbols.delete(s);
                // scoate cardul din grid
                const card = document.getElementById("symbol-grid").querySelector(`[data-sym="${s}"]`);
                if (card) card.remove();
                // daca era selectat, ascunde panoul
                if (currentSymbol === s) {
                    currentSymbol = null;
                    document.getElementById("detail-panel").style.display = "none";
                }
            }
        };
        symWrap.appendChild(item);
    });

    const tfWrap = document.getElementById("tf-checks");
    TFS_ALL.forEach(tf => {
        const item = document.createElement("div");
        item.className = "cb-item" + (selectedTFs.has(tf) ? " checked" : "");
        item.dataset.val = tf;
        item.innerHTML = `<input type="checkbox" ${selectedTFs.has(tf) ? "checked" : ""}><span>${tf}</span>`;
        item.onclick = () => {
            const checked = item.classList.toggle("checked");
            if (checked) selectedTFs.add(tf);
            else         selectedTFs.delete(tf);
            // daca scannerul ruleaza, trimite noile setari imediat
            if (document.body.classList.contains("scanning")) {
                fetch("/autotrader/start", {
                    method:"POST",
                    headers:{"Content-Type":"application/json"},
                    body: JSON.stringify({
                        tfs:     Array.from(selectedTFs),
                        symbols: Array.from(selectedSymbols),
                    })
                });
            }
        };
        tfWrap.appendChild(item);
    });
}

// ── Interval slider ───────────────────────────────────────────────────────
document.getElementById("interval-range").addEventListener("input", function() {
    const val = INTERVALS[parseInt(this.value)];
    document.getElementById("interval-val").textContent = val;
});

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
        tfs:          Array.from(selectedTFs),
        bars:         parseInt(document.getElementById("bars-input").value) || 500,
        symbols:      Array.from(selectedSymbols),
        auto_execute: document.getElementById("auto-exec-toggle").checked,
    };
    fetch("/autotrader/start", {
        method:"POST",
        headers:{"Content-Type":"application/json"},
        body: JSON.stringify(body)
    }).then(() => {
        document.getElementById("btn-start").disabled = true;
        document.getElementById("btn-stop").disabled  = false;
        showToast("▶ Scanner pornit — prima scanare in curs...", "#00695c");
        // arata imediat cardurile cu "se scanează"
        document.body.classList.add("scanning");
        document.getElementById("scanner-banner").classList.add("visible");
        updateGrid({});
    });
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

    // Auto-select first non-HOLD if a new decision appeared
    const decs = data.decisions || [];
    if (decs.length > 0) {
        const newest = decs[0].timestamp;
        if (newest !== lastDecisionTs) {
            lastDecisionTs = newest;
            const first = decs[0];
            if (first.signal !== "HOLD" && data.results[first.symbol]) {
                selectCard(first.symbol, data.results[first.symbol]);
            }
        }
    }
}

function updateGrid(results) {
    const grid = document.getElementById("symbol-grid");
    // Build map of existing cards
    const existing = {};
    grid.querySelectorAll(".sym-card").forEach(c => { existing[c.dataset.sym] = c; });

    // Arata doar simbolurile selectate in topbar
    const activeSyms = SYMBOLS_ALL.filter(s => selectedSymbols.has(s));
    const syms = activeSyms.length > 0 ? activeSyms : SYMBOLS_ALL;

    // Sterge carduri pentru simboluri debifate
    Object.keys(existing).forEach(sym => {
        if (!selectedSymbols.has(sym)) existing[sym].remove();
    });

    syms.forEach(sym => {
        const res = results[sym] || null;
        let card = existing[sym];
        if (!card) {
            card = document.createElement("div");
            card.className = "sym-card sig-hold";
            card.dataset.sym = sym;
            card.onclick = () => selectCard(sym, results[sym] || null);
            grid.appendChild(card);
        }

        const sig  = res ? res.signal : "HOLD";
        const conf = res ? res.confidence : 0;
        const trend = res && res.best_tf ? res.best_tf.trend : "—";
        const ts   = res ? res.timestamp.substring(11,19) : "";

        const isScanning = document.body.classList.contains("scanning");
        const noResult   = !res;
        card.className = `sym-card sig-${sig.toLowerCase()}` + (currentSymbol === sym ? " selected" : "") + (isScanning && noResult ? " scanning-card" : "");
        card.innerHTML = isScanning && noResult
            ? `<div class="card-name">${sym}</div>
               <div class="card-scanning"><span class="scan-spin">⟳</span> scanez...</div>`
            : `<div class="card-name">${sym}</div>
               <div class="card-signal ${sig.toLowerCase()}">${sig}</div>
               <div class="card-conf">${conf > 0 ? conf.toFixed(1)+"% incredere" : "—"}</div>
               <div class="card-trend">${trendRo(trend)}</div>
               <div class="card-time">${ts}</div>`;
        // Re-attach click (innerHTML clears it)
        card.onclick = () => selectCard(sym, results[sym] || null);
    });

    // If current symbol updated, refresh detail panel
    if (currentSymbol && results[currentSymbol]) {
        refreshDetailIfSelected(results[currentSymbol]);
    }
}

function trendRo(t) {
    return {ASCENDING:"▲ Ascendent", DESCENDING:"▼ Descendent", RANGING:"— Lateral"}[t] || t;
}

function selectCard(sym, res) {
    currentSymbol = sym;
    // Update card selection
    document.querySelectorAll(".sym-card").forEach(c => {
        c.classList.toggle("selected", c.dataset.sym === sym);
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
    document.getElementById("dp-chart-frame").src = `/autotrader/chart/${symbol}/${tf}`;
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
        tr.innerHTML = `
            <td>${d.timestamp.substring(0,19).replace("T"," ")}</td>
            <td>${d.symbol}</td>
            <td class="${cls}">${d.signal}</td>
            <td>${d.confidence ? d.confidence.toFixed(1)+"%" : "—"}</td>
            <td class="${d.executed ? "dec-yes" : "dec-no"}">${d.executed ? "DA" : "NU"}</td>
            <td style="color:#666;font-size:0.76rem">${(d.result||"").substring(0,80)}</td>
        `;
        tbody.appendChild(tr);
    });
}

// ── Init ──────────────────────────────────────────────────────────────────
buildControls();
// sincronizeaza UI cu starea reala de pe server
fetch("/autotrader/status").then(r => r.json()).then(data => {
    const sc = data.scanner;
    const toggle = document.getElementById("auto-exec-toggle");
    if (toggle) {
        toggle.checked = sc.auto_execute || false;
        document.getElementById("auto-ex-warn").style.display = sc.auto_execute ? "inline" : "none";
    }
    const h4tog = document.getElementById("h4-filter-toggle");
    if (h4tog) h4tog.checked = sc.use_h4_filter !== false;
    const sestog = document.getElementById("session-filter-toggle");
    if (sestog) sestog.checked = sc.use_session_filter !== false;
    // sincronizeaza selectedSymbols si selectedTFs cu serverul
    if (sc.symbols) {
        selectedSymbols = new Set(sc.symbols);
        document.querySelectorAll("#sym-checks .cb-item").forEach(el => {
            const v = el.dataset.val;
            el.classList.toggle("checked", selectedSymbols.has(v));
        });
    }
    if (sc.tfs) {
        selectedTFs = new Set(sc.tfs);
        document.querySelectorAll("#tf-checks .cb-item").forEach(el => {
            const v = el.dataset.val;
            el.classList.toggle("checked", selectedTFs.has(v));
        });
    }
});
pollStatus();
pollTimer = setInterval(pollStatus, 1000);

// ── FTMO status poll (la 10s) ─────────────────────────────────────────────
async function pollFtmo() {
    try {
        const r = await fetch("/ftmo_status");
        const d = await r.json();
        const el = document.getElementById("ftmo-indicator");
        if (!el) return;
        if (!d.ftmo_enabled) {
            el.style.display = "none";
            return;
        }
        el.style.display = "block";
        if (d.ok) {
            let txt = `✓ FTMO OK`;
            if (d.next_news) txt += ` · stiri in ${d.next_news.in_minutes}min (${d.next_news.time} UTC)`;
            if (d.daily_used_pct > 0) txt += ` · DD zilnic: ${d.daily_used_pct}%/5%`;
            el.textContent = txt;
            el.style.background = d.daily_used_pct > 3 ? "#b71c1c" : "#1b5e20";
            el.style.color = d.daily_used_pct > 3 ? "#ef9a9a" : "#a5d6a7";
        } else {
            el.textContent = `⛔ ${d.message}`;
            el.style.background = "#b71c1c";
            el.style.color = "#ef9a9a";
        }
    } catch(e) {}
}
pollFtmo();
setInterval(pollFtmo, 10000);

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
                    tfs: Array.from(selectedTFs),
                    bars: 500,
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
def autotrader_page():
    import json
    symbols_json = json.dumps(SYMBOLS)
    preselect    = request.args.get("symbol", "")
    decide_now   = request.args.get("decide", "0")
    html = AUTOTRADER_HTML.replace("{{ symbols_json }}", symbols_json) \
                          .replace("{{ preselect_symbol }}", preselect) \
                          .replace("{{ decide_now }}", decide_now)
    return Response(html, content_type="text/html; charset=utf-8")


@autotrader_bp.route("/autotrader/status")
def autotrader_status():
    with _scanner_lock:
        res_copy  = dict(results)
        dec_copy  = list(decisions)
        scan_copy = dict(scanner)
    payload = {
        "scanner":   scan_copy,
        "results":   res_copy,
        "decisions": dec_copy[:20],
    }
    return Response(json.dumps(payload, cls=NpEncoder), content_type="application/json")


@autotrader_bp.route("/autotrader/start", methods=["POST"])
def autotrader_start():
    body = request.get_json(silent=True) or {}
    if "interval" in body:
        scanner["interval"] = int(body["interval"])
    if "tfs" in body:
        scanner["tfs"] = [t for t in body["tfs"] if t in ALL_TFS]
    if "bars" in body:
        scanner["bars"] = int(body["bars"])
    if "symbols" in body:
        scanner["symbols"] = [s for s in body["symbols"] if s in SYMBOLS]
    if "auto_execute" in body:
        scanner["auto_execute"] = bool(body["auto_execute"])
    start_scanner()
    return Response(json.dumps({"ok": True}), content_type="application/json")


@autotrader_bp.route("/autotrader/stop", methods=["POST"])
def autotrader_stop():
    stop_scanner()
    return Response(json.dumps({"ok": True}), content_type="application/json")


@autotrader_bp.route("/autotrader/set", methods=["POST"])
def autotrader_set():
    """Modifica setari fara sa reporneasca scannerul."""
    body = request.get_json(silent=True) or {}
    if "auto_execute" in body:
        scanner["auto_execute"] = bool(body["auto_execute"])
    if "interval" in body:
        scanner["interval"] = int(body["interval"])
    if "tfs" in body:
        scanner["tfs"] = [t for t in body["tfs"] if t in ALL_TFS]
    if "bars" in body:
        scanner["bars"] = int(body["bars"])
    if "symbols" in body:
        scanner["symbols"] = [s for s in body["symbols"] if s in SYMBOLS]
    if "use_h4_filter" in body:
        scanner["use_h4_filter"] = bool(body["use_h4_filter"])
    if "use_session_filter" in body:
        scanner["use_session_filter"] = bool(body["use_session_filter"])
    return Response(json.dumps({"ok": True, "auto_execute": scanner["auto_execute"],
                                "use_h4_filter": scanner["use_h4_filter"],
                                "use_session_filter": scanner["use_session_filter"]}),
                    content_type="application/json")


@autotrader_bp.route("/autotrader/execute", methods=["POST"])
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


@autotrader_bp.route("/autotrader/chart/<symbol>/<tf>")
def autotrader_chart(symbol, tf):
    if symbol not in SYMBOLS or tf not in ALL_TFS:
        return Response("<div style='color:#ef5350;padding:12px'>Simbol sau TF invalid.</div>",
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
