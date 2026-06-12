"""
AutoTrader Blueprint — dashboard de auto-trading integrat cu ChartVisualizer
"""

import threading
import time
import json
import logging
from datetime import datetime

import notifier as _tg

import numpy as np
import pandas as pd
from flask import Blueprint, Response, request, jsonify

from app import (
    SYMBOLS, SYMBOLS_CRYPTO, ALL_TFS, RISK_DOLLARS, MIN_TF_VOTES, MIN_CONFIDENCE,
    fetch, find_pivots, detect_trend, calc_entry, calc_sl_tp, place_trade, place_pending_order,
    NpEncoder, MT5_AVAILABLE, mt5, build_chart,
    get_upcoming_red_news, close_all_positions_for_news, FTMO_NEWS_BLOCK_MIN,
    get_h4_direction, in_trading_session, calc_adx, ADX_MIN,
    calc_entry_smc, find_order_blocks, find_fvg, detect_bos,
    # P4 imports
    record_perf, get_perf_stats, check_auto_disable, is_auto_disabled,
    _check_daily_atr_overextension, resolve_strategy_conflict,
    record_daily_pnl, get_today_pnl, check_daily_drawdown_halt, is_daily_drawdown_halted,
)
import app as _app
from app import login_required

log = logging.getLogger(__name__)

autotrader_bp = Blueprint("autotrader", __name__)

# ── Scanner state ─────────────────────────────────────────────────────────────
scanner = {
    "running":        False,
    "interval":       60,
    "auto_execute":        False,
    "use_strategy_params": False,   # Aplica sl_atr_mult + risk_dollars per-strategie la executie
    "one_per_strategy":  False,
    "use_h4_filter":     False,
    "use_session_filter": False,
    "early_exit":        False,
    "time_based_exit":   True,    # inchide trade-uri vechi (>max_trade_hours)
    "breakeven_hours":   2.0,     # dupa N ore → muta SL la BE daca e in profit
    "max_trade_hours":   6.0,     # dupa N ore → inchide oricum (libereaza capital)
    "tp_ratio":          1.5,     # RR default (1.5 = TP mai apropiat → trade mai scurt)
    "combined_mode":     False,   # vot majoritar: >50% BUY/SELL → executa, strategiile individuale nu mai executa
    "pending_mode":      False,   # ordine pending la zona EOB in loc de market order
    # ── Scalp Boost Mode ──
    "scalp_boost":       False,   # combined rapid: lots×mult, TP 1:1, max 30min
    "scalp_lot_mult":    2.0,     # multiplicator loturi vs risc normal
    "scalp_max_min":     30,      # inchide oricum dupa N minute
    "scalp_tp_rr":       1.0,     # RR pentru TP (1:1)
    "scalp_min_agree":   3,       # minim N strategii trebuie sa fie de acord
    "symbols":        list(SYMBOLS),
    "last_scan":      None,
    "scan_count":     0,

    # ── Premium (recomandate) ──
    "smc": {
        "enabled":  True,
        "tfs":      ["H1", "H4"],
        "tf_bars":  {"M15":500,"H1":500,"H4":500,"D1":500},
        "elements": {"structure": True, "bos": True, "ob": True, "fvg": True,
                     "sweep": True, "ob_volume": True, "session": True, "atr_regime": True},
        "min_confidence": 85.0,
    },
    "eob": {
        "enabled":  True,
        "tfs":      ["M5", "M15", "H1", "H4"],
        "tf_bars":  {"M1":2000,"M5":2000,"M15":2000,"H1":2000,"H4":2000,"D1":2000},
        "elements": {
            "eob_htf":    True,
            "unicorn":    True,
            "eob_ltf":    True,
            "session":    True,   # gate London/NY
            "adx_filter": True,   # ADX > 18 pe HTF
        },
        "min_confidence": 85.0,
    },
    "trend_rider": {
        "enabled":  True,
        "tfs":      ["H1", "H4"],
        "tf_bars":  {"H1":500,"H4":500,"D1":500},
        "elements": {"ema": True, "macd": True, "supertrend": True,
                     "adx": True, "momentum": True, "volatility": True},
        "min_confidence": 85.0,
    },
    "candle_sniper": {
        "enabled":  True,
        "tfs":      ["M1"],          # forteaza M1 (strategia oricum decide)
        "tf_bars":  {"M1": 200, "M5": 200, "M15": 100},
        "elements": {"p1_pivot": True, "p2_inside": True, "p3_liquidity": True,
                     "p4_momentum": True, "p5_failed": True, "auto_tf": True,
                     "htf_bias": True},
        "min_confidence": 60.0,
    },
    "burst_scalper": {
        "enabled":  True,
        "tfs":      ["M1"],          # default M1; configurabil M2/M5
        "tf_bars":  {"M1": 200, "M2": 200, "M5": 200},
        "elements": {"ema_trend": True, "stoch_cross": True, "rsi_confirm": True,
                     "atr_filter": True, "candle_dir": True},
        "min_confidence": 60.0,
    },
    # ── Legacy (dezactivate default — TrendRider le inlocuieste) ──
    "classic": {
        "enabled":  False,
        "tfs":      ["H1", "H4", "D1"],
        "tf_bars":  {"M15":500,"H1":500,"H4":500,"D1":500},
        "elements": {"ema": True, "fib": True, "adx": True, "rsi": True},
        "min_confidence": 85.0,
    },
    "macd": {
        "enabled":  False,
        "tfs":      ["H1", "H4", "D1"],
        "tf_bars":  {"H1":500,"H4":500,"D1":500},
        "elements": {"macd_cross": True, "ema200": True, "histogram": True, "adx_filter": True, "cooldown": True},
        "min_confidence": 85.0,
    },
    "supertrend": {
        "enabled":  False,
        "tfs":      ["H1", "H4", "D1"],
        "tf_bars":  {"H1":300,"H4":300,"D1":300},
        "elements": {"supertrend": True, "ema50": True, "adx": True, "false_flip": True},
        "min_confidence": 85.0,
    },
    "ema_cross": {
        "enabled":  False,
        "tfs":      ["H1", "H4"],
        "tf_bars":  {"H1":200,"H4":200},
        "elements": {"ema_cross": True, "ema50": True, "momentum": True, "cooldown": True, "ema50_dist": True},
        "min_confidence": 85.0,
    },
    "bollinger": {
        "enabled":  False,
        "tfs":      ["H1", "H4"],
        "tf_bars":  {"M15":300,"H1":300,"H4":300},
        "elements": {"band_touch": True, "double_touch": True, "rsi_confirm": True, "squeeze": True, "adx_range": True, "bbw_filter": True},
        "min_confidence": 85.0,
    },
    # ── Session breakout (LTF) ──
    "london_breakout": {
        "enabled":  False,
        "tfs":      ["M15", "H1"],
        "tf_bars":  {"M5":200,"M15":200,"H1":200},
        "elements": {"session_gate": True, "asian_range": True, "breakout": True, "retest": True, "h4_trend": True, "time_window": True, "day_of_week": True},
        "min_confidence": 85.0,
    },
    "ny_breakout": {
        "enabled":  False,
        "tfs":      ["M15", "H1"],
        "tf_bars":  {"M5":200,"M15":200,"H1":200},
        "elements": {"session_gate": True, "pre_ny_range": True, "breakout": True, "retest": True, "h4_trend": True, "time_window": True, "ny_pairs": True, "london_moved_check": True, "us_open_pause": True},
        "min_confidence": 85.0,
    },
    "china_session": {
        "enabled":  False,
        "tfs":      ["M15", "H1"],
        "tf_bars":  {"M5":150,"M15":150,"H1":150},
        "elements": {"session_gate": True, "session_range": True, "rsi_extreme": True, "reversal": True, "china_pairs": True, "bb_width": True, "escape_exit": True},
        "min_confidence": 80.0,
    },
    # ── Reversal / Price action ──
    "rsi_divergence": {
        "enabled":  False,
        "tfs":      ["H4", "D1"],
        "tf_bars":  {"H1":300,"H4":300,"D1":300},
        "elements": {"bullish_div": True, "bearish_div": True, "rsi_zone": True, "bar_quality": True, "multi_div": True},
        "min_confidence": 85.0,
    },
    "engulfing": {
        "enabled":  False,
        "tfs":      ["H4", "D1"],
        "tf_bars":  {"H1":200,"H4":200,"D1":200},
        "elements": {"engulfing": True, "pin_bar": True, "key_level": True, "level_quality": True, "engulf_size": True},
        "min_confidence": 85.0,
    },
    "ichimoku": {
        "enabled":  False,
        "tfs":      ["H4", "D1"],
        "tf_bars":  {"H1":300,"H4":300,"D1":300},
        "elements": {"tk_cross": True, "kumo": True, "chikou": True, "tk_outside": True, "kumo_twist": True},
        "min_confidence": 85.0,
    },
}

# ── Snapshot defaults pentru butonul "Restore All to Defaults" ───────────────
import copy as _copy
SCANNER_DEFAULTS_SNAPSHOT = _copy.deepcopy(scanner)
results   = {}   # {symbol: {"classic": ..., "smc": ...}}
decisions = []
_scanner_thread = None
_scanner_lock = threading.Lock()


# ── Log persistent pe disc ────────────────────────────────────────────────────
import os as _os
_LOG_FILE = _os.path.join(_os.path.dirname(__file__), "review_log.json")

def _is_duplicate_decision(entry: dict, window: int = 5) -> bool:
    """
    Returneaza True daca aceeasi combinatie symbol+strategy+signal apare
    deja in ultimele `window` intrari din decisions (evita spam la fiecare ciclu).
    Exceptie: executiile reale (executed=True) se logheaza intotdeauna.
    """
    if entry.get("executed"):
        return False
    sig     = entry.get("signal", "")
    sym     = entry.get("symbol", "")
    strat   = entry.get("strategy", "")
    for prev in decisions[:window]:
        if (prev.get("symbol") == sym
                and prev.get("strategy") == strat
                and prev.get("signal") == sig):
            return True
    return False


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


# ── Statistics ────────────────────────────────────────────────────────────────
def compute_stats(window_trades: int = 100) -> dict:
    """
    Calculeaza statistici din review_log.json + perf_stats + MT5.
    Returneaza dict complet cu sectiunile: today, last_N, by_strategy,
    by_symbol, open_risk, session_quality.
    """
    try:
        log_data = []
        if _os.path.exists(_LOG_FILE):
            with open(_LOG_FILE, "r", encoding="utf-8") as f:
                log_data = json.load(f)
    except Exception:
        log_data = []

    now = datetime.now()
    today_str = now.strftime("%Y-%m-%d")

    # Separa executiile reale (executed=True) si semnalele detectate
    executed    = [e for e in log_data if e.get("executed") and
                   e.get("signal") in ("BUY", "SELL")]
    today_exec  = [e for e in executed if e.get("timestamp", "").startswith(today_str)]
    last_n_exec = executed[:window_trades]

    def _win_rate(entries):
        wins = sum(1 for e in entries if "succes" in e.get("result", "").lower()
                   or "plasat" in e.get("result", "").lower())
        total = len(entries)
        return round(wins / total * 100, 1) if total else 0.0, wins, total

    def _profit_factor(entries):
        """Estimeaza PF din confidence ca proxy pentru wins/losses."""
        wins   = [e for e in entries if "succes" in e.get("result","").lower()
                  or "plasat" in e.get("result","").lower()]
        losses = [e for e in entries if "esuat" in e.get("result","").lower()
                  or "eroare" in e.get("result","").lower()
                  or "invalid" in e.get("result","").lower()]
        if not losses:
            return 99.0 if wins else 0.0
        return round(len(wins) / len(losses), 2)

    wr_today, wins_today, total_today = _win_rate(today_exec)
    wr_last,  wins_last,  total_last  = _win_rate(last_n_exec)
    pf_last = _profit_factor(last_n_exec)

    # Streak curent
    streak = 0; streak_type = ""
    for e in executed[:20]:
        is_win = ("succes" in e.get("result","").lower() or "plasat" in e.get("result","").lower())
        if streak == 0:
            streak = 1; streak_type = "W" if is_win else "L"
        elif (is_win and streak_type == "W") or (not is_win and streak_type == "L"):
            streak += 1
        else:
            break

    # Per strategie (din perf_stats)
    by_strat = {}
    try:
        ps = get_perf_stats()
        for sk, st in ps.items():
            by_strat[sk] = {
                "trades":      st.get("trades", 0),
                "wins":        st.get("wins", 0),
                "win_rate":    round(st.get("wins", 0) / max(st.get("trades", 1), 1) * 100, 1),
                "profit_factor": st.get("profit_factor", 0.0),
            }
    except Exception:
        pass

    # Per simbol (din log recent)
    sym_count = {}
    for e in last_n_exec:
        s = e.get("symbol", "?")
        if s not in sym_count:
            sym_count[s] = {"trades": 0, "wins": 0}
        sym_count[s]["trades"] += 1
        if "succes" in e.get("result","").lower() or "plasat" in e.get("result","").lower():
            sym_count[s]["wins"] += 1
    top_symbols = sorted(
        [{"symbol": s, **v, "win_rate": round(v["wins"]/max(v["trades"],1)*100,1)}
         for s, v in sym_count.items()],
        key=lambda x: x["trades"], reverse=True
    )[:8]

    # Risc deschis curent (MT5)
    open_risk = 0.0
    open_trades_count = 0
    try:
        if MT5_AVAILABLE and mt5:
            positions = mt5.positions_get() or []
            open_trades_count = len(positions)
            for pos in positions:
                open_risk += float(getattr(pos, "volume", 0)) * _app.RISK_DOLLARS
    except Exception:
        pass

    # P&L azi (din get_today_pnl)
    today_pnl = 0.0
    try:
        today_pnl = get_today_pnl()
    except Exception:
        pass

    # Distributie semnale pe ore (azi)
    hour_dist = {}
    for e in today_exec:
        try:
            h = int(e.get("timestamp","T00")[:13].split("T")[1].split(":")[0])
            hour_dist[h] = hour_dist.get(h, 0) + 1
        except Exception:
            pass
    best_hour = max(hour_dist, key=hour_dist.get) if hour_dist else None

    # Strategii active in scannerul curent
    active_strats = [k for k, v in scanner.items() if isinstance(v, dict) and v.get("enabled")]

    return {
        "today": {
            "trades":   total_today,
            "wins":     wins_today,
            "losses":   total_today - wins_today,
            "win_rate": wr_today,
            "pnl_usd":  round(today_pnl, 2),
        },
        "last_n": {
            "n":          window_trades,
            "trades":     total_last,
            "wins":       wins_last,
            "losses":     total_last - wins_last,
            "win_rate":   wr_last,
            "profit_factor": pf_last,
        },
        "streak":        {"count": streak, "type": streak_type},
        "by_strategy":   by_strat,
        "top_symbols":   top_symbols,
        "open_risk":     round(open_risk, 2),
        "open_trades":   open_trades_count,
        "best_hour":     best_hour,
        "active_strats": active_strats,
        "scalp_boost":   scanner.get("scalp_boost", False),
        "ts":            now.isoformat(),
    }


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


# ── Time-based exit — inchide trade-uri vechi care nu se misca ───────────────
def time_based_exit(auto_ex: bool):
    """
    Inchide trade-urile prea vechi:
      - Dupa `breakeven_hours`: muta SL la breakeven (nu mai pierzi)
      - Dupa `max_trade_hours`: inchide complet, indiferent de P/L
        (libereaza capital, evita drifting fara directie)
    Config:
      scanner["breakeven_hours"]  (default 2)
      scanner["max_trade_hours"]  (default 6)
      scanner["time_based_exit"]  (default True)
    """
    if not MT5_AVAILABLE or mt5 is None or not auto_ex:
        return []
    if not scanner.get("time_based_exit", True):
        return []

    from datetime import datetime, timezone
    positions = mt5.positions_get() or []
    if not positions:
        return []

    be_hours  = float(scanner.get("breakeven_hours", 2.0))
    max_hours = float(scanner.get("max_trade_hours", 6.0))
    now_ts    = int(datetime.now(timezone.utc).timestamp())
    actions   = []

    for pos in positions:
        try:
            age_s = now_ts - int(pos.time)
            age_h = age_s / 3600.0
            if age_h < be_hours:
                continue

            pos_type   = "BUY" if pos.type == 0 else "SELL"
            price_open = pos.price_open
            price_now  = pos.price_current

            # ── Faza 1: breakeven SL (intre be_hours si max_hours) ──
            if age_h < max_hours:
                # Muta SL la pretul de intrare daca e in profit si SL nu e deja la BE
                sl_at_be = abs(pos.sl - price_open) < 1e-6 if pos.sl else False
                in_profit = (pos_type == "BUY" and price_now > price_open) or \
                            (pos_type == "SELL" and price_now < price_open)
                if in_profit and not sl_at_be:
                    req = {
                        "action":   mt5.TRADE_ACTION_SLTP,
                        "position": pos.ticket,
                        "sl":       price_open,
                        "tp":       pos.tp,
                    }
                    res = mt5.order_send(req)
                    if res and res.retcode == mt5.TRADE_RETCODE_DONE:
                        log.info(f"[TimeExit] #{pos.ticket} {pos.symbol} SL → breakeven ({age_h:.1f}h)")
                        actions.append({"ticket": pos.ticket, "symbol": pos.symbol,
                                        "action": "breakeven", "age_h": round(age_h, 1)})
                continue

            # ── Faza 2: hard close (age > max_hours) ──
            tick = mt5.symbol_info_tick(pos.symbol)
            if tick is None:
                continue
            info = mt5.symbol_info(pos.symbol)
            close_price = tick.bid if pos_type == "BUY" else tick.ask
            order_type  = mt5.ORDER_TYPE_SELL if pos_type == "BUY" else mt5.ORDER_TYPE_BUY
            fm = info.filling_mode if info else 0
            if fm & 2:    filling = mt5.ORDER_FILLING_IOC
            elif fm & 1:  filling = mt5.ORDER_FILLING_FOK
            else:         filling = mt5.ORDER_FILLING_RETURN

            req = {
                "action":       mt5.TRADE_ACTION_DEAL,
                "symbol":       pos.symbol,
                "volume":       pos.volume,
                "type":         order_type,
                "price":        close_price,
                "position":     pos.ticket,
                "deviation":    30,
                "magic":        pos.magic,
                "comment":      "time_exit",
                "type_time":    mt5.ORDER_TIME_GTC,
                "type_filling": filling,
            }
            res = mt5.order_send(req)
            if res and res.retcode == mt5.TRADE_RETCODE_DONE:
                log.info(f"[TimeExit] #{pos.ticket} {pos.symbol} close timeout "
                         f"({age_h:.1f}h > {max_hours}h) profit={pos.profit:.2f}$")
                actions.append({"ticket": pos.ticket, "symbol": pos.symbol,
                                "action": "closed_timeout", "age_h": round(age_h, 1),
                                "profit": round(pos.profit, 2)})
        except Exception as exc:
            log.warning(f"[TimeExit] {pos.symbol}: {exc}")

    return actions


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


def _cancel_stale_pending_orders():
    """
    Cureata ordinele pending plasate de scanner care nu mai sunt valide.
    Un pending e "stale" daca:
      - pretul curent a trecut DEJA de SL (zona distrusa inainte sa se declanseze)
      - pending e mai vechi decat expiration si brokerul nu l-a sters
    Returneaza lista de {symbol, ticket, reason} pentru log.
    """
    if not MT5_AVAILABLE or mt5 is None:
        return []
    try:
        if not mt5.initialize():
            return []
    except Exception:
        return []

    cancelled = []
    all_pending = mt5.orders_get() or []
    now_ts = int(datetime.now().timestamp())

    # Tipuri pending MT5: 2=BUY_LIMIT, 3=SELL_LIMIT, 4=BUY_STOP, 5=SELL_STOP, 6=BUY_STOP_LIMIT, 7=SELL_STOP_LIMIT
    BUY_TYPES  = {2, 4, 6}
    SELL_TYPES = {3, 5, 7}

    for op in all_pending:
        cmt = (op.comment or "")
        # Numai pending plasate de scanner (comment se termina cu "_P")
        if not cmt.endswith("_P"):
            continue
        symbol = op.symbol
        tick = mt5.symbol_info_tick(symbol)
        if tick is None:
            continue

        reason = None

        # Verifica expirarea manuala (brokerul ar trebui sa stearga singur, dar fallback)
        exp = getattr(op, "time_expiration", 0)
        if exp and exp > 0 and now_ts > exp + 60:
            reason = f"expirat (expiration={exp} < now={now_ts})"

        # Zona distrusa: pretul curent e deja dincolo de SL
        if reason is None and op.sl and op.sl > 0:
            if op.type in BUY_TYPES:
                # Pentru BUY, SL e sub entry → daca pretul curent e sub SL, zona e distrusa
                if tick.bid < op.sl:
                    reason = f"zona distrusa: bid={tick.bid} < SL={op.sl}"
            elif op.type in SELL_TYPES:
                # Pentru SELL, SL e peste entry → daca pretul curent e peste SL, zona e distrusa
                if tick.ask > op.sl:
                    reason = f"zona distrusa: ask={tick.ask} > SL={op.sl}"

        if reason is None:
            continue

        req = {
            "action": mt5.TRADE_ACTION_REMOVE,
            "order":  op.ticket,
        }
        result = mt5.order_send(req)
        if result and result.retcode == mt5.TRADE_RETCODE_DONE:
            cancelled.append({"symbol": symbol, "ticket": op.ticket, "reason": reason})
            log.info(f"Pending anulat {symbol} #{op.ticket}: {reason}")
        else:
            rc = result.retcode if result else -1
            log.warning(f"Nu am putut anula pending {symbol} #{op.ticket}: retcode={rc}")

    return cancelled


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
            if auto_ex and scanner.get("early_exit", False):
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

            # ── Time-based exit (trade-uri vechi → breakeven/close) ───────
            if auto_ex and scanner.get("time_based_exit", True):
                try:
                    time_actions = time_based_exit(auto_ex)
                    for ta in time_actions:
                        entry = {
                            "timestamp":  datetime.now().isoformat(),
                            "symbol":     ta["symbol"],
                            "signal":     f"TIME {ta['action'].upper()}",
                            "confidence": 0,
                            "executed":   True,
                            "result":     f"age={ta['age_h']}h"
                                          + (f" profit={ta.get('profit',0)}$" if 'profit' in ta else ""),
                            "strategy":   "—",
                        }
                        _log_action(entry)
                        with _scanner_lock:
                            decisions.insert(0, entry)
                            while len(decisions) > 50:
                                decisions.pop()
                except Exception as exc:
                    log.warning(f"time_based_exit call: {exc}")

            # ── Cleanup pending orders invalide (zona distrusa / expirate) ─
            if scanner.get("pending_mode", False) and auto_ex:
                try:
                    stale = _cancel_stale_pending_orders()
                    for st in stale:
                        entry = {
                            "timestamp": datetime.now().isoformat(),
                            "symbol":    st["symbol"],
                            "signal":    f"PENDING CANCEL #{st['ticket']}",
                            "confidence": 0,
                            "executed":  True,
                            "result":    st["reason"],
                            "strategy":  "eob",
                            "pending":   True,
                        }
                        _log_action(entry)
                        with _scanner_lock:
                            decisions.insert(0, entry)
                            while len(decisions) > 50: decisions.pop()
                except Exception as exc:
                    log.warning(f"Pending cleanup err: {exc}")

            # ── Incarcare strategii active din registry ────────────────────
            import strategies as _strat_pkg
            enabled_strategies = _strat_pkg.get_enabled(scanner)

            for sym in symbols:
                if not scanner["running"]:
                    break
                try:
                    sym_results = results.get(sym, {})

                    # ── P4: Daily ATR Overextension Filter ────────────────
                    atr_overext, atr_msg = _check_daily_atr_overextension(sym)
                    if atr_overext:
                        log.info(f"P4 ATR overext skip {sym}: {atr_msg}")
                        scanner.setdefault("atr_skip", {})[sym] = atr_msg
                        with _scanner_lock:
                            results[sym] = sym_results
                        continue
                    else:
                        scanner.get("atr_skip", {}).pop(sym, None)

                    # Colecteaza rezultatele din scan-ul CURENT (nu din sym_results vechi)
                    current_scan_results = {}

                    # ── Symbol exclusivity: daca un strategy "exclusive_symbols"
                    # match pe simbol, doar el ruleaza. Restul filtrate pe whitelist.
                    sym_strats = []
                    exclusive_pick = None
                    for sk, so in enabled_strategies:
                        if not so.matches_symbol(sym):
                            continue
                        if getattr(so, "exclusive_symbols", False):
                            exclusive_pick = (sk, so)
                            break
                        sym_strats.append((sk, so))
                    if exclusive_pick is not None:
                        sym_strats = [exclusive_pick]
                        log.debug(f"Exclusive: {sym} rezervat pentru {exclusive_pick[0]}")

                    for strat_key, strat_obj in sym_strats:
                        # ── P4: Auto-disable check ────────────────────────
                        if is_auto_disabled(strat_key):
                            log.debug(f"P4: {strat_key} auto-disabled — skip")
                            continue

                        cfg = scanner.get(strat_key, {})
                        tfs_s    = cfg.get("tfs", strat_obj.default_tfs)
                        tf_bars  = cfg.get("tf_bars", {})
                        elems    = cfg.get("elements", {k: True for k in strat_obj.elements})
                        min_conf = cfg.get("min_confidence", 66.0)

                        # Aplica SL multiplier per-strategie (override temporar global)
                        import app as _app_mod
                        _use_sp = scanner.get("use_strategy_params", False)
                        _strat_sl_mult = cfg.get("sl_atr_mult") if _use_sp else None
                        _global_sl_bak = _app_mod.SL_ATR_MULT
                        if _strat_sl_mult is not None:
                            _app_mod.SL_ATR_MULT = float(_strat_sl_mult)
                        try:
                            res = strat_obj.analyze(
                                sym, tfs_s, strat_obj.default_bars, tf_bars, elems, min_conf,
                                use_h4_filter=scanner.get("use_h4_filter", False),
                                use_session_filter=scanner.get("use_session_filter", False),
                            )
                        finally:
                            _app_mod.SL_ATR_MULT = _global_sl_bak

                        # ── Optional common filters (opt_*) ─────────────────
                        # Daca strategia returneaza BUY/SELL si user a bifat
                        # filtre opt_* in elements, le aplica acum.
                        if res.get("signal") in ("BUY", "SELL"):
                            try:
                                opt_keys = {k[4:] for k, v in (elems or {}).items()
                                            if v and k.startswith("opt_")}
                                if opt_keys:
                                    from strategies.common_filters import apply_all as _opt_apply
                                    # Foloseste best_tf-ul pentru date
                                    best_tf_info = res.get("best_tf") or {}
                                    use_tf = best_tf_info.get("tf") if isinstance(best_tf_info, dict) else tfs_s[0]
                                    df_opt, _ = fetch(sym, use_tf, 200)
                                    if df_opt is not None and len(df_opt) >= 50:
                                        df_opt.columns = [c.lower() for c in df_opt.columns]
                                        # H4 direction pentru htf_alignment
                                        h4_dir = None
                                        try:
                                            from app import get_h4_direction as _gh4
                                            h4_dir = _gh4(sym)
                                        except Exception:
                                            pass
                                        ok, reasons = _opt_apply(
                                            df_opt, res["signal"], opt_keys, h4_dir=h4_dir,
                                        )
                                        if not ok:
                                            # Demote la HOLD
                                            res["signal"] = "HOLD"
                                            res.setdefault("justification", []).extend(reasons)
                                            res.setdefault("justification", []).append(
                                                f"[opt filter] {strat_key} demoted la HOLD"
                                            )
                                        else:
                                            res.setdefault("justification", []).extend(reasons)
                            except Exception as _ef:
                                log.debug(f"opt filters {strat_key}: {_ef}")

                        res["strategy"] = strat_key
                        res["auto_executed"] = False
                        current_scan_results[strat_key] = res
                        sym_results[strat_key] = res

                        # ── P4: Auto-disable evaluation (non-blocking) ────
                        try:
                            if check_auto_disable(strat_key):
                                log.warning(
                                    f"P4 AutoDisable: {strat_key} dezactivat automat"
                                    " (PF < 1.0 pe ultimele 50 trade-uri)"
                                )
                        except Exception:
                            pass

                    # ── P4: Conflict Resolution pe rezultatele DIN SCAN-UL CURENT ──
                    active_sigs = {k: v for k, v in current_scan_results.items()
                                   if v.get("signal") in ("BUY", "SELL")}
                    if len(active_sigs) >= 2:
                        resolved = resolve_strategy_conflict(current_scan_results)
                        current_scan_results.update(resolved)
                        sym_results.update(resolved)

                    # ── Combined Mode: vot per STRATEGIE (nu per TF) ──
                    # Fiecare strategie = 1 vot, ponderat cu conviction-ul sau best_tf.
                    # TF-urile multiple ale aceleiasi strategii se agreg intern:
                    #   - directia dominanta a strategiei (majority vote intern)
                    #   - conviction = best conviction din TF-urile in acord
                    #   - coherence = TF-uri in acord / total TF-uri active
                    combined_mode = scanner.get("combined_mode", False)
                    if combined_mode:
                        SESSION_STRATS = {"london_breakout", "ny_breakout", "china_session", "vwap_bounce"}

                        # ── Nivel 1: agreg per strategie → 1 vot per strategie ──
                        strat_votes   = []   # {strat, signal, conviction, coherence, sl, tp, tf}
                        all_tf_results = []  # pastrat pentru executie (gasirea best_res)

                        for strat_key, res in current_scan_results.items():
                            if strat_key in SESSION_STRATS:
                                continue
                            tfs = res.get("tfs", [])
                            if not tfs:
                                continue

                            # Colecteaza toate TF-urile pentru executie
                            for tf_r in tfs:
                                all_tf_results.append({
                                    "strat":      strat_key,
                                    "tf":         tf_r.get("tf", "?"),
                                    "signal":     tf_r.get("signal", "HOLD"),
                                    "conviction": tf_r.get("conviction", 0),
                                    "sl":         tf_r.get("sl"),
                                    "tp":         tf_r.get("tp"),
                                    "price":      tf_r.get("price"),
                                })

                            # Vot intern per strategie
                            MIN_CONV_TO_VOTE = 1
                            buy_tfs  = [t for t in tfs if t.get("signal") == "BUY"  and t.get("conviction", 0) >= MIN_CONV_TO_VOTE]
                            sell_tfs = [t for t in tfs if t.get("signal") == "SELL" and t.get("conviction", 0) >= MIN_CONV_TO_VOTE]
                            active_tfs = len(buy_tfs) + len(sell_tfs)

                            if not buy_tfs and not sell_tfs:
                                strat_votes.append({"strat": strat_key, "signal": "HOLD",
                                                    "conviction": 0, "coherence": 0,
                                                    "sl": None, "tp": None, "tf": "?",
                                                    "risk_score": 0})
                                continue

                            if len(buy_tfs) >= len(sell_tfs):
                                strat_sig  = "BUY"
                                best_strat_tf = max(buy_tfs, key=lambda x: x.get("conviction", 0))
                                strat_conv = best_strat_tf.get("conviction", 0)
                                coherence  = len(buy_tfs) / active_tfs if active_tfs else 0
                            else:
                                strat_sig  = "SELL"
                                best_strat_tf = max(sell_tfs, key=lambda x: x.get("conviction", 0))
                                strat_conv = best_strat_tf.get("conviction", 0)
                                coherence  = len(sell_tfs) / active_tfs if active_tfs else 0

                            strat_votes.append({
                                "strat":     strat_key,
                                "signal":    strat_sig,
                                "conviction": strat_conv,
                                "coherence": coherence,
                                "sl":        best_strat_tf.get("sl"),
                                "tp":        best_strat_tf.get("tp"),
                                "tf":        best_strat_tf.get("tf", "?"),
                                "price":     best_strat_tf.get("price"),
                                "risk_score": res.get("risk_score", 0),
                            })

                        # ── Nivel 2: vot intre strategii ──
                        n_buy   = sum(1 for v in strat_votes if v["signal"] == "BUY")
                        n_sell  = sum(1 for v in strat_votes if v["signal"] == "SELL")
                        n_hold  = sum(1 for v in strat_votes if v["signal"] == "HOLD")
                        total   = len(strat_votes)

                        # Scor ponderat: conviction × coherence (doar pentru afisaj, nu pentru decizie)
                        buy_score  = sum(v["conviction"] * v["coherence"] for v in strat_votes if v["signal"] == "BUY")
                        sell_score = sum(v["conviction"] * v["coherence"] for v in strat_votes if v["signal"] == "SELL")
                        active_score = buy_score + sell_score

                        n_buy_qual  = n_buy
                        n_sell_qual = n_sell

                        combined_signal = None
                        combined_voters = []

                        # ── Regula de decizie (flat, bazata pe total strategii) ──
                        # BUY:  ≥60% din total strategii spun BUY
                        #       + spread(BUY%-SELL%) ≥ 21% (e.g. 60%BUY+40%SELL → spread 20% → NU)
                        #       → 60%BUY + 40%HOLD + 0%SELL = OK, dar 60%BUY + 40%SELL = NU
                        # SELL: simetric
                        buy_pct_flat  = n_buy  / total if total else 0
                        sell_pct_flat = n_sell / total if total else 0
                        spread_pct    = buy_pct_flat - sell_pct_flat  # pozitiv=BUY dominant

                        # Combined Risk Score — medie ponderata de conviction a risk_score-urilor
                        # strategiilor active; penalizat daca spread-ul e sub 40% ideal
                        _cv_active = [v for v in strat_votes if v["signal"] != "HOLD"]
                        if _cv_active:
                            _cv_total_conv = sum(v["conviction"] for v in _cv_active)
                            if _cv_total_conv > 0:
                                _cv_weighted = sum(v.get("risk_score", 0) * v["conviction"] for v in _cv_active) / _cv_total_conv
                            else:
                                _cv_weighted = sum(v.get("risk_score", 0) for v in _cv_active) / len(_cv_active)
                            _cv_spread_penalty = max(0.0, (0.40 - abs(spread_pct)) / 0.40 * 15.0)
                            combined_risk_score = round(max(0.0, _cv_weighted - _cv_spread_penalty), 1)
                        else:
                            combined_risk_score = 0.0

                        _combined_skip_reason = None
                        if total >= 2:
                            if buy_pct_flat >= 0.60 and spread_pct >= 0.21:
                                combined_signal = "BUY"
                                combined_voters = [v["strat"] for v in strat_votes if v["signal"] == "BUY"]
                            elif sell_pct_flat >= 0.60 and spread_pct <= -0.21:
                                combined_signal = "SELL"
                                combined_voters = [v["strat"] for v in strat_votes if v["signal"] == "SELL"]
                            else:
                                bp = round(buy_pct_flat * 100, 1)
                                sp = round(sell_pct_flat * 100, 1)
                                spr = round(abs(spread_pct) * 100, 1)
                                if buy_pct_flat > sell_pct_flat:
                                    if buy_pct_flat < 0.60:
                                        _combined_skip_reason = f"BUY {bp}% < 60% minim"
                                    else:
                                        _combined_skip_reason = f"BUY {bp}% OK dar spread {spr}% < 21%"
                                elif sell_pct_flat > buy_pct_flat:
                                    if sell_pct_flat < 0.60:
                                        _combined_skip_reason = f"SELL {sp}% < 60% minim"
                                    else:
                                        _combined_skip_reason = f"SELL {sp}% OK dar spread {spr}% < 21%"
                                else:
                                    _combined_skip_reason = f"Egalitate BUY/SELL ({bp}%/{sp}%)"
                                _log_action({
                                    "timestamp": datetime.now().isoformat(),
                                    "symbol":    sym,
                                    "signal":    "HOLD",
                                    "confidence": max(bp, sp),
                                    "executed":  False,
                                    "result":    f"Prag neindeplinit — {_combined_skip_reason}",
                                    "strategy":  "combined",
                                    "vote_map":  {"buy": n_buy, "sell": n_sell, "hold": n_hold, "total": total},
                                })

                        # Detalii per strategie pentru UI
                        strat_details = {}
                        for sk, sr in current_scan_results.items():
                            bf = sr.get("best_tf") or {}
                            strat_details[sk] = {
                                "signal":     sr.get("signal", "HOLD"),
                                "confidence": sr.get("confidence", 0),
                                "best_tf":    bf.get("tf", "—"),
                                "conviction": bf.get("conviction", 0),
                                "reasons":    (bf.get("reasons") or sr.get("justification") or [])[:3],
                                "sl":         bf.get("sl"),
                                "tp":         bf.get("tp"),
                                "price":      bf.get("price"),
                            }

                        sym_results["_combined_votes"] = {
                            "buy":        [v["strat"] for v in strat_votes if v["signal"] == "BUY"],
                            "sell":       [v["strat"] for v in strat_votes if v["signal"] == "SELL"],
                            "hold":       [v["strat"] for v in strat_votes if v["signal"] == "HOLD"],
                            "signal":     combined_signal or "HOLD",
                            "total":      total,          # nr strategii (nu TF-uri)
                            "n_buy":      n_buy,          # strategii BUY
                            "n_sell":     n_sell,         # strategii SELL
                            "n_hold":     n_hold,         # strategii HOLD
                            "n_buy_qual": n_buy_qual,
                            "n_sell_qual": n_sell_qual,
                            "buy_score":  round(buy_score, 2),
                            "sell_score": round(sell_score, 2),
                            "pct_buy":    round(n_buy / total * 100, 1) if total else 0,
                            "pct_sell":   round(n_sell / total * 100, 1) if total else 0,
                            "pct_buy_w":  round(buy_score  / active_score * 100, 1) if active_score else 0,
                            "pct_sell_w": round(sell_score / active_score * 100, 1) if active_score else 0,
                            "spread_pct": round(abs(spread_pct) * 100, 1),
                            "buy_pct_flat":  round(buy_pct_flat  * 100, 1),
                            "sell_pct_flat": round(sell_pct_flat * 100, 1),
                            "strategies": strat_details,
                            "strat_votes": strat_votes,
                            "tf_rows":    all_tf_results,
                            "skip_reason": _combined_skip_reason,
                            "combined_risk_score": combined_risk_score,
                        }

                        # ── Gaseste best_res: prefer strategia cu conv*coherence max si SL+TP valid ──
                        _best_res_combined  = None
                        _best_conv_combined = -1
                        for v in strat_votes:
                            if combined_signal and v["signal"] == combined_signal and v.get("sl") and v.get("tp"):
                                score = v["conviction"] * v.get("coherence", 1)
                                if score > _best_conv_combined:
                                    _best_conv_combined = score
                                    _best_res_combined  = v

                        # Logheaza semnalul chiar daca auto_execute e OFF
                        if combined_signal and not auto_ex:
                            _comb_voters_off = list({r["strat"] for r in all_tf_results if r["signal"] == combined_signal})
                            _comb_conf_off   = round(n_buy / total * 100, 1) if combined_signal == "BUY" else round(n_sell / total * 100, 1)
                            _comb_cid_off    = _store_analysis(
                                sym, "combined", combined_signal,
                                _best_res_combined.get("tf") if _best_res_combined else None,
                                _best_res_combined.get("price") if _best_res_combined else None,
                                _best_res_combined.get("sl")    if _best_res_combined else None,
                                _best_res_combined.get("tp")    if _best_res_combined else None,
                                _comb_conf_off,
                                [f"Combined {combined_signal}: {len(_comb_voters_off)}/{total} strategii",
                                 f"Votanti: {', '.join(_comb_voters_off)}"],
                                voters=_comb_voters_off,
                                vote_map={"buy": n_buy, "sell": n_sell, "hold": n_hold, "total": total},
                                scan_result={
                                    "mode": "combined",
                                    "strat_votes":   strat_votes,
                                    "strat_details": strat_details,
                                },
                            )
                            _log_action({
                                "timestamp":  datetime.now().isoformat(),
                                "symbol":     sym,
                                "signal":     combined_signal,
                                "confidence": _comb_conf_off,
                                "executed":   False,
                                "result":     "Auto Execute dezactivat — semnal detectat dar neexecutat",
                                "strategy":   "combined",
                                "tf":         _best_res_combined.get("tf", "?") if _best_res_combined else "?",
                                "sl":         _best_res_combined.get("sl") if _best_res_combined else None,
                                "tp":         _best_res_combined.get("tp") if _best_res_combined else None,
                                "voters":     _comb_voters_off,
                                "vote_map":   {"buy": n_buy, "sell": n_sell, "hold": n_hold, "total": total},
                                "chart_id":   _comb_cid_off,
                            })
                            sym_results["_combined_votes"]["auto_executed"] = False
                            sym_results["_combined_votes"]["skip_reason"] = "auto_execute_off"

                        # Executa doar daca auto_execute e ON si nu e haltat
                        if auto_ex and not is_daily_drawdown_halted() and combined_signal:
                            best_res  = _best_res_combined
                            best_conv = _best_conv_combined

                            if not best_res:
                                log.warning(
                                    f"[Combined] {sym} {combined_signal} — skip: niciun tf_result cu SL+TP valid. "
                                    f"BUY votes: {[r for r in all_tf_results if r['signal']=='BUY']}"
                                )
                                _log_action({
                                    "timestamp": datetime.now().isoformat(),
                                    "symbol": sym, "signal": combined_signal,
                                    "confidence": round((n_buy if combined_signal=="BUY" else n_sell)/total*100,1),
                                    "executed": False,
                                    "result": "Skip: niciun TF cu SL+TP valid pentru executie",
                                    "strategy": "combined",
                                    "vote_map": {"buy": n_buy, "sell": n_sell, "hold": n_hold, "total": total},
                                })

                            if best_res:
                                pct_signal = round(
                                    (n_buy if combined_signal == "BUY" else n_sell) / total * 100, 1
                                )
                                # Mapare: strategiile care au votat directia castigatoare
                                voters = list({
                                    r["strat"] for r in all_tf_results
                                    if r["signal"] == combined_signal
                                })

                                ok, msg = place_trade(
                                    sym, combined_signal,
                                    best_res["sl"], best_res["tp"], RISK_DOLLARS,
                                    strategy="combined",
                                    one_per_strategy=False,
                                )

                                _comb_cid = _store_analysis(
                                    sym, "combined", combined_signal,
                                    best_res.get("tf"), best_res.get("price"),
                                    best_res.get("sl"), best_res.get("tp"),
                                    pct_signal,
                                    [f"Combined {combined_signal}: {len(voters)}/{total} strategii",
                                     f"Votanti: {', '.join(voters)}"],
                                    voters=voters,
                                    vote_map={"buy": n_buy, "sell": n_sell,
                                              "hold": n_hold, "total": total},
                                    scan_result={
                                        "mode": "combined",
                                        "strat_votes":   strat_votes,
                                        "strat_details": strat_details,
                                    },
                                )
                                if ok:
                                    _link_ticket_to_chart(msg, _comb_cid)

                                # ── Logheaza actiunea in review_log ──
                                _log_action({
                                    "timestamp":  datetime.now().isoformat(),
                                    "symbol":     sym,
                                    "signal":     combined_signal,
                                    "confidence": pct_signal,
                                    "executed":   ok,
                                    "result":     msg,
                                    "strategy":   "combined",
                                    "tf":         best_res.get("tf", "?"),
                                    "sl":         best_res.get("sl"),
                                    "tp":         best_res.get("tp"),
                                    "voters":     voters,
                                    "vote_map": {
                                        "buy":  n_buy,
                                        "sell": n_sell,
                                        "hold": n_hold,
                                        "total": total,
                                    },
                                    "chart_id":   _comb_cid,
                                })

                                # ── Inregistreaza performanta ca "combined" ──
                                record_perf("combined", sym, combined_signal, 0.0, ok)

                                # ── Marcheaza executia pe _combined_votes ──
                                sym_results["_combined_votes"]["auto_executed"] = ok
                                sym_results["_combined_votes"]["voters"] = voters

                                if ok:
                                    _tg.notify_only(
                                        symbol=sym, signal=combined_signal, strategy="combined",
                                        sl=best_res["sl"], tp=best_res["tp"],
                                        price=best_res.get("price", 0),
                                        confidence=pct_signal,
                                        tf=best_res.get("tf", "?"),
                                        reasons=[
                                            f"Combined Mode: {n_buy if combined_signal=='BUY' else n_sell}/{total} ({pct_signal}%)",
                                            f"Votanti: {', '.join(voters)}",
                                        ],
                                    )
                                log.info(
                                    f"[Combined] {sym} {combined_signal} "
                                    f"BUY={n_buy}/{total} ({sym_results['_combined_votes']['pct_buy']}%) "
                                    f"SELL={n_sell}/{total} HOLD={n_hold}/{total} "
                                    f"voters={voters} → {msg}"
                                )

                    # ── Auto-execute — DOAR semnale din scan-ul CURENT ────
                    if auto_ex and is_daily_drawdown_halted():
                        halt_ok, halt_msg = check_daily_drawdown_halt()
                        if halt_ok:
                            log.warning(f"[Scanner] {halt_msg} — auto-execute oprit")
                            _log_action({
                                "timestamp": datetime.now().isoformat(),
                                "symbol": sym, "signal": "HALT",
                                "confidence": 0, "executed": False,
                                "result": halt_msg, "strategy": "system",
                            })

                    # ── Scalp Boost Mode ─────────────────────────────────────
                    scalp_boost = scanner.get("scalp_boost", False)
                    if auto_ex and scalp_boost and not is_daily_drawdown_halted() and not combined_mode:
                        _s_min_agree = int(scanner.get("scalp_min_agree", 3))
                        _s_tp_rr     = float(scanner.get("scalp_tp_rr", 1.0))
                        _s_lot_mult  = float(scanner.get("scalp_lot_mult", 2.0))

                        # Numara voturi din scan curent
                        _sb_buy  = [(k, r) for k, r in current_scan_results.items()
                                    if r.get("signal") == "BUY"  and r.get("best_tf")]
                        _sb_sell = [(k, r) for k, r in current_scan_results.items()
                                    if r.get("signal") == "SELL" and r.get("best_tf")]

                        _sb_signal = None
                        _sb_voters = []
                        _sb_best   = None
                        if len(_sb_buy) >= _s_min_agree and len(_sb_buy) > len(_sb_sell):
                            _sb_signal = "BUY"
                            _sb_voters = [k for k, _ in _sb_buy]
                            # Alege cel mai convingator result
                            _sb_best = max(
                                [r for _, r in _sb_buy if r.get("best_tf")],
                                key=lambda r: r.get("confidence", 0)
                            )
                        elif len(_sb_sell) >= _s_min_agree and len(_sb_sell) > len(_sb_buy):
                            _sb_signal = "SELL"
                            _sb_voters = [k for k, _ in _sb_sell]
                            _sb_best = max(
                                [r for _, r in _sb_sell if r.get("best_tf")],
                                key=lambda r: r.get("confidence", 0)
                            )

                        if _sb_signal and _sb_best:
                            _bf = _sb_best.get("best_tf", {})
                            _sl = _bf.get("sl")
                            _price = _bf.get("price") or float(
                                [v for v in current_scan_results.values()
                                 if v.get("best_tf")][0].get("best_tf", {}).get("price", 0)
                                if [v for v in current_scan_results.values() if v.get("best_tf")]
                                else 0
                            )
                            if _sl and _price:
                                # TP strict 1:N (scalp_tp_rr)
                                _risk = abs(_price - _sl)
                                _tp = round(_price + _risk * _s_tp_rr, 5) if _sb_signal == "BUY" \
                                      else round(_price - _risk * _s_tp_rr, 5)
                                _scalp_risk = _app.RISK_DOLLARS * _s_lot_mult

                                ok_s, msg_s = place_trade(
                                    sym, _sb_signal, _sl, _tp, _scalp_risk,
                                    strategy="scalp_boost",
                                    one_per_strategy=False,
                                )
                                _sb_conf = round(len(_sb_voters) / max(len(current_scan_results), 1) * 100, 1)
                                _sb_votes = [
                                    {"strat": k, "signal": _sb_signal,
                                     "tf": (current_scan_results.get(k, {}).get("best_tf") or {}).get("tf", "?"),
                                     "sl": (current_scan_results.get(k, {}).get("best_tf") or {}).get("sl"),
                                     "tp": (current_scan_results.get(k, {}).get("best_tf") or {}).get("tp"),
                                     "price": (current_scan_results.get(k, {}).get("best_tf") or {}).get("price"),
                                     "conviction": (current_scan_results.get(k, {}).get("best_tf") or {}).get("conviction", 0),
                                     "coherence": 1.0}
                                    for k in _sb_voters
                                ]
                                _sb_details = {
                                    k: {
                                        "signal": current_scan_results.get(k, {}).get("signal", "HOLD"),
                                        "confidence": current_scan_results.get(k, {}).get("confidence", 0),
                                        "best_tf": (current_scan_results.get(k, {}).get("best_tf") or {}).get("tf", "?"),
                                        "conviction": (current_scan_results.get(k, {}).get("best_tf") or {}).get("conviction", 0),
                                        "reasons": ((current_scan_results.get(k, {}).get("best_tf") or {}).get("reasons")
                                                    or current_scan_results.get(k, {}).get("justification") or [])[:5],
                                        "sl": (current_scan_results.get(k, {}).get("best_tf") or {}).get("sl"),
                                        "tp": (current_scan_results.get(k, {}).get("best_tf") or {}).get("tp"),
                                        "price": (current_scan_results.get(k, {}).get("best_tf") or {}).get("price"),
                                    }
                                    for k in _sb_voters
                                }
                                _sb_cid  = _store_analysis(
                                    sym, "scalp_boost", _sb_signal,
                                    _bf.get("tf"), _price, _sl, _tp, _sb_conf,
                                    [f"ScalpBoost {_sb_signal}: {len(_sb_voters)} strategii acord",
                                     f"Lot×{_s_lot_mult}, TP 1:{_s_tp_rr}",
                                     f"Votanti: {', '.join(_sb_voters)}"],
                                    voters=_sb_voters,
                                    scan_result={
                                        "mode": "scalp_boost",
                                        "strat_votes":   _sb_votes,
                                        "strat_details": _sb_details,
                                    },
                                )
                                if ok_s:
                                    _link_ticket_to_chart(msg_s, _sb_cid)
                                _sb_entry = {
                                    "timestamp":  datetime.now().isoformat(),
                                    "symbol":     sym,
                                    "signal":     _sb_signal,
                                    "confidence": _sb_conf,
                                    "executed":   ok_s,
                                    "result":     msg_s,
                                    "strategy":   "scalp_boost",
                                    "voters":     _sb_voters,
                                    "sl":         _sl,
                                    "tp":         _tp,
                                    "lot_mult":   _s_lot_mult,
                                    "chart_id":   _sb_cid,
                                }
                                _log_action(_sb_entry)
                                with _scanner_lock:
                                    if not _is_duplicate_decision(_sb_entry, window=6):
                                        decisions.insert(0, _sb_entry)
                                        while len(decisions) > 50: decisions.pop()
                                if ok_s:
                                    record_perf("scalp_boost", sym, _sb_signal, 0.0, True)
                                    log.info(
                                        f"[ScalpBoost] {sym} {_sb_signal}"
                                        f" voters={_sb_voters} lot×{_s_lot_mult}"
                                        f" TP:{_tp} SL:{_sl}"
                                    )

                    if auto_ex and not is_daily_drawdown_halted() and not combined_mode and not scalp_boost:
                        _pending_mode = scanner.get("pending_mode", False)
                        _pending_expiry = scanner.get("pending_expiry_hours", 24)

                        # ── Mod normal: fiecare strategie executa individual ─
                        for strat_key, res in current_scan_results.items():
                            ok, msg = False, ""
                            executed_as_pending = False

                            # ── Pending Mode: EOB cu zona de abordare ──────────
                            if _pending_mode and strat_key == "eob":
                                pe = res.get("pending_entry")
                                if pe and res.get("signal") == "HOLD":
                                    # Expiration dinamica: TF-ul zonei determina cat timp e valid setup-ul
                                    _tf_expiry_map = {
                                        "M5": 3, "M15": 6, "M30": 12,
                                        "H1": 24, "H4": 72, "D1": 120, "W1": 240,
                                    }
                                    _zone_tf = pe.get("tf", "H1")
                                    _dyn_exp = _tf_expiry_map.get(_zone_tf, _pending_expiry)
                                    # User override: daca a setat manual in UI > valoare TF → foloseste user
                                    _exp_h = max(_dyn_exp, _pending_expiry) if _pending_expiry > 24 else _dyn_exp

                                    ok, msg = place_pending_order(
                                        sym, pe["signal"],
                                        pe["price"], pe["sl"], pe["tp"],
                                        RISK_DOLLARS, strategy="eob",
                                        expiry_hours=_exp_h,
                                    )
                                    executed_as_pending = True
                                    entry = {
                                        "timestamp":  datetime.now().isoformat(),
                                        "symbol":     sym,
                                        "signal":     f"PENDING {pe['order_type']} ({_zone_tf}/{_exp_h}h)",
                                        "confidence": 0,
                                        "executed":   ok,
                                        "result":     msg,
                                        "strategy":   "eob",
                                        "pending":    True,
                                    }
                                    _log_action(entry)
                                    with _scanner_lock:
                                        if not _is_duplicate_decision(entry, window=8):
                                            decisions.insert(0, entry)
                                            while len(decisions) > 50: decisions.pop()
                                    continue

                            # ── Market order normal ───────────────────────────
                            if res.get("signal") not in ("BUY", "SELL"):
                                continue
                            bf = res.get("best_tf")
                            if not bf:
                                continue
                            _use_sp2 = scanner.get("use_strategy_params", False)
                            _strat_cfg2 = scanner.get(strat_key, {})
                            _exec_risk = float(_strat_cfg2["risk_dollars"]) if (_use_sp2 and "risk_dollars" in _strat_cfg2) else RISK_DOLLARS
                            ok, msg = place_trade(
                                sym, res["signal"], bf["sl"], bf["tp"], _exec_risk,
                                strategy=strat_key,
                                one_per_strategy=scanner.get("one_per_strategy", False),
                            )
                            res["auto_executed"] = ok
                            if strat_key in sym_results:
                                sym_results[strat_key]["auto_executed"] = ok
                            if ok:
                                record_perf(strat_key, sym, res["signal"], 0.0, True)
                                _tg.notify_only(
                                    symbol=sym, signal=res["signal"], strategy=strat_key,
                                    sl=bf["sl"], tp=bf["tp"],
                                    price=bf.get("price", 0),
                                    confidence=res.get("confidence", 0),
                                    tf=bf.get("tf", "?"),
                                    reasons=res.get("justification", []),
                                )
                            _cid = _store_analysis(
                                sym, strat_key, res["signal"],
                                bf.get("tf"), bf.get("price"),
                                bf.get("sl"), bf.get("tp"),
                                res.get("confidence", 0),
                                res.get("justification", []),
                                scan_result=res,
                            )
                            if ok:
                                _link_ticket_to_chart(msg, _cid)
                            entry = {
                                "timestamp":  datetime.now().isoformat(),
                                "symbol":     sym,
                                "signal":     res["signal"],
                                "confidence": res.get("confidence", 0),
                                "executed":   ok,
                                "result":     msg,
                                "strategy":   strat_key,
                                "chart_id":   _cid,
                            }
                            _log_action(entry)
                            with _scanner_lock:
                                if not _is_duplicate_decision(entry, window=8):
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
    # Reincearca incarcarea cache-ului acum ca MT5 e (probabil) conectat
    try:
        _bootstrap_cache()
    except Exception:
        pass
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
<script src="https://cdn.plot.ly/plotly-2.35.2.min.js" charset="utf-8"></script>
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
.toggle { position:relative; display:inline-block; width:42px; height:22px; flex-shrink:0; }
.toggle input { opacity:0; width:0; height:0; }
.slider {
    position:absolute; cursor:pointer; inset:0;
    background:#c62828; border-radius:22px; transition:background 0.2s;
}
.slider:before {
    content:""; position:absolute; width:16px; height:16px;
    left:3px; bottom:3px; background:#fff; border-radius:50%;
    transition:transform 0.2s, background 0.2s;
}
input:checked + .slider { background:#2e7d32; }
input:checked + .slider:before { transform:translateX(20px); background:#fff; }
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

/* Chart div */
.chart-frame-wrap { width:100%; min-height:700px; background:#111; border-radius:4px; overflow:hidden; border:1px solid #222; }
#dp-chart-div { width:100%; height:720px; background:#111; }
#dp-chart-loading { display:none; align-items:center; justify-content:center; height:720px; color:#555; font-size:0.9rem; }
#dp-chart-loading.visible { display:flex; }
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
    <button class="btn btn-sm btn-green" id="btn-start" onclick="startScanner()">▶ Start</button>
    <button class="btn btn-sm btn-red"   id="btn-stop"  onclick="stopScanner()" disabled>■ Stop</button>
    <a href="/trades"      class="btn btn-sm btn-teal">📊 Trades</a>
    <a href="/autoorders/" class="btn btn-sm" style="background:#1a1030;color:#ce93d8;border:1px solid #6a1b9a">⬡ AutoOrders</a>
    <a href="/backtest"    class="btn btn-sm" style="background:#0d1f0d;color:#66bb6a;border:1px solid #2e7d32">📈 Backtest</a>
    <a href="/"            class="btn btn-sm btn-grey">← Chart</a>
    <button class="btn btn-sm btn-grey" onclick="toggleCompactSettings()" id="btn-settings" style="border:1px solid #2a2a3e">⚙ Setari</button>
    <a href="/logout"      class="btn btn-sm btn-grey" style="margin-left:4px;color:#ef9a9a">⏻ Logout</a>
</div>

<!-- ══ COMPACT SETTINGS PANEL ══════════════════════════════════════════════ -->
<div id="compact-settings" style="display:none;background:#0c0c1a;border-bottom:1px solid #1c1c30;padding:16px 24px;display:none">
  <div style="display:grid;grid-template-columns:300px 1fr;gap:24px;max-width:900px">

    <!-- COL 1: Telegram + Cloudflare -->
    <div>
      <div style="font-size:.72rem;font-weight:700;color:#3a4860;text-transform:uppercase;letter-spacing:.8px;margin-bottom:10px">📱 Telegram Bot</div>
      <div style="display:flex;flex-direction:column;gap:7px">
        <div style="display:flex;align-items:center;justify-content:space-between">
          <span style="font-size:.78rem;color:#8090a0">Notificari</span>
          <label class="toggle"><input type="checkbox" id="cs-tg-enabled" onchange="csTgSave()"><span class="slider"></span></label>
        </div>
        <div>
          <div style="font-size:.7rem;color:#4a5870;margin-bottom:2px">Bot Token</div>
          <input id="cs-tg-token" type="password" placeholder="123456:ABC-DEF..."
                 style="width:100%;background:#101020;border:1px solid #1e1e35;color:#c0c8e0;padding:4px 8px;border-radius:5px;font-size:.78rem">
        </div>
        <div>
          <div style="font-size:.7rem;color:#4a5870;margin-bottom:2px">Chat ID</div>
          <input id="cs-tg-chatid" type="text" placeholder="ex: 123456789"
                 style="width:100%;background:#101020;border:1px solid #1e1e35;color:#c0c8e0;padding:4px 8px;border-radius:5px;font-size:.78rem">
        </div>
        <div style="display:flex;align-items:center;justify-content:space-between">
          <span style="font-size:.78rem;color:#8090a0">Aprobare manuala</span>
          <label class="toggle"><input type="checkbox" id="cs-tg-approval" onchange="csTgSave()"><span class="slider"></span></label>
        </div>
        <div style="display:flex;gap:6px;margin-top:2px">
          <button onclick="csTgSave()" style="flex:1;padding:4px;border-radius:5px;border:1px solid #3949ab;background:#1a1a2e;color:#7986cb;font-size:.76rem;cursor:pointer">💾 Salveaza</button>
          <button onclick="csTgTest()" style="flex:1;padding:4px;border-radius:5px;border:1px solid #2a3a2a;background:#0f1a0f;color:#66bb6a;font-size:.76rem;cursor:pointer">✉ Test</button>
        </div>
        <div id="cs-tg-msg" style="font-size:.72rem;min-height:14px;text-align:center"></div>
      </div>

      <div style="font-size:.72rem;font-weight:700;color:#3a4860;text-transform:uppercase;letter-spacing:.8px;margin:14px 0 10px">🌐 Cloudflare Tunnel</div>
      <div style="display:flex;align-items:center;gap:8px">
        <span class="status-dot" id="cs-cf-dot"></span>
        <span id="cs-cf-text" style="font-size:.78rem;color:#555;flex:1">Se verifica...</span>
        <button id="cs-cf-btn" onclick="csCfToggle()" style="padding:3px 10px;border-radius:4px;font-size:.75rem;cursor:pointer;border:1px solid #2a6a2a;background:#1b3a1b;color:#66bb6a">▶ Start</button>
      </div>
      <div id="cs-cf-url" style="display:none;margin-top:6px;font-size:.72rem;color:#26a69a;word-break:break-all;padding:5px 8px;background:#0a1a0a;border-radius:4px;border:1px solid #1b3a1b">
        <a id="cs-cf-url-link" href="#" target="_blank" style="color:#4db6ac;text-decoration:none"></a>
        <button onclick="csCfCopy()" style="margin-left:6px;background:#111;border:1px solid #333;color:#888;padding:1px 6px;border-radius:3px;font-size:.68rem;cursor:pointer">📋</button>
      </div>
    </div>

    <!-- COL 2: Quick Guide -->
    <div>
      <div style="font-size:.72rem;font-weight:700;color:#3a4860;text-transform:uppercase;letter-spacing:.8px;margin-bottom:10px">📖 Quick Guide</div>
      <div style="font-size:.76rem;color:#6070a0;line-height:2">
        <div style="color:#546e7a;font-weight:600;margin-bottom:3px">CONFIDENCE</div>
        Conviction × 8.5 = baza<br>
        +5% per TF confirmare<br>
        −8% per TF conflict<br>
        <div style="border-top:1px solid #1a1a2e;margin:8px 0"></div>
        <div style="color:#546e7a;font-weight:600;margin-bottom:3px">CONVICTION SCALE</div>
        <span style="color:#ef5350">0–3</span> slab — HOLD<br>
        <span style="color:#ffa726">4–6</span> decent — ~50%<br>
        <span style="color:#66bb6a">7–9</span> solid — ~70%<br>
        <span style="color:#26a69a">10+</span> excelent — 85%+<br>
        <div style="border-top:1px solid #1a1a2e;margin:8px 0"></div>
        <div style="color:#546e7a;font-weight:600;margin-bottom:3px">RISK SCORE</div>
        conv/14 × (1+agree−conflict) × 100<br>
        ≥70 = setup solid · ≥85 = excelent<br>
        <div style="border-top:1px solid #1a1a2e;margin:8px 0"></div>
        <div style="color:#546e7a;font-weight:600;margin-bottom:3px">TIMEFRAMES</div>
        <b style="color:#c0c8e0">Scalp</b>: M1–M5 · VWAP, Session<br>
        <b style="color:#c0c8e0">Intraday</b>: M15–H1 · SMC, Classic<br>
        <b style="color:#c0c8e0">Swing</b>: H4–D1 · MACD, Ichimoku, EOB<br>
        <div style="border-top:1px solid #1a1a2e;margin:8px 0"></div>
        <div style="color:#546e7a;font-weight:600;margin-bottom:3px">FTMO 10k</div>
        Risk ≤$50/trade · Max DD $1000<br>
        Daily loss $500 · Max lot 1.0
      </div>
    </div>

  </div>
</div>
<!-- ══════════════════════════════════════════════════════════════════════════ -->

<!-- SETTINGS PANEL (scanner config vechi — pastrat in DOM dar nefolosit) -->
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
            <input type="range" id="interval-range" min="0" max="5" step="1" value="5" style="width:90px" oninput="onIntervalChange(this)">
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
        <div class="config-row" style="border:1px solid #1a2a3a;border-radius:6px;padding:8px 10px;margin-top:2px;background:#0f1520">
            <label>Parametri per strategie<span class="sub">SL mult + risk propriu la executie</span></label>
            <label class="toggle"><input type="checkbox" id="use-strategy-params-toggle" onchange="sendGlobal('use_strategy_params',this.checked)"><span class="slider"></span></label>
        </div>
        <div class="config-row">
            <label>1 trade / strategie<span class="sub">Permite mai multe strategii pe acelasi simbol</span></label>
            <label class="toggle"><input type="checkbox" id="one-per-strategy-toggle" onchange="sendGlobal('one_per_strategy',this.checked)"><span class="slider"></span></label>
        </div>
        <div class="config-row" style="border:1px solid #3a3a2a;border-radius:6px;padding:8px 10px;margin-top:4px;background:#1a1a0f">
            <label>Time-based exit<span class="sub">Inchide trade vechi dupa max_trade_hours, SL la BE dupa breakeven_hours</span></label>
            <label class="toggle"><input type="checkbox" id="time-exit-toggle" onchange="sendGlobal('time_based_exit',this.checked)"><span class="slider"></span></label>
        </div>
        <div class="config-row">
            <label>Breakeven dupa (ore)<span class="sub">Muta SL la pretul de intrare daca trade in profit</span></label>
            <input type="number" id="breakeven-hours-input" value="2" min="0.25" max="24" step="0.25" style="width:60px" onchange="sendGlobal('breakeven_hours',parseFloat(this.value))">
        </div>
        <div class="config-row">
            <label>Close dupa (ore)<span class="sub">Inchide oricum — libereaza capital</span></label>
            <input type="number" id="max-trade-hours-input" value="6" min="0.5" max="72" step="0.5" style="width:60px" onchange="sendGlobal('max_trade_hours',parseFloat(this.value))">
        </div>
        <div class="config-row" style="border:1px solid #2a3a2a;border-radius:6px;padding:8px 10px;margin-top:4px;background:#0f1f0f">
            <label>Combined Mode<span class="sub">Conviction-weighted — scor 60%+ executa</span></label>
            <label class="toggle"><input type="checkbox" id="combined-mode-toggle" onchange="sendGlobal('combined_mode',this.checked);toggleCombinedPanel(this.checked)"><span class="slider"></span></label>
        </div>

        <div class="config-row">
            <label>Crypto<span class="sub">Adauga simboluri crypto</span></label>
            <div style="display:flex;align-items:center;gap:8px">
                <label class="toggle"><input type="checkbox" id="crypto-toggle" onchange="toggleCrypto(this.checked)"><span class="slider"></span></label>
                <button onclick="setCryptoOnly()" style="background:#1a1a1a;border:1px solid #333;color:#e65100;padding:3px 10px;border-radius:4px;cursor:pointer;font-size:0.75rem;white-space:nowrap">₿ Crypto Only</button>
                <button onclick="setForexOnly()" style="background:#1a1a1a;border:1px solid #333;color:#1976d2;padding:3px 10px;border-radius:4px;cursor:pointer;font-size:0.75rem;white-space:nowrap">Forex Only</button>
            </div>
        </div>

        <!-- ── Risk & Trading Style ──────────────────────────────── -->
        <div style="margin-top:14px;border-top:1px solid #2a2a2a;padding-top:10px">
            <div style="font-size:0.74rem;color:#546e7a;font-weight:600;margin-bottom:8px;letter-spacing:0.5px">RISK &amp; TRADING STYLE</div>

            <!-- Preset list -->
            <div style="display:flex;flex-direction:column;gap:2px;margin-bottom:10px">
                <button id="preset-scalp"        onclick="applyPreset('scalp')"        class="preset-row-btn"><span class="preset-name">⚡ Scalp</span><span class="preset-sub">M1/M5 · SL strans 1.0x · TP 1:1 · lot mic</span></button>
                <button id="preset-intraday"     onclick="applyPreset('intraday')"     class="preset-row-btn"><span class="preset-name">📈 Intraday</span><span class="preset-sub">M15/H1 · risc moderat · TP 1:2</span></button>
                <button id="preset-swing"        onclick="applyPreset('swing')"        class="preset-row-btn"><span class="preset-name">🌊 Swing</span><span class="preset-sub">H1/H4 · SL larg 2x · TP 1:3 · pozitii lungi</span></button>
                <button id="preset-sniper"       onclick="applyPreset('sniper')"       class="preset-row-btn"><span class="preset-name">🎯 Sniper</span><span class="preset-sub">3 strat. premium · conf 85%+ · rar dar precis</span></button>
                <button id="preset-session"      onclick="applyPreset('session')"      class="preset-row-btn"><span class="preset-name">🏙 Session</span><span class="preset-sub">London / NY / China breakout only</span></button>
                <button id="preset-reversal"     onclick="applyPreset('reversal')"     class="preset-row-btn"><span class="preset-name">🐻 Reversal</span><span class="preset-sub">Contra-trend · RSI divergente + pin bars</span></button>
                <button id="preset-conservative" onclick="applyPreset('conservative')" class="preset-row-btn"><span class="preset-name">💎 Conservative</span><span class="preset-sub">3 strat. sigure · conf 80%+ · risc minim FTMO</span></button>
                <button id="preset-aggressive"   onclick="applyPreset('aggressive')"   class="preset-row-btn"><span class="preset-name">🚀 Aggressive</span><span class="preset-sub">Toate strategiile active + Combined Mode</span></button>
                <button id="preset-scalp_boost" onclick="applyPreset('scalp_boost')"  class="preset-row-btn" style="border-color:#3a2200;background:#1a0f00"><span class="preset-name" style="color:#ff9800">⚡ Scalp Boost</span><span class="preset-sub">Lots×mult · TP 1:1 · 3+ strategii agree · max 30 min</span></button>
            </div>
            <style>
                .preset-row-btn {
                    display:flex; flex-direction:column; align-items:flex-start;
                    width:100%; padding:6px 10px;
                    border-radius:4px; border:1px solid #2a2a2a;
                    background:#141414; color:#ccc;
                    font-size:0.75rem; cursor:pointer; text-align:left; gap:1px;
                }
                .preset-row-btn:hover { border-color:#444; background:#1e1e1e; }
                .preset-name { font-weight:600; color:#ccc; }
                .preset-sub  { font-size:0.67rem; color:#555; }
            </style>
            <div style="font-size:0.68rem;color:#555;margin-bottom:10px;padding:4px 6px;background:#0f1419;border-radius:3px;line-height:1.4">
                <b style="color:#26a69a">FTMO 10k:</b> risk &le;$50/trade · max DD $1000 · daily loss $500<br>
                <span style="color:#666">Preseturile activeaza strategii + TF + SL + lots optim</span>
            </div>

            <!-- Risk mode -->
            <div class="config-row">
                <label>Mod risc<span class="sub">$ fix sau % equity</span></label>
                <div style="display:flex;gap:5px">
                    <button id="risk-mode-usd" onclick="setRiskMode(false)" style="padding:3px 8px;border-radius:4px;border:1px solid #ffeb3b;background:#1a1a1a;color:#ffeb3b;font-size:0.73rem;cursor:pointer">$ Fix</button>
                    <button id="risk-mode-pct" onclick="setRiskMode(true)"  style="padding:3px 8px;border-radius:4px;border:1px solid #333;background:#1a1a1a;color:#666;font-size:0.73rem;cursor:pointer">% Equity</button>
                </div>
            </div>

            <!-- Risk amount -->
            <div class="config-row" id="risk-usd-row">
                <label>Risc / trade<span class="sub">USD per tranzactie</span></label>
                <div style="display:flex;align-items:center;gap:4px">
                    <span style="color:#666;font-size:0.8rem">$</span>
                    <input type="number" id="risk-dollars-input" value="50" min="1" max="10000" step="1"
                        style="width:65px;background:#1e1e1e;color:#ffeb3b;border:1px solid #333;border-radius:4px;padding:3px 6px;font-size:0.88rem;font-weight:700"
                        onchange="sendGlobal('risk_dollars', parseFloat(this.value))">
                </div>
            </div>
            <div class="config-row" id="risk-pct-row" style="display:none">
                <label>Risc / trade<span class="sub">% din equity</span></label>
                <div style="display:flex;align-items:center;gap:4px">
                    <input type="number" id="risk-pct-input" value="1.0" min="0.1" max="10" step="0.1"
                        style="width:55px;background:#1e1e1e;color:#ffeb3b;border:1px solid #333;border-radius:4px;padding:3px 6px;font-size:0.88rem;font-weight:700"
                        onchange="sendGlobal('risk_pct', parseFloat(this.value))">
                    <span style="color:#666;font-size:0.8rem">%</span>
                </div>
            </div>

            <!-- SL Multiplier -->
            <div class="config-row">
                <label>SL Multiplier<span class="sub">ATR × val (mic=SL strâns)</span></label>
                <div style="display:flex;align-items:center;gap:6px">
                    <input type="range" id="sl-mult-range" min="0.5" max="3.0" step="0.1" value="2.0"
                        style="width:70px" oninput="onSlMultChange(this)">
                    <span id="sl-mult-label" style="color:#ab47bc;font-size:0.85rem;font-weight:700;min-width:28px">2.0</span>
                </div>
            </div>

            <!-- Max Lot -->
            <div class="config-row">
                <label>Max lot / trade<span class="sub">Leverage 100 → 5+</span></label>
                <input type="number" id="max-lot-input" value="1.0" min="0.01" max="100" step="0.01"
                    style="width:65px;background:#1e1e1e;color:#26a69a;border:1px solid #333;border-radius:4px;padding:3px 6px;font-size:0.88rem;font-weight:700"
                    onchange="sendGlobal('max_lot_global', parseFloat(this.value))">
            </div>
        </div>

        </div>
    </div>

    <!-- Selector + detalii strategie -->
    <div class="settings-section" id="settings-strategies">
        <div style="display:flex;align-items:center;justify-content:space-between;margin-bottom:6px;gap:6px">
            <h4 style="margin:0">Strategii</h4>
            <div style="display:flex;gap:6px">
                <button id="toggle-all-strats-btn" onclick="toggleAllStrategies()"
                    style="padding:4px 10px;border-radius:4px;border:1px solid #26a69a;background:#0d1f1d;color:#26a69a;font-size:0.72rem;cursor:pointer;white-space:nowrap">
                    ✓ Activeaza toate
                </button>
                <button onclick="restoreDefaults()" title="Revine la setarile implementate implicit pentru toate strategiile"
                    style="padding:4px 10px;border-radius:4px;border:1px solid #3a2a2a;background:#1a0d0d;color:#ffab91;font-size:0.72rem;cursor:pointer;white-space:nowrap">
                    ↻ Restore Defaults
                </button>
            </div>
        </div>
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

    <!-- Quick Guide -->
    <div class="settings-section" id="settings-legend">
        <h4>Quick Guide</h4>
        <div style="font-size:0.76rem;color:#aaa;line-height:1.9">
            <div style="color:#546e7a;font-weight:600;margin-bottom:4px">CONFIDENCE</div>
            Conviction x 8.5 = baza<br>
            +5% per TF confirmare<br>
            -8% per TF conflict<br>
            <div style="border-top:1px solid #2a2a2a;margin:6px 0"></div>
            <div style="color:#546e7a;font-weight:600;margin-bottom:4px">CONVICTION SCALE</div>
            <span style="color:#ef5350">0-3</span> slab — HOLD<br>
            <span style="color:#ffa726">4-6</span> decent — ~50%<br>
            <span style="color:#66bb6a">7-9</span> solid — ~70%<br>
            <span style="color:#26a69a">10+</span> excelent — 85%+<br>
            <div style="border-top:1px solid #2a2a2a;margin:6px 0"></div>
            <div style="color:#546e7a;font-weight:600;margin-bottom:4px">TIMEFRAMES</div>
            <b>Scalp</b>: M1-M5 (VWAP, Session)<br>
            <b>Intraday</b>: M15-H1 (SMC, Classic)<br>
            <b>Swing</b>: H4-D1 (MACD, Ichimoku)<br>
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

    <!-- TABS STRATEGII — generate dinamic in JS (buildStratTabs) -->
    <div id="strat-tabs-bar" style="display:flex;gap:0;flex-wrap:wrap;margin-bottom:0;border-bottom:2px solid #2a2a2a"></div>

    <!-- Sectiuni strategii — generate dinamic in JS (buildStratTabs) -->
    <div id="strat-sections"></div>


    <!-- Combined Mode Panel — vizibil in main content cand e activ -->
    <div id="combined-panel" style="display:none;margin:14px 0;background:#0f1a0f;border:1px solid #2a3a2a;border-radius:8px;padding:14px 18px">
        <div style="display:flex;align-items:center;gap:10px;margin-bottom:12px">
            <span style="font-size:0.85rem;font-weight:700;color:#66bb6a">🗳 Combined Mode</span>
            <span style="font-size:0.75rem;color:#555">BUY ≥60% din toate strategiile + spread(BUY−SELL) ≥21% → executa</span>
            <button onclick="refreshCombinedVotes();refreshCombinedLog()" style="margin-left:auto;background:#1b2e1b;border:1px solid #2a3a2a;color:#66bb6a;padding:3px 12px;border-radius:4px;font-size:0.75rem;cursor:pointer">↻ Refresh</button>
        </div>
        <div id="combined-votes-container" style="font-size:0.78rem;color:#555">
            Se asteapta primul scan cu Combined Mode activ...
        </div>
        <!-- Action log -->
        <div style="margin-top:16px;border-top:1px solid #1e2e1e;padding-top:12px">
            <div style="display:flex;align-items:center;gap:10px;margin-bottom:8px">
                <span style="font-size:0.8rem;font-weight:600;color:#555">📋 Actiuni recente Combined</span>
                <button onclick="refreshCombinedLog()" style="margin-left:auto;background:#111;border:1px solid #2a2a2a;color:#555;padding:2px 10px;border-radius:4px;font-size:0.72rem;cursor:pointer">↻</button>
            </div>
            <div id="combined-action-log" style="font-size:0.75rem;color:#555;max-height:260px;overflow-y:auto">
                Se incarca...
            </div>
        </div>
    </div>

    <!-- Pending Orders Panel -->
    <div id="pending-panel" style="display:none;margin:14px 0;background:#0a1020;border:1px solid #1a2a4a;border-radius:8px;padding:14px 18px">
        <div style="display:flex;align-items:center;gap:10px;margin-bottom:12px">
            <span style="font-size:0.85rem;font-weight:700;color:#7c4dff">⏳ Pending Orders — EOB</span>
            <span style="font-size:0.75rem;color:#555">Ordine Limit/Stop active in MT5, plasate automat la zone EOB</span>
            <button onclick="refreshPendingOrders()" style="margin-left:auto;background:#111;border:1px solid #2a2a4a;color:#7c4dff;padding:3px 12px;border-radius:4px;font-size:0.75rem;cursor:pointer">↻ Refresh</button>
        </div>
        <div style="display:flex;gap:10px;margin-bottom:10px;font-size:0.75rem">
            <label style="color:#888">Expira in</label>
            <input type="number" id="pending-expiry-input" value="24" min="1" max="168" step="1"
                style="width:55px;background:#1e1e1e;color:#7c4dff;border:1px solid #333;border-radius:4px;padding:2px 6px;font-size:0.82rem"
                onchange="sendGlobal('pending_expiry_hours', parseFloat(this.value))">
            <label style="color:#888">ore</label>
        </div>
        <div id="pending-orders-container" style="font-size:0.78rem;color:#555">
            Se asteapta primul scan cu Pending Mode activ...
        </div>
    </div>

    <!-- Detail panel -->
    <div id="detail-panel">
        <div class="detail-header">
            <span class="detail-sym" id="dp-symbol">—</span>
            <span class="badge hold" id="dp-badge">HOLD</span>
            <span class="conf-text" id="dp-conf"></span>
            <span id="dp-risk" style="font-size:0.72rem;font-weight:700;padding:2px 8px;border-radius:4px;background:#1a1a1a;margin-left:4px"></span>
        </div>
        <div class="detail-body">
            <div class="detail-left">
                <table class="tf-table">
                    <thead id="dp-tf-header"><tr>
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
                    <div id="dp-chart-loading" class="visible">Se incarca graficul...</div>
                    <div id="dp-chart-div"></div>
                </div>
            </div>
        </div>
    </div>

    <!-- Documentatie scoring -->
    <div class="decisions-section" style="margin-bottom:8px">
        <details style="cursor:pointer">
            <summary style="font-size:0.88rem;color:#90a4ae;padding:8px 0;user-select:none">
                📖 Cum se calculeaza scorul fiecarei strategii
            </summary>
            <div style="font-size:0.8rem;color:#aaa;line-height:1.7;padding:10px 0 4px 0">
                <table style="width:100%;border-collapse:collapse">
                <thead><tr style="color:#546e7a;font-size:0.75rem">
                    <th style="text-align:left;padding:4px 8px;border-bottom:1px solid #1e2a2a">Strategie</th>
                    <th style="text-align:left;padding:4px 8px;border-bottom:1px solid #1e2a2a">Element</th>
                    <th style="text-align:center;padding:4px 8px;border-bottom:1px solid #1e2a2a">Puncte</th>
                    <th style="text-align:left;padding:4px 8px;border-bottom:1px solid #1e2a2a">Conditie</th>
                </tr></thead>
                <tbody>
                <!-- CLASSIC -->
                <tr><td rowspan="4" style="padding:4px 8px;color:#26a69a;vertical-align:top;font-weight:600">🔵 Clasica</td>
                    <td style="padding:3px 8px">EMA cross</td><td style="text-align:center;padding:3px 8px">+2</td>
                    <td style="padding:3px 8px;color:#888">EMA20 &gt; EMA50 (BUY) sau EMA20 &lt; EMA50 (SELL)</td></tr>
                <tr><td style="padding:3px 8px">Fibonacci zone</td><td style="text-align:center;padding:3px 8px">+2</td>
                    <td style="padding:3px 8px;color:#888">Pretul e in zona de retracement 38.2%–61.8%</td></tr>
                <tr><td style="padding:3px 8px">ADX</td><td style="text-align:center;padding:3px 8px">+2</td>
                    <td style="padding:3px 8px;color:#888">ADX &gt; prag configurat (trend puternic)</td></tr>
                <tr><td style="padding:3px 8px">RSI neutru</td><td style="text-align:center;padding:3px 8px">+2</td>
                    <td style="padding:3px 8px;color:#888">RSI intre 40–60 (nu supracumparat/supravandut)</td></tr>
                <!-- SMC -->
                <tr style="background:#0a1010"><td rowspan="4" style="padding:4px 8px;color:#ff9800;vertical-align:top;font-weight:600">🟠 SMC</td>
                    <td style="padding:3px 8px">BOS</td><td style="text-align:center;padding:3px 8px">+2</td>
                    <td style="padding:3px 8px;color:#888">Break of Structure: pretul a depasit ultimul pivot H/L</td></tr>
                <tr style="background:#0a1010"><td style="padding:3px 8px">Order Block</td><td style="text-align:center;padding:3px 8px">+3</td>
                    <td style="padding:3px 8px;color:#888">Pretul e in interiorul unui OB activ (neaspart)</td></tr>
                <tr style="background:#0a1010"><td style="padding:3px 8px">FVG</td><td style="text-align:center;padding:3px 8px">+2</td>
                    <td style="padding:3px 8px;color:#888">Pretul e langa un Fair Value Gap neumplut</td></tr>
                <tr style="background:#0a1010"><td style="padding:3px 8px">Market Structure</td><td style="text-align:center;padding:3px 8px">+1</td>
                    <td style="padding:3px 8px;color:#888">HH+HL (bullish) sau LH+LL (bearish) in ultimele 50 bars</td></tr>
                <!-- EOB -->
                <tr><td rowspan="5" style="padding:4px 8px;color:#9c27b0;vertical-align:top;font-weight:600">🟣 EOB + Unicorn</td>
                    <td style="padding:3px 8px">HTF EOB (H4/D1)</td><td style="text-align:center;padding:3px 8px">+3</td>
                    <td style="padding:3px 8px;color:#888">Lumanare bullish+wick jos ≥25% engulfata bearish → pretul in zona (max 0.5% distanta)</td></tr>
                <tr><td style="padding:3px 8px">Unicorn BOS (MTF)</td><td style="text-align:center;padding:3px 8px">+2</td>
                    <td style="padding:3px 8px;color:#888">BOS confirmat pe H1/M15 in directia EOB HTF</td></tr>
                <tr><td style="padding:3px 8px">Unicorn FVG (MTF)</td><td style="text-align:center;padding:3px 8px">+1</td>
                    <td style="padding:3px 8px;color:#888">Pretul e IN FVG creat dupa BOS (confirmare Unicorn completa)</td></tr>
                <tr><td style="padding:3px 8px">EOB din EOB (LTF)</td><td style="text-align:center;padding:3px 8px">+4</td>
                    <td style="padding:3px 8px;color:#888">EOB mic pe M1/M5 in interiorul FVG de pe MTF → intrare precisa</td></tr>
                <tr><td style="padding:3px 8px;color:#555;font-style:italic" colspan="3">Semnal activ daca: o directie &gt; 50% din scorul total. Nu se dau semnal daca HTF are ambele directii (cel mai recent castiga).</td></tr>
                <!-- MACD -->
                <tr style="background:#0a1010"><td rowspan="3" style="padding:4px 8px;color:#29b6f6;vertical-align:top;font-weight:600">📊 MACD Cross</td>
                    <td style="padding:3px 8px">MACD cross</td><td style="text-align:center;padding:3px 8px">+3</td>
                    <td style="padding:3px 8px;color:#888">MACD line a trecut semnalul (golden/death cross)</td></tr>
                <tr style="background:#0a1010"><td style="padding:3px 8px">EMA200</td><td style="text-align:center;padding:3px 8px">+2</td>
                    <td style="padding:3px 8px;color:#888">Pretul e de aceeasi parte cu EMA200</td></tr>
                <tr style="background:#0a1010"><td style="padding:3px 8px">Histogram</td><td style="text-align:center;padding:3px 8px">+1</td>
                    <td style="padding:3px 8px;color:#888">Histograma creste in directia semnalului</td></tr>
                <!-- Bollinger -->
                <tr><td rowspan="3" style="padding:4px 8px;color:#ab47bc;vertical-align:top;font-weight:600">〰 Bollinger</td>
                    <td style="padding:3px 8px">Band touch</td><td style="text-align:center;padding:3px 8px">+3</td>
                    <td style="padding:3px 8px;color:#888">Pretul a atins banda superioara/inferioara</td></tr>
                <tr><td style="padding:3px 8px">RSI confirma</td><td style="text-align:center;padding:3px 8px">+2</td>
                    <td style="padding:3px 8px;color:#888">RSI overbought (&gt;70) la band top sau oversold (&lt;30) la band bot</td></tr>
                <tr><td style="padding:3px 8px">Squeeze</td><td style="text-align:center;padding:3px 8px">+1</td>
                    <td style="padding:3px 8px;color:#888">Benzile sunt inguste (volatilitate scazuta, potential breakout)</td></tr>
                <!-- Supertrend -->
                <tr style="background:#0a1010"><td rowspan="3" style="padding:4px 8px;color:#ffca28;vertical-align:top;font-weight:600">⚡ Supertrend</td>
                    <td style="padding:3px 8px">Supertrend</td><td style="text-align:center;padding:3px 8px">+3</td>
                    <td style="padding:3px 8px;color:#888">Pretul deasupra/sub linia Supertrend (ATR x multiplicator)</td></tr>
                <tr style="background:#0a1010"><td style="padding:3px 8px">EMA50</td><td style="text-align:center;padding:3px 8px">+2</td>
                    <td style="padding:3px 8px;color:#888">Pretul pe aceeasi parte cu EMA50</td></tr>
                <tr style="background:#0a1010"><td style="padding:3px 8px">ADX</td><td style="text-align:center;padding:3px 8px">+1</td>
                    <td style="padding:3px 8px;color:#888">ADX &gt; 20 (trend existent, nu lateralizare)</td></tr>
                <!-- Londra / NY -->
                <tr><td rowspan="3" style="padding:4px 8px;color:#ef5350;vertical-align:top;font-weight:600">🕰 London/NY Breakout</td>
                    <td style="padding:3px 8px">Asian range</td><td style="text-align:center;padding:3px 8px">+3</td>
                    <td style="padding:3px 8px;color:#888">Definit High/Low sesiunii asiatice (00:00–07:00 UTC)</td></tr>
                <tr><td style="padding:3px 8px">Breakout</td><td style="text-align:center;padding:3px 8px">+3</td>
                    <td style="padding:3px 8px;color:#888">Pretul a depasit High sau Low-ul asiatic cu body confirmat</td></tr>
                <tr><td style="padding:3px 8px">Session gate</td><td style="text-align:center;padding:3px 8px">+0/blocat</td>
                    <td style="padding:3px 8px;color:#888">Semnal valid doar in fereastra London (07:00–10:00) sau NY (12:00–15:00 UTC)</td></tr>
                <!-- RSI Divergenta -->
                <tr style="background:#0a1010"><td rowspan="3" style="padding:4px 8px;color:#66bb6a;vertical-align:top;font-weight:600">📉 RSI Divergenta</td>
                    <td style="padding:3px 8px">Divergenta</td><td style="text-align:center;padding:3px 8px">+4</td>
                    <td style="padding:3px 8px;color:#888">Pret face HH dar RSI face LH (bearish div) sau invers (bullish div)</td></tr>
                <tr style="background:#0a1010"><td style="padding:3px 8px">RSI zona</td><td style="text-align:center;padding:3px 8px">+2</td>
                    <td style="padding:3px 8px;color:#888">RSI &gt; 65 pentru divergenta bearish sau &lt; 35 pentru bullish</td></tr>
                <tr style="background:#0a1010"><td style="padding:3px 8px;color:#555;font-style:italic" colspan="3">Cel mai puternic semnal — divergenta + RSI extrem = semnal de reversal</td></tr>
                <!-- Engulfing -->
                <tr><td rowspan="2" style="padding:4px 8px;color:#ffa726;vertical-align:top;font-weight:600">🕯 Engulfing</td>
                    <td style="padding:3px 8px">Engulfing candle</td><td style="text-align:center;padding:3px 8px">+3</td>
                    <td style="padding:3px 8px;color:#888">O lumanare inglobeaza complet corpul precedentei (directie opusa)</td></tr>
                <tr><td style="padding:3px 8px">Key level</td><td style="text-align:center;padding:3px 8px">+2</td>
                    <td style="padding:3px 8px;color:#888">Engulfing apare la un nivel de suport/rezistenta important (pivot)</td></tr>
                <!-- Ichimoku -->
                <tr style="background:#0a1010"><td rowspan="3" style="padding:4px 8px;color:#4dd0e1;vertical-align:top;font-weight:600">☁ Ichimoku</td>
                    <td style="padding:3px 8px">TK Cross</td><td style="text-align:center;padding:3px 8px">+3</td>
                    <td style="padding:3px 8px;color:#888">Tenkan-sen a trecut Kijun-sen</td></tr>
                <tr style="background:#0a1010"><td style="padding:3px 8px">Kumo</td><td style="text-align:center;padding:3px 8px">+2</td>
                    <td style="padding:3px 8px;color:#888">Pretul deasupra/sub norul Kumo (confirma directia)</td></tr>
                <tr style="background:#0a1010"><td style="padding:3px 8px">Chikou</td><td style="text-align:center;padding:3px 8px">+1</td>
                    <td style="padding:3px 8px;color:#888">Chikou span deasupra/sub pret (26 perioade in urma)</td></tr>
                <!-- EMA Cross -->
                <tr><td rowspan="2" style="padding:4px 8px;color:#80cbc4;vertical-align:top;font-weight:600">✕ EMA Cross</td>
                    <td style="padding:3px 8px">EMA cross</td><td style="text-align:center;padding:3px 8px">+3</td>
                    <td style="padding:3px 8px;color:#888">EMA8 a trecut EMA21 (fast cross pentru intrari rapide)</td></tr>
                <tr><td style="padding:3px 8px">Momentum</td><td style="text-align:center;padding:3px 8px">+2</td>
                    <td style="padding:3px 8px;color:#888">ROC pozitiv in directia cross-ului</td></tr>
                </tbody>
                </table>
                <div style="margin-top:8px;font-size:0.76rem;color:#546e7a">
                    <b>Confidence %</b> = conviction × 8.5 + agreement bonus - conflict penalty &nbsp;|&nbsp;
                    <b>Semnal activ</b> daca Confidence ≥ prag configurat (default 66%) &nbsp;|&nbsp;
                    <b>HOLD</b> = fara setup pe acel TF (nu conteaza la vot)
                </div>
            </div>
        </details>
    </div>

    <!-- Decisions log -->
    <div class="decisions-section">
        <div style="display:flex;align-items:center;gap:10px;margin-bottom:8px">
            <h3 style="margin:0">Istoric decizii (ultimele 20)</h3>
            <span id="active-mode-badge" style="font-size:0.72rem;padding:2px 8px;border-radius:10px;background:#1a1a00;border:1px solid #555;color:#aaa">—</span>
        </div>
        <table class="dec-table">
            <thead><tr>
                <th>Timestamp</th><th>Simbol</th><th>Signal</th>
                <th>Strategie</th><th>Mod</th>
                <th>Incredere</th><th>Executat</th><th>Rezultat</th>
                <th>Analiza</th>
            </tr></thead>
            <tbody id="decisions-body"></tbody>
        </table>
    </div>

</div><!-- /main-content -->

<script>
const SYMBOLS_ALL    = {{ symbols_json }};
const SYMBOLS_CRYPTO = {{ crypto_json }};
const SYMBOLS_FOREX  = {{ forex_json }};
const TFS_ALL        = ["M1","M5","M15","M30","H1","H4","D1"];
const INTERVALS     = [1, 5, 10, 15, 20, 30];
const DEFAULT_BARS  = 500;

const CLASSIC_ELEMENTS = {ema:"EMA (trend aliniat)", fib:"FIB (Fibonacci zone)", adx:"ADX (forta trend)", rsi:"RSI (zona neutra)"};
const SMC_ELEMENTS     = {bos:"BOS (Break of Structure)", ob:"OB (Order Block)", fvg:"FVG (Fair Value Gap)", structure:"STR (Market Structure)"};

let selectedSymbols = new Set(SYMBOLS_ALL);
// State per strategie
let stratState = {
    classic: { tfs: new Set(["M15","H1","H4"]), tfBars: {M15:500,H1:500,H4:500,D1:500}, elements: {ema:true,fib:true,adx:true,rsi:true}, enabled: true, minConfidence: 66 },
    smc:     { tfs: new Set(["M5","M15","H1","H4"]), tfBars: {M5:500,M15:500,H1:500,H4:500}, elements: {bos:true,ob:true,fvg:true,structure:true}, enabled: true, minConfidence: 66 },
};

let currentSymbol   = null;
let currentStrategy = "classic";
let currentSignal   = null;
let currentChartTf  = null;
let pollTimer       = null;
let lastDecisionTs  = null;
let settingsOpen    = true;
let mt5SymbolsAll   = [];

// ── Settings toggle ────────────────────────────────────────────────────────
function toggleSettings() {
    settingsOpen = !settingsOpen;
    document.getElementById("settings-panel").classList.toggle("collapsed", !settingsOpen);
}

// ── Compact Settings Panel ─────────────────────────────────────────────────
let _csOpen = false;

function toggleCompactSettings() {
    _csOpen = !_csOpen;
    const panel = document.getElementById("compact-settings");
    panel.style.display = _csOpen ? "block" : "none";
    const btn = document.getElementById("btn-settings");
    btn.style.borderColor = _csOpen ? "#3949ab" : "#2a2a3e";
    btn.style.color        = _csOpen ? "#7986cb" : "";
    if (_csOpen) { csLoad(); csCfStatus(); }
}

// ── Telegram ────────────────────────────────────────────────────────────────
async function csLoad() {
    try {
        const r = await fetch("/telegram/config");
        const d = await r.json();
        document.getElementById("cs-tg-token").value    = d.bot_token  || "";
        document.getElementById("cs-tg-chatid").value   = d.chat_id    || "";
        document.getElementById("cs-tg-enabled").checked    = !!d.enabled;
        document.getElementById("cs-tg-approval").checked   = !!d.require_approval;
    } catch(e) {}
}

async function csTgSave() {
    const body = {
        bot_token:        document.getElementById("cs-tg-token").value.trim(),
        chat_id:          document.getElementById("cs-tg-chatid").value.trim(),
        enabled:          document.getElementById("cs-tg-enabled").checked,
        require_approval: document.getElementById("cs-tg-approval").checked,
    };
    const el = document.getElementById("cs-tg-msg");
    try {
        const r = await fetch("/telegram/config", { method:"POST",
            headers:{"Content-Type":"application/json"}, body:JSON.stringify(body) });
        const d = await r.json();
        el.style.color = d.ok ? "#66bb6a" : "#ef5350";
        el.textContent = d.ok ? "✓ Salvat" : (d.message || "Eroare");
    } catch(e) { el.style.color="#ef5350"; el.textContent="Eroare retea"; }
    setTimeout(() => el.textContent="", 3000);
}

async function csTgTest() {
    const el = document.getElementById("cs-tg-msg");
    el.style.color = "#ffeb3b"; el.textContent = "Se trimite...";
    try {
        const r = await fetch("/telegram/test", { method:"POST" });
        const d = await r.json();
        el.style.color = d.ok ? "#66bb6a" : "#ef5350";
        el.textContent = d.ok ? "✓ Mesaj trimis!" : (d.message || "Eroare");
    } catch(e) { el.style.color="#ef5350"; el.textContent="Eroare retea"; }
    setTimeout(() => el.textContent="", 4000);
}

// ── Cloudflare ──────────────────────────────────────────────────────────────
let _cfRunning = false;

async function csCfStatus() {
    try {
        const r = await fetch("/settings/cloudflare/status");
        const d = await r.json();
        _cfRunning = d.running;
        const dot  = document.getElementById("cs-cf-dot");
        const txt  = document.getElementById("cs-cf-text");
        const btn  = document.getElementById("cs-cf-btn");
        const urlD = document.getElementById("cs-cf-url");
        dot.className = "status-dot" + (d.running ? " on" : "");
        txt.textContent = d.running ? "Activ" : "Oprit";
        btn.textContent = d.running ? "■ Opreste" : "▶ Start";
        btn.style.borderColor = d.running ? "#6a2a2a" : "#2a6a2a";
        btn.style.background  = d.running ? "#3a1b1b" : "#1b3a1b";
        btn.style.color       = d.running ? "#ef5350" : "#66bb6a";
        if (d.url) {
            urlD.style.display = "block";
            const lnk = document.getElementById("cs-cf-url-link");
            lnk.href = d.url; lnk.textContent = d.url;
        } else {
            urlD.style.display = "none";
        }
    } catch(e) {}
}

async function csCfToggle() {
    const endpoint = _cfRunning ? "/settings/cloudflare/stop" : "/settings/cloudflare/start";
    await fetch(endpoint, { method:"POST" });
    setTimeout(csCfStatus, 1500);
}

function csCfCopy() {
    const url = document.getElementById("cs-cf-url-link").textContent;
    if (url) navigator.clipboard.writeText(url).catch(()=>{});
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
                    slMult: d.sl_atr_mult ?? null,   // null = foloseste globalul
                    riskUsd: d.risk_dollars ?? null, // null = foloseste globalul
                };
            }
        });
        buildStratTabs();
        buildStratList();
        buildStratDetail(_selectedStrat);
        syncTabsWithState();
    } catch(e) { console.warn("loadStratDefs:", e); }
}
let _selectedStrat = "classic";

let _allStratsActive = false;

function toggleAllStrategies() {
    _allStratsActive = !_allStratsActive;
    const body = {};
    STRAT_DEFS.forEach(({key}) => { body[key] = { enabled: _allStratsActive }; });
    fetch("/autotrader/set", {method:"POST", headers:{"Content-Type":"application/json"},
        body: JSON.stringify(body)}).then(() => {
        fetch("/autotrader/status").then(r => r.json()).then(data => {
            syncStratStateFromScanner(data.scanner);
            if (typeof buildStratList === "function") buildStratList();
            if (typeof syncTabsWithState === "function") syncTabsWithState();
            if (data.results) updateGrid(data.results);
        });
    });
    const btn = document.getElementById("toggle-all-strats-btn");
    if (btn) {
        btn.textContent = _allStratsActive ? "✗ Dezactiveaza toate" : "✓ Activeaza toate";
        btn.style.borderColor = _allStratsActive ? "#ef5350" : "#26a69a";
        btn.style.color       = _allStratsActive ? "#ef5350" : "#26a69a";
        btn.style.background  = _allStratsActive ? "#1f0d0d" : "#0d1f1d";
    }
    showToast(_allStratsActive ? "Toate strategiile activate" : "Toate strategiile dezactivate");
}

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
            <td style="text-align:right;padding:9px 12px">
                <label class="toggle" onclick="event.stopPropagation()" style="margin:0;vertical-align:middle">
                    <input type="checkbox" id="strat-toggle-${key}" ${enabled ? "checked" : ""}
                        onchange="toggleStratEnabled('${key}', this.checked)">
                    <span class="slider"></span>
                </label>
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
            <div class="config-row" style="margin-bottom:6px">
                <label>Confidence min<span class="sub">% minim semnal</span></label>
                <input type="number" id="${strat}-confidence" value="${st.minConfidence||66}" min="0" max="100" step="1" style="width:55px"
                       onchange="stratState['${strat}'].minConfidence=parseFloat(this.value);sendStrategySet('${strat}','min_confidence',parseFloat(this.value))">
            </div>
            <div class="config-row" style="margin-bottom:6px">
                <label>SL Multiplier<span class="sub">ATR × (gol = global)</span></label>
                <input type="number" id="${strat}-sl-mult" value="${st.slMult||''}" min="0.3" max="5" step="0.1"
                       placeholder="global" style="width:65px;background:#1e1e1e;color:#ab47bc;border:1px solid #333;border-radius:4px;padding:3px 6px;font-size:0.82rem"
                       onchange="const v=parseFloat(this.value)||null;stratState['${strat}'].slMult=v;sendStrategySet('${strat}','sl_atr_mult',v)">
            </div>
            <div class="config-row" style="margin-bottom:10px">
                <label>Risk per trade<span class="sub">USD (gol = global $${Math.round(window._globalRiskUsd||50)})</span></label>
                <input type="number" id="${strat}-risk-usd" value="${st.riskUsd||''}" min="1" max="10000" step="1"
                       placeholder="global" style="width:65px;background:#1e1e1e;color:#26c6da;border:1px solid #333;border-radius:4px;padding:3px 6px;font-size:0.82rem"
                       onchange="const v=parseFloat(this.value)||null;stratState['${strat}'].riskUsd=v;sendStrategySet('${strat}','risk_dollars',v)">
            </div>
            <div style="font-size:0.72rem;color:#555;font-weight:600;letter-spacing:0.5px;margin-bottom:5px">TIMEFRAME-URI (strategia alege cel mai bun automat)</div>
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
    STRAT_DEFS.forEach(({key}) => {
        const gid  = `symbol-grid-${key}`;
        const grid = document.getElementById(gid);
        if (grid && !grid.querySelector(`[data-sym="${sym}"]`)) {
            const card = document.createElement("div");
            card.className = "sym-card sig-hold"; card.dataset.sym = sym;
            card.innerHTML = `<div class="card-name">${sym}</div><div class="card-scanning">astept scanare...</div>`;
            card.onclick = () => selectCard(sym, null, key);
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
    const sec = INTERVALS[parseInt(el.value)];
    document.getElementById("interval-label").textContent = sec + "s";
    sendGlobal("interval", sec);
    try { localStorage.setItem("at_interval_idx", el.value); } catch(e){}
}

function sendGlobal(key, value) {
    fetch("/autotrader/set", {method:"POST", headers:{"Content-Type":"application/json"},
          body: JSON.stringify({[key]: value})});
}

// FTMO 10k optimized presets
// Rules: daily loss $500 (5%), max DD $1000 (10%), conservative risk
const PRESETS = {
    scalp: {
        label: "⚡ Scalp",
        sl_atr_mult: 1.0, max_lot_global: 0.5, tp_ratio: 1.0,
        risk_dollars: 30, max_open_trades: 3,
        // Toate strategiile rapide pe M5 + M15 pentru mai multe semnale
        enabled: ["supertrend", "ema_cross", "vwap_bounce", "london_breakout",
                  "ny_breakout", "china_session", "engulfing", "bollinger",
                  "keltner_channel", "candleforge", "smc"],
        disabled: ["classic", "macd", "ichimoku", "rsi_divergence", "eob"],
        tfs: {
            supertrend:      ["M5", "M15"],
            ema_cross:       ["M5", "M15"],
            vwap_bounce:     ["M5", "M15"],
            london_breakout: ["M5", "M15"],
            ny_breakout:     ["M5", "M15"],
            china_session:   ["M5", "M15"],
            engulfing:       ["M5", "M15"],
            bollinger:       ["M5", "M15"],
            keltner_channel: ["M5", "M15"],
            candleforge:     ["M5", "M15"],
            smc:             ["M5", "M15"],
        },
    },
    intraday: {
        label: "📈 Intraday",
        sl_atr_mult: 1.5, max_lot_global: 1.0, tp_ratio: 1.5,
        risk_dollars: 50, max_open_trades: 3,
        // Strategii echilibrate, FTMO recomandate
        enabled: ["classic", "smc", "supertrend", "ema_cross", "macd",
                  "bollinger", "engulfing", "candleforge", "eob"],
        disabled: ["ichimoku", "london_breakout", "ny_breakout", "china_session",
                   "vwap_bounce", "rsi_divergence", "keltner_channel"],
        tfs: {
            classic:     ["M15", "H1"],
            smc:         ["M15", "H1"],
            supertrend:  ["M15", "H1"],
            ema_cross:   ["M15", "H1"],
            macd:        ["H1", "H4"],
            bollinger:   ["M15", "H1"],
            engulfing:   ["M15", "H1"],
            candleforge: ["M15", "H1"],
            eob:         ["M15", "H1"],
        },
    },
    swing: {
        label: "🌊 Swing",
        sl_atr_mult: 2.0, max_lot_global: 1.5, tp_ratio: 2.0,
        risk_dollars: 100, max_open_trades: 4,
        enabled: ["classic", "smc", "ichimoku", "macd", "ema_cross",
                  "rsi_divergence", "engulfing"],
        disabled: ["supertrend", "bollinger", "london_breakout", "ny_breakout",
                   "china_session", "vwap_bounce", "keltner_channel",
                   "candleforge", "eob"],
        tfs: {
            classic:        ["H1", "H4"],
            smc:            ["H1", "H4"],
            ichimoku:       ["H4", "D1"],
            macd:           ["H4", "D1"],
            ema_cross:      ["H1", "H4"],
            rsi_divergence: ["H4", "D1"],
            engulfing:      ["H1", "H4"],
        },
    },
    sniper: {
        label: "🎯 Sniper",
        sl_atr_mult: 1.5, max_lot_global: 1.0, tp_ratio: 2.0,
        risk_dollars: 50, max_open_trades: 1,
        // Doar 3 strategii sigure, confidence 85%+, rar dar precis
        enabled: ["classic", "smc", "engulfing"],
        disabled: ["supertrend", "ema_cross", "macd", "ichimoku", "bollinger",
                   "rsi_divergence", "keltner_channel", "candleforge", "eob",
                   "london_breakout", "ny_breakout", "china_session", "vwap_bounce"],
        tfs:         { classic: ["H1"], smc: ["H1"], engulfing: ["H1"] },
        min_conf:    { classic: 85.0, smc: 85.0, engulfing: 85.0 },
    },
    session: {
        label: "🏙 Session",
        sl_atr_mult: 1.2, max_lot_global: 1.0, tp_ratio: 1.5,
        risk_dollars: 50, max_open_trades: 2,
        // Exclusiv breakout-uri de sesiune
        enabled: ["london_breakout", "ny_breakout", "china_session"],
        disabled: ["classic", "smc", "supertrend", "ema_cross", "macd", "ichimoku",
                   "bollinger", "engulfing", "rsi_divergence", "keltner_channel",
                   "candleforge", "eob", "vwap_bounce"],
        tfs: {
            london_breakout: ["M15"],
            ny_breakout:     ["M15"],
            china_session:   ["M15"],
        },
    },
    reversal: {
        label: "🐻 Reversal",
        sl_atr_mult: 1.5, max_lot_global: 0.5, tp_ratio: 2.0,
        risk_dollars: 40, max_open_trades: 2,
        // Contra-trend: divergente + pin bars
        enabled: ["rsi_divergence", "engulfing", "bollinger"],
        disabled: ["classic", "smc", "supertrend", "ema_cross", "macd", "ichimoku",
                   "keltner_channel", "candleforge", "eob", "london_breakout",
                   "ny_breakout", "china_session", "vwap_bounce"],
        tfs: {
            rsi_divergence: ["M15", "H1"],
            engulfing:      ["M15", "H1"],
            bollinger:      ["M15", "H1"],
        },
    },
    conservative: {
        label: "💎 Conservative",
        sl_atr_mult: 2.0, max_lot_global: 0.5, tp_ratio: 2.0,
        risk_dollars: 30, max_open_trades: 2,
        // FTMO-safest: 3 strategii premium, min_confidence 80%
        enabled: ["classic", "macd", "smc"],
        disabled: ["supertrend", "ema_cross", "ichimoku", "bollinger", "engulfing",
                   "rsi_divergence", "keltner_channel", "candleforge", "eob",
                   "london_breakout", "ny_breakout", "china_session", "vwap_bounce"],
        tfs:      { classic: ["H1", "H4"], macd: ["H1", "H4"], smc: ["H1", "H4"] },
        min_conf: { classic: 80.0, macd: 80.0, smc: 80.0 },
    },
    aggressive: {
        label: "🚀 Aggressive",
        sl_atr_mult: 1.5, max_lot_global: 2.0, tp_ratio: 1.5,
        risk_dollars: 100, max_open_trades: 5,
        combined_mode: true,
        enabled: ["classic", "smc", "supertrend", "ema_cross", "macd", "bollinger",
                  "ichimoku", "engulfing", "rsi_divergence", "keltner_channel",
                  "candleforge", "eob", "london_breakout", "ny_breakout",
                  "china_session", "vwap_bounce"],
        disabled: [],
        tfs: {
            classic:         ["M15", "H1"],
            smc:             ["M15", "H1"],
            supertrend:      ["M15", "H1"],
            ema_cross:       ["M15", "H1"],
            macd:            ["H1", "H4"],
            bollinger:       ["M15", "H1"],
            ichimoku:        ["H4", "D1"],
            engulfing:       ["M15", "H1"],
            rsi_divergence:  ["H1", "H4"],
            keltner_channel: ["M15", "H1"],
            candleforge:     ["M15", "H1"],
            eob:             ["M15", "H1"],
            london_breakout: ["M15"],
            ny_breakout:     ["M15"],
            china_session:   ["M15"],
            vwap_bounce:     ["M5", "M15"],
        },
    },
    scalp_boost: {
        label: "⚡ Scalp Boost",
        sl_atr_mult: 1.0, max_lot_global: 1.0, tp_ratio: 1.0,
        risk_dollars: 30, max_open_trades: 3,
        scalp_boost: true,
        enabled: ["supertrend", "ema_cross", "vwap_bounce", "london_breakout",
                  "ny_breakout", "china_session", "engulfing", "bollinger",
                  "keltner_channel", "candleforge", "smc"],
        disabled: ["classic", "macd", "ichimoku", "rsi_divergence", "eob"],
        tfs: {
            supertrend:      ["M5", "M15"],
            ema_cross:       ["M5", "M15"],
            vwap_bounce:     ["M5", "M15"],
            london_breakout: ["M5", "M15"],
            ny_breakout:     ["M5", "M15"],
            china_session:   ["M5", "M15"],
            engulfing:       ["M5", "M15"],
            bollinger:       ["M5", "M15"],
            keltner_channel: ["M5", "M15"],
            candleforge:     ["M5", "M15"],
            smc:             ["M5", "M15"],
        },
    },
};

let _activePreset = null;

function _highlightPresets(activeName) {
    Object.keys(PRESETS).forEach(k => {
        const btn = document.getElementById("preset-"+k);
        if (!btn) return;
        const isActive = k === activeName;
        btn.style.borderColor = isActive ? "#ffeb3b" : (k === "scalp_boost" ? "#3a2200" : "#2a2a2a");
        btn.style.color       = isActive ? "#ffeb3b" : (k === "scalp_boost" ? "#ff9800" : "#ccc");
        btn.style.background  = isActive ? "#1a1a00" : (k === "scalp_boost" ? "#1a0f00" : "#141414");
    });
}

function applyPreset(name) {
    const p = PRESETS[name];
    if (!p) return;

    // Toggle: daca acelasi preset e deja activ, dezactiveaza-l
    if (_activePreset === name) {
        _activePreset = null;
        const body = {};
        (p.enabled || []).forEach(k => { body[k] = { enabled: false }; });
        if (p.scalp_boost !== undefined) body.scalp_boost = false;
        if (p.combined_mode !== undefined) body.combined_mode = false;
        fetch("/autotrader/set", {method:"POST", headers:{"Content-Type":"application/json"},
            body: JSON.stringify(body)}).then(() => {
            fetch("/autotrader/status").then(r => r.json()).then(data => {
                syncStratStateFromScanner(data.scanner);
                if (typeof buildStratList === "function") buildStratList();
                if (typeof syncTabsWithState === "function") syncTabsWithState();
                if (data.results) updateGrid(data.results);
            });
        });
        _highlightPresets(null);
        showToast(`Preset ${p.label} dezactivat`);
        return;
    }

    _activePreset = name;

    // Compose single body with all strategy configs
    const body = {
        sl_atr_mult:     p.sl_atr_mult,
        max_lot_global:  p.max_lot_global,
        tp_ratio:        p.tp_ratio,
        risk_dollars:    p.risk_dollars,
        max_open_trades: p.max_open_trades,
    };
    if (p.combined_mode !== undefined) body.combined_mode = p.combined_mode;
    if (p.scalp_boost   !== undefined) body.scalp_boost   = p.scalp_boost;
    // Activeaza strategii + TF-uri + min_confidence
    (p.enabled || []).forEach(k => {
        const cfg = { enabled: true };
        if (p.tfs && p.tfs[k]) cfg.tfs = p.tfs[k];
        if (p.min_conf && p.min_conf[k] != null) cfg.min_confidence = p.min_conf[k];
        body[k] = cfg;
    });
    (p.disabled || []).forEach(k => {
        body[k] = { enabled: false };
    });

    fetch("/autotrader/set", {method:"POST", headers:{"Content-Type":"application/json"},
        body: JSON.stringify(body)}).then(() => {
        // Reload UI state — sincronizeaza stratState din scanner + rebuild UI
        fetch("/autotrader/status").then(r => r.json()).then(data => {
            syncStratStateFromScanner(data.scanner);
            if (typeof buildStratList === "function") buildStratList();
            if (typeof syncTabsWithState === "function") syncTabsWithState();
            if (data.results) updateGrid(data.results);
        });
    });

    // Update UI inputs
    document.getElementById("sl-mult-range").value = p.sl_atr_mult;
    document.getElementById("sl-mult-label").textContent = p.sl_atr_mult.toFixed(1);
    document.getElementById("max-lot-input").value = p.max_lot_global.toFixed(2);
    document.getElementById("tp-ratio-input").value = p.tp_ratio.toFixed(1);
    document.getElementById("risk-dollars-input").value = p.risk_dollars;
    document.getElementById("max-trades-input").value = p.max_open_trades;

    _highlightPresets(name);
    // Sync combined-mode UI toggle if preset sets it
    if (p.combined_mode !== undefined) {
        const cm = document.getElementById("combined-mode-toggle");
        if (cm) { cm.checked = !!p.combined_mode; toggleCombinedPanel(!!p.combined_mode); }
    }
    // Sync scalp boost UI if preset sets it
    if (p.scalp_boost !== undefined) toggleScalpBoost(!!p.scalp_boost, false);
    showToast(`Preset ${p.label} aplicat — ${p.enabled.length} strategii active`);
}

function setRiskMode(usePct) {
    sendGlobal("use_risk_pct", usePct);
    document.getElementById("risk-usd-row").style.display = usePct ? "none" : "flex";
    document.getElementById("risk-pct-row").style.display = usePct ? "flex" : "none";
    document.getElementById("risk-mode-usd").style.borderColor = usePct ? "#333" : "#ffeb3b";
    document.getElementById("risk-mode-usd").style.color       = usePct ? "#666" : "#ffeb3b";
    document.getElementById("risk-mode-pct").style.borderColor = usePct ? "#ffeb3b" : "#333";
    document.getElementById("risk-mode-pct").style.color       = usePct ? "#ffeb3b" : "#666";
}

function onSlMultChange(el) {
    const v = parseFloat(el.value);
    document.getElementById("sl-mult-label").textContent = v.toFixed(1);
    sendGlobal("sl_atr_mult", v);
}

const SESSION_STRATEGIES = ["london_breakout", "ny_breakout", "china_session", "vwap_bounce"];

function togglePendingPanel(enabled) {
    const panel = document.getElementById("pending-panel");
    if (panel) panel.style.display = enabled ? "block" : "none";
    if (enabled) refreshPendingOrders();
}

async function refreshPendingOrders() {
    const container = document.getElementById("pending-orders-container");
    if (!container) return;
    container.innerHTML = '<span style="color:#555">Se incarca...</span>';
    try {
        const r = await fetch("/autotrader/pending_orders");
        const d = await r.json();
        if (!d.ok) {
            container.innerHTML = `<span style="color:#ef5350">${d.message || "Eroare"}</span>`;
            return;
        }
        const orders = d.orders || [];
        if (!orders.length) {
            container.innerHTML = '<span style="color:#444">Niciun ordin pending activ.</span>';
            return;
        }
        let html = `<table style="width:100%;border-collapse:collapse;font-size:0.76rem">
            <thead><tr style="color:#444;border-bottom:1px solid #1e1e2e">
                <th style="padding:5px 10px;text-align:left">Ticket</th>
                <th style="padding:5px 8px;text-align:left">Simbol</th>
                <th style="padding:5px 8px;text-align:center">Tip</th>
                <th style="padding:5px 8px;text-align:right">Pret intrare</th>
                <th style="padding:5px 8px;text-align:right">SL</th>
                <th style="padding:5px 8px;text-align:right">TP</th>
                <th style="padding:5px 8px;text-align:right">Loturi</th>
                <th style="padding:5px 8px;text-align:center">Actiune</th>
            </tr></thead><tbody>`;
        orders.forEach(o => {
            const isBuy = o.type.startsWith("BUY");
            const clr = isBuy ? "#26a69a" : "#ef5350";
            html += `<tr style="border-bottom:1px solid #1a1a2a">
                <td style="padding:6px 10px;color:#555;font-size:0.7rem">#${o.ticket}</td>
                <td style="padding:6px 8px;font-weight:700;color:#eee">${o.symbol}</td>
                <td style="padding:6px 8px;text-align:center">
                    <span style="color:${clr};font-weight:700;font-size:0.78rem">${o.type}</span>
                </td>
                <td style="padding:6px 8px;text-align:right;color:#eee;font-family:monospace">${o.price}</td>
                <td style="padding:6px 8px;text-align:right;color:#ef5350;font-family:monospace">${o.sl || "—"}</td>
                <td style="padding:6px 8px;text-align:right;color:#26a69a;font-family:monospace">${o.tp || "—"}</td>
                <td style="padding:6px 8px;text-align:right;color:#888">${o.volume}</td>
                <td style="padding:6px 8px;text-align:center">
                    <button onclick="cancelPending(${o.ticket})"
                        style="background:#2a0a0a;border:1px solid #ef5350;color:#ef5350;padding:2px 10px;border-radius:4px;font-size:0.7rem;cursor:pointer">
                        ✕ Anuleaza
                    </button>
                </td>
            </tr>`;
        });
        html += '</tbody></table>';
        container.innerHTML = html;
    } catch(e) {
        container.innerHTML = `<span style="color:#ef5350">Eroare: ${e.message}</span>`;
    }
}

async function cancelPending(ticket) {
    if (!confirm(`Anulezi ordinul pending #${ticket}?`)) return;
    try {
        const r = await fetch("/autotrader/cancel_pending", {
            method: "POST",
            headers: {"Content-Type": "application/json"},
            body: JSON.stringify({ticket})
        });
        const d = await r.json();
        showToast(d.message || (d.ok ? "Anulat" : "Eroare"));
        refreshPendingOrders();
    } catch(e) {
        showToast("Eroare: " + e.message);
    }
}

// ── Scalp Boost ──────────────────────────────────────────────────────────────
function toggleScalpBoost(enabled, sendUpdate = true) {
    if (sendUpdate) sendGlobal("scalp_boost", enabled);
    const btn = document.getElementById("preset-scalp_boost");
    if (btn) {
        btn.style.borderColor = enabled ? "#ffeb3b" : "#3a2200";
        btn.style.color       = enabled ? "#ffeb3b" : "#ff9800";
        btn.style.background  = enabled ? "#1a1a00" : "#1a0f00";
    }
    if (enabled && _activePreset !== "scalp_boost") _activePreset = "scalp_boost";
    if (!enabled && _activePreset === "scalp_boost") _activePreset = null;
}

async function refreshScalpBoostVotes() {
    try {
        const r = await fetch("/autotrader/status");
        const d = await r.json();
        const dec = (d.decisions || []).filter(e => e.strategy === "scalp_boost");
        const sc  = d.scanner || {};

        // Update badges
        const lb = document.getElementById("sb-lot-mult-badge");
        const tb = document.getElementById("sb-tp-rr-badge");
        const mb = document.getElementById("sb-min-agree-badge");
        if (lb) lb.textContent = sc.scalp_lot_mult || 2;
        if (tb) tb.textContent = sc.scalp_tp_rr    || 1;
        if (mb) mb.textContent = sc.scalp_min_agree|| 3;

        // Voturi per simbol din results
        const res = d.results || {};
        const rows = [];
        for (const [sym, symRes] of Object.entries(res)) {
            const votes = {BUY:[], SELL:[], HOLD:[]};
            for (const [sk, sr] of Object.entries(symRes)) {
                if (sk.startsWith("_")) continue;
                const sig = sr.signal || "HOLD";
                if (votes[sig]) votes[sig].push(sk);
            }
            const tot = votes.BUY.length + votes.SELL.length + votes.HOLD.length;
            if (!tot) continue;
            const dominant = votes.BUY.length > votes.SELL.length ? "BUY" : votes.SELL.length > votes.BUY.length ? "SELL" : "—";
            const minAgree = sc.scalp_min_agree || 3;
            const ready    = Math.max(votes.BUY.length, votes.SELL.length) >= minAgree;
            rows.push(`<div style="display:flex;align-items:center;gap:8px;padding:4px 0;border-bottom:1px solid #2a1800">
                <span style="color:#ccc;font-weight:600;min-width:80px">${sym}</span>
                <span style="color:${votes.BUY.length>0?'#26a69a':'#444'};font-size:0.75rem">BUY ${votes.BUY.length}</span>
                <span style="color:${votes.SELL.length>0?'#ef5350':'#444'};font-size:0.75rem">SELL ${votes.SELL.length}</span>
                <span style="color:#555;font-size:0.75rem">HOLD ${votes.HOLD.length}</span>
                ${ready ? `<span style="background:${dominant==='BUY'?'#1b3a2e':'#3a1b1b'};color:${dominant==='BUY'?'#26a69a':'#ef5350'};padding:1px 8px;border-radius:10px;font-size:0.72rem;font-weight:700">${dominant} READY ⚡</span>` : ''}
            </div>`);
        }
        const c = document.getElementById("scalp-boost-votes");
        if (c) c.innerHTML = rows.length ? rows.join("") : '<div style="color:#555">Niciun simbol cu voturi suficiente...</div>';

        // Action log
        const logEl = document.getElementById("scalp-action-log");
        if (logEl) {
            logEl.innerHTML = dec.length ? dec.slice(0,10).map(e => {
                const col = e.executed ? "#26a69a" : "#666";
                const sigCol = e.signal === "BUY" ? "#26a69a" : e.signal === "SELL" ? "#ef5350" : "#888";
                return `<div style="padding:3px 0;border-bottom:1px solid #1e1200;display:flex;gap:8px;align-items:center">
                    <span style="color:#444;font-size:0.67rem">${(e.timestamp||"").slice(11,16)}</span>
                    <span style="color:#888;min-width:70px">${e.symbol||"—"}</span>
                    <span style="color:${sigCol};font-weight:600;font-size:0.75rem">${e.signal}</span>
                    <span style="color:#7a5000;font-size:0.7rem">×${e.lot_mult||2}</span>
                    <span style="color:${col};font-size:0.72rem">${e.executed?"✓ Executat":"✗"}</span>
                    <span style="color:#444;font-size:0.68rem;margin-left:auto">${(e.voters||[]).join(",")||""}</span>
                </div>`;
            }).join("") : '<div style="color:#555;font-size:0.75rem">Niciun trade scalp inca...</div>';
        }
    } catch(e) { console.warn("refreshScalpBoostVotes:", e); }
}

// Auto-refresh scalp votes la 30s cand scalp boost e activ
setInterval(() => {
    if (_activePreset === "scalp_boost") refreshScalpBoostVotes();
}, 30000);

function toggleCombinedPanel(enabled) {
    const panel = document.getElementById("combined-panel");
    if (!panel) return;
    panel.style.display = enabled ? "block" : "none";
    // Ascunde tab-urile individuale cand combined mode e activ
    const tabsBar = document.getElementById("strat-tabs-bar");
    const stratSections = document.getElementById("strat-sections");
    if (tabsBar) tabsBar.style.opacity = enabled ? "0.35" : "1";
    if (stratSections) stratSections.style.opacity = enabled ? "0.35" : "1";

    // Dezactiveaza/reactiveaza strategiile de sesiune in UI
    if (enabled) {
        SESSION_STRATEGIES.forEach(sk => {
            const chk = document.getElementById(`strat-enabled-${sk}`);
            if (chk && chk.checked) {
                chk.checked = false;
                chk.dispatchEvent(new Event("change"));
            }
        });
        refreshCombinedVotes();
    }
}

async function refreshCombinedVotes() {
    try {
        const r = await fetch("/autotrader/status");
        const d = await r.json();
        const res = d.results || {};
        const container = document.getElementById("combined-votes-container");
        if (!container) return;

        // Filtreaza simbolurile care au _combined_votes
        const syms = Object.keys(res).filter(s => {
            const r = res[s];
            return r && typeof r === "object" && r._combined_votes;
        });

        if (!syms.length) {
            container.innerHTML = '<div style="color:#555;padding:10px">Niciun rezultat inca. Asigura-te ca scannerul ruleaza si Combined Mode e activ.</div>';
            return;
        }

        function badge(sig) {
            const c = sig==="BUY"?"#26a69a":sig==="SELL"?"#ef5350":"#666";
            const bg = sig==="BUY"?"#0d2a1a":sig==="SELL"?"#2a0d0d":"#1a1a1a";
            return `<span style="background:${bg};color:${c};padding:2px 10px;border-radius:4px;font-weight:700;font-size:0.78rem;letter-spacing:0.5px">${sig}</span>`;
        }

        let html = '';

        for (const sym of syms.sort()) {
            const v = res[sym]._combined_votes;
            const cSig    = v.signal;
            const total   = v.total   || 1;
            const nBuy    = v.n_buy   || 0;
            const nSell   = v.n_sell  || 0;
            const nHold   = v.n_hold  || 0;
            const buyScore  = v.buy_score  || 0;
            const sellScore = v.sell_score || 0;
            const pctBuy   = v.pct_buy  || Math.round(nBuy  / total * 100);
            const pctSell  = v.pct_sell || Math.round(nSell / total * 100);
            const pctHold  = 100 - pctBuy - pctSell;
            // Procente ponderate (bazate pe conviction sum)
            const activeScore = buyScore + sellScore;
            const pctBuyW  = v.pct_buy_w  || (activeScore > 0 ? Math.round(buyScore  / activeScore * 100) : 0);
            const pctSellW = v.pct_sell_w || (activeScore > 0 ? Math.round(sellScore / activeScore * 100) : 0);
            const bClr = cSig==="BUY"?"#1b5e20":cSig==="SELL"?"#b71c1c":"#2a2a2a";
            const hBg  = cSig==="BUY"?"#0a1f0a":cSig==="SELL"?"#1f0a0a":"#141414";

            const strats = v.strategies || {};
            const stratKeys = Object.keys(strats);
            const pctBuyFlat  = v.buy_pct_flat  ?? Math.round(nBuy  / total * 100);
            const pctSellFlat = v.sell_pct_flat ?? Math.round(nSell / total * 100);
            const spreadPct   = v.spread_pct    ?? Math.abs(pctBuyFlat - pctSellFlat);
            const domPct      = Math.max(pctBuyFlat, pctSellFlat);
            const ok60        = domPct >= 60;
            const ok21        = spreadPct >= 21;
            const threshBuy60 = `<span style="color:${ok60?'#66bb6a':'#555'}">${ok60?'✓':'✗'} ≥60% (${domPct}%)</span>`;
            const threshSpread = `<span style="color:${ok21?'#66bb6a':'#555'}">${ok21?'✓':'✗'} spread≥21% (${spreadPct}%)</span>`;

            html += `<div style="background:#111;border:1px solid ${bClr};border-radius:8px;margin-bottom:14px;overflow:hidden">
                <div style="background:${hBg};padding:10px 16px;display:flex;align-items:center;gap:14px;border-bottom:1px solid ${bClr}">
                    <span style="font-weight:700;font-size:0.95rem;color:#eee;min-width:100px">${sym}</span>
                    <div style="flex:1;display:flex;flex-direction:column;gap:3px">
                        <div style="display:flex;height:9px;border-radius:3px;overflow:hidden;background:#0a0a0a" title="Voturi brute: BUY / SELL / HOLD">
                            <div style="width:${pctBuy}%;background:#26a69a;transition:width 0.3s" title="BUY ${pctBuy}%"></div>
                            <div style="width:${pctSell}%;background:#ef5350;transition:width 0.3s" title="SELL ${pctSell}%"></div>
                            <div style="width:${Math.max(0,pctHold)}%;background:#888;transition:width 0.3s" title="HOLD ${pctHold}%"></div>
                        </div>
                        <div style="display:flex;height:6px;border-radius:3px;overflow:hidden;background:#0a0a0a;opacity:0.55" title="Scor ponderat conviction (BUY/SELL)">
                            <div style="width:${pctBuyW}%;background:#26a69a;transition:width 0.3s"></div>
                            <div style="width:${pctSellW}%;background:#ef5350;transition:width 0.3s"></div>
                        </div>
                    </div>
                    <div style="display:flex;flex-direction:column;gap:3px;font-size:0.74rem;white-space:nowrap">
                        <div style="display:flex;gap:10px">
                            <span style="color:#26a69a;font-weight:600">BUY ${nBuy}/${total} <span style="opacity:0.6">(${pctBuyFlat}%)</span></span>
                            <span style="color:#ef5350;font-weight:600">SELL ${nSell}/${total} <span style="opacity:0.6">(${pctSellFlat}%)</span></span>
                            <span style="color:#555">HOLD ${nHold}</span>
                        </div>
                        <div style="display:flex;gap:8px;font-size:0.68rem">
                            ${threshBuy60}
                            ${threshSpread}
                        </div>
                    </div>
                    <div style="display:flex;flex-direction:column;align-items:center;gap:4px;margin-left:6px">
                        ${badge(cSig)}
                        ${(()=>{
                            const crs = v.combined_risk_score || 0;
                            if (!crs) return "";
                            const cc = crs >= 70 ? "#66bb6a" : crs >= 50 ? "#ffeb3b" : "#ef5350";
                            return `<span style="font-size:0.68rem;font-weight:700;color:${cc};border:1px solid ${cc};border-radius:3px;padding:1px 5px">RISC ${crs.toFixed(0)}</span>`;
                        })()}
                    </div>
                </div>`;

            const tfRows = v.tf_rows || [];
            if (tfRows.length) {
                html += `<table style="width:100%;border-collapse:collapse;font-size:0.76rem">
                    <thead><tr style="color:#444;border-bottom:1px solid #1e1e1e">
                        <th style="padding:6px 16px;text-align:left;font-weight:400;width:160px">Strategie</th>
                        <th style="padding:6px 8px;text-align:center;font-weight:400;width:55px">TF</th>
                        <th style="padding:6px 8px;text-align:center;font-weight:400;width:80px">Semnal</th>
                        <th style="padding:6px 8px;text-align:center;font-weight:400;width:50px">Conv.</th>
                    </tr></thead><tbody>`;

                // Grupeaza per strategie pentru aliniere vizuala
                let lastStrat = null;
                for (const row of tfRows) {
                    const sig  = row.signal || "HOLD";
                    const tf   = row.tf || "—";
                    const conv = row.conviction || 0;
                    const sk   = row.strat || "";
                    const rowBg = sig==="BUY"?"#091509":sig==="SELL"?"#150909":"transparent";
                    const sigClr = sig==="BUY"?"#26a69a":sig==="SELL"?"#ef5350":"#444";

                    const def   = STRAT_DEFS.find(d => d.key === sk) || {};
                    const icon  = def.icon  || "◆";
                    const color = def.color || "#888";
                    const name  = def.name  || sk;

                    const isNewStrat = sk !== lastStrat;
                    lastStrat = sk;

                    const stratCell = isNewStrat
                        ? `<td style="padding:7px 16px;color:${color};font-weight:600;white-space:nowrap;border-top:1px solid #1e1e1e">${icon} ${name}</td>`
                        : `<td style="padding:3px 16px;color:#333;border-top:none">↳</td>`;

                    const borderTop = isNewStrat ? "border-top:1px solid #1e1e1e;" : "";
                    const stars = conv > 0 ? "★".repeat(Math.min(conv, 5)) + (conv > 5 ? `+${conv-5}` : "") : "—";

                    html += `<tr style="background:${rowBg}">
                        ${stratCell}
                        <td style="padding:5px 8px;text-align:center;color:#777;font-family:monospace;${borderTop}">${tf}</td>
                        <td style="padding:5px 8px;text-align:center;${borderTop}">
                            <span style="color:${sigClr};font-weight:700;font-size:0.8rem">${sig}</span>
                        </td>
                        <td style="padding:5px 8px;text-align:center;color:${sigClr};font-size:0.7rem;${borderTop}">${stars}</td>
                    </tr>`;
                }
                html += '</tbody></table>';
            } else {
                html += '<div style="padding:10px 16px;color:#444;font-size:0.75rem">Nicio strategie activa</div>';
            }

            html += '</div>';
        }

        container.innerHTML = html;
        refreshCombinedLog();
    } catch(e) {
        console.warn("refreshCombinedVotes error:", e);
        container.innerHTML = '<div style="color:#ef5350;padding:10px">Eroare: ' + e.message + '</div>';
    }
}

async function refreshCombinedLog() {
    const el = document.getElementById("combined-action-log");
    if (!el) return;
    try {
        const r = await fetch("/autotrader/combined_log?n=40");
        const entries = await r.json();
        if (!entries.length) {
            el.innerHTML = '<div style="color:#333;padding:6px">Nicio actiune inca.</div>';
            return;
        }
        let html = '';
        for (const e of entries) {
            const ts = e.timestamp ? e.timestamp.replace("T", " ").slice(0,19) : "—";
            const sig = e.signal || "?";
            const sigClr = sig === "BUY" ? "#26a69a" : sig === "SELL" ? "#ef5350" : "#555";
            const ok = e.executed;
            const vm = e.vote_map || {};
            const voters = e.voters ? e.voters.join(", ") : "";
            const resultClr = ok ? "#66bb6a" : "#ef535088";
            const icon = ok ? "✅" : "❌";
            const vmStr = vm.total ? `BUY ${vm.buy||0}/${vm.total} · SELL ${vm.sell||0}/${vm.total} · HOLD ${vm.hold||0}/${vm.total}` : "";
            html += `<div style="border-bottom:1px solid #1a1a1a;padding:7px 6px;display:flex;flex-direction:column;gap:2px">
                <div style="display:flex;align-items:center;gap:8px">
                    <span style="color:#333;font-family:monospace;font-size:0.7rem">${ts}</span>
                    <span style="font-weight:700;color:${sigClr};font-size:0.78rem">${e.symbol || "?"} ${sig}</span>
                    <span style="margin-left:auto">${icon}</span>
                </div>
                <div style="color:#555;font-size:0.7rem">${e.result || "—"}</div>
                ${vmStr ? `<div style="color:#444;font-size:0.68rem">${vmStr}</div>` : ""}
                ${voters ? `<div style="color:#3a5a3a;font-size:0.67rem">Votanti: ${voters}</div>` : ""}
            </div>`;
        }
        el.innerHTML = html;
    } catch(e) {
        el.innerHTML = '<div style="color:#ef5350">Eroare: ' + e.message + '</div>';
    }
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
    };
    // Trimite TOATE strategiile, nu doar classic + smc
    STRAT_DEFS.forEach(({key}) => {
        const st = stratState[key];
        if (st) {
            body[key] = {
                enabled:  st.enabled,
                tfs:      [...(st.tfs || [])],
                tf_bars:  st.tfBars || {},
                elements: st.elements || {},
            };
        }
    });
    fetch("/autotrader/start", {method:"POST", headers:{"Content-Type":"application/json"}, body:JSON.stringify(body)})
    .then(() => {
        document.getElementById("btn-start").disabled = true;
        document.getElementById("btn-stop").disabled  = false;
        showToast("Scanner pornit", "#00695c");
        document.body.classList.add("scanning");
        document.getElementById("scanner-banner").classList.add("visible");
        updateGrid({});
    });
}

// ── Strategy tabs — dinamice pentru toate strategiile ─────────────────────
let _activeTab = "classic";

function buildStratTabs() {
    const bar      = document.getElementById("strat-tabs-bar");
    const sections = document.getElementById("strat-sections");
    if (!bar || !sections) return;

    // Curata
    bar.innerHTML      = "";
    sections.innerHTML = "";

    STRAT_DEFS.forEach(({key, label, color, icon}, idx) => {
        // Tab button
        const btn = document.createElement("button");
        btn.id = `tab-${key}`;
        btn.onclick = () => switchTab(key);
        btn.style.cssText = `padding:8px 18px;border:none;border-bottom:2px solid transparent;
            margin-bottom:-2px;background:#111;color:#555;font-weight:600;
            font-size:0.85rem;cursor:pointer;transition:all 0.2s`;
        btn.innerHTML = `${icon} ${label}`;
        bar.appendChild(btn);

        // Sectiune grid
        const sec = document.createElement("div");
        sec.id = `section-${key}`;
        sec.style.cssText = "display:none;margin-bottom:16px";
        const grid = document.createElement("div");
        grid.className = "symbol-grid";
        grid.id = `symbol-grid-${key}`;
        sec.appendChild(grid);
        sections.appendChild(sec);
    });

    // Activeaza primul tab sau cel curent
    switchTab(_activeTab);
    // Umple gridurile cu simbolurile existente
    selectedSymbols.forEach(sym => _ensureSymCard(sym));
}

function _ensureSymCard(sym) {
    STRAT_DEFS.forEach(({key}) => {
        const grid = document.getElementById(`symbol-grid-${key}`);
        if (grid && !grid.querySelector(`[data-sym="${sym}"]`)) {
            const card = document.createElement("div");
            card.className = "sym-card sig-hold";
            card.dataset.sym = sym;
            card.innerHTML = `<div class="card-name">${sym}</div><div class="card-scanning">astept scanare...</div>`;
            card.onclick = () => showSymDetail(sym, key);
            grid.appendChild(card);
        }
    });
}

function switchTab(strat) {
    _activeTab      = strat;
    currentStrategy = strat;
    STRAT_DEFS.forEach(({key, color}) => {
        const active = key === strat;
        const tab    = document.getElementById(`tab-${key}`);
        const sec    = document.getElementById(`section-${key}`);
        if (tab) {
            tab.style.color             = active ? color : "#555";
            tab.style.borderBottomColor = active ? color : "transparent";
            tab.style.background        = active ? "#161616" : "#111";
        }
        if (sec) sec.style.display = active ? "block" : "none";
    });
}

function syncTabsWithState() {
    // Opacitate tab dupa enabled/disabled
    STRAT_DEFS.forEach(({key}) => {
        const tab = document.getElementById(`tab-${key}`);
        if (tab) {
            const on = stratState[key] ? stratState[key].enabled : true;
            tab.style.opacity = on ? "1" : "0.5";
        }
    });
    switchTab(_activeTab);
    // Actualizeaza toggle-urile din lista de strategii
    STRAT_DEFS.forEach(({key}) => {
        const tog = document.getElementById(`strat-toggle-${key}`);
        if (tog) tog.checked = stratState[key] ? stratState[key].enabled : true;
    });
}

function toggleStratEnabled(key, enabled) {
    if (!stratState[key]) return;
    stratState[key].enabled = enabled;
    sendStrategySet(key, "enabled", enabled);
    syncTabsWithState();
    // Sincronizeaza si checkbox-ul din panoul de detalii daca e deschis
    const chk = document.getElementById(`${key}-enabled`);
    if (chk) chk.checked = enabled;
    showToast(enabled ? `✓ ${key.toUpperCase()} activat` : `✗ ${key.toUpperCase()} oprit`, enabled ? "#26a69a" : "#ef5350");
}

function setCryptoOnly() {
    // Sterge tot forex, pastreaza/adauga doar crypto
    [...selectedSymbols].forEach(s => {
        if (!SYMBOLS_CRYPTO.includes(s)) {
            selectedSymbols.delete(s);
            document.querySelectorAll(`[data-sym="${s}"]`).forEach(c => c.remove());
        }
    });
    SYMBOLS_CRYPTO.forEach(s => { if (!selectedSymbols.has(s)) addSymbol(s); });
    const ct = document.getElementById("crypto-toggle"); if(ct) ct.checked = true;
    saveWatchlist();
    buildSymChips();
    fetch("/autotrader/set",{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify({symbols:[...selectedSymbols]})});
    showToast("₿ Crypto Only activ", "#e65100");
}

function setForexOnly() {
    // Sterge tot crypto, pastreaza/adauga doar forex
    SYMBOLS_CRYPTO.forEach(s => {
        selectedSymbols.delete(s);
        document.querySelectorAll(`[data-sym="${s}"]`).forEach(c => c.remove());
    });
    SYMBOLS_FOREX.forEach(s => { if (!selectedSymbols.has(s)) addSymbol(s); });
    const ct = document.getElementById("crypto-toggle"); if(ct) ct.checked = false;
    saveWatchlist();
    buildSymChips();
    fetch("/autotrader/set",{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify({symbols:[...selectedSymbols]})});
    showToast("Forex Only activ", "#1976d2");
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

function restoreDefaults() {
    if (!confirm("Revii la setarile implementate implicit pentru TOATE strategiile?\\n\\nAcest lucru reseteaza:\\n• TF-urile\\n• Elementele activate\\n• min_confidence (85% pentru majoritatea)\\n• Starea enabled/disabled\\n\\nNu afecteaza symbols, auto_execute sau setarile de risk.")) return;
    fetch("/autotrader/restore_defaults", {method:"POST"})
        .then(r => r.json())
        .then(d => {
            if (d.ok) {
                showToast("↻ Setari implicite restaurate (" + (d.strategies_reset||[]).length + " strategii)", "#26a69a");
                setTimeout(() => location.reload(), 600);
            } else {
                showToast("✗ Eroare: " + (d.message||""), "#b71c1c");
            }
        })
        .catch(e => showToast("✗ Eroare retea", "#b71c1c"));
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

function highlightActivePreset(data) {
    // Detecteaza ce preset se potriveste cu starea curenta a scannerului
    if (!data || !data.scanner) return;
    const sc = data.scanner;
    const currentEnabled = new Set(
        Object.keys(stratState).filter(k => stratState[k].enabled)
    );
    let bestMatch = null, bestScore = -1;
    Object.keys(PRESETS).forEach(name => {
        const p = PRESETS[name];
        const expected = new Set(p.enabled || []);
        // Scor: matchuri corecte - diferente
        let score = 0;
        expected.forEach(k => { if (currentEnabled.has(k)) score++; else score--; });
        currentEnabled.forEach(k => { if (!expected.has(k)) score--; });
        // Bonus daca parametrii globali se potrivesc
        if (Math.abs((data.sl_atr_mult || 2) - p.sl_atr_mult) < 0.1) score += 2;
        if (Math.abs((data.tp_ratio || 1) - p.tp_ratio) < 0.1) score += 1;
        if (score > bestScore) { bestScore = score; bestMatch = name; }
    });
    // Aplica highlight numai daca potrivire exacta (toate strategiile match + params)
    const exactMatch = bestScore >= (PRESETS[bestMatch]?.enabled?.length || 0) + 2;
    ["scalp","intraday","swing","sniper","session","reversal","conservative","aggressive"].forEach(k => {
        const btn = document.getElementById("preset-"+k);
        if (!btn) return;
        const isActive = exactMatch && k === bestMatch;
        btn.style.borderColor = isActive ? "#ffeb3b" : "#333";
        btn.style.color       = isActive ? "#ffeb3b" : "#aaa";
        btn.style.background  = isActive ? "#1a1a00" : "#1a1a1a";
    });
}

function syncStratStateFromScanner(sc) {
    // Sincronizeaza stratState client-side cu ce e pe server pentru TOATE strategiile
    if (!sc) return false;
    let changed = false;
    Object.keys(stratState).forEach(key => {
        const cfg = sc[key];
        if (!cfg) return;
        const st = stratState[key];
        if (cfg.enabled != null && st.enabled !== cfg.enabled) {
            st.enabled = cfg.enabled;
            changed = true;
        }
        if (cfg.tfs) {
            const newSet = new Set(cfg.tfs);
            if (st.tfs.size !== newSet.size || [...st.tfs].some(t => !newSet.has(t))) {
                st.tfs = newSet;
                changed = true;
            }
        }
        if (cfg.tf_bars) Object.assign(st.tfBars, cfg.tf_bars);
        if (cfg.elements) Object.assign(st.elements, cfg.elements);
        if (cfg.min_confidence != null && st.minConfidence !== cfg.min_confidence) {
            st.minConfidence = cfg.min_confidence;
            changed = true;
        }
        if (cfg.sl_atr_mult != null) st.slMult = cfg.sl_atr_mult;
        if (cfg.risk_dollars != null) st.riskUsd = cfg.risk_dollars;
    });
    return changed;
}

function updateUI(data) {
    const sc = data.scanner;

    // Sync strategy state live — interfata reflecta ce ruleaza pe server
    if (syncStratStateFromScanner(sc)) {
        if (typeof buildStratList === "function") buildStratList();
        // Re-highlight active preset based on current global config
        highlightActivePreset(data);
    }

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
    updateActiveModeBadge(data);

    // Combined mode votes refresh
    if (sc.combined_mode) {
        const panel = document.getElementById("combined-panel");
        if (panel && panel.style.display !== "none") refreshCombinedVotes();
    }

    // Pending mode sync
    const pmToggle = document.getElementById("pending-mode-toggle");
    if (pmToggle && pmToggle.checked !== !!sc.pending_mode) {
        pmToggle.checked = !!sc.pending_mode;
        togglePendingPanel(!!sc.pending_mode);
    }
    if (sc.pending_mode) {
        const pp = document.getElementById("pending-panel");
        if (pp && pp.style.display !== "none") refreshPendingOrders();
    }

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
    const rs   = res ? (res.risk_score || 0) : 0;
    const isScanning = document.body.classList.contains("scanning");

    card.className = `sym-card sig-${sig.toLowerCase()}` +
        (currentSymbol===sym && currentStrategy===strat ? " selected" : "") +
        (isScanning && !res ? " scanning-card" : "");

    const stratDef = STRAT_DEFS.find(s => s.key === strat) || {color:"#888"};
    const stratColor = stratDef.color;
    const rsColor = rs >= 70 ? "#66bb6a" : rs >= 50 ? "#ffeb3b" : "#ef5350";
    const rsHtml = rs > 0 ? `<div style="color:${rsColor};font-size:0.67rem;font-weight:700;margin-top:1px">RISC ${rs.toFixed(0)}</div>` : "";
    card.innerHTML = isScanning && !res
        ? `<div class="card-name">${sym}</div><div class="card-scanning"><span class="scan-spin">⟳</span> scanez...</div>`
        : `<div class="card-name">${sym}</div>
           <div class="card-signal ${sig.toLowerCase()}">${sig}</div>
           <div class="card-conf">${conf>0?conf.toFixed(1)+"% incredere":"—"}</div>
           ${rsHtml}
           <div class="card-time" style="color:${stratColor}">${ts}</div>`;
    card.onclick = () => selectCard(sym, res, strat);
}

function updateGrid(results) {
    const syms = [...selectedSymbols];
    syncTabsWithState();

    // Construieste grilele pentru toate strategiile active
    STRAT_DEFS.forEach(({key: strat}) => {
        const grid = document.getElementById(`symbol-grid-${strat}`);
        if (!grid) return;
        // Sterge simboluri disparute
        grid.querySelectorAll(".sym-card").forEach(c => {
            if (!selectedSymbols.has(c.dataset.sym)) c.remove();
        });
        if (stratState[strat] && !stratState[strat].enabled) { grid.innerHTML = ""; return; }
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

    const dpRisk = document.getElementById("dp-risk");
    if (dpRisk) {
        const rs = res.risk_score || 0;
        const rsColor = rs >= 70 ? "#66bb6a" : rs >= 50 ? "#ffeb3b" : "#ef5350";
        dpRisk.textContent = rs > 0 ? `RISC ${rs.toFixed(0)}` : "";
        dpRisk.style.color = rsColor;
        dpRisk.style.borderColor = rsColor;
        dpRisk.style.border = rs > 0 ? `1px solid ${rsColor}` : "none";
    }

    // TF table
    const tbody  = document.getElementById("dp-tf-table");
    const thead  = document.getElementById("dp-tf-header");
    tbody.innerHTML = "";

    if (res.strategy === "eob") {
        // EOB: tabel detaliat cu contributia fiecarui TF
        thead.innerHTML = `<tr>
            <th>TF</th><th>Faza</th><th>Gasit</th>
            <th style="color:#26a69a;text-align:center">BUY</th>
            <th style="color:#ef5350;text-align:center">SELL</th>
        </tr>`;
        (res.tfs || []).forEach(r => {
            const buyPts  = r.score_buy  || 0;
            const sellPts = r.score_sell || 0;
            const buyStr  = buyPts  > 0 ? `<b style="color:#26a69a">+${buyPts}</b>`  : `<span style="color:#444">·</span>`;
            const sellStr = sellPts > 0 ? `<b style="color:#ef5350">+${sellPts}</b>` : `<span style="color:#444">·</span>`;
            const phaseColor = {"HTF":"#9c27b0","MTF":"#ff9800","MTF (Unicorn)":"#ff9800","LTF":"#00bcd4"}[r.phase] || "#888";
            const found = (r.found || "—");
            const tr = document.createElement("tr");
            tr.innerHTML = `
                <td><b>${r.tf}</b></td>
                <td style="color:${phaseColor};font-size:0.78em;white-space:nowrap">${r.phase || "—"}</td>
                <td style="font-size:0.77em;color:#bbb;max-width:240px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap"
                    title="${found}">${found}</td>
                <td style="text-align:center">${buyStr}</td>
                <td style="text-align:center">${sellStr}</td>
            `;
            tbody.appendChild(tr);
        });
    } else {
        // Default: tabel clasic signal/trend/convingere
        thead.innerHTML = `<tr>
            <th>TF</th><th>Signal</th><th>Trend</th><th>Convingere</th>
        </tr>`;
        (res.tfs || []).forEach(r => {
            const cls = r.signal === "BUY" ? "sig-buy" : r.signal === "SELL" ? "sig-sell" : "sig-hold";
            const tr = document.createElement("tr");
            tr.innerHTML = `
                <td>${r.tf}</td>
                <td class="${cls}">${r.signal}</td>
                <td>${trendRo(r.trend)}</td>
                <td>${"★".repeat(r.conviction||0)}</td>
            `;
            tbody.appendChild(tr);
        });
    }

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
    const chartTf  = (currentChartTf && availTfs.includes(currentChartTf)) ? currentChartTf : (bf.tf || (availTfs.length > 0 ? availTfs[0] : "M5"));
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
            currentChartTf = tf;
            loadChartTf(symbol, tf, true);  // forceReload — TF schimbat intentionat
        };
        wrap.appendChild(btn);
    });
}

let _chartLoadSym = null;
let _chartLoadTf  = null;

function _getSavedZoom(div) {
    // Citeste range-ul curent direct din div-ul Plotly randat
    try {
        const xr = div.layout && div.layout.xaxis && div.layout.xaxis.range;
        if (xr && xr.length === 2) return [xr[0], xr[1]];
    } catch(e) {}
    return null;
}

function loadChartTf(symbol, tf, forceReload) {
    const bars = (stratState[currentStrategy]||stratState.classic).tfBars[tf] || DEFAULT_BARS;

    const loading = document.getElementById("dp-chart-loading");
    const div     = document.getElementById("dp-chart-div");

    // Acelasi simbol+TF deja afisat si nu e reload fortat → pastreaza graficul intact
    if (!forceReload && _chartLoadSym === symbol && _chartLoadTf === tf && div._hasChart) return;

    // Salveaza zoom-ul curent INAINTE de reload (doar daca e acelasi simbol+TF)
    const sameContext = (symbol === _chartLoadSym && tf === _chartLoadTf);
    const savedZoom   = (sameContext && div._hasChart) ? _getSavedZoom(div) : null;

    _chartLoadSym = symbol;
    _chartLoadTf  = tf;

    if (!div._hasChart) {
        if (loading) { loading.classList.add("visible"); div.style.display = "none"; }
    }

    fetch(`/autotrader/chartjson/${symbol}/${tf}?bars=${bars}`)
        .then(r => r.json())
        .then(fig => {
            if (_chartLoadSym !== symbol || _chartLoadTf !== tf) return;

            if (loading) { loading.classList.remove("visible"); div.style.display = ""; }
            div._hasChart = true;

            // Sterge DOAR range-ul din xaxis — pastreaza restul (gridcolor, uirevision etc.)
            // Altfel Plotly aplica range-ul generat de server si suprascrie zoom-ul
            if (savedZoom && fig.layout && fig.layout.xaxis) {
                delete fig.layout.xaxis.range;
                delete fig.layout.xaxis.autorange;
            }

            // dragmode:"pan" din layout activeaza scroll-zoom automat in Plotly
            Plotly.react(div, fig.data, fig.layout, {
                responsive: true,
                displayModeBar: true,
                scrollZoom: true,
                modeBarButtonsToRemove: ["lasso2d", "select2d"],
            }).then(() => {
                // Restaureaza zoom-ul utilizatorului dupa ce Plotly a terminat render-ul
                if (savedZoom) {
                    Plotly.relayout(div, { "xaxis.range": savedZoom });
                }
            });
        })
        .catch(e => {
            if (loading) { loading.classList.remove("visible"); div.style.display = ""; }
            console.warn("Chart load error:", e);
        });
}

// ── Manual execute ────────────────────────────────────────────────────────
function executeManual(signal) {
    if (!currentSymbol) return;
    const exRes = document.getElementById("execute-result");
    exRes.style.display = "none";

    fetch("/autotrader/status")
        .then(r => r.json())
        .then(data => {
            const symData = data.results[currentSymbol] || {};
            const res = symData[currentStrategy] || symData;
            const bf  = res && res.best_tf ? res.best_tf : {};
            return fetch("/autotrader/execute", {
                method: "POST",
                headers: {"Content-Type":"application/json"},
                body: JSON.stringify({
                    symbol:   currentSymbol,
                    signal:   signal,
                    sl:       bf.sl || 0,
                    tp:       bf.tp || 0,
                    strategy: currentStrategy,
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
const MODE_FROM_TF = {
    "M1":"⚡ Scalp","M5":"⚡ Scalp","M15":"📈 Intraday",
    "H1":"📈 Intraday","H4":"🌊 Swing","D1":"🌊 Swing","W1":"🌊 Swing"
};
const MODE_COLORS = {
    "⚡ Scalp":    {bg:"#1a1400",border:"#f9a825",color:"#f9a825"},
    "📈 Intraday": {bg:"#0a1a0a",border:"#66bb6a",color:"#66bb6a"},
    "🌊 Swing":    {bg:"#0a0f1a",border:"#64b5f6",color:"#64b5f6"},
};

function getModeFromDecision(d) {
    if (d.tf) return MODE_FROM_TF[d.tf] || null;
    // Fallback: deriva din strategie
    const scalp  = ["london_breakout","ny_breakout","china_session","vwap_bounce"];
    const swing  = ["ichimoku","rsi_divergence","macd"];
    if (scalp.includes(d.strategy))  return "⚡ Scalp";
    if (swing.includes(d.strategy))  return "🌊 Swing";
    return "📈 Intraday";
}

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
        const stratDef = STRAT_DEFS.find(s => s.key === d.strategy) || {label: d.strategy, color: "#888", icon: "⚪"};
        const stratLabel = `<span style="color:${stratDef.color}">${stratDef.icon} ${stratDef.label}</span>`;

        const mode = getModeFromDecision(d);
        const mc   = mode ? (MODE_COLORS[mode] || {bg:"#111",border:"#555",color:"#aaa"}) : null;
        const modeBadge = mode
            ? `<span style="font-size:0.68rem;padding:1px 6px;border-radius:8px;background:${mc.bg};border:1px solid ${mc.border};color:${mc.color};white-space:nowrap">${mode}</span>`
            : `<span style="color:#444">—</span>`;

        const analysisBtn = d.chart_id
            ? `<a href="/autotrader/analysis/${d.chart_id}" target="_blank"
                  title="Deschide analiza pe grafic"
                  style="display:inline-flex;align-items:center;gap:3px;
                         padding:3px 8px;border-radius:5px;
                         background:#161b22;border:1px solid #30363d;
                         color:#58a6ff;font-size:0.76rem;text-decoration:none;
                         white-space:nowrap;transition:background .15s"
                  onmouseover="this.style.background='#21262d'"
                  onmouseout="this.style.background='#161b22'">📊 Grafic</a>`
            : `<span style="color:#333;font-size:0.72rem">—</span>`;
        tr.innerHTML = `
            <td>${d.timestamp.substring(0,19).replace("T"," ")}</td>
            <td>${d.symbol}</td>
            <td class="${cls}">${d.signal}</td>
            <td>${stratLabel}${d.tf ? `<span style="color:#555;font-size:0.72rem"> · ${d.tf}</span>` : ""}</td>
            <td>${modeBadge}</td>
            <td>${d.confidence ? d.confidence.toFixed(1)+"%" : "—"}</td>
            <td class="${d.executed ? "dec-yes" : "dec-no"}">${d.executed ? "DA" : "NU"}</td>
            <td style="color:#666;font-size:0.76rem">${(d.result||"").substring(0,80)}</td>
            <td>${analysisBtn}</td>
        `;
        tbody.appendChild(tr);
    });
}

function updateActiveModeBadge(data) {
    const el = document.getElementById("active-mode-badge");
    if (!el || !data) return;
    // Detecteaza presetul activ din highlightActivePreset logic
    const active = ["scalp","intraday","swing","sniper","session","reversal","conservative","aggressive"]
        .find(k => {
            const btn = document.getElementById("preset-"+k);
            return btn && btn.style.borderColor === "rgb(255, 235, 59)"; // #ffeb3b = active
        });
    if (active) {
        const p = PRESETS[active];
        const mc = MODE_COLORS[p?.label] || {bg:"#1a1a00",border:"#ffeb3b",color:"#ffeb3b"};
        el.textContent = `Mod activ: ${p?.label || active}`;
        el.style.background   = mc.bg;
        el.style.borderColor  = mc.border;
        el.style.color        = mc.color;
    } else {
        el.textContent = "Mod: personalizat";
        el.style.background  = "#111";
        el.style.borderColor = "#444";
        el.style.color       = "#666";
    }
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
    // Restaureaza intervalul din scanner (sursa de adevar = server)
    if (data.scanner && data.scanner.interval) {
        const idx = INTERVALS.indexOf(data.scanner.interval);
        const el  = document.getElementById("interval-range");
        if (idx >= 0 && el) {
            el.value = idx;
            document.getElementById("interval-label").textContent = data.scanner.interval + "s";
        }
    }
    if (data.risk_dollars != null) { document.getElementById("risk-dollars-input").value = data.risk_dollars; window._globalRiskUsd = data.risk_dollars; }
    if (data.risk_pct != null) document.getElementById("risk-pct-input").value = data.risk_pct;
    if (data.use_risk_pct != null) setRiskMode(data.use_risk_pct);
    if (data.sl_atr_mult != null) {
        document.getElementById("sl-mult-range").value = data.sl_atr_mult;
        document.getElementById("sl-mult-label").textContent = parseFloat(data.sl_atr_mult).toFixed(1);
    }
    if (data.max_lot_global != null) document.getElementById("max-lot-input").value = data.max_lot_global;
    const sc=data.scanner;
    const toggle=document.getElementById("auto-exec-toggle");
    if (toggle) { toggle.checked=sc.auto_execute||false; document.getElementById("auto-ex-warn").style.display=sc.auto_execute?"inline":"none"; }
    const ops=document.getElementById("one-per-strategy-toggle");
    if (ops) ops.checked=sc.one_per_strategy||false;
    const ee=document.getElementById("early-exit-toggle");
    if (ee) ee.checked=sc.early_exit||false;
    const te=document.getElementById("time-exit-toggle");
    if (te) te.checked=sc.time_based_exit!==false;
    const beh=document.getElementById("breakeven-hours-input");
    if (beh && sc.breakeven_hours!=null) beh.value=sc.breakeven_hours;
    const mth=document.getElementById("max-trade-hours-input");
    if (mth && sc.max_trade_hours!=null) mth.value=sc.max_trade_hours;
    const cm=document.getElementById("combined-mode-toggle");
    if (cm) { cm.checked=sc.combined_mode||false; toggleCombinedPanel(sc.combined_mode||false); }
    // use_strategy_params sync
    const usp = document.getElementById("use-strategy-params-toggle");
    if (usp && sc.use_strategy_params != null) usp.checked = sc.use_strategy_params;
    // Scalp Boost sync
    if (sc.scalp_boost != null) toggleScalpBoost(sc.scalp_boost, false);
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
pollTimer = setInterval(pollStatus, 3000);

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
@autotrader_bp.route("/autotrader_legacy")
@login_required
def autotrader_legacy_page():
    """UI-ul clasic (v1) — pastrat pentru fallback."""
    import json
    symbols_json = json.dumps(SYMBOLS)
    crypto_json  = json.dumps(SYMBOLS_CRYPTO)
    forex_json   = json.dumps(SYMBOLS)
    preselect    = request.args.get("symbol", "")
    decide_now   = request.args.get("decide", "0")
    html = AUTOTRADER_HTML.replace("{{ symbols_json }}", symbols_json) \
                          .replace("{{ crypto_json }}", crypto_json) \
                          .replace("{{ forex_json }}", forex_json) \
                          .replace("{{ preselect_symbol }}", preselect) \
                          .replace("{{ decide_now }}", decide_now)
    return Response(html, content_type="text/html; charset=utf-8")


@autotrader_bp.route("/autotrader")
@autotrader_bp.route("/autotrader_v2")
@login_required
def autotrader_page():
    """UI-ul minimalist v2 — devenit default. /autotrader_legacy pentru vechi."""
    import os
    tpl_path = os.path.join(os.path.dirname(__file__), "templates", "autotrader_v2.html")
    with open(tpl_path, encoding="utf-8") as f:
        html = f.read()
    return Response(html, content_type="text/html; charset=utf-8")


@autotrader_bp.route("/autotrader/presets")
@login_required
def autotrader_presets():
    """Returneaza JSON-ul tuturor preset-urilor (din presets.json)."""
    import os
    p = os.path.join(os.path.dirname(__file__), "presets.json")
    try:
        with open(p, encoding="utf-8") as f:
            return Response(f.read(), content_type="application/json")
    except Exception as e:
        return Response(json.dumps({"error": str(e)}), content_type="application/json", status=500)


@autotrader_bp.route("/autotrader/apply_preset", methods=["POST"])
@login_required
def autotrader_apply_preset():
    """
    Aplica un preset (nume sau JSON inline) -> seteaza scanner + globale.
    Body: {"name": "scalp"} sau {"policy": {...}}
    """
    import os
    body = request.get_json(silent=True) or {}
    policy = body.get("policy")
    name = body.get("name")
    if not policy and name:
        try:
            with open(os.path.join(os.path.dirname(__file__), "presets.json"), encoding="utf-8") as f:
                presets = json.load(f)
            policy = presets.get(name)
        except Exception as e:
            return jsonify({"ok": False, "error": str(e)}), 400
    if not policy:
        return jsonify({"ok": False, "error": "policy lipsa"}), 400

    # Globale (sl_atr_mult, risk_dollars, max_open_trades, tp_ratio, max_lot, scalp_boost, combined_mode)
    g = policy.get("global", {})
    if "sl_atr_mult" in g:    _app.SL_ATR_MULT = float(g["sl_atr_mult"])
    if "max_lot_global" in g: _app.MAX_LOT_GLOBAL = float(g["max_lot_global"])
    if "tp_ratio" in g:       _app.TP_RATIO = float(g["tp_ratio"])
    if "risk_dollars" in g:   _app.RISK_DOLLARS = float(g["risk_dollars"])
    if "max_open_trades" in g: _app.MAX_OPEN_TRADES = int(g["max_open_trades"])
    if "combined_mode" in g:  scanner["combined_mode"] = bool(g["combined_mode"])
    if "scalp_boost" in g:    scanner["scalp_boost"] = bool(g["scalp_boost"])

    enabled = set(policy.get("enabled", []))
    disabled = set(policy.get("disabled", []))
    tfs_map = policy.get("per_strategy_tfs", {})
    confs_map = policy.get("per_strategy_min_confidence", {})

    import strategies as _sp
    all_keys = {s.key for s in _sp.list_all()}
    for key in all_keys:
        if key not in scanner:
            scanner[key] = {"enabled": False, "tfs": [], "tf_bars": {},
                            "elements": {}, "min_confidence": 66.0}
        if key in enabled:
            scanner[key]["enabled"] = True
        elif key in disabled:
            scanner[key]["enabled"] = False
        if key in tfs_map:
            scanner[key]["tfs"] = [t for t in tfs_map[key] if t in ALL_TFS]
        if key in confs_map:
            scanner[key]["min_confidence"] = float(confs_map[key])

    return jsonify({"ok": True, "applied": name or "custom"})


@autotrader_bp.route("/autotrader/strategies")
@login_required
def autotrader_strategies():
    """Returneaza lista tuturor strategiilor + config din scanner.
    element_labels include ATAT elementele specifice strategiei, CAT SI
    filtrele comune opt_* (volume, RSI, EMA, etc.)."""
    import strategies as _sp
    import json as _json
    defs = []
    for s in _sp.list_all():
        cfg = scanner.get(s.key, {})
        all_elems = s._all_elements()   # specific + opt_*
        # Default: elementele specifice TRUE, STRICT/BLOCKING + opt_* FALSE (opt-in)
        elements_default = {}
        for k, lbl in all_elems.items():
            if k.startswith("opt_"):
                elements_default[k] = False
            elif "STRICT" in lbl.upper() or "BLOCKING" in lbl.upper():
                elements_default[k] = False
            else:
                elements_default[k] = True
        elements_current = {**elements_default, **cfg.get("elements", {})}
        # min_confidence: scanner config > strategy_defaults > 66
        strat_default_conf = (s.strategy_defaults or {}).get("min_confidence", 66.0)
        defs.append({
            "key":            s.key,
            "name":           s.name,
            "icon":           s.icon,
            "color":          s.color,
            "enabled":        cfg.get("enabled", False),
            "tfs":            cfg.get("tfs", s.default_tfs),
            "tf_bars":        cfg.get("tf_bars", {}),
            "elements":       elements_current,
            "element_labels": all_elems,
            "min_confidence": cfg.get("min_confidence", strat_default_conf),
            "sl_atr_mult":    cfg.get("sl_atr_mult"),
            "risk_dollars":   cfg.get("risk_dollars"),
        })
    return Response(_json.dumps(defs), mimetype="application/json")


@autotrader_bp.route("/autotrader/ticket_chart")
@login_required
def autotrader_ticket_chart():
    """Returneaza map {ticket: chart_id} pentru pozitiile deschise."""
    with _ticket_chart_map_lock:
        data = {str(k): v for k, v in _ticket_chart_map.items()}
    return Response(json.dumps(data), content_type="application/json")


@autotrader_bp.route("/autotrader/stats")
@login_required
def autotrader_stats():
    """Statistici de performanta din review_log + perf_stats + MT5 live."""
    try:
        data = compute_stats(window_trades=100)
        return Response(json.dumps(data, cls=NpEncoder), content_type="application/json")
    except Exception as exc:
        return Response(json.dumps({"error": str(exc)}), content_type="application/json")


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
        "risk_dollars":    _app.RISK_DOLLARS,
        "use_risk_pct":    _app.USE_RISK_PCT,
        "risk_pct":        _app.RISK_PCT,
        "sl_atr_mult":     _app.SL_ATR_MULT,
        "max_lot_global":  _app.MAX_LOT_GLOBAL,
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
        if "sl_atr_mult" in cfg:
            v = cfg["sl_atr_mult"]
            if v is None:
                scanner[key].pop("sl_atr_mult", None)  # reset la global
            else:
                scanner[key]["sl_atr_mult"] = max(0.3, min(5.0, float(v)))
        if "risk_dollars" in cfg:
            v = cfg["risk_dollars"]
            if v is None:
                scanner[key].pop("risk_dollars", None)  # reset la global
            else:
                scanner[key]["risk_dollars"] = max(1.0, min(10000.0, float(v)))


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
    if "one_per_strategy" in body:
        scanner["one_per_strategy"] = bool(body["one_per_strategy"])
    if "early_exit" in body:
        scanner["early_exit"] = bool(body["early_exit"])
    _apply_strategy_config(body)
    start_scanner()
    return Response(json.dumps({"ok": True}), content_type="application/json")


@autotrader_bp.route("/autotrader/stop", methods=["POST"])
@login_required
def autotrader_stop():
    stop_scanner()
    return Response(json.dumps({"ok": True}), content_type="application/json")


@autotrader_bp.route("/autotrader/combined_log")
@login_required
def autotrader_combined_log():
    """Returneaza ultimele N actiuni combined din review_log.json."""
    try:
        n = int(request.args.get("n", 50))
        if _os.path.exists(_LOG_FILE):
            with open(_LOG_FILE, "r", encoding="utf-8") as f:
                data = json.load(f)
        else:
            data = []
        combined = [e for e in data if e.get("strategy") == "combined"]
        combined.sort(key=lambda x: x.get("timestamp", ""), reverse=True)
        return jsonify(combined[:n])
    except Exception as e:
        return jsonify([])


@autotrader_bp.route("/autotrader/set", methods=["POST"])
@login_required
def autotrader_set():
    """Modifica setari fara sa reporneasca scannerul."""
    body = request.get_json(silent=True) or {}
    if "auto_execute" in body:
        scanner["auto_execute"] = bool(body["auto_execute"])
    if "use_strategy_params" in body:
        scanner["use_strategy_params"] = bool(body["use_strategy_params"])
    if "one_per_strategy" in body:
        scanner["one_per_strategy"] = bool(body["one_per_strategy"])
    if "early_exit" in body:
        scanner["early_exit"] = bool(body["early_exit"])
    if "time_based_exit" in body:
        scanner["time_based_exit"] = bool(body["time_based_exit"])
    if "breakeven_hours" in body:
        scanner["breakeven_hours"] = max(0.25, min(24.0, float(body["breakeven_hours"])))
    if "max_trade_hours" in body:
        scanner["max_trade_hours"] = max(0.5, min(72.0, float(body["max_trade_hours"])))
    if "pending_mode" in body:
        scanner["pending_mode"] = bool(body["pending_mode"])
    if "pending_expiry_hours" in body:
        scanner["pending_expiry_hours"] = max(1, int(body["pending_expiry_hours"]))
    if "combined_mode" in body:
        scanner["combined_mode"] = bool(body["combined_mode"])
        # Strategiile de sesiune nu au sens in combined mode (adauga HOLD-uri false)
        # Le dezactivam automat cand combined mode e activat
        SESSION_STRATEGIES = ["london_breakout", "ny_breakout", "china_session", "vwap_bounce"]
        if scanner["combined_mode"]:
            for sk in SESSION_STRATEGIES:
                if sk in scanner.get("strategies", {}):
                    scanner["strategies"][sk]["enabled"] = False
            log.info(f"[Combined] Strategii de sesiune dezactivate automat: {SESSION_STRATEGIES}")
    if "interval" in body:
        scanner["interval"] = int(body["interval"])
    if "symbols" in body:
        scanner["symbols"] = [s for s in body["symbols"] if s]
    if "max_open_trades" in body:
        _app.MAX_OPEN_TRADES = max(1, int(body["max_open_trades"]))
    if "scalp_boost" in body:
        scanner["scalp_boost"] = bool(body["scalp_boost"])
    if "scalp_lot_mult" in body:
        scanner["scalp_lot_mult"] = max(0.5, min(10.0, float(body["scalp_lot_mult"])))
    if "scalp_max_min" in body:
        scanner["scalp_max_min"] = max(5, min(120, int(body["scalp_max_min"])))
    if "scalp_tp_rr" in body:
        scanner["scalp_tp_rr"] = max(0.5, min(3.0, float(body["scalp_tp_rr"])))
    if "scalp_min_agree" in body:
        scanner["scalp_min_agree"] = max(2, min(8, int(body["scalp_min_agree"])))
    if "tp_ratio" in body:
        _app.TP_RATIO = max(0.1, float(body["tp_ratio"]))
    if "risk_dollars" in body:
        _app.RISK_DOLLARS = max(1.0, float(body["risk_dollars"]))
        _app._save_risk_config()
    if "use_risk_pct" in body:
        _app.USE_RISK_PCT = bool(body["use_risk_pct"])
        _app._save_risk_config()
    if "risk_pct" in body:
        _app.RISK_PCT = max(0.1, min(10.0, float(body["risk_pct"])))
        _app._save_risk_config()
    if "sl_atr_mult" in body:
        _app.SL_ATR_MULT = max(0.5, min(5.0, float(body["sl_atr_mult"])))
        _app._save_risk_config()
    if "max_lot_global" in body:
        _app.MAX_LOT_GLOBAL = max(0.01, min(100.0, float(body["max_lot_global"])))
        _app._save_risk_config()
    _apply_strategy_config(body)
    return Response(json.dumps({"ok": True}), content_type="application/json")


@autotrader_bp.route("/autotrader/restore_defaults", methods=["POST"])
@login_required
def autotrader_restore_defaults():
    """
    Revine la setarile implementate implicit (snapshot SCANNER_DEFAULTS_SNAPSHOT).
    Afecteaza doar configul strategiilor (tfs, elements, min_confidence, enabled).
    Nu schimba symbols, auto_execute sau alte flag-uri globale.
    """
    import copy
    try:
        for key, default_cfg in SCANNER_DEFAULTS_SNAPSHOT.items():
            # Doar cheile per-strategie (dict cu tfs/elements/min_confidence)
            if isinstance(default_cfg, dict) and "min_confidence" in default_cfg:
                if key in scanner:
                    scanner[key] = copy.deepcopy(default_cfg)
        log.info("[AutoTrader] Defaults restaurate pentru toate strategiile")
        return Response(json.dumps({
            "ok": True,
            "message": "Setari implicite restaurate",
            "strategies_reset": [k for k, v in SCANNER_DEFAULTS_SNAPSHOT.items()
                                 if isinstance(v, dict) and "min_confidence" in v],
        }), content_type="application/json")
    except Exception as exc:
        log.warning(f"[AutoTrader] restore_defaults error: {exc}")
        return Response(json.dumps({"ok": False, "message": str(exc)}),
                        content_type="application/json")


@autotrader_bp.route("/autotrader/execute", methods=["POST"])
@login_required
def autotrader_execute():
    try:
        body = request.get_json(silent=True) or {}
        symbol   = body.get("symbol", "").upper()
        signal   = body.get("signal", "")
        sl       = float(body.get("sl", 0) or 0)
        tp       = float(body.get("tp", 0) or 0)
        strategy = body.get("strategy", "manual")

        if not symbol or signal not in ("BUY", "SELL"):
            return Response(
                json.dumps({"ok": False, "message": f"Parametri invalizi: symbol={symbol} signal={signal}"}),
                content_type="application/json"
            )

        ok, msg = place_trade(symbol, signal, sl, tp, RISK_DOLLARS, strategy=strategy)
        decision = {
            "timestamp":  datetime.now().isoformat(),
            "symbol":     symbol,
            "signal":     signal,
            "strategy":   strategy,
            "confidence": results.get(symbol, {}).get(strategy, {}).get("confidence", 0),
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


@autotrader_bp.route("/autotrader/pending_orders")
@login_required
def autotrader_pending_orders():
    """Returneaza toate ordinele pending active din MT5 plasate de AutoTrader."""
    if not MT5_AVAILABLE or mt5 is None:
        return Response(json.dumps({"ok": False, "orders": [], "message": "MT5 indisponibil"}),
                        content_type="application/json")
    try:
        orders = mt5.orders_get() or []
        result = []
        for o in orders:
            comment = getattr(o, "comment", "") or ""
            if not comment.startswith("CV_"):
                continue
            result.append({
                "ticket":      o.ticket,
                "symbol":      o.symbol,
                "type":        ["BUY_LIM","SELL_LIM","BUY_STP","SELL_STP","BUY_STOP_LIM","SELL_STOP_LIM"][o.type] if o.type < 6 else str(o.type),
                "volume":      o.volume_initial,
                "price":       o.price_open,
                "sl":          o.sl,
                "tp":          o.tp,
                "comment":     comment,
                "time_setup":  str(o.time_setup),
                "time_expiration": str(o.time_expiration),
            })
        return Response(json.dumps({"ok": True, "orders": result}), content_type="application/json")
    except Exception as e:
        return Response(json.dumps({"ok": False, "orders": [], "message": str(e)}),
                        content_type="application/json")


@autotrader_bp.route("/autotrader/cancel_pending", methods=["POST"])
@login_required
def autotrader_cancel_pending():
    """Anuleaza un ordin pending dupa ticket."""
    if not MT5_AVAILABLE or mt5 is None:
        return Response(json.dumps({"ok": False, "message": "MT5 indisponibil"}),
                        content_type="application/json")
    try:
        body   = request.get_json(silent=True) or {}
        ticket = int(body.get("ticket", 0))
        if not ticket:
            return Response(json.dumps({"ok": False, "message": "ticket lipsa"}),
                            content_type="application/json")
        req = {
            "action": mt5.TRADE_ACTION_REMOVE,
            "order":  ticket,
        }
        result = mt5.order_send(req)
        if result and result.retcode == mt5.TRADE_RETCODE_DONE:
            return Response(json.dumps({"ok": True, "message": f"Ordin #{ticket} anulat"}),
                            content_type="application/json")
        code = result.retcode if result else -1
        return Response(json.dumps({"ok": False, "message": f"Eroare anulare: retcode={code}"}),
                        content_type="application/json")
    except Exception as e:
        return Response(json.dumps({"ok": False, "message": str(e)}),
                        content_type="application/json")


@autotrader_bp.route("/autotrader/place_pending", methods=["POST"])
@login_required
def autotrader_place_pending():
    """Plaseaza manual un ordin pending."""
    try:
        body        = request.get_json(silent=True) or {}
        symbol      = body.get("symbol", "").upper()
        signal      = body.get("signal", "")
        entry_price = float(body.get("entry_price", 0) or 0)
        sl          = float(body.get("sl", 0) or 0)
        tp          = float(body.get("tp", 0) or 0)
        expiry_h    = float(body.get("expiry_hours", 24) or 24)
        strategy    = body.get("strategy", "manual")

        if not symbol or signal not in ("BUY", "SELL") or not entry_price or not sl or not tp:
            return Response(json.dumps({"ok": False, "message": "Parametri invalizi"}),
                            content_type="application/json")

        ok, msg = place_pending_order(symbol, signal, entry_price, sl, tp,
                                      RISK_DOLLARS, strategy=strategy,
                                      expiry_hours=expiry_h)
        return Response(json.dumps({"ok": ok, "message": msg}), content_type="application/json")
    except Exception as e:
        return Response(json.dumps({"ok": False, "message": str(e)}),
                        content_type="application/json")


@autotrader_bp.route("/autotrader/perf")
@login_required
def autotrader_perf():
    """
    P4: Statistici de performanta per strategie.
    GET /autotrader/perf?strategy=classic&symbol=EURUSD
    """
    strategy = request.args.get("strategy", None)
    symbol   = request.args.get("symbol", None)

    if strategy:
        # Stats pentru o singura strategie
        stats = get_perf_stats(strategy, symbol)
        stats["strategy"] = strategy
        stats["symbol"]   = symbol or "toate"
        stats["auto_disabled"] = _app.is_auto_disabled(strategy)
        return Response(json.dumps(stats, cls=NpEncoder), mimetype="application/json")

    # Toate strategiile
    all_stats = {}
    strat_keys = [
        "classic", "smc", "eob", "macd", "bollinger", "supertrend",
        "london_breakout", "ny_breakout", "china_session",
        "rsi_divergence", "engulfing", "ichimoku", "ema_cross",
    ]
    for sk in strat_keys:
        stats = get_perf_stats(sk, symbol)
        stats["auto_disabled"] = _app.is_auto_disabled(sk)
        all_stats[sk] = stats

    return Response(json.dumps({
        "strategies": all_stats,
        "total_logged": len(_app._perf_log),
        "auto_disabled": list(_app._auto_disabled_strategies),
    }, cls=NpEncoder), mimetype="application/json")


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


def _build_trade_chart_fig(symbol, tf, bars):
    """Construieste figura Plotly cu overlay trade-uri. Returneaza fig sau None."""
    fig_or_html = build_chart(symbol, tf, bars, compact=False, return_fig=True)
    if isinstance(fig_or_html, str):
        return None  # eroare — nu avem date

    fig = fig_or_html

    # Suprapune pozitiile MT5 deschise pentru acest simbol
    try:
        if MT5_AVAILABLE and mt5:
            positions = mt5.positions_get(symbol=symbol) or []
            for pos in positions:
                sig_is_buy  = pos.type == 0
                entry_clr   = "#26a69a" if sig_is_buy else "#ef5350"
                sl_clr      = "#ef5350"
                tp_clr      = "#26a69a"
                ticket      = pos.ticket
                entry_price = pos.price_open
                sl_price    = pos.sl
                tp_price    = pos.tp
                profit      = pos.profit
                lots        = pos.volume
                direction   = "BUY" if sig_is_buy else "SELL"
                profit_str  = f"+{profit:.2f}$" if profit >= 0 else f"{profit:.2f}$"
                profit_clr  = "#26a69a" if profit >= 0 else "#ef5350"

                fig.add_hline(y=entry_price, line=dict(color=entry_clr, width=2), row=1, col=1)
                fig.add_annotation(
                    xref="paper", yref="y", x=1.002, y=entry_price,
                    text=f"<b>#{ticket} {direction}</b> {entry_price:.5f}<br>"
                         f"<span style='color:{profit_clr}'>{profit_str}</span> | {lots}L",
                    showarrow=False, font=dict(color=entry_clr, size=9),
                    xanchor="left", yanchor="middle", bgcolor="rgba(17,17,17,0.85)", row=1, col=1,
                )
                if sl_price and sl_price > 0:
                    fig.add_hline(y=sl_price, line=dict(color=sl_clr, width=1.5, dash="dash"), row=1, col=1)
                    fig.add_annotation(
                        xref="paper", yref="y", x=1.002, y=sl_price,
                        text=f"SL {sl_price:.5f}", showarrow=False,
                        font=dict(color=sl_clr, size=9), xanchor="left", yanchor="middle",
                        bgcolor="rgba(17,17,17,0.85)", row=1, col=1,
                    )
                    y0, y1 = min(entry_price, sl_price), max(entry_price, sl_price)
                    fig.add_hrect(y0=y0, y1=y1, row=1, col=1,
                                  fillcolor="rgba(239,83,80,0.08)", line=dict(width=0))
                if tp_price and tp_price > 0:
                    fig.add_hline(y=tp_price, line=dict(color=tp_clr, width=1.5, dash="dash"), row=1, col=1)
                    fig.add_annotation(
                        xref="paper", yref="y", x=1.002, y=tp_price,
                        text=f"TP {tp_price:.5f}", showarrow=False,
                        font=dict(color=tp_clr, size=9), xanchor="left", yanchor="middle",
                        bgcolor="rgba(17,17,17,0.85)", row=1, col=1,
                    )
                    y0, y1 = min(entry_price, tp_price), max(entry_price, tp_price)
                    fig.add_hrect(y0=y0, y1=y1, row=1, col=1,
                                  fillcolor="rgba(38,166,154,0.06)", line=dict(width=0))
    except Exception as _pos_ex:
        log.debug(f"chart overlay positions {symbol}: {_pos_ex}")

    return fig


# ── Analysis persistence — grupat pe cont MT5 ─────────────────────────────────
_analysis_cache      = {}
_analysis_cache_lock = threading.Lock()
_MAX_ANALYSIS_CACHE  = 500          # maxim in memorie
_MAX_DISK_ENTRIES    = 2000         # maxim per cont pe disc
_TTL_DAYS            = 60           # sterge analizele mai vechi de N zile

# Mapare ticket MT5 → chart_id (pentru butonul Analiza din cardul de trade)
_ticket_chart_map      = {}
_ticket_chart_map_lock = threading.Lock()

# Directorul radacina pentru cache pe disc
_CACHE_ROOT = _os.path.join(_os.path.dirname(__file__), ".cache")


def _get_account_folder() -> str:
    """
    Returneaza calea catre folderul contului curent MT5.
    Format: .cache/{login}_{server}/
    Daca MT5 nu e disponibil, foloseste 'unknown'.
    """
    folder_name = "unknown"
    try:
        if MT5_AVAILABLE and mt5:
            acc = mt5.account_info()
            if acc:
                server_clean = (acc.server or "").replace(" ", "_").replace("/", "_")
                folder_name  = f"{acc.login}_{server_clean}"
    except Exception:
        pass
    path = _os.path.join(_CACHE_ROOT, folder_name)
    _os.makedirs(path, exist_ok=True)
    return path


def _analysis_cache_file() -> str:
    return _os.path.join(_get_account_folder(), "analysis_cache.json")


def _ticket_chart_file() -> str:
    return _os.path.join(_get_account_folder(), "ticket_chart.json")


def _save_analysis_to_disk():
    """Scrie _analysis_cache pe disc (in folderul contului curent)."""
    try:
        path = _analysis_cache_file()
        with _analysis_cache_lock:
            data = dict(_analysis_cache)
        # Elimina scan_result (poate fi mare si nu e necesar persistent)
        slim = {}
        for cid, snap in data.items():
            s = dict(snap)
            s.pop("scan_result", None)
            slim[cid] = s
        # Pastreaza doar ultimele _MAX_DISK_ENTRIES (cele mai recente = key sortata desc)
        if len(slim) > _MAX_DISK_ENTRIES:
            keep = sorted(slim.keys(), reverse=True)[:_MAX_DISK_ENTRIES]
            slim = {k: slim[k] for k in keep}
        with open(path, "w", encoding="utf-8") as f:
            import json as _json2
            _json2.dump(slim, f, ensure_ascii=False, indent=2)
    except Exception as _de:
        log.warning(f"_save_analysis_to_disk: {_de}")


def _load_analysis_from_disk():
    """Incarca _analysis_cache de pe disc la pornirea serverului."""
    try:
        path = _analysis_cache_file()
        if not _os.path.exists(path):
            return
        import json as _json2
        from datetime import datetime as _dt3, timezone as _tz3, timedelta as _td3
        cutoff = _dt3.now(_tz3.utc) - _td3(days=_TTL_DAYS)
        with open(path, "r", encoding="utf-8") as f:
            data = _json2.load(f)
        loaded = 0
        with _analysis_cache_lock:
            for cid, snap in data.items():
                ts_str = snap.get("timestamp", "")
                try:
                    ts = _dt3.fromisoformat(ts_str)
                    if ts.tzinfo is None:
                        ts = ts.replace(tzinfo=_tz3.utc)
                    if ts < cutoff:
                        continue
                except Exception:
                    pass
                _analysis_cache[cid] = snap
                loaded += 1
        log.info(f"Analysis cache: {loaded} intrari incarcate din {path}")
    except Exception as _de:
        log.warning(f"_load_analysis_from_disk: {_de}")


def _save_ticket_chart_to_disk():
    """Scrie _ticket_chart_map pe disc."""
    try:
        path = _ticket_chart_file()
        with _ticket_chart_map_lock:
            data = {str(k): v for k, v in _ticket_chart_map.items()}
        import json as _json2
        with open(path, "w", encoding="utf-8") as f:
            _json2.dump(data, f, indent=2)
    except Exception as _de:
        log.warning(f"_save_ticket_chart_to_disk: {_de}")


def _load_ticket_chart_from_disk():
    """Incarca _ticket_chart_map de pe disc la pornire."""
    try:
        path = _ticket_chart_file()
        if not _os.path.exists(path):
            return
        import json as _json2
        with open(path, "r", encoding="utf-8") as f:
            data = _json2.load(f)
        with _ticket_chart_map_lock:
            for k, v in data.items():
                _ticket_chart_map[int(k)] = v
        log.info(f"Ticket→chart map: {len(data)} intrari incarcate din {path}")
    except Exception as _de:
        log.warning(f"_load_ticket_chart_from_disk: {_de}")


def _link_ticket_to_chart(msg: str, chart_id: str):
    """Parseaza ticket-ul din mesajul 'OK #12345 — ...' si il mapeaza la chart_id."""
    try:
        if not msg.startswith("OK #"):
            return
        ticket = int(msg.split("#")[1].split(" ")[0])
        with _ticket_chart_map_lock:
            _ticket_chart_map[ticket] = chart_id
            if len(_ticket_chart_map) > 500:
                oldest = sorted(_ticket_chart_map.keys())[:100]
                for k in oldest:
                    del _ticket_chart_map[k]
        _save_ticket_chart_to_disk()
    except Exception:
        pass


def _store_analysis(symbol, strategy, signal, tf, price, sl, tp, confidence,
                    justification, scan_result=None, voters=None, vote_map=None):
    """Stocheaza snapshot-ul analizei unui semnal in memorie + pe disc."""
    from datetime import datetime as _dt2
    ts = _dt2.now()
    chart_id = f"{ts.strftime('%Y%m%d%H%M%S')}_{symbol}_{strategy}_{signal}"

    # Extrage datele per-TF/per-strategie necesare pentru tab-uri, separate de
    # scan_result (care nu e persistat pe disc).  Aceste date sunt mici si
    # supravietuiesc repornirii serverului.
    _sr = scan_result or {}
    _raw_tfs = _sr.get("tfs") or []
    tfs_slim = [
        {
            "tf":         t.get("tf", "?"),
            "signal":     t.get("signal", "HOLD"),
            "conviction": t.get("conviction", 0),
            "sl":         t.get("sl"),
            "tp":         t.get("tp"),
            "price":      t.get("price"),
            "reasons":    (t.get("reasons") or [])[:12],
        }
        for t in _raw_tfs
    ]
    # Pentru combined/scalp_boost: strat_votes + strat_details
    strat_votes   = _sr.get("strat_votes") or []
    strat_details = _sr.get("strat_details") or {}

    snap = {
        "symbol":        symbol,
        "strategy":      strategy,
        "signal":        signal,
        "tf":            tf or "H1",
        "price":         price,
        "sl":            sl,
        "tp":            tp,
        "confidence":    confidence,
        "justification": justification or [],
        "voters":        voters or [],
        "vote_map":      vote_map or {},
        "timestamp":     ts.isoformat(),
        "tfs_slim":      tfs_slim,
        "strat_votes":   strat_votes,
        "strat_details": strat_details,
        "best_tf":       _sr.get("best_tf"),
        "scan_result":   scan_result,   # doar in memorie, stripit pe disc
    }
    with _analysis_cache_lock:
        _analysis_cache[chart_id] = snap
        if len(_analysis_cache) > _MAX_ANALYSIS_CACHE:
            oldest = sorted(_analysis_cache.keys())[:50]
            for k in oldest:
                del _analysis_cache[k]
    # Salveaza pe disc intr-un thread separat (nu blocheaza scanner-ul)
    threading.Thread(target=_save_analysis_to_disk, daemon=True).start()
    return chart_id


# ── Incarca datele la import (cand Flask porneste) ────────────────────────────
def _bootstrap_cache():
    """Apelat o singura data la pornire — incarca cache de pe disc."""
    _load_analysis_from_disk()
    _load_ticket_chart_from_disk()


# Incearca sa incarce imediat; daca MT5 nu e gata inca, se va reincerca la primul trade
try:
    _bootstrap_cache()
except Exception:
    pass


def _analysis_tab_items(snap):
    """
    Returneaza o lista de tab-uri pentru pagina de analiza:
    [{key, label, signal, conviction, sl, tp, price, reasons, is_best}]
    key = TF (ex. 'H1') sau strat_key pentru combined/scalp_boost.

    Sursele de date (in ordine de prioritate):
      1. snap["tfs_slim"]      — extras la store-time, persistent pe disc
      2. snap["scan_result"]   — in memorie (pierdut la restart server)
    """
    sr       = snap.get("scan_result") or {}
    strategy = snap.get("strategy", "")
    signal   = snap.get("signal", "HOLD")

    # ── Combined / ScalpBoost: tab per strategie ──────────────────────────────
    if strategy in ("combined", "scalp_boost"):
        # Prefer datele direct din snap (persistente)
        strat_votes   = snap.get("strat_votes") or sr.get("strat_votes") or []
        strat_details = snap.get("strat_details") or sr.get("strat_details") or {}
        tabs = []
        for v in strat_votes:
            strat_key = v.get("strat", "?")
            det       = strat_details.get(strat_key, {})
            reasons   = det.get("reasons") or []
            tabs.append({
                "key":        strat_key,
                "label":      strat_key.upper(),
                "tf":         v.get("tf") or det.get("best_tf") or snap.get("tf") or "H1",
                "signal":     v.get("signal", "HOLD"),
                "conviction": v.get("conviction", 0),
                "sl":         v.get("sl"),
                "tp":         v.get("tp"),
                "price":      v.get("price"),
                "reasons":    reasons,
                "is_best":    v.get("signal") == signal,
            })
        if not tabs:
            tabs = [{
                "key": snap.get("tf", "H1"), "label": snap.get("tf", "H1"),
                "tf": snap.get("tf", "H1"), "signal": signal,
                "conviction": 0, "sl": snap.get("sl"), "tp": snap.get("tp"),
                "price": snap.get("price"), "reasons": snap.get("justification", []),
                "is_best": True,
            }]
        return tabs

    # ── Individual strategy: tab per TF ──────────────────────────────────────
    # Prefer snap["tfs_slim"] (persistent) fata de scan_result["tfs"] (in-memory)
    tfs_list = snap.get("tfs_slim") or sr.get("tfs") or []
    if not tfs_list:
        # fallback: un singur tab cu datele principale
        best_tf = (snap.get("best_tf") or sr.get("best_tf") or {})
        return [{
            "key":        snap.get("tf", "H1"),
            "label":      snap.get("tf", "H1"),
            "tf":         snap.get("tf", "H1"),
            "signal":     signal,
            "conviction": best_tf.get("conviction", 0),
            "sl":         snap.get("sl"),
            "tp":         snap.get("tp"),
            "price":      snap.get("price"),
            "reasons":    snap.get("justification", []),
            "is_best":    True,
        }]

    best_tf_name = (snap.get("best_tf") or sr.get("best_tf") or {}).get("tf") or snap.get("tf")
    tabs = []
    for t in tfs_list:
        tf_name = t.get("tf", "?")
        tabs.append({
            "key":        tf_name,
            "label":      tf_name,
            "tf":         tf_name,
            "signal":     t.get("signal", "HOLD"),
            "conviction": t.get("conviction", 0),
            "sl":         t.get("sl"),
            "tp":         t.get("tp"),
            "price":      t.get("price"),
            "reasons":    t.get("reasons") or [],
            "is_best":    tf_name == best_tf_name,
        })
    tabs.sort(key=lambda x: (0 if x["is_best"] else 1, x["key"]))
    return tabs


def _build_analysis_page(chart_id):
    """Construieste pagina HTML completa de analiza cu tab per TF/strategie."""
    with _analysis_cache_lock:
        snap = _analysis_cache.get(chart_id)
    if not snap:
        return (
            "<html><body style='background:#0d1117;color:#ef5350;"
            "font-family:sans-serif;padding:40px'>"
            "<h2>Analiza nu mai este disponibila (cache expirat sau server repornit).</h2>"
            "<p style='color:#8b949e;margin-top:12px'>"
            "Analiza este pastrata in memorie doar in sesiunea curenta.</p>"
            "<button onclick='window.close()' style='margin-top:16px;padding:8px 18px;"
            "background:#21262d;border:1px solid #30363d;color:#c9d1d9;"
            "border-radius:6px;cursor:pointer'>Inchide</button>"
            "</body></html>"
        )

    symbol   = snap["symbol"]
    strategy = snap["strategy"]
    signal   = snap["signal"]
    conf     = snap.get("confidence", 0)
    just     = snap.get("justification", [])
    voters   = snap.get("voters", [])
    vote_map = snap.get("vote_map", {})
    ts_str   = snap["timestamp"][:19].replace("T", " ")
    price    = snap.get("price")
    sl       = snap.get("sl")
    tp       = snap.get("tp")

    sig_clr   = "#26a69a" if signal == "BUY" else "#ef5350" if signal == "SELL" else "#888"
    conf_pct  = min(100.0, max(0.0, conf))
    price_fmt = f"{price:.5f}" if price else "—"
    sl_fmt    = f"{sl:.5f}"    if sl    else "—"
    tp_fmt    = f"{tp:.5f}"    if tp    else "—"
    rr_str    = ""
    if price and sl and tp:
        _risk = abs(price - sl)
        if _risk > 0:
            rr_str = f"1:{abs(tp - price) / _risk:.2f}"

    # ── Tab items (per TF sau per strategie pentru combined) ──────────────────
    tabs = _analysis_tab_items(snap)

    # ── Vote map HTML ─────────────────────────────────────────────────────────
    vm = vote_map or {}
    vote_html = ""
    if vm.get("total"):
        _bp = round(vm.get("buy", 0) / vm["total"] * 100)
        _sp = round(vm.get("sell", 0) / vm["total"] * 100)
        vote_html = (
            f'<div style="display:flex;height:7px;background:#21262d;border-radius:4px;'
            f'overflow:hidden;margin:10px 0 4px">'
            f'<div style="width:{_bp}%;background:#26a69a"></div>'
            f'<div style="width:{_sp}%;background:#ef5350"></div></div>'
            f'<div style="font-size:.74rem;color:#8b949e">'
            f'BUY {vm.get("buy",0)}/{vm["total"]} &nbsp;·&nbsp; '
            f'SELL {vm.get("sell",0)}/{vm["total"]} &nbsp;·&nbsp; '
            f'HOLD {vm.get("hold",0)}/{vm["total"]}</div>'
        )

    voters_html = ""
    if voters:
        v_chips = "".join(
            f'<span style="background:#21262d;border:1px solid #30363d;border-radius:4px;'
            f'padding:2px 7px;font-size:.71rem;color:#8b949e">{v}</span>'
            for v in voters
        )
        voters_html = f'<div style="margin-top:8px;display:flex;flex-wrap:wrap;gap:4px">{v_chips}</div>'

    # ── Build tab buttons + per-tab data (JSON embedded in HTML) ─────────────
    import json as _json_mod
    tabs_data = {}
    tab_btns  = []
    for i, t in enumerate(tabs):
        k        = t["key"]
        lbl      = t["label"]
        t_sig    = t["signal"]
        t_clr    = "#26a69a" if t_sig == "BUY" else "#ef5350" if t_sig == "SELL" else "#666"
        is_best  = t.get("is_best", False)
        best_tag = " ★" if is_best else ""
        t_conv   = t.get("conviction", 0)
        t_sl     = t.get("sl")
        t_tp     = t.get("tp")
        t_price  = t.get("price")
        t_rr     = ""
        if t_price and t_sl and t_tp:
            _r = abs(t_price - t_sl)
            if _r > 0:
                t_rr = f"1:{abs(t_tp - t_price)/_r:.2f}"

        tabs_data[k] = {
            "tf":        t.get("tf", k),
            "signal":    t_sig,
            "conviction": t_conv,
            "reasons":   t.get("reasons") or [],
            "sl":        t_sl,
            "tp":        t_tp,
            "price":     t_price,
            "rr":        t_rr,
        }

        active_cls = "tab-btn active" if i == 0 else "tab-btn"
        tab_btns.append(
            f'<button class="{active_cls}" onclick="switchTab(\'{k}\')" id="tbtn-{k}" '
            f'style="border-color:{t_clr}22;color:{t_clr if t_sig != "HOLD" else "#555"}">'
            f'{lbl}{best_tag}'
            f'<span class="tab-sig" style="background:{t_clr}22;color:{t_clr};'
            f'border:1px solid {t_clr}44">{t_sig}</span>'
            f'<span class="tab-conv" style="color:#484f58">conv {t_conv}</span>'
            f'</button>'
        )

    tabs_json  = _json_mod.dumps(tabs_data, ensure_ascii=False)
    first_key  = tabs[0]["key"] if tabs else ""
    first_tf   = tabs[0].get("tf", first_key) if tabs else snap.get("tf", "H1")

    return f"""<!DOCTYPE html>
<html lang="ro">
<head>
<meta charset="utf-8">
<title>Analiza {symbol} {signal} — {strategy}</title>
<script src="https://cdn.plot.ly/plotly-2.35.2.min.js" charset="utf-8"></script>
<style>
*{{box-sizing:border-box;margin:0;padding:0}}
body{{background:#0d1117;color:#c9d1d9;font-family:'Segoe UI',sans-serif;
      display:flex;flex-direction:column;height:100vh;overflow:hidden}}
.topbar{{background:#161b22;border-bottom:1px solid #21262d;
         padding:10px 16px;display:flex;align-items:center;gap:10px;flex-wrap:wrap;flex-shrink:0}}
.sym{{font-size:1.2rem;font-weight:700}}
.badge{{padding:2px 12px;border-radius:14px;font-size:.85rem;font-weight:700;
        background:{sig_clr}22;border:2px solid {sig_clr};color:{sig_clr}}}
.tag{{padding:2px 8px;border-radius:8px;font-size:.75rem;
      background:#0d1117;border:1px solid #30363d;color:#8b949e}}
.ts{{font-size:.71rem;color:#484f58;margin-left:auto}}
.back{{padding:4px 11px;background:#21262d;border:1px solid #30363d;border-radius:5px;
       color:#8b949e;font-size:.78rem;cursor:pointer;text-decoration:none;flex-shrink:0}}
.back:hover{{background:#30363d;color:#c9d1d9}}

/* ── Tab bar ── */
.tabbar{{background:#161b22;border-bottom:1px solid #21262d;
         padding:0 14px;display:flex;gap:4px;overflow-x:auto;flex-shrink:0}}
.tab-btn{{display:flex;align-items:center;gap:6px;padding:8px 12px;
          background:none;border:none;border-bottom:2px solid transparent;
          color:#8b949e;font-size:.8rem;cursor:pointer;white-space:nowrap;
          transition:color .15s,border-color .15s;font-family:inherit}}
.tab-btn:hover{{color:#c9d1d9}}
.tab-btn.active{{border-bottom-color:{sig_clr};color:#c9d1d9}}
.tab-sig{{padding:1px 6px;border-radius:4px;font-size:.68rem;font-weight:700}}
.tab-conv{{font-size:.67rem}}

/* ── Content ── */
.content{{display:flex;flex:1;overflow:hidden}}
.chart-col{{flex:1;display:flex;flex-direction:column;overflow:hidden;padding:8px}}
#chart-div{{flex:1;min-height:0}}
.chart-loading{{display:flex;align-items:center;justify-content:center;
                height:100%;color:#484f58;font-size:.9rem}}

.side-col{{width:320px;flex-shrink:0;overflow-y:auto;
           background:#0d1117;border-left:1px solid #21262d;padding:12px}}
@media(max-width:900px){{
  .content{{flex-direction:column}}
  .side-col{{width:100%;border-left:none;border-top:1px solid #21262d;
             overflow-y:visible;max-height:260px}}
}}

.panel{{background:#161b22;border:1px solid #21262d;border-radius:7px;
        padding:12px;margin-bottom:10px}}
.panel h4{{font-size:.69rem;text-transform:uppercase;color:#484f58;letter-spacing:.6px;
           margin-bottom:9px;border-bottom:1px solid #21262d;padding-bottom:5px}}
.cbar-wrap{{background:#21262d;border-radius:4px;height:6px;margin-bottom:5px;overflow:hidden}}
.cbar{{height:100%;background:{sig_clr};border-radius:4px}}
.clabel{{font-size:.88rem;font-weight:700;color:{sig_clr}}}
.pgrid{{display:grid;grid-template-columns:1fr 1fr 1fr;gap:6px;margin-top:8px}}
.pitem{{background:#0d1117;border-radius:5px;padding:7px 4px;text-align:center}}
.pitem .pl{{font-size:.62rem;color:#484f58;margin-bottom:2px}}
.pitem .pv{{font-size:.82rem;font-weight:600}}
.rrbadge{{display:inline-block;background:#1f2937;border:1px solid #374151;
          padding:3px 10px;border-radius:5px;font-size:.82rem;font-weight:600;
          color:#d1d5db;margin-top:8px}}
ul.jlist{{list-style:none}}
ul.jlist li{{padding:4px 0;border-bottom:1px solid #21262d;
             font-size:.78rem;color:#8b949e;line-height:1.4}}
ul.jlist li:last-child{{border-bottom:none}}
ul.jlist li::before{{content:"▸ ";color:#30363d}}
.legend-row{{display:flex;gap:10px;flex-wrap:wrap;font-size:.68rem;color:#484f58;
             padding:4px 8px;flex-shrink:0}}
.leg-dot{{display:inline-block;width:9px;height:9px;border-radius:2px;margin-right:3px}}

/* per-tf detail (updated by JS) */
#tf-detail-sl{{color:#ef5350;font-weight:600}}
#tf-detail-tp{{color:#26a69a;font-weight:600}}
#tf-detail-price{{color:{sig_clr};font-weight:600}}
</style>
</head>
<body>

<!-- Top bar -->
<div class="topbar">
  <a class="back" onclick="window.close()">✕</a>
  <span class="sym">{symbol}</span>
  <span class="badge">{signal}</span>
  <span class="tag">{strategy.upper()}</span>
  <span class="ts">{ts_str}</span>
</div>

<!-- Tab bar -->
<div class="tabbar">{"".join(tab_btns)}</div>

<!-- Main content -->
<div class="content">

  <!-- Chart column -->
  <div class="chart-col">
    <div id="chart-div"><div class="chart-loading">Se incarca graficul...</div></div>
    <div class="legend-row">
      <span><span class="leg-dot" style="background:{sig_clr}"></span>Entry</span>
      <span><span class="leg-dot" style="background:#ef5350"></span>Stop Loss</span>
      <span><span class="leg-dot" style="background:#26a69a"></span>Take Profit</span>
      <span><span class="leg-dot" style="background:rgba(156,39,176,.55)"></span>OB Bear</span>
      <span><span class="leg-dot" style="background:rgba(33,150,243,.55)"></span>OB Bull</span>
      <span><span class="leg-dot" style="background:rgba(255,193,7,.45)"></span>FVG</span>
    </div>
  </div>

  <!-- Side panel -->
  <div class="side-col">

    <!-- Incredere globala -->
    <div class="panel">
      <h4>Incredere globala</h4>
      <div class="cbar-wrap"><div class="cbar" style="width:{conf_pct:.1f}%"></div></div>
      <div class="clabel">{conf_pct:.1f}%</div>
      <div class="pgrid">
        <div class="pitem">
          <div class="pl">ENTRY</div>
          <div class="pv" id="tf-detail-price">{price_fmt}</div>
        </div>
        <div class="pitem">
          <div class="pl">STOP LOSS</div>
          <div class="pv" id="tf-detail-sl">{sl_fmt}</div>
        </div>
        <div class="pitem">
          <div class="pl">TAKE PROFIT</div>
          <div class="pv" id="tf-detail-tp">{tp_fmt}</div>
        </div>
      </div>
      {f'<div class="rrbadge" id="tf-rr-badge">R:R &nbsp;{rr_str}</div>' if rr_str else '<div class="rrbadge" id="tf-rr-badge" style="display:none"></div>'}
      {vote_html}
      {voters_html}
    </div>

    <!-- Motive per TF (actualizate de JS) -->
    <div class="panel">
      <h4 id="reasons-title">Motive — <span id="reasons-tf-lbl">{first_key}</span></h4>
      <ul class="jlist" id="reasons-list">
        <li style="color:#484f58">Se incarca...</li>
      </ul>
    </div>

  </div>
</div>

<script>
const CHART_ID  = {_json_mod.dumps(chart_id)};
const SYMBOL    = {_json_mod.dumps(symbol)};
const STRATEGY  = {_json_mod.dumps(strategy)};
const SIGNAL    = {_json_mod.dumps(signal)};
const TABS_DATA = {tabs_json};
let   activeKey = {_json_mod.dumps(first_key)};
const chartLoaded = {{}};

function sigClr(s) {{
  return s === "BUY" ? "#26a69a" : s === "SELL" ? "#ef5350" : "#666";
}}

function switchTab(key) {{
  activeKey = key;
  // update tab styling
  document.querySelectorAll(".tab-btn").forEach(b => b.classList.remove("active"));
  const btn = document.getElementById("tbtn-" + key);
  if (btn) btn.classList.add("active");

  const t = TABS_DATA[key];
  if (!t) return;

  // update side panel levels
  const fmt = v => v != null ? v.toFixed(5) : "—";
  document.getElementById("tf-detail-price").textContent = fmt(t.price);
  document.getElementById("tf-detail-sl").textContent    = fmt(t.sl);
  document.getElementById("tf-detail-tp").textContent    = fmt(t.tp);
  const rrEl = document.getElementById("tf-rr-badge");
  if (rrEl) {{
    if (t.rr) {{ rrEl.textContent = "R:R  " + t.rr; rrEl.style.display = "inline-block"; }}
    else       {{ rrEl.style.display = "none"; }}
  }}

  // update reasons
  document.getElementById("reasons-tf-lbl").textContent = key;
  const ul = document.getElementById("reasons-list");
  if (t.reasons && t.reasons.length) {{
    ul.innerHTML = t.reasons.map(r => `<li>${{r}}</li>`).join("");
  }} else {{
    ul.innerHTML = "<li style='color:#484f58'>Fara motive inregistrate pe acest TF.</li>";
  }}

  // load chart (cached after first load)
  if (chartLoaded[key]) {{
    Plotly.react("chart-div", chartLoaded[key].data, chartLoaded[key].layout, {{responsive:true}});
    return;
  }}
  document.getElementById("chart-div").innerHTML =
    "<div class='chart-loading'>Se incarca " + key + "...</div>";

  fetch(`/autotrader/analysis_chart/${{CHART_ID}}/${{encodeURIComponent(key)}}`)
    .then(r => r.json())
    .then(fig => {{
      chartLoaded[key] = fig;
      Plotly.newPlot("chart-div", fig.data, fig.layout, {{responsive:true, displayModeBar:false}});
    }})
    .catch(e => {{
      document.getElementById("chart-div").innerHTML =
        "<div class='chart-loading' style='color:#ef5350'>Eroare: " + e.message + "</div>";
    }});
}}

// load first tab on page start
switchTab({_json_mod.dumps(first_key)});
</script>
</body></html>"""


@autotrader_bp.route("/autotrader/analysis/<chart_id>")
@login_required
def autotrader_analysis_page(chart_id):
    """Pagina HTML completa de analiza pentru un semnal stocat."""
    html = _build_analysis_page(chart_id)
    return Response(html, content_type="text/html; charset=utf-8")


@autotrader_bp.route("/autotrader/analysis_chart/<chart_id>/<tab_key>")
@login_required
def autotrader_analysis_chart(chart_id, tab_key):
    """
    Returneaza figura Plotly JSON pentru un tab specific (TF sau strategie) din analiza.
    Overlays: Entry/SL/TP din snapshot + OB/FVG live din MT5.
    """
    with _analysis_cache_lock:
        snap = _analysis_cache.get(chart_id)
    if not snap:
        return Response('{"error":"Analiza expirata"}',
                        content_type="application/json", status=404)

    symbol   = snap["symbol"]
    strategy = snap["strategy"]
    signal   = snap["signal"]
    sig_clr  = "#26a69a" if signal == "BUY" else "#ef5350"

    # Rezolva TF real si datele tab-ului
    tabs     = _analysis_tab_items(snap)
    tab_data = next((t for t in tabs if t["key"] == tab_key), None)
    if tab_data is None:
        # fallback: tab_key IS the TF
        tab_data = {"tf": tab_key, "sl": snap.get("sl"), "tp": snap.get("tp"),
                    "price": snap.get("price"), "signal": signal}

    tf    = tab_data.get("tf") or snap.get("tf") or "H1"
    t_sl  = tab_data.get("sl")
    t_tp  = tab_data.get("tp")
    t_px  = tab_data.get("price")
    t_sig = tab_data.get("signal", signal)

    if tf not in ALL_TFS:
        tf = "H1"

    fig = _build_trade_chart_fig(symbol, tf, 120)
    if fig is None:
        return Response('{"error":"No data"}', content_type="application/json", status=500)

    # ── Entry / SL / TP overlay ───────────────────────────────────────────────
    t_clr = "#26a69a" if t_sig == "BUY" else "#ef5350" if t_sig == "SELL" else sig_clr
    if t_px:
        fig.add_hline(y=t_px, line=dict(color=t_clr, width=2, dash="dot"), row=1, col=1)
        fig.add_annotation(
            xref="paper", yref="y", x=0.01, y=t_px,
            text=f"<b>ENTRY {t_sig}</b>  {t_px:.5f}",
            showarrow=False, font=dict(color=t_clr, size=11),
            xanchor="left", yanchor="bottom",
            bgcolor="rgba(13,17,23,0.92)", bordercolor=t_clr, borderwidth=1,
            row=1, col=1,
        )
    if t_sl:
        fig.add_hline(y=t_sl, line=dict(color="#ef5350", width=1.5, dash="dash"),
                      row=1, col=1)
        fig.add_annotation(
            xref="paper", yref="y", x=0.01, y=t_sl,
            text=f"SL  {t_sl:.5f}", showarrow=False,
            font=dict(color="#ef5350", size=9), xanchor="left", yanchor="top",
            bgcolor="rgba(13,17,23,0.85)", row=1, col=1,
        )
        if t_px:
            fig.add_hrect(y0=min(t_px, t_sl), y1=max(t_px, t_sl), row=1, col=1,
                          fillcolor="rgba(239,83,80,0.09)", line=dict(width=0))
    if t_tp:
        fig.add_hline(y=t_tp, line=dict(color="#26a69a", width=1.5, dash="dash"),
                      row=1, col=1)
        fig.add_annotation(
            xref="paper", yref="y", x=0.01, y=t_tp,
            text=f"TP  {t_tp:.5f}", showarrow=False,
            font=dict(color="#26a69a", size=9), xanchor="left", yanchor="top",
            bgcolor="rgba(13,17,23,0.85)", row=1, col=1,
        )
        if t_px:
            fig.add_hrect(y0=min(t_px, t_tp), y1=max(t_px, t_tp), row=1, col=1,
                          fillcolor="rgba(38,166,154,0.07)", line=dict(width=0))

    # ── OB + FVG overlay (pentru strategii cu structure) ─────────────────────
    if strategy in ("eob", "smc", "combined", "scalp_boost"):
        try:
            df_ov, _ = fetch(symbol, tf, 200)
            if df_ov is not None and len(df_ov) > 50:
                for ob in (find_order_blocks(df_ov) or [])[-5:]:
                    _oh = ob.get("high") or ob.get("hi")
                    _ol = ob.get("low")  or ob.get("lo")
                    _ot = ob.get("type", "")
                    if _oh and _ol:
                        _fc = ("rgba(156,39,176,0.17)" if "bear" in _ot.lower()
                               else "rgba(33,150,243,0.17)")
                        _bc = ("rgba(156,39,176,0.65)" if "bear" in _ot.lower()
                               else "rgba(33,150,243,0.65)")
                        fig.add_hrect(y0=_ol, y1=_oh, row=1, col=1,
                                      fillcolor=_fc,
                                      line=dict(color=_bc, width=1, dash="dot"))
                for fvg in (find_fvg(df_ov) or [])[-4:]:
                    _fh = fvg.get("hi") or fvg.get("high")
                    _fl = fvg.get("lo") or fvg.get("low")
                    if _fh and _fl:
                        fig.add_hrect(y0=_fl, y1=_fh, row=1, col=1,
                                      fillcolor="rgba(255,193,7,0.09)",
                                      line=dict(color="rgba(255,193,7,0.32)",
                                                width=1, dash="dot"))
        except Exception as _e:
            log.debug(f"analysis_chart overlay {symbol}/{tf}: {_e}")

    fig.update_layout(
        title=dict(
            text=(f"<b>{symbol}</b>  ·  {t_sig}  ·  "
                  f"{strategy.upper()}  ·  {tf}"),
            font=dict(size=13, color=t_clr),
        ),
        height=None,
        autosize=True,
        margin=dict(l=50, r=90, t=45, b=25),
    )
    return Response(fig.to_json(), content_type="application/json")


@autotrader_bp.route("/autotrader/chartjson/<symbol>/<tf>")
@login_required
def autotrader_chartjson(symbol, tf):
    """Returneaza figura Plotly ca JSON — folosit de Plotly.react() client-side."""
    if tf not in ALL_TFS:
        return Response('{"error":"TF invalid"}', content_type="application/json", status=400)
    bars = int(request.args.get("bars", 300))
    fig  = _build_trade_chart_fig(symbol, tf, bars)
    if fig is None:
        return Response('{"error":"No data"}', content_type="application/json", status=500)
    return Response(fig.to_json(), content_type="application/json")


@autotrader_bp.route("/autotrader/chart/<symbol>/<tf>")
@login_required
def autotrader_chart(symbol, tf):
    """Fallback HTML iframe route — pastrat pentru compatibilitate."""
    if tf not in ALL_TFS:
        return Response("<div style='color:#ef5350;padding:12px'>TF invalid.</div>",
                        content_type="text/html")
    bars = int(request.args.get("bars", 300))
    fig  = _build_trade_chart_fig(symbol, tf, bars)
    if fig is None:
        chart_html = f"<div style='color:#ef5350;padding:12px'>Nu s-au putut incarca datele pentru {symbol}/{tf}.</div>"
    else:
        chart_html = fig.to_html(full_html=False, include_plotlyjs="cdn")
    full_html = f"""<!DOCTYPE html>
<html><head><meta charset="utf-8">
<style>* {{ margin:0; padding:0; box-sizing:border-box; }} body {{ background:#111; }}</style>
</head><body>{chart_html}</body></html>"""
    return Response(full_html, content_type="text/html; charset=utf-8")
