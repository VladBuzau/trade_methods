"""
Strategie Clasica: EMA trend + Fibonacci zone + ADX putere + RSI neutru.
Migrata din autotrader.py → analyze_symbol_full().
"""
from __future__ import annotations
import logging
import re as _re

import numpy as np

from strategies.base import Strategy

log = logging.getLogger(__name__)


class ClassicStrategy(Strategy):
    key   = "classic"
    name  = "Clasica"
    icon  = "🔵"
    color = "#26a69a"

    default_tfs  = ["M5", "M15", "H1"]
    default_bars = 500
    elements     = {
        "ema": "EMA (trend aliniat)",
        "fib": "FIB (Fibonacci zone)",
        "adx": "ADX (forta trend)",
        "rsi": "RSI (zona neutra)",
    }

    def analyze(self, symbol, tfs, bars=500, tf_bars=None, elements=None,
                min_confidence=66.0, **kwargs):
        from app import (
            fetch, find_pivots, detect_trend, calc_entry, calc_sl_tp,
            get_h4_direction, in_trading_session, MIN_TF_VOTES, MIN_CONFIDENCE,
        )
        import app as _app

        if elements is None:
            elements = {k: True for k in self.elements}

        use_h4  = kwargs.get("use_h4_filter", True)
        use_ses = kwargs.get("use_session_filter", True)
        h4_dir  = get_h4_direction(symbol) if use_h4 else "ANY"
        session_ok = in_trading_session() if use_ses else True

        tf_results = []
        for tf in tfs:
            try:
                n_bars = (tf_bars or {}).get(tf, bars)
                df, _  = fetch(symbol, tf, n_bars)
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

                tf_results.append({
                    "tf":         tf,
                    "signal":     signal,
                    "trend":      trend,
                    "conviction": len(reasons),
                    "reasons":    reasons,
                    "price":      round(float(price), 5),
                    "sl":         sl,
                    "tp":         tp,
                })
            except Exception as exc:
                log.warning(f"ClassicStrategy {symbol}/{tf}: {exc}")

        if not tf_results:
            return self._empty_result(symbol, "Fara date suficiente")

        # Vot majoritar cu filtru H4
        buy_v  = [r for r in tf_results if r["signal"] == "BUY"]
        sell_v = [r for r in tf_results if r["signal"] == "SELL"]
        n_buy, n_sell, n_total = len(buy_v), len(sell_v), len(tf_results)
        confidence = self._confidence(max(n_buy, n_sell), n_total)

        final_signal = "HOLD"
        best_tf = None
        min_votes = kwargs.get("min_votes", MIN_TF_VOTES)

        if n_buy >= min_votes and n_buy > n_sell and confidence >= min_confidence:
            if h4_dir in ("BUY", "ANY"):
                final_signal = "BUY"
                best_tf = max(buy_v, key=lambda x: x["conviction"])
        elif n_sell >= min_votes and n_sell > n_buy and confidence >= min_confidence:
            if h4_dir in ("SELL", "ANY"):
                final_signal = "SELL"
                best_tf = max(sell_v, key=lambda x: x["conviction"])

        # Justificare
        justification = []
        trend_counts = {}
        for r in tf_results:
            trend_counts[r["trend"]] = trend_counts.get(r["trend"], 0) + 1
        dominant_trend = max(trend_counts, key=trend_counts.get) if trend_counts else "RANGING"
        trend_ro = {"ASCENDING": "ASCENDENT", "DESCENDING": "DESCENDENT", "RANGING": "LATERAL"}.get(dominant_trend, dominant_trend)
        justification.append(f"Trend {trend_ro} confirmat pe {trend_counts.get(dominant_trend,0)} din {n_total} TF-uri")

        for r in tf_results:
            if r["signal"] != "HOLD" and r["reasons"]:
                t_ro = {"ASCENDING": "ASCENDENT", "DESCENDING": "DESCENDENT", "RANGING": "LATERAL"}.get(r["trend"], r["trend"])
                clean = [x.replace(" ✓", "") for x in r["reasons"]]
                justification.append(f"{r['tf']}: Trend {t_ro}, {', '.join(clean)}")

        if final_signal == "HOLD":
            if h4_dir is None:
                justification.append("H4 lateral / ADX slab — asteapta trend clar")
            elif n_buy >= min_votes and confidence >= min_confidence and h4_dir == "SELL":
                justification.append("BUY blocat — H4 e BEARISH (contra-trend)")
            elif n_sell >= min_votes and confidence >= min_confidence and h4_dir == "BUY":
                justification.append("SELL blocat — H4 e BULLISH (contra-trend)")
            else:
                justification.append(
                    f"Semnale insuficiente: {n_buy} BUY / {n_sell} SELL pe {n_total} TF-uri"
                )

        if not session_ok:
            from datetime import datetime as _dt, timezone as _tz
            _now = _dt.now(_tz.utc).strftime("%H:%M")
            justification.append(f"⚠ In afara sesiunii ({_now} UTC)")

        return {
            "symbol":        symbol,
            "strategy":      self.key,
            "timestamp":     self._now(),
            "signal":        final_signal,
            "h4_dir":        h4_dir,
            "session_ok":    session_ok,
            "n_buy":         n_buy,
            "n_sell":        n_sell,
            "n_total":       n_total,
            "confidence":    confidence,
            "best_tf":       best_tf,
            "tfs":           tf_results,
            "justification": justification,
            "auto_executed": False,
        }
