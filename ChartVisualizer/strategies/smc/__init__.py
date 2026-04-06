"""
Strategie SMC (Smart Money Concepts): BOS + Order Block + FVG + Market Structure.
Migrata din autotrader.py → analyze_symbol_smc().
"""
from __future__ import annotations
import logging

from strategies.base import Strategy

log = logging.getLogger(__name__)


class SMCStrategy(Strategy):
    key   = "smc"
    name  = "SMC"
    icon  = "🟠"
    color = "#ff9800"

    default_tfs  = ["M15", "H1", "H4"]
    default_bars = 500
    elements     = {
        "bos":       "BOS (Break of Structure)",
        "ob":        "OB (Order Block)",
        "fvg":       "FVG (Fair Value Gap)",
        "structure": "STR (Market Structure)",
    }

    def analyze(self, symbol, tfs, bars=500, tf_bars=None, elements=None,
                min_confidence=66.0, **kwargs):
        from app import fetch, find_pivots, calc_entry_smc, calc_sl_tp

        if elements is None:
            elements = {k: True for k in self.elements}

        tf_results = []
        for tf in tfs:
            try:
                n_bars = (tf_bars or {}).get(tf, bars)
                df, _  = fetch(symbol, tf, n_bars)
                if df is None or len(df) < 50:
                    continue

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
                log.warning(f"SMCStrategy {symbol}/{tf}: {exc}")

        if not tf_results:
            return self._empty_result(symbol, "Fara date suficiente")

        return self._build_result(symbol, tf_results, min_confidence, min_votes=1)
