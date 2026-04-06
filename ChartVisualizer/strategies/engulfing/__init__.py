"""
Engulfing / Pin Bar: lumânari de reversal la nivel cheie (suport/rezistenta).
Bullish Engulfing: lumânare verde care inglobeaza complet lumânarea rosie anterioara.
Bearish Engulfing: invers.
Pin Bar: fitil lung (>2x body) la un nivel cheie.
"""
from __future__ import annotations
import logging
import numpy as np
from strategies.base import Strategy

log = logging.getLogger(__name__)


class EngulfingStrategy(Strategy):
    key   = "engulfing"
    name  = "Engulfing / Pin Bar"
    icon  = "🕯️"
    color = "#ff7043"

    default_tfs  = ["H1", "H4", "D1"]
    default_bars = 200
    elements     = {
        "engulfing": "Engulfing candle (inglobare completa)",
        "pin_bar":   "Pin Bar (fitil lung la nivel cheie)",
        "key_level": "Confirmare la suport/rezistenta",
    }

    @staticmethod
    def _is_near_level(price, levels, tolerance_pct=0.003):
        for lvl in levels:
            if abs(price - lvl) / lvl < tolerance_pct:
                return True
        return False

    def analyze(self, symbol, tfs, bars=200, tf_bars=None, elements=None,
                min_confidence=66.0, **kwargs):
        from app import fetch, find_pivots, calc_sl_tp

        if elements is None:
            elements = {k: True for k in self.elements}

        tf_results = []
        for tf in tfs:
            try:
                n_bars = (tf_bars or {}).get(tf, bars)
                df, _  = fetch(symbol, tf, n_bars)
                if df is None or len(df) < 20:
                    continue

                o = df["open"].values
                h = df["high"].values
                l = df["low"].values
                c = df["close"].values
                price = float(c[-1])

                # Niveluri cheie: pivoti recenti
                ph_idx, pl_idx = find_pivots(df, lookback=5)
                key_levels = (
                    [float(h[i]) for i in ph_idx[-5:]] +
                    [float(l[i]) for i in pl_idx[-5:]]
                )

                reasons    = []
                conviction = 0
                sig        = "HOLD"

                # ── Engulfing ────────────────────────────────────────────
                if elements.get("engulfing", True) and len(c) >= 2:
                    prev_o, prev_c = o[-2], c[-2]
                    curr_o, curr_c = o[-1], c[-1]

                    bullish_engulf = (prev_c < prev_o and curr_c > curr_o and
                                      curr_o <= prev_c and curr_c >= prev_o)
                    bearish_engulf = (prev_c > prev_o and curr_c < curr_o and
                                      curr_o >= prev_c and curr_c <= prev_o)

                    if bullish_engulf:
                        reasons.append(f"Bullish Engulfing la {price:.5f}")
                        conviction += 2
                        sig = "BUY"
                    elif bearish_engulf:
                        reasons.append(f"Bearish Engulfing la {price:.5f}")
                        conviction += 2
                        sig = "SELL"

                # ── Pin Bar ───────────────────────────────────────────────
                if elements.get("pin_bar", True):
                    body   = abs(c[-1] - o[-1])
                    candle = h[-1] - l[-1]
                    upper_wick = h[-1] - max(c[-1], o[-1])
                    lower_wick = min(c[-1], o[-1]) - l[-1]

                    if candle > 0 and body > 0:
                        # Pin bar bullish: fitil inferior lung
                        if lower_wick > 2 * body and lower_wick > upper_wick * 1.5:
                            if sig != "SELL":  # nu suprascriem engulfing bearish
                                reasons.append(f"Bullish Pin Bar (fitil jos {lower_wick:.5f})")
                                conviction += 2
                                sig = "BUY"
                        # Pin bar bearish: fitil superior lung
                        elif upper_wick > 2 * body and upper_wick > lower_wick * 1.5:
                            if sig != "BUY":
                                reasons.append(f"Bearish Pin Bar (fitil sus {upper_wick:.5f})")
                                conviction += 2
                                sig = "SELL"

                if sig == "HOLD":
                    tf_results.append({"tf": tf, "signal": "HOLD", "conviction": 0,
                                       "reasons": [], "price": price, "sl": None, "tp": None})
                    continue

                # ── Confirmare la nivel cheie ──────────────────────────
                if elements.get("key_level", True) and key_levels:
                    if self._is_near_level(price, key_levels):
                        reasons.append("La nivel cheie (pivot)")
                        conviction += 1
                    else:
                        # Nu e la nivel cheie — semnal slab, reducem conviction
                        conviction = max(1, conviction - 1)
                        reasons.append("Nu e la nivel cheie — semnal mai slab")

                sl, tp = calc_sl_tp(df, ph_idx, pl_idx, sig, price)

                tf_results.append({
                    "tf":         tf,
                    "signal":     sig,
                    "conviction": conviction,
                    "reasons":    reasons,
                    "price":      round(price, 5),
                    "sl":         sl,
                    "tp":         tp,
                })
            except Exception as exc:
                log.warning(f"EngulfingStrategy {symbol}/{tf}: {exc}")

        if not tf_results:
            return self._empty_result(symbol, "Fara date suficiente")

        return self._build_result(symbol, tf_results, min_confidence, min_votes=1)
