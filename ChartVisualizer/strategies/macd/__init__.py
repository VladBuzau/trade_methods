"""
Strategie MACD Cross: MACD linie > semnal + EMA200 filtru directie.
Timeframe recomandat: M15, H1, H4.
"""
from __future__ import annotations
import logging

import numpy as np

from strategies.base import Strategy

log = logging.getLogger(__name__)


class MACDStrategy(Strategy):
    key   = "macd"
    name  = "MACD Cross"
    icon  = "📈"
    color = "#7c4dff"

    default_tfs  = ["M15", "H1", "H4"]
    default_bars = 500
    elements     = {
        "macd_cross": "MACD crossover (linie > semnal)",
        "ema200":     "EMA 200 (filtru directie)",
        "histogram":  "Histogram (momentum confirmare)",
    }

    def analyze(self, symbol, tfs, bars=500, tf_bars=None, elements=None,
                min_confidence=66.0, **kwargs):
        from app import fetch, calc_sl_tp, find_pivots

        if elements is None:
            elements = {k: True for k in self.elements}

        tf_results = []
        for tf in tfs:
            try:
                n_bars = (tf_bars or {}).get(tf, bars)
                df, _  = fetch(symbol, tf, n_bars)
                if df is None or len(df) < 200:
                    continue

                close = df["close"]
                price = float(close.iloc[-1])

                # MACD (12, 26, 9)
                ema12 = close.ewm(span=12, adjust=False).mean()
                ema26 = close.ewm(span=26, adjust=False).mean()
                macd  = ema12 - ema26
                signal_line = macd.ewm(span=9, adjust=False).mean()
                histogram   = macd - signal_line

                # EMA 200
                ema200 = close.ewm(span=200, adjust=False).mean()

                macd_now  = float(macd.iloc[-1])
                macd_prev = float(macd.iloc[-2])
                sig_now   = float(signal_line.iloc[-1])
                sig_prev  = float(signal_line.iloc[-2])
                hist_now  = float(histogram.iloc[-1])
                hist_prev = float(histogram.iloc[-2])
                ema200_v  = float(ema200.iloc[-1])

                reasons    = []
                conviction = 0
                sig        = "HOLD"

                # Crossover detectie
                cross_up   = macd_prev < sig_prev and macd_now > sig_now
                cross_down = macd_prev > sig_prev and macd_now < sig_now

                if cross_up and elements.get("macd_cross", True):
                    reasons.append("MACD cross UP")
                    conviction += 2
                    sig = "BUY"
                elif cross_down and elements.get("macd_cross", True):
                    reasons.append("MACD cross DOWN")
                    conviction += 2
                    sig = "SELL"

                if sig == "HOLD":
                    tf_results.append({"tf": tf, "signal": "HOLD", "conviction": 0,
                                       "reasons": [], "price": price, "sl": None, "tp": None})
                    continue

                # EMA 200 filtru
                if elements.get("ema200", True):
                    if sig == "BUY" and price > ema200_v:
                        reasons.append("Pret deasupra EMA200")
                        conviction += 1
                    elif sig == "SELL" and price < ema200_v:
                        reasons.append("Pret sub EMA200")
                        conviction += 1
                    else:
                        sig = "HOLD"
                        reasons.append("EMA200 contra-trend — blocat")
                        tf_results.append({"tf": tf, "signal": "HOLD", "conviction": 0,
                                           "reasons": reasons, "price": price, "sl": None, "tp": None})
                        continue

                # Histogram momentum
                if elements.get("histogram", True):
                    if sig == "BUY" and hist_now > hist_prev:
                        reasons.append("Histogram creste (momentum BUY)")
                        conviction += 1
                    elif sig == "SELL" and hist_now < hist_prev:
                        reasons.append("Histogram scade (momentum SELL)")
                        conviction += 1

                ph_idx, pl_idx = find_pivots(df, lookback=5)
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
                log.warning(f"MACDStrategy {symbol}/{tf}: {exc}")

        if not tf_results:
            return self._empty_result(symbol, "Fara date suficiente")

        return self._build_result(symbol, tf_results, min_confidence, min_votes=1)
