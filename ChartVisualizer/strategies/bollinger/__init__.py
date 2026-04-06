"""
Strategie Bollinger Bands: pret atinge banda extrema + RSI confirmare + revenire la medie.
Timeframe recomandat: M15, H1, H4.
"""
from __future__ import annotations
import logging

import numpy as np

from strategies.base import Strategy

log = logging.getLogger(__name__)


class BollingerStrategy(Strategy):
    key   = "bollinger"
    name  = "Bollinger Bands"
    icon  = "🎯"
    color = "#00bcd4"

    default_tfs  = ["M15", "H1", "H4"]
    default_bars = 300
    elements     = {
        "band_touch":  "Atingere banda extrema (2σ)",
        "rsi_confirm": "RSI confirmare (30/70)",
        "squeeze":     "BB Squeeze (volatilitate scazuta)",
    }

    def analyze(self, symbol, tfs, bars=300, tf_bars=None, elements=None,
                min_confidence=66.0, **kwargs):
        from app import fetch, calc_sl_tp, find_pivots

        if elements is None:
            elements = {k: True for k in self.elements}

        tf_results = []
        for tf in tfs:
            try:
                n_bars = (tf_bars or {}).get(tf, bars)
                df, _  = fetch(symbol, tf, n_bars)
                if df is None or len(df) < 50:
                    continue

                close = df["close"]
                price = float(close.iloc[-1])

                # Bollinger Bands (20, 2)
                period = 20
                sma    = close.rolling(period).mean()
                std    = close.rolling(period).std()
                upper  = sma + 2 * std
                lower  = sma - 2 * std

                upper_v = float(upper.iloc[-1])
                lower_v = float(lower.iloc[-1])
                sma_v   = float(sma.iloc[-1])
                std_v   = float(std.iloc[-1])

                # RSI
                delta = close.diff()
                gain  = delta.clip(lower=0).rolling(14).mean()
                loss  = (-delta.clip(upper=0)).rolling(14).mean()
                rsi   = 100 - 100 / (1 + gain / loss.replace(0, np.nan))
                rsi_v = float(rsi.iloc[-1])

                reasons    = []
                conviction = 0
                sig        = "HOLD"

                # Atingere banda
                at_lower = price <= lower_v
                at_upper = price >= upper_v

                if elements.get("band_touch", True):
                    if at_lower:
                        reasons.append(f"Pret la banda inferioara ({lower_v:.5f})")
                        conviction += 2
                        sig = "BUY"
                    elif at_upper:
                        reasons.append(f"Pret la banda superioara ({upper_v:.5f})")
                        conviction += 2
                        sig = "SELL"

                if sig == "HOLD":
                    tf_results.append({"tf": tf, "signal": "HOLD", "conviction": 0,
                                       "reasons": [], "price": price, "sl": None, "tp": None})
                    continue

                # RSI confirmare
                if elements.get("rsi_confirm", True):
                    if sig == "BUY" and rsi_v < 35:
                        reasons.append(f"RSI oversold ({rsi_v:.1f})")
                        conviction += 1
                    elif sig == "SELL" and rsi_v > 65:
                        reasons.append(f"RSI overbought ({rsi_v:.1f})")
                        conviction += 1
                    elif sig == "BUY" and rsi_v > 55:
                        sig = "HOLD"
                        tf_results.append({"tf": tf, "signal": "HOLD", "conviction": 0,
                                           "reasons": ["RSI nu confirma BUY"], "price": price,
                                           "sl": None, "tp": None})
                        continue
                    elif sig == "SELL" and rsi_v < 45:
                        sig = "HOLD"
                        tf_results.append({"tf": tf, "signal": "HOLD", "conviction": 0,
                                           "reasons": ["RSI nu confirma SELL"], "price": price,
                                           "sl": None, "tp": None})
                        continue

                # Squeeze (latime banda ingusta = volatilitate scazuta = breakout iminent)
                if elements.get("squeeze", True):
                    band_width = (upper_v - lower_v) / sma_v if sma_v > 0 else 0
                    if band_width < 0.02:
                        reasons.append("BB Squeeze — volatilitate scazuta, breakout iminent")
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
                log.warning(f"BollingerStrategy {symbol}/{tf}: {exc}")

        if not tf_results:
            return self._empty_result(symbol, "Fara date suficiente")

        return self._build_result(symbol, tf_results, min_confidence, min_votes=1)
