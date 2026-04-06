"""
RSI Divergence: pret face un nou high/low dar RSI nu confirma → reversal iminent.
Bullish divergence: pret lower low + RSI higher low → BUY
Bearish divergence: pret higher high + RSI lower high → SELL
"""
from __future__ import annotations
import logging
import numpy as np
from strategies.base import Strategy

log = logging.getLogger(__name__)


class RSIDivergenceStrategy(Strategy):
    key   = "rsi_divergence"
    name  = "RSI Divergence"
    icon  = "📉"
    color = "#e91e63"

    default_tfs  = ["M15", "H1", "H4"]
    default_bars = 300
    elements     = {
        "bullish_div": "Divergenta Bullish (pret LL, RSI HL)",
        "bearish_div": "Divergenta Bearish (pret HH, RSI LH)",
        "rsi_zone":    "RSI in zona extrema (< 35 / > 65)",
    }

    @staticmethod
    def _find_rsi_divergence(close, rsi, lookback=30):
        """
        Cauta divergenta pe ultimele `lookback` bare.
        Returneaza ("BUY", reasons) sau ("SELL", reasons) sau ("HOLD", [])
        """
        if len(close) < lookback + 5:
            return "HOLD", []

        c = close.values[-lookback:]
        r = rsi.values[-lookback:]

        # Gaseste local minima/maxima simple
        def local_min_idx(arr):
            return [i for i in range(1, len(arr)-1) if arr[i] < arr[i-1] and arr[i] < arr[i+1]]
        def local_max_idx(arr):
            return [i for i in range(1, len(arr)-1) if arr[i] > arr[i-1] and arr[i] > arr[i+1]]

        price_mins = local_min_idx(c)
        price_maxs = local_max_idx(c)
        rsi_mins   = local_min_idx(r)
        rsi_maxs   = local_max_idx(r)

        # Bullish divergence: ultimele 2 price_mins — al doilea mai jos, RSI mai sus
        if len(price_mins) >= 2 and len(rsi_mins) >= 2:
            p1, p2 = price_mins[-2], price_mins[-1]
            # gasim cel mai apropiat rsi_min de fiecare price_min
            r1 = min(rsi_mins, key=lambda x: abs(x - p1))
            r2 = min(rsi_mins, key=lambda x: abs(x - p2))
            if c[p2] < c[p1] and r[r2] > r[r1]:
                reasons = [
                    f"Divergenta Bullish: pret {c[p1]:.5f}→{c[p2]:.5f} (LL), RSI {r[r1]:.1f}→{r[r2]:.1f} (HL)",
                ]
                return "BUY", reasons

        # Bearish divergence: ultimele 2 price_maxs — al doilea mai sus, RSI mai jos
        if len(price_maxs) >= 2 and len(rsi_maxs) >= 2:
            p1, p2 = price_maxs[-2], price_maxs[-1]
            r1 = min(rsi_maxs, key=lambda x: abs(x - p1))
            r2 = min(rsi_maxs, key=lambda x: abs(x - p2))
            if c[p2] > c[p1] and r[r2] < r[r1]:
                reasons = [
                    f"Divergenta Bearish: pret {c[p1]:.5f}→{c[p2]:.5f} (HH), RSI {r[r1]:.1f}→{r[r2]:.1f} (LH)",
                ]
                return "SELL", reasons

        return "HOLD", []

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
                if df is None or len(df) < 60:
                    continue

                close = df["close"]
                price = float(close.iloc[-1])

                # RSI 14
                delta = close.diff()
                gain  = delta.clip(lower=0).rolling(14).mean()
                loss  = (-delta.clip(upper=0)).rolling(14).mean()
                rsi   = 100 - 100 / (1 + gain / loss.replace(0, np.nan))
                rsi_v = float(rsi.iloc[-1])

                sig, reasons = self._find_rsi_divergence(close, rsi, lookback=40)
                conviction   = 0

                if sig == "HOLD":
                    tf_results.append({"tf": tf, "signal": "HOLD", "conviction": 0,
                                       "reasons": [], "price": price, "sl": None, "tp": None})
                    continue

                conviction += 2

                # Filtru zona extrema RSI
                if elements.get("rsi_zone", True):
                    if sig == "BUY" and rsi_v < 35:
                        reasons.append(f"RSI oversold ({rsi_v:.1f} < 35)")
                        conviction += 1
                    elif sig == "SELL" and rsi_v > 65:
                        reasons.append(f"RSI overbought ({rsi_v:.1f} > 65)")
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
                log.warning(f"RSIDivergenceStrategy {symbol}/{tf}: {exc}")

        if not tf_results:
            return self._empty_result(symbol, "Fara date suficiente")

        return self._build_result(symbol, tf_results, min_confidence, min_votes=1)
