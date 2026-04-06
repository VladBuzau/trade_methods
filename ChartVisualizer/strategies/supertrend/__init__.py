"""
Strategie Supertrend: ATR-based trend indicator, semnal clar BUY/SELL la schimbare directie.
Timeframe recomandat: H1, H4, D1.
"""
from __future__ import annotations
import logging

import numpy as np
import pandas as pd

from strategies.base import Strategy

log = logging.getLogger(__name__)


class SupertrendStrategy(Strategy):
    key   = "supertrend"
    name  = "Supertrend"
    icon  = "⚡"
    color = "#ffeb3b"

    default_tfs  = ["H1", "H4"]
    default_bars = 300
    elements     = {
        "supertrend": "Supertrend (ATR 10, factor 3.0)",
        "ema50":      "EMA 50 confirmare directie",
        "adx":        "ADX > 20 (trend puternic)",
    }

    @staticmethod
    def _calc_supertrend(df: pd.DataFrame, period: int = 10, multiplier: float = 3.0):
        """Calculeaza Supertrend. Returneaza seria (True=bullish, False=bearish)."""
        high  = df["high"]
        low   = df["low"]
        close = df["close"]

        # ATR
        tr = pd.concat([
            high - low,
            (high - close.shift()).abs(),
            (low  - close.shift()).abs(),
        ], axis=1).max(axis=1)
        atr = tr.ewm(span=period, adjust=False).mean()

        hl2    = (high + low) / 2
        upper  = hl2 + multiplier * atr
        lower  = hl2 - multiplier * atr

        # Recalcul iterativ
        final_upper = upper.copy()
        final_lower = lower.copy()
        supertrend  = pd.Series(True, index=df.index)   # True = bullish

        for i in range(1, len(df)):
            # Upper band
            if upper.iloc[i] < final_upper.iloc[i-1] or close.iloc[i-1] > final_upper.iloc[i-1]:
                final_upper.iloc[i] = upper.iloc[i]
            else:
                final_upper.iloc[i] = final_upper.iloc[i-1]
            # Lower band
            if lower.iloc[i] > final_lower.iloc[i-1] or close.iloc[i-1] < final_lower.iloc[i-1]:
                final_lower.iloc[i] = lower.iloc[i]
            else:
                final_lower.iloc[i] = final_lower.iloc[i-1]
            # Direction
            if supertrend.iloc[i-1] and close.iloc[i] < final_lower.iloc[i]:
                supertrend.iloc[i] = False
            elif not supertrend.iloc[i-1] and close.iloc[i] > final_upper.iloc[i]:
                supertrend.iloc[i] = True
            else:
                supertrend.iloc[i] = supertrend.iloc[i-1]

        return supertrend, final_upper, final_lower

    def analyze(self, symbol, tfs, bars=300, tf_bars=None, elements=None,
                min_confidence=66.0, **kwargs):
        from app import fetch, calc_sl_tp, find_pivots, calc_adx

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

                reasons    = []
                conviction = 0
                sig        = "HOLD"

                # Supertrend
                if elements.get("supertrend", True):
                    st, st_upper, st_lower = self._calc_supertrend(df)
                    bullish_now  = bool(st.iloc[-1])
                    bullish_prev = bool(st.iloc[-2])

                    # Semnal la schimbare de directie
                    if not bullish_prev and bullish_now:
                        reasons.append("Supertrend: BEARISH → BULLISH")
                        conviction += 3
                        sig = "BUY"
                    elif bullish_prev and not bullish_now:
                        reasons.append("Supertrend: BULLISH → BEARISH")
                        conviction += 3
                        sig = "SELL"
                    elif bullish_now:
                        reasons.append("Supertrend bullish (trend in desfasurare)")
                        conviction += 1
                        sig = "BUY"
                    else:
                        reasons.append("Supertrend bearish (trend in desfasurare)")
                        conviction += 1
                        sig = "SELL"

                if sig == "HOLD":
                    tf_results.append({"tf": tf, "signal": "HOLD", "conviction": 0,
                                       "reasons": [], "price": price, "sl": None, "tp": None})
                    continue

                # EMA 50 confirmare
                if elements.get("ema50", True):
                    ema50 = float(close.ewm(span=50, adjust=False).mean().iloc[-1])
                    if sig == "BUY" and price > ema50:
                        reasons.append("Pret deasupra EMA50")
                        conviction += 1
                    elif sig == "SELL" and price < ema50:
                        reasons.append("Pret sub EMA50")
                        conviction += 1
                    else:
                        sig = "HOLD"
                        reasons.append("EMA50 contra-trend — blocat")
                        tf_results.append({"tf": tf, "signal": "HOLD", "conviction": 0,
                                           "reasons": reasons, "price": price,
                                           "sl": None, "tp": None})
                        continue

                # ADX confirmare
                if elements.get("adx", True):
                    try:
                        adx_val = float(calc_adx(df).iloc[-1])
                        if adx_val > 20:
                            reasons.append(f"ADX {adx_val:.1f} — trend puternic")
                            conviction += 1
                        else:
                            reasons.append(f"ADX {adx_val:.1f} — trend slab")
                    except Exception:
                        pass

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
                log.warning(f"SupertrendStrategy {symbol}/{tf}: {exc}")

        if not tf_results:
            return self._empty_result(symbol, "Fara date suficiente")

        return self._build_result(symbol, tf_results, min_confidence, min_votes=1)
