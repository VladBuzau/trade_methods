"""
EMA Cross 8/21: crossover EMA rapida/lenta + volum confirmare.
Bun pentru scalping pe M5/M15 si swing pe H1.
"""
from __future__ import annotations
import logging
import numpy as np
from strategies.base import Strategy

log = logging.getLogger(__name__)


class EMACrossStrategy(Strategy):
    key   = "ema_cross"
    name  = "EMA Cross 8/21"
    icon  = "✂️"
    color = "#66bb6a"

    default_tfs  = ["M5", "M15", "H1"]
    default_bars = 200
    elements     = {
        "ema_cross":  "EMA 8/21 crossover",
        "ema50":      "EMA 50 (filtru trend major)",
        "momentum":   "Momentum confirmare (close > open)",
    }

    def analyze(self, symbol, tfs, bars=200, tf_bars=None, elements=None,
                min_confidence=66.0, **kwargs):
        from app import fetch, calc_sl_tp, find_pivots

        if elements is None:
            elements = {k: True for k in self.elements}

        tf_results = []
        for tf in tfs:
            try:
                n_bars = (tf_bars or {}).get(tf, bars)
                df, _  = fetch(symbol, tf, n_bars)
                if df is None or len(df) < 55:
                    continue

                close = df["close"]
                price = float(close.iloc[-1])

                ema8  = close.ewm(span=8,  adjust=False).mean()
                ema21 = close.ewm(span=21, adjust=False).mean()
                ema50 = close.ewm(span=50, adjust=False).mean()

                e8_now,  e8_prev  = float(ema8.iloc[-1]),  float(ema8.iloc[-2])
                e21_now, e21_prev = float(ema21.iloc[-1]), float(ema21.iloc[-2])
                e50_v             = float(ema50.iloc[-1])

                reasons    = []
                conviction = 0
                sig        = "HOLD"

                # EMA Cross
                if elements.get("ema_cross", True):
                    cross_up   = e8_prev < e21_prev and e8_now > e21_now
                    cross_down = e8_prev > e21_prev and e8_now < e21_now

                    if cross_up:
                        reasons.append(f"EMA8 {e8_now:.5f} cross UP peste EMA21 {e21_now:.5f}")
                        conviction += 2
                        sig = "BUY"
                    elif cross_down:
                        reasons.append(f"EMA8 {e8_now:.5f} cross DOWN sub EMA21 {e21_now:.5f}")
                        conviction += 2
                        sig = "SELL"
                    elif e8_now > e21_now:
                        reasons.append("EMA8 deasupra EMA21 (trend BUY in desfasurare)")
                        conviction += 1
                        sig = "BUY"
                    else:
                        reasons.append("EMA8 sub EMA21 (trend SELL in desfasurare)")
                        conviction += 1
                        sig = "SELL"

                # EMA 50 filtru
                if elements.get("ema50", True):
                    if sig == "BUY" and price > e50_v:
                        reasons.append(f"Pret deasupra EMA50 ({e50_v:.5f})")
                        conviction += 1
                    elif sig == "SELL" and price < e50_v:
                        reasons.append(f"Pret sub EMA50 ({e50_v:.5f})")
                        conviction += 1
                    else:
                        sig = "HOLD"
                        reasons.append("EMA50 contra-trend — blocat")
                        tf_results.append({"tf": tf, "signal": "HOLD", "conviction": 0,
                                           "reasons": reasons, "price": price,
                                           "sl": None, "tp": None})
                        continue

                # Momentum: ultima lumânare in directia semnalului
                if elements.get("momentum", True):
                    last_open  = float(df["open"].iloc[-1])
                    last_close = float(df["close"].iloc[-1])
                    if sig == "BUY" and last_close > last_open:
                        reasons.append("Momentum: ultima lumânare bullish")
                        conviction += 1
                    elif sig == "SELL" and last_close < last_open:
                        reasons.append("Momentum: ultima lumânare bearish")
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
                log.warning(f"EMACrossStrategy {symbol}/{tf}: {exc}")

        if not tf_results:
            return self._empty_result(symbol, "Fara date suficiente")

        return self._build_result(symbol, tf_results, min_confidence, min_votes=1)
