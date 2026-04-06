"""
Ichimoku Cloud: sistem complet de trend japonez.
Semnale: Tenkan/Kijun cross, pret deasupra/dedesubt cloud (Kumo), Chikou confirmare.
"""
from __future__ import annotations
import logging
import numpy as np
import pandas as pd
from strategies.base import Strategy

log = logging.getLogger(__name__)


class IchimokuStrategy(Strategy):
    key   = "ichimoku"
    name  = "Ichimoku"
    icon  = "☁️"
    color = "#26c6da"

    default_tfs  = ["H1", "H4", "D1"]
    default_bars = 300
    elements     = {
        "tk_cross":   "TK Cross (Tenkan > Kijun)",
        "kumo":       "Pret deasupra/dedesubt Cloud",
        "chikou":     "Chikou deasupra/dedesubt pretului",
    }

    @staticmethod
    def _ichimoku(df: pd.DataFrame):
        h = df["high"]
        l = df["low"]

        tenkan  = (h.rolling(9).max()  + l.rolling(9).min())  / 2
        kijun   = (h.rolling(26).max() + l.rolling(26).min()) / 2
        senkou_a = ((tenkan + kijun) / 2).shift(26)
        senkou_b = ((h.rolling(52).max() + l.rolling(52).min()) / 2).shift(26)
        chikou   = df["close"].shift(-26)

        return tenkan, kijun, senkou_a, senkou_b, chikou

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
                if df is None or len(df) < 100:
                    continue

                tenkan, kijun, senkou_a, senkou_b, chikou = self._ichimoku(df)

                price     = float(df["close"].iloc[-1])
                tenkan_v  = float(tenkan.iloc[-1])
                kijun_v   = float(kijun.iloc[-1])
                sa_v      = float(senkou_a.iloc[-1]) if not pd.isna(senkou_a.iloc[-1]) else None
                sb_v      = float(senkou_b.iloc[-1]) if not pd.isna(senkou_b.iloc[-1]) else None
                chikou_v  = float(chikou.iloc[-27]) if len(chikou) > 27 and not pd.isna(chikou.iloc[-27]) else None
                price_26  = float(df["close"].iloc[-27]) if len(df) > 27 else None

                tenkan_prev = float(tenkan.iloc[-2])
                kijun_prev  = float(kijun.iloc[-2])

                reasons    = []
                conviction = 0
                sig        = "HOLD"

                # ── TK Cross ──────────────────────────────────────────────
                if elements.get("tk_cross", True):
                    cross_up   = tenkan_prev < kijun_prev and tenkan_v > kijun_v
                    cross_down = tenkan_prev > kijun_prev and tenkan_v < kijun_v

                    if cross_up:
                        reasons.append(f"TK Cross UP: Tenkan {tenkan_v:.5f} > Kijun {kijun_v:.5f}")
                        conviction += 2
                        sig = "BUY"
                    elif cross_down:
                        reasons.append(f"TK Cross DOWN: Tenkan {tenkan_v:.5f} < Kijun {kijun_v:.5f}")
                        conviction += 2
                        sig = "SELL"
                    elif tenkan_v > kijun_v:
                        sig = "BUY"
                        reasons.append("Tenkan deasupra Kijun (trend bullish)")
                        conviction += 1
                    else:
                        sig = "SELL"
                        reasons.append("Tenkan sub Kijun (trend bearish)")
                        conviction += 1

                # ── Kumo filter ───────────────────────────────────────────
                if elements.get("kumo", True) and sa_v and sb_v:
                    cloud_top = max(sa_v, sb_v)
                    cloud_bot = min(sa_v, sb_v)
                    if sig == "BUY" and price > cloud_top:
                        reasons.append(f"Pret deasupra Cloud ({cloud_top:.5f})")
                        conviction += 2
                    elif sig == "SELL" and price < cloud_bot:
                        reasons.append(f"Pret sub Cloud ({cloud_bot:.5f})")
                        conviction += 2
                    elif cloud_bot <= price <= cloud_top:
                        sig = "HOLD"
                        reasons.append("Pret in interiorul Cloud — semnal blocat")
                        tf_results.append({"tf": tf, "signal": "HOLD", "conviction": 0,
                                           "reasons": reasons, "price": price,
                                           "sl": None, "tp": None})
                        continue
                    else:
                        sig = "HOLD"
                        reasons.append("Cloud contra-trend — semnal blocat")
                        tf_results.append({"tf": tf, "signal": "HOLD", "conviction": 0,
                                           "reasons": reasons, "price": price,
                                           "sl": None, "tp": None})
                        continue

                if sig == "HOLD":
                    tf_results.append({"tf": tf, "signal": "HOLD", "conviction": 0,
                                       "reasons": [], "price": price, "sl": None, "tp": None})
                    continue

                # ── Chikou confirmare ─────────────────────────────────────
                if elements.get("chikou", True) and chikou_v and price_26:
                    if sig == "BUY" and chikou_v > price_26:
                        reasons.append("Chikou deasupra pretului de 26 bare")
                        conviction += 1
                    elif sig == "SELL" and chikou_v < price_26:
                        reasons.append("Chikou sub pretul de 26 bare")
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
                log.warning(f"IchimokuStrategy {symbol}/{tf}: {exc}")

        if not tf_results:
            return self._empty_result(symbol, "Fara date suficiente")

        return self._build_result(symbol, tf_results, min_confidence, min_votes=1)
