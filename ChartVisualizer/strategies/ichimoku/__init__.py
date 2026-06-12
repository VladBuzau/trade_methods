"""
Ichimoku Cloud v2: sistem complet de trend japonez.
Semnale: Tenkan/Kijun cross, pret deasupra/dedesubt cloud (Kumo), Chikou confirmare.

P3 upgrades:
  - TK Cross Outside Cloud: cross-ul Tenkan/Kijun are valoare doar in afara Kumo
    (in interiorul cloudului = zgomot)
  - Kumo Twist Early Warning: Senkou A si B se intersecteaza → schimbare de trend iminenta
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

    default_tfs  = ["H4", "D1"]
    default_bars = 300
    elements     = {
        "tk_cross":      "TK Cross (Tenkan > Kijun)",
        "kumo":          "Pret deasupra/dedesubt Cloud",
        "chikou":        "Chikou deasupra/dedesubt pretului",
        "tk_outside":    "TK Cross valid doar in afara Kumo",
        "kumo_twist":    "Kumo Twist early warning — Senkou A/B se intersecteaza",
        "fresh_cross":   "STRICT: doar cross-uri proaspete (nu trend deja stabilit)",
        "cloud_thick":   "Cloud gros (Kumo width ≥ 0.3% pret) — trend solid",
        "above_below":   "Pret deasupra/sub Cloud BLOCKING (nu doar scoring)",
    }

    @staticmethod
    def _ichimoku(df: pd.DataFrame):
        h = df["high"]
        l = df["low"]

        tenkan   = (h.rolling(9).max()  + l.rolling(9).min())  / 2
        kijun    = (h.rolling(26).max() + l.rolling(26).min()) / 2
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

                # Cloud boundaries
                cloud_top = max(sa_v, sb_v) if sa_v and sb_v else None
                cloud_bot = min(sa_v, sb_v) if sa_v and sb_v else None

                # ── P3: Kumo Twist Early Warning ──
                if elements.get("kumo_twist", True) and sa_v and sb_v:
                    # Check daca Senkou A si B se intersecteaza in urmatoarele 5 bare
                    twist_coming = False
                    for offset in range(1, 6):
                        idx = -1 - offset
                        if abs(idx) <= len(senkou_a):
                            sa_fut = senkou_a.iloc[idx]
                            sb_fut = senkou_b.iloc[idx]
                            if not pd.isna(sa_fut) and not pd.isna(sb_fut):
                                if (sa_v > sb_v and sa_fut < sb_fut) or (sa_v < sb_v and sa_fut > sb_fut):
                                    twist_coming = True
                                    break
                    if twist_coming:
                        reasons.append(
                            "Kumo Twist detectat (Senkou A/B se intersecteaza curand)"
                            " — schimbare de trend iminenta, precautie!"
                        )
                        # Nu blocam, dar adaugam ca nota de precautie

                # ── TK Cross ──────────────────────────────────────────────
                if elements.get("tk_cross", True):
                    cross_up   = tenkan_prev < kijun_prev and tenkan_v > kijun_v
                    cross_down = tenkan_prev > kijun_prev and tenkan_v < kijun_v
                    fresh_only = elements.get("fresh_cross", False)

                    if cross_up:
                        sig = "BUY"
                        conviction += 3
                        if elements.get("tk_outside", True) and cloud_top and cloud_bot:
                            tk_in_cloud = cloud_bot <= tenkan_v <= cloud_top
                            if tk_in_cloud:
                                reasons.append(f"TK Cross UP in interiorul Kumo — zgomot (-2)")
                                conviction -= 2
                            else:
                                reasons.append(f"TK Cross UP in afara cloudului (+3)")
                        else:
                            reasons.append(f"TK Cross UP: Tenkan {tenkan_v:.5f} > Kijun {kijun_v:.5f} (+3)")

                    elif cross_down:
                        sig = "SELL"
                        conviction += 3
                        if elements.get("tk_outside", True) and cloud_top and cloud_bot:
                            tk_in_cloud = cloud_bot <= tenkan_v <= cloud_top
                            if tk_in_cloud:
                                reasons.append(f"TK Cross DOWN in interiorul Kumo — zgomot (-2)")
                                conviction -= 2
                            else:
                                reasons.append(f"TK Cross DOWN in afara cloudului (+3)")
                        else:
                            reasons.append(f"TK Cross DOWN: Tenkan {tenkan_v:.5f} < Kijun {kijun_v:.5f} (+3)")

                    elif not fresh_only and tenkan_v > kijun_v:
                        sig = "BUY"
                        reasons.append("Tenkan deasupra Kijun (trend bullish)")
                        conviction += 1
                    elif not fresh_only and tenkan_v < kijun_v:
                        sig = "SELL"
                        reasons.append("Tenkan sub Kijun (trend bearish)")
                        conviction += 1
                    else:
                        # fresh_only ON + nu e cross proaspat -> HOLD
                        if fresh_only:
                            reasons.append("fresh_cross ON: niciun cross proaspat — HOLD")

                # ── Kumo filter — scoring sau BLOCKING ────────────────────
                if elements.get("kumo", True) and cloud_top and cloud_bot:
                    blocking = elements.get("above_below", False)
                    if sig == "BUY" and price > cloud_top:
                        reasons.append(f"Pret deasupra Cloud ({cloud_top:.5f}) (+2)")
                        conviction += 2
                    elif sig == "SELL" and price < cloud_bot:
                        reasons.append(f"Pret sub Cloud ({cloud_bot:.5f}) (+2)")
                        conviction += 2
                    elif cloud_bot <= price <= cloud_top:
                        if blocking:
                            reasons.append("above_below ON: Pret in Cloud — HOLD")
                            sig = "HOLD"
                        else:
                            reasons.append("Pret in interiorul Cloud — indecis (-2)")
                            conviction -= 2
                    else:
                        if blocking:
                            reasons.append("above_below ON: Cloud contra-trend — HOLD")
                            sig = "HOLD"
                        else:
                            reasons.append("Cloud contra-trend (-3)")
                            conviction -= 3

                # ── Cloud thickness ──────────────────────────────────────
                if elements.get("cloud_thick", False) and cloud_top and cloud_bot and price > 0:
                    cloud_pct = (cloud_top - cloud_bot) / price * 100
                    if cloud_pct < 0.30:
                        reasons.append(f"Cloud subtire ({cloud_pct:.2f}%<0.30%) — trend slab (-1)")
                        conviction -= 1
                    else:
                        reasons.append(f"Cloud gros ({cloud_pct:.2f}%) — trend solid (+1)")
                        conviction += 1

                if sig == "HOLD":
                    tf_results.append({"tf": tf, "signal": "HOLD", "conviction": 0,
                                       "reasons": reasons, "price": price, "sl": None, "tp": None})
                    continue

                # ── Chikou confirmare ─────────────────────────────────────
                if elements.get("chikou", True) and chikou_v and price_26:
                    if sig == "BUY" and chikou_v > price_26:
                        reasons.append("Chikou deasupra pretului de 26 bare ✓")
                        conviction += 1
                    elif sig == "SELL" and chikou_v < price_26:
                        reasons.append("Chikou sub pretul de 26 bare ✓")
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
