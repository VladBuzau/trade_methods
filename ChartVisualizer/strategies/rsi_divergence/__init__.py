"""
RSI Divergence v2: pret face un nou high/low dar RSI nu confirma → reversal iminent.
Bullish divergence: pret lower low + RSI higher low → BUY
Bearish divergence: pret higher high + RSI lower high → SELL

P3 upgrades:
  - Bar count quality filter: divergenta optima intre 10-50 bare (prea scurta = zgomot,
    prea lunga = slab de actiune)
  - Double/Triple divergence bonus: 3 pivot-uri → +2 conviction, 2 → +1
"""
from __future__ import annotations
import logging
import numpy as np
from scipy.signal import find_peaks
from strategies.base import Strategy

log = logging.getLogger(__name__)

DIV_MIN_BARS = 10   # divergenta prea scurta = zgomot
DIV_MAX_BARS = 50   # divergenta prea lunga = nu mai e relevanta


class RSIDivergenceStrategy(Strategy):
    key   = "rsi_divergence"
    name  = "RSI Divergence"
    icon  = "📉"
    color = "#e91e63"

    default_tfs  = ["H1", "H4"]
    default_bars = 300
    elements     = {
        "bullish_div":      "Divergenta Bullish (pret LL, RSI HL)",
        "bearish_div":      "Divergenta Bearish (pret HH, RSI LH)",
        "rsi_zone":         "RSI in zona extrema (< 35 / > 65)",
        "bar_quality":      "Bar count quality filter (10-50 bare ideal)",
        "multi_div":        "Double/Triple divergenta bonus (+1/+2)",
        "rsi_zone_strict":  "STRICT: RSI obligatoriu < 35 (BUY) / > 65 (SELL)",
        "min_multi_2":      "Cere minim 2 divergente (ignora single)",
        "div_fresh":        "Divergenta fresh (ultimul pivot in ultimele 5 bare)",
    }

    @staticmethod
    def _find_rsi_divergence(close, rsi, lookback=60, bar_quality=True, multi_div=True):
        """
        Cauta divergenta pe ultimele `lookback` bare.
        Foloseste scipy.signal.find_peaks cu prominence pentru a elimina pivoti falsi.
        Returneaza (sig, reasons, div_bars, div_count) sau ("HOLD", [], 0, 0).
        """
        if len(close) < lookback + 5:
            return "HOLD", [], 0, 0

        c = close.values[-lookback:]
        r = rsi.values[-lookback:]

        # Prominence adaptiva: 5% din range-ul pretului, minim 2 unitati RSI
        price_range = float(np.max(c) - np.min(c))
        price_prom  = max(price_range * 0.05, 1e-8)
        rsi_prom    = 2.0   # unitati RSI

        ph_c, _  = find_peaks( c, distance=3, prominence=price_prom)
        pl_c, _  = find_peaks(-c, distance=3, prominence=price_prom)
        ph_r, _  = find_peaks( r, distance=3, prominence=rsi_prom)
        pl_r, _  = find_peaks(-r, distance=3, prominence=rsi_prom)

        price_maxs = ph_c.tolist()
        price_mins = pl_c.tolist()
        rsi_maxs   = ph_r.tolist()
        rsi_mins   = pl_r.tolist()

        # ── Bullish divergence ─────────────────────────────────────────────
        if len(price_mins) >= 2 and len(rsi_mins) >= 2:
            # Verifica toate combinatiile de minim (suporta triple div)
            bull_count = 0
            bull_reasons = []
            last_pm = price_mins[-1]

            for k in range(len(price_mins) - 1, 0, -1):
                pm_curr = price_mins[k]
                pm_prev = price_mins[k-1]
                rm_curr = min(rsi_mins, key=lambda x: abs(x - pm_curr))
                rm_prev = min(rsi_mins, key=lambda x: abs(x - pm_prev))

                if c[pm_curr] < c[pm_prev] and r[rm_curr] > r[rm_prev]:
                    bar_span = pm_curr - pm_prev
                    bull_count += 1
                    bull_reasons.append(
                        f"Div bullish #{bull_count}: pret {c[pm_prev]:.5f}→{c[pm_curr]:.5f}"
                        f" (LL), RSI {r[rm_prev]:.1f}→{r[rm_curr]:.1f} (HL)"
                        f" [span {bar_span} bare]"
                    )
                    last_span = bar_span
                    if bull_count >= 2:  # triple → suficient
                        break

            if bull_count > 0:
                return "BUY", bull_reasons, last_span, bull_count

        # ── Bearish divergence ─────────────────────────────────────────────
        if len(price_maxs) >= 2 and len(rsi_maxs) >= 2:
            bear_count = 0
            bear_reasons = []

            for k in range(len(price_maxs) - 1, 0, -1):
                pm_curr = price_maxs[k]
                pm_prev = price_maxs[k-1]
                rm_curr = min(rsi_maxs, key=lambda x: abs(x - pm_curr))
                rm_prev = min(rsi_maxs, key=lambda x: abs(x - pm_prev))

                if c[pm_curr] > c[pm_prev] and r[rm_curr] < r[rm_prev]:
                    bar_span = pm_curr - pm_prev
                    bear_count += 1
                    bear_reasons.append(
                        f"Div bearish #{bear_count}: pret {c[pm_prev]:.5f}→{c[pm_curr]:.5f}"
                        f" (HH), RSI {r[rm_prev]:.1f}→{r[rm_curr]:.1f} (LH)"
                        f" [span {bar_span} bare]"
                    )
                    last_span = bar_span
                    if bear_count >= 2:
                        break

            if bear_count > 0:
                return "SELL", bear_reasons, last_span, bear_count

        return "HOLD", [], 0, 0

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

                sig, reasons, div_bars, div_count = self._find_rsi_divergence(
                    close, rsi, lookback=60,
                    bar_quality=elements.get("bar_quality", True),
                    multi_div=elements.get("multi_div", True),
                )
                conviction = 0

                if sig == "HOLD":
                    tf_results.append({"tf": tf, "signal": "HOLD", "conviction": 0,
                                       "reasons": [], "price": price, "sl": None, "tp": None})
                    continue

                # ── min_multi_2: cere minim 2 divergente confirmate ──
                if elements.get("min_multi_2", False) and div_count < 2:
                    reasons.append(f"min_multi_2: doar {div_count} div, nu trec (≥2 cerut)")
                    tf_results.append({"tf": tf, "signal": "HOLD", "conviction": 0,
                                       "reasons": reasons, "price": price, "sl": None, "tp": None})
                    continue

                # ── div_fresh: ultimul pivot in ultimele 5 bare ──
                if elements.get("div_fresh", False) and div_bars > 50:
                    reasons.append(f"div_fresh: span {div_bars} prea vechi (>50 bare)")
                    tf_results.append({"tf": tf, "signal": "HOLD", "conviction": 0,
                                       "reasons": reasons, "price": price, "sl": None, "tp": None})
                    continue

                conviction += 2

                # ── P3: Bar Count Quality Filter ──
                if elements.get("bar_quality", True) and div_bars > 0:
                    if div_bars < DIV_MIN_BARS:
                        reasons.append(
                            f"Divergenta prea scurta ({div_bars} bare < {DIV_MIN_BARS})"
                            " — posibil zgomot, -1 conviction"
                        )
                        conviction -= 1
                    elif DIV_MIN_BARS <= div_bars <= DIV_MAX_BARS:
                        reasons.append(
                            f"Calitate divergenta excelenta ({div_bars} bare in fereastra optima)"
                            " ✓"
                        )
                        conviction += 1
                    else:
                        reasons.append(
                            f"Divergenta veche ({div_bars} bare > {DIV_MAX_BARS})"
                            " — relevanta redusa"
                        )

                # ── P3: Multi-divergence bonus ──
                if elements.get("multi_div", True):
                    if div_count >= 3:
                        reasons.append("Triple divergenta detectata — semnal foarte puternic (+2)")
                        conviction += 2
                    elif div_count == 2:
                        reasons.append("Double divergenta detectata — semnal puternic (+1)")
                        conviction += 1

                # Filtru zona extrema RSI
                if elements.get("rsi_zone", True):
                    if sig == "BUY" and rsi_v < 35:
                        reasons.append(f"RSI oversold ({rsi_v:.1f} < 35) ✓")
                        conviction += 1
                    elif sig == "SELL" and rsi_v > 65:
                        reasons.append(f"RSI overbought ({rsi_v:.1f} > 65) ✓")
                        conviction += 1
                    elif (sig == "BUY" and rsi_v > 55) or (sig == "SELL" and rsi_v < 45):
                        reasons.append(
                            f"RSI ({rsi_v:.1f}) contra-divergenta — semnal slab, -1"
                        )
                        conviction -= 1

                # ── rsi_zone_strict: blocheaza daca nu e in extrema ──
                if elements.get("rsi_zone_strict", False):
                    if (sig == "BUY" and rsi_v >= 35) or (sig == "SELL" and rsi_v <= 65):
                        reasons.append(f"rsi_zone_strict: RSI {rsi_v:.1f} nu e extrem — HOLD")
                        tf_results.append({"tf": tf, "signal": "HOLD", "conviction": 0,
                                           "reasons": reasons, "price": price, "sl": None, "tp": None})
                        continue

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
