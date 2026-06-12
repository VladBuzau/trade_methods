"""
Engulfing / Pin Bar v2: lumânari de reversal la nivel cheie (suport/rezistenta).
Bullish Engulfing: lumânare verde care inglobeaza complet lumânarea rosie anterioara.
Bearish Engulfing: invers.
Pin Bar: fitil lung (>2x body) la un nivel cheie.

P3 upgrades:
  - Key Level Quality: nivelul cheie trebuie testat de 2+ ori in ultimele 100 bare
  - Engulfing Size: lumânarea engulfing trebuie sa aiba corp >= 1.5x corpul lumânarii precedente
"""
from __future__ import annotations
import logging
import numpy as np
from strategies.base import Strategy

log = logging.getLogger(__name__)

KEY_LEVEL_MIN_TESTS = 2   # nivel cheie trebuie testat de cel putin 2 ori
ENGULF_SIZE_MIN     = 1.5 # corpul engulfing >= 1.5x corpul precedent


class EngulfingStrategy(Strategy):
    key   = "engulfing"
    name  = "Engulfing / Pin Bar"
    icon  = "🕯️"
    color = "#ff7043"

    default_tfs  = ["M15", "H1"]
    default_bars = 200
    elements     = {
        "engulfing":          "Engulfing candle (inglobare completa)",
        "pin_bar":            "Pin Bar (fitil lung la nivel cheie)",
        "key_level":          "Confirmare la suport/rezistenta",
        "level_quality":      "Key Level Quality: nivel testat 2+ ori in 100 bare",
        "engulf_size":        "Engulfing Size: corp >= 1.5x corpul precedent",
        "prev_trend_oppose":  "Trend anterior contrar semnalului (true reversal)",
        "min_body_pct":       "Corp >= 40% din range (anti doji)",
        "at_extreme_only":    "STRICT: doar la high/low recent 20 bare",
    }

    @staticmethod
    def _is_near_level(price, levels, tolerance_pct=0.003):
        for lvl in levels:
            if abs(price - lvl) / max(lvl, 1e-10) < tolerance_pct:
                return True, lvl
        return False, None

    @staticmethod
    def _count_level_tests(df, level: float, lookback: int = 100,
                            tolerance_pct: float = 0.003) -> int:
        """
        Numara de cate ori pretul a testat nivelul `level` in ultimele `lookback` bare.
        Versiune vectorizata numpy — fara Python loop.
        """
        highs = df["high"].values[-lookback:]
        lows  = df["low"].values[-lookback:]
        tol   = level * tolerance_pct
        return int(np.sum((np.abs(highs - level) <= tol) | (np.abs(lows - level) <= tol)))

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

                # ── Engulfing ─────────────────────────────────────────────
                if elements.get("engulfing", True) and len(c) >= 2:
                    prev_o, prev_c = o[-2], c[-2]
                    curr_o, curr_c = o[-1], c[-1]
                    prev_body = abs(prev_c - prev_o)
                    curr_body = abs(curr_c - curr_o)

                    bullish_engulf = (prev_c < prev_o and curr_c > curr_o and
                                      curr_o <= prev_c and curr_c >= prev_o)
                    bearish_engulf = (prev_c > prev_o and curr_c < curr_o and
                                      curr_o >= prev_c and curr_c <= prev_o)

                    if bullish_engulf:
                        reasons.append(f"Bullish Engulfing la {price:.5f}")
                        conviction += 2
                        sig = "BUY"
                        # ── P3: Engulfing Size check ──
                        if elements.get("engulf_size", True) and prev_body > 0:
                            size_ratio = curr_body / prev_body
                            if size_ratio >= ENGULF_SIZE_MIN:
                                reasons.append(
                                    f"Engulfing size {size_ratio:.2f}x"
                                    f" (>= {ENGULF_SIZE_MIN}x) — forta puternica ✓"
                                )
                                conviction += 1
                            else:
                                reasons.append(
                                    f"Engulfing size mic ({size_ratio:.2f}x"
                                    f" < {ENGULF_SIZE_MIN}x) — semnal slab"
                                )
                                conviction -= 1

                    elif bearish_engulf:
                        reasons.append(f"Bearish Engulfing la {price:.5f}")
                        conviction += 2
                        sig = "SELL"
                        if elements.get("engulf_size", True) and prev_body > 0:
                            size_ratio = curr_body / prev_body
                            if size_ratio >= ENGULF_SIZE_MIN:
                                reasons.append(
                                    f"Engulfing size {size_ratio:.2f}x ✓"
                                )
                                conviction += 1
                            else:
                                reasons.append(
                                    f"Engulfing size mic ({size_ratio:.2f}x) — semnal slab"
                                )
                                conviction -= 1

                # ── Pin Bar ───────────────────────────────────────────────
                if elements.get("pin_bar", True):
                    body         = abs(c[-1] - o[-1])
                    candle       = h[-1] - l[-1]
                    upper_wick   = h[-1] - max(c[-1], o[-1])
                    lower_wick   = min(c[-1], o[-1]) - l[-1]

                    if candle > 0 and body > 0:
                        if lower_wick > 2 * body and lower_wick > upper_wick * 1.5:
                            if sig != "SELL":
                                reasons.append(f"Bullish Pin Bar (fitil jos {lower_wick:.5f})")
                                conviction += 2
                                sig = "BUY"
                        elif upper_wick > 2 * body and upper_wick > lower_wick * 1.5:
                            if sig != "BUY":
                                reasons.append(f"Bearish Pin Bar (fitil sus {upper_wick:.5f})")
                                conviction += 2
                                sig = "SELL"

                if sig == "HOLD":
                    tf_results.append({"tf": tf, "signal": "HOLD", "conviction": 0,
                                       "reasons": [], "price": price, "sl": None, "tp": None})
                    continue

                # ── Min body pct (anti doji) ──────────────────────────────
                if elements.get("min_body_pct", False) and len(c) >= 1:
                    rng = float(h[-1] - l[-1])
                    body = abs(float(c[-1]) - float(o[-1]))
                    if rng > 0 and (body / rng) < 0.40:
                        reasons.append(f"Corp slab ({body/rng*100:.0f}%<40%) — semnal demoted")
                        tf_results.append({"tf": tf, "signal": "HOLD", "conviction": 0,
                                           "reasons": reasons, "price": price, "sl": None, "tp": None})
                        continue

                # ── Trend anterior trebuie sa fie OPUS semnalului (reversal real) ──
                if elements.get("prev_trend_oppose", False) and len(c) >= 10:
                    trend_close_old = float(c[-10])
                    trend_close_new = float(c[-2])  # inainte de bara curenta
                    prev_dir = "up" if trend_close_new > trend_close_old else "down"
                    if (sig == "BUY" and prev_dir != "down") or (sig == "SELL" and prev_dir != "up"):
                        reasons.append(f"Trend anterior nu e contrar (was {prev_dir}) — nu e reversal")
                        conviction -= 2
                    else:
                        reasons.append(f"Trend anterior {prev_dir} contrar — reversal valid (+1)")
                        conviction += 1

                # ── At extreme only ─────────────────────────────────────────
                if elements.get("at_extreme_only", False):
                    recent_high = float(np.max(h[-20:]))
                    recent_low = float(np.min(l[-20:]))
                    rng = recent_high - recent_low
                    if rng > 0:
                        if sig == "BUY":
                            pos = (price - recent_low) / rng
                            if pos > 0.30:
                                reasons.append(f"BUY nu la low extrem (pos {pos*100:.0f}%) — demoted")
                                tf_results.append({"tf": tf, "signal": "HOLD", "conviction": 0,
                                                   "reasons": reasons, "price": price, "sl": None, "tp": None})
                                continue
                        elif sig == "SELL":
                            pos = (recent_high - price) / rng
                            if pos > 0.30:
                                reasons.append(f"SELL nu la high extrem (pos {pos*100:.0f}%) — demoted")
                                tf_results.append({"tf": tf, "signal": "HOLD", "conviction": 0,
                                                   "reasons": reasons, "price": price, "sl": None, "tp": None})
                                continue

                # ── Confirmare la nivel cheie ──────────────────────────────
                if elements.get("key_level", True) and key_levels:
                    near, matched_level = self._is_near_level(price, key_levels)
                    if near and matched_level:
                        reasons.append(f"La nivel cheie (pivot {matched_level:.5f})")
                        conviction += 1

                        # ── P3: Level Quality — 2+ tests ──
                        if elements.get("level_quality", True):
                            n_tests = self._count_level_tests(
                                df, matched_level, lookback=100
                            )
                            if n_tests >= KEY_LEVEL_MIN_TESTS:
                                reasons.append(
                                    f"Nivel testat de {n_tests}x in 100 bare"
                                    " — nivel cheie confirmat ✓"
                                )
                                conviction += 1
                            else:
                                reasons.append(
                                    f"Nivel testat doar {n_tests}x in 100 bare"
                                    f" (< {KEY_LEVEL_MIN_TESTS}) — nivel slab"
                                )
                                conviction -= 1
                    else:
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
