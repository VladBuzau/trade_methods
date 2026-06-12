"""
Candle Sniper — strategie EOB pe M1/M5 bazata pe microstructura lumanarilor.

Strategia se auto-configureaza:
  - Auto-selecteaza TF-ul (M1 vs M5) pe baza calitatii semnalului
  - Auto-calculeaza SL pe baza structurii pattern-ului (swing high/low)
  - User-ul controleaza DOAR suma riscata (risk_dollars) → lot_size se calculeaza

Concept:
  La INCHIDEREA bara curente citim ultimele 3-5 lumanari si cautam un pattern
  clar de microstructura. SL e ancorat la nivelul structural unde patternul
  se invalideaza (nu un multiplu ATR arbitrar).

Patterns detectate:
  P1. 3-Bar Pivot Reversal — SL = low/high al barei pivot
  P2. Inside Bar Breakout  — SL = low/high al mother bar
  P3. Liquidity Grab        — SL = wick extreme + buffer
  P4. Momentum Continuation — SL = inceputul secventei (oldest of 3)
  P5. Failed Breakout       — SL = extrem-ul barei breaker

Logica EOB: la fiecare apel evaluam bara INCHISA + structura anterioara.
Auto TF: ruleaza ambele M1 si M5, alege TF-ul cu setup-ul cel mai puternic.
"""
from __future__ import annotations
import logging
from datetime import datetime, timezone

import numpy as np

from strategies.base import Strategy

log = logging.getLogger(__name__)


# ── Configurabile globale ─────────────────────────────────────────────────────
ATR_PERIOD       = 14

# Mod ULTRA-SHORT (default ON) — trade-uri de 3-4 minute pe M1:
#   - Forteaza M1 (skip M5 din auto-select)
#   - SL/TP pure ATR-based (skip structural anchor)
#   - SL = 0.5 × ATR M1 (~ 0.5-1 pip pe FX) — TIGHT
#   - TP = 0.7 × ATR M1 (RR 1.4)
#   - Trade-urile se inchid in 2-5 bare M1 (= 2-5 minute)
ULTRA_SHORT      = True
ULTRA_SL_MULT    = 0.5    # SL = 0.5 × ATR M1 — foarte tight
ULTRA_TP_MULT    = 0.7    # TP = 0.7 × ATR M1 — RR 1.4

# Mod LONG (cand ULTRA_SHORT = False)
SL_BUFFER_ATR    = 0.15
SL_MIN_ATR       = 0.6
TP_RR_RATIO      = 1.5

MIN_BODY_BREAK   = 0.60
MIN_BODY_REVERSE = 0.55
LIQUIDITY_WICK   = 0.40
LOOKBACK_BARS    = 5


# ── Helpers ───────────────────────────────────────────────────────────────────

def _atr(df, period: int = ATR_PERIOD) -> float:
    if df is None or len(df) < period + 2:
        return 0.0
    h = df["high"].values
    l = df["low"].values
    c = df["close"].values
    tr = np.maximum(h[1:] - l[1:],
                    np.maximum(np.abs(h[1:] - c[:-1]),
                               np.abs(l[1:] - c[:-1])))
    if len(tr) < period:
        return float(np.mean(tr))
    return float(np.mean(tr[-period:]))


def _bar(df, idx: int) -> dict:
    row = df.iloc[idx]
    o = float(row["open"])
    h = float(row["high"])
    l = float(row["low"])
    c = float(row["close"])
    rng = h - l
    body = abs(c - o)
    body_pct = (body / rng) if rng > 1e-10 else 0.0
    return {
        "open": o, "high": h, "low": l, "close": c,
        "range": rng, "body": body, "body_pct": body_pct,
        "is_bull": c > o,
        "is_bear": c < o,
        "wick_top": h - max(o, c),
        "wick_bot": min(o, c) - l,
    }


# ── Pattern detectors ─────────────────────────────────────────────────────────
# Fiecare returneaza: (signal, conviction, msg, sl_anchor)
# sl_anchor = pretul de la nivel structural unde patternul se invalideaza
# SL final = sl_anchor ± SL_BUFFER_ATR × ATR

def _p1_three_bar_pivot(df) -> tuple[str, int, str, float]:
    if len(df) < 4:
        return "HOLD", 0, "", 0.0

    b2 = _bar(df, -3)
    b1 = _bar(df, -2)
    b0 = _bar(df, -1)

    if b0["range"] < 1e-10 or b0["body_pct"] < MIN_BODY_REVERSE:
        return "HOLD", 0, "", 0.0

    if b2["is_bear"] and b1["is_bear"] and b0["is_bull"]:
        prior_high = max(b2["high"], b1["high"])
        if b0["close"] > prior_high:
            sl_anchor = b0["low"]   # SL sub low-ul barei pivot
            return "BUY", 9, (
                f"3-bar pivot BUY: 2 bare bear + bull corp {b0['body_pct']*100:.0f}% "
                f"close > {prior_high:.5f}"
            ), sl_anchor

    if b2["is_bull"] and b1["is_bull"] and b0["is_bear"]:
        prior_low = min(b2["low"], b1["low"])
        if b0["close"] < prior_low:
            sl_anchor = b0["high"]
            return "SELL", 9, (
                f"3-bar pivot SELL: 2 bare bull + bear corp {b0['body_pct']*100:.0f}% "
                f"close < {prior_low:.5f}"
            ), sl_anchor

    return "HOLD", 0, "", 0.0


def _p2_inside_bar_break(df) -> tuple[str, int, str, float]:
    if len(df) < 4:
        return "HOLD", 0, "", 0.0

    mother = _bar(df, -3)
    inside = _bar(df, -2)
    cur    = _bar(df, -1)

    if mother["range"] < 1e-10 or cur["range"] < 1e-10:
        return "HOLD", 0, "", 0.0

    if inside["high"] > mother["high"] or inside["low"] < mother["low"]:
        return "HOLD", 0, "", 0.0

    if inside["range"] / mother["range"] > 0.85:
        return "HOLD", 0, "", 0.0

    if cur["body_pct"] < MIN_BODY_BREAK:
        return "HOLD", 0, "", 0.0

    if cur["is_bull"] and cur["close"] > mother["high"]:
        sl_anchor = mother["low"]   # SL sub mother bar
        return "BUY", 9, (
            f"Inside bar BUY break mother [{mother['low']:.5f}-{mother['high']:.5f}] "
            f"corp {cur['body_pct']*100:.0f}%"
        ), sl_anchor

    if cur["is_bear"] and cur["close"] < mother["low"]:
        sl_anchor = mother["high"]
        return "SELL", 9, (
            f"Inside bar SELL break mother [{mother['low']:.5f}-{mother['high']:.5f}] "
            f"corp {cur['body_pct']*100:.0f}%"
        ), sl_anchor

    return "HOLD", 0, "", 0.0


def _p3_liquidity_grab(df) -> tuple[str, int, str, float]:
    if len(df) < LOOKBACK_BARS + 2:
        return "HOLD", 0, "", 0.0

    cur = _bar(df, -1)
    if cur["range"] < 1e-10:
        return "HOLD", 0, "", 0.0

    prior = df.iloc[-LOOKBACK_BARS-1:-1]
    prior_high = float(prior["high"].max())
    prior_low  = float(prior["low"].min())

    wick_top_pct = cur["wick_top"] / cur["range"] if cur["range"] > 0 else 0
    if cur["high"] > prior_high and cur["close"] < prior_high and wick_top_pct >= LIQUIDITY_WICK:
        if cur["is_bear"]:
            sl_anchor = cur["high"]   # SL peste extremul wick-ului
            return "SELL", 10, (
                f"Liquidity grab SELL: wick {wick_top_pct*100:.0f}% peste {prior_high:.5f}"
            ), sl_anchor

    wick_bot_pct = cur["wick_bot"] / cur["range"] if cur["range"] > 0 else 0
    if cur["low"] < prior_low and cur["close"] > prior_low and wick_bot_pct >= LIQUIDITY_WICK:
        if cur["is_bull"]:
            sl_anchor = cur["low"]
            return "BUY", 10, (
                f"Liquidity grab BUY: wick {wick_bot_pct*100:.0f}% sub {prior_low:.5f}"
            ), sl_anchor

    return "HOLD", 0, "", 0.0


def _p4_momentum_3bar(df) -> tuple[str, int, str, float]:
    if len(df) < 4:
        return "HOLD", 0, "", 0.0

    b2 = _bar(df, -3)
    b1 = _bar(df, -2)
    b0 = _bar(df, -1)

    if min(b2["body_pct"], b1["body_pct"], b0["body_pct"]) < 0.40:
        return "HOLD", 0, "", 0.0

    if not (b2["body_pct"] <= b1["body_pct"] <= b0["body_pct"]):
        return "HOLD", 0, "", 0.0

    if b2["is_bull"] and b1["is_bull"] and b0["is_bull"]:
        if b0["close"] > b1["close"] > b2["close"]:
            sl_anchor = b2["low"]   # SL sub inceputul momentum-ului
            return "BUY", 8, (
                f"Momentum BUY: corpuri "
                f"{b2['body_pct']*100:.0f}/{b1['body_pct']*100:.0f}/{b0['body_pct']*100:.0f}%"
            ), sl_anchor

    if b2["is_bear"] and b1["is_bear"] and b0["is_bear"]:
        if b0["close"] < b1["close"] < b2["close"]:
            sl_anchor = b2["high"]
            return "SELL", 8, (
                f"Momentum SELL: corpuri "
                f"{b2['body_pct']*100:.0f}/{b1['body_pct']*100:.0f}/{b0['body_pct']*100:.0f}%"
            ), sl_anchor

    return "HOLD", 0, "", 0.0


def _p5_failed_breakout(df) -> tuple[str, int, str, float]:
    if len(df) < LOOKBACK_BARS + 3:
        return "HOLD", 0, "", 0.0

    prior = df.iloc[-LOOKBACK_BARS-2:-2]
    prior_high = float(prior["high"].max())
    prior_low  = float(prior["low"].min())

    breaker = _bar(df, -2)
    cur     = _bar(df, -1)

    if breaker["body_pct"] < 0.50 or cur["body_pct"] < 0.45:
        return "HOLD", 0, "", 0.0

    if breaker["is_bull"] and breaker["close"] > prior_high:
        if cur["is_bear"] and cur["close"] < prior_high:
            sl_anchor = breaker["high"]   # SL peste varful breaker-ului
            return "SELL", 9, (
                f"Failed break SELL: breaker peste {prior_high:.5f}, close {cur['close']:.5f} sub"
            ), sl_anchor

    if breaker["is_bear"] and breaker["close"] < prior_low:
        if cur["is_bull"] and cur["close"] > prior_low:
            sl_anchor = breaker["low"]
            return "BUY", 9, (
                f"Failed break BUY: breaker sub {prior_low:.5f}, close {cur['close']:.5f} peste"
            ), sl_anchor

    return "HOLD", 0, "", 0.0


# ── HTF context ───────────────────────────────────────────────────────────────

def _htf_bias(df_htf) -> str:
    if df_htf is None or len(df_htf) < 60:
        return "NEUTRAL"
    close = df_htf["close"]
    ema20 = float(close.ewm(span=20, adjust=False).mean().iloc[-1])
    ema50 = float(close.ewm(span=50, adjust=False).mean().iloc[-1])
    diff_pct = abs(ema20 - ema50) / ema50 if ema50 > 0 else 0
    if diff_pct < 0.0003:
        return "NEUTRAL"
    return "BULL" if ema20 > ema50 else "BEAR"


# ── Evaluare TF: ruleaza toate patternurile pe un TF, returneaza setup-ul top ─

def _evaluate_tf(df, tf_name: str) -> dict | None:
    """
    Returneaza cel mai puternic setup gasit pe TF, cu SL ancorat structural.
    None daca niciun pattern detectat.
    """
    if df is None or len(df) < LOOKBACK_BARS + 5:
        return None

    atr = _atr(df, ATR_PERIOD)
    if atr <= 0:
        return None

    # Filtru ATR squeeze
    atr_long = _atr(df, 50)
    if atr_long > 0 and atr < atr_long * 0.4:
        return None

    detectors = [
        ("P1 Pivot",     _p1_three_bar_pivot),
        ("P2 InsideBar", _p2_inside_bar_break),
        ("P3 Liquidity", _p3_liquidity_grab),
        ("P4 Momentum",  _p4_momentum_3bar),
        ("P5 FailedBrk", _p5_failed_breakout),
    ]

    candidates = []
    for name, fn in detectors:
        try:
            sig, conv, msg, sl_anchor = fn(df)
            if sig in ("BUY", "SELL"):
                candidates.append({
                    "signal": sig, "conviction": conv, "msg": msg,
                    "sl_anchor": sl_anchor, "pattern": name,
                })
        except Exception as exc:
            log.debug(f"CandleSniper {name} on {tf_name}: {exc}")

    if not candidates:
        return None

    buys  = [c for c in candidates if c["signal"] == "BUY"]
    sells = [c for c in candidates if c["signal"] == "SELL"]

    if buys and sells:
        b_sum = sum(c["conviction"] for c in buys)
        s_sum = sum(c["conviction"] for c in sells)
        if b_sum == s_sum:
            return None
        chosen = buys if b_sum > s_sum else sells
    else:
        chosen = candidates

    best = max(chosen, key=lambda c: c["conviction"])
    same_dir = [c for c in candidates if c["signal"] == best["signal"]]

    # Bonus aliniere
    conv_final = best["conviction"]
    if len(same_dir) >= 2:
        conv_final = min(conv_final + 2, 13)

    price = float(df["close"].iloc[-1])

    # ── SL/TP — 2 moduri ─────────────────────────────────────────────────────
    if ULTRA_SHORT:
        # Mod 3-4 minute: SL/TP pure ATR-based, ignora structura
        sl_dist = atr * ULTRA_SL_MULT
        tp_dist = atr * ULTRA_TP_MULT
        if best["signal"] == "BUY":
            sl = price - sl_dist
            tp = price + tp_dist
        else:
            sl = price + sl_dist
            tp = price - tp_dist
    else:
        # Mod swing: SL ancorat structural + buffer
        buf = atr * SL_BUFFER_ATR
        if best["signal"] == "BUY":
            sl = best["sl_anchor"] - buf
            sl_dist = price - sl
        else:
            sl = best["sl_anchor"] + buf
            sl_dist = sl - price

        min_sl_dist = atr * SL_MIN_ATR
        if sl_dist < min_sl_dist:
            sl_dist = min_sl_dist
            sl = price - sl_dist if best["signal"] == "BUY" else price + sl_dist

        tp_dist = sl_dist * TP_RR_RATIO
        tp = price + tp_dist if best["signal"] == "BUY" else price - tp_dist

    return {
        "tf":         tf_name,
        "signal":     best["signal"],
        "conviction": conv_final,
        "pattern":    best["pattern"],
        "msg":        best["msg"],
        "price":      price,
        "sl":         sl,
        "tp":         tp,
        "sl_dist":    sl_dist,
        "tp_dist":    tp_dist,
        "atr":        atr,
        "n_converging": len(same_dir),
    }


# ── Strategie principala ──────────────────────────────────────────────────────

class CandleSniperStrategy(Strategy):
    key   = "candle_sniper"
    name  = "Candle Sniper"
    icon  = "🎯"
    color = "#00bcd4"

    # TF si bars sunt informative — strategia auto-selecteaza M1 vs M5
    default_tfs            = ["M1", "M5"]
    default_bars           = 200
    default_step           = 1        # M1 — fiecare bara (EOB logic)
    default_duration_years = 0.5
    default_max_hours      = 0.5      # 30 min max

    elements = {
        "p1_pivot":         "P1: 3-bar pivot reversal",
        "p2_inside":        "P2: Inside bar breakout",
        "p3_liquidity":     "P3: Liquidity grab + reversal",
        "p4_momentum":      "P4: Momentum continuation",
        "p5_failed":        "P5: Failed breakout (trap)",
        "auto_tf":          "Auto-select TF (M1 vs M5) — strategia decide",
        "htf_bias":         "Bias HTF — penalizeaza counter-trend",
        "htf_bias_strict":  "STRICT: HTF bias OBLIGATORIU (skip counter-trend total)",
        "p3_only":          "STRICT: doar pattern P3 (liquidity grab) — high quality",
        "min_2_patterns":   "Cere ≥2 patterns confluente (anti-noise)",
    }

    strategy_defaults = {
        "min_confidence": 60.0,
        # User controleaza DOAR risk_dollars (suma riscata per trade)
    }

    def analyze(self, symbol, tfs, bars=200, tf_bars=None, elements=None,
                min_confidence=60.0, **kwargs):
        from app import fetch

        if elements is None:
            elements = {k: True for k in self.elements}

        # ── Strategia ruleaza 24/7 — fara filtru de sesiune ───────────────────

        # ── AUTO-SELECT TF ────────────────────────────────────────────────────
        # Mod ULTRA_SHORT (3-4 min trades): M1 forced, fara fallback M5
        # Mod swing: incearca ambele M1 si M5, alege setup-ul mai puternic
        tf_candidates = ("M1",) if ULTRA_SHORT else ("M1", "M5")

        tf_results: dict[str, dict] = {}
        for tf_candidate in tf_candidates:
            try:
                n_bars = (tf_bars or {}).get(tf_candidate, bars)
                df, _ = fetch(symbol, tf_candidate, max(n_bars, 100))
                setup = _evaluate_tf(df, tf_candidate)
                if setup:
                    tf_results[tf_candidate] = setup
            except Exception as exc:
                log.debug(f"CandleSniper fetch {symbol}/{tf_candidate}: {exc}")

        if not tf_results:
            return self._empty_result(symbol, "Niciun pattern pe M1 sau M5")

        # Alege TF cu conviction max (tie-breaker: mai multe patterns converg)
        best_tf_setup = max(
            tf_results.values(),
            key=lambda s: (s["conviction"], s["n_converging"])
        )

        # ── HTF bias — penalizare counter-trend ──────────────────────────────
        htf_dir = "NEUTRAL"
        counter_trend = False
        if elements.get("htf_bias", True):
            # HTF = un nivel peste TF-ul ales
            htf_tf = "M15" if best_tf_setup["tf"] == "M5" else "M5"
            try:
                df_htf, _ = fetch(symbol, htf_tf, 100)
                htf_dir = _htf_bias(df_htf)
            except Exception:
                htf_dir = "NEUTRAL"

            counter_trend = (
                (best_tf_setup["signal"] == "BUY"  and htf_dir == "BEAR") or
                (best_tf_setup["signal"] == "SELL" and htf_dir == "BULL")
            )
            if counter_trend:
                best_tf_setup["conviction"] = max(best_tf_setup["conviction"] - 2, 0)

        # ── Construieste explicatia ──────────────────────────────────────────
        reasons = [
            f"Auto TF: {best_tf_setup['tf']} (verificate M1+M5)",
            f"Pattern: {best_tf_setup['pattern']}",
            best_tf_setup["msg"],
            f"ATR: {best_tf_setup['atr']:.5f} | SL ancorat structural"
            f" (dist {best_tf_setup['sl_dist']:.5f})",
            f"TP RR 1:{TP_RR_RATIO} → dist {best_tf_setup['tp_dist']:.5f}",
            f"Patternuri converg: {best_tf_setup['n_converging']}",
            f"HTF bias: {htf_dir}" + (" — counter-trend penalizat" if counter_trend else ""),
        ]

        # Daca avem semnale pe ambele TF-uri, mentioneaza
        if len(tf_results) > 1:
            other_tf = [t for t in tf_results if t != best_tf_setup["tf"]][0]
            other = tf_results[other_tf]
            reasons.append(
                f"Si {other_tf} a vazut {other['signal']} (conv {other['conviction']}) — "
                f"ales {best_tf_setup['tf']}"
            )

        tf_result = {
            "tf":         best_tf_setup["tf"],
            "signal":     best_tf_setup["signal"],
            "conviction": best_tf_setup["conviction"],
            "reasons":    reasons,
            "price":      round(best_tf_setup["price"], 5),
            "sl":         round(best_tf_setup["sl"], 5),
            "tp":         round(best_tf_setup["tp"], 5),
        }

        return self._build_result(
            symbol, [tf_result], min_confidence, min_votes=1,
            extra={
                "auto_tf":     best_tf_setup["tf"],
                "sl_distance": round(best_tf_setup["sl_dist"], 5),
                "tp_distance": round(best_tf_setup["tp_dist"], 5),
                "pattern":     best_tf_setup["pattern"],
            },
        )
