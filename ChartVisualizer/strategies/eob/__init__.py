"""
EOB — Enhanced Order Block v3 STRICT
─────────────────────────────────────
Gate-uri obligatorii (ALL trebuie sa treaca → HOLD dacă oricare esuaza):
  G1. Sesiune activa: London 06-16 UTC sau NY 12-21 UTC
  G2. HTF trend directional (nu NEUTRAL) + ALINIAT cu semnalul
  G3. Zona EOB HTF gasita SI pretul in interior (max 5% buffer)
  G4. MTF BOS confirmat grade B sau A (grade C → HOLD)
  G5. Entry Timing Score >= 6/10 pe LTF (>= 4 fara LTF)

Scoring (MAX_SCORE = 20):
  HTF  — pana la 10 pt  (decay + sweet spot + vol + stacking)
  MTF  — pana la  6 pt  (BOS grade A/B + FVG confluence + sweep)
  LTF  — pana la  6 pt  (DNA compression + displacement)

Confidence = scor / 20 × 100
Semnal de calitate: >= 85% (17/20 pt)
"""
from __future__ import annotations
import logging
from datetime import datetime, timezone

import numpy as np

from strategies.base import Strategy

log = logging.getLogger(__name__)

_EOB_ZONES_CACHE: dict[str, dict] = {}

def get_eob_context(symbol: str) -> dict | None:
    return _EOB_ZONES_CACHE.get(symbol.upper())

_TF_ORDER = ["M1", "M2", "M3", "M5", "M15", "M30", "H1", "H4", "D1", "W1"]

def _tf_rank(tf: str) -> int:
    return _TF_ORDER.index(tf) if tf in _TF_ORDER else 99

def _split_tfs(tfs):
    s = sorted(tfs, key=_tf_rank)
    ltf = [t for t in s if _tf_rank(t) <= _tf_rank("M5")]
    mtf = [t for t in s if _tf_rank("M15") <= _tf_rank(t) <= _tf_rank("H1")]
    htf = [t for t in s if _tf_rank(t) >= _tf_rank("H4")]
    if not htf and s: htf = [s[-1]]
    if not mtf and len(s) >= 2: mtf = [s[-2]]
    if not ltf and s: ltf = [s[0]]
    return ltf, mtf, htf


# ── Helpers ───────────────────────────────────────────────────────────────────

def _atr(df, period=14) -> float:
    if df is None or len(df) < period:
        return 0.0
    h = df["high"].values
    l = df["low"].values
    c = df["close"].values
    tr = np.maximum(h[1:] - l[1:],
         np.maximum(abs(h[1:] - c[:-1]), abs(l[1:] - c[:-1])))
    if len(tr) < period:
        return float(np.mean(tr))
    return float(np.mean(tr[-period:]))


def _adx(df, period=14) -> float:
    """ADX vectorizat — returneaza ultimul ADX value."""
    h = df["high"].values
    l = df["low"].values
    c = df["close"].values
    if len(c) < period * 2 + 5:
        return 0.0
    tr = np.maximum(h[1:] - l[1:],
                    np.maximum(abs(h[1:] - c[:-1]), abs(l[1:] - c[:-1])))
    plus_dm  = np.where((h[1:] - h[:-1]) > (l[:-1] - l[1:]),
                        np.maximum(h[1:] - h[:-1], 0.0), 0.0)
    minus_dm = np.where((l[:-1] - l[1:]) > (h[1:] - h[:-1]),
                        np.maximum(l[:-1] - l[1:], 0.0), 0.0)
    def rma(v, p):
        out = np.zeros(len(v), dtype=float)
        out[p-1] = np.mean(v[:p])
        for i in range(p, len(v)):
            out[i] = (out[i-1] * (p-1) + v[i]) / p
        return out
    atr_v   = rma(tr, period)
    safe    = np.where(atr_v == 0, 1.0, atr_v)
    plus_d  = 100.0 * rma(plus_dm, period) / safe
    minus_d = 100.0 * rma(minus_dm, period) / safe
    denom   = plus_d + minus_d
    dx      = 100.0 * np.abs(plus_d - minus_d) / np.where(denom == 0, 1.0, denom)
    adx_v   = rma(dx, period)
    return float(adx_v[-1])


def _in_session() -> bool:
    """London 06-16 UTC, NY 12-21 UTC. Returneaza True daca e ora activa."""
    import strategies as _strats
    hour = _strats.get_utc_now().hour
    return (6 <= hour < 16) or (12 <= hour < 21)


# ── FAZA 1: EOB Detection ─────────────────────────────────────────────────────

def detect_eob_approach_zones(df, direction: str, price_now: float, lookback: int = 100) -> list[dict]:
    """
    Zone EOB pending — STRICT (acelasi filtru ca detect_eob_zones_v2):
      Body>=40%, Wick>=40%, Range>=1.0×ATR si >=1.2×avg_range,
      confirmare DUBLA (2 bare in directia impulsului).
    """
    if df is None or len(df) < 30:
        return []

    opens  = df["open"].values
    highs  = df["high"].values
    lows   = df["low"].values
    closes = df["close"].values
    n = len(closes)
    start = max(2, n - lookback)

    ranges_all = highs - lows
    tr = np.maximum(highs[1:] - lows[1:],
                    np.maximum(np.abs(highs[1:] - closes[:-1]),
                               np.abs(lows[1:]  - closes[:-1])))
    atr14 = float(np.mean(tr[-14:])) if len(tr) >= 14 else float(np.mean(tr) or 0)

    vol_col = ("tick_volume" if "tick_volume" in df.columns
               else ("real_volume" if "real_volume" in df.columns else None))
    vol_sma20 = None
    if vol_col:
        vols = df[vol_col].values.astype(float)
        vol_sma20 = np.convolve(vols, np.ones(20)/20, mode="same")

    raw_zones = []
    for i in range(start, n - 2):
        rng  = highs[i] - lows[i]
        if rng < 1e-10:
            continue
        body = abs(closes[i] - opens[i])
        if body / rng < 0.40:
            continue
        if atr14 > 0 and rng < atr14 * 1.0:
            continue
        recent = ranges_all[max(0, i-20):i]
        avg_local = float(np.mean(recent)) if len(recent) > 0 else rng
        if avg_local > 0 and rng < avg_local * 1.2:
            continue

        bars_ago = n - 1 - i
        if bars_ago > 80:
            continue
        decay = max(0.3, 1.0 - bars_ago / 80.0)

        vol_ratio = 1.0
        vol_grade = "NORMAL"
        if vol_sma20 is not None and vol_sma20[i] > 0:
            vol_ratio = float(df[vol_col].iloc[i]) / vol_sma20[i]
            if vol_ratio >= 1.5:
                vol_grade = "STRONG"
            elif vol_ratio < 1.0:
                vol_grade = "WEAK"

        c1_rng  = highs[i+1] - lows[i+1]
        c1_body = abs(closes[i+1] - opens[i+1])
        c1_strong = c1_rng > 1e-10 and (c1_body / c1_rng) >= 0.50
        if not c1_strong:
            continue

        if direction == "BEARISH" and closes[i] > opens[i]:
            wick_bot = opens[i] - lows[i]
            if (wick_bot / rng) < 0.40:
                continue
            if not (closes[i+1] < opens[i] and closes[i+2] < closes[i+1]):
                continue
            z_lo = float(opens[i])
            z_hi = float(highs[i])
            if price_now < z_lo * 0.998:
                raw_zones.append({
                    "type": "BEARISH", "zone_low": round(z_lo, 5), "zone_high": round(z_hi, 5),
                    "sl": round(z_hi + (z_hi - z_lo) * 0.1, 5),
                    "pending_entry": round(z_lo, 5),
                    "order_type": "SELL_LIMIT",
                    "bars_ago": bars_ago, "decay": round(decay, 2),
                    "vol_ratio": round(vol_ratio, 2), "vol_grade": vol_grade,
                    "stacked": False,
                })

        elif direction == "BULLISH" and closes[i] < opens[i]:
            wick_top = highs[i] - opens[i]
            if (wick_top / rng) < 0.40:
                continue
            if not (closes[i+1] > opens[i] and closes[i+2] > closes[i+1]):
                continue
            z_lo = float(lows[i])
            z_hi = float(opens[i])
            if price_now > z_hi * 1.002:
                raw_zones.append({
                    "type": "BULLISH", "zone_low": round(z_lo, 5), "zone_high": round(z_hi, 5),
                    "sl": round(z_lo - (z_hi - z_lo) * 0.1, 5),
                    "pending_entry": round(z_hi, 5),
                    "order_type": "BUY_LIMIT",
                    "bars_ago": bars_ago, "decay": round(decay, 2),
                    "vol_ratio": round(vol_ratio, 2), "vol_grade": vol_grade,
                    "stacked": False,
                })

    for a in raw_zones:
        for b in raw_zones:
            if a is b:
                continue
            overlap_lo = max(a["zone_low"],  b["zone_low"])
            overlap_hi = min(a["zone_high"], b["zone_high"])
            if overlap_hi > overlap_lo:
                width_a   = a["zone_high"] - a["zone_low"]
                overlap_w = overlap_hi - overlap_lo
                if width_a > 0 and (overlap_w / width_a) >= 0.5:
                    a["stacked"] = True
                    break

    zones = [
        z for z in raw_zones
        if z["decay"] >= 0.60 and (z["vol_grade"] == "STRONG" or z["stacked"])
    ]
    return sorted(zones, key=lambda z: z["decay"] * (1 / max(z["bars_ago"], 1)), reverse=True)


def detect_eob_zones_v2(df, lookback: int = 100) -> list[dict]:
    """
    Zone EOB cu validare STRICTA: lumanarea trebuie sa fie semnificativa,
    nu o pin-bar mica langa una uriasa.

    Criterii STRICTE (toate trebuie indeplinite):
      G1. Body >= 40% din range (era 20%)
      G2. Wick respins >= 40% din range (era 30%)
      G3. Range absolut >= 1.0 × ATR(14) — lumanarea trebuie sa fie semnificativa
      G4. Range relativ >= 1.2 × media ranges ultimele 20 bare — domina contextul
      G5. CONFIRMARE DUBLA: 2 bare urmatoare in directia impulsului (nu doar 1)
      G6. Body bara confirmare 1 >= 50%, bara 2 close depaseste bara confirmare 1
    """
    if df is None or len(df) < 30:
        return []

    opens  = df["open"].values
    highs  = df["high"].values
    lows   = df["low"].values
    closes = df["close"].values
    n = len(closes)
    start = max(2, n - lookback)
    price_now = float(closes[-1])

    # ── ATR si average range pentru filtre de marime ──
    ranges_all = highs - lows
    tr = np.maximum(highs[1:] - lows[1:],
                    np.maximum(np.abs(highs[1:] - closes[:-1]),
                               np.abs(lows[1:]  - closes[:-1])))
    atr14 = float(np.mean(tr[-14:])) if len(tr) >= 14 else float(np.mean(tr) or 0)

    vol_col = ("tick_volume" if "tick_volume" in df.columns
               else ("real_volume" if "real_volume" in df.columns else None))
    vol_sma20 = None
    if vol_col:
        vols = df[vol_col].values.astype(float)
        vol_sma20 = np.convolve(vols, np.ones(20)/20, mode="same")

    zones = []

    # Confirmarea cere 2 bare urmatoare → opream cu 2 inainte de capat
    for i in range(start, n - 2):
        rng = highs[i] - lows[i]
        if rng < 1e-10:
            continue

        # G1: body >= 40%
        body = abs(closes[i] - opens[i])
        if body / rng < 0.40:
            continue

        # G3: range absolut >= 1.0 × ATR(14)
        if atr14 > 0 and rng < atr14 * 1.0:
            continue

        # G4: range relativ vs ultimele 20 bare (excluzand bara curenta)
        recent_window = ranges_all[max(0, i-20):i]
        avg_range_local = float(np.mean(recent_window)) if len(recent_window) > 0 else rng
        if avg_range_local > 0 and rng < avg_range_local * 1.2:
            continue

        bars_ago = n - 1 - i
        if bars_ago > 80:
            continue
        decay = max(0.3, 1.0 - bars_ago / 80.0)

        vol_ratio = 1.0
        vol_grade = "NORMAL"
        if vol_sma20 is not None and vol_sma20[i] > 0:
            raw_vol   = float(df[vol_col].iloc[i])
            vol_ratio = raw_vol / vol_sma20[i]
            if vol_ratio >= 1.5:
                vol_grade = "STRONG"
            elif vol_ratio < 1.0:
                vol_grade = "WEAK"

        # ── G5+G6: CONFIRMARE DUBLA — 2 bare in directia impulsului ──
        c1_rng  = highs[i+1] - lows[i+1]
        c1_body = abs(closes[i+1] - opens[i+1])
        c1_strong = c1_rng > 1e-10 and (c1_body / c1_rng) >= 0.50

        c2_rng  = highs[i+2] - lows[i+2]
        if not c1_strong or c2_rng < 1e-10:
            continue

        if closes[i] > opens[i]:
            # Lumanare bullish cu wick BOT → reversal BEARISH
            wick_bot = opens[i] - lows[i]
            # G2: wick >= 40%
            if (wick_bot / rng) < 0.40:
                continue
            # Bara 1: bear puternica + close < open (sub corp)
            if not (closes[i+1] < opens[i] and closes[i+1] < opens[i+1]):
                continue
            # Bara 2: continua jos (close < close bara 1)
            if not (closes[i+2] < closes[i+1]):
                continue
            z_lo = float(opens[i])
            z_hi = float(highs[i])
            if price_now <= z_hi * 1.002:
                zones.append({
                    "type": "BEARISH", "zone_low": round(z_lo, 5), "zone_high": round(z_hi, 5),
                    "sl":   round(z_hi + (z_hi - z_lo) * 0.1, 5),
                    "index": i, "bars_ago": bars_ago, "decay": round(decay, 2),
                    "wick_pct": round(wick_bot/rng*100, 1),
                    "body_pct": round(body/rng*100, 1),
                    "rng_vs_atr": round(rng/atr14, 2) if atr14 > 0 else 0,
                    "rng_vs_avg": round(rng/avg_range_local, 2) if avg_range_local > 0 else 0,
                    "vol_ratio": round(vol_ratio, 2), "vol_grade": vol_grade,
                    "engulf_close": round(float(closes[i+1]), 5),
                    "confirm2_close": round(float(closes[i+2]), 5),
                })

        elif closes[i] < opens[i]:
            # Lumanare bearish cu wick TOP → reversal BULLISH
            wick_top = highs[i] - opens[i]
            if (wick_top / rng) < 0.40:
                continue
            if not (closes[i+1] > opens[i] and closes[i+1] > opens[i+1]):
                continue
            if not (closes[i+2] > closes[i+1]):
                continue
            z_lo = float(lows[i])
            z_hi = float(opens[i])
            if price_now >= z_lo * 0.998:
                zones.append({
                    "type": "BULLISH", "zone_low": round(z_lo, 5), "zone_high": round(z_hi, 5),
                    "sl":   round(z_lo - (z_hi - z_lo) * 0.1, 5),
                    "index": i, "bars_ago": bars_ago, "decay": round(decay, 2),
                    "wick_pct": round(wick_top/rng*100, 1),
                    "body_pct": round(body/rng*100, 1),
                    "rng_vs_atr": round(rng/atr14, 2) if atr14 > 0 else 0,
                    "rng_vs_avg": round(rng/avg_range_local, 2) if avg_range_local > 0 else 0,
                    "vol_ratio": round(vol_ratio, 2), "vol_grade": vol_grade,
                    "engulf_close": round(float(closes[i+1]), 5),
                    "confirm2_close": round(float(closes[i+2]), 5),
                })

    for z in zones:
        z["stacked"] = False
        z["stack_count"] = 1
    for i, z1 in enumerate(zones):
        for j, z2 in enumerate(zones):
            if i >= j or z1["type"] != z2["type"]:
                continue
            lo = max(z1["zone_low"],  z2["zone_low"])
            hi = min(z1["zone_high"], z2["zone_high"])
            if hi > lo:
                min_size = min(z1["zone_high"]-z1["zone_low"], z2["zone_high"]-z2["zone_low"])
                if min_size > 0 and (hi - lo) / min_size >= 0.5:
                    z1["stacked"] = True
                    z2["stacked"] = True
                    z1["stack_count"] += 1
                    z2["stack_count"] += 1

    return zones


def _htf_trend_context(df, lookback: int = 20) -> tuple[str, float]:
    """HH/HL counting → STRONG_BULL / BULL / NEUTRAL / BEAR / STRONG_BEAR."""
    if df is None or len(df) < lookback + 5:
        return "NEUTRAL", 0.5

    closes = df["close"].values[-lookback:]
    highs  = df["high"].values[-lookback:]
    lows   = df["low"].values[-lookback:]

    bull_count = bear_count = 0
    for i in range(1, len(closes)):
        if highs[i] > highs[i-1] and lows[i] > lows[i-1]:
            bull_count += 1
        elif highs[i] < highs[i-1] and lows[i] < lows[i-1]:
            bear_count += 1

    total = bull_count + bear_count
    if total == 0:
        return "NEUTRAL", 0.5

    bull_ratio = bull_count / total
    if bull_ratio >= 0.70:
        return "STRONG_BULL", bull_ratio
    elif bull_ratio >= 0.55:
        return "BULL", bull_ratio
    elif bull_ratio <= 0.30:
        return "STRONG_BEAR", bull_ratio
    elif bull_ratio <= 0.45:
        return "BEAR", bull_ratio
    return "NEUTRAL", bull_ratio


def _eob_score_htf(zones, direction: str, price_now: float,
                   trend_ctx: str, bull_ratio: float) -> tuple[int, list[str], dict | None]:
    """
    Scor HTF pentru zona EOB. Returneaza (score, reasons, best_zone).
    Buffer zona: max 5% din marimea zonei (strictificat de la 10%).
    """
    target_zones = [z for z in zones if z["type"] == direction]
    if not target_zones:
        return 0, [f"HTF: nicio zona EOB {direction}"], None

    valid = []
    for z in target_zones:
        z_size = z["zone_high"] - z["zone_low"]
        buf    = z_size * 0.05  # 5% buffer (era 10%)
        in_zone = (z["zone_low"] - buf) <= price_now <= (z["zone_high"] + buf)
        if in_zone:
            valid.append(z)

    if not valid:
        best = target_zones[-1]
        return 0, [f"HTF: zona {direction} [{best['zone_low']}–{best['zone_high']}] — pret in afara (>5% buffer)"], None

    best = valid[-1]
    reasons = []
    score = 0

    base_score = int(5 * best["decay"])  # 3-5 pt bazat pe prospetime
    score += base_score
    reasons.append(f"HTF EOB {direction} [{best['zone_low']}–{best['zone_high']}]"
                   f" wick={best['wick_pct']}% age={best['bars_ago']}bare"
                   f" decay={best['decay']} (+{base_score}pt)")

    # Sweet spot (treimea de mijloc a zonei)
    z_lo, z_hi = best["zone_low"], best["zone_high"]
    mid_lo = z_lo + (z_hi - z_lo) / 3
    mid_hi = z_hi - (z_hi - z_lo) / 3
    if mid_lo <= price_now <= mid_hi:
        score += 2
        reasons.append("Sweet spot (treimea medie a zonei) (+2)")

    # Volum institutional
    if best["vol_grade"] == "STRONG":
        score += 2
        reasons.append(f"Volum institutional: {best['vol_ratio']}x SMA20 (+2)")
    elif best["vol_grade"] == "WEAK":
        score -= 1
        reasons.append(f"Volum slab: {best['vol_ratio']}x SMA20 (-1)")

    # Stacking confirmare
    if best.get("stacked") and best.get("stack_count", 1) > 1:
        score += 2
        reasons.append(f"EOB stacking: {best['stack_count']} zone suprapuse (+2)")

    return max(0, score), reasons, best


# ── FAZA 2: MTF ───────────────────────────────────────────────────────────────

def _bos_quality_grade(df, direction: str) -> tuple[str, int, str]:
    """
    Gradeaza BOS: A (+3), B (+1), C (0 — gate esuat).
    Grade C intoarce scor 0 dar semnaleaza si refuzul gate-ului.
    """
    from app import find_pivots, detect_bos
    try:
        ph_idx, pl_idx = find_pivots(df, lookback=5)
        bos = detect_bos(df, ph_idx, pl_idx)
        if bos != direction:
            return "NONE", 0, f"Nu exista BOS {direction}"

        closes = df["close"].values
        highs  = df["high"].values
        lows   = df["low"].values
        atr14  = _atr(df, 14)
        if atr14 <= 0:
            return "B", 1, "BOS confirmat (ATR indisponibil)"

        if direction == "BEARISH":
            if not pl_idx.size:
                return "B", 1, "BOS Bearish confirmat"
            broken_level  = float(lows[pl_idx[-1]])
            bos_strength  = (broken_level - closes[-1]) / atr14
        else:
            if not ph_idx.size:
                return "B", 1, "BOS Bullish confirmat"
            broken_level  = float(highs[ph_idx[-1]])
            bos_strength  = (closes[-1] - broken_level) / atr14

        if bos_strength >= 0.5:
            return "A", 3, f"BOS Grade A: strength={bos_strength:.2f} (+3)"
        elif bos_strength >= 0.2:
            return "B", 1, f"BOS Grade B: strength={bos_strength:.2f} (+1)"
        else:
            return "C", 0, f"BOS Grade C: strength={bos_strength:.2f} — prea slab (gate esuat)"
    except Exception as exc:
        return "B", 1, f"BOS confirmat (grade n/a: {exc})"


def _detect_liquidity_sweep(df, direction: str) -> tuple[bool, str]:
    """Spike rapid dincolo de un nivel recent + reversal rapid."""
    if df is None or len(df) < 10:
        return False, "Date insuficiente"
    closes = df["close"].values
    highs  = df["high"].values
    lows   = df["low"].values

    window = min(20, len(df))
    for i in range(-window, -2):
        body = abs(closes[i] - df["open"].values[i])
        rng  = highs[i] - lows[i]
        if rng < 1e-10:
            continue
        wick_ratio = 1 - (body / rng)
        if wick_ratio < 0.6:
            continue

        if direction == "BEARISH":
            wick_up = highs[i] - max(df["open"].values[i], closes[i])
            if wick_up / rng >= 0.6 and closes[i] < df["open"].values[i]:
                return True, f"Liquidity sweep bearish: wick {wick_up/rng*100:.0f}% sus"
        else:
            wick_dn = min(df["open"].values[i], closes[i]) - lows[i]
            if wick_dn / rng >= 0.6 and closes[i] > df["open"].values[i]:
                return True, f"Liquidity sweep bullish: wick {wick_dn/rng*100:.0f}% jos"

    return False, f"Niciun liquidity sweep {direction}"


def _fvg_eob_confluence(fvg, eob_zone) -> tuple[float, str]:
    """Overlap procentual intre FVG si zona EOB."""
    if not fvg or not eob_zone:
        return 0.0, "FVG sau EOB zona lipsa"
    flo, fhi = fvg.get("bottom", 0), fvg.get("top", 0)
    elo, ehi = eob_zone["zone_low"], eob_zone["zone_high"]
    overlap  = max(0, min(fhi, ehi) - max(flo, elo))
    min_size = min(fhi - flo, ehi - elo)
    if min_size <= 0:
        return 0.0, "Zone de marime 0"
    ratio = overlap / min_size
    return ratio, f"FVG+EOB overlap: {ratio*100:.0f}%"


# ── FAZA 3: LTF ───────────────────────────────────────────────────────────────

def _candle_dna(df, lookback: int = 10) -> dict:
    """
    DNA lumânari: body_ratio trend → accelerare/epuizare,
    range descrescator → compresie (explozie iminenta).
    """
    if df is None or len(df) < lookback + 2:
        return {"momentum": "NEUTRAL", "state": "UNKNOWN", "compression": False, "score": 0,
                "compression_len": 0, "body_ratios": []}

    opens  = df["open"].values[-lookback:]
    closes = df["close"].values[-lookback:]
    highs  = df["high"].values[-lookback:]
    lows   = df["low"].values[-lookback:]

    ranges      = highs - lows
    bodies      = np.abs(closes - opens)
    safe_ranges = np.where(ranges > 1e-10, ranges, 1.0)
    body_ratios = np.where(ranges > 1e-10, bodies / safe_ranges, 0.0)

    compression_len = 0
    for i in range(len(ranges)-1, 0, -1):
        if ranges[i] <= ranges[i-1] * 1.1:
            compression_len += 1
        else:
            break
    compression = compression_len >= 3

    if len(body_ratios) >= 3:
        recent = body_ratios[-3:]
        if recent[-1] > recent[-2] > recent[0]:
            momentum = "ACCELERATING"
        elif recent[-1] < recent[-2] < recent[0]:
            momentum = "EXHAUSTING"
        else:
            momentum = "NEUTRAL"
    else:
        momentum = "NEUTRAL"

    state = "NORMAL"
    if len(body_ratios) >= 2 and body_ratios[-2] >= 0.7 and body_ratios[-1] <= 0.3:
        state = "DOJI_AFTER_IMPULSE"

    score = 0
    if compression:
        score += 2
    if momentum == "ACCELERATING":
        score += 2
    elif momentum == "EXHAUSTING":
        score -= 1
    if state == "DOJI_AFTER_IMPULSE":
        score -= 2

    return {
        "momentum":        momentum,
        "state":           state,
        "compression":     compression,
        "compression_len": compression_len,
        "body_ratios":     [round(r, 2) for r in body_ratios[-5:]],
        "score":           score,
    }


def _displacement_v2(df, direction: str, lookback: int = 3,
                     body_pct: float = 0.60) -> tuple[bool, str]:
    """
    Corp >= 60% + close depaseste high/low ultimelor 3 bare.
    """
    if df is None or len(df) < lookback + 2:
        return False, "Date insuficiente pentru displacement V2"

    opens  = df["open"].values
    closes = df["close"].values
    highs  = df["high"].values
    lows   = df["low"].values

    for i in range(-lookback, 0):
        rng  = highs[i] - lows[i]
        if rng < 1e-10:
            continue
        body  = abs(closes[i] - opens[i])
        if body / rng < body_pct:
            continue
        if direction == "BUY"  and closes[i] <= opens[i]: continue
        if direction == "SELL" and closes[i] >= opens[i]: continue

        prev_start = max(0, len(closes) + i - 3)
        prev_end   = len(closes) + i
        if prev_end <= prev_start:
            continue

        if direction == "BUY":
            prev_high = highs[prev_start:prev_end].max()
            if closes[i] > prev_high:
                return True, f"Displacement V2 BUY: corp={body/rng*100:.0f}% close>{prev_high:.5f}"
        else:
            prev_low = lows[prev_start:prev_end].min()
            if closes[i] < prev_low:
                return True, f"Displacement V2 SELL: corp={body/rng*100:.0f}% close<{prev_low:.5f}"

    # Fallback: displacement simplu
    for i in range(-lookback, 0):
        rng  = highs[i] - lows[i]
        if rng < 1e-10: continue
        body  = abs(closes[i] - opens[i])
        if body / rng < body_pct: continue
        if direction == "BUY"  and closes[i] > opens[i]:
            return True, f"Displacement BUY basic: corp={body/rng*100:.0f}%"
        if direction == "SELL" and closes[i] < opens[i]:
            return True, f"Displacement SELL basic: corp={body/rng*100:.0f}%"

    return False, f"Niciun displacement {direction} in {lookback} bare"


def _entry_timing_score(dna: dict, disp_ok: bool, confluence_ratio: float,
                        sweep_ok: bool) -> tuple[int, str]:
    """
    Entry Timing Score 0-10.
    Gate: >= 6/10 pentru semnal valid (>= 4 fara LTF).
    """
    score = 0
    parts = []

    if dna.get("momentum") == "ACCELERATING":
        score += 2
        parts.append("momentum+2")
    if dna.get("compression"):
        score += 2
        parts.append(f"compresie({dna.get('compression_len',0)}bare)+2")
    if disp_ok:
        score += 3
        parts.append("displacement+3")
    if confluence_ratio >= 0.5:
        score += 2
        parts.append(f"confluenta{confluence_ratio*100:.0f}%+2")
    elif confluence_ratio >= 0.2:
        score += 1
        parts.append("confluenta_partiala+1")
    if sweep_ok:
        score += 1
        parts.append("sweep+1")

    desc = f"Entry score: {score}/10 [{', '.join(parts) if parts else 'nimic'}]"
    return score, desc


def _range_projection(df) -> tuple[float, str]:
    atr14 = _atr(df, 14)
    atr5  = _atr(df, 5)
    if atr14 <= 0 or atr5 <= 0:
        return 1.0, "ATR indisponibil"
    exp = atr14 / atr5
    return atr14 * exp, f"Range estimat: ATR14={atr14:.5f} × {exp:.1f} = {atr14*exp:.5f}"


def _time_to_move(dna: dict, compression_len: int) -> tuple[int, str]:
    if dna.get("state") == "DOJI_AFTER_IMPULSE":
        return 2, "Doji dupa impuls → miscare iminent 2-3 bare"
    if dna.get("compression") and compression_len >= 5:
        return 1, "Compresie >=5 bare → explozie iminent 1-2 bare"
    elif dna.get("compression"):
        return 3, f"Compresie {compression_len} bare → miscare 3-5 bare"
    if dna.get("momentum") == "ACCELERATING":
        return 2, "Momentum accelerat → continuare imediata"
    return 5, "Timing incert — asteapta confirmare"


def _invalidation_map(price_now: float, signal: str, htf_zone: dict,
                      mtf_fvg=None, bos_level: float | None = None) -> dict:
    z_lo = htf_zone["zone_low"]
    z_hi = htf_zone["zone_high"]
    z_mid = (z_lo + z_hi) / 2

    if signal == "SELL":
        hard_stop       = round(z_hi * 1.001, 5)
        soft_invalidate = round(z_mid, 5)
        bos_recapture   = bos_level or round(z_hi * 1.003, 5)
    else:
        hard_stop       = round(z_lo * 0.999, 5)
        soft_invalidate = round(z_mid, 5)
        bos_recapture   = bos_level or round(z_lo * 0.997, 5)

    return {
        "hard_stop":       hard_stop,
        "soft_invalidate": soft_invalidate,
        "bos_recapture":   bos_recapture,
        "description": (
            f"Hard stop: {hard_stop} | "
            f"Soft invalidare: {soft_invalidate} | "
            f"BOS recapture: {bos_recapture}"
        )
    }


# ── Strategia principala ──────────────────────────────────────────────────────
class EOBStrategy(Strategy):
    key   = "eob"
    name  = "EOB Pure"           # eliminam Unicorn — doar EOB logic
    icon  = "🟣"
    color = "#9c27b0"

    default_tfs            = ["M15", "H1"]
    default_bars           = 2000
    default_step           = 1        # fiecare bara H1/M15 — nu ratam zone
    default_duration_years = 2.0      # 2 ani — strategie swing, nevoie istoric
    default_max_hours      = 24.0     # 24h max — EOB tine pozitia mai mult

    # Note: liniar single-trade e enforced de backtest engine si autotrader.
    # Nu adaugam lock duplicat ca poate cauza blocaje fara auto-release.

    elements = {
        "eob_htf":     "EOB HTF (H4/D1) — zona context [OBLIGATORIU]",
        "eob_ltf":     "EOB din EOB — intrare pe LTF (M1/M5)",
        "session":     "Session Filter — London/NY only [gate]",
        "adx_filter":  "ADX > 18 pe HTF (trend valabil) [gate]",
        "adx_strict":  "STRICT: ADX > 25 pe HTF (vs default 18)",
        "fresh_eob":   "STRICT: EOB age < 50 bare (anti zone vechi)",
        "session_only_lon_ny_overlap": "STRICT: doar overlap LO+NY 13-15 UTC",
        "no_news_window": "STRICT: skip 30 min in jurul stirilor red",
    }

    # Scor maxim: HTF(10) + LTF(6) = 16 (fara Unicorn MTF de 6 puncte)
    MAX_SCORE = 16

    def analyze(self, symbol, tfs, bars=2000, tf_bars=None, elements=None,
                min_confidence=85.0, **kwargs):
        from app import fetch, find_fvg

        if elements is None:
            elements = {k: True for k in self.elements}

        ltf_list, mtf_list, htf_list = _split_tfs(tfs)
        all_tfs = list(dict.fromkeys(htf_list + mtf_list + ltf_list))

        # ── G1: Session gate ──────────────────────────────────────────────
        if elements.get("session", True) and not _in_session():
            return self._empty_result(symbol, "G1: Sesiune inactiva (London 06-16 / NY 12-21 UTC)")

        price_now   = None
        reasons     = []
        score_sell  = score_buy = 0
        htf_bear_zone = htf_bull_zone = None
        mtf_bear_fvg  = mtf_bull_fvg  = None
        dna_result = {"momentum": "NEUTRAL", "compression": False, "score": 0, "compression_len": 0}
        entry_score_sell = entry_score_buy = 0
        inv_map = None
        htf_trend_ctx = "NEUTRAL"

        tf_c = {tf: {"tf": tf, "phase": "—", "signal": "—",
                     "score_buy": 0, "score_sell": 0, "found": "—",
                     "zone": None, "sl": 0, "tp": 0, "price": 0}
                for tf in all_tfs}

        # ─────────────────────────────────────────────────────────────────
        # FAZA 1: HTF — zona EOB + trend context
        # ─────────────────────────────────────────────────────────────────
        if elements.get("eob_htf", True):
            for tf in htf_list:
                tf_c[tf]["phase"] = "HTF"
                try:
                    n_bars = (tf_bars or {}).get(tf, bars)
                    df, _  = fetch(symbol, tf, n_bars)
                    if df is None or len(df) < 30:
                        tf_c[tf]["found"] = "Date insuficiente"
                        continue

                    price_now = float(df["close"].values[-1])
                    tf_c[tf]["price"] = price_now

                    # G5 optional: ADX gate
                    if elements.get("adx_filter", True):
                        adx_val = _adx(df, 14)
                        if adx_val < 18:
                            tf_c[tf]["found"] = f"ADX {adx_val:.1f} < 18 — range, skip HTF"
                            reasons.append(f"G5 ADX: {adx_val:.1f} < 18 pe {tf} — skip")
                            continue
                        reasons.append(f"ADX {adx_val:.1f} >= 18 pe {tf} ✓")

                    zones = detect_eob_zones_v2(df, lookback=100)
                    trend_ctx, bull_ratio = _htf_trend_context(df, lookback=20)
                    htf_trend_ctx = trend_ctx
                    reasons.append(f"HTF trend: {trend_ctx} ({bull_ratio*100:.0f}% bull)")

                    # ── G2: Trend directional (nu NEUTRAL) ──
                    if trend_ctx == "NEUTRAL":
                        tf_c[tf]["found"] = "G2: HTF trend NEUTRAL — skip"
                        reasons.append(f"G2: HTF trend NEUTRAL pe {tf} — semnale ignorate")
                        continue

                    # BEARISH EOB scoring (doar daca trend e BEAR)
                    if trend_ctx in ("BEAR", "STRONG_BEAR"):
                        s_sell, r_sell, best_bear = _eob_score_htf(
                            zones, "BEARISH", price_now, trend_ctx, bull_ratio
                        )
                        if best_bear:
                            htf_bear_zone = best_bear
                            score_sell += s_sell
                            tf_c[tf]["score_sell"] += s_sell
                            tf_c[tf]["signal"] = "SELL"
                            tf_c[tf]["zone"]   = best_bear
                            reasons.extend(r_sell)

                    # BULLISH EOB scoring (doar daca trend e BULL)
                    if trend_ctx in ("BULL", "STRONG_BULL"):
                        s_buy, r_buy, best_bull = _eob_score_htf(
                            zones, "BULLISH", price_now, trend_ctx, bull_ratio
                        )
                        if best_bull:
                            htf_bull_zone = best_bull
                            score_buy += s_buy
                            tf_c[tf]["score_buy"] += s_buy
                            if tf_c[tf]["signal"] == "—":
                                tf_c[tf]["signal"] = "BUY"
                            if not tf_c[tf]["zone"]:
                                tf_c[tf]["zone"] = best_bull
                            reasons.extend(r_buy)

                    found = []
                    if htf_bear_zone: found.append(f"BEARISH [{htf_bear_zone['zone_low']}–{htf_bear_zone['zone_high']}]")
                    if htf_bull_zone: found.append(f"BULLISH [{htf_bull_zone['zone_low']}–{htf_bull_zone['zone_high']}]")
                    tf_c[tf]["found"] = "; ".join(found) if found else "Nicio zona EOB activa sau trend NEUTRAL"

                except Exception as exc:
                    log.warning(f"EOB HTF {symbol}/{tf}: {exc}")
                    tf_c[tf]["found"] = f"Eroare: {exc}"

        # ── G3: HTF zona obligatorie ──────────────────────────────────────
        if htf_bear_zone is None and htf_bull_zone is None:
            if price_now is None:
                return self._empty_result(symbol, "Nu am date de pret")

            # Cauta zone pending (pretul se apropie dar nu e inca in zona)
            pending_entry = None
            for tf in htf_list:
                try:
                    n_bars_p = (tf_bars or {}).get(tf, bars)
                    df_p, _  = fetch(symbol, tf, n_bars_p)
                    if df_p is None:
                        continue
                    bear_pending = detect_eob_approach_zones(df_p, "BEARISH", price_now, lookback=100)
                    bull_pending = detect_eob_approach_zones(df_p, "BULLISH", price_now, lookback=100)
                    if bear_pending:
                        z = bear_pending[0]
                        tp_p = round(z["pending_entry"] - abs(z["pending_entry"] - z["sl"]) * 1.5, 5)
                        pending_entry = {
                            "signal": "SELL", "order_type": z["order_type"],
                            "price": z["pending_entry"], "sl": z["sl"], "tp": tp_p,
                            "zone_low": z["zone_low"], "zone_high": z["zone_high"],
                            "tf": tf, "decay": z["decay"], "bars_ago": z["bars_ago"],
                        }
                        break
                    if bull_pending:
                        z = bull_pending[0]
                        tp_p = round(z["pending_entry"] + abs(z["pending_entry"] - z["sl"]) * 1.5, 5)
                        pending_entry = {
                            "signal": "BUY", "order_type": z["order_type"],
                            "price": z["pending_entry"], "sl": z["sl"], "tp": tp_p,
                            "zone_low": z["zone_low"], "zone_high": z["zone_high"],
                            "tf": tf, "decay": z["decay"], "bars_ago": z["bars_ago"],
                        }
                        break
                except Exception:
                    pass

            result = self._empty_result(symbol,
                "G3: nicio zona EOB activa cu pretul in interior — astept retrasare")
            if pending_entry:
                result["pending_entry"] = pending_entry
                result["justification"].append(
                    f"Zona {pending_entry['order_type']} @ {pending_entry['price']}"
                    f" [{pending_entry['zone_low']}–{pending_entry['zone_high']}]"
                    f" SL={pending_entry['sl']}  TP={pending_entry['tp']}"
                )
            return result

        # ─────────────────────────────────────────────────────────────────
        # FAZA 2: MTF — Unicorn (BOS + FVG + Sweep)
        # ELIMINAT din EOB Pure. Cod pastrat dar dezactivat default.
        # Daca vrei sa-l revii, seteaza elements["unicorn"] = True
        # ─────────────────────────────────────────────────────────────────
        bos_grade_sell = bos_grade_buy = "NONE"

        if elements.get("unicorn", False):  # ← default OFF (era True)
            for tf in mtf_list:
                tf_c[tf]["phase"] = "MTF (Unicorn)"
                try:
                    n_bars = (tf_bars or {}).get(tf, bars)
                    df, _  = fetch(symbol, tf, n_bars)
                    if df is None or len(df) < 50:
                        tf_c[tf]["found"] = "Date insuficiente"
                        continue
                    if price_now is None:
                        price_now = float(df["close"].values[-1])
                    tf_c[tf]["price"] = price_now

                    fvgs = find_fvg(df, lookback=80)
                    found_parts = []

                    # BOS SELL
                    if htf_bear_zone:
                        grade, bos_sc, bos_msg = _bos_quality_grade(df, "BEARISH")
                        bos_grade_sell = grade
                        sweep_ok, sweep_msg = _detect_liquidity_sweep(df, "BEARISH")
                        bear_fvgs = [f for f in fvgs if f["type"] == "BEARISH"]
                        s = 0
                        if grade in ("A", "B"):
                            s += bos_sc
                        if bear_fvgs:
                            fvg = bear_fvgs[-1]
                            conf_ratio, conf_msg = _fvg_eob_confluence(fvg, htf_bear_zone)
                            in_fvg = fvg["bottom"] <= price_now <= fvg["top"]
                            if in_fvg:
                                s += 2
                                mtf_bear_fvg = fvg
                            if conf_ratio >= 0.5:
                                s += 2
                            if sweep_ok:
                                s += 1
                            found_parts.append(f"{bos_msg} | FVG {'IN' if in_fvg else 'out'}"
                                               f" | {conf_msg} | {sweep_msg}")
                            reasons.append(f"MTF {tf} SELL: {bos_msg} conf={conf_ratio*100:.0f}% sweep={sweep_ok}")
                        elif not bear_fvgs:
                            reasons.append(f"MTF {tf} SELL: {bos_msg} — fara FVG bearish")
                        score_sell += s
                        tf_c[tf]["score_sell"] += s

                    # BOS BUY
                    if htf_bull_zone:
                        grade, bos_sc, bos_msg = _bos_quality_grade(df, "BULLISH")
                        bos_grade_buy = grade
                        sweep_ok, sweep_msg = _detect_liquidity_sweep(df, "BULLISH")
                        bull_fvgs = [f for f in fvgs if f["type"] == "BULLISH"]
                        s = 0
                        if grade in ("A", "B"):
                            s += bos_sc
                        if bull_fvgs:
                            fvg = bull_fvgs[-1]
                            conf_ratio, conf_msg = _fvg_eob_confluence(fvg, htf_bull_zone)
                            in_fvg = fvg["bottom"] <= price_now <= fvg["top"]
                            if in_fvg:
                                s += 2
                                mtf_bull_fvg = fvg
                            if conf_ratio >= 0.5:
                                s += 2
                            if sweep_ok:
                                s += 1
                            found_parts.append(f"{bos_msg} | FVG {'IN' if in_fvg else 'out'}"
                                               f" | {conf_msg} | {sweep_msg}")
                            reasons.append(f"MTF {tf} BUY: {bos_msg} conf={conf_ratio*100:.0f}% sweep={sweep_ok}")
                        elif not bull_fvgs:
                            reasons.append(f"MTF {tf} BUY: {bos_msg} — fara FVG bullish")
                        score_buy += s
                        tf_c[tf]["score_buy"] += s

                    if tf_c[tf]["score_sell"] > 0:
                        tf_c[tf]["signal"] = "SELL"
                    elif tf_c[tf]["score_buy"] > 0:
                        tf_c[tf]["signal"] = "BUY"
                    tf_c[tf]["found"] = "; ".join(found_parts) if found_parts else "Fara BOS/FVG confirmat"

                except Exception as exc:
                    log.warning(f"EOB MTF {symbol}/{tf}: {exc}")
                    tf_c[tf]["found"] = f"Eroare: {exc}"

        # ── G4: BOS obligatoriu — ACTIV doar daca Unicorn e activ ──────────
        # In modul EOB Pure (Unicorn OFF), nu blocam pe BOS grade
        if elements.get("unicorn", False):
            dominant_signal = "SELL" if score_sell > score_buy else ("BUY" if score_buy > score_sell else "NONE")
            if dominant_signal == "SELL" and htf_bear_zone and bos_grade_sell == "C":
                return self._empty_result(symbol,
                    f"G4: BOS SELL Grade C — prea slab pentru intrare")
            if dominant_signal == "BUY" and htf_bull_zone and bos_grade_buy == "C":
                return self._empty_result(symbol,
                    f"G4: BOS BUY Grade C — prea slab pentru intrare")

        # ─────────────────────────────────────────────────────────────────
        # FAZA 3: LTF — Candle DNA + Displacement
        # ─────────────────────────────────────────────────────────────────
        if elements.get("eob_ltf", True) and ltf_list:
            for tf in sorted(ltf_list, key=_tf_rank):
                tf_c[tf]["phase"] = "LTF"
                try:
                    n_bars = (tf_bars or {}).get(tf, bars)
                    df, _  = fetch(symbol, tf, min(n_bars, 100))
                    if df is None or len(df) < 15:
                        tf_c[tf]["found"] = "Date insuficiente"
                        continue
                    if price_now is None:
                        price_now = float(df["close"].values[-1])
                    tf_c[tf]["price"] = price_now

                    dna_result    = _candle_dna(df, lookback=10)
                    compression_len = dna_result.get("compression_len", 0)
                    exp_range, proj_msg = _range_projection(df)
                    ttm, ttm_msg  = _time_to_move(dna_result, compression_len)

                    reasons.append(f"LTF {tf}: momentum={dna_result['momentum']}"
                                   f" compresie={dna_result['compression']}({compression_len}bare)"
                                   f" state={dna_result['state']}")

                    found_parts = [
                        f"DNA: {dna_result['momentum']}/{dna_result['state']}",
                        f"Compresie: {compression_len}bare",
                        ttm_msg,
                    ]

                    disp_sell_ok, disp_sell_msg = _displacement_v2(df, "SELL")
                    disp_buy_ok,  disp_buy_msg  = _displacement_v2(df, "BUY")

                    if htf_bear_zone:
                        conf_r, _ = _fvg_eob_confluence(mtf_bear_fvg, htf_bear_zone) if mtf_bear_fvg else (0.0, "")
                        sweep_ok2, _ = _detect_liquidity_sweep(df, "BEARISH")
                        e_score, e_desc = _entry_timing_score(dna_result, disp_sell_ok, conf_r, sweep_ok2)
                        entry_score_sell = e_score
                        score_sell += dna_result["score"] + (3 if disp_sell_ok else 0)
                        tf_c[tf]["score_sell"] += dna_result["score"] + (3 if disp_sell_ok else 0)
                        found_parts.append(f"SELL: {disp_sell_msg} | {e_desc}")
                        reasons.append(f"LTF SELL: {e_desc}")

                    if htf_bull_zone:
                        conf_r, _ = _fvg_eob_confluence(mtf_bull_fvg, htf_bull_zone) if mtf_bull_fvg else (0.0, "")
                        sweep_ok2, _ = _detect_liquidity_sweep(df, "BULLISH")
                        e_score, e_desc = _entry_timing_score(dna_result, disp_buy_ok, conf_r, sweep_ok2)
                        entry_score_buy = e_score
                        score_buy += dna_result["score"] + (3 if disp_buy_ok else 0)
                        tf_c[tf]["score_buy"] += dna_result["score"] + (3 if disp_buy_ok else 0)
                        found_parts.append(f"BUY: {disp_buy_msg} | {e_desc}")
                        reasons.append(f"LTF BUY: {e_desc}")

                    if tf_c[tf]["signal"] == "—":
                        if tf_c[tf]["score_sell"] > tf_c[tf]["score_buy"]:
                            tf_c[tf]["signal"] = "SELL"
                        elif tf_c[tf]["score_buy"] > tf_c[tf]["score_sell"]:
                            tf_c[tf]["signal"] = "BUY"

                    tf_c[tf]["found"] = " | ".join(found_parts)

                except Exception as exc:
                    log.warning(f"EOB LTF {symbol}/{tf}: {exc}")
                    tf_c[tf]["found"] = f"Eroare: {exc}"

        # ─────────────────────────────────────────────────────────────────
        # SEMNAL FINAL
        # ─────────────────────────────────────────────────────────────────
        if price_now is None:
            return self._empty_result(symbol, "Nu am date de pret")

        tfs_for_charts = [
            {
                "tf": c["tf"], "phase": c["phase"], "signal": c["signal"],
                "found": c["found"], "score_buy": c["score_buy"],
                "score_sell": c["score_sell"],
                "trend": ("BULLISH" if c["score_buy"] > c["score_sell"]
                          else "BEARISH" if c["score_sell"] > c["score_buy"] else "—"),
                "conviction": c["score_buy"] + c["score_sell"],
                "zone": c["zone"], "price": c["price"] or price_now,
                "sl": c["sl"], "tp": 0,
            }
            for c in [tf_c[tf] for tf in all_tfs]
        ]

        def _make_hold(msg):
            reasons.append(msg)
            return {
                "symbol": symbol, "strategy": self.key, "timestamp": self._now(),
                "signal": "HOLD",
                "confidence": round(min(max(score_sell, score_buy), self.MAX_SCORE) / self.MAX_SCORE * 100, 1),
                "n_buy": score_buy, "n_sell": score_sell,
                "n_total": score_buy + score_sell,
                "best_tf": None, "tfs": tfs_for_charts,
                "justification": reasons, "auto_executed": False,
            }

        # Conflict simetric — nu intra
        if score_sell > 0 and score_buy > 0 and abs(score_sell - score_buy) < 4:
            return _make_hold(
                f"CONFLICT: SELL={score_sell} BUY={score_buy} — diferenta < 4pt, semnal anulat"
            )

        if score_buy == 0 and score_sell == 0:
            return _make_hold("Nicio aliniere EOB detectata")

        if score_sell > score_buy:
            signal    = "SELL"
            raw_score = score_sell
            entry_zone = htf_bear_zone
            entry_tf   = ltf_list[0] if ltf_list else htf_list[0]

            # ── G5: Entry Score >= 6 (>= 4 fara LTF) ──
            min_es = 4 if not ltf_list else 6
            if entry_score_sell < min_es:
                return _make_hold(
                    f"G5: SELL Entry Score {entry_score_sell}/10 < {min_es} minim"
                    f" — asteapta setup mai clar"
                )

            sl = entry_zone["sl"] if entry_zone else round(price_now * 1.003, 5)
            tp = _calc_tp(price_now, sl, signal, ratio=1.5)
            best_tf = {"tf": entry_tf, "signal": "SELL", "trend": "BEARISH",
                       "price": price_now, "sl": sl, "tp": tp,
                       "conviction": score_sell, "reasons": reasons}
            for t in tfs_for_charts:
                if t["tf"] == entry_tf:
                    t["sl"] = sl; t["tp"] = tp
            inv_map = _invalidation_map(price_now, "SELL", htf_bear_zone, mtf_bear_fvg) if htf_bear_zone else None

        elif score_buy > score_sell:
            signal    = "BUY"
            raw_score = score_buy
            entry_zone = htf_bull_zone
            entry_tf   = ltf_list[0] if ltf_list else htf_list[0]

            min_es = 4 if not ltf_list else 6
            if entry_score_buy < min_es:
                return _make_hold(
                    f"G5: BUY Entry Score {entry_score_buy}/10 < {min_es} minim"
                    f" — asteapta setup mai clar"
                )

            sl = entry_zone["sl"] if entry_zone else round(price_now * 0.997, 5)
            tp = _calc_tp(price_now, sl, signal, ratio=1.5)
            best_tf = {"tf": entry_tf, "signal": "BUY", "trend": "BULLISH",
                       "price": price_now, "sl": sl, "tp": tp,
                       "conviction": score_buy, "reasons": reasons}
            for t in tfs_for_charts:
                if t["tf"] == entry_tf:
                    t["sl"] = sl; t["tp"] = tp
            inv_map = _invalidation_map(price_now, "BUY", htf_bull_zone, mtf_bull_fvg) if htf_bull_zone else None

        else:
            return _make_hold("Score egal BUY/SELL — conflict")

        confidence = round(min(raw_score, self.MAX_SCORE) / self.MAX_SCORE * 100, 1)
        reasons.append(f"Score SELL={score_sell} BUY={score_buy} → {signal} {confidence}%"
                       f" | HTF trend: {htf_trend_ctx}")
        if inv_map:
            reasons.append(f"Invalidation: {inv_map['description']}")

        _EOB_ZONES_CACHE[symbol.upper()] = {
            "signal": signal,
            "htf_bear_zone": htf_bear_zone, "htf_bull_zone": htf_bull_zone,
            "mtf_bear_fvg":  mtf_bear_fvg,  "mtf_bull_fvg":  mtf_bull_fvg,
            "inv_map":       inv_map,
        }

        return {
            "symbol": symbol, "strategy": self.key, "timestamp": self._now(),
            "signal": signal, "confidence": confidence,
            "n_buy": score_buy, "n_sell": score_sell,
            "n_total": score_buy + score_sell,
            "best_tf": best_tf, "tfs": tfs_for_charts,
            "justification": reasons, "auto_executed": False,
            "eob_context": {
                "htf_bear_zone": htf_bear_zone, "htf_bull_zone": htf_bull_zone,
                "mtf_bear_fvg": mtf_bear_fvg, "mtf_bull_fvg": mtf_bull_fvg,
                "inv_map": inv_map,
                "dna": dna_result,
                "entry_score_sell": entry_score_sell,
                "entry_score_buy": entry_score_buy,
            },
        }


def _calc_tp(price: float, sl: float, signal: str, ratio: float = 1.5) -> float:
    risk = abs(price - sl)
    return round(price - risk * ratio if signal == "SELL" else price + risk * ratio, 5)
