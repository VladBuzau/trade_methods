"""
ai_agent/features/structural.py — features structurale (multi-bar patterns).

Calculeaza ~30 features care descriu STRUCTURI vizuale ale graficului:
  - Support/Resistance proximity & quality
  - Trend structure (HH/HL/LH/LL counts)
  - Swings (last high/low, range, frequency)
  - Fractals (5/13/21 bar)
  - Multi-TF context (H4 trend, D1 direction)
  - Order Blocks & Liquidity sweeps
  - Market regime (trending vs ranging, ADX, BB squeeze)
  - Fibonacci levels

Aceste features sunt apoi MERGED cu cele 41 tehnice existente -> ~70 total
care alimenteaza XGBoost.
"""
from __future__ import annotations
import numpy as np
import pandas as pd


def _atr(df: pd.DataFrame, period: int = 14) -> float:
    h, l, c = df["high"].values, df["low"].values, df["close"].values
    if len(c) < period + 1:
        return 0.0
    tr = np.maximum(h[1:] - l[1:],
                    np.maximum(np.abs(h[1:] - c[:-1]),
                               np.abs(l[1:] - c[:-1])))
    return float(np.mean(tr[-period:]))


def _adx(df: pd.DataFrame, period: int = 14) -> float:
    h = df["high"].values
    l = df["low"].values
    c = df["close"].values
    n = len(c)
    if n < period * 2 + 5:
        return 0.0
    tr = np.maximum(h[1:] - l[1:],
                   np.maximum(np.abs(h[1:] - c[:-1]),
                              np.abs(l[1:] - c[:-1])))
    plus_dm  = np.where((h[1:] - h[:-1]) > (l[:-1] - l[1:]),
                        np.maximum(h[1:] - h[:-1], 0), 0)
    minus_dm = np.where((l[:-1] - l[1:]) > (h[1:] - h[:-1]),
                        np.maximum(l[:-1] - l[1:], 0), 0)
    def rma(v, p):
        out = np.zeros_like(v, dtype=float)
        if len(v) < p:
            return out
        out[p-1] = np.mean(v[:p])
        for i in range(p, len(v)):
            out[i] = (out[i-1] * (p-1) + v[i]) / p
        return out
    atr_s = rma(tr, period)
    den = np.where(atr_s == 0, 1, atr_s)
    plus_d  = 100 * rma(plus_dm, period) / den
    minus_d = 100 * rma(minus_dm, period) / den
    sum_d = plus_d + minus_d
    dx = 100 * np.abs(plus_d - minus_d) / np.where(sum_d == 0, 1, sum_d)
    adx = rma(dx, period)
    return float(adx[-1]) if len(adx) > 0 else 0.0


def _find_pivots(df: pd.DataFrame, window: int = 5) -> tuple[list, list]:
    """Returneaza pivot highs si pivot lows ca liste de (idx, price)."""
    h = df["high"].values
    l = df["low"].values
    n = len(h)
    ph, pl = [], []
    for i in range(window, n - window):
        if h[i] == max(h[i-window:i+window+1]):
            ph.append((i, float(h[i])))
        if l[i] == min(l[i-window:i+window+1]):
            pl.append((i, float(l[i])))
    return ph, pl


def _structure_count(ph: list, pl: list, lookback_bars: int, n: int) -> dict:
    """Counts HH/HL/LH/LL în ultimele lookback_bars bare."""
    cutoff = n - lookback_bars
    recent_ph = [(i, p) for i, p in ph if i >= cutoff]
    recent_pl = [(i, p) for i, p in pl if i >= cutoff]
    hh = hl = lh = ll = 0
    for k in range(1, len(recent_ph)):
        if recent_ph[k][1] > recent_ph[k-1][1]:
            hh += 1
        else:
            lh += 1
    for k in range(1, len(recent_pl)):
        if recent_pl[k][1] > recent_pl[k-1][1]:
            hl += 1
        else:
            ll += 1
    return {"hh": hh, "hl": hl, "lh": lh, "ll": ll}


def _cluster_levels(prices: list[float], tol_pct: float = 0.0015) -> list[dict]:
    """Grupeaza pretul apropiate (in toleranta) intr-o zona cu count."""
    if not prices:
        return []
    sorted_p = sorted(prices)
    clusters = [[sorted_p[0]]]
    for p in sorted_p[1:]:
        ref = np.mean(clusters[-1])
        if abs(p - ref) / ref <= tol_pct:
            clusters[-1].append(p)
        else:
            clusters.append([p])
    return [{"price": float(np.mean(c)), "touches": len(c)} for c in clusters]


def compute_structural_features(df: pd.DataFrame, idx: int) -> dict:
    """
    Computeaza features structurale pentru bara idx din df.
    df trebuie sa contina open/high/low/close/volume.
    """
    if idx < 50 or idx >= len(df):
        return {}
    sub = df.iloc[max(0, idx-200):idx+1].copy().reset_index(drop=True)
    n = len(sub)
    price = float(sub["close"].iloc[-1])
    atr = _atr(sub, 14)
    if atr <= 0:
        atr = 1e-10

    feats = {}

    # ── 1. ADX & Trend strength ────────────────────────────────────────────
    adx_val = _adx(sub, 14)
    feats["s_adx"] = round(adx_val, 2)
    if adx_val > 30:
        feats["s_regime"] = 2     # strong trend
    elif adx_val > 22:
        feats["s_regime"] = 1     # moderate trend
    else:
        feats["s_regime"] = 0     # range

    # ── 2. Pivots & swings ─────────────────────────────────────────────────
    ph, pl = _find_pivots(sub, window=5)

    feats["s_n_pivots_high"] = len(ph)
    feats["s_n_pivots_low"]  = len(pl)

    # Last swing high / low
    if ph:
        last_ph_idx, last_ph_price = ph[-1]
        feats["s_last_swing_high_dist_atr"] = round((last_ph_price - price) / atr, 2)
        feats["s_last_swing_high_age"]      = n - 1 - last_ph_idx
    else:
        feats["s_last_swing_high_dist_atr"] = 0.0
        feats["s_last_swing_high_age"]      = -1
    if pl:
        last_pl_idx, last_pl_price = pl[-1]
        feats["s_last_swing_low_dist_atr"] = round((price - last_pl_price) / atr, 2)
        feats["s_last_swing_low_age"]      = n - 1 - last_pl_idx
    else:
        feats["s_last_swing_low_dist_atr"] = 0.0
        feats["s_last_swing_low_age"]      = -1

    # ── 3. Market structure (HH/HL/LH/LL) ──────────────────────────────────
    struct = _structure_count(ph, pl, lookback_bars=100, n=n)
    feats["s_hh_count"] = struct["hh"]
    feats["s_hl_count"] = struct["hl"]
    feats["s_lh_count"] = struct["lh"]
    feats["s_ll_count"] = struct["ll"]
    bull_score = struct["hh"] + struct["hl"]
    bear_score = struct["lh"] + struct["ll"]
    if bull_score - bear_score >= 2:
        feats["s_market_structure"] = 1   # bull
    elif bear_score - bull_score >= 2:
        feats["s_market_structure"] = -1  # bear
    else:
        feats["s_market_structure"] = 0   # neutral

    # ── 4. S/R proximity (clustered pivots) ────────────────────────────────
    pivot_prices_high = [p for _, p in ph[-20:]]
    pivot_prices_low  = [p for _, p in pl[-20:]]
    resistances = _cluster_levels(pivot_prices_high)
    supports    = _cluster_levels(pivot_prices_low)

    # Closest resistance above
    above = [r for r in resistances if r["price"] > price]
    if above:
        nearest_r = min(above, key=lambda r: r["price"] - price)
        feats["s_dist_to_resistance_atr"] = round((nearest_r["price"] - price) / atr, 2)
        feats["s_resistance_touches"]     = nearest_r["touches"]
    else:
        feats["s_dist_to_resistance_atr"] = 10.0   # foarte departe
        feats["s_resistance_touches"]     = 0

    below = [s for s in supports if s["price"] < price]
    if below:
        nearest_s = max(below, key=lambda s: s["price"])
        feats["s_dist_to_support_atr"] = round((price - nearest_s["price"]) / atr, 2)
        feats["s_support_touches"]     = nearest_s["touches"]
    else:
        feats["s_dist_to_support_atr"] = 10.0
        feats["s_support_touches"]     = 0

    # ── 5. Pozitia in range ─────────────────────────────────────────────────
    recent_high = float(sub["high"].iloc[-50:].max())
    recent_low  = float(sub["low"].iloc[-50:].min())
    rng = recent_high - recent_low
    if rng > 0:
        feats["s_range_position"] = round((price - recent_low) / rng, 3)  # 0=low, 1=high
        feats["s_range_size_atr"] = round(rng / atr, 2)
    else:
        feats["s_range_position"] = 0.5
        feats["s_range_size_atr"] = 0

    # ── 6. Fractals 13-bar ──────────────────────────────────────────────────
    ph13, pl13 = _find_pivots(sub, window=13)
    if ph13:
        f_high = ph13[-1][1]
        feats["s_fractal13_high_dist_atr"] = round((f_high - price) / atr, 2)
    else:
        feats["s_fractal13_high_dist_atr"] = 10.0
    if pl13:
        f_low = pl13[-1][1]
        feats["s_fractal13_low_dist_atr"] = round((price - f_low) / atr, 2)
    else:
        feats["s_fractal13_low_dist_atr"] = 10.0

    # ── 7. Fibonacci retracement ───────────────────────────────────────────
    # Ultima miscare semnificativa = ultimul pivot vs precedent
    if len(ph) >= 1 and len(pl) >= 1:
        # Folosim cel mai recent pivot (high sau low)
        last_pivot_high = ph[-1] if ph else (0, price)
        last_pivot_low  = pl[-1] if pl else (0, price)
        if last_pivot_high[0] > last_pivot_low[0]:
            # Trend up: fib calculat de la low -> high
            move_high = last_pivot_high[1]
            move_low  = last_pivot_low[1]
            move_range = move_high - move_low
        else:
            # Trend down: fib calculat de la high -> low
            move_high = last_pivot_high[1]
            move_low  = last_pivot_low[1]
            move_range = move_low - move_high   # negativ
        if abs(move_range) > 1e-8:
            fib_618 = move_high - move_range * 0.618
            fib_500 = move_high - move_range * 0.500
            fib_382 = move_high - move_range * 0.382
            feats["s_fib_618_dist_atr"] = round(abs(price - fib_618) / atr, 2)
            feats["s_fib_500_dist_atr"] = round(abs(price - fib_500) / atr, 2)
            feats["s_fib_382_dist_atr"] = round(abs(price - fib_382) / atr, 2)
        else:
            feats["s_fib_618_dist_atr"] = 10.0
            feats["s_fib_500_dist_atr"] = 10.0
            feats["s_fib_382_dist_atr"] = 10.0
    else:
        feats["s_fib_618_dist_atr"] = 10.0
        feats["s_fib_500_dist_atr"] = 10.0
        feats["s_fib_382_dist_atr"] = 10.0

    # ── 8. Liquidity sweep recent (high/low broken in <10 bare) ────────────
    high_50 = sub["high"].iloc[-60:-10].max() if n >= 60 else float("inf")
    low_50  = sub["low"].iloc[-60:-10].min()  if n >= 60 else float("-inf")
    recent_high_10 = float(sub["high"].iloc[-10:].max())
    recent_low_10  = float(sub["low"].iloc[-10:].min())
    feats["s_swept_high_recent"] = 1 if recent_high_10 > high_50 else 0
    feats["s_swept_low_recent"]  = 1 if recent_low_10 < low_50 else 0

    # ── 9. Order Block detection (simplificat) ─────────────────────────────
    # OB bullish = ultima lumânare bearish înainte de o miscare puternica up
    ob_bull_nearby = 0
    ob_bear_nearby = 0
    o = sub["open"].values
    c = sub["close"].values
    h = sub["high"].values
    l = sub["low"].values
    for k in range(max(0, n-30), n-3):
        try:
            # Bullish OB: bearish candle + miscare puternica up dupa
            if c[k] < o[k]:
                future_high = max(h[k+1:k+4])
                bearish_body = o[k] - c[k]
                move_up = future_high - o[k]
                if move_up > 2 * bearish_body and bearish_body > 0:
                    ob_price = (o[k] + c[k]) / 2
                    if abs(price - ob_price) / atr < 2.0:
                        ob_bull_nearby = 1
                        break
            # Bearish OB
            if c[k] > o[k]:
                future_low = min(l[k+1:k+4])
                bullish_body = c[k] - o[k]
                move_down = o[k] - future_low
                if move_down > 2 * bullish_body and bullish_body > 0:
                    ob_price = (o[k] + c[k]) / 2
                    if abs(price - ob_price) / atr < 2.0:
                        ob_bear_nearby = 1
                        break
        except Exception:
            pass
    feats["s_ob_bull_nearby"] = ob_bull_nearby
    feats["s_ob_bear_nearby"] = ob_bear_nearby

    # ── 10. Higher TF context (din EMA pe 200 si 50) ───────────────────────
    # Aproximam HTF folosind EMA mai lente
    if n >= 200:
        ema200 = sub["close"].ewm(span=200, adjust=False).mean().iloc[-1]
        ema50  = sub["close"].ewm(span=50, adjust=False).mean().iloc[-1]
        # "H4 simulat" = EMA peste perioada mai mare
        feats["s_htf_ema200_above"] = 1 if price > ema200 else 0
        feats["s_htf_ema_slope"]    = round(float((ema50 - ema200) / ema200) * 100, 3)
    else:
        feats["s_htf_ema200_above"] = 0
        feats["s_htf_ema_slope"]    = 0.0

    # ── 11. Volatility contraction (BB squeeze intensity) ──────────────────
    if n >= 50:
        bbw_series = []
        for kk in range(max(0, n-100), n):
            window = sub["close"].iloc[max(0, kk-19):kk+1]
            if len(window) >= 20:
                std_w = float(window.std())
                mean_w = float(window.mean())
                if mean_w > 0:
                    bbw_series.append(std_w / mean_w)
        if len(bbw_series) >= 20:
            cur_bbw = bbw_series[-1]
            p_min, p_max = float(np.min(bbw_series)), float(np.max(bbw_series))
            if p_max - p_min > 0:
                feats["s_bbw_percentile"] = round((cur_bbw - p_min) / (p_max - p_min), 3)
            else:
                feats["s_bbw_percentile"] = 0.5
        else:
            feats["s_bbw_percentile"] = 0.5
    else:
        feats["s_bbw_percentile"] = 0.5

    return feats
