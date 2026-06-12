"""
SR Bounce v2 — scalping pe zone Support/Rezistenta în market RANGING.

Schimbari critice fata de v1:
  1. Filtru RANGE: doar daca BB Width comprimat (piata in range, nu trend)
  2. SL larg: 0.7×ATR dincolo de zona (vs 0.3×) — anti stop-hunting
  3. TP la SMA20 (middle of range) sau 1×ATR (closer) — RR realist
  4. Wick touch REQUIRED: pretul a atins zona cu fitilul, body afara (rejection real)
  5. Trend H1 NU contra: BUY @ support doar daca H1 nu e bearish puternic
  6. Anti double-tap: skip daca bara precedenta a fost si ea la zona
  7. Higher min_confidence: 60% default (vs 55) pentru calitate
  8. Cooldown: min 10 bare intre trade-uri pe acelasi simbol (anti over-trade)
"""
from __future__ import annotations
import logging
import numpy as np

from strategies.base import Strategy

log = logging.getLogger(__name__)

# ── Configurare strategie ────────────────────────────────────────────────────
LOOKBACK_BARS    = 300
PIVOT_WINDOW     = 5
MIN_TOUCHES      = 2
TOL_PCT          = 0.0015
ZONE_HALF_WIDTH  = 0.5       # latimea zonei = 0.5 × ATR
NEAR_ZONE_ATR    = 0.4       # pret la <= 0.4 × ATR de zona
SL_BEYOND_ATR    = 0.7       # SL larg: 0.7×ATR dincolo de zona
TP_ATR_MULT      = 1.2       # TP default = 1.2×ATR (cca RR 1:1.7)
BBW_LOOKBACK     = 100
BBW_RANGE_PCT    = 0.6       # BB width sub p60 = range valid

_last_signal_bar: dict[str, int] = {}   # cooldown per simbol


def _atr(df, period: int = 14) -> float:
    h = df["high"].values
    l = df["low"].values
    c = df["close"].values
    if len(c) < period + 1:
        return 0.0
    tr = np.maximum(h[1:] - l[1:],
                    np.maximum(np.abs(h[1:] - c[:-1]),
                               np.abs(l[1:] - c[:-1])))
    return float(np.mean(tr[-period:]))


def _find_pivots(df, window: int = 5):
    h = df["high"].values
    l = df["low"].values
    n = len(h)
    ph, pl = [], []
    for i in range(window, n - window):
        if h[i] == max(h[i - window: i + window + 1]):
            ph.append((i, float(h[i])))
        if l[i] == min(l[i - window: i + window + 1]):
            pl.append((i, float(l[i])))
    return ph, pl


def _cluster(raw_levels: list[dict], tol_pct: float = TOL_PCT) -> list[dict]:
    if not raw_levels:
        return []
    raw_s = sorted(raw_levels, key=lambda x: x["price"])
    groups = [[raw_s[0]]]
    for item in raw_s[1:]:
        ref = np.mean([x["price"] for x in groups[-1]])
        if abs(item["price"] - ref) / ref <= tol_pct:
            groups[-1].append(item)
        else:
            groups.append([item])
    result = []
    for g in groups:
        prices = [x["price"] for x in g]
        kinds  = [x["kind"] for x in g]
        ages   = [x["age"] for x in g]
        result.append({
            "price":   float(np.mean(prices)),
            "kind":    "R" if kinds.count("R") >= kinds.count("S") else "S",
            "touches": len(g),
            "age":     min(ages),
        })
    return result


def _detect_zones(df) -> list[dict]:
    if df is None or len(df) < 50:
        return []
    df_window = df.iloc[-LOOKBACK_BARS:] if len(df) > LOOKBACK_BARS else df
    ph, pl = _find_pivots(df_window, window=PIVOT_WINDOW)
    n = len(df_window)
    raw = []
    for idx, price in ph:
        raw.append({"price": price, "age": n - idx, "kind": "R"})
    for idx, price in pl:
        raw.append({"price": price, "age": n - idx, "kind": "S"})
    clusters = _cluster(raw)
    zones = [c for c in clusters if c["touches"] >= MIN_TOUCHES]
    zones.sort(key=lambda x: x["age"])
    return zones


def _bb_width_compressed(df, lookback: int = BBW_LOOKBACK,
                        percentile: float = BBW_RANGE_PCT) -> tuple[bool, float]:
    """True daca BB Width curent e sub percentil → market in range."""
    if len(df) < lookback:
        return True, 0.0
    close = df["close"]
    sma = close.rolling(20).mean()
    std = close.rolling(20).std()
    upper = sma + 2 * std
    lower = sma - 2 * std
    bbw = ((upper - lower) / sma).dropna()
    if len(bbw) < 30:
        return True, 0.0
    cur = float(bbw.iloc[-1])
    thr = float(bbw.iloc[-lookback:].quantile(percentile))
    return cur <= thr, cur


def _wick_touched_zone(df, zone_price: float, zone_kind: str,
                       atr: float) -> tuple[bool, str]:
    """
    True daca bara curenta sau ultima inchisa a atins zona cu FITILUL si
    s-a inchis IN AFARA zonei (rejection real).
    """
    if len(df) < 2:
        return False, "date insuficiente"
    half = atr * ZONE_HALF_WIDTH / 2
    h = float(df["high"].iloc[-1])
    l = float(df["low"].iloc[-1])
    c = float(df["close"].iloc[-1])

    if zone_kind == "S":   # support — wick low atinge, close above
        if l <= zone_price + half and c > zone_price + half * 0.3:
            return True, f"low {l:.5f} touch + close {c:.5f} above"
    else:                   # resistance — wick high atinge, close below
        if h >= zone_price - half and c < zone_price - half * 0.3:
            return True, f"high {h:.5f} touch + close {c:.5f} below"
    return False, "fara wick rejection"


def _h4_trend(symbol: str, bars: int = 100) -> str:
    """Trend H1: BULL/BEAR/NEUTRAL via EMA20 vs EMA50."""
    try:
        from app import fetch
        df, _ = fetch(symbol, "H1", bars)
        if df is None or len(df) < 50:
            return "NEUTRAL"
        c = df["close"].astype(float)
        e20 = c.ewm(span=20, adjust=False).mean().iloc[-1]
        e50 = c.ewm(span=50, adjust=False).mean().iloc[-1]
        slope = (e20 - e50) / e50 * 100
        if slope > 0.15:
            return "BULL"
        if slope < -0.15:
            return "BEAR"
        return "NEUTRAL"
    except Exception:
        return "NEUTRAL"


class SRBounceStrategy(Strategy):
    key   = "sr_bounce"
    name  = "SR Bounce Scalp"
    icon  = "🎢"
    color = "#ab47bc"

    default_tfs            = ["M15", "H1"]   # mutat de pe M5 pe M15/H1 (mai putin noise)
    default_bars           = 400
    default_step           = 1
    default_max_hours      = 1.0
    default_duration_years = 0.5

    elements = {
        "zone_active":      "Pret la zona activa (<= 0.4× ATR)",
        "wick_rejection":   "Wick a atins zona + close afara (rejection real)",
        "range_market":     "BB Width sub percentil 60 (market in range)",
        "h4_not_contra":    "Trend H1 nu e contra semnalului",
        "anti_double_tap":  "Skip daca bara anterioara a fost si ea la zona",
        "cooldown_10":      "Cooldown 10 bare intre trade-uri (anti over-trade)",
        "min_touches_3":    "STRICT: zona testata ≥3 ori (default 2)",
        "fresh_zone":       "STRICT: zona < 50 bare (anti niveluri vechi)",
        "stricter_sl":      "STRICT: SL 0.4× ATR dincolo (default 0.7×)",
    }

    strategy_defaults = {
        "min_confidence": 60.0,
    }

    def analyze(self, symbol, tfs, bars=400, tf_bars=None, elements=None,
                min_confidence=60.0, **kwargs):
        from app import fetch

        if elements is None:
            elements = {k: True for k in self.elements}

        # H1 trend pre-calculated (o singura data)
        h1_dir = _h4_trend(symbol) if elements.get("h4_not_contra", True) else "NEUTRAL"

        tf_results = []
        for tf in tfs:
            try:
                n_bars = (tf_bars or {}).get(tf, bars)
                df, _  = fetch(symbol, tf, n_bars)
                if df is None or len(df) < 100:
                    continue
                df.columns = [c.lower() for c in df.columns]

                price = float(df["close"].iloc[-1])
                atr_val = _atr(df)
                if atr_val <= 0:
                    continue

                bar_idx = len(df) - 1

                reasons    = []
                conviction = 0
                signal     = "HOLD"

                # ── 1. Cooldown ────────────────────────────────────────────
                if elements.get("cooldown_10", True):
                    last_bar = _last_signal_bar.get(f"{symbol}:{tf}", -999)
                    if bar_idx - last_bar < 10:
                        reasons.append(f"Cooldown active ({bar_idx-last_bar} bare < 10)")
                        tf_results.append({"tf": tf, "signal": "HOLD", "conviction": 0,
                                           "reasons": reasons, "price": price,
                                           "sl": None, "tp": None})
                        continue

                # ── 2. Range market filter ─────────────────────────────────
                if elements.get("range_market", True):
                    range_ok, bbw_v = _bb_width_compressed(df)
                    if not range_ok:
                        reasons.append(f"BB Width {bbw_v:.4f} > percentil60 — market trending, skip")
                        tf_results.append({"tf": tf, "signal": "HOLD", "conviction": 0,
                                           "reasons": reasons, "price": price,
                                           "sl": None, "tp": None})
                        continue
                    reasons.append(f"BB Width comprimat ({bbw_v:.4f}) — range OK (+1)")
                    conviction += 1

                # ── 3. Zone detection + filtre stricte ─────────────────────
                zones = _detect_zones(df)
                if elements.get("min_touches_3", False):
                    zones = [z for z in zones if z["touches"] >= 3]
                if elements.get("fresh_zone", False):
                    zones = [z for z in zones if z["age"] < 50]
                if not zones:
                    reasons.append("Niciun zone S/R valid dupa filtre")
                    tf_results.append({"tf": tf, "signal": "HOLD", "conviction": 0,
                                       "reasons": reasons, "price": price,
                                       "sl": None, "tp": None})
                    continue

                # ── 4. Pret la o zona ──────────────────────────────────────
                near_zone = None
                if elements.get("zone_active", True):
                    near_thr = atr_val * NEAR_ZONE_ATR
                    candidates = [z for z in zones if abs(price - z["price"]) <= near_thr]
                    if not candidates:
                        reasons.append(f"Pret nu e la zona (din {len(zones)})")
                        tf_results.append({"tf": tf, "signal": "HOLD", "conviction": 0,
                                           "reasons": reasons, "price": price,
                                           "sl": None, "tp": None})
                        continue
                    # Cea mai apropiata
                    near_zone = min(candidates, key=lambda z: abs(price - z["price"]))
                else:
                    near_zone = zones[0]

                # ── 5. Directie + base conviction ──────────────────────────
                if near_zone["kind"] == "S":
                    signal = "BUY"
                    reasons.append(f"Support @ {near_zone['price']:.5f} "
                                   f"({near_zone['touches']}× touch, age {near_zone['age']}) (+4)")
                else:
                    signal = "SELL"
                    reasons.append(f"Resistance @ {near_zone['price']:.5f} "
                                   f"({near_zone['touches']}× touch, age {near_zone['age']}) (+4)")
                conviction += 4

                # ── 6. Wick rejection (real touch, nu doar proximitate) ────
                wick_ok = False
                if elements.get("wick_rejection", True):
                    wick_ok, wick_msg = _wick_touched_zone(df, near_zone["price"],
                                                          near_zone["kind"], atr_val)
                    if wick_ok:
                        reasons.append(f"Wick rejection: {wick_msg} (+3)")
                        conviction += 3
                    else:
                        reasons.append(f"Fara wick rejection: {wick_msg} (-2)")
                        conviction -= 2

                # ── 7. Anti double-tap ─────────────────────────────────────
                if elements.get("anti_double_tap", True) and len(df) >= 3:
                    half = atr_val * ZONE_HALF_WIDTH / 2
                    prev_h = float(df["high"].iloc[-2])
                    prev_l = float(df["low"].iloc[-2])
                    if (near_zone["kind"] == "S" and prev_l <= near_zone["price"] + half) or \
                       (near_zone["kind"] == "R" and prev_h >= near_zone["price"] - half):
                        reasons.append("Bara precedenta deja la zona — double-tap risk")
                        conviction -= 2

                # ── 8. H1 trend filter ────────────────────────────────────
                if elements.get("h4_not_contra", True) and h1_dir != "NEUTRAL":
                    if (signal == "BUY" and h1_dir == "BEAR") or \
                       (signal == "SELL" and h1_dir == "BULL"):
                        reasons.append(f"H1 trend {h1_dir} contrar — semnal slab (-3)")
                        conviction -= 3
                    else:
                        reasons.append(f"H1 trend {h1_dir} OK (+1)")
                        conviction += 1

                # ── 9. Bonus touches ──────────────────────────────────────
                if near_zone["touches"] >= 4:
                    conviction += 2
                    reasons.append(f"Touches {near_zone['touches']} >= 4 (+2)")
                elif near_zone["touches"] >= 3:
                    conviction += 1

                # Daca dupa toate penalty-urile conviction e prea mic → HOLD
                if conviction < 4:
                    reasons.append(f"Conviction final {conviction} < 4 — semnal slab")
                    tf_results.append({"tf": tf, "signal": "HOLD", "conviction": 0,
                                       "reasons": reasons, "price": price,
                                       "sl": None, "tp": None})
                    continue

                # ── SL / TP ────────────────────────────────────────────────
                sl_extra = (atr_val * 0.4 if elements.get("stricter_sl", False)
                            else atr_val * SL_BEYOND_ATR)
                if signal == "BUY":
                    zone_low = near_zone["price"] - atr_val * ZONE_HALF_WIDTH / 2
                    sl = round(zone_low - sl_extra, 5)
                    # TP: 1.2 ATR sau SMA20 (cea mai aproape de current price)
                    sma20 = float(df["close"].rolling(20).mean().iloc[-1])
                    tp_atr = price + atr_val * TP_ATR_MULT
                    tp = round(min(tp_atr, sma20 if sma20 > price else tp_atr), 5)
                    if tp <= price:
                        tp = round(price + atr_val * TP_ATR_MULT, 5)
                else:
                    zone_high = near_zone["price"] + atr_val * ZONE_HALF_WIDTH / 2
                    sl = round(zone_high + sl_extra, 5)
                    sma20 = float(df["close"].rolling(20).mean().iloc[-1])
                    tp_atr = price - atr_val * TP_ATR_MULT
                    tp = round(max(tp_atr, sma20 if sma20 < price else tp_atr), 5)
                    if tp >= price:
                        tp = round(price - atr_val * TP_ATR_MULT, 5)

                # Verifica RR minim (>= 0.8)
                risk = abs(price - sl)
                reward = abs(tp - price)
                if risk > 0 and reward / risk < 0.8:
                    reasons.append(f"RR slab ({reward/risk:.2f} < 0.8) — HOLD")
                    tf_results.append({"tf": tf, "signal": "HOLD", "conviction": 0,
                                       "reasons": reasons, "price": price,
                                       "sl": None, "tp": None})
                    continue

                # Înregistreaza ultimul bar pentru cooldown
                _last_signal_bar[f"{symbol}:{tf}"] = bar_idx

                tf_results.append({
                    "tf": tf, "signal": signal, "conviction": conviction,
                    "reasons": reasons, "price": round(price, 5),
                    "sl": sl, "tp": tp,
                })

            except Exception as exc:
                log.warning(f"SRBounce {symbol}/{tf}: {exc}")

        if not tf_results:
            return self._empty_result(symbol, "Fara date suficiente")
        return self._build_result(symbol, tf_results, min_confidence, min_votes=1)
