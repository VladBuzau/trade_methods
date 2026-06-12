"""
MTF S/R — Multi-Timeframe Support & Resistance
───────────────────────────────────────────────────────────────────────────
Logica:
  1. Detecteaza pivoti (highs si lows) pe M15, H1, H4, D1
  2. Clustereaza nivelurile care se suprapun (toleranta ~0.15% din pret)
  3. Zona VALIDA = confirmata pe cel putin 2 TF-uri simultan
  4. Zona PUTERNICA = 3+ TF-uri (opacitate mai mare pe grafic)
  5. Semnal pending: pret se apropie de zona (< 3x ATR distanta)
     → Resistance => SELL LIMIT
     → Support    => BUY  LIMIT

Chart overlay:
  - Toate zonele MTF valide colorate pe grafic (intensitate ~ tf_count)
  - Zona de approach (pretul aproape) highlight mai puternic + eticheta entry
"""
from __future__ import annotations
import numpy as np
from ..base import Strategy

# ── Configurare ───────────────────────────────────────────────────────────────
_SR_TFS    = ["M15", "H1", "H4", "D1"]      # TF-uri scanate
_LOOKBACK  = {"M15": 300, "H1": 250, "H4": 150, "D1": 100}
_PIVOT_WIN = {"M15": 5,   "H1": 5,   "H4": 4,   "D1": 3}

_TOL_PCT        = 0.0015   # 0.15% — toleranta clustering (un nivel valid intr-o banda)
_ZONE_ATR_MULT  = 0.6      # latimea zonei = 0.6 * ATR
_APPROACH_ATR   = 3.0      # distanta maxima pana la zona pentru semnal pending
_MIN_TOUCHES    = 2        # minim 2 pivoti pe TF ca nivelul sa conteze
_MIN_TF_COUNT   = 2        # minim 2 TF-uri ca zona sa fie valida


# ── Helpers ──────────────────────────────────────────────────────────────────
def _atr(df, period: int = 14) -> float:
    hi = df["high"].values
    lo = df["low"].values
    cl = df["close"].values
    tr = np.maximum(hi[1:] - lo[1:],
         np.maximum(np.abs(hi[1:] - cl[:-1]),
                    np.abs(lo[1:] - cl[:-1])))
    if len(tr) < period:
        return float(np.mean(tr)) if len(tr) else 1e-5
    return float(np.mean(tr[-period:]))


def _find_pivots(df, window: int = 5) -> tuple[list, list]:
    """Returneaza pivot highs si pivot lows ca liste de (bar_idx, price)."""
    hi = df["high"].values
    lo = df["low"].values
    n  = len(hi)
    ph, pl = [], []
    for i in range(window, n - window):
        # Strict pivot high — cel mai mare in fereastra
        if hi[i] == max(hi[i - window: i + window + 1]):
            ph.append((i, float(hi[i])))
        # Strict pivot low
        if lo[i] == min(lo[i - window: i + window + 1]):
            pl.append((i, float(lo[i])))
    return ph, pl


def _cluster_levels(raw: list[dict], tol_pct: float = _TOL_PCT) -> list[dict]:
    """
    Grupeaza niveluri din liste de {"price", "tf", "age", "kind"}.
    Returneaza clustere: {"price", "zone_low", "zone_high", "tfs", "tf_count",
                          "touches", "age", "kind"}.
    """
    if not raw:
        return []
    raw_s = sorted(raw, key=lambda x: x["price"])
    groups: list[list[dict]] = []
    cur = [raw_s[0]]
    for item in raw_s[1:]:
        ref = np.mean([x["price"] for x in cur])
        if abs(item["price"] - ref) / ref <= tol_pct:
            cur.append(item)
        else:
            groups.append(cur)
            cur = [item]
    groups.append(cur)

    result = []
    for g in groups:
        prices = [x["price"] for x in g]
        tfs    = list({x["tf"] for x in g})
        ages   = [x["age"] for x in g]
        kinds  = [x["kind"] for x in g]
        mean_p = float(np.mean(prices))
        kind   = "R" if kinds.count("R") >= kinds.count("S") else "S"
        result.append({
            "price":    mean_p,
            "tfs":      tfs,
            "tf_count": len(tfs),
            "touches":  len(g),
            "age":      min(ages),
            "kind":     kind,
        })
    return result


# ── Zone detection (single TF — pentru overlay simplu) ────────────────────────
def detect_sr_zones(df, lookback: int = 200) -> list[dict]:
    """
    Zone S/R pe dataframe-ul curent (un singur TF).
    Folosit ca fn_zones fallback sau pentru preview rapid.
    Returneaza zone cu minim 2 touch-uri, in formatul standard AutoOrders.
    """
    df = df.iloc[-lookback:] if len(df) > lookback else df
    atr_val = _atr(df)
    half_w  = atr_val * _ZONE_ATR_MULT / 2
    ph, pl  = _find_pivots(df, window=5)
    n = len(df)
    raw = []
    for idx, price in ph:
        raw.append({"price": price, "tf": "CRT", "age": n - idx, "kind": "R"})
    for idx, price in pl:
        raw.append({"price": price, "tf": "CRT", "age": n - idx, "kind": "S"})

    clusters = _cluster_levels(raw, tol_pct=_TOL_PCT)
    zones = []
    for c in clusters:
        if c["touches"] < _MIN_TOUCHES:
            continue
        zones.append({
            "type":      "BEARISH" if c["kind"] == "R" else "BULLISH",
            "zone_low":  round(c["price"] - half_w, 5),
            "zone_high": round(c["price"] + half_w, 5),
            "price":     round(c["price"], 5),
            "touches":   c["touches"],
            "age":       c["age"],
            "tf_count":  1,
            "tfs":       c["tfs"],
        })
    return sorted(zones, key=lambda x: x["age"])


# ── Zone detection MTF — principala ──────────────────────────────────────────
def detect_sr_mtf_zones(symbol: str, fetch_fn, df_current, price_now: float) -> dict:
    """
    Detecteaza zone S/R confirmate pe multiple TF-uri.
    Returneaza {"zones": [...toate zonele MTF valide...], "approach": [...zone aproape de pret...]}.
    Format zone: {type, zone_low, zone_high, price, tf_count, tfs, touches, age, order_type, pending_entry}
    """
    atr_val = _atr(df_current)
    if atr_val <= 0:
        return {"zones": [], "approach": []}

    raw: list[dict] = []
    for tf in _SR_TFS:
        bars = _LOOKBACK.get(tf, 150)
        win  = _PIVOT_WIN.get(tf, 5)
        try:
            df, _ = fetch_fn(symbol, tf, bars)
            if df is None or df.empty or len(df) < win * 3:
                continue
            ph, pl = _find_pivots(df, window=win)
            n = len(df)
            for idx, price in ph:
                raw.append({"price": price, "tf": tf, "age": n - idx, "kind": "R"})
            for idx, price in pl:
                raw.append({"price": price, "tf": tf, "age": n - idx, "kind": "S"})
        except Exception:
            continue

    if not raw:
        return {"zones": [], "approach": []}

    clusters = _cluster_levels(raw, tol_pct=_TOL_PCT)
    valid    = [c for c in clusters if c["tf_count"] >= _MIN_TF_COUNT]

    half_w       = atr_val * _ZONE_ATR_MULT / 2
    approach_thr = atr_val * _APPROACH_ATR

    all_zones: list[dict]      = []
    approach_zones: list[dict] = []

    for c in valid:
        is_res = c["price"] > price_now
        entry  = (round(c["price"] - half_w * 0.15, 5) if is_res
                  else round(c["price"] + half_w * 0.15, 5))
        zone = {
            "type":          "BEARISH" if is_res else "BULLISH",
            "zone_low":      round(c["price"] - half_w, 5),
            "zone_high":     round(c["price"] + half_w, 5),
            "price":         round(c["price"], 5),
            "tf_count":      c["tf_count"],
            "tfs":           sorted(c["tfs"], key=lambda t: _SR_TFS.index(t) if t in _SR_TFS else 99),
            "touches":       c["touches"],
            "age":           c["age"],
            "order_type":    "SELL_LIMIT" if is_res else "BUY_LIMIT",
            "pending_entry": entry,
        }
        all_zones.append(zone)
        if abs(c["price"] - price_now) <= approach_thr:
            approach_zones.append(zone)

    all_zones.sort(key=lambda x: abs(x["price"] - price_now))
    approach_zones.sort(key=lambda x: abs(x["price"] - price_now))
    return {"zones": all_zones[:30], "approach": approach_zones[:8]}


# ── Strategy class ────────────────────────────────────────────────────────────
class SRMTFStrategy(Strategy):
    key   = "sr_mtf"
    name  = "S/R Multi-Timeframe"
    icon  = "📐"
    color = "#7e57c2"

    default_tfs  = ["H1"]
    default_bars = 300
    elements     = {
        "min_tf_3":         "STRICT: zona pe ≥3 TF-uri (default ≥2)",
        "min_touches_3":    "STRICT: ≥3 atingeri (default ≥2)",
        "fresh_zones_only": "Doar zone proaspete (age < 50 bare)",
        "strict_distance":  "STRICT: pret la <1.5x ATR de zona (default 3x)",
        "min_rr_2":         "Cere RR >= 2:1 (default 1:2)",
    }

    def analyze(self, symbol, tfs, bars=300, tf_bars=None, elements=None,
                min_confidence=40.0, **kwargs):
        from app import fetch

        if elements is None:
            elements = {k: False for k in self.elements}  # STRICT opts default OFF

        tf     = tfs[0] if tfs else "H1"
        n_bars = (tf_bars or {}).get(tf, bars)

        try:
            df, _ = fetch(symbol, tf, n_bars)
        except Exception as e:
            return self._result_hold(symbol, f"fetch error: {e}")

        if df is None or df.empty:
            return self._result_hold(symbol, "Nu s-au putut incarca datele")

        price_now = float(df["close"].iloc[-1])
        atr_val   = _atr(df)
        if atr_val <= 0:
            return self._result_hold(symbol, "ATR zero — date insuficiente")

        mtf = detect_sr_mtf_zones(symbol, fetch, df, price_now)
        approach = mtf.get("approach", [])

        # ── Filtre STRICT opt-in ──
        if elements.get("strict_distance", False):
            approach = [a for a in approach
                        if abs(a["price"] - price_now) / atr_val <= 1.5]
        if elements.get("fresh_zones_only", False):
            approach = [a for a in approach if a["age"] < 50]
        if elements.get("min_tf_3", False):
            approach = [a for a in approach if a["tf_count"] >= 3]
        if elements.get("min_touches_3", False):
            approach = [a for a in approach if a["touches"] >= 3]

        if not approach:
            n_total = len(mtf.get("zones", []))
            return self._result_hold(
                symbol,
                f"Nicio zona MTF dupa filtre stricte. Total: {n_total}",
            )

        # Alege zona cea mai puternica (tf_count desc, distanta asc)
        best = max(approach, key=lambda x: (x["tf_count"], -abs(x["price"] - price_now)))

        # Confidence granular (nu mai satureaza la 95):
        #   baza 25 + tf_count*10 (max 40) + touches*2 (max 10) + fresh_bonus (max 10) - dist_penalty
        dist_atr_raw = abs(best["price"] - price_now) / atr_val if atr_val > 0 else 0
        fresh_bonus  = max(0, 10 - best["age"] * 0.15)          # 10 pt zona noua, 0 dupa ~66 bare
        dist_penalty = min(20, dist_atr_raw * 8)                # -8 per ATR distanta, max -20
        conf = 25.0 + min(best["tf_count"], 4) * 10 + min(best["touches"], 5) * 2 + fresh_bonus - dist_penalty
        conf = max(0.0, min(conf, 95.0))
        if conf < min_confidence:
            return self._result_hold(
                symbol,
                f"Confidence {conf:.0f}% < prag {min_confidence:.0f}%"
                f"  (zona {best['order_type']} pe {'+'.join(best['tfs'])})",
            )

        signal = "SELL" if best["type"] == "BEARISH" else "BUY"
        entry  = best["pending_entry"]
        half_w = atr_val * _ZONE_ATR_MULT / 2

        # RR target: default 2:1, sau 3:1 cu min_rr_2 (mai conservator)
        rr_target = 3.0 if elements.get("min_rr_2", False) else 2.0
        if signal == "SELL":
            sl = round(best["zone_high"] + atr_val * 0.5, 5)
            tp = round(entry - (sl - entry) * rr_target, 5)
        else:
            sl = round(best["zone_low"] - atr_val * 0.5, 5)
            tp = round(entry + (entry - sl) * rr_target, 5)

        tfs_str  = "+".join(best["tfs"])
        dist_atr = round(abs(best["price"] - price_now) / atr_val, 2)

        return {
            "symbol":        symbol,
            "strategy":      self.key,
            "timestamp":     self._now(),
            "signal":        "HOLD",           # nu intra market — doar pending
            "confidence":    round(conf, 1),
            "risk_score":    round(100 - conf, 1),
            "justification": [
                f"Zona S/R MTF: {best['order_type']} confirmata pe {best['tf_count']} TF-uri ({tfs_str})",
                f"Touches: {best['touches']}  |  Distanta: {dist_atr}x ATR  |  Varsta: {best['age']} bare",
                f"Entry: {entry}  SL: {sl}  TP: {tp}",
                f"Confidence: {conf:.0f}%",
            ],
            "pending_entry": {
                "signal":     signal,
                "order_type": best["order_type"],
                "price":      entry,
                "sl":         sl,
                "tp":         tp,
                "tf":         tf,
                "tf_count":   best["tf_count"],
                "tfs":        best["tfs"],
                "touches":    best["touches"],
            },
            "best_tf":       None,
            "tfs":           [],
            "auto_executed": False,
        }

    def _result_hold(self, symbol: str, reason: str) -> dict:
        return {
            "symbol":        symbol,
            "strategy":      self.key,
            "timestamp":     self._now(),
            "signal":        "HOLD",
            "confidence":    0.0,
            "risk_score":    0.0,
            "justification": [reason],
            "pending_entry": None,
            "best_tf":       None,
            "tfs":           [],
            "auto_executed": False,
        }
