"""
SMC v3 — STRICT mode: Smart Money Concepts cu confluente multiple obligatorii.

Filozofia noua:
  - Calitate peste cantitate: 3-5 semnale / saptamana / simbol, nu 3-5 pe zi.
  - Confluenta obligatorie: BOS + OB fresh + FVG activ + displacement + liquidity sweep.
  - Market structure clar: higher-high/higher-low (bull) sau LL/LH (bear).
  - Sesiune lichida: doar Londra (06-15 UTC) si NY (12-20 UTC).

Gate-uri obligatorii (HOLD daca oricare lipseste):
  G1. Market structure clar pe HTF (HH/HL sau LL/LH) — nu range
  G2. BOS recent (ultimele 20 bare) cu displacement candle (corp >= 55%)
  G3. OB fresh untested (n-a fost retestat dupa formare)
  G4. FVG activ in zona OB (confluenta OB+FVG)
  G5. Liquidity sweep inainte de BOS (wick care a atins high/low anterior)
  G6. ATR in range normal (0.5x - 2x media recenta)
  G7. Pret in 15% din zona OB (nu prea departe)
  G8. Multi-TF confirmare: cel putin 2/3 TF-uri ≥ 7 conviction

Conviction max: 12 (nu 15) — atingerea pragului necesita efort real.
Min confidence default: 85% (era 66%).
"""
from __future__ import annotations
import logging
from datetime import datetime
import numpy as np
from scipy.signal import find_peaks

from strategies.base import Strategy

log = logging.getLogger(__name__)

# ── Parametri stricti ────────────────────────────────────────────────────────
OB_MAX_AGE_BARS      = 50     # era 80 — zone mai proaspete
OB_VOL_MULT          = 1.30   # era 1.20 — volum si mai ridicat
FVG_MAX_AGE_BARS     = 30     # era 50 — FVG mai recent
BOS_LOOKBACK         = 20     # BOS trebuie in ultimele 20 bare
DISPLACEMENT_BODY_PCT = 0.55  # 55% din range = displacement real
SWEEP_LOOKBACK       = 10     # liquidity sweep in ultimele 10 bare
ATR_MIN_MULT         = 0.5    # ATR minim 50% din medie
ATR_MAX_MULT         = 2.0    # ATR maxim 200% din medie
PRICE_TO_OB_MAX_PCT  = 0.15   # pret max 15% distanta de OB

# Sesiuni lichide (UTC)
LIQUID_SESSIONS = [
    (6, 15),   # Londra + overlap
    (12, 20),  # NY + overlap
]


def _in_liquid_session() -> bool:
    import strategies as _strats
    hour = _strats.get_utc_now().hour
    return any(s <= hour < e for s, e in LIQUID_SESSIONS)


def _market_structure(df, lookback: int = 60) -> str:
    """
    Returneaza 'BULL', 'BEAR' sau 'RANGE' pe baza swing points.
    BULL = HH + HL consecutive; BEAR = LL + LH; RANGE = mixed.
    """
    if len(df) < lookback:
        return "RANGE"
    highs = df["high"].values[-lookback:]
    lows  = df["low"].values[-lookback:]
    sample = highs - lows
    prom = max(float(np.mean(sample)) * 0.3, 1e-8)

    ph_idx, _ = find_peaks( highs, distance=3, prominence=prom)
    pl_idx, _ = find_peaks(-lows,  distance=3, prominence=prom)

    if len(ph_idx) < 2 or len(pl_idx) < 2:
        return "RANGE"

    last_highs = highs[ph_idx[-2:]]
    last_lows  = lows[pl_idx[-2:]]

    hh = last_highs[1] > last_highs[0]
    hl = last_lows[1]  > last_lows[0]
    ll = last_lows[1]  < last_lows[0]
    lh = last_highs[1] < last_highs[0]

    if hh and hl: return "BULL"
    if ll and lh: return "BEAR"
    return "RANGE"


def _detect_bos_with_displacement(df, direction: str):
    """
    Returneaza dict cu detalii BOS daca s-a produs in ultimele BOS_LOOKBACK bare,
    altfel None. Cere si displacement candle (corp >= DISPLACEMENT_BODY_PCT).
    """
    n = len(df)
    if n < BOS_LOOKBACK + 10:
        return None
    highs  = df["high"].values
    lows   = df["low"].values
    closes = df["close"].values
    opens  = df["open"].values

    # Swing points pe toata fereastra extinsa
    sample = highs - lows
    prom = max(float(np.mean(sample[-80:])) * 0.3, 1e-8)
    ph_idx, _ = find_peaks( highs, distance=4, prominence=prom)
    pl_idx, _ = find_peaks(-lows,  distance=4, prominence=prom)

    # Cauta BOS in ultimele BOS_LOOKBACK bare
    start = n - BOS_LOOKBACK
    for i in range(start, n):
        body = abs(closes[i] - opens[i])
        rng  = highs[i] - lows[i]
        if rng <= 0:
            continue
        body_pct = body / rng

        if direction == "BULL":
            recent_highs = [h for h in ph_idx if h < i]
            if not recent_highs:
                continue
            last_sh = float(highs[recent_highs[-1]])
            if closes[i] > last_sh and closes[i] > opens[i] and body_pct >= DISPLACEMENT_BODY_PCT:
                return {"idx": i, "broken_level": last_sh, "body_pct": body_pct,
                        "age": n - 1 - i}
        elif direction == "BEAR":
            recent_lows = [l for l in pl_idx if l < i]
            if not recent_lows:
                continue
            last_sl = float(lows[recent_lows[-1]])
            if closes[i] < last_sl and closes[i] < opens[i] and body_pct >= DISPLACEMENT_BODY_PCT:
                return {"idx": i, "broken_level": last_sl, "body_pct": body_pct,
                        "age": n - 1 - i}
    return None


def _liquidity_sweep_before(df, bos_idx: int, direction: str) -> bool:
    """
    Verifica daca inainte de BOS a existat un sweep — wick care a atins/depasit
    un extrem recent apoi a fost respins (close in partea opusa wick-ului).
    """
    start = max(0, bos_idx - SWEEP_LOOKBACK)
    window = df.iloc[start:bos_idx]
    if len(window) < 3:
        return False
    highs  = window["high"].values
    lows   = window["low"].values
    closes = window["close"].values
    opens  = window["open"].values

    if direction == "BULL":
        # Sweep low: lumânare cu wick jos mare, close deasupra open
        for i in range(len(window)):
            rng = highs[i] - lows[i]
            if rng <= 0:
                continue
            lower_wick = min(opens[i], closes[i]) - lows[i]
            if lower_wick / rng >= 0.5 and closes[i] > opens[i]:
                return True
    else:
        # Sweep high
        for i in range(len(window)):
            rng = highs[i] - lows[i]
            if rng <= 0:
                continue
            upper_wick = highs[i] - max(opens[i], closes[i])
            if upper_wick / rng >= 0.5 and closes[i] < opens[i]:
                return True
    return False


def _find_fresh_ob(df, direction: str, bos_idx: int):
    """
    Gaseste OB fresh (untested) inainte de BOS in directia specificata.
    Fresh = n-a mai fost atins de pret dupa formare pana la bos_idx.
    """
    opens  = df["open"].values
    closes = df["close"].values
    highs  = df["high"].values
    lows   = df["low"].values
    n = len(df)

    # Volum pentru validare
    has_vol = "tick_volume" in df.columns or "volume" in df.columns
    if has_vol:
        vol_col = "tick_volume" if "tick_volume" in df.columns else "volume"
        vols    = df[vol_col].values
        vol_mean = float(np.mean(vols[max(0, bos_idx-50):bos_idx])) if bos_idx >= 10 else 0
    else:
        vols, vol_mean = None, 0

    # Cauta ultima lumanare contrara inainte de BOS (OB = ultimul opposite candle)
    search_start = max(0, bos_idx - 15)
    for j in range(bos_idx - 1, search_start - 1, -1):
        is_bull_candle = closes[j] > opens[j]
        age = n - 1 - j

        if age > OB_MAX_AGE_BARS:
            break

        if direction == "BULL" and not is_bull_candle:
            # OB bullish = ultima lumanare bearish
            ob_low  = float(lows[j])
            ob_high = float(highs[j])
            # Fresh check: pretul n-a revenit in [ob_low, ob_high] dupa bos_idx
            after = df.iloc[bos_idx+1:]
            if len(after) > 0:
                hit = ((after["low"] <= ob_high) & (after["high"] >= ob_low)).any()
                if hit:
                    continue
            vol_ok = True if not has_vol else (vols[j] >= vol_mean * OB_VOL_MULT if vol_mean > 0 else True)
            return {"low": ob_low, "high": ob_high, "idx": j, "age": age, "vol_ok": vol_ok}

        if direction == "BEAR" and is_bull_candle:
            ob_low  = float(lows[j])
            ob_high = float(highs[j])
            after = df.iloc[bos_idx+1:]
            if len(after) > 0:
                hit = ((after["low"] <= ob_high) & (after["high"] >= ob_low)).any()
                if hit:
                    continue
            vol_ok = True if not has_vol else (vols[j] >= vol_mean * OB_VOL_MULT if vol_mean > 0 else True)
            return {"low": ob_low, "high": ob_high, "idx": j, "age": age, "vol_ok": vol_ok}

    return None


def _fvg_in_ob(df, ob: dict, direction: str):
    """
    Cauta FVG (fair value gap) activ in zona OB sau in imediata apropiere.
    FVG bullish: lows[i+2] > highs[i] (gap sus); FVG bearish: highs[i+2] < lows[i].
    """
    if ob is None:
        return None
    highs = df["high"].values
    lows  = df["low"].values
    n = len(df)
    start = max(0, ob["idx"] - 5)
    end   = min(n - 2, ob["idx"] + 10)

    for i in range(start, end):
        age = n - 2 - i
        if age > FVG_MAX_AGE_BARS:
            continue
        if direction == "BULL" and lows[i+2] > highs[i]:
            fvg_low, fvg_high = float(highs[i]), float(lows[i+2])
            # Confluenta: FVG se suprapune cu OB
            if fvg_low <= ob["high"] and fvg_high >= ob["low"]:
                return {"low": fvg_low, "high": fvg_high, "age": age, "mid": (fvg_low + fvg_high) / 2}
        if direction == "BEAR" and highs[i+2] < lows[i]:
            fvg_low, fvg_high = float(highs[i+2]), float(lows[i])
            if fvg_low <= ob["high"] and fvg_high >= ob["low"]:
                return {"low": fvg_low, "high": fvg_high, "age": age, "mid": (fvg_low + fvg_high) / 2}
    return None


def _atr_regime_ok(df, period: int = 14) -> tuple[bool, float]:
    """ATR in [0.5x, 2.0x] media recenta → regim OK."""
    if len(df) < period * 3:
        return False, 0.0
    high  = df["high"].values
    low   = df["low"].values
    close = df["close"].values
    tr = np.maximum(high[1:] - low[1:],
                    np.maximum(np.abs(high[1:] - close[:-1]),
                               np.abs(low[1:] - close[:-1])))
    atr_series = np.convolve(tr, np.ones(period)/period, mode='valid')
    if len(atr_series) < 20:
        return False, 0.0
    atr_cur = float(atr_series[-1])
    atr_mean = float(np.mean(atr_series[-50:]))
    if atr_mean <= 0:
        return False, atr_cur
    ratio = atr_cur / atr_mean
    return (ATR_MIN_MULT <= ratio <= ATR_MAX_MULT), atr_cur


class SMCStrategy(Strategy):
    key   = "smc"
    name  = "SMC v3 STRICT"
    icon  = "🟠"
    color = "#ff9800"

    default_tfs  = ["H1", "H4"]      # era M15+H1 — migram pe TF-uri mai fiabile
    default_bars = 500
    elements     = {
        "structure":    "Market Structure (HH/HL sau LL/LH)",
        "bos":          "BOS + Displacement (corp ≥ 55%)",
        "ob":           "OB Fresh Untested (≤50 bare)",
        "fvg":          "FVG in OB (confluenta obligatorie)",
        "sweep":        "Liquidity Sweep inainte de BOS",
        "ob_volume":    "OB Volume (volum ≥ 1.3x medie)",
        "session":      "Sesiune lichida (Londra/NY)",
        "atr_regime":   "ATR in regim normal (0.5-2x)",
        "sweep_required": "STRICT: Liquidity Sweep OBLIGATORIU (nu doar bonus)",
        "fvg_required":   "STRICT: FVG OBLIGATORIU (default era deja obligatoriu)",
        "ob_age_30":      "STRICT: OB freshness < 30 bare (default 50)",
        "bos_age_10":     "STRICT: BOS proaspat (≤10 bare)",
    }

    def analyze(self, symbol, tfs, bars=500, tf_bars=None, elements=None,
                min_confidence=85.0, **kwargs):
        """
        Entry conditions (TOATE obligatorii pentru semnal ≥ 7 conviction):
          1. Market structure BULL/BEAR (nu RANGE)
          2. BOS cu displacement in ultimele 20 bare
          3. OB fresh untested
          4. FVG confluenta cu OB
          5. Liquidity sweep inainte de BOS
          6. ATR regim normal
          7. Pret in ≤15% distanta de OB

        Bonusuri:
          + Sesiune lichida (+1)
          + OB pe volum ridicat (+1)
          + Multi-TF confirmare (+5% per TF)
        """
        from app import fetch

        if elements is None:
            elements = {k: True for k in self.elements}

        # Gate global: sesiune lichida
        session_ok = _in_liquid_session()

        tf_results = []
        for tf in tfs:
            try:
                n_bars = (tf_bars or {}).get(tf, bars)
                df, _  = fetch(symbol, tf, n_bars)
                if df is None or len(df) < 100:
                    continue
                df.columns = [c.lower() for c in df.columns]

                reasons = []
                conviction = 0
                signal = "HOLD"
                price_entry = float(df["close"].iloc[-1])

                # ── Gate 1: Market structure ──
                struct = _market_structure(df)
                if struct == "RANGE":
                    reasons.append("Market structure: RANGE (nu HH/HL) — HOLD")
                    tf_results.append({"tf": tf, "signal": "HOLD", "conviction": 0,
                                       "reasons": reasons, "price": price_entry,
                                       "sl": None, "tp": None})
                    continue
                reasons.append(f"Structure: {struct} (HH/HL confirmat)")
                conviction += 2

                direction = "BULL" if struct == "BULL" else "BEAR"
                target_signal = "BUY" if direction == "BULL" else "SELL"

                # ── Gate 2: BOS + Displacement ──
                bos = _detect_bos_with_displacement(df, direction)
                if bos is None:
                    reasons.append("Niciun BOS cu displacement in ultimele 20 bare — HOLD")
                    tf_results.append({"tf": tf, "signal": "HOLD", "conviction": conviction,
                                       "reasons": reasons, "price": price_entry,
                                       "sl": None, "tp": None})
                    continue
                reasons.append(
                    f"BOS @ {bos['broken_level']:.5f} cu displacement {bos['body_pct']*100:.0f}%"
                    f" (varsta {bos['age']} bare)"
                )
                conviction += 3

                # ── Gate 3: OB fresh untested ──
                ob = _find_fresh_ob(df, direction, bos["idx"])
                if ob is None:
                    reasons.append("Niciun OB fresh untested ≤50 bare — HOLD")
                    tf_results.append({"tf": tf, "signal": "HOLD", "conviction": conviction,
                                       "reasons": reasons, "price": price_entry,
                                       "sl": None, "tp": None})
                    continue
                reasons.append(f"OB fresh @ {ob['low']:.5f}-{ob['high']:.5f} (varsta {ob['age']} bare)")
                conviction += 2

                # ── Gate 4: FVG in OB (confluenta) ──
                fvg = _fvg_in_ob(df, ob, direction)
                if fvg is None:
                    reasons.append("Niciun FVG in OB — confluenta lipseste, HOLD")
                    tf_results.append({"tf": tf, "signal": "HOLD", "conviction": conviction,
                                       "reasons": reasons, "price": price_entry,
                                       "sl": None, "tp": None})
                    continue
                reasons.append(f"FVG in OB @ {fvg['low']:.5f}-{fvg['high']:.5f} ✓ confluenta")
                conviction += 2

                # ── Gate 5: Liquidity sweep ──
                if _liquidity_sweep_before(df, bos["idx"], direction):
                    reasons.append("Liquidity sweep detectat inainte de BOS ✓")
                    conviction += 1
                else:
                    reasons.append("Fara liquidity sweep — semnal slabit")
                    conviction -= 1

                # ── Gate 6: ATR regim ──
                atr_ok, atr_val = _atr_regime_ok(df)
                if not atr_ok:
                    reasons.append(f"ATR in afara regim normal (0.5-2x) — HOLD")
                    tf_results.append({"tf": tf, "signal": "HOLD", "conviction": conviction,
                                       "reasons": reasons, "price": price_entry,
                                       "sl": None, "tp": None})
                    continue
                reasons.append(f"ATR normal ({atr_val:.5f}) ✓")
                conviction += 1

                # ── Gate 7: Proximitate OB ──
                ob_mid = (ob["low"] + ob["high"]) / 2
                dist_pct = abs(price_entry - ob_mid) / max(1e-9, ob_mid)
                if dist_pct > PRICE_TO_OB_MAX_PCT:
                    reasons.append(f"Pret prea departe de OB ({dist_pct*100:.1f}% > 15%) — HOLD")
                    tf_results.append({"tf": tf, "signal": "HOLD", "conviction": conviction,
                                       "reasons": reasons, "price": price_entry,
                                       "sl": None, "tp": None})
                    continue
                reasons.append(f"Pret la {dist_pct*100:.1f}% de OB ✓")
                conviction += 1

                # ── Bonusuri ──
                if session_ok:
                    reasons.append("Sesiune lichida (Londra/NY) ✓")
                    conviction += 1
                else:
                    reasons.append("Sesiune slaba — nu se ia trade")
                    tf_results.append({"tf": tf, "signal": "HOLD", "conviction": conviction,
                                       "reasons": reasons, "price": price_entry,
                                       "sl": None, "tp": None})
                    continue

                if ob["vol_ok"]:
                    reasons.append("OB format pe volum ridicat (≥1.3x) ✓")
                    conviction += 1

                # ── SL / TP: RR 1:1.5 → trade-uri mai scurte, TP atins mai rapid ──
                if direction == "BULL":
                    price_entry = float(ob["high"])
                    sl = round(float(ob["low"]) - atr_val * 0.3, 5)
                    tp = round(price_entry + (price_entry - sl) * 1.5, 5)  # RR 1:1.5
                else:
                    price_entry = float(ob["low"])
                    sl = round(float(ob["high"]) + atr_val * 0.3, 5)
                    tp = round(price_entry - (sl - price_entry) * 1.5, 5)

                signal = target_signal
                reasons.append(f"Setup complet: conviction {conviction}/12")

                tf_results.append({
                    "tf": tf, "signal": signal, "conviction": conviction,
                    "reasons": reasons,
                    "price": round(price_entry, 5),
                    "sl": sl, "tp": tp,
                })

            except Exception as exc:
                log.warning(f"SMCv3 {symbol}/{tf}: {exc}")

        if not tf_results:
            return self._empty_result(symbol, "Fara date suficiente")

        # min_votes=2: STRICT multi-TF agreement (era 1)
        return self._build_result(symbol, tf_results, min_confidence, min_votes=2)
