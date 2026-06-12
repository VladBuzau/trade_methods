"""
VWAP Mean Reversion — bot intraday short-term bazat pe research academic.

Bazat pe dovezi:
  - Mean reversion bate momentum intraday (NYSE studies, reversal-uri 3.29%/zi)
  - Win rate documentat: ranging 65-75%, squeeze 70-80%
  - CRITICAL: edge functioneaza DOAR in volatilitate ridicata (filtru regim)
  - VWAP = "fair price" intraday, RSI/BB = extreme, ADX = regim, volume = confirmare

Setup:
  LONG:  pret <= lower BB(-2sigma) SAU sub VWAP-2sigma
         + RSI < 30 (oversold)
         + ADX < 25 (range, nu trend)  [regime filter]
         + ATR curent > media ATR 50  [volatilitate suficienta]
         + volume > 1.2x medie (confirmare optionala)
  SHORT: mirror

SL: sub swing low / sub -1sigma VWAP band
TP: la VWAP (mijloc range) sau RR 1.5:1 - 3:1
Sesiune: skip 22-01 UTC (dead), bonus NY session

Best on: EURUSD NY session, GBPUSD, USDJPY, XAUUSD.
"""
from __future__ import annotations
import logging
import numpy as np

from strategies.base import Strategy

log = logging.getLogger(__name__)


# ── Indicatori ──────────────────────────────────────────────────────────────
def _vwap_session(df) -> tuple[float, float, float]:
    """
    VWAP cu benzi de deviatie. Calculat pe ultimele ~bare ca proxy de sesiune.
    Returneaza (vwap, vwap_upper_2sigma, vwap_lower_2sigma).
    """
    # Typical price
    tp = (df["high"] + df["low"] + df["close"]) / 3.0
    vol = df["volume"].replace(0, 1)  # evita div by zero
    # VWAP pe ultimele N bare (proxy sesiune intraday)
    N = min(len(df), 96)  # ~o sesiune pe M5/M15
    tp_n = tp.iloc[-N:]
    vol_n = vol.iloc[-N:]
    cum_vol = vol_n.sum()
    if cum_vol <= 0:
        v = float(df["close"].iloc[-1])
        return v, v, v
    vwap = float((tp_n * vol_n).sum() / cum_vol)
    # Deviatia ponderata pe volum
    var = float(((tp_n - vwap) ** 2 * vol_n).sum() / cum_vol)
    std = var ** 0.5
    return vwap, vwap + 2 * std, vwap - 2 * std


def _rsi(close, period: int = 14) -> float:
    delta = close.diff()
    gain = delta.clip(lower=0).ewm(alpha=1/period, adjust=False).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1/period, adjust=False).mean()
    rs = gain / loss.replace(0, np.nan)
    rsi = 100 - 100 / (1 + rs)
    return float(rsi.iloc[-1]) if not np.isnan(rsi.iloc[-1]) else 50.0


def _atr(df, period: int = 14) -> tuple[float, float]:
    """Returneaza (atr_curent, atr_medie_50)."""
    h, l, c = df["high"].values, df["low"].values, df["close"].values
    if len(c) < period + 1:
        return 0.0, 0.0
    tr = np.maximum(h[1:] - l[1:],
                    np.maximum(np.abs(h[1:] - c[:-1]),
                               np.abs(l[1:] - c[:-1])))
    atr_series = np.convolve(tr, np.ones(period)/period, mode="valid")
    if len(atr_series) == 0:
        return 0.0, 0.0
    cur = float(atr_series[-1])
    avg = float(np.mean(atr_series[-50:])) if len(atr_series) >= 50 else float(np.mean(atr_series))
    return cur, avg


def _adx(df, period: int = 14) -> float:
    h = df["high"].values
    l = df["low"].values
    c = df["close"].values
    n = len(c)
    if n < period * 2 + 5:
        return 0.0
    tr = np.maximum(h[1:] - l[1:],
                    np.maximum(np.abs(h[1:] - c[:-1]),
                               np.abs(l[1:] - c[:-1])))
    plus_dm = np.where((h[1:] - h[:-1]) > (l[:-1] - l[1:]),
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
    plus_d = 100 * rma(plus_dm, period) / den
    minus_d = 100 * rma(minus_dm, period) / den
    sum_d = plus_d + minus_d
    dx = 100 * np.abs(plus_d - minus_d) / np.where(sum_d == 0, 1, sum_d)
    adx = rma(dx, period)
    return float(adx[-1]) if len(adx) > 0 else 0.0


def _bollinger(close, period: int = 20, k: float = 2.0):
    sma = close.rolling(period).mean()
    std = close.rolling(period).std()
    upper = sma + k * std
    lower = sma - k * std
    return float(upper.iloc[-1]), float(sma.iloc[-1]), float(lower.iloc[-1])


def _find_pivots(df, window: int = 5):
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


class VWAPReversionStrategy(Strategy):
    key   = "vwap_reversion"
    name  = "VWAP Mean Reversion"
    icon  = "📊"
    color = "#26a69a"

    default_tfs            = ["M5", "M15"]
    default_bars           = 300
    default_step           = 1
    default_max_hours      = 4.0    # short-term, inchide in 4h
    default_duration_years = 1.0

    elements = {
        "bb_extreme":      "Pret la banda BB extrema (±2σ)",
        "vwap_extreme":    "Pret dincolo de VWAP ±2σ band",
        "rsi_confirm":     "RSI < 30 (BUY) / > 70 (SELL)",
        "regime_adx":      "REGIM: ADX < 25 (range, nu trend) [research]",
        "regime_atr":      "REGIM: ATR curent > medie (volatilitate suficienta) [research]",
        "volume_confirm":  "Volume > 1.2x medie (confirmare institutionala)",
        "session_filter":  "Skip sesiuni moarte (22-01 UTC)",
        "tp_at_vwap":      "TP la VWAP (mean reversion target)",
        "require_both_bb_rsi": "STRICT: BB extreme + RSI obligatoriu (nu doar unul)",
        "atr_regime_strict":   "STRICT: ATR > 1.2x medie (volatilitate ridicata only)",
    }

    strategy_defaults = {
        "min_confidence": 60.0,
    }

    def analyze(self, symbol, tfs, bars=300, tf_bars=None, elements=None,
                min_confidence=60.0, **kwargs):
        from app import fetch, calc_sl_tp, find_pivots
        from datetime import datetime, timezone

        if elements is None:
            elements = {k: not lbl.upper().startswith("STRICT")
                        for k, lbl in self.elements.items()}

        tf_results = []
        for tf in tfs:
            try:
                n_bars = (tf_bars or {}).get(tf, bars)
                df, _ = fetch(symbol, tf, n_bars)
                if df is None or len(df) < 100:
                    continue
                df.columns = [c.lower() for c in df.columns]
                if "volume" not in df.columns:
                    df["volume"] = 1.0

                price = float(df["close"].iloc[-1])
                close = df["close"]

                reasons = []
                conviction = 0
                signal = "HOLD"

                # ── REGIME FILTER 1: Session (foloseste TIMPUL BARII, nu ceasul) ──
                if elements.get("session_filter", True):
                    # Timpul barii: din coloana "time" (live) SAU din index (backtest)
                    bar_hour = None
                    try:
                        if "time" in df.columns:
                            t = df["time"].iloc[-1]
                            bar_hour = t.hour if hasattr(t, "hour") else None
                        elif hasattr(df.index[-1], "hour"):
                            bar_hour = df.index[-1].hour    # DatetimeIndex (backtest)
                    except Exception:
                        bar_hour = None
                    if bar_hour is None:
                        bar_hour = datetime.now(timezone.utc).hour
                    if bar_hour >= 22 or bar_hour < 1:
                        reasons.append(f"Sesiune moarta ({bar_hour:02d} UTC) — skip")
                        tf_results.append({"tf": tf, "signal": "HOLD", "conviction": 0,
                                           "reasons": reasons, "price": price,
                                           "sl": None, "tp": None})
                        continue

                # ── REGIME FILTER 2: ADX (range only) ─────────────────────
                adx_val = _adx(df, 14)
                if elements.get("regime_adx", True):
                    if adx_val >= 25:
                        reasons.append(f"ADX {adx_val:.1f} >= 25 — TREND, mean reversion riscant, skip")
                        tf_results.append({"tf": tf, "signal": "HOLD", "conviction": 0,
                                           "reasons": reasons, "price": price,
                                           "sl": None, "tp": None})
                        continue
                    # Regim range = baza solida pentru mean reversion (research)
                    reasons.append(f"ADX {adx_val:.1f} < 25 — range ideal MR (+3)")
                    conviction += 3

                # ── REGIME FILTER 3: ATR (volatilitate suficienta) ───────
                atr_cur, atr_avg = _atr(df, 14)
                if elements.get("regime_atr", True) and atr_avg > 0:
                    atr_ratio = atr_cur / atr_avg
                    atr_min = 1.2 if elements.get("atr_regime_strict", False) else 0.6
                    if atr_ratio < atr_min:
                        reasons.append(f"ATR ratio {atr_ratio:.2f} < {atr_min} — volatilitate prea mica, skip")
                        tf_results.append({"tf": tf, "signal": "HOLD", "conviction": 0,
                                           "reasons": reasons, "price": price,
                                           "sl": None, "tp": None})
                        continue
                    reasons.append(f"ATR ratio {atr_ratio:.2f} — volatilitate OK (+1)")
                    conviction += 1

                # ── Indicatori ───────────────────────────────────────────
                vwap, vwap_up, vwap_low = _vwap_session(df)
                bb_up, bb_mid, bb_low = _bollinger(close, 20, 2.0)
                rsi_val = _rsi(close, 14)

                # ── Determina directie din BB extreme (aproape de banda) ──
                bb_signal = "HOLD"
                bb_range = bb_up - bb_low
                if elements.get("bb_extreme", True) and bb_range > 0:
                    # Zona: ultimele 20% spre banda (nu doar atingere exacta)
                    dist_low  = (price - bb_low) / bb_range
                    dist_up   = (bb_up - price) / bb_range
                    if price <= bb_low:
                        bb_signal = "BUY"
                        reasons.append(f"Pret sub banda BB inferioara {bb_low:.5f} (+5)")
                        conviction += 5
                    elif price >= bb_up:
                        bb_signal = "SELL"
                        reasons.append(f"Pret peste banda BB superioara {bb_up:.5f} (+5)")
                        conviction += 5
                    elif dist_low < 0.15:
                        bb_signal = "BUY"
                        reasons.append(f"Pret aproape de banda inferioara ({dist_low*100:.0f}% range) (+4)")
                        conviction += 4
                    elif dist_up < 0.15:
                        bb_signal = "SELL"
                        reasons.append(f"Pret aproape de banda superioara ({dist_up*100:.0f}% range) (+4)")
                        conviction += 4

                # ── VWAP extreme (confirmare sau directie alternativa) ────
                vwap_signal = "HOLD"
                if elements.get("vwap_extreme", True):
                    if price <= vwap_low:
                        vwap_signal = "BUY"
                        if bb_signal != "SELL":
                            reasons.append(f"Pret sub VWAP-2σ ({vwap_low:.5f}) — oversold (+2)")
                            conviction += 2
                    elif price >= vwap_up:
                        vwap_signal = "SELL"
                        if bb_signal != "BUY":
                            reasons.append(f"Pret peste VWAP+2σ ({vwap_up:.5f}) — overbought (+2)")
                            conviction += 2

                # Combina semnale (BB are prioritate)
                signal = bb_signal if bb_signal != "HOLD" else vwap_signal

                if signal == "HOLD":
                    reasons.append("Pret nu e la extreme BB/VWAP")
                    tf_results.append({"tf": tf, "signal": "HOLD", "conviction": 0,
                                       "reasons": reasons, "price": price,
                                       "sl": None, "tp": None})
                    continue

                # ── RSI confirmation ─────────────────────────────────────
                rsi_ok = False
                if elements.get("rsi_confirm", True):
                    if signal == "BUY" and rsi_val < 30:
                        reasons.append(f"RSI {rsi_val:.1f} < 30 oversold — confirma BUY (+2)")
                        conviction += 2
                        rsi_ok = True
                    elif signal == "SELL" and rsi_val > 70:
                        reasons.append(f"RSI {rsi_val:.1f} > 70 overbought — confirma SELL (+2)")
                        conviction += 2
                        rsi_ok = True
                    elif signal == "BUY" and rsi_val < 40:
                        reasons.append(f"RSI {rsi_val:.1f} < 40 — partial BUY (+1)")
                        conviction += 1
                        rsi_ok = True
                    elif signal == "SELL" and rsi_val > 60:
                        reasons.append(f"RSI {rsi_val:.1f} > 60 — partial SELL (+1)")
                        conviction += 1
                        rsi_ok = True
                    else:
                        reasons.append(f"RSI {rsi_val:.1f} — nu confirma {signal} (-1)")
                        conviction -= 1

                # STRICT: cere ambele BB extreme + RSI
                if elements.get("require_both_bb_rsi", False):
                    bb_at_extreme = (signal == "BUY" and price <= bb_low) or \
                                    (signal == "SELL" and price >= bb_up)
                    if not (bb_at_extreme and rsi_ok):
                        reasons.append("require_both_bb_rsi: nu am BB extreme + RSI — HOLD")
                        tf_results.append({"tf": tf, "signal": "HOLD", "conviction": 0,
                                           "reasons": reasons, "price": price,
                                           "sl": None, "tp": None})
                        continue

                # ── Volume confirmation ──────────────────────────────────
                if elements.get("volume_confirm", True):
                    v_now = float(df["volume"].iloc[-1])
                    v_avg = float(df["volume"].iloc[-20:].mean())
                    if v_avg > 0 and v_now / v_avg >= 1.2:
                        reasons.append(f"Volume {v_now/v_avg:.2f}x medie — confirmare (+1)")
                        conviction += 1
                    elif v_avg > 0 and v_now / v_avg < 0.7:
                        reasons.append(f"Volume slab ({v_now/v_avg:.2f}x) — semnal mai slab (-1)")
                        conviction -= 1

                # ── SL / TP (research: SL sub swing, TP la VWAP / RR 1.5-3) ─
                # SL/TP tunabile (default research: SL strans, TP moderat)
                SL_ATR = float(kwargs.get("sl_atr", elements.get("_sl_atr", 0.8)))
                TP_ATR = float(kwargs.get("tp_atr", elements.get("_tp_atr", 1.6)))
                ph_idx, pl_idx = find_pivots(df, lookback=5)
                if signal == "BUY":
                    sl = round(price - atr_cur * SL_ATR, 5)
                    # TP = min(VWAP, 1.6ATR) daca tp_at_vwap, altfel 1.6ATR fix
                    if elements.get("tp_at_vwap", True) and vwap > price:
                        tp = round(min(vwap, price + atr_cur * TP_ATR), 5)
                    else:
                        tp = round(price + atr_cur * TP_ATR, 5)
                    if tp <= price:
                        tp = round(price + atr_cur * TP_ATR, 5)
                    risk = price - sl
                else:  # SELL
                    sl = round(price + atr_cur * SL_ATR, 5)
                    if elements.get("tp_at_vwap", True) and vwap < price:
                        tp = round(max(vwap, price - atr_cur * TP_ATR), 5)
                    else:
                        tp = round(price - atr_cur * TP_ATR, 5)
                    if tp >= price:
                        tp = round(price - atr_cur * TP_ATR, 5)
                    risk = sl - price

                reward = abs(tp - price)
                if risk <= 0 or reward <= 0:
                    tf_results.append({"tf": tf, "signal": "HOLD", "conviction": 0,
                                       "reasons": reasons, "price": price,
                                       "sl": None, "tp": None})
                    continue
                reasons.append(f"RR {reward/risk:.2f}:1 · SL 1.2ATR · TP {'VWAP/1.2ATR' if elements.get('tp_at_vwap', True) else 'RR1.5'}")

                conviction = max(conviction, 0)
                tf_results.append({
                    "tf": tf, "signal": signal, "conviction": conviction,
                    "reasons": reasons, "price": round(price, 5),
                    "sl": sl, "tp": tp,
                })

            except Exception as exc:
                log.warning(f"VWAPReversion {symbol}/{tf}: {exc}")

        if not tf_results:
            return self._empty_result(symbol, "Fara date suficiente")
        return self._build_result(symbol, tf_results, min_confidence, min_votes=1)
