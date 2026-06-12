"""
TrendRider — unified trend-following strategy.
Inlocuieste: Classic, EMA Cross, MACD, Supertrend.

Folozofie: un singur sistem complet pentru tradinguri in trend, cu 5 confluente:
  1. EMA Alignment  — EMA 8 > 21 > 50 (bull) sau invers (bear)
  2. MACD Confirm   — linie > semnal (bull) si histograma in crestere
  3. Supertrend     — linia sub pret (bull) sau deasupra (bear)
  4. ADX > 25       — trend puternic, nu range
  5. Momentum (ROC) — pretul a crescut/scazut in ultimele N bare

Toate 5 trebuie aliniate pentru a genera semnal → puncte conviction per TF.
TP/SL dinamice bazate pe ATR + trailing natural via Supertrend.
"""
from __future__ import annotations
import logging
import numpy as np

from strategies.base import Strategy, volatility_ok

log = logging.getLogger(__name__)

ADX_MIN = 25.0          # sub 25 = range, skip
ROC_BARS = 10           # momentum lookback
SL_ATR_MULT = 1.5
TP_RR = 1.5             # RR 1:1.5 — trade scurt, TP apropiat


# ── Indicatori vectorizati (fara pandas-ta ca sa evitam dependenta noua) ─────
def _ema(series, period):
    return series.ewm(span=period, adjust=False).mean()


def _atr(df, period=14):
    high, low, close = df["high"].values, df["low"].values, df["close"].values
    tr = np.maximum(high[1:] - low[1:],
                    np.maximum(np.abs(high[1:] - close[:-1]),
                               np.abs(low[1:]  - close[:-1])))
    if len(tr) < period:
        return float(np.mean(tr)) if len(tr) else 0.0
    return float(np.mean(tr[-period:]))


def _adx(df, period=14):
    """ADX vectorizat — returneaza ultimul ADX."""
    high  = df["high"].values
    low   = df["low"].values
    close = df["close"].values
    if len(close) < period * 2 + 5:
        return 0.0
    tr    = np.maximum(high[1:] - low[1:],
                       np.maximum(np.abs(high[1:] - close[:-1]),
                                  np.abs(low[1:]  - close[:-1])))
    plus_dm  = np.where((high[1:] - high[:-1]) > (low[:-1] - low[1:]),
                        np.maximum(high[1:] - high[:-1], 0), 0)
    minus_dm = np.where((low[:-1] - low[1:]) > (high[1:] - high[:-1]),
                        np.maximum(low[:-1] - low[1:], 0), 0)
    def rma(v, p):
        out = np.zeros_like(v, dtype=float)
        out[p-1] = np.mean(v[:p])
        for i in range(p, len(v)):
            out[i] = (out[i-1] * (p-1) + v[i]) / p
        return out
    atr    = rma(tr, period)
    plus_d = 100 * rma(plus_dm, period) / np.where(atr == 0, 1, atr)
    minus_d = 100 * rma(minus_dm, period) / np.where(atr == 0, 1, atr)
    dx = 100 * np.abs(plus_d - minus_d) / np.where((plus_d + minus_d) == 0, 1, plus_d + minus_d)
    adx = rma(dx, period)
    return float(adx[-1])


def _macd(series, fast=12, slow=26, signal=9):
    ema_f = series.ewm(span=fast,  adjust=False).mean()
    ema_s = series.ewm(span=slow,  adjust=False).mean()
    line  = ema_f - ema_s
    sig   = line.ewm(span=signal, adjust=False).mean()
    hist  = line - sig
    return float(line.iloc[-1]), float(sig.iloc[-1]), float(hist.iloc[-1]), float(hist.iloc[-2])


def _supertrend(df, period=10, mult=3.0):
    """Returneaza (dir_last, line_last): dir=1 bull, -1 bear."""
    high, low, close = df["high"].values, df["low"].values, df["close"].values
    n = len(close)
    if n < period * 3:
        return 0, 0.0
    tr = np.maximum(high[1:] - low[1:],
                    np.maximum(np.abs(high[1:] - close[:-1]),
                               np.abs(low[1:]  - close[:-1])))
    atr = np.zeros(n)
    atr[period] = np.mean(tr[:period])
    for i in range(period+1, n):
        atr[i] = (atr[i-1] * (period-1) + tr[i-1]) / period

    hl2 = (high + low) / 2
    upper = hl2 + mult * atr
    lower = hl2 - mult * atr
    final_upper = upper.copy()
    final_lower = lower.copy()
    direction = np.ones(n)
    for i in range(period+1, n):
        if upper[i] < final_upper[i-1] or close[i-1] > final_upper[i-1]:
            final_upper[i] = upper[i]
        else:
            final_upper[i] = final_upper[i-1]
        if lower[i] > final_lower[i-1] or close[i-1] < final_lower[i-1]:
            final_lower[i] = lower[i]
        else:
            final_lower[i] = final_lower[i-1]
        if direction[i-1] == 1 and close[i] < final_lower[i]:
            direction[i] = -1
        elif direction[i-1] == -1 and close[i] > final_upper[i]:
            direction[i] = 1
        else:
            direction[i] = direction[i-1]
    line = final_lower[-1] if direction[-1] == 1 else final_upper[-1]
    return int(direction[-1]), float(line)


class TrendRiderStrategy(Strategy):
    key   = "trend_rider"
    name  = "TrendRider"
    icon  = "🚀"
    color = "#2e7d32"

    default_tfs  = ["H1", "H4"]
    default_bars = 500
    elements     = {
        "ema":            "EMA 8/21/50 Alignment",
        "macd":           "MACD Line > Signal + histograma crescatoare",
        "supertrend":     "Supertrend direction (trailing stop natural)",
        "adx":            "ADX > 25 (trend puternic)",
        "momentum":       "ROC > 0 in directia semnalului",
        "volatility":     "ATR regim normal (filter)",
        "require_all_5":  "STRICT: toate 5 confluente obligatorii",
        "pullback_only":  "STRICT: doar pullback la EMA21 (anti chase)",
        "roc_strict":     "ROC > 0.5% (default 0.15%)",
        "adx_30":         "STRICT: ADX > 30 (vs default 25)",
    }

    def analyze(self, symbol, tfs, bars=500, tf_bars=None, elements=None,
                min_confidence=85.0, **kwargs):
        from app import fetch

        if elements is None:
            elements = {k: True for k in self.elements}

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

                # ── Gate: volatility ──
                if elements.get("volatility", True):
                    vol_ok, vol_msg = volatility_ok(df)
                    if not vol_ok:
                        reasons.append(vol_msg)
                        tf_results.append({"tf": tf, "signal": "HOLD",
                                           "conviction": 0, "reasons": reasons,
                                           "price": float(df["close"].iloc[-1]),
                                           "sl": None, "tp": None})
                        continue
                    reasons.append(vol_msg)

                # ── Calcule ──
                close = df["close"]
                ema8  = _ema(close, 8)
                ema21 = _ema(close, 21)
                ema50 = _ema(close, 50)
                ema8_v  = float(ema8.iloc[-1])
                ema21_v = float(ema21.iloc[-1])
                ema50_v = float(ema50.iloc[-1])
                price   = float(close.iloc[-1])

                bull_align = ema8_v > ema21_v > ema50_v
                bear_align = ema8_v < ema21_v < ema50_v

                # MACD
                macd_line, macd_sig, hist_cur, hist_prev = _macd(close)
                macd_bull = macd_line > macd_sig and hist_cur > hist_prev
                macd_bear = macd_line < macd_sig and hist_cur < hist_prev

                # Supertrend
                st_dir, st_line = _supertrend(df)

                # ADX
                adx_val = _adx(df)

                # ROC momentum
                if len(close) > ROC_BARS:
                    roc = (price - float(close.iloc[-ROC_BARS-1])) / float(close.iloc[-ROC_BARS-1]) * 100
                else:
                    roc = 0.0

                # ── Voting ──
                bull_score = bear_score = 0
                if elements.get("ema", True):
                    if bull_align:
                        bull_score += 2; reasons.append(f"EMA 8>21>50 bullish")
                    elif bear_align:
                        bear_score += 2; reasons.append(f"EMA 8<21<50 bearish")
                    else:
                        reasons.append("EMA nu e aliniata")

                if elements.get("macd", True):
                    if macd_bull:
                        bull_score += 2; reasons.append("MACD bull + hist crescatoare")
                    elif macd_bear:
                        bear_score += 2; reasons.append("MACD bear + hist descrescatoare")

                if elements.get("supertrend", True):
                    if st_dir == 1:
                        bull_score += 2; reasons.append(f"Supertrend bull @ {st_line:.5f}")
                    elif st_dir == -1:
                        bear_score += 2; reasons.append(f"Supertrend bear @ {st_line:.5f}")

                if elements.get("adx", True):
                    adx_threshold = 30.0 if elements.get("adx_30", False) else ADX_MIN
                    if adx_val >= adx_threshold:
                        reasons.append(f"ADX {adx_val:.1f} > {adx_threshold} (trend valid)")
                    else:
                        reasons.append(f"ADX {adx_val:.1f} < {adx_threshold} (range) — HOLD")
                        tf_results.append({"tf": tf, "signal": "HOLD",
                                           "conviction": 0, "reasons": reasons,
                                           "price": price, "sl": None, "tp": None})
                        continue

                if elements.get("momentum", True):
                    roc_threshold = 0.5 if elements.get("roc_strict", False) else 0.15
                    if roc > roc_threshold:
                        bull_score += 1; reasons.append(f"ROC +{roc:.2f}% in {ROC_BARS} bare")
                    elif roc < -roc_threshold:
                        bear_score += 1; reasons.append(f"ROC {roc:.2f}% in {ROC_BARS} bare")

                # ── Decizia ──
                signal = "HOLD"
                price_entry = price
                sl = tp = None
                atr_val = _atr(df)

                # Threshold scor: 5 default, 8 cu require_all_5 (toate 5 active)
                # max posibil = 2 ema + 2 macd + 2 supertrend + 1 momentum = 7
                # require_all_5 cere 7 (toate confluentele)
                score_threshold = 7 if elements.get("require_all_5", False) else 5

                # ── Pullback-only: cere pretul sa fie aproape de EMA21, nu departe ──
                pullback_ok = True
                if elements.get("pullback_only", False) and ema21_v > 0:
                    dist_pct = abs(price - ema21_v) / ema21_v * 100
                    if dist_pct > 0.5:  # > 0.5% de EMA21 = nu e pullback
                        pullback_ok = False
                        reasons.append(f"pullback_only: pret {dist_pct:.2f}% de EMA21 — nu pullback")

                if bull_score >= score_threshold and bull_score > bear_score and pullback_ok:
                    signal = "BUY"
                    sl = round(min(st_line, price - atr_val * SL_ATR_MULT), 5) if st_dir == 1 else \
                         round(price - atr_val * SL_ATR_MULT, 5)
                    tp = round(price + (price - sl) * TP_RR, 5)
                    conviction = bull_score + 2
                elif bear_score >= score_threshold and bear_score > bull_score and pullback_ok:
                    signal = "SELL"
                    sl = round(max(st_line, price + atr_val * SL_ATR_MULT), 5) if st_dir == -1 else \
                         round(price + atr_val * SL_ATR_MULT, 5)
                    tp = round(price - (sl - price) * TP_RR, 5)
                    conviction = bear_score + 2
                else:
                    reasons.append(f"Scor insuficient (bull={bull_score}, bear={bear_score}) < {score_threshold}")

                tf_results.append({
                    "tf": tf, "signal": signal, "conviction": conviction,
                    "reasons": reasons,
                    "price": round(price_entry, 5),
                    "sl": sl, "tp": tp,
                })

            except Exception as exc:
                log.warning(f"TrendRider {symbol}/{tf}: {exc}")

        if not tf_results:
            return self._empty_result(symbol, "Fara date")
        return self._build_result(symbol, tf_results, min_confidence, min_votes=2)
