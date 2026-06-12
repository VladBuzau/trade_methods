"""
Bollinger Bands v3: Conviction scoring — nimic nu blocheaza hard.
Mean reversion: atingere banda + confirmare RSI/ADX/squeeze.
"""
from __future__ import annotations
import logging
import numpy as np
from strategies.base import Strategy

log = logging.getLogger(__name__)


class BollingerStrategy(Strategy):
    key   = "bollinger"
    name  = "Bollinger Bands"
    icon  = "🎯"
    color = "#00bcd4"

    default_tfs  = ["M15", "H1", "H4"]
    default_bars = 300
    elements     = {
        "band_touch":     "Atingere banda extrema (2σ)",
        "extreme_only":   "STRICT: doar atingere directa (<5% spread banda)",
        "double_touch":   "Double Touch banda (+2)",
        "rsi_confirm":    "RSI confirmare (30/70)",
        "rsi_strict":     "STRICT: RSI oversold (<30) / overbought (>70)",
        "adx_range":      "ADX < 25 (mean reversion in range)",
        "bbw_filter":     "BB Width filter (squeeze detector)",
        "mid_band_tp":    "TP la linia medie SMA20",
        "candle_confirm": "Confirmare lumânare reversal (corp opus benzii)",
    }

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
                if df is None or len(df) < 50:
                    continue

                close = df["close"]
                price = float(close.iloc[-1])
                period = 20

                sma   = close.rolling(period).mean()
                std   = close.rolling(period).std()
                upper = sma + 2 * std
                lower = sma - 2 * std

                upper_v = float(upper.iloc[-1])
                lower_v = float(lower.iloc[-1])
                sma_v   = float(sma.iloc[-1])

                # RSI
                delta = close.diff()
                gain  = delta.clip(lower=0).rolling(14).mean()
                loss  = (-delta.clip(upper=0)).rolling(14).mean()
                rsi   = 100 - 100 / (1 + gain / loss.replace(0, np.nan))
                rsi_v = float(rsi.iloc[-1])

                reasons    = []
                conviction = 0
                sig        = "HOLD"

                # Band touch — determina directia
                if elements.get("band_touch", True):
                    band_range = upper_v - lower_v
                    dist_lower = price - lower_v
                    dist_upper = upper_v - price
                    extreme_only = elements.get("extreme_only", False)
                    # Zona apropiata de banda
                    if band_range > 0 and dist_lower / band_range < 0.05:
                        reasons.append(f"Pret la banda inferioara ({lower_v:.5f}) — BUY (+3)")
                        conviction += 3
                        sig = "BUY"
                    elif band_range > 0 and dist_upper / band_range < 0.05:
                        reasons.append(f"Pret la banda superioara ({upper_v:.5f}) — SELL (+3)")
                        conviction += 3
                        sig = "SELL"
                    elif not extreme_only and band_range > 0 and dist_lower / band_range < 0.15:
                        reasons.append(f"Pret aproape de banda inferioara ({lower_v:.5f}) — BUY (+2)")
                        conviction += 2
                        sig = "BUY"
                    elif not extreme_only and band_range > 0 and dist_upper / band_range < 0.15:
                        reasons.append(f"Pret aproape de banda superioara ({upper_v:.5f}) — SELL (+2)")
                        conviction += 2
                        sig = "SELL"
                    else:
                        reasons.append(f"Pret in interiorul benzilor ({lower_v:.5f} — {upper_v:.5f})")

                if sig == "HOLD":
                    tf_results.append({"tf": tf, "signal": "HOLD", "conviction": 0,
                                       "reasons": reasons, "price": price, "sl": None, "tp": None})
                    continue

                # ADX range confirmation — scoring
                if elements.get("adx_range", True):
                    try:
                        h = df["high"].values; l = df["low"].values; c = df["close"].values
                        tr_arr = [max(h[i]-l[i], abs(h[i]-c[i-1]), abs(l[i]-c[i-1])) for i in range(1, len(c))]
                        atr_recent = np.mean(tr_arr[-5:])  if len(tr_arr) >= 5  else 1e-10
                        atr_sma20  = np.mean(tr_arr[-20:]) if len(tr_arr) >= 20 else 1e-10
                        if atr_sma20 > 0:
                            ratio = atr_recent / atr_sma20
                            if ratio < 0.8:
                                reasons.append(f"ATR comprimat (ratio {ratio:.2f}) — squeeze, mean reversion ideal (+2)")
                                conviction += 2
                            elif ratio < 1.3:
                                reasons.append(f"ATR normal (ratio {ratio:.2f}) — range OK (+1)")
                                conviction += 1
                            else:
                                reasons.append(f"ATR expandat (ratio {ratio:.2f}) — trending, mean reversion riscant (-2)")
                                conviction -= 2
                    except Exception:
                        pass

                # BB Width filter — scoring
                if elements.get("bbw_filter", True) and sma_v > 0:
                    bbw_series = ((upper - lower) / sma).dropna()
                    if len(bbw_series) >= 20:
                        bbw_now = float(bbw_series.iloc[-1])
                        bbw_p80 = float(bbw_series.quantile(0.80))
                        bbw_p20 = float(bbw_series.quantile(0.20))
                        if bbw_now < bbw_p20:
                            reasons.append(f"BB Squeeze: width {bbw_now:.4f} — volatilitate minima (+2)")
                            conviction += 2
                        elif bbw_now > bbw_p80:
                            reasons.append(f"BB Width lata {bbw_now:.4f} > p80 — miscare consumata (-1)")
                            conviction -= 1
                        else:
                            reasons.append(f"BB Width normal {bbw_now:.4f}")

                # Double Touch — scoring
                if elements.get("double_touch", True):
                    lookback = 10
                    if sig == "BUY":
                        prev_lower = lower.iloc[-lookback-1:-1]
                        prev_close = close.iloc[-lookback-1:-1]
                        touches = int((prev_close <= prev_lower).sum())
                    else:
                        prev_upper = upper.iloc[-lookback-1:-1]
                        prev_close = close.iloc[-lookback-1:-1]
                        touches = int((prev_close >= prev_upper).sum())
                    if touches >= 2:
                        reasons.append(f"Double Touch: {touches} atingeri in {lookback} bare (+2)")
                        conviction += 2
                    elif touches >= 1:
                        reasons.append(f"Touch anterior: {touches} atingere(i) (+1)")
                        conviction += 1

                # RSI confirmare — scoring (strict opt: doar oversold/overbought)
                if elements.get("rsi_confirm", True):
                    rsi_strict = elements.get("rsi_strict", False)
                    if sig == "BUY" and rsi_v < 30:
                        reasons.append(f"RSI oversold {rsi_v:.1f} — ideal (+3)")
                        conviction += 3
                    elif not rsi_strict and sig == "BUY" and rsi_v < 40:
                        reasons.append(f"RSI {rsi_v:.1f} — zona favorabila BUY (+2)")
                        conviction += 2
                    elif not rsi_strict and sig == "BUY" and rsi_v < 50:
                        reasons.append(f"RSI {rsi_v:.1f} — partial (+1)")
                        conviction += 1
                    elif sig == "BUY" and rsi_v > 60:
                        reasons.append(f"RSI {rsi_v:.1f} — nu confirma BUY (-1)")
                        conviction -= 1
                    elif sig == "SELL" and rsi_v > 70:
                        reasons.append(f"RSI overbought {rsi_v:.1f} — ideal (+3)")
                        conviction += 3
                    elif not rsi_strict and sig == "SELL" and rsi_v > 60:
                        reasons.append(f"RSI {rsi_v:.1f} — zona favorabila SELL (+2)")
                        conviction += 2
                    elif not rsi_strict and sig == "SELL" and rsi_v > 50:
                        reasons.append(f"RSI {rsi_v:.1f} — partial (+1)")
                        conviction += 1
                    elif sig == "SELL" and rsi_v < 40:
                        reasons.append(f"RSI {rsi_v:.1f} — nu confirma SELL (-1)")
                        conviction -= 1
                    # rsi_strict: penalizeaza daca nu e in extrema
                    if rsi_strict:
                        if sig == "BUY" and rsi_v >= 30:
                            reasons.append(f"RSI strict: {rsi_v:.1f} >= 30 — nu oversold (-2)")
                            conviction -= 2
                        elif sig == "SELL" and rsi_v <= 70:
                            reasons.append(f"RSI strict: {rsi_v:.1f} <= 70 — nu overbought (-2)")
                            conviction -= 2

                # Candle confirm — corp lumanare in directia revenirii
                if elements.get("candle_confirm", False):
                    o = float(df["open"].iloc[-1])
                    c = float(close.iloc[-1])
                    body_dir = "up" if c > o else ("down" if c < o else "flat")
                    if sig == "BUY" and body_dir == "up":
                        reasons.append("Candle bullish confirma BUY (+1)")
                        conviction += 1
                    elif sig == "SELL" and body_dir == "down":
                        reasons.append("Candle bearish confirma SELL (+1)")
                        conviction += 1
                    elif body_dir != "flat":
                        reasons.append(f"Candle nu confirma directia ({body_dir} vs {sig}) (-1)")
                        conviction -= 1

                conviction = max(conviction, 0)
                ph_idx, pl_idx = find_pivots(df, lookback=5)
                sl, tp = calc_sl_tp(df, ph_idx, pl_idx, sig, price)
                # TP la SMA midline (opțional, default ON)
                if elements.get("mid_band_tp", True):
                    tp = round(sma_v, 5)

                tf_results.append({
                    "tf": tf, "signal": sig, "conviction": conviction,
                    "reasons": reasons, "price": round(price, 5),
                    "sl": sl, "tp": tp,
                })
            except Exception as exc:
                log.warning(f"BollingerStrategy {symbol}/{tf}: {exc}")

        if not tf_results:
            return self._empty_result(symbol, "Fara date suficiente")

        return self._build_result(symbol, tf_results, min_confidence, min_votes=1)
