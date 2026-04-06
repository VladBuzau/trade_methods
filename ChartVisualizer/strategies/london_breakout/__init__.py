"""
Strategie London Breakout: breakout din range-ul Asian Session (00:00-06:00 UTC)
la deschiderea sesiunii Londra (06:00-09:00 UTC).
Functioneaza cel mai bine pe perechi GBP, EUR + XAUUSD.
"""
from __future__ import annotations
import logging
from datetime import datetime, timezone

import numpy as np

from strategies.base import Strategy

log = logging.getLogger(__name__)

# Fereastra activa (UTC)
LONDON_OPEN_HOUR  = 6
LONDON_CLOSE_HOUR = 10
ASIAN_START_HOUR  = 0
ASIAN_END_HOUR    = 6


class LondonBreakoutStrategy(Strategy):
    key   = "london_breakout"
    name  = "London Breakout"
    icon  = "🇬🇧"
    color = "#ef5350"

    default_tfs  = ["M15", "H1"]
    default_bars = 200
    elements     = {
        "asian_range":  "Range Asian Session (00-06 UTC)",
        "breakout":     "Breakout + retest confirmare",
        "session_gate": "Activ doar in sesiunea Londra",
    }

    def analyze(self, symbol, tfs, bars=200, tf_bars=None, elements=None,
                min_confidence=66.0, **kwargs):
        from app import fetch, calc_sl_tp, find_pivots

        if elements is None:
            elements = {k: True for k in self.elements}

        now_utc = datetime.now(timezone.utc)
        in_london = LONDON_OPEN_HOUR <= now_utc.hour < LONDON_CLOSE_HOUR

        # Daca e activ session_gate si nu suntem in sesiunea Londra
        if elements.get("session_gate", True) and not in_london:
            result = self._empty_result(symbol, f"In afara sesiunii Londra ({now_utc.strftime('%H:%M')} UTC, activ 06-10)")
            result["out_of_session"] = True
            return result

        tf_results = []
        for tf in tfs:
            try:
                n_bars = (tf_bars or {}).get(tf, bars)
                df, _  = fetch(symbol, tf, n_bars)
                if df is None or len(df) < 50:
                    continue

                price = float(df["close"].iloc[-1])

                # Calculeaza range-ul Asian Session din ultimele bare
                # Folosim timestamp-urile din df daca sunt disponibile
                # Fallback: ultimele 24 bare pe M15 = 6 ore
                if "time" in df.columns or df.index.name == "time":
                    try:
                        ts_col = df["time"] if "time" in df.columns else df.index
                        ts_col = ts_col.apply(lambda x: x if hasattr(x, "hour") else datetime.fromtimestamp(x, tz=timezone.utc))
                        asian_mask = ts_col.apply(lambda x: ASIAN_START_HOUR <= x.hour < ASIAN_END_HOUR)
                        asian_df = df[asian_mask]
                    except Exception:
                        asian_df = df.iloc[-24:]
                else:
                    asian_df = df.iloc[-24:]

                if len(asian_df) < 3:
                    asian_df = df.iloc[-24:]

                asian_high = float(asian_df["high"].max())
                asian_low  = float(asian_df["low"].min())
                asian_range = asian_high - asian_low

                if asian_range <= 0:
                    continue

                reasons    = []
                conviction = 0
                sig        = "HOLD"

                # Breakout
                if elements.get("asian_range", True):
                    reasons.append(f"Asian range: {asian_low:.5f} — {asian_high:.5f} ({asian_range:.5f})")

                if elements.get("breakout", True):
                    buffer = asian_range * 0.1  # 10% buffer anti-fakeout
                    if price > asian_high + buffer:
                        reasons.append(f"Breakout sus din range Asian ({price:.5f} > {asian_high:.5f})")
                        conviction += 3
                        sig = "BUY"
                    elif price < asian_low - buffer:
                        reasons.append(f"Breakout jos din range Asian ({price:.5f} < {asian_low:.5f})")
                        conviction += 3
                        sig = "SELL"

                if sig == "HOLD":
                    tf_results.append({"tf": tf, "signal": "HOLD", "conviction": 0,
                                       "reasons": reasons, "price": price,
                                       "sl": None, "tp": None})
                    continue

                # SL/TP bazat pe range
                if sig == "BUY":
                    sl = asian_low - buffer
                    tp = price + asian_range * 1.5
                else:
                    sl = asian_high + buffer
                    tp = price - asian_range * 1.5

                tf_results.append({
                    "tf":         tf,
                    "signal":     sig,
                    "conviction": conviction,
                    "reasons":    reasons,
                    "price":      round(price, 5),
                    "sl":         round(sl, 5),
                    "tp":         round(tp, 5),
                })
            except Exception as exc:
                log.warning(f"LondonBreakoutStrategy {symbol}/{tf}: {exc}")

        if not tf_results:
            return self._empty_result(symbol, "Fara date suficiente")

        return self._build_result(symbol, tf_results, min_confidence, min_votes=1)
