"""
New York Breakout: breakout din range-ul pre-NY (09:00-13:00 UTC) la deschiderea NY.
Sesiunea New York (13:00-17:00 UTC) aduce cel mai mare volum pe Forex.
Complement la London Breakout.
"""
from __future__ import annotations
import logging
from datetime import datetime, timezone
from strategies.base import Strategy

log = logging.getLogger(__name__)

NY_OPEN_HOUR   = 13
NY_CLOSE_HOUR  = 18
PRE_NY_START   = 9
PRE_NY_END     = 13


class NYBreakoutStrategy(Strategy):
    key   = "ny_breakout"
    name  = "NY Breakout"
    icon  = "🗽"
    color = "#42a5f5"

    default_tfs  = ["M15", "H1"]
    default_bars = 200
    elements     = {
        "pre_ny_range":  "Range pre-NY (09-13 UTC)",
        "breakout":      "Breakout + buffer anti-fakeout",
        "session_gate":  "Activ doar in sesiunea NY",
    }

    def analyze(self, symbol, tfs, bars=200, tf_bars=None, elements=None,
                min_confidence=66.0, **kwargs):
        from app import fetch, calc_sl_tp, find_pivots

        if elements is None:
            elements = {k: True for k in self.elements}

        now_utc  = datetime.now(timezone.utc)
        in_ny    = NY_OPEN_HOUR <= now_utc.hour < NY_CLOSE_HOUR

        if elements.get("session_gate", True) and not in_ny:
            result = self._empty_result(
                symbol,
                f"In afara sesiunii NY ({now_utc.strftime('%H:%M')} UTC, activ 13-18)"
            )
            result["out_of_session"] = True
            return result

        tf_results = []
        for tf in tfs:
            try:
                n_bars = (tf_bars or {}).get(tf, bars)
                df, _  = fetch(symbol, tf, n_bars)
                if df is None or len(df) < 30:
                    continue

                price = float(df["close"].iloc[-1])

                # Range pre-NY: bare cu timestamp intre 09-13 UTC
                if "time" in df.columns or df.index.name == "time":
                    try:
                        ts_col = df["time"] if "time" in df.columns else df.index
                        ts_col = ts_col.apply(
                            lambda x: x if hasattr(x, "hour")
                            else datetime.fromtimestamp(x, tz=timezone.utc)
                        )
                        pre_ny_mask = ts_col.apply(
                            lambda x: PRE_NY_START <= x.hour < PRE_NY_END
                        )
                        pre_df = df[pre_ny_mask]
                    except Exception:
                        pre_df = df.iloc[-16:]
                else:
                    pre_df = df.iloc[-16:]

                if len(pre_df) < 2:
                    pre_df = df.iloc[-16:]

                pre_high = float(pre_df["high"].max())
                pre_low  = float(pre_df["low"].min())
                pre_range = pre_high - pre_low

                if pre_range <= 0:
                    continue

                reasons    = []
                conviction = 0
                sig        = "HOLD"

                if elements.get("pre_ny_range", True):
                    reasons.append(f"Range pre-NY: {pre_low:.5f} — {pre_high:.5f} ({pre_range:.5f})")

                if elements.get("breakout", True):
                    buffer = pre_range * 0.1
                    if price > pre_high + buffer:
                        reasons.append(f"Breakout UP din range pre-NY ({price:.5f} > {pre_high:.5f})")
                        conviction += 3
                        sig = "BUY"
                    elif price < pre_low - buffer:
                        reasons.append(f"Breakout DOWN din range pre-NY ({price:.5f} < {pre_low:.5f})")
                        conviction += 3
                        sig = "SELL"

                if sig == "HOLD":
                    tf_results.append({"tf": tf, "signal": "HOLD", "conviction": 0,
                                       "reasons": reasons, "price": price,
                                       "sl": None, "tp": None})
                    continue

                if sig == "BUY":
                    sl = pre_low - buffer
                    tp = price + pre_range * 1.5
                else:
                    sl = pre_high + buffer
                    tp = price - pre_range * 1.5

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
                log.warning(f"NYBreakoutStrategy {symbol}/{tf}: {exc}")

        if not tf_results:
            return self._empty_result(symbol, "Fara date suficiente")

        return self._build_result(symbol, tf_results, min_confidence, min_votes=1)
