"""
NY Breakout v2: breakout din range-ul pre-NY (09:00-13:00 UTC)
la deschiderea sesiunii New York (13:00-18:00 UTC).

Sesiunea NY aduce cel mai mare volum din zi pe Forex.
Overlap London-NY (13:00-16:00 UTC) = cel mai lichid moment.

Logica de intrare:
  1. session_gate       — fereastra 13-18 UTC
  2. pre_ny_range       — calitate range pre-NY vs ATR zilnic
  3. breakout           — depasire range + displacement candle (obligatoriu)
  4. retest             — retest la nivel de breakout (+conviction)
  5. h4_trend           — aliniere EMA20/EMA50 pe H4 (+/-conviction)
  6. time_window        — bonus overlap London-NY 13:00-15:00 UTC (prime time)
  7. ny_pairs           — bonus conviction pentru perechi USD majore
  8. london_moved_check — filtru: London a consumat deja >80% din ATR → skip (P3)
  9. us_open_pause      — pauza 15:25-15:40 UTC (US market open volatilitate)  (P3)

Perechi recomandate: EURUSD, GBPUSD, USDCAD, USDCHF, XAUUSD, USDJPY
"""
from __future__ import annotations
import logging
from datetime import datetime, timezone

from strategies.base import Strategy

log = logging.getLogger(__name__)

NY_OPEN_HOUR  = 13
NY_CLOSE_HOUR = 18
PRE_NY_START  = 9
PRE_NY_END    = 13

RANGE_MIN_ATR_FRAC = 0.15
RANGE_MAX_ATR_FRAC = 1.20

# Perechi cu expunere maxima la sesiunea NY
NY_PAIRS = {
    "EURUSD", "GBPUSD", "USDCAD", "USDCHF",
    "USDJPY", "XAUUSD", "GBPJPY", "EURJPY",
    "US30", "NAS100", "SPX500", "USOIL",
}


# ---------------------------------------------------------------------------
# Helpers (aceleasi ca london_breakout, duplicate intentionat)
# ---------------------------------------------------------------------------

def _extract_pre_ny_df(df):
    """Returneaza barele din fereastra pre-NY (09:00-13:00 UTC)."""
    try:
        if "time" in df.columns:
            ts = df["time"]
        elif df.index.name == "time":
            ts = df.index.to_series()
        else:
            return df.iloc[-16:]
        ts = ts.apply(
            lambda x: x if hasattr(x, "hour")
            else datetime.fromtimestamp(float(x), tz=timezone.utc)
        )
        mask   = ts.apply(lambda x: PRE_NY_START <= x.hour < PRE_NY_END)
        result = df[mask]
        return result if len(result) >= 3 else df.iloc[-16:]
    except Exception:
        return df.iloc[-16:]


def _has_displacement(df, direction: str, lookback: int = 3,
                      body_pct: float = 0.55) -> tuple[bool, str]:
    if df is None or len(df) < lookback + 1:
        return False, "Date insuficiente pentru displacement"
    opens  = df["open"].values
    closes = df["close"].values
    highs  = df["high"].values
    lows   = df["low"].values
    for i in range(-lookback, 0):
        rng = highs[i] - lows[i]
        if rng < 1e-10:
            continue
        body  = abs(closes[i] - opens[i])
        ratio = body / rng
        if ratio < body_pct:
            continue
        if direction == "BUY" and closes[i] > opens[i]:
            return True, f"Displacement BUY: corp {ratio*100:.1f}% din range (bar {i})"
        if direction == "SELL" and closes[i] < opens[i]:
            return True, f"Displacement SELL: corp {ratio*100:.1f}% din range (bar {i})"
    return False, f"Niciun displacement {direction} in ultimele {lookback} lumanari"


def _detect_retest(df, direction: str, level: float,
                   tol_frac: float = 0.15) -> tuple[bool, str]:
    if df is None or len(df) < 8:
        return False, "Date insuficiente pentru retest"
    closes = df["close"].values
    highs  = df["high"].values
    lows   = df["low"].values
    window    = min(30, len(df))
    tolerance = max((highs[-window:].max() - lows[-window:].min()) * tol_frac, 1e-8)

    if direction == "BUY":
        bk_idx = None
        for i in range(-window, -3):
            if closes[i] > level:
                bk_idx = i
                break
        if bk_idx is None:
            return False, "Retest BUY: nu s-a detectat breakout anterior"
        retest_ok = any(lows[i] <= level + tolerance for i in range(bk_idx + 1, 0))
        if not retest_ok:
            return False, "Retest BUY: pretul nu a revenit la nivel"
        if closes[-1] > level:
            return True, f"Retest BUY confirmat la {level:.5f}"
        return False, "Retest BUY: pretul nu a recuperat dupa retest"
    else:
        bk_idx = None
        for i in range(-window, -3):
            if closes[i] < level:
                bk_idx = i
                break
        if bk_idx is None:
            return False, "Retest SELL: nu s-a detectat breakout anterior"
        retest_ok = any(highs[i] >= level - tolerance for i in range(bk_idx + 1, 0))
        if not retest_ok:
            return False, "Retest SELL: pretul nu a revenit la nivel"
        if closes[-1] < level:
            return True, f"Retest SELL confirmat la {level:.5f}"
        return False, "Retest SELL: pretul nu a recuperat dupa retest"


# ---------------------------------------------------------------------------
# Strategie
# ---------------------------------------------------------------------------

class NYBreakoutStrategy(Strategy):
    key   = "ny_breakout"
    name  = "NY Breakout"
    icon  = "🗽"
    color = "#42a5f5"

    default_tfs  = ["M5", "M15"]
    default_bars = 200
    elements     = {
        "session_gate":        "Activ doar in sesiunea NY (13-18 UTC)",
        "pre_ny_range":        "Calitate range pre-NY (09-13 UTC) vs ATR zilnic",
        "breakout":            "Breakout + displacement candle (obligatoriu)",
        "retest":              "Retest la nivel breakout (+3 conviction)",
        "h4_trend":            "Aliniere EMA20/EMA50 pe H4 (+/-1 conviction)",
        "time_window":         "Bonus overlap London-NY 13:00-15:00 UTC (+2)",
        "ny_pairs":            "Bonus conviction perechi USD majore (+1)",
        "london_moved_check":  "Filtru London Already Moved (>80% ATR consumat → skip)",
        "us_open_pause":       "Pauza US Market Open 15:25-15:40 UTC (volatilitate)",
        "retest_only":         "STRICT: doar trade dupa retest (anti fake breakout)",
        "h4_strict":           "STRICT: H4 trend OBLIGATORIU aliniat",
        "overlap_only":        "STRICT: doar in fereastra overlap LO-NY 13-15 UTC",
    }

    def analyze(self, symbol, tfs, bars=200, tf_bars=None, elements=None,
                min_confidence=66.0, **kwargs):
        from app import fetch

        if elements is None:
            elements = {k: True for k in self.elements}

        import strategies as _strats
        now_utc = _strats.get_utc_now()
        hour_f  = now_utc.hour + now_utc.minute / 60.0
        in_ny   = NY_OPEN_HOUR <= now_utc.hour < NY_CLOSE_HOUR

        # --- Session gate — scoring -------------------------------------------
        session_penalty = 0
        if elements.get("session_gate", True) and not in_ny:
            session_penalty = -3

        if elements.get("us_open_pause", True):
            if 15.0 + 25/60 <= hour_f <= 15.0 + 40/60:
                session_penalty -= 2

        # --- Time window bonus --------------------------------------------
        if elements.get("time_window", True):
            if 13.0 <= hour_f < 15.0:
                time_bonus = 2
                time_note  = "overlap London-NY 13:00-15:00 UTC (prime time)"
            elif 15.0 <= hour_f < 16.5:
                time_bonus = 1
                time_note  = "sesiune NY activa 15:00-16:30 UTC"
            else:
                time_bonus = 0
                time_note  = "fereastra tarzie NY 16:30+ UTC"
        else:
            time_bonus = 0
            time_note  = ""

        # --- NY pairs bonus -----------------------------------------------
        ny_bonus = 0
        if elements.get("ny_pairs", True):
            sym_clean = symbol.replace(".", "").replace("-", "").upper()
            if any(p in sym_clean for p in NY_PAIRS):
                ny_bonus = 1

        # --- H4 EMA trend -------------------------------------------------
        h4_direction = None
        if elements.get("h4_trend", True):
            try:
                df_h4, _ = fetch(symbol, "H4", 60)
                if df_h4 is not None and len(df_h4) >= 50:
                    ema20 = float(df_h4["close"].ewm(span=20, adjust=False).mean().iloc[-1])
                    ema50 = float(df_h4["close"].ewm(span=50, adjust=False).mean().iloc[-1])
                    h4_direction = "BUY" if ema20 > ema50 else "SELL"
            except Exception as exc:
                log.debug(f"H4 EMA {symbol}: {exc}")

        # --- ATR zilnic ---------------------------------------------------
        atr_d1 = None
        try:
            df_d1, _ = fetch(symbol, "D1", 20)
            if df_d1 is not None and len(df_d1) >= 10:
                atr_d1 = float(
                    (df_d1["high"] - df_d1["low"]).rolling(14).mean().iloc[-1]
                )
        except Exception:
            pass

        # --- Analiza per TF -----------------------------------------------
        tf_results = []

        for tf in tfs:
            try:
                n_bars = (tf_bars or {}).get(tf, bars)
                df, _  = fetch(symbol, tf, n_bars)
                if df is None or len(df) < 50:
                    continue

                price = float(df["close"].iloc[-1])

                pre_df    = _extract_pre_ny_df(df)
                pre_high  = float(pre_df["high"].max())
                pre_low   = float(pre_df["low"].min())
                pre_range = pre_high - pre_low

                if pre_range <= 0:
                    continue

                reasons    = []
                conviction = session_penalty
                if session_penalty:
                    reasons.append(f"In afara sesiunii NY ({now_utc.strftime('%H:%M')} UTC) ({session_penalty})")

                reasons.append(
                    f"Range pre-NY: {pre_low:.5f} — {pre_high:.5f}"
                    f" ({pre_range:.5f})"
                )

                # ── P3: London Already Moved filter ──
                # Daca London a consumat deja >80% din ATR zilnic, miscarea e probabil finalizata
                if elements.get("london_moved_check", True) and atr_d1 and atr_d1 > 0:
                    today_range = float(df["high"].iloc[-24:].max()) - float(df["low"].iloc[-24:].min())
                    consumed_pct = today_range / atr_d1 if atr_d1 > 0 else 0
                    if consumed_pct > 0.80:
                        reasons.append(
                            f"London Already Moved: {consumed_pct*100:.0f}% din ATR zilnic consumat"
                            " (>80%) — miscare NY probabil limitata, skip"
                        )
                        tf_results.append({
                            "tf": tf, "signal": "HOLD", "conviction": 0,
                            "reasons": reasons, "price": price,
                            "sl": None, "tp": None,
                        })
                        continue
                    else:
                        reasons.append(
                            f"ATR disponibil: {consumed_pct*100:.0f}% consumat — spatiu miscare NY ✓"
                        )

                # Calitate range vs ATR
                if elements.get("pre_ny_range", True) and atr_d1 and atr_d1 > 0:
                    frac = pre_range / atr_d1
                    if frac < RANGE_MIN_ATR_FRAC:
                        reasons.append(
                            f"Range pre-NY prea ingust ({frac*100:.0f}%"
                            " din ATR zilnic) — skipped"
                        )
                        tf_results.append({
                            "tf": tf, "signal": "HOLD", "conviction": 0,
                            "reasons": reasons, "price": price,
                            "sl": None, "tp": None,
                        })
                        continue
                    elif frac > RANGE_MAX_ATR_FRAC:
                        reasons.append(
                            f"Range pre-NY prea larg ({frac*100:.0f}%"
                            " din ATR zilnic) — skipped"
                        )
                        tf_results.append({
                            "tf": tf, "signal": "HOLD", "conviction": 0,
                            "reasons": reasons, "price": price,
                            "sl": None, "tp": None,
                        })
                        continue
                    else:
                        reasons.append(
                            f"Calitate range: {frac*100:.0f}% din ATR zilnic — OK"
                        )
                        conviction += 1

                # Breakout
                buffer = pre_range * 0.10
                if price > pre_high + buffer:
                    sig            = "BUY"
                    breakout_level = pre_high
                    sl             = pre_low - buffer
                elif price < pre_low - buffer:
                    sig            = "SELL"
                    breakout_level = pre_low
                    sl             = pre_high + buffer
                else:
                    reasons.append(
                        f"Pretul ({price:.5f}) in interiorul range-ului pre-NY"
                    )
                    tf_results.append({
                        "tf": tf, "signal": "HOLD", "conviction": 0,
                        "reasons": reasons, "price": price,
                        "sl": None, "tp": None,
                    })
                    continue

                reasons.append(
                    f"Breakout {'sus' if sig == 'BUY' else 'jos'}: {price:.5f}"
                    f" {'>' if sig == 'BUY' else '<'} {breakout_level:.5f} + buffer"
                )
                conviction += 3

                # Displacement candle — OBLIGATORIU
                if elements.get("breakout", True):
                    disp_ok, disp_msg = _has_displacement(df, sig, lookback=3)
                    reasons.append(disp_msg)
                    if disp_ok:
                        conviction += 2
                    else:
                        tf_results.append({
                            "tf": tf, "signal": "HOLD", "conviction": 0,
                            "reasons": reasons, "price": price,
                            "sl": None, "tp": None,
                        })
                        continue

                # Retest (+3 conviction)
                if elements.get("retest", True):
                    ret_ok, ret_msg = _detect_retest(df, sig, breakout_level)
                    reasons.append(ret_msg)
                    if ret_ok:
                        conviction += 3

                # H4 trend alignment
                if elements.get("h4_trend", True) and h4_direction:
                    if h4_direction == sig:
                        reasons.append(
                            f"H4 EMA aliniat cu {sig} — confluenta trend"
                        )
                        conviction += 1
                    else:
                        reasons.append(
                            f"H4 EMA contrar ({h4_direction}) — contra-trend"
                        )
                        conviction -= 1

                # Time window bonus
                if time_note:
                    reasons.append(f"Time bonus: {time_note} (+{time_bonus})")
                conviction += time_bonus

                # NY pairs bonus
                if ny_bonus:
                    reasons.append(
                        f"Pereche USD majora — sesiune NY (+{ny_bonus})"
                    )
                    conviction += ny_bonus

                # TP: ATR D1 × 0.8 sau range × 1.5
                if atr_d1 and atr_d1 > 0:
                    tp_dist = atr_d1 * 0.8
                    reasons.append(f"TP bazat pe ATR D1 ({atr_d1:.5f} × 0.8)")
                else:
                    tp_dist = pre_range * 1.5
                    reasons.append("TP bazat pe range × 1.5 (ATR D1 indisponibil)")

                tp = price + tp_dist if sig == "BUY" else price - tp_dist

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
