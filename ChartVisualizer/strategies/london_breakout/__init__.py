"""
London Breakout v2: breakout din range-ul Asian Session (00:00-06:00 UTC)
la deschiderea sesiunii Londra (06:00-09:00 UTC).

Logica de intrare (in ordine):
  1. session_gate  — verificare fereastra activa
  2. asian_range   — calitate range (nu prea mic, nu prea mare vs ATR zilnic)
  3. breakout      — pret depaseste range + displacement candle obligatoriu
  4. retest        — optional, detecteaza retest la nivel (conviction bonus)
  5. h4_trend      — aliniere EMA20/EMA50 pe H4 (conviction bonus/minus)
  6. time_window   — bonus conviction 06:00-07:30 UTC (prime time)
  7. day_of_week   — scoring pe zi: Luni -2, Vineri -1, Mar-Joi +1 (P3)

Functioneaza cel mai bine pe: GBPUSD, EURUSD, GBPJPY, XAUUSD.
"""
from __future__ import annotations
import logging
from datetime import datetime, timezone

from strategies.base import Strategy

log = logging.getLogger(__name__)

LONDON_OPEN_HOUR  = 6
LONDON_CLOSE_HOUR = 9
ASIAN_START_HOUR  = 0
ASIAN_END_HOUR    = 6

# Calitate range Asian exprimata ca fractie din ATR zilnic
RANGE_MIN_ATR_FRAC = 0.15   # < 15% din ATR = prea ingust (fakeout probabil)
RANGE_MAX_ATR_FRAC = 1.20   # > 120% din ATR = prea volatil (stop-uri uriase)


# ---------------------------------------------------------------------------
# Functii helper
# ---------------------------------------------------------------------------

def _extract_asian_df(df):
    """Returneaza barele din sesiunea Asiatica (00:00-06:00 UTC)."""
    try:
        if "time" in df.columns:
            ts = df["time"]
        elif df.index.name == "time":
            ts = df.index.to_series()
        else:
            return df.iloc[-24:]
        ts = ts.apply(
            lambda x: x if hasattr(x, "hour")
            else datetime.fromtimestamp(float(x), tz=timezone.utc)
        )
        mask   = ts.apply(lambda x: ASIAN_START_HOUR <= x.hour < ASIAN_END_HOUR)
        result = df[mask]
        return result if len(result) >= 3 else df.iloc[-24:]
    except Exception:
        return df.iloc[-24:]


def _has_displacement(df, direction: str, lookback: int = 3,
                      body_pct: float = 0.55) -> tuple[bool, str]:
    """
    Verifica daca in ultimele `lookback` bare exista o lumanare
    cu corp >= body_pct din range in directia data.
    """
    if df is None or len(df) < lookback + 1:
        return False, "Date insuficiente pentru displacement"

    opens  = df["open"].values
    closes = df["close"].values
    highs  = df["high"].values
    lows   = df["low"].values

    for i in range(-lookback, 0):
        rng  = highs[i] - lows[i]
        if rng < 1e-10:
            continue
        body  = abs(closes[i] - opens[i])
        ratio = body / rng
        if ratio < body_pct:
            continue
        is_bull = closes[i] > opens[i]
        is_bear = closes[i] < opens[i]
        if direction == "BUY" and is_bull:
            return True, f"Displacement BUY: corp {ratio*100:.1f}% din range (bar {i})"
        if direction == "SELL" and is_bear:
            return True, f"Displacement SELL: corp {ratio*100:.1f}% din range (bar {i})"

    return False, f"Niciun displacement {direction} in ultimele {lookback} lumanari"


def _detect_retest(df, direction: str, level: float,
                   tol_frac: float = 0.15) -> tuple[bool, str]:
    """
    Detecteaza un pattern break-and-retest pe df:
      BUY:  ... close > level (breakout) ... low aproape de level (retest) ... close > level (bounce)
      SELL: ... close < level (breakout) ... high aproape de level (retest) ... close < level (bounce)

    Returneaza (True, msg) daca pattern-ul e prezent in ultimele 30 bare.
    """
    if df is None or len(df) < 8:
        return False, "Date insuficiente pentru retest"

    closes = df["close"].values
    highs  = df["high"].values
    lows   = df["low"].values

    window    = min(30, len(df))
    recent_rng = highs[-window:].max() - lows[-window:].min()
    tolerance  = max(recent_rng * tol_frac, 1e-8)

    if direction == "BUY":
        # 1. Gasim primul bar din ultimele 30 care a inchis deasupra level-ului
        bk_idx = None
        for i in range(-window, -3):
            if closes[i] > level:
                bk_idx = i
                break
        if bk_idx is None:
            return False, "Retest BUY: nu s-a detectat breakout anterior"

        # 2. Dupa acel bar, un low a atins nivelul (retest)
        retest_ok = False
        for i in range(bk_idx + 1, 0):
            if lows[i] <= level + tolerance:
                retest_ok = True
                break
        if not retest_ok:
            return False, "Retest BUY: pretul nu a revenit la nivel dupa breakout"

        # 3. Pretul curent e din nou deasupra level-ului
        if closes[-1] > level:
            return True, f"Retest BUY confirmat la {level:.5f} — bounce din nivel"
        return False, "Retest BUY: pretul nu a recuperat dupa retest"

    else:  # SELL
        bk_idx = None
        for i in range(-window, -3):
            if closes[i] < level:
                bk_idx = i
                break
        if bk_idx is None:
            return False, "Retest SELL: nu s-a detectat breakout anterior"

        retest_ok = False
        for i in range(bk_idx + 1, 0):
            if highs[i] >= level - tolerance:
                retest_ok = True
                break
        if not retest_ok:
            return False, "Retest SELL: pretul nu a revenit la nivel dupa breakout"

        if closes[-1] < level:
            return True, f"Retest SELL confirmat la {level:.5f} — respingere din nivel"
        return False, "Retest SELL: pretul nu a recuperat dupa retest"


# ---------------------------------------------------------------------------
# Strategie
# ---------------------------------------------------------------------------

class LondonBreakoutStrategy(Strategy):
    key   = "london_breakout"
    name  = "London Breakout"
    icon  = "🇬🇧"
    color = "#ef5350"

    default_tfs  = ["M5", "M15"]
    default_bars = 200
    elements     = {
        "session_gate":   "Activ doar in sesiunea Londra (06-09 UTC)",
        "asian_range":    "Calitate range Asian vs ATR zilnic",
        "breakout":       "Breakout + displacement candle (obligatoriu)",
        "retest":         "Retest la nivel breakout (+3 conviction)",
        "h4_trend":       "Aliniere EMA20/EMA50 pe H4 (+/-1 conviction)",
        "time_window":    "Bonus conviction prime-time 06:00-07:30 UTC",
        "day_of_week":    "Scoring zi saptamana: Luni -2, Vineri -1, Mar-Joi +1",
        "retest_only":    "STRICT: doar trade dupa retest (anti fake breakout)",
        "h4_strict":      "STRICT: H4 trend OBLIGATORIU aliniat (nu doar bonus)",
        "skip_mon_fri":   "STRICT: skip Luni + Vineri (prea volatil/iesire pozitii)",
    }

    def analyze(self, symbol, tfs, bars=200, tf_bars=None, elements=None,
                min_confidence=66.0, **kwargs):
        from app import fetch

        if elements is None:
            elements = {k: True for k in self.elements}

        import strategies as _strats
        now_utc = _strats.get_utc_now()
        hour_f  = now_utc.hour + now_utc.minute / 60.0
        in_london = LONDON_OPEN_HOUR <= now_utc.hour < LONDON_CLOSE_HOUR

        # --- Session gate — scoring, nu blocare ----------------------------
        session_penalty = 0
        if elements.get("session_gate", True) and not in_london:
            session_penalty = -3  # penalizare mare, dar nu blocheaza

        # --- P3: Day-of-week scoring --------------------------------------
        # 0=Mon, 1=Tue, 2=Wed, 3=Thu, 4=Fri, 5=Sat, 6=Sun
        dow_bonus = 0
        dow_note  = ""
        if elements.get("day_of_week", True):
            dow = now_utc.weekday()
            if dow == 0:
                dow_bonus = -2
                dow_note  = "Luni — gap deschidere, directie neclara (-2)"
            elif dow == 4:
                dow_bonus = -1
                dow_note  = "Vineri — position squaring, risc inchideri (-1)"
            elif dow in (1, 2, 3):
                dow_bonus = 1
                dow_note  = f"{'Marti' if dow==1 else 'Miercuri' if dow==2 else 'Joi'} — lichiditate maxima (+1)"
            elif dow >= 5:
                dow_bonus = -3
                dow_note  = "Weekend — piata inchisa"
        # STRICT: skip Mon/Fri entirely
        if elements.get("skip_mon_fri", False) and now_utc.weekday() in (0, 4):
            return self._empty_result(symbol, f"skip_mon_fri: zi {now_utc.weekday()} skip")

        # --- Time window bonus (calculat o singura data) ------------------
        if elements.get("time_window", True):
            if 6.0 <= hour_f < 7.5:
                time_bonus = 2
                time_note  = "prime time London 06:00-07:30 UTC"
            elif 7.5 <= hour_f < 8.5:
                time_bonus = 1
                time_note  = "sesiune activa 07:30-08:30 UTC"
            else:
                time_bonus = 0
                time_note  = "fereastra tarzie Londra 08:30+ UTC"
        else:
            time_bonus = 0
            time_note  = ""

        # --- H4 EMA trend (o singura data, nu per TF) ---------------------
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

        # --- ATR zilnic (pentru calitate range si TP) ---------------------
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

                # Range Asian
                asian_df    = _extract_asian_df(df)
                asian_high  = float(asian_df["high"].max())
                asian_low   = float(asian_df["low"].min())
                asian_range = asian_high - asian_low

                if asian_range <= 0:
                    continue

                reasons    = []
                conviction = session_penalty
                if session_penalty:
                    reasons.append(f"In afara sesiunii Londra ({now_utc.strftime('%H:%M')} UTC) ({session_penalty})")

                reasons.append(
                    f"Asian range: {asian_low:.5f} — {asian_high:.5f} "
                    f"({asian_range:.5f})"
                )

                # Calitate range vs ATR zilnic
                if elements.get("asian_range", True) and atr_d1 and atr_d1 > 0:
                    frac = asian_range / atr_d1
                    if frac < RANGE_MIN_ATR_FRAC:
                        reasons.append(
                            f"Range Asian prea ingust ({frac*100:.0f}% din ATR zilnic)"
                            " — skipped"
                        )
                        tf_results.append({
                            "tf": tf, "signal": "HOLD", "conviction": 0,
                            "reasons": reasons, "price": price,
                            "sl": None, "tp": None,
                        })
                        continue
                    elif frac > RANGE_MAX_ATR_FRAC:
                        reasons.append(
                            f"Range Asian prea larg ({frac*100:.0f}% din ATR zilnic)"
                            " — skipped"
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

                # Breakout detection
                buffer = asian_range * 0.10
                if price > asian_high + buffer:
                    sig             = "BUY"
                    breakout_level  = asian_high
                    sl              = asian_low - buffer
                elif price < asian_low - buffer:
                    sig             = "SELL"
                    breakout_level  = asian_low
                    sl              = asian_high + buffer
                else:
                    reasons.append(
                        f"Pretul ({price:.5f}) in interiorul range-ului — fara breakout"
                    )
                    tf_results.append({
                        "tf": tf, "signal": "HOLD", "conviction": 0,
                        "reasons": reasons, "price": price,
                        "sl": None, "tp": None,
                    })
                    continue

                dir_label = "sus" if sig == "BUY" else "jos"
                reasons.append(
                    f"Breakout {dir_label}: {price:.5f} "
                    f"{'>' if sig=='BUY' else '<'} {breakout_level:.5f} + buffer"
                )
                conviction += 3

                # Displacement candle — OBLIGATORIU
                if elements.get("breakout", True):
                    disp_ok, disp_msg = _has_displacement(df, sig, lookback=3)
                    reasons.append(disp_msg)
                    if disp_ok:
                        conviction += 2
                    else:
                        # Fara displacement = fakeout potential, nu intra
                        tf_results.append({
                            "tf": tf, "signal": "HOLD", "conviction": 0,
                            "reasons": reasons, "price": price,
                            "sl": None, "tp": None,
                        })
                        continue

                # Retest (optional, +3 conviction)
                ret_ok = False
                if elements.get("retest", True):
                    ret_ok, ret_msg = _detect_retest(df, sig, breakout_level)
                    reasons.append(ret_msg)
                    if ret_ok:
                        conviction += 3

                # STRICT: retest_only → skip daca nu am retest
                if elements.get("retest_only", False) and not ret_ok:
                    reasons.append("retest_only ON: niciun retest detectat — HOLD")
                    tf_results.append({"tf": tf, "signal": "HOLD", "conviction": 0,
                                       "reasons": reasons, "price": price,
                                       "sl": None, "tp": None})
                    continue

                # H4 trend alignment
                h4_match = False
                if elements.get("h4_trend", True) and h4_direction:
                    if h4_direction == sig:
                        reasons.append(f"H4 EMA aliniat cu {sig} — confluenta trend")
                        conviction += 1
                        h4_match = True
                    else:
                        reasons.append(f"H4 EMA contrar ({h4_direction}) — contra-trend, risc crescut")
                        conviction -= 1

                # STRICT: h4_strict → BLOCHEAZA daca H4 nu confirma
                if elements.get("h4_strict", False):
                    if not h4_match:
                        reasons.append("h4_strict ON: H4 nu aliniat — HOLD")
                        tf_results.append({"tf": tf, "signal": "HOLD", "conviction": 0,
                                           "reasons": reasons, "price": price,
                                           "sl": None, "tp": None})
                        continue

                # Time window bonus
                if time_note:
                    reasons.append(f"Time bonus: {time_note} (+{time_bonus})")
                conviction += time_bonus

                # Day-of-week scoring
                if dow_note:
                    reasons.append(f"Zi: {dow_note}")
                conviction += dow_bonus

                # TP: ATR zilnic × 0.8 daca disponibil, altfel range × 1.5
                if atr_d1 and atr_d1 > 0:
                    tp_dist = atr_d1 * 0.8
                    reasons.append(f"TP bazat pe ATR D1 ({atr_d1:.5f} × 0.8)")
                else:
                    tp_dist = asian_range * 1.5
                    reasons.append("TP bazat pe Asian range × 1.5 (ATR D1 indisponibil)")

                if sig == "BUY":
                    tp = price + tp_dist
                else:
                    tp = price - tp_dist

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
