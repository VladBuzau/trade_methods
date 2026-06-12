"""
China Session Strategy v2: mean reversion in sesiunea Shanghai.
Activ: 01:00-07:00 UTC (09:00-15:00 CST).

Caracteristica sesiunii asiatice: consolidare si range mic,
pretul oscileaza intre suport si rezistenta fara trend puternic.

Logica:
  1. session_gate  — fereastra 01-07 UTC
  2. session_range — detectie ranging vs trending (via ATR)
                     daca sesiunea e trending → HOLD (mean reversion periculoasa)
  3. rsi_extreme   — RSI > 65 la top range (SELL) sau < 35 la bottom (BUY)
  4. reversal      — lumanare de reversal obligatorie la extrem
  5. china_pairs   — bonus conviction pentru XAUUSD, AUDUSD, NZDUSD, USDJPY
  6. bb_width      — Bollinger Width confirmare: width mica = range valid, mare = skip (P3)
  7. escape_exit   — Breakout Escape: daca pretul iese din range cu displacement → HOLD (P3)

Perechi recomandate: XAUUSD, AUDUSD, NZDUSD, USDJPY, USDCNH
"""
from __future__ import annotations
import logging
from datetime import datetime, timezone

from strategies.base import Strategy

log = logging.getLogger(__name__)

CHINA_OPEN_HOUR  = 1   # 01:00 UTC = 09:00 CST
CHINA_CLOSE_HOUR = 7   # 07:00 UTC = 15:00 CST

# Perechi cu expunere directa la sesiunea chineza
CHINA_PAIRS = {
    "XAUUSD", "AUDUSD", "NZDUSD", "USDJPY",
    "USDCNH", "AUDNZD", "AUDJPY", "NZDJPY",
    "AUDCAD", "AUDCHF",
}

# recent_range / ATR >= acest prag → sesiune trending, nu range
TRENDING_THRESHOLD = 0.50

# "langa extrem" = in aceasta fractie din range de la margine
EXTREME_ZONE_PCT = 0.20


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _rsi(series, period: int = 14):
    delta = series.diff()
    gain  = delta.clip(lower=0).rolling(period).mean()
    loss  = (-delta.clip(upper=0)).rolling(period).mean()
    rs    = gain / loss.replace(0, float("nan"))
    return 100 - (100 / (1 + rs))


def _reversal_candle(df, direction: str, lookback: int = 2,
                     min_body_pct: float = 0.40) -> tuple[bool, str]:
    """
    Cauta o lumanare de reversal in ultimele `lookback` bare:
      BUY  reversal: corp bullish (close > open), min_body_pct din range
      SELL reversal: corp bearish (close < open), min_body_pct din range
    """
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
        if ratio < min_body_pct:
            continue
        if direction == "BUY" and closes[i] > opens[i]:
            return True, f"Reversal bullish: corp {ratio*100:.1f}% din range (bar {i})"
        if direction == "SELL" and closes[i] < opens[i]:
            return True, f"Reversal bearish: corp {ratio*100:.1f}% din range (bar {i})"

    return False, f"Nicio lumanare de reversal {direction} in ultimele {lookback} bare"


# ---------------------------------------------------------------------------
# Strategie
# ---------------------------------------------------------------------------

class ChinaSessionStrategy(Strategy):
    key   = "china_session"
    name  = "China Session"
    icon  = "🇨🇳"
    color = "#ff6f00"

    default_tfs  = ["M5", "M15"]
    default_bars = 150
    elements     = {
        "session_gate":     "Activ doar in sesiunea Shanghai (01-07 UTC)",
        "session_range":    "Detectie ranging vs trending — skip daca trending",
        "rsi_extreme":      "RSI < 35 (BUY) sau > 65 (SELL) la extrem range",
        "reversal":         "Lumanare de reversal obligatorie (gate)",
        "china_pairs":      "Bonus conviction perechi legate de China (+1)",
        "bb_width":         "BB Width confirmare: benzi inguste = range valid, late = skip",
        "escape_exit":      "Breakout Escape: displacement candle in afara range → skip",
        "rsi_strict":       "STRICT: RSI < 30 / > 70 (vs default 35/65)",
        "china_pairs_only": "STRICT: doar AUD/USD, NZD/USD, AUDJPY, copper/gold",
    }

    def analyze(self, symbol, tfs, bars=150, tf_bars=None, elements=None,
                min_confidence=60.0, **kwargs):
        from app import fetch

        if elements is None:
            elements = {k: True for k in self.elements}

        import strategies as _strats
        now_utc  = _strats.get_utc_now()
        in_china = CHINA_OPEN_HOUR <= now_utc.hour < CHINA_CLOSE_HOUR

        # --- Session gate — scoring -------------------------------------------
        session_penalty = 0
        if elements.get("session_gate", True) and not in_china:
            session_penalty = -3

        # --- China pairs bonus (calculat o singura data) ------------------
        china_bonus = 0
        if elements.get("china_pairs", True):
            sym_clean = symbol.replace(".", "").replace("-", "").upper()
            if any(p in sym_clean for p in CHINA_PAIRS):
                china_bonus = 1

        # --- Analiza per TF -----------------------------------------------
        tf_results = []

        for tf in tfs:
            try:
                n_bars = (tf_bars or {}).get(tf, bars)
                df, _  = fetch(symbol, tf, n_bars)
                if df is None or len(df) < 30:
                    continue

                price = float(df["close"].iloc[-1])

                # ATR(14)
                atr = float(
                    (df["high"] - df["low"]).rolling(14).mean().iloc[-1]
                )
                if atr <= 0:
                    continue

                # Range sesiune: ultimele 8 bare (~2h pe M15, ~8h pe H1)
                n_rng        = min(8, len(df))
                sess_high    = float(df["high"].iloc[-n_rng:].max())
                sess_low     = float(df["low"].iloc[-n_rng:].min())
                sess_range   = sess_high - sess_low

                reasons    = []
                conviction = session_penalty
                if session_penalty:
                    reasons.append(f"In afara sesiunii China ({now_utc.strftime('%H:%M')} UTC) ({session_penalty})")

                reasons.append(
                    f"Range sesiune: {sess_low:.5f} — {sess_high:.5f}"
                    f" ({sess_range:.5f})"
                )

                # ── P3: Breakout Escape filter ──
                # Daca o lumanare recenta a iesit cu displacement din range → nu mai e range
                if elements.get("escape_exit", True):
                    escape = False
                    for i in range(-3, 0):
                        bar_open  = float(df["open"].iloc[i])
                        bar_close = float(df["close"].iloc[i])
                        bar_high  = float(df["high"].iloc[i])
                        bar_low   = float(df["low"].iloc[i])
                        bar_range = bar_high - bar_low
                        if bar_range < 1e-10:
                            continue
                        body_pct = abs(bar_close - bar_open) / bar_range
                        # Displacement candle iesind din range = escape
                        if body_pct >= 0.60:
                            if bar_close > sess_high or bar_close < sess_low:
                                escape = True
                                reasons.append(
                                    f"Breakout Escape: displacement ({body_pct*100:.0f}%)"
                                    " iese din range — mean reversion invalida, skip"
                                )
                                break
                    if escape:
                        tf_results.append({
                            "tf": tf, "signal": "HOLD", "conviction": 0,
                            "reasons": reasons, "price": price,
                            "sl": None, "tp": None,
                        })
                        continue

                # ── P3: BB Width confirmare ──
                if elements.get("bb_width", True):
                    try:
                        sma20    = df["close"].rolling(20).mean()
                        std20    = df["close"].rolling(20).std()
                        bb_upper = sma20 + 2 * std20
                        bb_lower = sma20 - 2 * std20
                        bbw_series = ((bb_upper - bb_lower) / sma20).dropna()
                        if len(bbw_series) >= 20:
                            bbw_now = float(bbw_series.iloc[-1])
                            bbw_p50 = float(bbw_series.quantile(0.50))
                            bbw_p80 = float(bbw_series.quantile(0.80))
                            if bbw_now > bbw_p80:
                                reasons.append(
                                    f"BB Width {bbw_now:.4f} > percentila 80 ({bbw_p80:.4f})"
                                    " — benzi late, piata in trend, skip"
                                )
                                tf_results.append({
                                    "tf": tf, "signal": "HOLD", "conviction": 0,
                                    "reasons": reasons, "price": price,
                                    "sl": None, "tp": None,
                                })
                                continue
                            elif bbw_now < bbw_p50:
                                reasons.append(
                                    f"BB Width {bbw_now:.4f} — benzi inguste, range confirmat ✓"
                                )
                                conviction += 1
                    except Exception:
                        pass

                # Ranging vs trending
                if elements.get("session_range", True):
                    range_frac = sess_range / atr if atr > 0 else 1.0
                    if range_frac >= TRENDING_THRESHOLD:
                        reasons.append(
                            f"Sesiune TRENDING (range {range_frac*100:.0f}%"
                            " din ATR) — mean reversion riscanta, skip"
                        )
                        tf_results.append({
                            "tf": tf, "signal": "HOLD", "conviction": 0,
                            "reasons": reasons, "price": price,
                            "sl": None, "tp": None,
                        })
                        continue
                    else:
                        reasons.append(
                            f"Sesiune RANGING (range {range_frac*100:.0f}%"
                            " din ATR) — mean reversion valid"
                        )
                        conviction += 1

                # Zona extrema
                zone        = sess_range * EXTREME_ZONE_PCT
                near_top    = price >= sess_high - zone
                near_bottom = price <= sess_low  + zone

                if not near_top and not near_bottom:
                    reasons.append(
                        f"Pret ({price:.5f}) in zona medie — astept extrem"
                    )
                    tf_results.append({
                        "tf": tf, "signal": "HOLD", "conviction": 0,
                        "reasons": reasons, "price": price,
                        "sl": None, "tp": None,
                    })
                    continue

                sig = "SELL" if near_top else "BUY"
                reasons.append(
                    f"Pret la {'top' if sig == 'SELL' else 'bottom'} range"
                    f" ({price:.5f})"
                )
                conviction += 2

                # RSI extrema
                if elements.get("rsi_extreme", True):
                    rsi_val = float(_rsi(df["close"], 14).iloc[-1])
                    if sig == "SELL":
                        if rsi_val > 65:
                            reasons.append(
                                f"RSI overbought: {rsi_val:.1f} — confirmare SELL"
                            )
                            conviction += 2
                        elif rsi_val > 55:
                            reasons.append(
                                f"RSI ridicat: {rsi_val:.1f} — semnal slab"
                            )
                            conviction += 1
                        else:
                            reasons.append(
                                f"RSI neutru: {rsi_val:.1f} — fara confirmare RSI"
                            )
                    else:  # BUY
                        if rsi_val < 35:
                            reasons.append(
                                f"RSI oversold: {rsi_val:.1f} — confirmare BUY"
                            )
                            conviction += 2
                        elif rsi_val < 45:
                            reasons.append(
                                f"RSI scazut: {rsi_val:.1f} — semnal slab"
                            )
                            conviction += 1
                        else:
                            reasons.append(
                                f"RSI neutru: {rsi_val:.1f} — fara confirmare RSI"
                            )

                # Reversal candle — OBLIGATORIU
                if elements.get("reversal", True):
                    rev_ok, rev_msg = _reversal_candle(df, sig, lookback=2)
                    reasons.append(rev_msg)
                    if rev_ok:
                        conviction += 2
                    else:
                        tf_results.append({
                            "tf": tf, "signal": "HOLD", "conviction": 0,
                            "reasons": reasons, "price": price,
                            "sl": None, "tp": None,
                        })
                        continue

                # China pairs bonus
                if china_bonus:
                    reasons.append(
                        f"Pereche influientata de sesiunea chineza (+{china_bonus})"
                    )
                    conviction += china_bonus

                # SL/TP
                buf = sess_range * 0.05
                if sig == "SELL":
                    sl = sess_high + buf
                    tp = sess_low  + buf   # TP la bottom range
                else:
                    sl = sess_low  - buf
                    tp = sess_high - buf   # TP la top range

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
                log.warning(f"ChinaSessionStrategy {symbol}/{tf}: {exc}")

        if not tf_results:
            return self._empty_result(symbol, "Fara date suficiente")

        return self._build_result(symbol, tf_results, min_confidence, min_votes=1)
