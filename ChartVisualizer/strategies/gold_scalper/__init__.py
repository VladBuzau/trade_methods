"""
Gold Scalper — strategie dedicata XAUUSD pentru scalping intraday.

Optimizata pentru caracteristicile gold-ului:
  - Volatilitate mare → SL/TP dinamic ATR-based
  - Spread larg (~0.30) → filtru obligatoriu, evita piata moarta
  - Trendeaza puternic in sesiune US → 3 setup-uri complementare
  - Reactioneaza la news macro → fereastra activa concentrata London-NY

Fereastra activa: 12:00-20:00 UTC (overlap London-NY + sesiunea NY).

Trei setup-uri (oricare poate genera semnal):
  1. liquidity_sweep — wick depaseste high/low ultimele 24 bare M5 (~2h),
                       apoi lumanare reversal cu corp >50% → trade contra spike
  2. ema_pullback    — trend confirmat (M15 EMA20 > EMA50 sau invers),
                       retest M5 EMA20 cu touch + confirm candle → trade in trend
  3. range_bounce    — piata calma (ATR sub mediana, BB width sub p50):
                       BUY la BB lower band, SELL la BB upper band

SL/TP dinamic:
  - SL = 1.0 × ATR M5
  - TP = 1.5 × ATR M5 (RR 1.5)
  - Max durata trade: 4 ore (gold nu te tine mult)

Filtre obligatorii:
  - session_window: doar 12:00-20:00 UTC
  - news_blackout : skip 30 min inainte/dupa NFP, FOMC, CPI (ora-cheie 12:30, 14:00, 18:00 UTC)
  - atr_floor    : ATR M5 > 0.20 USD (evita piata moarta Asia)
  - atr_ceiling  : ATR M5 < 3.00 USD (evita spike-uri news, slippage masiv)

Frecventa asteptata: 3-6 trade-uri/zi pe gold in sesiune activa.
"""
from __future__ import annotations
import logging
from datetime import datetime, timezone

from strategies.base import Strategy

log = logging.getLogger(__name__)

# ── Sesiune activa ────────────────────────────────────────────────────────────
# Extinsa: de la London open (08:00) pana la final NY (22:00 UTC)
# Gold are volum decent de la 08:00 (London) nu doar de la 12:00
ACTIVE_OPEN_HOUR  = 8    # London open — gold tradeaza activ
ACTIVE_CLOSE_HOUR = 21   # NY close — lichiditate inca buna pana la 21

# ── Ferestre de news (UTC) — skip 15 min inainte si dupa (era 30 min) ─────────
# Reduse la ±15 min: nu vrem sa pierdem prea mult timp din sesiune
NEWS_WINDOWS_UTC = [
    (12.17, 12.75),  # 12:30 ± 15 min — NFP, CPI, PPI
    (13.75, 14.25),  # 14:00 ± 15 min — ISM PMI, JOLTS
    (17.75, 18.25),  # 18:00 ± 15 min — FOMC, Powell
]

# ── ATR thresholds (in USD per ounce) ─────────────────────────────────────────
ATR_FLOOR_USD   = 0.15   # relaxat: 15 cents (era 20) — gold mai volatil de noapte
ATR_CEILING_USD = 5.00   # relaxat: 5 USD (era 3) — spike-uri mari dar valide

# ── SL/TP multipliers ─────────────────────────────────────────────────────────
# RR 2.0:1 = breakeven la 33% WR (math sustenabila)
# Cu WR 40%: EV = 0.40×100$ - 0.60×50$ = 40-30 = +10$/trade
SL_ATR_MULT = 1.5
TP_ATR_MULT = 3.0   # era 1.8 → RR 1.2 (math negativa la 41% WR)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _atr(df, period: int = 14) -> float:
    """ATR(period) — ultima valoare."""
    import numpy as np
    if len(df) < period + 2:
        return 0.0
    high  = df["high"].values
    low   = df["low"].values
    close = df["close"].values
    tr = np.maximum(high[1:] - low[1:],
                    np.maximum(np.abs(high[1:] - close[:-1]),
                               np.abs(low[1:]  - close[:-1])))
    if len(tr) < period:
        return 0.0
    return float(np.mean(tr[-period:]))


def _ema(series, period: int) -> float:
    """EMA(period) — ultima valoare."""
    return float(series.ewm(span=period, adjust=False).mean().iloc[-1])


def _bb_bands(close_series, period: int = 20, mult: float = 2.0):
    """Bollinger Bands — returneaza (upper, middle, lower, width_now, width_p50)."""
    sma = close_series.rolling(period).mean()
    std = close_series.rolling(period).std()
    upper  = sma + mult * std
    lower  = sma - mult * std
    width  = (upper - lower) / sma
    if width.dropna().empty:
        return None
    return (
        float(upper.iloc[-1]),
        float(sma.iloc[-1]),
        float(lower.iloc[-1]),
        float(width.iloc[-1]),
        float(width.dropna().quantile(0.50)),
    )


def _in_news_blackout(hour_f: float) -> tuple[bool, str]:
    """True daca suntem in fereastra de blackout pentru news."""
    for start, end in NEWS_WINDOWS_UTC:
        if start <= hour_f < end:
            return True, f"News blackout {start:.1f}-{end:.1f} UTC (ora curenta {hour_f:.2f})"
    return False, ""


def _liquidity_sweep_signal(df_m5) -> tuple[str, str, int]:
    """
    Setup 1: Liquidity Sweep Reversal pe M5.
      - Wick depaseste high/low ultimele 24 bare M5 (~2h)
      - Lumanarea curenta inchide INAPOI in range cu corp >50%
    Returneaza: (signal, motiv, conviction_bonus).
    """
    if len(df_m5) < 30:
        return "HOLD", "Date M5 insuficiente pentru sweep", 0

    last = df_m5.iloc[-1]
    last_open  = float(last["open"])
    last_close = float(last["close"])
    last_high  = float(last["high"])
    last_low   = float(last["low"])
    last_range = last_high - last_low
    if last_range < 1e-9:
        return "HOLD", "Bara curenta fara range", 0

    body_pct = abs(last_close - last_open) / last_range

    prior = df_m5.iloc[-25:-1]
    prior_high = float(prior["high"].max())
    prior_low  = float(prior["low"].min())

    # Sweep sus → SELL (wick peste prior_high, dar close inapoi sub)
    if last_high > prior_high and last_close < prior_high and last_close < last_open:
        if body_pct >= 0.25:   # relaxat: 25% corp suficient (era 0.35)
            return "SELL", (
                f"Sweep SELL: wick peste {prior_high:.2f} ({last_high:.2f}), "
                f"close {last_close:.2f} sub varf, corp {body_pct*100:.0f}%"
            ), 10

    # Sweep jos → BUY (wick sub prior_low, dar close inapoi peste)
    if last_low < prior_low and last_close > prior_low and last_close > last_open:
        if body_pct >= 0.25:
            return "BUY", (
                f"Sweep BUY: wick sub {prior_low:.2f} ({last_low:.2f}), "
                f"close {last_close:.2f} peste suport, corp {body_pct*100:.0f}%"
            ), 10

    return "HOLD", "Fara sweep valid pe M5", 0


def _ema_pullback_signal(df_m5, df_m15, atr_m5: float) -> tuple[str, str, int]:
    """
    Setup 2: Pullback la EMA20 in trend M15.
      - M15 EMA20 vs EMA50 → directie trend
      - M5 SE APROPIE de EMA20 (in ±0.3 ATR) si lumanara confirma directia
    """
    if df_m15 is None or len(df_m15) < 60 or len(df_m5) < 30:
        return "HOLD", "Date insuficiente pentru EMA pullback", 0

    ema20_m15 = _ema(df_m15["close"], 20)
    ema50_m15 = _ema(df_m15["close"], 50)

    diff_pct = abs(ema20_m15 - ema50_m15) / ema50_m15 if ema50_m15 > 0 else 0
    if diff_pct < 0.0002:   # relaxat: 0.02% — acceptam si trend slab (era 0.05%)
        return "HOLD", f"EMA M15 prea apropiate ({diff_pct*100:.2f}%) — fara trend", 0

    trend_dir = "BUY" if ema20_m15 > ema50_m15 else "SELL"

    ema20_m5 = _ema(df_m5["close"], 20)
    last     = df_m5.iloc[-1]
    last_o   = float(last["open"])
    last_c   = float(last["close"])
    last_h   = float(last["high"])
    last_l   = float(last["low"])
    last_rng = last_h - last_l
    if last_rng < 1e-9:
        return "HOLD", "Bara curenta fara range", 0

    body_pct = abs(last_c - last_o) / last_rng

    # APROPIERE de EMA — acceptam pana la 0.6 ATR distanta (era 0.3)
    proximity = min(abs(last_h - ema20_m5), abs(last_l - ema20_m5),
                    abs(last_c - ema20_m5))
    near_ema = proximity <= atr_m5 * 0.6

    if not near_ema:
        return "HOLD", f"M5 prea departe de EMA20 ({proximity:.2f} > {atr_m5*0.6:.2f})", 0

    if trend_dir == "BUY" and last_c > last_o and body_pct >= 0.20:
        return "BUY", (
            f"Pullback BUY: M15 trend up, M5 langa EMA20 ({ema20_m5:.2f}), "
            f"confirm bullish corp {body_pct*100:.0f}%"
        ), 9

    if trend_dir == "SELL" and last_c < last_o and body_pct >= 0.20:
        return "SELL", (
            f"Pullback SELL: M15 trend down, M5 langa EMA20 ({ema20_m5:.2f}), "
            f"confirm bearish corp {body_pct*100:.0f}%"
        ), 9

    return "HOLD", "Apropiere EMA20 fara confirmare directie", 0


def _range_bounce_signal(df_m5) -> tuple[str, str, int]:
    """
    Setup 3: Range Bounce pe Bollinger Bands.
      - Piata calma (BB width sub mediana)
      - BUY daca close <= BB lower (cu lumanare bullish)
      - SELL daca close >= BB upper (cu lumanare bearish)
    """
    if len(df_m5) < 30:
        return "HOLD", "Date M5 insuficiente pentru range", 0

    bb = _bb_bands(df_m5["close"], 20, 2.0)
    if bb is None:
        return "HOLD", "BB nu se poate calcula", 0

    upper, middle, lower, w_now, w_p50 = bb

    # Filtru relaxat: pana la 2.0 × mediana — excludem doar trend-urile puternice
    w_threshold = w_p50 * 2.0
    if w_now > w_threshold:
        return "HOLD", (
            f"BB width {w_now:.4f} prea mare ({w_threshold:.4f}) — trend puternic"
        ), 0

    last = df_m5.iloc[-1]
    last_o = float(last["open"])
    last_c = float(last["close"])
    last_h = float(last["high"])
    last_l = float(last["low"])

    # Apropiere de BB lower/upper — 15% din banda (era 10%)
    band_width = upper - lower
    near_lower = (last_l - lower) <= band_width * 0.15
    near_upper = (upper - last_h) <= band_width * 0.15

    if near_lower and last_c > last_o:
        return "BUY", (
            f"Range BUY: low {last_l:.2f} langa BB lower {lower:.2f}, "
            f"close bullish"
        ), 9

    if near_upper and last_c < last_o:
        return "SELL", (
            f"Range SELL: high {last_h:.2f} langa BB upper {upper:.2f}, "
            f"close bearish"
        ), 9

    return "HOLD", "Pretul nu e langa BB extrem", 0


# ---------------------------------------------------------------------------
# Strategie
# ---------------------------------------------------------------------------

class GoldScalperStrategy(Strategy):
    key   = "gold_scalper"
    name  = "Gold Scalper"
    icon  = "🥇"
    color = "#ffd700"

    # ── Filtre simbol (rezerva XAUUSD pentru aceasta strategie) ──────────────
    symbol_whitelist  = ["XAU", "GOLD"]
    exclusive_symbols = True   # NICIO alta strategie nu mai ruleaza pe XAUUSD

    default_tfs            = ["M5", "M15"]
    default_bars           = 200
    default_step           = 1        # M5 — fiecare bara
    default_duration_years = 1.0      # 1 an Gold (XAUUSD volatil, mai mult istoric)
    default_max_hours      = 4.0      # 4h max
    elements = {
        "session_window":      "Activ doar 12:00-20:00 UTC (London-NY overlap + NY)",
        "news_blackout":       "Skip ferestre 12:30 / 14:00 / 18:00 UTC (NFP, FOMC, ISM)",
        "atr_filter":          "ATR M5 in [0.20, 3.00] USD (skip mort si spike)",
        "liquidity_sweep":     "Setup 1: wick peste prior 24h + reversal candle",
        "ema_pullback":        "Setup 2: M15 trend + retest M5 EMA20",
        "range_bounce":        "Setup 3: piata calma → BB extremes",
        "xauusd_only":         "STRICT: ruleaza DOAR pe XAUUSD",
        "session_strict":      "STRICT: doar 13-17 UTC (NY power hours)",
        "setup1_only":         "STRICT: doar liquidity sweep (cel mai precis)",
    }

    strategy_defaults = {
        "min_confidence": 55.0,        # mai permisiv — vrem frecventa
        "sl_atr_mult":    SL_ATR_MULT,
        "tp_atr_mult":    TP_ATR_MULT,
    }

    def analyze(self, symbol, tfs, bars=200, tf_bars=None, elements=None,
                min_confidence=60.0, **kwargs):
        from app import fetch

        if elements is None:
            elements = {k: True for k in self.elements}

        # ── Filtru simbol — strategia e doar pentru gold ──────────────────────
        sym_clean = symbol.replace(".", "").replace("-", "").upper()
        if "XAU" not in sym_clean and "GOLD" not in sym_clean:
            return self._empty_result(
                symbol, f"Strategie dedicata XAUUSD — {symbol} ignorat"
            )

        # ── Timpul curent (real sau backtest) ─────────────────────────────────
        import strategies as _strats
        now_utc = _strats.get_utc_now()
        hour_f  = now_utc.hour + now_utc.minute / 60.0

        # ── Session window gate ──────────────────────────────────────────────
        if elements.get("session_window", True):
            if not (ACTIVE_OPEN_HOUR <= now_utc.hour < ACTIVE_CLOSE_HOUR):
                return self._empty_result(
                    symbol,
                    f"In afara ferestrei active {ACTIVE_OPEN_HOUR:02d}:00-{ACTIVE_CLOSE_HOUR:02d}:00"
                    f" UTC (acum {now_utc.strftime('%H:%M')})"
                )

        # ── News blackout ────────────────────────────────────────────────────
        if elements.get("news_blackout", True):
            blocked, msg = _in_news_blackout(hour_f)
            if blocked:
                return self._empty_result(symbol, msg)

        # ── Fetch M5 si M15 (M15 doar pentru EMA pullback) ────────────────────
        try:
            df_m5, _ = fetch(symbol, "M5", (tf_bars or {}).get("M5", bars))
            df_m15, _ = fetch(symbol, "M15", (tf_bars or {}).get("M15", 100))
        except Exception as exc:
            log.warning(f"GoldScalper {symbol} fetch: {exc}")
            return self._empty_result(symbol, f"Fetch error: {exc}")

        if df_m5 is None or len(df_m5) < 30:
            return self._empty_result(symbol, "Date M5 insuficiente")

        price = float(df_m5["close"].iloc[-1])

        # ── ATR filter ───────────────────────────────────────────────────────
        atr = _atr(df_m5, 14)
        if elements.get("atr_filter", True):
            if atr < ATR_FLOOR_USD:
                return self._empty_result(
                    symbol, f"ATR M5 {atr:.3f} sub prag {ATR_FLOOR_USD} — piata moarta"
                )
            if atr > ATR_CEILING_USD:
                return self._empty_result(
                    symbol, f"ATR M5 {atr:.3f} peste prag {ATR_CEILING_USD} — spike news"
                )

        # ── Ruleaza cele 3 setup-uri si alege cel mai puternic ────────────────
        candidates: list[tuple[str, str, int, str]] = []  # (sig, msg, conv, setup_name)

        if elements.get("liquidity_sweep", True):
            sig, msg, conv = _liquidity_sweep_signal(df_m5)
            if sig in ("BUY", "SELL"):
                candidates.append((sig, msg, conv, "Liquidity Sweep"))

        if elements.get("ema_pullback", True):
            sig, msg, conv = _ema_pullback_signal(df_m5, df_m15, atr)
            if sig in ("BUY", "SELL"):
                candidates.append((sig, msg, conv, "EMA Pullback"))

        if elements.get("range_bounce", True):
            sig, msg, conv = _range_bounce_signal(df_m5)
            if sig in ("BUY", "SELL"):
                candidates.append((sig, msg, conv, "Range Bounce"))

        if not candidates:
            return self._empty_result(symbol, "Niciun setup activ in fereastra")

        # Daca avem setup-uri opuse, vot majoritar pe conviction
        buys  = [c for c in candidates if c[0] == "BUY"]
        sells = [c for c in candidates if c[0] == "SELL"]
        if buys and sells:
            buy_sum  = sum(c[2] for c in buys)
            sell_sum = sum(c[2] for c in sells)
            if buy_sum == sell_sum:
                return self._empty_result(symbol, "Setup-uri in conflict — egal, skip")
            chosen = buys if buy_sum > sell_sum else sells
        else:
            chosen = candidates

        # Cel mai puternic setup din directia castigatoare
        best = max(chosen, key=lambda c: c[2])
        sig, msg, conv, setup_name = best

        # Bonus daca ≥2 setup-uri converg pe aceeasi directie
        same_dir_count = sum(1 for c in candidates if c[0] == sig)
        if same_dir_count >= 2:
            conv = min(conv + 2, 13)   # cap la 13 pentru a evita confidence > 95%

        # ── SL/TP dinamic ATR-based ──────────────────────────────────────────
        sl_dist = atr * SL_ATR_MULT
        tp_dist = atr * TP_ATR_MULT

        if sig == "BUY":
            sl = price - sl_dist
            tp = price + tp_dist
        else:
            sl = price + sl_dist
            tp = price - tp_dist

        reasons = [
            f"Setup: {setup_name}",
            msg,
            f"ATR M5: {atr:.3f} USD — SL {sl_dist:.2f}, TP {tp_dist:.2f} (RR 1:{TP_ATR_MULT/SL_ATR_MULT:.1f})",
            f"Setup-uri converg: {same_dir_count}/{len(candidates)}",
        ]

        # Construim un singur tf_result (M5 e TF principal pentru scalper)
        tf_result = {
            "tf":         "M5",
            "signal":     sig,
            "conviction": conv,
            "reasons":    reasons,
            "price":      round(price, 2),
            "sl":         round(sl, 2),
            "tp":         round(tp, 2),
        }

        # Aliniem la formatul standard al strategiilor — 1 vot e suficient
        return self._build_result(
            symbol, [tf_result], min_confidence, min_votes=1
        )
