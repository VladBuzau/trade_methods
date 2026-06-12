"""
common_filters.py — biblioteca de filtre OPTIONALE reutilizabile pentru toate strategiile.

Fiecare filtru e o functie care primeste contextul de tradinng si returneaza
(pass: bool, reason: str). Strategiile le pot apela individual via base helper
_apply_optional_filters() — toate sunt OPT-IN prin elements dict.

Pattern:
  def filter_name(df, signal, **ctx) -> tuple[bool, str]:
      ...

ctx poate include: h4_dir, htf_df, atr, current_price, news_minutes_away, etc.
"""
from __future__ import annotations
from datetime import datetime, timezone
import numpy as np


# ─── Volume ──────────────────────────────────────────────────────────────────
def volume_above_avg(df, signal: str, *, period: int = 20,
                     mult: float = 1.2, **ctx) -> tuple[bool, str]:
    """Volum bara curenta vs medie 20-bare. Skip daca volume < mult × avg."""
    try:
        if "volume" not in df.columns or len(df) < period + 1:
            return True, "volume: indisponibil (skip filter)"
        cur = float(df["volume"].iloc[-1])
        avg = float(df["volume"].iloc[-period-1:-1].mean())
        if avg <= 0:
            return True, "volume: medie zero (skip filter)"
        ratio = cur / avg
        if ratio < mult:
            return False, f"volume slab ({ratio:.2f}× < {mult}×) — piata moarta"
        return True, f"volume OK ({ratio:.2f}× medie)"
    except Exception:
        return True, "volume filter exceptie, skip"


# ─── RSI Confluence ──────────────────────────────────────────────────────────
def _rsi(values, period: int = 14) -> float:
    if len(values) < period + 1:
        return 50.0
    deltas = np.diff(values)
    gain = np.where(deltas > 0, deltas, 0.0)
    loss = np.where(deltas < 0, -deltas, 0.0)
    avg_gain = float(np.mean(gain[-period:]))
    avg_loss = float(np.mean(loss[-period:]))
    if avg_loss == 0:
        return 100.0
    rs = avg_gain / avg_loss
    return 100.0 - (100.0 / (1.0 + rs))


def rsi_confluence(df, signal: str, *, period: int = 14,
                   buy_max: float = 65.0, sell_min: float = 35.0, **ctx) -> tuple[bool, str]:
    """
    BUY blocat daca RSI > 65 (deja suprasaturat sus → reversal probabil).
    SELL blocat daca RSI < 35 (deja oversold jos → bounce probabil).
    """
    try:
        rsi = _rsi(df["close"].values, period)
        if signal == "BUY" and rsi > buy_max:
            return False, f"RSI overbought ({rsi:.0f} > {buy_max:.0f}) — risc reversal"
        if signal == "SELL" and rsi < sell_min:
            return False, f"RSI oversold ({rsi:.0f} < {sell_min:.0f}) — risc bounce"
        return True, f"RSI OK ({rsi:.0f})"
    except Exception:
        return True, "RSI filter exceptie, skip"


# ─── EMA Trend Alignment ─────────────────────────────────────────────────────
def ema_trend_align(df, signal: str, *, period: int = 200, **ctx) -> tuple[bool, str]:
    """
    BUY permis doar daca pretul > EMA200 (trend bull).
    SELL permis doar daca pretul < EMA200 (trend bear).
    """
    try:
        if len(df) < period + 1:
            return True, "EMA200: date insuficiente (skip)"
        ema = df["close"].ewm(span=period, adjust=False).mean().iloc[-1]
        price = float(df["close"].iloc[-1])
        if signal == "BUY" and price < ema:
            return False, f"BUY contra-trend (preț {price:.5f} < EMA{period} {ema:.5f})"
        if signal == "SELL" and price > ema:
            return False, f"SELL contra-trend (preț {price:.5f} > EMA{period} {ema:.5f})"
        return True, f"EMA{period} aliniat"
    except Exception:
        return True, "EMA trend filter exceptie, skip"


# ─── HTF Alignment (H4) ──────────────────────────────────────────────────────
def htf_alignment(df, signal: str, *, h4_dir: str | None = None, **ctx) -> tuple[bool, str]:
    """
    Semnalul trebuie sa fie aliniat cu directia H4.
    h4_dir e calculat de app.get_h4_direction() si pasat in ctx.
    """
    if not h4_dir or h4_dir not in ("BULL", "BEAR"):
        return True, "H4 dir indisponibil — skip"
    if signal == "BUY" and h4_dir != "BULL":
        return False, f"H4 = {h4_dir} contra BUY"
    if signal == "SELL" and h4_dir != "BEAR":
        return False, f"H4 = {h4_dir} contra SELL"
    return True, f"H4 = {h4_dir} aliniat"


# ─── Momentum (acceleration, nu doar directie) ───────────────────────────────
def momentum_acceleration(df, signal: str, *, lookback: int = 5,
                          min_change_pct: float = 0.0, **ctx) -> tuple[bool, str]:
    """
    Cere ca ultimele N bare sa fi miscat in directia semnalului cu min_change_pct.
    Evita semnale in lateral / consolidare.
    """
    try:
        if len(df) < lookback + 1:
            return True, "momentum: date insuficiente"
        c = df["close"].values
        change = (c[-1] - c[-1 - lookback]) / c[-1 - lookback] * 100.0
        if signal == "BUY" and change < min_change_pct:
            return False, f"momentum slab pt BUY ({change:+.2f}% < {min_change_pct}%)"
        if signal == "SELL" and change > -min_change_pct:
            return False, f"momentum slab pt SELL ({change:+.2f}% > -{min_change_pct}%)"
        return True, f"momentum {change:+.2f}% pe {lookback} bare"
    except Exception:
        return True, "momentum filter exceptie, skip"


# ─── Swing Distance ──────────────────────────────────────────────────────────
def swing_distance_ok(df, signal: str, *, max_pct: float = 1.5, **ctx) -> tuple[bool, str]:
    """
    Pretul actual nu trebuie sa fie prea departe (>max_pct%) de:
      - high recent 20 bare (pentru SELL)
      - low recent 20 bare (pentru BUY)
    Evita intrari forate la varful unei miscari.
    """
    try:
        if len(df) < 20:
            return True, "swing dist: date insuficiente"
        price = float(df["close"].iloc[-1])
        if signal == "BUY":
            recent_low = float(df["low"].iloc[-20:].min())
            dist_pct = abs(price - recent_low) / price * 100
            if dist_pct > max_pct:
                return False, f"BUY prea departe de low ({dist_pct:.2f}% > {max_pct}%)"
            return True, f"BUY aproape de low ({dist_pct:.2f}%)"
        if signal == "SELL":
            recent_high = float(df["high"].iloc[-20:].max())
            dist_pct = abs(price - recent_high) / price * 100
            if dist_pct > max_pct:
                return False, f"SELL prea departe de high ({dist_pct:.2f}% > {max_pct}%)"
            return True, f"SELL aproape de high ({dist_pct:.2f}%)"
        return True, "swing dist: signal HOLD"
    except Exception:
        return True, "swing dist filter exceptie, skip"


# ─── News Proximity ──────────────────────────────────────────────────────────
def news_proximity(df, signal: str, *, max_minutes: int = 5,
                   news_minutes_away: float | None = None, **ctx) -> tuple[bool, str]:
    """Skip daca exista stire red in proximitate (±max_minutes)."""
    if news_minutes_away is None:
        return True, "news: nu am info (skip)"
    try:
        m = float(news_minutes_away)
        if abs(m) <= max_minutes:
            return False, f"news in {m:.0f} min — skip"
        return True, f"news in {m:.0f} min (OK)"
    except Exception:
        return True, "news filter exceptie, skip"


# ─── Weekend Close Pending ───────────────────────────────────────────────────
def weekend_close_pending(df, signal: str, *, hour_utc: int = 20, **ctx) -> tuple[bool, str]:
    """Vineri dupa hour_utc UTC nu mai intra (evita gap weekend)."""
    now = datetime.now(timezone.utc)
    if now.weekday() == 4 and now.hour >= hour_utc:  # vineri = 4
        return False, f"weekend close pending (vineri {now.hour:02d}:00 UTC)"
    return True, "weekend OK"


# ─── Time of Day ─────────────────────────────────────────────────────────────
def time_of_day_active(df, signal: str, **ctx) -> tuple[bool, str]:
    """Skip in ferestrele moarte: 22-01 UTC + Asia-only (01-06 UTC)."""
    hour = datetime.now(timezone.utc).hour
    if hour >= 22 or hour < 6:
        return False, f"in ora moarta ({hour:02d}:00 UTC)"
    return True, f"sesiune activa ({hour:02d}:00 UTC)"


# ─── Spread Acceptable ───────────────────────────────────────────────────────
def spread_acceptable(df, signal: str, *, current_spread_pips: float | None = None,
                      max_pips: float | None = None, **ctx) -> tuple[bool, str]:
    """Skip daca spread > max_pips. Daca nu primim datele, skip filter."""
    if current_spread_pips is None or max_pips is None:
        return True, "spread: nu am info"
    if current_spread_pips > max_pips:
        return False, f"spread {current_spread_pips:.1f} > {max_pips:.1f} pips"
    return True, f"spread OK ({current_spread_pips:.1f} pips)"


# ─── Body Strength (corp lumanare) ───────────────────────────────────────────
def body_strength(df, signal: str, *, min_body_pct: float = 0.4, **ctx) -> tuple[bool, str]:
    """Corp lumanare curenta >= min_body_pct din range — evita doji/spinning tops."""
    try:
        h = float(df["high"].iloc[-1])
        l = float(df["low"].iloc[-1])
        o = float(df["open"].iloc[-1])
        c = float(df["close"].iloc[-1])
        rng = h - l
        if rng <= 0:
            return False, "body: range zero"
        body = abs(c - o)
        ratio = body / rng
        if ratio < min_body_pct:
            return False, f"corp slab ({ratio*100:.0f}% < {min_body_pct*100:.0f}%) — doji"
        # Verifica si directia
        if signal == "BUY" and c < o:
            return False, "BUY pe lumanare bearish"
        if signal == "SELL" and c > o:
            return False, "SELL pe lumanare bullish"
        return True, f"corp {ratio*100:.0f}%"
    except Exception:
        return True, "body filter exceptie, skip"


# ─── Registry ────────────────────────────────────────────────────────────────
# {key: (label_ro, function, optional_default_params)}
OPTIONAL_FILTERS: dict[str, tuple[str, callable, dict]] = {
    "volume_above_avg":  ("Volum > 1.2× medie",          volume_above_avg,       {}),
    "rsi_confluence":    ("RSI nu suprasaturat",          rsi_confluence,         {}),
    "ema_trend_align":   ("Aliniat cu EMA200",            ema_trend_align,        {}),
    "htf_alignment":     ("Aliniat cu H4",                htf_alignment,          {}),
    "momentum_filter":   ("Momentum (5 bare)",            momentum_acceleration,  {}),
    "swing_distance":    ("Aproape de pivot/swing",       swing_distance_ok,      {}),
    "news_proximity":    ("Evită ±5 min știri red",       news_proximity,         {}),
    "weekend_close":     ("Skip vineri ≥20 UTC",          weekend_close_pending,  {}),
    "time_of_day":       ("Skip ore moarte (22-06 UTC)",  time_of_day_active,     {}),
    "spread_filter":     ("Spread acceptabil",            spread_acceptable,      {}),
    "body_strength":     ("Corp lumanare ≥ 40%",          body_strength,          {}),
}


def apply_all(df, signal: str, enabled_keys: set[str] | dict[str, bool] | None = None,
              **ctx) -> tuple[bool, list[str]]:
    """
    Aplica TOATE filtrele activate (din enabled_keys).
    Returneaza (pass_all: bool, reasons: list[str]).

    enabled_keys poate fi un set de chei sau un dict {key: bool}.
    """
    if signal not in ("BUY", "SELL"):
        return True, []

    if enabled_keys is None:
        return True, []
    if isinstance(enabled_keys, dict):
        active = {k for k, v in enabled_keys.items() if v}
    else:
        active = set(enabled_keys)

    reasons: list[str] = []
    for key in active:
        if key not in OPTIONAL_FILTERS:
            continue
        label, fn, defaults = OPTIONAL_FILTERS[key]
        kwargs = {**defaults, **ctx}
        try:
            ok, reason = fn(df, signal, **kwargs)
        except TypeError:
            # Functia nu accepta unele kwargs — incearca fara
            ok, reason = fn(df, signal)
        if not ok:
            reasons.append(f"❌ {label}: {reason}")
            return False, reasons  # short-circuit
        reasons.append(f"✓ {label}: {reason}")
    return True, reasons


def labels() -> dict[str, str]:
    """Returneaza {key: label} pentru UI."""
    return {k: v[0] for k, v in OPTIONAL_FILTERS.items()}
