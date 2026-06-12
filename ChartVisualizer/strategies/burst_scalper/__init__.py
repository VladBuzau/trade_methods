"""
Burst Scalper — strategie de scalping rapid pe TF mic (default M1).

Bazata pe combinatia clasica folosita in bot-urile comerciale de scalping
(ScalperX EA, Hamster Scalping, MAXI Pro, multe Pine Script):
  - EMA(5) vs EMA(20) → directie micro-trend
  - Stochastic (5,3,3) → timing entry (cross %K x %D in zona extrema)
  - RSI(7) → confirmare momentum (>50 BUY, <50 SELL)
  - ATR(14) → SL/TP dinamic + filtru piata moarta

Trade duration: 1-5 bare pe TF-ul ales (M1 = 1-5 minute).
TP/SL: tight, RR 1.5:1 (TP > SL).

CONFIGURARE:
  - TF: configurabil (M1 default; M2, M5 alternative)
  - min_confidence: 55-70 recomandat
  - sl_atr_mult: 1.0 (SL la 1×ATR)
  - tp_atr_mult: 1.5 (TP la 1.5×ATR)
  - max_bars_hold: 5 (force exit dupa 5 bare daca nu hit TP/SL)

ENTRY RULES:

  BUY:
    1. EMA(5) > EMA(20)                      [trend up]
    2. Stoch %K cross above %D in ultimele 2 bare, %K era < 30 [oversold reversal]
    3. RSI(7) > 50 si crescator               [momentum bullish]
    4. ATR(14) > 0.5 × ATR(50) media            [piata vie]
    5. Bara curenta close > open               [confirmare candle]

  SELL: oglinda exact (toate conditiile inverse)

EXIT RULES:
  - SL: 1.0 × ATR(14)
  - TP: 1.5 × ATR(14) → RR 1.5
  - Max 5 bare hold (configurabil) — daca nu hit TP/SL, force exit
"""
from __future__ import annotations
import logging

import numpy as np
import pandas as pd

from strategies.base import Strategy

log = logging.getLogger(__name__)


# ── Defaults v3 PRO LINEAR ────────────────────────────────────────────────────
# Filozofie: A+ setups only, exit rapid, single trade, no revenge.
DEFAULT_TF        = "M1"
EMA_FAST          = 5
EMA_SLOW          = 20
STOCH_K           = 5
STOCH_D           = 3
STOCH_SMOOTH      = 3
RSI_PERIOD        = 7
ATR_PERIOD        = 14
ATR_AVG_PERIOD    = 50

# ── Exit rapid (cei mai buni trader-i: profit rapid, stop disciplinat) ────────
SL_ATR_MULT       = 0.5    # SL super tight (era 1.0)
TP_ATR_MULT       = 0.7    # TP rapid (era 2.0) → RR 1.4 dar exit in <2 bare
MAX_BARS_HOLD     = 2      # max 2 bare = max 2 minute pe M1 (era 5)

# ── Stochastic extreme zones (only A+ entries) ────────────────────────────────
STOCH_OVERSOLD    = 20.0   # era 30 — extreme reversal only
STOCH_OVERBOUGHT  = 80.0   # era 70

# ── QUALITY-FIRST: intra doar la setup-uri A+ certe ──────────────────────────
# Filozofia: NU urmarim un numar de trade-uri/zi. Acceptam zile cu 0 trade-uri.
# Rather wait for conviction max decat sa fortam intrari mediocre.
ADX_PERIOD        = 14
ADX_MIN           = 25.0   # strong trend confirmat (DI+/DI- spread mare)
HTF_TIMEFRAME     = "M15"
HTF_EMA_FAST      = 20
HTF_EMA_SLOW      = 50
VOLUME_SPIKE_MULT = 1.5    # institutional confirm puternic (>50% peste medie)
COOLDOWN_BARS     = 5      # cooldown standard
LOSS_COOLDOWN     = 15     # no revenge — 15 min dupa loss
MIN_SCORE         = 10     # 10/11 — A+ ONLY (era 9, era 8)

# ── ANTI-FOMO FILTERS — stricte ──────────────────────────────────────────────
PULLBACK_MAX_DIST_ATR  = 0.6   # entry doar foarte aproape de EMA20
RESISTANCE_LOOKBACK    = 20
RESISTANCE_BUFFER_ATR  = 0.6   # 0.6 ATR de swing high/low → blocat
ENTRY_MIN_BODY_PCT     = 0.55  # bara intrare cu corp puternic


# ── Indicatori ────────────────────────────────────────────────────────────────

def _ema(s: pd.Series, period: int) -> pd.Series:
    return s.ewm(span=period, adjust=False).mean()


def _rsi(s: pd.Series, period: int = RSI_PERIOD) -> pd.Series:
    delta = s.diff()
    gain  = delta.clip(lower=0).rolling(period).mean()
    loss  = (-delta.clip(upper=0)).rolling(period).mean()
    rs    = gain / loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def _stochastic(df: pd.DataFrame, k_period: int = STOCH_K,
                d_period: int = STOCH_D, smooth: int = STOCH_SMOOTH):
    low_min  = df["low"].rolling(k_period).min()
    high_max = df["high"].rolling(k_period).max()
    k_raw = 100 * (df["close"] - low_min) / (high_max - low_min).replace(0, np.nan)
    k = k_raw.rolling(smooth).mean()
    d = k.rolling(d_period).mean()
    return k, d


def _atr(df: pd.DataFrame, period: int = ATR_PERIOD) -> float:
    if len(df) < period + 2:
        return 0.0
    h = df["high"].values
    l = df["low"].values
    c = df["close"].values
    tr = np.maximum(h[1:] - l[1:],
                    np.maximum(np.abs(h[1:] - c[:-1]),
                               np.abs(l[1:] - c[:-1])))
    if len(tr) < period:
        return float(np.mean(tr))
    return float(np.mean(tr[-period:]))


def _adx(df: pd.DataFrame, period: int = ADX_PERIOD) -> float:
    """ADX vectorizat — returneaza ultimul ADX value (0-100)."""
    h = df["high"].values
    l = df["low"].values
    c = df["close"].values
    if len(c) < period * 2 + 5:
        return 0.0
    tr = np.maximum(h[1:] - l[1:],
                    np.maximum(np.abs(h[1:] - c[:-1]),
                               np.abs(l[1:] - c[:-1])))
    plus_dm  = np.where((h[1:] - h[:-1]) > (l[:-1] - l[1:]),
                        np.maximum(h[1:] - h[:-1], 0.0), 0.0)
    minus_dm = np.where((l[:-1] - l[1:]) > (h[1:] - h[:-1]),
                        np.maximum(l[:-1] - l[1:], 0.0), 0.0)
    def rma(v, p):
        out = np.zeros(len(v))
        out[p-1] = np.mean(v[:p])
        for i in range(p, len(v)):
            out[i] = (out[i-1] * (p-1) + v[i]) / p
        return out
    atr_v   = rma(tr, period)
    safe    = np.where(atr_v == 0, 1.0, atr_v)
    plus_d  = 100.0 * rma(plus_dm, period) / safe
    minus_d = 100.0 * rma(minus_dm, period) / safe
    denom   = plus_d + minus_d
    dx      = 100.0 * np.abs(plus_d - minus_d) / np.where(denom == 0, 1.0, denom)
    adx_v   = rma(dx, period)
    return float(adx_v[-1])


def _volume_spike(df: pd.DataFrame) -> float:
    """Returneaza ratio volume_curent / SMA20. >1.0 = peste medie."""
    vol_col = ("tick_volume" if "tick_volume" in df.columns
               else ("real_volume" if "real_volume" in df.columns else None))
    if vol_col is None or len(df) < 21:
        return 1.0
    vols = df[vol_col].values.astype(float)
    sma  = float(np.mean(vols[-21:-1])) if len(vols) >= 21 else float(np.mean(vols))
    if sma <= 0:
        return 1.0
    return float(vols[-1] / sma)


def _htf_bias(symbol: str, fetch_fn, bars: int = 100) -> str:
    """Bias HTF (M15) prin EMA20 vs EMA50. Returneaza BULL/BEAR/NEUTRAL."""
    try:
        df, _ = fetch_fn(symbol, HTF_TIMEFRAME, bars)
        if df is None or len(df) < HTF_EMA_SLOW + 5:
            return "NEUTRAL"
        ema_f = float(_ema(df["close"], HTF_EMA_FAST).iloc[-1])
        ema_s = float(_ema(df["close"], HTF_EMA_SLOW).iloc[-1])
        diff_pct = abs(ema_f - ema_s) / ema_s if ema_s > 0 else 0
        if diff_pct < 0.0002:
            return "NEUTRAL"
        return "BULL" if ema_f > ema_s else "BEAR"
    except Exception:
        return "NEUTRAL"


# ── Detectie cross Stochastic %K x %D ─────────────────────────────────────────

def _stoch_cross_up(k: pd.Series, d: pd.Series, lookback: int = 2) -> bool:
    """%K trece peste %D in ultimele lookback bare, plecand din zona oversold."""
    if len(k) < lookback + 2:
        return False
    for i in range(-lookback, 0):
        if pd.isna(k.iloc[i-1]) or pd.isna(d.iloc[i-1]):
            continue
        if k.iloc[i-1] < d.iloc[i-1] and k.iloc[i] > d.iloc[i]:
            # plecat din oversold (< 30)
            if k.iloc[i-1] < STOCH_OVERSOLD:
                return True
    return False


def _stoch_cross_down(k: pd.Series, d: pd.Series, lookback: int = 2) -> bool:
    """%K trece sub %D in ultimele lookback bare, plecand din zona overbought."""
    if len(k) < lookback + 2:
        return False
    for i in range(-lookback, 0):
        if pd.isna(k.iloc[i-1]) or pd.isna(d.iloc[i-1]):
            continue
        if k.iloc[i-1] > d.iloc[i-1] and k.iloc[i] < d.iloc[i]:
            if k.iloc[i-1] > STOCH_OVERBOUGHT:
                return True
    return False


# ── Strategie ─────────────────────────────────────────────────────────────────

class BurstScalperStrategy(Strategy):
    key   = "burst_scalper"
    name  = "Burst Scalper"
    icon  = "⚡"
    color = "#ff5722"

    default_tfs            = [DEFAULT_TF]
    default_bars           = 200      # 200 bare M1 = ~3h history pt indicatori
    default_step           = 1        # M1 — verifica fiecare bara
    default_duration_years = 0.5      # jumate de an = ~120k bare M1 (suficient)
    default_max_hours      = 0.05     # 3 min max (max_bars_hold=2 pe M1)

    elements = {
        "ema_trend":      "EMA(5) vs EMA(20) — directie micro-trend",
        "stoch_cross":    "Stochastic %K cross %D in zona extrema",
        "rsi_confirm":    "RSI(7) confirma directia (>50 BUY / <50 SELL)",
        "atr_filter":     "ATR floor — skip piata moarta",
        "candle_dir":     "Bara curenta in directia semnalului",
        "htf_bias":       "Bias HTF (M15 EMA20/50) — blocheaza counter-trend",
        "adx_filter":     "ADX > 20 — doar trending markets, skip choppy",
        "volume_spike":   "Volume > 1.2x SMA20 — confirmare institutionala",
        "cooldown":       "Min 3 bare intre trade-uri — evita over-trading",
        "adx_strict":     "STRICT: ADX > 30 (vs default 20)",
        "cooldown_5":     "STRICT: 5 bare cooldown (vs 3)",
        "max_loss_pause": "STRICT: 2 pierderi consecutive → pauza 1h",
    }

    strategy_defaults = {
        "min_confidence": 65.0,
        "sl_atr_mult":    SL_ATR_MULT,
        "tp_atr_mult":    TP_ATR_MULT,
        "max_bars_hold":  MAX_BARS_HOLD,
        "min_score":      MIN_SCORE,
    }

    # ── Tracking pentru disciplina ───────────────────────────────────────────
    # Per simbol: ts ultim trade, ts ultim loss, contor trade-uri active
    _last_trade_ts: dict = {}
    _last_loss_ts:  dict = {}
    _active_trade:  dict = {}   # {sym: True} → un singur trade activ per simbol

    def analyze(self, symbol, tfs, bars=200, tf_bars=None, elements=None,
                min_confidence=60.0, **kwargs):
        from app import fetch

        if elements is None:
            elements = {k: True for k in self.elements}

        # ── Alege TF: prima din lista (configurabil) ─────────────────────────
        tf = (tfs[0] if tfs else DEFAULT_TF).upper()
        n_bars = (tf_bars or {}).get(tf, bars)

        try:
            df, _ = fetch(symbol, tf, max(n_bars, 100))
        except Exception as exc:
            return self._empty_result(symbol, f"Fetch error {tf}: {exc}")

        if df is None or len(df) < EMA_SLOW + 5:
            return self._empty_result(symbol, f"Date insuficiente pe {tf}")

        # ── Calcul indicatori ────────────────────────────────────────────────
        close = df["close"]
        ema_f = _ema(close, EMA_FAST)
        ema_s = _ema(close, EMA_SLOW)
        rsi   = _rsi(close, RSI_PERIOD)
        k, d  = _stochastic(df, STOCH_K, STOCH_D, STOCH_SMOOTH)
        atr   = _atr(df, ATR_PERIOD)
        atr_avg = _atr(df, ATR_AVG_PERIOD)

        if atr <= 0:
            return self._empty_result(symbol, "ATR indisponibil")

        price = float(close.iloc[-1])
        last_open  = float(df["open"].iloc[-1])
        last_close = float(close.iloc[-1])

        last_ema_f = float(ema_f.iloc[-1])
        last_ema_s = float(ema_s.iloc[-1])
        last_rsi   = float(rsi.iloc[-1]) if not pd.isna(rsi.iloc[-1]) else 50.0
        prev_rsi   = float(rsi.iloc[-2]) if len(rsi) >= 2 and not pd.isna(rsi.iloc[-2]) else 50.0

        # ── Filtre comune ────────────────────────────────────────────────────
        reasons = [f"TF: {tf}, Price: {price:.5f}, ATR: {atr:.5f}"]

        # Gate ATR
        if elements.get("atr_filter", True):
            if atr_avg > 0 and atr < atr_avg * 0.5:
                return self._empty_result(
                    symbol,
                    f"ATR squeeze {atr:.5f} < 50% medie {atr_avg:.5f} — piata moarta"
                )

        # Gate ADX — skip cand piata e choppy/range
        adx_val = 0.0
        if elements.get("adx_filter", True):
            adx_val = _adx(df, ADX_PERIOD)
            if adx_val < ADX_MIN:
                return self._empty_result(
                    symbol,
                    f"ADX {adx_val:.1f} < {ADX_MIN} — piata sideways/choppy, skip"
                )
            reasons.append(f"ADX {adx_val:.1f} >= {ADX_MIN} (trending) ✓")

        # ── Auto-clear single-trade lock daca max_hold a trecut ───────────────
        # Strategia nu primeste notificari de inchidere → daca au trecut
        # MAX_BARS_HOLD bare de la ultim trade, pozitia a fost inchisa (TP/SL/expirat)
        tf_min_local = {"M1":1,"M5":5,"M15":15,"M30":30,"H1":60,"H4":240,"D1":1440}.get(tf, 1)
        if self._active_trade.get(symbol, False):
            last_ts = self._last_trade_ts.get(symbol)
            if last_ts is not None:
                try:
                    elapsed_s = (df.index[-1] - last_ts).total_seconds()
                    max_hold_s = (MAX_BARS_HOLD + 1) * tf_min_local * 60
                    if elapsed_s >= max_hold_s:
                        self._active_trade[symbol] = False  # auto-release
                        # Daca pierderea a fost recenta, marcheaza last_loss_ts
                        # (estimare conservatoare: presupunem worst case loss)
                        # In productie, real-time, ar fi un callback de la executor.
                except Exception:
                    self._active_trade[symbol] = False  # safety release

        # ── Gate Single-Trade Lock (PRINCIPIU PRO: o singura pozitie activa) ──
        if self._active_trade.get(symbol, False):
            return self._empty_result(
                symbol, "Single-trade lock: deja un trade activ pe acest simbol"
            )

        # ── Gate Cooldown ─────────────────────────────────────────────────────
        if elements.get("cooldown", True):
            tf_min = {"M1":1,"M5":5,"M15":15,"M30":30,"H1":60,"H4":240,"D1":1440}.get(tf, 1)
            cur_ts = df.index[-1]

            # Cooldown standard intre trade-uri
            cooldown_s = COOLDOWN_BARS * tf_min * 60
            last_ts = self._last_trade_ts.get(symbol)
            if last_ts is not None:
                try:
                    delta_s = (cur_ts - last_ts).total_seconds()
                    if delta_s < cooldown_s:
                        return self._empty_result(
                            symbol,
                            f"Cooldown: {delta_s:.0f}s de la ultim trade (min {cooldown_s}s)"
                        )
                except Exception:
                    pass

            # NO REVENGE TRADING: cooldown extins dupa loss
            loss_cd_s = LOSS_COOLDOWN * tf_min * 60
            last_loss = self._last_loss_ts.get(symbol)
            if last_loss is not None:
                try:
                    delta_loss = (cur_ts - last_loss).total_seconds()
                    if delta_loss < loss_cd_s:
                        return self._empty_result(
                            symbol,
                            f"No-revenge: doar {delta_loss:.0f}s de la ultim SL (min {loss_cd_s}s)"
                        )
                except Exception:
                    pass

        # HTF bias — directia generala pe M15
        htf_dir = "NEUTRAL"
        if elements.get("htf_bias", True):
            htf_dir = _htf_bias(symbol, fetch)
            reasons.append(f"HTF {HTF_TIMEFRAME} bias: {htf_dir}")

        # Volume spike
        vol_ratio = 1.0
        if elements.get("volume_spike", True):
            vol_ratio = _volume_spike(df)
            reasons.append(f"Volume ratio: {vol_ratio:.2f}x SMA20")

        # ── Verificare conditii BUY ───────────────────────────────────────────
        buy_score = 0
        sell_score = 0

        # 1. EMA trend
        ema_bullish = last_ema_f > last_ema_s
        ema_bearish = last_ema_f < last_ema_s

        if elements.get("ema_trend", True):
            if ema_bullish:
                buy_score += 3
                reasons.append(f"EMA{EMA_FAST}>EMA{EMA_SLOW}: {last_ema_f:.5f} > {last_ema_s:.5f}")
            elif ema_bearish:
                sell_score += 3
                reasons.append(f"EMA{EMA_FAST}<EMA{EMA_SLOW}: {last_ema_f:.5f} < {last_ema_s:.5f}")

        # 2. Stochastic cross
        if elements.get("stoch_cross", True):
            if _stoch_cross_up(k, d, lookback=2):
                buy_score += 4
                reasons.append(f"Stoch %K cross UP din oversold (K={k.iloc[-1]:.1f})")
            if _stoch_cross_down(k, d, lookback=2):
                sell_score += 4
                reasons.append(f"Stoch %K cross DOWN din overbought (K={k.iloc[-1]:.1f})")

        # 3. RSI confirmation
        if elements.get("rsi_confirm", True):
            if last_rsi > 50 and last_rsi >= prev_rsi:
                buy_score += 2
                reasons.append(f"RSI {last_rsi:.1f} > 50 si crescator")
            if last_rsi < 50 and last_rsi <= prev_rsi:
                sell_score += 2
                reasons.append(f"RSI {last_rsi:.1f} < 50 si descrescator")

        # 4. Candle direction
        if elements.get("candle_dir", True):
            if last_close > last_open:
                buy_score += 1
            if last_close < last_open:
                sell_score += 1

        # 5. Volume spike — bonus pt. confirmare institutionala
        if elements.get("volume_spike", True):
            if vol_ratio >= VOLUME_SPIKE_MULT:
                # Bonus pe directia castigatoare
                if buy_score > sell_score:
                    buy_score += 1
                elif sell_score > buy_score:
                    sell_score += 1
                reasons.append(f"Volume spike OK: {vol_ratio:.2f}x (+1)")
            else:
                reasons.append(f"Volume slab: {vol_ratio:.2f}x — skip bonus")

        # ── HTF bias HARD GATE ────────────────────────────────────────────────
        # PRO RULE: Trade with the tide. Daca HTF e BULL, NU vindem. Daca BEAR, NU cumparam.
        # NEUTRAL inca permis (range/consolidare → mean reversion).
        if elements.get("htf_bias", True) and htf_dir != "NEUTRAL":
            if htf_dir == "BULL" and sell_score > buy_score:
                return self._empty_result(
                    symbol,
                    f"HTF BULL gate: SELL ({sell_score}) blocat — trade with the tide"
                )
            if htf_dir == "BEAR" and buy_score > sell_score:
                return self._empty_result(
                    symbol,
                    f"HTF BEAR gate: BUY ({buy_score}) blocat — trade with the tide"
                )

        # ── Decizie ──────────────────────────────────────────────────────────
        min_score = MIN_SCORE  # 7/10 (era 6)

        if buy_score == sell_score:
            return self._empty_result(symbol, f"Scoruri egale BUY={buy_score} SELL={sell_score} — skip")

        if buy_score > sell_score and buy_score >= min_score:
            signal = "BUY"
            conviction = min(buy_score, 11)
        elif sell_score > buy_score and sell_score >= min_score:
            signal = "SELL"
            conviction = min(sell_score, 11)
        else:
            best = max(buy_score, sell_score)
            return self._empty_result(
                symbol,
                f"Scor max {best}/11 sub prag {min_score} | BUY={buy_score} SELL={sell_score}"
            )

        # ════════════════════════════════════════════════════════════════════
        # ANTI-FOMO FILTERS — blocheaza entry la varful retracement-ului
        # ════════════════════════════════════════════════════════════════════

        # FILTRU 1: Pullback Required — pret aproape de EMA20 (nu prea departe)
        # Logica: cei mai buni trader-i cumpara la EMA, nu departe de ea.
        # Daca pretul e prea sus → momentum-ul s-a epuizat, intrare late.
        ema_dist_atr = abs(price - last_ema_s) / atr if atr > 0 else 0
        if ema_dist_atr > PULLBACK_MAX_DIST_ATR:
            return self._empty_result(
                symbol,
                f"Anti-FOMO: pret la {ema_dist_atr:.2f}xATR de EMA20 "
                f"(max {PULLBACK_MAX_DIST_ATR}xATR) — late entry, skip"
            )

        # FILTRU 2: Not at Recent High/Low — evita rezistenta/suportul
        # Logica: daca pretul e la swing high, urmeaza reversal probabil.
        if len(df) >= RESISTANCE_LOOKBACK + 2:
            recent_high = float(df["high"].iloc[-RESISTANCE_LOOKBACK-1:-1].max())
            recent_low  = float(df["low"].iloc[-RESISTANCE_LOOKBACK-1:-1].min())
            buf = atr * RESISTANCE_BUFFER_ATR

            if signal == "BUY" and price >= recent_high - buf:
                return self._empty_result(
                    symbol,
                    f"Anti-FOMO: BUY blocat la rezistenta — pret {price:.5f} "
                    f"in {RESISTANCE_BUFFER_ATR}xATR de swing high {recent_high:.5f}"
                )
            if signal == "SELL" and price <= recent_low + buf:
                return self._empty_result(
                    symbol,
                    f"Anti-FOMO: SELL blocat la suport — pret {price:.5f} "
                    f"in {RESISTANCE_BUFFER_ATR}xATR de swing low {recent_low:.5f}"
                )

        # FILTRU 3: Strong Entry Bar — bara curenta cu corp semnificativ
        # Logica: bara doji/wick = indecisie. Vrem candle puternic in directia.
        last_high = float(df["high"].iloc[-1])
        last_low  = float(df["low"].iloc[-1])
        bar_range = last_high - last_low
        bar_body  = abs(last_close - last_open)
        body_pct  = (bar_body / bar_range) if bar_range > 1e-10 else 0
        if body_pct < ENTRY_MIN_BODY_PCT:
            return self._empty_result(
                symbol,
                f"Anti-FOMO: bara intrare slaba (corp {body_pct*100:.0f}% < "
                f"{ENTRY_MIN_BODY_PCT*100:.0f}%) — indecizie, skip"
            )
        # Verifica si directia barei vs semnal
        if signal == "BUY" and last_close < last_open:
            return self._empty_result(symbol, "Anti-FOMO: BUY dar bara curenta e bear")
        if signal == "SELL" and last_close > last_open:
            return self._empty_result(symbol, "Anti-FOMO: SELL dar bara curenta e bull")

        reasons.append(f"Anti-FOMO ✓: EMA dist {ema_dist_atr:.2f}xATR, "
                       f"corp {body_pct*100:.0f}%, departe de S/R")

        # Inregistreaza timestamp + marcheaza trade activ (single-trade lock)
        try:
            self._last_trade_ts[symbol] = df.index[-1]
            self._active_trade[symbol] = True
        except Exception:
            pass

        # ── SL/TP ATR-based ──────────────────────────────────────────────────
        sl_dist = atr * SL_ATR_MULT
        tp_dist = atr * TP_ATR_MULT

        if signal == "BUY":
            sl = price - sl_dist
            tp = price + tp_dist
        else:
            sl = price + sl_dist
            tp = price - tp_dist

        reasons.append(f"SL {sl_dist:.5f} ({SL_ATR_MULT}×ATR), TP {tp_dist:.5f} ({TP_ATR_MULT}×ATR), RR 1:{TP_ATR_MULT/SL_ATR_MULT:.2f}")
        reasons.append(f"Max hold: {MAX_BARS_HOLD} bare {tf}")

        tf_result = {
            "tf":         tf,
            "signal":     signal,
            "conviction": conviction,
            "reasons":    reasons,
            "price":      round(price, 5),
            "sl":         round(sl, 5),
            "tp":         round(tp, 5),
        }

        return self._build_result(
            symbol, [tf_result], min_confidence, min_votes=1,
            extra={
                "max_bars_hold": MAX_BARS_HOLD,
                "buy_score":     buy_score,
                "sell_score":    sell_score,
                "atr":           round(atr, 5),
                "sl_distance":   round(sl_dist, 5),
                "tp_distance":   round(tp_dist, 5),
            },
        )
