"""
ai_agent/features/builder.py — calculatoare de features tehnice.

Pentru fiecare (symbol, TF, timestamp), produce un dict cu ~40 features:
  - Momentum:   RSI, ROC, MACD
  - Volatility: ATR ratio, BB width, BB position
  - Trend:      EMA slopes, EMA distances
  - Patterns:   candle body %, wick ratios, doji
  - Volume:     volume z-score, volume vs SMA
  - Time:       hour, day-of-week
  - Cross-TF:   trend pe HTF (cand TF e M5/M15)

Toate features-urile sunt numerice (NaN-safe).
"""
from __future__ import annotations
import json
import time
import logging
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from ai_agent.db.schema import init_db, get_conn
from ai_agent.config import FEATURE_LOOKBACK_BARS

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)


# ─── Indicatori vectorizati ─────────────────────────────────────────────────
def _ema(s: pd.Series, span: int) -> pd.Series:
    return s.ewm(span=span, adjust=False).mean()


def _rsi(s: pd.Series, period: int = 14) -> pd.Series:
    d = s.diff()
    gain = d.clip(lower=0).ewm(alpha=1/period, adjust=False).mean()
    loss = (-d.clip(upper=0)).ewm(alpha=1/period, adjust=False).mean()
    rs = gain / loss.replace(0, np.nan)
    return 100 - 100 / (1 + rs)


def _macd(s: pd.Series, fast: int = 12, slow: int = 26, sig: int = 9):
    ef = _ema(s, fast)
    es = _ema(s, slow)
    line = ef - es
    signal = _ema(line, sig)
    return line, signal, line - signal


def _atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    h, l, c = df["high"], df["low"], df["close"]
    tr = pd.concat([h - l,
                    (h - c.shift()).abs(),
                    (l - c.shift()).abs()], axis=1).max(axis=1)
    return tr.ewm(alpha=1/period, adjust=False).mean()


def _bbands(s: pd.Series, period: int = 20, k: float = 2.0):
    sma = s.rolling(period).mean()
    std = s.rolling(period).std()
    upper = sma + k * std
    lower = sma - k * std
    return upper, sma, lower


# ─── Feature computation ────────────────────────────────────────────────────
def compute_features_for_bar(df: pd.DataFrame, idx: int) -> dict | None:
    """
    Compute features-urile pentru bara `idx` din df. Returneaza None daca
    nu sunt suficiente bare istoric (min 50).
    """
    if idx < 50 or idx >= len(df):
        return None
    # Sub-frame pana la bara curenta (inclusiv)
    sub = df.iloc[:idx + 1]
    c = sub["close"]
    h = sub["high"]
    l = sub["low"]
    o = sub["open"]
    v = sub["volume"]

    price = float(c.iloc[-1])
    if price <= 0:
        return None

    # ── Momentum ────────────────────────────────────────────────────────
    rsi14 = _rsi(c, 14)
    rsi7  = _rsi(c, 7)
    macd_line, macd_sig, macd_hist = _macd(c)

    # ── Volatility ──────────────────────────────────────────────────────
    atr14 = _atr(sub, 14)
    atr_now = float(atr14.iloc[-1])
    atr_avg50 = float(atr14.iloc[-50:].mean()) if len(atr14) >= 50 else atr_now
    atr_ratio = (atr_now / atr_avg50) if atr_avg50 > 0 else 1.0

    bbu, bbm, bbl = _bbands(c, 20, 2.0)
    bbu_v = float(bbu.iloc[-1]) if not pd.isna(bbu.iloc[-1]) else price
    bbl_v = float(bbl.iloc[-1]) if not pd.isna(bbl.iloc[-1]) else price
    bbm_v = float(bbm.iloc[-1]) if not pd.isna(bbm.iloc[-1]) else price
    bb_width = (bbu_v - bbl_v) / bbm_v if bbm_v > 0 else 0.0
    # BB position: 0 = la lower, 1 = la upper
    bb_pos = ((price - bbl_v) / (bbu_v - bbl_v)) if (bbu_v - bbl_v) > 0 else 0.5

    # Realized volatility (std of log returns)
    log_ret = np.log(c / c.shift(1))
    rvol_20 = float(log_ret.iloc[-20:].std() * np.sqrt(252 * 24)) if len(log_ret) >= 20 else 0.0

    # ── Trend (EMA slopes) ──────────────────────────────────────────────
    ema8  = _ema(c, 8).iloc[-1]
    ema21 = _ema(c, 21).iloc[-1]
    ema50 = _ema(c, 50).iloc[-1]
    ema200_avail = len(c) >= 200
    ema200 = _ema(c, 200).iloc[-1] if ema200_avail else ema50

    # Distante normalizate (cat % e pretul fata de EMA)
    def _dist(a, b):
        return (a / b - 1.0) * 100 if b > 0 else 0.0

    # ── ROC (rate of change) ────────────────────────────────────────────
    def _roc(n):
        if len(c) <= n:
            return 0.0
        return float((c.iloc[-1] / c.iloc[-1 - n] - 1.0) * 100)

    # ── Candle patterns ─────────────────────────────────────────────────
    o_now, c_now = float(o.iloc[-1]), float(c.iloc[-1])
    h_now, l_now = float(h.iloc[-1]), float(l.iloc[-1])
    rng = h_now - l_now
    body = abs(c_now - o_now)
    upper_wick = h_now - max(c_now, o_now)
    lower_wick = min(c_now, o_now) - l_now
    body_pct  = (body / rng) if rng > 0 else 0.0
    upper_pct = (upper_wick / rng) if rng > 0 else 0.0
    lower_pct = (lower_wick / rng) if rng > 0 else 0.0
    is_bull = 1 if c_now > o_now else 0
    is_doji = 1 if body_pct < 0.1 else 0
    is_pin_bull = 1 if (lower_pct > 0.6 and body_pct < 0.3) else 0
    is_pin_bear = 1 if (upper_pct > 0.6 and body_pct < 0.3) else 0

    # ── Volume ──────────────────────────────────────────────────────────
    v_now = float(v.iloc[-1]) if not pd.isna(v.iloc[-1]) else 0.0
    if len(v) >= 20:
        v_mean = float(v.iloc[-20:].mean())
        v_std  = float(v.iloc[-20:].std())
        v_z = (v_now - v_mean) / v_std if v_std > 0 else 0.0
        v_ratio = v_now / v_mean if v_mean > 0 else 1.0
    else:
        v_z, v_ratio = 0.0, 1.0

    # ── Time features ───────────────────────────────────────────────────
    ts = int(sub.index[-1] if isinstance(sub.index[-1], (int, float))
             else sub.iloc[-1].name)
    dt = datetime.fromtimestamp(ts, tz=timezone.utc) if ts < 10**11 else datetime.fromtimestamp(ts/1000, tz=timezone.utc)
    hour = dt.hour
    dow  = dt.weekday()

    # ── Combined feature dict ──────────────────────────────────────────
    # ── Structural features (30 noi, in plus fata de cele 41 tehnice) ────────
    try:
        from ai_agent.features.structural import compute_structural_features
        struct_feats = compute_structural_features(df, idx)
    except Exception:
        struct_feats = {}

    feats = {
        # Momentum
        "rsi_14":           round(float(rsi14.iloc[-1]) if not pd.isna(rsi14.iloc[-1]) else 50.0, 2),
        "rsi_7":            round(float(rsi7.iloc[-1])  if not pd.isna(rsi7.iloc[-1])  else 50.0, 2),
        "rsi_14_delta":     round(float(rsi14.iloc[-1] - rsi14.iloc[-5])
                                  if len(rsi14) >= 5 and not pd.isna(rsi14.iloc[-1]) else 0.0, 2),
        "macd_line":        round(float(macd_line.iloc[-1]), 6),
        "macd_signal":      round(float(macd_sig.iloc[-1]), 6),
        "macd_hist":        round(float(macd_hist.iloc[-1]), 6),
        "macd_hist_delta":  round(float(macd_hist.iloc[-1] - macd_hist.iloc[-2])
                                  if len(macd_hist) >= 2 else 0.0, 6),
        # Volatility
        "atr_14":           round(atr_now, 6),
        "atr_ratio":        round(atr_ratio, 3),
        "bb_width":         round(bb_width, 4),
        "bb_position":      round(bb_pos, 3),
        "rvol_20":          round(rvol_20, 4),
        # Trend
        "ema_8_dist":       round(_dist(price, ema8), 4),
        "ema_21_dist":      round(_dist(price, ema21), 4),
        "ema_50_dist":      round(_dist(price, ema50), 4),
        "ema_200_dist":     round(_dist(price, ema200), 4),
        "ema_8_21_slope":   round(_dist(ema8, ema21), 4),
        "ema_21_50_slope":  round(_dist(ema21, ema50), 4),
        "ema_50_200_slope": round(_dist(ema50, ema200), 4),
        "trend_bull_align": 1 if (ema8 > ema21 > ema50) else 0,
        "trend_bear_align": 1 if (ema8 < ema21 < ema50) else 0,
        # ROC
        "roc_5":            round(_roc(5), 3),
        "roc_10":           round(_roc(10), 3),
        "roc_20":           round(_roc(20), 3),
        # Candle
        "body_pct":         round(body_pct, 3),
        "upper_wick_pct":   round(upper_pct, 3),
        "lower_wick_pct":   round(lower_pct, 3),
        "is_bull":          is_bull,
        "is_doji":          is_doji,
        "is_pin_bull":      is_pin_bull,
        "is_pin_bear":      is_pin_bear,
        # Volume
        "volume":           round(v_now, 2),
        "volume_z":         round(v_z, 3),
        "volume_ratio":     round(v_ratio, 3),
        # Time
        "hour":             hour,
        "day_of_week":      dow,
        "is_weekend":       1 if dow >= 5 else 0,
        "is_london_session":  1 if 7 <= hour < 16 else 0,
        "is_ny_session":      1 if 13 <= hour < 22 else 0,
        "is_asia_session":    1 if 0 <= hour < 9 else 0,
        # Meta
        "_n_bars_available": int(len(c)),
    }
    # Merge structural features (cu prefix s_*)
    feats.update(struct_feats)
    return feats


# ─── Database loaders ───────────────────────────────────────────────────────
def load_prices(symbol: str, timeframe: str) -> pd.DataFrame:
    with get_conn() as conn:
        rows = conn.execute(
            "SELECT ts, open, high, low, close, volume FROM prices "
            "WHERE symbol=? AND timeframe=? ORDER BY ts",
            (symbol, timeframe),
        ).fetchall()
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame([dict(r) for r in rows])
    df.set_index("ts", inplace=True)
    return df


def save_features_batch(rows: list[tuple]) -> int:
    """rows = [(symbol, tf, ts, json_features), ...]"""
    if not rows:
        return 0
    with get_conn() as conn:
        c = conn.executemany(
            "INSERT OR REPLACE INTO features (symbol, timeframe, ts, features) "
            "VALUES (?,?,?,?)",
            rows,
        )
        return c.rowcount


# ─── Main runner ────────────────────────────────────────────────────────────
def build_for_symbol_tf(symbol: str, timeframe: str,
                        start_idx: int = 200,
                        batch_size: int = 5000) -> int:
    """Computeaza features pentru toate barele dintr-un (symbol, tf)."""
    df = load_prices(symbol, timeframe)
    if df.empty:
        log.warning(f"{symbol}/{timeframe}: 0 bare in DB")
        return 0
    n = len(df)
    log.info(f"{symbol}/{timeframe}: {n} bare → calculam features de la idx {start_idx}")

    batch = []
    total = 0
    for i in range(start_idx, n):
        try:
            feats = compute_features_for_bar(df, i)
            if feats is None:
                continue
            ts = int(df.index[i])
            batch.append((symbol, timeframe, ts, json.dumps(feats)))
            if len(batch) >= batch_size:
                total += save_features_batch(batch)
                batch = []
                log.info(f"  ...{total}/{n - start_idx}")
        except Exception as e:
            log.debug(f"  bar {i}: {e}")
    if batch:
        total += save_features_batch(batch)
    log.info(f"{symbol}/{timeframe}: {total} features inserate")
    return total


def build_all(symbols: list[str] | None = None,
              timeframes: list[str] | None = None) -> dict:
    from ai_agent.config import SYMBOLS, TIMEFRAMES
    init_db()
    syms = symbols or SYMBOLS
    tfs  = timeframes or TIMEFRAMES
    res = {}
    for sym in syms:
        for tf in tfs:
            t0 = time.time()
            n = build_for_symbol_tf(sym, tf)
            res[f"{sym}/{tf}"] = n
            log.info(f"  done in {time.time()-t0:.1f}s")
    return res


if __name__ == "__main__":
    import sys
    if len(sys.argv) >= 3:
        build_for_symbol_tf(sys.argv[1], sys.argv[2])
    elif len(sys.argv) >= 2:
        build_all([sys.argv[1]])
    else:
        results = build_all()
        print()
        print("=" * 60)
        for k, v in results.items():
            print(f"  {k}: {v} features")
