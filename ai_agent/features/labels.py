"""
ai_agent/features/labels.py — generare labels pentru supervised learning.

Pentru fiecare (symbol, TF, ts), calculam ce s-a intamplat DUPA acel moment:
  - future_return_4h, _24h, _72h     (% miscare pret)
  - hit_long_TP, hit_long_SL          (BUY cu RR 1:1 a atins TP sau SL primul?)
  - hit_short_TP, hit_short_SL        (mirror pt SELL)

Aceste labels sunt target-urile pentru model. Predict-ul "next bar direction"
e mai usor dar mai putin util; predict-ul "hit_TP_first" e direct actionabil.

Foloseste ATR la momentul tranzactiei pentru a defini SL/TP (RR 1:1 cu ATR×1.5).
"""
from __future__ import annotations
import sys
import time
import logging
from datetime import timedelta

import numpy as np
import pandas as pd

from ai_agent.db.schema import init_db, get_conn
from ai_agent.config import LABEL_HORIZONS_HOURS

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

ATR_MULT_SL = 1.5   # SL = entry ± 1.5 × ATR
ATR_MULT_TP = 1.5   # TP = entry ± 1.5 × ATR (RR 1:1)


def _bars_per_hour(tf: str) -> int:
    """Cate bare ies intr-o ora pe TF-ul respectiv."""
    return {"M1": 60, "M5": 12, "M15": 4, "M30": 2,
            "H1":  1, "H4":  1, "D1": 1}.get(tf, 1)


def _atr_series(df: pd.DataFrame, period: int = 14) -> pd.Series:
    h, l, c = df["high"], df["low"], df["close"]
    tr = pd.concat([h - l, (h - c.shift()).abs(), (l - c.shift()).abs()], axis=1).max(axis=1)
    return tr.ewm(alpha=1/period, adjust=False).mean()


def load_prices_df(symbol: str, tf: str) -> pd.DataFrame:
    with get_conn() as conn:
        rows = conn.execute(
            "SELECT ts, open, high, low, close, volume FROM prices "
            "WHERE symbol=? AND timeframe=? ORDER BY ts",
            (symbol, tf),
        ).fetchall()
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame([dict(r) for r in rows])
    return df.set_index("ts")


def compute_labels(df: pd.DataFrame, tf: str) -> list[tuple]:
    """
    Returneaza rows pentru tabela labels:
      (symbol_placeholder, tf, ts, horizon_h, return_pct, hit_long, hit_short)
    Symbol-ul il setam afara.
    """
    if df.empty:
        return []
    atr = _atr_series(df, 14)
    closes = df["close"].values
    highs  = df["high"].values
    lows   = df["low"].values

    n = len(df)
    rows = []

    for horizon_h in LABEL_HORIZONS_HOURS:
        bars_ahead = horizon_h * _bars_per_hour(tf)
        if bars_ahead < 1:
            bars_ahead = 1
        for i in range(15, n - bars_ahead):
            entry = closes[i]
            if entry <= 0:
                continue
            atr_now = float(atr.iloc[i])
            if atr_now <= 0:
                continue
            # Slice viitor
            future_h = highs[i+1 : i+1+bars_ahead]
            future_l = lows[i+1  : i+1+bars_ahead]
            future_c = closes[i+1: i+1+bars_ahead]
            if len(future_c) == 0:
                continue

            # Return total pe horizon
            ret_pct = (future_c[-1] / entry - 1.0) * 100

            # BUY scenario
            buy_tp = entry + atr_now * ATR_MULT_TP
            buy_sl = entry - atr_now * ATR_MULT_SL
            hit_long = 0
            for j in range(len(future_h)):
                if future_h[j] >= buy_tp:
                    hit_long = 1
                    break
                if future_l[j] <= buy_sl:
                    hit_long = 0
                    break
            else:
                # Nu am atins niciun nivel — folosim return final
                hit_long = 1 if future_c[-1] > entry else 0

            # SELL scenario
            sell_tp = entry - atr_now * ATR_MULT_TP
            sell_sl = entry + atr_now * ATR_MULT_SL
            hit_short = 0
            for j in range(len(future_h)):
                if future_l[j] <= sell_tp:
                    hit_short = 1
                    break
                if future_h[j] >= sell_sl:
                    hit_short = 0
                    break
            else:
                hit_short = 1 if future_c[-1] < entry else 0

            ts = int(df.index[i])
            rows.append((tf, ts, horizon_h, round(float(ret_pct), 4),
                         int(hit_long), int(hit_short)))
    return rows


def save_labels(symbol: str, rows: list[tuple]) -> int:
    """rows = [(tf, ts, horizon_h, return_pct, hit_long, hit_short), ...]"""
    if not rows:
        return 0
    with get_conn() as conn:
        c = conn.executemany(
            "INSERT OR REPLACE INTO labels "
            "(symbol, timeframe, ts, horizon_h, return_pct, hit_long, hit_short) "
            "VALUES (?,?,?,?,?,?,?)",
            [(symbol, r[0], r[1], r[2], r[3], r[4], r[5]) for r in rows],
        )
        return c.rowcount


def build_for_symbol_tf(symbol: str, tf: str) -> int:
    df = load_prices_df(symbol, tf)
    if df.empty:
        log.warning(f"{symbol}/{tf}: 0 bare in DB")
        return 0
    log.info(f"{symbol}/{tf}: {len(df)} bare → genereaza labels pe horizons {LABEL_HORIZONS_HOURS}h")
    rows = compute_labels(df, tf)
    n = save_labels(symbol, rows)
    log.info(f"  → {n} labels inserate")
    return n


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
    if len(sys.argv) >= 3:
        build_for_symbol_tf(sys.argv[1], sys.argv[2])
    elif len(sys.argv) >= 2:
        build_all([sys.argv[1]])
    else:
        results = build_all()
        print()
        print("=" * 60)
        for k, v in results.items():
            print(f"  {k}: {v} labels")
