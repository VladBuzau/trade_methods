"""
ai_agent/data/mt5_dumper.py — descarca istoric MT5 → SQLite.

Foloseste conexiunea MT5 din ChartVisualizer/app.py daca e deja activa, sau
initializeaza una proprie.

Usage:
    python -m ai_agent.data.mt5_dumper            # all symbols, all TFs
    python -m ai_agent.data.mt5_dumper EURUSD H1  # un simbol + un TF
"""
from __future__ import annotations
import sys
import time
import logging
from datetime import datetime, timezone, timedelta

import pandas as pd
import MetaTrader5 as mt5

from ai_agent.config import SYMBOLS, TIMEFRAMES, HISTORICAL_YEARS
from ai_agent.db.schema import init_db, get_conn

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

_TF_MAP = {
    "M1":  mt5.TIMEFRAME_M1,
    "M5":  mt5.TIMEFRAME_M5,
    "M15": mt5.TIMEFRAME_M15,
    "M30": mt5.TIMEFRAME_M30,
    "H1":  mt5.TIMEFRAME_H1,
    "H4":  mt5.TIMEFRAME_H4,
    "D1":  mt5.TIMEFRAME_D1,
}


def ensure_mt5() -> bool:
    if not mt5.initialize():
        log.error(f"MT5 initialize failed: {mt5.last_error()}")
        return False
    info = mt5.account_info()
    log.info(f"MT5 connected — account {info.login if info else 'demo'}")
    return True


def fetch_chunked(symbol: str, tf_str: str, start: datetime, end: datetime,
                  chunk_days: int = 30) -> pd.DataFrame:
    """
    MT5 returneaza max ~10k bare per call. Spargem in chunks de chunk_days zile.
    """
    tf_const = _TF_MAP.get(tf_str)
    if tf_const is None:
        return pd.DataFrame()

    all_dfs = []
    cur = start
    while cur < end:
        chunk_end = min(cur + timedelta(days=chunk_days), end)
        try:
            rates = mt5.copy_rates_range(symbol, tf_const, cur, chunk_end)
            if rates is not None and len(rates) > 0:
                df = pd.DataFrame(rates)
                all_dfs.append(df)
        except Exception as e:
            log.warning(f"  chunk {cur.date()}→{chunk_end.date()}: {e}")
        cur = chunk_end
        time.sleep(0.05)   # politicos cu MT5

    if not all_dfs:
        return pd.DataFrame()
    return pd.concat(all_dfs, ignore_index=True).drop_duplicates(subset=["time"])


def insert_prices(symbol: str, tf: str, df: pd.DataFrame) -> int:
    """Insert bulk in tabela prices (REPLACE pe (symbol, tf, ts))."""
    if df.empty:
        return 0
    rows = [
        (symbol, tf, int(r["time"]), float(r["open"]), float(r["high"]),
         float(r["low"]), float(r["close"]),
         float(r.get("tick_volume", 0) or r.get("real_volume", 0) or 0))
        for r in df.to_dict("records")
    ]
    with get_conn() as conn:
        conn.executemany(
            "INSERT OR REPLACE INTO prices "
            "(symbol, timeframe, ts, open, high, low, close, volume) "
            "VALUES (?,?,?,?,?,?,?,?)",
            rows,
        )
    return len(rows)


def dump_symbol_tf(symbol: str, tf: str, years: float = HISTORICAL_YEARS) -> int:
    """Descarca tot istoricul pentru (symbol, tf)."""
    end = datetime.now(timezone.utc)
    start = end - timedelta(days=int(years * 365))
    log.info(f"Fetching {symbol}/{tf} de la {start.date()} → {end.date()}")
    df = fetch_chunked(symbol, tf, start, end, chunk_days=180)
    if df.empty:
        log.warning(f"  → 0 bare returnate")
        return 0
    n = insert_prices(symbol, tf, df)
    log.info(f"  → {n} bare inserate (range: {df['time'].min()}..{df['time'].max()})")
    return n


def dump_all(symbols: list[str] | None = None,
             tfs: list[str] | None = None,
             years: float = HISTORICAL_YEARS) -> dict:
    """Dump complet pentru toate combinatiile symbol × tf."""
    init_db()
    if not ensure_mt5():
        return {}
    syms = symbols or SYMBOLS
    tfs_l = tfs or TIMEFRAMES
    results = {}
    total = len(syms) * len(tfs_l)
    i = 0
    for sym in syms:
        for tf in tfs_l:
            i += 1
            log.info(f"[{i}/{total}] {sym}/{tf}")
            n = dump_symbol_tf(sym, tf, years)
            results[f"{sym}/{tf}"] = n
    return results


if __name__ == "__main__":
    if len(sys.argv) >= 3:
        results = dump_all([sys.argv[1]], [sys.argv[2]])
    elif len(sys.argv) >= 2:
        results = dump_all([sys.argv[1]])
    else:
        results = dump_all()
    print()
    print("=" * 60)
    total_rows = sum(results.values())
    print(f"Total bare inserate: {total_rows}")
    for k, v in sorted(results.items()):
        print(f"  {k}: {v}")
