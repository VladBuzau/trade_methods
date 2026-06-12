"""
ai_agent/db/schema.py — schema SQLite si helper connection.

Tabele:
  prices        — OHLCV per symbol/TF/timestamp
  news          — articole + sentiment scores
  tweets        — tweets de la conturi tracked
  macro         — FRED series (yield curve, VIX, DXY, etc.)
  features      — feature matrix per symbol/timestamp (computed)
  labels        — target labels (future returns)
  predictions   — predictiile modelului (timestamp + score)
  trades_paper  — paper trades pentru evaluare
"""
from __future__ import annotations
import sqlite3
from contextlib import contextmanager
from ai_agent.config import DB_PATH


SCHEMA_SQL = """
-- ─── PRICES ─────────────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS prices (
    symbol     TEXT    NOT NULL,
    timeframe  TEXT    NOT NULL,   -- M5/M15/H1/H4/D1
    ts         INTEGER NOT NULL,   -- unix timestamp UTC
    open       REAL    NOT NULL,
    high       REAL    NOT NULL,
    low        REAL    NOT NULL,
    close      REAL    NOT NULL,
    volume     REAL,
    PRIMARY KEY (symbol, timeframe, ts)
);
CREATE INDEX IF NOT EXISTS idx_prices_sym_tf ON prices(symbol, timeframe);
CREATE INDEX IF NOT EXISTS idx_prices_ts ON prices(ts);

-- ─── NEWS ───────────────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS news (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    source        TEXT    NOT NULL,   -- 'alphavantage', 'newsapi', 'reddit'
    external_id   TEXT,               -- pentru dedup
    ts            INTEGER NOT NULL,   -- timestamp publicat
    symbols       TEXT,               -- CSV de simboluri legate (ex: 'EURUSD,GBPUSD')
    title         TEXT,
    body          TEXT,
    url           TEXT,
    sentiment     REAL,               -- score [-1, +1] din FinBERT
    impact_score  REAL,               -- proprietary impact (relevance × magnitude)
    raw_json      TEXT,
    UNIQUE(source, external_id)
);
CREATE INDEX IF NOT EXISTS idx_news_ts ON news(ts);
CREATE INDEX IF NOT EXISTS idx_news_source ON news(source);

-- ─── TWEETS ─────────────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS tweets (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    username      TEXT    NOT NULL,
    tweet_id      TEXT    UNIQUE,
    ts            INTEGER NOT NULL,
    text          TEXT,
    retweets      INTEGER,
    likes         INTEGER,
    sentiment     REAL,
    impact_score  REAL,
    raw_json      TEXT
);
CREATE INDEX IF NOT EXISTS idx_tweets_user_ts ON tweets(username, ts);

-- ─── MACRO ─────────────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS macro (
    series_id  TEXT    NOT NULL,
    ts         INTEGER NOT NULL,
    value      REAL,
    PRIMARY KEY (series_id, ts)
);
CREATE INDEX IF NOT EXISTS idx_macro_ts ON macro(ts);

-- ─── FEATURES ──────────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS features (
    symbol     TEXT    NOT NULL,
    timeframe  TEXT    NOT NULL,
    ts         INTEGER NOT NULL,
    features   TEXT    NOT NULL,    -- JSON serialized dict
    PRIMARY KEY (symbol, timeframe, ts)
);

-- ─── LABELS ─────────────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS labels (
    symbol     TEXT    NOT NULL,
    timeframe  TEXT    NOT NULL,
    ts         INTEGER NOT NULL,
    horizon_h  INTEGER NOT NULL,
    return_pct REAL,           -- (close_future / close_now - 1) * 100
    hit_long   INTEGER,        -- 1 daca BUY ar fi profitat
    hit_short  INTEGER,        -- 1 daca SELL ar fi profitat
    PRIMARY KEY (symbol, timeframe, ts, horizon_h)
);

-- ─── PREDICTIONS ────────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS predictions (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    model_id    TEXT    NOT NULL,
    symbol      TEXT    NOT NULL,
    timeframe   TEXT    NOT NULL,
    ts          INTEGER NOT NULL,
    pred_dir    INTEGER,         -- 1 = BUY, -1 = SELL, 0 = HOLD
    confidence  REAL,
    horizon_h   INTEGER,
    features_used TEXT,
    UNIQUE(model_id, symbol, timeframe, ts, horizon_h)
);

-- ─── PAPER TRADES (validation) ─────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS trades_paper (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    model_id    TEXT,
    symbol      TEXT,
    entry_ts    INTEGER,
    entry_price REAL,
    exit_ts     INTEGER,
    exit_price  REAL,
    side        TEXT,            -- 'BUY' / 'SELL'
    sl          REAL,
    tp          REAL,
    pnl         REAL,
    outcome     TEXT             -- 'TP', 'SL', 'TIMEOUT'
);
"""


def init_db():
    """Creeaza toate tabelele daca nu exista."""
    with sqlite3.connect(str(DB_PATH)) as conn:
        conn.executescript(SCHEMA_SQL)
        conn.commit()


@contextmanager
def get_conn():
    """Context manager pentru conexiune SQLite."""
    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def stats() -> dict:
    """Returneaza nr de inregistrari per tabel."""
    init_db()
    counts = {}
    with get_conn() as conn:
        for table in ["prices", "news", "tweets", "macro",
                      "features", "labels", "predictions", "trades_paper"]:
            try:
                r = conn.execute(f"SELECT COUNT(*) as c FROM {table}").fetchone()
                counts[table] = r["c"]
            except Exception:
                counts[table] = 0
    return counts


if __name__ == "__main__":
    init_db()
    print(f"DB created: {DB_PATH}")
    for t, c in stats().items():
        print(f"  {t}: {c} rows")
