"""
ai_agent/config.py — configurare centralizata pentru intregul agent AI.

Cheile API se citesc din env vars (nu hardcoded). Variabilele se pot seta in
.env (cu python-dotenv) sau direct in environment.
"""
from __future__ import annotations
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "db"
LOGS_DIR = ROOT / "logs"
MODELS_DIR = ROOT / "models" / "saved"

for d in (DATA_DIR, LOGS_DIR, MODELS_DIR):
    d.mkdir(parents=True, exist_ok=True)

# ── Database ────────────────────────────────────────────────────────────────
DB_PATH = DATA_DIR / "agent.db"

# ── Symbols of interest ─────────────────────────────────────────────────────
SYMBOLS = [
    "EURUSD", "GBPUSD", "USDJPY", "AUDUSD", "USDCAD",
    "EURGBP", "EURJPY", "GBPJPY", "XAUUSD",
    "BTCUSD", "ETHUSD",
]

TIMEFRAMES = ["M5", "M15", "H1", "H4", "D1"]

# Cati ani de istoric default
HISTORICAL_YEARS = 5

# ── External API keys (env vars) ────────────────────────────────────────────
# AlphaVantage News — https://www.alphavantage.co/support/#api-key
ALPHA_VANTAGE_KEY = os.environ.get("ALPHA_VANTAGE_KEY", "")

# NewsAPI.org — https://newsapi.org
NEWSAPI_KEY = os.environ.get("NEWSAPI_KEY", "")

# Twitter/X — https://developer.twitter.com
TWITTER_BEARER_TOKEN = os.environ.get("TWITTER_BEARER_TOKEN", "")

# FRED (Federal Reserve Economic Data) — gratis cu inregistrare
# https://fred.stlouisfed.org/docs/api/api_key.html
FRED_KEY = os.environ.get("FRED_KEY", "")

# Reddit — gratis, prin praw
REDDIT_CLIENT_ID     = os.environ.get("REDDIT_CLIENT_ID", "")
REDDIT_CLIENT_SECRET = os.environ.get("REDDIT_CLIENT_SECRET", "")
REDDIT_USER_AGENT    = "ai_agent_v0.1 by u/anonymous"

# ── News sources de interes ─────────────────────────────────────────────────
TWITTER_ACCOUNTS_TO_TRACK = [
    "realDonaldTrump",      # politica US (impact USD)
    "federalreserve",       # Fed comunicari
    "ECB",                  # ECB
    "BankofEngland",        # BoE
    "BankofJapan",          # BoJ
    "elonmusk",             # crypto + tweets de impact
]

REDDIT_SUBREDDITS = [
    "wallstreetbets",
    "forex",
    "Forex",
    "Bitcoin",
    "CryptoCurrency",
]

# ── Macro indicators (FRED series IDs) ──────────────────────────────────────
FRED_SERIES = {
    "DGS10":  "10-Year Treasury yield",
    "DGS2":   "2-Year Treasury yield",
    "T10Y2Y": "Yield curve spread (10Y - 2Y)",
    "DTWEXBGS": "Trade Weighted USD Index",
    "VIXCLS": "VIX",
    "DFF":    "Federal Funds Rate",
    "M2SL":   "M2 Money Supply",
}

# ── Feature engineering ─────────────────────────────────────────────────────
LABEL_HORIZONS_HOURS = [4, 24, 72]   # predict return pe 4h, 24h, 72h
FEATURE_LOOKBACK_BARS = 200          # cati bare istoric folosim pt features

# ── Model training ──────────────────────────────────────────────────────────
TRAIN_WALK_WINDOW_DAYS = 365         # 1 an train
TRAIN_TEST_WINDOW_DAYS = 30          # 1 luna validate
MIN_AUC_THRESHOLD      = 0.45        # cobori pentru testing/demo
                                     # ATENȚIE: < 0.55 = model fără edge real,
                                     # NU folosi pentru capital propriu


def show_status():
    """Print configurarea curenta (fara secret keys)."""
    print(f"DB:        {DB_PATH}")
    print(f"Symbols:   {len(SYMBOLS)}")
    print(f"TFs:       {TIMEFRAMES}")
    print(f"History:   {HISTORICAL_YEARS} ani")
    print(f"API keys configured:")
    print(f"  AlphaVantage: {'YES' if ALPHA_VANTAGE_KEY else 'NO'}")
    print(f"  NewsAPI:      {'YES' if NEWSAPI_KEY else 'NO'}")
    print(f"  Twitter:      {'YES' if TWITTER_BEARER_TOKEN else 'NO'}")
    print(f"  FRED:         {'YES' if FRED_KEY else 'NO'}")
    print(f"  Reddit:       {'YES' if REDDIT_CLIENT_ID else 'NO'}")


if __name__ == "__main__":
    show_status()
