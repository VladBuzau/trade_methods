"""
ai_agent/data/news_collector.py — colector de stiri din surse multiple.

Surse implementate:
  - AlphaVantage News (FREE tier 25/zi; $50/mo unlimited)
  - NewsAPI.org           ($150/mo)
  - Reddit prin praw     (FREE)

Twitter/X separat (twitter_collector.py — necesita auth flow special).

Usage:
    python -m ai_agent.data.news_collector alphavantage EURUSD
    python -m ai_agent.data.news_collector reddit wallstreetbets 50
"""
from __future__ import annotations
import sys
import time
import json
import logging
import hashlib
from datetime import datetime, timezone, timedelta

import requests

from ai_agent.config import (
    ALPHA_VANTAGE_KEY, NEWSAPI_KEY,
    REDDIT_CLIENT_ID, REDDIT_CLIENT_SECRET, REDDIT_USER_AGENT,
    REDDIT_SUBREDDITS, SYMBOLS,
)
from ai_agent.db.schema import init_db, get_conn

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)


def _insert_news(rows: list[dict]) -> int:
    """Insert in tabela news cu dedup pe (source, external_id)."""
    if not rows:
        return 0
    with get_conn() as conn:
        c = conn.executemany(
            "INSERT OR IGNORE INTO news "
            "(source, external_id, ts, symbols, title, body, url, sentiment, impact_score, raw_json) "
            "VALUES (?,?,?,?,?,?,?,?,?,?)",
            [(r["source"], r["external_id"], r["ts"], r.get("symbols", ""),
              r.get("title", ""), r.get("body", ""), r.get("url", ""),
              r.get("sentiment"), r.get("impact_score"),
              json.dumps(r.get("raw", {}), default=str))
             for r in rows],
        )
        return c.rowcount


# ─── AlphaVantage News ──────────────────────────────────────────────────────
def fetch_alphavantage(symbol: str, limit: int = 50) -> int:
    """
    AlphaVantage NEWS_SENTIMENT. Returneaza articole + sentiment-ul lor (deja
    calculat de AlphaVantage). Free tier: 25 calls/zi.
    Doc: https://www.alphavantage.co/documentation/#news-sentiment
    """
    if not ALPHA_VANTAGE_KEY:
        log.error("ALPHA_VANTAGE_KEY lipseste din env")
        return 0

    # AlphaVantage foloseste ticker-uri (gen 'AAPL'), nu forex pairs.
    # Pentru forex, folosim parametrul `topics`. Pentru noi ar fi:
    #   topics=forex, financial_markets
    # Mappam symbolul nostru la topics.
    if symbol in ("EURUSD", "GBPUSD", "USDJPY", "AUDUSD", "USDCAD"):
        params = {"function": "NEWS_SENTIMENT", "topics": "forex,economy_macro",
                  "limit": limit, "apikey": ALPHA_VANTAGE_KEY}
    elif symbol in ("BTCUSD", "ETHUSD"):
        params = {"function": "NEWS_SENTIMENT", "topics": "blockchain,economy_macro",
                  "limit": limit, "apikey": ALPHA_VANTAGE_KEY}
    elif symbol == "XAUUSD":
        params = {"function": "NEWS_SENTIMENT", "topics": "economy_macro,financial_markets",
                  "limit": limit, "apikey": ALPHA_VANTAGE_KEY}
    else:
        params = {"function": "NEWS_SENTIMENT", "tickers": symbol,
                  "limit": limit, "apikey": ALPHA_VANTAGE_KEY}

    try:
        r = requests.get("https://www.alphavantage.co/query",
                         params=params, timeout=30)
        r.raise_for_status()
        data = r.json()
    except Exception as e:
        log.error(f"AlphaVantage fetch failed: {e}")
        return 0

    feed = data.get("feed", [])
    if not feed:
        log.info(f"AlphaVantage: 0 articole pt {symbol}")
        return 0

    rows = []
    for art in feed:
        ts_str = art.get("time_published", "")
        try:
            ts = int(datetime.strptime(ts_str, "%Y%m%dT%H%M%S")
                     .replace(tzinfo=timezone.utc).timestamp())
        except Exception:
            continue
        # External ID: hash din URL ca sa fie unic
        ext_id = hashlib.md5(art.get("url", ts_str).encode()).hexdigest()[:16]
        # Sentiment overall: AlphaVantage da [-1, 1]
        sent = float(art.get("overall_sentiment_score", 0) or 0)
        # Lista simboluri legate
        related_syms = [t.get("ticker", "") for t in art.get("ticker_sentiment", [])]
        rows.append({
            "source": "alphavantage",
            "external_id": ext_id,
            "ts": ts,
            "symbols": ",".join(related_syms[:5]),
            "title": art.get("title", "")[:500],
            "body": art.get("summary", "")[:2000],
            "url": art.get("url", "")[:500],
            "sentiment": sent,
            "impact_score": None,
            "raw": art,
        })
    n = _insert_news(rows)
    log.info(f"AlphaVantage {symbol}: {n} articole noi (din {len(rows)})")
    return n


# ─── NewsAPI.org ────────────────────────────────────────────────────────────
def fetch_newsapi(query: str, days_back: int = 1) -> int:
    """NewsAPI: doar query general (gen 'EUR USD inflation')."""
    if not NEWSAPI_KEY:
        log.error("NEWSAPI_KEY lipseste din env")
        return 0
    from_dt = (datetime.now(timezone.utc) - timedelta(days=days_back)).date().isoformat()
    try:
        r = requests.get("https://newsapi.org/v2/everything", params={
            "q": query, "from": from_dt, "language": "en",
            "sortBy": "publishedAt", "pageSize": 100,
            "apiKey": NEWSAPI_KEY,
        }, timeout=30)
        r.raise_for_status()
        data = r.json()
    except Exception as e:
        log.error(f"NewsAPI fetch: {e}")
        return 0

    rows = []
    for art in data.get("articles", []):
        try:
            ts = int(datetime.fromisoformat(
                art["publishedAt"].replace("Z", "+00:00")).timestamp())
        except Exception:
            continue
        ext_id = hashlib.md5(art["url"].encode()).hexdigest()[:16]
        rows.append({
            "source": "newsapi",
            "external_id": ext_id,
            "ts": ts,
            "symbols": query,
            "title": (art.get("title") or "")[:500],
            "body":  (art.get("description") or "")[:2000],
            "url":   (art.get("url") or "")[:500],
            "sentiment": None,   # FinBERT separat
            "impact_score": None,
            "raw": art,
        })
    n = _insert_news(rows)
    log.info(f"NewsAPI '{query}': {n} articole noi")
    return n


# ─── Reddit ─────────────────────────────────────────────────────────────────
def fetch_reddit(subreddit: str, limit: int = 100) -> int:
    """Folosim praw — necesita REDDIT_CLIENT_ID + SECRET (free)."""
    try:
        import praw
    except ImportError:
        log.error("praw nu e instalat: pip install praw")
        return 0
    if not REDDIT_CLIENT_ID or not REDDIT_CLIENT_SECRET:
        log.error("REDDIT_CLIENT_ID/SECRET lipsesc din env")
        return 0

    reddit = praw.Reddit(
        client_id=REDDIT_CLIENT_ID,
        client_secret=REDDIT_CLIENT_SECRET,
        user_agent=REDDIT_USER_AGENT,
    )
    rows = []
    try:
        for post in reddit.subreddit(subreddit).new(limit=limit):
            rows.append({
                "source": "reddit",
                "external_id": post.id,
                "ts": int(post.created_utc),
                "symbols": "",   # le inferam mai tarziu cu NER
                "title": post.title[:500],
                "body":  (post.selftext or "")[:2000],
                "url":   f"https://reddit.com{post.permalink}",
                "sentiment": None,
                "impact_score": float(post.score),  # upvotes ca proxy initial
                "raw": {"score": post.score, "comments": post.num_comments,
                        "subreddit": subreddit},
            })
    except Exception as e:
        log.error(f"Reddit fetch r/{subreddit}: {e}")
        return 0
    n = _insert_news(rows)
    log.info(f"Reddit r/{subreddit}: {n} posturi noi (din {len(rows)})")
    return n


def fetch_all_reddit(limit_per_sub: int = 50) -> dict:
    init_db()
    res = {}
    for sub in REDDIT_SUBREDDITS:
        res[sub] = fetch_reddit(sub, limit_per_sub)
        time.sleep(1)
    return res


if __name__ == "__main__":
    init_db()
    if len(sys.argv) < 2:
        print("Usage: news_collector.py <alphavantage|newsapi|reddit> [arg] [limit]")
        sys.exit(1)
    source = sys.argv[1].lower()
    arg    = sys.argv[2] if len(sys.argv) >= 3 else ""
    limit  = int(sys.argv[3]) if len(sys.argv) >= 4 else 50
    if source == "alphavantage":
        fetch_alphavantage(arg or "EURUSD", limit)
    elif source == "newsapi":
        fetch_newsapi(arg or "EUR USD inflation", days_back=1)
    elif source == "reddit":
        fetch_reddit(arg or "forex", limit)
    elif source == "all_reddit":
        res = fetch_all_reddit(limit)
        for k, v in res.items():
            print(f"  r/{k}: {v}")
    else:
        print(f"Sursa necunoscuta: {source}")
