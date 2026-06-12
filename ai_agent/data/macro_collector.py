"""
ai_agent/data/macro_collector.py — date macro din FRED (gratuit).

FRED series ID-uri configurate in config.FRED_SERIES. Inregistrare cont:
    https://fred.stlouisfed.org/docs/api/api_key.html
"""
from __future__ import annotations
import sys
import time
import logging
from datetime import datetime, timezone

import requests

from ai_agent.config import FRED_KEY, FRED_SERIES
from ai_agent.db.schema import init_db, get_conn

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)


def fetch_fred_series(series_id: str, start: str = "2015-01-01") -> int:
    """Descarca tot istoricul unui series ID din FRED."""
    if not FRED_KEY:
        log.error("FRED_KEY lipseste din env")
        return 0
    try:
        r = requests.get(
            "https://api.stlouisfed.org/fred/series/observations",
            params={
                "series_id":         series_id,
                "api_key":           FRED_KEY,
                "file_type":         "json",
                "observation_start": start,
            },
            timeout=30,
        )
        r.raise_for_status()
        data = r.json()
    except Exception as e:
        log.error(f"FRED fetch {series_id}: {e}")
        return 0

    obs = data.get("observations", [])
    rows = []
    for o in obs:
        try:
            val = float(o["value"])
        except (ValueError, KeyError):
            continue  # '.' = missing
        try:
            ts = int(datetime.strptime(o["date"], "%Y-%m-%d")
                     .replace(tzinfo=timezone.utc).timestamp())
        except Exception:
            continue
        rows.append((series_id, ts, val))

    if not rows:
        log.warning(f"  {series_id}: 0 observatii valide")
        return 0
    with get_conn() as conn:
        conn.executemany(
            "INSERT OR REPLACE INTO macro (series_id, ts, value) VALUES (?,?,?)",
            rows,
        )
    log.info(f"  {series_id}: {len(rows)} observatii (range {obs[0]['date']}..{obs[-1]['date']})")
    return len(rows)


def fetch_all(start: str = "2015-01-01") -> dict:
    init_db()
    res = {}
    for sid, desc in FRED_SERIES.items():
        log.info(f"Fetching {sid} ({desc})")
        res[sid] = fetch_fred_series(sid, start)
        time.sleep(0.3)   # politicos
    return res


if __name__ == "__main__":
    init_db()
    if len(sys.argv) >= 2:
        fetch_fred_series(sys.argv[1])
    else:
        res = fetch_all()
        print()
        print("=" * 60)
        for sid, n in res.items():
            print(f"  {sid}: {n} obs")
