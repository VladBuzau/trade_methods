"""
ai_agent/analysis/news_impact.py — analizator istoric de impact al stirilor.

Foloseste:
  - Date economice istorice (NFP, FOMC, CPI, ECB rate decisions, etc.)
    — generate programatic pentru ultimii N ani din date cunoscute
  - Pretul istoric MT5 deja descarcat in DB

Pentru fiecare eveniment, analizeaza:
  - Pretul cu T-5min, T-1min (inainte)
  - T+5min, T+15min, T+30min, T+1h, T+4h (dupa)
  - Calculeaza miscari in pips
  - Agregheaza per tip eveniment

Output: "NFP face EUR/USD sa se miste mediu ±28 pips in primele 30 min".
"""
from __future__ import annotations
import logging
from datetime import datetime, timezone, timedelta
import calendar

import numpy as np
import pandas as pd

from ai_agent.db.schema import init_db, get_conn

log = logging.getLogger(__name__)


# ─── Generator de date evenimente recurente ─────────────────────────────────
def _first_friday(year: int, month: int) -> datetime:
    """Prima vinerea din lună."""
    c = calendar.monthcalendar(year, month)
    for week in c:
        if week[calendar.FRIDAY]:
            return datetime(year, month, week[calendar.FRIDAY], 12, 30, tzinfo=timezone.utc)
    return None


def _wed_in_month(year: int, month: int, ordinal: int) -> datetime:
    """A N-a miercuri din lună (FOMC e de obicei a 2-a sau a 3-a miercuri)."""
    c = calendar.monthcalendar(year, month)
    wednesdays = [week[calendar.WEDNESDAY] for week in c if week[calendar.WEDNESDAY]]
    if ordinal <= len(wednesdays):
        return datetime(year, month, wednesdays[ordinal-1], 18, 0, tzinfo=timezone.utc)
    return None


def generate_known_events(start_year: int = 2022, end_year: int = 2026) -> list[dict]:
    """
    Genereaza lista cu evenimente economice MAJORE recurente:
      - NFP (US Non-Farm Payrolls): 1st Friday/month 12:30 UTC (13:30 in DST)
      - US CPI: cca 10-15 a fiecărei luni 12:30 UTC
      - FOMC: 8 meeting-uri pe an (Jan, Mar, May, Jun, Jul, Sep, Nov, Dec) - 18:00 UTC
      - ECB: ~8 meeting-uri pe an - 12:15 UTC
    Returneaza events sortate cronologic.
    """
    events = []
    for year in range(start_year, end_year + 1):
        for month in range(1, 13):
            # NFP (1st Friday)
            nfp = _first_friday(year, month)
            if nfp:
                events.append({
                    "name": "NFP",
                    "full_name": "US Non-Farm Payrolls",
                    "currency": "USD",
                    "impact": "high",
                    "ts": int(nfp.timestamp()),
                })
            # US CPI (cca a 10-15-a, miercuri sau joi 12:30)
            try:
                cpi = datetime(year, month, 12, 12, 30, tzinfo=timezone.utc)
                events.append({
                    "name": "CPI",
                    "full_name": "US CPI (Consumer Price Index)",
                    "currency": "USD",
                    "impact": "high",
                    "ts": int(cpi.timestamp()),
                })
            except Exception:
                pass

        # FOMC meetings (aproximativ luni: Jan, Mar, May, Jun, Jul, Sep, Nov, Dec)
        fomc_months = [1, 3, 5, 6, 7, 9, 11, 12]
        for fmonth in fomc_months:
            # FOMC e de obicei a 2-a sau a 3-a miercuri
            fomc = _wed_in_month(year, fmonth, 2) or _wed_in_month(year, fmonth, 3)
            if fomc:
                events.append({
                    "name": "FOMC",
                    "full_name": "FOMC Rate Decision",
                    "currency": "USD",
                    "impact": "high",
                    "ts": int(fomc.timestamp()),
                })

        # ECB Rate Decision (1st Thursday of meeting months)
        ecb_months = [1, 3, 4, 6, 7, 9, 10, 12]
        for emonth in ecb_months:
            c = calendar.monthcalendar(year, emonth)
            thursdays = [w[calendar.THURSDAY] for w in c if w[calendar.THURSDAY]]
            if thursdays:
                # Usually first Thursday
                ecb = datetime(year, emonth, thursdays[0], 12, 15, tzinfo=timezone.utc)
                events.append({
                    "name": "ECB",
                    "full_name": "ECB Rate Decision",
                    "currency": "EUR",
                    "impact": "high",
                    "ts": int(ecb.timestamp()),
                })

    events.sort(key=lambda x: x["ts"])
    return events


# ─── Match cu pretul si calcul impact ───────────────────────────────────────
def _get_price_near(symbol: str, tf: str, ts: int,
                    minutes_offset: int) -> float | None:
    """Returneaza inchiderea barei cea mai apropiata de (ts + minutes_offset)."""
    target_ts = ts + minutes_offset * 60
    with get_conn() as conn:
        # Cauta bara cu ts cel mai apropiat (in jur de target ±1 ora)
        row = conn.execute(
            "SELECT close FROM prices WHERE symbol=? AND timeframe=? "
            "AND ts >= ? AND ts <= ? ORDER BY ABS(ts - ?) LIMIT 1",
            (symbol, tf, target_ts - 3600, target_ts + 3600, target_ts),
        ).fetchone()
    return float(row["close"]) if row else None


def compute_event_impact(event: dict, symbol: str, tf: str) -> dict | None:
    """
    Pentru un eveniment + simbol, calculeaza miscarile in pips la diferite
    orizonturi de timp dupa publicare.
    """
    ts = event["ts"]
    price_before = _get_price_near(symbol, tf, ts, minutes_offset=-5)
    if price_before is None or price_before <= 0:
        return None
    horizons_min = [5, 15, 30, 60, 240]
    moves = {}
    for m in horizons_min:
        p = _get_price_near(symbol, tf, ts, minutes_offset=m)
        if p is None:
            moves[f"move_{m}m_pips"] = None
            continue
        # Pips conversion (forex 4-digit = pip × 10000, JPY × 100, gold × 10)
        diff = p - price_before
        if "JPY" in symbol:
            pips = diff * 100
        elif "XAU" in symbol or "GOLD" in symbol:
            pips = diff * 10
        else:
            pips = diff * 10000
        moves[f"move_{m}m_pips"] = round(pips, 1)
    return moves


def analyze_event_type(event_name: str, symbol: str, tf: str = "M5",
                       start_year: int = 2022, end_year: int = 2026) -> dict:
    """
    Pentru toate evenimentele cu name=event_name din intervalul de timp,
    calculeaza statistici de impact pe (symbol, tf).
    """
    all_events = generate_known_events(start_year, end_year)
    matching = [e for e in all_events if e["name"] == event_name]

    impacts = []
    for ev in matching:
        imp = compute_event_impact(ev, symbol, tf)
        if imp:
            imp["ts"] = ev["ts"]
            imp["dt"] = datetime.fromtimestamp(ev["ts"], tz=timezone.utc).isoformat()
            impacts.append(imp)

    if not impacts:
        return {"event": event_name, "symbol": symbol, "n": 0,
                "error": "Niciun match (lipsa date pret la momentele evenimentelor)"}

    # Agregare per horizon
    horizons = ["move_5m_pips", "move_15m_pips", "move_30m_pips",
                "move_60m_pips", "move_240m_pips"]
    stats = {}
    for h in horizons:
        vals = [i[h] for i in impacts if i.get(h) is not None]
        if not vals:
            stats[h] = {"n": 0}
            continue
        abs_vals = [abs(v) for v in vals]
        ups = sum(1 for v in vals if v > 0)
        downs = sum(1 for v in vals if v < 0)
        stats[h] = {
            "n":             len(vals),
            "mean_pips":     round(float(np.mean(vals)), 1),
            "mean_abs_pips": round(float(np.mean(abs_vals)), 1),
            "median_pips":   round(float(np.median(vals)), 1),
            "max_up_pips":   round(float(np.max(vals)), 1),
            "max_down_pips": round(float(np.min(vals)), 1),
            "ups":           ups,
            "downs":         downs,
            "bias_pct":      round(ups / len(vals) * 100, 1),
            "std":           round(float(np.std(vals)), 1),
        }

    return {
        "event":      event_name,
        "symbol":     symbol,
        "tf":         tf,
        "n":          len(impacts),
        "n_total":    len(matching),
        "stats":      stats,
        "events":     impacts[-30:],   # ultimele 30 pentru afisare
    }


def analyze_all_events(symbol: str, tf: str = "M5",
                       start_year: int = 2022, end_year: int = 2026) -> dict:
    """Ruleaza analiza pentru toate tipurile de evenimente cunoscute."""
    init_db()
    results = {}
    for event_name in ["NFP", "CPI", "FOMC", "ECB"]:
        r = analyze_event_type(event_name, symbol, tf, start_year, end_year)
        if r.get("n", 0) > 0:
            results[event_name] = r
    return {
        "symbol":  symbol,
        "tf":      tf,
        "period":  f"{start_year}-{end_year}",
        "results": results,
    }


if __name__ == "__main__":
    import sys, json
    sym = sys.argv[1] if len(sys.argv) > 1 else "EURUSD"
    tf  = sys.argv[2] if len(sys.argv) > 2 else "M5"
    r = analyze_all_events(sym, tf)
    print(json.dumps(r, indent=2, default=str))
