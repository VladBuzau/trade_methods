"""
ai_agent/monitoring/predictions_log.py — analiza predictiilor live.

Pentru fiecare predictie veche (in DB), o compara cu pretul actual pentru a
calcula accuracy real. Daca model devine inacurat → trigger retrain.

Usage:
    python -m ai_agent.monitoring.predictions_log
"""
from __future__ import annotations
import logging
from datetime import datetime, timezone, timedelta

import pandas as pd

from ai_agent.db.schema import init_db, get_conn

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)


def evaluate_predictions(days_back: int = 7) -> dict:
    """
    Pentru predictiile mai vechi de `horizon_h` ore (ca sa avem outcome),
    determina daca au fost corecte.
    """
    init_db()
    cutoff_ts = int((datetime.now(timezone.utc) - timedelta(days=days_back)).timestamp())

    with get_conn() as conn:
        preds = conn.execute(
            "SELECT * FROM predictions WHERE ts >= ? ORDER BY ts",
            (cutoff_ts,),
        ).fetchall()

    if not preds:
        return {"n": 0, "message": "Nicio predictie in fereastra"}

    results_per_model: dict[str, dict] = {}

    for p in preds:
        model_id = p["model_id"]
        horizon_h = p["horizon_h"]
        ts = p["ts"]
        symbol = p["symbol"]
        tf = p["timeframe"]
        pred_dir = p["pred_dir"]
        confidence = p["confidence"]

        # Pretul la momentul predictiei vs pretul cu horizon_h ore mai tarziu
        with get_conn() as conn:
            entry_price = conn.execute(
                "SELECT close FROM prices WHERE symbol=? AND timeframe=? AND ts<=? "
                "ORDER BY ts DESC LIMIT 1", (symbol, tf, ts),
            ).fetchone()
            exit_ts = ts + horizon_h * 3600
            exit_price = conn.execute(
                "SELECT close FROM prices WHERE symbol=? AND timeframe=? AND ts>=? "
                "ORDER BY ts ASC LIMIT 1", (symbol, tf, exit_ts),
            ).fetchone()

        if not entry_price or not exit_price:
            continue

        entry = float(entry_price["close"])
        exit  = float(exit_price["close"])
        ret_pct = (exit / entry - 1.0) * 100

        # Predictia a fost corecta?
        correct = (
            (pred_dir == 1 and ret_pct > 0) or
            (pred_dir == -1 and ret_pct < 0) or
            (pred_dir == 0)   # HOLD nu se evalueaza
        )

        r = results_per_model.setdefault(model_id, {
            "n": 0, "correct": 0, "wins": [], "losses": [],
            "by_confidence_bucket": {"50-60": [0, 0], "60-70": [0, 0],
                                      "70-80": [0, 0], "80-90": [0, 0],
                                      "90-100": [0, 0]},
        })
        if pred_dir == 0:
            continue
        r["n"] += 1
        if correct:
            r["correct"] += 1
            r["wins"].append(round(ret_pct, 3))
        else:
            r["losses"].append(round(ret_pct, 3))

        # Confidence bucket
        if confidence < 60: bucket = "50-60"
        elif confidence < 70: bucket = "60-70"
        elif confidence < 80: bucket = "70-80"
        elif confidence < 90: bucket = "80-90"
        else: bucket = "90-100"
        r["by_confidence_bucket"][bucket][0] += 1
        if correct:
            r["by_confidence_bucket"][bucket][1] += 1

    # Sumar
    summary = {}
    for model_id, r in results_per_model.items():
        acc = r["correct"] / r["n"] * 100 if r["n"] > 0 else 0.0
        wins_avg = sum(r["wins"]) / len(r["wins"]) if r["wins"] else 0.0
        losses_avg = sum(r["losses"]) / len(r["losses"]) if r["losses"] else 0.0
        buckets_str = {}
        for b, (n, c) in r["by_confidence_bucket"].items():
            buckets_str[b] = f"{c}/{n} ({c/n*100:.0f}%)" if n > 0 else "—"
        summary[model_id] = {
            "predictions":  r["n"],
            "accuracy":     round(acc, 1),
            "wins":         len(r["wins"]),
            "losses":       len(r["losses"]),
            "avg_win_pct":  round(wins_avg, 3),
            "avg_loss_pct": round(losses_avg, 3),
            "by_confidence": buckets_str,
        }
    return {"n": len(preds), "models": summary, "days_back": days_back}


def print_report(days_back: int = 7):
    r = evaluate_predictions(days_back)
    print("=" * 60)
    print(f"PREDICTIONS REPORT — last {days_back}d ({r['n']} predictions)")
    print("=" * 60)
    for model_id, s in r.get("models", {}).items():
        print(f"\nModel: {model_id}")
        print(f"  Total:     {s['predictions']}")
        print(f"  Acuratete: {s['accuracy']}%")
        print(f"  Wins:      {s['wins']} (mediu +{s['avg_win_pct']}%)")
        print(f"  Losses:    {s['losses']} (mediu {s['avg_loss_pct']}%)")
        print(f"  Per confidence bucket:")
        for b, info in s["by_confidence"].items():
            print(f"    {b}%: {info}")


if __name__ == "__main__":
    import sys
    days = int(sys.argv[1]) if len(sys.argv) > 1 else 7
    print_report(days)
