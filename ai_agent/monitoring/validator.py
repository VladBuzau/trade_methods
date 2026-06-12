"""
ai_agent/monitoring/validator.py — testeaza statistic daca modelul AI are edge real.

3 metode:
  1. Backtest signal-by-signal pe date OUT-OF-SAMPLE (ultimele N zile)
  2. Comparatie vs baselines: random, always-BUY, always-SELL
  3. Binomial p-value: e win rate diferit semnificativ de 50%?

Output: raport text cu concluzie clara ("AI e RANDOM" / "AI are EDGE STATISTIC").
"""
from __future__ import annotations
import logging
from datetime import datetime, timezone, timedelta
import math

import numpy as np
import pandas as pd

from ai_agent.db.schema import init_db, get_conn
from ai_agent.features.builder import compute_features_for_bar
from ai_agent.models.inference import load_model, predict_dual

log = logging.getLogger(__name__)


def _binomial_p_value(wins: int, total: int, p0: float = 0.5) -> float:
    """
    P-value pentru "este win_rate diferit de p0?" (two-tailed).
    Foloseste normal approximation pentru N mare (>30).
    """
    if total <= 0:
        return 1.0
    p_hat = wins / total
    se = math.sqrt(p0 * (1 - p0) / total)
    if se == 0:
        return 1.0
    z = abs(p_hat - p0) / se
    # P(|Z| > z) = 2 * (1 - Phi(z))
    p_value = 2 * (1 - 0.5 * (1 + math.erf(z / math.sqrt(2))))
    return p_value


def _evaluate_predictor(name: str, signals: list[tuple],
                       prices_after: list[dict]) -> dict:
    """
    signals = list of (bar_idx, "BUY"|"SELL"|"HOLD")
    prices_after = pentru fiecare bar_idx, dict cu high/low/close N bare in viitor.
    Calculeaza wins/losses + statistici.
    """
    actionable = [(idx, sig) for idx, sig in signals if sig in ("BUY", "SELL")]
    wins = 0
    losses = 0
    pnl_total = 0.0
    pnl_per_trade = []

    for idx, sig in actionable:
        if idx >= len(prices_after) or not prices_after[idx]:
            continue
        info = prices_after[idx]
        entry = info["entry"]
        max_up = info["max_up"]
        max_down = info["max_down"]
        # Daca a mers mai mult in directia semnalului → win
        if sig == "BUY":
            won = max_up > max_down
            pnl = max_up - max_down  # PnL aprox in pip units
        else:  # SELL
            won = max_down > max_up
            pnl = max_down - max_up
        if won:
            wins += 1
        else:
            losses += 1
        pnl_total += pnl
        pnl_per_trade.append(pnl / entry * 10000)  # in pips approx

    total = wins + losses
    win_rate = (wins / total * 100) if total else 0.0
    p_value = _binomial_p_value(wins, total) if total > 30 else 1.0
    avg_pnl_pips = float(np.mean(pnl_per_trade)) if pnl_per_trade else 0.0

    return {
        "name":         name,
        "n_signals":    len(actionable),
        "wins":         wins,
        "losses":       losses,
        "hold_count":   len(signals) - len(actionable),
        "win_rate":     round(win_rate, 1),
        "avg_pnl_pips": round(avg_pnl_pips, 2),
        "p_value":      round(p_value, 4),
        "significant":  p_value < 0.05,
    }


def run_validation(symbol: str, tf: str, horizon: int = 24,
                   oos_days: int = 60, eval_bars: int = 24) -> dict:
    """
    Pipeline:
      1. Genereaza N predictii AI pe ultimele oos_days zile
      2. Genereaza N predictii Random (50/50)
      3. Genereaza N predictii Always-BUY si Always-SELL
      4. Evalueaza fiecare pe urmatoarele eval_bars bare
      5. Compara + p-value
    """
    init_db()
    log.info(f"Validare {symbol}/{tf} h{horizon} pe ultimele {oos_days}d")

    # Load prices
    with get_conn() as conn:
        rows = conn.execute(
            "SELECT ts, open, high, low, close, volume FROM prices "
            "WHERE symbol=? AND timeframe=? ORDER BY ts",
            (symbol, tf),
        ).fetchall()
    if not rows:
        return {"error": "No prices in DB"}

    df = pd.DataFrame([dict(r) for r in rows])

    # OOS window: ultimele oos_days
    bars_per_day = {"M1":1440, "M5":288, "M15":96, "M30":48,
                    "H1":24, "H4":6, "D1":1}.get(tf, 24)
    n_oos = oos_days * bars_per_day
    if len(df) < n_oos + 200 + eval_bars:
        return {"error": f"Insuficiente bare ({len(df)} < {n_oos + 200 + eval_bars})"}

    # Start: 200 bare in trecut pt features, terminam cu eval_bars buffer la final
    start_idx = len(df) - n_oos - eval_bars
    end_idx   = len(df) - eval_bars

    # Pregateste prices_after pentru fiecare bara din OOS
    prices_after = {}
    for i in range(start_idx, end_idx):
        entry = float(df["close"].iloc[i])
        future_h = df["high"].iloc[i+1:i+1+eval_bars].max()
        future_l = df["low"].iloc[i+1:i+1+eval_bars].min()
        prices_after[i] = {
            "entry":    entry,
            "max_up":   float(future_h) - entry,
            "max_down": entry - float(future_l),
        }

    # 1. AI predictions
    ai_signals = []
    skipped_no_model = 0
    bundle_long  = load_model(symbol, tf, horizon, "y_hit_long")
    bundle_short = load_model(symbol, tf, horizon, "y_hit_short")
    if bundle_long is None or bundle_short is None:
        return {"error": f"Modele lipsa pentru {symbol}/{tf}/h{horizon}"}

    df_indexed = df.set_index(df.index)  # int index
    for i in range(start_idx, end_idx):
        if i % 100 == 0:
            log.info(f"  bara {i - start_idx}/{n_oos}")
        try:
            feats = compute_features_for_bar(df_indexed, i)
            if feats is None:
                continue
            pred = predict_dual(feats, symbol, tf, horizon)
            if pred:
                ai_signals.append((i, pred["signal"]))
            else:
                skipped_no_model += 1
        except Exception:
            pass

    # Baselines deterministe (fara random):
    always_buy  = [(i, "BUY") for i in range(start_idx, end_idx)]
    always_sell = [(i, "SELL") for i in range(start_idx, end_idx)]

    # Evaluate
    results = {
        "ai":          _evaluate_predictor("AI (XGBoost)", ai_signals, prices_after),
        "always_buy":  _evaluate_predictor("Always BUY", always_buy, prices_after),
        "always_sell": _evaluate_predictor("Always SELL", always_sell, prices_after),
    }

    # Concluzie automata
    ai = results["ai"]
    conclusion = []
    if ai["n_signals"] < 30:
        conclusion.append("⚠ Prea putine semnale AI pentru test statistic (<30)")
    if ai["p_value"] >= 0.05:
        conclusion.append("❌ AI win rate NU e diferit statistic de 50% — fara edge")
    else:
        conclusion.append(f"✓ AI win rate {ai['win_rate']}% e diferit de 50% (p={ai['p_value']:.3f})")

    # Compara vs trivial (always BUY/SELL)
    best_trivial = max(results["always_buy"], results["always_sell"],
                       key=lambda r: r["win_rate"])
    diff_vs_trivial = ai["win_rate"] - best_trivial["win_rate"]
    if diff_vs_trivial < 1:
        conclusion.append(f"❌ AI nu bate strategia triviala '{best_trivial['name']}' "
                          f"({best_trivial['win_rate']}%)")
    else:
        conclusion.append(f"✓ AI bate {best_trivial['name']} cu {diff_vs_trivial:+.1f}%")

    verdict = "❌ FARA EDGE"
    if ai["p_value"] < 0.05 and diff_vs_trivial > 0:
        verdict = "✅ POTENTIAL EDGE"
    elif ai["p_value"] < 0.10:
        verdict = "⚠ MARGINAL"

    return {
        "symbol":      symbol,
        "tf":          tf,
        "horizon_h":   horizon,
        "oos_days":    oos_days,
        "n_bars_oos":  n_oos,
        "eval_bars":   eval_bars,
        "verdict":     verdict,
        "conclusion":  conclusion,
        "results":     results,
        "skipped_no_model": skipped_no_model,
    }


if __name__ == "__main__":
    import sys, json
    sym = sys.argv[1] if len(sys.argv) > 1 else "EURUSD"
    tf  = sys.argv[2] if len(sys.argv) > 2 else "H1"
    hz  = int(sys.argv[3]) if len(sys.argv) > 3 else 24
    days = int(sys.argv[4]) if len(sys.argv) > 4 else 60
    r = run_validation(sym, tf, hz, days)
    print(json.dumps(r, indent=2, default=str))
