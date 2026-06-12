"""
quick_backtest.py — backtester standalone pentru iterare rapida pe strategii.

Replica logica engine-ului ChartVisualizer:
  pentru fiecare bara (fereastra glisanta) -> analyze() -> daca BUY/SELL,
  simuleaza trade-ul bara cu bara pana la SL/TP/timeout.

Salveaza rezultate in ai_agent/analysis/results/<strategy>_<symbol>_<tf>.json
si printeaza metrici + diagnostic.

Usage: python -m ai_agent.analysis.quick_backtest vwap_reversion EURUSD M15 3000
"""
from __future__ import annotations
import sys, json, time
from pathlib import Path
from collections import Counter

import numpy as np
import pandas as pd

RESULTS_DIR = Path(__file__).resolve().parent / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def run(strategy_key: str, symbol: str, tf: str, n_bars: int = 3000,
        window: int = 300, max_hold_bars: int = 16,
        min_conf: float = 55.0, spread_pips: float = 1.2,
        risk_dollars: float = 50.0, elements: dict | None = None) -> dict:
    # Setup path pentru ChartVisualizer
    cv = Path(__file__).resolve().parents[2] / "ChartVisualizer"
    if str(cv) not in sys.path:
        sys.path.insert(0, str(cv))

    import MetaTrader5 as mt5
    mt5.initialize()
    _TF = {"M1": mt5.TIMEFRAME_M1, "M5": mt5.TIMEFRAME_M5, "M15": mt5.TIMEFRAME_M15,
           "M30": mt5.TIMEFRAME_M30, "H1": mt5.TIMEFRAME_H1, "H4": mt5.TIMEFRAME_H4,
           "D1": mt5.TIMEFRAME_D1}
    rates = mt5.copy_rates_from_pos(symbol, _TF[tf], 0, n_bars)
    full = pd.DataFrame(rates)
    full["time"] = pd.to_datetime(full["time"], unit="s")
    full = full.set_index("time")
    full = full.rename(columns={"tick_volume": "volume"})

    import app
    import strategies as sp
    strat = sp.get_strategy(strategy_key)
    if strat is None:
        return {"error": f"strategy {strategy_key} not found"}

    if elements is None:
        elements = {k: (not lbl.upper().startswith("STRICT"))
                    for k, lbl in strat.elements.items()}

    # Pip size
    if "JPY" in symbol:
        pip = 0.01
    elif "XAU" in symbol:
        pip = 0.1
    else:
        pip = 0.0001
    spread = spread_pips * pip

    trades = []
    i = window
    N = len(full)
    while i < N - 1:
        # Slice istoric pana la bara i (fereastra glisanta)
        def mf(s, t, bars, _e=i):
            return full.iloc[max(0, _e - int(bars)):_e].copy(), "qbt"
        app.fetch = mf
        try:
            res = strat.analyze(symbol, [tf], bars=window, elements=elements,
                                min_confidence=min_conf)
        except Exception:
            i += 1
            continue
        sig = res.get("signal", "HOLD")
        if sig not in ("BUY", "SELL"):
            i += 1
            continue
        bft = res.get("best_tf") or {}
        sl = bft.get("sl")
        tp = bft.get("tp")
        if sl is None or tp is None:
            i += 1
            continue

        # Entry la open-ul barei urmatoare (no look-ahead)
        entry = float(full.iloc[i + 1]["open"])
        if sig == "BUY":
            entry += spread / 2     # cost
            sl_p, tp_p = float(sl), float(tp)
        else:
            entry -= spread / 2
            sl_p, tp_p = float(sl), float(tp)

        # Simuleaza bara cu bara
        outcome = None
        exit_p = entry
        exit_bar = i + 1
        for j in range(i + 1, min(i + 1 + max_hold_bars, N)):
            bar = full.iloc[j]
            hi, lo = float(bar["high"]), float(bar["low"])
            if sig == "BUY":
                if lo <= sl_p:
                    outcome, exit_p, exit_bar = "SL", sl_p, j; break
                if hi >= tp_p:
                    outcome, exit_p, exit_bar = "TP", tp_p, j; break
            else:
                if hi >= sl_p:
                    outcome, exit_p, exit_bar = "SL", sl_p, j; break
                if lo <= tp_p:
                    outcome, exit_p, exit_bar = "TP", tp_p, j; break
        if outcome is None:
            outcome = "TIMEOUT"
            exit_p = float(full.iloc[min(i + max_hold_bars, N - 1)]["close"])
            exit_bar = min(i + max_hold_bars, N - 1)

        # PnL in dolari (risc fix per trade)
        risk_dist = abs(entry - sl_p)
        if risk_dist <= 0:
            i += 1
            continue
        if sig == "BUY":
            pnl_price = exit_p - entry
        else:
            pnl_price = entry - exit_p
        # Position sized asa incat SL = -risk_dollars
        pnl = pnl_price / risk_dist * risk_dollars

        trades.append({
            "signal": sig, "entry": round(entry, 5), "sl": round(sl_p, 5),
            "tp": round(tp_p, 5), "exit": round(exit_p, 5), "outcome": outcome,
            "pnl": round(pnl, 2), "conf": res.get("confidence", 0),
            "bars_held": exit_bar - (i + 1),
        })
        # Avanseaza dincolo de trade (un trade odata, ca live)
        i = exit_bar + 1

    # Metrici
    if not trades:
        return {"strategy": strategy_key, "symbol": symbol, "tf": tf,
                "n_trades": 0, "msg": "0 trades"}
    wins = [t for t in trades if t["pnl"] > 0]
    losses = [t for t in trades if t["pnl"] <= 0]
    gp = sum(t["pnl"] for t in wins)
    gl = abs(sum(t["pnl"] for t in losses))
    pf = gp / gl if gl > 0 else 999
    oc = Counter(t["outcome"] for t in trades)
    # Equity curve + max DD
    eq = 10000.0
    peak = eq
    max_dd = 0.0
    for t in trades:
        eq += t["pnl"]
        peak = max(peak, eq)
        max_dd = max(max_dd, peak - eq)
    expectancy = sum(t["pnl"] for t in trades) / len(trades)

    result = {
        "strategy": strategy_key, "symbol": symbol, "tf": tf,
        "n_trades": len(trades),
        "win_rate": round(len(wins) / len(trades) * 100, 1),
        "profit_factor": round(pf, 2),
        "net_pnl": round(sum(t["pnl"] for t in trades), 2),
        "expectancy": round(expectancy, 2),
        "max_drawdown": round(max_dd, 2),
        "avg_bars_held": round(np.mean([t["bars_held"] for t in trades]), 1),
        "outcomes": dict(oc),
        "by_outcome": {
            o: {"n": sum(1 for t in trades if t["outcome"] == o),
                "avg_pnl": round(np.mean([t["pnl"] for t in trades if t["outcome"] == o]), 2),
                "total": round(sum(t["pnl"] for t in trades if t["outcome"] == o), 0)}
            for o in oc
        },
        "spread_pips": spread_pips,
    }
    # Salveaza
    out = RESULTS_DIR / f"{strategy_key}_{symbol}_{tf}.json"
    out.write_text(json.dumps({**result, "trades": trades[-200:]}, indent=2), encoding="utf-8")
    result["saved"] = str(out)
    return result


if __name__ == "__main__":
    sk = sys.argv[1] if len(sys.argv) > 1 else "vwap_reversion"
    sym = sys.argv[2] if len(sys.argv) > 2 else "EURUSD"
    tf = sys.argv[3] if len(sys.argv) > 3 else "M15"
    nb = int(sys.argv[4]) if len(sys.argv) > 4 else 3000
    r = run(sk, sym, tf, nb)
    print("=" * 55)
    print(f"  {sk} · {sym} · {tf} · {nb} bare")
    print("=" * 55)
    for k in ["n_trades", "win_rate", "profit_factor", "net_pnl",
              "expectancy", "max_drawdown", "avg_bars_held"]:
        if k in r:
            print(f"  {k:16}: {r[k]}")
    if "by_outcome" in r:
        print("  Outcomes:")
        for o, info in r["by_outcome"].items():
            print(f"    {o:8}: n={info['n']:4} avg={info['avg_pnl']:+7.2f} total={info['total']:+.0f}")
