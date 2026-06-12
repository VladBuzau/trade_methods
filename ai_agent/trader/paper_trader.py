"""
ai_agent/trader/paper_trader.py — execuție virtuală (paper trading).

Pentru fiecare semnal acceptat:
  1. Deschide trade virtual (cu SL/TP calculate din ATR)
  2. Monitorizeaza la fiecare scan: pretul curent atinge SL sau TP?
  3. Inchide trade + actualizeaza balance virtual
  4. Logheaza in DB (tabela trades_paper)
"""
from __future__ import annotations
import uuid
import logging
from datetime import datetime, timezone

from ai_agent.db.schema import get_conn
from ai_agent.trader import state

log = logging.getLogger(__name__)


def open_trade(symbol: str, tf: str, signal: str, entry_price: float,
               sl: float, tp: float, confidence: int, model_id: str) -> dict:
    """Deschide un trade virtual + salveaza in DB."""
    ticket = f"PT-{uuid.uuid4().hex[:8]}"
    now = int(datetime.now(timezone.utc).timestamp())
    trade = {
        "ticket":      ticket,
        "symbol":      symbol,
        "tf":          tf,
        "side":        signal,
        "entry_price": entry_price,
        "sl":          sl,
        "tp":          tp,
        "ts_entry":    now,
        "confidence":  confidence,
        "model_id":    model_id,
        "status":      "open",
        "pnl":         None,
        "exit_price":  None,
        "ts_exit":     None,
        "outcome":     None,
    }
    state.add_open_trade(trade)
    log.info(f"[PAPER] OPEN {signal} {symbol}@{entry_price:.5f} SL={sl:.5f} TP={tp:.5f} ({ticket})")
    return trade


def check_open_trades(price_lookup) -> list[dict]:
    """
    Verifica toate trade-urile deschise: hit SL/TP? sau timp expirat?
    price_lookup = dict {(symbol, tf): {"high": float, "low": float, "close": float}}
    Returneaza lista trade-urilor inchise in acest pass.
    """
    closed_now = []
    rt = state.get_runtime()
    settings = state.load()
    now = int(datetime.now(timezone.utc).timestamp())

    for t in rt["open_paper_trades"][:]:   # copy ca sa pot modifica
        sym_tf = (t["symbol"], t["tf"])
        if sym_tf not in price_lookup:
            continue
        h = price_lookup[sym_tf]["high"]
        l = price_lookup[sym_tf]["low"]
        c = price_lookup[sym_tf]["close"]

        outcome = None
        exit_price = None

        if t["side"] == "BUY":
            if l <= t["sl"]:
                outcome = "SL"
                exit_price = t["sl"]
            elif h >= t["tp"]:
                outcome = "TP"
                exit_price = t["tp"]
        else:  # SELL
            if h >= t["sl"]:
                outcome = "SL"
                exit_price = t["sl"]
            elif l <= t["tp"]:
                outcome = "TP"
                exit_price = t["tp"]

        # Timeout (24h max)
        if not outcome and (now - t["ts_entry"]) > 86400:
            outcome = "TIMEOUT"
            exit_price = c

        if outcome:
            # PnL ca % din balance × leverage simplificat
            if t["side"] == "BUY":
                pnl_pct = (exit_price - t["entry_price"]) / t["entry_price"] * 100
            else:
                pnl_pct = (t["entry_price"] - exit_price) / t["entry_price"] * 100

            # Risc fix per trade (% din balance)
            risk_amt = settings["paper_balance"] * settings["risk_per_trade_pct"] / 100
            sl_distance_pct = abs(t["entry_price"] - t["sl"]) / t["entry_price"] * 100
            # Leverage implicit: cat trebuie sa fie pozitia ca sa pierdem risk_amt la SL
            if sl_distance_pct > 0:
                position_size_pct = (risk_amt / settings["paper_balance"] * 100) / sl_distance_pct * 100
            else:
                position_size_pct = 0
            pnl_usd = settings["paper_balance"] * position_size_pct / 100 * pnl_pct / 100

            closed = {**t,
                      "status":     "closed",
                      "outcome":    outcome,
                      "exit_price": round(exit_price, 5),
                      "ts_exit":    now,
                      "pnl":        round(pnl_usd, 2),
                      "pnl_pct":    round(pnl_pct, 3),
                      "duration_s": now - t["ts_entry"]}
            state.remove_open_trade(t["ticket"])
            state.push_closed_trade(closed)

            # Save to DB
            try:
                with get_conn() as conn:
                    conn.execute(
                        "INSERT INTO trades_paper "
                        "(model_id, symbol, entry_ts, entry_price, exit_ts, exit_price, "
                        " side, sl, tp, pnl, outcome) VALUES (?,?,?,?,?,?,?,?,?,?,?)",
                        (closed["model_id"], closed["symbol"],
                         closed["ts_entry"], closed["entry_price"],
                         closed["ts_exit"], closed["exit_price"],
                         closed["side"], closed["sl"], closed["tp"],
                         closed["pnl"], closed["outcome"]),
                    )
            except Exception as e:
                log.error(f"DB save paper trade: {e}")

            # Update balance
            new_balance = settings["paper_balance"] + closed["pnl"]
            state.update({"paper_balance": round(new_balance, 2)})

            log.info(f"[PAPER] CLOSE {closed['ticket']} {outcome} pnl=${closed['pnl']:+.2f} "
                     f"new_balance=${new_balance:.2f}")
            closed_now.append(closed)
    return closed_now


def metrics() -> dict:
    """Returneaza statistici cumulate paper trades."""
    rt = state.get_runtime()
    closed = rt["closed_paper_trades"]
    if not closed:
        return {"n": 0, "wins": 0, "losses": 0, "win_rate": 0.0,
                "net_pnl": 0.0, "profit_factor": 0.0}
    wins = [t for t in closed if t["pnl"] > 0]
    losses = [t for t in closed if t["pnl"] <= 0]
    gross_p = sum(t["pnl"] for t in wins)
    gross_l = abs(sum(t["pnl"] for t in losses))
    pf = (gross_p / gross_l) if gross_l > 0 else (999.0 if gross_p > 0 else 0)
    return {
        "n":              len(closed),
        "wins":           len(wins),
        "losses":         len(losses),
        "win_rate":       round(len(wins) / len(closed) * 100, 1),
        "net_pnl":        round(sum(t["pnl"] for t in closed), 2),
        "profit_factor":  round(pf, 2),
        "avg_win":        round(sum(t["pnl"] for t in wins) / len(wins), 2) if wins else 0,
        "avg_loss":       round(sum(t["pnl"] for t in losses) / len(losses), 2) if losses else 0,
    }
