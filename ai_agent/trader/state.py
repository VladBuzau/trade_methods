"""
ai_agent/trader/state.py — stare persistenta AI Trader.

Settings salvate intr-un JSON la ai_agent/db/trader_state.json:
  - watchlist:           [["EURUSD", "H1"], ["XAUUSD", "H4"]]
  - scan_interval_sec:   60
  - running:             bool
  - paper_trading:       bool — daca executa virtual trades
  - paper_balance:       float — soldul virtual
  - risk_per_trade_pct:  % din balance per trade
  - min_confidence:      % min pt a executa
  - horizon_h:           24 — horizon predictie
  - max_open_trades:     5

Stare runtime (in-memory):
  - last_scan_ts
  - scan_count
  - open_paper_trades: [{ticket, symbol, side, entry, sl, tp, ts_entry, ...}]
"""
from __future__ import annotations
import json
import threading
from pathlib import Path
from datetime import datetime, timezone

from ai_agent.config import DATA_DIR

STATE_PATH = DATA_DIR / "trader_state.json"

_DEFAULTS = {
    "watchlist":         [["EURUSD", "H1"], ["XAUUSD", "H1"]],
    "scan_interval_sec": 60,
    "running":           False,
    "paper_trading":     True,        # True = virtual, False = real MT5
    "real_mt5_execute":  False,       # explicit toggle pt MT5 real
    "paper_balance":     10000.0,
    "risk_per_trade_pct": 1.0,
    "risk_dollars":      50.0,        # pt MT5 real - fix $ per trade
    "min_confidence":    60,
    "horizon_h":         24,
    "max_open_trades":   5,
}

_lock = threading.RLock()
_runtime = {
    "last_scan_ts":      None,
    "scan_count":        0,
    "last_predictions":  [],   # ultimele 50 predictii (in-memory ring)
    "open_paper_trades": [],
    "closed_paper_trades": [],  # in-memory ring 200
    "scanner_thread":    None,
}


def load() -> dict:
    """Citeste settings de pe disc (sau defaults)."""
    if not STATE_PATH.exists():
        return dict(_DEFAULTS)
    try:
        with open(STATE_PATH, encoding="utf-8") as f:
            data = json.load(f)
        # Merge cu defaults pt chei noi
        for k, v in _DEFAULTS.items():
            data.setdefault(k, v)
        return data
    except Exception:
        return dict(_DEFAULTS)


def save(settings: dict) -> None:
    with _lock:
        STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
        with open(STATE_PATH, "w", encoding="utf-8") as f:
            json.dump(settings, f, indent=2)


def update(patch: dict) -> dict:
    """Aplica patch peste settings + salveaza."""
    with _lock:
        cur = load()
        cur.update(patch)
        save(cur)
        return cur


def get_runtime() -> dict:
    """Snapshot al starii runtime (thread-safe)."""
    with _lock:
        return {
            "last_scan_ts":     _runtime["last_scan_ts"],
            "scan_count":       _runtime["scan_count"],
            "last_predictions": list(_runtime["last_predictions"]),
            "open_paper_trades": list(_runtime["open_paper_trades"]),
            "closed_paper_trades": list(_runtime["closed_paper_trades"]),
        }


def set_runtime(**kwargs):
    with _lock:
        _runtime.update(kwargs)


def push_prediction(p: dict, keep: int = 50):
    """Adauga predictie in ring buffer in-memory."""
    with _lock:
        _runtime["last_predictions"].append(p)
        if len(_runtime["last_predictions"]) > keep:
            _runtime["last_predictions"] = _runtime["last_predictions"][-keep:]


def add_open_trade(t: dict):
    with _lock:
        _runtime["open_paper_trades"].append(t)


def remove_open_trade(ticket: str) -> dict | None:
    with _lock:
        for i, t in enumerate(_runtime["open_paper_trades"]):
            if t.get("ticket") == ticket:
                return _runtime["open_paper_trades"].pop(i)
    return None


def push_closed_trade(t: dict, keep: int = 200):
    with _lock:
        _runtime["closed_paper_trades"].append(t)
        if len(_runtime["closed_paper_trades"]) > keep:
            _runtime["closed_paper_trades"] = _runtime["closed_paper_trades"][-keep:]


def inc_scan():
    with _lock:
        _runtime["scan_count"] += 1
        _runtime["last_scan_ts"] = int(datetime.now(timezone.utc).timestamp())
