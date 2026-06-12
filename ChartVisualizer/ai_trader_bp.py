"""
ai_trader_bp.py — Flask blueprint pentru AI Trader (separat de AutoTrader).

Endpoints:
  GET  /ai_trader                  → pagina
  GET  /ai_trader/status           → snapshot complet (settings + runtime + metrics)
  POST /ai_trader/start            → porneste scanner-ul AI Trader
  POST /ai_trader/stop             → opreste scanner
  POST /ai_trader/settings         → update settings (watchlist, interval, risk, etc.)
  POST /ai_trader/add_symbol       → adauga (symbol, tf) la watchlist
  POST /ai_trader/remove_symbol    → scoate (symbol, tf) din watchlist
  POST /ai_trader/reset_balance    → reset paper balance la 10000
  POST /ai_trader/close_all        → inchide toate trade-urile virtuale deschise
"""
from __future__ import annotations
import os
import sys
import json
import logging
from pathlib import Path

from flask import Blueprint, request, Response

log = logging.getLogger(__name__)

ai_trader_bp = Blueprint("ai_trader", __name__)

# Adauga FINALLBOSS la path
_FINALLBOSS = Path(__file__).resolve().parent.parent
if str(_FINALLBOSS) not in sys.path:
    sys.path.insert(0, str(_FINALLBOSS))


def _login_required(f):
    from functools import wraps
    from flask import session, redirect, url_for
    @wraps(f)
    def wrapper(*a, **kw):
        try:
            from app import _is_direct_localhost
            if _is_direct_localhost():
                return f(*a, **kw)
        except Exception:
            pass
        if not session.get("logged_in"):
            return redirect(url_for("login"))
        return f(*a, **kw)
    return wrapper


def _json(obj, status=200):
    return Response(json.dumps(obj, default=str), status=status,
                    content_type="application/json")


# ── Page ───────────────────────────────────────────────────────────────────
@ai_trader_bp.route("/ai_trader")
@_login_required
def ai_trader_page():
    tpl = os.path.join(os.path.dirname(__file__), "templates", "ai_trader.html")
    if not os.path.exists(tpl):
        return Response("Template lipsa", status=500)
    with open(tpl, encoding="utf-8") as f:
        return Response(f.read(), content_type="text/html; charset=utf-8")


# ── Status ─────────────────────────────────────────────────────────────────
@ai_trader_bp.route("/ai_trader/status")
@_login_required
def ai_trader_status():
    try:
        from ai_agent.trader import state, paper_trader, scanner
        scanner.ensure_scanner_started()   # idempotent
        return _json({
            "settings":  state.load(),
            "runtime":   state.get_runtime(),
            "metrics":   paper_trader.metrics(),
        })
    except Exception as e:
        return _json({"error": str(e)}, 500)


# ── Start / Stop ───────────────────────────────────────────────────────────
@ai_trader_bp.route("/ai_trader/start", methods=["POST"])
@_login_required
def ai_trader_start():
    try:
        from ai_agent.trader import state, scanner
        scanner.ensure_scanner_started()
        state.update({"running": True})
        return _json({"ok": True, "running": True})
    except Exception as e:
        return _json({"ok": False, "error": str(e)}, 500)


@ai_trader_bp.route("/ai_trader/stop", methods=["POST"])
@_login_required
def ai_trader_stop():
    try:
        from ai_agent.trader import state
        state.update({"running": False})
        return _json({"ok": True, "running": False})
    except Exception as e:
        return _json({"ok": False, "error": str(e)}, 500)


# ── Settings ───────────────────────────────────────────────────────────────
@ai_trader_bp.route("/ai_trader/settings", methods=["POST"])
@_login_required
def ai_trader_settings():
    body = request.get_json(silent=True) or {}
    try:
        from ai_agent.trader import state
        allowed = {"scan_interval_sec", "paper_trading", "real_mt5_execute",
                   "risk_per_trade_pct", "risk_dollars",
                   "min_confidence", "horizon_h", "max_open_trades", "watchlist"}
        patch = {k: v for k, v in body.items() if k in allowed}
        new_state = state.update(patch)
        return _json({"ok": True, "settings": new_state})
    except Exception as e:
        return _json({"ok": False, "error": str(e)}, 500)


# ── Watchlist ──────────────────────────────────────────────────────────────
@ai_trader_bp.route("/ai_trader/add_symbol", methods=["POST"])
@_login_required
def ai_trader_add_symbol():
    body = request.get_json(silent=True) or {}
    symbol = body.get("symbol", "").upper().strip()
    tf     = body.get("tf", "H1")
    if not symbol:
        return _json({"ok": False, "error": "symbol lipsa"}, 400)
    try:
        from ai_agent.trader import state
        s = state.load()
        wl = s.get("watchlist", [])
        pair = [symbol, tf]
        if pair not in wl:
            wl.append(pair)
            state.update({"watchlist": wl})
        return _json({"ok": True, "watchlist": wl})
    except Exception as e:
        return _json({"ok": False, "error": str(e)}, 500)


@ai_trader_bp.route("/ai_trader/remove_symbol", methods=["POST"])
@_login_required
def ai_trader_remove_symbol():
    body = request.get_json(silent=True) or {}
    symbol = body.get("symbol", "").upper().strip()
    tf     = body.get("tf", "")
    try:
        from ai_agent.trader import state
        s = state.load()
        wl = [p for p in s.get("watchlist", []) if not (p[0] == symbol and p[1] == tf)]
        state.update({"watchlist": wl})
        return _json({"ok": True, "watchlist": wl})
    except Exception as e:
        return _json({"ok": False, "error": str(e)}, 500)


# ── Reset / Close ──────────────────────────────────────────────────────────
@ai_trader_bp.route("/ai_trader/reset_balance", methods=["POST"])
@_login_required
def ai_trader_reset_balance():
    try:
        from ai_agent.trader import state
        state.update({"paper_balance": 10000.0})
        return _json({"ok": True})
    except Exception as e:
        return _json({"ok": False, "error": str(e)}, 500)


@ai_trader_bp.route("/ai_trader/close_all", methods=["POST"])
@_login_required
def ai_trader_close_all():
    try:
        from ai_agent.trader import state
        rt = state.get_runtime()
        closed_count = 0
        for t in rt["open_paper_trades"][:]:
            # Inchidem la pretul de entry (PnL 0) — manually closed
            t["status"] = "closed"
            t["outcome"] = "MANUAL"
            t["pnl"] = 0.0
            t["exit_price"] = t["entry_price"]
            state.remove_open_trade(t["ticket"])
            state.push_closed_trade(t)
            closed_count += 1
        return _json({"ok": True, "closed": closed_count})
    except Exception as e:
        return _json({"ok": False, "error": str(e)}, 500)
