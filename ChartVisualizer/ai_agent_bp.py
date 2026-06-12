"""
ai_agent_bp.py — Flask blueprint pentru GUI-ul agentului AI.

Endpoints:
  GET  /ai_agent                        → pagina principala (HTML)
  GET  /ai_agent/models                 → lista modele antrenate (JSON)
  GET  /ai_agent/db_stats               → statistici DB (n rows per tabel)
  POST /ai_agent/train                  → trigger training pentru (sym, tf, hz, target)
  POST /ai_agent/full_pipeline          → dump + features + labels + train
  GET  /ai_agent/job/<job_id>           → status job background
  GET  /ai_agent/predictions            → ultimele predictii
  GET  /ai_agent/accuracy_report        → raport accuracy live
  POST /ai_agent/dump_data              → trigger MT5 dump
"""
from __future__ import annotations
import os
import sys
import json
import uuid
import threading
import logging
from pathlib import Path
from datetime import datetime, timezone, timedelta

from flask import Blueprint, request, Response

log = logging.getLogger(__name__)

ai_agent_bp = Blueprint("ai_agent", __name__)

# Adauga FINALLBOSS la path
_FINALLBOSS = Path(__file__).resolve().parent.parent
if str(_FINALLBOSS) not in sys.path:
    sys.path.insert(0, str(_FINALLBOSS))


# ── Job tracking ────────────────────────────────────────────────────────────
_jobs: dict[str, dict] = {}
_jobs_lock = threading.Lock()


def _set_job(job_id, **kwargs):
    with _jobs_lock:
        if job_id not in _jobs:
            _jobs[job_id] = {}
        _jobs[job_id].update(kwargs)


def _get_job(job_id):
    with _jobs_lock:
        return dict(_jobs.get(job_id, {}))


def _login_required(f):
    """Reuse pattern from app.py — bypass for localhost."""
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


# ── Endpoints ───────────────────────────────────────────────────────────────
@ai_agent_bp.route("/ai_agent")
@_login_required
def ai_agent_page():
    tpl = os.path.join(os.path.dirname(__file__), "templates", "ai_agent.html")
    if not os.path.exists(tpl):
        return Response("Template lipsa", status=500)
    with open(tpl, encoding="utf-8") as f:
        return Response(f.read(), content_type="text/html; charset=utf-8")


@ai_agent_bp.route("/ai_agent/models")
@_login_required
def ai_agent_models():
    try:
        from ai_agent.models.inference import list_available_models
        return Response(json.dumps(list_available_models(), default=str),
                        content_type="application/json")
    except Exception as e:
        return Response(json.dumps({"error": str(e)}),
                        content_type="application/json", status=500)


@ai_agent_bp.route("/ai_agent/db_stats")
@_login_required
def ai_agent_db_stats():
    try:
        from ai_agent.db.schema import stats
        return Response(json.dumps(stats()), content_type="application/json")
    except Exception as e:
        return Response(json.dumps({"error": str(e)}),
                        content_type="application/json", status=500)


@ai_agent_bp.route("/ai_agent/train", methods=["POST"])
@_login_required
def ai_agent_train():
    body = request.get_json(silent=True) or {}
    symbol  = body.get("symbol", "EURUSD")
    tf      = body.get("tf", "H1")
    horizon = int(body.get("horizon", 24))
    target  = body.get("target", "y_hit_long")
    train_days = int(body.get("train_days", 365))
    test_days  = int(body.get("test_days", 30))

    job_id = str(uuid.uuid4())[:8]
    _set_job(job_id, status="running", progress=0,
             message=f"Training {symbol}/{tf} h{horizon} {target}")

    def _run():
        try:
            from ai_agent.models.xgb_trainer import run_walk_forward
            result = run_walk_forward(symbol, tf, horizon,
                                      train_days=train_days,
                                      test_days=test_days,
                                      target=target)
            import time as _t
            _set_job(job_id, finished_at=_t.time(), status="done", progress=100,
                     result=result,
                     message=(f"AUC mediu {result.get('mean_auc', '—')} "
                              f"({result.get('n_folds', 0)} folduri)"))
        except Exception as e:
            import traceback
            log.error(f"Train job {job_id}: {e}\n{traceback.format_exc()}")
            import time as _t
            _set_job(job_id, finished_at=_t.time(), status="error", message=str(e))

    threading.Thread(target=_run, daemon=True).start()
    return Response(json.dumps({"ok": True, "job_id": job_id}),
                    content_type="application/json")


@ai_agent_bp.route("/ai_agent/full_pipeline", methods=["POST"])
@_login_required
def ai_agent_full_pipeline():
    """Dump + features + labels + train pentru un (symbol, tf)."""
    body = request.get_json(silent=True) or {}
    symbol  = body.get("symbol", "EURUSD")
    tf      = body.get("tf", "H1")
    horizon = int(body.get("horizon", 24))
    years   = float(body.get("years", 2.0))

    job_id = str(uuid.uuid4())[:8]
    _set_job(job_id, status="running", progress=0,
             message=f"Pipeline {symbol}/{tf} (years={years})")

    def _run():
        try:
            _set_job(job_id, progress=5, message="[1/5] MT5 dump...")
            from ai_agent.data.mt5_dumper import dump_symbol_tf, ensure_mt5
            if not ensure_mt5():
                raise RuntimeError("MT5 init failed")
            n_bars = dump_symbol_tf(symbol, tf, years=years)
            _set_job(job_id, progress=30, message=f"  → {n_bars} bare dump")

            _set_job(job_id, progress=35, message="[2/5] Features...")
            from ai_agent.features.builder import build_for_symbol_tf as build_f
            n_feat = build_f(symbol, tf)
            _set_job(job_id, progress=55, message=f"  → {n_feat} features")

            _set_job(job_id, progress=60, message="[3/5] Labels...")
            from ai_agent.features.labels import build_for_symbol_tf as build_l
            n_lbl = build_l(symbol, tf)
            _set_job(job_id, progress=70, message=f"  → {n_lbl} labels")

            _set_job(job_id, progress=75, message="[4/5] Train long...")
            from ai_agent.models.xgb_trainer import run_walk_forward
            r_long = run_walk_forward(symbol, tf, horizon,
                                      train_days=int(years*365 * 0.7),
                                      test_days=30,
                                      target="y_hit_long")

            _set_job(job_id, progress=88, message="[5/5] Train short...")
            r_short = run_walk_forward(symbol, tf, horizon,
                                       train_days=int(years*365 * 0.7),
                                       test_days=30,
                                       target="y_hit_short")

            import time as _t
            _set_job(job_id, finished_at=_t.time(), status="done", progress=100,
                     message=(f"Complete · long AUC {r_long.get('mean_auc', '—')} · "
                              f"short AUC {r_short.get('mean_auc', '—')}"),
                     result={
                         "n_bars": n_bars, "n_features": n_feat, "n_labels": n_lbl,
                         "long":  r_long, "short": r_short,
                     })
        except Exception as e:
            import traceback
            log.error(f"Pipeline {job_id}: {e}\n{traceback.format_exc()}")
            import time as _t
            _set_job(job_id, finished_at=_t.time(), status="error", message=str(e))

    threading.Thread(target=_run, daemon=True).start()
    return Response(json.dumps({"ok": True, "job_id": job_id}),
                    content_type="application/json")


@ai_agent_bp.route("/ai_agent/dump_data", methods=["POST"])
@_login_required
def ai_agent_dump_data():
    """Dump bulk pentru toate (symbol, tf) configurate."""
    body = request.get_json(silent=True) or {}
    symbols  = body.get("symbols")    # optional override
    tfs      = body.get("tfs")
    years    = float(body.get("years", 2.0))

    job_id = str(uuid.uuid4())[:8]
    _set_job(job_id, status="running", progress=0, message="Dump start...")

    def _run():
        try:
            from ai_agent.data.mt5_dumper import dump_all
            results = dump_all(symbols=symbols, tfs=tfs, years=years)
            total = sum(results.values())
            import time as _t
            _set_job(job_id, finished_at=_t.time(), status="done", progress=100,
                     message=f"Total {total} bare in {len(results)} pairs",
                     result=results)
        except Exception as e:
            import time as _t
            _set_job(job_id, finished_at=_t.time(), status="error", message=str(e))

    threading.Thread(target=_run, daemon=True).start()
    return Response(json.dumps({"ok": True, "job_id": job_id}),
                    content_type="application/json")


@ai_agent_bp.route("/ai_agent/job/<job_id>")
@_login_required
def ai_agent_job_status(job_id):
    job = _get_job(job_id)
    if not job:
        return Response(json.dumps({"status": "not_found"}),
                        content_type="application/json")
    return Response(json.dumps(job, default=str),
                    content_type="application/json")


@ai_agent_bp.route("/ai_agent/active_job")
@_login_required
def ai_agent_active_job():
    """Returneaza cel mai recent job ACTIV (running) sau ultimul terminat in <60s."""
    import time as _t
    with _jobs_lock:
        latest = None
        latest_id = None
        for jid, j in _jobs.items():
            if j.get("status") == "running":
                latest_id = jid; latest = j
                break
        if latest is None:
            # fallback: ultimul finalizat recent
            best_age = 999
            for jid, j in _jobs.items():
                ts = j.get("finished_at", 0) or 0
                age = _t.time() - ts if ts else 999
                if j.get("status") in ("done", "error") and age < 60 and age < best_age:
                    best_age = age
                    latest_id = jid; latest = j
    if latest is None:
        return Response(json.dumps({"active": False}),
                        content_type="application/json")
    return Response(json.dumps({"active": True, "job_id": latest_id, "job": latest},
                               default=str),
                    content_type="application/json")


@ai_agent_bp.route("/ai_agent/predictions")
@_login_required
def ai_agent_predictions():
    """Returneaza ultimele predictii."""
    try:
        from ai_agent.db.schema import get_conn
        limit = int(request.args.get("limit", 50))
        with get_conn() as conn:
            rows = conn.execute(
                "SELECT * FROM predictions ORDER BY ts DESC LIMIT ?",
                (limit,),
            ).fetchall()
        out = [dict(r) for r in rows]
        return Response(json.dumps(out, default=str),
                        content_type="application/json")
    except Exception as e:
        return Response(json.dumps({"error": str(e)}),
                        content_type="application/json", status=500)


@ai_agent_bp.route("/ai_agent/news_impact", methods=["POST"])
@_login_required
def ai_agent_news_impact():
    """Analizeaza istoric impactul stirilor majore pe pretul unui simbol."""
    body = request.get_json(silent=True) or {}
    symbol = body.get("symbol", "EURUSD")
    tf     = body.get("tf", "M5")
    years  = int(body.get("years", 3))
    end_year = datetime.now(timezone.utc).year
    start_year = end_year - years

    job_id = str(uuid.uuid4())[:8]
    _set_job(job_id, status="running", progress=0,
             message=f"News Impact {symbol}/{tf} pe {years}ani...")

    def _run():
        try:
            from ai_agent.analysis.news_impact import analyze_all_events
            r = analyze_all_events(symbol, tf, start_year, end_year)
            import time as _t
            _set_job(job_id, finished_at=_t.time(), status="done", progress=100,
                     result=r,
                     message=f"Analiza completa: {sum(v['n'] for v in r['results'].values())} evenimente")
        except Exception as e:
            import time as _t, traceback
            log.error(f"news_impact {job_id}: {e}\n{traceback.format_exc()}")
            _set_job(job_id, finished_at=_t.time(), status="error", message=str(e))

    threading.Thread(target=_run, daemon=True).start()
    return Response(json.dumps({"ok": True, "job_id": job_id}),
                    content_type="application/json")


@ai_agent_bp.route("/ai_agent/validate", methods=["POST"])
@_login_required
def ai_agent_validate():
    """Ruleaza validare statistica (AI vs random vs trivial)."""
    body = request.get_json(silent=True) or {}
    symbol  = body.get("symbol", "EURUSD")
    tf      = body.get("tf", "H1")
    horizon = int(body.get("horizon", 24))
    oos_days = int(body.get("oos_days", 60))

    job_id = str(uuid.uuid4())[:8]
    _set_job(job_id, status="running", progress=0,
             message=f"Validare {symbol}/{tf} h{horizon} pe {oos_days}d OOS...")

    def _run():
        try:
            from ai_agent.monitoring.validator import run_validation
            result = run_validation(symbol, tf, horizon, oos_days)
            import time as _t
            _set_job(job_id, finished_at=_t.time(), status="done", progress=100,
                     result=result,
                     message=f"Verdict: {result.get('verdict', '—')}")
        except Exception as e:
            import time as _t, traceback
            log.error(f"validate {job_id}: {e}\n{traceback.format_exc()}")
            _set_job(job_id, finished_at=_t.time(), status="error", message=str(e))

    threading.Thread(target=_run, daemon=True).start()
    return Response(json.dumps({"ok": True, "job_id": job_id}),
                    content_type="application/json")


@ai_agent_bp.route("/ai_agent/accuracy_report")
@_login_required
def ai_agent_accuracy_report():
    """Raport accuracy live (predictions vs actual)."""
    try:
        from ai_agent.monitoring.predictions_log import evaluate_predictions
        days = int(request.args.get("days", 7))
        return Response(json.dumps(evaluate_predictions(days), default=str),
                        content_type="application/json")
    except Exception as e:
        return Response(json.dumps({"error": str(e)}),
                        content_type="application/json", status=500)
