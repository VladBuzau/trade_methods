"""
ai_agent/monitoring/auto_retrain.py — orchestreaza pipeline-ul de retrain.

Workflow:
  1. Download date noi (incremental MT5)
  2. Recalculeaza features pe ultimele N zile
  3. Recalculeaza labels
  4. Re-antreneaza modelul daca:
       - acuratetea live a scazut sub prag, SAU
       - modelul are > 30 zile (drift garantat)
  5. Compara mean AUC vs modelul vechi — pastreaza modelul mai bun

Usage:
    python -m ai_agent.monitoring.auto_retrain                # ruleaza tot
    python -m ai_agent.monitoring.auto_retrain EURUSD H1      # specific
"""
from __future__ import annotations
import sys
import json
import logging
from datetime import datetime, timezone, timedelta
from pathlib import Path

import joblib

from ai_agent.config import (
    SYMBOLS, TIMEFRAMES, LABEL_HORIZONS_HOURS,
    MODELS_DIR, MIN_AUC_THRESHOLD,
)

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

MAX_MODEL_AGE_DAYS = 30
MIN_LIVE_ACCURACY = 50.0   # daca live acc < 50% → retrain


def model_age_days(symbol: str, tf: str, horizon: int = 24,
                   target: str = "y_hit_long") -> float | None:
    path = MODELS_DIR / f"{symbol}_{tf}_h{horizon}_{target}.joblib"
    if not path.exists():
        return None
    try:
        bundle = joblib.load(path)
        ta = bundle.get("trained_at")
        if not ta:
            return None
        dt = datetime.fromisoformat(ta)
        return (datetime.now(timezone.utc) - dt).total_seconds() / 86400
    except Exception:
        return None


def needs_retrain(symbol: str, tf: str, horizon: int = 24,
                  target: str = "y_hit_long") -> tuple[bool, str]:
    """Returneaza (needs_retrain, reason)."""
    age = model_age_days(symbol, tf, horizon, target)
    if age is None:
        return True, "Modelul lipseste"
    if age > MAX_MODEL_AGE_DAYS:
        return True, f"Model vechi ({age:.1f}d > {MAX_MODEL_AGE_DAYS}d)"
    # Check live accuracy
    try:
        from ai_agent.monitoring.predictions_log import evaluate_predictions
        rep = evaluate_predictions(days_back=14)
        key = f"{symbol}_{tf}_h{horizon}"
        info = rep.get("models", {}).get(key)
        if info and info["predictions"] >= 20:
            if info["accuracy"] < MIN_LIVE_ACCURACY:
                return True, (f"Live accuracy {info['accuracy']}% < "
                              f"{MIN_LIVE_ACCURACY}% (n={info['predictions']})")
    except Exception as e:
        log.debug(f"check live acc: {e}")
    return False, f"OK (age {age:.1f}d, no drift detected)"


def retrain_pipeline(symbol: str, tf: str, horizon: int = 24,
                     target: str = "y_hit_long") -> dict:
    """Pipeline complet: dump → features → labels → train."""
    log.info(f"\n{'='*60}\nRETRAIN {symbol}/{tf} h{horizon} {target}\n{'='*60}")
    needs, reason = needs_retrain(symbol, tf, horizon, target)
    if not needs:
        log.info(f"  Skip: {reason}")
        return {"action": "skipped", "reason": reason}

    log.info(f"  Trigger: {reason}")

    # 1. Update prices (incremental — descarca doar bare noi)
    log.info("  [1/4] Update MT5 prices...")
    from ai_agent.data.mt5_dumper import dump_symbol_tf, ensure_mt5
    if not ensure_mt5():
        return {"action": "failed", "reason": "MT5 init failed"}
    dump_symbol_tf(symbol, tf, years=2.0)

    # 2. Rebuild features (recent only)
    log.info("  [2/4] Rebuild features...")
    from ai_agent.features.builder import build_for_symbol_tf as build_f
    build_f(symbol, tf)

    # 3. Rebuild labels
    log.info("  [3/4] Rebuild labels...")
    from ai_agent.features.labels import build_for_symbol_tf as build_l
    build_l(symbol, tf)

    # 4. Re-train
    log.info("  [4/4] Re-train XGBoost...")
    from ai_agent.models.xgb_trainer import run_walk_forward
    result = run_walk_forward(symbol, tf, horizon, target=target)
    if not result:
        return {"action": "failed", "reason": "training failed"}
    return {
        "action":   "retrained",
        "reason":   reason,
        "new_auc":  result.get("mean_auc"),
        "saved":    "model_path" in result,
    }


def run_all(symbols: list[str] | None = None,
            timeframes: list[str] | None = None,
            horizons: list[int] | None = None) -> list[dict]:
    syms = symbols or SYMBOLS
    tfs  = timeframes or TIMEFRAMES
    hzs  = horizons or LABEL_HORIZONS_HOURS
    results = []
    for sym in syms:
        for tf in tfs:
            for hz in hzs:
                for target in ("y_hit_long", "y_hit_short"):
                    try:
                        r = retrain_pipeline(sym, tf, hz, target)
                        r["model"] = f"{sym}/{tf}/h{hz}/{target}"
                        results.append(r)
                    except Exception as e:
                        log.error(f"  EROARE: {e}")
                        results.append({
                            "model": f"{sym}/{tf}/h{hz}/{target}",
                            "action": "error", "reason": str(e),
                        })
    return results


if __name__ == "__main__":
    if len(sys.argv) >= 3:
        r = retrain_pipeline(sys.argv[1], sys.argv[2],
                             int(sys.argv[3]) if len(sys.argv) > 3 else 24)
        print(json.dumps(r, indent=2))
    else:
        results = run_all()
        print("\n" + "=" * 60)
        print(f"AUTO-RETRAIN SUMMARY ({len(results)} models)")
        print("=" * 60)
        for r in results:
            print(f"  {r['model']:50s} {r['action']:10s} {r.get('reason', '')[:60]}")
