"""
ai_agent/models/xgb_trainer.py — antrenare XGBoost cu walk-forward CV.

Pipeline:
  1. Citeste features + labels din DB pentru (symbol, tf)
  2. Aliniaza pe ts (inner join)
  3. Walk-forward split:
       - train window: 365 zile
       - test window: 30 zile
       - roll forward 30 zile la fiecare iteratie
  4. Antreneaza XGBoost pe fiecare fold, masoara AUC pe test
  5. Daca media AUC > MIN_AUC_THRESHOLD → salveaza modelul final
       (refit pe ultimul TRAIN_WALK_WINDOW_DAYS)

Output:
  - models/saved/<symbol>_<tf>_<horizon>.joblib  (modelul pentru inference)
  - logs/training_report.json                     (metrici per fold)
"""
from __future__ import annotations
import sys
import json
import time
import logging
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import joblib

from ai_agent.db.schema import init_db, get_conn
from ai_agent.config import (
    MODELS_DIR, LOGS_DIR,
    LABEL_HORIZONS_HOURS,
    TRAIN_WALK_WINDOW_DAYS, TRAIN_TEST_WINDOW_DAYS,
    MIN_AUC_THRESHOLD,
)

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)


def load_dataset(symbol: str, tf: str, horizon_h: int) -> pd.DataFrame:
    """
    Joineaza features + labels pe (symbol, tf, ts) pentru un anumit horizon.
    Returneaza DataFrame: ts, feature_1...feature_N, y_hit_long, y_return_pct
    """
    with get_conn() as conn:
        # Features
        f_rows = conn.execute(
            "SELECT ts, features FROM features WHERE symbol=? AND timeframe=? ORDER BY ts",
            (symbol, tf),
        ).fetchall()
        # Labels
        l_rows = conn.execute(
            "SELECT ts, return_pct, hit_long, hit_short FROM labels "
            "WHERE symbol=? AND timeframe=? AND horizon_h=? ORDER BY ts",
            (symbol, tf, horizon_h),
        ).fetchall()
    if not f_rows or not l_rows:
        return pd.DataFrame()

    # Expand JSON features
    feats_df = pd.DataFrame([
        {"ts": r["ts"], **json.loads(r["features"])}
        for r in f_rows
    ])
    feats_df = feats_df.drop(columns=["_n_bars_available"], errors="ignore")

    labels_df = pd.DataFrame([dict(r) for r in l_rows])
    labels_df = labels_df.rename(columns={
        "hit_long":  "y_hit_long",
        "hit_short": "y_hit_short",
        "return_pct":"y_return_pct",
    })

    df = feats_df.merge(labels_df, on="ts", how="inner")
    df = df.sort_values("ts").reset_index(drop=True)
    return df


def _bars_per_day(tf: str) -> int:
    return {"M1": 60*24, "M5": 12*24, "M15": 4*24, "M30": 2*24,
            "H1": 24, "H4": 6, "D1": 1}.get(tf, 24)


def walk_forward_split(df: pd.DataFrame, tf: str,
                       train_days: int, test_days: int) -> list:
    """
    Genereaza folds (train_idx, test_idx) progresivi:
      Fold 1: train = first train_days; test = next test_days
      Fold 2: roll forward test_days, repeat
    """
    n = len(df)
    bpd = _bars_per_day(tf)
    train_size = train_days * bpd
    test_size  = test_days  * bpd
    folds = []
    start = 0
    while start + train_size + test_size <= n:
        tr_end = start + train_size
        ts_end = tr_end + test_size
        folds.append({
            "train_idx": range(start, tr_end),
            "test_idx":  range(tr_end, ts_end),
            "train_start_ts": int(df.iloc[start]["ts"]),
            "train_end_ts":   int(df.iloc[tr_end-1]["ts"]),
            "test_start_ts":  int(df.iloc[tr_end]["ts"]),
            "test_end_ts":    int(df.iloc[ts_end-1]["ts"]),
        })
        start += test_size   # roll forward
    return folds


def train_model(X_train, y_train, X_val=None, y_val=None):
    """Antreneaza un XGBoost binary classifier."""
    import xgboost as xgb
    params = {
        "objective":        "binary:logistic",
        "eval_metric":      "auc",
        "max_depth":        4,
        "learning_rate":    0.05,
        "n_estimators":     200,
        "min_child_weight": 10,    # anti-overfit
        "subsample":        0.8,
        "colsample_bytree": 0.8,
        "reg_alpha":        0.1,
        "reg_lambda":       1.0,
        "random_state":     42,
        "n_jobs":           -1,
        "verbosity":        0,
    }
    model = xgb.XGBClassifier(**params)
    eval_set = [(X_val, y_val)] if X_val is not None and len(X_val) > 0 else None
    model.fit(X_train, y_train, eval_set=eval_set, verbose=False)
    return model


def evaluate(model, X, y) -> dict:
    """Returneaza AUC, accuracy, precision/recall pentru clasa 1 (BUY win)."""
    from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score
    probs = model.predict_proba(X)[:, 1]
    preds = (probs >= 0.5).astype(int)
    try:
        auc = float(roc_auc_score(y, probs))
    except Exception:
        auc = 0.5
    return {
        "auc":       round(auc, 4),
        "accuracy":  round(float(accuracy_score(y, preds)), 4),
        "precision": round(float(precision_score(y, preds, zero_division=0)), 4),
        "recall":    round(float(recall_score(y, preds, zero_division=0)), 4),
        "n":         int(len(y)),
        "class_balance": round(float(y.mean()), 4),
    }


def run_walk_forward(symbol: str, tf: str, horizon_h: int = 24,
                     train_days: int = TRAIN_WALK_WINDOW_DAYS,
                     test_days: int = TRAIN_TEST_WINDOW_DAYS,
                     target: str = "y_hit_long") -> dict:
    """
    Pipeline complet: load → split → train each fold → aggregate metrics.

    target = 'y_hit_long' (default: invata cand BUY castiga)
           sau 'y_hit_short' (cand SELL castiga)
    """
    init_db()
    log.info(f"Loading dataset {symbol}/{tf} horizon={horizon_h}h target={target}")
    df = load_dataset(symbol, tf, horizon_h)
    if df.empty:
        log.error(f"Niciun rand pentru {symbol}/{tf}")
        return {}
    log.info(f"  → {len(df)} rows, {len(df.columns)-4} features")

    # Drop coloane non-features
    feat_cols = [c for c in df.columns
                 if c not in ("ts", "y_hit_long", "y_hit_short", "y_return_pct")]
    X_all = df[feat_cols].astype(float).fillna(0.0)
    y_all = df[target].astype(int)

    folds = walk_forward_split(df, tf, train_days, test_days)
    if not folds:
        log.error(f"Insuficiente date pentru walk-forward "
                  f"({train_days}d train + {test_days}d test cere "
                  f"~{(train_days+test_days)*_bars_per_day(tf)} bare, ai {len(df)})")
        return {}

    log.info(f"  → {len(folds)} folduri")

    fold_results = []
    for i, fold in enumerate(folds, 1):
        X_tr = X_all.iloc[list(fold["train_idx"])]
        y_tr = y_all.iloc[list(fold["train_idx"])]
        X_ts = X_all.iloc[list(fold["test_idx"])]
        y_ts = y_all.iloc[list(fold["test_idx"])]

        if y_tr.nunique() < 2 or y_ts.nunique() < 2:
            log.warning(f"  Fold {i}: clasa unica in train/test — skip")
            continue
        if len(X_tr) < 100:
            continue

        t0 = time.time()
        model = train_model(X_tr, y_tr, X_ts, y_ts)
        train_metrics = evaluate(model, X_tr, y_tr)
        test_metrics  = evaluate(model, X_ts, y_ts)
        t_elapsed = time.time() - t0
        log.info(f"  Fold {i:2d}: train AUC={train_metrics['auc']} | "
                 f"test AUC={test_metrics['auc']} | "
                 f"acc={test_metrics['accuracy']} | n={test_metrics['n']} | "
                 f"{t_elapsed:.1f}s")
        fold_results.append({
            "fold":         i,
            "train_start":  fold["train_start_ts"],
            "train_end":    fold["train_end_ts"],
            "test_start":   fold["test_start_ts"],
            "test_end":     fold["test_end_ts"],
            "train":        train_metrics,
            "test":         test_metrics,
        })

    if not fold_results:
        log.error("Niciun fold valid")
        return {}

    mean_auc = float(np.mean([f["test"]["auc"] for f in fold_results]))
    median_auc = float(np.median([f["test"]["auc"] for f in fold_results]))
    log.info(f"  → Mean test AUC: {mean_auc:.4f}  (median {median_auc:.4f})")

    summary = {
        "symbol":     symbol,
        "tf":         tf,
        "horizon_h":  horizon_h,
        "target":     target,
        "n_features": len(feat_cols),
        "n_folds":    len(fold_results),
        "mean_auc":   round(mean_auc, 4),
        "median_auc": round(median_auc, 4),
        "min_auc":    round(min(f["test"]["auc"] for f in fold_results), 4),
        "max_auc":    round(max(f["test"]["auc"] for f in fold_results), 4),
        "folds":      fold_results,
        "trained_at": datetime.now(timezone.utc).isoformat(),
    }

    # Salveaza model final daca AUC e suficient
    if mean_auc >= MIN_AUC_THRESHOLD:
        log.info(f"  ✓ AUC mediu {mean_auc:.3f} ≥ {MIN_AUC_THRESHOLD} — refit pe ultimele {train_days}d")
        # Refit pe ultimele train_days bare
        bpd = _bars_per_day(tf)
        cutoff = max(0, len(X_all) - train_days * bpd)
        X_final = X_all.iloc[cutoff:]
        y_final = y_all.iloc[cutoff:]
        final_model = train_model(X_final, y_final)
        model_path = MODELS_DIR / f"{symbol}_{tf}_h{horizon_h}_{target}.joblib"
        joblib.dump({
            "model":        final_model,
            "feature_cols": feat_cols,
            "summary":      summary,
            "trained_at":   datetime.now(timezone.utc).isoformat(),
        }, model_path)
        log.info(f"  → Model salvat: {model_path}")
        summary["model_path"] = str(model_path)

        # Feature importance
        imp = sorted(zip(feat_cols, final_model.feature_importances_),
                     key=lambda x: x[1], reverse=True)
        summary["top_features"] = [(k, round(float(v), 4)) for k, v in imp[:10]]
        log.info("  Top 10 features:")
        for k, v in imp[:10]:
            log.info(f"    {k:25s} {v:.4f}")
    else:
        log.warning(f"  ✗ AUC mediu {mean_auc:.3f} < {MIN_AUC_THRESHOLD} — model NU se salveaza")

    # Salveaza raport
    report_path = LOGS_DIR / f"train_report_{symbol}_{tf}_h{horizon_h}_{target}.json"
    with open(report_path, "w") as f:
        json.dump(summary, f, indent=2, default=str)
    log.info(f"  Report: {report_path}")
    return summary


def train_all(symbols: list[str] | None = None,
              timeframes: list[str] | None = None,
              horizons: list[int] | None = None) -> list[dict]:
    from ai_agent.config import SYMBOLS, TIMEFRAMES
    init_db()
    syms = symbols or SYMBOLS
    tfs  = timeframes or TIMEFRAMES
    hzs  = horizons or LABEL_HORIZONS_HOURS
    results = []
    for sym in syms:
        for tf in tfs:
            for hz in hzs:
                for target in ("y_hit_long", "y_hit_short"):
                    log.info(f"\n{'='*60}\n  {sym}/{tf} h={hz} target={target}\n{'='*60}")
                    try:
                        r = run_walk_forward(sym, tf, hz, target=target)
                        if r:
                            results.append(r)
                    except Exception as e:
                        log.error(f"  EROARE: {e}")
    return results


if __name__ == "__main__":
    if len(sys.argv) >= 3:
        # python -m ai_agent.models.xgb_trainer EURUSD H1 [horizon] [target]
        sym = sys.argv[1]
        tf  = sys.argv[2]
        hz  = int(sys.argv[3]) if len(sys.argv) >= 4 else 24
        tg  = sys.argv[4] if len(sys.argv) >= 5 else "y_hit_long"
        run_walk_forward(sym, tf, hz, target=tg)
    else:
        results = train_all()
        print()
        print("=" * 60)
        for r in results:
            print(f"  {r['symbol']}/{r['tf']} h{r['horizon_h']} {r['target']}: "
                  f"AUC={r['mean_auc']}")
