"""
ai_agent/models/inference.py — folosire modele antrenate pentru predictii live.

Pipeline:
  1. Incarca modelul .joblib pentru (symbol, tf, horizon, target)
  2. Calculeaza features pe bara curenta
  3. model.predict_proba(features) → probabilitate
  4. Returneaza decizia: BUY / SELL / HOLD + confidence

Folosit de:
  - strategies_ai/ai_agent_strategy.py — wrapper ca strategie ChartVisualizer
  - eventual scanner direct pentru augmentation
"""
from __future__ import annotations
import logging
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from ai_agent.config import MODELS_DIR

log = logging.getLogger(__name__)

_MODEL_CACHE: dict[str, dict] = {}


def model_path(symbol: str, tf: str, horizon: int = 24,
               target: str = "y_hit_long") -> Path:
    return MODELS_DIR / f"{symbol}_{tf}_h{horizon}_{target}.joblib"


def load_model(symbol: str, tf: str, horizon: int = 24,
               target: str = "y_hit_long") -> dict | None:
    """Cache-uit. Returneaza None daca modelul nu exista."""
    key = f"{symbol}/{tf}/h{horizon}/{target}"
    if key in _MODEL_CACHE:
        return _MODEL_CACHE[key]
    path = model_path(symbol, tf, horizon, target)
    if not path.exists():
        log.warning(f"Model lipseste: {path}")
        return None
    try:
        bundle = joblib.load(path)
        _MODEL_CACHE[key] = bundle
        return bundle
    except Exception as e:
        log.error(f"Load model fail: {e}")
        return None


def predict_one(features: dict, symbol: str, tf: str,
                horizon: int = 24,
                target: str = "y_hit_long") -> dict | None:
    """
    features = dict cu cele 41 features (de la features.builder.compute_features_for_bar)
    Returneaza: {prob, signal, confidence_pct} sau None daca model nu exista
    """
    bundle = load_model(symbol, tf, horizon, target)
    if bundle is None:
        return None

    model = bundle["model"]
    cols  = bundle["feature_cols"]

    # Aliniaza features cu coloanele asteptate de model
    row = [features.get(c, 0.0) for c in cols]
    X = np.array([row], dtype=float)

    try:
        prob = float(model.predict_proba(X)[0, 1])
    except Exception as e:
        log.error(f"predict_proba: {e}")
        return None

    # Decision rule:
    #   target=y_hit_long  → prob mare = BUY, prob mic = SELL
    #   target=y_hit_short → prob mare = SELL, prob mic = BUY
    if target == "y_hit_long":
        if prob >= 0.55:
            signal, conf = "BUY", int(prob * 100)
        elif prob <= 0.45:
            signal, conf = "SELL", int((1 - prob) * 100)
        else:
            signal, conf = "HOLD", 0
    else:   # y_hit_short
        if prob >= 0.55:
            signal, conf = "SELL", int(prob * 100)
        elif prob <= 0.45:
            signal, conf = "BUY", int((1 - prob) * 100)
        else:
            signal, conf = "HOLD", 0

    return {
        "signal":     signal,
        "confidence": conf,
        "prob_class1":round(prob, 4),
        "model_key":  f"{symbol}/{tf}/h{horizon}/{target}",
    }


def predict_dual(features: dict, symbol: str, tf: str,
                 horizon: int = 24) -> dict | None:
    """
    Combina predictiile de la 2 modele (y_hit_long si y_hit_short) pentru
    o decizie mai robusta:
      - Daca AMBELE indica BUY → BUY high confidence
      - Daca AMBELE indica SELL → SELL high confidence
      - Daca opuse → HOLD (incertitudine)
    """
    p_long  = predict_one(features, symbol, tf, horizon, target="y_hit_long")
    p_short = predict_one(features, symbol, tf, horizon, target="y_hit_short")
    if not p_long and not p_short:
        return None
    if not p_long:
        return p_short
    if not p_short:
        return p_long

    # Combina
    sig_l = p_long["signal"]
    sig_s = p_short["signal"]

    if sig_l == sig_s and sig_l != "HOLD":
        # Acord — confidence max
        conf = max(p_long["confidence"], p_short["confidence"])
        return {
            "signal":     sig_l,
            "confidence": min(95, conf + 5),  # bonus pt acord
            "prob_long":  p_long["prob_class1"],
            "prob_short": p_short["prob_class1"],
            "agreement":  True,
        }
    if sig_l == "HOLD" and sig_s != "HOLD":
        return p_short
    if sig_s == "HOLD" and sig_l != "HOLD":
        return p_long
    # Conflict
    return {
        "signal":     "HOLD",
        "confidence": 0,
        "prob_long":  p_long["prob_class1"],
        "prob_short": p_short["prob_class1"],
        "agreement":  False,
        "conflict":   f"long={sig_l} vs short={sig_s}",
    }


def list_available_models() -> list[dict]:
    """Returneaza toate modelele salvate cu metadata."""
    out = []
    for p in sorted(MODELS_DIR.glob("*.joblib")):
        try:
            b = joblib.load(p)
            s = b.get("summary", {})
            out.append({
                "file":    p.name,
                "symbol":  s.get("symbol"),
                "tf":      s.get("tf"),
                "horizon": s.get("horizon_h"),
                "target":  s.get("target"),
                "mean_auc":s.get("mean_auc"),
                "n_folds": s.get("n_folds"),
                "trained_at": b.get("trained_at"),
            })
        except Exception:
            pass
    return out


if __name__ == "__main__":
    print("Available models:")
    for m in list_available_models():
        print(f"  {m['file']:50s}  AUC={m['mean_auc']}  folds={m['n_folds']}")
