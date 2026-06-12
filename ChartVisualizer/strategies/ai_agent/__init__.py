"""
AI Agent Strategy — wrapper care apeleaza modelele XGBoost din ai_agent/.

Modelele trebuie antrenate in prealabil:
  python -m ai_agent.models.xgb_trainer EURUSD H1 24 y_hit_long
  python -m ai_agent.models.xgb_trainer EURUSD H1 24 y_hit_short
  ...

Strategia incarca pe (symbol, tf) modelele dual (long + short) si combina
predictiile. Daca modelul nu exista pentru un (symbol, tf), returneaza HOLD
silent (nu blocheaza scanner-ul).

Elements:
  - use_dual_models    (default ON): combina predictia long si short pt acord
  - require_high_conf  (default OFF): doar trade-uri cu confidence >= 70
  - horizon_24h        (default ON): foloseste horizon 24h (default)
  - horizon_72h        (default OFF): foloseste horizon 72h in loc de 24
  - log_predictions    (default ON): logging in DB predictions table

Pentru ca scanner-ul ChartVisualizer sa gaseasca ai_agent, adaugam parent
dir la sys.path.
"""
from __future__ import annotations
import sys
import logging
import os
from pathlib import Path

# Adauga FINALLBOSS la sys.path ca sa putem importa ai_agent
_PARENT = Path(__file__).resolve().parents[3]   # ../../../../FINALLBOSS
if str(_PARENT) not in sys.path:
    sys.path.insert(0, str(_PARENT))

from strategies.base import Strategy

log = logging.getLogger(__name__)

# Import lazy — daca ai_agent lipseste sau xgboost nu e instalat, strategia
# raporteaza problema dar nu crapa scanner-ul.
_AI_AVAILABLE = False
_AI_IMPORT_ERROR = None
try:
    from ai_agent.features.builder import compute_features_for_bar
    from ai_agent.models.inference import predict_one, predict_dual, list_available_models
    _AI_AVAILABLE = True
except Exception as e:
    _AI_IMPORT_ERROR = str(e)
    log.warning(f"ai_agent indisponibil: {e}")


class AIAgentStrategy(Strategy):
    key   = "ai_agent"
    name  = "AI Agent (XGBoost)"
    icon  = "🤖"
    color = "#00e5ff"

    default_tfs            = ["H1"]
    default_bars           = 300
    default_step           = 1
    default_max_hours      = 24.0
    default_duration_years = 2.0

    elements = {
        "use_dual_models":   "Combina modelele long+short (acord = bonus)",
        "require_high_conf": "STRICT: doar predictii cu confidence >= 70%",
        "horizon_72h":       "STRICT: foloseste horizon 72h in loc de 24h",
        "log_predictions":   "Salveaza predictiile in DB pentru post-analiza",
    }

    strategy_defaults = {
        "min_confidence": 55.0,
    }

    def analyze(self, symbol, tfs, bars=300, tf_bars=None, elements=None,
                min_confidence=55.0, **kwargs):
        from app import fetch

        if elements is None:
            elements = {k: not lbl.upper().startswith(("STRICT", "BLOCKING"))
                        for k, lbl in self.elements.items()}

        if not _AI_AVAILABLE:
            return self._empty_result(symbol,
                f"AI agent indisponibil: {_AI_IMPORT_ERROR or 'unknown'}")

        # Horizon-ul de folosit
        horizon = 72 if elements.get("horizon_72h", False) else 24

        tf_results = []
        for tf in tfs:
            try:
                n_bars = (tf_bars or {}).get(tf, bars)
                df, _  = fetch(symbol, tf, n_bars)
                if df is None or len(df) < 100:
                    continue
                df.columns = [c.lower() for c in df.columns]

                # Pregateste df pentru compute_features_for_bar
                # Builder-ul asteapta index integer (ts), nu datetime
                df_norm = df.reset_index(drop=True).copy()
                if "time" not in df_norm.columns:
                    df_norm.index = range(len(df_norm))

                feats = compute_features_for_bar(df_norm, len(df_norm) - 1)
                if feats is None:
                    continue

                # Apel model
                if elements.get("use_dual_models", True):
                    pred = predict_dual(feats, symbol, tf, horizon=horizon)
                else:
                    pred = predict_one(feats, symbol, tf,
                                       horizon=horizon, target="y_hit_long")

                if pred is None:
                    reasons = [f"Niciun model XGBoost pentru {symbol}/{tf}/h{horizon}"]
                    tf_results.append({"tf": tf, "signal": "HOLD", "conviction": 0,
                                       "reasons": reasons,
                                       "price": float(df["close"].iloc[-1]),
                                       "sl": None, "tp": None})
                    continue

                sig = pred["signal"]
                conf = pred.get("confidence", 0)

                # require_high_conf strict
                if elements.get("require_high_conf", False) and conf < 70:
                    reasons = [f"AI: {sig} {conf}% — sub pragul 70%, HOLD"]
                    tf_results.append({"tf": tf, "signal": "HOLD", "conviction": 0,
                                       "reasons": reasons,
                                       "price": float(df["close"].iloc[-1]),
                                       "sl": None, "tp": None})
                    continue

                # Logging optional
                if elements.get("log_predictions", True):
                    try:
                        _log_prediction(symbol, tf, horizon, pred, feats)
                    except Exception as _e:
                        log.debug(f"log_prediction: {_e}")

                if sig == "HOLD":
                    tf_results.append({"tf": tf, "signal": "HOLD", "conviction": 0,
                                       "reasons": [f"AI HOLD (prob ambigua)"],
                                       "price": float(df["close"].iloc[-1]),
                                       "sl": None, "tp": None})
                    continue

                # Conviction mapping: confidence% / 10 → 5-9 conviction
                conviction = max(0, min(10, conf // 10))

                # SL/TP din ATR (consistent cu alte strategii)
                price = float(df["close"].iloc[-1])
                from app import find_pivots, calc_sl_tp
                ph_idx, pl_idx = find_pivots(df, lookback=5)
                sl, tp = calc_sl_tp(df, ph_idx, pl_idx, sig, price)

                reasons = [
                    f"AI predicate {sig} cu {conf}% confidence",
                    f"Model: horizon={horizon}h",
                ]
                if pred.get("agreement"):
                    reasons.append("Long+Short models in acord ✓")
                if pred.get("conflict"):
                    reasons.append(f"Models conflict: {pred['conflict']}")
                if "prob_long" in pred:
                    reasons.append(f"prob_long={pred['prob_long']} prob_short={pred['prob_short']}")
                elif "prob_class1" in pred:
                    reasons.append(f"prob={pred['prob_class1']}")

                tf_results.append({
                    "tf": tf, "signal": sig, "conviction": conviction,
                    "reasons": reasons, "price": round(price, 5),
                    "sl": sl, "tp": tp,
                })
            except Exception as exc:
                log.warning(f"AI Agent {symbol}/{tf}: {exc}")

        if not tf_results:
            return self._empty_result(symbol, "Fara rezultate (model lipsa sau date insuficiente)")
        return self._build_result(symbol, tf_results, min_confidence, min_votes=1)


# ─── DB logging ─────────────────────────────────────────────────────────────
def _log_prediction(symbol: str, tf: str, horizon: int,
                    pred: dict, features: dict) -> None:
    """Salveaza predictia in tabela predictions pentru post-analiza."""
    from ai_agent.db.schema import get_conn
    import json
    import time as _t

    sig_to_int = {"BUY": 1, "SELL": -1, "HOLD": 0}
    with get_conn() as conn:
        conn.execute(
            "INSERT OR IGNORE INTO predictions "
            "(model_id, symbol, timeframe, ts, pred_dir, confidence, "
            "horizon_h, features_used) VALUES (?,?,?,?,?,?,?,?)",
            (
                f"{symbol}_{tf}_h{horizon}",
                symbol, tf, int(_t.time()),
                sig_to_int.get(pred["signal"], 0),
                int(pred.get("confidence", 0)),
                horizon,
                json.dumps({k: features.get(k, 0) for k in list(features.keys())[:10]}),
            ),
        )
