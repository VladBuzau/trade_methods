"""
Base class pentru toate strategiile de tranzactionare.
Orice strategie noua mosteneste Strategy si implementeaza analyze().
"""
from __future__ import annotations
from datetime import datetime
from typing import Any


class Strategy:
    # ── Identitate (suprascrie in subclasa) ──────────────────────────────────
    key   : str  = "base"
    name  : str  = "Base"
    icon  : str  = "⚪"
    color : str  = "#888888"

    default_tfs   : list[str] = ["H1"]
    default_bars  : int       = 500
    elements      : dict[str, str] = {}   # {key: "Label afisat in UI"}

    # ── Interfata publica ─────────────────────────────────────────────────────
    def analyze(
        self,
        symbol     : str,
        tfs        : list[str],
        bars       : int               = 500,
        tf_bars    : dict[str, int]    = None,
        elements   : dict[str, bool]   = None,
        min_confidence : float         = 66.0,
        **kwargs,
    ) -> dict[str, Any]:
        """
        Analizeaza simbolul si returneaza mereu acelasi format:

        {
            "symbol":        str,
            "strategy":      str,           ← self.key
            "timestamp":     ISO str,
            "signal":        "BUY"|"SELL"|"HOLD",
            "confidence":    float  0-100,
            "best_tf":       dict | None,   ← {"tf","signal","sl","tp","price","reasons"}
            "tfs":           list[dict],    ← rezultate per TF
            "justification": list[str],     ← motive in romana
            "auto_executed": bool,
        }
        """
        raise NotImplementedError(f"{self.__class__.__name__}.analyze() nu e implementat")

    # ── Helpers comune ────────────────────────────────────────────────────────
    @staticmethod
    def _now() -> str:
        return datetime.now().isoformat()

    @staticmethod
    def _confidence(n_win: int, n_total: int) -> float:
        if n_total == 0:
            return 0.0
        return round((n_win / n_total) * 100, 1)

    def _empty_result(self, symbol: str, reason: str = "Fara date") -> dict:
        return {
            "symbol":        symbol,
            "strategy":      self.key,
            "timestamp":     self._now(),
            "signal":        "HOLD",
            "confidence":    0.0,
            "best_tf":       None,
            "tfs":           [],
            "justification": [reason],
            "auto_executed": False,
        }

    def _build_result(
        self,
        symbol      : str,
        tf_results  : list[dict],
        min_confidence : float,
        min_votes   : int = 1,
        extra       : dict = None,
    ) -> dict:
        """
        Helper: calculeaza semnal final din lista de rezultate per TF.
        Fiecare element din tf_results trebuie sa aiba: signal, conviction, reasons, sl, tp, price, tf.
        """
        buy_v  = [r for r in tf_results if r["signal"] == "BUY"]
        sell_v = [r for r in tf_results if r["signal"] == "SELL"]
        n_buy, n_sell, n_total = len(buy_v), len(sell_v), len(tf_results)

        confidence  = self._confidence(max(n_buy, n_sell), n_total)
        final       = "HOLD"
        best_tf     = None

        if n_buy >= min_votes and n_buy > n_sell and confidence >= min_confidence:
            final   = "BUY"
            best_tf = max(buy_v, key=lambda x: x.get("conviction", 0))
        elif n_sell >= min_votes and n_sell > n_buy and confidence >= min_confidence:
            final   = "SELL"
            best_tf = max(sell_v, key=lambda x: x.get("conviction", 0))

        justification = []
        if best_tf:
            justification += [r.replace(" ✓", "") for r in best_tf.get("reasons", [])]
            justification.append(
                f"Confidence {confidence}% ({max(n_buy, n_sell)}/{n_total} TF-uri)"
            )
        else:
            justification.append(
                f"Semnal insuficient: {n_buy} BUY / {n_sell} SELL pe {n_total} TF-uri"
            )

        result = {
            "symbol":        symbol,
            "strategy":      self.key,
            "timestamp":     self._now(),
            "signal":        final,
            "confidence":    confidence,
            "n_buy":         n_buy,
            "n_sell":        n_sell,
            "n_total":       n_total,
            "best_tf":       best_tf,
            "tfs":           tf_results,
            "justification": justification,
            "auto_executed": False,
        }
        if extra:
            result.update(extra)
        return result
