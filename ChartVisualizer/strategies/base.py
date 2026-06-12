"""
Base class pentru toate strategiile de tranzactionare (v2 STRICT).

Schimbari majore fata de v1:
  - Default min_confidence: 66 → 85 (semnale de calitate, nu cantitate)
  - Min votes TF-uri: 1 → 2 (semnalul valid doar daca ≥2 TF-uri confirma)
  - Filtre comune: volatility gate, session gate, spread gate (toate strategiile le pot folosi)
  - Defaults implicite salvate ca snapshot pentru buton "Restore Defaults"
"""
from __future__ import annotations
from datetime import datetime, timezone
from typing import Any


# ── Defaults globale (sursa de adevar pentru butonul "Restore Defaults") ──────
STRICT_DEFAULTS = {
    "min_confidence":     85.0,
    "min_votes_per_tf":   2,
    "auto_execute":       False,
    "risk_dollars":       50.0,
    "max_open_trades":    3,
    "one_per_strategy":   True,
    "session_filter":     True,    # skip trades in sesiuni dead (22-01 UTC)
    "volatility_filter":  True,    # skip ATR squeeze / expansion extrema
    "spread_filter":      True,    # skip spread > 2x medie
    "weekend_close":      True,
}

# Sesiuni dead (UTC) — zero lichiditate
DEAD_SESSIONS = [(22, 24), (0, 1)]


# ── Filtre comune ─────────────────────────────────────────────────────────────
def is_dead_session() -> bool:
    """True daca e in fereastra moarta 22-01 UTC."""
    hour = datetime.now(timezone.utc).hour
    if hour >= 22 or hour < 1:
        return True
    return False


def volatility_ok(df, period: int = 14, min_mult: float = 0.5, max_mult: float = 2.5) -> tuple[bool, str]:
    """
    ATR curent trebuie in [min_mult, max_mult] × media ATR pe 50 bare.
    Squeeze extrem (< 0.5x) = fake breakout; expansion (> 2.5x) = stop-uri uriase.
    """
    try:
        import numpy as np
        if len(df) < period * 4:
            return True, "ATR: date insuficiente, skip filter"
        high  = df["high"].values
        low   = df["low"].values
        close = df["close"].values
        tr = np.maximum(high[1:] - low[1:],
                        np.maximum(np.abs(high[1:] - close[:-1]),
                                   np.abs(low[1:]  - close[:-1])))
        atr = np.convolve(tr, np.ones(period)/period, mode='valid')
        if len(atr) < 20:
            return True, "ATR: date insuficiente"
        cur = float(atr[-1])
        avg = float(np.mean(atr[-50:])) if len(atr) >= 50 else float(np.mean(atr))
        if avg <= 0:
            return False, "ATR medie zero"
        ratio = cur / avg
        if ratio < min_mult:
            return False, f"ATR squeeze ({ratio:.2f}× < {min_mult}×) — piata moarta"
        if ratio > max_mult:
            return False, f"ATR expansion ({ratio:.2f}× > {max_mult}×) — volatil extrem"
        return True, f"ATR regim normal ({ratio:.2f}×)"
    except Exception:
        return True, "ATR filter: exceptie, skip"


class Strategy:
    # ── Identitate (suprascrie in subclasa) ──────────────────────────────────
    key   : str  = "base"
    name  : str  = "Base"
    icon  : str  = "⚪"
    color : str  = "#888888"

    default_tfs   : list[str] = ["H1"]
    default_bars  : int       = 500
    elements      : dict[str, str] = {}   # {key: "Label afisat in UI"}

    # ── Backtest defaults (auto-aplicate cand strategia e selectata) ─────────
    # step=1 = verifica fiecare bara (default sigur, nu pierdem semnale).
    # Override doar daca strategia merita pas mai mare (ex: scanare lenta dorita)
    default_step             : int   = 1      # bare intre analyze() in backtest
    default_duration_years   : float = 1.0    # ani de history pt backtest
    default_max_hours        : float = 10.0   # max durata trade in ore

    # ── Filtrare pe simboluri ────────────────────────────────────────────────
    # symbol_whitelist: lista de fragmente; daca e setata, strategia ruleaza
    #                   DOAR pe simboluri care contin unul din fragmente.
    # exclusive_symbols: cand True + simbolul matchuieste whitelist-ul, NICIO
    #                    alta strategie nu mai ruleaza pe acel simbol.
    symbol_whitelist  : list[str] | None = None
    exclusive_symbols : bool             = False

    # Defaults specifice strategiei — overrride in subclasa pt "Restore"
    strategy_defaults: dict = {}

    @classmethod
    def matches_symbol(cls, symbol: str) -> bool:
        """True daca strategia poate rula pe acest simbol."""
        if not cls.symbol_whitelist:
            return True
        sym_clean = symbol.replace(".", "").replace("-", "").upper()
        return any(p.upper() in sym_clean for p in cls.symbol_whitelist)

    # ── Interfata publica ─────────────────────────────────────────────────────
    def analyze(
        self,
        symbol     : str,
        tfs        : list[str],
        bars       : int               = 500,
        tf_bars    : dict[str, int]    = None,
        elements   : dict[str, bool]   = None,
        min_confidence : float         = 85.0,   # STRICT: era 66
        **kwargs,
    ) -> dict[str, Any]:
        raise NotImplementedError(f"{self.__class__.__name__}.analyze() nu e implementat")

    def get_defaults(self) -> dict:
        """Returneaza defaults pentru UI ("Restore Defaults" button)."""
        base = {
            "min_confidence": 85.0,
            "tfs":            list(self.default_tfs),
            "bars":           self.default_bars,
            "elements":       {k: True for k in self.elements},
        }
        base.update(self.strategy_defaults)
        return base

    # ── Optional common filters (volume, RSI, EMA, etc.) ────────────────────
    def _all_elements(self) -> dict[str, str]:
        """
        Returneaza TOATE elementele disponibile: cele specifice strategiei +
        filtrele comune din common_filters.OPTIONAL_FILTERS.
        Optional filters au prefix "opt_" pentru a nu se ciocni cu elementele
        specifice strategiei.
        """
        from strategies.common_filters import labels as _opt_labels
        out = dict(self.elements)
        for k, lbl in _opt_labels().items():
            out[f"opt_{k}"] = lbl
        return out

    def _apply_optional_filters(self, df, signal: str, elements: dict | None,
                                **ctx) -> tuple[bool, list[str]]:
        """
        Aplica toate filtrele opt_* care sunt bifate in elements.
        Returneaza (pass_all, list[reasons]).
        """
        if not elements or signal not in ("BUY", "SELL"):
            return True, []
        from strategies.common_filters import apply_all
        active_keys = set()
        for k, v in elements.items():
            if v and k.startswith("opt_"):
                active_keys.add(k[4:])   # strip "opt_" prefix
        return apply_all(df, signal, enabled_keys=active_keys, **ctx)

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
            "risk_score":    0.0,
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
        min_votes   : int = 2,    # STRICT: era 1 — acum default 2
        extra       : dict = None,
    ) -> dict:
        """
        Conviction-based confidence (scoring v2 STRICT).

        Scara conviction → confidence (mai severa):
          conviction 0-4   → setup slab    → ~15-35%
          conviction 5-7   → setup decent  → ~40-60%
          conviction 8-10  → setup solid   → ~65-82%
          conviction 11+   → setup excelent → 85-100%

        Cerinte STRICT:
          - min_votes = 2: semnal valid doar daca ≥2 TF-uri in aceeasi directie
          - Conflicte TF penalizate -12% (era -8%)
          - Bonus agreement doar pentru TF-uri identice (nu HOLD)
        """
        buy_v  = [r for r in tf_results if r["signal"] == "BUY"]
        sell_v = [r for r in tf_results if r["signal"] == "SELL"]
        n_buy, n_sell, n_total = len(buy_v), len(sell_v), len(tf_results)
        n_active = n_buy + n_sell

        best_buy  = max(buy_v,  key=lambda x: x.get("conviction", 0)) if buy_v  else None
        best_sell = max(sell_v, key=lambda x: x.get("conviction", 0)) if sell_v else None

        final   = "HOLD"
        best_tf = None
        n_win   = 0

        if best_buy and best_sell:
            buy_score  = best_buy["conviction"]  + n_buy  * 0.5
            sell_score = best_sell["conviction"] + n_sell * 0.5
            if buy_score >= sell_score:
                final, best_tf, n_win = "BUY",  best_buy,  n_buy
            else:
                final, best_tf, n_win = "SELL", best_sell, n_sell
        elif best_buy:
            final, best_tf, n_win = "BUY",  best_buy,  n_buy
        elif best_sell:
            final, best_tf, n_win = "SELL", best_sell, n_sell

        if best_tf is None:
            reasons_summary = []
            for r in tf_results:
                tf_name = r.get("tf", "?")
                r_list  = r.get("reasons", [])
                if r_list:
                    reasons_summary.append(f"{tf_name}: " + " | ".join(r_list[:3]))
            result = self._empty_result(symbol, "Toate TF-urile returneaza HOLD")
            result["tfs"] = tf_results
            result["justification"] = reasons_summary or ["Nicio conditie activa"]
            return result

        conv = best_tf.get("conviction", 0)

        # STRICT: conviction * 7.5 (era 8.5) — mai greu sa atingi 85%
        # conv 8 = 60%, conv 10 = 75%, conv 11 = 82.5%, conv 12 = 90%
        base_conf = min(conv * 7.5, 90.0)

        # Agreement bonus: +4% per TF aditional (era +5%)
        agree_bonus = max(0, (n_win - 1)) * 4.0

        # Penalizare conflict: -12% per TF opus (era -8%)
        n_disagree = n_active - n_win
        disagree_penalty = n_disagree * 12.0

        confidence = round(max(0.0, min(base_conf + agree_bonus - disagree_penalty, 100.0)), 1)

        # Risk Score: mai strict, conv/13 (era /14) — atins mai greu
        import numpy as np
        _rs_conv_norm   = min(conv / 13.0, 1.0)
        _rs_agree       = min((n_win - 1) * 0.12, 0.36)
        _rs_conflict    = min(n_disagree * 0.25, 0.75)  # era 0.20
        _rs_tf_depth    = min((n_total - 1) * 0.04, 0.12)
        risk_score = round(
            min(100.0, max(0.0,
                _rs_conv_norm * (1.0 + _rs_agree - _rs_conflict + _rs_tf_depth) * 100.0
            )), 1
        )

        # Prag efectiv: STRICT — minim 70%, scaleaza pana la user_min cu TF-uri multe
        # 1 TF=70%, 2 TFs=75%, 3 TFs=80%, 4+ TFs=user_min
        effective_min = min(min_confidence, max(70.0, 70.0 + (n_total - 1) * 5.0))

        # STRICT GATE 1: min_votes
        if n_win < min_votes:
            final   = "HOLD"
            best_tf = None

        # STRICT GATE 2: confidence sub prag
        if confidence < effective_min:
            final   = "HOLD"
            best_tf = None

        justification = []
        if best_tf:
            justification += [r.replace(" ✓", "") for r in best_tf.get("reasons", [])]
            justification.append(
                f"Confidence {confidence}% (conviction {conv}"
                f", {n_win}/{n_active} TF-uri active"
                + (f", +{agree_bonus:.0f}% agreement" if agree_bonus > 0 else "")
                + (f", -{disagree_penalty:.0f}% conflict" if disagree_penalty > 0 else "")
                + ")"
            )
        else:
            prag_info = f"{effective_min:.0f}%" + (f" (auto din {n_total} TF)" if effective_min < min_confidence else "")
            vote_info = f", min {min_votes} voturi" if min_votes > 1 else ""
            justification.append(
                f"Confidence {confidence}% sub prag {prag_info}{vote_info}"
                f" (conviction {conv}, {n_buy} BUY / {n_sell} SELL)"
            )

        result = {
            "symbol":        symbol,
            "strategy":      self.key,
            "timestamp":     self._now(),
            "signal":        final,
            "confidence":    confidence,
            "risk_score":    risk_score if final != "HOLD" else 0.0,
            "conviction":    conv,
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
