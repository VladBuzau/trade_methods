"""
Auto-discovery pentru strategii de tranzactionare.
Fiecare strategie = subfolder propriu cu __init__.py.
Adaugi un folder nou → apare automat in UI la urmatoarea pornire.
"""
from __future__ import annotations
import importlib
import logging
from datetime import datetime, timezone
from pathlib import Path

from .base import Strategy

log = logging.getLogger(__name__)

# ── Backtest time injection ───────────────────────────────────────────────────
# Set by backtest engine before each analyze() call so all session checks
# use the bar's actual timestamp instead of real wall-clock time.
BT_BAR_UTC: datetime | None = None


def get_utc_now() -> datetime:
    """UTC now — returns BT_BAR_UTC during backtest, real clock otherwise."""
    return BT_BAR_UTC if BT_BAR_UTC is not None else datetime.now(timezone.utc)

_REGISTRY: dict[str, Strategy] = {}


def _discover():
    pkg_dir = Path(__file__).parent
    for item in sorted(pkg_dir.iterdir()):
        # Accepta doar subfoldere cu __init__.py (nu fisiere .py din radacina)
        if not item.is_dir():
            continue
        if item.name.startswith("_"):
            continue
        init_file = item / "__init__.py"
        if not init_file.exists():
            continue
        try:
            mod = importlib.import_module(f".{item.name}", package=__name__)
            for attr_name in dir(mod):
                obj = getattr(mod, attr_name)
                if (
                    isinstance(obj, type)
                    and issubclass(obj, Strategy)
                    and obj is not Strategy
                    and obj.key != "base"
                ):
                    instance = obj()
                    _REGISTRY[instance.key] = instance
                    log.debug(f"Strategie inregistrata: {instance.key} ({instance.name})")
        except Exception as exc:
            log.warning(f"Nu am putut incarca strategia '{item.name}': {exc}")


_discover()


_AUTOTRADER_EXCLUDED = {"ai_agent"}   # ruleaza separat in /ai_trader


def list_all() -> list[Strategy]:
    """Lista strategiilor pentru scanner-ul AutoTrader (fara ai_agent)."""
    order = [
        # Core
        "eob", "smc", "sr_mtf",
        "trend_rider",          # meta: inlocuieste classic/ema/macd/supertrend (eliminate)
        # Scalpers / short-term
        "vwap_reversion",       # VWAP mean reversion + regime filter (research-based)
        "sr_bounce",            # S/R bounce scalp, trades max 30-60 min
        "burst_scalper",        # EMA+Stoch+RSI pe M1
        "candle_sniper",        # EOB pe M1/M5
        "gold_scalper",         # dedicat XAUUSD
        # Reversal / patterns
        "bollinger", "engulfing", "rsi_divergence", "ichimoku",
        # Session breakouts
        "london_breakout", "ny_breakout", "china_session",
    ]
    result = [_REGISTRY[k] for k in order if k in _REGISTRY]
    for k, s in _REGISTRY.items():
        if k not in order and k not in _AUTOTRADER_EXCLUDED:
            result.append(s)
    return result


def get_strategy(key: str) -> Strategy | None:
    return _REGISTRY.get(key)


def get_enabled(scanner_config: dict) -> list[tuple[str, Strategy]]:
    return [
        (s.key, s) for s in list_all()
        if scanner_config.get(s.key, {}).get("enabled", False)
    ]


def as_defs_json() -> list[dict]:
    return [
        {"key": s.key, "name": s.name, "icon": s.icon, "color": s.color}
        for s in list_all()
    ]
