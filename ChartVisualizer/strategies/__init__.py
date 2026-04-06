"""
Auto-discovery pentru strategii de tranzactionare.
Fiecare strategie = subfolder propriu cu __init__.py.
Adaugi un folder nou → apare automat in UI la urmatoarea pornire.
"""
from __future__ import annotations
import importlib
import logging
from pathlib import Path

from .base import Strategy

log = logging.getLogger(__name__)

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


def list_all() -> list[Strategy]:
    order = [
        "classic", "smc",
        "macd", "bollinger", "supertrend",
        "london_breakout", "ny_breakout",
        "rsi_divergence", "engulfing", "ichimoku", "ema_cross",
    ]
    result = [_REGISTRY[k] for k in order if k in _REGISTRY]
    for k, s in _REGISTRY.items():
        if k not in order:
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
