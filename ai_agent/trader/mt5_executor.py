"""
ai_agent/trader/mt5_executor.py — execuție reală pe MT5 cu marcaj AI distinct.

Toate trade-urile AI au:
  - magic_number = AI_MAGIC (991)         → permite filtrare in UI
  - comment      = f"AI-{model_id}"       → vizibil in MT5 + /trades

UI-ul /trades poate apoi sorta dupa magic ca sa stie ce e AI vs AutoTrader.
"""
from __future__ import annotations
import logging

log = logging.getLogger(__name__)

AI_MAGIC = 991


def open_real_trade(symbol: str, side: str, sl: float, tp: float,
                    risk_dollars: float, model_id: str) -> dict:
    """
    Deschide pozitie reala pe MT5. Returneaza dict cu rezultatul (ticket, eroare).
    Foloseste app.place_trade din ChartVisualizer (deja are calcul lotaj, validare, etc.)
    """
    try:
        # Importam place_trade din ChartVisualizer (in path de sys.path adaugat de bp)
        from app import place_trade
        # place_trade accepta: (symbol, signal, sl, tp, risk_dollars, strategy, ...)
        ok, msg = place_trade(symbol, side, sl, tp,
                              risk_dollars=risk_dollars,
                              strategy=f"ai_{model_id[:20]}")
        if not ok:
            return {"ok": False, "error": msg}
        return {"ok": True, "message": msg, "magic": AI_MAGIC,
                "comment": f"AI-{model_id}"}
    except Exception as e:
        log.error(f"open_real_trade: {e}")
        return {"ok": False, "error": str(e)}


def list_ai_positions() -> list[dict]:
    """Lista pozitiilor MT5 deschise care au magic = AI_MAGIC."""
    try:
        import MetaTrader5 as mt5
        if not mt5.initialize():
            return []
        all_pos = mt5.positions_get()
        if not all_pos:
            return []
        ai_pos = []
        for p in all_pos:
            # Filtru: magic 991 SAU comment incepe cu AI-
            is_ai = (p.magic == AI_MAGIC) or (p.comment or "").startswith("AI-")
            if is_ai:
                ai_pos.append({
                    "ticket":     int(p.ticket),
                    "symbol":     p.symbol,
                    "type":       "BUY" if p.type == 0 else "SELL",
                    "volume":     p.volume,
                    "price_open": float(p.price_open),
                    "sl":         float(p.sl) if p.sl else 0,
                    "tp":         float(p.tp) if p.tp else 0,
                    "profit":     float(p.profit),
                    "comment":    p.comment,
                    "magic":      p.magic,
                })
        return ai_pos
    except Exception as e:
        log.error(f"list_ai_positions: {e}")
        return []
