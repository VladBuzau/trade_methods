"""
OrderManager — gestiunea ordinelor AutoOrders v2.
Persistenta: orders_store.json
Dedup: max 1 BUY + 1 SELL per (symbol, strategy) simultan.
"""
from __future__ import annotations
import json
import uuid
import threading
import logging
from datetime import datetime
from pathlib import Path
from typing import Optional

log = logging.getLogger(__name__)

STATUS_ACTIVE = {"pending_mt5", "pending_virtual"}
STATUS_CLOSED = {"filled", "cancelled", "expired", "closed_tp", "closed_sl", "replaced"}
ALL_STATUSES  = STATUS_ACTIVE | STATUS_CLOSED


def _now() -> str:
    return datetime.now().isoformat()


class OrderManager:
    """
    Gestioneaza ordinele AutoOrders:
    - Persistenta atomica in orders_store.json
    - Dedup: max 1 BUY + 1 SELL per (symbol, strategy)
    - Sync cu MT5: detecteaza filled/expired automat
    """

    def __init__(self, store_path: str = "orders_store.json"):
        self._path  = Path(store_path)
        self._lock  = threading.Lock()
        self._orders: dict[str, dict] = {}
        self._load()

    # ── Can place? ────────────────────────────────────────────────────────────

    def can_place(self, symbol: str, strategy: str, direction: str) -> bool:
        """True daca nu exista un ordin activ pentru (symbol, strategy, direction)."""
        with self._lock:
            return not self._find_active(symbol, strategy, direction)

    # ── CRUD ──────────────────────────────────────────────────────────────────

    def place(
        self,
        symbol:      str,
        strategy:    str,
        direction:   str,       # "BUY" | "SELL"
        mode:        str,       # "mt5_pending" | "virtual"
        entry:       float,
        sl:          float,
        tp:          float,
        source_zone: dict,
        mt5_ticket:  int   = 0,
        risk_usd:    float = 100.0,
    ) -> dict:
        strat_tag = strategy.upper().replace("_", "")[:6]
        oid = uuid.uuid4().hex[:10].upper()
        rr  = round(abs(tp - entry) / abs(entry - sl), 2) if abs(entry - sl) > 0 else 0
        order = {
            "id":          oid,
            "symbol":      symbol,
            "strategy":    strategy,
            "direction":   direction,
            "mode":        mode,
            "entry":       round(float(entry), 5),
            "sl":          round(float(sl), 5),
            "tp":          round(float(tp), 5),
            "rr":          rr,
            "risk_usd":    float(risk_usd),
            "status":      "pending_mt5" if mode == "mt5_pending" else "pending_virtual",
            "mt5_ticket":  int(mt5_ticket),
            "comment":     f"AO_{strat_tag}_{direction}_P",
            "source_zone": source_zone or {},
            "created":     _now(),
            "filled":      None,
            "closed":      None,
            "fill_price":  None,
            "pnl":         None,
            "reason":      None,
        }
        with self._lock:
            self._orders[oid] = order
            self._save()
        log.info(f"[OM] place {oid} {symbol} {strategy} {direction} {mode} @ {entry}")
        return order

    def cancel(self, order_id: str, reason: str = "manual") -> bool:
        with self._lock:
            o = self._orders.get(order_id)
            if not o or o["status"] not in STATUS_ACTIVE:
                return False
            o["status"] = "cancelled"
            o["closed"] = _now()
            o["reason"] = reason
            self._save()
        log.info(f"[OM] cancel {order_id} reason={reason}")
        return True

    def replace(
        self,
        order_id:   str,
        new_entry:  float,
        new_sl:     float,
        new_tp:     float,
        source_zone: dict,
        reason:     str = "zone_moved",
    ) -> Optional[dict]:
        """Marcheza ordinul vechi ca replaced si plaseaza unul nou identic cu zona noua."""
        with self._lock:
            old = self._orders.get(order_id)
            if not old or old["status"] not in STATUS_ACTIVE:
                return None
            old_copy = dict(old)
            old["status"] = "replaced"
            old["closed"] = _now()
            old["reason"] = reason
            self._save()
        return self.place(
            old_copy["symbol"], old_copy["strategy"], old_copy["direction"],
            old_copy["mode"], new_entry, new_sl, new_tp, source_zone,
            risk_usd=old_copy.get("risk_usd", 100.0),
        )

    def mark_filled(self, order_id: str, fill_price: float, mt5_ticket: int = 0):
        with self._lock:
            o = self._orders.get(order_id)
            if o:
                o["status"]     = "filled"
                o["filled"]     = _now()
                o["fill_price"] = round(float(fill_price), 5)
                if mt5_ticket:
                    o["mt5_ticket"] = int(mt5_ticket)
                self._save()

    def mark_closed(self, order_id: str, status: str, pnl: float = None, reason: str = None):
        with self._lock:
            o = self._orders.get(order_id)
            if o:
                o["status"] = status
                o["closed"] = _now()
                if pnl is not None:
                    o["pnl"] = round(float(pnl), 2)
                if reason:
                    o["reason"] = reason
                self._save()

    def update_ticket(self, order_id: str, mt5_ticket: int):
        with self._lock:
            o = self._orders.get(order_id)
            if o:
                o["mt5_ticket"] = int(mt5_ticket)
                self._save()

    def modify(self, order_id: str, sl: float = None, tp: float = None) -> bool:
        with self._lock:
            o = self._orders.get(order_id)
            if not o or o["status"] not in STATUS_ACTIVE:
                return False
            if sl is not None:
                o["sl"] = round(float(sl), 5)
            if tp is not None:
                o["tp"] = round(float(tp), 5)
            entry = o["entry"]
            if abs(entry - o["sl"]) > 0:
                o["rr"] = round(abs(o["tp"] - entry) / abs(entry - o["sl"]), 2)
            self._save()
        return True

    # ── Queries ───────────────────────────────────────────────────────────────

    def get_all(
        self,
        status_filter: set   = None,
        symbol:        str   = None,
        strategy:      str   = None,
        limit:         int   = 200,
    ) -> list[dict]:
        with self._lock:
            orders = list(self._orders.values())
        if status_filter:
            orders = [o for o in orders if o["status"] in status_filter]
        if symbol:
            orders = [o for o in orders if o["symbol"] == symbol]
        if strategy:
            orders = [o for o in orders if o["strategy"] == strategy]
        orders = sorted(orders, key=lambda x: x["created"], reverse=True)
        return orders[:limit]

    def get_active(self, symbol: str = None, strategy: str = None) -> list[dict]:
        return self.get_all(STATUS_ACTIVE, symbol, strategy)

    def get_by_ticket(self, ticket: int) -> Optional[dict]:
        ticket = int(ticket)
        with self._lock:
            for o in self._orders.values():
                if o.get("mt5_ticket") == ticket:
                    return dict(o)
        return None

    def get_by_id(self, order_id: str) -> Optional[dict]:
        with self._lock:
            o = self._orders.get(order_id)
            return dict(o) if o else None

    def stats(self) -> dict:
        with self._lock:
            orders = list(self._orders.values())
        active   = [o for o in orders if o["status"] in STATUS_ACTIVE]
        filled   = [o for o in orders if o["status"] == "filled"]
        closed   = [o for o in orders if o["status"] in STATUS_CLOSED]
        pnl_list = [o["pnl"] for o in orders if o.get("pnl") is not None]
        return {
            "total":    len(orders),
            "active":   len(active),
            "filled":   len(filled),
            "closed":   len(closed),
            "total_pnl": round(sum(pnl_list), 2) if pnl_list else 0,
        }

    # ── MT5 sync ──────────────────────────────────────────────────────────────

    def sync_mt5(self, mt5) -> list[str]:
        """
        Verifica ordinele pending_mt5 versus starea reala din MT5.
        - Ticket in pozitii → mark_filled
        - Ticket disparut din pending/pozitii → mark expired
        Returneaza lista de order_id modificate.
        """
        changed = []
        try:
            if not mt5.initialize():
                return []
            pending_tix = {o.ticket for o in (mt5.orders_get()    or [])}
            pos_tix     = {p.ticket for p in (mt5.positions_get() or [])}

            for o in self.get_all({"pending_mt5"}):
                ticket = o.get("mt5_ticket", 0)
                if not ticket:
                    continue
                if ticket in pos_tix:
                    pos = next((p for p in (mt5.positions_get() or [])
                                if p.ticket == ticket), None)
                    price = pos.price_open if pos else o["entry"]
                    self.mark_filled(o["id"], price, ticket)
                    changed.append(o["id"])
                elif ticket not in pending_tix:
                    self.mark_closed(o["id"], "expired", reason="not_in_mt5")
                    changed.append(o["id"])
        except Exception as exc:
            log.warning(f"[OM] sync_mt5: {exc}")
        return changed

    # ── Persistence ───────────────────────────────────────────────────────────

    def _find_active(self, symbol: str, strategy: str, direction: str) -> list[dict]:
        return [
            o for o in self._orders.values()
            if o["symbol"]    == symbol
            and o["strategy"] == strategy
            and o["direction"]== direction
            and o["status"]   in STATUS_ACTIVE
        ]

    def _load(self):
        try:
            if self._path.exists():
                data = json.loads(self._path.read_text(encoding="utf-8"))
                self._orders = {o["id"]: o for o in data.get("orders", [])}
                log.info(f"[OM] loaded {len(self._orders)} orders from {self._path}")
        except Exception as exc:
            log.warning(f"[OM] load failed: {exc}")
            self._orders = {}

    def _save(self):
        try:
            tmp = self._path.with_suffix(".tmp")
            tmp.write_text(
                json.dumps({"orders": list(self._orders.values())},
                           indent=2, ensure_ascii=False),
                encoding="utf-8",
            )
            tmp.replace(self._path)
        except Exception as exc:
            log.warning(f"[OM] save failed: {exc}")
