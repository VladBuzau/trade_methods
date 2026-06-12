"""
AutoOrders — pagina dedicata ordinelor pending/predictive.
Thread si setari separate de AutoTrader.

── Cum adaugi o strategie noua ──────────────────────────────────────────────
1. Importeaza clasa in _load_strategies() de mai jos
2. Adaug-o in _STRATEGIES cu un dict:
      {"label": "Nume afisat", "obj": StrategiaInstantiata(), "pending": True}
   pending=True inseamna ca strategia intoarce "pending_entry" in rezultat.
   pending=False = strategia genereaza semnal market normal (afisam doar pe chart).
3. Gata. Tot restul (chart, scan, UI) se adapteaza automat.
─────────────────────────────────────────────────────────────────────────────
"""
import threading
import time
import json
import logging
from datetime import datetime

import plotly.graph_objects as go
from plotly.subplots import make_subplots
from flask import Blueprint, request, Response

from app import (
    fetch, MT5_AVAILABLE, mt5, NpEncoder, login_required,
    place_pending_order, place_trade, ALL_TFS, SYMBOLS, SYMBOLS_CRYPTO,
)
from order_manager import OrderManager, STATUS_ACTIVE, STATUS_CLOSED

log = logging.getLogger(__name__)

autoorders_bp = Blueprint("autoorders", __name__)

# ── Registry strategii ─────────────────────────────────────────────────────────
# Adauga orice strategie care suporta pending / predictive entry
_STRATEGIES: dict = {}


def _load_strategies():
    try:
        from strategies.eob import EOBStrategy, detect_eob_zones_v2, detect_eob_approach_zones
        _STRATEGIES["eob"] = {
            "label":   "EOB — Enhanced Order Block",
            "obj":     EOBStrategy(),
            "pending": True,
            # Functii optionale pentru chart overlay (None = nu se deseneaza)
            "fn_zones":    detect_eob_zones_v2,
            "fn_approach": detect_eob_approach_zones,
        }
    except Exception as exc:
        log.warning(f"[AutoOrders] EOB load: {exc}")

    # ── S/R Multi-Timeframe ──────────────────────────────────────────────────
    try:
        from strategies.sr_mtf import SRMTFStrategy, detect_sr_zones, detect_sr_mtf_zones
        _STRATEGIES["sr_mtf"] = {
            "label":       "📐 S/R Multi-Timeframe",
            "obj":         SRMTFStrategy(),
            "pending":     True,
            "fn_zones":    detect_sr_zones,      # fallback single-TF (df, lookback)
            "fn_approach": None,                  # nu e folosit — MTF are logica proprie
            "fn_zones_mtf": detect_sr_mtf_zones, # (symbol, fetch, df, price_now) → {zones, approach}
        }
    except Exception as exc:
        log.warning(f"[AutoOrders] SR-MTF load: {exc}")

    # ── Adauga strategii noi aici ────────────────────────────────────────────
    # try:
    #     from strategies.smc import SMCStrategy
    #     _STRATEGIES["smc"] = {
    #         "label":   "SMC — Smart Money Concept",
    #         "obj":     SMCStrategy(),
    #         "pending": False,
    #         "fn_zones": None, "fn_approach": None,
    #     }
    # except Exception as exc:
    #     log.warning(f"[AutoOrders] SMC load: {exc}")
    # ────────────────────────────────────────────────────────────────────────


_load_strategies()

# ── OrderManager singleton ─────────────────────────────────────────────────────
_order_manager = OrderManager(store_path="orders_store.json")

_TF_EXPIRY = {"M5": 3, "M15": 6, "M30": 12, "H1": 24, "H4": 72, "D1": 120, "W1": 240}

# ── State separat — nu partajeaza nimic cu autotrader.scanner ─────────────────
_state = {
    "running":            False,
    "strategy":           next(iter(_STRATEGIES), "eob"),
    "strategies_active":  [next(iter(_STRATEGIES), "eob")],   # E6: multi-strategy
    "symbol":             "EURUSD",
    "symbols_watch":      ["EURUSD"],                          # E6: multi-symbol
    "tf":                 "H1",
    "bars":               2000,
    "interval":           60,
    "auto_execute":       False,
    "exec_mode":          "mt5_pending",   # "mt5_pending" | "virtual"
    "expiry_hours":       24,
    "min_confidence":     60.0,
    "risk_usd":           100.0,
    "replace_thr":        0.3,             # ATR multiplier pt zone invalidation
    "virtual_tick_s":     5,
    "last_scan":          None,
    "decisions":          [],
    "_last_res":          None,
}
_lock   = threading.Lock()
_thread: threading.Thread | None = None


# ── Zone extractor unificat ───────────────────────────────────────────────────
def _get_zones_for_strategy(symbol: str, strat_key: str) -> dict:
    """
    Returneaza {'buy': zone|None, 'sell': zone|None, 'atr': float, 'price': float}.
    zone = {'entry', 'sl', 'tp', 'zone_low', 'zone_high'}
    Analiza strict pe TF-ul curent, fara cross-TF mixing.
    """
    strat = _STRATEGIES.get(strat_key, {})
    tf    = _state["tf"]
    bars  = _state["bars"]
    try:
        df, _ = fetch(symbol, tf, bars)
        if df is None or df.empty:
            return {"buy": None, "sell": None, "atr": 0, "price": 0}
        df.columns = [c.lower() for c in df.columns]
        price_now = float(df["close"].iloc[-1])
        from strategies.sr_mtf import _atr as _calc_atr
        atr = _calc_atr(df)

        buy_zone = sell_zone = None
        fn_approach  = strat.get("fn_approach")
        fn_zones_mtf = strat.get("fn_zones_mtf")

        if fn_approach:
            bear_ap = fn_approach(df, "BEARISH", price_now)
            bull_ap = fn_approach(df, "BULLISH", price_now)
            if bear_ap:
                z  = bear_ap[0]
                e  = z.get("pending_entry", z["zone_low"])
                sl = round(z["zone_high"] + atr * 0.5, 5)
                tp = round(e - abs(sl - e) * 2, 5)
                sell_zone = {"entry": e, "sl": sl, "tp": tp,
                             "zone_low": z["zone_low"], "zone_high": z["zone_high"]}
            if bull_ap:
                z  = bull_ap[0]
                e  = z.get("pending_entry", z["zone_high"])
                sl = round(z["zone_low"] - atr * 0.5, 5)
                tp = round(e + abs(e - sl) * 2, 5)
                buy_zone = {"entry": e, "sl": sl, "tp": tp,
                            "zone_low": z["zone_low"], "zone_high": z["zone_high"]}
        elif fn_zones_mtf:
            mtf = fn_zones_mtf(symbol, fetch, df, price_now)
            for z in mtf.get("approach", []):
                if z.get("type") == "BEARISH" and sell_zone is None:
                    e  = z.get("pending_entry", z["zone_low"])
                    sl = round(z["zone_high"] + atr * 0.5, 5)
                    tp = round(e - abs(sl - e) * 2, 5)
                    sell_zone = {"entry": e, "sl": sl, "tp": tp,
                                 "zone_low": z["zone_low"], "zone_high": z["zone_high"]}
                elif z.get("type") != "BEARISH" and buy_zone is None:
                    e  = z.get("pending_entry", z["zone_high"])
                    sl = round(z["zone_low"] - atr * 0.5, 5)
                    tp = round(e + abs(e - sl) * 2, 5)
                    buy_zone = {"entry": e, "sl": sl, "tp": tp,
                                "zone_low": z["zone_low"], "zone_high": z["zone_high"]}
                if buy_zone and sell_zone:
                    break
        return {"buy": buy_zone, "sell": sell_zone, "atr": atr, "price": price_now}
    except Exception as exc:
        log.warning(f"[AutoOrders] _get_zones {symbol}/{strat_key}: {exc}")
        return {"buy": None, "sell": None, "atr": 0, "price": 0}


def _cancel_mt5_order(ticket: int) -> bool:
    """Anuleaza un ordin pending direct din MT5."""
    if not MT5_AVAILABLE or not mt5:
        return False
    try:
        mt5.initialize()
        req = {"action": mt5.TRADE_ACTION_REMOVE, "order": int(ticket)}
        res = mt5.order_send(req)
        return bool(res and res.retcode == mt5.TRADE_RETCODE_DONE)
    except Exception as exc:
        log.warning(f"[AutoOrders] _cancel_mt5_order {ticket}: {exc}")
        return False


def _place_mt5_pending_for_order(order: dict) -> int:
    """Plaseaza un ordin pending MT5 si returneaza ticket-ul (0 = esec)."""
    try:
        ok, msg, ticket = place_pending_order(
            order["symbol"], order["direction"],
            order["entry"], order["sl"], order["tp"],
            risk_dollars=order.get("risk_usd", 100.0),
            strategy=order["strategy"],
            expiry_hours=_TF_EXPIRY.get(_state["tf"], _state["expiry_hours"]),
            comment=order["comment"],
        ) if _has_ticket_return() else _place_pending_compat(order)
        return ticket if ok else 0
    except Exception as exc:
        log.warning(f"[AutoOrders] _place_mt5_pending_for_order: {exc}")
        return 0


def _has_ticket_return() -> bool:
    """Verifica daca place_pending_order returneaza 3-tuple (ok, msg, ticket)."""
    import inspect
    try:
        sig = inspect.signature(place_pending_order)
        return True
    except Exception:
        return False


def _place_pending_compat(order: dict):
    """Fallback daca place_pending_order returneaza (ok, msg)."""
    ok, msg = place_pending_order(
        order["symbol"], order["direction"],
        order["entry"], order["sl"], order["tp"],
        risk_dollars=order.get("risk_usd", 100.0),
        strategy=order["strategy"],
        expiry_hours=_TF_EXPIRY.get(_state["tf"], _state["expiry_hours"]),
    )
    return ok, msg, 0


def _check_replacements(symbol: str, strat_key: str, zones: dict, atr: float):
    """
    Verifica ordinele active pentru (symbol, strat_key).
    - Daca zona a disparut → cancel
    - Daca zona s-a mutat > replace_thr * ATR → replace
    """
    replace_thr = _state.get("replace_thr", 0.3)
    for o in _order_manager.get_active(symbol, strat_key):
        direction = o["direction"]
        zone = zones.get("buy" if direction == "BUY" else "sell")

        if zone is None:
            if o["mode"] == "mt5_pending" and o.get("mt5_ticket"):
                _cancel_mt5_order(o["mt5_ticket"])
            _order_manager.cancel(o["id"], reason="zone_gone")
            log.info(f"[AutoOrders] cancelled {o['id']} zone gone")
            continue

        dist = abs(zone["entry"] - o["entry"])
        if atr > 0 and dist > replace_thr * atr:
            if o["mode"] == "mt5_pending" and o.get("mt5_ticket"):
                _cancel_mt5_order(o["mt5_ticket"])
            new_ord = _order_manager.replace(
                o["id"], zone["entry"], zone["sl"], zone["tp"],
                zone, reason="zone_moved"
            )
            if new_ord and o["mode"] == "mt5_pending":
                ticket = _place_mt5_pending_for_order(new_ord)
                if ticket:
                    _order_manager.update_ticket(new_ord["id"], ticket)
            log.info(f"[AutoOrders] replaced {o['id']} dist={dist:.5f} > {replace_thr}*ATR")


# ── Scanner v2 — multi symbol × strategy ─────────────────────────────────────
def _scan_all():
    """
    Scanner principal: itereaza symbols_watch × strategies_active.
    Plaseaza max 1 BUY + 1 SELL per (symbol, strategy) via OrderManager.
    Mentine sidebar analysis pentru simbolul/strategia activa.
    """
    symbols    = _state.get("symbols_watch") or [_state["symbol"]]
    strat_keys = _state.get("strategies_active") or [_state["strategy"]]
    exec_mode  = _state.get("exec_mode", "mt5_pending")
    risk_usd   = _state.get("risk_usd", 100.0)
    auto_ex    = _state["auto_execute"]

    # Sync MT5 state la fiecare scan
    if MT5_AVAILABLE and mt5:
        try:
            _order_manager.sync_mt5(mt5)
        except Exception:
            pass

    decisions_batch = []

    for symbol in symbols:
        for strat_key in strat_keys:
            if strat_key not in _STRATEGIES:
                continue
            try:
                zones = _get_zones_for_strategy(symbol, strat_key)
                atr   = zones["atr"]

                # Verifica replacement pentru ordinele active
                _check_replacements(symbol, strat_key, zones, atr)

                if not auto_ex:
                    continue

                # Plaseaza ordine noi unde lipsesc
                for direction, zone_key in [("BUY", "buy"), ("SELL", "sell")]:
                    zone = zones.get(zone_key)
                    if zone is None:
                        continue
                    if not _order_manager.can_place(symbol, strat_key, direction):
                        continue

                    order = _order_manager.place(
                        symbol, strat_key, direction, exec_mode,
                        zone["entry"], zone["sl"], zone["tp"],
                        zone, risk_usd=risk_usd,
                    )

                    if exec_mode == "mt5_pending":
                        ticket = _place_mt5_pending_for_order(order)
                        executed = ticket > 0
                        if ticket:
                            _order_manager.update_ticket(order["id"], ticket)
                        msg = f"MT5 ticket #{ticket}" if executed else "MT5 esuat"
                    else:
                        executed = True
                        msg = "Virtual agent watchlist"

                    decisions_batch.append({
                        "ts":       datetime.now().strftime("%H:%M:%S"),
                        "signal":   f"AUTO {direction}",
                        "executed": executed,
                        "result":   f"{symbol} {strat_key} @ {zone['entry']:.5f} — {msg}",
                        "strategy": strat_key,
                    })

            except Exception as exc:
                log.warning(f"[AutoOrders] scan_all {symbol}/{strat_key}: {exc}")

    # Sidebar analysis — simbolul/strategia activa (pentru display, nu pt plasare)
    try:
        cur_strat = _STRATEGIES.get(_state["strategy"])
        if cur_strat:
            res = cur_strat["obj"].analyze(
                _state["symbol"], [_state["tf"]],
                bars=_state["bars"], min_confidence=_state["min_confidence"]
            )
            with _lock:
                _state["_last_res"] = res
    except Exception:
        pass

    with _lock:
        _state["last_scan"] = datetime.now().isoformat()
        for d in decisions_batch:
            _state["decisions"].insert(0, d)
        while len(_state["decisions"]) > 80:
            _state["decisions"].pop()


def _scanner_loop():
    log.info("[AutoOrders] Scanner v2 pornit.")
    while _state["running"]:
        _scan_all()
        interval = _state["interval"]
        for _ in range(max(1, interval // 2)):
            if not _state["running"]:
                break
            time.sleep(2)
    log.info("[AutoOrders] Scanner v2 oprit.")


# ── Virtual Agent — monitorizeaza ordine virtual si executa market ────────────
class _VirtualAgent:
    """Thread care citeste tick-uri la fiecare virtual_tick_s secunde
    si executa market order cand pretul atinge entry-ul unui virtual order."""

    def __init__(self):
        self._running = False
        self._thread: threading.Thread | None = None

    def start(self):
        if self._running:
            return
        self._running = True
        self._thread = threading.Thread(
            target=self._loop, daemon=True, name="virtual_agent"
        )
        self._thread.start()
        log.info("[VirtualAgent] pornit")

    def stop(self):
        self._running = False
        log.info("[VirtualAgent] oprit")

    def _loop(self):
        while self._running:
            try:
                self._check()
            except Exception as exc:
                log.warning(f"[VirtualAgent] loop exc: {exc}")
            time.sleep(max(1, _state.get("virtual_tick_s", 5)))

    def _check(self):
        if not MT5_AVAILABLE or not mt5:
            return
        mt5.initialize()
        for order in _order_manager.get_all({"pending_virtual"}):
            try:
                tick = mt5.symbol_info_tick(order["symbol"])
                if not tick:
                    continue
                price = tick.ask if order["direction"] == "BUY" else tick.bid
                entry = order["entry"]
                hit = (
                    (order["direction"] == "BUY"  and price <= entry) or
                    (order["direction"] == "SELL" and price >= entry)
                )
                if hit:
                    self._fire(order, price)
            except Exception as exc:
                log.warning(f"[VirtualAgent] check order {order['id']}: {exc}")

    def _fire(self, order: dict, fill_price: float):
        try:
            ok, msg = place_trade(
                order["symbol"], order["direction"],
                order["sl"], order["tp"],
                order.get("risk_usd", 100.0),
                strategy="manual",
            )
            if ok:
                _order_manager.mark_filled(order["id"], fill_price)
                log.info(
                    f"[VirtualAgent] fired {order['id']} "
                    f"{order['symbol']} {order['direction']} @ {fill_price}"
                )
                with _lock:
                    _state["decisions"].insert(0, {
                        "ts":       datetime.now().strftime("%H:%M:%S"),
                        "signal":   f"VIRT EXEC {order['direction']}",
                        "executed": True,
                        "result":   f"{order['symbol']} @ {fill_price:.5f} — {msg}",
                        "strategy": order["strategy"],
                    })
            else:
                log.warning(f"[VirtualAgent] fire failed {order['id']}: {msg}")
        except Exception as exc:
            log.warning(f"[VirtualAgent] fire exc {order['id']}: {exc}")


_virtual_agent = _VirtualAgent()


# ── S/R per TF curent (fara cross-TF) ────────────────────────────────────────
def _find_sr_zones_simple(df, price_now: float):
    """
    Gaseste cel mai relevant support (BUY) si resistance (SELL) pe TF-ul curent.
    Nu foloseste alte TF-uri — analiza strict pe datele primite.
    Returneaza (buy_zone | None, sell_zone | None).
    """
    try:
        from strategies.sr_mtf import _find_pivots, _cluster_levels, _atr, _TOL_PCT, _ZONE_ATR_MULT
    except ImportError:
        return None, None

    atr_val = _atr(df)
    if atr_val <= 0:
        return None, None
    half_w = atr_val * _ZONE_ATR_MULT / 2

    ph, pl = _find_pivots(df, window=5)
    n = len(df)
    raw = []
    for idx, price in ph:
        raw.append({"price": price, "tf": "CRT", "age": n - idx, "kind": "R"})
    for idx, price in pl:
        raw.append({"price": price, "tf": "CRT", "age": n - idx, "kind": "S"})

    clusters = _cluster_levels(raw, tol_pct=_TOL_PCT)
    valid    = [c for c in clusters if c["touches"] >= 2]

    resistances = sorted(
        [c for c in valid if c["price"] > price_now and c["kind"] == "R"],
        key=lambda x: x["price"])
    supports = sorted(
        [c for c in valid if c["price"] < price_now and c["kind"] == "S"],
        key=lambda x: -x["price"])

    sell_zone = None
    if resistances:
        r = resistances[0]
        sell_zone = {
            "zone_low":  round(r["price"] - half_w, 5),
            "zone_high": round(r["price"] + half_w, 5),
            "entry":     round(r["price"], 5),
            "touches":   r["touches"],
            "age":       r["age"],
        }
    buy_zone = None
    if supports:
        s = supports[0]
        buy_zone = {
            "zone_low":  round(s["price"] - half_w, 5),
            "zone_high": round(s["price"] + half_w, 5),
            "entry":     round(s["price"], 5),
            "touches":   s["touches"],
            "age":       s["age"],
        }
    return buy_zone, sell_zone


# ── Chart builder ─────────────────────────────────────────────────────────────
def _build_chart(symbol: str, tf: str, bars: int, strat_key: str) -> dict:
    """
    Chart complet construit de la zero: candlestick + volum + zona BUY + zona SELL.
    Fiecare TF analizat strict pe datele lui — fara cross-TF mixing.
    """
    import pandas as pd

    df, source = fetch(symbol, tf, bars)
    if df is None or df.empty:
        return {}

    # ── Normalizeaza indexul la DatetimeIndex (MT5 poate returna int Unix) ────
    df.columns = [c.lower() for c in df.columns]
    if not isinstance(df.index, pd.DatetimeIndex):
        try:
            df.index = pd.to_datetime(df.index, unit="s")
        except Exception:
            try:
                df.index = pd.to_datetime(df.index)
            except Exception:
                pass

    price_now = float(df["close"].iloc[-1])
    dates     = df.index
    n         = len(df)

    has_vol = "tick_volume" in df.columns or "volume" in df.columns
    n_rows  = 2 if has_vol else 1
    row_h   = [0.78, 0.22] if has_vol else [1.0]

    fig = make_subplots(
        rows=n_rows, cols=1,
        row_heights=row_h,
        shared_xaxes=True,
        vertical_spacing=0.02,
    )

    # ── Candlestick ──────────────────────────────────────────────────────────
    fig.add_trace(go.Candlestick(
        x=dates,
        open=df["open"], high=df["high"],
        low=df["low"],   close=df["close"],
        name="OHLC",
        increasing_line_color="#26a69a", decreasing_line_color="#ef5350",
        increasing_fillcolor="#26a69a",  decreasing_fillcolor="#ef5350",
        showlegend=False,
    ), row=1, col=1)

    # ── Volum ────────────────────────────────────────────────────────────────
    if has_vol:
        vol_col  = "tick_volume" if "tick_volume" in df.columns else "volume"
        v_colors = [
            "#26a69a" if df["close"].iloc[i] >= df["open"].iloc[i] else "#ef5350"
            for i in range(n)
        ]
        fig.add_trace(go.Bar(
            x=dates, y=df[vol_col],
            marker_color=v_colors,
            name="Vol", showlegend=False, opacity=0.5,
        ), row=2, col=1)

    # ── Zone pe TF-ul curent — in functie de strategia activa ────────────────
    strat       = _STRATEGIES.get(strat_key, {})
    fn_zones    = strat.get("fn_zones")
    fn_approach = strat.get("fn_approach")
    fn_zones_mtf = strat.get("fn_zones_mtf")

    buy_zone  = None
    sell_zone = None

    if fn_approach:
        try:
            bear_ap = fn_approach(df, "BEARISH", price_now)
            bull_ap = fn_approach(df, "BULLISH", price_now)
            if bear_ap:
                z = bear_ap[0]
                sell_zone = {"entry": z.get("pending_entry", z["zone_low"]),
                             "zone_low": z["zone_low"], "zone_high": z["zone_high"],
                             "label": "SELL"}
            if bull_ap:
                z = bull_ap[0]
                buy_zone  = {"entry": z.get("pending_entry", z["zone_high"]),
                             "zone_low": z["zone_low"], "zone_high": z["zone_high"],
                             "label": "BUY"}
        except Exception as exc:
            log.warning(f"[AutoOrders] chart EOB zones: {exc}")

    elif fn_zones_mtf:
        try:
            mtf = fn_zones_mtf(symbol, fetch, df, price_now)
            for z in mtf.get("approach", []):
                if z.get("type") == "BEARISH" and sell_zone is None:
                    sell_zone = {"entry": z.get("pending_entry", z["zone_low"]),
                                 "zone_low": z["zone_low"], "zone_high": z["zone_high"],
                                 "label": f"SELL  {'+'.join(z.get('tfs',[]))}"}
                elif z.get("type") != "BEARISH" and buy_zone is None:
                    buy_zone  = {"entry": z.get("pending_entry", z["zone_high"]),
                                 "zone_low": z["zone_low"], "zone_high": z["zone_high"],
                                 "label": f"BUY  {'+'.join(z.get('tfs',[]))}"}
                if sell_zone and buy_zone:
                    break
        except Exception as exc:
            log.warning(f"[AutoOrders] chart S/R MTF zones: {exc}")

    else:
        buy_zone, sell_zone = _find_sr_zones_simple(df, price_now)
        if sell_zone: sell_zone["label"] = "SELL"
        if buy_zone:  buy_zone["label"]  = "BUY"

    # ── Zona SELL ─────────────────────────────────────────────────────────────
    if sell_zone:
        fig.add_hrect(
            y0=sell_zone["zone_low"], y1=sell_zone["zone_high"], row=1, col=1,
            fillcolor="rgba(239,83,80,0.18)",
            line=dict(width=0),
        )
        # Eticheta pe zona (stanga)
        fig.add_annotation(
            x=0.01, y=(sell_zone["zone_low"] + sell_zone["zone_high"]) / 2,
            xref="paper", yref="y",
            text=f"<b>▼ SELL ZONE</b>",
            showarrow=False,
            font=dict(color="#ff5252", size=11, family="monospace"),
            bgcolor="rgba(20,5,5,0.75)",
            bordercolor="#ef5350", borderwidth=1, borderpad=3,
            xanchor="left",
        )
        # Price tag axa Y (dreapta)
        fig.add_annotation(
            x=1, y=sell_zone["entry"], xref="paper", yref="y",
            text=f"<b>{sell_zone['entry']:.5f}</b>",
            showarrow=False,
            font=dict(color="#ffffff", size=10, family="monospace"),
            bgcolor="#c62828", bordercolor="#ff5252",
            borderwidth=1, borderpad=4, xanchor="left",
        )

    # ── Zona BUY ──────────────────────────────────────────────────────────────
    if buy_zone:
        fig.add_hrect(
            y0=buy_zone["zone_low"], y1=buy_zone["zone_high"], row=1, col=1,
            fillcolor="rgba(38,166,154,0.18)",
            line=dict(width=0),
        )
        # Eticheta pe zona (stanga)
        fig.add_annotation(
            x=0.01, y=(buy_zone["zone_low"] + buy_zone["zone_high"]) / 2,
            xref="paper", yref="y",
            text=f"<b>▲ BUY ZONE</b>",
            showarrow=False,
            font=dict(color="#69f0ae", size=11, family="monospace"),
            bgcolor="rgba(5,20,15,0.75)",
            bordercolor="#26a69a", borderwidth=1, borderpad=3,
            xanchor="left",
        )
        # Price tag axa Y (dreapta)
        fig.add_annotation(
            x=1, y=buy_zone["entry"], xref="paper", yref="y",
            text=f"<b>{buy_zone['entry']:.5f}</b>",
            showarrow=False,
            font=dict(color="#ffffff", size=10, family="monospace"),
            bgcolor="#00695c", bordercolor="#69f0ae",
            borderwidth=1, borderpad=4, xanchor="left",
        )

    # ── Ordine pending MT5 — doar linia de entry ──────────────────────────────
    try:
        if MT5_AVAILABLE and mt5 and mt5.initialize():
            type_lbl = {2: "BUY LIM", 3: "SELL LIM", 4: "BUY STP", 5: "SELL STP"}
            for op in (mt5.orders_get(symbol=symbol) or []):
                is_buy  = op.type in (2, 4, 6)
                c_entry = "#26a69a" if is_buy else "#ef5350"
                lbl     = type_lbl.get(op.type, "PENDING")
                fig.add_annotation(
                    x=1, y=op.price_open, xref="paper", yref="y",
                    text=f"<b>{'▲' if is_buy else '▼'} #{op.ticket} {lbl}  {op.price_open:.5f}</b>",
                    showarrow=False,
                    font=dict(color="#ffffff", size=9, family="monospace"),
                    bgcolor="#1a237e" if is_buy else "#4a1010",
                    bordercolor=c_entry, borderwidth=1, borderpad=3,
                    xanchor="left",
                )
    except Exception as exc:
        log.warning(f"[AutoOrders] MT5 pending overlay: {exc}")

    # ── Layout ───────────────────────────────────────────────────────────────
    p_lo = float(df["low"].iloc[-view_n:].min())
    p_hi = float(df["high"].iloc[-view_n:].max())
    pad  = (p_hi - p_lo) * 0.08

    # X range — ca string ISO ca sa evitam orice mismatch de tip cu Plotly
    view_n = min(150, n)
    try:
        x_st_str  = str(dates[-view_n])[:19]
        x_end_str = str(dates[-1])[:19]
        bar_secs  = int((dates[-1] - dates[-2]).total_seconds())
        x_pad_str = str(dates[-1] + pd.Timedelta(seconds=bar_secs * 20))[:19]
    except Exception:
        x_st_str = x_end_str = x_pad_str = None

    spike = dict(showspikes=True, spikemode="across", spikesnap="cursor",
                 spikecolor="rgba(180,180,180,0.3)", spikethickness=1, spikedash="dot")

    x_axis = dict(gridcolor="#13131f", rangeslider=dict(visible=False), **spike)
    if x_st_str:
        x_axis.update(type="date", range=[x_st_str, x_pad_str], autorange=False)

    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="#07070e",
        plot_bgcolor="#07070e",
        height=520,
        dragmode="pan",
        hovermode="x unified",
        margin=dict(l=10, r=145, t=38, b=10),
        title=dict(
            text=(f"<b>{symbol}</b> — {tf}"
                  f"  |  <span style='color:#5c6bc0'>{strat.get('label', strat_key)}</span>"
                  f"  |  {n} bare"
                  + (f"  |  <span style='color:#ef5350'>▼ {sell_zone['entry']:.5f}</span>" if sell_zone else "")
                  + (f"  |  <span style='color:#26a69a'>▲ {buy_zone['entry']:.5f}</span>" if buy_zone else "")),
            font=dict(size=11, color="#c0c8e0"),
        ),
        xaxis=x_axis,
        yaxis=dict(range=[p_lo - pad, p_hi + pad], autorange=False,
                   gridcolor="#13131f", side="right", fixedrange=False, **spike),
        yaxis2=dict(gridcolor="#0f0f1e", side="right"),
    )

    # ── Calculeaza SL/TP pentru butoanele rapide ─────────────────────────────
    from strategies.sr_mtf import _atr as _calc_atr
    atr_val = _calc_atr(df)

    out = json.loads(fig.to_json())
    out["_price"] = round(price_now, 5)
    out["_atr"]   = round(atr_val, 5)

    if sell_zone:
        sl_s = round(sell_zone["zone_high"], 5)
        tp_s = round(sell_zone["entry"] - abs(sl_s - sell_zone["entry"]) * 2, 5)
        out["_sell_zone"] = {**sell_zone, "sl": sl_s, "tp": tp_s}

    if buy_zone:
        sl_b = round(buy_zone["zone_low"], 5)
        tp_b = round(buy_zone["entry"] + abs(buy_zone["entry"] - sl_b) * 2, 5)
        out["_buy_zone"] = {**buy_zone, "sl": sl_b, "tp": tp_b}

    return out


def _hex2rgb(hex_color: str) -> str:
    """Converteste #rrggbb in 'r,g,b' pentru rgba()."""
    h = hex_color.lstrip("#")
    return f"{int(h[0:2],16)},{int(h[2:4],16)},{int(h[4:6],16)}"


# ── MT5 helper — ordine pending active ───────────────────────────────────────
def _get_pending_orders(symbol: str, strat_key: str) -> list:
    if not MT5_AVAILABLE or mt5 is None:
        return []
    try:
        if not mt5.initialize():
            return []
        orders  = mt5.orders_get(symbol=symbol) or []
        now_ts  = int(datetime.now().timestamp())
        tag     = strat_key.upper()[:6]
        type_names = {
            2: "BUY_LIMIT", 3: "SELL_LIMIT",
            4: "BUY_STOP",  5: "SELL_STOP",
            6: "BUY_STP_LIM", 7: "SELL_STP_LIM",
        }
        out = []
        for op in orders:
            cmt = (op.comment or "").upper()
            if not (cmt.endswith("_P") and tag in cmt):
                continue
            exp_ts = getattr(op, "time_expiration", 0) or 0
            if exp_ts > 0:
                rem = exp_ts - now_ts
                if rem > 0:
                    h, r       = divmod(rem, 3600)
                    expiry_str = f"{int(h)}h {int(r // 60)}m"
                    expiry_dt  = datetime.fromtimestamp(exp_ts).strftime("%d.%m %H:%M")
                else:
                    expiry_str = "expirat"
                    expiry_dt  = "—"
            else:
                expiry_str = "GTC"
                expiry_dt  = "GTC"
            out.append({
                "ticket":    op.ticket,
                "symbol":    op.symbol,
                "type":      type_names.get(op.type, str(op.type)),
                "price":     op.price_open,
                "sl":        op.sl,
                "tp":        op.tp,
                "volume":    op.volume_current,
                "comment":   op.comment or "",
                "expiry":    expiry_str,
                "expiry_dt": expiry_dt,
                "exp_ts":    exp_ts,
                "is_buy":    op.type in (2, 4, 6),
                "rem_s":     max(0, exp_ts - now_ts) if exp_ts else 0,
            })
        return sorted(out, key=lambda x: x["exp_ts"] or 9_000_000_000)
    except Exception as exc:
        log.warning(f"[AutoOrders] _get_pending_orders: {exc}")
        return []


# ── Routes ────────────────────────────────────────────────────────────────────
@autoorders_bp.route("/autoorders/")
@login_required
def autoorders_page():
    all_syms   = SYMBOLS + SYMBOLS_CRYPTO
    tfs        = ALL_TFS + ["W1"]
    strat_opts = {k: v["label"] for k, v in _STRATEGIES.items()}
    with _lock:
        cur_state = dict(_state)
    return Response(_render(all_syms, tfs, strat_opts, cur_state),
                    content_type="text/html; charset=utf-8")


@autoorders_bp.route("/autoorders/status")
@login_required
def autoorders_status():
    with _lock:
        state_snap = {k: v for k, v in _state.items() if k != "_last_res"}
        res        = _state.get("_last_res")
        decs       = list(_state["decisions"][:50])
    strat_key = _state["strategy"]
    pending   = _get_pending_orders(_state["symbol"], strat_key)
    out_res   = None
    if res:
        pe = res.get("pending_entry")
        out_res = {
            "signal":        res.get("signal"),
            "confidence":    res.get("confidence", 0),
            "risk_score":    res.get("risk_score", 0),
            "justification": res.get("justification", []),
            "pending_entry": pe,
        }
    return Response(json.dumps({
        "ok": True,
        "state": state_snap,
        "result": out_res,
        "decisions": decs,
        "pending_orders": pending,
        "strategies": {k: v["label"] for k, v in _STRATEGIES.items()},
        "order_stats":   _order_manager.stats(),
    }, cls=NpEncoder), content_type="application/json")


@autoorders_bp.route("/autoorders/chart")
@login_required
def autoorders_chart():
    sym       = request.args.get("symbol",   _state["symbol"])
    tf        = request.args.get("tf",       _state["tf"])
    bars      = int(request.args.get("bars", _state["bars"]))
    strat_key = request.args.get("strategy", _state["strategy"])
    return Response(
        json.dumps(_build_chart(sym, tf, bars, strat_key), cls=NpEncoder),
        content_type="application/json",
    )


@autoorders_bp.route("/autoorders/start", methods=["POST"])
@login_required
def autoorders_start():
    global _thread
    _apply(request.get_json(silent=True) or {})
    if not _state["running"]:
        _state["running"] = True
        _thread = threading.Thread(target=_scanner_loop, daemon=True, name="autoorders")
        _thread.start()
    if _state["exec_mode"] == "virtual":
        _virtual_agent.start()
    return Response('{"ok":true}', content_type="application/json")


@autoorders_bp.route("/autoorders/stop", methods=["POST"])
@login_required
def autoorders_stop():
    _state["running"] = False
    _virtual_agent.stop()
    return Response('{"ok":true}', content_type="application/json")


@autoorders_bp.route("/autoorders/set", methods=["POST"])
@login_required
def autoorders_set():
    _apply(request.get_json(silent=True) or {})
    return Response('{"ok":true}', content_type="application/json")


@autoorders_bp.route("/autoorders/cancel_pending", methods=["POST"])
@login_required
def autoorders_cancel():
    body   = request.get_json(silent=True) or {}
    ticket = int(body.get("ticket", 0))
    if not ticket:
        return Response('{"ok":false,"message":"ticket lipsa"}',
                        content_type="application/json")
    if not MT5_AVAILABLE or mt5 is None:
        return Response('{"ok":false,"message":"MT5 indisponibil"}',
                        content_type="application/json")
    try:
        mt5.initialize()
        res = mt5.order_send({"action": mt5.TRADE_ACTION_REMOVE, "order": ticket})
        ok  = bool(res and res.retcode == mt5.TRADE_RETCODE_DONE)
        msg = "Anulat" if ok else f"retcode={res.retcode if res else -1}"
        return Response(json.dumps({"ok": ok, "message": msg}),
                        content_type="application/json")
    except Exception as exc:
        return Response(json.dumps({"ok": False, "message": str(exc)}),
                        content_type="application/json")


@autoorders_bp.route("/autoorders/suggest")
@login_required
def autoorders_suggest():
    """
    Sugereaza SL/TP/lot bazat pe ATR al TF-ului curent.
    SL = 1.5 * ATR  (spatiu sa respire pretul)
    TP = 3.0 * ATR  (RR 1:2)
    Lot = risk_$ / (sl_pips * pip_value_per_lot)
    """
    symbol     = request.args.get("symbol", _state["symbol"]).upper()
    tf         = request.args.get("tf", _state["tf"])
    direction  = request.args.get("dir", "BUY").upper()
    risk_usd   = float(request.args.get("risk", 100))

    try:
        df, _ = fetch(symbol, tf, 300)
        if df is None or df.empty:
            return Response('{"ok":false,"message":"Date indisponibile"}',
                            content_type="application/json")
        df.columns = [c.lower() for c in df.columns]
        from strategies.sr_mtf import _atr as _calc_atr
        atr_val = _calc_atr(df)
        if atr_val <= 0:
            return Response('{"ok":false,"message":"ATR zero"}',
                            content_type="application/json")

        price_now = float(df["close"].iloc[-1])

        # Incearca sa ia pretul live din MT5
        if MT5_AVAILABLE and mt5:
            try:
                mt5.initialize()
                tick = mt5.symbol_info_tick(symbol)
                if tick:
                    price_now = tick.ask if direction == "BUY" else tick.bid
            except Exception:
                pass

        sl_dist = round(atr_val * 1.5, 5)
        tp_dist = round(atr_val * 3.0, 5)

        if direction == "BUY":
            sl = round(price_now - sl_dist, 5)
            tp = round(price_now + tp_dist, 5)
        else:
            sl = round(price_now + sl_dist, 5)
            tp = round(price_now - tp_dist, 5)

        # Valoare pip — incearca MT5, fallback estimare din ATR/pips
        pip_value_per_lot = 10.0  # default EURUSD/majors
        pip_size = 0.0001         # default forex major
        try:
            if MT5_AVAILABLE and mt5:
                mt5.initialize()
                info = mt5.symbol_info(symbol)
                if info:
                    pip_size = info.point * 10 if info.digits in (4, 5) else info.point
                    # pip_value = contract_size * pip_size / price (pentru non-USD quote)
                    pip_value_per_lot = round(info.trade_contract_size * pip_size, 6)
        except Exception:
            pass

        sl_pips = abs(price_now - sl) / pip_size if pip_size > 0 else 0
        lot_size = 0.0
        if sl_pips > 0 and pip_value_per_lot > 0:
            lot_size = round(risk_usd / (sl_pips * pip_value_per_lot), 2)
            lot_size = max(0.01, lot_size)

        return Response(json.dumps({
            "ok":        True,
            "price":     round(price_now, 5),
            "sl":        sl,
            "tp":        tp,
            "atr":       round(atr_val, 5),
            "sl_pips":   round(sl_pips, 1),
            "tp_pips":   round(abs(tp - price_now) / pip_size, 1) if pip_size > 0 else 0,
            "lot":       lot_size,
            "rr":        "1:2",
            "risk_usd":  risk_usd,
        }, cls=NpEncoder), content_type="application/json")

    except Exception as exc:
        log.warning(f"[AutoOrders] suggest: {exc}")
        return Response(json.dumps({"ok": False, "message": str(exc)}),
                        content_type="application/json")


@autoorders_bp.route("/autoorders/orders")
@login_required
def autoorders_orders():
    """Returneaza toate ordinele din OrderManager, grupate pe status."""
    status_req = request.args.get("status", "all")
    symbol_req = request.args.get("symbol")
    strat_req  = request.args.get("strategy")

    if status_req == "active":
        orders = _order_manager.get_all(STATUS_ACTIVE, symbol_req, strat_req)
    elif status_req == "filled":
        orders = _order_manager.get_all({"filled"}, symbol_req, strat_req)
    elif status_req == "history":
        orders = _order_manager.get_all(STATUS_CLOSED, symbol_req, strat_req)
    else:
        orders = _order_manager.get_all(None, symbol_req, strat_req)

    return Response(
        json.dumps({"ok": True, "orders": orders, "stats": _order_manager.stats()},
                   cls=NpEncoder),
        content_type="application/json",
    )


@autoorders_bp.route("/autoorders/orders/cancel", methods=["POST"])
@login_required
def autoorders_orders_cancel():
    body      = request.get_json(silent=True) or {}
    order_id  = str(body.get("order_id", ""))
    reason    = str(body.get("reason", "manual"))

    o = _order_manager.get_by_id(order_id)
    if not o:
        return Response('{"ok":false,"message":"Ordin negasit"}',
                        content_type="application/json")

    # Anuleaza si in MT5 daca e pending
    if o.get("mode") == "mt5_pending" and o.get("mt5_ticket"):
        _cancel_mt5_order(o["mt5_ticket"])

    ok = _order_manager.cancel(order_id, reason=reason)
    return Response(json.dumps({"ok": ok}), content_type="application/json")


@autoorders_bp.route("/autoorders/orders/modify", methods=["POST"])
@login_required
def autoorders_orders_modify():
    body     = request.get_json(silent=True) or {}
    order_id = str(body.get("order_id", ""))
    sl       = float(body["sl"]) if "sl" in body else None
    tp       = float(body["tp"]) if "tp" in body else None

    o = _order_manager.get_by_id(order_id)
    if not o:
        return Response('{"ok":false,"message":"Ordin negasit"}',
                        content_type="application/json")

    # Modifica si in MT5
    if o.get("mode") == "mt5_pending" and o.get("mt5_ticket") and MT5_AVAILABLE and mt5:
        try:
            mt5.initialize()
            req = {
                "action":   mt5.TRADE_ACTION_MODIFY,
                "order":    o["mt5_ticket"],
                "sl":       sl if sl is not None else o["sl"],
                "tp":       tp if tp is not None else o["tp"],
            }
            mt5.order_send(req)
        except Exception as exc:
            log.warning(f"[AutoOrders] modify MT5: {exc}")

    ok = _order_manager.modify(order_id, sl=sl, tp=tp)
    return Response(json.dumps({"ok": ok}), content_type="application/json")


@autoorders_bp.route("/autoorders/orders/sync", methods=["POST"])
@login_required
def autoorders_orders_sync():
    if not MT5_AVAILABLE or mt5 is None:
        return Response('{"ok":false,"message":"MT5 indisponibil"}',
                        content_type="application/json")
    changed = _order_manager.sync_mt5(mt5)
    return Response(json.dumps({"ok": True, "changed": len(changed), "ids": changed}),
                    content_type="application/json")


@autoorders_bp.route("/autoorders/tick")
@login_required
def autoorders_tick():
    """Returneaza bid/ask curent pentru un simbol."""
    symbol = request.args.get("symbol", _state["symbol"]).upper()
    if not MT5_AVAILABLE or mt5 is None:
        return Response('{"ok":false}', content_type="application/json")
    try:
        mt5.initialize()
        tick = mt5.symbol_info_tick(symbol)
        if tick:
            return Response(
                json.dumps({"ok": True, "bid": tick.bid, "ask": tick.ask, "symbol": symbol}),
                content_type="application/json",
            )
    except Exception as exc:
        log.warning(f"[AutoOrders] tick: {exc}")
    return Response('{"ok":false}', content_type="application/json")


@autoorders_bp.route("/autoorders/place_order", methods=["POST"])
@login_required
def autoorders_place_order():
    """Plaseaza un ordin market sau pending direct din UI."""
    body       = request.get_json(silent=True) or {}
    symbol     = str(body.get("symbol", "")).upper()
    signal     = str(body.get("signal", "")).upper()
    order_type = str(body.get("order_type", "market")).lower()
    sl         = float(body.get("sl",    0) or 0)
    tp         = float(body.get("tp",    0) or 0)
    entry      = float(body.get("entry", 0) or 0)
    risk_dollars = float(body.get("risk_dollars", 50) or 50)

    if not symbol or signal not in ("BUY", "SELL"):
        return Response(
            json.dumps({"ok": False, "message": f"Parametri invalizi: symbol={symbol!r} signal={signal!r}"}),
            content_type="application/json",
        )
    if sl <= 0 or tp <= 0:
        return Response(
            json.dumps({"ok": False, "message": "SL si TP sunt obligatorii"}),
            content_type="application/json",
        )

    if order_type == "market":
        ok, msg = place_trade(symbol, signal, sl, tp, risk_dollars, strategy="manual")
    else:
        if entry <= 0:
            return Response(
                json.dumps({"ok": False, "message": "Entry price obligatoriu pentru ordine pending"}),
                content_type="application/json",
            )
        ok, msg = place_pending_order(
            symbol, signal, entry, sl, tp,
            risk_dollars=risk_dollars, strategy="manual", expiry_hours=24,
        )

    # Logam decizia in decisions
    decision = {
        "ts":       datetime.now().strftime("%H:%M:%S"),
        "signal":   f"MANUAL {'MKT' if order_type=='market' else 'PENDING'} {signal}",
        "executed": ok,
        "result":   msg,
        "strategy": "manual",
    }
    with _lock:
        _state["decisions"].insert(0, decision)
        while len(_state["decisions"]) > 80:
            _state["decisions"].pop()

    return Response(json.dumps({"ok": ok, "message": msg}), content_type="application/json")


def _apply(body: dict):
    if "strategy" in body:
        k = str(body["strategy"])
        if k in _STRATEGIES:
            _state["strategy"] = k
    if "strategies_active" in body:
        sa = [k for k in body["strategies_active"] if k in _STRATEGIES]
        if sa:
            _state["strategies_active"] = sa
            _state["strategy"] = sa[0]
    if "symbols_watch" in body:
        sw = [s.strip().upper() for s in body["symbols_watch"] if s.strip()]
        if sw:
            _state["symbols_watch"] = sw
            _state["symbol"] = sw[0]
    if "symbol"         in body: _state["symbol"]         = str(body["symbol"]).upper()
    if "tf"             in body: _state["tf"]             = str(body["tf"])
    if "bars"           in body: _state["bars"]           = max(100, min(5000, int(body["bars"])))
    if "interval"       in body: _state["interval"]       = max(10, int(body["interval"]))
    if "auto_execute"   in body: _state["auto_execute"]   = bool(body["auto_execute"])
    if "exec_mode"      in body and body["exec_mode"] in ("mt5_pending", "virtual"):
        _state["exec_mode"] = str(body["exec_mode"])
    if "expiry_hours"   in body: _state["expiry_hours"]   = max(1, int(body["expiry_hours"]))
    if "min_confidence" in body: _state["min_confidence"] = max(0.0, min(100.0, float(body["min_confidence"])))
    if "risk_usd"       in body: _state["risk_usd"]       = max(1.0, float(body["risk_usd"]))
    if "replace_thr"    in body: _state["replace_thr"]    = max(0.1, min(2.0, float(body["replace_thr"])))


# ── HTML ──────────────────────────────────────────────────────────────────────
def _render(symbols: list, tfs: list, strat_opts: dict, state: dict) -> str:
    cur_sym       = state.get("symbol", "EURUSD")
    cur_tf        = state.get("tf", "H1")
    cur_str       = state.get("strategy", next(iter(strat_opts), "eob"))
    cur_bars      = state.get("bars", 1000)
    cur_int       = state.get("interval", 60)
    cur_conf      = state.get("min_confidence", 60.0)
    cur_risk      = state.get("risk_usd", 100.0)
    cur_mode      = state.get("exec_mode", "mt5_pending")
    cur_strats    = set(state.get("strategies_active", [cur_str]))
    cur_sw        = ",".join(state.get("symbols_watch", [cur_sym]))
    cur_repl      = state.get("replace_thr", 0.3)

    # Multi-strategy checkboxes
    strat_checks = "".join(
        f'<label style="display:flex;align-items:center;gap:6px;cursor:pointer;font-size:.77rem;color:#8090a0">'
        f'<input type="checkbox" id="sc-{k}" value="{k}" '
        f'{"checked" if k in cur_strats else ""} onchange="syncStratChecks()">'
        f'{v}</label>'
        for k, v in strat_opts.items()
    )

    sym_opts   = "".join(
        f'<option value="{s}"{"selected" if s == cur_sym else ""}>{s}</option>'
        for s in symbols)
    tf_opts    = "".join(
        f'<option value="{t}"{"selected" if t == cur_tf else ""}>{t}</option>'
        for t in tfs)
    strat_html = "".join(
        f'<option value="{k}"{"selected" if k == cur_str else ""}>{v}</option>'
        for k, v in strat_opts.items())

    return f"""<!DOCTYPE html>
<html lang="ro">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>AutoOrders</title>
<script src="https://cdn.plot.ly/plotly-2.35.2.min.js" charset="utf-8"></script>
<style>
*{{box-sizing:border-box;margin:0;padding:0}}
body{{background:#07070e;color:#a0aabb;font-family:'Inter',sans-serif;font-size:.8rem;display:flex;flex-direction:column;height:100vh;overflow:hidden}}
.hdr{{display:flex;align-items:center;gap:12px;padding:8px 16px;background:#0c0c1a;border-bottom:1px solid #1c1c30;flex-shrink:0}}
.hdr-title{{font-size:.98rem;font-weight:700;color:#c0c8e0;letter-spacing:.6px}}
.hdr a{{color:#5c6bc0;text-decoration:none;font-size:.76rem}}
.hdr a:hover{{color:#7986cb}}
.status-dot{{width:8px;height:8px;border-radius:50%;background:#222;transition:background .4s}}
.status-dot.on{{background:#26a69a;box-shadow:0 0 6px #26a69a80}}
.tabs{{display:flex;gap:2px;padding:6px 14px 0;background:#0c0c1a;border-bottom:1px solid #1c1c30;flex-shrink:0}}
.tab{{padding:5px 14px;border-radius:6px 6px 0 0;cursor:pointer;color:#6070a0;border:1px solid transparent;border-bottom:none;font-size:.79rem;transition:color .2s}}
.tab.active{{background:#0f0f20;color:#c0c8e0;border-color:#1c1c30}}
.tab:hover:not(.active){{color:#9090c0}}
.main{{display:flex;flex:1;overflow:hidden}}
.sidebar{{width:228px;flex-shrink:0;background:#0c0c1a;border-right:1px solid #1a1a2e;padding:12px 10px;overflow-y:auto;display:flex;flex-direction:column;gap:9px}}
.sidebar label{{display:block;color:#5a6880;font-size:.7rem;margin-bottom:2px}}
.sidebar select,.sidebar input[type=number]{{width:100%;background:#10101e;border:1px solid #1e1e35;color:#c0c8e0;padding:4px 8px;border-radius:5px;font-size:.79rem}}
.sidebar select:focus,.sidebar input:focus{{outline:none;border-color:#3949ab}}
.toggle-row{{display:flex;justify-content:space-between;align-items:center}}
.toggle{{position:relative;width:34px;height:18px;display:inline-block}}
.toggle input{{opacity:0;width:0;height:0}}
.slider{{position:absolute;inset:0;background:#1a1a2e;border-radius:18px;cursor:pointer;transition:.2s}}
.slider::before{{position:absolute;content:"";height:12px;width:12px;left:3px;bottom:3px;background:#444;border-radius:50%;transition:.2s}}
input:checked+.slider{{background:#283593}}
input:checked+.slider::before{{transform:translateX(16px);background:#7986cb}}
.btn{{padding:5px 12px;border-radius:6px;border:none;cursor:pointer;font-size:.78rem;font-weight:500;transition:background .15s}}
.btn-start{{background:#1a237e;color:#c5cae9}}.btn-start:hover{{background:#283593}}
.btn-stop{{background:#1a1a2e;color:#ef9a9a;border:1px solid #3a2222}}.btn-stop:hover{{background:#2a1a1a}}
.btn-scan{{background:#0d2020;color:#4db6ac;border:1px solid #1a3535}}.btn-scan:hover{{background:#122828}}
.btn-sm{{padding:3px 8px;border-radius:4px;border:1px solid #2a2a3e;background:#10101e;color:#8090a0;cursor:pointer;font-size:.72rem}}.btn-sm:hover{{color:#ef5350;border-color:#ef5350}}
.content{{flex:1;display:flex;flex-direction:column;overflow:hidden}}
.tab-panel{{display:none;flex:1;flex-direction:column;overflow:hidden}}
.tab-panel.active{{display:flex}}
.analysis-card{{background:#0f0f1e;border:1px solid #1a1a2e;border-radius:6px;padding:8px 10px;display:flex;flex-direction:column;gap:5px}}
.alabel{{color:#4a5870;font-size:.68rem;margin-bottom:1px}}
.signal-badge{{display:inline-block;padding:2px 10px;border-radius:4px;font-weight:600;font-size:.76rem}}
.signal-badge.buy{{background:#0d2a24;color:#26a69a}}
.signal-badge.sell{{background:#2a1010;color:#ef5350}}
.signal-badge.hold{{background:#141420;color:#5c6070}}
.signal-badge.pending{{background:#1a1030;color:#ab47bc}}
.dec-log{{overflow-y:auto;max-height:130px;display:flex;flex-direction:column;gap:1px}}
.dec-row{{display:flex;gap:6px;padding:2px 0;border-bottom:1px solid #10101e;align-items:baseline}}
.dec-row .ts{{color:#2a3050;font-size:.66rem;white-space:nowrap}}
.dec-row .dsig{{font-weight:600;font-size:.71rem;min-width:70px}}
.dec-row .dres{{color:#4a5060;font-size:.7rem;flex:1;overflow:hidden;white-space:nowrap;text-overflow:ellipsis}}
.pending-wrap{{flex:1;overflow:auto;padding:8px 12px}}
.pending-tbl{{width:100%;border-collapse:collapse;font-size:.76rem}}
.pending-tbl th{{padding:5px 10px;color:#4a5070;text-align:left;border-bottom:1px solid #1a1a2e;font-weight:500;white-space:nowrap;font-size:.7rem}}
.pending-tbl td{{padding:5px 10px;border-bottom:1px solid #0f0f1e;vertical-align:middle}}
.pending-tbl tr:hover td{{background:#0c0c18}}
.exp-bar{{height:3px;border-radius:2px;background:#1a1a2e;margin-top:3px;overflow:hidden;width:90px}}
.exp-fill{{height:100%;border-radius:2px;transition:width .6s}}
.sec{{font-size:.68rem;font-weight:700;color:#3a4860;text-transform:uppercase;letter-spacing:.9px;margin-bottom:4px}}
.sep{{height:1px;background:#13131f;margin:2px 0}}
::-webkit-scrollbar{{width:4px;height:4px}}
::-webkit-scrollbar-track{{background:#080810}}
::-webkit-scrollbar-thumb{{background:#1c1c30;border-radius:2px}}
.legend-dot{{display:inline-block;width:10px;height:10px;border-radius:1px;vertical-align:middle;margin-right:3px}}
.btn-tf{{padding:3px 7px;border-radius:4px;border:1px solid #1e1e35;background:#0f0f1e;color:#5a6880;cursor:pointer;font-size:.74rem;font-weight:600;transition:all .15s}}
.btn-tf:hover{{border-color:#3949ab;color:#9090c0}}
.btn-tf.active{{background:#1a1a30;border-color:#5c6bc0;color:#c0c8e0}}
.exec-btn{{padding:3px 10px;border-radius:4px;border:1px solid #1e1e35;background:#0f0f1e;color:#5a6880;cursor:pointer;font-size:.72rem;font-weight:600;transition:all .15s}}
.exec-btn.active-mt5{{border-color:#5c6bc0;color:#7986cb;background:#0f0f20}}
.exec-btn.active-virt{{border-color:#ab47bc;color:#ce93d8;background:#160f1e}}
.orders-tbl{{width:100%;border-collapse:collapse;font-size:.74rem}}
.orders-tbl th{{padding:4px 8px;color:#3a4860;text-align:left;border-bottom:1px solid #1a1a2e;font-weight:500;white-space:nowrap;font-size:.68rem}}
.orders-tbl td{{padding:4px 8px;border-bottom:1px solid #0c0c18;vertical-align:middle;white-space:nowrap}}
.orders-tbl tr:hover td{{background:#0c0c18}}
.subtab{{padding:4px 12px;border-radius:5px 5px 0 0;cursor:pointer;color:#4a5870;font-size:.76rem;border:1px solid transparent;border-bottom:none;transition:color .15s}}
.subtab.active{{background:#0f0f1e;color:#c0c8e0;border-color:#1c1c30}}
.stat-chip{{display:inline-block;padding:1px 7px;border-radius:10px;font-size:.68rem;font-weight:600;margin-left:4px}}
</style>
</head>
<body>

<!-- HEADER -->
<div class="hdr">
  <span class="hdr-title">⬡ AutoOrders</span>
  <a href="/autotrader">← AutoTrader</a>
  <div style="flex:1"></div>
  <span id="status-text" style="color:#3a4860;font-size:.74rem">Inactiv</span>
  <span class="status-dot" id="status-dot"></span>
  <span id="last-scan-ts" style="color:#252535;font-size:.68rem;margin-left:10px"></span>
</div>

<!-- TABS -->
<div class="tabs">
  <div class="tab active" onclick="switchTab('analysis')">📊 Chart</div>
  <div class="tab" onclick="switchTab('orders')">📋 MT5 Pending</div>
  <div class="tab" onclick="switchTab('ao-orders')">🗂 Orders
    <span class="stat-chip" id="chip-active" style="background:#1a1a30;color:#5c6bc0">0</span>
  </div>
</div>

<div class="main">

  <!-- SIDEBAR -->
  <div class="sidebar">

    <div class="sec">Strategie</div>
    <div>
      <label>Strategie activa</label>
      <select id="cfg-strat" onchange="applyConfig()">{strat_html}</select>
    </div>

    <div class="sep"></div>
    <div class="sec">Configurare</div>

    <div>
      <label>Simbol</label>
      <select id="cfg-symbol" onchange="applyConfig()">{sym_opts}</select>
    </div>
    <div>
      <label>Timeframe</label>
      <select id="cfg-tf" onchange="applyConfig();updateTFBtns()">{tf_opts}</select>
    </div>
    <div style="display:flex;flex-wrap:wrap;gap:3px;margin-top:2px">
      <button class="btn-tf" id="tf-M1"  onclick="setTF('M1')">1M</button>
      <button class="btn-tf" id="tf-M2"  onclick="setTF('M2')">2M</button>
      <button class="btn-tf" id="tf-M5"  onclick="setTF('M5')">5M</button>
      <button class="btn-tf" id="tf-M15" onclick="setTF('M15')">15M</button>
      <button class="btn-tf" id="tf-H1"  onclick="setTF('H1')">1H</button>
      <button class="btn-tf" id="tf-H4"  onclick="setTF('H4')">4H</button>
      <button class="btn-tf" id="tf-D1"  onclick="setTF('D1')">1D</button>
    </div>
    <div>
      <label>Bare</label>
      <input id="cfg-bars" type="number" value="{cur_bars}" min="100" max="5000" onchange="applyConfig()">
    </div>
    <div>
      <label>Interval scan (sec)</label>
      <input id="cfg-interval" type="number" value="{cur_int}" min="10" max="3600" onchange="applyConfig()">
    </div>
    <div>
      <label>Min Confidence %</label>
      <input id="cfg-conf" type="number" value="{cur_conf}" min="0" max="100" step="5" onchange="applyConfig()">
    </div>

    <div class="sep"></div>
    <div class="sec">AutoOrders v2</div>
    <div>
      <div class="alabel" style="color:#4a5870;font-size:.7rem;margin-bottom:3px">Strategii active</div>
      <div style="display:flex;flex-direction:column;gap:5px">{strat_checks}</div>
    </div>
    <div>
      <div class="alabel" style="color:#4a5870;font-size:.7rem;margin-bottom:3px">Mod executie</div>
      <div style="display:flex;gap:5px">
        <button class="exec-btn {'active-mt5' if cur_mode=='mt5_pending' else ''}" id="exec-mt5" onclick="setExecMode('mt5_pending')">MT5 Pending</button>
        <button class="exec-btn {'active-virt' if cur_mode=='virtual' else ''}" id="exec-virt" onclick="setExecMode('virtual')">Virtual</button>
      </div>
    </div>
    <div>
      <label>Simboluri watch (CSV)</label>
      <input id="cfg-sw" type="text" value="{cur_sw}" placeholder="EURUSD,GBPUSD"
             style="width:100%;background:#10101e;border:1px solid #1e1e35;color:#c0c8e0;padding:4px 8px;border-radius:5px;font-size:.79rem"
             onchange="applySW()">
    </div>
    <div>
      <label>Zone replace (ATR×)</label>
      <input id="cfg-repl" type="number" step="0.05" value="{cur_repl}" min="0.1" max="2"
             style="width:100%;background:#10101e;border:1px solid #1e1e35;color:#c0c8e0;padding:4px 8px;border-radius:5px;font-size:.79rem"
             onchange="applyRepl()">
    </div>

    <div class="sep"></div>

    <div class="toggle-row">
      <span style="color:#8090a0">Auto Execute</span>
      <label class="toggle">
        <input type="checkbox" id="cfg-auto" onchange="applyConfig()">
        <span class="slider"></span>
      </label>
    </div>

    <div style="display:flex;gap:6px;margin-top:2px">
      <button class="btn btn-start" onclick="startScanner()">▶ Start</button>
      <button class="btn btn-stop"  onclick="stopScanner()">■ Stop</button>
      <button class="btn btn-scan"  onclick="scanNow()" title="Scan acum">⟳</button>
    </div>

    <div class="sep"></div>
    <div class="sec">Ultima analiza</div>
    <div class="analysis-card">
      <div>
        <div class="alabel">Semnal</div>
        <span class="signal-badge hold" id="an-signal">—</span>
      </div>
      <div style="display:flex;gap:14px">
        <div>
          <div class="alabel">Confidence</div>
          <span id="an-conf" style="color:#c0c8e0;font-weight:600">—</span>
        </div>
        <div>
          <div class="alabel">Risk Score</div>
          <span id="an-risk" style="color:#9c27b0;font-weight:600">—</span>
        </div>
      </div>
      <div>
        <div class="alabel">Pending Entry</div>
        <div id="an-pending" style="color:#ab47bc;font-size:.71rem;line-height:1.4">—</div>
      </div>
      <div>
        <div class="alabel">Justificare</div>
        <div id="an-just" style="color:#505a70;font-size:.7rem;line-height:1.45">—</div>
      </div>
    </div>

    <div class="sep"></div>
    <div class="sec">Decizii recente</div>
    <div class="dec-log" id="dec-log"></div>

    <div class="sep"></div>
    <!-- ── ORDER MANUAL ──────────────────────────────────────────────── -->
    <div class="sec" style="display:flex;align-items:center;justify-content:space-between">
      <span>Order Manual</span>
      <span id="tick-price" style="color:#4a5870;font-size:.7rem">—</span>
    </div>

    <!-- BUY / SELL -->
    <div style="display:flex;gap:4px">
      <button class="btn-dir" id="btn-buy"  onclick="setDir('BUY')"
              style="flex:1;padding:6px;border-radius:5px;border:2px solid #00c853;background:#0a2a14;color:#00e676;font-weight:700;cursor:pointer;transition:.15s;font-size:.82rem">
        ▲ BUY
      </button>
      <button class="btn-dir" id="btn-sell" onclick="setDir('SELL')"
              style="flex:1;padding:6px;border-radius:5px;border:2px solid #333;background:#1e0a0a;color:#ef5350;font-weight:700;cursor:pointer;transition:.15s;font-size:.82rem;opacity:.45">
        ▼ SELL
      </button>
    </div>

    <!-- Tip ordin -->
    <div>
      <label>Tip ordin</label>
      <select id="ord-type" onchange="toggleEntryField()" style="width:100%;background:#10101e;border:1px solid #1e1e35;color:#c0c8e0;padding:4px 8px;border-radius:5px;font-size:.79rem">
        <option value="market">⚡ Market (instant)</option>
        <option value="pending" selected>📌 Pending (Limit/Stop)</option>
      </select>
    </div>

    <!-- Click pe chart → seteaza pret -->
    <div style="display:flex;gap:4px;align-items:center">
      <span style="color:#4a5870;font-size:.7rem;flex:1">Click chart →</span>
      <select id="click-target" style="background:#0f0f1e;border:1px solid #1a1a2e;color:#8090a0;padding:2px 6px;border-radius:4px;font-size:.72rem">
        <option value="entry">Entry</option>
        <option value="sl">SL</option>
        <option value="tp">TP</option>
      </select>
    </div>

    <!-- Entry price (doar pending) -->
    <div id="entry-row">
      <label>Entry price <span style="color:#3a4860;font-size:.66rem">(click chart)</span></label>
      <input id="ord-entry" type="number" step="0.00001" placeholder="0.00000"
             style="width:100%;background:#10101e;border:1px solid #1e1e35;color:#ab47bc;padding:4px 8px;border-radius:5px;font-size:.79rem">
    </div>

    <!-- SL -->
    <div>
      <label>Stop Loss <span style="color:#ef5350;font-size:.66rem">●</span></label>
      <input id="ord-sl" type="number" step="0.00001" placeholder="0.00000"
             style="width:100%;background:#10101e;border:1px solid #1e1e35;color:#ef9a9a;padding:4px 8px;border-radius:5px;font-size:.79rem">
    </div>

    <!-- TP -->
    <div>
      <label>Take Profit <span style="color:#26a69a;font-size:.66rem">●</span></label>
      <input id="ord-tp" type="number" step="0.00001" placeholder="0.00000"
             style="width:100%;background:#10101e;border:1px solid #1e1e35;color:#80cbc4;padding:4px 8px;border-radius:5px;font-size:.79rem">
    </div>

    <!-- Risk -->
    <div>
      <label>Risk ($)</label>
      <input id="ord-risk" type="number" step="1" value="100" min="1"
             style="width:100%;background:#10101e;border:1px solid #1e1e35;color:#c0c8e0;padding:4px 8px;border-radius:5px;font-size:.79rem"
             onchange="autoSuggest()">
    </div>

    <!-- Sugestie lot calculat -->
    <div id="lot-suggest" style="display:none;background:#0a0a1a;border:1px solid #1a1a30;border-radius:5px;padding:5px 8px;font-size:.71rem;line-height:1.6">
      <div style="display:flex;justify-content:space-between">
        <span style="color:#4a5870">Lot sugerat:</span>
        <span id="sug-lot" style="color:#c0c8e0;font-weight:700;font-family:monospace">—</span>
      </div>
      <div style="display:flex;justify-content:space-between">
        <span style="color:#4a5870">SL / TP (pips):</span>
        <span id="sug-pips" style="color:#8090a0;font-family:monospace">—</span>
      </div>
      <div style="display:flex;justify-content:space-between">
        <span style="color:#4a5870">ATR curent:</span>
        <span id="sug-atr" style="color:#5c6bc0;font-family:monospace">—</span>
      </div>
      <div style="display:flex;justify-content:space-between">
        <span style="color:#4a5870">RR:</span>
        <span style="color:#ab47bc;font-family:monospace">1 : 2</span>
      </div>
    </div>

    <!-- Place button -->
    <button id="btn-place" onclick="placeOrder()"
            style="width:100%;padding:7px;border-radius:6px;border:2px solid #3949ab;background:#1a237e;color:#c5cae9;font-weight:700;font-size:.82rem;cursor:pointer;transition:all .15s;margin-top:2px">
      ▶ Plaseaza Order
    </button>
    <div style="text-align:center;color:#2a3a5a;font-size:.67rem;margin-top:1px">se cere confirmare inainte de executie</div>
    <div id="order-result" style="font-size:.72rem;min-height:18px;text-align:center;margin-top:2px"></div>
    <!-- ──────────────────────────────────────────────────────────────── -->

  </div><!-- /sidebar -->

  <!-- CONTENT -->
  <div class="content">

    <!-- TAB: Analiza -->
    <div class="tab-panel active" id="tab-analysis">
      <div id="chart-div" style="flex:1;min-height:0"></div>
      <!-- Butoane rapide deschidere pozitie -->
      <div style="display:flex;gap:8px;padding:8px 12px;background:#0a0a16;border-top:1px solid #1a1a2e;flex-shrink:0">
        <button id="btn-sell-zone" onclick="quickOrder('SELL')" disabled
                style="flex:1;padding:8px 0;border-radius:6px;border:1px solid #3a1a1a;
                       background:#1a0808;color:#5a3030;font-weight:700;font-size:.82rem;
                       cursor:not-allowed;transition:all .15s;letter-spacing:.3px">
          ▼ SELL ZONE
        </button>
        <button id="btn-buy-zone" onclick="quickOrder('BUY')" disabled
                style="flex:1;padding:8px 0;border-radius:6px;border:1px solid #1a3a1a;
                       background:#081a08;color:#306030;font-weight:700;font-size:.82rem;
                       cursor:not-allowed;transition:all .15s;letter-spacing:.3px">
          ▲ BUY ZONE
        </button>
        <div style="color:#2a3040;font-size:.68rem;align-self:center;white-space:nowrap">← pre-fill</div>
        <div style="flex:1"></div>
        <button class="btn-sm" onclick="refreshChart()" style="align-self:center">⟳</button>
      </div>
    </div>

    <!-- TAB: Ordine Pending -->
    <div class="tab-panel" id="tab-orders">
      <div id="chart-div2" style="height:260px;flex-shrink:0"></div>
      <div style="background:#0c0c1a;border-top:1px solid #1a1a2e;padding:5px 12px;display:flex;align-items:center;gap:10px;flex-shrink:0">
        <span style="color:#3a4860;font-size:.7rem;font-weight:700;text-transform:uppercase;letter-spacing:.6px">Ordine Pending Active</span>
        <div style="flex:1"></div>
        <button class="btn-sm" onclick="refreshAll()">⟳ Refresh</button>
      </div>
      <div class="pending-wrap">
        <table class="pending-tbl">
          <thead>
            <tr>
              <th>Ticket</th><th>Simbol</th><th>Tip</th>
              <th style="text-align:right">Entry</th>
              <th style="text-align:right">SL</th>
              <th style="text-align:right">TP</th>
              <th style="text-align:right">Loturi</th>
              <th>Data expirare</th>
              <th>Timp ramas</th>
              <th></th>
            </tr>
          </thead>
          <tbody id="pending-tbody">
            <tr><td colspan="10" style="color:#2a3040;padding:16px;text-align:center">Se incarca...</td></tr>
          </tbody>
        </table>
      </div>
    </div>

    <!-- TAB: AO Orders (OrderManager) -->
    <div class="tab-panel" id="tab-ao-orders">
      <div style="display:flex;align-items:center;gap:4px;padding:5px 12px;background:#0c0c1a;border-bottom:1px solid #1a1a2e;flex-shrink:0">
        <div class="subtab active" id="subtab-active"  onclick="switchOrdersTab('active')">Active</div>
        <div class="subtab"        id="subtab-filled"  onclick="switchOrdersTab('filled')">Filled</div>
        <div class="subtab"        id="subtab-history" onclick="switchOrdersTab('history')">History</div>
        <div style="flex:1"></div>
        <button class="btn-sm" onclick="syncOrders()">⟳ Sync MT5</button>
        <button class="btn-sm" onclick="_loadOrders(_curOrderTab)" style="margin-left:4px">↻ Refresh</button>
        <span id="orders-count" style="color:#2a3550;font-size:.68rem;margin-left:10px"></span>
      </div>
      <div style="flex:1;overflow:auto;padding:8px 12px">
        <table class="orders-tbl">
          <thead>
            <tr>
              <th>ID / Ticket</th><th>Simbol</th><th>Strategie</th>
              <th>Dir</th><th>Mod</th>
              <th style="text-align:right">Entry</th>
              <th style="text-align:right">SL</th>
              <th style="text-align:right">TP</th>
              <th style="text-align:right">RR</th>
              <th>Status</th><th>Creat</th><th></th>
            </tr>
          </thead>
          <tbody id="orders-tbody">
            <tr><td colspan="12" style="color:#2a3040;padding:18px;text-align:center">Se incarca...</td></tr>
          </tbody>
        </table>
      </div>
    </div>

  </div><!-- /content -->
</div><!-- /main -->

<script>
let _c1Drawn = false, _c2Drawn = false;
let _activeTab = 'analysis';
const SIG_COL = {{BUY:'#26a69a',SELL:'#ef5350',HOLD:'#3a4050'}};
let _zoneData = {{buy: null, sell: null}};

function switchTab(tab) {{
  _activeTab = tab;
  const idx = tab==='analysis'?0:tab==='orders'?1:2;
  document.querySelectorAll('.tab').forEach((t,i) => t.classList.toggle('active', i===idx));
  document.querySelectorAll('.tab-panel').forEach(p=>p.classList.remove('active'));
  document.getElementById('tab-'+tab).classList.add('active');
  if (tab==='orders')    {{ refreshChart2(); refreshAll(); }}
  if (tab==='ao-orders') {{ _loadOrders(_curOrderTab); }}
}}

function setTF(tf) {{
  document.getElementById('cfg-tf').value = tf;
  updateTFBtns();
  applyConfig();
}}

function updateTFBtns() {{
  const cur = document.getElementById('cfg-tf').value;
  ['M1','M2','M5','M15','H1','H4','D1'].forEach(t => {{
    const el = document.getElementById('tf-'+t);
    if (el) el.classList.toggle('active', t === cur);
  }});
}}

function getCfg() {{
  return {{
    strategy:       document.getElementById('cfg-strat').value,
    symbol:         document.getElementById('cfg-symbol').value,
    tf:             document.getElementById('cfg-tf').value,
    bars:           parseInt(document.getElementById('cfg-bars').value)||500,
    interval:       parseInt(document.getElementById('cfg-interval').value)||60,
    min_confidence: parseFloat(document.getElementById('cfg-conf').value)||60,
    auto_execute:   document.getElementById('cfg-auto').checked,
  }};
}}

function applyConfig() {{
  fetch('/autoorders/set',{{method:'POST',headers:{{'Content-Type':'application/json'}},body:JSON.stringify(getCfg())}});
  refreshChart();
}}

function startScanner() {{
  fetch('/autoorders/start',{{method:'POST',headers:{{'Content-Type':'application/json'}},body:JSON.stringify(getCfg())}});
}}
function stopScanner() {{
  fetch('/autoorders/stop',{{method:'POST'}});
}}
function scanNow() {{
  // Forteaza un scan imediat prin start cu interval scurt
  const cfg = getCfg();
  cfg.interval = 10;
  fetch('/autoorders/start',{{method:'POST',headers:{{'Content-Type':'application/json'}},body:JSON.stringify(cfg)}});
}}

async function _fetchChart(height) {{
  const sym   = document.getElementById('cfg-symbol').value;
  const tf    = document.getElementById('cfg-tf').value;
  const bars  = document.getElementById('cfg-bars').value||500;
  const strat = document.getElementById('cfg-strat').value;
  const r = await fetch(`/autoorders/chart?symbol=${{sym}}&tf=${{tf}}&bars=${{bars}}&strategy=${{strat}}`);
  const fig = await r.json();
  if (!fig||!fig.data) return null;
  if (height && fig.layout) fig.layout.height = height;
  return fig;
}}

const _pltConfig = {{
  responsive: true,
  scrollZoom: true,
  doubleClick: 'reset',
  displayModeBar: true,
  modeBarButtonsToRemove: ['select2d','lasso2d','autoScale2d','hoverClosestCartesian','hoverCompareCartesian','toggleSpikelines'],
  displaylogo: false,
}};

function _updateZoneBtns() {{
  const bBtn = document.getElementById('btn-buy-zone');
  const sBtn = document.getElementById('btn-sell-zone');
  const z    = _zoneData;

  if (bBtn) {{
    const has = !!z.buy;
    bBtn.disabled = !has;
    bBtn.textContent = has
      ? `▲ BUY @ ${{z.buy.entry.toFixed(5)}}  SL ${{z.buy.sl.toFixed(5)}}`
      : '▲ BUY ZONE';
    bBtn.style.background   = has ? '#0d2a24' : '#081a08';
    bBtn.style.borderColor  = has ? '#26a69a' : '#1a3a1a';
    bBtn.style.color        = has ? '#26a69a' : '#306030';
    bBtn.style.cursor       = has ? 'pointer'  : 'not-allowed';
  }}
  if (sBtn) {{
    const has = !!z.sell;
    sBtn.disabled = !has;
    sBtn.textContent = has
      ? `▼ SELL @ ${{z.sell.entry.toFixed(5)}}  SL ${{z.sell.sl.toFixed(5)}}`
      : '▼ SELL ZONE';
    sBtn.style.background   = has ? '#2a1010' : '#1a0808';
    sBtn.style.borderColor  = has ? '#ef5350' : '#3a1a1a';
    sBtn.style.color        = has ? '#ef5350' : '#5a3030';
    sBtn.style.cursor       = has ? 'pointer'  : 'not-allowed';
  }}
}}

function quickOrder(dir) {{
  const z = dir === 'BUY' ? _zoneData.buy : _zoneData.sell;
  if (!z) return;

  // Seteaza directia
  setDir(dir);

  // Seteaza tip ordin → Pending (limit la pretul zonei)
  document.getElementById('ord-type').value = 'pending';
  toggleEntryField();

  // Completeaza Entry / SL / TP din zona detectata
  const entryEl = document.getElementById('ord-entry');
  const slEl    = document.getElementById('ord-sl');
  const tpEl    = document.getElementById('ord-tp');
  if (entryEl) entryEl.value = z.entry.toFixed(5);
  if (slEl)    slEl.value    = z.sl.toFixed(5);
  if (tpEl)    tpEl.value    = z.tp.toFixed(5);

  // Flash vizual pe campuri
  [entryEl, slEl, tpEl].forEach(el => {{
    if (!el) return;
    el.style.boxShadow = dir==='BUY' ? '0 0 0 2px #00c853' : '0 0 0 2px #f44336';
    setTimeout(() => el.style.boxShadow = '', 900);
  }});

  // Afiseaza mesaj in result
  const res = document.getElementById('order-result');
  res.style.color   = dir === 'BUY' ? '#00e676' : '#ff5252';
  res.textContent   = `Zona ${{dir}} pre-completata — verifica si apasa Plaseaza Order`;
  setTimeout(() => {{ if (res.textContent.includes('pre-completata')) res.textContent=''; }}, 4000);

  // Scroll sidebar la butonul de plasare
  const btn = document.getElementById('btn-place');
  if (btn) btn.scrollIntoView({{behavior:'smooth', block:'nearest'}});
}}

async function refreshChart() {{
  const fig = await _fetchChart(null);
  if (!fig) return;
  // Extrage zonele detectate si ATR
  _zoneData.buy  = fig._buy_zone  || null;
  _zoneData.sell = fig._sell_zone || null;
  if (fig._atr) _lastAtr = fig._atr;
  _updateZoneBtns();
  _fillFormFromZone();
  autoSuggest();
  const div = document.getElementById('chart-div');
  if (_c1Drawn) {{
    Plotly.react('chart-div', fig.data, fig.layout, _pltConfig);
  }} else {{
    Plotly.newPlot('chart-div', fig.data, fig.layout, _pltConfig);
    div.on('plotly_click', _onChartClick);
    div.on('plotly_relayout', _onRelayout);
    _c1Drawn = true;
  }}
}}

async function refreshChart2() {{
  const fig = await _fetchChart(250);
  if (!fig) return;
  const cfg2 = Object.assign({{}}, _pltConfig, {{displayModeBar:false}});
  if (_c2Drawn) Plotly.react('chart-div2', fig.data, fig.layout, cfg2);
  else {{ Plotly.newPlot('chart-div2', fig.data, fig.layout, cfg2); _c2Drawn=true; }}
}}

async function refreshAll() {{
  try {{
    const r = await fetch('/autoorders/status');
    const d = await r.json();
    if (!d.ok) return;
    _updateBar(d.state);
    if (d.result)      _updateAnalysis(d.result);
    if (d.decisions)   _updateDecisions(d.decisions);
    if (d.order_stats) _updateOrderChip(d.order_stats);
    _updateTable(d.pending_orders||[]);
    if (_activeTab==='orders')    refreshChart2();
    if (_activeTab==='ao-orders') _loadOrders(_curOrderTab);
  }} catch(e) {{ console.warn('poll err',e); }}
}}

function _updateBar(state) {{
  const on = state.running;
  document.getElementById('status-text').textContent = on ? 'Ruleaza' : 'Inactiv';
  document.getElementById('status-dot').className    = 'status-dot'+(on?' on':'');
  const ts = state.last_scan ? new Date(state.last_scan).toLocaleTimeString('ro') : '';
  document.getElementById('last-scan-ts').textContent = ts ? `Scanat ${{ts}}` : '';
  // sync UI
  const st = document.getElementById('cfg-strat');
  if (st && state.strategy && st.value !== state.strategy) st.value = state.strategy;
  document.getElementById('cfg-symbol').value   = state.symbol;
  document.getElementById('cfg-tf').value       = state.tf;
  document.getElementById('cfg-bars').value     = state.bars;
  document.getElementById('cfg-interval').value = state.interval;
  document.getElementById('cfg-conf').value     = state.min_confidence;
  document.getElementById('cfg-auto').checked   = state.auto_execute;
}}

function _updateAnalysis(res) {{
  const sig = res.signal||'HOLD';
  const el  = document.getElementById('an-signal');
  el.textContent = sig;
  el.className   = 'signal-badge '+(sig==='BUY'?'buy':sig==='SELL'?'sell':sig.includes('PENDING')?'pending':'hold');
  document.getElementById('an-conf').textContent = res.confidence!=null ? res.confidence+'%' : '—';
  document.getElementById('an-risk').textContent = res.risk_score!=null  ? res.risk_score+'%'  : '—';
  const pe = res.pending_entry;
  document.getElementById('an-pending').textContent = pe
    ? `${{pe.order_type}} @ ${{pe.price}}\\nSL ${{pe.sl}}  TP ${{pe.tp}}  (${{pe.tf||'?'}})`
    : '—';
  const just = (res.justification||[]).slice(0,5).join('\\n');
  document.getElementById('an-just').textContent = just||'—';
}}

function _updateDecisions(decs) {{
  document.getElementById('dec-log').innerHTML = decs.map(d => {{
    const col = d.signal.includes('PENDING')?'#9c27b0':(SIG_COL[d.signal]||'#6070a0');
    const stratBadge = d.strategy ? `<span style="color:#2a3550;font-size:.65rem">${{d.strategy}}</span>` : '';
    return `<div class="dec-row">
      <span class="ts">${{d.ts}}</span>
      <span class="dsig" style="color:${{col}}">${{d.signal}}</span>
      ${{stratBadge}}
      <span class="dres">${{d.result}}</span>
    </div>`;
  }}).join('');
}}

function _updateTable(orders) {{
  const tbody = document.getElementById('pending-tbody');
  if (!orders.length) {{
    tbody.innerHTML='<tr><td colspan="10" style="color:#2a3040;padding:18px;text-align:center">Niciun ordin pending activ</td></tr>';
    return;
  }}
  const MAX_H = 120;
  tbody.innerHTML = orders.map(o => {{
    const clr  = o.is_buy?'#26a69a':'#ef5350';
    const icon = o.is_buy?'▲':'▼';
    const remH = (o.rem_s||0)/3600;
    const pct  = o.expiry==='GTC'?80:Math.min(100,(remH/MAX_H)*100);
    const bClr = pct>50?'#26a69a':pct>20?'#fb8c00':'#ef5350';
    const expStyle = o.expiry==='expirat'?'color:#ef5350':'';
    return `<tr>
      <td style="color:#2a3860;font-family:monospace;font-size:.72rem">#${{o.ticket}}</td>
      <td style="font-weight:600;color:#c0c8e0">${{o.symbol}}</td>
      <td style="color:${{clr}};font-weight:700">${{icon}} ${{o.type}}</td>
      <td style="text-align:right;color:${{clr}};font-family:monospace">${{o.price.toFixed(5)}}</td>
      <td style="text-align:right;color:#ef5350;font-family:monospace">${{o.sl?o.sl.toFixed(5):'—'}}</td>
      <td style="text-align:right;color:#26a69a;font-family:monospace">${{o.tp?o.tp.toFixed(5):'—'}}</td>
      <td style="text-align:right;color:#6070a0">${{o.volume}}</td>
      <td>
        <div style="font-size:.72rem;${{expStyle}}">${{o.expiry_dt}}</div>
        <div class="exp-bar"><div class="exp-fill" style="width:${{pct.toFixed(1)}}%;background:${{bClr}}"></div></div>
      </td>
      <td style="font-weight:500;font-size:.76rem;${{expStyle}}">${{o.expiry}}</td>
      <td><button class="btn-sm" onclick="cancelOrder(${{o.ticket}})">✕</button></td>
    </tr>`;
  }}).join('');
}}

async function cancelOrder(ticket) {{
  if (!confirm(`Anulezi ordinul pending #${{ticket}}?`)) return;
  const r = await fetch('/autoorders/cancel_pending',{{
    method:'POST',headers:{{'Content-Type':'application/json'}},
    body:JSON.stringify({{ticket}}),
  }});
  const d = await r.json();
  alert(d.ok?`Ordin #${{ticket}} anulat`:`Eroare: ${{d.message}}`);
  refreshAll();
  if (_activeTab==='orders') refreshChart2();
}}

// ── Order Manual ──────────────────────────────────────────────────────────────
let _orderDir = 'BUY';
let _lastAtr   = null;
let _suggestTO = null;

async function autoSuggest() {{
  clearTimeout(_suggestTO);
  _suggestTO = setTimeout(async () => {{
    const sym  = document.getElementById('cfg-symbol').value;
    const tf   = document.getElementById('cfg-tf').value;
    const risk = parseFloat(document.getElementById('ord-risk').value) || 100;
    try {{
      const r = await fetch(`/autoorders/suggest?symbol=${{sym}}&tf=${{tf}}&dir=${{_orderDir}}&risk=${{risk}}`);
      const d = await r.json();
      if (!d.ok) return;
      _lastAtr = d.atr;
      // Auto-fill SL / TP
      const slEl = document.getElementById('ord-sl');
      const tpEl = document.getElementById('ord-tp');
      if (slEl) {{ slEl.value = d.sl.toFixed(5); slEl.style.boxShadow='0 0 0 2px #3949ab'; setTimeout(()=>slEl.style.boxShadow='',800); }}
      if (tpEl) {{ tpEl.value = d.tp.toFixed(5); tpEl.style.boxShadow='0 0 0 2px #3949ab'; setTimeout(()=>tpEl.style.boxShadow='',800); }}
      // Afiseaza card-ul de sugestii
      document.getElementById('sug-lot').textContent  = d.lot > 0 ? d.lot.toFixed(2) + ' lots' : '< 0.01';
      document.getElementById('sug-pips').textContent = `${{d.sl_pips.toFixed(0)}} / ${{d.tp_pips.toFixed(0)}}`;
      document.getElementById('sug-atr').textContent  = d.atr.toFixed(5);
      document.getElementById('lot-suggest').style.display = 'block';
    }} catch(e) {{ console.warn('suggest err', e); }}
  }}, 200);
}}

function setDir(dir) {{
  _orderDir = dir;
  const buy  = document.getElementById('btn-buy');
  const sell = document.getElementById('btn-sell');
  const BASE = 'flex:1;padding:6px;border-radius:5px;font-weight:700;cursor:pointer;transition:.15s;font-size:.82rem;';
  if (dir === 'BUY') {{
    buy.style.cssText  = BASE + 'border:2px solid #00c853;background:#0a2a14;color:#00e676;opacity:1;box-shadow:0 0 8px #00c85340';
    sell.style.cssText = BASE + 'border:2px solid #333;background:#1e0a0a;color:#ef5350;opacity:.45';
  }} else {{
    sell.style.cssText = BASE + 'border:2px solid #f44336;background:#2a0a0a;color:#ff5252;opacity:1;box-shadow:0 0 8px #f4433640';
    buy.style.cssText  = BASE + 'border:2px solid #333;background:#0a2a14;color:#00e676;opacity:.45';
  }}
  _fillFormFromZone();
  autoSuggest();
}}

function toggleEntryField() {{
  const isPending = document.getElementById('ord-type').value === 'pending';
  document.getElementById('entry-row').style.display = isPending ? 'block' : 'none';
}}

// Auto-scale Y pe bara vizibile (TradingView behavior)
function _onRelayout(evt) {{
  // Detecteaza schimbari pe axa X (pan / zoom)
  const xChanged = evt['xaxis.range[0]'] !== undefined || evt['xaxis.autorange'] !== undefined;
  if (!xChanged) return;
  const div  = document.getElementById('chart-div');
  const gd   = div;
  if (!gd._fullData) return;

  // Ia intervalul X vizibil
  const xRange = gd._fullLayout.xaxis.range;
  if (!xRange || xRange.length < 2) return;
  const xMin = new Date(xRange[0]).getTime();
  const xMax = new Date(xRange[1]).getTime();

  // Gaseste min/max Y al lumânarilor vizibile (trace 0 = candlestick)
  const trace = gd._fullData[0];
  if (!trace || !trace.x) return;
  let lo = Infinity, hi = -Infinity;
  for (let i = 0; i < trace.x.length; i++) {{
    const t = new Date(trace.x[i]).getTime();
    if (t < xMin || t > xMax) continue;
    if (trace.low  && trace.low[i]  < lo) lo = trace.low[i];
    if (trace.high && trace.high[i] > hi) hi = trace.high[i];
  }}
  if (!isFinite(lo) || !isFinite(hi)) return;
  const pad = (hi - lo) * 0.08;
  Plotly.relayout('chart-div', {{'yaxis.range': [lo - pad, hi + pad], 'yaxis.autorange': false}});
}}

// Chart click → seteaza campul selectat
function _onChartClick(data) {{
  if (!data.points || !data.points.length) return;
  const price  = data.points[0].y;
  if (price == null) return;
  const target = document.getElementById('click-target').value;
  const field  = target === 'entry' ? 'ord-entry' : target === 'sl' ? 'ord-sl' : 'ord-tp';
  const el = document.getElementById(field);
  if (el) {{
    el.value = price.toFixed(5);
    el.style.boxShadow = '0 0 0 2px #3949ab';
    setTimeout(() => el.style.boxShadow = '', 600);
  }}
}}

// Fetch bid/ask la fiecare 5s
let _lastBid = 0, _lastAsk = 0;
async function fetchTick() {{
  const sym = document.getElementById('cfg-symbol').value;
  try {{
    const r = await fetch(`/autoorders/tick?symbol=${{sym}}`);
    const d = await r.json();
    if (d.ok) {{
      _lastBid = d.bid;
      _lastAsk = d.ask;
      document.getElementById('tick-price').textContent =
        `Bid ${{d.bid.toFixed(5)}} / Ask ${{d.ask.toFixed(5)}}`;
      _fillEntryFromTick();
    }}
  }} catch(e) {{}}
}}

function _fillFormFromZone() {{
  const z = _orderDir === 'BUY' ? _zoneData.buy : _zoneData.sell;
  if (!z) return;
  const entryEl = document.getElementById('ord-entry');
  const slEl    = document.getElementById('ord-sl');
  const tpEl    = document.getElementById('ord-tp');
  if (entryEl) entryEl.value = z.entry.toFixed(5);
  if (slEl)    slEl.value    = z.sl.toFixed(5);
  if (tpEl)    tpEl.value    = z.tp.toFixed(5);
}}

function _fillEntryFromTick() {{
  const el = document.getElementById('ord-entry');
  if (!el) return;
  // Nu suprascrie daca userul a editat manual (valoare != 0 si != pretul curent)
  const cur = parseFloat(el.value) || 0;
  const mkt = _orderDir === 'BUY' ? _lastAsk : _lastBid;
  if (mkt <= 0) return;
  // Suprascrie doar daca e gol / zero / era pretul anterior de market
  if (cur === 0 || cur === _lastBid || cur === _lastAsk) {{
    el.value = mkt.toFixed(5);
  }}
}}

async function placeOrder() {{
  const symbol   = document.getElementById('cfg-symbol').value;
  const ordType  = document.getElementById('ord-type').value;
  const sl       = parseFloat(document.getElementById('ord-sl').value)    || 0;
  const tp       = parseFloat(document.getElementById('ord-tp').value)    || 0;
  const entry    = parseFloat(document.getElementById('ord-entry').value) || 0;
  const risk     = parseFloat(document.getElementById('ord-risk').value)  || 100;
  const resultEl = document.getElementById('order-result');
  const btn      = document.getElementById('btn-place');
  const dir      = _orderDir;

  resultEl.textContent = '';
  if (!sl || !tp) {{ resultEl.style.color='#ef5350'; resultEl.textContent='⚠ SL si TP sunt obligatorii'; return; }}
  if (ordType==='pending' && !entry) {{ resultEl.style.color='#ef5350'; resultEl.textContent='⚠ Entry price obligatoriu pentru Pending'; return; }}

  // ── Dialog confirmare explicit ────────────────────────────────────────────
  const isMarket  = ordType === 'market';
  const dirSymbol = dir === 'BUY' ? '▲' : '▼';
  const dirColor  = dir === 'BUY' ? 'color:#00e676' : 'color:#ff5252';
  const typeLabel = isMarket ? '⚡ MARKET (se executa INSTANT la pret curent!)' : `📌 PENDING LIMIT/STOP @ ${{entry.toFixed(5)}}`;
  const warnLine  = isMarket ? '\\n⚠️  ATENTIE: ordinul MARKET se executa imediat la pretul curent!\\n' : '';

  const msg =
    `${{dirSymbol}} ${{dir}} ${{symbol}}\\n` +
    `Tip: ${{typeLabel}}\\n` +
    `${{warnLine}}` +
    `SL:   ${{sl.toFixed(5)}}\\n` +
    `TP:   ${{tp.toFixed(5)}}\\n` +
    `Risk: $${{risk}}\\n\\n` +
    `Confirmi plasarea ordinului?`;

  if (!confirm(msg)) return;

  btn.disabled = true;
  btn.textContent = '⏳ Se trimite...';

  try {{
    const body = {{ symbol, signal: dir, order_type: ordType, sl, tp, entry, risk_dollars: risk }};
    const r = await fetch('/autoorders/place_order', {{
      method: 'POST',
      headers: {{'Content-Type': 'application/json'}},
      body: JSON.stringify(body),
    }});
    const d = await r.json();
    resultEl.style.color = d.ok ? '#26a69a' : '#ef5350';
    resultEl.textContent = (d.ok ? '✓ ' : '✗ ') + (d.message || (d.ok ? 'Trimis' : 'Eroare'));
    if (d.ok) {{
      ['ord-sl','ord-tp','ord-entry'].forEach(id => {{ const el=document.getElementById(id); if(el) el.value=''; }});
      refreshAll();
      if (_activeTab==='orders') refreshChart2();
    }}
  }} catch(e) {{
    resultEl.style.color='#ef5350';
    resultEl.textContent = '✗ Eroare de retea';
  }} finally {{
    btn.disabled = false;
    btn.textContent = '▶ Plaseaza Order';
  }}
}}

// ── Orders Tab (OrderManager) ─────────────────────────────────────────────────
let _curOrderTab = 'active';

function switchOrdersTab(tab) {{
  _curOrderTab = tab;
  ['active','filled','history'].forEach(t => {{
    const el = document.getElementById('subtab-'+t);
    if (el) el.classList.toggle('active', t===tab);
  }});
  _loadOrders(tab);
}}

async function _loadOrders(status) {{
  status = status || _curOrderTab;
  const sym = document.getElementById('cfg-symbol').value;
  try {{
    const r = await fetch(`/autoorders/orders?status=${{status}}&symbol=${{sym}}`);
    const d = await r.json();
    if (!d.ok) return;
    _renderOrdersTable(d.orders, status);
    const cnt = d.orders.length;
    const el = document.getElementById('orders-count');
    if (el) el.textContent = cnt + ' ordin' + (cnt===1?'':'e');
    if (d.stats) _updateOrderChip(d.stats);
  }} catch(e) {{ console.warn('orders load err',e); }}
}}

function _renderOrdersTable(orders, status) {{
  const tbody = document.getElementById('orders-tbody');
  if (!orders || !orders.length) {{
    tbody.innerHTML = '<tr><td colspan="12" style="color:#2a3040;padding:18px;text-align:center">Niciun ordin</td></tr>';
    return;
  }}
  const STATUS_COL = {{
    pending_mt5:'#5c6bc0', pending_virtual:'#ab47bc',
    filled:'#26a69a', cancelled:'#3a4060', expired:'#3a4060',
    closed_tp:'#26a69a', closed_sl:'#ef5350', replaced:'#5a6040',
  }};
  tbody.innerHTML = orders.map(o => {{
    const clr   = o.direction==='BUY'?'#26a69a':'#ef5350';
    const icon  = o.direction==='BUY'?'▲':'▼';
    const sClr  = STATUS_COL[o.status]||'#5a6070';
    const ts    = o.created ? o.created.slice(0,16).replace('T',' ') : '—';
    const isAct = o.status==='pending_mt5'||o.status==='pending_virtual';
    const ticket = o.mt5_ticket ? `<br><span style="color:#1e2a4a;font-size:.64rem">#${{o.mt5_ticket}}</span>` : '';
    const cancelBtn = isAct
      ? `<button class="btn-sm" onclick="cancelOrder2('${{o.id}}')">✕</button>`
      : '';
    return `<tr>
      <td style="font-family:monospace;font-size:.66rem;color:#2a3860">${{o.id}}${{ticket}}</td>
      <td style="font-weight:600;color:#c0c8e0">${{o.symbol}}</td>
      <td style="color:#4a5870;font-size:.69rem">${{o.strategy}}</td>
      <td style="color:${{clr}};font-weight:700">${{icon}} ${{o.direction}}</td>
      <td style="color:#3a4060;font-size:.68rem">${{o.mode}}</td>
      <td style="text-align:right;color:${{clr}};font-family:monospace">${{(+o.entry).toFixed(5)}}</td>
      <td style="text-align:right;color:#ef9a9a;font-family:monospace">${{(+o.sl).toFixed(5)}}</td>
      <td style="text-align:right;color:#80cbc4;font-family:monospace">${{(+o.tp).toFixed(5)}}</td>
      <td style="text-align:right;color:#8090a0">${{o.rr||'—'}}</td>
      <td><span style="color:${{sClr}};font-size:.69rem;font-weight:600">${{o.status}}</span></td>
      <td style="color:#2a3040;font-size:.67rem">${{ts}}</td>
      <td>${{cancelBtn}}</td>
    </tr>`;
  }}).join('');
}}

async function cancelOrder2(orderId) {{
  if (!confirm('Anulezi ordinul ' + orderId + '?')) return;
  const r = await fetch('/autoorders/orders/cancel', {{
    method:'POST', headers:{{'Content-Type':'application/json'}},
    body: JSON.stringify({{order_id: orderId, reason:'manual'}}),
  }});
  const d = await r.json();
  if (!d.ok) alert('Eroare la anulare');
  _loadOrders(_curOrderTab);
}}

async function syncOrders() {{
  const r = await fetch('/autoorders/orders/sync', {{method:'POST'}});
  const d = await r.json();
  if (d.ok) _loadOrders(_curOrderTab);
}}

function syncStratChecks() {{
  const checked = [];
  document.querySelectorAll('input[id^="sc-"]').forEach(cb => {{
    if (cb.checked) checked.push(cb.value);
  }});
  if (!checked.length) return;
  fetch('/autoorders/set', {{
    method:'POST', headers:{{'Content-Type':'application/json'}},
    body: JSON.stringify({{strategies_active: checked}}),
  }});
}}

function setExecMode(mode) {{
  const mt5Btn  = document.getElementById('exec-mt5');
  const virtBtn = document.getElementById('exec-virt');
  if (mt5Btn)  {{ mt5Btn.classList.toggle('active-mt5',  mode==='mt5_pending'); }}
  if (virtBtn) {{ virtBtn.classList.toggle('active-virt', mode==='virtual'); }}
  fetch('/autoorders/set', {{
    method:'POST', headers:{{'Content-Type':'application/json'}},
    body: JSON.stringify({{exec_mode: mode}}),
  }});
}}

function applySW() {{
  const val  = document.getElementById('cfg-sw').value;
  const syms = val.split(',').map(s=>s.trim().toUpperCase()).filter(Boolean);
  if (!syms.length) return;
  fetch('/autoorders/set', {{
    method:'POST', headers:{{'Content-Type':'application/json'}},
    body: JSON.stringify({{symbols_watch: syms}}),
  }});
}}

function applyRepl() {{
  const val = parseFloat(document.getElementById('cfg-repl').value);
  if (isNaN(val)) return;
  fetch('/autoorders/set', {{
    method:'POST', headers:{{'Content-Type':'application/json'}},
    body: JSON.stringify({{replace_thr: val}}),
  }});
}}

function _updateOrderChip(stats) {{
  const el = document.getElementById('chip-active');
  if (!el) return;
  el.textContent      = stats.active || 0;
  el.style.background = (stats.active||0) > 0 ? '#1a237e' : '#1a1a30';
  el.style.color      = (stats.active||0) > 0 ? '#7986cb' : '#5c6bc0';
}}

// ── Init ──────────────────────────────────────────────────────────────────────
// Default tip ordin = pending → arata entry row
document.getElementById('ord-type').value = 'pending';
toggleEntryField();
updateTFBtns();
fetchTick();
setInterval(fetchTick, 5000);
refreshChart();
refreshAll();
setInterval(refreshAll, 10000);
</script>
</body>
</html>"""
