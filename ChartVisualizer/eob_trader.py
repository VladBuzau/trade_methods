"""
EOBTrader — pagina si scanner dedicat exclusiv strategiei EOB cu ordine pending.
Thread si setari complet separate de AutoTrader. Nu influenteaza si nu e influentat
de scannerul principal.
"""
import threading
import time
import json
import logging
from datetime import datetime

import plotly.graph_objects as go
from flask import Blueprint, request, Response

from app import (
    fetch, MT5_AVAILABLE, mt5, NpEncoder, login_required,
    place_pending_order, ALL_TFS, SYMBOLS, SYMBOLS_CRYPTO,
)
from strategies.eob import EOBStrategy, detect_eob_zones_v2, detect_eob_approach_zones

log = logging.getLogger(__name__)

eob_trader_bp = Blueprint("eob_trader", __name__)

# ── State separat — nu partajeaza nimic cu autotrader.scanner ─────────────────
_state = {
    "running":        False,
    "symbol":         "EURUSD",
    "tf":             "H1",
    "bars":           500,
    "interval":       60,
    "auto_execute":   False,
    "expiry_hours":   24,
    "min_confidence": 60.0,
    "last_scan":      None,
    "decisions":      [],
}
_lock   = threading.Lock()
_thread: threading.Thread | None = None
_strat  = EOBStrategy()

_TF_EXPIRY = {"M5": 3, "M15": 6, "M30": 12, "H1": 24, "H4": 72, "D1": 120, "W1": 240}


# ── Scanner ───────────────────────────────────────────────────────────────────
def _scan_once():
    sym      = _state["symbol"]
    tf       = _state["tf"]
    bars     = _state["bars"]
    min_conf = _state["min_confidence"]
    auto_ex  = _state["auto_execute"]
    exp_h    = _TF_EXPIRY.get(tf, _state["expiry_hours"])

    try:
        res = _strat.analyze(sym, [tf], bars=bars, min_confidence=min_conf)
    except Exception as exc:
        log.warning(f"EOBTrader scan: {exc}")
        return

    pe  = res.get("pending_entry")
    sig = res.get("signal", "HOLD")

    if auto_ex and pe and sig == "HOLD":
        ok, msg = place_pending_order(
            sym, pe["signal"], pe["price"], pe["sl"], pe["tp"],
            strategy="eob", expiry_hours=exp_h,
        )
        decision = {
            "ts": datetime.now().strftime("%H:%M:%S"),
            "signal": f"PENDING {pe.get('order_type','?')}",
            "executed": ok, "result": msg,
        }
    elif sig in ("BUY", "SELL"):
        decision = {
            "ts": datetime.now().strftime("%H:%M:%S"),
            "signal": sig,
            "executed": False,
            "result": f"Confidence {res.get('confidence',0)}%",
        }
    else:
        just = (res.get("justification") or ["—"])[0]
        decision = {
            "ts": datetime.now().strftime("%H:%M:%S"),
            "signal": "HOLD", "executed": False, "result": just,
        }

    with _lock:
        _state["last_scan"] = datetime.now().isoformat()
        _state["_last_res"] = res
        _state["decisions"].insert(0, decision)
        while len(_state["decisions"]) > 80:
            _state["decisions"].pop()


def _scanner_loop():
    log.info("EOBTrader pornit.")
    while _state["running"]:
        _scan_once()
        interval = _state["interval"]
        for _ in range(max(1, interval // 2)):
            if not _state["running"]:
                break
            time.sleep(2)
    log.info("EOBTrader oprit.")


# ── Chart builder ─────────────────────────────────────────────────────────────
def _build_chart(symbol: str, tf: str, bars: int) -> dict:
    df, _ = fetch(symbol, tf, bars)
    if df is None or df.empty:
        return {}

    dates     = df.index.tolist()
    price_now = float(df["close"].iloc[-1])
    shapes, annots = [], []

    # ── EOB zones v2 (zone active — pret e deja in zona sau langa ea) ─────────
    try:
        zones = detect_eob_zones_v2(df, lookback=min(bars, 120))
        for z in zones[:30]:
            is_bear = z["type"] == "BEARISH"
            fill    = "rgba(239,83,80,0.10)"  if is_bear else "rgba(38,166,154,0.10)"
            border  = "#ef5350" if is_bear else "#26a69a"
            idx     = z.get("index", 0)
            x0      = dates[min(idx, len(dates)-1)]
            shapes.append(dict(
                type="rect", xref="x", yref="y",
                x0=x0, x1=dates[-1],
                y0=z["zone_low"], y1=z["zone_high"],
                fillcolor=fill, line=dict(color=border, width=1, dash="dot"),
                layer="below",
            ))
            grade = z.get("vol_grade", "")
            annots.append(dict(
                x=dates[-1], y=z["zone_high"], xref="x", yref="y",
                text=f"{z['type'][:4]} {grade} d={z.get('decay',0):.1f}",
                showarrow=False, font=dict(size=8, color=border),
                xanchor="right", yanchor="bottom",
            ))
    except Exception as exc:
        log.warning(f"EOB zones chart: {exc}")

    # ── Approach zones (candidati pending — pret se apropie dar nu e in zona) ─
    try:
        bear_ap = detect_eob_approach_zones(df, "BEARISH", price_now)
        bull_ap = detect_eob_approach_zones(df, "BULLISH", price_now)
        for z in (bear_ap + bull_ap)[:8]:
            is_bear = z["type"] == "BEARISH"
            fill    = "rgba(239,83,80,0.22)" if is_bear else "rgba(38,166,154,0.22)"
            border  = "#ff1744" if is_bear else "#00e676"
            ago_idx = max(0, len(dates) - z.get("bars_ago", 1) - 3)
            shapes.append(dict(
                type="rect", xref="x", yref="y",
                x0=dates[ago_idx], x1=dates[-1],
                y0=z["zone_low"], y1=z["zone_high"],
                fillcolor=fill, line=dict(color=border, width=1.5),
                layer="below",
            ))
            annots.append(dict(
                x=dates[-1], y=z["pending_entry"], xref="x", yref="y",
                text=f"⟶ {z['order_type']}",
                showarrow=False, font=dict(size=9, color=border, family="monospace"),
                xanchor="right", yanchor="top",
                bgcolor="rgba(0,0,0,0.5)", bordercolor=border, borderwidth=1,
            ))
    except Exception as exc:
        log.warning(f"Approach zones chart: {exc}")

    # ── Pending orders MT5 (linii orizontale entry/SL/TP) ─────────────────────
    try:
        if MT5_AVAILABLE and mt5 and mt5.initialize():
            pending = mt5.orders_get(symbol=symbol) or []
            for op in pending:
                cmt = (op.comment or "").upper()
                if "EOB" not in cmt:
                    continue
                is_buy  = op.type in (2, 4, 6)
                c_entry = "#26a69a" if is_buy else "#ef5350"
                type_lbl = {2:"BUY LIM", 3:"SELL LIM", 4:"BUY STP", 5:"SELL STP"}.get(op.type, "PENDING")
                # Entry line
                shapes.append(dict(
                    type="line", xref="x", yref="y",
                    x0=dates[0], x1=dates[-1],
                    y0=op.price_open, y1=op.price_open,
                    line=dict(color=c_entry, width=1.5, dash="dash"),
                ))
                annots.append(dict(
                    x=dates[0], y=op.price_open, xref="x", yref="y",
                    text=f"#{op.ticket} {type_lbl}",
                    showarrow=False, font=dict(size=9, color=c_entry),
                    xanchor="left", yanchor="bottom",
                    bgcolor="rgba(0,0,0,0.6)", bordercolor=c_entry, borderwidth=1,
                ))
                if op.sl:
                    shapes.append(dict(
                        type="line", xref="x", yref="y",
                        x0=dates[0], x1=dates[-1], y0=op.sl, y1=op.sl,
                        line=dict(color="#ef5350", width=0.8, dash="dot"),
                    ))
                if op.tp:
                    shapes.append(dict(
                        type="line", xref="x", yref="y",
                        x0=dates[0], x1=dates[-1], y0=op.tp, y1=op.tp,
                        line=dict(color="#26a69a", width=0.8, dash="dot"),
                    ))
    except Exception as exc:
        log.warning(f"MT5 pending overlay: {exc}")

    fig = go.Figure(data=[go.Candlestick(
        x=dates,
        open=df["open"], high=df["high"], low=df["low"], close=df["close"],
        name="OHLC",
        increasing_line_color="#26a69a", decreasing_line_color="#ef5350",
        increasing_fillcolor="#26a69a",  decreasing_fillcolor="#ef5350",
    )])
    fig.update_layout(
        paper_bgcolor="#08080f", plot_bgcolor="#08080f",
        font=dict(color="#8892a4", size=10),
        margin=dict(l=50, r=10, t=20, b=30),
        xaxis=dict(rangeslider=dict(visible=False), gridcolor="#13131f", showgrid=True, color="#8892a4"),
        yaxis=dict(gridcolor="#13131f", showgrid=True, color="#8892a4"),
        shapes=shapes, annotations=annots,
        height=480, showlegend=False,
    )
    return json.loads(fig.to_json())


# ── MT5 pending orders helper ─────────────────────────────────────────────────
def _get_pending_orders(symbol: str):
    if not MT5_AVAILABLE or mt5 is None:
        return []
    try:
        if not mt5.initialize():
            return []
        orders = mt5.orders_get(symbol=symbol) or []
        now_ts = int(datetime.now().timestamp())
        out = []
        type_names = {2:"BUY_LIMIT", 3:"SELL_LIMIT", 4:"BUY_STOP", 5:"SELL_STOP", 6:"BUY_STP_LIM", 7:"SELL_STP_LIM"}
        for op in orders:
            cmt = (op.comment or "").upper()
            if "EOB" not in cmt:
                continue
            exp_ts = getattr(op, "time_expiration", 0) or 0
            if exp_ts > 0:
                rem = exp_ts - now_ts
                if rem > 0:
                    h, r = divmod(rem, 3600)
                    expiry_str = f"{int(h)}h {int(r//60)}m"
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
        return sorted(out, key=lambda x: x["exp_ts"] or 9e9)
    except Exception as exc:
        log.warning(f"_get_pending_orders: {exc}")
        return []


# ── Routes ────────────────────────────────────────────────────────────────────
@eob_trader_bp.route("/eobtrader/")
@login_required
def eob_page():
    all_syms = SYMBOLS + SYMBOLS_CRYPTO
    tfs      = ALL_TFS + ["W1"]
    return Response(_render(all_syms, tfs), content_type="text/html; charset=utf-8")


@eob_trader_bp.route("/eobtrader/status")
@login_required
def eob_status():
    with _lock:
        state_snap = {k: v for k, v in _state.items() if k not in ("_last_res",)}
        res = _state.get("_last_res")
        decs = list(_state["decisions"][:50])
    pending = _get_pending_orders(_state["symbol"])
    out_res = None
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
        "ok": True, "state": state_snap,
        "result": out_res, "decisions": decs, "pending_orders": pending,
    }, cls=NpEncoder), content_type="application/json")


@eob_trader_bp.route("/eobtrader/chart")
@login_required
def eob_chart():
    sym  = request.args.get("symbol", _state["symbol"])
    tf   = request.args.get("tf", _state["tf"])
    bars = int(request.args.get("bars", _state["bars"]))
    return Response(json.dumps(_build_chart(sym, tf, bars), cls=NpEncoder),
                    content_type="application/json")


@eob_trader_bp.route("/eobtrader/start", methods=["POST"])
@login_required
def eob_start():
    global _thread
    _apply(request.get_json(silent=True) or {})
    if not _state["running"]:
        _state["running"] = True
        _thread = threading.Thread(target=_scanner_loop, daemon=True, name="eob-trader")
        _thread.start()
    return Response('{"ok":true}', content_type="application/json")


@eob_trader_bp.route("/eobtrader/stop", methods=["POST"])
@login_required
def eob_stop():
    _state["running"] = False
    return Response('{"ok":true}', content_type="application/json")


@eob_trader_bp.route("/eobtrader/set", methods=["POST"])
@login_required
def eob_set():
    _apply(request.get_json(silent=True) or {})
    return Response('{"ok":true}', content_type="application/json")


@eob_trader_bp.route("/eobtrader/cancel_pending", methods=["POST"])
@login_required
def eob_cancel():
    body   = request.get_json(silent=True) or {}
    ticket = int(body.get("ticket", 0))
    if not ticket:
        return Response('{"ok":false,"message":"ticket lipsa"}', content_type="application/json")
    if not MT5_AVAILABLE or mt5 is None:
        return Response('{"ok":false,"message":"MT5 indisponibil"}', content_type="application/json")
    try:
        mt5.initialize()
        res = mt5.order_send({"action": mt5.TRADE_ACTION_REMOVE, "order": ticket})
        ok  = bool(res and res.retcode == mt5.TRADE_RETCODE_DONE)
        msg = "Anulat" if ok else f"retcode={res.retcode if res else -1}"
        return Response(json.dumps({"ok": ok, "message": msg}), content_type="application/json")
    except Exception as exc:
        return Response(json.dumps({"ok": False, "message": str(exc)}), content_type="application/json")


def _apply(body: dict):
    if "symbol"         in body: _state["symbol"]         = str(body["symbol"]).upper()
    if "tf"             in body: _state["tf"]             = str(body["tf"])
    if "bars"           in body: _state["bars"]           = max(100, min(5000, int(body["bars"])))
    if "interval"       in body: _state["interval"]       = max(10, int(body["interval"]))
    if "auto_execute"   in body: _state["auto_execute"]   = bool(body["auto_execute"])
    if "expiry_hours"   in body: _state["expiry_hours"]   = max(1, int(body["expiry_hours"]))
    if "min_confidence" in body: _state["min_confidence"] = max(0.0, min(100.0, float(body["min_confidence"])))


# ── HTML ──────────────────────────────────────────────────────────────────────
def _render(symbols, tfs) -> str:
    sym_opts = "".join(f'<option value="{s}">{s}</option>' for s in symbols)
    tf_opts  = "".join(f'<option value="{t}">{t}</option>' for t in tfs)
    return f"""<!DOCTYPE html>
<html lang="ro">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>EOB Trader</title>
<script src="https://cdn.plot.ly/plotly-2.27.0.min.js"></script>
<style>
*{{box-sizing:border-box;margin:0;padding:0}}
body{{background:#07070e;color:#a0aabb;font-family:'Inter',sans-serif;font-size:.8rem;display:flex;flex-direction:column;height:100vh;overflow:hidden}}
/* header */
.hdr{{display:flex;align-items:center;gap:12px;padding:8px 16px;background:#0c0c1a;border-bottom:1px solid #1c1c30;flex-shrink:0}}
.hdr-title{{font-size:1rem;font-weight:600;color:#c0c8e0;letter-spacing:.5px}}
.hdr a{{color:#5c6bc0;text-decoration:none;font-size:.78rem}}
.hdr a:hover{{color:#7986cb}}
.status-dot{{width:8px;height:8px;border-radius:50%;background:#333;margin-left:4px}}
.status-dot.on{{background:#26a69a}}
/* tabs */
.tabs{{display:flex;gap:2px;padding:6px 14px 0;background:#0c0c1a;border-bottom:1px solid #1c1c30;flex-shrink:0}}
.tab{{padding:5px 14px;border-radius:6px 6px 0 0;cursor:pointer;color:#6070a0;border:1px solid transparent;border-bottom:none;font-size:.8rem}}
.tab.active{{background:#0f0f20;color:#c0c8e0;border-color:#1c1c30}}
.tab:hover:not(.active){{color:#9090c0}}
/* main layout */
.main{{display:flex;flex:1;overflow:hidden}}
/* sidebar */
.sidebar{{width:220px;flex-shrink:0;background:#0c0c1a;border-right:1px solid #1a1a2e;padding:12px;overflow-y:auto;display:flex;flex-direction:column;gap:10px}}
.sidebar label{{display:block;color:#6070a0;font-size:.72rem;margin-bottom:3px}}
.sidebar select,.sidebar input{{width:100%;background:#10101e;border:1px solid #1e1e35;color:#c0c8e0;padding:4px 8px;border-radius:5px;font-size:.8rem}}
.sidebar select:focus,.sidebar input:focus{{outline:none;border-color:#3949ab}}
/* toggle */
.toggle-row{{display:flex;justify-content:space-between;align-items:center;margin-top:2px}}
.toggle{{position:relative;width:34px;height:18px;display:inline-block}}
.toggle input{{opacity:0;width:0;height:0}}
.slider{{position:absolute;inset:0;background:#1a1a2e;border-radius:18px;cursor:pointer;transition:.2s}}
.slider:before{{position:absolute;content:"";height:12px;width:12px;left:3px;bottom:3px;background:#555;border-radius:50%;transition:.2s}}
input:checked+.slider{{background:#1a237e}}
input:checked+.slider:before{{transform:translateX(16px);background:#7986cb}}
/* buttons */
.btn{{padding:5px 12px;border-radius:6px;border:none;cursor:pointer;font-size:.78rem;font-weight:500}}
.btn-start{{background:#1a237e;color:#c5cae9}}
.btn-start:hover{{background:#283593}}
.btn-stop{{background:#1a1a2e;color:#ef9a9a;border:1px solid #3a2222}}
.btn-stop:hover{{background:#2a1a1a}}
.btn-sm{{padding:3px 8px;border-radius:4px;border:1px solid #2a2a3e;background:#10101e;color:#9090c0;cursor:pointer;font-size:.72rem}}
.btn-sm:hover{{color:#ef5350;border-color:#ef5350}}
/* content panels */
.content{{flex:1;display:flex;flex-direction:column;overflow:hidden}}
.tab-panel{{display:none;flex:1;overflow:hidden;flex-direction:column}}
.tab-panel.active{{display:flex}}
/* chart */
#chart-div,#chart-div2{{flex:1;min-height:0}}
/* analysis card */
.analysis-card{{background:#0f0f1e;border:1px solid #1a1a2e;border-radius:6px;padding:8px}}
.analysis-card .label{{color:#6070a0;font-size:.7rem;margin-bottom:2px}}
.signal-badge{{display:inline-block;padding:2px 10px;border-radius:4px;font-weight:600;font-size:.78rem}}
.signal-badge.buy{{background:#0d2a24;color:#26a69a}}
.signal-badge.sell{{background:#2a1010;color:#ef5350}}
.signal-badge.hold{{background:#1a1a2e;color:#5c6070}}
.signal-badge.pending{{background:#1a1030;color:#9c27b0}}
/* decisions log */
.dec-log{{overflow-y:auto;max-height:140px;display:flex;flex-direction:column;gap:2px}}
.dec-row{{display:flex;gap:8px;padding:2px 0;border-bottom:1px solid #10101e;align-items:center}}
.dec-row .ts{{color:#3a4050;font-size:.68rem;white-space:nowrap}}
.dec-row .dsig{{font-weight:500;font-size:.72rem;min-width:60px}}
.dec-row .dres{{color:#5c6070;font-size:.72rem;flex:1;overflow:hidden;white-space:nowrap;text-overflow:ellipsis}}
/* pending table */
.pending-wrap{{flex:1;overflow:auto;padding:8px}}
.pending-tbl{{width:100%;border-collapse:collapse;font-size:.76rem}}
.pending-tbl th{{padding:5px 10px;color:#5c6070;text-align:left;border-bottom:1px solid #1a1a2e;font-weight:500;white-space:nowrap}}
.pending-tbl td{{padding:5px 10px;border-bottom:1px solid #10101e;vertical-align:middle}}
.pending-tbl tr:hover td{{background:#0f0f1e}}
.buy-clr{{color:#26a69a}}
.sell-clr{{color:#ef5350}}
.exp-bar{{height:4px;border-radius:2px;background:#1a1a2e;margin-top:3px;overflow:hidden}}
.exp-fill{{height:100%;border-radius:2px;background:#3949ab;transition:width .5s}}
/* section title */
.sec-title{{font-size:.72rem;font-weight:600;color:#4a5080;text-transform:uppercase;letter-spacing:.8px;margin-bottom:5px}}
/* scrollbar */
::-webkit-scrollbar{{width:4px;height:4px}}
::-webkit-scrollbar-track{{background:#0a0a14}}
::-webkit-scrollbar-thumb{{background:#1e1e35;border-radius:2px}}
.separator{{height:1px;background:#1a1a2e;margin:4px 0}}
</style>
</head>
<body>

<div class="hdr">
  <span class="hdr-title">⬡ EOB Trader</span>
  <a href="/autotrader">← AutoTrader</a>
  <div style="flex:1"></div>
  <span id="status-text" style="color:#4a5080;font-size:.75rem">Inactiv</span>
  <span class="status-dot" id="status-dot"></span>
  <span id="last-scan-ts" style="color:#2a3040;font-size:.7rem;margin-left:8px"></span>
</div>

<div class="tabs">
  <div class="tab active" onclick="switchTab('analysis')">📊 Analiza EOB</div>
  <div class="tab" onclick="switchTab('orders')">📋 Ordine Pending</div>
</div>

<div class="main">
  <!-- SIDEBAR -->
  <div class="sidebar">
    <div>
      <div class="sec-title">Configurare</div>
      <label>Simbol</label>
      <select id="cfg-symbol" onchange="applyConfig()">{sym_opts}</select>
    </div>
    <div>
      <label>Timeframe</label>
      <select id="cfg-tf" onchange="applyConfig()">{tf_opts}</select>
    </div>
    <div>
      <label>Bare</label>
      <input id="cfg-bars" type="number" value="500" min="100" max="5000" onchange="applyConfig()">
    </div>
    <div>
      <label>Interval scan (sec)</label>
      <input id="cfg-interval" type="number" value="60" min="10" max="3600" onchange="applyConfig()">
    </div>
    <div>
      <label>Min Confidence %</label>
      <input id="cfg-conf" type="number" value="60" min="0" max="100" step="5" onchange="applyConfig()">
    </div>
    <div class="separator"></div>
    <div class="toggle-row">
      <span>Auto Execute</span>
      <label class="toggle">
        <input type="checkbox" id="cfg-auto" onchange="applyConfig()">
        <span class="slider"></span>
      </label>
    </div>
    <div style="display:flex;gap:6px;margin-top:4px">
      <button class="btn btn-start" onclick="startScanner()">▶ Start</button>
      <button class="btn btn-stop"  onclick="stopScanner()">■ Stop</button>
    </div>
    <div class="separator"></div>
    <div class="sec-title">Ultima analiza</div>
    <div class="analysis-card" id="analysis-card">
      <div class="label">Semnal</div>
      <span class="signal-badge hold" id="an-signal">—</span>
      <div class="label" style="margin-top:6px">Confidence</div>
      <span id="an-conf" style="color:#c0c8e0;font-weight:500">—</span>
      <div class="label" style="margin-top:6px">Risk Score</div>
      <span id="an-risk" style="color:#9c27b0;font-weight:500">—</span>
      <div class="label" style="margin-top:6px">Pending Entry</div>
      <div id="an-pending" style="color:#9c27b0;font-size:.72rem">—</div>
      <div class="label" style="margin-top:6px">Justificare</div>
      <div id="an-just" style="color:#6070a0;font-size:.72rem;line-height:1.4">—</div>
    </div>
    <div class="separator"></div>
    <div class="sec-title">Decizii recente</div>
    <div class="dec-log" id="dec-log"></div>
  </div>

  <!-- CONTENT -->
  <div class="content">

    <!-- TAB: Analiza EOB -->
    <div class="tab-panel active" id="tab-analysis">
      <div id="chart-div"></div>
      <div style="padding:6px 10px;background:#0c0c1a;border-top:1px solid #1a1a2e;display:flex;gap:10px;align-items:center;flex-shrink:0">
        <span style="color:#3a4050;font-size:.7rem">
          <span style="display:inline-block;width:10px;height:10px;background:rgba(38,166,154,0.3);border:1px solid #26a69a;margin-right:3px;vertical-align:middle"></span>Zona EOB bullish
          <span style="display:inline-block;width:10px;height:10px;background:rgba(239,83,80,0.3);border:1px solid #ef5350;margin-left:8px;margin-right:3px;vertical-align:middle"></span>Zona EOB bearish
          <span style="display:inline-block;width:10px;height:10px;background:rgba(0,230,118,0.35);border:1px solid #00e676;margin-left:8px;margin-right:3px;vertical-align:middle"></span>Approach (pending)
          <span style="color:#5c5c7c;margin-left:8px">— Entry pending &nbsp;&#183;&#183; SL/TP</span>
        </span>
        <div style="flex:1"></div>
        <button class="btn-sm" onclick="refreshChart()">⟳ Refresh chart</button>
      </div>
    </div>

    <!-- TAB: Ordine Pending -->
    <div class="tab-panel" id="tab-orders">
      <div id="chart-div2"></div>
      <div style="background:#0c0c1a;border-top:1px solid #1a1a2e;padding:5px 12px;display:flex;align-items:center;gap:10px;flex-shrink:0">
        <span style="color:#4a5080;font-size:.72rem;font-weight:600;text-transform:uppercase;letter-spacing:.6px">Ordine Pending Actve</span>
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
              <th>Expira</th>
              <th>Timp ramas</th>
              <th></th>
            </tr>
          </thead>
          <tbody id="pending-tbody">
            <tr><td colspan="10" style="color:#3a4050;padding:12px">Se incarca...</td></tr>
          </tbody>
        </table>
      </div>
    </div>

  </div><!-- /content -->
</div><!-- /main -->

<script>
let _chartDrawn  = false;
let _chart2Drawn = false;
let _pollTimer   = null;
let _activeTab   = 'analysis';

// ── Tab switching ─────────────────────────────────────────────────────────────
function switchTab(tab) {{
  _activeTab = tab;
  document.querySelectorAll('.tab').forEach((t,i) => t.classList.toggle('active', i === (tab==='analysis' ? 0 : 1)));
  document.querySelectorAll('.tab-panel').forEach(p => p.classList.remove('active'));
  document.getElementById('tab-' + tab).classList.add('active');
  if (tab === 'orders' && !_chart2Drawn) {{
    refreshChart2();
    _chart2Drawn = true;
  }}
}}

// ── Config helpers ────────────────────────────────────────────────────────────
function getCfg() {{
  return {{
    symbol:         document.getElementById('cfg-symbol').value,
    tf:             document.getElementById('cfg-tf').value,
    bars:           parseInt(document.getElementById('cfg-bars').value) || 500,
    interval:       parseInt(document.getElementById('cfg-interval').value) || 60,
    min_confidence: parseFloat(document.getElementById('cfg-conf').value) || 60,
    auto_execute:   document.getElementById('cfg-auto').checked,
  }};
}}

function applyConfig() {{
  fetch('/eobtrader/set', {{method:'POST', headers:{{'Content-Type':'application/json'}}, body:JSON.stringify(getCfg())}});
  _chartDrawn = false;
  refreshChart();
}}

function startScanner() {{
  const cfg = getCfg();
  fetch('/eobtrader/start', {{method:'POST', headers:{{'Content-Type':'application/json'}}, body:JSON.stringify(cfg)}});
}}

function stopScanner() {{
  fetch('/eobtrader/stop', {{method:'POST'}});
}}

// ── Charts ────────────────────────────────────────────────────────────────────
async function refreshChart() {{
  const sym  = document.getElementById('cfg-symbol').value;
  const tf   = document.getElementById('cfg-tf').value;
  const bars = document.getElementById('cfg-bars').value || 500;
  const r    = await fetch(`/eobtrader/chart?symbol=${{sym}}&tf=${{tf}}&bars=${{bars}}`);
  const fig  = await r.json();
  if (!fig || !fig.data) return;
  if (_chartDrawn) {{
    Plotly.react('chart-div', fig.data, fig.layout);
  }} else {{
    Plotly.newPlot('chart-div', fig.data, fig.layout, {{responsive:true, displayModeBar:false}});
    _chartDrawn = true;
  }}
}}

async function refreshChart2() {{
  const sym  = document.getElementById('cfg-symbol').value;
  const tf   = document.getElementById('cfg-tf').value;
  const bars = document.getElementById('cfg-bars').value || 500;
  const r    = await fetch(`/eobtrader/chart?symbol=${{sym}}&tf=${{tf}}&bars=${{bars}}`);
  const fig  = await r.json();
  if (!fig || !fig.data) return;
  if (fig.layout) {{
    fig.layout.height = 280;
  }}
  if (_chart2Drawn) {{
    Plotly.react('chart-div2', fig.data, fig.layout);
  }} else {{
    Plotly.newPlot('chart-div2', fig.data, fig.layout, {{responsive:true, displayModeBar:false}});
    _chart2Drawn = true;
  }}
}}

// ── Status poll ───────────────────────────────────────────────────────────────
async function refreshAll() {{
  try {{
    const r = await fetch('/eobtrader/status');
    const d = await r.json();
    if (!d.ok) return;
    updateStatusBar(d.state);
    if (d.result)   updateAnalysis(d.result);
    if (d.decisions) updateDecisions(d.decisions);
    updatePendingTable(d.pending_orders || []);
    if (_activeTab === 'orders') refreshChart2();
  }} catch(e) {{ console.warn('poll err', e); }}
}}

function updateStatusBar(state) {{
  const running = state.running;
  document.getElementById('status-text').textContent = running ? 'Ruleaza' : 'Inactiv';
  document.getElementById('status-dot').className    = 'status-dot' + (running ? ' on' : '');
  const ts = state.last_scan ? new Date(state.last_scan).toLocaleTimeString('ro') : '';
  document.getElementById('last-scan-ts').textContent = ts ? `Scanat: ${{ts}}` : '';
  // Sync config if changed externally
  document.getElementById('cfg-symbol').value = state.symbol;
  document.getElementById('cfg-tf').value     = state.tf;
  document.getElementById('cfg-bars').value   = state.bars;
  document.getElementById('cfg-interval').value = state.interval;
  document.getElementById('cfg-conf').value   = state.min_confidence;
  document.getElementById('cfg-auto').checked = state.auto_execute;
}}

function updateAnalysis(res) {{
  const sig = (res.signal || 'HOLD');
  const el  = document.getElementById('an-signal');
  el.textContent = sig;
  el.className   = 'signal-badge ' + (sig === 'BUY' ? 'buy' : sig === 'SELL' ? 'sell' : sig.startsWith('PENDING') ? 'pending' : 'hold');
  document.getElementById('an-conf').textContent = (res.confidence ?? '—') + (res.confidence != null ? '%' : '');
  document.getElementById('an-risk').textContent = (res.risk_score  ?? '—') + (res.risk_score  != null ? '%' : '');
  const pe = res.pending_entry;
  document.getElementById('an-pending').textContent = pe
    ? `${{pe.order_type}} @ ${{pe.price}}  SL ${{pe.sl}}  TP ${{pe.tp}}  (${{pe.tf || '?'}})`
    : '—';
  const just = (res.justification || []).slice(0, 5).join(' | ');
  document.getElementById('an-just').textContent = just || '—';
}}

function updateDecisions(decs) {{
  const log = document.getElementById('dec-log');
  const SIG_COL = {{BUY:'#26a69a', SELL:'#ef5350', HOLD:'#3a4050'}};
  log.innerHTML = decs.map(d => {{
    const col = d.signal.startsWith('PENDING') ? '#9c27b0' : (SIG_COL[d.signal] || '#6070a0');
    return `<div class="dec-row">
      <span class="ts">${{d.ts}}</span>
      <span class="dsig" style="color:${{col}}">${{d.signal}}</span>
      <span class="dres">${{d.result}}</span>
    </div>`;
  }}).join('');
}}

function updatePendingTable(orders) {{
  const tbody = document.getElementById('pending-tbody');
  if (!orders.length) {{
    tbody.innerHTML = '<tr><td colspan="10" style="color:#3a4050;padding:16px;text-align:center">Niciun ordin pending EOB activ</td></tr>';
    return;
  }}
  tbody.innerHTML = orders.map(o => {{
    const clr      = o.is_buy ? '#26a69a' : '#ef5350';
    const typeIcon = o.is_buy ? '▲' : '▼';
    // expiry progress bar (% ramas din durata totala)
    const MAX_H = 120;
    const remH  = (o.rem_s || 0) / 3600;
    const pct   = o.expiry === 'GTC' ? 100 : Math.min(100, (remH / MAX_H) * 100);
    const barClr = pct > 50 ? '#26a69a' : pct > 20 ? '#fb8c00' : '#ef5350';
    const expStyle = o.expiry === 'expirat' ? 'color:#ef5350' : '';
    return `<tr>
      <td style="color:#3a4060;font-family:monospace">#${{o.ticket}}</td>
      <td style="font-weight:500;color:#c0c8e0">${{o.symbol}}</td>
      <td style="color:${{clr}};font-weight:600">${{typeIcon}} ${{o.type}}</td>
      <td style="text-align:right;color:${{clr}};font-family:monospace">${{o.price.toFixed(5)}}</td>
      <td style="text-align:right;color:#ef5350;font-family:monospace">${{o.sl ? o.sl.toFixed(5) : '—'}}</td>
      <td style="text-align:right;color:#26a69a;font-family:monospace">${{o.tp ? o.tp.toFixed(5) : '—'}}</td>
      <td style="text-align:right;color:#8090a0">${{o.volume}}</td>
      <td>
        <div style="${{expStyle}}">${{o.expiry_dt}}</div>
        <div class="exp-bar"><div class="exp-fill" style="width:${{pct.toFixed(1)}}%;background:${{barClr}}"></div></div>
      </td>
      <td style="font-weight:500;${{expStyle}}">${{o.expiry}}</td>
      <td><button class="btn-sm" onclick="cancelOrder(${{o.ticket}})">✕</button></td>
    </tr>`;
  }}).join('');
}}

async function cancelOrder(ticket) {{
  if (!confirm(`Anulezi ordinul pending #${{ticket}}?`)) return;
  const r = await fetch('/eobtrader/cancel_pending', {{
    method:'POST', headers:{{'Content-Type':'application/json'}},
    body: JSON.stringify({{ticket}}),
  }});
  const d = await r.json();
  alert(d.ok ? `Ordin #${{ticket}} anulat` : `Eroare: ${{d.message}}`);
  refreshAll();
  if (_activeTab === 'orders') refreshChart2();
}}

function refreshChart() {{
  const sym  = document.getElementById('cfg-symbol').value;
  const tf   = document.getElementById('cfg-tf').value;
  const bars = document.getElementById('cfg-bars').value || 500;
  fetch(`/eobtrader/chart?symbol=${{sym}}&tf=${{tf}}&bars=${{bars}}`)
    .then(r => r.json())
    .then(fig => {{
      if (!fig || !fig.data) return;
      if (_chartDrawn) Plotly.react('chart-div', fig.data, fig.layout);
      else {{ Plotly.newPlot('chart-div', fig.data, fig.layout, {{responsive:true,displayModeBar:false}}); _chartDrawn=true; }}
    }}).catch(()=>{{}});
}}

// ── Init ──────────────────────────────────────────────────────────────────────
refreshChart();
refreshAll();
_pollTimer = setInterval(refreshAll, 10000);   // poll la 10s
</script>
</body>
</html>"""
