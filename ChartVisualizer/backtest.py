"""
Walk-forward backtester pentru strategiile ChartVisualizer.
Foloseste exact interfata analyze() a fiecarei strategii, cu date
istorice MT5 injectate prin monkey-patch pe app.fetch.

Nu modifica nimic din codul de trading live.
"""
from __future__ import annotations

import json
import logging
import os
import threading
import uuid
from datetime import datetime, timedelta, timezone
from functools import lru_cache

import numpy as np
import pandas as pd
from flask import Blueprint, Response, request, session, redirect, url_for

log = logging.getLogger(__name__)

backtest_bp = Blueprint("backtest", __name__)

# ── Directoare ────────────────────────────────────────────────────────────────
_RESULTS_DIR = os.path.join(os.path.dirname(__file__), "backtest_results")
os.makedirs(_RESULTS_DIR, exist_ok=True)

# ── Stare globala jobs ─────────────────────────────────────────────────────────
_bt_jobs: dict[str, dict] = {}
_bt_lock = threading.Lock()

# ── Date mock (thread-local pentru rulari paralele viitoare) ───────────────────
_mock_data: dict[tuple, pd.DataFrame] = {}
_mock_end_idx: dict[tuple, int] = {}


# ══════════════════════════════════════════════════════════════════════════════
#  Date istorice
# ══════════════════════════════════════════════════════════════════════════════

def _tf_const(tf_str: str):
    """Converteste string TF ('H1') la constanta MT5."""
    import app as _app
    if _app.mt5 is None:
        return None
    _MAP = {
        "M1":  _app.mt5.TIMEFRAME_M1,
        "M5":  _app.mt5.TIMEFRAME_M5,
        "M15": _app.mt5.TIMEFRAME_M15,
        "M30": _app.mt5.TIMEFRAME_M30,
        "H1":  _app.mt5.TIMEFRAME_H1,
        "H4":  _app.mt5.TIMEFRAME_H4,
        "D1":  _app.mt5.TIMEFRAME_D1,
        "W1":  _app.mt5.TIMEFRAME_W1,
    }
    return _MAP.get(tf_str.upper())


# Dimensiunea chunk-ului per request MT5, in zile, in functie de TF
_CHUNK_DAYS: dict[str, int] = {
    "M1":  7,    # 1 saptamana per request
    "M5":  30,   # 1 luna per request
    "M15": 60,   # 2 luni per request
    "M30": 90,
    "H1":  180,
    "H4":  365,
    "D1":  365 * 2,
    "W1":  365 * 4,
}


def _fetch_history(symbol: str, tf: str, from_dt: datetime, to_dt: datetime,
                   update_fn=None) -> pd.DataFrame | None:
    """
    Fetch OHLCV din MT5 in chunk-uri mici (evita timeout / buffer gol).
    update_fn(msg) — callback optional pentru progress updates.
    """
    import app as _app
    import time as _time

    if not _app.MT5_AVAILABLE or _app.mt5 is None:
        return None
    tf_c = _tf_const(tf)
    if tf_c is None:
        return None

    # MT5 trebuie re-initializat in thread-uri noi
    try:
        if _app.mt5.terminal_info() is None:
            _app.mt5.initialize()
    except Exception:
        pass

    # Asigura ca simbolul e vizibil/selectat in MT5
    try:
        info = _app.mt5.symbol_info(symbol)
        if info and not info.visible:
            _app.mt5.symbol_select(symbol, True)
    except Exception:
        pass

    chunk_days = _CHUNK_DAYS.get(tf.upper(), 90)
    chunks: list[pd.DataFrame] = []

    cursor = from_dt
    total_days = (to_dt - from_dt).days or 1
    fetched_days = 0

    while cursor < to_dt:
        chunk_end = min(cursor + timedelta(days=chunk_days), to_dt)

        rates = _app.mt5.copy_rates_range(symbol, tf_c, cursor, chunk_end)
        if rates is not None and len(rates) > 0:
            df_chunk = pd.DataFrame(rates)
            df_chunk["time"] = pd.to_datetime(df_chunk["time"], unit="s")
            df_chunk = df_chunk.set_index("time")
            df_chunk = df_chunk.rename(columns={"tick_volume": "volume"})
            chunks.append(df_chunk[["open", "high", "low", "close", "volume"]])
            fetched_days += (chunk_end - cursor).days
            if update_fn:
                pct = min(99, int(fetched_days / total_days * 100))
                update_fn(f"Descarcare {symbol}/{tf}: {cursor.strftime('%Y-%m')} "
                          f"→ {chunk_end.strftime('%Y-%m')} ({pct}%)")
        else:
            log.debug(f"[Backtest] chunk gol {symbol}/{tf} {cursor.date()}→{chunk_end.date()}")

        cursor = chunk_end
        _time.sleep(0.05)  # pauza mica intre requesturi

    if not chunks:
        return None

    df = pd.concat(chunks).sort_index()
    df = df[~df.index.duplicated(keep="last")]  # elimina duplicate la granita chunk-urilor
    return df


# ══════════════════════════════════════════════════════════════════════════════
#  Mock fetch — injecteaza fereastra istorica in strategii
# ══════════════════════════════════════════════════════════════════════════════

def _make_mock_fetch(symbol: str):
    """Returneaza o functie fetch() care citeste din _mock_data la slice-ul curent."""
    sym = symbol.upper()

    def _fetch(s, tf, bars):
        key = (sym, tf.upper())
        full = _mock_data.get(key)
        if full is None:
            return None, "backtest: no data"
        end   = _mock_end_idx.get(key, len(full))
        start = max(0, end - int(bars))
        return full.iloc[start:end].copy(), "backtest"

    return _fetch


# ══════════════════════════════════════════════════════════════════════════════
#  Metrici
# ══════════════════════════════════════════════════════════════════════════════

def compute_metrics(trades: list[dict], risk_dollars: float,
                    initial_equity: float = 10_000.0) -> dict:
    if not trades:
        return {
            "total_trades": 0, "wins": 0, "losses": 0, "expires": 0, "timeouts": 0,
            "win_rate": 0.0, "profit_factor": 0.0,
            "net_pnl": 0.0, "gross_profit": 0.0, "gross_loss": 0.0,
            "max_drawdown": 0.0, "avg_rr": 0.0, "sharpe": 0.0,
            "equity_curve": [initial_equity], "equity_dates": [],
        }

    wins     = [t for t in trades if t["outcome"] == "TP"]
    losses   = [t for t in trades if t["outcome"] == "SL"]
    expires  = [t for t in trades if t["outcome"] == "EXPIRED"]
    timeouts = [t for t in trades if t["outcome"] == "TIMEOUT"]

    # Pierderile includ SL, expire-urile negative si timeout-urile negative
    neutral  = expires + timeouts
    gross_profit = sum(t["pnl"] for t in wins)
    gross_loss   = abs(sum(t["pnl"] for t in losses + neutral if t["pnl"] < 0))
    net_pnl      = sum(t["pnl"] for t in trades)

    win_rate = len(wins) / len(trades) * 100
    pf = (gross_profit / gross_loss) if gross_loss > 0 else (
        999.0 if gross_profit > 0 else 0.0)

    # Equity curve
    eq = [initial_equity]
    eq_dates = []
    for t in trades:
        eq.append(round(eq[-1] + t["pnl"], 2))
        eq_dates.append(t.get("exit_time", "")[:10])

    # Max drawdown
    peak, max_dd = eq[0], 0.0
    for e in eq:
        if e > peak:
            peak = e
        dd = peak - e
        if dd > max_dd:
            max_dd = dd

    # Avg R:R efectiv (doar trade-urile castigatoare)
    rrs    = [t["rr"] for t in wins if t.get("rr", 0) > 0]
    avg_rr = float(np.mean(rrs)) if rrs else 0.0

    # Sharpe anual (bazat pe PnL per trade)
    pnls = [t["pnl"] for t in trades]
    if len(pnls) > 1:
        mean_r = float(np.mean(pnls))
        std_r  = float(np.std(pnls))
        # ~252 trade-uri ca proxy pentru "zile de trading"
        n_year = 252
        sharpe = (mean_r / std_r * np.sqrt(n_year)) if std_r > 0 else 0.0
    else:
        sharpe = 0.0

    # Consecutive losses max
    max_consec_loss = 0
    cur_loss = 0
    for t in trades:
        if t["pnl"] < 0:
            cur_loss += 1
            max_consec_loss = max(max_consec_loss, cur_loss)
        else:
            cur_loss = 0

    return {
        "total_trades":     len(trades),
        "wins":             len(wins),
        "losses":           len(losses),
        "expires":          len(expires),
        "timeouts":         len(timeouts),
        "win_rate":         round(win_rate, 1),
        "profit_factor":    round(min(pf, 999.0), 2),
        "net_pnl":          round(net_pnl, 2),
        "gross_profit":     round(gross_profit, 2),
        "gross_loss":       round(gross_loss, 2),
        "max_drawdown":     round(max_dd, 2),
        "avg_rr":           round(avg_rr, 2),
        "sharpe":           round(sharpe, 2),
        "max_consec_loss":  max_consec_loss,
        "equity_curve":     eq,
        "equity_dates":     eq_dates,
    }


# ══════════════════════════════════════════════════════════════════════════════
#  Motor walk-forward
# ══════════════════════════════════════════════════════════════════════════════

def _simulate_trade(primary_df: pd.DataFrame, open_trade: dict,
                    from_bar: int, to_bar: int, max_bars: int = 99999) -> dict | None:
    """
    Verifica bara cu bara daca SL, TP sau durata maxima a fost atinsa.
    Returneaza trade-ul inchis sau None daca e inca deschis.
    """
    sig        = open_trade["signal"]
    sl         = open_trade["sl"]
    tp_price   = open_trade["tp"]
    entry      = open_trade["entry_price"]
    risk       = open_trade["risk"]
    entry_bar  = open_trade["entry_bar"]

    for i in range(from_bar, min(to_bar + 1, len(primary_df))):
        row = primary_df.iloc[i]
        low_, high_ = float(row["low"]), float(row["high"])
        duration = i - entry_bar

        # ── Durata maxima atinsa — inchide la close-ul barei ──────────────────
        if duration >= max_bars:
            last_p  = float(primary_df.iloc[i]["close"])
            sl_dist = abs(entry - sl)
            if sl_dist > 0:
                pnl = ((last_p - entry) / sl_dist * risk if sig == "BUY"
                       else (entry - last_p) / sl_dist * risk)
            else:
                pnl = 0.0
            return {**open_trade,
                    "exit_bar":      i,
                    "exit_time":     str(primary_df.index[i]),
                    "outcome":       "TIMEOUT",
                    "pnl":           round(pnl, 2),
                    "rr":            0.0,
                    "duration_bars": duration}

        if sig == "BUY":
            if low_ <= sl:
                sl_dist  = abs(entry - sl)
                tp_dist  = abs(tp_price - entry)
                rr_ratio = tp_dist / sl_dist if sl_dist > 0 else 0
                return {**open_trade,
                        "exit_bar":   i,
                        "exit_time":  str(primary_df.index[i]),
                        "outcome":    "SL",
                        "pnl":        round(-risk, 2),
                        "rr":         round(rr_ratio, 2),
                        "duration_bars": i - entry_bar}
            if high_ >= tp_price:
                sl_dist  = abs(entry - sl)
                tp_dist  = abs(tp_price - entry)
                rr_ratio = tp_dist / sl_dist if sl_dist > 0 else 0
                return {**open_trade,
                        "exit_bar":   i,
                        "exit_time":  str(primary_df.index[i]),
                        "outcome":    "TP",
                        "pnl":        round(risk * rr_ratio, 2),
                        "rr":         round(rr_ratio, 2),
                        "duration_bars": i - entry_bar}

        elif sig == "SELL":
            if high_ >= sl:
                sl_dist  = abs(sl - entry)
                tp_dist  = abs(entry - tp_price)
                rr_ratio = tp_dist / sl_dist if sl_dist > 0 else 0
                return {**open_trade,
                        "exit_bar":   i,
                        "exit_time":  str(primary_df.index[i]),
                        "outcome":    "SL",
                        "pnl":        round(-risk, 2),
                        "rr":         round(rr_ratio, 2),
                        "duration_bars": i - entry_bar}
            if low_ <= tp_price:
                sl_dist  = abs(sl - entry)
                tp_dist  = abs(entry - tp_price)
                rr_ratio = tp_dist / sl_dist if sl_dist > 0 else 0
                return {**open_trade,
                        "exit_bar":   i,
                        "exit_time":  str(primary_df.index[i]),
                        "outcome":    "TP",
                        "pnl":        round(risk * rr_ratio, 2),
                        "rr":         round(rr_ratio, 2),
                        "duration_bars": i - entry_bar}

    return None  # inca deschis


_TF_MIN_GLOBAL = {"M1":1,"M5":5,"M15":15,"M30":30,"H1":60,"H4":240,"D1":1440,"W1":10080}


def _process_one_bar(state: dict) -> bool:
    """
    Proceseaza UNA bara la indexul state['bar_idx'].
    Modifica state in place. Returneaza False daca nu mai sunt bare.

    Folosit atat in run_backtest_job (auto-run) cat si in replay mode.
    """
    import app as _app
    import strategies as _sp

    bar_idx     = state["bar_idx"]
    primary_df  = state["primary_df"]
    if bar_idx >= len(primary_df):
        return False

    # ── Phase 1: Verifica trade deschis pentru SL/TP ────────────────────────
    if state["open_trade"] is not None:
        closed = _simulate_trade(
            primary_df, state["open_trade"],
            state["last_check_bar"] + 1, bar_idx,
            state["max_dur_bars"]
        )
        if closed is not None:
            state["trades"].append(closed)
            state["open_trade"] = None
        else:
            state["open_trade"]["entry_bar_last"] = bar_idx
        state["last_check_bar"] = bar_idx
        if state["open_trade"] is not None:
            return True   # inca in trade — fara analiza noua

    # ── Phase 2: Update TF cutoffs ──────────────────────────────────────────
    primary_ts       = primary_df.index[bar_idx]
    primary_close_ts = primary_ts + pd.Timedelta(minutes=state["primary_tf_min"])

    for tf in state["all_tfs"]:
        full_df = _mock_data.get((state["symbol"], tf))
        if full_df is None:
            continue
        tf_min_h  = _TF_MIN_GLOBAL.get(tf.upper(), 60)
        cutoff_ts = primary_close_ts - pd.Timedelta(minutes=tf_min_h)
        idx = int(full_df.index.searchsorted(cutoff_ts, side="right"))
        _mock_end_idx[(state["symbol"], tf)] = max(0, idx)

    _sp.BT_BAR_UTC = (primary_close_ts.to_pydatetime()
                      if hasattr(primary_close_ts, "to_pydatetime")
                      else primary_close_ts)

    # ── Phase 3: Ruleaza strategia ──────────────────────────────────────────
    try:
        res = state["strat"].analyze(
            state["symbol"], state["tfs"], state["bars_window"],
            tf_bars=None, elements=None,
            min_confidence=state["min_confidence"],
            use_h4_filter=False,
            use_session_filter=False,
        )
    except Exception as exc:
        log.debug(f"[BT replay] analyze {state['symbol']} bar={bar_idx}: {exc}")
        return True

    if res.get("signal") not in ("BUY", "SELL"):
        return True

    if bar_idx + 1 >= len(primary_df):
        return True

    bft    = res.get("best_tf") or {}
    sl     = bft.get("sl")
    tp_val = bft.get("tp")
    if sl is None:
        return True

    # ── Phase 4: Deschide trade ─────────────────────────────────────────────
    raw_entry = float(primary_df.iloc[bar_idx + 1]["open"])
    if res["signal"] == "BUY":
        entry_price = raw_entry + state["spread"]
    else:
        entry_price = raw_entry - state["spread"]

    sl_dist = abs(entry_price - float(sl))
    if sl_dist <= 0:
        return True

    if res["signal"] == "BUY":
        if tp_val is None or float(tp_val) <= entry_price:
            tp_val = entry_price + sl_dist * state["tp_ratio"]
    else:
        if tp_val is None or float(tp_val) >= entry_price:
            tp_val = entry_price - sl_dist * state["tp_ratio"]

    entry_bar_real = bar_idx + 1
    # Pastram justificarea — de ce a luat decizia (lista de reasons din analyze)
    reasons = res.get("justification") or []
    if not reasons and res.get("best_tf"):
        reasons = res["best_tf"].get("reasons", [])

    state["open_trade"] = {
        "signal":      res["signal"],
        "sl":          float(sl),
        "tp":          float(tp_val),
        "entry_price": entry_price,
        "entry_bar":   entry_bar_real,
        "entry_time":  str(primary_df.index[entry_bar_real]),
        "risk":        state["risk_dollars"],
        "confidence":  res.get("confidence", 0),
        "tf":          state["primary_tf"],
        "strategy":    state["strat_key"],
        "reasons":     list(reasons),       # ← motivele intrarii
        "exit_bar":    None, "exit_time":  None,
        "outcome":     None, "pnl":        None,
        "rr":          None, "duration_bars": None,
    }
    state["last_check_bar"] = entry_bar_real - 1
    return True


def _build_replay_snapshot(state: dict, bars_visible: int = 150) -> dict:
    """Construieste snapshot-ul live din state (pentru replay endpoints)."""
    primary_df = state["primary_df"]
    bar_idx    = state["bar_idx"]
    start_idx  = max(0, bar_idx - bars_visible + 1)
    end_idx    = bar_idx + 1
    sub = primary_df.iloc[start_idx:end_idx]

    bars_payload = [
        {"t": str(idx), "o": float(r["open"]), "h": float(r["high"]),
         "l": float(r["low"]), "c": float(r["close"])}
        for idx, r in sub.iterrows()
    ]

    trades_list = state["trades"]
    wins   = sum(1 for t in trades_list if t["outcome"] == "TP")
    losses = sum(1 for t in trades_list if t["outcome"] == "SL")
    equity = sum(t["pnl"] for t in trades_list if t.get("pnl") is not None)

    recent_trades = [
        {"signal": t.get("signal"), "entry_time": t.get("entry_time"),
         "exit_time": t.get("exit_time"), "entry_price": t.get("entry_price"),
         "sl": t.get("sl"), "tp": t.get("tp"),
         "outcome": t.get("outcome"), "pnl": t.get("pnl"),
         "rr": t.get("rr"), "duration_bars": t.get("duration_bars"),
         "confidence": t.get("confidence"),
         "reasons": t.get("reasons", [])}
        for t in trades_list[-200:]
    ]

    open_payload = None
    if state["open_trade"] is not None:
        ot = state["open_trade"]
        open_payload = {
            "signal": ot.get("signal"), "entry_time": ot.get("entry_time"),
            "entry_price": ot.get("entry_price"),
            "sl": ot.get("sl"), "tp": ot.get("tp"),
            "confidence": ot.get("confidence"),
            "reasons": ot.get("reasons", []),
        }

    return {
        "bar_idx":    bar_idx,
        "bar_time":   str(primary_df.index[bar_idx]),
        "primary_tf": state["primary_tf"],
        "total_bars": state["total_bars"],
        "bars":       bars_payload,
        "open_trade": open_payload,
        "trades":     recent_trades,
        "metrics": {
            "total":  len(trades_list),
            "wins":   wins,
            "losses": losses,
            "wr":     round(wins / max(1, len(trades_list)) * 100, 1) if trades_list else 0.0,
            "equity": round(equity, 2),
            "risk":   state["risk_dollars"],
        },
    }


def _replay_init_job(job_id: str, params: dict):
    """
    Thread: incarca date istorice, pregateste state pentru replay,
    apoi iese (state ramine in _bt_jobs pana la cancel/cleanup).
    """
    import app as _app
    import strategies as _sp

    symbol             = params["symbol"].upper()
    strat_key          = params["strategy"]
    tfs                = params["tfs"]
    bars_window        = int(params.get("bars_window", 200))
    duration_years     = float(params.get("duration_years", 0.5))
    risk_dollars       = float(params.get("risk_dollars", 50.0))
    tp_ratio           = float(params.get("tp_ratio", 1.5))
    min_confidence     = float(params.get("min_confidence", 60.0))
    spread             = float(params.get("spread", 0.0003))
    max_duration_hours = float(params.get("max_duration_hours", 1.0))

    def _update(status, progress=None, message=None):
        with _bt_lock:
            _bt_jobs[job_id]["status"] = status
            if progress is not None:
                _bt_jobs[job_id]["progress"] = progress
            if message:
                _bt_jobs[job_id]["message"] = message

    try:
        _update("loading", 0, "Replay: incarcare date...")

        to_dt   = datetime.now(timezone.utc)
        from_dt = to_dt - timedelta(days=int(duration_years * 365))
        all_tfs = list(dict.fromkeys(tfs))

        for tf in all_tfs:
            df = _fetch_history(symbol, tf, from_dt, to_dt,
                                update_fn=lambda m: _update("loading", None, m))
            if df is None or len(df) < 100:
                _update("error", message=f"MT5 nu a returnat date pentru {symbol}/{tf}")
                return
            _mock_data[(symbol, tf)] = df

        primary_tf  = tfs[0]
        primary_df  = _mock_data[(symbol, primary_tf)]
        total_bars  = len(primary_df)
        _tf_min     = _TF_MIN_GLOBAL.get(primary_tf.upper(), 60)
        max_dur_bars = int(max_duration_hours * 60 / _tf_min) if max_duration_hours > 0 else 99999

        strat = _sp.get_strategy(strat_key)
        if strat is None:
            _update("error", message=f"Strategie necunoscuta: {strat_key}")
            return

        # Activeaza mock_fetch global (replay-ul foloseste analyze() prin app.fetch)
        _app.fetch    = _make_mock_fetch(symbol)
        _app.TP_RATIO = tp_ratio

        with _bt_lock:
            _bt_jobs[job_id].update({
                "status":          "ready",
                "mode":            "replay",
                "primary_df":      primary_df,
                "all_tfs":         all_tfs,
                "primary_tf":      primary_tf,
                "primary_tf_min":  _tf_min,
                "bar_idx":         bars_window,
                "last_check_bar":  bars_window - 1,
                "open_trade":      None,
                "trades":          [],
                "max_dur_bars":    max_dur_bars,
                "strat_key":       strat_key,
                "strat":           strat,
                "spread":          spread,
                "risk_dollars":    risk_dollars,
                "tp_ratio":        tp_ratio,
                "min_confidence":  min_confidence,
                "tfs":             tfs,
                "bars_window":     bars_window,
                "symbol":          symbol,
                "total_bars":      total_bars,
                "progress":        100,
                "message":         f"Gata: {total_bars} bare {primary_tf}, start la bara {bars_window}",
            })
            _bt_jobs[job_id]["live"] = _build_replay_snapshot(_bt_jobs[job_id])

    except Exception as exc:
        log.exception(f"[BT replay init] {job_id}: {exc}")
        _update("error", message=f"Eroare init replay: {exc}")


def _replay_advance(job_id: str, n_bars: int = 1, mode: str = "fixed") -> dict | None:
    """
    Avanseaza replay-ul.
      mode='fixed'        → exact n_bars
      mode='until_trade'  → pana se deschide un trade nou
      mode='until_close'  → pana se inchide trade-ul curent
    Returneaza snapshot dupa avansare.
    """
    with _bt_lock:
        job = _bt_jobs.get(job_id)
        if not job or job.get("status") != "ready":
            return None
        if job.get("mode") != "replay":
            return None

    advanced = 0
    max_bars = 50000  # safety cap

    if mode == "until_close":
        if job["open_trade"] is None:
            with _bt_lock:
                job["live"] = _build_replay_snapshot(job)
            return job["live"]
        initial_count = len(job["trades"])

    elif mode == "until_trade":
        if job["open_trade"] is not None:
            # E deja un trade deschis → avanseaza 1 bara
            n_bars, mode = 1, "fixed"

    while advanced < (n_bars if mode == "fixed" else max_bars):
        if job["bar_idx"] >= job["total_bars"] - 1:
            break
        more = _process_one_bar(job)
        if not more:
            break
        job["bar_idx"] += 1
        advanced += 1

        if mode == "until_trade" and job["open_trade"] is not None:
            break
        if mode == "until_close" and len(job["trades"]) > initial_count:
            break

    with _bt_lock:
        job["live"] = _build_replay_snapshot(job)
        if job["bar_idx"] >= job["total_bars"] - 1:
            job["status"] = "done"
            job["message"] = f"Replay terminat: {len(job['trades'])} trade-uri"

    return job["live"]


def run_backtest_job(job_id: str, params: dict):
    """
    Ruleaza backtestul intr-un thread separat.
    La fiecare bara primary_tf, actualizeaza slice-urile tuturor TF-urilor si
    apeleaza strat.analyze() cu fereastra corecta.
    """
    import app as _app
    import strategies as _sp

    symbol             = params["symbol"].upper()
    strat_key          = params["strategy"]
    tfs                = params["tfs"]
    bars_window        = int(params.get("bars_window", 500))
    duration_years     = float(params.get("duration_years", 2.0))
    step               = int(params.get("step", 4))
    risk_dollars       = float(params.get("risk_dollars", 50.0))
    tp_ratio           = float(params.get("tp_ratio", 2.0))
    min_confidence     = float(params.get("min_confidence", 66.0))
    spread             = float(params.get("spread", 0.0003))        # spread in price units
    max_duration_hours = float(params.get("max_duration_hours", 10.0))  # ore max per trade
    elements_override  = params.get("elements") or None  # {key: bool} pasati la analyze()

    def _update(status: str, progress: int = None, message: str = None):
        with _bt_lock:
            _bt_jobs[job_id]["status"] = status
            if progress is not None:
                _bt_jobs[job_id]["progress"] = progress
            if message:
                _bt_jobs[job_id]["message"] = message

    def _publish_live(primary_df, bar_idx, trades_list, open_trade,
                      risk_dollars, primary_tf, bars_visible: int = 150):
        """
        Publica snapshot live al backtestului in job state.
        - bare recente OHLC (ultimele ~150)
        - trade deschis (cu entry/sl/tp)
        - trade-uri inchise pana acum (esential)
        - metrici curente (wins/losses/wr/equity)
        Snapshot e mic — JSON ~30-80 KB pentru 150 bare + 200 trade-uri.
        """
        try:
            start_idx = max(0, bar_idx - bars_visible + 1)
            end_idx   = bar_idx + 1
            sub = primary_df.iloc[start_idx:end_idx]
            bars_payload = [
                {
                    "t": str(idx),
                    "o": float(row["open"]),
                    "h": float(row["high"]),
                    "l": float(row["low"]),
                    "c": float(row["close"]),
                }
                for idx, row in sub.iterrows()
            ]

            wins   = sum(1 for t in trades_list if t["outcome"] == "TP")
            losses = sum(1 for t in trades_list if t["outcome"] == "SL")
            equity = sum(t["pnl"] for t in trades_list if t.get("pnl") is not None)

            # Trade-uri esentiale (max ultimele 200, fiecare cu doar field-urile relevante)
            recent_trades = [
                {
                    "signal":     t.get("signal"),
                    "entry_time": t.get("entry_time"),
                    "exit_time":  t.get("exit_time"),
                    "entry_price": t.get("entry_price"),
                    "sl":         t.get("sl"),
                    "tp":         t.get("tp"),
                    "outcome":    t.get("outcome"),
                    "pnl":        t.get("pnl"),
                    "rr":         t.get("rr"),
                    "duration_bars": t.get("duration_bars"),
                    "confidence": t.get("confidence"),
                    "reasons":    t.get("reasons", []),
                }
                for t in trades_list[-200:]
            ]

            open_payload = None
            if open_trade is not None:
                open_payload = {
                    "signal":      open_trade.get("signal"),
                    "entry_time":  open_trade.get("entry_time"),
                    "entry_price": open_trade.get("entry_price"),
                    "sl":          open_trade.get("sl"),
                    "tp":          open_trade.get("tp"),
                    "confidence":  open_trade.get("confidence"),
                    "reasons":     open_trade.get("reasons", []),
                }

            snapshot = {
                "bar_idx":     bar_idx,
                "bar_time":    str(primary_df.index[bar_idx]),
                "primary_tf":  primary_tf,
                "bars":        bars_payload,
                "open_trade":  open_payload,
                "trades":      recent_trades,
                "metrics": {
                    "total":   len(trades_list),
                    "wins":    wins,
                    "losses":  losses,
                    "wr":      round(wins / max(1, len(trades_list)) * 100, 1) if trades_list else 0.0,
                    "equity":  round(equity, 2),
                    "risk":    risk_dollars,
                },
            }
            with _bt_lock:
                _bt_jobs[job_id]["live"] = snapshot
        except Exception as exc:
            log.debug(f"[BT live] publish error: {exc}")

    try:
        _update("running", 0, "Incarcare date istorice din MT5...")

        # ── 1. Fetch date istorice ────────────────────────────────────────────
        to_dt   = datetime.now(timezone.utc)
        from_dt = to_dt - timedelta(days=int(duration_years * 365))

        all_tfs = list(dict.fromkeys(tfs))  # dedup, pastram ordinea

        min_required = 100
        for tf in all_tfs:
            def _prog(msg, _tf=tf):
                _update("running", None, msg)
            df = _fetch_history(symbol, tf, from_dt, to_dt, update_fn=_prog)
            if df is None or len(df) < min_required:
                err = (f"MT5 nu a returnat date pentru {symbol}/{tf}.\n\n"
                       f"Solutii:\n"
                       f"• Foloseste TF mai mare (H1, H4) pentru durate lungi\n"
                       f"• Reduce durata la 6 luni sau 1 an\n"
                       f"• In MT5: selecteaza simbolul si da scroll inapoi pe graficul {tf} "
                       f"ca sa fortezi descarcarea istoricului\n"
                       f"• Incearca din nou dupa cateva secunde")
                _update("error", message=err)
                return
            if len(df) < bars_window + 10:
                new_window = max(min_required, len(df) - 10)
                log.warning(f"[Backtest] {symbol}/{tf}: doar {len(df)} bare — fereastra redusa {bars_window}→{new_window}")
                bars_window = min(bars_window, new_window)
            _mock_data[(symbol, tf)] = df
            log.info(f"[Backtest] {symbol}/{tf}: {len(df)} bare ({from_dt.date()} → {to_dt.date()})")

        primary_tf  = tfs[0]
        primary_df  = _mock_data[(symbol, primary_tf)]
        total_bars  = len(primary_df)

        # Conversie durata maxima ore → bare pentru TF primar
        _TF_MIN = {"M1":1,"M5":5,"M15":15,"M30":30,"H1":60,"H4":240,"D1":1440,"W1":10080}
        _tf_min = _TF_MIN.get(primary_tf.upper(), 60)
        max_dur_bars = int(max_duration_hours * 60 / _tf_min) if max_duration_hours > 0 else 99999

        _update("running", 5, f"Date: {total_bars} bare {primary_tf} ({from_dt.date()} → {to_dt.date()})"
                f" | spread={spread} | max={max_duration_hours}h ({max_dur_bars} bare)")

        # ── 2. Strategie ──────────────────────────────────────────────────────
        strat = _sp.get_strategy(strat_key)
        if strat is None:
            _update("error", message=f"Strategie necunoscuta: {strat_key}")
            return

        # ── 3. Setup mock & walk-forward ──────────────────────────────────────
        original_fetch = _app.fetch
        original_tp    = _app.TP_RATIO
        _app.TP_RATIO  = tp_ratio
        mock_fetch     = _make_mock_fetch(symbol)

        trades: list[dict]  = []
        open_trade: dict | None = None
        last_check_bar      = 0

        scan_positions = list(range(bars_window, total_bars, step))
        total_scans    = len(scan_positions)

        try:
            _app.fetch = mock_fetch

            # Frecventa publicare snapshot live (la fiecare N scan-uri)
            live_pub_every = max(1, total_scans // 200)  # ~200 update-uri pe parcurs

            for scan_i, bar_idx in enumerate(scan_positions):

                # Progress update ~20 ori pe parcurs
                if scan_i % max(1, total_scans // 20) == 0:
                    pct = 5 + int(scan_i / total_scans * 85)
                    _update("running", pct,
                            f"Bara {bar_idx}/{total_bars} | Trade-uri: {len(trades)}")

                # ── Simulare trade deschis: verifica SL/TP pana la bara curenta ──
                if open_trade is not None:
                    closed = _simulate_trade(primary_df, open_trade,
                                             last_check_bar + 1, bar_idx, max_dur_bars)
                    if closed is not None:
                        trades.append(closed)
                        open_trade = None
                        # Publish imediat la inchiderea unui trade (event)
                        _publish_live(primary_df, bar_idx, trades, open_trade,
                                      risk_dollars, primary_tf)
                    else:
                        open_trade["entry_bar_last"] = bar_idx
                    last_check_bar = bar_idx

                    if open_trade is not None:
                        # Snapshot periodic chiar si cand trade-ul e deschis
                        if scan_i % live_pub_every == 0:
                            _publish_live(primary_df, bar_idx, trades, open_trade,
                                          risk_dollars, primary_tf)
                        # Trade inca deschis → nu scana semnal nou
                        continue

                # Snapshot periodic
                if scan_i % live_pub_every == 0:
                    _publish_live(primary_df, bar_idx, trades, open_trade,
                                  risk_dollars, primary_tf)

                # ── Actualizeaza slice-urile TF auxiliare ──────────────────────
                # primary_ts e OPEN-ul barii primare curente.
                # La acest moment, bara primara s-a inchis → timpul curent =
                #   primary_close = primary_ts + primary_tf_min.
                # Pentru TF mai mari, includem DOAR barele deja inchise la acel
                # moment: tf_bar_open + tf_min <= primary_close.
                # Asta inseamna ca, pe orice TF, ULTIMA bara vizibila e cea mai
                # recent INCHISA — fara bare partiale, fara look-ahead.
                primary_ts       = primary_df.index[bar_idx]
                primary_close_ts = primary_ts + pd.Timedelta(minutes=_tf_min)

                for tf in all_tfs:
                    full_df = _mock_data.get((symbol, tf))
                    if full_df is None:
                        continue
                    tf_min_h = _TF_MIN.get(tf.upper(), 60)
                    cutoff_ts = primary_close_ts - pd.Timedelta(minutes=tf_min_h)
                    idx = int(full_df.index.searchsorted(cutoff_ts, side="right"))
                    _mock_end_idx[(symbol, tf)] = max(0, idx)

                # Injecteaza timpul curent (close-ul barii primare) pentru session checks
                _sp.BT_BAR_UTC = (primary_close_ts.to_pydatetime()
                                  if hasattr(primary_close_ts, "to_pydatetime")
                                  else primary_close_ts)

                # ── Analiza ────────────────────────────────────────────────────
                try:
                    res = strat.analyze(
                        symbol, tfs, bars_window,
                        tf_bars=None, elements=elements_override,
                        min_confidence=min_confidence,
                        use_h4_filter=False,
                        use_session_filter=False,
                    )
                except Exception as exc:
                    log.debug(f"[BT] analyze {symbol}/{primary_tf} bar={bar_idx}: {exc}")
                    continue

                if res.get("signal") not in ("BUY", "SELL"):
                    continue

                # ── Aplica opt_* filters (consistent cu scanner-ul) ──────────
                if elements_override:
                    try:
                        opt_keys = {k[4:] for k, v in elements_override.items()
                                    if v and k.startswith("opt_")}
                        if opt_keys:
                            from strategies.common_filters import apply_all as _opt_apply
                            best_tf_info = res.get("best_tf") or {}
                            use_tf = (best_tf_info.get("tf")
                                      if isinstance(best_tf_info, dict) else tfs[0])
                            # Pentru backtest folosim primary_df (deja istoric, mock)
                            df_opt = primary_df.iloc[max(0, bar_idx-200):bar_idx+1].copy()
                            if len(df_opt) >= 50:
                                df_opt.columns = [c.lower() for c in df_opt.columns]
                                ok, _reasons = _opt_apply(df_opt, res["signal"], opt_keys)
                                if not ok:
                                    continue  # filtru a respins semnalul
                    except Exception as _ef:
                        log.debug(f"[BT] opt filters: {_ef}")

                # Nu mai avem o bara urmatoare — nu putem intra
                if bar_idx + 1 >= len(primary_df):
                    continue

                bft    = res.get("best_tf") or {}
                sl     = bft.get("sl")
                tp_val = bft.get("tp")

                if sl is None:
                    continue

                # ── Intrare realista: open-ul barei URMATOARE + spread ─────────
                # (in live trading ordinul se executa la open-ul barei N+1, nu la close-ul N)
                raw_entry = float(primary_df.iloc[bar_idx + 1]["open"])
                if res["signal"] == "BUY":
                    entry_price = raw_entry + spread   # cumpar la ask
                else:
                    entry_price = raw_entry - spread   # vand la bid

                sl_dist = abs(entry_price - float(sl))
                if sl_dist <= 0:
                    continue

                # Recalculeaza TP cu tp_ratio daca e invalid sau nu mai e valid dupa spread
                if res["signal"] == "BUY":
                    if tp_val is None or float(tp_val) <= entry_price:
                        tp_val = entry_price + sl_dist * tp_ratio
                else:
                    if tp_val is None or float(tp_val) >= entry_price:
                        tp_val = entry_price - sl_dist * tp_ratio

                entry_bar_real = bar_idx + 1  # bara la care s-a executat efectiv
                # Pastram motivele intrarii pentru transparenta in UI
                reasons_entry = res.get("justification") or []
                if not reasons_entry and res.get("best_tf"):
                    reasons_entry = res["best_tf"].get("reasons", [])
                open_trade = {
                    "signal":      res["signal"],
                    "sl":          float(sl),
                    "tp":          float(tp_val),
                    "entry_price": entry_price,
                    "entry_bar":   entry_bar_real,
                    "entry_time":  str(primary_df.index[entry_bar_real]),
                    "risk":        risk_dollars,
                    "confidence":  res.get("confidence", 0),
                    "tf":          primary_tf,
                    "strategy":    strat_key,
                    "reasons":     list(reasons_entry),
                    "exit_bar":    None,
                    "exit_time":   None,
                    "outcome":     None,
                    "pnl":         None,
                    "rr":          None,
                    "duration_bars": None,
                }
                # next call _simulate_trade va incepe de la (last_check_bar+1) = entry_bar_real
                # (verificam si bara de intrare — high/low de pe ea pot atinge SL/TP)
                last_check_bar = entry_bar_real - 1

                # Publish snapshot la deschiderea trade-ului
                _publish_live(primary_df, bar_idx, trades, open_trade,
                              risk_dollars, primary_tf)

        finally:
            _app.fetch    = original_fetch
            _app.TP_RATIO = original_tp
            _sp.BT_BAR_UTC = None  # reseteaza injectia de timp
            # Curata mock data pentru acest simbol
            for k in [k for k in _mock_data if k[0] == symbol]:
                del _mock_data[k]
            for k in [k for k in _mock_end_idx if k[0] == symbol]:
                del _mock_end_idx[k]

        # ── Trade deschis la sfarsit → EXPIRED ────────────────────────────────
        if open_trade is not None:
            last_price = float(primary_df.iloc[-1]["close"])
            entry      = open_trade["entry_price"]
            sl_d       = abs(entry - open_trade["sl"])
            if open_trade["signal"] == "BUY":
                raw_pnl = (last_price - entry) / sl_d * risk_dollars if sl_d > 0 else 0
            else:
                raw_pnl = (entry - last_price) / sl_d * risk_dollars if sl_d > 0 else 0
            trades.append({**open_trade,
                            "exit_bar":      len(primary_df) - 1,
                            "exit_time":     str(primary_df.index[-1]),
                            "outcome":       "EXPIRED",
                            "pnl":           round(raw_pnl, 2),
                            "rr":            0.0,
                            "duration_bars": len(primary_df) - 1 - open_trade["entry_bar"]})

        # ── 4. Metrici ────────────────────────────────────────────────────────
        _update("running", 92, f"Calculare metrici ({len(trades)} trade-uri)...")
        metrics = compute_metrics(trades, risk_dollars)

        result = {
            "job_id":     job_id,
            "params":     params,
            "symbol":     symbol,
            "strategy":   strat_key,
            "tfs":        tfs,
            "from_dt":    str(from_dt.date()),
            "to_dt":      str(to_dt.date()),
            "total_bars": total_bars,
            "step":       step,
            "trades":     trades,
            "metrics":    metrics,
            "created_at": datetime.now().isoformat(),
        }

        # Salveaza pe disc
        result_path = os.path.join(_RESULTS_DIR, f"{job_id}.json")
        with open(result_path, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2, default=str)

        with _bt_lock:
            _bt_jobs[job_id].update({
                "status":   "done",
                "progress": 100,
                "result":   result,
                "message":  (f"Complet: {len(trades)} trade-uri | "
                             f"WR {metrics['win_rate']}% | "
                             f"PF {metrics['profit_factor']} | "
                             f"Net {metrics['net_pnl']:+.0f}$"),
            })
        log.info(f"[Backtest] {job_id} done: trades={len(trades)}, "
                 f"WR={metrics['win_rate']}%, PF={metrics['profit_factor']}, "
                 f"net={metrics['net_pnl']:.2f}$")

    except Exception as exc:
        import traceback
        log.error(f"[Backtest] {job_id} EROARE: {exc}\n{traceback.format_exc()}")
        _update("error", message=str(exc))


# ══════════════════════════════════════════════════════════════════════════════
#  Flask routes
# ══════════════════════════════════════════════════════════════════════════════

def _login_required(f):
    from functools import wraps
    @wraps(f)
    def wrapper(*a, **kw):
        # Bypass auth pentru localhost direct (consistent cu app.py)
        try:
            from app import _is_direct_localhost
            if _is_direct_localhost():
                return f(*a, **kw)
        except Exception:
            pass
        if not session.get("logged_in"):
            return redirect(url_for("login"))
        return f(*a, **kw)
    return wrapper


@backtest_bp.route("/backtest_v2")
@backtest_bp.route("/backtest")
@_login_required
def backtest_v2_page():
    """UI minimalist v2 — sidebar + tabs (setup/results/element_impact)."""
    import os as _os
    tpl = _os.path.join(_os.path.dirname(__file__), "templates", "backtest_v2.html")
    with open(tpl, encoding="utf-8") as f:
        return Response(f.read(), content_type="text/html; charset=utf-8")


@backtest_bp.route("/backtest_legacy")
@_login_required
def backtest_page():
    import strategies as _sp
    import app as _app
    strats = [{"key": s.key, "name": s.name, "icon": s.icon, "color": s.color,
               "default_tfs":      s.default_tfs,
               "default_bars":     s.default_bars,
               "default_step":     getattr(s, "default_step", 4),
               "default_duration": getattr(s, "default_duration_years", 1.0),
               "default_max_hours": getattr(s, "default_max_hours", 10.0),
               "default_min_conf": s.strategy_defaults.get("min_confidence", 66.0)}
              for s in _sp.list_all()]

    # Ultimele rezultate salvate
    saved = []
    for fn in sorted(os.listdir(_RESULTS_DIR), reverse=True)[:10]:
        if fn.endswith(".json"):
            try:
                with open(os.path.join(_RESULTS_DIR, fn), encoding="utf-8") as f:
                    d = json.load(f)
                m = d.get("metrics", {})
                saved.append({
                    "job_id":   d.get("job_id", fn[:-5]),
                    "label":    f"{d.get('symbol','?')} {d.get('strategy','?')} {','.join(d.get('tfs',[]))}",
                    "from_dt":  d.get("from_dt", ""),
                    "to_dt":    d.get("to_dt", ""),
                    "trades":   m.get("total_trades", 0),
                    "win_rate": m.get("win_rate", 0),
                    "pf":       m.get("profit_factor", 0),
                    "net_pnl":  m.get("net_pnl", 0),
                })
            except Exception:
                pass

    html = _build_page(strats, saved)
    resp = Response(html, content_type="text/html; charset=utf-8")
    # Anti-cache: forteaza browser-ul sa ia mereu versiunea proaspata din server
    resp.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
    resp.headers["Pragma"]        = "no-cache"
    resp.headers["Expires"]       = "0"
    return resp


def _run_signal_log_job(job_id: str, params: dict):
    """
    Mod 'Signal Log': pentru fiecare bara din istoricul descarcat, ruleaza
    strat.analyze() pe fereastra glisanta de `bars_window` bare si salveaza
    rezultatul (signal + confidence + reasoning). Apoi valideaza fiecare semnal
    BUY/SELL pe urmatoarele `eval_bars` bare (high/low atinse vs entry).

    NU deschide trades — doar inregistreaza si valideaza statistic.
    """
    import app as _app
    import strategies as _sp

    def _update(status: str, progress: int = None, message: str = None):
        with _bt_lock:
            _bt_jobs[job_id]["status"] = status
            if progress is not None:
                _bt_jobs[job_id]["progress"] = progress
            if message:
                _bt_jobs[job_id]["message"] = message

    try:
        symbol           = params["symbol"].upper()
        strat_key        = params["strategy"]
        tfs              = params["tfs"]
        bars_window      = int(params.get("bars_window", 500))
        duration_years   = float(params.get("duration_years", 1.0))
        eval_bars        = int(params.get("eval_bars", 24))     # bare pt validare
        min_confidence   = float(params.get("min_confidence", 60.0))
        elements_override= params.get("elements") or None
        max_signals      = int(params.get("max_signals", 5000))

        strat = _sp.get_strategy(strat_key)
        if strat is None:
            _update("error", message=f"Strategie necunoscuta: {strat_key}")
            return

        primary_tf = tfs[0]
        _update("running", 5, "Descarcare istoric...")

        now = datetime.now(timezone.utc)
        from_dt = now - timedelta(days=int(duration_years * 365))
        all_dfs: dict[str, pd.DataFrame] = {}
        for tf in tfs:
            df = _fetch_history(symbol, tf, from_dt, now, force_full=True)
            if df is None or df.empty:
                _update("error", message=f"Fara date pentru {symbol}/{tf}")
                return
            all_dfs[tf] = df

        primary_df = all_dfs[primary_tf]
        n_total = len(primary_df)
        if n_total <= bars_window + eval_bars:
            _update("error", message=f"Istoric insuficient ({n_total} bare < {bars_window+eval_bars})")
            return

        _update("running", 10, f"Stream {n_total - bars_window - eval_bars} bare...")

        # Monkey-patch fetch (la fel ca in run_backtest_job)
        global _mock_data, _mock_end_idx
        _mock_data = {(symbol, tf): df for tf, df in all_dfs.items()}
        _mock_end_idx = {(symbol, tf): 0 for tf in tfs}
        original_fetch = _app.fetch
        _app.fetch = _make_mock_fetch(symbol)

        signals = []
        try:
            start_bar = bars_window
            end_bar   = n_total - eval_bars
            for bar_idx in range(start_bar, end_bar):
                if len(signals) >= max_signals:
                    break
                # Sync end_idx pentru toate TFs
                bar_ts = primary_df.index[bar_idx]
                for tf in tfs:
                    df_tf = all_dfs[tf]
                    pos = df_tf.index.searchsorted(bar_ts, side="right")
                    _mock_end_idx[(symbol, tf)] = int(pos)

                try:
                    res = strat.analyze(
                        symbol, tfs, bars_window,
                        tf_bars=None, elements=elements_override,
                        min_confidence=min_confidence,
                        use_h4_filter=False, use_session_filter=False,
                    )
                except Exception as exc:
                    log.debug(f"[SignalLog] analyze {bar_idx}: {exc}")
                    continue

                sig = res.get("signal", "HOLD")
                conf = res.get("confidence", 0.0)

                # Validare: pentru BUY/SELL verifica miscarea pe urmatoarele eval_bars
                outcome = None
                max_up = max_down = 0.0
                entry_price = float(primary_df.iloc[bar_idx]["close"])
                if sig in ("BUY", "SELL"):
                    eval_slice = primary_df.iloc[bar_idx+1 : bar_idx+1+eval_bars]
                    if len(eval_slice) > 0:
                        high_max = float(eval_slice["high"].max())
                        low_min  = float(eval_slice["low"].min())
                        max_up   = round(high_max - entry_price, 6)
                        max_down = round(entry_price - low_min, 6)
                        # Validare simpla: directia dominanta in urmatoarele eval_bars
                        if sig == "BUY":
                            outcome = "valid" if max_up > max_down else "invalid"
                        else:  # SELL
                            outcome = "valid" if max_down > max_up else "invalid"

                signals.append({
                    "bar_idx":     bar_idx,
                    "time":        str(bar_ts),
                    "price":       round(entry_price, 5),
                    "signal":      sig,
                    "confidence":  round(conf, 1),
                    "max_up":      max_up,
                    "max_down":    max_down,
                    "outcome":     outcome,  # None pentru HOLD, "valid"/"invalid" pt BUY/SELL
                    "reasons":     (res.get("justification") or [])[:3],
                })

                # Progress
                if bar_idx % 50 == 0:
                    pct = int(10 + (bar_idx - start_bar) / (end_bar - start_bar) * 85)
                    _update("running", pct, f"Procesat {bar_idx - start_bar} bare ({len(signals)} semnale)")
        finally:
            _app.fetch = original_fetch
            _mock_data.clear()
            _mock_end_idx.clear()

        # Statistici agregate
        buy_signals  = [s for s in signals if s["signal"] == "BUY"]
        sell_signals = [s for s in signals if s["signal"] == "SELL"]
        hold_count   = sum(1 for s in signals if s["signal"] == "HOLD")
        valid_buys   = sum(1 for s in buy_signals  if s["outcome"] == "valid")
        valid_sells  = sum(1 for s in sell_signals if s["outcome"] == "valid")
        n_buys, n_sells = len(buy_signals), len(sell_signals)

        # Confidence buckets
        buckets = {"50-60": [], "60-70": [], "70-80": [], "80-90": [], "90-100": []}
        for s in signals:
            if s["outcome"] is None:
                continue
            c = s["confidence"]
            if   c < 60:  buckets["50-60"].append(s)
            elif c < 70:  buckets["60-70"].append(s)
            elif c < 80:  buckets["70-80"].append(s)
            elif c < 90:  buckets["80-90"].append(s)
            else:         buckets["90-100"].append(s)
        confidence_stats = {}
        for k, lst in buckets.items():
            n = len(lst)
            v = sum(1 for s in lst if s["outcome"] == "valid")
            confidence_stats[k] = {
                "total":    n,
                "valid":    v,
                "accuracy": round(v/n*100, 1) if n else 0.0,
            }

        result = {
            "job_id":     job_id,
            "symbol":     symbol,
            "strategy":   strat_key,
            "tfs":        tfs,
            "eval_bars":  eval_bars,
            "total_bars": n_total,
            "scanned":    len(signals),
            "summary": {
                "buys":           n_buys,
                "sells":          n_sells,
                "holds":          hold_count,
                "buys_valid":     valid_buys,
                "sells_valid":    valid_sells,
                "buy_accuracy":   round(valid_buys/n_buys*100, 1)  if n_buys else 0.0,
                "sell_accuracy":  round(valid_sells/n_sells*100, 1) if n_sells else 0.0,
                "overall_accuracy": round(
                    (valid_buys + valid_sells) / max(1, n_buys + n_sells) * 100, 1),
            },
            "confidence_stats": confidence_stats,
            "signals":  signals[-500:],   # ultimele 500 pentru afisare (toate ar fi prea mult)
            "all_signals_count": len(signals),
        }
        # Salveaza pe disc
        path = os.path.join(_RESULTS_DIR, f"siglog_{job_id}.json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2, default=str)

        with _bt_lock:
            _bt_jobs[job_id].update({
                "status": "done", "progress": 100, "result": result,
                "message": (f"Complet: {len(signals)} semnale | "
                           f"Acuratete BUY {result['summary']['buy_accuracy']}% | "
                           f"SELL {result['summary']['sell_accuracy']}%"),
            })
    except Exception as exc:
        import traceback
        log.error(f"[SignalLog] {job_id} EROARE: {exc}\n{traceback.format_exc()}")
        _update("error", message=str(exc))


@backtest_bp.route("/backtest/signal_log", methods=["POST"])
@_login_required
def backtest_signal_log_endpoint():
    """Lanseaza job de signal log (rolling window, fara trades)."""
    params = request.get_json(silent=True) or {}
    for k in ("symbol", "strategy", "tfs"):
        if not params.get(k):
            return Response(json.dumps({"ok": False, "message": f"Lipsa: {k}"}),
                            content_type="application/json")
    job_id = str(uuid.uuid4())[:8]
    with _bt_lock:
        _bt_jobs[job_id] = {"status": "pending", "progress": 0,
                            "message": "Signal log pornit..."}
    t = threading.Thread(target=_run_signal_log_job, args=(job_id, params), daemon=True)
    t.start()
    return Response(json.dumps({"ok": True, "job_id": job_id}),
                    content_type="application/json")


@backtest_bp.route("/backtest/element_impact", methods=["POST"])
@_login_required
def backtest_element_impact():
    """
    Lanseaza N backtests in paralel — fiecare cu UN element opt_* diferit
    activat — apoi returneaza un raport care arata contributia fiecaruia.

    Body: {symbol, strategy, tfs, duration_years, baseline_elements: {key: bool}}
    """
    params = request.get_json(silent=True) or {}
    required = ["symbol", "strategy", "tfs"]
    for k in required:
        if not params.get(k):
            return Response(json.dumps({"ok": False, "message": f"Lipsa: {k}"}),
                            content_type="application/json")

    import strategies as _sp
    from strategies.common_filters import OPTIONAL_FILTERS
    strat = _sp.get_strategy(params["strategy"])
    if strat is None:
        return Response(json.dumps({"ok": False,
                                    "message": f"Strategie necunoscuta: {params['strategy']}"}),
                        content_type="application/json")

    baseline = params.get("baseline_elements") or {
        k: True for k in strat.elements
    }  # by default toate specifice ON, toate opt_* OFF
    opt_keys_to_test = [f"opt_{k}" for k in OPTIONAL_FILTERS.keys()]

    # Construim runs: 1 baseline + N (cate unul cu un opt_* activat)
    runs = [{"name": "baseline", "elements": dict(baseline)}]
    for opt_k in opt_keys_to_test:
        elems = dict(baseline)
        elems[opt_k] = True
        runs.append({"name": opt_k, "elements": elems})

    # Job id parent pentru a coordona
    parent_id = str(uuid.uuid4())[:8]
    with _bt_lock:
        _bt_jobs[parent_id] = {
            "status": "running", "progress": 0,
            "message": f"Lanseaza {len(runs)} runs paralele...",
            "element_impact": {"runs": []},
        }

    def _run_all():
        results = []
        for i, run in enumerate(runs):
            run_id = f"{parent_id}_{i}"
            with _bt_lock:
                _bt_jobs[run_id] = {"status": "pending", "progress": 0,
                                    "message": f"Run {run['name']}"}
            run_params = {**params, "elements": run["elements"]}
            run_backtest_job(run_id, run_params)
            with _bt_lock:
                job = _bt_jobs.get(run_id, {})
                metrics = (job.get("result") or {}).get("metrics") or {}
                trades_count = len((job.get("result") or {}).get("trades") or [])
            results.append({
                "name":          run["name"],
                "trades":        trades_count,
                "win_rate":      metrics.get("win_rate", 0),
                "profit_factor": metrics.get("profit_factor", 0),
                "total_pnl":     metrics.get("net_pnl", 0),
                "max_dd":        metrics.get("max_drawdown", 0),
                "expectancy":    metrics.get("expectancy", 0),
            })
            with _bt_lock:
                _bt_jobs[parent_id]["progress"] = int((i+1)/len(runs)*100)
                _bt_jobs[parent_id]["message"] = f"Run {i+1}/{len(runs)} completat"
        # Compara fiecare opt_ vs baseline
        baseline_res = results[0]
        diffs = []
        for r in results[1:]:
            diffs.append({
                "element": r["name"],
                "delta_pf":      round(r["profit_factor"] - baseline_res["profit_factor"], 2),
                "delta_winrate": round(r["win_rate"]     - baseline_res["win_rate"], 1),
                "delta_pnl":     round(r["total_pnl"]    - baseline_res["total_pnl"], 2),
                "trades":        r["trades"],
                "trades_baseline": baseline_res["trades"],
            })
        diffs.sort(key=lambda x: x["delta_pf"], reverse=True)
        with _bt_lock:
            _bt_jobs[parent_id]["status"] = "done"
            _bt_jobs[parent_id]["element_impact"] = {
                "baseline": baseline_res,
                "runs":     results,
                "ranking":  diffs,
            }

    t = threading.Thread(target=_run_all, daemon=True)
    t.start()
    return Response(json.dumps({"ok": True, "parent_id": parent_id, "n_runs": len(runs)}),
                    content_type="application/json")


@backtest_bp.route("/backtest/run", methods=["POST"])
@_login_required
def backtest_run():
    params = request.get_json(silent=True) or {}
    required = ["symbol", "strategy", "tfs"]
    for k in required:
        if not params.get(k):
            return Response(json.dumps({"ok": False, "message": f"Lipsa: {k}"}),
                            content_type="application/json")

    # ── Validare symbol whitelist ────────────────────────────────────────────
    # Daca strategia are whitelist setat (ex: gold_scalper → XAU/GOLD),
    # blocam rularea pe simboluri care nu match-uiesc.
    import strategies as _sp
    strat = _sp.get_strategy(params["strategy"])
    if strat is None:
        return Response(json.dumps({"ok": False,
                                    "message": f"Strategie necunoscuta: {params['strategy']}"}),
                        content_type="application/json")

    sym = params["symbol"].upper()
    if not strat.matches_symbol(sym):
        wl = ", ".join(getattr(strat, "symbol_whitelist", []) or [])
        return Response(json.dumps({
            "ok": False,
            "message": (f"Strategia '{strat.name}' ruleaza doar pe simboluri "
                        f"care contin: {wl}. Simbolul {sym} nu match-uieste.")
        }), content_type="application/json")

    job_id = str(uuid.uuid4())[:8]
    with _bt_lock:
        _bt_jobs[job_id] = {"status": "pending", "progress": 0, "message": "Se porneste..."}

    t = threading.Thread(target=run_backtest_job, args=(job_id, params), daemon=True)
    t.start()

    return Response(json.dumps({"ok": True, "job_id": job_id}),
                    content_type="application/json")


@backtest_bp.route("/backtest/status/<job_id>")
@_login_required
def backtest_status(job_id: str):
    with _bt_lock:
        job = _bt_jobs.get(job_id)
    if job is None:
        # Cauta pe disc (atat backtest cat si signal_log)
        for fn in (f"{job_id}.json", f"siglog_{job_id}.json"):
            path = os.path.join(_RESULTS_DIR, fn)
            if os.path.exists(path):
                try:
                    with open(path, encoding="utf-8") as f:
                        disk_result = json.load(f)
                    return Response(json.dumps({
                        "status": "done", "progress": 100,
                        "result": disk_result,
                    }, default=str), content_type="application/json")
                except Exception:
                    pass
        return Response(json.dumps({"status": "not_found"}), content_type="application/json")

    resp = {"status": job["status"], "progress": job["progress"],
            "message": job.get("message", "")}
    # Include result/element_impact daca jobul e gata — UI-ul are nevoie
    if job["status"] == "done":
        if "result" in job:
            resp["result"] = job["result"]
        if "element_impact" in job:
            resp["element_impact"] = job["element_impact"]
    return Response(json.dumps(resp, default=str), content_type="application/json")


# ══════════════════════════════════════════════════════════════════════════════
#  REPLAY MODE — control manual bar-cu-bar
# ══════════════════════════════════════════════════════════════════════════════

@backtest_bp.route("/backtest/replay/start", methods=["POST"])
@_login_required
def backtest_replay_start():
    """Initializeaza un job de replay — incarca date, asteapta comenzi step."""
    params = request.get_json(silent=True) or {}
    required = ["symbol", "strategy", "tfs"]
    for k in required:
        if k not in params:
            return Response(json.dumps({"ok": False, "message": f"Lipseste: {k}"}),
                            content_type="application/json")

    import strategies as _sp
    strat = _sp.get_strategy(params["strategy"])
    if strat is None:
        return Response(json.dumps({"ok": False,
                                    "message": f"Strategie necunoscuta: {params['strategy']}"}),
                        content_type="application/json")

    sym = params["symbol"].upper()
    if not strat.matches_symbol(sym):
        wl = ", ".join(getattr(strat, "symbol_whitelist", []) or [])
        return Response(json.dumps({"ok": False,
            "message": f"Strategia '{strat.name}' ruleaza doar pe: {wl}"}),
            content_type="application/json")

    job_id = str(uuid.uuid4())[:8]
    with _bt_lock:
        _bt_jobs[job_id] = {"status": "loading", "progress": 0,
                            "message": "Replay: pornire...", "mode": "replay"}

    threading.Thread(target=_replay_init_job, args=(job_id, params), daemon=True).start()
    return Response(json.dumps({"ok": True, "job_id": job_id, "mode": "replay"}),
                    content_type="application/json")


@backtest_bp.route("/backtest/replay/step/<job_id>", methods=["POST"])
@_login_required
def backtest_replay_step(job_id: str):
    """Avanseaza N bare (default 1)."""
    body = request.get_json(silent=True) or {}
    n = max(1, int(body.get("bars", 1)))
    snap = _replay_advance(job_id, n_bars=n, mode="fixed")
    if snap is None:
        return Response(json.dumps({"ok": False, "message": "Job not ready"}),
                        content_type="application/json")
    return Response(json.dumps({"ok": True, "snapshot": snap}, default=str),
                    content_type="application/json")


@backtest_bp.route("/backtest/replay/until_trade/<job_id>", methods=["POST"])
@_login_required
def backtest_replay_until_trade(job_id: str):
    """Avanseaza pana se deschide un trade nou."""
    snap = _replay_advance(job_id, mode="until_trade")
    if snap is None:
        return Response(json.dumps({"ok": False, "message": "Job not ready"}),
                        content_type="application/json")
    return Response(json.dumps({"ok": True, "snapshot": snap}, default=str),
                    content_type="application/json")


@backtest_bp.route("/backtest/replay/until_close/<job_id>", methods=["POST"])
@_login_required
def backtest_replay_until_close(job_id: str):
    """Avanseaza pana se inchide trade-ul curent."""
    snap = _replay_advance(job_id, mode="until_close")
    if snap is None:
        return Response(json.dumps({"ok": False, "message": "Job not ready"}),
                        content_type="application/json")
    return Response(json.dumps({"ok": True, "snapshot": snap}, default=str),
                    content_type="application/json")


@backtest_bp.route("/backtest/replay/state/<job_id>")
@_login_required
def backtest_replay_state(job_id: str):
    """Returneaza state-ul curent fara a avansa (pt polling status)."""
    with _bt_lock:
        job = _bt_jobs.get(job_id)
    if not job:
        return Response(json.dumps({"ok": False, "message": "Job not found"}),
                        content_type="application/json")
    return Response(json.dumps({
        "ok": True,
        "status":  job.get("status"),
        "progress": job.get("progress", 0),
        "message": job.get("message", ""),
        "snapshot": job.get("live"),
    }, default=str), content_type="application/json")


@backtest_bp.route("/backtest/live/<job_id>")
@_login_required
def backtest_live(job_id: str):
    """
    Snapshot live al backtestului in timpul rularii:
      - bare recente (~150) cu OHLC
      - trade deschis (entry/sl/tp)
      - trade-uri inchise pana acum (max ultimele 200)
      - metrici curente (wins, losses, WR, equity)
    """
    with _bt_lock:
        job = _bt_jobs.get(job_id)
    if job is None:
        return Response(json.dumps({"ok": False, "message": "Job not found"}),
                        content_type="application/json")
    live = job.get("live")
    if live is None:
        return Response(json.dumps({"ok": False, "message": "Nu exista snapshot inca"}),
                        content_type="application/json")
    return Response(json.dumps({"ok": True, "snapshot": live, "status": job["status"]},
                               default=str),
                    content_type="application/json")


@backtest_bp.route("/backtest/result/<job_id>")
@_login_required
def backtest_result(job_id: str):
    # Incearca din memorie
    with _bt_lock:
        job = _bt_jobs.get(job_id)
    if job and job.get("result"):
        return Response(json.dumps(job["result"], default=str),
                        content_type="application/json")
    # De pe disc
    path = os.path.join(_RESULTS_DIR, f"{job_id}.json")
    if os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            return Response(f.read(), content_type="application/json")
    return Response(json.dumps({"error": "not found"}), content_type="application/json",
                    status=404)


@backtest_bp.route("/backtest/cancel/<job_id>", methods=["POST"])
@_login_required
def backtest_cancel(job_id: str):
    with _bt_lock:
        if job_id in _bt_jobs:
            if _bt_jobs[job_id]["status"] == "running":
                _bt_jobs[job_id]["status"] = "cancelled"
    return Response(json.dumps({"ok": True}), content_type="application/json")


@backtest_bp.route("/backtest/delete/<job_id>", methods=["POST"])
@_login_required
def backtest_delete(job_id: str):
    with _bt_lock:
        _bt_jobs.pop(job_id, None)
    path = os.path.join(_RESULTS_DIR, f"{job_id}.json")
    deleted = False
    if os.path.exists(path):
        os.remove(path)
        deleted = True
    return Response(json.dumps({"ok": True, "deleted": deleted}),
                    content_type="application/json")


@backtest_bp.route("/backtest/saved_list")
@_login_required
def backtest_saved_list():
    """Lista cu toate rezultatele salvate pe disc (backtest + signal_log)."""
    saved = []
    try:
        files = sorted(os.listdir(_RESULTS_DIR),
                       key=lambda f: os.path.getmtime(os.path.join(_RESULTS_DIR, f)),
                       reverse=True)[:50]
    except Exception:
        files = []
    for fn in files:
        if not fn.endswith(".json"):
            continue
        is_siglog = fn.startswith("siglog_")
        try:
            full_path = os.path.join(_RESULTS_DIR, fn)
            with open(full_path, encoding="utf-8") as f:
                d = json.load(f)
            mtime = os.path.getmtime(full_path)
            entry = {
                "job_id":   d.get("job_id", fn.replace(".json", "")),
                "type":     "signal_log" if is_siglog else "backtest",
                "symbol":   d.get("symbol", "?"),
                "strategy": d.get("strategy", "?"),
                "tfs":      d.get("tfs", []),
                "saved_at": datetime.fromtimestamp(mtime).isoformat(),
                "filename": fn,
            }
            if is_siglog:
                s = d.get("summary", {})
                entry.update({
                    "label":           (f"[SigLog] {entry['symbol']} {entry['strategy']} "
                                        f"{','.join(entry['tfs'])} · {d.get('all_signals_count',0)} signals"),
                    "trades":          d.get("all_signals_count", 0),
                    "win_rate":        s.get("overall_accuracy", 0),
                    "pf":              0,
                    "net_pnl":         0,
                    "buy_accuracy":    s.get("buy_accuracy", 0),
                    "sell_accuracy":   s.get("sell_accuracy", 0),
                })
            else:
                m = d.get("metrics", {})
                entry.update({
                    "label":    (f"[BT] {entry['symbol']} {entry['strategy']} "
                                 f"{','.join(entry['tfs'])} · {d.get('from_dt','')}→{d.get('to_dt','')}"),
                    "from_dt":  d.get("from_dt", ""),
                    "to_dt":    d.get("to_dt", ""),
                    "trades":   m.get("total_trades", 0),
                    "win_rate": m.get("win_rate", 0),
                    "pf":       m.get("profit_factor", 0),
                    "net_pnl":  m.get("net_pnl", 0),
                })
            saved.append(entry)
        except Exception:
            pass
    return Response(json.dumps(saved), content_type="application/json")


@backtest_bp.route("/backtest/load/<filename>")
@_login_required
def backtest_load_saved(filename: str):
    """Returneaza continutul unui rezultat salvat (backtest sau signal_log)."""
    # Sanitize filename — accept doar nume simple .json
    if "/" in filename or "\\" in filename or not filename.endswith(".json"):
        return Response(json.dumps({"ok": False, "message": "filename invalid"}),
                        content_type="application/json", status=400)
    full = os.path.join(_RESULTS_DIR, filename)
    if not os.path.exists(full):
        return Response(json.dumps({"ok": False, "message": "not found"}),
                        content_type="application/json", status=404)
    try:
        with open(full, encoding="utf-8") as f:
            d = json.load(f)
        return Response(json.dumps({"ok": True, "result": d, "type":
                                    "signal_log" if filename.startswith("siglog_") else "backtest"}),
                        content_type="application/json")
    except Exception as e:
        return Response(json.dumps({"ok": False, "message": str(e)}),
                        content_type="application/json", status=500)


# ══════════════════════════════════════════════════════════════════════════════
#  HTML
# ══════════════════════════════════════════════════════════════════════════════

def _build_page(strats: list, saved: list) -> str:
    import app as _app
    import strategies as _sp
    from datetime import datetime as _dt

    strats_json = json.dumps(strats)
    saved_json  = json.dumps(saved)
    all_symbols = list(getattr(_app, "SYMBOLS", [])) + list(getattr(_app, "SYMBOLS_CRYPTO", []))
    syms_json   = json.dumps(sorted(set(all_symbols)))
    n_strats    = len(strats)
    page_ts     = _dt.now().strftime("%H:%M:%S")

    TFS_ALL = ["M1","M5","M15","M30","H1","H4","D1","W1"]

    return f"""<!DOCTYPE html>
<html lang="ro">
<head>
<meta charset="UTF-8">
<title>Backtest — ChartVisualizer</title>
<script src="https://cdn.plot.ly/plotly-2.27.0.min.js"></script>
<style>
*{{box-sizing:border-box;margin:0;padding:0}}
body{{background:#0d0d0d;color:#ddd;font-family:'Segoe UI',sans-serif;font-size:13px}}
a{{color:#64b5f6;text-decoration:none}}
h2{{font-size:1.1rem;color:#fff;margin-bottom:10px}}
h3{{font-size:0.9rem;color:#aaa;margin-bottom:8px;font-weight:600;letter-spacing:.5px}}

/* Layout */
.page{{display:flex;height:100vh;overflow:hidden}}
.sidebar{{width:320px;min-width:320px;background:#111;border-right:1px solid #222;
          display:flex;flex-direction:column;overflow-y:auto}}
.main{{flex:1;overflow-y:auto;padding:16px}}

/* Sidebar */
.sidebar-header{{padding:14px 16px 10px;border-bottom:1px solid #222;display:flex;align-items:center;gap:10px}}
.sidebar-header h1{{font-size:1rem;color:#fff}}
.nav-back{{color:#666;font-size:0.8rem}}
.config-section{{padding:14px 16px;border-bottom:1px solid #1a1a1a}}
.config-row{{display:flex;justify-content:space-between;align-items:center;
             margin-bottom:8px;gap:8px}}
.config-row label{{color:#aaa;font-size:0.82rem;flex:1}}
.config-row label .sub{{display:block;font-size:0.7rem;color:#555;margin-top:1px}}
input[type=number],input[type=text],select{{
  background:#1a1a1a;color:#ddd;border:1px solid #2a2a2a;
  border-radius:4px;padding:4px 8px;font-size:0.82rem;outline:none}}
input[type=number]:focus,input[type=text]:focus,select:focus{{border-color:#444}}
.tf-grid{{display:flex;flex-wrap:wrap;gap:5px;margin-top:4px}}
.tf-btn{{padding:3px 9px;border-radius:4px;border:1px solid #2a2a2a;background:#1a1a1a;
          color:#777;cursor:pointer;font-size:0.78rem;transition:.15s}}
.tf-btn.active{{border-color:#64b5f6;color:#64b5f6;background:#0a1a2a}}
.run-btn{{width:100%;padding:11px;background:#1565c0;color:#fff;font-size:0.92rem;
          font-weight:700;border:none;border-radius:6px;cursor:pointer;transition:.2s;
          letter-spacing:.3px}}
.run-btn:hover{{background:#1976d2}}
.run-btn:disabled{{background:#1a1a1a;color:#555;cursor:not-allowed}}
.run-btn.danger{{background:#b71c1c}}
.run-btn.danger:hover{{background:#c62828}}

/* Progress */
.progress-wrap{{padding:12px 16px;border-bottom:1px solid #1a1a1a;display:none}}
.progress-bar-outer{{height:6px;background:#1a1a1a;border-radius:3px;overflow:hidden;margin-bottom:6px}}
.progress-bar-inner{{height:100%;background:#1565c0;border-radius:3px;transition:width .3s}}
.progress-msg{{color:#777;font-size:0.78rem}}

/* Saved list */
.saved-section{{padding:12px 16px;flex:1}}
.saved-item{{padding:7px 10px;border:1px solid #1a1a1a;border-radius:5px;
             margin-bottom:6px;cursor:pointer;transition:.15s}}
.saved-item:hover{{border-color:#333;background:#141414}}
.saved-item .label{{color:#ccc;font-size:0.8rem;font-weight:600}}
.saved-item .meta{{color:#555;font-size:0.73rem;margin-top:2px}}
.saved-item .meta .wr{{color:#66bb6a}}.saved-item .meta .pf{{color:#ffa726}}
.saved-item .meta .pnl{{}}

/* Main area */
.empty-state{{display:flex;flex-direction:column;align-items:center;
              justify-content:center;height:60%;color:#444;gap:12px}}
.empty-state .icon{{font-size:2.5rem}}
.empty-state p{{font-size:0.9rem}}

/* Metric cards */
.metrics-grid{{display:grid;grid-template-columns:repeat(auto-fill,minmax(130px,1fr));gap:10px;margin-bottom:16px}}
.metric-card{{background:#141414;border:1px solid #222;border-radius:7px;padding:12px 14px}}
.metric-card .val{{font-size:1.3rem;font-weight:700;color:#fff}}
.metric-card .lbl{{font-size:0.72rem;color:#666;margin-top:3px}}
.metric-card.green .val{{color:#66bb6a}}
.metric-card.red .val{{color:#ef5350}}
.metric-card.orange .val{{color:#ffa726}}
.metric-card.blue .val{{color:#64b5f6}}

/* Chart */
.chart-wrap{{background:#111;border:1px solid #1a1a1a;border-radius:7px;
             padding:10px;margin-bottom:16px;height:300px}}

/* Live view */
.live-wrap{{display:none;flex-direction:column;gap:12px}}
.live-header{{display:flex;align-items:center;gap:12px;padding-bottom:8px}}
.live-pulse{{width:8px;height:8px;border-radius:50%;background:#ef5350;
             animation:pulse 1.2s ease-in-out infinite}}
@keyframes pulse {{ 0%,100%{{opacity:1}}50%{{opacity:.3}} }}
.live-title{{font-size:0.95rem;color:#fff;font-weight:600}}
.live-bar-info{{color:#666;font-size:0.78rem;margin-left:auto}}
.live-metrics{{display:grid;grid-template-columns:repeat(6,1fr);gap:8px}}
.live-metric{{background:#141414;border:1px solid #222;border-radius:5px;padding:8px 10px}}
.live-metric .v{{font-size:1.05rem;font-weight:700;color:#fff}}
.live-metric .l{{font-size:0.7rem;color:#666;margin-top:2px}}
.live-metric.green .v{{color:#66bb6a}}
.live-metric.red .v{{color:#ef5350}}
.live-metric.blue .v{{color:#64b5f6}}
.live-chart{{background:#0a0a0a;border:1px solid #1a1a1a;border-radius:7px;
             height:480px;padding:8px}}
.live-history{{background:#111;border:1px solid #1a1a1a;border-radius:7px;
               padding:10px;max-height:280px;overflow-y:auto}}
.live-open{{padding:8px 12px;background:#0a1a2a;border:1px solid #1565c0;
            border-radius:5px;font-size:0.83rem;color:#90caf9;
            display:flex;align-items:center;gap:14px}}
.live-open.empty{{background:#141414;border-color:#222;color:#555}}

/* Replay step buttons sub chart */
button.rb{{padding:9px 6px;background:#2a2a2a;color:#ddd;border:1px solid #333;
          border-radius:5px;font-size:0.82rem;font-weight:600;cursor:pointer;
          transition:all .15s}}
button.rb:hover{{background:#1565c0;border-color:#1976d2;color:#fff}}
button.rb:active{{transform:scale(.97)}}

/* Trade table */
.trade-table{{width:100%;border-collapse:collapse;font-size:0.79rem}}
.trade-table th{{background:#141414;color:#777;padding:7px 10px;text-align:left;
                  border-bottom:1px solid #222;font-weight:600;letter-spacing:.3px;
                  cursor:pointer;user-select:none}}
.trade-table th:hover{{color:#aaa}}
.trade-table td{{padding:6px 10px;border-bottom:1px solid #181818;color:#bbb}}
.trade-table tr:hover td{{background:#141414}}
.badge{{padding:2px 7px;border-radius:3px;font-size:0.72rem;font-weight:700}}
.badge.TP{{background:#1b3a1b;color:#66bb6a}}
.badge.SL{{background:#3a1b1b;color:#ef5350}}
.badge.EXP{{background:#2a2a1a;color:#ffa726}}
.badge.TO{{background:#1a1a2e;color:#90caf9}}
.badge.BUY{{background:#0a1a2a;color:#64b5f6}}
.badge.SELL{{background:#2a0a1a;color:#f48fb1}}
.result-header{{display:flex;align-items:center;justify-content:space-between;margin-bottom:12px}}
.result-header h2{{margin:0}}
.result-meta{{font-size:0.78rem;color:#555}}
</style>
</head>
<body>
<div class="page">

<!-- ═══ SIDEBAR ═══════════════════════════════════════════════════════════ -->
<div class="sidebar">
  <div class="sidebar-header">
    <div>
      <h1>📊 Backtest <span style="font-size:0.65rem;color:#444;font-weight:400">v2.LIVE</span></h1>
      <a class="nav-back" href="/">← Inapoi la ChartVisualizer</a>
      <div style="font-size:0.65rem;color:#333;margin-top:3px">
        {n_strats} strategii | server start {page_ts}
      </div>
    </div>
  </div>

  <!-- Config -->
  <div class="config-section">
    <h3>SIMBOL</h3>
    <div class="config-row">
      <input type="text" id="sym-input" placeholder="EURUSD"
             list="sym-list" style="width:100%;text-transform:uppercase"
             oninput="this.value=this.value.toUpperCase()">
      <datalist id="sym-list"></datalist>
    </div>

    <h3 style="margin-top:12px">STRATEGIE</h3>
    <select id="strat-select" style="width:100%;margin-bottom:10px">
      <option value="">— selecteaza —</option>
    </select>

    <h3>TIMEFRAME-URI <span id="tf-hint" style="font-size:0.7rem;color:#1565c0;font-weight:normal;display:none">— auto din strategie</span></h3>
    <div class="tf-grid" id="tf-grid">
      {"".join(f'<button class="tf-btn" data-tf="{tf}" onclick="toggleTF(this)">{tf}</button>' for tf in TFS_ALL)}
    </div>
    <div id="tf-strat-info" style="font-size:0.7rem;color:#666;margin-top:6px;display:none"></div>
  </div>

  <div class="config-section">
    <h3>PARAMETRI</h3>
    <div class="config-row">
      <label>Lumanari fereastra<span class="sub">bare pe analiza</span></label>
      <input type="number" id="bars-window" value="500" min="100" max="2000" step="50" style="width:70px">
    </div>
    <div class="config-row">
      <label>Durata<span class="sub">ani de history</span></label>
      <input type="number" id="duration" value="2" min="0.5" max="4" step="0.5" style="width:60px">
    </div>
    <div class="config-row">
      <label>Pas scanare<span class="sub">bare intre analyze()</span></label>
      <input type="number" id="step" value="4" min="1" max="100" step="1" style="width:60px">
    </div>
    <div class="config-row">
      <label>Risk / trade<span class="sub">USD</span></label>
      <input type="number" id="risk-usd" value="50" min="1" max="10000" step="1" style="width:70px">
    </div>
    <div class="config-row">
      <label>TP / SL ratio<span class="sub">2.0 = 1:2</span></label>
      <input type="number" id="tp-ratio" value="2.0" min="0.5" max="10" step="0.1" style="width:60px">
    </div>
    <div class="config-row">
      <label>Min confidence %<span class="sub">prag semnal</span></label>
      <input type="number" id="min-conf" value="66" min="0" max="100" step="1" style="width:60px">
    </div>
    <div class="config-row">
      <label>Spread<span class="sub">price: 0.0003 FX, 0.30 XAUUSD</span></label>
      <input type="number" id="spread" value="0.0003" min="0" max="10" step="0.0001" style="width:80px">
    </div>
    <div class="config-row">
      <label>Max durata trade<span class="sub">ore (0 = nelimitat)</span></label>
      <input type="number" id="max-dur" value="10" min="0" max="240" step="1" style="width:60px">
    </div>
  </div>

  <div class="config-section" style="background:#0e1620;border-left:3px solid #1565c0">
    <h3 style="color:#90caf9">⏯ MOD EXECUTIE</h3>
    <label style="display:flex;align-items:center;gap:8px;cursor:pointer;padding:6px 0">
      <input type="checkbox" id="replay-mode" onchange="toggleReplayMode()" style="width:18px;height:18px;cursor:pointer">
      <span style="font-size:0.85rem">
        <strong style="color:#fff">Replay manual</strong>
        <span class="sub" style="display:block;color:#666;font-size:0.72rem;margin-top:2px">
          Step-by-step cu butoane (+1 bara, urmatorul trade, etc.)
        </span>
      </span>
    </label>
    <button class="run-btn" id="run-btn" onclick="startBacktest()" style="margin-top:8px">▶ Ruleaza Backtest</button>
  </div>

  <!-- Replay controls eliminate din sidebar — sunt doar sub chart in main area.
       Pastram un container ascuns ca JS-ul existent sa nu dea null reference. -->
  <div id="replay-controls" style="display:none">
    <input type="hidden" id="auto-delay" value="800">
    <button id="auto-btn" style="display:none"></button>
    <div id="replay-status" style="display:none"></div>
  </div>

  <!-- Progress -->
  <div class="progress-wrap" id="progress-wrap">
    <div class="progress-bar-outer">
      <div class="progress-bar-inner" id="progress-bar" style="width:0%"></div>
    </div>
    <div class="progress-msg" id="progress-msg">Se porneste...</div>
    <div id="error-box" style="display:none;margin-top:8px;padding:8px 10px;
         background:#1f0808;border:1px solid #7f0000;border-radius:5px;
         color:#ef9a9a;font-size:0.78rem;white-space:pre-wrap;word-break:break-word"></div>
    <div style="display:flex;gap:6px;margin-top:8px">
      <button class="run-btn danger" style="flex:1;padding:6px" id="cancel-btn" onclick="cancelJob()">⏹ Stop</button>
      <button class="run-btn" style="flex:0;padding:6px 10px;background:#222;color:#888;display:none"
              id="close-err-btn" onclick="closeError()">✕</button>
    </div>
  </div>

  <!-- Salvate -->
  <div class="saved-section">
    <h3>RULARI ANTERIOARE</h3>
    <div id="saved-list"></div>
  </div>
</div>

<!-- ═══ MAIN ═══════════════════════════════════════════════════════════════ -->
<div class="main" id="main-area">
  <div class="empty-state" id="empty-state">
    <div class="icon">📈</div>
    <p>Configureaza si apasa <strong>Ruleaza Backtest</strong> pentru a incepe.</p>
    <p style="color:#333;font-size:0.78rem">
      Motor walk-forward: analizeaza fiecare fereastra de N lumanari, simuleaza SL/TP bara cu bara.
    </p>
  </div>

  <!-- Live view (apare in timpul rularii) -->
  <div class="live-wrap" id="live-wrap">
    <div class="live-header">
      <div class="live-pulse"></div>
      <div class="live-title" id="live-title">LIVE Backtest</div>
      <div class="live-bar-info" id="live-bar-info">Bara: — | Time: —</div>
    </div>

    <!-- Trade deschis -->
    <div class="live-open empty" id="live-open">Niciun trade deschis</div>

    <!-- De ce s-a luat decizia (vizibil cand exista trade deschis) -->
    <div id="live-reasons" style="display:none;background:#0a1018;border:1px solid #1565c0;
         border-radius:5px;padding:8px 12px;font-size:0.78rem;color:#bbb;line-height:1.6;margin-top:-6px">
      <div style="color:#90caf9;font-weight:600;margin-bottom:5px;font-size:0.75rem">
        🔍 Motivele intrarii:
      </div>
      <div id="live-reasons-body"></div>
    </div>

    <!-- Metrici live -->
    <div class="live-metrics">
      <div class="live-metric"><div class="v" id="lm-total">0</div><div class="l">Total trade-uri</div></div>
      <div class="live-metric green"><div class="v" id="lm-wins">0</div><div class="l">Wins (TP)</div></div>
      <div class="live-metric red"><div class="v" id="lm-losses">0</div><div class="l">Losses (SL)</div></div>
      <div class="live-metric blue"><div class="v" id="lm-wr">0%</div><div class="l">Win Rate</div></div>
      <div class="live-metric" id="lm-eq-card"><div class="v" id="lm-equity">0$</div><div class="l">Equity (P&L)</div></div>
      <div class="live-metric"><div class="v" id="lm-risk">50$</div><div class="l">Risk/trade</div></div>
    </div>

    <!-- Chart live -->
    <div class="live-chart" id="live-chart"></div>

    <!-- ═══ Replay controls SUB CHART (vizibil doar in mod replay) ═══ -->
    <div id="replay-controls-main" style="display:none;background:#0e1620;
         border:1px solid #1565c0;border-radius:7px;padding:12px;margin-bottom:12px">
      <div style="display:flex;align-items:center;gap:12px;margin-bottom:10px">
        <strong style="color:#90caf9;font-size:0.95rem">⏯ CONTROL REPLAY</strong>
        <span id="replay-status-main" style="color:#888;font-size:0.78rem;margin-left:auto"></span>
      </div>

      <!-- Auto-play row -->
      <div style="display:flex;align-items:center;gap:10px;margin-bottom:10px">
        <button id="auto-btn-main" onclick="toggleAutoPlay()"
                style="flex:1;padding:12px;background:#7b1fa2;color:#fff;border:none;
                       border-radius:5px;font-size:0.92rem;font-weight:700;cursor:pointer">
          ▶ Auto-play (trade cu trade)
        </button>
        <label style="color:#888;font-size:0.78rem">Pauza
          <input type="number" id="auto-delay-main" value="800" min="100" max="5000" step="100"
                 style="width:70px;margin-left:5px;background:#1a1a1a;color:#ddd;
                        border:1px solid #2a2a2a;border-radius:4px;padding:5px 7px"
                 onchange="document.getElementById('auto-delay').value=this.value">
          ms
        </label>
      </div>

      <!-- Step buttons row -->
      <div style="display:grid;grid-template-columns:repeat(5,1fr);gap:6px;margin-bottom:6px">
        <button class="rb" onclick="replayStep(1)">+1 bara</button>
        <button class="rb" onclick="replayStep(2)">+2 bare</button>
        <button class="rb" onclick="replayStep(3)">+3 bare</button>
        <button class="rb" onclick="replayStep(10)">+10 bare</button>
        <button class="rb" onclick="replayStep(50)">+50 bare</button>
      </div>

      <!-- Smart navigation -->
      <div style="display:grid;grid-template-columns:1fr 1fr;gap:6px">
        <button onclick="replayUntilTrade()"
                style="padding:11px;background:#1565c0;color:#fff;border:none;
                       border-radius:5px;font-size:0.85rem;font-weight:600;cursor:pointer">
          ⏭ Pana la urmatorul trade
        </button>
        <button onclick="replayUntilClose()"
                style="padding:11px;background:#0a4f3a;color:#fff;border:none;
                       border-radius:5px;font-size:0.85rem;font-weight:600;cursor:pointer">
          ⏹ Pana se inchide trade-ul
        </button>
      </div>
    </div>

    <!-- Istoric scrollabil -->
    <div class="live-history">
      <h3 style="margin-bottom:6px">ISTORIC TRADE-URI</h3>
      <table class="trade-table" id="live-history-table">
        <thead><tr>
          <th>Intrare</th><th>Iesire</th><th>Semnal</th><th>Outcome</th>
          <th>P&L</th><th>R:R</th><th>Durata</th>
        </tr></thead>
        <tbody id="live-history-body"></tbody>
      </table>
    </div>
  </div>

  <div id="result-area" style="display:none"></div>
</div>

</div><!-- .page -->

<script>
const STRATS = {strats_json};
const SAVED  = {saved_json};
const SYMS   = {syms_json};

let _selectedTFs = [];
let _currentJobId = null;
let _pollInterval = null;

// ── Init ──────────────────────────────────────────────────────────────────────
(function init() {{
  // Symbols datalist
  const dl = document.getElementById("sym-list");
  SYMS.forEach(s => {{ const o = document.createElement("option"); o.value = s; dl.appendChild(o); }});

  // Strategy select
  const sel = document.getElementById("strat-select");
  STRATS.forEach(s => {{
    const o = document.createElement("option");
    o.value = s.key;
    o.textContent = s.icon + " " + s.name;
    sel.appendChild(o);
  }});
  sel.addEventListener("change", onStratChange);

  // Saved list — incarca proaspat de pe server
  refreshSavedList();
}})();

function _flashInput(id) {{
  const el = document.getElementById(id);
  if (!el) return;
  const orig = el.style.boxShadow;
  el.style.boxShadow = "0 0 0 2px #66bb6a";
  setTimeout(() => {{ el.style.boxShadow = orig; }}, 600);
}}

function onStratChange() {{
  const key = document.getElementById("strat-select").value;
  const def = STRATS.find(s => s.key === key);
  const hintEl = document.getElementById("tf-hint");
  const infoEl = document.getElementById("tf-strat-info");
  if (!def) {{
    if (hintEl) hintEl.style.display = "none";
    if (infoEl) infoEl.style.display = "none";
    return;
  }}

  // ── 1. TF auto-aplicate ──────────────────────────────────────────────
  _selectedTFs = [...def.default_tfs];
  syncTFButtons();

  // ── 2. PARAMETRII auto-aplicati din strategie ────────────────────────
  if (def.default_bars     != null) {{
    document.getElementById("bars-window").value = def.default_bars;
    _flashInput("bars-window");
  }}
  if (def.default_step     != null) {{
    document.getElementById("step").value = def.default_step;
    _flashInput("step");
  }}
  if (def.default_duration != null) {{
    document.getElementById("duration").value = def.default_duration;
    _flashInput("duration");
  }}
  if (def.default_max_hours != null) {{
    document.getElementById("max-dur").value = def.default_max_hours;
    _flashInput("max-dur");
  }}
  if (def.default_min_conf != null) {{
    document.getElementById("min-conf").value = def.default_min_conf;
    _flashInput("min-conf");
  }}

  // ── 3. Indicator vizual: TF + parametri auto din strategie ───────────
  if (hintEl) hintEl.style.display = "inline";
  if (infoEl) {{
    infoEl.style.display = "block";
    infoEl.innerHTML =
      `<span style="color:#90caf9">${{def.icon}} ${{def.name}}</span> ` +
      `recomanda:<br>` +
      `<span style="color:#aaa">TF:</span> <strong style="color:#fff">${{def.default_tfs.join(", ")}}</strong> | ` +
      `<span style="color:#aaa">Bars:</span> <strong style="color:#fff">${{def.default_bars}}</strong> | ` +
      `<span style="color:#aaa">Pas:</span> <strong style="color:#fff">${{def.default_step}}</strong><br>` +
      `<span style="color:#aaa">Durata:</span> <strong style="color:#fff">${{def.default_duration}}y</strong> | ` +
      `<span style="color:#aaa">Max:</span> <strong style="color:#fff">${{def.default_max_hours}}h</strong> | ` +
      `<span style="color:#aaa">Conf:</span> <strong style="color:#fff">${{def.default_min_conf}}%</strong>` +
      `<br><span style="color:#555">Toate auto-aplicate. Edit manual pentru override.</span>`;
  }}

  // Flash pe butoanele TF auto-selectate
  document.querySelectorAll(".tf-btn.active").forEach(b => {{
    const orig = b.style.boxShadow;
    b.style.boxShadow = "0 0 0 2px #66bb6a";
    setTimeout(() => {{ b.style.boxShadow = orig; }}, 600);
  }});
}}

function toggleTF(btn) {{
  const tf = btn.dataset.tf;
  const idx = _selectedTFs.indexOf(tf);
  if (idx >= 0) {{
    _selectedTFs.splice(idx, 1);
  }} else {{
    _selectedTFs.push(tf);
  }}
  syncTFButtons();
}}

function syncTFButtons() {{
  document.querySelectorAll(".tf-btn").forEach(b => {{
    b.classList.toggle("active", _selectedTFs.includes(b.dataset.tf));
  }});
}}

// ── Backtest run ──────────────────────────────────────────────────────────────
let _livePollInterval = null;
let _replayMode = false;
let _replayJobId = null;

function toggleReplayMode() {{
  _replayMode = document.getElementById("replay-mode").checked;
  const stepInput = document.getElementById("step");
  if (_replayMode) {{
    stepInput.disabled = true;
    stepInput.style.opacity = 0.4;
    document.getElementById("run-btn").textContent = "▶ Start Replay";
  }} else {{
    stepInput.disabled = false;
    stepInput.style.opacity = 1;
    document.getElementById("run-btn").textContent = "▶ Ruleaza Backtest";
  }}
}}

function replayStep(nBars) {{
  if (!_replayJobId) return;
  setReplayStatus(`Avansare ${{nBars}} bare...`);
  fetch("/backtest/replay/step/" + _replayJobId, {{
    method: "POST",
    headers: {{"Content-Type": "application/json"}},
    body: JSON.stringify({{bars: nBars}})
  }})
  .then(r => r.json())
  .then(d => {{
    if (!d.ok) {{ setReplayStatus("❌ " + (d.message||"")); return; }}
    renderLive(d.snapshot);
    setReplayStatus(`Bara ${{d.snapshot.bar_idx}}/${{d.snapshot.total_bars}} | ${{d.snapshot.metrics.total}} trade-uri`);
  }})
  .catch(e => setReplayStatus("Eroare retea: " + e));
}}

function replayUntilTrade() {{
  if (!_replayJobId) return;
  setReplayStatus("Cautare urmatorul trade...");
  fetch("/backtest/replay/until_trade/" + _replayJobId, {{method: "POST"}})
  .then(r => r.json())
  .then(d => {{
    if (!d.ok) {{ setReplayStatus("❌ " + (d.message||"")); return; }}
    renderLive(d.snapshot);
    const ot = d.snapshot.open_trade;
    setReplayStatus(ot
      ? `Bara ${{d.snapshot.bar_idx}} | Nou trade: ${{ot.signal}} @ ${{ot.entry_price?.toFixed(5)}}`
      : `Bara ${{d.snapshot.bar_idx}} | Niciun trade nou — finale data poate?`);
  }})
  .catch(e => setReplayStatus("Eroare retea: " + e));
}}

function replayUntilClose() {{
  if (!_replayJobId) return;
  setReplayStatus("Astept inchiderea trade-ului...");
  fetch("/backtest/replay/until_close/" + _replayJobId, {{method: "POST"}})
  .then(r => r.json())
  .then(d => {{
    if (!d.ok) {{ setReplayStatus("❌ " + (d.message||"")); return; }}
    renderLive(d.snapshot);
    const m = d.snapshot.metrics;
    setReplayStatus(`Bara ${{d.snapshot.bar_idx}} | Total: ${{m.total}} (W ${{m.wins}}/L ${{m.losses}}) | Equity: ${{m.equity}}$`);
  }})
  .catch(e => setReplayStatus("Eroare retea: " + e));
}}

function setReplayStatus(msg) {{
  const el  = document.getElementById("replay-status");
  const el2 = document.getElementById("replay-status-main");
  if (el)  el.textContent  = msg;
  if (el2) el2.textContent = msg;
}}

// ── Auto-play: cicleaza automat until_trade → until_close → repeat ──────────
let _autoPlayActive = false;
let _autoPlayLastBarIdx = -1;
let _autoPlayStuckCount = 0;

function toggleAutoPlay() {{
  if (_autoPlayActive) {{
    stopAutoPlay();
  }} else {{
    startAutoPlay();
  }}
}}

function _setAutoBtns(text, bg) {{
  ["auto-btn", "auto-btn-main"].forEach(id => {{
    const b = document.getElementById(id);
    if (b) {{ b.textContent = text; b.style.background = bg; }}
  }});
}}

function startAutoPlay() {{
  if (!_replayJobId) return;
  _autoPlayActive = true;
  _autoPlayLastBarIdx = -1;
  _autoPlayStuckCount = 0;
  _setAutoBtns("⏸ Pauza Auto-play", "#c2185b");
  setReplayStatus("▶ Auto-play pornit — cicleaza trade cu trade...");
  autoPlayTick();
}}

function stopAutoPlay() {{
  _autoPlayActive = false;
  _setAutoBtns("▶ Auto-play (trade cu trade)", "#7b1fa2");
  setReplayStatus("⏸ Auto-play oprit");
}}

function autoPlayTick() {{
  if (!_autoPlayActive || !_replayJobId) return;
  const delayMain = parseInt((document.getElementById("auto-delay-main")||{{}}).value);
  const delaySide = parseInt((document.getElementById("auto-delay")||{{}}).value);
  const delay = Math.max(100, delayMain || delaySide || 800);

  // Determina ce comanda sa rulam: daca avem trade deschis → until_close, altfel until_trade
  fetch("/backtest/replay/state/" + _replayJobId)
  .then(r => r.json())
  .then(d => {{
    if (!d.ok) {{ stopAutoPlay(); return; }}
    if (d.status === "done") {{
      stopAutoPlay();
      setReplayStatus("✓ Auto-play complet — replay-ul s-a terminat");
      return;
    }}
    const snap = d.snapshot;
    if (!snap) {{ stopAutoPlay(); return; }}

    // Detecteaza stuck (acelasi bar_idx la 3 incercari → stop)
    if (snap.bar_idx === _autoPlayLastBarIdx) {{
      _autoPlayStuckCount++;
      if (_autoPlayStuckCount >= 3) {{
        stopAutoPlay();
        setReplayStatus("⚠ Auto-play oprit (stuck) — fara progres la bara " + snap.bar_idx);
        return;
      }}
    }} else {{
      _autoPlayStuckCount = 0;
      _autoPlayLastBarIdx = snap.bar_idx;
    }}

    const endpoint = snap.open_trade
      ? "/backtest/replay/until_close/" + _replayJobId
      : "/backtest/replay/until_trade/" + _replayJobId;

    fetch(endpoint, {{method: "POST"}})
    .then(r => r.json())
    .then(d2 => {{
      if (!d2.ok) {{ stopAutoPlay(); setReplayStatus("❌ " + (d2.message||"")); return; }}
      renderLive(d2.snapshot);
      const m = d2.snapshot.metrics;
      const phase = d2.snapshot.open_trade ? "🟢 Trade deschis" : "🔴 Cautare next";
      setReplayStatus(`${{phase}} | Bara ${{d2.snapshot.bar_idx}}/${{d2.snapshot.total_bars}} | T: ${{m.total}} W/L: ${{m.wins}}/${{m.losses}} | Eq: ${{m.equity}}$`);

      // Daca am atins finalul datelor → stop
      if (d2.snapshot.bar_idx >= d2.snapshot.total_bars - 1) {{
        stopAutoPlay();
        setReplayStatus("✓ Final data — auto-play oprit");
        return;
      }}
      // Repeta dupa delay
      if (_autoPlayActive) {{
        setTimeout(autoPlayTick, delay);
      }}
    }})
    .catch(e => {{ stopAutoPlay(); setReplayStatus("Eroare retea: " + e); }});
  }})
  .catch(e => {{ stopAutoPlay(); setReplayStatus("Eroare retea: " + e); }});
}}

function pollReplayInit() {{
  if (!_replayJobId) return;
  fetch("/backtest/replay/state/" + _replayJobId)
  .then(r => r.json())
  .then(d => {{
    if (!d.ok) return;
    setProgress(d.progress || 0, d.message || "");
    if (d.status === "ready") {{
      clearInterval(_pollInterval); _pollInterval = null;
      // Initializare gata — afiseaza ambele panouri (sidebar + sub chart)
      document.getElementById("replay-controls").style.display = "block";
      document.getElementById("replay-controls-main").style.display = "block";
      document.getElementById("progress-wrap").style.display = "none";
      if (d.snapshot) renderLive(d.snapshot);
      setReplayStatus(`✓ Date incarcate. Bara start ${{d.snapshot?.bar_idx||0}}/${{d.snapshot?.total_bars||0}}`);
      document.getElementById("run-btn").disabled = false;
    }} else if (d.status === "error") {{
      clearInterval(_pollInterval); _pollInterval = null;
      showError(d.message);
    }}
  }});
}}

function startReplay() {{
  if (_pollInterval) {{ clearInterval(_pollInterval); _pollInterval = null; }}
  if (_livePollInterval) {{ clearInterval(_livePollInterval); _livePollInterval = null; }}

  const sym    = document.getElementById("sym-input").value.trim().toUpperCase();
  const strat  = document.getElementById("strat-select").value;
  const bars   = parseInt(document.getElementById("bars-window").value);
  const dur    = parseFloat(document.getElementById("duration").value);
  const risk   = parseFloat(document.getElementById("risk-usd").value);
  const tpr    = parseFloat(document.getElementById("tp-ratio").value);
  const conf   = parseFloat(document.getElementById("min-conf").value);
  const sprd   = parseFloat(document.getElementById("spread").value);
  const maxDur = parseFloat(document.getElementById("max-dur").value);

  if (!sym)   {{ alert("Introdu simbol"); return; }}
  if (!strat) {{ alert("Selecteaza strategie"); return; }}
  if (_selectedTFs.length === 0) {{ alert("Selecteaza TF"); return; }}

  document.getElementById("run-btn").disabled = true;
  document.getElementById("progress-wrap").style.display = "block";
  document.getElementById("replay-controls").style.display = "none";
  document.getElementById("replay-controls-main").style.display = "none";
  setProgress(0, "Replay: pornire...");

  fetch("/backtest/replay/start", {{
    method: "POST",
    headers: {{"Content-Type": "application/json"}},
    body: JSON.stringify({{
      symbol: sym, strategy: strat, tfs: _selectedTFs,
      bars_window: bars, duration_years: dur,
      risk_dollars: risk, tp_ratio: tpr, min_confidence: conf,
      spread: sprd, max_duration_hours: maxDur,
    }})
  }})
  .then(r => r.json())
  .then(d => {{
    if (!d.ok) {{ alert("Eroare: " + d.message); resetRunBtn(); return; }}
    _replayJobId = d.job_id;
    showLiveView();
    _pollInterval = setInterval(pollReplayInit, 800);
  }})
  .catch(e => {{ alert("Eroare retea: " + e); resetRunBtn(); }});
}}

function showLiveView() {{
  document.getElementById("empty-state").style.display = "none";
  document.getElementById("result-area").style.display = "none";
  document.getElementById("live-wrap").style.display = "flex";
}}

function hideLiveView() {{
  document.getElementById("live-wrap").style.display = "none";
  if (_livePollInterval) {{ clearInterval(_livePollInterval); _livePollInterval = null; }}
}}

function pollLive() {{
  if (!_currentJobId) return;
  fetch("/backtest/live/" + _currentJobId)
  .then(r => r.json())
  .then(d => {{
    if (!d.ok || !d.snapshot) return;
    renderLive(d.snapshot);
    if (d.status === "done" || d.status === "error" || d.status === "cancelled") {{
      // ultim render apoi opreste polling-ul live
      if (_livePollInterval) {{ clearInterval(_livePollInterval); _livePollInterval = null; }}
    }}
  }})
  .catch(() => {{}});
}}

function renderLive(snap) {{
  // ── Bar info ────────────────────────────────────────────────────────────
  const barTime = (snap.bar_time || "").substring(0, 19);
  document.getElementById("live-bar-info").textContent =
    `Bara: ${{snap.bar_idx}} | Time: ${{barTime}} | TF: ${{snap.primary_tf}}`;

  // ── Open trade ──────────────────────────────────────────────────────────
  const openEl = document.getElementById("live-open");
  const reasonsBox = document.getElementById("live-reasons");
  const reasonsBody = document.getElementById("live-reasons-body");
  if (snap.open_trade) {{
    const ot = snap.open_trade;
    const sigBadge = ot.signal === "BUY" ? "🟢 BUY" : "🔴 SELL";
    openEl.classList.remove("empty");
    openEl.innerHTML =
      `<strong>${{sigBadge}}</strong>` +
      `<span>Entry: ${{ot.entry_price?.toFixed(5)}}</span>` +
      `<span>SL: ${{ot.sl?.toFixed(5)}}</span>` +
      `<span>TP: ${{ot.tp?.toFixed(5)}}</span>` +
      `<span>Time: ${{(ot.entry_time||"").substring(0,19)}}</span>` +
      (ot.confidence ? `<span>Conf: ${{ot.confidence}}%</span>` : "");

    // Afiseaza motivele intrarii
    if (reasonsBox && reasonsBody) {{
      const reasons = ot.reasons || [];
      if (reasons.length > 0) {{
        reasonsBody.innerHTML = reasons.map(r => `• ${{r}}`).join("<br>");
        reasonsBox.style.display = "block";
      }} else {{
        reasonsBox.style.display = "none";
      }}
    }}
  }} else {{
    openEl.classList.add("empty");
    openEl.textContent = "Niciun trade deschis";
    if (reasonsBox) reasonsBox.style.display = "none";
  }}

  // ── Metrici ─────────────────────────────────────────────────────────────
  const m = snap.metrics || {{}};
  document.getElementById("lm-total").textContent = m.total || 0;
  document.getElementById("lm-wins").textContent  = m.wins  || 0;
  document.getElementById("lm-losses").textContent = m.losses || 0;
  document.getElementById("lm-wr").textContent    = (m.wr || 0).toFixed(1) + "%";
  const eqEl = document.getElementById("lm-equity");
  const eqCard = document.getElementById("lm-eq-card");
  eqEl.textContent = (m.equity >= 0 ? "+" : "") + (m.equity || 0).toFixed(2) + "$";
  eqCard.classList.remove("green", "red");
  eqCard.classList.add(m.equity >= 0 ? "green" : "red");
  document.getElementById("lm-risk").textContent = (m.risk || 50) + "$";

  // ── Chart Plotly cu candele — ZOOM ADAPTIV pe trade ─────────────────────
  const bars = snap.bars || [];
  if (bars.length > 0) {{
    const ot = snap.open_trade;

    // ZOOM ADAPTIV: cand e trade deschis, focusam pe ultimele 40 bare;
    // altfel aratam ultimele 80 bare (mai compact, mai detaliu)
    const focusBars = ot ? 40 : 80;
    const startSlice = Math.max(0, bars.length - focusBars);
    const visBars = bars.slice(startSlice);

    const candleTrace = {{
      x: visBars.map(b => b.t),
      open: visBars.map(b => b.o),
      high: visBars.map(b => b.h),
      low:  visBars.map(b => b.l),
      close: visBars.map(b => b.c),
      type: "candlestick",
      increasing: {{line: {{color: "#26a69a"}}, fillcolor: "#26a69a"}},
      decreasing: {{line: {{color: "#ef5350"}}, fillcolor: "#ef5350"}},
      showlegend: false,
      hoverinfo: "x+y",
    }};

    const traces = [candleTrace];
    const shapes = [];
    const annotations = [];

    const visStart = visBars[0].t, visEnd = visBars[visBars.length-1].t;
    const closedTrades = (snap.trades || []).filter(t =>
      t.entry_time && t.entry_time >= visStart && t.entry_time <= visEnd
    );

    // Marker entry/exit pentru trade-uri inchise vizibile
    if (closedTrades.length > 0) {{
      const xEntry = closedTrades.map(t => t.entry_time);
      const yEntry = closedTrades.map(t => t.entry_price);
      const colEntry = closedTrades.map(t => t.signal === "BUY" ? "#26a69a" : "#ef5350");
      const symEntry = closedTrades.map(t => t.signal === "BUY" ? "triangle-up" : "triangle-down");
      traces.push({{
        x: xEntry, y: yEntry, type: "scatter", mode: "markers",
        marker: {{symbol: symEntry, size: 14, color: colEntry,
                  line: {{color: "#fff", width: 2}}}},
        name: "Entry", hoverinfo: "x+y",
      }});

      const xExit = closedTrades.filter(t => t.exit_time).map(t => t.exit_time);
      const yExit = closedTrades.filter(t => t.exit_time).map(t =>
        t.outcome === "TP" ? t.tp : t.sl
      );
      const colExit = closedTrades.filter(t => t.exit_time).map(t =>
        t.outcome === "TP" ? "#26a69a" : t.outcome === "SL" ? "#ef5350" : "#ffa726"
      );
      traces.push({{
        x: xExit, y: yExit, type: "scatter", mode: "markers",
        marker: {{symbol: "x", size: 13, color: colExit,
                  line: {{color: "#000", width: 2}}}},
        name: "Exit", hoverinfo: "x+y",
      }});
    }}

    // ── Trade DESCHIS — vizibilitate maxima ─────────────────────────────────
    if (ot) {{
      // Clamp entry_time la fereastra vizibila (cand category axis, x trebuie sa fie
      // o categorie existenta — daca entry e in afara, folosim primul bar vizibil)
      let x0 = ot.entry_time;
      const visTimes = visBars.map(b => b.t);
      if (!x0 || !visTimes.includes(x0)) {{
        x0 = visStart;
      }}
      const x1 = visEnd;
      const isBuy = ot.signal === "BUY";

      // Zone fill SL (rosu transparent) si TP (verde transparent)
      shapes.push({{
        type: "rect", x0, x1, y0: ot.entry_price, y1: ot.tp,
        fillcolor: "rgba(38, 166, 154, 0.10)",
        line: {{width: 0}}, layer: "below"
      }});
      shapes.push({{
        type: "rect", x0, x1, y0: ot.sl, y1: ot.entry_price,
        fillcolor: "rgba(239, 83, 80, 0.10)",
        line: {{width: 0}}, layer: "below"
      }});

      // Linie Entry — albastru gros
      shapes.push({{type:"line", x0, x1, y0: ot.entry_price, y1: ot.entry_price,
                    line: {{color: "#64b5f6", width: 2, dash: "dot"}}}});
      // Linie SL — rosu gros
      shapes.push({{type:"line", x0, x1, y0: ot.sl, y1: ot.sl,
                    line: {{color: "#ef5350", width: 2, dash: "dash"}}}});
      // Linie TP — verde gros
      shapes.push({{type:"line", x0, x1, y0: ot.tp, y1: ot.tp,
                    line: {{color: "#66bb6a", width: 2, dash: "dash"}}}});

      // Linie verticala la entry
      shapes.push({{type:"line", x0: x0, x1: x0, yref:"paper", y0:0, y1:1,
                    line: {{color: isBuy ? "#26a69a" : "#ef5350", width: 1, dash: "dot"}}}});

      // Etichete pe partea dreapta (price tags)
      annotations.push({{
        x: x1, y: ot.entry_price, xref: "x", yref: "y",
        text: `ENTRY ${{ot.entry_price?.toFixed(5)}}`,
        showarrow: false, xanchor: "left", xshift: 5,
        font: {{color: "#64b5f6", size: 11, family: "monospace"}},
        bgcolor: "#0a1a2a", bordercolor: "#64b5f6", borderwidth: 1, borderpad: 3
      }});
      annotations.push({{
        x: x1, y: ot.tp, xref: "x", yref: "y",
        text: `TP ${{ot.tp?.toFixed(5)}}`,
        showarrow: false, xanchor: "left", xshift: 5,
        font: {{color: "#66bb6a", size: 11, family: "monospace"}},
        bgcolor: "#0a1a0a", bordercolor: "#66bb6a", borderwidth: 1, borderpad: 3
      }});
      annotations.push({{
        x: x1, y: ot.sl, xref: "x", yref: "y",
        text: `SL ${{ot.sl?.toFixed(5)}}`,
        showarrow: false, xanchor: "left", xshift: 5,
        font: {{color: "#ef5350", size: 11, family: "monospace"}},
        bgcolor: "#1a0a0a", bordercolor: "#ef5350", borderwidth: 1, borderpad: 3
      }});

      // Marker BIG la entry pe chart
      traces.push({{
        x: [x0], y: [ot.entry_price], type: "scatter", mode: "markers+text",
        marker: {{symbol: isBuy ? "triangle-up" : "triangle-down",
                  size: 22, color: isBuy ? "#26a69a" : "#ef5350",
                  line: {{color: "#fff", width: 2}}}},
        text: [isBuy ? "BUY" : "SELL"],
        textposition: isBuy ? "bottom center" : "top center",
        textfont: {{color: "#fff", size: 11, family: "Arial Black"}},
        showlegend: false, hoverinfo: "x+y",
      }});
    }}

    // ── X-axis: tick labels selectate inteligent (fiecare a 5-a/10-a bara) ──
    // Nu putem folosi date axis cu rangebreaks (goluri weekend etc.)
    // Solutie TradingView-style: type='category' → bare egal spatiate, fara goluri.
    const nVis = visBars.length;
    const labelEvery = Math.max(1, Math.floor(nVis / 8));   // ~8 etichete vizibile
    const tickvals = [];
    const ticktext = [];
    for (let i = 0; i < nVis; i += labelEvery) {{
      tickvals.push(visBars[i].t);
      // Format: "MM-DD HH:MM" sau doar "HH:MM" daca e aceeasi zi
      const ts = visBars[i].t;
      const datePart = ts.substring(5, 10);   // MM-DD
      const timePart = ts.substring(11, 16);  // HH:MM
      ticktext.push(timePart + "\\n" + datePart);
    }}

    const layout = {{
      paper_bgcolor: "#0a0a0a",
      plot_bgcolor:  "#0a0a0a",
      font: {{color: "#aaa", size: 11}},
      margin: {{l: 55, r: 110, t: 10, b: 50}},
      xaxis: {{
        type: "category",                      // ← key: bare egal spatiate, fara goluri
        rangeslider: {{visible: false}},
        gridcolor: "#1a1a1a", showgrid: true,
        tickmode: "array",
        tickvals: tickvals,
        ticktext: ticktext,
        tickfont: {{size: 9, color: "#888"}},
      }},
      yaxis: {{gridcolor: "#1a1a1a", side: "right", autorange: true}},
      shapes: shapes,
      annotations: annotations,
      showlegend: false,
    }};

    Plotly.react("live-chart", traces, layout,
                 {{displayModeBar: false, responsive: true}});
  }}

  // ── Tabel istoric (clickable rows pentru a vedea motivele) ─────────────
  const trades = (snap.trades || []).slice().reverse().slice(0, 50);
  const tbody = document.getElementById("live-history-body");
  tbody.innerHTML = trades.map((t, idx) => {{
    const sigCls = t.signal === "BUY" ? "BUY" : "SELL";
    const outCls = t.outcome === "TP" ? "TP" : t.outcome === "SL" ? "SL"
                 : t.outcome === "EXPIRED" ? "EXP" : "TO";
    const pnlCol = (t.pnl || 0) >= 0 ? "color:#66bb6a" : "color:#ef5350";
    const hasReasons = (t.reasons||[]).length > 0;
    const reasonsHtml = (t.reasons||[]).map(r => `• ${{r}}`).join("<br>");

    return `<tr style="cursor:${{hasReasons?'pointer':'default'}}"
                onclick="${{hasReasons?`toggleTradeDetails(${{idx}})`:''}}"
                title="${{hasReasons?'Click pentru detalii decizie':''}}">
      <td>${{(t.entry_time||"").substring(0,19)}} ${{hasReasons?'<span style="color:#666">▾</span>':''}}</td>
      <td>${{(t.exit_time||"").substring(0,19)||"—"}}</td>
      <td><span class="badge ${{sigCls}}">${{t.signal||""}}</span></td>
      <td><span class="badge ${{outCls}}">${{t.outcome||"—"}}</span></td>
      <td style="${{pnlCol}};font-weight:600">${{(t.pnl>=0?"+":"") + (t.pnl||0)}}$</td>
      <td>${{t.rr || "—"}}</td>
      <td>${{t.duration_bars || 0}}b</td>
    </tr>
    <tr id="td-${{idx}}" style="display:none">
      <td colspan="7" style="background:#0a1018;padding:10px 14px;border-bottom:2px solid #1565c0">
        <div style="color:#90caf9;font-weight:600;margin-bottom:6px;font-size:0.78rem">
          🔍 De ce a luat strategia aceasta decizie:
        </div>
        <div style="color:#bbb;font-size:0.78rem;line-height:1.6">${{reasonsHtml||'(fara motive salvate)'}}</div>
        <div style="margin-top:8px;color:#666;font-size:0.72rem">
          Confidence: ${{t.confidence||0}}% | Entry: ${{t.entry_price?.toFixed?.(5)||'?'}}
          | SL: ${{t.sl?.toFixed?.(5)||'?'}} | TP: ${{t.tp?.toFixed?.(5)||'?'}}
        </div>
      </td>
    </tr>`;
  }}).join("");
}}

function toggleTradeDetails(idx) {{
  const row = document.getElementById("td-" + idx);
  if (!row) return;
  row.style.display = (row.style.display === "none") ? "table-row" : "none";
}}

function startBacktest() {{
  // Daca e mod replay activat, deleaga la startReplay()
  if (_replayMode) {{
    startReplay();
    return;
  }}
  // Opreste orice poll anterior
  if (_pollInterval) {{ clearInterval(_pollInterval); _pollInterval = null; }}
  if (_livePollInterval) {{ clearInterval(_livePollInterval); _livePollInterval = null; }}

  const sym    = document.getElementById("sym-input").value.trim().toUpperCase();
  const strat  = document.getElementById("strat-select").value;
  const bars   = parseInt(document.getElementById("bars-window").value);
  const dur    = parseFloat(document.getElementById("duration").value);
  const step   = parseInt(document.getElementById("step").value);
  const risk   = parseFloat(document.getElementById("risk-usd").value);
  const tpr    = parseFloat(document.getElementById("tp-ratio").value);
  const conf   = parseFloat(document.getElementById("min-conf").value);
  const sprd   = parseFloat(document.getElementById("spread").value);
  const maxDur = parseFloat(document.getElementById("max-dur").value);

  if (!sym)   {{ alert("Introdu un simbol (ex: EURUSD)"); return; }}
  if (!strat) {{ alert("Selecteaza o strategie"); return; }}
  if (_selectedTFs.length === 0) {{ alert("Selecteaza cel putin un TF"); return; }}

  document.getElementById("run-btn").disabled = true;
  document.getElementById("progress-wrap").style.display = "block";
  setProgress(0, "Se trimite request...");

  fetch("/backtest/run", {{
    method: "POST",
    headers: {{"Content-Type": "application/json"}},
    body: JSON.stringify({{
      symbol: sym, strategy: strat, tfs: _selectedTFs,
      bars_window: bars, duration_years: dur, step: step,
      risk_dollars: risk, tp_ratio: tpr, min_confidence: conf,
      spread: sprd, max_duration_hours: maxDur,
    }})
  }})
  .then(r => r.json())
  .then(d => {{
    if (!d.ok) {{ alert("Eroare: " + d.message); resetRunBtn(); return; }}
    _currentJobId = d.job_id;
    showLiveView();              // arata vederea live cu chart + metrici + istoric
    pollStatus();
    _pollInterval = setInterval(pollStatus, 1500);
    // Live snapshot polling — mai des decat status (300ms)
    _livePollInterval = setInterval(pollLive, 300);
  }})
  .catch(e => {{ alert("Eroare retea: " + e); resetRunBtn(); }});
}}

function pollStatus() {{
  if (!_currentJobId) return;
  fetch("/backtest/status/" + _currentJobId)
  .then(r => r.json())
  .then(d => {{
    setProgress(d.progress || 0, d.message || "");
    if (d.status === "done") {{
      clearInterval(_pollInterval);
      // un ultim live poll, apoi load rezultatul final
      setTimeout(() => {{
        if (_livePollInterval) {{ clearInterval(_livePollInterval); _livePollInterval = null; }}
        loadResult(_currentJobId);
      }}, 400);
    }} else if (d.status === "error") {{
      clearInterval(_pollInterval);
      if (_livePollInterval) {{ clearInterval(_livePollInterval); _livePollInterval = null; }}
      hideLiveView();
      showError(d.message || "Eroare necunoscuta");
    }} else if (d.status === "cancelled") {{
      clearInterval(_pollInterval);
      if (_livePollInterval) {{ clearInterval(_livePollInterval); _livePollInterval = null; }}
      // NU mai ascundem live view la cancel — pastram chart-ul cu trade-urile de pana acum
      const pulseEl = document.querySelector(".live-pulse");
      if (pulseEl) {{ pulseEl.style.background = "#ffa726"; pulseEl.style.animation = "none"; }}
      const titleEl = document.getElementById("live-title");
      if (titleEl) titleEl.textContent = "⏹ Oprit — chart cu trade-uri partiale";
      setProgress(0, "Oprit de utilizator. Chart pastrat.");
      resetRunBtn();
    }}
  }});
}}

function cancelJob() {{
  if (!_currentJobId) return;
  fetch("/backtest/cancel/" + _currentJobId, {{method:"POST"}});
  clearInterval(_pollInterval);
  setProgress(0, "Oprit.");
  resetRunBtn();
}}

function setProgress(pct, msg) {{
  document.getElementById("progress-bar").style.width = pct + "%";
  if (msg) document.getElementById("progress-msg").textContent = msg;
}}

function showError(msg) {{
  const eb = document.getElementById("error-box");
  eb.textContent = "❌ " + msg;
  eb.style.display = "block";
  document.getElementById("progress-bar").style.width = "0%";
  document.getElementById("progress-msg").textContent = "Eroare — vezi detalii mai jos";
  document.getElementById("cancel-btn").style.display = "none";
  document.getElementById("close-err-btn").style.display = "block";
  document.getElementById("run-btn").disabled = false;
  // progress-wrap ramine vizibil cu eroarea
}}

function closeError() {{
  document.getElementById("error-box").style.display = "none";
  document.getElementById("cancel-btn").style.display = "block";
  document.getElementById("close-err-btn").style.display = "none";
  document.getElementById("progress-wrap").style.display = "none";
}}

function resetRunBtn() {{
  document.getElementById("run-btn").disabled = false;
  document.getElementById("error-box").style.display = "none";
  document.getElementById("cancel-btn").style.display = "block";
  document.getElementById("close-err-btn").style.display = "none";
  document.getElementById("progress-wrap").style.display = "none";
}}

// ── Load & render result ──────────────────────────────────────────────────────
function loadResult(jobId) {{
  fetch("/backtest/result/" + jobId)
  .then(r => r.json())
  .then(data => {{
    resetRunBtn();
    // NU mai ascundem live view-ul — vrem sa ramana vizibil chart-ul cu trade-uri.
    // Doar oprim polling-ul si schimbam header-ul ("LIVE" → "FINAL").
    if (_livePollInterval) {{ clearInterval(_livePollInterval); _livePollInterval = null; }}
    const pulseEl = document.querySelector(".live-pulse");
    if (pulseEl) {{ pulseEl.style.background = "#66bb6a"; pulseEl.style.animation = "none"; }}
    const titleEl = document.getElementById("live-title");
    if (titleEl) titleEl.textContent = "✓ Backtest complet — chart cu toate trade-urile";

    renderResult(data, true);  // true = inserează rezultatul SUB live view (nu inlocuieste)
    refreshSavedList();
  }})
  .catch(e => {{ alert("Nu am putut incarca rezultatul: " + e); }});
}}

function refreshSavedList() {{
  fetch("/backtest/saved_list")
  .then(r => r.json())
  .then(list => {{
    SAVED.splice(0, SAVED.length, ...list);
    renderSaved();
  }})
  .catch(() => {{}});
}}

function deleteSaved(jobId, event) {{
  event.stopPropagation();
  if (!confirm("Stergi acest rezultat?")) return;
  fetch("/backtest/delete/" + jobId, {{method:"POST"}})
  .then(() => refreshSavedList())
  .catch(() => {{}});
}}

function renderResult(data, keepLive=false) {{
  const m = data.metrics;
  document.getElementById("empty-state").style.display = "none";
  document.getElementById("result-area").style.display = "block";
  if (!keepLive) {{
    document.getElementById("live-wrap").style.display = "none";
  }}

  const pfColor  = m.profit_factor >= 1.3 ? "green" : m.profit_factor >= 1.0 ? "orange" : "red";
  const wrColor  = m.win_rate >= 55 ? "green" : m.win_rate >= 40 ? "orange" : "red";
  const pnlColor = m.net_pnl >= 0 ? "green" : "red";
  const shColor  = m.sharpe >= 1.0 ? "green" : m.sharpe >= 0 ? "orange" : "red";

  const stratDef = STRATS.find(s => s.key === data.strategy) || {{}};
  const color = stratDef.color || "#64b5f6";

  document.getElementById("result-area").innerHTML = `
    <div class="result-header">
      <h2 style="color:${{color}}">${{stratDef.icon||"📊"}} ${{data.symbol}} — ${{data.strategy.toUpperCase()}}
        <span style="color:#555;font-size:0.8rem;font-weight:400"> | ${{data.tfs.join(", ")}}</span>
      </h2>
      <div class="result-meta">${{data.from_dt}} → ${{data.to_dt}} | ${{data.total_bars}} bare | pas ${{data.step}}</div>
    </div>

    <div class="metrics-grid">
      <div class="metric-card"><div class="val">${{m.total_trades}}</div><div class="lbl">Total trade-uri</div></div>
      <div class="metric-card ${{wrColor}}"><div class="val">${{m.win_rate}}%</div><div class="lbl">Win Rate</div></div>
      <div class="metric-card ${{pfColor}}"><div class="val">${{m.profit_factor}}</div><div class="lbl">Profit Factor</div></div>
      <div class="metric-card ${{pnlColor}}"><div class="val">${{m.net_pnl > 0 ? "+" : ""}}${{m.net_pnl}}$</div><div class="lbl">Net P&L</div></div>
      <div class="metric-card red"><div class="val">-${{m.max_drawdown}}$</div><div class="lbl">Max Drawdown</div></div>
      <div class="metric-card blue"><div class="val">${{m.avg_rr}}:1</div><div class="lbl">Avg R:R efectiv</div></div>
      <div class="metric-card ${{shColor}}"><div class="val">${{m.sharpe}}</div><div class="lbl">Sharpe Ratio</div></div>
      <div class="metric-card"><div class="val">${{m.max_consec_loss}}</div><div class="lbl">Max pierderi consec.</div></div>
      <div class="metric-card"><div class="val" style="font-size:1rem">${{m.wins}}W / ${{m.losses}}L / ${{(m.timeouts||0)+(m.expires||0)}}T</div><div class="lbl">TP / SL / Timeout</div></div>
    </div>

    <div class="chart-wrap" id="equity-chart"></div>

    <h3 style="margin-bottom:8px;color:#777">TRADE-URI (${{data.trades.length}})</h3>
    <table class="trade-table">
      <thead><tr>
        <th onclick="sortTable('entry_time')">Intrare</th>
        <th onclick="sortTable('exit_time')">Iesire</th>
        <th>Semnal</th>
        <th>Outcome</th>
        <th onclick="sortTable('pnl')">P&L $</th>
        <th onclick="sortTable('rr')">R:R</th>
        <th onclick="sortTable('confidence')">Confidence</th>
        <th onclick="sortTable('duration_bars')">Durata (bare)</th>
      </tr></thead>
      <tbody id="trades-tbody"></tbody>
    </table>
  `;

  // Equity curve
  const dates  = m.equity_dates.length > 0 ? m.equity_dates : m.equity_curve.map((_,i) => i);
  const eq     = m.equity_curve;
  const colors = eq.map((v,i) => i === 0 ? "#555" : v >= eq[i-1] ? "#66bb6a" : "#ef5350");
  Plotly.newPlot("equity-chart", [{{
    x: dates, y: eq, type: "scatter", mode: "lines",
    line: {{color: color, width: 2}},
    fill: "tozeroy", fillcolor: color.replace(")", ",0.08)").replace("rgb","rgba"),
    name: "Equity",
  }}], {{
    paper_bgcolor: "#111", plot_bgcolor: "#111",
    margin: {{t:10, b:30, l:55, r:10}},
    font: {{color:"#666", size:11}},
    xaxis: {{gridcolor:"#1a1a1a", zeroline:false}},
    yaxis: {{gridcolor:"#1a1a1a", zeroline:false}},
    showlegend: false,
  }}, {{responsive:true, displayModeBar:false}});

  // Trades
  _currentTrades = data.trades;
  _sortKey = "entry_time"; _sortAsc = true;
  renderTradesTable();
}}

let _currentTrades = [];
let _sortKey = "entry_time";
let _sortAsc = true;

function sortTable(key) {{
  if (_sortKey === key) _sortAsc = !_sortAsc;
  else {{ _sortKey = key; _sortAsc = true; }}
  renderTradesTable();
}}

function renderTradesTable() {{
  const sorted = [..._currentTrades].sort((a,b) => {{
    const va = a[_sortKey], vb = b[_sortKey];
    if (va == null) return 1; if (vb == null) return -1;
    return _sortAsc ? (va > vb ? 1 : -1) : (va < vb ? 1 : -1);
  }});
  const tbody = document.getElementById("trades-tbody");
  if (!tbody) return;
  tbody.innerHTML = sorted.map(t => {{
    const pnlCol = t.pnl >= 0 ? "#66bb6a" : "#ef5350";
    const badge  = t.outcome === "TP" ? "TP" : t.outcome === "SL" ? "SL" : t.outcome === "TIMEOUT" ? "TO" : "EXP";
    const badgeTxt = t.outcome === "TIMEOUT" ? "TIMEOUT" : t.outcome;
    const sigBadge = t.signal === "BUY" ? "BUY" : "SELL";
    return `<tr>
      <td>${{(t.entry_time||"").slice(0,16)}}</td>
      <td>${{(t.exit_time||"").slice(0,16)}}</td>
      <td><span class="badge ${{sigBadge}}">${{t.signal}}</span></td>
      <td><span class="badge ${{badge}}">${{badgeTxt}}</span></td>
      <td style="color:${{pnlCol}};font-weight:700">${{t.pnl >= 0 ? "+" : ""}}${{t.pnl}}$</td>
      <td>${{t.rr ? t.rr.toFixed(2) : "—"}}</td>
      <td>${{t.confidence ? t.confidence.toFixed(0) + "%" : "—"}}</td>
      <td>${{t.duration_bars ?? "—"}}</td>
    </tr>`;
  }}).join("");
}}

// ── Saved list ─────────────────────────────────────────────────────────────────
function renderSaved() {{
  const el = document.getElementById("saved-list");
  if (!SAVED.length) {{
    el.innerHTML = "<div style='color:#444;font-size:0.78rem'>Nicio rulare salvata</div>";
    return;
  }}
  el.innerHTML = SAVED.map(s => {{
    const pnlStr = s.net_pnl >= 0 ? `+${{s.net_pnl}}$` : `${{s.net_pnl}}$`;
    const pnlCol = s.net_pnl >= 0 ? "#66bb6a" : "#ef5350";
    return `<div class="saved-item" onclick="loadResult('${{s.job_id}}')" style="position:relative">
      <button onclick="deleteSaved('${{s.job_id}}',event)"
              style="position:absolute;top:6px;right:6px;background:transparent;border:none;
                     color:#555;cursor:pointer;font-size:0.85rem;padding:0 4px;line-height:1"
              title="Sterge">✕</button>
      <div class="label" style="padding-right:20px">${{s.label}}</div>
      <div class="meta">
        ${{s.from_dt}} → ${{s.to_dt}} |
        ${{s.trades}} trades |
        <span class="wr">WR ${{s.win_rate}}%</span> |
        <span class="pf">PF ${{s.pf}}</span> |
        <span style="color:${{pnlCol}}">${{pnlStr}}</span>
      </div>
    </div>`;
  }}).join("");
}}
</script>
</body>
</html>"""
