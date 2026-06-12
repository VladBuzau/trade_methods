"""
ai_agent/trader/scanner.py — buclă proprie de scanare AI Trader.

Independent de scanner-ul ChartVisualizer. Ruleaza intr-un thread daemon care:
  1. La fiecare scan_interval_sec, parcurge watchlist [(symbol, tf), ...]
  2. Pentru fiecare pereche:
     a. Fetch ultimele N bare (via MT5)
     b. Calculeaza features (cele 41)
     c. predict_dual(features, symbol, tf, horizon)
     d. Daca BUY/SELL cu confidence >= min_confidence → executa paper trade
  3. Verifica trade-urile deschise pt SL/TP hit
  4. Loghează predictiile + actualizeaza state runtime

Pornit/oprit din UI via state.update({"running": bool}).
"""
from __future__ import annotations
import time
import logging
import threading
from datetime import datetime, timezone

import numpy as np

from ai_agent.trader import state, paper_trader

log = logging.getLogger(__name__)


def _atr(df, period=14):
    h, l, c = df["high"].values, df["low"].values, df["close"].values
    if len(c) < period + 1:
        return 0.0
    tr = np.maximum(h[1:] - l[1:],
                    np.maximum(np.abs(h[1:] - c[:-1]),
                               np.abs(l[1:] - c[:-1])))
    return float(np.mean(tr[-period:]))


def _scan_one(symbol: str, tf: str, settings: dict) -> dict | None:
    """Ruleaza un scan complet pe (symbol, tf). Returneaza dict cu rezultatul."""
    try:
        import MetaTrader5 as mt5
        from ai_agent.features.builder import compute_features_for_bar
        from ai_agent.models.inference import predict_dual

        # 1. Fetch
        _TF_MAP = {
            "M1": mt5.TIMEFRAME_M1, "M5": mt5.TIMEFRAME_M5,
            "M15": mt5.TIMEFRAME_M15, "M30": mt5.TIMEFRAME_M30,
            "H1": mt5.TIMEFRAME_H1, "H4": mt5.TIMEFRAME_H4,
            "D1": mt5.TIMEFRAME_D1,
        }
        tf_const = _TF_MAP.get(tf)
        if tf_const is None:
            return None
        rates = mt5.copy_rates_from_pos(symbol, tf_const, 0, 300)
        if rates is None or len(rates) < 100:
            return {"symbol": symbol, "tf": tf, "error": "no data"}

        import pandas as pd
        df = pd.DataFrame(rates)
        df.rename(columns={"tick_volume": "volume"}, inplace=True)
        if "volume" not in df.columns and "real_volume" in df.columns:
            df["volume"] = df["real_volume"]
        df_norm = df.reset_index(drop=True).copy()

        # 2. Features
        feats = compute_features_for_bar(df_norm, len(df_norm) - 1)
        if feats is None:
            return {"symbol": symbol, "tf": tf, "error": "no features"}

        # 3. Predict
        pred = predict_dual(feats, symbol, tf, horizon=settings["horizon_h"])
        if pred is None:
            return {"symbol": symbol, "tf": tf, "error": "no model trained"}

        current_price = float(df["close"].iloc[-1])
        atr = _atr(df, 14)

        result = {
            "ts":         int(datetime.now(timezone.utc).timestamp()),
            "symbol":     symbol,
            "tf":         tf,
            "price":      round(current_price, 5),
            "atr":        round(atr, 5),
            "signal":     pred.get("signal"),
            "confidence": pred.get("confidence", 0),
            "prob_long":  pred.get("prob_long"),
            "prob_short": pred.get("prob_short"),
            "agreement":  pred.get("agreement", False),
        }

        # 4. Decizie executie
        rt = state.get_runtime()
        n_open = len(rt["open_paper_trades"])
        if (pred["signal"] in ("BUY", "SELL")
                and pred["confidence"] >= settings["min_confidence"]
                and n_open < settings["max_open_trades"]):
            # Evita duplicate pe acelasi (symbol, tf)
            already_open = any(
                t["symbol"] == symbol and t["tf"] == tf
                for t in rt["open_paper_trades"]
            )
            if not already_open:
                sl_dist = atr * 1.5
                tp_dist = atr * 1.5   # RR 1:1
                if pred["signal"] == "BUY":
                    sl = current_price - sl_dist
                    tp = current_price + tp_dist
                else:
                    sl = current_price + sl_dist
                    tp = current_price - tp_dist
                model_id = f"{symbol}_{tf}_h{settings['horizon_h']}"

                # Paper trade (intotdeauna logat, chiar daca real MT5 e activ)
                if settings.get("paper_trading", True):
                    trade = paper_trader.open_trade(
                        symbol, tf, pred["signal"],
                        current_price, round(sl, 5), round(tp, 5),
                        pred["confidence"], model_id=model_id,
                    )
                    result["trade_opened"] = trade["ticket"]

                # Real MT5 execution (opt-in)
                if settings.get("real_mt5_execute", False):
                    from ai_agent.trader.mt5_executor import open_real_trade
                    r_real = open_real_trade(
                        symbol, pred["signal"],
                        round(sl, 5), round(tp, 5),
                        risk_dollars=settings.get("risk_dollars", 50.0),
                        model_id=model_id,
                    )
                    if r_real.get("ok"):
                        result["real_mt5_opened"] = True
                        log.info(f"[REAL MT5] AI trade opened: {symbol} {pred['signal']}")
                    else:
                        result["real_mt5_error"] = r_real.get("error", "unknown")

        # Returneaza si OHLC pentru check_open_trades
        result["_ohlc"] = {
            "high":  float(df["high"].iloc[-1]),
            "low":   float(df["low"].iloc[-1]),
            "close": float(df["close"].iloc[-1]),
        }
        return result
    except Exception as e:
        log.error(f"_scan_one {symbol}/{tf}: {e}")
        return {"symbol": symbol, "tf": tf, "error": str(e)}


def scanner_loop():
    """Thread principal — ruleaza cat timp state.running == True."""
    log.info("AI Trader scanner pornit")
    while True:
        try:
            settings = state.load()
            if not settings.get("running", False):
                time.sleep(2)
                continue

            wl = settings.get("watchlist", [])
            if not wl:
                time.sleep(5)
                continue

            price_lookup = {}
            for sym_tf in wl:
                if len(sym_tf) != 2:
                    continue
                sym, tf = sym_tf
                result = _scan_one(sym, tf, settings)
                if result and not result.get("error"):
                    state.push_prediction({
                        k: v for k, v in result.items() if not k.startswith("_")
                    })
                    if "_ohlc" in result:
                        price_lookup[(sym, tf)] = result["_ohlc"]
                elif result and result.get("error"):
                    # Logăm si erorile (utile pentru debug)
                    state.push_prediction(result)

            # Check trade-uri deschise
            if price_lookup:
                paper_trader.check_open_trades(price_lookup)

            state.inc_scan()
            time.sleep(max(5, settings.get("scan_interval_sec", 60)))
        except Exception as e:
            log.error(f"scanner_loop: {e}")
            time.sleep(10)


_started = False


def ensure_scanner_started():
    """Porneste thread-ul daca nu ruleaza (idempotent)."""
    global _started
    if _started:
        return
    _started = True
    t = threading.Thread(target=scanner_loop, daemon=True, name="ai_trader_scanner")
    t.start()
    state.set_runtime(scanner_thread=t.name)
    log.info("AI Trader scanner thread started")
