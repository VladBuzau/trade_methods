"""
ChartVisualizer — grafic + auto-trader integrat
"""

from flask import Flask, render_template_string, request, Response, send_file, session, redirect, url_for
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import json, logging, math, os
from datetime import datetime, timezone, timedelta
from scipy.signal import find_peaks

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

# ── Telegram Notifier ─────────────────────────────────────────────────────────
import notifier as _tg

# ── MetaTrader5 ───────────────────────────────────────────────────────────────
try:
    import MetaTrader5 as mt5
    MT5_AVAILABLE = mt5.initialize()
    log.info("MT5 conectat" if MT5_AVAILABLE else "MT5 pornit dar nu conectat")
except ImportError:
    mt5 = None
    MT5_AVAILABLE = False
    log.info("MT5 nu e instalat")

app = Flask(__name__)
app.secret_key = "cv_secret_2024_xK9mP"  # schimba cu ceva random

# Porneste polling-ul Telegram in background (nu blocheaza serverul)
_tg.start_polling()

# ── Auth config ───────────────────────────────────────────────────────────────
import hashlib, functools

def _load_auth_config():
    cfg_path = os.path.join(os.path.dirname(__file__), "config.json")
    if os.path.exists(cfg_path):
        try:
            with open(cfg_path, encoding="utf-8") as f:
                cfg = json.load(f)
            if "users" in cfg:
                return cfg["users"]
        except Exception:
            pass
    # fallback default
    return {"admin": hashlib.sha256("admin123".encode()).hexdigest()}

def _check_password(username, password):
    users = _load_auth_config()
    hashed = hashlib.sha256(password.encode()).hexdigest()
    return users.get(username) == hashed

def _is_direct_localhost() -> bool:
    """
    True doar pentru conexiuni directe la 127.0.0.1/::1, NU prin cloudflared.
    Cloudflare tunel forwardeaza si el de pe 127.0.0.1, deci ne uitam dupa
    headere CF-* sau host *.trycloudflare.com pentru a-l deosebi.
    """
    if request.remote_addr not in ("127.0.0.1", "::1", "localhost"):
        return False
    if any(request.headers.get(h) for h in
           ("CF-Connecting-IP", "CF-Ray", "CF-Visitor", "X-Forwarded-For", "X-Real-IP")):
        return False
    host = (request.headers.get("Host", "") or "").lower()
    if "trycloudflare.com" in host or "cloudflare" in host:
        return False
    return True


def login_required(f):
    @functools.wraps(f)
    def decorated(*args, **kwargs):
        # Bypass auth pentru conexiuni directe localhost (NU prin cloudflare)
        if _is_direct_localhost():
            return f(*args, **kwargs)
        if not session.get("logged_in"):
            return redirect("/login?next=" + request.path)
        return f(*args, **kwargs)
    return decorated


# ── Constante imutabile centralizate in config.py ───────────────────────────
# (timeframes, symbols, spread limits, strategy priorities, FTMO/performance)
from config import (
    MT5_TF, ALL_TFS, MULTI_BARS,
    SYMBOLS, SYMBOLS_CRYPTO,
    MAX_SPREAD_PIPS as _CFG_MAX_SPREAD_PIPS,
    MAX_SPREAD_DEFAULT as _CFG_MAX_SPREAD_DEFAULT,
    STRATEGY_PRIORITY as _CFG_STRATEGY_PRIORITY,
    ADX_MIN as _CFG_ADX_MIN,
    FTMO_NEWS_BLOCK_MIN as _CFG_FTMO_NEWS_BLOCK_MIN,
    PERF_MIN_TRADES as _CFG_PERF_MIN_TRADES,
    PERF_LOOKBACK as _CFG_PERF_LOOKBACK,
    PERF_MIN_PF as _CFG_PERF_MIN_PF,
)

RISK_DOLLARS    = 50.0   # valoare runtime, poate fi modificata din UI
RISK_PCT        = 1.0    # % din equity pentru position sizing dinamic (1% default)
USE_RISK_PCT    = False  # True = foloseste % equity; False = suma fixa RISK_DOLLARS

# Spread limits — importate din config.py (vezi MAX_SPREAD_PIPS)
MAX_SPREAD_PIPS = _CFG_MAX_SPREAD_PIPS
MAX_SPREAD_DEFAULT = _CFG_MAX_SPREAD_DEFAULT

# Drawdown Reducer: reduce riscul dupa pierderi consecutive
_consecutive_losses  = 0
_drawdown_factor     = 1.0   # 1.0 = normal, 0.5 = jumatate risc, 0.25 = sfert
_perf_log: list[dict] = []   # log performanta trade-uri {strategy, symbol, pnl, win}

# P4: Auto-disable set — strategii dezactivate automat din cauza performantei slabe
_auto_disabled_strategies: set = set()

# P4: Prioritate strategii — importat din config.py
STRATEGY_PRIORITY = _CFG_STRATEGY_PRIORITY

# Performance thresholds — importate din config.py
PERF_MIN_TRADES = _CFG_PERF_MIN_TRADES
PERF_LOOKBACK   = _CFG_PERF_LOOKBACK
PERF_MIN_PF     = _CFG_PERF_MIN_PF

# P4: Max Daily Drawdown — daca pierdem > 3% din equity azi, oprim tradingul
MAX_DAILY_DRAWDOWN_PCT = 100.0  # DEMO: dezactivat (schimba la 3.0 pe cont live)
_daily_pnl: dict[str, float] = {}   # {"2024-01-15": -120.50, ...}
_daily_drawdown_halted = False       # True = trading oprit pe azi


def _get_today_key() -> str:
    return __import__("datetime").datetime.now().strftime("%Y-%m-%d")


def record_daily_pnl(pnl: float):
    """Inregistreaza un PnL realizat azi. Apelat dupa fiecare trade inchis."""
    global _daily_drawdown_halted
    today = _get_today_key()
    _daily_pnl[today] = _daily_pnl.get(today, 0.0) + pnl
    # Verifica daca depasim limita
    if not _daily_drawdown_halted:
        check_daily_drawdown_halt()


def get_today_pnl() -> float:
    """Returneaza PnL-ul total de azi."""
    return _daily_pnl.get(_get_today_key(), 0.0)


def check_daily_drawdown_halt() -> tuple[bool, str]:
    """
    Verifica daca trebuie oprit tradingul pe azi.
    Citeste direct din MT5 tranzactiile inchise azi (realized PnL).
    Returneaza (True, msg) daca trading-ul e oprit.
    """
    global _daily_drawdown_halted
    if not MT5_AVAILABLE or mt5 is None:
        _daily_drawdown_halted = False
        return False, "MT5 indisponibil"

    try:
        import datetime as _dt
        now = _dt.datetime.now()
        today_start = _dt.datetime(now.year, now.month, now.day, 0, 0, 0)

        account = mt5.account_info()
        if account is None or account.equity <= 0:
            _daily_drawdown_halted = False
            return False, "Nu s-a putut obtine equity"

        # Citeste toate deal-urile inchise de azi (tip OUT = 1)
        deals = mt5.history_deals_get(today_start, now)
        if deals is None:
            _daily_drawdown_halted = False
            return False, "Nu s-au putut citi deal-urile"

        # Sumeaza profit-ul net al deal-urilor de iesire (entry_type == 1 = out)
        today_pnl = sum(
            d.profit + d.swap + d.commission
            for d in deals
            if d.entry == 1  # DEAL_ENTRY_OUT
        )

        if today_pnl >= 0:
            _daily_drawdown_halted = False
            return False, f"PnL azi: +{today_pnl:.2f}$ — ok"

        loss_pct = abs(today_pnl) / account.equity * 100
        if loss_pct >= MAX_DAILY_DRAWDOWN_PCT:
            _daily_drawdown_halted = True
            return True, (
                f"MAX DAILY DRAWDOWN ATINS: -{loss_pct:.1f}% din equity "
                f"(pierdere {today_pnl:.2f}$, limita {MAX_DAILY_DRAWDOWN_PCT}%) — "
                f"TRADING OPRIT PE AZI"
            )

        _daily_drawdown_halted = False
        return False, f"PnL azi: {today_pnl:.2f}$ ({loss_pct:.1f}% din equity, limita {MAX_DAILY_DRAWDOWN_PCT}%)"

    except Exception as exc:
        _daily_drawdown_halted = False
        return False, f"Eroare drawdown check: {exc}"


def is_daily_drawdown_halted() -> bool:
    """True daca tradingul e oprit pe azi din cauza drawdown-ului zilnic."""
    # Reset automat la miezul noptii
    today = _get_today_key()
    if today not in _daily_pnl:
        global _daily_drawdown_halted
        _daily_drawdown_halted = False
    return _daily_drawdown_halted


# P4: Correlation Guard — blocheaza al doilea trade pe simboluri corelate (corr > 0.8)
CORR_LOOKBACK   = 20   # bare H1 pentru calculul corelatiei
CORR_THRESHOLD  = 0.8  # corelatie maxima acceptata intre doua pozitii deschise


_CRYPTO_SYMBOLS = {
    "BTCUSD","ETHUSD","XRPUSD","SOLUSD","BNBUSD","DOGEUSD",
    "ADAUSD","AVAXUSD","LINKUSD","LTCUSD",
}

def _check_correlation_guard(new_symbol: str, new_signal: str) -> tuple[bool, str]:
    """
    Verifica daca noul simbol e puternic corelat cu o pozitie deja deschisa
    in ACEEASI directie.
    Returneaza (True, msg) daca e blocat, (False, "") daca e ok.
    Crypto e dezactivat — toate crypto sunt corelate prin natura lor.
    """
    try:
        # Crypto sunt corelate structural — guard dezactivat pe ele
        if new_symbol in _CRYPTO_SYMBOLS:
            return False, ""
        if not MT5_AVAILABLE or mt5 is None:
            return False, ""
        open_positions = mt5.positions_get() or []
        if not open_positions:
            return False, ""

        # Simbolurile deschise + directia lor
        open_syms = {}
        for pos in open_positions:
            direction = "BUY" if pos.type == 0 else "SELL"
            open_syms[pos.symbol] = direction

        if not open_syms:
            return False, ""

        # Fetch close prices pentru noul simbol
        df_new, _ = fetch(new_symbol, "H1", CORR_LOOKBACK + 5)
        if df_new is None or len(df_new) < CORR_LOOKBACK:
            return False, ""

        new_closes = df_new["close"].iloc[-CORR_LOOKBACK:].values

        for open_sym, open_dir in open_syms.items():
            if open_sym == new_symbol:
                continue
            # Corelatia conteaza numai daca directiile sunt aceleasi
            if open_dir != new_signal:
                continue
            try:
                df_open, _ = fetch(open_sym, "H1", CORR_LOOKBACK + 5)
                if df_open is None or len(df_open) < CORR_LOOKBACK:
                    continue
                open_closes = df_open["close"].iloc[-CORR_LOOKBACK:].values
                if len(open_closes) != len(new_closes):
                    min_len = min(len(open_closes), len(new_closes))
                    open_closes = open_closes[-min_len:]
                    new_closes_cut = new_closes[-min_len:]
                else:
                    new_closes_cut = new_closes
                corr = float(np.corrcoef(new_closes_cut, open_closes)[0, 1])
                if corr > CORR_THRESHOLD:
                    return True, (
                        f"Correlation Guard: {new_symbol} corelat {corr:.2f} cu {open_sym} "
                        f"(ambele {new_signal}, threshold {CORR_THRESHOLD}) — skip"
                    )
            except Exception:
                continue

        return False, ""
    except Exception:
        return False, ""


def record_perf(strategy: str, symbol: str, signal: str, pnl: float, win: bool):
    """
    Inregistreaza rezultatul unui trade in log-ul de performanta.
    Apeleaza si _update_drawdown_reducer().
    """
    global _perf_log
    entry = {
        "ts":       __import__("datetime").datetime.now().isoformat(),
        "strategy": strategy,
        "symbol":   symbol,
        "signal":   signal,
        "pnl":      round(pnl, 2),
        "win":      win,
    }
    _perf_log.append(entry)
    _perf_log = _perf_log[-500:]  # pastreaza ultimele 500
    _update_drawdown_reducer(win)


def get_perf_stats(strategy: str = None, symbol: str = None) -> dict:
    """
    Calculeaza statistici de performanta: win_rate, avg_pnl, profit_factor.
    Filtreaza optional dupa strategy si/sau symbol.
    """
    log_filtered = _perf_log
    if strategy:
        log_filtered = [e for e in log_filtered if e["strategy"] == strategy]
    if symbol:
        log_filtered = [e for e in log_filtered if e["symbol"] == symbol]

    # Ulitmele PERF_LOOKBACK trade-uri
    recent = log_filtered[-PERF_LOOKBACK:]
    if not recent:
        return {"trades": 0, "win_rate": 0.0, "profit_factor": 0.0, "avg_pnl": 0.0}

    wins     = [e for e in recent if e["win"]]
    losses   = [e for e in recent if not e["win"]]
    gross_p  = sum(e["pnl"] for e in wins)
    gross_l  = abs(sum(e["pnl"] for e in losses)) or 1e-10
    pf       = round(gross_p / gross_l, 2)
    wr       = round(len(wins) / len(recent) * 100, 1)
    avg_pnl  = round(sum(e["pnl"] for e in recent) / len(recent), 2)

    return {
        "trades":        len(recent),
        "wins":          len(wins),
        "losses":        len(losses),
        "win_rate":      wr,
        "profit_factor": pf,
        "avg_pnl":       avg_pnl,
    }


def check_auto_disable(strategy: str) -> bool:
    """
    Verifica daca o strategie trebuie dezactivata automat.
    Criteriu: PF < PERF_MIN_PF pe ultimele PERF_LOOKBACK trade-uri
              si minim PERF_MIN_TRADES trade-uri disponibile.
    Returneaza True daca strategia trebuie dezactivata.
    """
    stats = get_perf_stats(strategy)
    if stats["trades"] < PERF_MIN_TRADES:
        return False  # nu avem suficiente date
    if stats["profit_factor"] < PERF_MIN_PF:
        _auto_disabled_strategies.add(strategy)
        return True
    # Daca si-a revenit, scoate din set
    _auto_disabled_strategies.discard(strategy)
    return False


def is_auto_disabled(strategy: str) -> bool:
    """Verifica daca strategia e in lista auto-disabled."""
    return strategy in _auto_disabled_strategies


def _check_daily_atr_overextension(symbol: str) -> tuple[bool, str]:
    """
    P4: Verifica daca ziua curenta e deja overextended.
    Returneaza (True, msg) daca piata e overextended si ar trebui sa evitam noi intrari.
    Criteriu: today_range > 1.5 × ATR14 (media ultimelor 14 zile)
    """
    try:
        df_d1, _ = fetch(symbol, "D1", 20)
        if df_d1 is None or len(df_d1) < 15:
            return False, ""
        today_high  = float(df_d1["high"].iloc[-1])
        today_low   = float(df_d1["low"].iloc[-1])
        today_range = today_high - today_low
        atr14 = float((df_d1["high"] - df_d1["low"]).rolling(14).mean().iloc[-2])
        if atr14 <= 0:
            return False, ""
        ratio = today_range / atr14
        if ratio > 1.5:
            return True, f"ATR zilnic overextended: {ratio:.2f}×ATR14 (>{1.5}×) — risc ridicat"
        return False, f"ATR zilnic normal: {ratio:.2f}×ATR14"
    except Exception:
        return False, ""


def resolve_strategy_conflict(strategy_results: dict) -> dict:
    """
    P4: Rezolva conflicte intre strategii care dau semnale opuse pe acelasi simbol.
    Input: {strat_key: result_dict}
    Output: {strat_key: result_dict} — strategiile perdante sunt setate pe HOLD.
    """
    buy_strats  = {k: v for k, v in strategy_results.items() if v.get("signal") == "BUY"}
    sell_strats = {k: v for k, v in strategy_results.items() if v.get("signal") == "SELL"}

    if not buy_strats or not sell_strats:
        return strategy_results  # fara conflict

    # Conflict detectat — selecteaza semnalul cu prioritate mai mica (= mai important)
    all_active = {**buy_strats, **sell_strats}
    winner = min(all_active.keys(), key=lambda k: STRATEGY_PRIORITY.get(k, 99))
    winner_signal = strategy_results[winner]["signal"]

    resolved = {}
    for k, v in strategy_results.items():
        if v.get("signal") not in ("BUY", "SELL"):
            resolved[k] = v
            continue
        if v["signal"] != winner_signal:
            # Suprascrie cu HOLD — reseteaza si confidence ca sa nu para contradictie
            import copy
            v2 = copy.copy(v)
            v2["signal"]     = "HOLD"
            v2["confidence"] = 0.0
            v2["best_tf"]    = None
            v2["justification"] = [
                f"Conflict cu {winner} ({winner_signal}) — anulat (prioritate mai mica)"
            ] + (v.get("justification") or [])
            resolved[k] = v2
        else:
            resolved[k] = v

    return resolved

def _load_risk_config():
    """Incarca setarile de risc din config.json daca exista."""
    global RISK_DOLLARS, RISK_PCT, USE_RISK_PCT, MAX_SPREAD_PIPS, SL_ATR_MULT, MAX_LOT_GLOBAL
    cfg_path = os.path.join(os.path.dirname(__file__), "config.json")
    try:
        if os.path.exists(cfg_path):
            with open(cfg_path, encoding="utf-8") as f:
                cfg = json.load(f)
            if "risk_dollars" in cfg:
                RISK_DOLLARS = float(cfg["risk_dollars"])
            if "risk_pct" in cfg:
                RISK_PCT = float(cfg["risk_pct"])
            if "use_risk_pct" in cfg:
                USE_RISK_PCT = bool(cfg["use_risk_pct"])
            if "max_spread_pips" in cfg:
                MAX_SPREAD_PIPS.update(cfg["max_spread_pips"])
            if "sl_atr_mult" in cfg:
                SL_ATR_MULT = float(cfg["sl_atr_mult"])
            if "max_lot_global" in cfg:
                MAX_LOT_GLOBAL = float(cfg["max_lot_global"])
    except Exception:
        pass

def _save_risk_config():
    """Salveaza setarile de risc in config.json."""
    cfg_path = os.path.join(os.path.dirname(__file__), "config.json")
    try:
        cfg = {}
        if os.path.exists(cfg_path):
            with open(cfg_path, encoding="utf-8") as f:
                cfg = json.load(f)
        cfg["risk_dollars"]    = RISK_DOLLARS
        cfg["risk_pct"]        = RISK_PCT
        cfg["use_risk_pct"]    = USE_RISK_PCT
        cfg["max_spread_pips"] = MAX_SPREAD_PIPS
        cfg["sl_atr_mult"]     = SL_ATR_MULT
        cfg["max_lot_global"]  = MAX_LOT_GLOBAL
        with open(cfg_path, "w", encoding="utf-8") as f:
            json.dump(cfg, f, indent=2)
    except Exception as e:
        log.warning(f"Nu am putut salva config: {e}")


def _get_effective_risk() -> float:
    """
    Returneaza riscul efectiv in USD pentru trade-ul curent.
    Daca USE_RISK_PCT: calculeaza % din equity MT5.
    Daca drawdown reducer activ: aplica factorul de reducere.
    """
    global _drawdown_factor
    if USE_RISK_PCT and MT5_AVAILABLE and mt5 is not None:
        try:
            acc = mt5.account_info()
            if acc and acc.equity > 0:
                base = acc.equity * (RISK_PCT / 100.0)
                return round(base * _drawdown_factor, 2)
        except Exception:
            pass
    return round(RISK_DOLLARS * _drawdown_factor, 2)


def _update_drawdown_reducer(win: bool):
    """Actualizeaza contorul de pierderi consecutive si factorul de reducere."""
    global _consecutive_losses, _drawdown_factor
    if win:
        _consecutive_losses = 0
        _drawdown_factor    = 1.0
        log.info("DrawdownReducer: reset — trade castigat")
    else:
        _consecutive_losses += 1
        if _consecutive_losses >= 5:
            _drawdown_factor = 0.25
            log.warning(f"DrawdownReducer: {_consecutive_losses} pierderi consecutive → risc 25%")
        elif _consecutive_losses >= 3:
            _drawdown_factor = 0.5
            log.warning(f"DrawdownReducer: {_consecutive_losses} pierderi consecutive → risc 50%")
        else:
            log.info(f"DrawdownReducer: {_consecutive_losses} pierderi consecutive (factor={_drawdown_factor})")


def _check_spread(symbol: str, tick, info) -> tuple[bool, str]:
    """
    Verifica daca spread-ul curent e acceptabil.
    Returneaza (True, msg) daca ok, (False, msg) daca prea mare.
    """
    try:
        pip_size    = info.point * (10 if info.digits in (5, 3) else 1)
        spread_pips = (tick.ask - tick.bid) / pip_size
        max_spread  = MAX_SPREAD_PIPS.get(symbol, MAX_SPREAD_DEFAULT)
        if spread_pips > max_spread:
            return False, f"Spread prea mare: {spread_pips:.1f} pips (max {max_spread} pips) — skip"
        return True, f"Spread ok: {spread_pips:.1f} pips"
    except Exception as exc:
        return True, f"Spread check n/a: {exc}"  # daca nu putem verifica, continuam

_load_risk_config()

TRADE_MAGIC      = 202800
_pending_symbols: set = set()  # chei "symbol" sau "symbol::strategy" — anti-race condition
MIN_TF_VOTES    = 2         # voturi minime absolute
MIN_CONFIDENCE  = 50.0      # % minim confidence pentru a intra (ex: 2/4 = 50%)
MAX_OPEN_TRADES = 5
TP_RATIO        = 1.0   # raport TP/SL: 1.0 = 1:1, 1.5 = 1:1.5, 2.0 = 1:2
SL_ATR_MULT     = 2.0   # multiplicator ATR pentru SL fallback (1.0=scalp, 2.0=swing)
MAX_LOT_GLOBAL  = 1.0   # lot maxim per trade (1.0=conservator, 5.0=leverage 100)

# Sesiuni active (UTC) — in afara acestor ferestre botul nu deschide trades
TRADING_SESSIONS = [
    (6, 0, 18, 0),   # Tokyo open → NY close (mai permisiv)
]

ADX_MIN = _CFG_ADX_MIN  # din config.py

# ── Reguli FTMO ───────────────────────────────────────────────────────────────
FTMO_DAILY_LOSS_PCT   = 0.05   # 5% din balanta initiala
FTMO_MAX_LOSS_PCT     = 0.10   # 10% drawdown total
FTMO_NEWS_BLOCK_MIN   = _CFG_FTMO_NEWS_BLOCK_MIN
FTMO_ENABLED          = True   # activeaza verificarile FTMO

# Cache stiri ForexFactory
_news_cache = {"events": [], "fetched_at": None}
_news_lock  = __import__("threading").Lock()

def fetch_red_news():
    """Descarca stirile rosii (High impact) din ForexFactory XML. Cache 30 min."""
    import urllib.request, xml.etree.ElementTree as ET
    from datetime import datetime, timezone, timedelta

    with _news_lock:
        now = datetime.now(timezone.utc)
        if (_news_cache["fetched_at"] and
                (now - _news_cache["fetched_at"]).total_seconds() < 14400):  # 4 ore
            return _news_cache["events"]

    try:
        urls = [
            "https://nfs.faireconomy.media/ff_calendar_thisweek.xml",
            "https://cdn-nfs.faireconomy.media/ff_calendar_thisweek.xml",
        ]
        data = None
        for url in urls:
            try:
                req = urllib.request.Request(url, headers={
                    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
                })
                with urllib.request.urlopen(req, timeout=10) as resp:
                    data = resp.read()
                break
            except:
                continue
        if not data:
            return _news_cache["events"]
        root = ET.fromstring(data)
        events = []
        for ev in root.findall("event"):
            impact = ev.findtext("impact", "").strip().lower()
            if impact != "high":
                continue
            title    = ev.findtext("title", "")
            country  = ev.findtext("country", "")
            date_str = ev.findtext("date", "")
            time_str = ev.findtext("time", "")
            try:
                # format: "04-01-2026" si "8:30am" — ForexFactory e in ET (America/New_York)
                dt_str = f"{date_str} {time_str}"
                dt_naive = datetime.strptime(dt_str, "%m-%d-%Y %I:%M%p")
                try:
                    from zoneinfo import ZoneInfo
                    _ET = ZoneInfo("America/New_York")
                    dt = dt_naive.replace(tzinfo=_ET).astimezone(timezone.utc)
                except Exception:
                    # fallback: EDT = UTC-4
                    dt = (dt_naive - timedelta(hours=-4)).replace(tzinfo=timezone.utc)
                events.append({"title": title, "country": country, "dt": dt})
            except:
                pass
        with _news_lock:
            _news_cache["events"]     = events
            _news_cache["fetched_at"] = datetime.now(timezone.utc)
        log.info(f"ForexFactory: {len(events)} stiri rosii incarcate")
        return events
    except Exception as e:
        log.warning(f"ForexFactory fetch eroare: {e}")
        return _news_cache["events"]  # returneaza cache vechi daca exista


def get_upcoming_red_news(minutes_ahead=5):
    """Returneaza stirile rosii din urmatoarele N minute."""
    from datetime import datetime, timezone, timedelta
    now    = datetime.now(timezone.utc)
    events = fetch_red_news()
    upcoming = []
    for ev in events:
        diff = (ev["dt"] - now).total_seconds() / 60
        if -FTMO_NEWS_BLOCK_MIN <= diff <= minutes_ahead:
            upcoming.append({**ev, "in_minutes": round(diff, 1),
                             "dt": ev["dt"].strftime("%H:%M UTC")})
    return upcoming


def close_all_positions_for_news():
    """Inchide toate pozitiile deschise inainte de stire rosie."""
    if not MT5_AVAILABLE or mt5 is None:
        return []
    positions = mt5.positions_get()
    if not positions:
        return []
    closed = []
    for pos in positions:
        tick  = mt5.symbol_info_tick(pos.symbol)
        info  = mt5.symbol_info(pos.symbol)
        if not tick or not info:
            continue
        close_price = tick.bid if pos.type == 0 else tick.ask
        fm = info.filling_mode
        if fm & 2:    filling = mt5.ORDER_FILLING_IOC
        elif fm & 1:  filling = mt5.ORDER_FILLING_FOK
        else:         filling = mt5.ORDER_FILLING_RETURN
        req = {
            "action":       mt5.TRADE_ACTION_DEAL,
            "symbol":       pos.symbol,
            "volume":       pos.volume,
            "type":         mt5.ORDER_TYPE_SELL if pos.type == 0 else mt5.ORDER_TYPE_BUY,
            "position":     pos.ticket,
            "price":        close_price,
            "deviation":    30,
            "magic":        pos.magic,
            "comment":      "news_close",
            "type_time":    mt5.ORDER_TIME_GTC,
            "type_filling": filling,
        }
        result = mt5.order_send(req)
        if result and result.retcode == mt5.TRADE_RETCODE_DONE:
            closed.append(pos.symbol)
            log.info(f"Pozitie inchisa inainte de stire: {pos.symbol} #{pos.ticket}")
        else:
            log.warning(f"Eroare inchidere {pos.symbol}: {result.retcode if result else -1}")
    return closed

class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, (np.integer,)): return int(obj)
        if isinstance(obj, (np.floating,)): return float(obj)
        if isinstance(obj, np.ndarray): return obj.tolist()
        return super().default(obj)

# ── Fetch MT5 ─────────────────────────────────────────────────────────────────
def fetch_mt5(symbol, tf, bars):
    if not MT5_AVAILABLE or mt5 is None:
        return None
    tf_const = MT5_TF.get(tf)
    if tf_const is None:
        return None
    rates = mt5.copy_rates_from_pos(symbol, tf_const, 0, int(bars) + 10)
    if rates is None or len(rates) == 0:
        return None
    df = pd.DataFrame(rates)
    df["time"] = pd.to_datetime(df["time"], unit="s")
    df = df.set_index("time")
    df = df.rename(columns={"tick_volume": "volume"})
    return df[["open","high","low","close","volume"]].tail(int(bars))

def fetch(symbol, tf, bars):
    df = fetch_mt5(symbol, tf, bars)
    if df is not None and len(df) >= 10:
        return df, "MT5"
    return None, "MT5 indisponibil"

# ── Pivot detection ───────────────────────────────────────────────────────────
def find_pivots(df, lookback=5):
    """
    Detecteaza pivot high/low folosind scipy.signal.find_peaks.
    - distance: minim `lookback` bare intre doi pivoti (elimina pivoti prea apropiati)
    - prominence: pivotul trebuie sa fie cu cel putin 25% din ATR mai proeminent
      fata de vecinii lui (elimina micro-varfuri nesemnificative)
    """
    highs = df["high"].values
    lows  = df["low"].values

    # Prominence bazata pe ATR — adaptiva la volatilitatea activului
    sample = min(50, len(highs))
    atr_est = float(np.mean(highs[-sample:] - lows[-sample:])) if sample > 1 else 1e-5
    prominence = max(atr_est * 0.25, 1e-8)
    distance   = max(lookback, 3)

    ph, _ = find_peaks( highs, distance=distance, prominence=prominence)
    pl, _ = find_peaks(-lows,  distance=distance, prominence=prominence)

    return ph.tolist(), pl.tolist()

# ── Trend detection (pe ultimele recent_bars) ─────────────────────────────────
def detect_trend(ph_idx, pl_idx, highs, lows, recent_bars=100):
    n = len(highs)
    cutoff = n - recent_bars
    ph_r = [i for i in ph_idx if i >= cutoff]
    pl_r = [i for i in pl_idx if i >= cutoff]
    if len(ph_r) < 2 or len(pl_r) < 2:
        return "RANGING"
    ph_p = [highs[i] for i in ph_r[-6:]]
    pl_p = [lows[i]  for i in pl_r[-6:]]
    hh = sum(ph_p[i] > ph_p[i-1] for i in range(1, len(ph_p)))
    lh = sum(ph_p[i] < ph_p[i-1] for i in range(1, len(ph_p)))
    hl = sum(pl_p[i] > pl_p[i-1] for i in range(1, len(pl_p)))
    ll = sum(pl_p[i] < pl_p[i-1] for i in range(1, len(pl_p)))
    bull = hh + hl
    bear = lh + ll
    if bull > bear and bull >= 2:
        return "ASCENDING"
    if bear > bull and bear >= 2:
        return "DESCENDING"
    return "RANGING"

# ── ADX calculation ───────────────────────────────────────────────────────────
def calc_adx(df, period=14):
    """Calculeaza ADX. Returneaza seria ADX."""
    high  = df["high"]
    low   = df["low"]
    close = df["close"]
    tr    = pd.concat([high - low,
                       (high - close.shift()).abs(),
                       (low  - close.shift()).abs()], axis=1).max(axis=1)
    dm_plus  = (high - high.shift()).clip(lower=0)
    dm_minus = (low.shift() - low).clip(lower=0)
    # anuleaza unde nu e clar
    dm_plus  = dm_plus.where(dm_plus > dm_minus, 0)
    dm_minus = dm_minus.where(dm_minus > dm_plus, 0)

    atr14    = tr.rolling(period).mean()
    di_plus  = 100 * dm_plus.rolling(period).mean()  / atr14.replace(0, np.nan)
    di_minus = 100 * dm_minus.rolling(period).mean() / atr14.replace(0, np.nan)
    dx       = 100 * (di_plus - di_minus).abs() / (di_plus + di_minus).replace(0, np.nan)
    adx      = dx.rolling(period).mean()
    return adx


# ── H4 trend filter ───────────────────────────────────────────────────────────
def get_h4_direction(symbol, bars=100):
    """Returneaza directia trendului pe H4: 'BUY', 'SELL' sau None (lateral)."""
    try:
        df, _ = fetch(symbol, "H4", bars)
        if df is None or len(df) < 60:
            return None
        ema20 = float(df["close"].ewm(span=20, adjust=False).mean().iloc[-1])
        ema50 = float(df["close"].ewm(span=50, adjust=False).mean().iloc[-1])
        adx   = calc_adx(df)
        adx_now = float(adx.iloc[-1]) if not adx.empty else 0
        if adx_now < ADX_MIN:
            return None   # piata laterala pe H4 — nu tranzactiona
        if ema20 > ema50 * 1.0001:
            return "BUY"
        if ema20 < ema50 * 0.9999:
            return "SELL"
        return None
    except Exception as e:
        log.warning(f"get_h4_direction {symbol}: {e}")
        return None


# ── Session filter ────────────────────────────────────────────────────────────
def in_trading_session():
    """Returneaza True daca suntem in fereastra London sau New York."""
    from datetime import datetime, timezone
    now = datetime.now(timezone.utc)
    h, m = now.hour, now.minute
    cur = h * 60 + m
    for sh, sm, eh, em in TRADING_SESSIONS:
        if sh * 60 + sm <= cur <= eh * 60 + em:
            return True
    return False



# ── Fibonacci retracement ────────────────────────────────────────────────────
FIB_LEVELS = [0.0, 0.236, 0.382, 0.5, 0.618, 0.786, 1.0]

def calc_fib_levels(df, ph_idx, pl_idx, trend):
    """
    Calculeaza nivelele Fibonacci de retracement din ultimul swing major.
    Returneaza dict cu nivelele si swing_high/swing_low, sau None.
    """
    highs  = df["high"].values
    lows   = df["low"].values
    n      = len(highs)
    cutoff = max(0, n - 200)
    ph_r   = [i for i in ph_idx if i >= cutoff]
    pl_r   = [i for i in pl_idx if i >= cutoff]

    if not ph_r or not pl_r:
        return None

    last_ph = highs[ph_r[-1]]
    last_pl = lows[pl_r[-1]]

    # Swing direction: in trend ascendent, swing e de la ultimul PL la ultimul PH
    if trend == "ASCENDING":
        # masurim de la swing_low (pl) la swing_high (ph)
        # dar PH trebuie sa fie DUPA PL pentru a fi un swing valid
        valid_pls = [i for i in pl_r if i < ph_r[-1]]
        if not valid_pls:
            return None
        swing_low  = lows[valid_pls[-1]]
        swing_high = last_ph
    elif trend == "DESCENDING":
        # masurim de la swing_high (ph) la swing_low (pl)
        valid_phs = [i for i in ph_r if i < pl_r[-1]]
        if not valid_phs:
            return None
        swing_high = highs[valid_phs[-1]]
        swing_low  = last_pl
    else:
        return None

    diff = swing_high - swing_low
    if diff <= 0:
        return None

    levels = {}
    for f in FIB_LEVELS:
        if trend == "ASCENDING":
            levels[f] = round(swing_high - diff * f, 5)  # retracement de la high in jos
        else:
            levels[f] = round(swing_low + diff * f, 5)   # retracement de la low in sus

    return {
        "swing_high": round(swing_high, 5),
        "swing_low":  round(swing_low, 5),
        "levels":     levels,  # {0.0: price, 0.382: price, ...}
        "trend":      trend,
    }


def price_near_fib(price, fib_data, tolerance=0.002):
    """
    Verifica daca pretul e in 'golden zone' Fibonacci (38.2% sau 61.8%).
    Returneaza (True, nivel_cel_mai_apropiat) sau (False, None).
    """
    if fib_data is None:
        return False, None
    golden = [0.382, 0.5, 0.618]
    best_level = None
    best_dist  = float("inf")
    for f in golden:
        fib_price = fib_data["levels"].get(f)
        if fib_price is None:
            continue
        dist = abs(price - fib_price) / fib_price
        if dist < best_dist:
            best_dist  = dist
            best_level = f
    if best_dist <= tolerance:
        return True, best_level
    return False, None


# ── Entry conditions (noua strategie: EMA50 pullback + Fibonacci + ADX) ──────
def calc_entry(df, ph_idx, pl_idx, trend, ema20, ema50, rsi):
    """
    Intrare pe pullback la EMA50 SAU nivel Fibonacci 38.2%/61.8%,
    confirmat de ADX > 25 si RSI in zona neutra.
    """
    highs     = df["high"].values
    lows      = df["low"].values
    price_now = float(df["close"].iloc[-1])
    rsi_now   = float(rsi.iloc[-1])
    ema20_now = float(ema20.iloc[-1])
    ema50_now = float(ema50.iloc[-1])
    entry_signal = "HOLD"
    entry_reason = []

    # ADX pe acest TF
    adx     = calc_adx(df)
    adx_now = float(adx.iloc[-1]) if not adx.empty else 0

    # Fibonacci levels
    fib_data = calc_fib_levels(df, ph_idx, pl_idx, trend)
    at_fib, fib_level = price_near_fib(price_now, fib_data)

    # Conditie ADX
    if adx_now < ADX_MIN:
        entry_reason.append(f"ADX {round(adx_now,1)} < {ADX_MIN} — piata laterala")
        return "HOLD", entry_reason, price_now

    # RSI: nu mai e un gate dur — e context. In uptrend puternic RSI sta la 60-75
    # si blocheaza toate semnalele. Lasam ClassicStrategy scoring sa il gestioneze.
    rsi_extreme_bull = rsi_now > 75   # overbought extrem — evita
    rsi_extreme_bear = rsi_now < 25   # oversold extrem — evita

    if trend == "ASCENDING":
        ema_aligned = ema20_now > ema50_now
        # Toleranta largita: ±0.5% in jurul EMA50 (era ±0.2%)
        near_ema50  = ema50_now * 0.995 <= price_now <= ema50_now * 1.005
        at_support  = near_ema50 or at_fib

        if ema_aligned:  entry_reason.append("EMA20>EMA50 ✓")
        if near_ema50:   entry_reason.append(f"pullback EMA50 ({round(ema50_now,5)}) ✓")
        if at_fib:       entry_reason.append(f"Fibonacci {round(fib_level*100,1)}% ({round(fib_data['levels'][fib_level],5)}) ✓")
        entry_reason.append(f"RSI {round(rsi_now,1)}" + (" — overbought extrem, risc" if rsi_extreme_bull else " ✓"))
        entry_reason.append(f"ADX {round(adx_now,1)} ✓")

        if ema_aligned and at_support and not rsi_extreme_bull:
            entry_signal = "BUY"

    elif trend == "DESCENDING":
        ema_aligned = ema20_now < ema50_now
        near_ema50  = ema50_now * 0.995 <= price_now <= ema50_now * 1.005
        at_resist   = near_ema50 or at_fib

        if ema_aligned:  entry_reason.append("EMA20<EMA50 ✓")
        if near_ema50:   entry_reason.append(f"pullback EMA50 ({round(ema50_now,5)}) ✓")
        if at_fib:       entry_reason.append(f"Fibonacci {round(fib_level*100,1)}% ({round(fib_data['levels'][fib_level],5)}) ✓")
        entry_reason.append(f"RSI {round(rsi_now,1)}" + (" — oversold extrem, risc" if rsi_extreme_bear else " ✓"))
        entry_reason.append(f"ADX {round(adx_now,1)} ✓")

        if ema_aligned and at_resist and not rsi_extreme_bear:
            entry_signal = "SELL"

    return entry_signal, entry_reason, price_now

# ══════════════════════════════════════════════════════════════════════════════
# ── SMC Strategy: Order Blocks, Fair Value Gap, Break of Structure ────────────
# ══════════════════════════════════════════════════════════════════════════════

def find_order_blocks(df, lookback=50):
    """
    Detecteaza Order Blocks (OB):
    - Bullish OB: ultima lumanare bearish inainte de o miscare bullish puternica
    - Bearish OB: ultima lumanare bullish inainte de o miscare bearish puternica
    Returneaza lista de OB: {type, high, low, index, broken}
    """
    obs = []
    closes = df["close"].values
    opens  = df["open"].values
    highs  = df["high"].values
    lows   = df["low"].values
    n = len(closes)
    start = max(3, n - lookback)

    for i in range(start, n - 2):
        # Bullish OB: lumanare bearish urmata de impuls bullish puternic
        if closes[i] < opens[i]:  # bearish candle
            # miscare bullish dupa: urmatoarele 2 close > high-ul OB
            if closes[i+1] > highs[i] and closes[i+2] > highs[i]:
                obs.append({
                    "type":   "BULLISH",
                    "high":   round(float(highs[i]), 5),
                    "low":    round(float(lows[i]), 5),
                    "mid":    round(float((highs[i] + lows[i]) / 2), 5),
                    "index":  i,
                    "broken": False,
                })
        # Bearish OB: lumanare bullish urmata de impuls bearish puternic
        elif closes[i] > opens[i]:  # bullish candle
            if closes[i+1] < lows[i] and closes[i+2] < lows[i]:
                obs.append({
                    "type":   "BEARISH",
                    "high":   round(float(highs[i]), 5),
                    "low":    round(float(lows[i]), 5),
                    "mid":    round(float((highs[i] + lows[i]) / 2), 5),
                    "index":  i,
                    "broken": False,
                })

    # Marcheaza OB-urile sparte (pretul a trecut prin ele)
    price_now = float(closes[-1])
    for ob in obs:
        if ob["type"] == "BULLISH" and price_now < ob["low"]:
            ob["broken"] = True
        elif ob["type"] == "BEARISH" and price_now > ob["high"]:
            ob["broken"] = True

    # Returneaza doar OB-urile active (nesparte), cele mai recente primele
    active = [ob for ob in obs if not ob["broken"]]
    return active[-5:]  # ultimele 5 active


def find_fvg(df, lookback=100):
    """
    Detecteaza Fair Value Gaps (FVG):
    - Bullish FVG: high[i-2] < low[i]  → gap intre lumanarea i-2 si i
    - Bearish FVG: low[i-2] > high[i]  → gap intre lumanarea i-2 si i
    Returneaza lista: {type, top, bottom, mid, index, filled}
    """
    fvgs = []
    highs  = df["high"].values
    lows   = df["low"].values
    closes = df["close"].values
    n = len(closes)
    start = max(2, n - lookback)

    for i in range(start, n):
        # Bullish FVG
        if highs[i-2] < lows[i]:
            fvgs.append({
                "type":   "BULLISH",
                "top":    round(float(lows[i]), 5),
                "bottom": round(float(highs[i-2]), 5),
                "mid":    round(float((lows[i] + highs[i-2]) / 2), 5),
                "index":  i,
                "filled": False,
            })
        # Bearish FVG
        elif lows[i-2] > highs[i]:
            fvgs.append({
                "type":   "BEARISH",
                "top":    round(float(lows[i-2]), 5),
                "bottom": round(float(highs[i]), 5),
                "mid":    round(float((lows[i-2] + highs[i]) / 2), 5),
                "index":  i,
                "filled": False,
            })

    price_now = float(closes[-1])
    for fvg in fvgs:
        # FVG umplut daca pretul a trecut prin mijloc
        if fvg["type"] == "BULLISH" and price_now < fvg["mid"]:
            fvg["filled"] = True
        elif fvg["type"] == "BEARISH" and price_now > fvg["mid"]:
            fvg["filled"] = True

    active = [f for f in fvgs if not f["filled"]]
    return active[-5:]


def detect_bos(df, ph_idx, pl_idx):
    """
    Break of Structure (BOS):
    - Bullish BOS: pretul a spart ultimul Higher High → trend bullish confirmat
    - Bearish BOS: pretul a spart ultimul Lower Low → trend bearish confirmat
    Returneaza: "BULLISH" / "BEARISH" / None
    """
    highs  = df["high"].values
    lows   = df["low"].values
    closes = df["close"].values
    n      = len(closes)
    cutoff = max(0, n - 100)

    ph_r = [i for i in ph_idx if i >= cutoff]
    pl_r = [i for i in pl_idx if i >= cutoff]
    if len(ph_r) < 2 or len(pl_r) < 2:
        return None

    price_now = float(closes[-1])

    # Bullish BOS: pretul > ultimul pivot high
    last_ph = float(highs[ph_r[-1]])
    prev_ph = float(highs[ph_r[-2]])
    if price_now > last_ph and last_ph > prev_ph:
        return "BULLISH"

    # Bearish BOS: pretul < ultimul pivot low
    last_pl = float(lows[pl_r[-1]])
    prev_pl = float(lows[pl_r[-2]])
    if price_now < last_pl and last_pl < prev_pl:
        return "BEARISH"

    return None


def price_in_ob(price, obs, direction, tolerance=0.001):
    """Verifica daca pretul e intr-un Order Block activ in directia dorita."""
    for ob in obs:
        if ob["type"] != direction:
            continue
        lo = ob["low"] * (1 - tolerance)
        hi = ob["high"] * (1 + tolerance)
        if lo <= price <= hi:
            return True, ob
    return False, None


def price_near_fvg(price, fvgs, direction, tolerance=0.002):
    """Verifica daca pretul e langa un FVG activ in directia dorita."""
    for fvg in fvgs:
        if fvg["type"] != direction:
            continue
        lo = fvg["bottom"] * (1 - tolerance)
        hi = fvg["top"] * (1 + tolerance)
        if lo <= price <= hi:
            return True, fvg
    return False, None


def calc_entry_smc(df, ph_idx, pl_idx, elements=None):
    """
    Strategie SMC:
    - BOS confirma directia
    - Order Block = zona de intrare
    - FVG = confirmare suplimentara
    elements = dict cu toggle-uri: {"bos": True, "ob": True, "fvg": True, "structure": True}
    """
    if elements is None:
        elements = {"bos": True, "ob": True, "fvg": True, "structure": True}

    closes = df["close"].values
    highs  = df["high"].values
    lows   = df["low"].values
    price_now = float(closes[-1])

    reasons = []
    signal  = "HOLD"
    score_buy  = 0
    score_sell = 0

    # 1. Break of Structure
    bos = detect_bos(df, ph_idx, pl_idx) if elements.get("bos") else None
    if bos == "BULLISH":
        score_buy += 2
        reasons.append("BOS Bullish ✓")
    elif bos == "BEARISH":
        score_sell += 2
        reasons.append("BOS Bearish ✓")

    # 2. Order Blocks
    obs = find_order_blocks(df) if elements.get("ob") else []
    in_bull_ob, bull_ob = price_in_ob(price_now, obs, "BULLISH")
    in_bear_ob, bear_ob = price_in_ob(price_now, obs, "BEARISH")
    if in_bull_ob:
        score_buy += 3
        reasons.append(f"Order Block Bullish [{bull_ob['low']}–{bull_ob['high']}] ✓")
    if in_bear_ob:
        score_sell += 3
        reasons.append(f"Order Block Bearish [{bear_ob['low']}–{bear_ob['high']}] ✓")

    # 3. Fair Value Gap
    fvgs = find_fvg(df) if elements.get("fvg") else []
    near_bull_fvg, bull_fvg = price_near_fvg(price_now, fvgs, "BULLISH")
    near_bear_fvg, bear_fvg = price_near_fvg(price_now, fvgs, "BEARISH")
    if near_bull_fvg:
        score_buy += 2
        reasons.append(f"FVG Bullish [{bull_fvg['bottom']}–{bull_fvg['top']}] ✓")
    if near_bear_fvg:
        score_sell += 2
        reasons.append(f"FVG Bearish [{bear_fvg['bottom']}–{bear_fvg['top']}] ✓")

    # 4. Market Structure (Higher Highs / Lower Lows)
    if elements.get("structure"):
        n  = len(closes)
        cutoff = max(0, n - 50)
        ph_r = [i for i in ph_idx if i >= cutoff]
        pl_r = [i for i in pl_idx if i >= cutoff]
        if len(ph_r) >= 2 and len(pl_r) >= 2:
            hh = highs[ph_r[-1]] > highs[ph_r[-2]]  # Higher High
            hl = lows[pl_r[-1]]  > lows[pl_r[-2]]   # Higher Low
            lh = highs[ph_r[-1]] < highs[ph_r[-2]]  # Lower High
            ll = lows[pl_r[-1]]  < lows[pl_r[-2]]   # Lower Low
            if hh and hl:
                score_buy += 1
                reasons.append("Structure: HH+HL ✓")
            elif lh and ll:
                score_sell += 1
                reasons.append("Structure: LH+LL ✓")

    # Decizie — necesita minim 2 puncte pentru a intra
    min_score = 2
    if score_buy >= min_score and score_buy > score_sell:
        signal = "BUY"
    elif score_sell >= min_score and score_sell > score_buy:
        signal = "SELL"

    conviction = max(score_buy, score_sell)
    return signal, reasons, price_now, conviction


# ── SL/TP din pivoti ──────────────────────────────────────────────────────────
def calc_sl_tp(df, ph_idx, pl_idx, signal, price):
    """
    SL: cel mai relevant pivot low/high din apropierea pretului curent.
    TP: urmatorul nivel structural (pivot high/low dincolo de price) daca exista
        si ofera minim 1.5:1 R:R; altfel fallback la R:R 2:1.
    Buffer ATR redus la 0.2x (era 0.3x) pentru SL mai precis.
    """
    highs  = df["high"].values
    lows   = df["low"].values
    n      = len(highs)
    cutoff = n - 150
    atr    = float(df["high"].sub(df["low"]).rolling(14).mean().iloc[-1])
    if atr <= 0 or np.isnan(atr):
        atr = abs(price) * 0.001

    ph_r = [i for i in ph_idx if i >= cutoff]
    pl_r = [i for i in pl_idx if i >= cutoff]

    sl = tp = 0.0

    sl_mult = SL_ATR_MULT
    sl_min  = max(sl_mult * 0.25, 0.3)

    if signal == "BUY":
        # SL: cel mai recent pivot low SUB price
        below = [i for i in pl_r if lows[i] < price]
        if below:
            sl = lows[below[-1]] - atr * 0.2
        else:
            sl = price - atr * sl_mult
        risk = max(price - sl, atr * sl_min)
        sl   = price - risk  # normalizat

        # TP: primul pivot high DEASUPRA price + minim 1.5:1 R:R
        above = [i for i in ph_r if highs[i] > price + risk * 0.5]
        if above:
            tp_struct = highs[above[0]]
            if tp_struct >= price + risk * 1.5:
                tp = tp_struct
            else:
                tp = price + risk * 2.0
        else:
            tp = price + risk * 2.0

    elif signal == "SELL":
        # SL: cel mai recent pivot high DEASUPRA price
        above = [i for i in ph_r if highs[i] > price]
        if above:
            sl = highs[above[-1]] + atr * 0.2
        else:
            sl = price + atr * sl_mult
        risk = max(sl - price, atr * sl_min)
        sl   = price + risk  # normalizat

        # TP: primul pivot low SUB price - minim 1.5:1 R:R
        below = [i for i in pl_r if lows[i] < price - risk * 0.5]
        if below:
            tp_struct = lows[below[-1]]
            if tp_struct <= price - risk * 1.5:
                tp = tp_struct
            else:
                tp = price - risk * 2.0
        else:
            tp = price - risk * 2.0

    return round(sl, 5), round(tp, 5)

# ── Analiza per TF ────────────────────────────────────────────────────────────
def get_signal_data(symbol, tf, bars=500):
    df, _ = fetch(symbol, tf, bars)
    if df is None:
        return None
    highs = df["high"].values
    lows  = df["low"].values
    ph_idx, pl_idx = find_pivots(df, lookback=5)
    trend = detect_trend(ph_idx, pl_idx, highs, lows, recent_bars=100)
    ema20 = df["close"].ewm(span=20, adjust=False).mean()
    ema50 = df["close"].ewm(span=50, adjust=False).mean()
    delta = df["close"].diff()
    gain  = delta.clip(lower=0).rolling(14).mean()
    loss  = (-delta.clip(upper=0)).rolling(14).mean()
    rsi   = 100 - 100 / (1 + gain / loss.replace(0, np.nan))
    signal, reasons, price = calc_entry(df, ph_idx, pl_idx, trend, ema20, ema50, rsi)
    sl, tp = calc_sl_tp(df, ph_idx, pl_idx, signal, price)
    return {
        "tf": tf,
        "signal": signal,
        "trend": trend,
        "conviction": len(reasons),
        "reasons": reasons,
        "price": round(price, 5),
        "sl": sl,
        "tp": tp,
    }

# ── Agregate multi-TF ─────────────────────────────────────────────────────────
def analyze_symbol(symbol, tfs=None, bars=500):
    if tfs is None:
        tfs = ["M1", "M5", "M15", "H1", "H4"]
    results = [r for tf in tfs if (r := get_signal_data(symbol, tf, bars)) is not None]
    buy_v  = [r for r in results if r["signal"] == "BUY"]
    sell_v = [r for r in results if r["signal"] == "SELL"]
    n_buy, n_sell = len(buy_v), len(sell_v)
    final = "HOLD"
    best  = None
    if n_buy >= MIN_TF_VOTES and n_buy > n_sell:
        final = "BUY"
        best  = max(buy_v,  key=lambda x: x["conviction"])
    elif n_sell >= MIN_TF_VOTES and n_sell > n_buy:
        final = "SELL"
        best  = max(sell_v, key=lambda x: x["conviction"])
    return {
        "symbol": symbol,
        "signal": final,
        "n_buy": n_buy, "n_sell": n_sell,
        "n_total": len(results),
        "best": best,
        "all": results,
    }

# ── Verificari FTMO ──────────────────────────────────────────────────────────
def check_ftmo_rules(symbol=""):
    """Returneaza (ok, motiv) — ok=False inseamna trade blocat."""
    if not FTMO_ENABLED or not MT5_AVAILABLE or mt5 is None:
        return True, ""

    from datetime import datetime, timezone
    now_utc = datetime.now(timezone.utc)

    # 1. Verificare news — folosim cache-ul real ForexFactory
    upcoming = get_upcoming_red_news(minutes_ahead=FTMO_NEWS_BLOCK_MIN)
    if upcoming:
        ev = upcoming[0]
        return False, f"Blocat: stire {ev['title']} ({ev['country']}) in {ev['in_minutes']} min"

    # 2. Verificare drawdown zilnic 5%
    acc = mt5.account_info()
    if acc:
        balance    = acc.balance
        equity     = acc.equity
        # drawdown zilnic: equity sub 95% din balance
        daily_floor = balance * (1 - FTMO_DAILY_LOSS_PCT)
        if equity <= daily_floor:
            return False, f"Blocat: drawdown zilnic atins (equity {equity:.2f} <= {daily_floor:.2f})"

        # 3. Verificare drawdown total 10%
        # folosim balance ca proxy pentru initial balance (nu avem istoricul exact)
        # daca equity < 90% din balance curent, blocam
        total_floor = balance * (1 - FTMO_MAX_LOSS_PCT)
        if equity <= total_floor:
            return False, f"Blocat: drawdown total 10% atins (equity {equity:.2f})"

    # 4. Weekend — nu tranzactiona vineri dupa 21:00 UTC si sambata/duminica
    # Crypto nu are weekend — sari verificarea pentru crypto
    CRYPTO_KEYWORDS = {"BTC","ETH","XRP","LTC","ADA","SOL","BNB","DOT","DOGE","MATIC","XLM","LINK","UNI","AVAX"}
    is_crypto = any(kw in symbol.upper() for kw in CRYPTO_KEYWORDS)
    if not is_crypto:
        weekday = now_utc.weekday()  # 4=vineri, 5=sambata, 6=duminica
        if weekday == 5:
            return False, "Blocat: sambata — piata inchisa"
        if weekday == 6 and now_utc.hour < 22:
            return False, "Blocat: duminica — piata inchisa pana la 22:00 UTC"
        if weekday == 4 and now_utc.hour >= 21:
            return False, "Blocat: vineri dupa 21:00 UTC — inchidere weekend"

    return True, ""


# ── Executa trade MT5 ─────────────────────────────────────────────────────────
def get_signal_file_path():
    """Returneaza calea catre fisierul de semnal in directorul MT5 Files"""
    if MT5_AVAILABLE and mt5 is not None:
        info = mt5.terminal_info()
        if info:
            return os.path.join(info.data_path, "MQL5", "Files", "cv_signal.json")
    # fallback langa app.py
    return os.path.join(os.path.dirname(__file__), "cv_signal.json")

def place_trade(symbol, signal, sl, tp, risk_dollars=50.0, strategy="classic",
                one_per_strategy=False):
    import math, os
    if not MT5_AVAILABLE or mt5 is None:
        return False, "MT5 indisponibil"

    if not mt5.initialize():
        return False, "MT5 initialize() esuat"

    # ── Blocare Combined Mode — strategiile individuale nu pot executa ──────────
    try:
        from autotrader import scanner as _sc
        _use_ses = _sc.get("use_session_filter", True)
        _use_h4  = _sc.get("use_h4_filter", True)
        if _sc.get("combined_mode", False) and strategy != "combined":
            return False, f"Combined Mode activ — executie individuala blocata ({strategy})"
    except Exception:
        _use_ses = True
        _use_h4  = True

    # Filtru sesiuni dezactivat — user controleaza manual scannerul

    # Verificare H4 direction (doar daca filtrul e activ)
    # H4 direction nu mai e filtru separat — e controlat din lista TF

    # Verificare reguli FTMO
    ftmo_ok, ftmo_msg = check_ftmo_rules(symbol)
    if not ftmo_ok:
        log.warning(f"FTMO block: {ftmo_msg}")
        return False, ftmo_msg

    # Verificare Max Daily Drawdown
    if is_daily_drawdown_halted():
        halted, halt_msg = check_daily_drawdown_halt()
        if halted:
            log.warning(f"Daily drawdown halt: {halt_msg}")
            return False, halt_msg

    # Verificare numar maxim pozitii deschise
    open_count = mt5.positions_total()
    if open_count >= MAX_OPEN_TRADES:
        return False, f"Limita atinsa: {open_count}/{MAX_OPEN_TRADES} pozitii deschise — asteapta sa se inchida una"

    # Verificare: max 1 trade per simbol (sau per simbol+strategie)
    # Ordinele manuale nu sunt blocate de aceasta regula
    existing = mt5.positions_get(symbol=symbol) or []
    if existing and strategy != "manual":
        if not one_per_strategy:
            # Mod clasic: orice trade pe simbol → skip
            return False, f"Deja exista trade deschis pe {symbol} — skip"
        else:
            # Mod 1-per-strategie: verifica daca ACEASTA strategie are deja trade
            strat_tag = strategy.upper()[:6]
            owned = [
                pos for pos in existing
                if f"CV_BUY_{strat_tag}"  in (pos.comment or "")
                or f"CV_SELL_{strat_tag}" in (pos.comment or "")
            ]
            if owned:
                return False, f"Strategia {strategy} are deja trade pe {symbol} — skip"

    # Verificare anti-race: (simbol, strategie) in curs de trimitere?
    pending_key = f"{symbol}::{strategy}" if one_per_strategy else symbol
    if pending_key in _pending_symbols:
        return False, f"{symbol} in curs de procesare — skip"

    # Tot ce urmeaza e in try/finally → _pending_symbols se curata MEREU
    _pending_symbols.add(pending_key)
    try:
        # Verificare corelatie USD — max 1 trade in aceeasi directie USD simultan
        USD_BASE  = {"USDJPY", "USDCHF", "USDCAD"}
        USD_QUOTE = {"EURUSD", "GBPUSD", "AUDUSD", "NZDUSD", "XAUUSD"}
        if symbol in USD_BASE:
            usd_direction = "BUY_USD" if signal == "BUY" else "SELL_USD"
        elif symbol in USD_QUOTE:
            usd_direction = "SELL_USD" if signal == "BUY" else "BUY_USD"
        else:
            usd_direction = None

        if usd_direction:
            all_positions = mt5.positions_get() or []
            usd_dir_count = sum(
                1 for pos in all_positions
                if (pos.symbol in USD_BASE and ("BUY_USD" if (pos.type == 0) else "SELL_USD") == usd_direction)
                or (pos.symbol in USD_QUOTE and ("SELL_USD" if (pos.type == 0) else "BUY_USD") == usd_direction)
            )
            if usd_dir_count >= 2:
                return False, f"Corelatie USD: deja exista {usd_dir_count} trade(s) in directia {usd_direction} — skip {symbol}"

        # Correlation Guard: blocheaza simboluri corelate > 0.8 in aceeasi directie
        corr_blocked, corr_msg = _check_correlation_guard(symbol, signal)
        if corr_blocked:
            log.info(corr_msg)
            return False, corr_msg

        info = mt5.symbol_info(symbol)
        if info is None:
            return False, f"Symbol {symbol} negasit in MT5"

        if not info.visible:
            mt5.symbol_select(symbol, True)

        tick = mt5.symbol_info_tick(symbol)
        if tick is None:
            return False, "Nu s-a putut obtine pretul"

        # Spread filter (skip pentru ordine manuale — userul vede spread-ul singur)
        if strategy != "manual":
            spread_ok, spread_msg = _check_spread(symbol, tick, info)
            if not spread_ok:
                return False, spread_msg
        log.info(f"place_trade {symbol}: spread check {'skipped (manual)' if strategy=='manual' else spread_msg}")

        exec_price = tick.ask if signal == "BUY" else tick.bid
        tick_val   = info.trade_tick_value
        tick_size  = info.trade_tick_size
        sl_dist    = abs(exec_price - sl)

        if tick_size <= 0 or tick_val <= 0:
            return False, f"Date simbol invalide (tick_size={tick_size})"

        try:
            stops_level = getattr(info, "trade_stops_level", None) or getattr(info, "stops_level", 0)
            broker_min  = stops_level * info.point
        except Exception:
            broker_min = 0
        pip_size = info.point * (10 if info.digits in (5, 3) else 1)
        # Crypto are preturi mari (100-60000$) — 0.15% ar forta SL enorm si lot mic
        # Folosim minim 10 pips sau broker_min, fara factorul procentual pe crypto
        is_crypto = any(c in symbol for c in ("BTC","ETH","XRP","SOL","BNB","DOG","ADA","AVAX","LINK","LTC"))
        if is_crypto:
            min_dist = max(broker_min, 10 * pip_size)
        else:
            min_dist = max(broker_min, 20 * pip_size, exec_price * 0.0015)

        if signal == "BUY":
            sl = min(sl, exec_price - min_dist)
        else:
            sl = max(sl, exec_price + min_dist)

        sl_dist = abs(exec_price - sl)
        if sl_dist <= 0:
            return False, "SL invalid dupa ajustare"

        if signal == "BUY":
            tp = exec_price + sl_dist * TP_RATIO
        else:
            tp = exec_price - sl_dist * TP_RATIO

        lot_step = info.volume_step
        min_lot  = info.volume_min
        max_lot  = min(info.volume_max, MAX_LOT_GLOBAL)

        # Position sizing: % equity sau suma fixa, cu drawdown reducer aplicat
        effective_risk = _get_effective_risk()
        if effective_risk != risk_dollars:
            log.info(f"EffectiveRisk: {effective_risk:.2f} USD (factor={_drawdown_factor:.2f})")

        lots = math.floor(effective_risk / (sl_dist / tick_size * tick_val) / lot_step) * lot_step
        lots = max(min_lot, min(max_lot, lots))
        lots = round(lots, 2)
        log.info(f"place_trade {symbol} {signal}: price={exec_price} sl={round(sl,info.digits)} sl_dist={round(sl_dist,info.digits)} ({round(sl_dist/pip_size,1)} pips) lots={lots}")

        order_type = mt5.ORDER_TYPE_BUY if signal == "BUY" else mt5.ORDER_TYPE_SELL

        fm = info.filling_mode
        if fm & 2:   filling = mt5.ORDER_FILLING_IOC
        elif fm & 1: filling = mt5.ORDER_FILLING_FOK
        else:        filling = mt5.ORDER_FILLING_RETURN

        req = {
            "action":       mt5.TRADE_ACTION_DEAL,
            "symbol":       symbol,
            "volume":       lots,
            "type":         order_type,
            "price":        exec_price,
            "sl":           round(sl, info.digits),
            "tp":           round(tp, info.digits),
            "deviation":    30,
            "magic":        TRADE_MAGIC,
            "comment":      f"CV_{signal}_{strategy.upper()[:6]}",
            "type_time":    mt5.ORDER_TIME_GTC,
            "type_filling": filling,
        }
        log.info(f"order_send: {req}")

        attempts = [
            {**req, "type_filling": mt5.ORDER_FILLING_IOC},
            {**req, "type_filling": mt5.ORDER_FILLING_FOK},
            {**req, "type_filling": mt5.ORDER_FILLING_RETURN},
            {k: v for k, v in req.items() if k != "type_filling"},
        ]
        last_code = -1
        for attempt in attempts:
            result = mt5.order_send(attempt)
            if result is not None and result.retcode == mt5.TRADE_RETCODE_DONE:
                ticket = result.order
                import threading as _thr
                _thr.Thread(target=save_trade_snapshot, args=(ticket, symbol, signal, exec_price,
                    round(sl, info.digits), round(tp, info.digits)), kwargs={"tf": "M5"}, daemon=True).start()
                return True, f"OK #{ticket} — {lots} loturi {signal} {symbol} @ {exec_price}  SL={round(sl,info.digits)}  TP={round(tp,info.digits)}"
            last_code = result.retcode if result else -1
            log.warning(f"attempt filling={attempt.get('type_filling','none')} retcode={last_code}")

        msgs = {
            10027: "AutoTrading dezactivat — Tools→Options→Expert Advisors→Allow algorithmic trading",
            10030: "Filling mode incompatibil cu brokerul (10030)",
            10018: "Piata inchisa",
            10019: "Fonduri insuficiente",
            10016: f"SL/TP invalid ({info.digits} zecimale)",
            10014: f"Volum invalid (lots={lots})",
            10006: "Ordin respins de broker",
            10013: "Parametri invalizi",
        }
        return False, msgs.get(last_code, f"Eroare MT5: {last_code}")

    finally:
        _pending_symbols.discard(pending_key)  # elibereaza MEREU, indiferent de ce s-a intamplat

def place_pending_order(symbol, signal, entry_price, sl, tp,
                        risk_dollars=50.0, strategy="eob", expiry_hours=24):
    """
    Plaseaza un ordin pending MT5 (Limit sau Stop) la pretul specificat.
    Tipul se alege automat:
      BUY  + entry < pret curent → BUY_LIMIT  (asteapta retrasarea)
      BUY  + entry > pret curent → BUY_STOP   (breakout bullish)
      SELL + entry > pret curent → SELL_LIMIT  (asteapta bounce-ul)
      SELL + entry < pret curent → SELL_STOP   (breakout bearish)

    Aplica FULL FTMO validation stack (identic cu place_trade):
      - check_ftmo_rules, is_daily_drawdown_halted, MAX_OPEN_TRADES
      - USD correlation & _check_correlation_guard
      - _check_spread, MIN_STOPS_LEVEL
      - anti-race _pending_symbols
      - dedup: skip daca exista deja pending cu acelasi simbol+strategie
    """
    import math
    from datetime import datetime as _dt, timedelta as _td

    if not MT5_AVAILABLE or mt5 is None:
        return False, "MT5 indisponibil"
    if not mt5.initialize():
        return False, "MT5 initialize() esuat"

    # Blocare Combined Mode — strategiile individuale nu pot plasa pending
    try:
        from autotrader import scanner as _sc
        if _sc.get("combined_mode", False) and strategy != "combined":
            return False, f"Combined Mode activ — pending blocat ({strategy})"
    except Exception:
        pass

    # Verificare reguli FTMO
    ftmo_ok, ftmo_msg = check_ftmo_rules(symbol)
    if not ftmo_ok:
        log.warning(f"FTMO block (pending): {ftmo_msg}")
        return False, ftmo_msg

    # Verificare Max Daily Drawdown
    if is_daily_drawdown_halted():
        halted, halt_msg = check_daily_drawdown_halt()
        if halted:
            log.warning(f"Daily drawdown halt (pending): {halt_msg}")
            return False, halt_msg

    # Verificare numar maxim pozitii + pending deschise
    open_count    = mt5.positions_total()
    pending_total = len(mt5.orders_get() or [])
    if open_count + pending_total >= MAX_OPEN_TRADES:
        return False, f"Limita atinsa: {open_count} pozitii + {pending_total} pending ≥ {MAX_OPEN_TRADES}"

    # Dedup: deja exista pending pe (simbol + strategie)?
    strat_tag = strategy.upper()[:6]
    existing_pending = mt5.orders_get(symbol=symbol) or []
    for op in existing_pending:
        cmt = (op.comment or "")
        if f"CV_BUY_{strat_tag}_P" in cmt or f"CV_SELL_{strat_tag}_P" in cmt:
            return False, f"Pending {strategy} deja exista pe {symbol} (ticket {op.ticket}) — skip"

    # Evita pending pe simbol cu pozitie activa de la aceeasi strategie
    existing_pos = mt5.positions_get(symbol=symbol) or []
    for pos in existing_pos:
        cmt = (pos.comment or "")
        if f"CV_BUY_{strat_tag}" in cmt or f"CV_SELL_{strat_tag}" in cmt:
            return False, f"Strategia {strategy} are deja pozitie pe {symbol} — skip pending"

    # Anti-race: (simbol, strategie) in curs de trimitere?
    pending_key = f"{symbol}::{strategy}::P"
    if pending_key in _pending_symbols:
        return False, f"{symbol} pending in curs de procesare — skip"

    _pending_symbols.add(pending_key)
    try:
        # Corelatie USD — max 2 trades in aceeasi directie USD (pozitii + pending)
        USD_BASE  = {"USDJPY", "USDCHF", "USDCAD"}
        USD_QUOTE = {"EURUSD", "GBPUSD", "AUDUSD", "NZDUSD", "XAUUSD"}
        if symbol in USD_BASE:
            usd_direction = "BUY_USD" if signal == "BUY" else "SELL_USD"
        elif symbol in USD_QUOTE:
            usd_direction = "SELL_USD" if signal == "BUY" else "BUY_USD"
        else:
            usd_direction = None

        if usd_direction:
            all_positions = mt5.positions_get() or []
            usd_dir_count = sum(
                1 for pos in all_positions
                if (pos.symbol in USD_BASE and ("BUY_USD" if (pos.type == 0) else "SELL_USD") == usd_direction)
                or (pos.symbol in USD_QUOTE and ("SELL_USD" if (pos.type == 0) else "BUY_USD") == usd_direction)
            )
            if usd_dir_count >= 2:
                return False, f"Corelatie USD: {usd_dir_count} trade(s) {usd_direction} — skip pending {symbol}"

        # Correlation Guard
        corr_blocked, corr_msg = _check_correlation_guard(symbol, signal)
        if corr_blocked:
            log.info(f"Pending corr-guard: {corr_msg}")
            return False, corr_msg

        info = mt5.symbol_info(symbol)
        if info is None:
            return False, f"Symbol {symbol} negasit in MT5"
        if not info.visible:
            mt5.symbol_select(symbol, True)

        tick = mt5.symbol_info_tick(symbol)
        if tick is None:
            return False, "Nu s-a putut obtine pretul curent"

        # Spread filter (skip pentru ordine manuale)
        if strategy != "manual":
            spread_ok, spread_msg = _check_spread(symbol, tick, info)
            if not spread_ok:
                return False, spread_msg
        log.info(f"place_pending_order {symbol}: spread check {'skipped (manual)' if strategy=='manual' else spread_msg}")

        current_price = tick.ask if signal == "BUY" else tick.bid

        # MIN_STOPS_LEVEL — distanta minima intre pretul curent si entry pending
        try:
            stops_level = getattr(info, "trade_stops_level", None) or getattr(info, "stops_level", 0)
            broker_min  = stops_level * info.point
        except Exception:
            broker_min = 0
        pip_size = info.point * (10 if info.digits in (5, 3) else 1)

        # Auto-select LIMIT vs STOP
        if signal == "BUY":
            if entry_price < current_price:
                order_type      = mt5.ORDER_TYPE_BUY_LIMIT
                order_type_name = "BUY_LIMIT"
            else:
                order_type      = mt5.ORDER_TYPE_BUY_STOP
                order_type_name = "BUY_STOP"
        else:
            if entry_price > current_price:
                order_type      = mt5.ORDER_TYPE_SELL_LIMIT
                order_type_name = "SELL_LIMIT"
            else:
                order_type      = mt5.ORDER_TYPE_SELL_STOP
                order_type_name = "SELL_STOP"

        # Distanta minima broker (pending trebuie sa fie la cel putin broker_min de pretul curent)
        entry_dist = abs(entry_price - current_price)
        if broker_min > 0 and entry_dist < broker_min:
            return False, (f"Pending prea aproape de pretul curent "
                           f"({round(entry_dist/pip_size,1)} pips < "
                           f"min broker {round(broker_min/pip_size,1)} pips)")

        # SL minim (similar cu place_trade)
        sl_dist = abs(entry_price - sl)
        is_crypto = any(c in symbol for c in ("BTC","ETH","XRP","SOL","BNB","DOG","ADA","AVAX","LINK","LTC"))
        if is_crypto:
            min_sl_dist = max(broker_min, 10 * pip_size)
        else:
            min_sl_dist = max(broker_min, 20 * pip_size, entry_price * 0.0015)
        if sl_dist < min_sl_dist:
            if signal == "BUY":
                sl = entry_price - min_sl_dist
            else:
                sl = entry_price + min_sl_dist
            sl_dist = abs(entry_price - sl)
            log.info(f"SL ajustat la min broker: {round(sl, info.digits)}")

        tick_val  = info.trade_tick_value
        tick_size = info.trade_tick_size
        if tick_size <= 0 or tick_val <= 0 or sl_dist <= 0:
            return False, "Date simbol invalide sau SL invalid"

        # Recalculeaza TP pe baza TP_RATIO
        if signal == "BUY":
            tp = entry_price + sl_dist * TP_RATIO
        else:
            tp = entry_price - sl_dist * TP_RATIO

        lot_step = info.volume_step
        min_lot  = info.volume_min
        max_lot  = min(info.volume_max, MAX_LOT_GLOBAL)
        effective_risk = _get_effective_risk()
        if effective_risk != risk_dollars:
            log.info(f"EffectiveRisk pending: {effective_risk:.2f} USD (factor={_drawdown_factor:.2f})")

        lots = math.floor(effective_risk / (sl_dist / tick_size * tick_val) / lot_step) * lot_step
        lots = max(min_lot, min(max_lot, lots))
        lots = round(lots, 2)

        expiry_ts = int((_dt.now() + _td(hours=expiry_hours)).timestamp())

        req = {
            "action":     mt5.TRADE_ACTION_PENDING,
            "symbol":     symbol,
            "volume":     lots,
            "type":       order_type,
            "price":      round(entry_price, info.digits),
            "sl":         round(sl, info.digits),
            "tp":         round(tp, info.digits),
            "magic":      TRADE_MAGIC,
            "comment":    f"CV_{signal}_{strat_tag}_P",
            "type_time":  mt5.ORDER_TIME_SPECIFIED,
            "expiration": expiry_ts,
        }
        log.info(f"place_pending_order {symbol} {order_type_name} @ {round(entry_price,info.digits)} "
                 f"SL={round(sl,info.digits)} TP={round(tp,info.digits)} lots={lots} exp={expiry_hours}h")

        result = mt5.order_send(req)
        if result and result.retcode == mt5.TRADE_RETCODE_DONE:
            return True, (f"Pending {order_type_name} {lots}L @ {round(entry_price, info.digits)}"
                          f"  SL={round(sl, info.digits)}  TP={round(tp, info.digits)}"
                          f"  expirat in {expiry_hours}h")

        # Fallback fara expirare explicita (broker nu suporta ORDER_TIME_SPECIFIED)
        req2 = {**req, "type_time": mt5.ORDER_TIME_GTC}
        req2.pop("expiration", None)
        result2 = mt5.order_send(req2)
        if result2 and result2.retcode == mt5.TRADE_RETCODE_DONE:
            return True, (f"Pending {order_type_name} {lots}L @ {round(entry_price, info.digits)}"
                          f"  SL={round(sl, info.digits)}  TP={round(tp, info.digits)}  (GTC)")

        err_code = (result.retcode if result else -1)
        err_msgs = {
            10027: "AutoTrading dezactivat — Tools→Options→Expert Advisors",
            10030: "Filling mode incompatibil cu brokerul",
            10018: "Piata inchisa",
            10019: "Fonduri insuficiente",
            10016: f"SL/TP invalid ({info.digits} zecimale sau prea aproape)",
            10014: f"Volum invalid (lots={lots})",
            10013: "Parametri invalizi",
            10006: "Ordin respins de broker",
            10015: "Pret invalid (pending prea aproape de pret curent)",
        }
        return False, f"Eroare pending: {err_msgs.get(err_code, f'retcode={err_code}')}"
    finally:
        _pending_symbols.discard(pending_key)


# ── Position Monitor (Trailing Stop ATR + Partial Close + Time-Based Exit) ────
import threading as _threading

_MONITOR_SETTINGS = {
    "trailing_atr_enabled": False,
    "trailing_atr_mult":    2.0,
    "partial_close_enabled": False,
    "partial_close_pct":    50,    # % din pozitie de inchis la TP1
    "time_exit_enabled":    False,
    "time_exit_bars":       20,    # bare pe TF principal fara miscare → close
    "time_exit_tf":         "H1",  # TF de referinta pentru time exit
}

_monitor_thread: _threading.Thread | None = None
_monitor_running = False

# Map ticket → {entry, sl, tp, signal, bars_flat, strategy, volume, symbol}
_monitored_positions: dict[int, dict] = {}
# Set de tickete deschise la ciclul precedent — pentru detectie inchidere
_prev_open_tickets: set[int] = set()


def start_position_monitor():
    global _monitor_thread, _monitor_running
    if _monitor_running:
        return
    _monitor_running = True
    _monitor_thread = _threading.Thread(target=_position_monitor_loop,
                                        daemon=True, name="pos-monitor")
    _monitor_thread.start()
    log.info("Position monitor pornit.")


def stop_position_monitor():
    global _monitor_running
    _monitor_running = False


def _position_monitor_loop():
    import time as _time
    while _monitor_running:
        try:
            _run_monitor_cycle()
        except Exception as exc:
            log.warning(f"Monitor cycle error: {exc}")
        _time.sleep(10)  # ruleaza la fiecare 10 secunde
    log.info("Position monitor oprit.")


def _detect_close_reason(ticket: int, pos_info: dict) -> tuple[str, float, float]:
    """
    Interogheaza istoricul MT5 pentru un ticket inchis.
    Returneaza (reason, close_price, pnl).
    reason: "SL" | "TP" | "MANUAL" | "NEWS" | "TIME_EXIT" | "STOP_OUT" | "OTHER"

    MT5 deal reason codes:
      0 = CLIENT  (terminal desktop — manual)
      1 = MOBILE  (app mobil — manual)
      2 = WEB     (web terminal — manual)
      3 = EXPERT  (EA/bot — automat intern)
      4 = SL      (stop loss)
      5 = TP      (take profit)
      6 = SO      (stop out / margin call)
    """
    try:
        # Cel mai precis: cauta direct dupa position id
        deals = mt5.history_deals_get(position=ticket) or []

        if not deals:
            # Fallback: ultimele 24h
            from datetime import datetime, timezone, timedelta
            date_to   = datetime.now(tz=timezone.utc)
            date_from = date_to - timedelta(days=1)
            all_deals = mt5.history_deals_get(date_from, date_to) or []
            deals = [d for d in all_deals if d.position_id == ticket]

        if not deals:
            # Ultima sansa: foloseste datele cunoscute pentru a ghici motivul
            pdata     = pos_info or {}
            sl        = pdata.get("sl", 0.0)
            tp        = pdata.get("tp", 0.0)
            signal    = pdata.get("signal", "BUY")
            # Nu putem determina motivul — returnam MANUAL ca default rezonabil
            return "MANUAL", 0.0, 0.0

        # Deal-ul de iesire = entry == 1 (DEAL_ENTRY_OUT)
        # Unele brokere folosesc valori diferite; luam ultimul deal cu profit != 0
        out_deal = None
        for d in sorted(deals, key=lambda x: x.time, reverse=True):
            if getattr(d, "entry", -1) == 1:   # DEAL_ENTRY_OUT = 1
                out_deal = d
                break
        if out_deal is None:
            # Fallback: ultimul deal al pozitiei
            out_deal = max(deals, key=lambda d: d.time)

        close_price = float(out_deal.price)
        pnl         = float(out_deal.profit + out_deal.commission + out_deal.swap)
        reason_code = int(getattr(out_deal, "reason", -1))
        comment     = (getattr(out_deal, "comment", "") or "").lower()

        # Prioritate: comment-ul nostru intern > reason code MT5
        if "news" in comment or "stire" in comment:
            reason = "NEWS"
        elif "weekend" in comment:
            reason = "NEWS"
        elif "time" in comment or "flat" in comment or "time_exit" in comment:
            reason = "TIME_EXIT"
        elif "partial" in comment:
            reason = "PARTIAL"
        elif reason_code == 4:
            reason = "SL"
        elif reason_code == 5:
            reason = "TP"
        elif reason_code == 6:
            reason = "STOP_OUT"
        elif reason_code in (0, 1, 2):
            # CLIENT / MOBILE / WEB = inchis manual de user
            reason = "MANUAL"
        elif reason_code == 3:
            # EXPERT = inchis de EA/bot (time exit, partial etc.)
            reason = "OTHER"
        else:
            reason = "MANUAL"   # default mai bun decat "OTHER"

        return reason, close_price, pnl
    except Exception as exc:
        log.warning(f"_detect_close_reason {ticket}: {exc}")
        return "MANUAL", 0.0, 0.0


_STRAT_TAG_MAP = {
    "CLASSI": "Classic",  "CLA": "Classic",
    "SMC":    "SMC",
    "EOB":    "EOB + Unicorn",
    "MACD":   "MACD",
    "BOLLIN": "Bollinger Bands", "BOL": "Bollinger Bands",
    "SUPERT": "Supertrend",      "SUP": "Supertrend",
    "LONDON": "London Breakout", "LON": "London Breakout",
    "NY_BRE": "NY Breakout",     "NY":  "NY Breakout",
    "CHINA_": "China Session",   "CHI": "China Session",
    "RSI_DI": "RSI Divergence",  "RSI": "RSI Divergence",
    "ENGULF": "Engulfing / Pin Bar", "ENG": "Engulfing / Pin Bar",
    "ICHIMO": "Ichimoku",        "ICH": "Ichimoku",
    "EMA_CR": "EMA Cross 8/21",  "EMA": "EMA Cross 8/21",
    "VWAP_B": "VWAP Bounce",     "VWA": "VWAP Bounce",
    "KELTNE": "Keltner Channel", "KEL": "Keltner Channel",
    "COMBIN": "Combined",        "COM": "Combined",
    "CANDLF": "CandleForge",     "CAN": "CandleForge",
}

def _tag_to_strat(comment: str) -> str:
    if not comment or not comment.startswith("CV_"):
        return "Classic"
    parts = comment.split("_", 2)
    tag = parts[2] if len(parts) >= 3 else ""
    if not tag:
        return "Classic"
    if tag in _STRAT_TAG_MAP:
        return _STRAT_TAG_MAP[tag]
    tag3 = tag[:3]
    for key, name in _STRAT_TAG_MAP.items():
        if key[:3] == tag3:
            return name
    return tag


def _run_monitor_cycle():
    global _prev_open_tickets
    if not MT5_AVAILABLE or mt5 is None:
        return
    if not mt5.initialize():
        return

    positions    = mt5.positions_get() or []
    current_tickets = {p.ticket for p in positions}

    # ── Detectie pozitii inchise fata de ciclul precedent ────────────────────
    closed_tickets = _prev_open_tickets - current_tickets
    for ticket in closed_tickets:
        pdata = _monitored_positions.get(ticket)
        if not pdata:
            continue
        reason, close_price, pnl = _detect_close_reason(ticket, pdata)
        symbol   = pdata.get("symbol", "?")
        signal   = pdata.get("signal", "?")
        strategy = pdata.get("strategy", "?")
        entry    = pdata.get("entry", 0.0)
        sl       = pdata.get("sl", 0.0)
        tp       = pdata.get("tp", 0.0)
        volume   = pdata.get("volume", 0.0)

        log.info(f"[Monitor] Pozitie inchisa: {symbol} ticket={ticket} motiv={reason} pnl={pnl:.2f}$")

        # Notificare Telegram
        try:
            import notifier as _tg
            _tg.notify_close(
                symbol=symbol, signal=signal, strategy=strategy,
                entry=entry, close_price=close_price,
                sl=sl, tp=tp, pnl=pnl, reason=reason, volume=volume,
            )
        except Exception as exc:
            log.warning(f"notify_close error: {exc}")

        # Log in review_log.json
        try:
            from autotrader import _log_action
            _log_action({
                "timestamp":   __import__("datetime").datetime.now().isoformat(),
                "symbol":      symbol,
                "signal":      f"CLOSE ({signal})",
                "confidence":  0,
                "executed":    True,
                "result":      f"{reason} | pnl={pnl:+.2f}$ | close={close_price:.5f}",
                "strategy":    strategy,
                "close_reason": reason,
                "pnl":         round(pnl, 2),
                "entry":       entry,
                "close_price": close_price,
                "sl":          sl,
                "tp":          tp,
                "volume":      volume,
            })
        except Exception as exc:
            log.warning(f"_log_action close error: {exc}")

        # Curata din monitored
        _monitored_positions.pop(ticket, None)

    _prev_open_tickets = current_tickets

    if not positions:
        return

    settings = _MONITOR_SETTINGS

    for pos in positions:
        ticket  = pos.ticket
        symbol  = pos.symbol
        pos_type = pos.type   # 0=BUY, 1=SELL
        signal  = "BUY" if pos_type == 0 else "SELL"
        entry   = pos.price_open
        sl_cur  = pos.sl
        tp_cur  = pos.tp
        volume  = pos.volume
        profit  = pos.profit

        # ── Salveaza datele pozitiei la prima vedere (pentru detectia inchiderii) ──
        if ticket not in _monitored_positions:
            strat = _tag_to_strat(pos.comment or "")
            _monitored_positions[ticket] = {
                "symbol":   symbol,
                "signal":   signal,
                "entry":    float(entry),
                "sl":       float(sl_cur),
                "tp":       float(tp_cur),
                "volume":   float(volume),
                "strategy": strat,
            }
        else:
            # Actualizeaza SL/TP daca au fost modificate (trailing stop etc.)
            _monitored_positions[ticket]["sl"] = float(sl_cur)
            _monitored_positions[ticket]["tp"] = float(tp_cur)

        info = mt5.symbol_info(symbol)
        tick = mt5.symbol_info_tick(symbol)
        if not info or not tick:
            continue

        price_now = tick.bid if signal == "BUY" else tick.ask
        pip_size  = info.point * (10 if info.digits in (5, 3) else 1)

        # ── ATR pentru trailing si time exit ──
        atr_val = None
        try:
            df_h, _ = fetch(symbol, settings["time_exit_tf"], 50)
            if df_h is not None and len(df_h) >= 15:
                atr_val = float((df_h["high"] - df_h["low"]).rolling(14).mean().iloc[-1])
        except Exception:
            pass

        # ── Partial Close (50% la TP1 = entry + 1R) ──
        if settings["partial_close_enabled"] and tp_cur > 0 and sl_cur > 0:
            risk      = abs(entry - sl_cur)
            tp1_level = (entry + risk) if signal == "BUY" else (entry - risk)
            tp1_hit   = (signal == "BUY" and price_now >= tp1_level) or \
                        (signal == "SELL" and price_now <= tp1_level)
            partial_done = _monitored_positions.get(ticket, {}).get("partial_done", False)

            if tp1_hit and not partial_done:
                partial_vol = round(volume * (settings["partial_close_pct"] / 100.0), 2)
                partial_vol = max(info.volume_min, partial_vol)
                if partial_vol < volume:
                    _partial_close_position(pos, partial_vol, price_now, info)
                    # Muta SL la breakeven
                    _modify_sl(pos, entry, info)
                    _monitored_positions.setdefault(ticket, {})["partial_done"] = True
                    log.info(f"Partial close {ticket} {symbol}: {partial_vol} loturi la TP1 ({tp1_level:.5f})")

        # ── Trailing Stop ATR ──
        if settings["trailing_atr_enabled"] and atr_val and sl_cur > 0:
            trail_dist = atr_val * settings["trailing_atr_mult"]
            if signal == "BUY":
                new_sl = price_now - trail_dist
                if new_sl > sl_cur and new_sl < price_now:
                    _modify_sl(pos, new_sl, info)
                    log.info(f"Trailing BUY {ticket} {symbol}: SL {sl_cur:.5f} → {new_sl:.5f}")
            else:
                new_sl = price_now + trail_dist
                if new_sl < sl_cur and new_sl > price_now:
                    _modify_sl(pos, new_sl, info)
                    log.info(f"Trailing SELL {ticket} {symbol}: SL {sl_cur:.5f} → {new_sl:.5f}")

        # ── Time-Based Exit ──
        if settings["time_exit_enabled"] and atr_val:
            move = abs(price_now - entry)
            flat_threshold = atr_val * 0.3
            mon  = _monitored_positions.setdefault(ticket, {})
            if move < flat_threshold:
                mon["bars_flat"] = mon.get("bars_flat", 0) + 1
            else:
                mon["bars_flat"] = 0

            if mon["bars_flat"] >= settings["time_exit_bars"]:
                _close_position_market(pos, price_now, info)
                _monitored_positions.pop(ticket, None)
                log.info(f"Time exit {ticket} {symbol}: flat {mon['bars_flat']} cicli → inchis la {price_now}")


def _partial_close_position(pos, volume: float, price: float, info):
    try:
        order_type = mt5.ORDER_TYPE_SELL if pos.type == 0 else mt5.ORDER_TYPE_BUY
        tick = mt5.symbol_info_tick(pos.symbol)
        close_price = tick.bid if pos.type == 0 else tick.ask
        req = {
            "action":   mt5.TRADE_ACTION_DEAL,
            "symbol":   pos.symbol,
            "volume":   volume,
            "type":     order_type,
            "price":    close_price,
            "position": pos.ticket,
            "deviation": 20,
            "magic":    TRADE_MAGIC,
            "comment":  "CV_PARTIAL_CLOSE",
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": mt5.ORDER_FILLING_IOC,
        }
        mt5.order_send(req)
    except Exception as exc:
        log.warning(f"Partial close error {pos.ticket}: {exc}")


def _close_position_market(pos, price: float, info):
    try:
        order_type = mt5.ORDER_TYPE_SELL if pos.type == 0 else mt5.ORDER_TYPE_BUY
        tick = mt5.symbol_info_tick(pos.symbol)
        close_price = tick.bid if pos.type == 0 else tick.ask
        req = {
            "action":   mt5.TRADE_ACTION_DEAL,
            "symbol":   pos.symbol,
            "volume":   pos.volume,
            "type":     order_type,
            "price":    close_price,
            "position": pos.ticket,
            "deviation": 20,
            "magic":    TRADE_MAGIC,
            "comment":  "CV_TIME_EXIT",
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": mt5.ORDER_FILLING_IOC,
        }
        mt5.order_send(req)
    except Exception as exc:
        log.warning(f"Market close error {pos.ticket}: {exc}")


def _modify_sl(pos, new_sl: float, info):
    try:
        req = {
            "action":   mt5.TRADE_ACTION_SLTP,
            "symbol":   pos.symbol,
            "position": pos.ticket,
            "sl":       round(new_sl, info.digits),
            "tp":       pos.tp,
        }
        mt5.order_send(req)
    except Exception as exc:
        log.warning(f"Modify SL error {pos.ticket}: {exc}")


# Porneste monitorul la startup
start_position_monitor()

# Wire notifier → place_trade (fara import circular)
def _tg_place_trade(symbol, signal, sl, tp):
    return place_trade(symbol, signal, sl, tp, risk_dollars=RISK_DOLLARS, strategy="telegram")
_tg._trade_callback = _tg_place_trade


# ── Build chart ───────────────────────────────────────────────────────────────
def build_chart(symbol, tf, lookback, compact=False, return_fig=False):
    df, source = fetch(symbol, tf, lookback)
    if df is None:
        return f"<div style='color:#ef5350;padding:12px'>Nu s-au putut incarca datele MT5 pentru {symbol}/{tf}.</div>"
    highs = df["high"].values
    lows  = df["low"].values
    dates = df.index
    ph_idx, pl_idx = find_pivots(df, lookback=5)
    trend = detect_trend(ph_idx, pl_idx, highs, lows, recent_bars=100)
    ema20 = df["close"].ewm(span=20, adjust=False).mean()
    ema50 = df["close"].ewm(span=50, adjust=False).mean()
    delta = df["close"].diff()
    gain  = delta.clip(lower=0).rolling(14).mean()
    loss  = (-delta.clip(upper=0)).rolling(14).mean()
    rsi   = 100 - 100 / (1 + gain / loss.replace(0, np.nan))
    entry_signal, entry_reason, price_now = calc_entry(df, ph_idx, pl_idx, trend, ema20, ema50, rsi)
    fib_data = calc_fib_levels(df, ph_idx, pl_idx, trend)
    height      = 460 if compact else 680
    row_heights = [0.72, 0.28] if compact else [0.78, 0.22]
    fig = make_subplots(rows=2, cols=1, row_heights=row_heights,
                        shared_xaxes=True, vertical_spacing=0.03)
    fig.add_trace(go.Candlestick(
        x=dates, open=df["open"], high=df["high"], low=df["low"], close=df["close"],
        name="OHLC", increasing_line_color="#26a69a", decreasing_line_color="#ef5350",
        increasing_fillcolor="#26a69a", decreasing_fillcolor="#ef5350",
        showlegend=not compact), row=1, col=1)
    fig.add_trace(go.Scatter(x=dates, y=ema20, line=dict(color="#ffeb3b", width=1.2),
        name="EMA20", opacity=0.9, showlegend=not compact), row=1, col=1)
    fig.add_trace(go.Scatter(x=dates, y=ema50, line=dict(color="#ff9800", width=1.2),
        name="EMA50", opacity=0.9, showlegend=not compact), row=1, col=1)
    if ph_idx:
        fig.add_trace(go.Scatter(x=[dates[i] for i in ph_idx], y=[highs[i]*1.0002 for i in ph_idx],
            mode="markers+text", marker=dict(symbol="triangle-down", size=8 if compact else 10, color="#ef5350"),
            text=["H"]*len(ph_idx), textposition="top center", textfont=dict(size=8, color="#ef5350"),
            name="Pivot High", showlegend=not compact), row=1, col=1)
    if pl_idx:
        fig.add_trace(go.Scatter(x=[dates[i] for i in pl_idx], y=[lows[i]*0.9998 for i in pl_idx],
            mode="markers+text", marker=dict(symbol="triangle-up", size=8 if compact else 10, color="#26a69a"),
            text=["L"]*len(pl_idx), textposition="bottom center", textfont=dict(size=8, color="#26a69a"),
            name="Pivot Low", showlegend=not compact), row=1, col=1)
    if len(ph_idx) >= 2:
        lc = "#ef5350" if trend == "DESCENDING" else "#ff9800"
        fig.add_trace(go.Scatter(x=[dates[i] for i in ph_idx[-6:]], y=[highs[i] for i in ph_idx[-6:]],
            mode="lines", line=dict(color=lc, width=1, dash="dot"), showlegend=False, opacity=0.7), row=1, col=1)
    if len(pl_idx) >= 2:
        lc = "#26a69a" if trend == "ASCENDING" else "#ef5350"
        fig.add_trace(go.Scatter(x=[dates[i] for i in pl_idx[-6:]], y=[lows[i] for i in pl_idx[-6:]],
            mode="lines", line=dict(color=lc, width=1, dash="dot"), showlegend=False, opacity=0.7), row=1, col=1)
    # ── Fibonacci lines ───────────────────────────────────────────────────────
    FIB_COLORS = {
        0.0:   ("#888",   "0%"),
        0.236: ("#b0bec5","23.6%"),
        0.382: ("#64b5f6","38.2%"),
        0.5:   ("#fff176","50%"),
        0.618: ("#ffb74d","61.8%"),  # golden ratio — cel mai important
        0.786: ("#ef9a9a","78.6%"),
        1.0:   ("#888",   "100%"),
    }
    if fib_data and not compact:
        for f, (color, label) in FIB_COLORS.items():
            fv = fib_data["levels"].get(f)
            if fv is None:
                continue
            is_golden = f in (0.382, 0.5, 0.618)
            fv_str = f"{fv:.5f}" if fv < 100 else f"{fv:.2f}"
            fig.add_hline(y=fv,
                line=dict(color=color, width=1.5 if is_golden else 0.8,
                          dash="solid" if is_golden else "dot"),
                row=1, col=1)
            fig.add_annotation(
                xref="paper", yref="y",
                x=1.002, y=fv,
                text=f"<b>{label}</b> {fv_str}",
                showarrow=False,
                font=dict(color=color, size=9 if not is_golden else 10),
                xanchor="left", yanchor="middle",
                bgcolor="rgba(17,17,17,0.7)",
                row=1, col=1,
            )
        # zona golden zone (38.2% - 61.8%) colorata subtil
        fib_382 = fib_data["levels"].get(0.382)
        fib_618 = fib_data["levels"].get(0.618)
        if fib_382 and fib_618:
            y0, y1 = min(fib_382, fib_618), max(fib_382, fib_618)
            fig.add_hrect(y0=y0, y1=y1,
                fillcolor="rgba(255,183,77,0.07)",
                line=dict(color="rgba(255,183,77,0.3)", width=1),
                row=1, col=1)

    fig.add_trace(go.Scatter(x=dates, y=rsi, line=dict(color="#ab47bc", width=1.2),
        name="RSI(14)", showlegend=not compact), row=2, col=1)
    fig.add_hline(y=70, line=dict(color="#ef5350", width=0.8, dash="dash"), row=2, col=1)
    fig.add_hline(y=30, line=dict(color="#26a69a", width=0.8, dash="dash"), row=2, col=1)
    fig.add_hline(y=50, line=dict(color="#555",    width=0.6, dash="dot"),  row=2, col=1)
    trend_label = {"ASCENDING":"▲ ASCENDING","DESCENDING":"▼ DESCENDING","RANGING":"— RANGING"}[trend]
    trend_col   = {"ASCENDING":"#26a69a","DESCENDING":"#ef5350","RANGING":"#aaa"}[trend]
    entry_badge = ""
    if entry_signal == "BUY":   entry_badge = "  <span style='color:#26a69a'>● BUY</span>"
    elif entry_signal == "SELL": entry_badge = "  <span style='color:#ef5350'>● SELL</span>"
    # Range initial: ultimele ~300 bare, dar cu type="date" explicit
    x_start = dates[-300] if len(dates) > 300 else dates[0]
    x_end   = dates[-1]
    # Padding dreapta ~30 bare (ca TradingView) — calculat din timeframe
    try:
        bar_delta = dates[-1] - dates[-2]
        x_end_padded = dates[-1] + bar_delta * 30
    except Exception:
        x_end_padded = dates[-1]
    fig.update_layout(
        template="plotly_dark", paper_bgcolor="#111", plot_bgcolor="#111",
        uirevision=f"{symbol}_{tf}",   # pastreaza zoom/pan al utilizatorului intre refresh-uri
        title=dict(text=(f"<b>{symbol}</b> — {tf}  |  Trend: <span style='color:{trend_col}'>{trend_label}</span>"
                         f"  |  Pivoti: {len(ph_idx)} H, {len(pl_idx)} L{entry_badge}"
                         f"  |  <span style='color:#888'>sursa: {source}</span>"),
                   font=dict(size=13 if compact else 14, color="#ddd")),
        xaxis_rangeslider_visible=False, height=height,
        dragmode="pan",
        hovermode="x unified",
        margin=dict(l=50, r=110, t=50, b=36 if compact else 50),
        legend=dict(orientation="h", yanchor="bottom", y=1.01, xanchor="left", x=0,
                    font=dict(size=9 if compact else 10), bgcolor="rgba(0,0,0,0)"),
        # Y — spike line orizontal transparent + etichetare pret pe dreapta
        yaxis=dict(
            gridcolor="#222", zerolinecolor="#333",
            uirevision=f"{symbol}_{tf}", autorange=True, fixedrange=False,
            showspikes=True,
            spikemode="across", spikesnap="cursor",
            spikecolor="rgba(180,180,180,0.35)",
            spikethickness=1, spikedash="dot",
            side="right",
        ),
        yaxis2=dict(
            gridcolor="#222", zerolinecolor="#333",
            range=[0,100], fixedrange=False,
            side="right",
            showspikes=True,
            spikemode="across", spikesnap="cursor",
            spikecolor="rgba(180,180,180,0.25)",
            spikethickness=1, spikedash="dot",
        ),
        xaxis=dict(
            type="date",
            uirevision=f"{symbol}_{tf}",
            range=[x_start, x_end_padded],   # padding dreapta
            autorange=False,
            rangeslider=dict(visible=False),
            showspikes=True,
            spikemode="across", spikesnap="cursor",
            spikecolor="rgba(180,180,180,0.35)",
            spikethickness=1, spikedash="dot",
        ),
        xaxis2=dict(
            type="date", gridcolor="#222",
            showspikes=True,
            spikemode="across", spikesnap="cursor",
            spikecolor="rgba(180,180,180,0.25)",
            spikethickness=1, spikedash="dot",
        ),
    )

    # ── Zone EOB vizibile pe grafic ───────────────────────────────────────────
    try:
        from strategies.eob import get_eob_context
        eob = get_eob_context(symbol)
        if eob:
            def _eob_zone(y0, y1, color_hex, label, opacity=0.20, dash="solid", width=2):
                r, g, b = int(color_hex[1:3],16), int(color_hex[3:5],16), int(color_hex[5:7],16)
                fig.add_hrect(
                    y0=y0, y1=y1, row=1, col=1,
                    fillcolor=f"rgba({r},{g},{b},{opacity})",
                    line=dict(color=color_hex, width=width, dash=dash),
                    annotation_text=f"  {label}  [{round(y0,5)} – {round(y1,5)}]",
                    annotation_font=dict(color=color_hex, size=10, family="monospace"),
                    annotation_bgcolor=f"rgba(0,0,0,0.6)",
                    annotation_bordercolor=color_hex,
                    annotation_borderwidth=1,
                    annotation_position="top left",
                )

            # HTF EOB — zona principala (mai vizibila)
            z = eob.get("htf_bear_zone")
            if z:
                _eob_zone(z["zone_low"], z["zone_high"], "#ef5350",
                          "🔴 HTF EOB SELL", opacity=0.25, width=2)
            z = eob.get("htf_bull_zone")
            if z:
                _eob_zone(z["zone_low"], z["zone_high"], "#26a69a",
                          "🟢 HTF EOB BUY", opacity=0.25, width=2)

            # LTF entry — EOB din EOB (punctata, mai subtire)
            z = eob.get("ltf_entry_bear")
            if z:
                _eob_zone(z["zone_low"], z["zone_high"], "#ff5252",
                          "⬇ LTF Entry SELL", opacity=0.35, dash="dot", width=1)
            z = eob.get("ltf_entry_bull")
            if z:
                _eob_zone(z["zone_low"], z["zone_high"], "#69f0ae",
                          "⬆ LTF Entry BUY", opacity=0.35, dash="dot", width=1)

            # MTF FVG — dezactivat acum, pastrat pentru viitor
            # z = eob.get("mtf_bear_fvg")
            # if z: _eob_zone(z["bottom"], z["top"], "#ff9800", "MTF FVG Bear", opacity=0.15, dash="dash")
            # z = eob.get("mtf_bull_fvg")
            # if z: _eob_zone(z["bottom"], z["top"], "#00bcd4", "MTF FVG Bull", opacity=0.15, dash="dash")

    except Exception as _eob_ex:
        log.debug(f"EOB zones chart inject: {_eob_ex}")

    if return_fig:
        return fig
    # Post-script: auto-rescale Y cand user-ul face zoom pe X (ca TradingView)
    auto_rescale_js = """
    (function(){
        var gd = document.getElementById('{plot_id}');
        if(!gd) return;
        function autoRescaleY(){
            try{
                var xr = gd.layout.xaxis.range;
                if(!xr || xr.length!==2) return;
                var x0 = new Date(xr[0]).getTime();
                var x1 = new Date(xr[1]).getTime();
                var traces = gd.data || [];
                var yMin = Infinity, yMax = -Infinity;
                for(var t=0; t<traces.length; t++){
                    var tr = traces[t];
                    if(tr.yaxis && tr.yaxis !== 'y') continue;
                    var xs = tr.x || [];
                    var lows  = tr.low  || tr.y || [];
                    var highs = tr.high || tr.y || [];
                    for(var i=0; i<xs.length; i++){
                        var xi = new Date(xs[i]).getTime();
                        if(xi < x0 || xi > x1) continue;
                        var lo = lows[i], hi = highs[i];
                        if(lo!=null && lo<yMin) yMin = lo;
                        if(hi!=null && hi>yMax) yMax = hi;
                    }
                }
                if(yMin<Infinity && yMax>-Infinity && yMax>yMin){
                    var pad = (yMax-yMin)*0.08;
                    Plotly.relayout(gd, {'yaxis.range':[yMin-pad, yMax+pad]});
                }
            }catch(e){}
        }
        gd.on('plotly_relayout', function(ev){
            if(ev['xaxis.range[0]']!==undefined || ev['xaxis.range']!==undefined){
                autoRescaleY();
            }
        });
        setTimeout(autoRescaleY, 150);
    })();
    """
    return fig.to_html(full_html=False, include_plotlyjs="cdn", post_script=auto_rescale_js)

SNAPSHOTS_DIR = os.path.join(os.path.dirname(__file__), "snapshots")
os.makedirs(SNAPSHOTS_DIR, exist_ok=True)

def save_trade_snapshot(ticket, symbol, signal, entry, sl, tp, tf="M5", analysis=None):
    """Genereaza si salveaza un grafic HTML cu Entry/SL/TP la momentul executiei."""
    try:
        from datetime import datetime
        df, source = fetch(symbol, tf, 200)
        if df is None:
            log.warning(f"snapshot {ticket}: nu s-au putut incarca datele")
            return

        highs  = df["high"].values
        lows   = df["low"].values
        dates  = df.index
        ph_idx, pl_idx = find_pivots(df, lookback=5)
        trend  = detect_trend(ph_idx, pl_idx, highs, lows, recent_bars=100)
        ema20  = df["close"].ewm(span=20, adjust=False).mean()
        ema50  = df["close"].ewm(span=50, adjust=False).mean()
        delta  = df["close"].diff()
        gain   = delta.clip(lower=0).rolling(14).mean()
        loss   = (-delta.clip(upper=0)).rolling(14).mean()
        rsi    = 100 - 100 / (1 + gain / loss.replace(0, np.nan))

        fig = make_subplots(rows=2, cols=1, row_heights=[0.78, 0.22],
                            shared_xaxes=True, vertical_spacing=0.03)

        # Candlestick
        fig.add_trace(go.Candlestick(
            x=dates, open=df["open"], high=df["high"], low=df["low"], close=df["close"],
            name="OHLC", increasing_line_color="#26a69a", decreasing_line_color="#ef5350",
            increasing_fillcolor="#26a69a", decreasing_fillcolor="#ef5350"), row=1, col=1)

        # EMA
        fig.add_trace(go.Scatter(x=dates, y=ema20, line=dict(color="#ffeb3b", width=1.2), name="EMA20"), row=1, col=1)
        fig.add_trace(go.Scatter(x=dates, y=ema50, line=dict(color="#ff9800", width=1.2), name="EMA50"), row=1, col=1)

        # Pivoti
        if ph_idx:
            fig.add_trace(go.Scatter(x=[dates[i] for i in ph_idx], y=[highs[i]*1.0002 for i in ph_idx],
                mode="markers", marker=dict(symbol="triangle-down", size=8, color="#ef5350"),
                name="Pivot H"), row=1, col=1)
        if pl_idx:
            fig.add_trace(go.Scatter(x=[dates[i] for i in pl_idx], y=[lows[i]*0.9998 for i in pl_idx],
                mode="markers", marker=dict(symbol="triangle-up", size=8, color="#26a69a"),
                name="Pivot L"), row=1, col=1)

        # ── Entry ca punct mare pe ultima candela ──
        entry_col  = "#26a69a" if signal == "BUY" else "#ef5350"
        entry_sym  = "triangle-up" if signal == "BUY" else "triangle-down"
        entry_date = dates[-1]
        fig.add_trace(go.Scatter(
            x=[entry_date], y=[entry],
            mode="markers+text",
            marker=dict(symbol=entry_sym, size=18, color=entry_col,
                        line=dict(color="#fff", width=1.5)),
            text=[f"  ENTRY {signal} @ {round(entry,5)}"],
            textposition="middle right",
            textfont=dict(color=entry_col, size=11),
            name=f"Entry {signal}", showlegend=True,
        ), row=1, col=1)

        # ── SL si TP ca linii orizontale ──
        fig.add_hline(y=sl, line=dict(color="#ef5350", width=1.5, dash="dash"),
            annotation_text=f"  SL @ {round(sl,5)}",
            annotation_font=dict(color="#ef5350", size=10), row=1, col=1)
        fig.add_hline(y=tp, line=dict(color="#26a69a", width=1.5, dash="dash"),
            annotation_text=f"  TP @ {round(tp,5)}",
            annotation_font=dict(color="#26a69a", size=10), row=1, col=1)

        # Zona colorata intre SL si TP
        sl_color = "rgba(239,83,80,0.07)"
        tp_color = "rgba(38,166,154,0.07)"
        fig.add_hrect(y0=sl, y1=entry, fillcolor=sl_color, line_width=0, row=1, col=1)
        fig.add_hrect(y0=entry, y1=tp, fillcolor=tp_color, line_width=0, row=1, col=1)

        # RSI
        fig.add_trace(go.Scatter(x=dates, y=rsi, line=dict(color="#ab47bc", width=1.2), name="RSI(14)"), row=2, col=1)
        fig.add_hline(y=70, line=dict(color="#ef5350", width=0.8, dash="dash"), row=2, col=1)
        fig.add_hline(y=30, line=dict(color="#26a69a", width=0.8, dash="dash"), row=2, col=1)

        trend_label = {"ASCENDING":"▲ ASCENDING","DESCENDING":"▼ DESCENDING","RANGING":"— RANGING"}.get(trend, trend)
        trend_col   = {"ASCENDING":"#26a69a","DESCENDING":"#ef5350","RANGING":"#aaa"}.get(trend, "#aaa")
        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        rr = abs(tp - entry) / abs(entry - sl) if abs(entry - sl) > 0 else 0
        title_text = (
            f"<b>#{ticket} — {symbol} {signal}</b>  |  "
            f"Entry: {round(entry,5)}  SL: {round(sl,5)}  TP: {round(tp,5)}  R:R 1:{round(rr,1)}<br>"
            f"<span style='font-size:11px;color:#888'>Trend: <span style='color:{trend_col}'>{trend_label}</span>  |  "
            f"TF: {tf}  |  {ts}</span>"
        )
        if analysis:
            just = analysis.get("justification", [])
            if just:
                title_text += f"<br><span style='font-size:10px;color:#666'>{' · '.join(just[:3])}</span>"

        fig.update_layout(
            template="plotly_dark", paper_bgcolor="#111", plot_bgcolor="#111",
            title=dict(text=title_text, font=dict(size=13, color="#ddd")),
            xaxis_rangeslider_visible=False, height=680,
            margin=dict(l=60, r=110, t=90, b=20),
            legend=dict(orientation="h", yanchor="bottom", y=1.01, xanchor="left", x=0,
                        font=dict(size=9), bgcolor="rgba(0,0,0,0)"),
            yaxis=dict(gridcolor="#222", zerolinecolor="#333"),
            yaxis2=dict(gridcolor="#222", zerolinecolor="#333", range=[0,100]),
            xaxis2=dict(gridcolor="#222"),
        )

        html_content = f"""<!DOCTYPE html><html><head><meta charset="utf-8">
<title>Snapshot #{ticket} {symbol}</title>
<style>body{{background:#111;margin:0;padding:0}}</style></head><body>
{fig.to_html(full_html=False, include_plotlyjs="cdn")}
</body></html>"""

        path = os.path.join(SNAPSHOTS_DIR, f"{ticket}.html")
        with open(path, "w", encoding="utf-8") as f:
            f.write(html_content)

        # salveaza si metadata JSON
        meta_path = os.path.join(SNAPSHOTS_DIR, f"{ticket}.json")
        meta = {
            "ticket":  ticket,
            "symbol":  symbol,
            "signal":  signal,
            "entry":   entry,
            "sl":      sl,
            "tp":      tp,
            "tf":      tf,
            "ts":      ts,
            "rr":      round(rr, 2),
        }
        with open(meta_path, "w") as f:
            json.dump(meta, f)

        log.info(f"Snapshot salvat: snapshots/{ticket}.html")
    except Exception as e:
        log.error(f"save_trade_snapshot eroare: {e}")


def build_multi(symbol, selected_tfs, bars_per_tf):
    parts = []
    for tf in ALL_TFS:
        if tf not in selected_tfs:
            continue
        bars = bars_per_tf.get(tf, 500)
        html = build_chart(symbol, tf, bars, compact=True)
        parts.append(f'<div style="margin-bottom:6px">{html}</div>')
    return "\n".join(parts)

# ── HTML ──────────────────────────────────────────────────────────────────────
HTML = """<!DOCTYPE html>
<html><head>
<meta charset="utf-8">
<title>ChartVisualizer</title>
<style>
* { box-sizing:border-box; margin:0; padding:0; }
body { background:#111; color:#eee; font-family:'Segoe UI',sans-serif; }

/* ── Navbar ── */
.navbar {
    background:#161616; border-bottom:1px solid #2a2a2a;
    padding:0 20px; height:44px;
    display:flex; align-items:center; justify-content:space-between;
}
.navbar-brand {
    font-size:1rem; font-weight:600; color:#eee; letter-spacing:0.3px;
    display:flex; align-items:center; gap:8px;
}
.navbar-brand .brand-dot { width:8px; height:8px; border-radius:50%; background:#1976d2; display:inline-block; }
.navbar-links { display:flex; gap:8px; align-items:center; }

/* ── Controls bar ── */
.controls-bar {
    background:#1a1a1a; border-bottom:1px solid #2a2a2a;
    padding:8px 20px; display:flex; align-items:flex-end; gap:20px; flex-wrap:wrap;
}
.ctrl-group { display:flex; gap:10px; align-items:flex-end; }
.ctrl-group + .ctrl-group { border-left:1px solid #2a2a2a; padding-left:20px; }
.ctrl-sep { width:1px; background:#2a2a2a; align-self:stretch; }
.ctrl-divider { border-left:1px solid #2a2a2a; height:28px; align-self:center; }
.ctrl-item { display:flex; flex-direction:column; gap:3px; }
.sym-search-wrap { position:relative; }
.sym-search-wrap input { width:110px; background:#242424; color:#eee; border:1px solid #383838; padding:5px 9px; border-radius:4px; font-size:0.84rem; }
.sym-search-wrap input:focus { outline:none; border-color:#1976d2; }
.sym-dropdown { position:absolute; top:100%; left:0; right:0; z-index:500; background:#1a1a1a; border:1px solid #444; border-radius:4px; max-height:260px; overflow-y:auto; box-shadow:0 4px 16px #000b; margin-top:2px; }
.sym-drop-item { padding:6px 10px; cursor:pointer; font-size:0.84rem; color:#ccc; border-bottom:1px solid #222; display:flex; justify-content:space-between; align-items:center; }
.sym-drop-item:hover, .sym-drop-item.active { background:#263238; color:#fff; }
.sym-drop-cat { font-size:0.7rem; padding:1px 6px; border-radius:3px; }
.sym-cat-forex  { background:#0d2d1a; color:#66bb6a; }
.sym-cat-crypto { background:#2d1a0d; color:#ff9800; }
.sym-cat-metal  { background:#2d2a0d; color:#ffd54f; }

/* ── Status strip ── */
.status-strip {
    background:#141414; border-bottom:1px solid #222;
    padding:3px 20px; font-size:0.72rem; color:#555;
    display:flex; align-items:center; gap:16px;
}
#chart-refresh-status { color:#555; }

select, input[type=number] {
    background:#242424; color:#eee; border:1px solid #383838;
    padding:5px 9px; border-radius:4px; font-size:0.84rem;
    transition: border-color 0.15s;
}
select:focus, input[type=number]:focus { outline:none; border-color:#1976d2; }
.btn {
    background:#1976d2; color:#fff; border:none;
    padding:6px 14px; border-radius:4px; cursor:pointer;
    font-size:0.83rem; text-decoration:none; display:inline-flex;
    align-items:center; gap:5px; transition:background 0.15s;
    white-space:nowrap;
}
.btn:hover { background:#1565c0; }
.btn-multi { background:#6a1b9a; }
.btn-multi:hover { background:#4a148c; }
.btn-active { outline:2px solid #ce93d8; }
.btn-decide { background:#37474f; }
.btn-decide:hover { background:#455a64; }
.btn-autotrader { background:#1565c0; }
.btn-autotrader:hover { background:#0d47a1; }
.btn-analyze { background:#00838f; }
.btn-analyze:hover { background:#006064; }
.btn-trade-buy  { background:#1b5e20; color:#a5d6a7; font-weight:bold; }
.btn-trade-sell { background:#b71c1c; color:#ef9a9a; font-weight:bold; }
label { font-size:0.75rem; color:#666; display:block; margin-bottom:2px; letter-spacing:0.2px; text-transform:uppercase; }

.tf-checks { display:flex; gap:4px; flex-wrap:wrap; align-items:center; }
.tf-check-item {
    display:flex; align-items:center; gap:4px;
    background:#242424; border:1px solid #383838; border-radius:4px;
    padding:4px 8px; cursor:pointer; font-size:0.8rem; color:#bbb; user-select:none;
    transition:background 0.15s, border-color 0.15s;
}
.tf-check-item input { display:none; }
.tf-check-item.checked { background:#4a148c; border-color:#9c27b0; color:#fff; }
.tf-bars-row { display:flex; gap:5px; flex-wrap:wrap; }
.tf-bars-item { display:flex; flex-direction:column; align-items:center; gap:1px; }
.tf-bars-item span { font-size:0.7rem; color:#666; text-transform:uppercase; }
.tf-bars-item input { width:58px; text-align:center; padding:4px; }
.chart-container { padding:10px 16px; }

/* Panoul de analiza automata */
#analyze-panel {
    background:#1a1a1a; border:1px solid #333; border-radius:6px;
    margin:10px 16px; padding:14px 18px; display:none;
}
#analyze-panel h3 { font-size:0.95rem; color:#ccc; margin-bottom:10px; }
.tf-vote-table { width:100%; border-collapse:collapse; font-size:0.83rem; margin-bottom:12px; }
.tf-vote-table th { color:#888; font-weight:400; padding:4px 8px; border-bottom:1px solid #333; text-align:left; }
.tf-vote-table td { padding:5px 8px; border-bottom:1px solid #222; }
.sig-buy  { color:#26a69a; font-weight:bold; }
.sig-sell { color:#ef5350; font-weight:bold; }
.sig-hold { color:#888; }
.verdict-box {
    background:#222; border-radius:6px; padding:12px 16px;
    display:flex; align-items:center; gap:20px; flex-wrap:wrap;
}
.verdict-big { font-size:1.4rem; font-weight:bold; }
.verdict-buy  { color:#26a69a; }
.verdict-sell { color:#ef5350; }
.verdict-hold { color:#888; }
.verdict-detail { font-size:0.82rem; color:#aaa; line-height:1.6; }
.verdict-detail b { color:#ccc; }
#trade-result { margin-top:10px; font-size:0.85rem; padding:8px 12px; border-radius:4px; display:none; }
.trade-ok   { background:#1b5e20; color:#a5d6a7; }
.trade-err  { background:#b71c1c; color:#ef9a9a; }
/* Manual trade */
#manual-panel {
    margin-top:12px; padding:12px 14px; background:#1e1e2e;
    border:1px solid #444; border-radius:6px; display:none;
}
#manual-panel h4 { font-size:0.85rem; color:#888; margin-bottom:10px; font-weight:400; }
.manual-row { display:flex; gap:10px; align-items:flex-end; flex-wrap:wrap; }
.manual-field { display:flex; flex-direction:column; gap:3px; }
.manual-field label { font-size:0.78rem; color:#888; }
.manual-field input { background:#2a2a2a; color:#eee; border:1px solid #555; padding:5px 8px; border-radius:4px; width:110px; font-size:0.85rem; }
.manual-field input:focus { outline:1px solid #9c27b0; }

/* ── Strategy Simulator ── */
#sim-panel { margin:8px 16px 0; }
#sim-header { display:flex; align-items:center; gap:10px; background:#1a1a1a; border:1px solid #2e2e2e; border-radius:6px 6px 0 0; padding:10px 16px; cursor:pointer; user-select:none; }
#sim-header:hover { background:#1e1e1e; }
#sim-body { background:#161616; border:1px solid #2e2e2e; border-top:none; border-radius:0 0 6px 6px; padding:12px 16px; }
.sim-conv-bar { display:inline-flex; gap:2px; vertical-align:middle; }
.sim-conv-dot { width:7px; height:7px; border-radius:50%; display:inline-block; }
#sim-table { width:100%; border-collapse:collapse; font-size:0.82rem; }
#sim-table thead tr { border-bottom:1px solid #2a2a2a; }
#sim-table tbody tr { border-bottom:1px solid #1e1e1e; transition:background 0.1s; }
#sim-table tbody tr:hover { background:#1e1e1e; }
#sim-table tbody tr.tf-best { background:#0d1f0d; }
#sim-table tbody tr.tf-best-sell { background:#1f0d0d; }
.conf-bar-wrap { height:4px; background:#222; border-radius:2px; width:80px; }
.conf-bar-fill { height:4px; border-radius:2px; }
/* ── Drawing Tools ── */
#draw-toolbar { padding:5px 16px; background:#141414; border-bottom:1px solid #222; display:none; gap:6px; align-items:center; flex-wrap:wrap; }
.draw-tool-btn { background:#242424; color:#aaa; border:1px solid #333; padding:4px 10px; border-radius:4px; cursor:pointer; font-size:0.82rem; transition:background 0.15s; }
.draw-tool-btn:hover { background:#333; color:#eee; }
.draw-tool-btn.active { background:#1976d2; color:#fff; border-color:#1976d2; }

/* ── Unified Toolbar ── */
.toolbar {
    background:#161616; border-bottom:1px solid #222;
    padding:0 14px; height:46px;
    display:flex; align-items:center; gap:8px;
    overflow-x:auto; overflow-y:hidden; flex-shrink:0;
    scrollbar-width:none;
}
.toolbar::-webkit-scrollbar { display:none; }
.tb-sep { width:1px; background:#2a2a2a; align-self:stretch; margin:8px 2px; flex-shrink:0; }
.tb-sep-big { margin:8px 6px; }
.tb-btn {
    background:#242424; color:#aaa; border:1px solid #2e2e2e;
    padding:4px 11px; border-radius:5px; cursor:pointer; font-size:0.82rem;
    white-space:nowrap; flex-shrink:0; text-decoration:none; display:inline-flex; align-items:center;
    transition:background 0.12s, color 0.12s;
}
.tb-btn:hover { background:#2e2e2e; color:#eee; }
.tb-btn-purple { background:#2d1a45; color:#ce93d8; border-color:#4a148c; }
.tb-btn-purple:hover { background:#3e1f60; }
.tb-btn-yellow { background:#2a2000; color:#ffd54f; border-color:#5d4037; }
.tb-btn-yellow:hover { background:#3a2e00; }
.tb-btn-run { background:#00695c; color:#e0f2f1; border-color:#00897b; font-weight:600; padding:4px 14px; }
.tb-btn-run:hover { background:#00796b; }
.tb-select { background:#242424; color:#eee; border:1px solid #2e2e2e; padding:4px 8px; border-radius:5px; font-size:0.82rem; }
.tb-field { display:flex; align-items:center; gap:4px; flex-shrink:0; }
.tb-label { font-size:0.72rem; color:#555; text-transform:uppercase; letter-spacing:0.3px; white-space:nowrap; }

/* ── TF Pills ── */
.tf-pills { display:flex; gap:3px; align-items:center; flex-shrink:0; }
.tf-pill {
    background:#1a1a1a; color:#666; border:1px solid #2a2a2a;
    padding:3px 9px; border-radius:4px; cursor:pointer; font-size:0.78rem;
    font-weight:600; letter-spacing:0.3px; transition:all 0.12s; white-space:nowrap;
}
.tf-pill:hover { background:#2a2a2a; color:#aaa; border-color:#444; }
.tf-pill.active { background:#1976d2; color:#fff; border-color:#1976d2; }

/* ── Signal Strip ── */
.signal-strip {
    background:#141414; border-bottom:1px solid #1e1e1e;
    padding:0 16px; min-height:38px;
    display:flex; align-items:center; gap:10px; flex-wrap:wrap;
}
.sig-badge {
    font-size:0.88rem; font-weight:800; letter-spacing:0.5px;
    padding:4px 14px; border-radius:5px; border:1px solid #333;
    background:#1c1c1c; color:#555; white-space:nowrap; flex-shrink:0;
    min-width:64px; text-align:center;
}
.sig-badge.buy  { background:#0a2010; color:#26a69a; border-color:#1b5e20; }
.sig-badge.sell { background:#200a0a; color:#ef5350; border-color:#5e1b1b; }
.sig-badge.hold { background:#1a1700; color:#ff9800; border-color:#4a3000; }
.strip-sep { width:1px; background:#222; align-self:stretch; margin:6px 2px; flex-shrink:0; }
.strip-field { display:flex; flex-direction:column; gap:1px; flex-shrink:0; }
.sf-label { font-size:0.67rem; color:#444; text-transform:uppercase; letter-spacing:0.3px; }
.sf-val { font-size:0.82rem; color:#ccc; font-weight:600; }
.sf-mono { font-family:monospace; }

/* ── Exec Buttons ── */
.exec-btn {
    padding:4px 14px; border-radius:5px; border:1px solid;
    cursor:pointer; font-size:0.82rem; font-weight:700;
    transition:opacity 0.12s; white-space:nowrap;
}
.exec-buy  { background:#0a2010; color:#26a69a; border-color:#1b5e20; }
.exec-buy:hover  { background:#0d2f18; }
.exec-sell { background:#200a0a; color:#ef5350; border-color:#5e1b1b; }
.exec-sell:hover { background:#2f0d0d; }

/* ── TF Table ── */
.th-cell { padding:5px 10px; font-size:0.75rem; color:#666; font-weight:500; text-align:left; border-bottom:1px solid #2a2a2a; white-space:nowrap; }
</style>
<script>
async function decideNow(symbol) {
    window.open(`/autotrader?symbol=${symbol}&decide=1`, '_blank');
}

function toggleTf(el) {
    el.classList.toggle('checked');
    el.querySelector('input').checked = el.classList.contains('checked');
}

// Analiza rapida (buton "Analizeaza" din toolbar)
async function quickAnalyze() {
    // Deschide panoul daca e colapsat
    const body = document.getElementById('sim-body');
    if (body && body.style.display === 'none') toggleSimPanel();
    document.getElementById('analyze-btn').textContent = '⏳...';
    document.getElementById('analyze-btn').disabled = true;
    // Sincronizeaza TF-ul din grafic cu panoul de analiza inainte de rulare
    syncSimTfFromChart();
    await runSimulation();
    document.getElementById('analyze-btn').textContent = 'Analizeaza';
    document.getElementById('analyze-btn').disabled = false;
}

// Sincronizeaza TF-ul selectat in grafic cu panoul de analiza
function syncSimTfFromChart() {
    const mainTf = document.getElementById('main-tf-select')?.value;
    if (mainTf) selectSimTf(mainTf);
}

// Selecteaza un singur TF (radio behavior) + refresh grafic + re-ruleaza analiza
let _simAutoRunTimer = null;
function selectSimTf(tf) {
    // Radio: deselecteaza tot, selecteaza doar tf-ul ales
    document.querySelectorAll('#sim-tfs .tf-check-item').forEach(el => {
        const input = el.querySelector('input');
        const match = input.value === tf;
        el.classList.toggle('checked', match);
        input.checked = match;
    });
    // Sincronizeaza dropdown-ul principal
    const mainSel = document.getElementById('main-tf-select');
    if (mainSel && mainSel.value !== tf) mainSel.value = tf;
    // Refresh imediat al graficului
    _chartCountdown = 0;
    // Re-ruleaza analiza dupa 300ms (debounce)
    clearTimeout(_simAutoRunTimer);
    _simAutoRunTimer = setTimeout(() => runSimulation(), 350);
}

// Cand utilizatorul schimba TF-ul din dropdown-ul principal
function onMainTfChange(tf) {
    selectSimTf(tf);
}

async function executeTrade(symbol, signal, sl, tp) {
    if(!confirm(`Executa ${signal} pe ${symbol}?  SL=${sl}  TP=${tp}  Risc=$${{{ RISK_DOLLARS }}}`)) return;
    const url = `/trade?symbol=${symbol}&signal=${signal}&sl=${sl}&tp=${tp}`;
    const resp = await fetch(url, {method:'POST'});
    const data = await resp.json();
    const box = document.getElementById('trade-result');
    box.style.display = 'block';
    box.className = data.ok ? 'trade-ok' : 'trade-err';
    box.innerHTML = data.message;
    if(!data.ok) {
        document.getElementById('force-trade-bar').style.display = 'flex';
    }
}

// ── Symbol search dropdown ────────────────────────────────────────────────────
const _ALL_SYMBOLS = {{ symbols | tojson }};
let _symDropIdx = -1;

function _symCategory(s) {
    const cr = ['BTC','ETH','XRP','SOL','BNB','DOG','ADA','AVAX','LINK','LTC'];
    if (cr.some(c => s.includes(c))) return ['crypto','Crypto'];
    if (['XAU','XAG','OIL','WTI'].some(c => s.includes(c))) return ['metal','Metal'];
    return ['forex','Forex'];
}

function _buildSymDropdown(matches, current) {
    const dd = document.getElementById('sym-dropdown');
    if (!matches.length) { dd.style.display='none'; return; }
    dd.innerHTML = matches.slice(0,30).map((s,i) => {
        const [catKey, catLabel] = _symCategory(s);
        const sel = s === current ? ' style="background:#1a2a3a"' : '';
        return `<div class="sym-drop-item" data-sym="${s}"${sel} onmousedown="selectChartSym('${s}')">
            <span>${s}</span>
            <span class="sym-drop-cat sym-cat-${catKey}">${catLabel}</span>
        </div>`;
    }).join('') + (matches.length > 30 ? `<div style="padding:5px 10px;color:#555;font-size:0.72rem">... inca ${matches.length-30} — continua sa scrii</div>` : '');
    dd.style.display = 'block';
    _symDropIdx = -1;
}

function onChartSymInput(inp) {
    const val = inp.value.trim().toUpperCase();
    const cur = document.getElementById('sym-hidden').value;
    const matches = val ? _ALL_SYMBOLS.filter(s => s.includes(val)) : _ALL_SYMBOLS;
    _buildSymDropdown(matches, cur);
}

function onChartSymFocus(inp) {
    onChartSymInput(inp);
}

function onChartSymKey(e) {
    const dd = document.getElementById('sym-dropdown');
    const items = dd.querySelectorAll('.sym-drop-item');
    if (e.key === 'ArrowDown') {
        _symDropIdx = Math.min(_symDropIdx+1, items.length-1);
        items.forEach((el,i) => el.classList.toggle('active', i===_symDropIdx));
        e.preventDefault();
    } else if (e.key === 'ArrowUp') {
        _symDropIdx = Math.max(_symDropIdx-1, 0);
        items.forEach((el,i) => el.classList.toggle('active', i===_symDropIdx));
        e.preventDefault();
    } else if (e.key === 'Enter') {
        e.preventDefault();
        if (_symDropIdx >= 0 && items[_symDropIdx]) {
            selectChartSym(items[_symDropIdx].dataset.sym);
        } else {
            const val = e.target.value.trim().toUpperCase();
            if (_ALL_SYMBOLS.includes(val)) selectChartSym(val);
        }
    } else if (e.key === 'Escape') {
        dd.style.display = 'none';
    }
}

function selectChartSym(sym) {
    document.getElementById('sym-hidden').value    = sym;
    document.getElementById('sym-search-input').value = sym;
    document.getElementById('sym-dropdown').style.display = 'none';
    _clearSimLines();
    refreshChart();
}

document.addEventListener('click', e => {
    if (!e.target.closest('#sym-search-wrap'))
        document.getElementById('sym-dropdown').style.display = 'none';
});

// ── Zoom presets ──────────────────────────────────────────────────────────────
function zoomToLast(n) {
    const gd = getPlotlyDiv();
    if (!gd || !gd.data || !gd.data[0]) return;
    const x = gd.data[0].x;
    if (!x || x.length < 2) return;
    const x0 = x[Math.max(0, x.length - n)];
    const x1 = x[x.length - 1];
    Plotly.relayout(gd, {
        'xaxis.range': [x0, x1],
        'yaxis.autorange': true,
        'yaxis2.range': [0, 100],   // RSI fix — nu autorange
    }).then(() => { if (_simLineValues) _applySimLines(); });
    document.querySelectorAll('#zoom-pills .tf-pill').forEach(b => b.classList.remove('active'));
    event?.target?.classList.add('active');
}

function fitChart() {
    const gd = getPlotlyDiv();
    if (!gd) return;
    Plotly.relayout(gd, {
        'xaxis.autorange': true,
        'yaxis.autorange': true,
        'yaxis2.range': [0, 100],   // RSI fix
    }).then(() => { if (_simLineValues) _applySimLines(); });
    document.querySelectorAll('#zoom-pills .tf-pill').forEach(b => b.classList.remove('active'));
    const allBtn = document.querySelector('#zoom-pills .tf-pill:last-child');
    if (allBtn) allBtn.classList.add('active');
}

// ── TF pills + TF table ────────────────────────────────────────────────────────
function selectMainTf(tf) {
    const sel = document.getElementById('main-tf-select');
    if (sel) sel.value = tf;
    document.querySelectorAll('#main-tf-pills .tf-pill').forEach(btn => {
        btn.classList.toggle('active', btn.dataset.tf === tf);
    });
    _clearSimLines();
    refreshChart();
}

function toggleTfTable() {
    const wrap = document.getElementById('tf-table-wrap');
    const btn  = document.getElementById('tf-table-toggle');
    if (!wrap) return;
    const open = wrap.style.display !== 'none';
    wrap.style.display = open ? 'none' : 'block';
    if (btn) btn.textContent = open ? '▾ TF' : '▴ TF';
}

// ── Live updates ──────────────────────────────────────────────────────────────
const CHART_REFRESH_S = 30;   // refresh grafic la 30s — pastreaza zoom utilizator
let _chartCountdown   = CHART_REFRESH_S;
let _prevBid = null;

function currentSymbol() {
    return document.getElementById('sym-hidden')?.value ||
           document.querySelector('[name=symbol]')?.value || 'EURUSD';
}

function buildChartUrl() {
    const params = new URLSearchParams(window.location.search);
    params.set('symbol', currentSymbol());
    return '/chart_html?' + params.toString();
}

async function updateTicker() {
    try {
        const r = await fetch(`/tick?symbol=${currentSymbol()}`);
        const d = await r.json();
        if (!d.bid) return;
        document.getElementById('tick-symbol').textContent = d.symbol;
        const bidEl = document.getElementById('tick-bid');
        const askEl = document.getElementById('tick-ask');
        bidEl.textContent = 'BID ' + d.bid;
        askEl.textContent = 'ASK ' + d.ask;
        document.getElementById('tick-spread').textContent = d.spread + ' pips spread';
        if (_prevBid !== null && _prevBid !== d.bid) {
            const col = d.bid > _prevBid ? '#26a69a' : '#ef5350';
            bidEl.style.color = col; askEl.style.color = col;
            setTimeout(() => { bidEl.style.color='#ef5350'; askEl.style.color='#26a69a'; }, 400);
        }
        _prevBid = d.bid;
    } catch(e) {}
}

function buildChartJsonUrl() {
    const symbol = currentSymbol();
    const tf   = document.getElementById('main-tf-select')?.value || 'M5';
    const bars = document.querySelector('[name=bars]')?.value || 500;
    return `/chart_json?symbol=${encodeURIComponent(symbol)}&tf=${tf}&bars=${bars}`;
}

let _chartRefreshing = false;

async function refreshChart() {
    if (_chartRefreshing) return;
    _chartRefreshing = true;
    const el = document.getElementById('chart-refresh-status');
    if (el) { el.textContent = '↻ ...'; el.style.color = '#ff9800'; }
    try {
        const resp = await fetch(buildChartJsonUrl());
        const data = await resp.json();
        if (data.error) throw new Error(data.error);

        let gd = getPlotlyDiv();
        if (!gd) {
            // Niciun grafic Plotly inca — creeaza div-ul si foloseste newPlot
            const container = document.querySelector('.chart-container');
            if (!container) throw new Error('Lipseste .chart-container');
            container.innerHTML = '<div id="main-plotly-div"></div>';
            gd = document.getElementById('main-plotly-div');
            await Plotly.newPlot(gd, data.data, data.layout, {responsive: true, scrollZoom: true});
        } else {
            // Actualizeaza graficul in-place fara sa stergem div-ul
            await Plotly.react(gd, data.data, data.layout, {responsive: true, scrollZoom: true});
        }
        if (el) { el.textContent = '↻ live'; el.style.color = '#26a69a'; }
        // Reaplica overlayurile dupa update
        if (_indicatorVisible) {
            const sym   = currentSymbol();
            const strat = document.getElementById('sim-strategy')?.value;
            const tf    = document.getElementById('main-tf-select')?.value || 'H1';
            const bars  = parseInt(document.getElementById('sim-bars')?.value || 2000);
            await applyIndicatorOverlay(sym, strat, tf, bars);
        }
        if (_simLineValues) {
            _applySimLines();
        } else if (_simZonesVisible && _simLastResult) {
            addZonesToChart(_simLastResult);
        }
    } catch(e) {
        if (el) { el.textContent = '↻ err'; el.style.color = '#ef5350'; }
        log.debug && log.debug('refreshChart err:', e);
    } finally {
        _chartRefreshing = false;
    }
}

function liveLoop() {
    updateTicker();
    _chartCountdown--;
    const chartEl = document.getElementById('chart-refresh-status');
    if (chartEl && _chartCountdown > 0) chartEl.textContent = `↻ ${_chartCountdown}s`;
    if (_chartCountdown <= 0) {
        _chartCountdown = CHART_REFRESH_S;
        refreshChart();
    }
}

// ── Right-click drag → pan (ca TradingView) ─────────────────────────────────
function _installRightDragPan() {
    const container = document.querySelector('.chart-container');
    if (!container) return;
    container.addEventListener('contextmenu', e => e.preventDefault(), true);
    container.addEventListener('mousedown', function(e) {
        if (e.button !== 2) return;
        e.preventDefault();
        e.stopPropagation();
        // Trimite un left-click sintetic catre target-ul plotly → triggereaza pan
        const fake = new MouseEvent('mousedown', {
            bubbles: true, cancelable: true,
            view: window,
            button: 0, buttons: 1,
            clientX: e.clientX, clientY: e.clientY,
            screenX: e.screenX, screenY: e.screenY,
            ctrlKey: e.ctrlKey, shiftKey: e.shiftKey, altKey: e.altKey,
        });
        e.target.dispatchEvent(fake);
    }, true);
}

window.addEventListener('load', () => {
    // Porneste live loop mereu — graficul se incarca automat via refreshChart()
    setInterval(liveLoop, 1000);
    _chartCountdown = 0;   // triggereaza refresh imediat la primul tick
    liveLoop();

    loadStrategiesInfo().then(() => {
        // Dupa ce strategiile si graficul sunt incarcate, ruleaza analiza automata
        setTimeout(() => runSimulation(), 1200);
    });

    initTgNavBtn();

    // Instaleaza right-click pan dupa ce graficul e randat (slight delay)
    setTimeout(_installRightDragPan, 2000);
});

async function forceManual(signal) {
    const symbol = document.getElementById('manual-symbol').value ||
                   document.querySelector('[name=symbol]').value;
    const sl = parseFloat(document.getElementById('manual-sl').value);
    const tp = parseFloat(document.getElementById('manual-tp').value);
    if(!sl || !tp) { alert('Completeaza SL si TP'); return; }
    const url = `/trade?symbol=${symbol}&signal=${signal}&sl=${sl}&tp=${tp}`;
    const resp = await fetch(url, {method:'POST'});
    const data = await resp.json();
    const box = document.getElementById('trade-result');
    box.style.display = 'block';
    box.className = data.ok ? 'trade-ok' : 'trade-err';
    box.innerHTML = data.message;
}

async function executeManual(signal) {
    const symbol = document.getElementById('manual-symbol').value;
    const sl     = parseFloat(document.getElementById('manual-sl').value);
    const tp     = parseFloat(document.getElementById('manual-tp').value);
    if(!symbol || isNaN(sl) || isNaN(tp) || sl===0 || tp===0) {
        alert('Completeaza SL si TP inainte de executie.');
        return;
    }
    if(!confirm(`TRADE MANUAL: ${signal} ${symbol}\nSL=${sl}  TP=${tp}  Risc=$${{{ RISK_DOLLARS }}}\n\nConfirmi?`)) return;
    const url = `/trade?symbol=${symbol}&signal=${signal}&sl=${sl}&tp=${tp}`;
    const resp = await fetch(url, {method:'POST'});
    const data = await resp.json();
    const box = document.getElementById('trade-result');
    box.style.display = 'block';
    box.className = data.ok ? 'trade-ok' : 'trade-err';
    box.textContent = data.message;
}

// ── Strategy Simulator ───────────────────────────────────────────────────────
let _simStrategies    = [];
let _simZonesVisible  = false;
let _indicatorVisible = false;
let _indicatorTraceCount = 0;
let _simLastResult    = null;
let _simLastTf        = null;

async function loadStrategiesInfo() {
    try {
        const r = await fetch('/strategies_info');
        _simStrategies = await r.json();
        const sel = document.getElementById('sim-strategy');
        if (!sel) return;
        sel.innerHTML = _simStrategies.map(s =>
            `<option value="${s.key}">${s.icon} ${s.name}</option>`
        ).join('');
        if (_simStrategies.length > 0) onSimStrategyChange(_simStrategies[0].key);
        // Sincronizeaza butoanele TF cu selectia din grafic (fara auto-run)
        const mainTf = document.getElementById('main-tf-select')?.value;
        if (mainTf) {
            document.querySelectorAll('#sim-tfs .tf-check-item').forEach(el => {
                const input = el.querySelector('input');
                const match = input.value === mainTf;
                el.classList.toggle('checked', match);
                input.checked = match;
            });
        }
    } catch(e) { /* ignore */ }
}

function onSimStrategyChange(key) {
    const strat = _simStrategies.find(s => s.key === key);
    if (!strat) return;
    const container = document.getElementById('sim-tfs');
    if (!container) return;
    // TF curent din grafic (sau primul default al strategiei)
    const currentTf = document.getElementById('main-tf-select')?.value || strat.default_tfs[0] || 'H1';
    const allTfs = ['M1','M5','M15','M30','H1','H4','D1'];
    container.innerHTML = allTfs.map(tf => {
        const active = tf === currentTf;
        return `<label class="tf-check-item ${active?'checked':''}" onclick="selectSimTf('${tf}')">
            <input type="checkbox" value="${tf}" ${active?'checked':''}> ${tf}
        </label>`;
    }).join('');
    const barsEl = document.getElementById('sim-bars');
    if (barsEl) barsEl.value = strat.default_bars || 2000;
    // Reset overlays
    removeIndicatorTraces();
    clearZonesFromChart();
    _simZonesVisible  = false;
    _indicatorVisible = false;
    const ib = document.getElementById('sim-overlay-ind-btn');
    const zb = document.getElementById('sim-overlay-btn');
    if (ib) { ib.style.background='#263238'; ib.textContent='📊 Indicatori'; ib.style.display='none'; }
    if (zb) { zb.style.background='#37474f'; zb.textContent='📍 Zones'; }
}

// Run simulation then draw lines for a specific direction (from floating buttons)
async function runSimulationAs(direction) {
    await runSimulation();
    // If result differs from requested direction, draw sim lines with forced direction anyway
    if (_simLastResult && _simLastResult.best_tf) {
        const best = _simLastResult.best_tf;
        if (best.sl && best.price) {
            _simLineValues = { entry: best.price, sl: best.sl, tp: best.tp || null, signal: direction };
            _applySimLines();
        }
    }
}

async function runSimulation() {
    const symbol = document.querySelector('[name=symbol]')?.value || 'EURUSD';
    const strat  = document.getElementById('sim-strategy')?.value;
    // Citeste TF-ul selectat (radio — unul singur)
    const selectedInput = document.querySelector('#sim-tfs .tf-check-item.checked input');
    const tfs = selectedInput ? [selectedInput.value] : ['M5'];
    const bars = document.querySelector('[name=bars]')?.value ||
                 document.getElementById('sim-bars')?.value || 2000;
    if (!tfs.length) { alert('Selecteaza un timeframe'); return; }

    const btn    = document.getElementById('sim-btn');
    const loadEl = document.getElementById('sim-loading');
    const errEl  = document.getElementById('sim-error');
    const resEl  = document.getElementById('sim-results');

    btn.disabled = true; btn.textContent = '⏳...';
    loadEl.style.display = 'block';
    errEl.style.display  = 'none';
    resEl.style.display  = 'none';

    // Reset overlays
    removeIndicatorTraces();
    clearZonesFromChart();
    _simZonesVisible  = false;
    _indicatorVisible = false;

    try {
        const url  = `/simulate?symbol=${symbol}&strategy=${strat}&tfs=${tfs.join(',')}&bars=${bars}`;
        const resp = await fetch(url);
        const data = await resp.json();
        if (data.error) throw new Error(data.error);
        _simLastResult = data;

        const chartTf = tfs[0];   // single TF
        _simLastTf = chartTf;

        showSimResult(data, symbol, strat, parseInt(bars));

        // Auto-aplica indicatorii pe grafic
        const gd = getPlotlyDiv();
        if (gd) {
            await applyIndicatorOverlay(symbol, strat, chartTf, parseInt(bars));
        }
    } catch(e) {
        errEl.textContent = 'Eroare: ' + e.message;
        errEl.style.display = 'block';
    } finally {
        btn.disabled = false; btn.textContent = '▶ Ruleaza';
        loadEl.style.display = 'none';
    }
}

// ── Sim Lines: entry / SL / TP pe grafic ─────────────────────────────────────
let _simLineValues = null;  // {entry, sl, tp, signal}

function _applySimLines() {
    const gd = getPlotlyDiv();
    if (!gd || !_simLineValues) return;
    const { entry, sl, tp, signal } = _simLineValues;
    const entryColor = signal === 'BUY' ? '#29b6f6' : '#ff8a65';
    const shapes = [];
    shapes.push({
        type:'line', x0:0, x1:1, xref:'paper', y0:entry, y1:entry, yref:'y',
        line:{color:entryColor, width:2, dash:'dot'},
        label:{text:' ENTRY', textposition:'end', font:{color:entryColor, size:10}}
    });
    if (sl) shapes.push({
        type:'line', x0:0, x1:1, xref:'paper', y0:sl, y1:sl, yref:'y',
        line:{color:'#ef5350', width:1.5, dash:'dash'},
        label:{text:' SL', textposition:'end', font:{color:'#ef5350', size:10}}
    });
    if (tp) shapes.push({
        type:'line', x0:0, x1:1, xref:'paper', y0:tp, y1:tp, yref:'y',
        line:{color:'#26a69a', width:1.5, dash:'dash'},
        label:{text:' TP', textposition:'end', font:{color:'#26a69a', size:10}}
    });
    try { Plotly.relayout(gd, {shapes}); } catch(e) {}
}

function _clearSimLines() {
    _simLineValues = null;
    const gd = getPlotlyDiv();
    if (!gd) return;
    try { Plotly.relayout(gd, {shapes:[]}); } catch(e) {}
}

// ── Lot preview refresh ───────────────────────────────────────────────────────
let _lpBest = null;
let _lpSymbol = null;

function _refreshLotPreview(symbol, best) {
    _lpBest   = best;
    _lpSymbol = symbol;
    const lpIds = ['lp-lots','lp-sl-pips','lp-tp-pips','lp-riskrew','lp-duration'];
    if (!best || !best.price || !best.sl) {
        lpIds.forEach(id => { const el = document.getElementById(id); if(el) el.textContent = '—'; });
        return;
    }
    lpIds.forEach(id => { const el = document.getElementById(id); if(el) el.textContent = '…'; });
    const lot      = document.getElementById('sim-lot')?.value || '';
    const maxRisk  = document.getElementById('sim-max-risk')?.value || '50';
    const params   = new URLSearchParams({
        symbol: symbol || 'EURUSD',
        entry:  best.price,
        sl:     best.sl,
        tp:     best.tp || 0,
        tf:     best.tf || 'H1',
        max_risk: maxRisk,
    });
    if (lot) params.set('lot', lot);
    fetch('/api/lot_preview?' + params)
        .then(r => r.json())
        .then(d => {
            if (d.error) { document.getElementById('lp-lots').textContent = '—'; return; }
            const lotEl = document.getElementById('lp-lots');
            if (lotEl) {
                lotEl.textContent = d.lots;
                lotEl.style.color = d.risk_exceeded ? '#ff9800' : '#42a5f5';
            }
            // Sync lot input with auto-calculated value (only if user didn't override)
            const lotInput = document.getElementById('sim-lot');
            if (lotInput && !lot) lotInput.value = d.lots;
            const slEl = document.getElementById('lp-sl-pips');
            if (slEl) slEl.textContent = d.sl_pips + ' pip';
            const tpEl = document.getElementById('lp-tp-pips');
            if (tpEl) tpEl.textContent = d.tp_pips ? d.tp_pips + ' pip' : '—';
            const rrEl = document.getElementById('lp-riskrew');
            if (rrEl) rrEl.innerHTML =
                `<span style="color:${d.risk_exceeded?'#ff9800':'#ef9a9a'}">-$${d.risk_usd}</span>` +
                (d.reward_usd ? ` / <span style="color:#a5d6a7">+$${d.reward_usd}</span>` : '');
            const durEl = document.getElementById('lp-duration');
            if (durEl) durEl.textContent = d.duration;
        })
        .catch(() => { const el = document.getElementById('lp-lots'); if(el) el.textContent = '—'; });
}

function onSimLotChange() {
    if (_lpBest && _lpSymbol) _refreshLotPreview(_lpSymbol, _lpBest);
}
function onSimMaxRiskChange() {
    // Reset lot manual override so it recalculates from new max risk
    const lotInput = document.getElementById('sim-lot');
    if (lotInput) lotInput.value = '';
    if (_lpBest && _lpSymbol) _refreshLotPreview(_lpSymbol, _lpBest);
}

function showSimResult(data, symbol, stratKey, bars) {
    const resEl = document.getElementById('sim-results');
    resEl.style.display = 'block';

    const sig  = data.signal || 'HOLD';
    const conf = data.confidence || 0;
    const best = data.best_tf;

    // Update header inline indicator
    const stratInfo = _simStrategies.find(s => s.key === stratKey);
    const lbl = document.getElementById('sim-strat-label');
    if (lbl && stratInfo) lbl.textContent = stratInfo.icon + ' ' + stratInfo.name;
    const inline = document.getElementById('sim-sig-inline');
    if (inline) {
        inline.style.display = 'inline';
        inline.textContent = sig === 'BUY' ? '▲ BUY' : sig === 'SELL' ? '▼ SELL' : '— HOLD';
        inline.style.color  = sig === 'BUY' ? '#26a69a' : sig === 'SELL' ? '#ef5350' : '#666';
    }

    // HOLD reason banner
    const holdDiv  = document.getElementById('sim-hold-reason');
    const holdText = document.getElementById('sim-hold-text');
    if (sig === 'HOLD') {
        holdDiv.style.display = 'block';
        // Colecteaza motivele de HOLD din toate TF-urile + justification
        const lines = [];
        (data.justification || []).forEach(j => lines.push('📌 ' + j));
        // Per-TF motivele pentru HOLD
        (data.tfs || []).filter(r => r.signal === 'HOLD' || r.signal === 'hold').forEach(r => {
            if (r.reasons?.length) {
                lines.push(`<span style="color:#555">${r.tf}:</span> ` +
                    r.reasons.slice(0,4).join(' · '));
            }
        });
        holdText.innerHTML = lines.length
            ? lines.map(l => `<div style="margin-bottom:3px">${l}</div>`).join('')
            : '<div>Nicio conditie activa pe TF-urile selectate</div>';
    } else {
        holdDiv.style.display = 'none';
    }

    // Summary card
    const sigEl  = document.getElementById('sim-sig');
    const confEl = document.getElementById('sim-conf');
    sigEl.textContent = sig==='BUY' ? '▲ BUY' : sig==='SELL' ? '▼ SELL' : '— HOLD';
    sigEl.style.color = sig==='BUY' ? '#26a69a' : sig==='SELL' ? '#ef5350' : '#666';
    confEl.innerHTML = `${conf.toFixed(1)}%
        <div class="conf-bar-wrap" style="margin-top:4px">
          <div class="conf-bar-fill" style="width:${Math.min(conf,100)}%;background:${conf>=70?'#26a69a':conf>=55?'#ff9800':'#ef5350'}"></div>
        </div>`;

    if (best) {
        document.getElementById('sim-entry').textContent = best.price ? best.price.toFixed(5) : '—';
        document.getElementById('sim-sltp').innerHTML =
            `<span style="color:#ef9a9a">${best.sl ? best.sl.toFixed(5) : '—'}</span>` +
            ` / <span style="color:#a5d6a7">${best.tp ? best.tp.toFixed(5) : '—'}</span>`;
        const rr = (best.sl && best.tp && best.price) ?
            Math.abs(best.tp - best.price) / Math.abs(best.price - best.sl) : null;
        document.getElementById('sim-rr').textContent = rr ? rr.toFixed(2)+':1' : '—';
        document.getElementById('sim-rr').style.color = rr>=2?'#26a69a':rr>=1?'#ff9800':'#ef5350';
    } else {
        document.getElementById('sim-entry').textContent = '—';
        document.getElementById('sim-sltp').textContent  = '—';
        document.getElementById('sim-rr').textContent    = '—';
    }

    document.getElementById('sim-buy-btn').style.display  = sig==='BUY'  ? 'inline-flex' : 'none';
    document.getElementById('sim-sell-btn').style.display = sig==='SELL' ? 'inline-flex' : 'none';

    _simLineValues = null;

    // Lot preview
    _refreshLotPreview(symbol, best);

    document.getElementById('sim-overlay-btn').style.display = 'inline-flex';
    document.getElementById('sim-overlay-ind-btn').style.display = 'inline-flex';

    // Trade section pre-fill + show execution panel
    if (symbol) document.getElementById('manual-symbol').value = symbol;
    if (best) {
        if (best.sl) document.getElementById('manual-sl').value = best.sl;
        if (best.tp) document.getElementById('manual-tp').value = best.tp;
    }
    document.getElementById('trade-exec-panel').style.display = 'block';
    document.getElementById('trade-result').style.display = 'none';
    document.getElementById('force-trade-bar').style.display = 'none';

    // TF rows
    const tfRows = data.tfs || [];
    let html = '';
    for (const r of tfRows) {
        const isBest = best && r.tf === best.tf && r.signal === sig;
        const rowCls = isBest ? (sig==='BUY'?'tf-best':'tf-best-sell') : '';
        const sigCol = r.signal==='BUY'?'#26a69a':r.signal==='SELL'?'#ef5350':'#3a3a3a';
        const conv   = r.conviction || 0;
        const convColor = r.signal==='BUY'?'#26a69a':r.signal==='SELL'?'#ef5350':'#444';
        const dots   = Array.from({length:10}, (_,i) =>
            `<span class="sim-conv-dot" style="background:${i<conv?convColor:'#2a2a2a'}"></span>`
        ).join('');
        const rr = (r.sl&&r.tp&&r.price) ?
            (Math.abs(r.tp-r.price)/Math.abs(r.price-r.sl)).toFixed(2) : null;
        const rrCol = rr>=2?'#26a69a':rr>=1?'#ff9800':'#3a3a3a';
        // Reasons: daca HOLD, arata toate motivele; altfel primele 3
        const isHoldRow = r.signal === 'HOLD';
        const maxR = isHoldRow ? 5 : 3;
        const reasons_short = (r.reasons||[]).slice(0, maxR)
            .map(s => s.replace(/[(][+-][0-9]+[)]/g,'').trim()).join(' · ');
        const reasons_full  = (r.reasons||[]).join(' | ');
        html += `<tr class="${rowCls}">
            <td style="padding:6px 10px;font-weight:${isBest?'bold':'400'};color:${isBest?'#fff':'#aaa'};white-space:nowrap">${r.tf}${isBest?' ★':''}</td>
            <td style="padding:6px 10px;text-align:center;color:${sigCol};font-weight:bold;font-size:0.82rem">${r.signal}</td>
            <td style="padding:6px 10px;text-align:center"><span class="sim-conv-bar">${dots}</span> <span style="color:#444;font-size:0.72rem">${conv}</span></td>
            <td style="padding:6px 10px;text-align:right;color:#aaa;font-size:0.8rem">${r.price?r.price.toFixed(5):'—'}</td>
            <td style="padding:6px 10px;text-align:right;color:#ef9a9a;font-size:0.8rem">${r.sl?r.sl.toFixed(5):'—'}</td>
            <td style="padding:6px 10px;text-align:right;color:#a5d6a7;font-size:0.8rem">${r.tp?r.tp.toFixed(5):'—'}</td>
            <td style="padding:6px 10px;text-align:right;color:${rrCol};font-size:0.8rem">${rr?rr+':1':'—'}</td>
            <td style="padding:6px 10px;color:${isHoldRow?'#777':'#555'};font-size:0.77rem;max-width:300px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap" title="${reasons_full}">${reasons_short||'—'}</td>
        </tr>`;
    }
    document.getElementById('sim-rows').innerHTML = html ||
        '<tr><td colspan="8" style="padding:14px;color:#444;text-align:center">Fara semnale pe TF-urile selectate</td></tr>';
}

function toggleSimPanel() {
    const body = document.getElementById('sim-body');
    const icon = document.getElementById('sim-collapse-icon');
    if (!body) return;
    const open = body.style.display !== 'none';
    body.style.display = open ? 'none' : 'block';
    icon.textContent = open ? '▶' : '▼';
}

// ── Indicator Overlay (strategy lines on chart) ──────────────────────────────
function getPlotlyDiv() {
    return document.querySelector('.chart-container .js-plotly-plot') ||
           document.querySelector('.chart-container [class*="plotly"]');
}

async function applyIndicatorOverlay(symbol, strat, tf, bars) {
    if (!symbol) symbol = document.querySelector('[name=symbol]')?.value || 'EURUSD';
    if (!strat)  strat  = document.getElementById('sim-strategy')?.value;
    if (!tf)     tf     = document.querySelector('[name=tf]')?.value || 'H1';
    if (!bars)   bars   = parseInt(document.getElementById('sim-bars')?.value || 2000);

    removeIndicatorTraces();
    const gd = getPlotlyDiv();
    if (!gd) return;
    try {
        const url  = `/strategy_overlay?symbol=${symbol}&strategy=${strat}&tf=${tf}&bars=${bars}`;
        const resp = await fetch(url);
        const data = await resp.json();
        if (!data.traces?.length) return;
        await Plotly.addTraces(gd, data.traces);
        _indicatorTraceCount = data.traces.length;
        _indicatorVisible    = true;
        // Forteaza RSI panel inapoi la [0,100] — addTraces poate extinde axa
        try { Plotly.relayout(gd, {'yaxis2.range': [0, 100]}); } catch(e) {}
        // Reaplica liniile de sim daca existau
        if (_simLineValues) _applySimLines();
        const btn = document.getElementById('sim-overlay-ind-btn');
        if (btn) { btn.style.background='#1976d2'; btn.textContent='📊 Indicatori: ON'; }
    } catch(e) {}
}

function removeIndicatorTraces() {
    const gd = getPlotlyDiv();
    if (!gd || !_indicatorTraceCount) return;
    try {
        const total = gd.data.length;
        const idxs  = Array.from({length: _indicatorTraceCount}, (_, i) => total - _indicatorTraceCount + i);
        Plotly.deleteTraces(gd, idxs);
    } catch(e) {}
    _indicatorTraceCount = 0;
    _indicatorVisible    = false;
}

async function toggleIndicatorOverlay() {
    const btn = document.getElementById('sim-overlay-ind-btn');
    if (_indicatorVisible) {
        removeIndicatorTraces();
        if (btn) { btn.style.background='#263238'; btn.textContent='📊 Indicatori'; }
    } else {
        const symbol = document.querySelector('[name=symbol]')?.value || 'EURUSD';
        const strat  = document.getElementById('sim-strategy')?.value;
        const tf     = document.querySelector('[name=tf]')?.value || (_simLastTf || 'H1');
        const bars   = parseInt(document.getElementById('sim-bars')?.value || 2000);
        await applyIndicatorOverlay(symbol, strat, tf, bars);
        if (btn) { btn.style.background='#1976d2'; btn.textContent='📊 Indicatori: ON'; }
    }
}

// ── Zone Overlay (SL/TP/Entry lines on chart) ─────────────────────────────────
function toggleZoneOverlay() {
    if (!_simLastResult) return;
    _simZonesVisible = !_simZonesVisible;
    const btn = document.getElementById('sim-overlay-btn');
    if (_simZonesVisible) {
        addZonesToChart(_simLastResult);
        if (btn) { btn.style.background='#1976d2'; btn.textContent='📍 Zones: ON'; }
    } else {
        clearZonesFromChart();
        if (btn) { btn.style.background='#37474f'; btn.textContent='📍 Zones'; }
    }
}

function addZonesToChart(data) {
    // addZonesToChart e alias pentru _applySimLines — daca _simLineValues e setat, il foloseste deja
    if (_simLineValues) { _applySimLines(); return; }
    const gd = getPlotlyDiv();
    if (!gd || !data.best_tf) return;
    const best = data.best_tf;
    const sig  = data.signal;
    const shapes = [];
    if (best.sl) shapes.push({
        type:'line', x0:0, x1:1, xref:'paper', y0:best.sl, y1:best.sl, yref:'y',
        line:{color:'#ef5350', width:1.5, dash:'dash'},
        label:{text:' SL', textposition:'end', font:{color:'#ef5350', size:10}}
    });
    if (best.tp) shapes.push({
        type:'line', x0:0, x1:1, xref:'paper', y0:best.tp, y1:best.tp, yref:'y',
        line:{color:'#26a69a', width:1.5, dash:'dash'},
        label:{text:' TP', textposition:'end', font:{color:'#26a69a', size:10}}
    });
    if (best.price) shapes.push({
        type:'line', x0:0, x1:1, xref:'paper', y0:best.price, y1:best.price, yref:'y',
        line:{color:sig==='BUY'?'#29b6f6':'#ff8a65', width:2, dash:'dot'},
        label:{text:' ENTRY', textposition:'end', font:{color:'#fff', size:10}}
    });
    try { Plotly.relayout(gd, {shapes}); } catch(e) {}
}

function clearZonesFromChart() {
    const gd = getPlotlyDiv();
    if (!gd) return;
    try { Plotly.relayout(gd, {shapes:[]}); } catch(e) {}
    _simZonesVisible = false;
}

// ── Drawing Tools ─────────────────────────────────────────────────────────────
function setDrawMode(mode, el) {
    const gd = getPlotlyDiv();
    document.querySelectorAll('.draw-tool-btn').forEach(b => b.classList.remove('active'));
    if (el) el.classList.add('active');
    if (!gd) return;
    try { Plotly.relayout(gd, {dragmode: mode}); } catch(e) {}
}

function clearAllDrawShapes() {
    clearZonesFromChart();
    const btn = document.getElementById('sim-overlay-btn');
    if (btn) { btn.style.background='#37474f'; btn.textContent='📍 Zones'; }
    const panBtn = document.getElementById('draw-pan-btn');
    document.querySelectorAll('.draw-tool-btn').forEach(b => b.classList.remove('active'));
    if (panBtn) panBtn.classList.add('active');
    setDrawMode('pan', null);
}

function showDrawToolbar() {
    const tb = document.getElementById('draw-toolbar');
    if (tb) { tb.style.display = tb.style.display==='flex' ? 'none' : 'flex'; }
}

async function execSimTrade(signal) {
    if (!_simLastResult?.best_tf) return;
    const symbol = document.querySelector('[name=symbol]')?.value || 'EURUSD';
    const best   = _simLastResult.best_tf;
    if(!confirm(`Executa ${signal} pe ${symbol}?\nSL=${best.sl}  TP=${best.tp}`)) return;
    const url  = `/trade?symbol=${symbol}&signal=${signal}&sl=${best.sl}&tp=${best.tp}`;
    const resp = await fetch(url, {method:'POST'});
    const data = await resp.json();
    // Asigura ca panoul de executie e vizibil
    const panel = document.getElementById('trade-exec-panel');
    if (panel) panel.style.display = 'block';
    const box = document.getElementById('trade-result');
    box.style.display = 'block';
    box.className  = data.ok ? 'trade-ok' : 'trade-err';
    box.textContent = data.message;
    if (!data.ok) document.getElementById('force-trade-bar').style.display = 'flex';
}

// ── Telegram Panel ────────────────────────────────────────────────────────────
async function toggleTgNotif() {
    const r = await fetch('/telegram/config');
    const d = await r.json();
    const nowEnabled = !!d.enabled;
    await fetch('/telegram/config', {
        method: 'POST',
        headers: {'Content-Type':'application/json'},
        body: JSON.stringify({...d, enabled: !nowEnabled}),
    });
    updateTgNavBtn(!nowEnabled);
}

function updateTgNavBtn(enabled) {
    const btn = document.getElementById('tg-navbar-btn');
    if (!btn) return;
    btn.textContent = enabled ? '📱 Telegram: ON' : '📵 Telegram: OFF';
    btn.style.background  = enabled ? '#1b3a1b' : '#1a1a1a';
    btn.style.color       = enabled ? '#26a69a' : '#555';
    btn.style.borderColor = enabled ? '#26a69a' : '#333';
}

async function initTgNavBtn() {
    try {
        const r = await fetch('/telegram/config');
        const d = await r.json();
        updateTgNavBtn(!!d.enabled);
    } catch(e) {}
}

function toggleTgPanel() {
    const body = document.getElementById('tg-body');
    const icon = document.getElementById('tg-collapse-icon');
    if (!body) return;
    const open = body.style.display !== 'none';
    body.style.display = open ? 'none' : 'block';
    icon.textContent = open ? '▶' : '▼';
    if (!open) { loadTgConfig(); loadTgPending(); }
}

function toggleTgEnabled() {
    const cb = document.getElementById('tg-enabled');
    cb.checked = !cb.checked;
    document.getElementById('tg-enable-lbl').classList.toggle('checked', cb.checked);
}

async function loadTgConfig() {
    try {
        const r = await fetch('/telegram/config');
        const d = await r.json();
        document.getElementById('tg-token').value  = d.bot_token || '';
        document.getElementById('tg-chatid').value = d.chat_id  || '';
        document.getElementById('tg-mode').value   = d.require_approval ? '1' : '0';
        const enabled = !!d.enabled;
        document.getElementById('tg-enabled').checked = enabled;
        document.getElementById('tg-enable-lbl').classList.toggle('checked', enabled);
        document.getElementById('tg-status-dot').style.background = enabled ? '#26a69a' : '#333';
    } catch(e) {}
}

async function saveTgConfig() {
    const token   = document.getElementById('tg-token').value.trim();
    const chatId  = document.getElementById('tg-chatid').value.trim();
    const enabled = document.getElementById('tg-enabled').checked;
    const reqApproval = document.getElementById('tg-mode').value === '1';
    const st = document.getElementById('tg-config-status');
    st.textContent = '⏳ Salvez...'; st.style.color = '#ff9800';
    try {
        const r = await fetch('/telegram/config', {
            method: 'POST',
            headers: {'Content-Type':'application/json'},
            body: JSON.stringify({bot_token: token, chat_id: chatId, enabled, require_approval: reqApproval}),
        });
        const d = await r.json();
        st.textContent = d.ok ? '✅ Salvat' : '❌ Eroare';
        st.style.color = d.ok ? '#26a69a' : '#ef5350';
        document.getElementById('tg-status-dot').style.background = enabled ? '#26a69a' : '#333';
    } catch(e) {
        st.textContent = '❌ ' + e.message; st.style.color = '#ef5350';
    }
}

async function testTgConnection() {
    const st = document.getElementById('tg-config-status');
    st.textContent = '⏳ Testez conexiunea...'; st.style.color = '#ff9800';
    try {
        const r = await fetch('/telegram/test');
        const d = await r.json();
        st.textContent = (d.ok ? '✅ ' : '❌ ') + d.msg;
        st.style.color = d.ok ? '#26a69a' : '#ef5350';
    } catch(e) {
        st.textContent = '❌ ' + e.message; st.style.color = '#ef5350';
    }
}

async function loadTgPending() {
    try {
        const r = await fetch('/telegram/pending');
        const items = await r.json();
        const pending = items.filter(i => i.status === 'pending');
        const badge = document.getElementById('tg-pending-badge');
        if (pending.length > 0) {
            badge.style.display = 'inline';
            badge.textContent = pending.length + ' pending';
        } else {
            badge.style.display = 'none';
        }
        const wrap = document.getElementById('tg-approvals-wrap');
        const list = document.getElementById('tg-approvals-list');
        if (items.length === 0) { wrap.style.display = 'none'; return; }
        wrap.style.display = 'block';
        list.innerHTML = items.map(item => {
            const isPending = item.status === 'pending';
            const sig = item.signal === 'BUY' ? '▲' : '▼';
            const sigCol = item.signal === 'BUY' ? '#26a69a' : '#ef5350';
            const statusColor = {pending:'#ff9800', approved:'#26a69a', denied:'#ef5350', expired:'#555'}[item.status] || '#888';
            const timeLeft = isPending ? Math.max(0, Math.round((item.expires - Date.now()/1000))) : 0;
            return `<div style="display:flex;align-items:center;gap:10px;background:#1a1a1a;border:1px solid #2a2a2a;border-radius:4px;padding:8px 12px;flex-wrap:wrap">
                <span style="color:${sigCol};font-weight:bold;font-size:0.9rem">${sig} ${item.signal}</span>
                <span style="color:#ccc;font-size:0.85rem">${item.symbol}</span>
                <span style="color:#666;font-size:0.78rem">${item.strategy} · ${item.tf} · ${item.confidence?.toFixed(0)}%</span>
                <span style="color:${statusColor};font-size:0.78rem;font-weight:600;text-transform:uppercase">${item.status}${isPending?' ('+timeLeft+'s)':''}</span>
                ${isPending ? `
                <button class="btn btn-trade-buy" style="font-size:0.75rem;padding:3px 10px" onclick="tgApprove('${item.token}')">✅ Executa</button>
                <button class="btn btn-trade-sell" style="font-size:0.75rem;padding:3px 10px" onclick="tgDeny('${item.token}')">❌ Respinge</button>
                ` : ''}
            </div>`;
        }).join('');
    } catch(e) {}
}

async function tgApprove(token) {
    await fetch(`/telegram/approve/${token}`, {method:'POST'});
    loadTgPending();
}
async function tgDeny(token) {
    await fetch(`/telegram/deny/${token}`, {method:'POST'});
    loadTgPending();
}

async function sendAnalysisToTelegram() {
    if (!_simLastResult) { alert('Ruleaza mai intai o analiza (buton ▶ Ruleaza)'); return; }
    const symbol = document.querySelector('[name=symbol]')?.value || 'EURUSD';
    const strat  = document.getElementById('sim-strategy')?.value || 'classic';
    const tf     = document.querySelector('.sim-tf-btn.active')?.dataset?.tf ||
                   document.getElementById('main-tf-select')?.value || 'M5';
    const bars   = parseInt(document.getElementById('sim-bars')?.value || 2000);
    const data   = _simLastResult;
    const best   = data.best_tf;
    const btn = event?.target;
    if (btn) { btn.textContent = '⏳...'; btn.disabled = true; }
    try {
        const payload = {
            symbol, strategy: strat, tf, bars,
            signal: data.signal, confidence: data.confidence || 0,
            sl: best?.sl, tp: best?.tp, price: best?.price,
            reasons: data.justification || [],
        };
        const r = await fetch('/telegram/send_analysis', {
            method: 'POST',
            headers: {'Content-Type':'application/json'},
            body: JSON.stringify(payload),
        });
        const res = await r.json();
        if (btn) {
            btn.textContent = res.ok
                ? (res.mode === 'approval' ? '✅ Trimis — asteapta aprobare' : '✅ Trimis')
                : '❌ Eroare';
            btn.disabled = false;
        }
        // Deschide panoul de pending daca e cerere de aprobare
        if (res.mode === 'approval') {
            const body = document.getElementById('tg-body');
            if (body && body.style.display === 'none') toggleTgPanel();
            setTimeout(loadTgPending, 1000);
        }
    } catch(e) {
        if (btn) { btn.textContent = '❌ ' + e.message; btn.disabled = false; }
    }
    // Reset buton dupa 3s
    setTimeout(() => { if (btn) { btn.textContent = '📤 Trimite analiza pe Telegram'; btn.disabled = false; } }, 3000);
}

// Refresh pending la fiecare 10s daca panoul e deschis
setInterval(() => {
    const body = document.getElementById('tg-body');
    if (body && body.style.display !== 'none') loadTgPending();
}, 10000);
</script>
</head><body>

<!-- ── Navbar ── -->
<div class="navbar">
    <div class="navbar-brand">
        <span class="brand-dot"></span>
        ChartVisualizer
    </div>
    <div style="display:flex;align-items:center;gap:20px">
        <div id="live-ticker" style="font-size:0.82rem;color:#aaa;display:flex;gap:14px;align-items:center">
            <span id="tick-symbol" style="color:#666;font-size:0.75rem"></span>
            <span id="tick-bid" style="color:#ef5350;font-weight:600"></span>
            <span id="tick-ask" style="color:#26a69a;font-weight:600"></span>
            <span id="tick-spread" style="color:#555;font-size:0.72rem"></span>
        </div>
        <div class="navbar-links">
            <a href="/autotrader"  class="btn btn-autotrader" style="font-size:0.8rem;padding:5px 12px">⚡ AutoTrader</a>
            <a href="/trades"      class="btn" style="background:#00695c;font-size:0.8rem;padding:5px 12px">📊 Trades</a>
            <a href="/autoorders/" class="btn" style="background:#1a1030;color:#ce93d8;border:1px solid #6a1b9a;font-size:0.8rem;padding:5px 12px">⬡ AutoOrders</a>
            <a href="/news"        class="btn" style="background:#6a1b9a;font-size:0.8rem;padding:5px 12px">📰 Stiri</a>
            <a href="/account"     class="btn" style="background:#333;color:#bbb;font-size:0.8rem;padding:5px 12px">Cont MT5</a>
            <a href="/settings"    class="btn" style="background:#1a1a2e;color:#7986cb;border:1px solid #3949ab;font-size:0.8rem;padding:5px 12px">⚙ Settings</a>
        </div>
    </div>
</div>

<!-- ── Hidden form fields (needed for page reload) ── -->
<form id="chart-form" method="get" style="display:none">
    <input type="hidden" name="mode" value="{{ mode }}">
    <input type="hidden" name="symbol" id="sym-form-sym" value="{{ symbol }}">
    <input type="hidden" name="tf"     id="sym-form-tf"  value="{{ tf }}">
    <input type="hidden" name="bars"   id="sym-form-bars" value="{{ bars }}">
    {% if mode == "multi" %}
    {% for t in selected_tfs %}<input type="hidden" name="mtf" value="{{ t }}">{% endfor %}
    {% for t, b in bars_map.items() %}<input type="hidden" name="bars_{{ t }}" value="{{ b }}">{% endfor %}
    {% endif %}
</form>
<input type="hidden" name="symbol" id="sym-hidden" value="{{ symbol }}">

<!-- ── Unified Toolbar ── -->
<div class="toolbar">

    <!-- Symbol -->
    <div class="sym-search-wrap" id="sym-search-wrap">
        <input type="text" id="sym-search-input" autocomplete="off" spellcheck="false"
            value="{{ symbol }}"
            oninput="onChartSymInput(this)"
            onfocus="onChartSymFocus(this)"
            onkeydown="onChartSymKey(event)"
            placeholder="Symbol...">
        <div class="sym-dropdown" id="sym-dropdown" style="display:none"></div>
    </div>

    <!-- TF pills (chart) -->
    {% if mode == "single" %}
    <div class="tf-pills" id="main-tf-pills">
        {% for t in all_tfs %}
        <button type="button" class="tf-pill {% if t == tf %}active{% endif %}"
            onclick="selectMainTf('{{ t }}')" data-tf="{{ t }}">{{ t }}</button>
        {% endfor %}
    </div>
    <select name="tf" id="main-tf-select" style="display:none">
        {% for t in all_tfs %}<option value="{{ t }}" {% if t == tf %}selected{% endif %}>{{ t }}</option>{% endfor %}
    </select>
    {% endif %}

    <!-- Bars -->
    <input type="number" id="chart-bars-input" value="{{ bars }}" min="50" max="5000" step="50"
        style="width:68px" title="Lumânări" onchange="refreshChart()">

    <!-- Chart actions -->
    <div class="tb-sep"></div>
    <button type="button" class="tb-btn" onclick="refreshChart()" title="Reîncarcă graficul">↻</button>
    <button type="button" class="tb-btn" onclick="fitChart()" title="Fit — resetează zoom la toate barele">⊡ Fit</button>
    <!-- Zoom presets -->
    <div class="tf-pills" id="zoom-pills">
        <button type="button" class="tf-pill" onclick="zoomToLast(100)" title="Ultimele 100 lumânări">100</button>
        <button type="button" class="tf-pill" onclick="zoomToLast(300)" title="Ultimele 300 lumânări">300</button>
        <button type="button" class="tf-pill" onclick="zoomToLast(500)" title="Ultimele 500 lumânări">500</button>
        <button type="button" class="tf-pill" onclick="fitChart()" title="Toate barele">All</button>
    </div>
    {% if mode == "single" %}
    <a href="?mode=multi&symbol={{ symbol }}" class="tb-btn tb-btn-purple">⊞ Multi</a>
    {% else %}
    <a href="?mode=single&symbol={{ symbol }}&tf=M5&bars=2000" class="tb-btn tb-btn-purple">⊟ Single</a>
    {% endif %}
    <button type="button" class="tb-btn tb-btn-yellow" onclick="decideNow('{{ symbol }}')">⚡ Decide</button>

    <!-- Big separator -->
    <div class="tb-sep tb-sep-big"></div>

    <!-- Analysis controls -->
    <select id="sim-strategy" onchange="onSimStrategyChange(this.value)" class="tb-select" style="min-width:140px"></select>
    <div id="sim-tfs" class="tf-pills" style="flex-wrap:nowrap"></div>
    <input type="number" id="sim-bars" value="2000" min="50" max="5000" step="50" style="width:60px" title="Bare analiza">
    <div class="tb-sep"></div>
    <div class="tb-field">
        <span class="tb-label">Lot</span>
        <input type="number" id="sim-lot" value="0.10" min="0.01" max="10" step="0.01" style="width:62px" oninput="onSimLotChange()">
    </div>
    <div class="tb-field">
        <span class="tb-label">Risc $</span>
        <input type="number" id="sim-max-risk" value="50" min="1" max="10000" step="1" style="width:58px" oninput="onSimMaxRiskChange()">
    </div>
    <button id="sim-btn" class="tb-btn tb-btn-run" onclick="runSimulation()">▶ Analiza</button>
    <button id="sim-overlay-ind-btn" class="tb-btn" style="display:none" onclick="toggleIndicatorOverlay()">📊</button>

    <!-- Status + refresh indicator -->
    <div style="margin-left:auto;display:flex;align-items:center;gap:10px">
        <div id="sim-loading" style="display:none;color:#555;font-size:0.78rem">⏳ analiza...</div>
        <div id="sim-error"   style="display:none;color:#ef5350;font-size:0.78rem;max-width:200px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap"></div>
        <span id="sim-strat-label" style="font-size:0.75rem;color:#444"></span>
        <span id="chart-refresh-status" style="font-size:0.7rem;color:#333">↻</span>
    </div>
</div>
<button id="analyze-btn" style="display:none" onclick="quickAnalyze()"></button>

<!-- ── Signal Strip ── -->
<div id="sim-results" style="display:none">

    <!-- HOLD banner -->
    <div id="sim-hold-reason" style="display:none;background:#1a1400;border-bottom:1px solid #3d2e00;padding:8px 20px;display:flex;align-items:center;gap:10px">
        <span style="font-size:0.72rem;color:#ff9800;font-weight:600;text-transform:uppercase;letter-spacing:0.4px;white-space:nowrap">⚠ HOLD</span>
        <div id="sim-hold-text" style="color:#666;font-size:0.78rem;line-height:1.5"></div>
    </div>

    <!-- Signal strip row -->
    <div class="signal-strip">
        <!-- Signal badge -->
        <div id="sim-sig" class="sig-badge">—</div>
        <span id="sim-sig-inline" style="display:none"></span>

        <!-- Confidence -->
        <div class="strip-field">
            <div class="sf-label">Confidence</div>
            <div id="sim-conf" class="sf-val">—</div>
        </div>

        <div class="strip-sep"></div>

        <!-- Entry / SL / TP -->
        <div class="strip-field">
            <div class="sf-label">Entry</div>
            <div id="sim-entry" class="sf-val sf-mono">—</div>
        </div>
        <div class="strip-field">
            <div class="sf-label">SL</div>
            <div id="sim-sl-val" class="sf-val sf-mono" style="color:#ef9a9a">—</div>
        </div>
        <div class="strip-field">
            <div class="sf-label">TP</div>
            <div id="sim-tp-val" class="sf-val sf-mono" style="color:#a5d6a7">—</div>
        </div>
        <div class="strip-field">
            <div class="sf-label">R:R</div>
            <div id="sim-rr" class="sf-val" style="font-weight:700">—</div>
        </div>
        <!-- Hidden combined sltp for compat -->
        <div id="sim-sltp" style="display:none">—</div>

        <div class="strip-sep"></div>

        <!-- Lot preview -->
        <div class="strip-field">
            <div class="sf-label">Lot</div>
            <div id="lp-lots" class="sf-val" style="color:#42a5f5;font-weight:700">—</div>
        </div>
        <div class="strip-field">
            <div class="sf-label">SL pip</div>
            <div id="lp-sl-pips" class="sf-val sf-mono" style="color:#ef9a9a">—</div>
        </div>
        <div class="strip-field">
            <div class="sf-label">TP pip</div>
            <div id="lp-tp-pips" class="sf-val sf-mono" style="color:#a5d6a7">—</div>
        </div>
        <div class="strip-field">
            <div class="sf-label">Risc / Reward</div>
            <div id="lp-riskrew" class="sf-val sf-mono">—</div>
        </div>
        <div class="strip-field">
            <div class="sf-label">Durată</div>
            <div id="lp-duration" class="sf-val" style="color:#888">—</div>
        </div>

        <div class="strip-sep"></div>

        <!-- Execute buttons -->
        <div style="display:flex;gap:6px;align-items:center">
            <button id="sim-buy-btn"  class="exec-btn exec-buy"  style="display:none" onclick="execSimTrade('BUY')">▲ BUY</button>
            <button id="sim-sell-btn" class="exec-btn exec-sell" style="display:none" onclick="execSimTrade('SELL')">▼ SELL</button>
            <button id="sim-overlay-btn" class="tb-btn" style="display:none;font-size:0.75rem;padding:4px 8px" onclick="toggleZoneOverlay()">📍</button>
        </div>

        <!-- TF breakdown toggle -->
        <button class="tb-btn" style="margin-left:auto;font-size:0.75rem;padding:4px 10px;color:#555"
            onclick="toggleTfTable()" id="tf-table-toggle">▾ TF</button>

        <!-- Trade SL/TP inputs (inline) -->
        <div style="display:flex;align-items:center;gap:6px;border-left:1px solid #1e1e1e;padding-left:12px;margin-left:4px">
            <input type="hidden" id="manual-symbol">
            <div class="tb-field">
                <span class="tb-label">SL</span>
                <input type="number" id="manual-sl" step="0.00001" style="width:90px">
            </div>
            <div class="tb-field">
                <span class="tb-label">TP</span>
                <input type="number" id="manual-tp" step="0.00001" style="width:90px">
            </div>
            <button class="exec-btn exec-buy"  onclick="executeManual('BUY')"  style="font-size:0.75rem;padding:4px 10px">▲</button>
            <button class="exec-btn exec-sell" onclick="executeManual('SELL')" style="font-size:0.75rem;padding:4px 10px">▼</button>
        </div>
    </div>

    <!-- Trade result -->
    <div id="trade-result"    style="display:none;padding:6px 20px;font-size:0.82rem;border-bottom:1px solid #1e1e1e"></div>
    <div id="force-trade-bar" style="display:none;padding:6px 20px;gap:8px;align-items:center;border-bottom:1px solid #1e1e1e;background:#1a0000">
        <span style="font-size:0.8rem;color:#aaa">Incearca oricum:</span>
        <button class="exec-btn exec-buy"  onclick="forceManual('BUY')">▲ Forteaza BUY</button>
        <button class="exec-btn exec-sell" onclick="forceManual('SELL')">▼ Forteaza SELL</button>
    </div>

    <!-- TF breakdown table (collapsible) -->
    <div id="tf-table-wrap" style="display:none;border-bottom:1px solid #1a1a1a;overflow-x:auto">
        <table id="sim-table" style="width:100%;border-collapse:collapse">
            <thead>
                <tr style="background:#0f0f0f">
                    <th class="th-cell" style="width:52px">TF</th>
                    <th class="th-cell" style="width:70px;text-align:center">Signal</th>
                    <th class="th-cell" style="width:110px;text-align:center">Conviction</th>
                    <th class="th-cell" style="text-align:right">Entry</th>
                    <th class="th-cell" style="text-align:right">Stop Loss</th>
                    <th class="th-cell" style="text-align:right">Take Profit</th>
                    <th class="th-cell" style="text-align:right;width:52px">R:R</th>
                    <th class="th-cell">Motive</th>
                </tr>
            </thead>
            <tbody id="sim-rows"></tbody>
        </table>
    </div>
</div>

<!-- ── hidden compat ── -->
<div id="trade-exec-panel" style="display:none"></div>

<!-- ── Drawing Tools Toolbar ── -->
<div id="draw-toolbar">
    <span style="font-size:0.72rem;color:#444;text-transform:uppercase;letter-spacing:0.5px">Draw:</span>
    <button id="draw-pan-btn" class="draw-tool-btn active" onclick="setDrawMode('pan',this)" title="Pan / Zoom">✋ Pan</button>
    <button class="draw-tool-btn" onclick="setDrawMode('drawline',this)" title="Linie dreapta">╱ Line</button>
    <button class="draw-tool-btn" onclick="setDrawMode('drawrect',this)" title="Dreptunghi / zona">▭ Rect</button>
    <button class="draw-tool-btn" onclick="setDrawMode('drawopenpath',this)" title="Linie libera">〰 Path</button>
    <button class="draw-tool-btn" onclick="setDrawMode('drawcircle',this)" title="Cerc">◯ Circle</button>
    <div style="width:1px;height:18px;background:#2a2a2a;margin:0 4px"></div>
    <button class="draw-tool-btn" style="color:#ef5350" onclick="clearAllDrawShapes()" title="Sterge toate formele">✕ Clear</button>
    <span style="font-size:0.72rem;color:#333;margin-left:8px">Click + drag pe grafic pentru a desena</span>
</div>

<div class="chart-container" style="position:relative">
    {% if chart %}
        {{ chart | safe }}
    {% else %}
        <p style="color:#888;font-size:0.85rem;padding:4px 0">Selecteaza un simbol si apasa Actualizeaza.</p>
    {% endif %}
    <!-- Floating chart action buttons — sus-dreapta, nu overlap-eaza cu axa X -->
    <div id="chart-float-btns" style="position:absolute;top:58px;right:120px;display:flex;gap:6px;z-index:100">
        <button onclick="runSimulationAs('BUY')"
            style="background:rgba(10,32,16,0.88);color:#26a69a;border:1px solid #1b5e20;padding:5px 12px;border-radius:5px;cursor:pointer;font-size:0.78rem;font-weight:700;backdrop-filter:blur(4px);box-shadow:0 2px 10px #000a;transition:background 0.12s"
            onmouseover="this.style.background='rgba(13,47,24,0.95)'" onmouseout="this.style.background='rgba(10,32,16,0.88)'">
            ▲ Sim BUY
        </button>
        <button onclick="runSimulationAs('SELL')"
            style="background:rgba(32,10,10,0.88);color:#ef5350;border:1px solid #5e1b1b;padding:5px 12px;border-radius:5px;cursor:pointer;font-size:0.78rem;font-weight:700;backdrop-filter:blur(4px);box-shadow:0 2px 10px #000a;transition:background 0.12s"
            onmouseover="this.style.background='rgba(47,13,13,0.95)'" onmouseout="this.style.background='rgba(32,10,10,0.88)'">
            ▼ Sim SELL
        </button>
    </div>
</div>

</body></html>"""

# ── Login page HTML ──────────────────────────────────────────────────────────
LOGIN_HTML = """<!DOCTYPE html>
<html lang="ro"><head>
<meta charset="utf-8">
<title>Login — ChartVisualizer</title>
<style>
* { box-sizing:border-box; margin:0; padding:0; }
body { background:#0e0e0e; color:#eee; font-family:'Segoe UI',sans-serif;
       display:flex; align-items:center; justify-content:center; min-height:100vh; }
.card { background:#1a1a1a; border:1px solid #2a2a2a; border-radius:10px;
        padding:36px 40px; width:100%; max-width:360px; box-shadow:0 8px 32px rgba(0,0,0,0.5); }
.logo { font-size:1.5rem; font-weight:700; color:#9c27b0; margin-bottom:6px; }
.sub  { font-size:0.83rem; color:#666; margin-bottom:28px; }
label { font-size:0.78rem; color:#888; text-transform:uppercase; display:block; margin-bottom:4px; }
input { width:100%; background:#242424; color:#eee; border:1px solid #383838;
        padding:10px 12px; border-radius:5px; font-size:0.95rem; margin-bottom:16px; }
input:focus { outline:none; border-color:#9c27b0; }
button { width:100%; background:#9c27b0; color:#fff; border:none; padding:11px;
         border-radius:5px; font-size:1rem; font-weight:600; cursor:pointer; transition:background 0.15s; }
button:hover { background:#7b1fa2; }
.err { background:#b71c1c22; border:1px solid #b71c1c; color:#ef9a9a;
       padding:9px 12px; border-radius:5px; font-size:0.85rem; margin-bottom:16px; display:none; }
.err.show { display:block; }
</style>
</head>
<body>
<div class="card">
    <div class="logo">⚡ ChartVisualizer</div>
    <div class="sub">Autentifica-te pentru a continua</div>
    {% if error %}<div class="err show">{{ error }}</div>{% endif %}
    <form method="POST" action="/login">
        <input type="hidden" name="next" value="{{ next }}">
        <label>Utilizator</label>
        <input type="text" name="username" autofocus autocomplete="username" placeholder="username">
        <label>Parola</label>
        <input type="password" name="password" autocomplete="current-password" placeholder="••••••••">
        <button type="submit">Intra</button>
    </form>
</div>
</body></html>"""

# ── Routes ────────────────────────────────────────────────────────────────────

@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        username = request.form.get("username", "").strip()
        password = request.form.get("password", "")
        next_url = request.form.get("next", "/")
        if _check_password(username, password):
            session["logged_in"] = True
            session["username"]  = username
            return redirect(next_url or "/")
        error = "Utilizator sau parola incorecta"
        return render_template_string(LOGIN_HTML, error=error, next=next_url)
    next_url = request.args.get("next", "/")
    if session.get("logged_in"):
        return redirect(next_url)
    return render_template_string(LOGIN_HTML, error=None, next=next_url)

@app.route("/logout")
def logout():
    session.clear()
    return redirect("/login")
@app.route("/")
@login_required
def index():
    symbol = request.args.get("symbol", "EURUSD").upper()
    mode   = request.args.get("mode", "single")
    tf     = request.args.get("tf", "M5")
    bars   = int(request.args.get("bars", 2000))
    selected_tfs = request.args.getlist("mtf") or ["M1","M5","M15","H1","H4"]
    bars_map = {t: int(request.args.get(f"bars_{t}", 2000)) for t in ALL_TFS}
    if mode == "multi":
        chart = build_multi(symbol, selected_tfs, bars_map)
    else:
        chart = build_chart(symbol, tf, bars, compact=False)
    return render_template_string(
        HTML.replace("{{ RISK_DOLLARS }}", str(RISK_DOLLARS))
            .replace("{{ MIN_TF_VOTES }}", str(MIN_TF_VOTES))
            .replace("RISK_DOLLARS_VAL", str(int(RISK_DOLLARS))),
        symbols=SYMBOLS + SYMBOLS_CRYPTO, all_tfs=ALL_TFS,
        symbol=symbol, mode=mode, tf=tf, bars=bars,
        selected_tfs=selected_tfs, bars_map=bars_map, chart=chart,
    )

@app.route("/set_risk", methods=["POST"])
@login_required
def set_risk():
    global RISK_DOLLARS
    data = request.get_json(silent=True) or {}
    val  = data.get("risk_dollars")
    if val is None:
        return Response(json.dumps({"ok": False, "msg": "Lipseste risk_dollars"}), mimetype="application/json")
    try:
        val = float(val)
        if val <= 0 or val > 10000:
            raise ValueError("Valoare invalida")
        RISK_DOLLARS = val
        _save_risk_config()
        log.info(f"RISK_DOLLARS setat la ${RISK_DOLLARS}")
        return Response(json.dumps({"ok": True, "risk_dollars": RISK_DOLLARS}), mimetype="application/json")
    except Exception as e:
        return Response(json.dumps({"ok": False, "msg": str(e)}), mimetype="application/json")

@app.route("/get_risk")
@login_required
def get_risk():
    return Response(json.dumps({"risk_dollars": RISK_DOLLARS}), mimetype="application/json")

@app.route("/ftmo_status")
@login_required
def ftmo_status():
    from datetime import datetime, timezone
    ok, msg = check_ftmo_rules()
    acc = mt5.account_info() if MT5_AVAILABLE and mt5 else None
    now_utc = datetime.now(timezone.utc)
    # urmatoarea stire rosie din cache ForexFactory
    upcoming_news = get_upcoming_red_news(minutes_ahead=120)
    next_news = None
    if upcoming_news:
        ev = upcoming_news[0]
        next_news = {"time": ev.get("dt", ""), "in_minutes": round(ev["in_minutes"])}
    elif True:
        # cauta urmatoarea stire (chiar si peste 2 min)
        all_red = fetch_red_news()
        from datetime import datetime, timezone as _tz
        _now = datetime.now(_tz.utc)
        for ev in sorted(all_red, key=lambda x: x["dt"]):
            diff_m = (ev["dt"] - _now).total_seconds() / 60
            if diff_m > 0:
                next_news = {"time": ev["dt"].strftime("%H:%M UTC"), "in_minutes": round(diff_m)}
                break
    return Response(json.dumps({
        "ok":           ok,
        "message":      msg,
        "ftmo_enabled": FTMO_ENABLED,
        "balance":      round(acc.balance, 2)  if acc else 0,
        "equity":       round(acc.equity,  2)  if acc else 0,
        "daily_floor":  round(acc.balance * (1 - FTMO_DAILY_LOSS_PCT), 2) if acc else 0,
        "daily_used_pct": round((1 - acc.equity / acc.balance) * 100, 2) if acc and acc.balance > 0 else 0,
        "next_news":    next_news,
        "time_utc":     now_utc.strftime("%H:%M:%S"),
        "weekday":      now_utc.strftime("%A"),
    }), mimetype="application/json")

@app.route("/tick")
@login_required
def tick_route():
    symbol = request.args.get("symbol", "EURUSD").upper()
    if not MT5_AVAILABLE or mt5 is None:
        return Response(json.dumps({}), mimetype="application/json")
    tick = mt5.symbol_info_tick(symbol)
    if tick is None:
        return Response(json.dumps({}), mimetype="application/json")
    info = mt5.symbol_info(symbol)
    digits = info.digits if info else 5
    spread = round((tick.ask - tick.bid) / (10 ** -digits)) if info else 0
    return Response(json.dumps({
        "symbol": symbol,
        "bid": round(tick.bid, digits),
        "ask": round(tick.ask, digits),
        "spread": int(spread),
    }), mimetype="application/json")

@app.route("/chart_html")
@login_required
def chart_html_route():
    symbol = request.args.get("symbol", "EURUSD").upper()
    mode   = request.args.get("mode", "single")
    tf     = request.args.get("tf", "M5")
    bars   = int(request.args.get("bars", 2000))
    selected_tfs = request.args.getlist("mtf") or ["M1","M5","M15","H1","H4"]
    bars_map = {t: int(request.args.get(f"bars_{t}", 2000)) for t in ALL_TFS}
    if mode == "multi":
        html = build_multi(symbol, selected_tfs, bars_map)
    else:
        html = build_chart(symbol, tf, bars, compact=False)
    return Response(html, mimetype="text/html")

@app.route("/chart_json")
@login_required
def chart_json_route():
    """Returns the Plotly figure as JSON so the client can update in-place via Plotly.react()."""
    symbol = request.args.get("symbol", "EURUSD").upper()
    tf     = request.args.get("tf", "M5")
    bars   = int(request.args.get("bars", 2000))
    try:
        fig = build_chart(symbol, tf, bars, compact=False, return_fig=True)
        if isinstance(fig, str):
            # build_chart returned an error string (no data)
            return Response(json.dumps({"error": fig}), mimetype="application/json")
        # Use fig.to_json() — Plotly handles datetime/Timestamp serialization natively
        return Response(fig.to_json(), mimetype="application/json")
    except Exception as exc:
        log.warning(f"chart_json {symbol}/{tf}: {exc}")
        return Response(json.dumps({"error": str(exc)}), mimetype="application/json")

@app.route("/analyze")
@login_required
def analyze_route():
    symbol  = request.args.get("symbol", "EURUSD").upper()
    tfs_str = request.args.get("tfs", "M1,M5,M15,H1,H4")
    tfs     = [t.strip() for t in tfs_str.split(",") if t.strip()]
    bars_map = {t: int(request.args.get(f"bars_{t}", 2000)) for t in tfs}
    results = []
    for tf in tfs:
        r = get_signal_data(symbol, tf, bars_map.get(tf, 500))
        if r:
            results.append(r)
    buy_v  = [r for r in results if r["signal"] == "BUY"]
    sell_v = [r for r in results if r["signal"] == "SELL"]
    n_buy, n_sell = len(buy_v), len(sell_v)
    final = "HOLD"
    best  = None
    if n_buy >= MIN_TF_VOTES and n_buy > n_sell:
        final = "BUY"
        best  = max(buy_v,  key=lambda x: x["conviction"])
    elif n_sell >= MIN_TF_VOTES and n_sell > n_buy:
        final = "SELL"
        best  = max(sell_v, key=lambda x: x["conviction"])
    data = {"symbol": symbol, "signal": final,
            "n_buy": n_buy, "n_sell": n_sell, "n_total": len(results),
            "best": best, "all": results}
    return Response(json.dumps(data, cls=NpEncoder), mimetype="application/json")

@app.route("/strategies_info")
@login_required
def strategies_info():
    from strategies import list_all
    data = [
        {
            "key": s.key, "name": s.name, "icon": s.icon, "color": s.color,
            "default_tfs": s.default_tfs, "default_bars": s.default_bars,
        }
        for s in list_all()
    ]
    return Response(json.dumps(data), mimetype="application/json")

def _compute_overlay(df, strategy_key):
    """Returns Plotly trace specs for a strategy's indicators (to overlay on chart)."""
    import pandas as pd

    if df is None or len(df) < 20:
        return {"traces": []}

    def _safe_list(series):
        vals = series.tolist() if hasattr(series, "tolist") else list(series)
        return [None if (v is None or (isinstance(v, float) and np.isnan(v))) else round(float(v), 8) for v in vals]

    try:
        dates = df.index.strftime("%Y-%m-%dT%H:%M:%S").tolist()
    except Exception:
        dates = [str(x) for x in df.index]

    close = df["close"]
    high  = df["high"]
    low   = df["low"]

    def line(name, series, color, width=1.3, dash=None):
        spec = {"type": "scatter", "x": dates, "y": _safe_list(series),
                "name": name, "line": {"color": color, "width": width},
                "mode": "lines", "hovertemplate": f"{name}: %{{y:.5f}}<extra></extra>"}
        if dash:
            spec["line"]["dash"] = dash
        return spec

    traces = []
    sk = strategy_key

    if sk in ("classic", "smc", "eob", "engulfing", "rsi_divergence"):
        traces.append(line("EMA 20", close.ewm(span=20, adjust=False).mean(), "#ffeb3b", 1.2))
        traces.append(line("EMA 50", close.ewm(span=50, adjust=False).mean(), "#ff9800", 1.5))

    elif sk == "bollinger":
        sma = close.rolling(20).mean()
        std = close.rolling(20).std()
        upper = sma + 2 * std
        lower = sma - 2 * std
        traces.append(line("BB Mid",   sma,   "#fff59d", 1.0))
        t_up = line("BB Upper", upper, "#4db6ac", 1.0, dash="dot")
        t_lo = line("BB Lower", lower, "#4db6ac", 1.0, dash="dot")
        t_lo["fill"] = "tonexty"; t_lo["fillcolor"] = "rgba(77,182,172,0.06)"
        traces += [t_up, t_lo]

    elif sk == "keltner_channel":
        ema20 = close.ewm(span=20, adjust=False).mean()
        pc    = close.shift(1)
        atr   = pd.concat([high - low, (high - pc).abs(), (low - pc).abs()], axis=1).max(axis=1).ewm(span=20, adjust=False).mean()
        t_up = line("KC Upper", ema20 + 1.5 * atr, "#ab47bc", 1.0, dash="dot")
        t_lo = line("KC Lower", ema20 - 1.5 * atr, "#ab47bc", 1.0, dash="dot")
        t_lo["fill"] = "tonexty"; t_lo["fillcolor"] = "rgba(171,71,188,0.07)"
        traces += [line("KC Mid", ema20, "#ce93d8", 1.5), t_up, t_lo]

    elif sk == "ema_cross":
        traces.append(line("EMA 8",  close.ewm(span=8,  adjust=False).mean(), "#4fc3f7", 1.2))
        traces.append(line("EMA 21", close.ewm(span=21, adjust=False).mean(), "#f48fb1", 1.5))
        traces.append(line("EMA 50", close.ewm(span=50, adjust=False).mean(), "#ff9800", 1.5, dash="dot"))

    elif sk == "macd":
        traces.append(line("EMA 12",  close.ewm(span=12,  adjust=False).mean(), "#4fc3f7", 1.0))
        traces.append(line("EMA 26",  close.ewm(span=26,  adjust=False).mean(), "#ff8a65", 1.0))
        traces.append(line("EMA 200", close.ewm(span=200, adjust=False).mean(), "#fff176", 1.8))

    elif sk == "supertrend":
        period, mult = 10, 3.0
        pc  = close.shift(1)
        atr = pd.concat([high - low, (high - pc).abs(), (low - pc).abs()], axis=1).max(axis=1).ewm(span=period, adjust=False).mean()
        hl2   = (high + low) / 2
        u_arr = (hl2 + mult * atr).to_numpy()
        l_arr = (hl2 - mult * atr).to_numpy()
        c_arr = close.to_numpy()
        fu, fl = u_arr.copy(), l_arr.copy()
        bull = np.ones(len(df), dtype=bool)
        for i in range(1, len(df)):
            fu[i] = u_arr[i] if (u_arr[i] < fu[i-1] or c_arr[i-1] > fu[i-1]) else fu[i-1]
            fl[i] = l_arr[i] if (l_arr[i] > fl[i-1] or c_arr[i-1] < fl[i-1]) else fl[i-1]
            if bull[i-1] and c_arr[i] < fl[i]:   bull[i] = False
            elif not bull[i-1] and c_arr[i] > fu[i]: bull[i] = True
            else: bull[i] = bull[i-1]
        st_bull = [float(fl[i]) if bull[i]  else None for i in range(len(df))]
        st_bear = [float(fu[i]) if not bull[i] else None for i in range(len(df))]
        traces.append({"type":"scatter","x":dates,"y":st_bull,"name":"ST Bull","line":{"color":"#26a69a","width":2},"mode":"lines","connectgaps":False,"hovertemplate":"ST Bull: %{y:.5f}<extra></extra>"})
        traces.append({"type":"scatter","x":dates,"y":st_bear,"name":"ST Bear","line":{"color":"#ef5350","width":2},"mode":"lines","connectgaps":False,"hovertemplate":"ST Bear: %{y:.5f}<extra></extra>"})
        traces.append(line("EMA 50", close.ewm(span=50, adjust=False).mean(), "#ff9800", 1.2, dash="dot"))

    elif sk == "ichimoku":
        tenkan   = (high.rolling(9).max()  + low.rolling(9).min())  / 2
        kijun    = (high.rolling(26).max() + low.rolling(26).min()) / 2
        senkou_a = ((tenkan + kijun) / 2).shift(26)
        senkou_b = ((high.rolling(52).max() + low.rolling(52).min()) / 2).shift(26)
        traces.append(line("Tenkan (9)",  tenkan,   "#e91e63", 1.0))
        traces.append(line("Kijun (26)",  kijun,    "#1565c0", 1.5))
        t_sa = line("Senkou A", senkou_a, "#26a69a", 1.0)
        t_sb = line("Senkou B", senkou_b, "#ef5350", 1.0)
        t_sb["fill"] = "tonexty"; t_sb["fillcolor"] = "rgba(128,128,128,0.12)"
        traces += [t_sa, t_sb]

    elif sk == "vwap_bounce":
        hlc3 = (high + low + close) / 3
        vol_col = next((c for c in ("tick_volume", "volume") if c in df.columns), None)
        if vol_col:
            vol     = df[vol_col]
            cum_vol = vol.cumsum().replace(0, np.nan)
            vwap    = (hlc3 * vol).cumsum() / cum_vol
            std_v   = pd.Series([hlc3.iloc[max(0,i-20):i+1].std() for i in range(len(df))], index=df.index)
            traces.append(line("VWAP",    vwap,           "#ff9800", 2.0))
            traces.append(line("VWAP+1σ", vwap + std_v,  "#ff9800", 1.0, dash="dot"))
            traces.append(line("VWAP−1σ", vwap - std_v,  "#ff9800", 1.0, dash="dot"))

    elif sk in ("london_breakout", "ny_breakout", "china_session"):
        traces.append(line("EMA 20", close.ewm(span=20, adjust=False).mean(), "#ffeb3b", 1.2))

    return {"traces": traces}


@app.route("/strategy_overlay")
@login_required
def strategy_overlay_route():
    symbol       = request.args.get("symbol", "EURUSD").upper()
    strategy_key = request.args.get("strategy", "classic")
    tf           = request.args.get("tf", "H1")
    bars         = int(request.args.get("bars", 2000))
    try:
        df, _ = fetch(symbol, tf, bars)
        result = _compute_overlay(df, strategy_key)
    except Exception as exc:
        log.warning(f"strategy_overlay {symbol}/{strategy_key}/{tf}: {exc}")
        result = {"traces": []}
    return Response(json.dumps(result, cls=NpEncoder), mimetype="application/json")


@app.route("/api/lot_preview")
@login_required
def api_lot_preview():
    symbol   = request.args.get("symbol", "EURUSD").upper()
    entry    = float(request.args.get("entry", 0) or 0)
    sl       = float(request.args.get("sl", 0) or 0)
    tp       = float(request.args.get("tp", 0) or 0)
    tf       = request.args.get("tf", "H1")
    lot_ovr  = request.args.get("lot")       # lot manual override
    risk_ovr = request.args.get("max_risk")  # max risk override in $

    if not MT5_AVAILABLE or mt5 is None:
        return Response(json.dumps({"error": "MT5 nedisponibil"}), mimetype="application/json")
    if sl == 0 or entry == 0:
        return Response(json.dumps({"error": "entry/sl lipsa"}), mimetype="application/json")

    try:
        info = mt5.symbol_info(symbol)
        if not info:
            return Response(json.dumps({"error": f"Simbol {symbol} negasit"}), mimetype="application/json")

        tick_val  = info.trade_tick_value
        tick_size = info.trade_tick_size
        pip_size  = info.point * (10 if info.digits in (5, 3) else 1)

        sl_dist = abs(entry - sl)
        tp_dist = abs(tp - entry) if tp else 0

        if tick_size <= 0 or tick_val <= 0 or sl_dist <= 0:
            return Response(json.dumps({"error": "Date simbol invalide"}), mimetype="application/json")

        lot_step = info.volume_step
        min_lot  = info.volume_min
        max_lot  = min(info.volume_max, 1.0)

        if lot_ovr:
            # Mod manual: lot specificat de user → calculam riscul real
            lots = max(min_lot, min(max_lot, round(float(lot_ovr), 2)))
        else:
            # Mod auto: calculam lot din max_risk sau din RISK_DOLLARS
            risk_target = float(risk_ovr) if risk_ovr else _get_effective_risk()
            lots = math.floor(risk_target / (sl_dist / tick_size * tick_val) / lot_step) * lot_step
            lots = max(min_lot, min(max_lot, lots))
            lots = round(lots, 2)

        sl_pips    = round(sl_dist / pip_size, 1)
        tp_pips    = round(tp_dist / pip_size, 1) if tp_dist else 0
        risk_usd   = round(sl_dist / tick_size * tick_val * lots, 2)
        reward_usd = round(tp_dist / tick_size * tick_val * lots, 2) if tp_dist else 0
        risk_exceeded = risk_ovr and risk_usd > float(risk_ovr) * 1.05

        TF_DURATION = {
            "M1": "5–20 min", "M5": "15–45 min", "M15": "30–90 min",
            "M30": "1–3 ore", "H1": "2–8 ore", "H4": "8–24 ore",
            "D1": "1–5 zile", "W1": "1–4 sapt",
        }

        return Response(json.dumps({
            "lots":          lots,
            "sl_pips":       sl_pips,
            "tp_pips":       tp_pips,
            "risk_usd":      risk_usd,
            "reward_usd":    reward_usd,
            "risk_exceeded": bool(risk_exceeded),
            "duration":      TF_DURATION.get(tf, "—"),
        }), mimetype="application/json")
    except Exception as exc:
        return Response(json.dumps({"error": str(exc)}), mimetype="application/json")


@app.route("/simulate")
@login_required
def simulate_route():
    symbol       = request.args.get("symbol", "EURUSD").upper()
    strategy_key = request.args.get("strategy", "classic")
    tfs_str      = request.args.get("tfs", "")
    bars         = int(request.args.get("bars", 2000))
    from strategies import get_strategy
    strat = get_strategy(strategy_key)
    if strat is None:
        return Response(json.dumps({"error": f"Strategie inexistenta: {strategy_key}"}),
                        mimetype="application/json")
    tfs = [t.strip() for t in tfs_str.split(",") if t.strip()] if tfs_str else strat.default_tfs
    try:
        result = strat.analyze(symbol, tfs, bars=bars, min_confidence=55.0)
    except Exception as exc:
        log.warning(f"simulate_route {symbol}/{strategy_key}: {exc}")
        return Response(json.dumps({"error": str(exc)}), mimetype="application/json")
    return Response(json.dumps(result, cls=NpEncoder), mimetype="application/json")


# ── Telegram Notification Endpoints ──────────────────────────────────────────

@app.route("/telegram/config", methods=["GET", "POST"])
@login_required
def telegram_config():
    if request.method == "POST":
        d = request.get_json(silent=True) or {}
        _tg.save_config(
            token=d.get("bot_token", ""),
            chat_id=str(d.get("chat_id", "")),
            enabled=bool(d.get("enabled", False)),
            require_approval=bool(d.get("require_approval", True)),
        )
        return Response(json.dumps({"ok": True}), mimetype="application/json")
    return Response(json.dumps(_tg.get_config()), mimetype="application/json")


@app.route("/telegram/test")
@login_required
def telegram_test():
    ok, msg = _tg.test_connection()
    return Response(json.dumps({"ok": ok, "msg": msg}), mimetype="application/json")


@app.route("/telegram/pending")
@login_required
def telegram_pending():
    return Response(json.dumps(_tg.get_all_recent(30), cls=NpEncoder), mimetype="application/json")


@app.route("/telegram/approve/<token>", methods=["POST"])
@login_required
def telegram_approve(token):
    ok, msg = _tg.manual_approve(token)
    return Response(json.dumps({"ok": ok, "msg": msg}), mimetype="application/json")


@app.route("/telegram/deny/<token>", methods=["POST"])
@login_required
def telegram_deny(token):
    ok, msg = _tg.manual_deny(token)
    return Response(json.dumps({"ok": ok, "msg": msg}), mimetype="application/json")


@app.route("/telegram/send_analysis", methods=["POST"])
@login_required
def telegram_send_analysis():
    """Trimite analiza curenta pe Telegram cu screenshot grafic."""
    d = request.get_json(silent=True) or {}
    symbol   = d.get("symbol", "EURUSD").upper()
    strategy = d.get("strategy", "classic")
    tf       = d.get("tf", "M5")
    bars     = int(d.get("bars", 2000))
    signal   = d.get("signal", "HOLD")
    confidence = float(d.get("confidence", 0))
    sl       = d.get("sl")
    tp       = d.get("tp")
    price    = d.get("price")
    reasons  = d.get("reasons", [])

    # Genereaza screenshot
    img_bytes = None
    try:
        fig = build_chart(symbol, tf, bars, return_fig=True)
        if not isinstance(fig, str):
            img_bytes = _tg.capture_chart(fig)
    except Exception as e:
        log.debug(f"telegram_send_analysis screenshot: {e}")

    if signal in ("BUY", "SELL") and sl and tp and price and _tg._require_approval:
        token = _tg.request_approval(
            symbol=symbol, signal=signal, strategy=strategy,
            sl=float(sl), tp=float(tp), price=float(price),
            confidence=confidence, tf=tf,
            img_bytes=img_bytes, reasons=reasons,
        )
        return Response(json.dumps({"ok": True, "mode": "approval", "token": token}),
                        mimetype="application/json")
    else:
        # Notificare simpla (HOLD sau aprobare dezactivata)
        _tg.notify_only(
            symbol=symbol, signal=signal, strategy=strategy,
            sl=float(sl) if sl else 0, tp=float(tp) if tp else 0,
            price=float(price) if price else 0,
            confidence=confidence, tf=tf,
            img_bytes=img_bytes, reasons=reasons,
        )
        return Response(json.dumps({"ok": True, "mode": "notify"}),
                        mimetype="application/json")


@app.route("/debug_trade")
@login_required
def debug_trade():
    symbol = request.args.get("symbol", "EURUSD").upper()
    if not MT5_AVAILABLE or mt5 is None:
        return Response(json.dumps({"error": "MT5 indisponibil"}), mimetype="application/json")
    try:
        info = mt5.symbol_info(symbol)
        tick = mt5.symbol_info_tick(symbol)
        term = mt5.terminal_info()
        acc  = mt5.account_info()
        data = {
            "symbol":        symbol,
            "filling_mode":  int(info.filling_mode) if info else None,
            "trade_mode":    int(info.trade_mode)   if info else None,
            "digits":        int(info.digits)        if info else None,
            "ask":           float(tick.ask)         if tick else None,
            "bid":           float(tick.bid)         if tick else None,
            "term_trade_allowed":    bool(term.trade_allowed)    if term else None,
            "acc_trade_allowed":     bool(acc.trade_allowed)     if acc else None,
            "acc_trade_expert":      bool(acc.trade_expert)      if acc else None,
            "ORDER_FILLING_FOK":     int(mt5.ORDER_FILLING_FOK),
            "ORDER_FILLING_IOC":     int(mt5.ORDER_FILLING_IOC),
            "ORDER_FILLING_RETURN":  int(mt5.ORDER_FILLING_RETURN),
        }
    except Exception as e:
        data = {"error": str(e)}
    return Response(json.dumps(data), mimetype="application/json")

@app.route("/mt5_status")
@login_required
def mt5_status():
    if not MT5_AVAILABLE or mt5 is None:
        return Response(json.dumps({"ok": False, "msg": "MT5 nu e disponibil"}), mimetype="application/json")
    info = mt5.terminal_info()
    acc  = mt5.account_info()
    data = {
        "ok": True,
        "connected": info.connected if info else False,
        "trade_allowed": info.trade_allowed if info else False,
        "balance": acc.balance if acc else 0,
        "equity": acc.equity if acc else 0,
        "server": acc.server if acc else "",
        "login": acc.login if acc else 0,
    }
    return Response(json.dumps(data), mimetype="application/json")

@app.route("/trade", methods=["POST"])
@login_required
def trade_route():
    symbol = request.args.get("symbol", "EURUSD").upper()
    signal = request.args.get("signal", "")
    sl     = float(request.args.get("sl", 0))
    tp     = float(request.args.get("tp", 0))
    if signal not in ("BUY", "SELL") or sl == 0 or tp == 0:
        return Response(json.dumps({"ok": False, "message": "Parametri invalizi"}),
                        mimetype="application/json")
    ok, msg = place_trade(symbol, signal, sl, tp, RISK_DOLLARS)
    return Response(json.dumps({"ok": ok, "message": msg}), mimetype="application/json")

ACCOUNT_HTML = """<!DOCTYPE html>
<html><head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Cont MT5</title>
<style>
* { box-sizing:border-box; margin:0; padding:0; }
body { background:#111; color:#eee; font-family:'Segoe UI',monospace; padding:16px; }
h2 { font-size:1rem; color:#888; font-weight:400; margin-bottom:14px; }
.grid { display:grid; grid-template-columns:repeat(auto-fit,minmax(160px,1fr)); gap:10px; margin-bottom:20px; }
.card { background:#1a1a1a; border:1px solid #333; border-radius:8px; padding:14px 16px; }
.card .label { font-size:0.75rem; color:#888; margin-bottom:4px; }
.card .value { font-size:1.3rem; font-weight:bold; color:#eee; }
.card .value.green { color:#26a69a; }
.card .value.red   { color:#ef5350; }
.card .value.yellow{ color:#ffeb3b; }
table { width:100%; border-collapse:collapse; font-size:0.83rem; }
th { color:#888; font-weight:400; padding:6px 10px; border-bottom:1px solid #333; text-align:left; }
td { padding:7px 10px; border-bottom:1px solid #1e1e1e; }
.buy  { color:#26a69a; font-weight:bold; }
.sell { color:#ef5350; font-weight:bold; }
.section { background:#1a1a1a; border:1px solid #333; border-radius:8px; padding:14px 16px; margin-bottom:14px; }
.section h3 { font-size:0.85rem; color:#888; font-weight:400; margin-bottom:10px; }
.close-btn { background:#b71c1c; color:#ef9a9a; border:none; padding:4px 10px; border-radius:4px; cursor:pointer; font-size:0.78rem; }
.close-btn:hover { background:#c62828; }
#msg { margin-top:10px; padding:8px 12px; border-radius:4px; display:none; font-size:0.85rem; }
.ok  { background:#1b5e20; color:#a5d6a7; }
.err { background:#b71c1c; color:#ef9a9a; }
.refresh { color:#1976d2; font-size:0.8rem; cursor:pointer; text-decoration:underline; margin-left:10px; }
a.back { color:#888; font-size:0.82rem; text-decoration:none; display:inline-block; margin-bottom:14px; }
a.back:hover { color:#ccc; }
.login-section { background:#1a1a1a; border:1px solid #444; border-radius:8px; padding:16px 20px; margin-bottom:16px; }
.login-section h3 { font-size:0.9rem; color:#aaa; margin-bottom:12px; font-weight:400; }
.login-row { display:flex; gap:10px; flex-wrap:wrap; align-items:flex-end; }
.login-field { display:flex; flex-direction:column; gap:4px; }
.login-field label { font-size:0.74rem; color:#777; text-transform:uppercase; }
.login-field input { background:#242424; color:#eee; border:1px solid #444; padding:6px 10px; border-radius:4px; font-size:0.85rem; width:160px; }
.login-field input:focus { outline:none; border-color:#1976d2; }
.btn-connect { background:#1976d2; color:#fff; border:none; padding:7px 18px; border-radius:4px; cursor:pointer; font-size:0.85rem; }
.btn-connect:hover { background:#1565c0; }
.btn-disconnect { background:#b71c1c; color:#ef9a9a; border:none; padding:5px 14px; border-radius:4px; cursor:pointer; font-size:0.82rem; margin-left:12px; }
.btn-disconnect:hover { background:#c62828; color:#fff; }
#login-msg { margin-top:10px; padding:7px 12px; border-radius:4px; display:none; font-size:0.84rem; }
</style>
<script>
let autoRefresh = null;

function startAutoRefresh() {
    if(autoRefresh) clearInterval(autoRefresh);
    autoRefresh = setInterval(loadData, 1000);
}

async function loadData() {
    try {
        const resp = await fetch('/account_data');
        const d = await resp.json();
        if(!d.ok) {
            // arata formularul de login daca MT5 nu e conectat
            document.getElementById('login-section').style.display = 'block';
            document.getElementById('content').style.display = 'none';
            document.getElementById('btn-disconnect').style.display = 'none';
            document.getElementById('account-title').textContent = 'neconectat';
            return;
        }
        // ascunde login daca e conectat
        document.getElementById('login-section').style.display = 'none';
        document.getElementById('content').style.display = 'block';
        document.getElementById('btn-disconnect').style.display = 'inline-block';

        // cards
        const pl_color = d.profit >= 0 ? 'green' : 'red';
        const eq_color = d.equity >= d.balance ? 'green' : 'red';
        document.getElementById('card-balance').textContent  = d.balance.toFixed(2) + ' ' + d.currency;
        document.getElementById('card-equity').textContent   = d.equity.toFixed(2)  + ' ' + d.currency;
        document.getElementById('card-equity').className     = 'value ' + eq_color;
        document.getElementById('card-margin').textContent   = d.margin.toFixed(2)   + ' ' + d.currency;
        document.getElementById('card-free').textContent     = d.free_margin.toFixed(2) + ' ' + d.currency;
        document.getElementById('card-profit').textContent   = (d.profit >= 0 ? '+' : '') + d.profit.toFixed(2) + ' ' + d.currency;
        document.getElementById('card-profit').className     = 'value ' + pl_color;
        document.getElementById('card-positions').textContent = d.positions.length;
        document.getElementById('card-level').textContent    = d.margin_level > 0 ? d.margin_level.toFixed(0) + '%' : '—';
        document.getElementById('account-title').textContent = d.login + ' · ' + d.server;

        // pozitii
        let rows = '';
        if(d.positions.length === 0) {
            rows = '<tr><td colspan="8" style="color:#666;text-align:center">Nicio pozitie deschisa</td></tr>';
        } else {
            for(const p of d.positions) {
                const pc = p.type==='BUY' ? 'buy' : 'sell';
                const plc = p.profit >= 0 ? 'green' : 'red';
                rows += `<tr>
                    <td><b>${p.symbol}</b></td>
                    <td class="${pc}">${p.type}</td>
                    <td>${p.volume}</td>
                    <td>${p.price_open}</td>
                    <td>${p.price_current}</td>
                    <td>${p.sl || '—'}</td>
                    <td>${p.tp || '—'}</td>
                    <td style="color:${p.profit>=0?'#26a69a':'#ef5350'};font-weight:bold">
                        ${p.profit>=0?'+':''}${p.profit.toFixed(2)}
                    </td>
                    <td><button class="close-btn" onclick="closePos(${p.ticket})">✕</button></td>
                </tr>`;
            }
        }
        document.getElementById('pos-rows').innerHTML = rows;
        document.getElementById('last-update').textContent = 'Actualizat: ' + new Date().toLocaleTimeString();
    } catch(e) {
        console.error(e);
    }
}

async function mt5Login() {
    const login    = document.getElementById('inp-login').value.trim();
    const password = document.getElementById('inp-pass').value.trim();
    const server   = document.getElementById('inp-server').value.trim();
    const path     = document.getElementById('inp-path').value.trim();
    if (!login || !password || !server) { alert('Completeaza Login, Parola si Server'); return; }
    const btn = document.getElementById('btn-login');
    btn.textContent = 'Se conecteaza...'; btn.disabled = true;
    try {
        const resp = await fetch('/mt5_login', {
            method: 'POST',
            headers: {'Content-Type':'application/json'},
            body: JSON.stringify({login: parseInt(login), password, server, path})
        });
        const d = await resp.json();
        const msg = document.getElementById('login-msg');
        msg.style.display = 'block';
        msg.className = d.ok ? 'ok' : 'err';
        msg.textContent = d.message;
        if (d.ok) { loadData(); document.getElementById('login-section').style.display='none'; }
    } catch(e) { alert('Eroare: ' + e); }
    finally { btn.textContent = 'Conecteaza'; btn.disabled = false; }
}

async function closePos(ticket) {
    if(!confirm('Inchizi pozitia ' + ticket + '?')) return;
    const resp = await fetch('/close_position?ticket=' + ticket, {method:'POST'});
    const d = await resp.json();
    const msg = document.getElementById('msg');
    msg.style.display = 'block';
    msg.className = d.ok ? 'ok' : 'err';
    msg.textContent = d.message;
    loadData();
}

async function mt5Disconnect() {
    if (!confirm('Deconectezi contul MT5 curent?')) return;
    const btn = document.getElementById('btn-disconnect');
    btn.textContent = '...'; btn.disabled = true;
    try {
        const resp = await fetch('/mt5_disconnect', {method:'POST'});
        const d = await resp.json();
        if (d.ok) {
            document.getElementById('content').style.display = 'none';
            document.getElementById('login-section').style.display = 'block';
            document.getElementById('account-title').textContent = 'neconectat';
        }
    } catch(e) {}
    btn.textContent = 'Deconecteaza'; btn.disabled = false;
}

async function setRisk(val) {
    const v = parseFloat(val);
    if (!v || v <= 0) return;
    const resp = await fetch('/set_risk', {
        method: 'POST',
        headers: {'Content-Type':'application/json'},
        body: JSON.stringify({risk_dollars: v})
    });
    const d = await resp.json();
    if (d.ok) {
        document.getElementById('risk-input').style.borderColor = '#26a69a';
        setTimeout(() => document.getElementById('risk-input').style.borderColor = '#333', 1500);
    }
}

window.onload = () => { loadData(); startAutoRefresh(); };
</script>
</head><body>
<a class="back" href="/">← ChartVisualizer</a>
<h2>Cont MT5 — <span id="account-title">...</span>
    <span class="refresh" onclick="loadData()">↻ refresh</span>
    <span id="last-update" style="color:#555;font-size:0.75rem;margin-left:10px"></span>
    <button class="btn-disconnect" id="btn-disconnect" onclick="mt5Disconnect()" style="display:none">⏏ Deconecteaza</button>
</h2>

<!-- Login MT5 -->
<div class="login-section" id="login-section">
    <h3>🔑 Conectare cont MT5</h3>
    <div class="login-row">
        <div class="login-field">
            <label>Login (nr. cont)</label>
            <input type="number" id="inp-login" placeholder="ex: 12345678">
        </div>
        <div class="login-field">
            <label>Parola</label>
            <input type="password" id="inp-pass" placeholder="parola cont">
        </div>
        <div class="login-field">
            <label>Server broker</label>
            <input type="text" id="inp-server" placeholder="ex: ICMarkets-Demo">
        </div>
        <div class="login-field">
            <label>Cale terminal (optional)</label>
            <input type="text" id="inp-path" placeholder="C:\\...\\terminal64.exe" style="width:220px">
        </div>
        <button class="btn-connect" id="btn-login" onclick="mt5Login()">Conecteaza</button>
    </div>
    <div id="login-msg"></div>
</div>

<div id="content">
    <div class="grid">
        <div class="card"><div class="label">Balance</div><div class="value" id="card-balance">...</div></div>
        <div class="card"><div class="label">Equity</div><div class="value" id="card-equity">...</div></div>
        <div class="card"><div class="label">Profit flotant</div><div class="value" id="card-profit">...</div></div>
        <div class="card"><div class="label">Margin</div><div class="value yellow" id="card-margin">...</div></div>
        <div class="card"><div class="label">Free Margin</div><div class="value" id="card-free">...</div></div>
        <div class="card"><div class="label">Margin Level</div><div class="value" id="card-level">...</div></div>
        <div class="card"><div class="label">Pozitii deschise</div><div class="value" id="card-positions">...</div></div>
        <div class="card" style="min-width:140px">
            <div class="label">Risc / Trade</div>
            <div style="display:flex;align-items:center;gap:6px;margin-top:4px">
                <span style="color:#aaa;font-size:0.85rem">$</span>
                <input id="risk-input" type="number" min="1" max="10000" step="1"
                    value="RISK_DOLLARS_VAL"
                    style="width:70px;background:#1e1e1e;color:#ffeb3b;border:1px solid #333;border-radius:4px;padding:3px 6px;font-size:0.95rem;font-weight:700"
                    onchange="setRisk(this.value)">
            </div>
        </div>
    </div>

    <div class="section">
        <h3>Pozitii deschise</h3>
        <table>
            <thead><tr>
                <th>Symbol</th><th>Tip</th><th>Volum</th>
                <th>Deschis la</th><th>Pret curent</th>
                <th>SL</th><th>TP</th><th>Profit</th><th></th>
            </tr></thead>
            <tbody id="pos-rows"></tbody>
        </table>
        <div id="msg"></div>
    </div>
</div>
</body></html>"""

@app.route("/account")
@login_required
def account_page():
    return ACCOUNT_HTML.replace("RISK_DOLLARS_VAL", str(int(RISK_DOLLARS)))

@app.route("/mt5_login", methods=["POST"])
@login_required
def mt5_login():
    global MT5_AVAILABLE
    import os
    body = request.get_json(silent=True) or {}
    login    = body.get("login")
    password = body.get("password", "")
    server   = body.get("server", "")
    path     = body.get("path", "").strip() or None

    if not login or not password or not server:
        return Response(json.dumps({"ok": False, "message": "Login, parola si server sunt obligatorii"}),
                        mimetype="application/json")
    if mt5 is None:
        return Response(json.dumps({"ok": False, "message": "Modulul MetaTrader5 nu e instalat"}),
                        mimetype="application/json")
    try:
        mt5.shutdown()
    except:
        pass
    try:
        kwargs = dict(login=int(login), password=str(password), server=str(server), timeout=30000)
        if path and os.path.exists(path):
            kwargs["path"] = path
        ok = mt5.initialize(**kwargs)
        if ok:
            MT5_AVAILABLE = True
            acc = mt5.account_info()
            name = acc.name if acc else ""
            bal  = acc.balance if acc else 0
            return Response(json.dumps({
                "ok": True,
                "message": f"Conectat: {name} | Cont #{login} | Balance: {bal:.2f}"
            }), mimetype="application/json")
        else:
            err = mt5.last_error()
            MT5_AVAILABLE = False
            return Response(json.dumps({
                "ok": False,
                "message": f"Eroare MT5: {err}"
            }), mimetype="application/json")
    except Exception as e:
        return Response(json.dumps({"ok": False, "message": str(e)}), mimetype="application/json")

@app.route("/mt5_disconnect", methods=["POST"])
@login_required
def mt5_disconnect():
    global MT5_AVAILABLE
    if mt5 is None:
        return Response(json.dumps({"ok": False, "message": "MT5 nu e disponibil"}), mimetype="application/json")
    try:
        mt5.shutdown()
    except Exception:
        pass
    MT5_AVAILABLE = False
    log.info("MT5 deconectat manual de utilizator")
    return Response(json.dumps({"ok": True, "message": "Deconectat"}), mimetype="application/json")

@app.route("/account_data")
@login_required
def account_data():
    if not MT5_AVAILABLE or mt5 is None:
        return Response(json.dumps({"ok": False, "msg": "MT5 indisponibil"}), mimetype="application/json")
    acc = mt5.account_info()
    if acc is None:
        return Response(json.dumps({"ok": False, "msg": "Nu s-a putut obtine info cont"}), mimetype="application/json")

    all_pos = mt5.positions_get()
    positions = []
    if all_pos:
        for p in all_pos:
            open_dt = datetime.fromtimestamp(p.time, tz=timezone.utc).strftime("%Y-%m-%d %H:%M:%S") if p.time else ""
            # Source detection
            cmt = (p.comment or "")
            magic = getattr(p, "magic", 0)
            if magic == 991 or cmt.startswith("AI-") or cmt.startswith("ai_"):
                source = "AI"
                source_icon = "🧠"
            elif cmt and not cmt.startswith("AI-"):
                source = "AutoTrader"
                source_icon = "⚡"
            else:
                source = "Manual"
                source_icon = "👤"
            positions.append({
                "ticket":        int(p.ticket),
                "symbol":        p.symbol,
                "type":          "BUY" if p.type == 0 else "SELL",
                "volume":        p.volume,
                "price_open":    round(p.price_open, 5),
                "price_current": round(p.price_current, 5),
                "sl":            round(p.sl, 5) if p.sl else 0,
                "tp":            round(p.tp, 5) if p.tp else 0,
                "profit":        round(p.profit, 2),
                "swap":          round(p.swap, 2) if hasattr(p, 'swap') else 0,
                "comment":       cmt,
                "magic":         int(magic),
                "source":        source,
                "source_icon":   source_icon,
                "open_time":     open_dt,
                "open_ts":       int(p.time) if p.time else 0,
            })
    positions.sort(key=lambda x: x["open_ts"], reverse=True)

    total_profit = sum(p["profit"] for p in positions)

    data = {
        "ok":           True,
        "login":        acc.login,
        "server":       acc.server,
        "currency":     acc.currency,
        "balance":      round(acc.balance, 2),
        "equity":       round(acc.equity, 2),
        "margin":       round(acc.margin, 2),
        "free_margin":  round(acc.margin_free, 2),
        "margin_level": round(acc.margin_level, 2) if acc.margin_level else 0,
        "profit":       round(total_profit, 2),
        "positions":    positions,
    }
    return Response(json.dumps(data, cls=NpEncoder), mimetype="application/json")

@app.route("/close_all_trades", methods=["POST"])
@login_required
def close_all_trades():
    if not MT5_AVAILABLE or mt5 is None:
        return Response(json.dumps({"ok": False, "message": "MT5 indisponibil"}), mimetype="application/json")
    positions = mt5.positions_get() or []
    closed = 0
    for pos in positions:
        tick = mt5.symbol_info_tick(pos.symbol)
        if not tick:
            continue
        info = mt5.symbol_info(pos.symbol)
        close_price = tick.bid if pos.type == 0 else tick.ask
        order_type  = mt5.ORDER_TYPE_SELL if pos.type == 0 else mt5.ORDER_TYPE_BUY
        fm = info.filling_mode if info else 0
        if fm & 2:    filling = mt5.ORDER_FILLING_IOC
        elif fm & 1:  filling = mt5.ORDER_FILLING_FOK
        else:         filling = mt5.ORDER_FILLING_RETURN
        req = {
            "action": mt5.TRADE_ACTION_DEAL, "symbol": pos.symbol,
            "volume": pos.volume, "type": order_type, "price": close_price,
            "position": pos.ticket, "deviation": 30, "magic": pos.magic,
            "comment": "manual_close_all", "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": filling,
        }
        result = mt5.order_send(req)
        if result and result.retcode == mt5.TRADE_RETCODE_DONE:
            closed += 1
    return Response(json.dumps({"ok": True, "closed": closed}), mimetype="application/json")


@app.route("/close_position", methods=["POST"])
@login_required
def close_position():
    if not MT5_AVAILABLE or mt5 is None:
        return Response(json.dumps({"ok": False, "message": "MT5 indisponibil"}), mimetype="application/json")
    ticket = int(request.args.get("ticket", 0))
    if not mt5.positions_get(ticket=ticket):
        return Response(json.dumps({"ok": False, "message": "Pozitia nu exista"}), mimetype="application/json")
    pos = mt5.positions_get(ticket=ticket)[0]
    tick = mt5.symbol_info_tick(pos.symbol)
    close_price = tick.bid if pos.type == 0 else tick.ask
    info = mt5.symbol_info(pos.symbol)
    fm = info.filling_mode if info else 2
    if fm & 2:    filling = mt5.ORDER_FILLING_IOC
    elif fm & 1:  filling = mt5.ORDER_FILLING_FOK
    else:         filling = mt5.ORDER_FILLING_RETURN
    req = {
        "action":       mt5.TRADE_ACTION_DEAL,
        "symbol":       pos.symbol,
        "volume":       pos.volume,
        "type":         mt5.ORDER_TYPE_SELL if pos.type == 0 else mt5.ORDER_TYPE_BUY,
        "position":     ticket,
        "price":        close_price,
        "deviation":    30,
        "magic":        pos.magic,
        "comment":      "close",
        "type_time":    mt5.ORDER_TIME_GTC,
        "type_filling": filling,
    }
    result = mt5.order_send(req)
    if result and result.retcode == mt5.TRADE_RETCODE_DONE:
        return Response(json.dumps({"ok": True, "message": f"Pozitia {ticket} inchisa"}), mimetype="application/json")
    code = result.retcode if result else -1
    return Response(json.dumps({"ok": False, "message": f"Eroare inchidere: {code}"}), mimetype="application/json")

@app.route("/modify_trade", methods=["POST"])
@login_required
def modify_trade():
    if not MT5_AVAILABLE or mt5 is None:
        return Response(json.dumps({"ok": False, "message": "MT5 indisponibil"}), mimetype="application/json")
    body   = request.get_json(silent=True) or {}
    ticket = int(body.get("ticket", 0))
    new_sl = float(body.get("sl", 0))
    new_tp = float(body.get("tp", 0))
    pos_list = mt5.positions_get(ticket=ticket)
    if not pos_list:
        return Response(json.dumps({"ok": False, "message": "Pozitia nu exista"}), mimetype="application/json")
    pos  = pos_list[0]
    info = mt5.symbol_info(pos.symbol)
    digits = info.digits if info else 5
    req = {
        "action":   mt5.TRADE_ACTION_SLTP,
        "position": ticket,
        "symbol":   pos.symbol,
        "sl":       round(new_sl, digits),
        "tp":       round(new_tp, digits),
    }
    result = mt5.order_send(req)
    if result and result.retcode == mt5.TRADE_RETCODE_DONE:
        return Response(json.dumps({"ok": True, "message": f"SL/TP modificat pe {pos.symbol}"}), mimetype="application/json")
    code = result.retcode if result else -1
    return Response(json.dumps({"ok": False, "message": f"Eroare modificare: {code}"}), mimetype="application/json")


def _build_trade_fig(ticket, tf_override=None, bars_override=None):
    """Construieste figura Plotly pentru un trade activ. Returneaza fig sau None."""
    if not MT5_AVAILABLE or mt5 is None:
        return None
    pos_list = mt5.positions_get(ticket=ticket)
    if not pos_list:
        return None
    pos    = pos_list[0]
    symbol = pos.symbol
    tf     = tf_override or request.args.get("tf", "M5")
    _default_bars = {"M1":800,"M5":500,"M15":300,"H1":200,"H4":150,"D1":100}.get(tf, 300)
    bars   = bars_override or int(request.args.get("bars", _default_bars))

    df, source = fetch(symbol, tf, bars)
    if df is None:
        return None

    tick = mt5.symbol_info_tick(symbol)
    price_now = float(tick.bid if pos.type == 0 else tick.ask) if tick else pos.price_current
    is_buy    = (pos.type == 0)
    col_entry = "rgba(255,152,0,0.5)"
    col_sl    = "rgba(239,83,80,0.45)"
    col_tp    = "rgba(38,166,154,0.45)"
    col_price = "rgba(255,235,59,0.6)"
    pl        = round(price_now - pos.price_open, 5) if is_buy else round(pos.price_open - price_now, 5)
    pl_pct    = round(pl / pos.price_open * 100, 3)

    # Calculeaza range Y bazat pe lumânari + linii trade (ignora valori aberante)
    candle_low  = float(df["low"].min())
    candle_high = float(df["high"].max())
    atr = float(df["high"].sub(df["low"]).rolling(14).mean().iloc[-1])
    # include SL si TP in range doar daca sunt rezonabile (in raza de 20xATR)
    y_points = [pos.price_open, price_now]
    if pos.sl and abs(pos.sl - price_now) < atr * 20: y_points.append(pos.sl)
    if pos.tp and abs(pos.tp - price_now) < atr * 20: y_points.append(pos.tp)
    y_min = min(candle_low,  min(y_points)) - atr * 0.5
    y_max = max(candle_high, max(y_points)) + atr * 0.5

    fig = make_subplots(rows=1, cols=1)
    fig.add_trace(go.Candlestick(
        x=df.index, open=df["open"], high=df["high"], low=df["low"], close=df["close"],
        name="OHLC", increasing_line_color="#26a69a", decreasing_line_color="#ef5350",
        increasing_fillcolor="#26a69a", decreasing_fillcolor="#ef5350",
        showlegend=False))

    info = mt5.symbol_info(symbol)
    digits = info.digits if info else 5

    def hline(y, color, dash, label, width=1):
        if abs(y - price_now) > atr * 25: return
        fig.add_hline(y=y,
            line=dict(color=color, width=width, dash=dash),
            annotation_text=f"  {label}: {round(y, digits)}",
            annotation_font=dict(color=color, size=10))

    hline(pos.price_open, col_entry, "solid", "Entry")
    if pos.sl: hline(pos.sl, col_sl, "dash", "SL")
    if pos.tp: hline(pos.tp, col_tp, "dash", "TP")
    hline(price_now, col_price, "solid", f"{round(price_now, digits)}")

    # Linie verticala + punct mic deasupra/dedesubt lumanarii de intrare
    if pos.time:
        try:
            open_dt = datetime.fromtimestamp(pos.time, tz=timezone.utc).replace(tzinfo=None)
            if len(df) > 0 and hasattr(df.index[0], 'tzinfo') and df.index[0].tzinfo is not None:
                open_dt = open_dt.replace(tzinfo=timezone.utc)

            # Gaseste bara cea mai apropiata de momentul deschiderii
            if open_dt in df.index:
                candle_extreme = float(df.loc[open_dt, "low"]) if is_buy else float(df.loc[open_dt, "high"])
            else:
                # Cea mai apropiata bara
                idx_pos = df.index.searchsorted(open_dt)
                idx_pos = min(idx_pos, len(df) - 1)
                candle_extreme = float(df["low"].iloc[idx_pos]) if is_buy else float(df["high"].iloc[idx_pos])

            dot_y = candle_extreme - atr * 0.5 if is_buy else candle_extreme + atr * 0.5
            marker_color = "#26a69a" if is_buy else "#ef5350"  # verde BUY, rosu SELL

            # Linie verticala subtire pe axa timpului
            fig.add_vline(
                x=open_dt.isoformat(),
                line=dict(color="rgba(255,255,255,0.2)", width=1, dash="dot"),
            )

            # Triunghi verde ▲ pentru BUY (sub lumânare), rosu ▼ pentru SELL (deasupra)
            fig.add_trace(go.Scatter(
                x=[open_dt],
                y=[dot_y],
                mode="markers",
                marker=dict(
                    symbol="triangle-up" if is_buy else "triangle-down",
                    size=12,
                    color=marker_color,
                    line=dict(color="#fff", width=1),
                ),
                showlegend=False,
                hovertemplate=f"Entry: {round(pos.price_open, digits)}<extra></extra>",
            ))
        except Exception as _e:
            log.warning(f"entry marker: {_e}")

    # zona profit/loss colorata (doar daca SL/TP sunt rezonabile)
    if pos.sl and pos.tp and abs(pos.tp - price_now) < atr * 20:
        fig.add_hrect(y0=pos.price_open, y1=pos.tp if is_buy else pos.sl,
            fillcolor="rgba(38,166,154,0.07)", line_width=0)
        fig.add_hrect(y0=pos.sl if is_buy else pos.tp, y1=pos.price_open,
            fillcolor="rgba(239,83,80,0.07)", line_width=0)

    pl_color = "#26a69a" if pl >= 0 else "#ef5350"
    title = (f"<b>{symbol}</b> {'BUY' if is_buy else 'SELL'} {pos.volume}L"
             f"  |  Entry: {round(pos.price_open, digits)}"
             f"  |  <span style='color:{pl_color}'>P&L: {'+' if pl>=0 else ''}{round(pos.profit,2)}$</span>"
             f"  |  {tf}")
    fig.update_layout(
        template="plotly_dark", paper_bgcolor="#111", plot_bgcolor="#111",
        title=dict(text=title, font=dict(size=12, color="#ddd")),
        xaxis_rangeslider_visible=False, height=380,
        dragmode="pan",
        margin=dict(l=50, r=120, t=45, b=20),
        yaxis=dict(gridcolor="#1e1e1e", range=[y_min, y_max]),
        xaxis=dict(gridcolor="#1e1e1e", uirevision=f"{symbol}_{tf}"),
        uirevision=f"{symbol}_{tf}",
    )
    return fig


@app.route("/trade_chartjson/<int:ticket>")
@login_required
def trade_chartjson(ticket):
    """Returneaza figura Plotly ca JSON — folosit de Plotly.react() client-side."""
    fig = _build_trade_fig(ticket)
    if fig is None:
        return Response('{"error":"no data"}', mimetype="application/json", status=500)
    return Response(fig.to_json(), mimetype="application/json")


@app.route("/trade_chart/<int:ticket>")
@login_required
def trade_chart(ticket):
    """Fallback HTML — compatibilitate."""
    fig = _build_trade_fig(ticket)
    if fig is None:
        return Response("<p style='color:#ef5350'>Date indisponibile</p>", mimetype="text/html")
    return Response(fig.to_html(full_html=False, include_plotlyjs="cdn"), mimetype="text/html")


TRADES_HTML = """<!DOCTYPE html>
<html><head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Trades Active</title>
<script src="https://cdn.plot.ly/plotly-2.35.2.min.js" charset="utf-8"></script>
<style>
* { box-sizing:border-box; margin:0; padding:0; }
body { background:#111; color:#eee; font-family:'Segoe UI',sans-serif; }
.navbar { background:#161616; border-bottom:1px solid #2a2a2a; padding:0 20px; height:44px; display:flex; align-items:center; justify-content:space-between; }
.navbar-brand { font-size:1rem; font-weight:600; color:#eee; }
.nav-links { display:flex; gap:8px; }
.btn { background:#1976d2; color:#fff; border:none; padding:6px 14px; border-radius:4px; cursor:pointer; font-size:0.82rem; text-decoration:none; display:inline-flex; align-items:center; gap:5px; }
.btn:hover { background:#1565c0; }
.btn-red  { background:#b71c1c; } .btn-red:hover  { background:#c62828; }
.btn-teal { background:#00695c; } .btn-teal:hover { background:#004d40; }
.btn-grey { background:#333; color:#bbb; } .btn-grey:hover { background:#444; }
.content { padding:16px 20px; }
.summary { display:grid; grid-template-columns:repeat(auto-fit,minmax(140px,1fr)); gap:10px; margin-bottom:20px; }
.sum-card { background:#1a1a1a; border:1px solid #2a2a2a; border-radius:6px; padding:12px 14px; }
.sum-card .lbl { font-size:0.72rem; color:#666; text-transform:uppercase; margin-bottom:4px; }
.sum-card .val { font-size:1.2rem; font-weight:bold; color:#eee; }
.val.green { color:#26a69a; } .val.red { color:#ef5350; } .val.yellow { color:#ffeb3b; }
.trade-card { background:#1a1a1a; border:1px solid #2a2a2a; border-radius:8px; margin-bottom:14px; overflow:hidden; }
.trade-header { display:flex; align-items:center; gap:14px; padding:12px 16px; border-bottom:1px solid #222; flex-wrap:wrap; cursor:pointer; user-select:none; }
.trade-header:hover { background:#1e1e1e; }
.th-sym  { font-size:1rem; font-weight:bold; color:#eee; min-width:80px; }
.th-type { font-size:0.9rem; font-weight:bold; padding:3px 10px; border-radius:4px; }
.th-type.buy  { background:#1b5e20; color:#a5d6a7; }
.th-type.sell { background:#b71c1c; color:#ef9a9a; }
.th-stat { display:flex; flex-direction:column; gap:1px; }
.th-stat .lbl { font-size:0.7rem; color:#666; }
.th-stat .val { font-size:0.85rem; font-weight:600; }
.th-pl { font-size:1.1rem; font-weight:bold; margin-left:auto; }
.th-toggle { color:#555; font-size:0.8rem; margin-left:8px; }
.trade-body { padding:14px 16px; display:none; }
.trade-body.open { display:block; }
.pbar-wrap { margin-bottom:14px; }
.pbar-label { display:flex; justify-content:space-between; font-size:0.72rem; color:#888; margin-bottom:4px; }
.pbar { height:8px; background:#2a2a2a; border-radius:4px; position:relative; overflow:visible; }
.pbar-fill-loss { position:absolute; left:0; top:0; height:100%; background:#ef5350; border-radius:4px; }
.pbar-fill-profit { position:absolute; top:0; height:100%; background:#26a69a; border-radius:4px; }
.pbar-cursor { position:absolute; top:-4px; width:3px; height:16px; background:#ffeb3b; border-radius:2px; transform:translateX(-50%); }
.modify-row { display:flex; gap:10px; align-items:flex-end; flex-wrap:wrap; margin-bottom:12px; }
.mf { display:flex; flex-direction:column; gap:3px; }
.mf label { font-size:0.72rem; color:#666; text-transform:uppercase; }
.mf input { background:#242424; color:#eee; border:1px solid #383838; padding:5px 8px; border-radius:4px; font-size:0.84rem; width:120px; }
.mf input:focus { outline:none; border-color:#1976d2; }
.modify-result { font-size:0.82rem; padding:5px 10px; border-radius:4px; display:none; margin-top:6px; }
.ok-msg  { background:#1b5e20; color:#a5d6a7; }
.err-msg { background:#b71c1c; color:#ef9a9a; }
.chart-section { margin-top:10px; }
.tf-tabs { display:flex; gap:6px; margin-bottom:6px; }
.tf-tab { background:#2a2a2a; color:#aaa; border:1px solid #383838; padding:3px 10px; border-radius:4px; cursor:pointer; font-size:0.78rem; }
.tf-tab.active { background:#37474f; color:#fff; border-color:#607d8b; }
.chart-wrap { border:1px solid #222; border-radius:4px; overflow:hidden; background:#111; }
.trade-chart-div { width:100%; height:390px; background:#111; }
.trade-chart-loading { height:390px; display:flex; align-items:center; justify-content:center; color:#555; font-size:0.85rem; }
.empty-state { text-align:center; padding:60px 20px; color:#555; font-size:0.9rem; }
.empty-state .icon { font-size:2.5rem; margin-bottom:10px; }
.section-title { font-size:0.85rem; font-weight:600; color:#888; text-transform:uppercase;
    letter-spacing:1px; margin:28px 0 12px; padding-bottom:6px; border-bottom:1px solid #222;
    display:flex; align-items:center; gap:10px; }
.section-title span { font-size:0.75rem; color:#555; font-weight:400; text-transform:none; letter-spacing:0; }
.hist-stats { display:flex; gap:16px; flex-wrap:wrap; margin-bottom:14px; }
.hist-stat { background:#1a1a1a; border:1px solid #2a2a2a; border-radius:5px; padding:8px 14px; font-size:0.82rem; }
.hist-stat .lbl { color:#555; font-size:0.7rem; margin-bottom:2px; text-transform:uppercase; }
.hist-stat .val { font-weight:bold; }
.hist-table { width:100%; border-collapse:collapse; font-size:0.8rem; }
.hist-table th { color:#555; font-weight:500; text-transform:uppercase; font-size:0.7rem;
    padding:6px 10px; border-bottom:1px solid #222; text-align:left; }
.hist-table td { padding:7px 10px; border-bottom:1px solid #1a1a1a; vertical-align:middle; }
.hist-table tr:hover td { background:#1a1a1a; }
.hist-table .type-buy  { color:#26a69a; font-weight:bold; }
.hist-table .type-sell { color:#ef5350; font-weight:bold; }
.hist-table .profit-pos { color:#26a69a; font-weight:bold; }
.hist-table .profit-neg { color:#ef5350; font-weight:bold; }
.hist-table .profit-zero { color:#888; }
.hist-load-more { background:#242424; border:1px solid #333; color:#888; padding:8px 20px;
    border-radius:4px; cursor:pointer; font-size:0.8rem; width:100%; margin-top:10px; }
.hist-load-more:hover { background:#2a2a2a; color:#aaa; }
a { color:inherit; text-decoration:none; }
a:hover { text-decoration:none; }

/* ── Perf per strategie ── */
.perf-table { width:100%; border-collapse:collapse; font-size:0.82rem; }
.perf-table th { color:#555; font-weight:400; padding:7px 10px; border-bottom:1px solid #222; text-align:left; cursor:default; }
.perf-table td { padding:7px 10px; border-bottom:1px solid #181818; }
.perf-row { cursor:pointer; transition:background 0.12s; }
.perf-row:hover { background:#1e1e1e; }
.perf-row.selected { background:#1a2a1a; }
.strat-stat-card { background:#1a1a1a; border:1px solid #2a2a2a; border-radius:6px; padding:10px 14px; min-width:110px; }
.strat-stat-card .sc-lbl { font-size:0.7rem; color:#555; margin-bottom:3px; }
.strat-stat-card .sc-val { font-size:1.1rem; font-weight:700; }
.detail-trades-table { width:100%; border-collapse:collapse; font-size:0.8rem; }
.detail-trades-table th { color:#444; font-weight:400; padding:6px 8px; border-bottom:1px solid #222; text-align:left; white-space:nowrap; }
.detail-trades-table td { padding:6px 8px; border-bottom:1px solid #161616; white-space:nowrap; }
.detail-trades-table tr:hover td { background:#1a1a1a; }
.badge-open  { background:#1a3a5a; color:#64b5f6; padding:2px 7px; border-radius:10px; font-size:0.7rem; }
.badge-closed { background:#1a1a1a; color:#555; padding:2px 7px; border-radius:10px; font-size:0.7rem; }
.badge-buy   { color:#26a69a; font-weight:600; }
.badge-sell  { color:#ef5350; font-weight:600; }
</style>
</head>
<body>
<div class="navbar">
    <div class="navbar-brand">📊 Trades Active</div>
    <div class="nav-links">
        <button class="btn btn-red" onclick="closeAllTrades()" id="btn-close-all">✕ Închide toate</button>
        <div id="close-result" style="font-size:0.82rem;padding:4px 10px;border-radius:4px;display:none"></div>
        <a href="/"            class="btn btn-grey">ChartVisualizer</a>
        <a href="/autotrader"  class="btn btn-grey">AutoTrader</a>
        <a href="/autoorders/" class="btn btn-sm" style="background:#1a1030;color:#ce93d8;border:1px solid #6a1b9a">⬡ AutoOrders</a>
        <a href="/account"     class="btn btn-grey">Cont</a>
    </div>
</div>

<div class="content">
    <div class="summary">
        <div class="sum-card"><div class="lbl">Pozitii deschise</div><div class="val" id="s-count">—</div></div>
        <div class="sum-card"><div class="lbl">Profit total</div><div class="val" id="s-profit">—</div></div>
        <div class="sum-card"><div class="lbl">Balance</div><div class="val" id="s-balance">—</div></div>
        <div class="sum-card"><div class="lbl">Equity</div><div class="val" id="s-equity">—</div></div>
        <div class="sum-card"><div class="lbl">Actualizat</div><div class="val yellow" id="s-time" style="font-size:0.85rem">—</div></div>
    </div>
    <div id="trades-container"></div>

    <div class="section-title">📊 Performanță per Strategie <span style="font-size:0.75rem;color:#555">(ultimele 91 zile + deschise)</span></div>
    <div id="perf-container"><div style="color:#555;font-size:0.82rem;padding:10px">Se incarca...</div></div>
    <div id="strat-detail-section" style="display:none">
        <div id="strat-detail-header" style="display:flex;align-items:center;gap:10px;margin:16px 0 8px;flex-wrap:wrap"></div>
        <div id="strat-stats-cards" style="display:flex;gap:8px;flex-wrap:wrap;margin-bottom:12px"></div>
        <div id="strat-trades-container"></div>
    </div>

    <div class="section-title">📋 Istoric Trades <span id="hist-subtitle"></span></div>
    <div class="hist-stats" id="hist-stats"></div>
    <div id="hist-container"><div style="color:#555;font-size:0.82rem;padding:10px">Se incarca istoricul...</div></div>
</div>

<script>
const TFS = ["M1","M5","M15","H1","H4"];
let tradeState = {};  // ticket -> {open, tf}

async function loadTrades() {
    try {
        const r = await fetch('/account_data');
        if (!r.ok) throw new Error(`HTTP ${r.status}`);
        const d = await r.json();
        if (!d.ok) {
            document.getElementById('trades-container').innerHTML =
                '<div class="empty-state"><div class="icon">⚠️</div>MT5 neconectat — <a href="/account" style="color:#1976d2">conecteaza-te</a></div>';
            return;
        }
        updateSummary(d);
        renderTrades(d);
    } catch(e) {
        console.warn('loadTrades error', e);
        document.getElementById('trades-container').innerHTML =
            '<div class="empty-state"><div class="icon">⚠️</div>Nu se pot incarca pozitiile: ' + String(e) + '</div>';
    }
}

function updateSummary(d) {
    const profit = d.positions.reduce((s, p) => s + p.profit, 0);
    const pc = profit >= 0 ? 'green' : 'red';
    document.getElementById('s-count').textContent   = d.positions.length;
    document.getElementById('s-profit').textContent  = (profit >= 0 ? '+' : '') + profit.toFixed(2) + ' ' + d.currency;
    document.getElementById('s-profit').className    = 'val ' + pc;
    document.getElementById('s-balance').textContent = d.balance.toFixed(2) + ' ' + d.currency;
    document.getElementById('s-equity').textContent  = d.equity.toFixed(2)  + ' ' + d.currency;
    document.getElementById('s-equity').className    = 'val ' + (d.equity >= d.balance ? 'green' : 'red');
    document.getElementById('s-time').textContent    = new Date().toLocaleTimeString();
}

let _ticketChartMap = {};
async function _fetchTicketChartMap() {
    try {
        const r = await fetch('/autotrader/ticket_chart');
        if (r.ok) _ticketChartMap = await r.json();
    } catch(e) {}
}
_fetchTicketChartMap();
setInterval(_fetchTicketChartMap, 15000);

function renderTrades(d) {
    const container = document.getElementById('trades-container');
    if (d.positions.length === 0) {
        container.innerHTML = '<div class="empty-state"><div class="icon">💤</div>Nicio pozitie deschisa momentan</div>';
        return;
    }
    const existing = {};
    container.querySelectorAll('.trade-card').forEach(c => { existing[c.dataset.ticket] = c; });

    // adauga/actualizeaza carduri
    d.positions.forEach(p => {
        const ticket = String(p.ticket);
        const state  = tradeState[ticket] || {open: false, tf: "M5"};
        const isBuy  = p.type === 'BUY';
        const plColor= p.profit >= 0 ? '#26a69a' : '#ef5350';
        const plSign = p.profit >= 0 ? '+' : '';

        // bara progres SL-TP
        let pbarHtml = '';
        if (p.sl && p.tp) {
            const total  = Math.abs(p.tp - p.sl);
            const moved  = isBuy ? (p.price_current - p.sl) : (p.sl - p.price_current);
            const pct    = Math.max(0, Math.min(100, (moved / total) * 100));
            const inProfit = isBuy ? p.price_current > p.price_open : p.price_current < p.price_open;
            const entryPct = Math.max(0, Math.min(100,
                isBuy ? ((p.price_open - p.sl) / total * 100) : ((p.sl - p.price_open) / total * 100)));
            pbarHtml = `
            <div class="pbar-wrap">
                <div class="pbar-label">
                    <span style="color:#ef5350">SL ${p.sl}</span>
                    <span style="color:#ffeb3b">Entry ${p.price_open}</span>
                    <span style="color:#26a69a">TP ${p.tp}</span>
                </div>
                <div class="pbar">
                    <div class="pbar-fill-loss"  style="width:${entryPct}%"></div>
                    <div class="pbar-fill-profit" style="left:${entryPct}%;width:${Math.max(0,pct-entryPct)}%"></div>
                    <div class="pbar-cursor" style="left:${pct}%"></div>
                </div>
            </div>`;
        }

        const c = (p.comment||'').toUpperCase();
        const STRAT_BADGES = {
            'COMBIN':   ['#0a1a2a','#42a5f5','Combined'],
            'EOB':      ['#2a0a3a','#9c27b0','EOB'],
            'SMC':      ['#3a2200','#ff9800','SMC'],
            'MACD':     ['#001a2a','#29b6f6','MACD'],
            'BOLL':     ['#1a002a','#ab47bc','Bollinger'],
            'SUPE':     ['#2a2000','#ffca28','Supertrend'],
            'LOND':     ['#2a0000','#ef5350','London'],
            'NY_B':     ['#2a0000','#ef5350','NY'],
            'RSI_':     ['#002a00','#66bb6a','RSI Div'],
            'ENGU':     ['#2a1500','#ffa726','Engulfing'],
            'ICHI':     ['#002a2a','#4dd0e1','Ichimoku'],
            'EMA_':     ['#002a1a','#80cbc4','EMA Cross'],
            'CANDL':    ['#1a2a00','#aed581','CandleForge'],
            'CLA':      ['#00302a','#26a69a','Classic'],
            'CV_':      ['#00302a','#26a69a','Classic'],
        };
        let stratBadge = '';
        for (const [key, [bg, col, label]] of Object.entries(STRAT_BADGES)) {
            if (c.includes(key)) {
                stratBadge = `<span style="background:${bg};color:${col};padding:2px 6px;border-radius:3px;font-size:0.7rem;font-weight:600">${label}</span>`;
                break;
            }
        }

        const openTime = p.open_time ? p.open_time.replace('T',' ') : '—';
        const cardHtml = `
        <div class="trade-header" onclick="toggleCard('${ticket}')">
            <span class="th-sym">${p.symbol}</span>
            <span class="th-type ${isBuy ? 'buy' : 'sell'}">${p.type}</span>
            ${stratBadge}
            <div class="th-stat"><span class="lbl">Deschis</span><span class="val" style="color:#607d8b;font-size:0.78rem">${openTime}</span></div>
            <div class="th-stat"><span class="lbl">Volume</span><span class="val">${p.volume}</span></div>
            <div class="th-stat"><span class="lbl">Entry</span><span class="val">${p.price_open}</span></div>
            <div class="th-stat"><span class="lbl">Curent</span><span class="val" style="color:#ffeb3b">${p.price_current}</span></div>
            <div class="th-stat"><span class="lbl">SL</span><span class="val" style="color:#ef5350">${p.sl || '—'}</span></div>
            <div class="th-stat"><span class="lbl">TP</span><span class="val" style="color:#26a69a">${p.tp || '—'}</span></div>
            <span class="th-pl" style="color:${plColor}">${plSign}${p.profit.toFixed(2)}$</span>
            <span class="th-toggle">${state.open ? '▲' : '▼'}</span>
        </div>
        <div class="trade-body ${state.open ? 'open' : ''}" id="body-${ticket}">
            ${pbarHtml}
            <div class="modify-row">
                <div class="mf"><label>Nou SL</label><input type="number" id="sl-${ticket}" value="${p.sl || ''}" step="0.00001" placeholder="Stop Loss"></div>
                <div class="mf"><label>Nou TP</label><input type="number" id="tp-${ticket}" value="${p.tp || ''}" step="0.00001" placeholder="Take Profit"></div>
                <button class="btn btn-teal" onclick="modifyTrade(${p.ticket})">✎ Modifica SL/TP</button>
                <button class="btn btn-red"  onclick="closeTrade(${p.ticket}, '${p.symbol}')">✕ Inchide</button>
                ${_ticketChartMap[String(p.ticket)] ? `<a href="/autotrader/analysis/${_ticketChartMap[String(p.ticket)]}" target="_blank"
                    style="display:inline-flex;align-items:center;gap:5px;padding:6px 13px;
                           background:#0d1a0d;border:1px solid #2a5a2a;border-radius:5px;
                           color:#4caf50;font-size:0.82rem;text-decoration:none;cursor:pointer;
                           transition:background .15s"
                    onmouseover="this.style.background='#1a2e1a'"
                    onmouseout="this.style.background='#0d1a0d'"
                    title="Analiza grafica la momentul deschiderii trade-ului">📊 Analiza</a>` : ''}
            </div>
            <div class="modify-result" id="res-${ticket}"></div>
            <div class="chart-section">
                <div class="tf-tabs" id="tabs-${ticket}">
                    ${TFS.map(tf => `<button class="tf-tab ${tf === state.tf ? 'active' : ''}" onclick="setTf('${ticket}','${p.symbol}',this,'${tf}')">${tf}</button>`).join('')}
                </div>
                <div class="chart-wrap">
                    <div id="chart-${ticket}" class="trade-chart-div">
                        <div class="trade-chart-loading">Se incarca graficul...</div>
                    </div>
                </div>
            </div>
        </div>`;

        if (existing[ticket]) {
            // actualizeaza doar header (fara sa distruga iframe)
            const card = existing[ticket];
            const oldHeader = card.querySelector('.trade-header');
            const newDiv = document.createElement('div');
            newDiv.innerHTML = cardHtml;
            card.replaceChild(newDiv.querySelector('.trade-header'), oldHeader);
            // actualizeaza bara progres daca panoul e deschis
            const pbarEl = card.querySelector('.pbar-wrap');
            const newPbar = newDiv.querySelector('.pbar-wrap');
            if (pbarEl && newPbar) pbarEl.outerHTML = newPbar.outerHTML;
            // actualizeaza butonul de analiza (apare dupa ce map-ul e incarcat)
            const oldMr = card.querySelector('.modify-row');
            const newMr = newDiv.querySelector('.modify-row');
            if (oldMr && newMr) oldMr.innerHTML = newMr.innerHTML;
        } else {
            const card = document.createElement('div');
            card.className = 'trade-card';
            card.dataset.ticket = ticket;
            card.innerHTML = cardHtml;
            container.appendChild(card);
        }
    });

    // sterge carduri pentru pozitii inchise
    const activeTickets = new Set(d.positions.map(p => String(p.ticket)));
    Object.keys(existing).forEach(t => {
        if (!activeTickets.has(t)) existing[t].remove();
    });
}

// ── Chart Plotly.react (fara iframe, fara reset zoom) ────────────────────────
const _tradeChartLoaded = {};  // ticket → true dupa primul render

function loadTradeChart(ticket, tf, forceReload, keepZoom) {
    const div = document.getElementById('chart-' + ticket);
    if (!div) return;
    const curTf = tf || (tradeState[String(ticket)] || {}).tf || 'M5';
    const key   = String(ticket) + '_' + curTf;

    if (!forceReload && _tradeChartLoaded[key] && div._hasChart) return;

    // Salveaza zoom DOAR daca TF-ul nu s-a schimbat (keepZoom=true)
    // La schimbarea TF-ului resetam zoom-ul ca sa vedem datele corecte
    let savedZoom = null;
    if (keepZoom && div._hasChart && div.layout && div.layout.xaxis) {
        const xr = div.layout.xaxis.range;
        if (Array.isArray(xr) && xr.length === 2) savedZoom = [xr[0], xr[1]];
    }

    fetch('/trade_chartjson/' + ticket + '?tf=' + curTf)
        .then(r => { if (!r.ok) throw new Error('HTTP ' + r.status); return r.json(); })
        .then(fig => {
            if (!fig || !Array.isArray(fig.data)) throw new Error('date invalide');
            // Sterge range-ul din layout — il restauram dupa render
            if (savedZoom && fig.layout && fig.layout.xaxis) {
                delete fig.layout.xaxis.range;
                delete fig.layout.xaxis.autorange;
            }
            // Plotly.react functioneaza pe div gol SAU pe div existent
            return Plotly.react(div, fig.data, fig.layout, {
                responsive:   true,
                displayModeBar: true,
                scrollZoom:   true,
                modeBarButtonsToRemove: ['lasso2d', 'select2d'],
            });
        })
        .then(() => {
            div._hasChart = true;
            _tradeChartLoaded[key] = true;
            if (savedZoom) Plotly.relayout(div, {'xaxis.range': savedZoom});
        })
        .catch(e => {
            console.warn('loadTradeChart error ticket=' + ticket + ' tf=' + curTf, e);
            div.innerHTML = '<div style="padding:20px;color:#ef5350;font-size:0.8rem">Eroare grafic: ' + e.message + '</div>';
        });
}

function toggleCard(ticket) {
    if (!tradeState[ticket]) tradeState[ticket] = {open: false, tf: "M5"};
    tradeState[ticket].open = !tradeState[ticket].open;
    const body = document.getElementById('body-' + ticket);
    body.classList.toggle('open', tradeState[ticket].open);
    // Incarca graficul la prima deschidere
    if (tradeState[ticket].open) loadTradeChart(ticket, tradeState[ticket].tf);
    const card = body.closest('.trade-card');
    const arrow = card.querySelector('.th-toggle');
    if (arrow) arrow.textContent = tradeState[ticket].open ? '▲' : '▼';
}

function setTf(ticket, symbol, btn, tf) {
    if (!tradeState[ticket]) tradeState[ticket] = {open: true, tf};
    tradeState[ticket].tf = tf;
    document.querySelectorAll(`#tabs-${ticket} .tf-tab`).forEach(b => b.classList.remove('active'));
    btn.classList.add('active');
    // Schimbare TF — forceReload=true, keepZoom=false (zoom nou pentru TF nou)
    Object.keys(_tradeChartLoaded).forEach(k => { if (k.startsWith(ticket + '_')) delete _tradeChartLoaded[k]; });
    loadTradeChart(ticket, tf, true, false);
}

async function modifyTrade(ticket) {
    const sl = parseFloat(document.getElementById('sl-' + ticket).value);
    const tp = parseFloat(document.getElementById('tp-' + ticket).value);
    if (isNaN(sl) || isNaN(tp)) { alert('Completeaza SL si TP'); return; }
    const r = await fetch('/modify_trade', {
        method: 'POST',
        headers: {'Content-Type':'application/json'},
        body: JSON.stringify({ticket, sl, tp})
    });
    const d = await r.json();
    const el = document.getElementById('res-' + ticket);
    el.style.display = 'block';
    el.className = 'modify-result ' + (d.ok ? 'ok-msg' : 'err-msg');
    el.textContent = d.message;
    setTimeout(() => el.style.display = 'none', 3000);
    if (d.ok) loadTrades();
}

async function closeTrade(ticket, symbol) {
    if (!confirm('Inchizi ' + symbol + ' #' + ticket + '?')) return;
    const r = await fetch('/close_position?ticket=' + ticket, {method:'POST'});
    const d = await r.json();
    if (d.ok) { delete tradeState[String(ticket)]; loadTrades(); }
    else alert('Eroare: ' + d.message);
}

// ── Istoric ───────────────────────────────────────────────────────────────────
let allHistory = [];

async function loadHistory() {
    try {
        const r = await fetch('/history_data');
        if (!r.ok) throw new Error(`HTTP ${r.status}`);
        const d = await r.json();
        if (!d.ok) {
            document.getElementById('hist-container').innerHTML =
                '<div style="color:#555;font-size:0.82rem;padding:10px">Nu s-a putut incarca istoricul: ' + (d.msg||'') + '</div>';
            return;
        }
        allHistory = d.history || [];
        // stats
        const net = d.total_net;
        const wr  = d.wins + d.losses > 0 ? Math.round(d.wins / (d.wins + d.losses) * 100) : 0;
        document.getElementById('hist-subtitle').textContent = `ultimele 90 zile · ${d.count} trades`;
        document.getElementById('hist-stats').innerHTML = `
            <div class="hist-stat"><div class="lbl">Net Total</div><div class="val ${net>=0?'profit-pos':'profit-neg'}">${net>=0?'+':''}${net.toFixed(2)} $</div></div>
            <div class="hist-stat"><div class="lbl">Castigate</div><div class="val profit-pos">${d.wins}</div></div>
            <div class="hist-stat"><div class="lbl">Pierdute</div><div class="val profit-neg">${d.losses}</div></div>
            <div class="hist-stat"><div class="lbl">Win Rate</div><div class="val ${wr>=50?'profit-pos':'profit-neg'}">${wr}%</div></div>
            <div class="hist-stat"><div class="lbl">Total</div><div class="val">${d.count}</div></div>
        `;
        renderHistory();
    } catch(e) {
        console.warn('history err', e);
        document.getElementById('hist-container').innerHTML =
            '<div style="color:#555;font-size:0.82rem;padding:10px">Eroare la incarcare istoric: ' + String(e) + '</div>';
    }
}

function renderHistory() {
    const slice = allHistory;
    let html = '<table class="hist-table"><thead><tr>';
    ['Data inchidere','Simbol','Tip','Volum','Entry','Close','Profit','Comision','Net','Strategie','📸'].forEach(h => {
        html += `<th>${h}</th>`;
    });
    html += '</tr></thead><tbody>';
    slice.forEach(h => {
        const netCls  = h.net > 0 ? 'profit-pos' : h.net < 0 ? 'profit-neg' : 'profit-zero';
        const profCls = h.profit > 0 ? 'profit-pos' : h.profit < 0 ? 'profit-neg' : 'profit-zero';
        const hasSnap = snapshotTickets.has(h.ticket);
        const snapBtn = hasSnap
            ? `<a href="/snapshot/${h.ticket}" target="_blank" style="background:#1a2a1a;border:1px solid #2e4a2e;color:#66bb6a;padding:3px 8px;border-radius:3px;font-size:0.75rem;text-decoration:none;white-space:nowrap">📸 Vezi</a>`
            : `<span style="color:#333;font-size:0.75rem">—</span>`;
        const c = (h.comment||'').toUpperCase();
        const HIST_BADGES = {
            'COMBIN':   ['#0a1a2a','#42a5f5','Combined'],
            'EOB':      ['#2a0a3a','#9c27b0','EOB'],
            'SMC':      ['#3a2200','#ff9800','SMC'],
            'MACD':     ['#001a2a','#29b6f6','MACD'],
            'BOLL':     ['#1a002a','#ab47bc','Boll'],
            'SUPE':     ['#2a2000','#ffca28','Super'],
            'LOND':     ['#2a0000','#ef5350','London'],
            'NY_B':     ['#2a0000','#ef5350','NY'],
            'RSI_':     ['#002a00','#66bb6a','RSI'],
            'ENGU':     ['#2a1500','#ffa726','Engulf'],
            'ICHI':     ['#002a2a','#4dd0e1','Ichi'],
            'EMA_':     ['#002a1a','#80cbc4','EMA'],
            'CANDL':    ['#1a2a00','#aed581','CandleForge'],
            'CLA':      ['#00302a','#26a69a','Classic'],
            'CV_':      ['#00302a','#26a69a','Classic'],
        };
        let stratBadge = `<span style="color:#444;font-size:0.72rem">${h.comment||'—'}</span>`;
        for (const [key, [bg, col, label]] of Object.entries(HIST_BADGES)) {
            if (c.includes(key)) {
                stratBadge = `<span style="background:${bg};color:${col};padding:2px 7px;border-radius:3px;font-size:0.72rem;font-weight:600">${label}</span>`;
                break;
            }
        }
        html += `<tr>
            <td style="color:#666;white-space:nowrap;font-size:0.8rem">${h.close_time || h.open_time}</td>
            <td style="font-weight:bold">${h.symbol}</td>
            <td class="type-${h.type.toLowerCase()}">${h.type}</td>
            <td style="color:#aaa">${h.volume}</td>
            <td style="color:#aaa">${h.price_open}</td>
            <td style="color:#aaa">${h.price_close ?? '—'}</td>
            <td class="${profCls}">${h.profit>=0?'+':''}${h.profit.toFixed(2)}</td>
            <td style="color:#666">${h.commission.toFixed(2)}</td>
            <td class="${netCls}" style="font-size:0.85rem">${h.net>=0?'+':''}${h.net.toFixed(2)}</td>
            <td>${stratBadge}</td>
            <td>${snapBtn}</td>
        </tr>`;
    });
    html += '</tbody></table>';
    document.getElementById('hist-container').innerHTML = html;
}

// ── Snapshots ─────────────────────────────────────────────────────────────────
let snapshotTickets = new Set();

async function loadSnapshots() {
    try {
        const r = await fetch('/snapshots_list');
        const d = await r.json();
        snapshotTickets = new Set(d.tickets || []);
    } catch(e) {}
}

// reload history la 30s (nu are nevoie de 1s)
loadSnapshots();
loadHistory();
loadPerf();
setInterval(loadHistory, 30000);
setInterval(loadSnapshots, 30000);
setInterval(loadPerf, 60000);

// ── Performanta per strategie ─────────────────────────────────────────────────
let _selectedStrat = null;

function pnlStr(v) {
    return (v > 0 ? '+' : '') + v.toFixed(2) + '$';
}
function pnlColor(v) {
    return v >= 0 ? '#26a69a' : '#ef5350';
}

async function loadPerf() {
    try {
        const r    = await fetch('/trades/perf');
        const data = await r.json();
        if (!Array.isArray(data)) return;
        const el = document.getElementById('perf-container');
        if (!data.length) {
            el.innerHTML = '<div style="color:#555;padding:10px;font-size:0.82rem">Nicio tranzactie gasita cu tag CV_</div>';
            return;
        }

        let html = `<table class="perf-table"><thead>
        <tr>
            <th>Strategie</th>
            <th style="text-align:center">Deschise</th>
            <th style="text-align:right">PnL Deschis</th>
            <th style="text-align:center">Închise</th>
            <th style="text-align:right">PnL Închis</th>
            <th style="text-align:center">Câștigate</th>
            <th style="text-align:center">Win Rate</th>
            <th style="text-align:right">Best</th>
            <th style="text-align:right">Worst</th>
            <th style="text-align:center">Durata medie</th>
            <th style="text-align:right">TOTAL</th>
        </tr></thead><tbody>`;

        for (const s of data) {
            const tc  = pnlColor(s.total_pnl);
            const oc  = pnlColor(s.open_pnl);
            const cc  = pnlColor(s.closed_pnl);
            const wr  = s.win_rate > 0 ? s.win_rate + '%' : '—';
            const wrc = s.win_rate >= 55 ? '#26a69a' : s.win_rate >= 40 ? '#ffeb3b' : s.win_rate > 0 ? '#ef5350' : '#555';
            const isSel = (_selectedStrat === s.strategy) ? ' selected' : '';
            html += `<tr class="perf-row${isSel}" onclick="selectStrat('${s.strategy.replace(/'/g,"\\'")}')">
                <td style="font-weight:600">${s.strategy}</td>
                <td style="text-align:center;color:#aaa">${s.open_count || '—'}</td>
                <td style="text-align:right;color:${oc}">${pnlStr(s.open_pnl)}</td>
                <td style="text-align:center;color:#aaa">${s.closed_count || '—'}</td>
                <td style="text-align:right;color:${cc}">${pnlStr(s.closed_pnl)}</td>
                <td style="text-align:center;color:#aaa">${s.wins > 0 ? s.wins + ' / ' + (s.wins+s.losses) : '—'}</td>
                <td style="text-align:center;color:${wrc};font-weight:600">${wr}</td>
                <td style="text-align:right;color:#26a69a">${s.best_trade != null ? '+'+s.best_trade+'$' : '—'}</td>
                <td style="text-align:right;color:#ef5350">${s.worst_trade != null ? s.worst_trade+'$' : '—'}</td>
                <td style="text-align:center;color:#777">${s.avg_duration_str || '—'}</td>
                <td style="text-align:right;color:${tc};font-weight:700">${pnlStr(s.total_pnl)}</td>
            </tr>`;
        }

        const totAll    = data.reduce((a,s)=>a+s.total_pnl, 0);
        const totOpen   = data.reduce((a,s)=>a+s.open_pnl,  0);
        const totClosed = data.reduce((a,s)=>a+s.closed_pnl,0);
        const totWins   = data.reduce((a,s)=>a+s.wins,   0);
        const totLosses = data.reduce((a,s)=>a+s.losses, 0);
        const totWR     = (totWins+totLosses) > 0 ? (totWins/(totWins+totLosses)*100).toFixed(1)+'%' : '—';
        html += `<tr style="border-top:2px solid #2a2a2a;background:#161616">
            <td style="font-weight:700;color:#aaa">TOTAL</td>
            <td></td>
            <td style="text-align:right;color:${pnlColor(totOpen)}">${pnlStr(totOpen)}</td>
            <td></td>
            <td style="text-align:right;color:${pnlColor(totClosed)}">${pnlStr(totClosed)}</td>
            <td style="text-align:center;color:#aaa">${totWins} / ${totWins+totLosses}</td>
            <td style="text-align:center;color:${totWR!='—'&&parseFloat(totWR)>=50?'#26a69a':'#ef5350'};font-weight:600">${totWR}</td>
            <td></td><td></td><td></td>
            <td style="text-align:right;color:${pnlColor(totAll)};font-weight:700;font-size:0.92rem">${pnlStr(totAll)}</td>
        </tr>`;

        el.innerHTML = html + '</tbody></table>';

        // Daca era o strategie selectata, reincarc detaliile
        if (_selectedStrat) selectStrat(_selectedStrat, false);

    } catch(e) {
        document.getElementById('perf-container').innerHTML =
            `<div style="color:#555;padding:10px;font-size:0.82rem">Eroare: ${e}</div>`;
    }
}

async function selectStrat(name, scroll=true) {
    _selectedStrat = name;
    // highlight row
    document.querySelectorAll('.perf-row').forEach(r => {
        r.classList.toggle('selected', r.textContent.trimStart().startsWith(name));
    });

    const section = document.getElementById('strat-detail-section');
    const header  = document.getElementById('strat-detail-header');
    const cards   = document.getElementById('strat-stats-cards');
    const tbody   = document.getElementById('strat-trades-container');

    header.innerHTML = `<span style="font-size:0.95rem;font-weight:700;color:#eee">${name}</span>
        <span style="font-size:0.78rem;color:#555">— se incarca...</span>`;
    section.style.display = 'block';
    if (scroll) section.scrollIntoView({behavior:'smooth', block:'start'});

    try {
        const r = await fetch('/trades/perf/detail?strategy=' + encodeURIComponent(name));
        const d = await r.json();
        if (d.error) { header.innerHTML = `<span style="color:#ef5350">${d.error}</span>`; return; }

        const st = d.stats || {};
        const trades = d.trades || [];

        // Header
        header.innerHTML = `
            <span style="font-size:0.95rem;font-weight:700;color:#eee">${name}</span>
            <span style="font-size:0.78rem;color:#555">${trades.length} trade-uri totale</span>
            <button onclick="_selectedStrat=null;document.getElementById('strat-detail-section').style.display='none'"
                style="margin-left:auto;background:none;border:1px solid #333;color:#666;padding:3px 10px;border-radius:4px;cursor:pointer;font-size:0.75rem">
                ✕ Inchide
            </button>`;

        // Stats cards
        const wr = st.win_rate || 0;
        const wrc = wr >= 55 ? '#26a69a' : wr >= 40 ? '#ffeb3b' : wr > 0 ? '#ef5350' : '#555';
        cards.innerHTML = `
            <div class="strat-stat-card"><div class="sc-lbl">Total PnL</div>
                <div class="sc-val" style="color:${pnlColor(st.total_pnl||0)}">${pnlStr(st.total_pnl||0)}</div></div>
            <div class="strat-stat-card"><div class="sc-lbl">PnL Închis</div>
                <div class="sc-val" style="color:${pnlColor(st.closed_pnl||0)}">${pnlStr(st.closed_pnl||0)}</div></div>
            <div class="strat-stat-card"><div class="sc-lbl">PnL Deschis</div>
                <div class="sc-val" style="color:${pnlColor(st.open_pnl||0)}">${pnlStr(st.open_pnl||0)}</div></div>
            <div class="strat-stat-card"><div class="sc-lbl">Win Rate</div>
                <div class="sc-val" style="color:${wrc}">${wr > 0 ? wr+'%' : '—'}</div></div>
            <div class="strat-stat-card"><div class="sc-lbl">Câștigate / Total</div>
                <div class="sc-val" style="color:#aaa">${st.wins||0} / ${(st.wins||0)+(st.losses||0)}</div></div>
            <div class="strat-stat-card"><div class="sc-lbl">Best trade</div>
                <div class="sc-val" style="color:#26a69a">${st.best_trade!=null ? '+'+st.best_trade+'$' : '—'}</div></div>
            <div class="strat-stat-card"><div class="sc-lbl">Worst trade</div>
                <div class="sc-val" style="color:#ef5350">${st.worst_trade!=null ? st.worst_trade+'$' : '—'}</div></div>
            <div class="strat-stat-card"><div class="sc-lbl">Durata medie</div>
                <div class="sc-val" style="color:#aaa">${st.avg_duration_str||'—'}</div></div>
        `;

        // Tabel trade-uri
        if (!trades.length) {
            tbody.innerHTML = '<div style="color:#555;padding:10px;font-size:0.82rem">Niciun trade gasit.</div>';
            return;
        }

        let th = `<table class="detail-trades-table"><thead><tr>
            <th>Status</th><th>Symbol</th><th>Dir</th><th>Volum</th>
            <th>Intrare</th><th>Iesire</th><th>Deschis la</th><th>Inchis la</th>
            <th>Durata</th><th>Profit brut</th><th>Comis.</th><th>Swap</th>
            <th style="text-align:right">NET</th>
        </tr></thead><tbody>`;

        for (const t of trades) {
            const nc  = pnlColor(t.net);
            const dir = t.direction === 'BUY'
                ? '<span class="badge-buy">▲ BUY</span>'
                : '<span class="badge-sell">▼ SELL</span>';
            const badge = t.status === 'OPEN'
                ? '<span class="badge-open">DESCHIS</span>'
                : '<span class="badge-closed">INCHIS</span>';
            const netStr = (t.net > 0 ? '+' : '') + t.net.toFixed(2) + '$';
            th += `<tr>
                <td>${badge}</td>
                <td style="font-weight:600">${t.symbol}</td>
                <td>${dir}</td>
                <td style="color:#777">${t.volume}</td>
                <td style="color:#aaa">${t.open_price}</td>
                <td style="color:#aaa">${t.close_price}</td>
                <td style="color:#666;font-size:0.74rem">${t.open_time}</td>
                <td style="color:#666;font-size:0.74rem">${t.close_time}</td>
                <td style="color:#666">${t.duration}</td>
                <td style="color:${pnlColor(t.profit)}">${(t.profit>0?'+':'')+t.profit.toFixed(2)}$</td>
                <td style="color:#555">${t.commission.toFixed(2)}$</td>
                <td style="color:#555">${t.swap.toFixed(2)}$</td>
                <td style="text-align:right;font-weight:700;color:${nc}">${netStr}</td>
            </tr>`;
        }
        tbody.innerHTML = th + '</tbody></table>';

    } catch(e) {
        header.innerHTML = `<span style="color:#ef5350">Eroare: ${e}</span>`;
    }
}

// ── Inchide toate ────────────────────────────────────────────────────────────
async function closeAllTrades() {
    if (!confirm("Inchizi TOATE pozitiile deschise?")) return;
    const btn = document.getElementById("btn-close-all");
    btn.disabled = true; btn.textContent = "⏳ Se inchid...";
    try {
        const r = await fetch("/close_all_trades", {method:"POST"});
        const d = await r.json();
        const res = document.getElementById("close-result");
        res.style.display = "inline-block";
        res.style.background = d.ok ? "#b71c1c" : "#333";
        res.style.color = "#fff";
        res.textContent = d.ok ? `Inchis ${d.closed} pozitii` : (d.message || "Eroare");
        setTimeout(() => { res.style.display="none"; }, 4000);
        if (d.ok) loadTrades();
    } catch(e) {}
    btn.disabled = false; btn.textContent = "✕ Închide toate";
}

// ── Live refresh ─────────────────────────────────────────────────────────────
// refresh date la 1s, graficele la 10s
setInterval(loadTrades, 1000);
// Refresh grafice la 15s — keepZoom=true ca sa nu se piarda zoom-ul utilizatorului
setInterval(() => {
    Object.keys(tradeState).forEach(ticket => {
        if (tradeState[ticket] && tradeState[ticket].open) {
            const div = document.getElementById('chart-' + ticket);
            if (div && div._hasChart) {
                loadTradeChart(ticket, tradeState[ticket].tf || 'M5', true, true);
            }
        }
    });
}, 15000);

loadTrades();
</script>
</body></html>"""


def _fetch_all_news():
    """Descarca TOATE stirile (nu doar High) din FF XML. Cache comun 4 ore."""
    import urllib.request, xml.etree.ElementTree as ET
    from datetime import datetime, timezone

    with _news_lock:
        now = datetime.now(timezone.utc)
        cached = _news_cache.get("all_events")
        fetched_at = _news_cache.get("fetched_at")
        if cached is not None and fetched_at and (now - fetched_at).total_seconds() < 14400:
            return cached, None

    urls = [
        "https://nfs.faireconomy.media/ff_calendar_thisweek.xml",
        "https://cdn-nfs.faireconomy.media/ff_calendar_thisweek.xml",
    ]
    data = None
    last_err = None
    for url in urls:
        try:
            req = urllib.request.Request(url, headers={
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
                "Accept": "application/xml,text/xml,*/*",
                "Cache-Control": "no-cache",
            })
            with urllib.request.urlopen(req, timeout=12) as resp:
                data = resp.read()
            break
        except Exception as e:
            last_err = str(e)
            log.warning(f"News URL {url} eroare: {e}")

    if not data:
        return _news_cache.get("all_events", []), last_err

    now = datetime.now(timezone.utc)
    events = []
    try:
        root = ET.fromstring(data)
        for ev in root.findall("event"):
            impact   = ev.findtext("impact",   "").strip().lower()
            title    = ev.findtext("title",    "")
            country  = ev.findtext("country",  "")
            date_str = ev.findtext("date",     "").strip()
            time_str = ev.findtext("time",     "").strip()
            forecast = ev.findtext("forecast", "")
            previous = ev.findtext("previous", "")
            actual   = ev.findtext("actual",   "")
            dt = None
            # ForexFactory e in ET (America/New_York) — convertim la UTC
            try:
                from zoneinfo import ZoneInfo as _ZI
                _ET = _ZI("America/New_York")
                def _to_utc(dt_naive): return dt_naive.replace(tzinfo=_ET).astimezone(timezone.utc)
            except Exception:
                from datetime import timedelta as _td
                def _to_utc(dt_naive): return (dt_naive + _td(hours=4)).replace(tzinfo=timezone.utc)
            for fmt in ["%m-%d-%Y %I:%M%p", "%m-%d-%Y %H:%M", "%Y-%m-%d %H:%M", "%m/%d/%Y %I:%M%p"]:
                try:
                    dt = _to_utc(datetime.strptime(f"{date_str} {time_str}", fmt))
                    break
                except:
                    pass
            if dt is None:
                try:
                    dt = _to_utc(datetime.strptime(date_str, "%m-%d-%Y"))
                except:
                    continue
            diff_min = (dt - now).total_seconds() / 60
            # time_et = ora FF (ET), time_ro = ora Romaniei (Europe/Bucharest)
            try:
                from zoneinfo import ZoneInfo as _ZI2
                dt_et = dt.astimezone(_ZI2("America/New_York"))
                dt_ro = dt.astimezone(_ZI2("Europe/Bucharest"))
                time_et_str  = dt_et.strftime("%H:%M")
                date_et_str  = dt_et.strftime("%Y-%m-%d")
                time_ro_str  = dt_ro.strftime("%H:%M")
            except Exception:
                from datetime import timedelta as _td2
                dt_et = dt - _td2(hours=4)
                dt_ro = dt + _td2(hours=3)
                time_et_str  = dt_et.strftime("%H:%M")
                date_et_str  = dt_et.strftime("%Y-%m-%d")
                time_ro_str  = dt_ro.strftime("%H:%M")
            events.append({
                "title":    title,
                "country":  country,
                "impact":   impact,
                "time_utc": dt.strftime("%Y-%m-%d %H:%M"),
                "time_et":  f"{date_et_str} {time_et_str}",
                "time_ro":  time_ro_str,
                "forecast": forecast,
                "previous": previous,
                "actual":   actual,
                "in_min":   round(diff_min),
                "past":     diff_min < 0,
            })
    except Exception as e:
        log.error(f"News XML parse error: {e}")
        return _news_cache.get("all_events", []), str(e)

    events.sort(key=lambda x: x["in_min"])
    with _news_lock:
        _news_cache["all_events"] = events
        _news_cache["fetched_at"] = datetime.now(timezone.utc)
    log.info(f"News cache actualizat: {len(events)} evenimente")
    return events, None


@app.route("/news_data")
@login_required
def news_data():
    events, err = _fetch_all_news()
    if not events and err:
        return Response(json.dumps({"ok": False, "error": f"ForexFactory indisponibil: {err}. Incearca din nou in 5 minute."}), mimetype="application/json")
    return Response(json.dumps({"ok": True, "events": events, "count": len(events)}), mimetype="application/json")


NEWS_HTML = """<!DOCTYPE html>
<html><head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Calendar Stiri Forex</title>
<style>
* { box-sizing:border-box; margin:0; padding:0; }
body { background:#111; color:#eee; font-family:'Segoe UI',sans-serif; }
.navbar { background:#161616; border-bottom:1px solid #2a2a2a; padding:0 20px; height:44px; display:flex; align-items:center; justify-content:space-between; }
.navbar-brand { font-size:1rem; font-weight:600; color:#eee; }
.nav-links { display:flex; gap:8px; }
.btn { background:#1976d2; color:#fff; border:none; padding:6px 14px; border-radius:4px; cursor:pointer; font-size:0.82rem; text-decoration:none; display:inline-flex; align-items:center; }
.btn:hover { background:#1565c0; }
.btn-grey { background:#333; color:#bbb; } .btn-grey:hover { background:#444; }
.content { padding:16px 20px; max-width:1100px; }
.filter-bar { display:flex; gap:8px; margin-bottom:16px; flex-wrap:wrap; align-items:center; }
.filter-btn { background:#2a2a2a; border:1px solid #383838; color:#aaa; padding:5px 14px; border-radius:4px; cursor:pointer; font-size:0.82rem; }
.filter-btn.active { color:#fff; }
.filter-btn.high.active  { background:#b71c1c; border-color:#ef5350; }
.filter-btn.medium.active { background:#e65100; border-color:#ff9800; }
.filter-btn.low.active   { background:#1b5e20; border-color:#26a69a; }
.filter-btn.all.active   { background:#1976d2; border-color:#42a5f5; }
.section-day { margin-bottom:20px; }
.day-header { font-size:0.8rem; color:#666; text-transform:uppercase; letter-spacing:1px; padding:6px 0; border-bottom:1px solid #222; margin-bottom:8px; }
.news-row {
    display:grid;
    grid-template-columns: 60px 60px 40px 1fr 90px 90px 90px 100px;
    gap:8px; align-items:center;
    padding:8px 10px; border-radius:4px;
    margin-bottom:4px; font-size:0.83rem;
    border-left:3px solid transparent;
    transition: background 0.15s;
}
.news-row:hover { background:#1a1a1a; }
.news-row.high   { border-left-color:#ef5350; background:#1a1212; }
.news-row.medium { border-left-color:#ff9800; background:#1a1510; }
.news-row.low    { border-left-color:#26a69a; background:#111a18; }
.news-row.past   { opacity:0.4; }
.news-row.upcoming { animation: glow 1.5s infinite alternate; }
@keyframes glow { from { background:#1a1212; } to { background:#2a1515; } }
.impact-dot { width:10px; height:10px; border-radius:50%; display:inline-block; }
.impact-dot.high   { background:#ef5350; box-shadow:0 0 6px #ef5350; }
.impact-dot.medium { background:#ff9800; }
.impact-dot.low    { background:#26a69a; }
.col-time    { color:#888; font-size:0.78rem; }
.col-country { font-size:0.75rem; background:#2a2a2a; padding:2px 6px; border-radius:3px; text-align:center; }
.col-title   { font-weight:500; color:#ddd; }
.col-val     { text-align:right; font-size:0.8rem; color:#aaa; }
.col-actual.positive { color:#26a69a; font-weight:bold; }
.col-actual.negative { color:#ef5350; font-weight:bold; }
.badge-soon { background:#b71c1c; color:#ef9a9a; font-size:0.7rem; padding:2px 7px; border-radius:10px; margin-left:6px; animation:pulse 1s infinite; }
@keyframes pulse { 0%,100%{opacity:1} 50%{opacity:0.5} }
.col-header { color:#555; font-size:0.72rem; text-transform:uppercase; }
.headers { display:grid; grid-template-columns:60px 60px 40px 1fr 90px 90px 90px 100px; gap:8px; padding:4px 10px; margin-bottom:6px; }
.col-time-ro { color:#4fc3f7; font-size:0.78rem; font-weight:500; }
.update-time { font-size:0.72rem; color:#555; margin-left:auto; }
.empty { color:#555; padding:40px; text-align:center; }
</style>
</head><body>
<div class="navbar">
    <div class="navbar-brand">📰 Calendar Stiri Forex</div>
    <div class="nav-links">
        <a href="/"            class="btn btn-grey">ChartVisualizer</a>
        <a href="/autotrader"  class="btn btn-grey">AutoTrader</a>
        <a href="/trades"      class="btn btn-grey">Trades</a>
        <a href="/autoorders/" class="btn btn-sm" style="background:#1a1030;color:#ce93d8;border:1px solid #6a1b9a">⬡ AutoOrders</a>
    </div>
</div>
<div class="content">
    <div class="filter-bar">
        <button class="filter-btn high active"   onclick="setFilter('high')"  >🔴 Impact Mare</button>
        <button class="filter-btn medium active" onclick="setFilter('medium')">🟡 Impact Mediu</button>
        <button class="filter-btn low"           onclick="setFilter('low')"   >🟢 Impact Mic</button>
        <button class="filter-btn all active"    onclick="setFilter('all')"   >Toate</button>
        <span class="update-time" id="update-time"></span>
    </div>
    <div class="headers">
        <span class="col-header" title="US Eastern Time (New York)">Ora (ET)</span>
        <span class="col-header" style="color:#4fc3f7" title="Ora Romaniei (Europe/Bucharest)">Ora (RO)</span>
        <span class="col-header">Tara</span>
        <span class="col-header">Eveniment</span>
        <span class="col-header" style="text-align:right">Prognoza</span>
        <span class="col-header" style="text-align:right">Anterior</span>
        <span class="col-header" style="text-align:right">Actual</span>
        <span class="col-header" style="text-align:right">Status</span>
    </div>
    <div id="news-container"><div class="empty">Se incarca...</div></div>
</div>
<script>
let activeFilters = new Set(["high","medium","low"]);
let allEvents = [];

function setFilter(f) {
    if (f === "all") {
        if (activeFilters.size === 3) activeFilters.clear();
        else { activeFilters = new Set(["high","medium","low"]); }
    } else {
        if (activeFilters.has(f)) activeFilters.delete(f);
        else activeFilters.add(f);
    }
    document.querySelectorAll(".filter-btn").forEach(b => {
        const bf = b.className.match(/high|medium|low|all/)?.[0];
        if (bf === "all") b.classList.toggle("active", activeFilters.size === 3);
        else if (bf) b.classList.toggle("active", activeFilters.has(bf));
    });
    renderNews(allEvents);
}

async function loadNews() {
    try {
        const r = await fetch("/news_data");
        const d = await r.json();
        if (!d.ok) {
            document.getElementById("news-container").innerHTML =
                `<div class="empty">⚠ ${d.error}<br><br><button onclick="loadNews()" style="background:#1976d2;color:#fff;border:none;padding:8px 16px;border-radius:4px;cursor:pointer">↻ Incearca din nou</button></div>`;
            return;
        }
        allEvents = d.events;
        renderNews(allEvents);
        document.getElementById("update-time").textContent = `Actualizat: ${new Date().toLocaleTimeString()} · ${d.count} evenimente`;
    } catch(e) { console.warn(e); }
}

function renderNews(events) {
    const filtered = events.filter(e => activeFilters.has(e.impact));
    if (!filtered.length) {
        document.getElementById("news-container").innerHTML = '<div class="empty">Niciun eveniment pentru filtrele selectate</div>';
        return;
    }
    // grupeaza pe zi (dupa data ET — asa cum apare pe ForexFactory)
    const byDay = {};
    filtered.forEach(e => {
        const day = (e.time_et || e.time_utc).split(" ")[0];
        if (!byDay[day]) byDay[day] = [];
        byDay[day].push(e);
    });
    let html = "";
    Object.keys(byDay).sort().forEach(day => {
        const d = new Date(day + "T00:00:00Z");
        const dayLabel = d.toLocaleDateString("ro-RO", {weekday:"long", day:"numeric", month:"long", timeZone:"UTC"});
        html += `<div class="section-day"><div class="day-header">${dayLabel}</div>`;
        byDay[day].forEach(e => {
            const soon   = !e.past && e.in_min >= 0 && e.in_min <= 5;
            const actualClass = e.actual
                ? (parseFloat(e.actual) >= parseFloat(e.forecast || e.previous || 0) ? "positive" : "negative")
                : "";
            const statusBadge = e.past
                ? `<span style="color:#555;font-size:0.75rem">Trecut</span>`
                : soon
                ? `<span class="badge-soon">⚠ in ${e.in_min}min</span>`
                : `<span style="color:#666;font-size:0.75rem">in ${e.in_min}min</span>`;
            html += `
            <div class="news-row ${e.impact} ${e.past ? "past" : ""} ${soon ? "upcoming" : ""}">
                <span class="col-time">${(e.time_et || e.time_utc).split(" ")[1]}</span>
                <span class="col-time-ro">${e.time_ro || "—"}</span>
                <span class="col-country">${e.country}</span>
                <span class="col-title">
                    <span class="impact-dot ${e.impact}"></span>
                    &nbsp;${e.title}
                </span>
                <span class="col-val">${e.forecast || "—"}</span>
                <span class="col-val">${e.previous || "—"}</span>
                <span class="col-val col-actual ${actualClass}">${e.actual || "—"}</span>
                <span class="col-val">${statusBadge}</span>
            </div>`;
        });
        html += "</div>";
    });
    document.getElementById("news-container").innerHTML = html;
}

loadNews();
setInterval(loadNews, 60000);  // refresh la 1 minut
</script>
</body></html>"""


@app.route("/news")
@login_required
def news_page():
    return NEWS_HTML


@app.route("/snapshot/<int:ticket>")
@login_required
def snapshot_view(ticket):
    path = os.path.join(SNAPSHOTS_DIR, f"{ticket}.html")
    if not os.path.exists(path):
        return "<html><body style='background:#111;color:#555;font-family:sans-serif;padding:40px;text-align:center'><h2>Snapshot negasit pentru #{}</h2><p>Trade-ul a fost executat inainte de implementarea snapshot-urilor.</p></body></html>".format(ticket), 404
    return send_file(path, mimetype="text/html")


@app.route("/snapshots_list")
@login_required
def snapshots_list():
    """Returneaza lista de ticket-uri care au snapshot."""
    try:
        tickets = set()
        for f in os.listdir(SNAPSHOTS_DIR):
            if f.endswith(".json"):
                try:
                    tickets.add(int(f.replace(".json", "")))
                except:
                    pass
        return Response(json.dumps({"tickets": list(tickets)}), mimetype="application/json")
    except Exception as e:
        return Response(json.dumps({"tickets": []}), mimetype="application/json")


@app.route("/debug_deals")
@login_required
def debug_deals():
    """Dump raw MT5 deals pentru ultimele 2 zile — ajuta la diagnosticare."""
    if not MT5_AVAILABLE or mt5 is None:
        return Response(json.dumps({"ok": False}), mimetype="application/json")
    try:
        from datetime import datetime, timedelta, timezone
        date_to   = datetime.now(timezone.utc)
        date_from = date_to - timedelta(days=2)
        deals = mt5.history_deals_get(date_from, date_to) or []
        rows = []
        for d in deals:
            rows.append({
                "ticket": d.ticket, "position_id": d.position_id,
                "symbol": d.symbol, "type": d.type, "entry": d.entry,
                "volume": d.volume, "price": round(d.price, 5),
                "profit": round(d.profit, 2), "comment": d.comment,
                "time": datetime.fromtimestamp(d.time, tz=timezone.utc).strftime("%Y-%m-%d %H:%M:%S"),
            })
        return Response(json.dumps({"ok": True, "count": len(rows), "deals": rows}, indent=2),
                        mimetype="application/json")
    except Exception as e:
        return Response(json.dumps({"ok": False, "msg": str(e)}), mimetype="application/json")


def history_deals_get_range(date_from, date_to, chunk_days=7):
    """Returnează toate history deals între două date folosind ferestre mici."""
    from datetime import timedelta
    deals = []
    current = date_from
    while current < date_to:
        next_dt = min(date_to, current + timedelta(days=chunk_days))
        chunk = mt5.history_deals_get(current, next_dt) or []
        if chunk:
            deals.extend(chunk)
        current = next_dt
    unique = {}
    for d in deals:
        unique[int(d.ticket)] = d
    return list(unique.values())


def history_orders_get_range(date_from, date_to, chunk_days=7):
    """Returnează toate history orders între două date folosind ferestre mici."""
    from datetime import timedelta
    orders = []
    current = date_from
    while current < date_to:
        next_dt = min(date_to, current + timedelta(days=chunk_days))
        chunk = mt5.history_orders_get(current, next_dt) or []
        if chunk:
            orders.extend(chunk)
        current = next_dt
    unique = {}
    for o in orders:
        unique[int(o.ticket)] = o
    return list(unique.values())


@app.route("/history_data")
@login_required
def history_data():
    if not MT5_AVAILABLE or mt5 is None:
        return Response(json.dumps({"ok": False, "msg": "MT5 indisponibil"}), mimetype="application/json")
    try:
        from datetime import datetime, timedelta, timezone
        now = datetime.now(timezone.utc)
        date_to   = now.replace(hour=0, minute=0, second=0, microsecond=0) + timedelta(days=1)
        date_from = date_to - timedelta(days=91)  # +1 zi buffer pentru timezone broker
        # Pas 1: ia TOATE deal-urile din perioada pe interval aliniat pe zile UTC
        all_deals = mt5.history_deals_get(date_from, date_to) or []

        # Pas 2: colecteaza TOATE position_id-urile sau fallback ticket-urile din fereastra (deschidere SAU inchidere)
        # Astfel prinde si trades deschise inainte de fereastra dar inchise in ea.
        position_keys = set()
        for d in all_deals:
            if d.position_id and d.position_id != 0 and d.symbol:
                position_keys.add(("position", int(d.position_id)))
            elif d.ticket and d.ticket != 0 and d.symbol:
                position_keys.add(("ticket", int(d.ticket)))

        # Pas 3: comentarii din orders
        orders_history = mt5.history_orders_get(date_from, date_to) or []
        order_comment_map = {}
        for o in orders_history:
            if o.comment:
                if o.position_id and o.position_id != 0:
                    key = ("position", int(o.position_id))
                elif getattr(o, 'ticket', None):
                    key = ("ticket", int(o.ticket))
                else:
                    key = None
                if key and key not in order_comment_map:
                    order_comment_map[key] = o.comment

        history = []
        for key in position_keys:
            kind, value = key
            if kind == "position":
                pos_deals = mt5.history_deals_get(position=value)
            else:
                pos_deals = mt5.history_deals_get(ticket=value)
            if not pos_deals:
                continue

            entry_deal = next((d for d in pos_deals if d.entry == 0), None)
            exit_deal  = next((d for d in pos_deals if d.entry in (1, 2, 3)), None)
            if entry_deal is None:
                continue

            profit     = sum(d.profit for d in pos_deals)
            commission = sum(d.commission for d in pos_deals)
            swap       = sum(d.swap for d in pos_deals)
            deal_type  = "BUY" if entry_deal.type == 0 else "SELL"
            open_time  = datetime.fromtimestamp(entry_deal.time, tz=timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
            close_time = datetime.fromtimestamp(exit_deal.time, tz=timezone.utc).strftime("%Y-%m-%d %H:%M:%S") if exit_deal else None
            open_ts    = int(entry_deal.time)
            close_ts   = int(exit_deal.time) if exit_deal else 0
            comment    = entry_deal.comment or order_comment_map.get(key, "")
            history.append({
                "ticket":      value,
                "symbol":      entry_deal.symbol,
                "type":        deal_type,
                "volume":      round(entry_deal.volume, 2),
                "price_open":  round(entry_deal.price, 5),
                "price_close": round(exit_deal.price, 5) if exit_deal else None,
                "profit":      round(profit, 2),
                "commission":  round(commission, 2),
                "swap":        round(swap, 2),
                "net":         round(profit + commission + swap, 2),
                "open_time":   open_time,
                "close_time":  close_time,
                "open_ts":     open_ts,
                "close_ts":    close_ts,
                "comment":     comment,
            })

        # sortat descrescator dupa timestamp INCHIDERE (sau DESCHIDERE daca nu e inchis)
        history.sort(key=lambda x: x["close_ts"] or x["open_ts"], reverse=True)
        # pastreaza doar pozitiile inchise (au exit deal)
        closed = [h for h in history if h["close_time"]]
        total_net = round(sum(h["net"] for h in closed), 2)
        wins  = len([h for h in closed if h["net"] > 0])
        losses= len([h for h in closed if h["net"] < 0])
        return Response(json.dumps({
            "ok": True,
            "history": closed[:500],
            "total_net": total_net,
            "wins": wins,
            "losses": losses,
            "count": len(closed),
        }), mimetype="application/json")
    except Exception as e:
        log.error(f"history_data error: {e}")
        return Response(json.dumps({"ok": False, "msg": str(e)}), mimetype="application/json")


@app.route("/trades")
@login_required
def trades_page():
    return TRADES_HTML


def _build_perf_data():
    """Construieste statistici + lista trade-uri per strategie. Returneaza (stats_list, trades_by_strat)."""
    from datetime import datetime, timedelta, timezone as _tz
    now = datetime.now(_tz.utc)
    date_to   = now + timedelta(days=1)
    date_from = now - timedelta(days=91)

    stats: dict[str, dict] = {}
    trades_by_strat: dict[str, list] = {}

    def _ensure(s):
        if s not in stats:
            stats[s] = {"strategy": s, "open_count": 0, "open_pnl": 0.0,
                        "closed_count": 0, "closed_pnl": 0.0, "wins": 0, "losses": 0,
                        "best_trade": None, "worst_trade": None, "avg_duration_min": 0,
                        "_durations": [], "_closed_nets": []}
        if s not in trades_by_strat:
            trades_by_strat[s] = []

    # ── Pozitii deschise ──
    for pos in (mt5.positions_get() or []):
        # Sari pozitiile non-trade (type > 1)
        if pos.type not in (0, 1):
            continue
        s = _tag_to_strat(pos.comment or "")
        _ensure(s)
        stats[s]["open_count"] += 1
        stats[s]["open_pnl"]   += pos.profit
        open_dt = datetime.fromtimestamp(pos.time, tz=_tz.utc).strftime("%Y-%m-%d %H:%M") if pos.time else ""
        trades_by_strat[s].append({
            "status":      "OPEN",
            "symbol":      pos.symbol,
            "direction":   "BUY" if pos.type == 0 else "SELL",
            "volume":      pos.volume,
            "open_price":  round(pos.price_open, 5),
            "close_price": round(pos.price_current, 5),
            "sl":          round(pos.sl, 5) if pos.sl else None,
            "tp":          round(pos.tp, 5) if pos.tp else None,
            "profit":      round(pos.profit, 2),
            "commission":  0.0,
            "swap":        round(pos.swap, 2),
            "net":         round(pos.profit + pos.swap, 2),
            "open_time":   open_dt,
            "close_time":  "—",
            "duration":    "in curs",
            "ticket":      int(pos.ticket),
            "comment":     pos.comment or "",
        })

    # ── Pozitii inchise ──
    all_deals = mt5.history_deals_get(date_from, date_to) or []

    # Pastreaza doar deal-uri reale de trading (type 0=BUY, 1=SELL) — exclude balance, credit, etc.
    trade_deals = [d for d in all_deals if d.type in (0, 1) and d.symbol]

    # Strategia se ia DOAR din deal-ul de intrare (entry==0) cu comment CV_
    pos_strat: dict[int, str] = {}
    for d in trade_deals:
        if d.entry == 0 and d.position_id and d.position_id != 0:
            strat = _tag_to_strat(d.comment or "")
            pid = int(d.position_id)
            # Preferam strategia non-Classic daca gasim un deal mai specific
            if pid not in pos_strat or pos_strat[pid] == "Classic":
                pos_strat[pid] = strat

    # Colecteaza toate position_id-urile unice din fereastra
    position_ids = {int(d.position_id) for d in trade_deals
                    if d.position_id and d.position_id != 0}

    for pid in position_ids:
        pos_deals_all = mt5.history_deals_get(position=pid)
        if not pos_deals_all:
            continue
        # Filtreaza si aici la deal-uri reale
        pos_deals = [d for d in pos_deals_all if d.type in (0, 1)]
        if not pos_deals:
            continue

        entries = [d for d in pos_deals if d.entry == 0]
        exits   = [d for d in pos_deals if d.entry in (1, 2, 3)]
        if not exits:
            continue  # inca deschisa

        # Strategia: mai intai din pos_strat, altfel din primul entry deal direct
        s = pos_strat.get(pid)
        if not s:
            # Incearca sa gaseasca direct din entry deals ale acestei pozitii
            for ed in entries:
                candidate = _tag_to_strat(ed.comment or "")
                if candidate != "Classic":
                    s = candidate
                    break
            if not s:
                s = "Classic"

        _ensure(s)

        entry_d = entries[0] if entries else pos_deals[0]
        exit_d  = exits[-1]

        net        = sum(d.profit + d.commission + d.swap for d in pos_deals)
        commission = sum(d.commission for d in pos_deals)
        swap       = sum(d.swap for d in pos_deals)
        gross      = sum(d.profit for d in pos_deals)
        volume     = entry_d.volume if entry_d else 0
        symbol     = entry_d.symbol if entry_d else (exit_d.symbol if exit_d else "")
        direction  = "BUY" if (entry_d and entry_d.type == 0) else "SELL"

        open_ts  = entry_d.time if entry_d else 0
        close_ts = exit_d.time  if exit_d  else 0
        open_dt  = datetime.fromtimestamp(open_ts,  tz=_tz.utc).strftime("%Y-%m-%d %H:%M") if open_ts else ""
        close_dt = datetime.fromtimestamp(close_ts, tz=_tz.utc).strftime("%Y-%m-%d %H:%M") if close_ts else ""
        dur_min  = round((close_ts - open_ts) / 60) if (open_ts and close_ts) else 0
        dur_str  = (f"{dur_min//60}h {dur_min%60}m" if dur_min >= 60 else f"{dur_min}m") if dur_min else "0m"

        stats[s]["closed_count"]    += 1
        stats[s]["closed_pnl"]      += net
        stats[s]["_durations"].append(dur_min)
        stats[s]["_closed_nets"].append(net)
        if net >= 0:
            stats[s]["wins"]    += 1
        else:
            stats[s]["losses"]  += 1

        trades_by_strat[s].append({
            "status":      "CLOSED",
            "symbol":      symbol,
            "direction":   direction,
            "volume":      volume,
            "open_price":  round(entry_d.price, 5) if entry_d else 0,
            "close_price": round(exit_d.price,  5) if exit_d  else 0,
            "sl":          None,
            "tp":          None,
            "profit":      round(gross, 2),
            "commission":  round(commission, 2),
            "swap":        round(swap, 2),
            "net":         round(net, 2),
            "open_time":   open_dt,
            "close_time":  close_dt,
            "duration":    dur_str,
            "ticket":      int(entry_d.ticket) if entry_d else 0,
            "comment":     (entry_d.comment or "") if entry_d else "",
        })

    # Finalizeaza stats
    result = []
    for s, st in stats.items():
        nets = st.pop("_closed_nets", [])
        durs = st.pop("_durations", [])
        st["open_pnl"]   = round(st["open_pnl"], 2)
        st["closed_pnl"] = round(st["closed_pnl"], 2)
        st["total_pnl"]  = round(st["open_pnl"] + st["closed_pnl"], 2)
        total_closed = st["wins"] + st["losses"]
        st["win_rate"]    = round(st["wins"] / total_closed * 100, 1) if total_closed > 0 else 0
        st["best_trade"]  = round(max(nets), 2) if nets else None
        st["worst_trade"] = round(min(nets), 2) if nets else None
        avg_d = round(sum(durs) / len(durs)) if durs else 0
        st["avg_duration_min"] = avg_d
        st["avg_duration_str"] = (f"{avg_d//60}h {avg_d%60}m" if avg_d >= 60 else f"{avg_d}m") if avg_d else "—"
        result.append(st)

    result.sort(key=lambda x: x["total_pnl"], reverse=True)

    for s in trades_by_strat:
        trades_by_strat[s].sort(
            key=lambda t: (0 if t["status"] == "OPEN" else 1, t["open_time"]),
            reverse=False
        )
        trades_by_strat[s].reverse()

    return result, trades_by_strat


@app.route("/trades/perf")
@login_required
def trades_perf():
    if not MT5_AVAILABLE or mt5 is None:
        return Response(json.dumps({"error": "MT5 indisponibil"}), mimetype="application/json")
    try:
        result, _ = _build_perf_data()
        return Response(json.dumps(result, cls=NpEncoder), mimetype="application/json")
    except Exception as e:
        log.error(f"trades_perf: {e}")
        return Response(json.dumps({"error": str(e)}), mimetype="application/json")


@app.route("/trades/perf/debug")
@login_required
def trades_perf_debug():
    """Debug: afiseaza primele 50 deal-uri cu comment, entry, type, position_id."""
    if not MT5_AVAILABLE or mt5 is None:
        return Response(json.dumps({"error": "MT5 indisponibil"}), mimetype="application/json")
    try:
        from datetime import datetime, timedelta, timezone as _tz
        now = datetime.now(_tz.utc)
        deals = mt5.history_deals_get(now - timedelta(days=7), now + timedelta(days=1)) or []
        out = []
        for d in deals[:100]:
            out.append({
                "ticket":      int(d.ticket),
                "position_id": int(d.position_id),
                "symbol":      d.symbol,
                "type":        int(d.type),
                "entry":       int(d.entry),
                "comment":     d.comment,
                "profit":      round(d.profit, 2),
                "commission":  round(d.commission, 2),
                "swap":        round(d.swap, 2),
                "strat_detected": _tag_to_strat(d.comment or ""),
                "time":        datetime.fromtimestamp(d.time, tz=_tz.utc).strftime("%Y-%m-%d %H:%M:%S") if d.time else "",
            })
        return Response(json.dumps(out, cls=NpEncoder), mimetype="application/json")
    except Exception as e:
        return Response(json.dumps({"error": str(e)}), mimetype="application/json")


@app.route("/trades/perf/detail")
@login_required
def trades_perf_detail():
    if not MT5_AVAILABLE or mt5 is None:
        return Response(json.dumps({"error": "MT5 indisponibil"}), mimetype="application/json")
    try:
        strategy = request.args.get("strategy", "")
        result, trades_by_strat = _build_perf_data()
        stats = next((r for r in result if r["strategy"] == strategy), None)
        trades = trades_by_strat.get(strategy, [])
        return Response(json.dumps({"stats": stats, "trades": trades}, cls=NpEncoder), mimetype="application/json")
    except Exception as e:
        log.error(f"trades_perf_detail: {e}")
        return Response(json.dumps({"error": str(e)}), mimetype="application/json")


# ── Settings page — Cloudflare + Telegram ────────────────────────────────────
import subprocess as _subprocess
import signal as _signal

import re as _re
import threading as _threading

_CF_URL  = None
_CF_LOCK = _threading.Lock()

def _cf_is_running():
    try:
        r = _subprocess.run(["tasklist", "/fi", "imagename eq cloudflared.exe"],
                            capture_output=True, text=True, timeout=3)
        return "cloudflared.exe" in r.stdout
    except Exception:
        return False

def _cf_kill():
    global _CF_URL
    try:
        _subprocess.run(["taskkill", "/f", "/im", "cloudflared.exe"],
                        capture_output=True, timeout=5)
    except Exception:
        pass
    with _CF_LOCK:
        _CF_URL = None

def _cf_launch():
    global _CF_URL
    try:
        proc = _subprocess.Popen(
            ["cloudflared", "tunnel", "--url", "http://localhost:5004"],
            stdout=_subprocess.PIPE, stderr=_subprocess.STDOUT,
            creationflags=_subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0,
        )
    except Exception as e:
        log.warning("cloudflared launch failed: %s", e)
        return
    log.info("Cloudflare tunnel pornit (PID %s)", proc.pid)
    for raw in proc.stdout:
        try:
            line = raw.decode("utf-8", errors="ignore").strip()
        except Exception:
            continue
        m = _re.search(r"https://[a-zA-Z0-9\-]+\.trycloudflare\.com", line)
        if m:
            with _CF_LOCK:
                if _CF_URL is None:
                    _CF_URL = m.group(0)
                    log.info("Cloudflare URL: %s", _CF_URL)
        if proc.poll() is not None:
            break
    log.info("Cloudflare tunnel oprit")

# cloudflared pornit din start.bat — nu auto-start din app

@app.route("/settings")
@login_required
def settings_page():
    return Response(SETTINGS_HTML, mimetype="text/html")

@app.route("/settings/cloudflare/start", methods=["POST"])
@login_required
def cf_start():
    if _cf_is_running():
        url = _CF_URL
        if url:
            return Response(json.dumps({"ok": True, "url": url}), mimetype="application/json")
    try:
        _cf_kill()
        __import__("time").sleep(0.5)
        _threading.Thread(target=_cf_launch, daemon=True).start()
        return Response(json.dumps({"ok": True, "url": None}), mimetype="application/json")
    except FileNotFoundError:
        return Response(json.dumps({"ok": False, "msg": "cloudflared nu e instalat"}), mimetype="application/json")
    except Exception as e:
        return Response(json.dumps({"ok": False, "msg": str(e)}), mimetype="application/json")

@app.route("/settings/cloudflare/stop", methods=["POST"])
@login_required
def cf_stop():
    try:
        _cf_kill()
        return Response(json.dumps({"ok": True}), mimetype="application/json")
    except Exception as e:
        return Response(json.dumps({"ok": False, "msg": str(e)}), mimetype="application/json")

@app.route("/settings/cloudflare/status")
@login_required
def cf_status():
    running = _cf_is_running()
    url = _CF_URL if running else None
    return Response(json.dumps({"running": running, "url": url}), mimetype="application/json")

SETTINGS_HTML = """<!DOCTYPE html>
<html><head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Settings — ChartVisualizer</title>
<style>
*{box-sizing:border-box;margin:0;padding:0}
body{background:#0d0d0d;color:#ccc;font-family:'Segoe UI',sans-serif;padding:24px}
.back{display:inline-block;color:#555;font-size:0.85rem;text-decoration:none;margin-bottom:20px}
.back:hover{color:#aaa}
h1{font-size:1.1rem;color:#aaa;font-weight:500;margin-bottom:24px}
.section{background:#111;border:1px solid #222;border-radius:8px;padding:20px 24px;margin-bottom:20px;max-width:680px}
.section h2{font-size:0.88rem;text-transform:uppercase;letter-spacing:1px;color:#555;margin-bottom:16px;padding-bottom:8px;border-bottom:1px solid #1e1e1e}
.row{display:flex;align-items:center;justify-content:space-between;padding:9px 0;border-bottom:1px solid #1a1a1a;gap:12px}
.row:last-child{border-bottom:none}
.label{font-size:0.84rem;color:#aaa}
.sub{display:block;font-size:0.72rem;color:#555;margin-top:2px}
input[type=text],input[type=password]{background:#1a1a1a;border:1px solid #333;color:#eee;padding:6px 10px;border-radius:4px;font-size:0.82rem;width:260px}
input[type=text]:focus,input[type=password]:focus{outline:none;border-color:#555}
.btn{padding:6px 14px;border-radius:5px;border:none;cursor:pointer;font-size:0.8rem;font-weight:600}
.btn-green{background:#1b3a1b;color:#66bb6a;border:1px solid #2a6a2a}
.btn-green:hover{background:#1f4a1f}
.btn-red{background:#3a1b1b;color:#ef5350;border:1px solid #6a2a2a}
.btn-red:hover{background:#4a1f1f}
.btn-blue{background:#1a1f3a;color:#7986cb;border:1px solid #3949ab}
.btn-blue:hover{background:#1f2a4a}
.status-dot{width:8px;height:8px;border-radius:50%;display:inline-block;margin-right:6px}
.dot-on{background:#66bb6a;box-shadow:0 0 6px #66bb6a}
.dot-off{background:#444}
#cf-url{font-size:0.78rem;color:#26a69a;word-break:break-all;margin-top:10px;padding:8px 12px;background:#0a1f0a;border-radius:4px;border:1px solid #1b3a1b;display:none}
#cf-url a{color:#4db6ac;text-decoration:none}
#cf-url a:hover{text-decoration:underline}
.toast{position:fixed;bottom:20px;right:20px;background:#222;color:#eee;padding:8px 18px;border-radius:6px;font-size:0.82rem;display:none;z-index:999}
.dot-pending{background:#ffeb3b;box-shadow:0 0 6px #ffeb3b;animation:pulse 0.8s infinite}
@keyframes pulse{0%,100%{opacity:1}50%{opacity:0.3}}
.spinner{display:inline-block;width:10px;height:10px;border:2px solid #444;border-top-color:#ffeb3b;border-radius:50%;animation:spin 0.7s linear infinite;vertical-align:middle;margin-right:5px}
@keyframes spin{to{transform:rotate(360deg)}}
</style>
</head><body>
<a class="back" href="/">← ChartVisualizer</a>
<h1>⚙ Settings</h1>

<!-- CLOUDFLARE TUNNEL -->
<div class="section">
    <h2>🌐 Cloudflare Tunnel</h2>
    <div class="row">
        <label class="label">Status
            <span class="sub">Acces extern securizat prin Cloudflare</span>
        </label>
        <div style="display:flex;align-items:center;gap:10px">
            <span id="cf-status-text" style="font-size:0.8rem;color:#555">
                <span class="status-dot dot-off" id="cf-dot"></span>Oprit
            </span>
            <button class="btn btn-green" id="cf-start-btn" onclick="cfStart()">▶ Porneste</button>
            <button class="btn btn-red"   id="cf-stop-btn"  onclick="cfStop()" style="display:none">■ Opreste</button>
        </div>
    </div>
    <div id="cf-url">
        <span style="color:#555;font-size:0.72rem">URL public:</span><br>
        <a id="cf-url-link" href="#" target="_blank"></a>
        <button onclick="copyUrl()" style="margin-left:10px;background:#1a1a1a;border:1px solid #333;color:#aaa;padding:2px 8px;border-radius:3px;font-size:0.72rem;cursor:pointer">📋 Copiaza</button>
    </div>
    <div style="font-size:0.72rem;color:#444;margin-top:12px">
        Necesita <code style="color:#666">cloudflared</code> instalat si in PATH.
        <a href="https://developers.cloudflare.com/cloudflare-one/connections/connect-apps/install-and-setup/installation/" target="_blank" style="color:#3949ab">Instalare →</a>
    </div>
</div>

<!-- TELEGRAM -->
<div class="section">
    <h2>📱 Telegram Bot</h2>
    <div class="row">
        <label class="label">Bot Token
            <span class="sub">De la @BotFather pe Telegram</span>
        </label>
        <input type="password" id="tg-token" placeholder="123456:ABC-DEF...">
    </div>
    <div class="row">
        <label class="label">Chat ID
            <span class="sub">ID-ul tau de utilizator sau grup</span>
        </label>
        <input type="text" id="tg-chatid" placeholder="-100123456789">
    </div>
    <div class="row">
        <label class="label">Notificari
            <span class="sub">Trimite alerte la fiecare semnal</span>
        </label>
        <select id="tg-enabled" style="background:#1a1a1a;border:1px solid #333;color:#eee;padding:6px 10px;border-radius:4px;font-size:0.82rem">
            <option value="0">Dezactivate</option>
            <option value="1">Active</option>
        </select>
    </div>
    <div class="row">
        <label class="label">Aprobare manuala
            <span class="sub">Trade-urile asteapta confirmare pe Telegram</span>
        </label>
        <select id="tg-approval" style="background:#1a1a1a;border:1px solid #333;color:#eee;padding:6px 10px;border-radius:4px;font-size:0.82rem">
            <option value="0">Nu (automat)</option>
            <option value="1">Da (aprob manual)</option>
        </select>
    </div>
    <div style="display:flex;gap:8px;margin-top:14px">
        <button class="btn btn-blue" onclick="saveTg()">💾 Salveaza</button>
        <button class="btn" style="background:#1a1a1a;color:#aaa;border:1px solid #333" onclick="testTg()">📤 Test mesaj</button>
    </div>
</div>

<div class="toast" id="toast"></div>

<script>
function toast(msg, ok=true){
    const t=document.getElementById('toast');
    t.textContent=msg;
    t.style.display='block';
    t.style.borderLeft='3px solid '+(ok?'#26a69a':'#ef5350');
    clearTimeout(t._t);
    t._t=setTimeout(()=>t.style.display='none',3000);
}

// ── CLOUDFLARE ──
let _cfPollTimer = null;
let _cfPolling   = false;

function _cfSetLoading(){
    document.getElementById('cf-dot').className = 'status-dot dot-pending';
    document.getElementById('cf-status-text').innerHTML =
        '<span class="status-dot dot-pending"></span><span class="spinner"></span> Se conecteaza...';
    document.getElementById('cf-start-btn').style.display = 'none';
    document.getElementById('cf-stop-btn').style.display  = 'none';
    const urlD = document.getElementById('cf-url');
    urlD.style.display = 'block';
    urlD.innerHTML = '<span style="color:#555;font-size:0.75rem"><span class="spinner"></span> Astept URL de la Cloudflare...</span>';
}

function _cfSetActive(url){
    document.getElementById('cf-dot').className = 'status-dot dot-on';
    document.getElementById('cf-status-text').innerHTML =
        '<span class="status-dot dot-on"></span>Activ';
    document.getElementById('cf-start-btn').style.display = 'none';
    document.getElementById('cf-stop-btn').style.display  = '';
    const urlD = document.getElementById('cf-url');
    urlD.style.display = 'block';
    urlD.innerHTML = '<span style="color:#555;font-size:0.72rem">URL public:</span><br>' +
        '<a id="cf-url-link" href="'+url+'" target="_blank" style="color:#4db6ac;text-decoration:none">'+url+'</a>' +
        '<button onclick="copyUrl()" style="margin-left:10px;background:#1a1a1a;border:1px solid #333;color:#aaa;padding:2px 8px;border-radius:3px;font-size:0.72rem;cursor:pointer">📋 Copiaza</button>';
}

function _cfSetOff(){
    document.getElementById('cf-dot').className = 'status-dot dot-off';
    document.getElementById('cf-status-text').innerHTML =
        '<span class="status-dot dot-off"></span>Oprit';
    document.getElementById('cf-start-btn').style.display = '';
    document.getElementById('cf-stop-btn').style.display  = 'none';
    document.getElementById('cf-url').style.display = 'none';
    document.getElementById('cf-url').innerHTML = '';
}

async function cfStart(){
    _cfSetLoading();
    const r = await fetch('/settings/cloudflare/start', {method:'POST'});
    const d = await r.json();
    if(!d.ok){
        toast('Eroare: ' + (d.msg||'necunoscuta'), false);
        _cfSetOff();
        return;
    }
    if(d.url){
        _cfSetActive(d.url);
    } else {
        _startPolling();
    }
}

async function cfStop(){
    clearTimeout(_cfPollTimer); _cfPolling = false;
    await fetch('/settings/cloudflare/stop', {method:'POST'});
    _cfSetOff();
    toast('Tunel oprit');
}

function _startPolling(){
    if(_cfPolling) return;
    _cfPolling = true;
    let attempts = 0;
    function poll(){
        if(!_cfPolling) return;
        attempts++;
        fetch('/settings/cloudflare/status').then(r=>r.json()).then(d=>{
            if(d.url){
                _cfPolling = false;
                _cfSetActive(d.url);
                toast('Tunel activ!');
            } else if(!d.running){
                _cfPolling = false;
                _cfSetOff();
                toast('Tunelul s-a oprit neasteptat', false);
            } else if(attempts < 40){
                _cfPollTimer = setTimeout(poll, 1500);
            } else {
                _cfPolling = false;
                toast('Timeout — URL nu a aparut in 60s', false);
                _cfSetOff();
            }
        }).catch(()=>{ _cfPollTimer = setTimeout(poll, 2000); });
    }
    poll();
}

function copyUrl(){
    const el = document.getElementById('cf-url-link');
    if(!el) return;
    navigator.clipboard.writeText(el.textContent).then(()=>toast('URL copiat!'));
}

// ── TELEGRAM ──
async function loadTg(){
    const r = await fetch('/telegram/config');
    const d = await r.json();
    document.getElementById('tg-token').value    = d.bot_token||'';
    document.getElementById('tg-chatid').value   = d.chat_id||'';
    document.getElementById('tg-enabled').value  = d.enabled ? '1':'0';
    document.getElementById('tg-approval').value = d.require_approval ? '1':'0';
}
async function saveTg(){
    const r = await fetch('/telegram/config',{
        method:'POST', headers:{'Content-Type':'application/json'},
        body: JSON.stringify({
            bot_token:        document.getElementById('tg-token').value.trim(),
            chat_id:          document.getElementById('tg-chatid').value.trim(),
            enabled:          document.getElementById('tg-enabled').value==='1',
            require_approval: document.getElementById('tg-approval').value==='1',
        })
    });
    const d = await r.json();
    toast(d.ok ? 'Salvat cu succes' : 'Eroare: '+(d.msg||''), d.ok);
}
async function testTg(){
    const r = await fetch('/telegram/test',{method:'POST'});
    const d = await r.json();
    toast(d.ok ? 'Mesaj test trimis!' : 'Eroare: '+(d.msg||d.error||''), d.ok);
}

// Init — verifica starea la incarcare
(async()=>{
    loadTg();
    const d = await fetch('/settings/cloudflare/status').then(r=>r.json());
    if(d.running && d.url){
        _cfSetActive(d.url);
    } else if(d.running && !d.url){
        _cfSetLoading();
        _startPolling();
    } else {
        _cfSetOff();
    }
})();
</script>
</body></html>"""


# AutoTrader blueprint — importat dupa ce toate variabilele si functiile sunt definite
try:
    from autotrader import autotrader_bp
    app.register_blueprint(autotrader_bp)
    log.info("AutoTrader blueprint inregistrat")
except Exception as e:
    log.warning(f"AutoTrader blueprint nu s-a putut incarca: {e}")

# AutoOrders blueprint — scanner si pagina dedicata ordinelor pending/predictive
try:
    from autoorders import autoorders_bp
    app.register_blueprint(autoorders_bp)
    log.info("AutoOrders blueprint inregistrat")
except Exception as e:
    log.warning(f"AutoOrders blueprint nu s-a putut incarca: {e}")

# Backtest blueprint — walk-forward backtester pentru strategii
try:
    from backtest import backtest_bp
    app.register_blueprint(backtest_bp)
    log.info("Backtest blueprint inregistrat")
except Exception as e:
    log.warning(f"Backtest blueprint nu s-a putut incarca: {e}")

# AI Agent blueprint — pagina /ai_agent (GUI pt training + monitoring modele)
try:
    from ai_agent_bp import ai_agent_bp
    app.register_blueprint(ai_agent_bp)
    log.info("AI Agent blueprint inregistrat")
except Exception as e:
    log.warning(f"AI Agent blueprint nu s-a putut incarca: {e}")

# AI Trader blueprint — pagina /ai_trader (scanner independent + paper trading)
try:
    from ai_trader_bp import ai_trader_bp
    app.register_blueprint(ai_trader_bp)
    log.info("AI Trader blueprint inregistrat")
except Exception as e:
    log.warning(f"AI Trader blueprint nu s-a putut incarca: {e}")

if __name__ == "__main__":
    import sys
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 5004
    print(f"ChartVisualizer pornit pe http://localhost:{port}")
    app.run(host="0.0.0.0", port=port, debug=False)
