"""
ai_agent/monitoring/kill_switch.py — opreste AI agent automat in caz de drawdown.

Verifica zilnic equity-ul cont MT5. Daca DD > prag:
  1. Dezactiveaza strategia ai_agent in scanner (scanner['ai_agent']['enabled'] = False)
  2. Trimite alert Telegram (daca bridge configurat)
  3. Loghează in fisier kill_switch.log

Pragul default: 10% drawdown din peak equity al ultimelor 30 zile.
"""
from __future__ import annotations
import logging
from datetime import datetime, timezone

from ai_agent.config import LOGS_DIR

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

MAX_DD_PCT = 10.0      # daca DD > 10% → kill switch
PEAK_LOOKBACK_DAYS = 30


def get_current_equity() -> float | None:
    """Cite equity-ul curent din MT5."""
    try:
        import MetaTrader5 as mt5
        if not mt5.initialize():
            return None
        acc = mt5.account_info()
        if acc is None:
            return None
        return float(acc.equity)
    except Exception as e:
        log.error(f"get_current_equity: {e}")
        return None


def get_peak_equity(days_back: int = PEAK_LOOKBACK_DAYS) -> float | None:
    """
    Peak equity din ultimele N zile, citit dintr-un fisier de log.
    Pe Windows poti combina cu un cron care log-uieste zilnic.
    """
    log_path = LOGS_DIR / "equity_log.csv"
    if not log_path.exists():
        return None
    try:
        import pandas as pd
        df = pd.read_csv(log_path, names=["timestamp", "equity"])
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        cutoff = datetime.now(timezone.utc) - pd.Timedelta(days=days_back)
        recent = df[df["timestamp"] >= cutoff]
        return float(recent["equity"].max()) if not recent.empty else None
    except Exception as e:
        log.error(f"get_peak_equity: {e}")
        return None


def log_equity_snapshot() -> None:
    """Salveaza equity curent in CSV (apelat zilnic)."""
    eq = get_current_equity()
    if eq is None:
        return
    log_path = LOGS_DIR / "equity_log.csv"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with open(log_path, "a", encoding="utf-8") as f:
        f.write(f"{datetime.now(timezone.utc).isoformat()},{eq}\n")
    log.info(f"Equity snapshot: ${eq:.2f}")


def check_drawdown() -> dict:
    """Verifica DD curent. Returneaza dict cu rezultatul."""
    log_equity_snapshot()
    eq = get_current_equity()
    peak = get_peak_equity()
    if eq is None or peak is None or peak <= 0:
        return {"ok": True, "reason": "Date insuficiente"}
    dd_pct = (peak - eq) / peak * 100
    result = {
        "ok":      dd_pct <= MAX_DD_PCT,
        "equity":  round(eq, 2),
        "peak":    round(peak, 2),
        "dd_pct":  round(dd_pct, 2),
        "max_dd":  MAX_DD_PCT,
    }
    if not result["ok"]:
        log.error(f"⚠ KILL SWITCH TRIGGERED: DD {dd_pct:.2f}% > {MAX_DD_PCT}%")
        _trigger_kill(result)
    else:
        log.info(f"DD OK: {dd_pct:.2f}% (peak ${peak:.0f}, current ${eq:.0f})")
    return result


def _trigger_kill(info: dict) -> None:
    """Dezactiveaza ai_agent in scanner + log + alert."""
    try:
        # Dezactiveaza via API daca ChartVisualizer ruleaza
        import requests
        requests.post("http://localhost:5004/autotrader/set",
                      json={"ai_agent": {"enabled": False}}, timeout=5)
        log.info("ai_agent disabled via /autotrader/set")
    except Exception as e:
        log.warning(f"nu am putut dezactiva via API: {e}")

    # Telegram alert (daca ClaudeTelegramBridge are credentiale)
    try:
        from pathlib import Path
        import json
        cfg_path = Path("c:/Users/vlady/Desktop/FINALLBOSS/ChartVisualizer/config.json")
        if cfg_path.exists():
            cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
            tg = cfg.get("telegram", {})
            token = tg.get("bot_token", "")
            chat_id = tg.get("chat_id", "")
            if token and chat_id:
                requests.post(
                    f"https://api.telegram.org/bot{token}/sendMessage",
                    json={"chat_id": chat_id,
                          "text": (f"🛑 *AI Agent KILLED*\n"
                                   f"DD: {info['dd_pct']}% > {info['max_dd']}%\n"
                                   f"Equity: ${info['equity']:.0f}\n"
                                   f"Peak:   ${info['peak']:.0f}"),
                          "parse_mode": "Markdown"},
                    timeout=10,
                )
    except Exception as e:
        log.warning(f"telegram alert fail: {e}")

    # Log file
    with open(LOGS_DIR / "kill_switch.log", "a", encoding="utf-8") as f:
        f.write(f"{datetime.now(timezone.utc).isoformat()} KILL "
                f"dd={info['dd_pct']}% eq={info['equity']} peak={info['peak']}\n")


if __name__ == "__main__":
    r = check_drawdown()
    import json as _j
    print(_j.dumps(r, indent=2))
