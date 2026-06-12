"""
Telegram Notifier — sistem de aprobare trade pe telefon.
100% gratuit: Telegram Bot API + requests (deja instalat).

Setup rapid (5 minute):
  1. Deschide Telegram → cauta @BotFather → /newbot → copiaza TOKEN
  2. Trimite /start botului tau, apoi deschide:
       https://api.telegram.org/bot<TOKEN>/getUpdates
     si copiaza "chat":{"id":XXXXXXX} → ala e CHAT_ID
  3. In ChartVisualizer → Settings → Telegram → introdu token + chat_id → Salveaza + Testeaza
"""
from __future__ import annotations
import json, os, time, threading, logging, uuid, io
import requests

log = logging.getLogger(__name__)

_CFG_PATH = os.path.join(os.path.dirname(__file__), "config.json")

# ── State global ──────────────────────────────────────────────────────────────
_bot_token: str  = ""
_chat_id:   str  = ""
_enabled:   bool = False
_require_approval: bool = True   # True = asteapta aprobare; False = doar notificare

_pending: dict[str, dict] = {}   # {token: trade_info}
_poll_offset: int | None  = None
_trade_callback = None           # injectat de app.py → place_trade()
_polling_started: bool = False


# ── Config ────────────────────────────────────────────────────────────────────

def load_config() -> None:
    global _bot_token, _chat_id, _enabled, _require_approval
    try:
        if os.path.exists(_CFG_PATH):
            with open(_CFG_PATH, encoding="utf-8") as f:
                cfg = json.load(f)
            tg = cfg.get("telegram", {})
            _bot_token        = tg.get("bot_token", "").strip()
            _chat_id          = str(tg.get("chat_id", "")).strip()
            _enabled          = bool(tg.get("enabled", False)) and bool(_bot_token) and bool(_chat_id)
            _require_approval = bool(tg.get("require_approval", True))
    except Exception as e:
        log.warning(f"Telegram config load: {e}")


def save_config(token: str, chat_id: str, enabled: bool, require_approval: bool = True) -> None:
    global _bot_token, _chat_id, _enabled, _require_approval
    _bot_token        = token.strip()
    _chat_id          = str(chat_id).strip()
    _enabled          = bool(enabled) and bool(_bot_token) and bool(_chat_id)
    _require_approval = bool(require_approval)
    try:
        cfg: dict = {}
        if os.path.exists(_CFG_PATH):
            with open(_CFG_PATH, encoding="utf-8") as f:
                cfg = json.load(f)
        cfg["telegram"] = {
            "bot_token":        _bot_token,
            "chat_id":          _chat_id,
            "enabled":          _enabled,
            "require_approval": _require_approval,
        }
        with open(_CFG_PATH, "w", encoding="utf-8") as f:
            json.dump(cfg, f, indent=2)
        log.info(f"Telegram config saved — enabled={_enabled}")
    except Exception as e:
        log.warning(f"Telegram config save: {e}")


def get_config() -> dict:
    return {
        "bot_token":        _bot_token,
        "chat_id":          _chat_id,
        "enabled":          _enabled,
        "require_approval": _require_approval,
    }


# ── Telegram API ──────────────────────────────────────────────────────────────

def _api(method: str, json_data: dict | None = None, files=None, data=None, timeout=12) -> dict | None:
    if not _bot_token:
        return None
    url = f"https://api.telegram.org/bot{_bot_token}/{method}"
    try:
        if files:
            r = requests.post(url, data=data, files=files, timeout=30)
        else:
            r = requests.post(url, json=json_data, timeout=timeout)
        r.raise_for_status()
        return r.json()
    except Exception as e:
        log.warning(f"Telegram API {method}: {e}")
        return None


def _markup(token: str) -> dict:
    """Inline keyboard: Executa / Respinge."""
    return {"inline_keyboard": [[
        {"text": "✅  Executa", "callback_data": f"approve_{token}"},
        {"text": "❌  Respinge", "callback_data": f"deny_{token}"},
    ]]}


# ── Mesaje simple ─────────────────────────────────────────────────────────────

def send_message(text: str, parse_mode: str = "HTML") -> None:
    if not _enabled:
        return
    _api("sendMessage", json_data={"chat_id": _chat_id, "text": text,
                                    "parse_mode": parse_mode, "disable_web_page_preview": True})


def send_photo(img_bytes: bytes, caption: str = "", reply_markup: dict | None = None) -> None:
    if not _enabled:
        return
    payload = {"chat_id": _chat_id, "caption": caption[:1024], "parse_mode": "HTML"}
    if reply_markup:
        payload["reply_markup"] = json.dumps(reply_markup)
    _api("sendPhoto", data=payload, files={"photo": ("chart.png", img_bytes, "image/png")})


# ── Cerere de aprobare ────────────────────────────────────────────────────────

def request_approval(
    symbol: str, signal: str, strategy: str,
    sl: float, tp: float, price: float,
    confidence: float, tf: str,
    img_bytes: bytes | None = None,
    reasons: list[str] | None = None,
) -> str | None:
    """
    Trimite notificare pe Telegram cu analiza + butoane Aprobare/Respingere.
    Returneaza token-ul cererii (sau None daca Telegram e dezactivat).
    """
    if not _enabled:
        return None

    token   = uuid.uuid4().hex[:12]
    expires = time.time() + 300   # 5 minute timeout

    _pending[token] = {
        "token": token, "symbol": symbol, "signal": signal,
        "strategy": strategy, "sl": sl, "tp": tp, "price": price,
        "confidence": confidence, "tf": tf,
        "status": "pending", "expires": expires,
        "reasons": reasons or [],
    }

    rr = round(abs(tp - price) / abs(price - sl), 2) if price and sl and tp and abs(price - sl) > 1e-9 else 0
    sig_emoji = "🟢 BUY" if signal == "BUY" else "🔴 SELL"

    reasons_text = ""
    if reasons:
        reasons_text = "\n📋 <b>Analiza:</b>\n" + "\n".join(f"  • {r}" for r in reasons[:6])

    caption = (
        f"<b>⚡ SEMNAL NOU — {symbol}</b>\n\n"
        f"📊 Strategie: <code>{strategy}</code>\n"
        f"📈 Signal: <b>{sig_emoji}</b>  |  TF: <code>{tf}</code>\n"
        f"🎯 Confidence: <b>{confidence:.1f}%</b>\n\n"
        f"💰 Entry:  <code>{price:.5f}</code>\n"
        f"🛑 SL:     <code>{sl:.5f}</code>\n"
        f"✅ TP:     <code>{tp:.5f}</code>\n"
        f"⚖️ R:R:    <b>{rr:.2f}:1</b>"
        f"{reasons_text}\n\n"
        f"<i>⏱ Expira in 5 minute</i>"
    )

    markup = _markup(token) if _require_approval else None

    if img_bytes:
        send_photo(img_bytes, caption=caption, reply_markup=markup)
    else:
        payload = {"chat_id": _chat_id, "text": caption, "parse_mode": "HTML",
                   "disable_web_page_preview": True}
        if markup:
            payload["reply_markup"] = markup
        _api("sendMessage", json_data=payload)

    log.info(f"Telegram approval request: {token} — {signal} {symbol} conf={confidence:.0f}%")
    return token


def notify_only(
    symbol: str, signal: str, strategy: str,
    sl: float, tp: float, price: float,
    confidence: float, tf: str,
    img_bytes: bytes | None = None,
    reasons: list[str] | None = None,
) -> None:
    """Trimite notificare simpla fara butoane de aprobare."""
    if not _enabled:
        return
    rr = round(abs(tp - price) / abs(price - sl), 2) if price and sl and tp and abs(price - sl) > 1e-9 else 0
    sig_emoji = "🟢 BUY" if signal == "BUY" else "🔴 SELL"
    reasons_text = ""
    if reasons:
        reasons_text = "\n📋 <b>Analiza:</b>\n" + "\n".join(f"  • {r}" for r in reasons[:6])
    text = (
        f"<b>✅ EXECUTAT — {symbol}</b>\n\n"
        f"📊 Strategie: <code>{strategy}</code>\n"
        f"📈 Signal: <b>{sig_emoji}</b>  |  TF: <code>{tf}</code>\n"
        f"🎯 Confidence: <b>{confidence:.1f}%</b>\n\n"
        f"💰 Entry:  <code>{price:.5f}</code>\n"
        f"🛑 SL:     <code>{sl:.5f}</code>\n"
        f"✅ TP:     <code>{tp:.5f}</code>\n"
        f"⚖️ R:R:    <b>{rr:.2f}:1</b>"
        f"{reasons_text}"
    )
    if img_bytes:
        send_photo(img_bytes, caption=text)
    else:
        _api("sendMessage", json_data={"chat_id": _chat_id, "text": text,
                                        "parse_mode": "HTML", "disable_web_page_preview": True})


# ── Notificare inchidere pozitie ─────────────────────────────────────────────

def notify_close(
    symbol: str,
    signal: str,           # "BUY" / "SELL" — directia originala
    strategy: str,
    entry: float,
    close_price: float,
    sl: float,
    tp: float,
    pnl: float,
    reason: str,           # "SL" | "TP" | "MANUAL" | "NEWS" | "TIME_EXIT" | "PARTIAL"
    volume: float = 0.0,
) -> None:
    """Trimite notificare Telegram la inchiderea unei pozitii."""
    if not _enabled:
        return

    reason_map = {
        "SL":         "🛑 Stop Loss atins",
        "TP":         "✅ Take Profit atins",
        "MANUAL":     "👤 Inchis manual",
        "NEWS":       "📰 Inchis inainte de stire",
        "TIME_EXIT":  "⏱ Time Exit (miscare plata)",
        "PARTIAL":    "⚡ Partial close (50% la TP1)",
        "STOP_OUT":   "💀 Stop Out",
        "OTHER":      "🔄 Inchis automat",
    }
    reason_text = reason_map.get(reason, f"🔄 {reason}")

    pnl_emoji = "🟢" if pnl >= 0 else "🔴"
    pnl_text  = f"{'+' if pnl >= 0 else ''}{pnl:.2f}$"
    sig_emoji = "📈 BUY" if signal == "BUY" else "📉 SELL"
    rr_entry  = abs(close_price - entry)
    rr_risk   = abs(entry - sl) if sl else 0
    rr_val    = round(rr_entry / rr_risk, 2) if rr_risk > 1e-9 else 0

    text = (
        f"<b>{pnl_emoji} INCHIS — {symbol}</b>\n\n"
        f"📊 Strategie: <code>{strategy}</code>\n"
        f"📈 Directie: <b>{sig_emoji}</b>\n"
        f"❓ Motiv: <b>{reason_text}</b>\n\n"
        f"💰 Entry:  <code>{entry:.5f}</code>\n"
        f"🏁 Close:  <code>{close_price:.5f}</code>\n"
        f"🛑 SL:     <code>{sl:.5f}</code>\n"
        f"✅ TP:     <code>{tp:.5f}</code>\n"
        f"📦 Volum:  <code>{volume}</code>\n\n"
        f"⚖️ R realizat: <b>{rr_val:.2f}R</b>\n"
        f"💵 P&L: <b>{pnl_text}</b>"
    )
    _api("sendMessage", json_data={
        "chat_id": _chat_id, "text": text,
        "parse_mode": "HTML", "disable_web_page_preview": True,
    })


# ── Pending approvals ─────────────────────────────────────────────────────────

def get_pending() -> dict:
    now = time.time()
    for t, v in list(_pending.items()):
        if v["status"] == "pending" and v["expires"] < now:
            v["status"] = "expired"
            try:
                send_message(f"⏰ Trade expirat: {v['signal']} {v['symbol']} (timeout 5 min)")
            except Exception:
                pass
    return {t: v for t, v in _pending.items() if v["status"] == "pending"}


def get_all_recent(limit: int = 20) -> list[dict]:
    items = sorted(_pending.values(), key=lambda x: x.get("expires", 0), reverse=True)
    return items[:limit]


def manual_approve(token: str) -> tuple[bool, str]:
    """Aprobare manuala din UI web (fara Telegram)."""
    trade = _pending.get(token)
    if not trade:
        return False, "Token negasit"
    if trade["status"] != "pending":
        return False, f"Status: {trade['status']}"
    if time.time() > trade["expires"]:
        trade["status"] = "expired"
        return False, "Trade expirat"
    return _execute_trade(trade, approved_by="web-ui")


def manual_deny(token: str) -> tuple[bool, str]:
    trade = _pending.get(token)
    if not trade:
        return False, "Token negasit"
    trade["status"] = "denied"
    send_message(f"❌ <b>RESPINS (web)</b> — {trade['signal']} {trade['symbol']}")
    return True, "Trade respins"


# ── Procesare callback Telegram ───────────────────────────────────────────────

def _execute_trade(trade: dict, approved_by: str = "") -> tuple[bool, str]:
    """Executa trade-ul aprobat."""
    global _trade_callback
    trade["status"] = "approved"
    trade["approved_by"] = approved_by

    if _trade_callback:
        try:
            ok, msg = _trade_callback(
                trade["symbol"], trade["signal"],
                trade["sl"], trade["tp"],
            )
            trade["execute_ok"]     = ok
            trade["execute_result"] = msg
            result_msg = f"✅ Executat: {msg}" if ok else f"❌ Eroare executie: {msg}"
            send_message(
                f"{'✅' if ok else '❌'} <b>{'EXECUTAT' if ok else 'EROARE'}</b> — "
                f"{trade['signal']} {trade['symbol']}\n{msg}"
            )
            return ok, msg
        except Exception as e:
            trade["execute_result"] = str(e)
            send_message(f"❌ Exceptie la executie: {e}")
            return False, str(e)
    else:
        send_message("⚠️ Executia automata nu e configurata in app.py.")
        return False, "trade_callback neconfigurat"


def _process_callback(callback_data: str, callback_query_id: str, user_name: str = "") -> None:
    if callback_data.startswith("approve_"):
        token = callback_data[8:]
        trade = _pending.get(token)
        if not trade:
            _api("answerCallbackQuery", json_data={"callback_query_id": callback_query_id,
                                                    "text": "Trade negasit sau expirat."})
            return
        if trade["status"] != "pending":
            _api("answerCallbackQuery", json_data={"callback_query_id": callback_query_id,
                                                    "text": f"Deja {trade['status']}."})
            return
        if time.time() > trade["expires"]:
            trade["status"] = "expired"
            _api("answerCallbackQuery", json_data={"callback_query_id": callback_query_id,
                                                    "text": "Trade expirat (>5 min)."})
            return
        _api("answerCallbackQuery", json_data={"callback_query_id": callback_query_id,
                                                "text": "✅ Aprobat! Se executa..."})
        send_message(f"⏳ Execut {trade['signal']} {trade['symbol']} (aprobat de {user_name or 'tine'})...")
        _execute_trade(trade, approved_by=user_name)

    elif callback_data.startswith("deny_"):
        token = callback_data[5:]
        trade = _pending.get(token)
        if not trade:
            _api("answerCallbackQuery", json_data={"callback_query_id": callback_query_id,
                                                    "text": "Trade negasit."})
            return
        trade["status"] = "denied"
        _api("answerCallbackQuery", json_data={"callback_query_id": callback_query_id,
                                                "text": "❌ Trade respins."})
        send_message(f"❌ <b>RESPINS</b> — {trade['signal']} {trade['symbol']} de {user_name or 'tine'}")


def _process_message(msg: dict) -> None:
    """Raspunde la comenzi /status, /pending, /help trimise botului."""
    text = (msg.get("text") or "").strip().lower()
    if text in ("/status", "/start"):
        pending = get_pending()
        lines = [f"<b>⚡ ChartVisualizer Bot</b>"]
        lines.append(f"Status: {'🟢 activ' if _enabled else '🔴 inactiv'}")
        lines.append(f"Mod: {'📱 aprobare' if _require_approval else '🔔 notificare'}")
        if pending:
            lines.append(f"\n⏳ <b>{len(pending)} trade(uri) in asteptare:</b>")
            for t, v in pending.items():
                lines.append(f"  • {v['signal']} {v['symbol']} conf={v['confidence']:.0f}%")
        else:
            lines.append("\n✅ Niciun trade in asteptare.")
        send_message("\n".join(lines))
    elif text == "/help":
        send_message(
            "<b>Comenzi disponibile:</b>\n"
            "/status — trade-uri in asteptare\n"
            "/help — aceasta lista"
        )


# ── Polling thread ────────────────────────────────────────────────────────────

def _poll_once() -> None:
    global _poll_offset
    result = _api("getUpdates", json_data={
        "offset": _poll_offset,
        "timeout": 0,
        "allowed_updates": ["callback_query", "message"],
    }, timeout=8)
    if not result or not result.get("ok"):
        return
    for upd in result.get("result", []):
        _poll_offset = upd["update_id"] + 1
        cb = upd.get("callback_query")
        if cb:
            _process_callback(
                cb.get("data", ""),
                cb.get("id", ""),
                cb.get("from", {}).get("first_name", ""),
            )
        msg = upd.get("message")
        if msg:
            _process_message(msg)


def _polling_loop() -> None:
    log.info("Telegram polling thread pornit")
    while True:
        try:
            if _enabled and _bot_token:
                _poll_once()
        except Exception as e:
            log.debug(f"Telegram poll: {e}")
        time.sleep(2)


def _bridge_polling_active() -> bool:
    """
    True daca ClaudeTelegramBridge ruleaza (are bridge.lock cu PID viu).
    In acest caz nu pornim polling propriu pentru a evita 409 Conflict.
    """
    # Env var (transmis explicit cand bridge porneste ChartVisualizer)
    if os.environ.get("CHARTVIS_DISABLE_TG_POLLING") == "1":
        return True
    # Lock file (cazul cand ChartVis e pornit manual peste un bridge deja activ)
    lock = os.path.join(os.path.dirname(__file__), "..", "ClaudeTelegramBridge", "bridge.lock")
    lock = os.path.abspath(lock)
    if not os.path.exists(lock):
        return False
    try:
        with open(lock, encoding="utf-8") as f:
            pid = int(f.read().strip())
        # Verifica PID
        try:
            import psutil  # type: ignore
            return psutil.pid_exists(pid)
        except ImportError:
            # Fara psutil: presupune lock valid (conservator)
            return True
    except Exception:
        return False


def start_polling() -> None:
    global _polling_started
    if _polling_started:
        return
    if _bridge_polling_active():
        log.info("ClaudeTelegramBridge activ — skip Telegram polling local")
        return
    _polling_started = True
    t = threading.Thread(target=_polling_loop, daemon=True, name="tg-poll")
    t.start()
    log.info("Telegram polling pornit")


# ── Screenshot grafic ─────────────────────────────────────────────────────────

def capture_chart(fig) -> bytes | None:
    """
    Genereaza PNG din figura Plotly.
    Necesita: pip install kaleido
    Daca kaleido nu e instalat, returneaza None (trimite text fara poza).
    """
    try:
        import plotly.io as pio
        img = pio.to_image(fig, format="png", width=1280, height=720, scale=1.5)
        return img
    except Exception as e:
        log.debug(f"capture_chart: {e} (instaleaza kaleido pentru poze)")
        return None


# ── Test conexiune ────────────────────────────────────────────────────────────

def test_connection() -> tuple[bool, str]:
    if not _bot_token:
        return False, "Bot token lipsa — adauga in Settings"
    r = _api("getMe")
    if not r or not r.get("ok"):
        return False, f"Token invalid: {r}"
    bot_name = r["result"].get("username", "?")
    if not _chat_id:
        return False, "Chat ID lipsa — trimite /start botului @" + bot_name
    r2 = _api("sendMessage", json_data={
        "chat_id": _chat_id,
        "text": (
            f"✅ <b>ChartVisualizer conectat!</b>\n\n"
            f"Bot: @{bot_name}\n"
            f"Mod: {'📱 Aprobare trade' if _require_approval else '🔔 Notificare simpla'}\n\n"
            f"Vei primi semnale de tranzactionare aici.\n"
            f"Comenzi: /status, /help"
        ),
        "parse_mode": "HTML",
    })
    if not r2 or not r2.get("ok"):
        err = (r2 or {}).get("description", str(r2))
        return False, f"Chat ID invalid sau bot neautorizat: {err}"
    return True, f"OK — bot @{bot_name} functioneaza"


# ── Init ──────────────────────────────────────────────────────────────────────
load_config()
