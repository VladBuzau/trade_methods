"""
ClaudeTelegramBridge — Telegram -> VSCode Claude Code injector.

Asculta mesajele Telegram, le injecteaza in fereastra VSCode care are
chat-ul Claude Code deschis (folosind clipboard + Ctrl+V + Enter).

Credentialele Telegram (bot_token, chat_id) se citesc din
ChartVisualizer/config.json — nu trebuie duplicat nimic.

Marker:
  Cand un mesaj e injectat din Telegram, se creeaza fisierul `pending.flag`.
  response_hook.py il citeste si stie ca raspunsul Claude trebuie trimis
  inapoi pe Telegram. Apoi sterge marker-ul.
"""
from __future__ import annotations
import os, sys, json, time, threading, logging, traceback, atexit, signal, hashlib
from pathlib import Path

import requests
import pyautogui
import pyperclip
import pygetwindow as gw
import subprocess

try:
    import psutil  # type: ignore
    _HAS_PSUTIL = True
except ImportError:
    psutil = None  # type: ignore
    _HAS_PSUTIL = False

# ── Logging ──────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)
log = logging.getLogger("bridge")

# pyautogui: nu opri scriptul daca mouse-ul ajunge in colt
pyautogui.FAILSAFE = False
pyautogui.PAUSE = 0.05

# ── Paths ────────────────────────────────────────────────────────────────────
HERE = Path(__file__).resolve().parent
CONFIG_PATH = HERE / "config.json"
PENDING_FLAG = HERE / "pending.flag"   # legacy, mentinut doar pentru cleanup
LAST_SENT_FLAG = HERE / "last_sent.txt"
APPROVALS_DIR = HERE / "approvals"
BRIDGE_LOCK = HERE / "bridge.lock"     # PID — notifier.py il citeste si skip polling
INJECTED_DIR = HERE / "injected"       # injected/<hash>.tg per mesaj injectat

# Marker invizibil (LEGACY, mentinut doar pentru compatibilitate).
# Acum semnalul "vine de pe Telegram" e fisierul injected/<hash>.tg.
TELEGRAM_MARKER = "​​​"


def _hash_text(text: str) -> str:
    """Hash normalizat (strip whitespace) — match exact intre bridge si hook."""
    norm = (text or "").strip()
    return hashlib.sha256(norm.encode("utf-8")).hexdigest()[:16]


def _record_injection(text: str) -> Path:
    """
    Scrie injected/<hash>.tg ca semnal autoritar pentru Telegram-origin.
    Fisierul contine JSON cu metadata pentru heartbeat:
      ts             - timestamp injectie
      tools_run      - contor tool calls (incrementat de progress_hook.py)
      tools_summary  - ultimele N tool names (debug)
      last_heartbeat - ts cand bridge-ul a trimis ultimul heartbeat
    """
    INJECTED_DIR.mkdir(parents=True, exist_ok=True)
    f = INJECTED_DIR / f"{_hash_text(text)}.tg"
    now = int(time.time())
    f.write_text(json.dumps({
        "ts": now,
        "tools_run": 0,
        "tools_summary": [],
        "last_heartbeat": now,
    }), encoding="utf-8")
    return f


# Heartbeat: bridge informeaza utilizatorul ce face Claude pe Telegram
# in timp ce executa task-uri lungi.
HEARTBEAT_INTERVAL_S = 30


def _heartbeat_loop() -> None:
    """
    Thread de fundal care, la fiecare 5s, scaneaza injected/*.tg.
    Daca un mesaj e nerezolvat de mai mult de HEARTBEAT_INTERVAL_S si nu am
    trimis recent un heartbeat -> trimite mini-update pe Telegram.
    """
    log.info("Heartbeat thread pornit (interval %ss).", HEARTBEAT_INTERVAL_S)
    while True:
        try:
            now = time.time()
            if INJECTED_DIR.exists():
                for f in INJECTED_DIR.glob("*.tg"):
                    try:
                        info = json.loads(f.read_text(encoding="utf-8"))
                    except Exception:
                        continue
                    if not isinstance(info, dict):
                        continue
                    last_hb = info.get("last_heartbeat", info.get("ts", 0))
                    if now - last_hb < HEARTBEAT_INTERVAL_S:
                        continue
                    tools = info.get("tools_run", 0)
                    summary = info.get("tools_summary", [])
                    last_tool = summary[-1] if summary else "(thinking)"
                    elapsed = int(now - info.get("ts", now))
                    msg = (
                        f"⏳ <b>Lucrez...</b> {elapsed}s elapsed\n"
                        f"Tool #{tools}: <code>{last_tool}</code>"
                    )
                    try:
                        tg_send(msg)
                    except Exception:
                        pass
                    info["last_heartbeat"] = now
                    try:
                        f.write_text(json.dumps(info), encoding="utf-8")
                    except Exception:
                        pass
        except Exception as e:
            log.debug(f"heartbeat: {e}")
        time.sleep(5)


def _cleanup_old_injections(max_age_s: int = 3600) -> None:
    """Sterge inject files mai vechi de max_age_s secunde."""
    try:
        if not INJECTED_DIR.exists():
            return
        cutoff = time.time() - max_age_s
        for f in INJECTED_DIR.glob("*.tg"):
            try:
                if f.stat().st_mtime < cutoff:
                    f.unlink()
            except Exception:
                pass
    except Exception:
        pass

# Credentialele Telegram vin din ChartVisualizer
CHARTVIS_DIR = HERE.parent / "ChartVisualizer"
CHARTVIS_CONFIG = CHARTVIS_DIR / "config.json"
CHARTVIS_APP = CHARTVIS_DIR / "app.py"
CHARTVIS_START_BAT = CHARTVIS_DIR / "start.bat"
CHARTVIS_TUNNEL_LOG = CHARTVIS_DIR / "cloudflare_tunnel.log"
CHARTVIS_LOCAL_URL = "http://localhost:5004"

# Regex pentru URL-ul public Cloudflare
import re as _re
_CF_URL_RE = _re.compile(r"https://[a-zA-Z0-9\-\.]+\.trycloudflare\.com")


# ── Config loading ───────────────────────────────────────────────────────────
def load_bridge_config() -> dict:
    if CONFIG_PATH.exists():
        with open(CONFIG_PATH, encoding="utf-8") as f:
            return json.load(f)
    return {}


def load_telegram_creds() -> tuple[str, str]:
    """Citeste bot_token si chat_id din ChartVisualizer/config.json."""
    if not CHARTVIS_CONFIG.exists():
        raise FileNotFoundError(
            f"Nu gasesc {CHARTVIS_CONFIG}. "
            "Configureaza Telegram in ChartVisualizer mai intai."
        )
    with open(CHARTVIS_CONFIG, encoding="utf-8") as f:
        cfg = json.load(f)
    tg = cfg.get("telegram", {})
    token = tg.get("bot_token", "").strip()
    chat_id = str(tg.get("chat_id", "")).strip()
    if not token or not chat_id:
        raise ValueError(
            "bot_token sau chat_id lipsesc din ChartVisualizer/config.json. "
            "Deschide ChartVisualizer -> Settings -> Telegram si seteaza-le."
        )
    return token, chat_id


# ── Telegram API ─────────────────────────────────────────────────────────────
_token: str = ""
_chat_id: str = ""
_poll_offset: int | None = None


def tg_api(method: str, json_data: dict | None = None, timeout: int = 12) -> dict | None:
    url = f"https://api.telegram.org/bot{_token}/{method}"
    try:
        r = requests.post(url, json=json_data, timeout=timeout)
        r.raise_for_status()
        return r.json()
    except Exception as e:
        log.warning(f"Telegram {method}: {e}")
        return None


def tg_send(text: str, parse_mode: str = "HTML") -> None:
    # Sparge mesajele lungi (Telegram max 4096)
    MAX = 4000
    chunks = [text[i : i + MAX] for i in range(0, len(text), MAX)] or [""]
    for chunk in chunks:
        tg_api(
            "sendMessage",
            json_data={
                "chat_id": _chat_id,
                "text": chunk,
                "parse_mode": parse_mode,
                "disable_web_page_preview": True,
            },
        )


# ── VSCode injection ─────────────────────────────────────────────────────────
def find_vscode_window(title_hint: str = "Visual Studio Code") -> object | None:
    """Gaseste prima fereastra VSCode (titlul contine 'Visual Studio Code')."""
    wins = [w for w in gw.getAllWindows() if title_hint in w.title]
    if not wins:
        return None
    # Prefer ferestre vizibile + cu suprafata > 0
    wins = [w for w in wins if w.width > 100 and w.height > 100]
    return wins[0] if wins else None


def inject_into_vscode(text: str, cfg: dict) -> tuple[bool, str]:
    """
    Injecteaza textul in fereastra VSCode activa.
    Pasi: focus window -> (optional) focus chat shortcut -> paste -> Enter.
    """
    title_hint = cfg.get("vscode_title_hint", "Visual Studio Code")
    focus_chat_shortcut = cfg.get(
        "focus_chat_shortcut", ""
    )  # ex: "ctrl+alt+c" daca ai setat un keybinding
    pre_delay = float(cfg.get("pre_inject_delay_s", 0.4))
    post_focus_delay = float(cfg.get("post_focus_delay_s", 0.3))

    win = find_vscode_window(title_hint)
    if not win:
        return False, f"Nu gasesc nicio fereastra VSCode (cautam: '{title_hint}')."

    try:
        # Activeaza fereastra (poate arunca daca minimizata)
        if win.isMinimized:
            win.restore()
        try:
            win.activate()
        except Exception:
            # Fallback: click pe titlebar
            try:
                pyautogui.click(win.left + 50, win.top + 10)
            except Exception:
                pass
        time.sleep(pre_delay)

        # Optional: trimite shortcut pentru focus pe chat
        if focus_chat_shortcut:
            keys = [k.strip().lower() for k in focus_chat_shortcut.split("+") if k.strip()]
            if keys:
                pyautogui.hotkey(*keys)
                time.sleep(post_focus_delay)

        # Pune in clipboard si paste
        pyperclip.copy(text)
        pyautogui.hotkey("ctrl", "v")
        time.sleep(0.20)
        # Inchide orice popup (slash menu, file mention, autocomplete) deschis
        # de caractere precum '/' sau '@' din text — Escape doar inchide popup,
        # nu deselectaza input-ul.
        pyautogui.press("escape")
        time.sleep(0.10)
        # Trimite. Folosesc end+enter ca sa fiu sigur ca focus-ul e pe sfarsit.
        pyautogui.press("end")
        time.sleep(0.05)
        pyautogui.press("enter")

        return True, f"OK - injectat in '{win.title}'"
    except Exception as e:
        tb = traceback.format_exc()
        log.error(f"Injectie esuata: {e}\n{tb}")
        return False, f"Eroare injectie: {e}"


# ── Trimite taste catre VSCode (Escape pentru abort, etc.) ───────────────────
def send_keys_to_vscode(keys: list[str], cfg: dict) -> tuple[bool, str]:
    """
    Focuseaza fereastra VSCode si trimite combinatia de taste.
    keys: lista pentru pyautogui.hotkey, ex ['escape'] sau ['ctrl', 'l'].
    """
    title_hint = cfg.get("vscode_title_hint", "Visual Studio Code")
    pre_delay = float(cfg.get("pre_inject_delay_s", 0.4))

    win = find_vscode_window(title_hint)
    if not win:
        return False, f"Nu gasesc fereastra VSCode (cautam: '{title_hint}')."
    try:
        if win.isMinimized:
            win.restore()
        try:
            win.activate()
        except Exception:
            try:
                pyautogui.click(win.left + 50, win.top + 10)
            except Exception:
                pass
        time.sleep(pre_delay)
        if len(keys) == 1:
            pyautogui.press(keys[0])
        else:
            pyautogui.hotkey(*keys)
        return True, f"Trimis [{'+'.join(keys)}] -> '{win.title}'"
    except Exception as e:
        return False, f"Eroare trimitere taste: {e}"


# ── Project management (ChartVisualizer) ─────────────────────────────────────
def _find_chartvis_processes() -> list:
    """Gaseste procesele Python care ruleaza ChartVisualizer/app.py."""
    if not _HAS_PSUTIL:
        return []
    found = []
    app_path_low = str(CHARTVIS_APP).lower().replace("\\", "/")
    for p in psutil.process_iter(["pid", "name", "cmdline", "cwd"]):
        try:
            name = (p.info.get("name") or "").lower()
            if "python" not in name:
                continue
            cmdline = p.info.get("cmdline") or []
            cmdline_str = " ".join(cmdline).lower().replace("\\", "/")
            if "app.py" in cmdline_str and "chartvisualizer" in cmdline_str:
                found.append(p)
                continue
            # Fallback: cwd-ul e ChartVisualizer si argv contine app.py
            cwd = (p.info.get("cwd") or "").lower().replace("\\", "/")
            if "chartvisualizer" in cwd and any("app.py" in c.lower() for c in cmdline):
                found.append(p)
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue
    return found


def _cloudflare_url(timeout_s: float = 0.0) -> str | None:
    """
    Citeste cloudflare_tunnel.log si extrage URL-ul public.
    timeout_s > 0 = poll cu interval 1s pana cand apare URL (sau timeout).
    """
    deadline = time.time() + timeout_s
    while True:
        if CHARTVIS_TUNNEL_LOG.exists():
            try:
                content = CHARTVIS_TUNNEL_LOG.read_text(encoding="utf-8", errors="ignore")
                m = _CF_URL_RE.search(content)
                if m:
                    return m.group(0)
            except Exception:
                pass
        if time.time() >= deadline:
            return None
        time.sleep(1.0)


def project_start() -> tuple[bool, str]:
    """Porneste ChartVisualizer (via start.bat, detached)."""
    if not CHARTVIS_START_BAT.exists():
        return False, f"Nu gasesc {CHARTVIS_START_BAT}"
    procs = _find_chartvis_processes()
    if procs:
        pids = ", ".join(str(p.info["pid"]) for p in procs)
        return False, f"Deja ruleaza (PID: {pids})"
    # Sterge log-ul vechi ca sa nu citim URL stale
    try:
        CHARTVIS_TUNNEL_LOG.unlink(missing_ok=True)
    except Exception:
        pass
    try:
        DETACHED = 0x00000008  # DETACHED_PROCESS
        CREATE_NEW = 0x00000200  # CREATE_NEW_PROCESS_GROUP
        # Dezactiveaza polling Telegram in notifier.py (bridge-ul deja polueaza)
        env = os.environ.copy()
        env["CHARTVIS_DISABLE_TG_POLLING"] = "1"
        subprocess.Popen(
            ["cmd.exe", "/c", "start", "", str(CHARTVIS_START_BAT)],
            cwd=str(CHARTVIS_DIR),
            creationflags=DETACHED | CREATE_NEW,
            shell=False,
            env=env,
        )
        return True, "Pornit (start.bat). Astept URL public..."
    except Exception as e:
        return False, f"Eroare pornire: {e}"


def project_stop() -> tuple[bool, str]:
    """Opreste procesele ChartVisualizer."""
    if not _HAS_PSUTIL:
        return False, "psutil nu e instalat — pip install psutil"
    procs = _find_chartvis_processes()
    if not procs:
        return True, "Niciun proces de oprit (nu rula nimic)."
    killed = []
    for p in procs:
        try:
            pid = p.info["pid"]
            p.terminate()
            try:
                p.wait(timeout=3)
            except psutil.TimeoutExpired:
                p.kill()
            killed.append(str(pid))
        except (psutil.NoSuchProcess, psutil.AccessDenied) as e:
            killed.append(f"err-{e}")
    return True, f"Oprit PID: {', '.join(killed)}"


def project_restart() -> tuple[bool, str]:
    """Restart: stop + start."""
    ok_stop, msg_stop = project_stop()
    time.sleep(1.5)
    ok_start, msg_start = project_start()
    full = f"Stop: {msg_stop}\nStart: {msg_start}"
    return (ok_stop and ok_start), full


def project_status() -> str:
    """Status ChartVisualizer + verificare port 5004 + URL Cloudflare."""
    lines = []
    if not _HAS_PSUTIL:
        lines.append("⚠️ psutil nu e instalat")
    else:
        procs = _find_chartvis_processes()
        if procs:
            for p in procs:
                try:
                    uptime = int(time.time() - p.create_time())
                    lines.append(f"🟢 PID {p.info['pid']} — uptime {uptime}s")
                except Exception:
                    lines.append(f"🟢 PID {p.info['pid']}")
        else:
            lines.append("🔴 Nu ruleaza")
    try:
        r = requests.get(CHARTVIS_LOCAL_URL + "/", timeout=2)
        lines.append(f"💻 Local: {CHARTVIS_LOCAL_URL}  (HTTP {r.status_code})")
    except Exception:
        lines.append(f"💻 Local: {CHARTVIS_LOCAL_URL}  (inaccesibil)")
    url = _cloudflare_url()  # fara poll
    if url:
        lines.append(f"🌐 Public: {url}")
    else:
        lines.append("🌐 Public: (URL Cloudflare indisponibil)")
    return "\n".join(lines)


# ── Mesaj handling ───────────────────────────────────────────────────────────
def handle_text_message(text: str, cfg: dict) -> None:
    text = (text or "").strip()
    if not text:
        return

    # Comenzi bridge (rezervate)
    low = text.lower()
    if low in ("/ping_bridge", "/bridge_status"):
        win = find_vscode_window(cfg.get("vscode_title_hint", "Visual Studio Code"))
        win_state = f"OK ('{win.title}')" if win else "NU EXISTA"
        try:
            pending_approvals = len(list(APPROVALS_DIR.glob("*.req")))
        except Exception:
            pending_approvals = 0
        tg_send(
            f"<b>ClaudeTelegramBridge</b>\n"
            f"VSCode: {win_state}\n"
            f"Aprobari in asteptare: {pending_approvals}\n"
            f"Mod: marker per mesaj (zero-width)"
        )
        return

    if low in ("/abort", "/stop_claude", "/cancel"):
        ok, msg = send_keys_to_vscode(["escape"], cfg)
        # Auto-deny orice aprobare in asteptare (claude e oprit oricum)
        cleaned = 0
        try:
            for f in APPROVALS_DIR.glob("*.req"):
                token = f.stem
                resp = APPROVALS_DIR / f"{token}.resp"
                resp.write_text(json.dumps({"decision": "deny",
                                            "by": "auto-abort",
                                            "ts": int(time.time())}),
                                encoding="utf-8")
                cleaned += 1
        except Exception:
            pass
        suffix = f" (auto-respins {cleaned} aprobare/aprobari)" if cleaned else ""
        tg_send(f"{'🛑' if ok else '❌'} {msg}{suffix}")
        return

    if low == "/reset_bridge":
        cleaned = 0
        try:
            PENDING_FLAG.unlink(missing_ok=True)
            for f in APPROVALS_DIR.glob("*.req"):
                f.unlink(missing_ok=True)
                cleaned += 1
            for f in APPROVALS_DIR.glob("*.resp"):
                f.unlink(missing_ok=True)
                cleaned += 1
        except Exception:
            pass
        tg_send(f"Reset: {cleaned} fisiere sterse din approvals/.")
        return

    if low.startswith("/help_bridge") or low == "/help":
        tg_send(
            "<b>ClaudeTelegramBridge</b>\n"
            "Orice text trimis (care nu e comanda) e injectat in chatul "
            "Claude Code din VSCode.\n\n"
            "<b>Proiect ChartVisualizer:</b>\n"
            "  /start — porneste proiectul + returneaza URL public\n"
            "  /stop — opreste proiectul\n"
            "  /restart — stop + start + URL nou\n"
            "  /status — stare + URL public + port\n"
            "  /link — doar URL-ul public Cloudflare\n\n"
            "<b>Claude:</b>\n"
            "  /abort — opreste raspunsul Claude in curs (Escape) +\n"
            "          auto-respinge aprobarile in asteptare\n\n"
            "<b>Bridge:</b>\n"
            "  /bridge_status — stare bridge + fereastra VSCode\n"
            "  /reset_bridge — cleanup approvals/\n"
            "  /help_bridge — acest mesaj"
        )
        return

    # ── Comenzi ChartVisualizer ──
    if low in ("/start", "/proj_start"):
        tg_send("⏳ Pornesc ChartVisualizer + tunel Cloudflare...")
        ok, msg = project_start()
        if not ok:
            # Daca deja ruleaza, returneaza URL-ul existent
            tg_send(f"⚠️ {msg}")
            url = _cloudflare_url(timeout_s=2)
            if url:
                tg_send(f"🌐 Public: {url}\n💻 Local: {CHARTVIS_LOCAL_URL}")
            return
        tg_send(f"✅ {msg}")
        url = _cloudflare_url(timeout_s=45)
        if url:
            tg_send(
                f"🌐 <b>Public:</b> {url}\n"
                f"💻 <b>Local:</b> {CHARTVIS_LOCAL_URL}\n"
                f"📊 <b>AutoTrader:</b> {url}/autotrader"
            )
        else:
            tg_send(
                "⚠️ URL Cloudflare nu a aparut in 45s.\n"
                "Posibil: cloudflared nu e instalat sau lent. /status pentru verificare."
            )
        return

    if low in ("/stop", "/proj_stop"):
        tg_send("⏳ Opresc ChartVisualizer...")
        ok, msg = project_stop()
        tg_send(f"{'✅' if ok else '❌'} {msg}")
        return

    if low in ("/restart", "/proj_restart"):
        tg_send("⏳ Restart ChartVisualizer...")
        ok, msg = project_restart()
        tg_send(f"{'✅' if ok else '❌'} <pre>{msg}</pre>")
        if ok:
            url = _cloudflare_url(timeout_s=45)
            if url:
                tg_send(
                    f"🌐 <b>Public:</b> {url}\n"
                    f"💻 <b>Local:</b> {CHARTVIS_LOCAL_URL}\n"
                    f"📊 <b>AutoTrader:</b> {url}/autotrader"
                )
            else:
                tg_send("⚠️ URL Cloudflare nu a aparut in 45s. /status pentru verificare.")
        return

    if low in ("/status", "/proj_status"):
        tg_send(f"<b>ChartVisualizer status</b>\n{project_status()}")
        return

    if low in ("/link", "/url"):
        url = _cloudflare_url(timeout_s=2)
        if url:
            tg_send(
                f"🌐 <b>Public:</b> {url}\n"
                f"💻 <b>Local:</b> {CHARTVIS_LOCAL_URL}\n"
                f"📊 <b>AutoTrader:</b> {url}/autotrader"
            )
        else:
            tg_send("❌ URL Cloudflare indisponibil (proiectul nu ruleaza sau cloudflared n-a pornit)")
        return

    # Inregistreaza injectia ca fisier (semnal autoritar pentru hook-uri).
    # Hash-ul textului = id-ul mesajului; hook-urile cauta injected/<hash>.tg.
    _record_injection(text)
    _cleanup_old_injections()

    ok, msg = inject_into_vscode(text, cfg)
    if ok:
        tg_send(f"📨 Trimis in Claude. Astept raspuns...")
    else:
        # Daca injectia esueaza, sterge inject file ca sa nu ramana stale
        try:
            (INJECTED_DIR / f"{_hash_text(text)}.tg").unlink(missing_ok=True)
        except Exception:
            pass
        tg_send(f"❌ {msg}")


# ── Callback queries (butoane aprobare) ──────────────────────────────────────
def handle_callback_query(cb: dict) -> None:
    """
    Proceseaza apasarile pe butoane (aprobare/respingere tool calls).
    Callback data: 'tool_approve_<token>' sau 'tool_deny_<token>'.
    """
    data = cb.get("data", "") or ""
    cb_id = cb.get("id", "")
    user = cb.get("from", {}).get("first_name", "")
    chat = cb.get("message", {}).get("chat", {})

    # Filtru chat
    if str(chat.get("id", "")) != _chat_id:
        tg_api("answerCallbackQuery", json_data={"callback_query_id": cb_id,
                                                  "text": "Chat neautorizat."})
        return

    decision: str | None = None
    token: str = ""
    if data.startswith("tool_approve_"):
        decision = "approve"
        token = data[len("tool_approve_"):]
    elif data.startswith("tool_deny_"):
        decision = "deny"
        token = data[len("tool_deny_"):]
    else:
        tg_api("answerCallbackQuery", json_data={"callback_query_id": cb_id,
                                                  "text": "Callback necunoscut."})
        return

    # Scrie raspunsul pentru approval_hook.py
    APPROVALS_DIR.mkdir(parents=True, exist_ok=True)
    resp_path = APPROVALS_DIR / f"{token}.resp"
    try:
        resp_path.write_text(json.dumps({
            "decision": decision,
            "by": user,
            "ts": int(time.time()),
        }), encoding="utf-8")
    except Exception as e:
        log.error(f"Nu pot scrie {resp_path}: {e}")
        tg_api("answerCallbackQuery", json_data={"callback_query_id": cb_id,
                                                  "text": f"Eroare salvare: {e}"})
        return

    # Confirma in Telegram (popup discret + edit mesaj original)
    text_popup = "✅ Aprobat" if decision == "approve" else "❌ Respins"
    tg_api("answerCallbackQuery", json_data={"callback_query_id": cb_id,
                                              "text": text_popup})

    # Editeaza mesajul original ca sa scoata butoanele
    msg = cb.get("message", {})
    chat_id = msg.get("chat", {}).get("id")
    msg_id = msg.get("message_id")
    if chat_id and msg_id:
        original_text = msg.get("text", "") or ""
        suffix = f"\n\n{text_popup} de {user or 'tine'}"
        tg_api("editMessageText", json_data={
            "chat_id": chat_id,
            "message_id": msg_id,
            "text": (original_text + suffix)[:4000],
            "parse_mode": "HTML",
        })


# ── Polling Telegram ─────────────────────────────────────────────────────────
def poll_once(cfg: dict) -> None:
    global _poll_offset
    result = tg_api(
        "getUpdates",
        json_data={
            "offset": _poll_offset,
            "timeout": 0,
            "allowed_updates": ["message", "callback_query"],
        },
        timeout=8,
    )
    if not result or not result.get("ok"):
        return
    for upd in result.get("result", []):
        _poll_offset = upd["update_id"] + 1

        # Callback query (apasare buton)
        cb = upd.get("callback_query")
        if cb:
            try:
                handle_callback_query(cb)
            except Exception as e:
                log.error(f"callback: {e}")
            continue

        # Mesaj text
        msg = upd.get("message") or {}
        if str(msg.get("chat", {}).get("id", "")) != _chat_id:
            log.info(f"Ignor mesaj de la chat_id strain: {msg.get('chat', {}).get('id')}")
            continue
        text = msg.get("text") or ""
        if text:
            handle_text_message(text, cfg)


def polling_loop(cfg: dict) -> None:
    log.info("Polling Telegram pornit.")
    while True:
        try:
            poll_once(cfg)
        except Exception as e:
            log.debug(f"poll: {e}")
        time.sleep(1.5)


# ── Main ─────────────────────────────────────────────────────────────────────
def main() -> int:
    global _token, _chat_id
    try:
        _token, _chat_id = load_telegram_creds()
    except Exception as e:
        log.error(str(e))
        print(f"\n[EROARE] {e}\n", file=sys.stderr)
        return 1

    cfg = load_bridge_config()

    # Test conexiune
    me = tg_api("getMe")
    if not me or not me.get("ok"):
        log.error("Token Telegram invalid.")
        return 2
    bot_name = me["result"].get("username", "?")
    log.info(f"Bot conectat: @{bot_name}")

    # Mesaj de pornire
    tg_send(
        f"🟢 <b>ClaudeTelegramBridge online</b>\n"
        f"Bot: @{bot_name}\n"
        f"Scrie orice mesaj — va fi injectat in VSCode Claude Code.\n"
        f"Comenzi: /help_bridge"
    )

    # Marker curatat la pornire
    try:
        PENDING_FLAG.unlink(missing_ok=True)
    except Exception:
        pass

    # Curata cereri/raspunsuri vechi de aprobare
    try:
        APPROVALS_DIR.mkdir(parents=True, exist_ok=True)
        for f in APPROVALS_DIR.glob("*.req"):
            f.unlink(missing_ok=True)
        for f in APPROVALS_DIR.glob("*.resp"):
            f.unlink(missing_ok=True)
    except Exception as e:
        log.warning(f"cleanup approvals: {e}")

    # Scrie lock file cu PID — notifier.py il verifica ca sa nu polueze
    # acelasi bot Telegram (altfel 409 Conflict).
    try:
        BRIDGE_LOCK.write_text(str(os.getpid()), encoding="utf-8")
        log.info(f"Lock scris: {BRIDGE_LOCK} (PID {os.getpid()})")
    except Exception as e:
        log.warning(f"Nu pot scrie {BRIDGE_LOCK}: {e}")

    def _cleanup_lock(*_args):
        try:
            BRIDGE_LOCK.unlink(missing_ok=True)
            log.info("Lock sters.")
        except Exception:
            pass

    atexit.register(_cleanup_lock)
    try:
        signal.signal(signal.SIGINT, lambda *_: (_cleanup_lock(), sys.exit(0)))
        signal.signal(signal.SIGTERM, lambda *_: (_cleanup_lock(), sys.exit(0)))
    except Exception:
        pass

    # Heartbeat thread — informari periodice pe Telegram in timpul task-urilor lungi
    threading.Thread(target=_heartbeat_loop, daemon=True, name="heartbeat").start()

    try:
        polling_loop(cfg)
    except KeyboardInterrupt:
        log.info("Oprit.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
