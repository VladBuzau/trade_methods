"""
approval_hook.py — PreToolUse hook pentru Claude Code.

Inregistrat in ~/.claude/settings.json sub "hooks.PreToolUse".

DETECTIE ORIGINE (per mesaj, nu global):
  Bridge-ul (bridge.py) adauga TELEGRAM_MARKER (3 zero-width spaces)
  la finalul fiecarui mesaj injectat. Hook-ul:

  1. Cite ultimul mesaj 'user' din transcript_path.
  2. Daca contine TELEGRAM_MARKER -> mesaj de pe Telegram -> aproba
     prin Telegram cu butoane ✅/❌.
  3. Daca NU contine marker -> mesaj scris direct in VSCode -> output
     {permissionDecision: "ask"} ca sa cada inapoi pe UI-ul nativ VSCode.

  TOOL SAFE (Edit/Write/Read/...) -> auto-allow indiferent de origine.
  TOOL RISKY (Bash, PowerShell, WebFetch, WebSearch) -> aprobare conform
  originii (Telegram sau VSCode).
"""
from __future__ import annotations
import sys, os, json, time, uuid, hashlib
from pathlib import Path

import requests

HERE = Path(__file__).resolve().parent
PENDING_FLAG = HERE / "pending.flag"   # legacy, ignorat
APPROVALS_DIR = HERE / "approvals"
INJECTED_DIR = HERE / "injected"       # semnal autoritar "vine de pe Telegram"
AUTO_APPROVE_FLAG = HERE / "auto_approve.flag"
CHARTVIS_CONFIG = HERE.parent / "ChartVisualizer" / "config.json"

# Marker invizibil (LEGACY). Sursa autoritara e fisierul injected/<hash>.tg
# scris de bridge.py inainte de fiecare injectie.
TELEGRAM_MARKER = "​​​"


def _hash_text(text: str) -> str:
    """Hash normalizat (strip whitespace). Trebuie identic cu cel din bridge.py."""
    norm = (text or "").strip()
    return hashlib.sha256(norm.encode("utf-8")).hexdigest()[:16]

TIMEOUT_S = 120          # cat timp asteapta raspunsul
POLL_INTERVAL_S = 0.5

# Tool-uri permise automat (modificari de cod = ne-distructive per ack user)
AUTO_ALLOW_TOOLS = {
    "Read", "Write", "Edit", "NotebookEdit",
    "Glob", "Grep", "TodoWrite",
    "AskUserQuestion",
}

# Tool-uri care merita aprobare explicita pe Telegram
RISKY_TOOLS = {
    "Bash", "WebFetch", "WebSearch", "PowerShell",
}


def _allow() -> None:
    sys.exit(0)


def _deny(reason: str) -> None:
    print(reason, file=sys.stderr)
    sys.exit(2)


def _load_telegram_creds() -> tuple[str, str] | None:
    if not CHARTVIS_CONFIG.exists():
        return None
    try:
        with open(CHARTVIS_CONFIG, encoding="utf-8") as f:
            cfg = json.load(f)
        tg = cfg.get("telegram", {})
        token = tg.get("bot_token", "").strip()
        chat_id = str(tg.get("chat_id", "")).strip()
        if not token or not chat_id:
            return None
        return token, chat_id
    except Exception:
        return None


def _tg_send_approval(token: str, chat_id: str, approval_token: str,
                      tool_name: str, summary: str) -> bool:
    """Trimite cerere de aprobare cu butoane inline."""
    url = f"https://api.telegram.org/bot{token}/sendMessage"
    text = (
        f"⚠️ <b>Aprobare necesara</b>\n\n"
        f"🔧 Tool: <code>{tool_name}</code>\n"
        f"📋 Detalii:\n<pre>{summary[:1500]}</pre>\n"
        f"<i>Expira in {TIMEOUT_S}s</i>"
    )
    markup = {
        "inline_keyboard": [[
            {"text": "✅ Aproba", "callback_data": f"tool_approve_{approval_token}"},
            {"text": "❌ Respinge", "callback_data": f"tool_deny_{approval_token}"},
        ]]
    }
    try:
        r = requests.post(url, json={
            "chat_id": chat_id,
            "text": text,
            "parse_mode": "HTML",
            "reply_markup": markup,
            "disable_web_page_preview": True,
        }, timeout=10)
        return r.ok
    except Exception:
        return False


def _tg_send_text(token: str, chat_id: str, text: str) -> None:
    try:
        requests.post(
            f"https://api.telegram.org/bot{token}/sendMessage",
            json={"chat_id": chat_id, "text": text, "parse_mode": "HTML"},
            timeout=10,
        )
    except Exception:
        pass


def _summarize_tool_input(tool_name: str, tool_input: dict) -> str:
    """Genereaza un rezumat scurt al actiunii."""
    if not isinstance(tool_input, dict):
        return str(tool_input)[:500]

    if tool_name == "Bash" or tool_name == "PowerShell":
        cmd = tool_input.get("command", "")
        desc = tool_input.get("description", "")
        return f"$ {cmd}\n\n({desc})" if desc else f"$ {cmd}"

    if tool_name == "WebFetch":
        url = tool_input.get("url", "")
        prompt = tool_input.get("prompt", "")[:200]
        return f"URL: {url}\nPrompt: {prompt}"

    if tool_name == "WebSearch":
        q = tool_input.get("query", "")
        return f"Query: {q}"

    # generic
    try:
        return json.dumps(tool_input, indent=2)[:1000]
    except Exception:
        return str(tool_input)[:500]


def _extract_user_text(content) -> str:
    """Extrage textul dintr-un mesaj user (poate fi string sau lista de blocks)."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for block in content:
            if not isinstance(block, dict):
                continue
            if block.get("type") == "text":
                t = block.get("text", "")
                if t:
                    parts.append(t)
        return "\n".join(parts)
    return ""


def _last_user_text(transcript_path: str) -> str:
    """Returneaza textul ultimului mesaj user din transcript."""
    p = Path(transcript_path)
    if not p.exists():
        return ""
    last_text = ""
    try:
        with open(p, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    entry = json.loads(line)
                except Exception:
                    continue
                if entry.get("type") != "user":
                    continue
                msg = entry.get("message", {}) or {}
                # Sare peste tool_result-uri (sunt tip user dar reprezinta output tool)
                content = msg.get("content")
                if isinstance(content, list) and content and isinstance(content[0], dict) \
                        and content[0].get("type") == "tool_result":
                    continue
                txt = _extract_user_text(content)
                if txt:
                    last_text = txt
    except Exception:
        pass
    return last_text


def _is_telegram_origin(transcript_path: str) -> bool:
    """
    True daca ultimul mesaj user a fost injectat de bridge.py.
    Semnal autoritar: fisierul injected/<hash(text)>.tg trebuie sa existe.
    """
    txt = _last_user_text(transcript_path)
    if not txt:
        return False
    return (INJECTED_DIR / f"{_hash_text(txt)}.tg").exists()


def _check_auto_commands(user_text: str) -> str | None:
    """
    Verifica daca user-ul a inclus o comanda de toggle pentru auto-approve:
      '@auto_on'  -> creeaza flag, returneaza 'on'
      '@auto_off' -> sterge flag, returneaza 'off'
      altfel returneaza None.
    """
    low = user_text.lower()
    if "@auto_on" in low or "@yolo_on" in low:
        try:
            AUTO_APPROVE_FLAG.write_text(str(int(time.time())), encoding="utf-8")
        except Exception:
            pass
        return "on"
    if "@auto_off" in low or "@yolo_off" in low:
        try:
            AUTO_APPROVE_FLAG.unlink(missing_ok=True)
        except Exception:
            pass
        return "off"
    return None


def _is_auto_approve(user_text: str) -> bool:
    """
    True daca tool-ul curent trebuie auto-aprobat (skip Telegram approval):
      - fisierul AUTO_APPROVE_FLAG exista, SAU
      - mesajul user contine @auto / @yolo (per-mesaj).
    """
    if AUTO_APPROVE_FLAG.exists():
        return True
    low = user_text.lower()
    return ("@auto" in low) or ("@yolo" in low)


def _ask_default() -> None:
    """Output JSON cu permissionDecision='ask' -> defer la UI nativ VSCode."""
    print(json.dumps({
        "hookSpecificOutput": {
            "hookEventName": "PreToolUse",
            "permissionDecision": "ask",
        }
    }))
    sys.exit(0)


def _wait_for_response(token_str: str) -> str:
    """
    Polling pe fisierul approvals/<token>.resp. Returneaza:
    'approve' | 'deny' | 'timeout'.
    """
    resp_path = APPROVALS_DIR / f"{token_str}.resp"
    deadline = time.time() + TIMEOUT_S
    while time.time() < deadline:
        if resp_path.exists():
            try:
                with open(resp_path, encoding="utf-8") as f:
                    data = json.load(f)
                decision = data.get("decision", "deny")
                try:
                    resp_path.unlink(missing_ok=True)
                except Exception:
                    pass
                return decision
            except Exception:
                time.sleep(POLL_INTERVAL_S)
                continue
        time.sleep(POLL_INTERVAL_S)
    return "timeout"


def main() -> int:
    # 1. Cite hook input
    try:
        raw = sys.stdin.read()
        data = json.loads(raw) if raw.strip() else {}
    except Exception:
        # Fail safe: allow (nu blocam Claude pe input invalid)
        _allow()
        return 0

    tool_name = data.get("tool_name", "?")
    tool_input = data.get("tool_input", {})
    transcript_path = data.get("transcript_path", "")

    # 2. Cite o singura data ultimul mesaj user din transcript
    user_text = _last_user_text(transcript_path) if transcript_path else ""
    # Origine = fisier injected/<hash>.tg existent (autoritar). Marker text-based
    # e ignorat — putea sa scurga prin clipboard/paste/arrow-up.
    is_telegram = bool(user_text) and (INJECTED_DIR / f"{_hash_text(user_text)}.tg").exists()

    # 3. Toggle persistent: @auto_on / @auto_off in mesaj
    toggle = _check_auto_commands(user_text)
    if toggle and is_telegram:
        creds_for_msg = _load_telegram_creds()
        if creds_for_msg:
            msg_txt = ("🤖 <b>Auto-approve: ON</b> (flag creat)"
                       if toggle == "on" else
                       "🛡 <b>Auto-approve: OFF</b> (flag sters)")
            _tg_send_text(creds_for_msg[0], creds_for_msg[1], msg_txt)

    # 4. Mesaj scris direct in VSCode -> AUTO-ALLOW (user a cerut bypass total)
    #    Inainte era _ask_default() (UI nativ VSCode). Acum permite totul.
    if not is_telegram:
        _allow()
        return 0

    # 5. Tool safe -> auto-allow chiar daca e Telegram
    if tool_name in AUTO_ALLOW_TOOLS:
        _allow()
        return 0

    # 6. AUTO mode (flag fisier sau prefix @auto/@yolo) -> auto-allow risky
    if _is_auto_approve(user_text):
        creds_auto = _load_telegram_creds()
        if creds_auto:
            summary_short = _summarize_tool_input(tool_name, tool_input)[:200]
            _tg_send_text(
                creds_auto[0], creds_auto[1],
                f"🤖 <b>Auto-aprobat:</b> {tool_name}\n<code>{summary_short}</code>",
            )
        _allow()
        return 0

    # 7. Tool risky pe ruta Telegram -> cere aprobare pe Telegram
    creds = _load_telegram_creds()
    if not creds:
        # Nu avem credentiale -> nu blocam (fallback allow ca sa nu blocheze workflow)
        _allow()
        return 0
    tg_token, chat_id = creds

    APPROVALS_DIR.mkdir(parents=True, exist_ok=True)
    approval_token = uuid.uuid4().hex[:10]
    summary = _summarize_tool_input(tool_name, tool_input)

    # Salveaza request (debug)
    req_path = APPROVALS_DIR / f"{approval_token}.req"
    try:
        req_path.write_text(json.dumps({
            "tool_name": tool_name,
            "tool_input": tool_input,
            "summary": summary,
            "ts": int(time.time()),
        }, indent=2), encoding="utf-8")
    except Exception:
        pass

    sent = _tg_send_approval(tg_token, chat_id, approval_token, tool_name, summary)
    if not sent:
        # Nu am putut trimite -> deny safe
        _deny(f"Bridge: nu am putut trimite cererea de aprobare pe Telegram pentru {tool_name}")
        return 2

    decision = _wait_for_response(approval_token)

    # Cleanup request file
    try:
        req_path.unlink(missing_ok=True)
    except Exception:
        pass

    if decision == "approve":
        _tg_send_text(tg_token, chat_id, f"✅ <b>Aprobat:</b> {tool_name}")
        _allow()
        return 0
    elif decision == "deny":
        _tg_send_text(tg_token, chat_id, f"❌ <b>Respins:</b> {tool_name}")
        _deny(f"Utilizatorul a respins {tool_name} prin Telegram.")
        return 2
    else:  # timeout
        _tg_send_text(tg_token, chat_id, f"⏱ <b>Timeout aprobare</b> ({tool_name}) — respins automat.")
        _deny(f"Aprobare pentru {tool_name} a expirat (>{TIMEOUT_S}s) — refuzat automat.")
        return 2


if __name__ == "__main__":
    sys.exit(main())
