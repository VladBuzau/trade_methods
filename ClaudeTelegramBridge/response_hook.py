"""
response_hook.py — Stop hook pentru Claude Code.

Inregistrat in ~/.claude/settings.json sub "hooks.Stop". Cand Claude
termina un raspuns, hook-ul:

  1. Verifica daca exista marker-ul `pending.flag` (creat de bridge.py
     cand a injectat un mesaj din Telegram). Daca NU exista -> exit
     liniar, nu deranjeaza conversatiile normale.

  2. Citeste transcript_path (JSONL), extrage ultimul mesaj assistant.

  3. Trimite textul respectiv pe Telegram (folosind credentialele din
     ChartVisualizer/config.json).

  4. Sterge marker-ul.

Hook input (stdin, JSON):
  {
    "session_id": "...",
    "transcript_path": "C:\\Users\\...\\.claude\\projects\\...\\xxx.jsonl",
    "stop_hook_active": false,
    "cwd": "..."
  }
"""
from __future__ import annotations
import sys, os, json, time, hashlib
from pathlib import Path

import requests

HERE = Path(__file__).resolve().parent
PENDING_FLAG = HERE / "pending.flag"   # legacy, ignorat
LAST_SENT_FLAG = HERE / "last_sent.txt"
INJECTED_DIR = HERE / "injected"       # semnal autoritar "vine de pe Telegram"
CHARTVIS_CONFIG = HERE.parent / "ChartVisualizer" / "config.json"

MAX_TG_CHUNK = 4000

# Marker invizibil (LEGACY). Sursa autoritara: injected/<hash>.tg.
TELEGRAM_MARKER = "​​​"


def _hash_text(text: str) -> str:
    """Hash normalizat (strip whitespace). Identic cu cel din bridge.py."""
    norm = (text or "").strip()
    return hashlib.sha256(norm.encode("utf-8")).hexdigest()[:16]


def _exit(msg: str = "", code: int = 0) -> None:
    if msg:
        print(msg, file=sys.stderr)
    sys.exit(code)


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


def _tg_send(token: str, chat_id: str, text: str) -> None:
    url = f"https://api.telegram.org/bot{token}/sendMessage"
    chunks = [text[i : i + MAX_TG_CHUNK] for i in range(0, len(text), MAX_TG_CHUNK)] or [""]
    for chunk in chunks:
        try:
            requests.post(
                url,
                json={
                    "chat_id": chat_id,
                    "text": chunk,
                    "disable_web_page_preview": True,
                },
                timeout=10,
            )
        except Exception:
            pass


def _extract_text(content) -> str:
    """Transcript content poate fi string sau lista de blocks."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for block in content:
            if not isinstance(block, dict):
                continue
            btype = block.get("type")
            if btype == "text":
                t = block.get("text", "")
                if t:
                    parts.append(t)
            elif btype == "thinking":
                # ignora thinking blocks
                continue
            elif btype == "tool_use":
                name = block.get("name", "?")
                parts.append(f"[tool: {name}]")
        return "\n".join(parts).strip()
    return ""


def _last_assistant_message(transcript_path: str) -> str:
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
                # Format Claude Code transcript: {"type": "assistant", "message": {"content": [...]}}
                if entry.get("type") != "assistant":
                    continue
                msg = entry.get("message", {})
                content = msg.get("content")
                txt = _extract_text(content)
                if txt:
                    last_text = txt
    except Exception:
        pass
    return last_text


def _last_user_text(transcript_path: str) -> str:
    """Returneaza textul ultimului mesaj user (sare peste tool_result)."""
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
                content = msg.get("content")
                if isinstance(content, list) and content and isinstance(content[0], dict) \
                        and content[0].get("type") == "tool_result":
                    continue
                # Reciclam _extract_text (handle string si list)
                txt = _extract_text(content)
                if txt:
                    last_text = txt
    except Exception:
        pass
    return last_text


def main() -> int:
    # Citeste input-ul hook-ului
    try:
        raw = sys.stdin.read()
        data = json.loads(raw) if raw.strip() else {}
    except Exception:
        data = {}

    transcript_path = data.get("transcript_path", "")
    if not transcript_path:
        return 0

    # Decide pe baza fisierului inject (autoritar). Daca nu exista, mesajul
    # nu vine de pe Telegram -> nu trimite nimic.
    user_text = _last_user_text(transcript_path)
    if not user_text:
        return 0
    inject_file = INJECTED_DIR / f"{_hash_text(user_text)}.tg"
    if not inject_file.exists():
        return 0

    # Cite metadata (pentru a include statistici in raspuns)
    tools_run = 0
    elapsed = 0
    try:
        info = json.loads(inject_file.read_text(encoding="utf-8"))
        if isinstance(info, dict):
            tools_run = info.get("tools_run", 0)
            ts = info.get("ts", 0)
            if ts:
                elapsed = int(time.time() - ts)
    except Exception:
        pass

    creds = _load_telegram_creds()
    if not creds:
        _exit("response_hook: credentialele Telegram nu sunt configurate", 0)
        return 0
    token, chat_id = creds

    text = _last_assistant_message(transcript_path)
    if not text:
        text = "[bridge] raspuns gol sau doar tool calls"

    # Header cu statistici scurte daca au fost tool calls
    if tools_run > 0:
        header = f"✅ <b>Gata</b> ({elapsed}s, {tools_run} tool calls)\n\n"
    else:
        header = ""

    _tg_send(token, chat_id, f"{header}💬 {text}")

    try:
        LAST_SENT_FLAG.write_text(text, encoding="utf-8")
    except Exception:
        pass

    # Consuma inject file — viitoarele mesaje cu acelasi text vor fi clasificate
    # corect (VSCode daca nu vine cu inject nou).
    try:
        inject_file.unlink(missing_ok=True)
    except Exception:
        pass

    return 0


if __name__ == "__main__":
    sys.exit(main())
