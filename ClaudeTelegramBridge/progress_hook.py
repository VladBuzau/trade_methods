"""
progress_hook.py — PostToolUse hook pentru Claude Code.

Inregistrat in ~/.claude/settings.json sub "hooks.PostToolUse".
Ruleaza dupa fiecare tool call. Daca tool call-ul a apartinut unui
mesaj injectat de pe Telegram (exista injected/<hash>.tg), incrementeaza
contorul de tool calls in fisier — bridge.py il citeste pentru heartbeat.

Hook input (stdin, JSON):
  {
    "session_id": "...",
    "transcript_path": "...",
    "tool_name": "Bash",
    "tool_input": {...},
    "tool_response": {...}
  }
"""
from __future__ import annotations
import sys, os, json, time, hashlib
from pathlib import Path

HERE = Path(__file__).resolve().parent
INJECTED_DIR = HERE / "injected"
MAX_SUMMARY_TOOLS = 15


def _hash_text(text: str) -> str:
    """Hash normalizat — identic cu cel din bridge.py."""
    norm = (text or "").strip()
    return hashlib.sha256(norm.encode("utf-8")).hexdigest()[:16]


def _extract_user_text(content) -> str:
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
                txt = _extract_user_text(content)
                if txt:
                    last_text = txt
    except Exception:
        pass
    return last_text


def main() -> int:
    try:
        raw = sys.stdin.read()
        data = json.loads(raw) if raw.strip() else {}
    except Exception:
        return 0

    transcript_path = data.get("transcript_path", "")
    tool_name = data.get("tool_name", "?")
    if not transcript_path:
        return 0

    user_text = _last_user_text(transcript_path)
    if not user_text:
        return 0

    inject_file = INJECTED_DIR / f"{_hash_text(user_text)}.tg"
    if not inject_file.exists():
        # Nu e mesaj de pe Telegram — silent
        return 0

    # Actualizeaza contor (tolereaza format JSON sau legacy)
    info: dict = {"ts": int(time.time()), "tools_run": 0, "tools_summary": []}
    try:
        existing = json.loads(inject_file.read_text(encoding="utf-8"))
        if isinstance(existing, dict):
            info = existing
    except Exception:
        pass

    info["tools_run"] = int(info.get("tools_run", 0)) + 1
    summary = list(info.get("tools_summary", []))
    summary.append(tool_name)
    info["tools_summary"] = summary[-MAX_SUMMARY_TOOLS:]

    try:
        inject_file.write_text(json.dumps(info), encoding="utf-8")
    except Exception:
        pass

    return 0


if __name__ == "__main__":
    sys.exit(main())
