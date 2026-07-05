import json
import os
import subprocess
import time
import urllib.parse
import urllib.request

TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN", "").strip()
REPO = os.environ.get("TG_BRIDGE_REPO", os.getcwd())
STATE = os.path.expanduser("~/.config/tg_bridge")
ALLOW_F = os.path.join(STATE, "allowed_chat")
SESS_F = os.path.join(STATE, "session")
API = f"https://api.telegram.org/bot{TOKEN}"
CLAUDE_TIMEOUT = 1800
POLL = 50
SYS = ("You are answering over Telegram on a phone. Keep replies concise and "
       "plain-text (no markdown tables, no huge code blocks). You are running "
       "in the user's research repo and may use tools to do real work.")


def api(method, **params):
    data = urllib.parse.urlencode(params).encode()
    req = urllib.request.Request(f"{API}/{method}", data=data)
    with urllib.request.urlopen(req, timeout=POLL + 15) as r:
        return json.load(r)


def send(chat, text):
    for i in range(0, len(text), 3800):
        try:
            api("sendMessage", chat_id=chat, text=text[i:i + 3800])
        except Exception as e:
            print("send err", e, flush=True)


def load(path, default=""):
    return open(path).read().strip() if os.path.exists(path) else default


def save(path, val):
    os.makedirs(STATE, exist_ok=True)
    with open(path, "w") as f:
        f.write(val)


def run_claude(prompt):
    sid = load(SESS_F)
    cmd = ["claude", "--print", "--output-format", "json",
           "--dangerously-skip-permissions", "--append-system-prompt", SYS]
    if sid:
        cmd += ["--resume", sid]
    cmd += [prompt]
    try:
        p = subprocess.run(cmd, cwd=REPO, capture_output=True, text=True,
                           timeout=CLAUDE_TIMEOUT)
    except subprocess.TimeoutExpired:
        return "(claude timed out after 30 min)"
    out = p.stdout.strip()
    if not out:
        return f"(claude error)\n{p.stderr.strip()[-1500:]}"
    try:
        obj = json.loads(out)
    except Exception:
        return out[-3500:]
    if obj.get("session_id"):
        save(SESS_F, obj["session_id"])
    if obj.get("is_error"):
        return f"(claude error) {obj.get('result', '')[:1500]}"
    return obj.get("result", "(empty)")


def handle(msg):
    chat = str(msg["chat"]["id"])
    text = msg.get("text", "")
    allowed = load(ALLOW_F)
    if not allowed:
        save(ALLOW_F, chat)
        allowed = chat
        send(chat, f"Locked to this chat (id {chat}). Send a message and I'll "
                   "work in the repo. /new resets the conversation.")
        if text in ("/start", ""):
            return
    if chat != allowed:
        return
    if text == "/start":
        send(chat, "Ready. Just message me. /new = fresh conversation.")
        return
    if text == "/new":
        if os.path.exists(SESS_F):
            os.remove(SESS_F)
        send(chat, "Started a fresh conversation.")
        return
    if not text:
        return
    try:
        api("sendChatAction", chat_id=chat, action="typing")
    except Exception:
        pass
    reply = run_claude(text)
    send(chat, reply or "(no output)")


def main():
    if not TOKEN:
        raise SystemExit("set TELEGRAM_BOT_TOKEN")
    print(f"bridge up, repo={REPO}", flush=True)
    offset = None
    while True:
        try:
            params = {"timeout": POLL}
            if offset is not None:
                params["offset"] = offset
            r = api("getUpdates", **params)
        except Exception as e:
            print("poll err", e, flush=True)
            time.sleep(5)
            continue
        for upd in r.get("result", []):
            offset = upd["update_id"] + 1
            msg = upd.get("message") or upd.get("edited_message")
            if msg and "chat" in msg:
                try:
                    handle(msg)
                except Exception as e:
                    print("handle err", e, flush=True)


if __name__ == "__main__":
    main()
