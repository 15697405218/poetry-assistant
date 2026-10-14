"""试 POST /chat/stream，把 SSE 按行打到终端。

先开着 uvicorn（不要 --reload），再在项目根执行：

  python scripts/test_stream.py --email 你的邮箱 --password 你的密码
  python scripts/test_stream.py --email 你的邮箱 --password 你的密码 --q 放翁是谁
  python scripts/test_stream.py --token 粘贴JWT --q 老苏那首中秋问月的
"""
from __future__ import annotations

import argparse
import json
import sys
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

BASE = "http://127.0.0.1:8000"


def request_json(method: str, path: str, token: str | None = None, data=None, form=None) -> dict:
    headers = {}
    body = None
    if token:
        headers["Authorization"] = f"Bearer {token}"
    if form is not None:
        body = urllib.parse.urlencode(form).encode()
        headers["Content-Type"] = "application/x-www-form-urlencoded"
    elif data is not None:
        body = json.dumps(data, ensure_ascii=False).encode("utf-8")
        headers["Content-Type"] = "application/json"
    req = urllib.request.Request(BASE + path, data=body, headers=headers, method=method)
    with urllib.request.urlopen(req, timeout=120) as resp:
        return json.loads(resp.read().decode("utf-8"))


def login(email: str, password: str) -> str:
    out = request_json(
        "POST",
        "/auth/login",
        form={"username": email, "password": password},
    )
    return out["access_token"]


def ensure_conversation(token: str, conversation_id: int | None) -> int:
    if conversation_id:
        return conversation_id
    created = request_json(
        "POST",
        "/conversations/",
        token=token,
        data={"title": "流式测试"},
    )
    return created["id"]


def stream_chat(token: str, conversation_id: int, content: str) -> None:
    body = json.dumps(
        {"conversation_id": conversation_id, "content": content},
        ensure_ascii=False,
    ).encode("utf-8")
    req = urllib.request.Request(
        BASE + "/chat/stream",
        data=body,
        headers={
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
            "Accept": "text/event-stream",
        },
        method="POST",
    )
    print(f"会话 {conversation_id} 问：{content}\n---")
    event = "message"
    data_lines: list[str] = []
    with urllib.request.urlopen(req, timeout=180) as resp:
        for raw in resp:
            line = raw.decode("utf-8").rstrip("\r\n")
            if line.startswith("event:"):
                event = line.split(":", 1)[1].strip()
                continue
            if line.startswith("data:"):
                data_lines.append(line[5:].lstrip())
                continue
            if line:
                continue
            data = "\n".join(data_lines)
            data_lines = []
            if event == "status":
                if data:
                    print(f"[状态] {data}", flush=True)
            elif event == "error":
                print(f"[错误] {data}", flush=True)
            elif data == "[DONE]":
                print("\n---\n[DONE]", flush=True)
            else:
                print(data, end="", flush=True)
            event = "message"


def main() -> None:
    global BASE
    p = argparse.ArgumentParser()
    p.add_argument("--email")
    p.add_argument("--password")
    p.add_argument("--token")
    p.add_argument("--conversation-id", type=int, default=0)
    p.add_argument("--q", default="放翁是谁")
    p.add_argument("--base", default=BASE)
    args = p.parse_args()
    BASE = args.base.rstrip("/")

    from app.services.terms import rewrite_query

    rw = rewrite_query(args.q)
    print("原句:", rw["original"])
    print("改写(给模型和检索):", rw["query"])

    if args.token:
        token = args.token
    elif args.email and args.password:
        token = login(args.email, args.password)
        print("登录成功")
    else:
        raise SystemExit("请传 --email/--password，或 --token")

    cid = ensure_conversation(token, args.conversation_id or None)
    print("会话 id:", cid)
    try:
        stream_chat(token, cid, args.q)
    except urllib.error.HTTPError as e:
        print("HTTP", e.code, e.read().decode("utf-8", errors="replace"))
    except urllib.error.URLError as e:
        print("连不上服务，先开 uvicorn:", e)


if __name__ == "__main__":
    main()
