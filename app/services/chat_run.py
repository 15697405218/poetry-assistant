"""对话图执行：API 同步兜底和 worker 共用，避免各写一套 retrieve/generate。"""
from __future__ import annotations

import re
from collections.abc import Callable, Iterator
from typing import Any

from langgraph.errors import GraphRecursionError

from app.db import SessionLocal
from app.models.message import Message
from app.services.graph import SYSTEM, graph, graph_config, used_library_tools
from app.services.llm import AuthError, RateLimitError, UpstreamError
from app.services.memory import compact_and_history

# 本库检索引用壳：只剥【主出处…】【参考…】【出处#…】和 [作者·标题]，不动李白、诗句本身。
_LIB_BRACKET = re.compile(r"【(?:主出处|参考)[^】]*】|【出处#[^】]*】")
_LIB_AUTHOR_TITLE = re.compile(r"\[\s*[\u4e00-\u9fff][^\[\]]*?·[^\[\]]+?\]")

Event = dict[str, Any]


def strip_library_cites(text: str) -> str:
    """去掉伪装成本库命中的引用壳，保留作者名和诗句正文。"""
    if not text:
        return text or ""
    out = _LIB_BRACKET.sub("", text)
    out = _LIB_AUTHOR_TITLE.sub("", out)
    out = re.sub(r"[ \t]{2,}", " ", out)
    out = re.sub(r" *\n", "\n", out)
    out = re.sub(r"\n{3,}", "\n\n", out)
    return out.strip()


def _state_used_library(g: Any, config: dict) -> bool:
    get_state = getattr(g, "get_state", None)
    if not callable(get_state):
        return False
    try:
        snap = get_state(config)
        values = getattr(snap, "values", None)
        if values is None and isinstance(snap, dict):
            values = snap.get("values") or snap
        messages = values.get("messages") if isinstance(values, dict) else None
        return used_library_tools(messages or [])
    except Exception:
        return False


def format_sse(data: str, event: str | None = None) -> str:
    # SSE 一行一条 data；正文里的换行必须拆开，否则第一行后面会被客户端丢掉
    head = f"event: {event}\n" if event else ""
    body = "".join(f"data: {line}\n" for line in str(data).split("\n"))
    return head + body + "\n"


def event_to_sse(ev: Event) -> str:
    kind = ev.get("type")
    if kind == "ping":
        return ": ping\n\n"
    if kind == "status":
        return format_sse(ev.get("data") or "", event="status")
    if kind == "token":
        return format_sse(ev.get("data") or "")
    if kind == "error":
        return f"event: error\ndata: {ev.get('data') or 'error'}\n\n"
    if kind == "done":
        return "data: [DONE]\n\n"
    return ""


def save_assistant_message(conversation_id: int, text: str) -> None:
    db = SessionLocal()
    try:
        db.add(Message(
            conversation_id=conversation_id,
            role="assistant",
            content=text,
        ))
        db.commit()
    finally:
        db.close()


def load_history(conversation_id: int) -> list[dict]:
    db = SessionLocal()
    try:
        return compact_and_history(db, conversation_id)
    finally:
        db.close()


def iter_graph_events(
    conversation_id: int,
    rewrite: str,
    original: str,
    *,
    graph_impl: Any = None,
    load_history_fn: Callable[[int], list[dict]] | None = None,
    save_assistant_fn: Callable[[int, str], None] | None = None,
) -> Iterator[Event]:
    """跑 LangGraph retrieve+generate，产出 status/token/error/done。

    worker 和 Redis 挂掉时的进程内兜底都走这里。
    """
    g = graph_impl if graph_impl is not None else graph
    history = (load_history_fn or load_history)(conversation_id)
    seed = [{"role": "system", "content": SYSTEM}] + history
    parts: list[str] = []
    lib_used = False
    cfg = graph_config(conversation_id)
    try:
        for ev in g.stream(
            {
                "messages": seed,
                "answer": "",
                "rewrite": rewrite,
                "original": original,
            },
            cfg,
            stream_mode="custom",
        ):
            if not isinstance(ev, dict):
                continue
            if ev.get("lib_used") or ev.get("status") == "正在检索库…":
                lib_used = True
            if "status" in ev:
                yield {"type": "status", "data": ev["status"]}
            token = ev.get("token")
            if token:
                parts.append(token)
                yield {"type": "token", "data": token}
    except AuthError as e:
        yield {"type": "error", "data": str(e)}
        return
    except RateLimitError as e:
        yield {"type": "error", "data": str(e)}
        return
    except UpstreamError as e:
        yield {"type": "error", "data": str(e)}
        return
    except GraphRecursionError:
        yield {"type": "error", "data": "模型反复调用工具，已停止"}
        return
    text = "".join(parts)
    if not lib_used and not _state_used_library(g, cfg):
        text = strip_library_cites(text)
    (save_assistant_fn or save_assistant_message)(conversation_id, text)
    yield {"type": "done", "data": text}


def run_chat_job(
    conversation_id: int,
    rewrite: str,
    original: str,
    **kwargs: Any,
) -> tuple[str, str | None]:
    """非流式：跑完图，返回 (正文, 错误)。"""
    parts: list[str] = []
    err: str | None = None
    final: str | None = None
    for ev in iter_graph_events(conversation_id, rewrite, original, **kwargs):
        if ev.get("type") == "token":
            parts.append(ev.get("data") or "")
        elif ev.get("type") == "error":
            err = ev.get("data") or "error"
        elif ev.get("type") == "done" and ev.get("data") is not None:
            final = ev.get("data") or ""
    return (final if final is not None else "".join(parts)), err
