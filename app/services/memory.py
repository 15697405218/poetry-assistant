"""会话记忆：进模型的全部上下文粗算超过 100k 才压一次。

计入：总 system + 已有摘要 + 未压缩原文。压完只留摘要 + 最近 KEEP_LOW 条，
再涨到下一个 100k 才压。token 按字数/2 粗算。
"""
from __future__ import annotations

from sqlalchemy.orm import Session

from app.config import settings
from app.models.conversation import Conversation
from app.models.message import Message
from app.services.graph import SYSTEM
from app.services.llm import AuthError, RateLimitError, UpstreamError, llm

KEEP_LOW = settings.keep_low
RAW_TOKEN_LIMIT = settings.raw_token_limit
SUMMARY_PREFIX = "以下是更早对话的精简摘要，回答时一并参考：\n"

SUMMARIZE_SYSTEM = """你是对话压缩器。把材料压成一段中文摘要，给后续诗词助手接着用。
只保留：用户在问什么、已确认的作者/篇名/结论、未完成的追问。不要编造，不要写成对用户的回复。不超过 400 字。只输出摘要正文。"""


def _msg_text(m: Message) -> str:
    if m.role == "user":
        return m.rewrite or m.content
    return m.content


def _approx_tokens(text: str) -> int:
    return max(1, (len(text or "") + 1) // 2)


def _window_tokens(rows: list[Message]) -> int:
    return sum(_approx_tokens(_msg_text(m)) for m in rows)


def _packed_tokens(summary: str, rows: list[Message]) -> int:
    n = _approx_tokens(SYSTEM)
    if summary.strip():
        n += _approx_tokens(SUMMARY_PREFIX + summary.strip())
    return n + _window_tokens(rows)


def _fold_block(rows: list[Message]) -> str:
    parts = []
    for m in rows:
        who = "用户" if m.role == "user" else "助手"
        text = (_msg_text(m) or "").strip()
        if len(text) > 400:
            text = text[:400] + "…"
        parts.append(f"{who}：{text}")
    return "\n".join(parts)


def _summarize(old: str, chunk: str) -> str | None:
    user = []
    if old.strip():
        user.append("已有摘要：\n" + old.strip())
    user.append("新增对话：\n" + chunk)
    try:
        out = llm.chat(
            [
                {"role": "system", "content": SUMMARIZE_SYSTEM},
                {"role": "user", "content": "\n\n".join(user)},
            ],
            temperature=settings.summarize_temperature,
            max_tokens=settings.summarize_max_tokens,
        )
    except (AuthError, RateLimitError, UpstreamError, Exception):
        return None
    text = (out or "").strip()
    return text or None


def _need_compact(summary: str, rows: list[Message]) -> bool:
    return _packed_tokens(summary, rows) > RAW_TOKEN_LIMIT and len(rows) > KEEP_LOW


def compact_and_history(db: Session, conversation_id: int) -> list[dict]:
    """必要时过压缩，返回给模型的 messages（不含总 system）。"""
    conv = db.query(Conversation).filter(Conversation.id == conversation_id).first()
    q = db.query(Message).filter(Message.conversation_id == conversation_id)
    if conv and conv.summarized_through_id:
        q = q.filter(Message.id > conv.summarized_through_id)
    rows = q.order_by(Message.id.asc()).all()
    if not rows:
        return []

    summary = (conv.summary or "").strip() if conv else ""
    if conv is not None and _need_compact(summary, rows):
        keep = rows[-KEEP_LOW:]
        overflow = rows[:-KEEP_LOW]
        if overflow:
            folded = _summarize(summary, _fold_block(overflow))
            if folded:
                conv.summary = folded
                conv.summarized_through_id = overflow[-1].id
                db.commit()
                rows = keep
                summary = folded

    history: list[dict] = []
    if summary:
        history.append({
            "role": "system",
            "content": SUMMARY_PREFIX + summary,
        })
    for m in rows:
        if m.role == "user":
            history.append({"role": "user", "content": m.rewrite or m.content})
        else:
            history.append({"role": m.role, "content": m.content})
    return history
