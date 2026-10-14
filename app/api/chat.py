from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from sqlalchemy.orm import Session
from fastapi.responses import StreamingResponse

from app.api.auth import get_current_user, get_db
from app.api.conversations import get_owned_conversation
from app.models.user import User
from app.models.message import Message
from app.schemas.chat import ChatIn
from app.services.chat_run import event_to_sse, format_sse, iter_graph_events, run_chat_job
from app.services.disambiguate import disambiguate_query
from app.services.job_queue import collect_job_text, enqueue_job, iter_job_events, queue_depth
from app.services.llm import AuthError, RateLimitError, UpstreamError
from app.services.redis_store import limit_chat
from app.services.terms import rewrite_query

router = APIRouter(prefix="/chat", tags=["chat"])


class DebugOut(BaseModel):
    reply: str


def _save_user_query(payload: ChatIn, db: Session) -> tuple[str, str]:
    rw = rewrite_query(payload.content)
    terms_q = rw["query"] or payload.content
    original = rw["original"] or payload.content
    dis = disambiguate_query(terms_q)
    if dis.get("confident") and str(dis.get("query") or "").strip():
        query = str(dis["query"]).strip()
    else:
        query = terms_q
    db.add(Message(
        conversation_id=payload.conversation_id,
        role="user",
        content=payload.content,
        rewrite=query or None,
    ))
    db.commit()
    return query, original


def _raise_job_error(msg: str) -> None:
    text = msg or "上游错误"
    if "认证" in text or "API key" in text:
        raise HTTPException(status_code=401, detail=text)
    if "重试超出" in text or "过于频繁" in text:
        raise HTTPException(status_code=429, detail=text)
    if "排队超时" in text:
        raise HTTPException(status_code=504, detail=text)
    if "Redis" in text:
        raise HTTPException(status_code=503, detail=text)
    raise HTTPException(status_code=502, detail=text)


def _queued_status() -> str:
    n = queue_depth()
    if n is None:
        return "排队中…"
    ahead = max(int(n) - 1, 0)
    if ahead:
        return f"排队中…前面还有 {ahead} 个任务"
    return "排队中…"


@router.post("", response_model=DebugOut)
def chat(
    payload: ChatIn,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    get_owned_conversation(payload.conversation_id, current_user, db)
    limit_chat(current_user.id)
    rewritten, original = _save_user_query(payload, db)
    job_id = enqueue_job(
        payload.conversation_id,
        current_user.id,
        rewritten,
        original,
    )
    if job_id:
        reply, err = collect_job_text(job_id)
        if err:
            _raise_job_error(err)
        return DebugOut(reply=reply)
    try:
        reply, err = run_chat_job(payload.conversation_id, rewritten, original)
    except AuthError as e:
        raise HTTPException(status_code=401, detail=str(e))
    except RateLimitError as e:
        raise HTTPException(status_code=429, detail=str(e))
    except UpstreamError as e:
        raise HTTPException(status_code=502, detail=str(e))
    if err:
        _raise_job_error(err)
    return DebugOut(reply=reply)


@router.post("/stream")
def chat_stream(
    payload: ChatIn,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    get_owned_conversation(payload.conversation_id, current_user, db)
    limit_chat(current_user.id)
    rewritten, original = _save_user_query(payload, db)
    conversation_id = payload.conversation_id
    user_id = current_user.id

    def event_gen():
        job_id = enqueue_job(conversation_id, user_id, rewritten, original)
        if job_id is None:
            for ev in iter_graph_events(conversation_id, rewritten, original):
                chunk = event_to_sse(ev)
                if chunk:
                    yield chunk
            return
        yield format_sse(_queued_status(), event="status")
        for ev in iter_job_events(job_id):
            chunk = event_to_sse(ev)
            if chunk:
                yield chunk

    return StreamingResponse(
        event_gen(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "Connection": "keep-alive"},
    )
