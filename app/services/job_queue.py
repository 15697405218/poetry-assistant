"""Redis 对话任务队列：List 排队 + 每任务 Stream 推 token/状态。

API 入队后 SSE 读 `poetry:job:{id}:events`；worker BLPOP `poetry:jobs` 再跑图。
Redis 不可用时返回 None / 中断事件，由调用方走进程内同步兜底，不空等。
"""
from __future__ import annotations

import json
import time
import uuid
from collections.abc import Iterator
from typing import Any

from app.config import settings
from app.observability import get_logger
from app.services.redis_store import drop_redis_queue, get_redis_queue

log = get_logger("poetry.queue")

JOBS_KEY = "poetry:jobs"
JOB_PREFIX = "poetry:job:"
EVENTS_SUFFIX = ":events"

Event = dict[str, Any]


def jobs_key() -> str:
    return JOBS_KEY


def meta_key(job_id: str) -> str:
    return f"{JOB_PREFIX}{job_id}"


def events_key(job_id: str) -> str:
    return f"{JOB_PREFIX}{job_id}{EVENTS_SUFFIX}"


def _ttl() -> int:
    return max(int(settings.job_ttl_seconds), 60)


def queue_depth() -> int | None:
    r = get_redis_queue()
    if r is None:
        return None
    try:
        return int(r.llen(JOBS_KEY))
    except Exception:
        drop_redis_queue()
        return None


def enqueue_job(
    conversation_id: int,
    user_id: int,
    rewrite: str,
    original: str,
) -> str | None:
    """入队成功返回 job_id；Redis 挂了返回 None（调用方应同步跑）。"""
    r = get_redis_queue()
    if r is None:
        return None
    job_id = uuid.uuid4().hex
    payload = json.dumps(
        {
            "job_id": job_id,
            "conversation_id": conversation_id,
            "user_id": user_id,
            "rewrite": rewrite,
            "original": original,
            "ts": time.time(),
        },
        ensure_ascii=False,
    )
    meta = json.dumps(
        {"status": "queued", "conversation_id": conversation_id, "user_id": user_id},
        ensure_ascii=False,
    )
    try:
        r.setex(meta_key(job_id), _ttl(), meta)
        r.rpush(JOBS_KEY, payload)
        return job_id
    except Exception:
        drop_redis_queue()
        log.warning("enqueue failed job_id=%s", job_id)
        return None


def pop_job(timeout: int = 5) -> dict | None:
    r = get_redis_queue()
    if r is None:
        return None
    try:
        item = r.blpop(JOBS_KEY, timeout=timeout)
    except Exception:
        drop_redis_queue()
        return None
    if not item:
        return None
    _, raw = item
    try:
        job = json.loads(raw)
    except json.JSONDecodeError:
        log.warning("bad job payload")
        return None
    if not isinstance(job, dict) or not job.get("job_id"):
        return None
    return job


def set_job_status(job_id: str, status: str) -> None:
    r = get_redis_queue()
    if r is None:
        return
    try:
        raw = r.get(meta_key(job_id))
        meta = json.loads(raw) if raw else {}
        if not isinstance(meta, dict):
            meta = {}
        meta["status"] = status
        r.setex(meta_key(job_id), _ttl(), json.dumps(meta, ensure_ascii=False))
    except Exception:
        drop_redis_queue()


def publish_event(job_id: str, ev: Event) -> bool:
    r = get_redis_queue()
    if r is None:
        return False
    key = events_key(job_id)
    try:
        r.xadd(
            key,
            {"e": json.dumps(ev, ensure_ascii=False)},
            maxlen=2000,
            approximate=True,
        )
        r.expire(key, _ttl())
        return True
    except Exception:
        drop_redis_queue()
        return False


def iter_job_events(job_id: str, wait_seconds: int | None = None) -> Iterator[Event]:
    """阻塞读任务 Stream。空闲时 yield ping；Redis 挂了或超时 yield error。"""
    wait = float(settings.job_wait_seconds if wait_seconds is None else wait_seconds)
    deadline = time.monotonic() + max(wait, 0.1)
    last_id = "0-0"
    while time.monotonic() < deadline:
        r = get_redis_queue()
        if r is None:
            yield {"type": "error", "data": "Redis 不可用，队列中断"}
            return
        remaining_ms = int(max(deadline - time.monotonic(), 0.05) * 1000)
        block = min(2000, remaining_ms)
        try:
            raw = r.xread({events_key(job_id): last_id}, block=block, count=50)
        except Exception:
            drop_redis_queue()
            yield {"type": "error", "data": "Redis 不可用，队列中断"}
            return
        if not raw:
            yield {"type": "ping"}
            continue
        for _name, entries in raw:
            for msg_id, fields in entries:
                last_id = msg_id
                try:
                    ev = json.loads(fields.get("e") or "{}")
                except json.JSONDecodeError:
                    continue
                if not isinstance(ev, dict):
                    continue
                yield ev
                if ev.get("type") in {"done", "error"}:
                    return
    yield {"type": "error", "data": "排队超时，请确认已启动 python scripts/worker.py"}


def collect_job_text(job_id: str, wait_seconds: int | None = None) -> tuple[str, str | None]:
    parts: list[str] = []
    for ev in iter_job_events(job_id, wait_seconds=wait_seconds):
        kind = ev.get("type")
        if kind == "token":
            parts.append(ev.get("data") or "")
        elif kind == "error":
            return "".join(parts), ev.get("data") or "error"
        elif kind == "done":
            return "".join(parts), None
    return "".join(parts), "排队超时，请确认已启动 python scripts/worker.py"
