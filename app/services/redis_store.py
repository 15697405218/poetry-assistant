"""Redis：对话限流、消歧/检索缓存、登录锁定、JWT 登出黑名单、对话任务队列。

连不上就降级跳过（登出除外；队列则由 API 改走进程内同步），数据在 Postgres。
"""
from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone

import redis
from fastapi import HTTPException, status

from app.config import settings

_client: redis.Redis | None = None
_queue_client: redis.Redis | None = None


def get_redis() -> redis.Redis | None:
    global _client
    if _client is not None:
        return _client
    try:
        client = redis.Redis.from_url(
            settings.redis_url,
            decode_responses=True,
            socket_connect_timeout=0.8,
            socket_timeout=0.8,
        )
        client.ping()
        _client = client
        return _client
    except Exception:
        return None


def get_redis_queue() -> redis.Redis | None:
    """给 BLPOP / XREAD 用：socket_timeout 不能是 0.8s，否则阻塞读会被掐断。"""
    global _queue_client
    if _queue_client is not None:
        return _queue_client
    try:
        client = redis.Redis.from_url(
            settings.redis_url,
            decode_responses=True,
            socket_connect_timeout=1.0,
            socket_timeout=None,
            health_check_interval=30,
        )
        client.ping()
        _queue_client = client
        return _queue_client
    except Exception:
        return None


def drop_redis_queue() -> None:
    global _queue_client
    _queue_client = None


def redis_queue_ok() -> bool:
    r = get_redis_queue()
    if r is None:
        return False
    try:
        return bool(r.ping())
    except Exception:
        drop_redis_queue()
        return False


def redis_ok() -> bool:
    r = get_redis()
    if r is None:
        return False
    try:
        return bool(r.ping())
    except Exception:
        global _client
        _client = None
        return False


def _drop() -> None:
    global _client
    _client = None


def cache_key(*parts: object) -> str:
    raw = "|".join("" if p is None else str(p) for p in parts)
    digest = hashlib.sha256(raw.encode("utf-8")).hexdigest()
    return digest


def cache_get(key: str):
    r = get_redis()
    if r is None:
        return None
    try:
        raw = r.get(key)
    except Exception:
        _drop()
        return None
    if not raw:
        return None
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return None


def cache_set(key: str, value: object, ttl: int | None = None) -> None:
    r = get_redis()
    if r is None:
        return
    ttl = ttl if ttl is not None else settings.cache_ttl_seconds
    try:
        r.setex(key, ttl, json.dumps(value, ensure_ascii=False, default=str))
    except Exception:
        _drop()


def limit_chat(user_id: int) -> None:
    r = get_redis()
    if r is None:
        return
    key = f"poetry:rl:chat:{user_id}"
    try:
        n = int(r.incr(key))
        if n == 1:
            r.expire(key, settings.chat_rate_window)
    except Exception:
        _drop()
        return
    if n > settings.chat_rate_limit:
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail="对话过于频繁，请稍后再试",
        )


def login_locked(email: str) -> bool:
    r = get_redis()
    if r is None:
        return False
    key = f"poetry:login:fail:{email.strip().lower()}"
    try:
        n = int(r.get(key) or 0)
    except Exception:
        _drop()
        return False
    return n >= settings.login_fail_max


def login_fail(email: str) -> None:
    r = get_redis()
    if r is None:
        return
    key = f"poetry:login:fail:{email.strip().lower()}"
    try:
        n = int(r.incr(key))
        if n == 1:
            r.expire(key, settings.login_lock_seconds)
    except Exception:
        _drop()


def login_ok(email: str) -> None:
    r = get_redis()
    if r is None:
        return
    try:
        r.delete(f"poetry:login:fail:{email.strip().lower()}")
    except Exception:
        _drop()


def limit_login_ip(ip: str) -> None:
    r = get_redis()
    if r is None:
        return
    key = f"poetry:rl:login:{ip or 'unknown'}"
    try:
        n = int(r.incr(key))
        if n == 1:
            r.expire(key, 600)
    except Exception:
        _drop()
        return
    if n > 30:
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail="登录尝试过多，请稍后再试",
        )


def blacklist_jti(jti: str, ttl_seconds: int) -> None:
    r = get_redis()
    if r is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Redis 不可用，无法登出",
        )
    ttl_seconds = max(int(ttl_seconds), 1)
    try:
        r.setex(f"poetry:jwt:bl:{jti}", ttl_seconds, "1")
    except Exception:
        _drop()
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Redis 不可用，无法登出",
        )


def jti_blacklisted(jti: str) -> bool:
    if not jti:
        return False
    r = get_redis()
    if r is None:
        return False
    try:
        return bool(r.exists(f"poetry:jwt:bl:{jti}"))
    except Exception:
        _drop()
        return False


def ttl_until(exp) -> int:
    if exp is None:
        return settings.access_token_expire_minutes * 60
    try:
        ts = int(exp)
    except (TypeError, ValueError):
        return settings.access_token_expire_minutes * 60
    left = ts - int(datetime.now(timezone.utc).timestamp())
    return max(left, 1)


def session_ttl() -> int:
    return max(int(settings.access_token_expire_minutes) * 60, 60)


def _sess_key(jti: str) -> str:
    return f"poetry:sess:{jti}"


def create_session(jti: str) -> None:
    r = get_redis()
    if r is None:
        return
    try:
        r.setex(_sess_key(jti), session_ttl(), "1")
    except Exception:
        _drop()


def touch_session(jti: str) -> bool:
    """活跃则把空闲 TTL 拨回满。Redis 在且会话已没了 → False。"""
    if not jti:
        return True
    r = get_redis()
    if r is None:
        return True
    key = _sess_key(jti)
    try:
        if not r.exists(key):
            return False
        r.expire(key, session_ttl())
        return True
    except Exception:
        _drop()
        return True


def drop_session(jti: str) -> None:
    r = get_redis()
    if r is None:
        return
    try:
        r.delete(_sess_key(jti))
    except Exception:
        _drop()
