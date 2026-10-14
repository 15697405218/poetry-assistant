"""LangGraph Redis Checkpointer。0.2.65 没有官方 RedisSaver，按 BaseCheckpointSaver 自己落 Redis。

会话正文仍以 Postgres 原文窗口 + 滚动摘要为准；这里存同一次 invoke 里 model⇄tools 的图状态。
thread_id = conv:{conversation_id}。Redis 不可用时退回进程内 MemorySaver。
"""
from __future__ import annotations

import pickle
from collections import defaultdict

import redis
from langchain_core.runnables import RunnableConfig

from langgraph.checkpoint.base import (
    ChannelVersions,
    Checkpoint,
    CheckpointMetadata,
    CheckpointTuple,
)
from langgraph.checkpoint.memory import InMemorySaver

from app.config import settings
from app.services.redis_store import session_ttl

_PREFIX = "poetry:lg:ckpt:"


class RedisCheckpointSaver(InMemorySaver):
    def __init__(self) -> None:
        super().__init__()
        self._r: redis.Redis | None | bool = None

    def _client(self) -> redis.Redis | None:
        if self._r is False:
            return None
        if self._r is not None:
            return self._r
        try:
            client = redis.Redis.from_url(
                settings.redis_url,
                decode_responses=False,
                socket_connect_timeout=0.8,
                socket_timeout=0.8,
            )
            client.ping()
            self._r = client
            return client
        except Exception:
            self._r = False
            return None

    def _key(self, thread_id: str) -> bytes:
        return f"{_PREFIX}{thread_id}".encode()

    def _ttl(self) -> int:
        return max(session_ttl(), int(settings.access_token_max_minutes) * 60)

    def _hydrate(self, thread_id: str) -> None:
        r = self._client()
        if r is None:
            return
        try:
            raw = r.get(self._key(thread_id))
        except Exception:
            self._r = False
            return
        if not raw:
            return
        try:
            data = pickle.loads(raw)
        except Exception:
            return
        nsmap: dict = defaultdict(dict)
        nsmap.update(data.get("storage") or {})
        self.storage[thread_id] = nsmap
        for k, v in (data.get("writes") or {}).items():
            self.writes[k] = v
        for k, v in (data.get("blobs") or {}).items():
            self.blobs[k] = v

    def _persist(self, thread_id: str) -> None:
        r = self._client()
        if r is None:
            return
        writes = {k: v for k, v in self.writes.items() if k[0] == thread_id}
        blobs = {k: v for k, v in self.blobs.items() if k[0] == thread_id}
        payload = pickle.dumps({
            "storage": dict(self.storage.get(thread_id) or {}),
            "writes": writes,
            "blobs": blobs,
        })
        try:
            r.setex(self._key(thread_id), self._ttl(), payload)
        except Exception:
            self._r = False

    def get_tuple(self, config: RunnableConfig) -> CheckpointTuple | None:
        thread_id = config["configurable"]["thread_id"]
        self._hydrate(thread_id)
        return super().get_tuple(config)

    def put(
        self,
        config: RunnableConfig,
        checkpoint: Checkpoint,
        metadata: CheckpointMetadata,
        new_versions: ChannelVersions,
    ) -> RunnableConfig:
        out = super().put(config, checkpoint, metadata, new_versions)
        self._persist(config["configurable"]["thread_id"])
        return out

    def put_writes(
        self,
        config: RunnableConfig,
        writes,
        task_id: str,
        task_path: str = "",
    ) -> None:
        super().put_writes(config, writes, task_id, task_path)
        self._persist(config["configurable"]["thread_id"])

    def delete_thread(self, thread_id: str) -> None:
        super().delete_thread(thread_id)
        r = self._client()
        if r is None:
            return
        try:
            r.delete(self._key(thread_id))
        except Exception:
            self._r = False
