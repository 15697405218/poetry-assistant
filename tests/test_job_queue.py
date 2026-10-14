"""任务队列单测：内存 FakeRedis，不碰真 Redis / GPU / Chroma。"""
from __future__ import annotations

import json

import app.services.job_queue as jq
from app.services.chat_run import event_to_sse, format_sse, iter_graph_events, run_chat_job
from app.services.llm import AuthError, UpstreamError
from app.worker import process_job


class FakeRedis:
    def __init__(self):
        self.lists: dict[str, list] = {}
        self.kv: dict[str, str] = {}
        self.streams: dict[str, list] = {}
        self.expired: dict[str, int] = {}

    def ping(self):
        return True

    def llen(self, key):
        return len(self.lists.get(key, []))

    def rpush(self, key, val):
        self.lists.setdefault(key, []).append(val)
        return len(self.lists[key])

    def blpop(self, keys, timeout=0):
        if isinstance(keys, str):
            keys = [keys]
        for k in keys:
            if self.lists.get(k):
                return (k, self.lists[k].pop(0))
        return None

    def setex(self, key, ttl, val):
        self.kv[key] = val
        self.expired[key] = ttl
        return True

    def get(self, key):
        return self.kv.get(key)

    def expire(self, key, ttl):
        self.expired[key] = ttl
        return True

    def xadd(self, name, fields, maxlen=None, approximate=True):
        stream = self.streams.setdefault(name, [])
        msg_id = f"{len(stream) + 1}-0"
        stream.append((msg_id, dict(fields)))
        if maxlen and len(stream) > maxlen:
            self.streams[name] = stream[-int(maxlen):]
        return msg_id

    def xread(self, streams, block=None, count=None):
        out = []
        for name, last_id in streams.items():
            entries = []
            for mid, fields in self.streams.get(name, []):
                if _id_after(mid, last_id):
                    entries.append((mid, fields))
                    if count and len(entries) >= count:
                        break
            if entries:
                out.append((name, entries))
        return out or None


def _id_after(mid: str, last_id: str) -> bool:
    if last_id in ("0-0", "0", "$"):
        return last_id != "$"
    return tuple(int(x) for x in mid.split("-")) > tuple(int(x) for x in last_id.split("-"))


class FakeGraph:
    def __init__(self, events, error=None):
        self.events = events
        self.error = error
        self.calls = []

    def stream(self, state, config, stream_mode="custom"):
        self.calls.append((state, config, stream_mode))
        if self.error:
            raise self.error
        yield from self.events


def test_enqueue_fifo_and_meta(monkeypatch):
    fake = FakeRedis()
    monkeypatch.setattr(jq, "get_redis_queue", lambda: fake)
    a = jq.enqueue_job(1, 9, "改写甲", "原句甲")
    b = jq.enqueue_job(2, 9, "改写乙", "原句乙")
    assert a and b and a != b
    assert jq.queue_depth() == 2
    first = jq.pop_job(timeout=0)
    second = jq.pop_job(timeout=0)
    assert first["conversation_id"] == 1
    assert first["rewrite"] == "改写甲"
    assert second["job_id"] == b
    assert json.loads(fake.get(jq.meta_key(a)))["status"] == "queued"
    assert jq.pop_job(timeout=0) is None


def test_publish_and_iter_events(monkeypatch):
    fake = FakeRedis()
    monkeypatch.setattr(jq, "get_redis_queue", lambda: fake)
    job_id = "abc123"
    jq.publish_event(job_id, {"type": "status", "data": "正在思考中…"})
    jq.publish_event(job_id, {"type": "token", "data": "放"})
    jq.publish_event(job_id, {"type": "token", "data": "翁"})
    jq.publish_event(job_id, {"type": "done"})
    kinds = [ev["type"] for ev in jq.iter_job_events(job_id, wait_seconds=1)]
    assert kinds == ["status", "token", "token", "done"]
    text, err = jq.collect_job_text(job_id, wait_seconds=1)
    assert text == "放翁"
    assert err is None


def test_iter_events_timeout(monkeypatch):
    fake = FakeRedis()
    monkeypatch.setattr(jq, "get_redis_queue", lambda: fake)
    evs = list(jq.iter_job_events("missing", wait_seconds=0.15))
    assert evs[-1]["type"] == "error"
    assert "worker" in evs[-1]["data"]
    assert any(e["type"] == "ping" for e in evs)


def test_enqueue_none_when_redis_down(monkeypatch):
    monkeypatch.setattr(jq, "get_redis_queue", lambda: None)
    assert jq.enqueue_job(1, 1, "q", "o") is None
    assert jq.queue_depth() is None


def test_iter_events_redis_drop(monkeypatch):
    monkeypatch.setattr(jq, "get_redis_queue", lambda: None)
    evs = list(jq.iter_job_events("x", wait_seconds=1))
    assert evs == [{"type": "error", "data": "Redis 不可用，队列中断"}]


def test_sse_newlines_and_ping():
    assert format_sse("a\nb", event="status") == "event: status\ndata: a\ndata: b\n\n"
    assert event_to_sse({"type": "token", "data": "月"}) == "data: 月\n\n"
    assert event_to_sse({"type": "ping"}) == ": ping\n\n"
    assert event_to_sse({"type": "done"}) == "data: [DONE]\n\n"
    assert event_to_sse({"type": "error", "data": "boom"}).startswith("event: error")


def test_iter_graph_events_saves_and_done():
    g = FakeGraph([{"status": "正在检索库…"}, {"token": "答"}, {"token": "案"}])
    saved = []
    evs = list(iter_graph_events(
        7,
        "改写",
        "原句",
        graph_impl=g,
        load_history_fn=lambda cid: [{"role": "user", "content": "问"}],
        save_assistant_fn=lambda cid, text: saved.append((cid, text)),
    ))
    assert [e["type"] for e in evs] == ["status", "token", "token", "done"]
    assert saved == [(7, "答案")]
    state, cfg, mode = g.calls[0]
    assert state["rewrite"] == "改写"
    assert state["original"] == "原句"
    assert cfg["configurable"]["thread_id"] == "conv:7"
    assert mode == "custom"


def test_iter_graph_events_error_no_save():
    g = FakeGraph([], error=AuthError("API key无效"))
    saved = []
    evs = list(iter_graph_events(
        1,
        "q",
        "o",
        graph_impl=g,
        load_history_fn=lambda cid: [],
        save_assistant_fn=lambda cid, text: saved.append(text),
    ))
    assert evs == [{"type": "error", "data": "API key无效"}]
    assert saved == []


def test_run_chat_job_upstream():
    g = FakeGraph([], error=UpstreamError("挂了"))
    text, err = run_chat_job(
        1,
        "q",
        "o",
        graph_impl=g,
        load_history_fn=lambda cid: [],
        save_assistant_fn=lambda cid, t: None,
    )
    assert text == ""
    assert err == "挂了"


def test_process_job_publishes(monkeypatch):
    published = []
    statuses = []
    monkeypatch.setattr(
        "app.worker.iter_graph_events",
        lambda cid, rw, orig: iter([
            {"type": "status", "data": "正在思考中…"},
            {"type": "token", "data": "好"},
            {"type": "done"},
        ]),
    )
    monkeypatch.setattr("app.worker.publish_event", lambda jid, ev: published.append((jid, ev)) or True)
    monkeypatch.setattr("app.worker.set_job_status", lambda jid, st: statuses.append(st))
    process_job({
        "job_id": "j1",
        "conversation_id": 3,
        "rewrite": "q",
        "original": "o",
        "ts": 1e12,
    })
    assert statuses[-1] == "done"
    assert published[0][1]["type"] == "status"
    assert published[-1][1]["type"] == "done"


def test_queued_status(monkeypatch):
    from app.api.chat import _queued_status

    monkeypatch.setattr("app.api.chat.queue_depth", lambda: 3)
    assert "前面还有 2 个任务" in _queued_status()
    monkeypatch.setattr("app.api.chat.queue_depth", lambda: None)
    assert _queued_status() == "排队中…"


def test_process_job_expired(monkeypatch):
    published = []
    monkeypatch.setattr("app.worker.publish_event", lambda jid, ev: published.append(ev) or True)
    monkeypatch.setattr("app.worker.set_job_status", lambda jid, st: None)
    process_job({
        "job_id": "old",
        "conversation_id": 1,
        "ts": 1.0,
    })
    assert published == [{"type": "error", "data": "任务过期"}]
