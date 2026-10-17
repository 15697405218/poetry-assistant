"""对话 GPU worker：BLPOP poetry:jobs，跑 retrieve+generate，往任务 Stream 推事件。

单独进程启动，和 uvicorn 并列：

  python scripts/worker.py
"""
from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor
from threading import Semaphore

from app.config import settings
from app.observability import get_logger, setup_logging
from app.services.chat_run import iter_graph_events
from app.services.job_queue import pop_job, publish_event, set_job_status
from app.services.redis_store import redis_queue_ok

log = get_logger("poetry.worker")


def process_job(job: dict) -> None:
    job_id = str(job.get("job_id") or "")
    cid = int(job.get("conversation_id") or 0)
    if not job_id or not cid:
        log.warning("skip malformed job %s", job)
        return
    created = float(job.get("ts") or 0)
    if created and (time.time() - created) > max(int(settings.job_ttl_seconds), 60):
        publish_event(job_id, {"type": "error", "data": "任务过期"})
        set_job_status(job_id, "expired")
        return
    set_job_status(job_id, "running")
    log.info("job start id=%s conv=%s", job_id, cid)
    try:
        for ev in iter_graph_events(
            cid,
            job.get("rewrite") or "",
            job.get("original") or "",
        ):
            if not publish_event(job_id, ev):
                log.warning("job %s publish failed, stop", job_id)
                set_job_status(job_id, "error")
                return
            if ev.get("type") in {"done", "error"}:
                set_job_status(job_id, "error" if ev.get("type") == "error" else "done")
                return
        publish_event(job_id, {"type": "done"})
        set_job_status(job_id, "done")
    except Exception as e:
        log.exception("job failed id=%s", job_id)
        publish_event(job_id, {"type": "error", "data": str(e)})
        set_job_status(job_id, "error")


def run_forever() -> None:
    setup_logging()
    n = max(int(settings.gpu_max_inflight), 1)
    log.info(
        "chat worker started inflight=%s redis=%s",
        n,
        settings.redis_url,
    )
    if n == 1:
        _loop_serial()
        return
    _loop_pool(n)


def _loop_serial() -> None:
    while True:
        job = pop_job(timeout=5)
        if job is None:
            if not redis_queue_ok():
                log.warning("Redis 不可用，5s 后重试")
                time.sleep(5)
            continue
        process_job(job)


def _loop_pool(n: int) -> None:
    sem = Semaphore(n)
    with ThreadPoolExecutor(max_workers=n) as pool:
        while True:
            sem.acquire()
            job = pop_job(timeout=5)
            if job is None:
                sem.release()
                if not redis_queue_ok():
                    log.warning("Redis 不可用，5s 后重试")
                    time.sleep(5)
                continue

            def _run(j: dict = job) -> None:
                try:
                    process_job(j)
                finally:
                    sem.release()

            pool.submit(_run)


def main() -> None:
    run_forever()


if __name__ == "__main__":
    main()
