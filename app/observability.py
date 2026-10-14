"""链路日志和阶段耗时。定位瓶颈用，不是完整 APM。"""
from __future__ import annotations

import logging
import time
from contextlib import contextmanager

_configured = False


def setup_logging() -> None:
    global _configured
    if _configured:
        return
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )
    _configured = True


def get_logger(name: str = "poetry") -> logging.Logger:
    setup_logging()
    return logging.getLogger(name)


@contextmanager
def stage(name: str, **extra):
    log = get_logger("poetry.trace")
    start = time.perf_counter()
    try:
        yield
    finally:
        ms = (time.perf_counter() - start) * 1000
        bits = " ".join(f"{k}={v}" for k, v in extra.items() if v is not None)
        log.info("stage=%s ms=%.1f %s", name, ms, bits)
