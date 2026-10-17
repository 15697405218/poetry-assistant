from fastapi import FastAPI

from app.observability import setup_logging
from app.db import Base, engine, ensure_schema

from app.models import User, Conversation, Message, Poem, PoemKey, Author

from app.api.auth import router as auth_router

from app.api.chat import router as chat_router

from app.api.conversations import router as conversations_router

setup_logging()
app = FastAPI(title="Poetry Assistant", description="a Poetry Assistant")
app.include_router(auth_router)
app.include_router(chat_router)
app.include_router(conversations_router)

Base.metadata.create_all(bind=engine)
ensure_schema()

@app.get("/health")
def health_check():
    from app.config import settings
    from app.services.job_queue import queue_depth
    from app.services.redis_store import redis_ok
    return {
        "status": "ok",
        "redis": redis_ok(),
        "index_version": settings.index_version,
        "fts_poems": _fts_count(),
        "vector_poems": _vector_count(),
        "queue": queue_depth(),
    }


def _fts_count() -> int:
    from app.services.fts import poem_count

    return poem_count()


def _vector_count() -> int:
    from app.services.vector_store import vector_poem_count

    return vector_poem_count()