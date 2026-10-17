"""诗词向量在 Postgres pgvector 里，和倒排同一份 poems 表。

查询走 SQL，多进程可共用，不再把 HNSW 载进每个 Python 进程。
问句仍要本地 embed（BGE）；索引在数据库服务端。
"""
from __future__ import annotations

from sqlalchemy import text

from app.config import settings
from app.db import SessionLocal, engine
from app.observability import get_logger

log = get_logger("poetry.vector")

EMBED_DIM = 1024


def ensure_pgvector() -> None:
    """扩展 + poems.embedding 列（兼容）+ poem_vectors 表（检索用）。"""
    with engine.begin() as conn:
        conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
        conn.execute(text(
            f"ALTER TABLE poems ADD COLUMN IF NOT EXISTS embedding vector({EMBED_DIM})"
        ))
        conn.execute(text(
            f"CREATE TABLE IF NOT EXISTS poem_vectors ("
            f"poem_id INTEGER PRIMARY KEY REFERENCES poems(id) ON DELETE CASCADE, "
            f"embedding vector({EMBED_DIM}) NOT NULL)"
        ))


def vector_poem_count() -> int:
    db = SessionLocal()
    try:
        n = db.scalar(text("SELECT COUNT(*) FROM poem_vectors"))
        return int(n or 0)
    except Exception:
        return 0
    finally:
        db.close()


def has_embedding_index() -> bool:
    db = SessionLocal()
    try:
        n = db.scalar(text(
            "SELECT COUNT(*) FROM pg_indexes WHERE indexname = 'poem_vectors_hnsw'"
        ))
        return int(n or 0) > 0
    except Exception:
        return False
    finally:
        db.close()


def create_embedding_index() -> None:
    with engine.begin() as conn:
        conn.execute(text("SET maintenance_work_mem = '2GB'"))
        conn.execute(text(
            "CREATE INDEX IF NOT EXISTS poem_vectors_hnsw "
            "ON poem_vectors USING hnsw (embedding vector_cosine_ops) "
            "WITH (m = 16, ef_construction = 64)"
        ))
        conn.execute(text("ANALYZE poem_vectors"))


def _vec_literal(values: list[float]) -> str:
    return "[" + ",".join(f"{x:.7f}" for x in values) + "]"


def embed_query(q: str) -> list[float]:
    from app.services.rag import make_ef, to_hans

    vec = make_ef()([to_hans(q or "").strip() or " "])[0]
    return [float(x) for x in vec]


def search_poems_vector(
    q: str,
    top_k: int = 5,
    author: str | None = None,
) -> list[dict]:
    q = (q or "").strip()
    if not q or top_k <= 0:
        return []
    try:
        vec = embed_query(q)
    except Exception:
        log.exception("embed_query failed")
        return []
    lit = _vec_literal(vec)
    ef = int(getattr(settings, "hnsw_search_ef", 64) or 64)
    db = SessionLocal()
    try:
        db.execute(text("SET hnsw.ef_search = :ef"), {"ef": ef})
        if author:
            rows = db.execute(
                text(
                    "SELECT p.id, p.document, p.author, p.title, p.dynasty, "
                    "(v.embedding <=> CAST(:vec AS vector)) AS distance "
                    "FROM poem_vectors v "
                    "JOIN poems p ON p.id = v.poem_id "
                    "WHERE p.author = :author "
                    "ORDER BY v.embedding <=> CAST(:vec AS vector) "
                    "LIMIT :k"
                ),
                {"vec": lit, "author": author, "k": top_k},
            ).all()
        else:
            rows = db.execute(
                text(
                    "SELECT p.id, p.document, p.author, p.title, p.dynasty, "
                    "(v.embedding <=> CAST(:vec AS vector)) AS distance "
                    "FROM poem_vectors v "
                    "JOIN poems p ON p.id = v.poem_id "
                    "ORDER BY v.embedding <=> CAST(:vec AS vector) "
                    "LIMIT :k"
                ),
                {"vec": lit, "k": top_k},
            ).all()
    except Exception:
        log.exception("pgvector search failed")
        return []
    finally:
        db.close()
    hits = []
    for row in rows:
        hits.append({
            "id": f"pg-{row.id}",
            "text": row.document,
            "author": row.author,
            "title": row.title,
            "dynasty": row.dynasty or "",
            "distance": float(row.distance) if row.distance is not None else None,
        })
    return hits
