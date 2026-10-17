"""把已有 Chroma embeddings 灌进 poems.embedding / authors，不再重算 34 万向量。

按 author + title + md5(document) 对齐。Chroma 的 id 和 Postgres 自增 id 对不上。
"""
from __future__ import annotations

import hashlib
from pathlib import Path

import chromadb
from sqlalchemy import text

from app.config import settings
from app.db import SessionLocal
from app.models.author import Author
from app.services.vector_store import (
    EMBED_DIM,
    create_embedding_index,
    ensure_pgvector,
    vector_poem_count,
)

BATCH = 400
MATCH_BATCH = 4000


def _md5(text_value: str) -> str:
    return hashlib.md5((text_value or "").encode("utf-8")).hexdigest()


def _vec_literal(values) -> str:
    return "[" + ",".join(f"{float(x):.7f}" for x in values) + "]"


def _chroma() -> chromadb.PersistentClient:
    persist = Path(settings.chroma_persist_dir)
    if not persist.is_dir():
        raise FileNotFoundError(f"Chroma 目录不存在: {persist}")
    return chromadb.PersistentClient(path=str(persist))


def migrate_authors() -> int:
    client = _chroma()
    try:
        col = client.get_collection("authors")
    except Exception as e:
        print("没有 authors 集合，跳过", e, flush=True)
        return 0
    ids = col.get(include=[])["ids"]
    print("authors 条数", len(ids), flush=True)
    db = SessionLocal()
    n = 0
    # autoflush=False：同一批里重名（如「无名」）查库看不到未提交的 add，会撞 ix_authors_name。
    seen: dict[str, Author] = {}
    try:
        for i in range(0, len(ids), BATCH):
            chunk = ids[i : i + BATCH]
            got = col.get(ids=chunk, include=["documents", "metadatas"])
            for doc, meta in zip(got["documents"], got["metadatas"]):
                meta = meta or {}
                name = str(meta.get("name") or "").strip()
                if not name:
                    continue
                bio = doc or ""
                path = str(meta.get("source_path") or "")
                row = seen.get(name)
                if row is None:
                    row = db.query(Author).filter(Author.name == name).first()
                if row:
                    row.bio = bio
                    row.source_path = path
                else:
                    row = Author(name=name, bio=bio, source_path=path)
                    db.add(row)
                seen[name] = row
                n += 1
            db.commit()
            print("authors 已写入", n, flush=True)
    finally:
        db.close()
    return n


def migrate_poem_embeddings() -> dict:
    ensure_pgvector()
    client = _chroma()
    col = client.get_collection("poetry")
    ids = col.get(include=[])["ids"]
    print("chroma poetry", len(ids), flush=True)

    db = SessionLocal()
    inserted = 0
    try:
        db.execute(text("DROP TABLE IF EXISTS poem_embedding_stage"))
        db.execute(text(
            f"CREATE TABLE poem_embedding_stage ("
            f"author TEXT, title TEXT, doc_md5 CHAR(32), "
            f"embedding vector({EMBED_DIM}))"
        ))
        db.commit()
        for i in range(0, len(ids), BATCH):
            chunk = ids[i : i + BATCH]
            got = col.get(ids=chunk, include=["embeddings", "documents", "metadatas"])
            rows = []
            for emb, doc, meta in zip(got["embeddings"], got["documents"], got["metadatas"]):
                if emb is None:
                    continue
                meta = meta or {}
                rows.append({
                    "a": str(meta.get("author") or ""),
                    "t": str(meta.get("title") or ""),
                    "m": _md5(doc or ""),
                    "e": _vec_literal(emb),
                })
            if rows:
                db.execute(
                    text(
                        "INSERT INTO poem_embedding_stage "
                        "(author, title, doc_md5, embedding) "
                        "VALUES (:a, :t, :m, CAST(:e AS vector))"
                    ),
                    rows,
                )
                db.commit()
            inserted += len(rows)
            if inserted % 5000 < BATCH:
                print("stage", inserted, "/", len(ids), flush=True)
        print("stage 完成", inserted, flush=True)
    finally:
        db.close()
    filled = match_staged_embeddings()
    return {"stage": inserted, **filled}


def _tune_match_session(db) -> None:
    db.execute(text("SET statement_timeout = 0"))
    db.execute(text("SET lock_timeout = 0"))
    db.execute(text("SET synchronous_commit = off"))
    db.execute(text("SET jit = off"))
    db.execute(text("SET work_mem = '32MB'"))
    db.execute(text("SET enable_hashjoin = off"))
    db.execute(text("SET enable_mergejoin = off"))


def _poem_id_range(db) -> tuple[int, int]:
    lo = db.scalar(text("SELECT id FROM poems ORDER BY id ASC LIMIT 1"))
    hi = db.scalar(text("SELECT id FROM poems ORDER BY id DESC LIMIT 1"))
    if lo is None or hi is None:
        raise RuntimeError("poems 为空，无法对齐向量")
    return int(lo), int(hi)


def match_staged_embeddings() -> dict:
    """按 id 分批把 stage 对齐到 poems.embedding，再建 HNSW。不重读 Chroma。"""
    db = SessionLocal()
    matched = 0
    leftover = 0
    try:
        n_stage = db.scalar(text("SELECT COUNT(*) FROM poem_embedding_stage"))
        print("stage 表", int(n_stage or 0), flush=True)
        _tune_match_session(db)
        db.execute(text("ALTER TABLE poems SET (autovacuum_enabled = false)"))
        db.commit()
        db.execute(text(
            "CREATE INDEX IF NOT EXISTS poem_embedding_stage_lookup "
            "ON poem_embedding_stage (author, title, doc_md5)"
        ))
        db.commit()

        lo, hi = _poem_id_range(db)
        print("poems id", lo, "..", hi, "批量", MATCH_BATCH, flush=True)

        for start in range(lo, hi + 1, MATCH_BATCH):
            end = min(start + MATCH_BATCH - 1, hi)
            n = db.execute(
                text(
                    "UPDATE poems p SET embedding = s.embedding "
                    "FROM poem_embedding_stage s "
                    "WHERE p.id BETWEEN :a AND :b "
                    "AND p.embedding IS NULL "
                    "AND p.author = s.author "
                    "AND p.title = s.title "
                    "AND md5(p.document) = s.doc_md5"
                ),
                {"a": start, "b": end},
            ).rowcount
            db.commit()
            matched += int(n or 0)
            print("对齐", start, "-", end, "本批", int(n or 0), "累计", matched, flush=True)

        for start in range(lo, hi + 1, MATCH_BATCH):
            end = min(start + MATCH_BATCH - 1, hi)
            n = db.execute(
                text(
                    "UPDATE poems p SET embedding = s.embedding "
                    "FROM poem_embedding_stage s "
                    "WHERE p.id BETWEEN :a AND :b "
                    "AND p.embedding IS NULL "
                    "AND p.author = s.author "
                    "AND p.title = s.title "
                    "AND NOT EXISTS ("
                    "  SELECT 1 FROM poems o "
                    "  WHERE o.id <> p.id "
                    "  AND o.author = p.author "
                    "  AND o.title = p.title"
                    ")"
                ),
                {"a": start, "b": end},
            ).rowcount
            db.commit()
            leftover += int(n or 0)
            print("标题回填", start, "-", end, "本批", int(n or 0), "累计", leftover, flush=True)

        filled = vector_poem_count()
        empty = db.scalar(text(
            "SELECT COUNT(*) FROM poems WHERE embedding IS NULL"
        ))
        print("poems 有向量", filled, "没有向量", int(empty or 0), flush=True)
        db.execute(text("ALTER TABLE poems SET (autovacuum_enabled = true)"))
        db.commit()
    finally:
        db.close()

    print("建 HNSW（可能要几分钟）", flush=True)
    create_embedding_index()
    print("HNSW 完成", flush=True)
    return {"filled": vector_poem_count(), "matched": matched, "leftover": leftover}


def migrate_from_chroma() -> dict:
    import app.models  # noqa: F401
    from app.db import Base, engine

    Base.metadata.create_all(bind=engine)
    ensure_pgvector()
    authors = migrate_authors()
    poems = migrate_poem_embeddings()
    return {"authors": authors, **poems}
