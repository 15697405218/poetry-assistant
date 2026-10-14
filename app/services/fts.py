"""Postgres 倒排 + pg_trgm 正文检索。替代 Chroma $contains 和启动现场扫 JSON。"""
from __future__ import annotations

from sqlalchemy import func, or_, select, text

from app.db import SessionLocal, engine
from app.models.poem import Poem, PoemKey
from app.observability import get_logger
from app.services.poetry_parse import hamming, to_hans

log = get_logger("poetry.fts")

_KIND_MATCH = {"line": "index", "title": "title", "variant": "variant"}


def ensure_fts() -> None:
    with engine.begin() as conn:
        conn.execute(text("CREATE EXTENSION IF NOT EXISTS pg_trgm"))
        conn.execute(text(
            "CREATE INDEX IF NOT EXISTS poems_body_trgm ON poems USING gin (body gin_trgm_ops)"
        ))
        conn.execute(text(
            "CREATE INDEX IF NOT EXISTS poems_document_trgm ON poems USING gin (document gin_trgm_ops)"
        ))
        conn.execute(text(
            "CREATE INDEX IF NOT EXISTS poem_keys_prefix_len "
            "ON poem_keys (prefix, char_length(key))"
        ))


def poem_count() -> int:
    db = SessionLocal()
    try:
        return int(db.scalar(select(func.count()).select_from(Poem)) or 0)
    except Exception:
        return 0
    finally:
        db.close()


def _hit(poem: Poem, match: str) -> dict:
    return {
        "id": f"fts-{poem.id}",
        "text": poem.document,
        "author": poem.author,
        "title": poem.title,
        "dynasty": poem.dynasty,
        "distance": 0.0,
        "match": match,
    }


def _like_pattern(q: str) -> str:
    escaped = q.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")
    return f"%{escaped}%"


def search_contains(q: str, limit: int = 15) -> list[dict]:
    q = to_hans(q or "").strip()
    if len(q) < 2:
        return []
    pat = _like_pattern(q)
    db = SessionLocal()
    try:
        rows = db.scalars(
            select(Poem)
            .where(or_(Poem.body.like(pat, escape="\\"), Poem.document.like(pat, escape="\\")))
            .limit(limit)
        ).all()
        return [_hit(p, "literal") for p in rows]
    finally:
        db.close()


def search_title(title: str, limit: int = 15) -> list[dict]:
    return lookup_index(title, limit=limit, kinds=("title",))


def lookup_index(
    needle: str,
    limit: int = 15,
    kinds: tuple[str, ...] = ("title", "line", "variant"),
) -> list[dict]:
    """精确 key，再对同长度、同二字前缀做汉明距离 1。"""
    q = to_hans(needle or "").strip()
    if len(q) < 2:
        return []
    db = SessionLocal()
    hits: list[dict] = []
    seen: set[tuple] = set()

    def take(poem: Poem, match: str) -> bool:
        key = (poem.author, poem.title, (poem.document or "")[:40])
        if key in seen:
            return False
        seen.add(key)
        hits.append(_hit(poem, match))
        return len(hits) >= limit

    try:
        exact = db.execute(
            select(PoemKey.kind, Poem)
            .join(Poem, Poem.id == PoemKey.poem_id)
            .where(PoemKey.key == q, PoemKey.kind.in_(kinds))
        ).all()
        for kind, poem in exact:
            if take(poem, _KIND_MATCH.get(kind, "index")):
                return hits
        if len(hits) >= limit or len(q) < 5:
            return hits
        if "line" not in kinds and "variant" not in kinds:
            return hits
        fuzzy_kinds = [k for k in kinds if k in {"line", "variant"}]
        cands = db.execute(
            select(PoemKey.key, PoemKey.kind, Poem)
            .join(Poem, Poem.id == PoemKey.poem_id)
            .where(
                PoemKey.prefix == q[:2],
                func.char_length(PoemKey.key) == len(q),
                PoemKey.kind.in_(fuzzy_kinds),
            )
        ).all()
        for key, kind, poem in cands:
            if hamming(q, key) != 1:
                continue
            if take(poem, _KIND_MATCH.get(kind, "index")):
                break
        return hits
    finally:
        db.close()
