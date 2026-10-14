import pytest
from sqlalchemy import delete

from app.db import SessionLocal
from app.models.poem import Poem, PoemKey
from app.services.fts import ensure_fts, lookup_index, search_contains
from app.services.poetry_parse import first_clause, hamming, inverted_keys


def test_inverted_keys_first_line_and_title():
    keys = inverted_keys(["静夜思"], "床前看月光色，疑是地上霜。", ["明月光"])
    kinds = {k: kind for k, kind in keys}
    assert kinds.get("床前看月光色") == "line"
    assert kinds.get("疑是地上霜") == "line"
    assert kinds.get("床前看月光") == "line"
    assert kinds.get("静夜思") == "title"
    assert kinds.get("明月光") == "variant"


def test_parse_helpers_still_match_old_index():
    assert first_clause("床前看月光，疑是地上霜。") == "床前看月光"
    assert hamming("床前明月光", "床前看月光") == 1


def _seed() -> int:
    ensure_fts()
    db = SessionLocal()
    try:
        db.execute(delete(PoemKey).where(PoemKey.key.in_(["__fts测床前看月光__", "静夜思"])))
        db.execute(delete(Poem).where(Poem.title == "__fts测静夜思__"))
        db.commit()
        poem = Poem(
            author="李白",
            title="__fts测静夜思__",
            dynasty="唐",
            body="床前看月光疑是地上霜",
            document="作者：李白\n标题：静夜思\n床前看月光疑是地上霜\n异文：明月光",
            title_aliases="静夜思",
            source_path="test",
        )
        poem.keys.append(PoemKey(key="床前看月光", kind="line", prefix="床前"))
        poem.keys.append(PoemKey(key="静夜思", kind="title", prefix="静夜"))
        db.add(poem)
        db.commit()
        return poem.id
    finally:
        db.close()


def _cleanup(poem_id: int) -> None:
    db = SessionLocal()
    try:
        db.execute(delete(PoemKey).where(PoemKey.poem_id == poem_id))
        db.execute(delete(Poem).where(Poem.id == poem_id))
        db.commit()
    finally:
        db.close()


def test_postgres_exact_and_hamming_and_contains():
    try:
        poem_id = _seed()
    except Exception as e:
        pytest.skip(f"Postgres/pg_trgm 不可用: {e}")
    try:
        exact = lookup_index("床前看月光", limit=5)
        assert any(h["title"] == "__fts测静夜思__" for h in exact)
        fuzzy = lookup_index("床前明月光", limit=5)
        assert any(h["title"] == "__fts测静夜思__" for h in fuzzy)
        titled = lookup_index("静夜思", limit=5, kinds=("title",))
        assert any(h["title"] == "__fts测静夜思__" for h in titled)
        lit = search_contains("疑是地上霜", limit=5)
        assert any(h["title"] == "__fts测静夜思__" for h in lit)
    finally:
        _cleanup(poem_id)
