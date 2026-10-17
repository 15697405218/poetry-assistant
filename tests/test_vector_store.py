import pytest
from sqlalchemy import text

from app.services.vector_store import EMBED_DIM, _vec_literal, ensure_pgvector


def test_vec_literal_roundtrip_shape():
    lit = _vec_literal([0.1, -0.2, 0.3])
    assert lit.startswith("[")
    assert lit.endswith("]")
    assert lit.count(",") == 2


def test_pgvector_extension_optional():
    try:
        ensure_pgvector()
    except Exception as e:
        pytest.skip(f"pgvector 不可用: {e}")
    from app.db import SessionLocal

    db = SessionLocal()
    try:
        n = db.scalar(text(
            "SELECT COUNT(*) FROM information_schema.columns "
            "WHERE table_name = 'poems' AND column_name = 'embedding'"
        ))
        assert int(n or 0) == 1
        dim = db.scalar(text(
            "SELECT atttypmod FROM pg_attribute "
            "WHERE attrelid = 'poems'::regclass AND attname = 'embedding'"
        ))
        # vector atttypmod is dim + 4 in some versions; just assert column exists
        assert dim is None or int(dim) in {EMBED_DIM, EMBED_DIM + 4}
    finally:
        db.close()
