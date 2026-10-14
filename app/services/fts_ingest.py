"""把诗词倒排写入 Postgres。不碰向量库，可单独跑。"""
from __future__ import annotations

from pathlib import Path

from sqlalchemy import delete, text

from app.config import settings
from app.db import Base, SessionLocal, engine, ensure_schema
from app.models.poem import Poem, PoemKey
from app.services.fts import ensure_fts
from app.services.poetry_parse import iter_poems

BATCH = 400


def ingest_fts(root: Path | None = None) -> None:
    import app.models  # noqa: F401

    Base.metadata.create_all(bind=engine)
    ensure_schema()
    root = root or Path(settings.poetry_data_dir)
    db = SessionLocal()
    try:
        db.execute(text("DROP INDEX IF EXISTS poems_body_trgm"))
        db.execute(text("DROP INDEX IF EXISTS poems_document_trgm"))
        db.execute(delete(PoemKey))
        db.execute(delete(Poem))
        db.commit()
        poems: list[Poem] = []
        n = 0
        keys_n = 0
        for rec in iter_poems(root):
            poem = Poem(
                author=rec["author"],
                title=rec["title"],
                dynasty=rec["dynasty"],
                body=rec["body"],
                document=rec["document"],
                title_aliases=rec["title_aliases"],
                source_path=rec["source_path"],
            )
            for key, kind in rec["keys"]:
                poem.keys.append(PoemKey(key=key, kind=kind, prefix=key[:2]))
                keys_n += 1
            poems.append(poem)
            if len(poems) >= BATCH:
                db.add_all(poems)
                db.commit()
                n += len(poems)
                poems = []
                print("倒排已写入", n, "keys", keys_n, flush=True)
        if poems:
            db.add_all(poems)
            db.commit()
            n += len(poems)
        total = db.scalar(text("SELECT COUNT(*) FROM poems"))
        db.commit()
        print("建 pg_trgm / 倒排索引", flush=True)
        ensure_fts()
        db.execute(text("ANALYZE poems"))
        db.execute(text("ANALYZE poem_keys"))
        db.commit()
        print("倒排最终", total, "keys", keys_n, flush=True)
    finally:
        db.close()
