from sqlalchemy import create_engine, inspect, text
from sqlalchemy.orm import DeclarativeBase, sessionmaker

from app.config import settings

engine = create_engine(settings.database_url)
SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False)

class Base(DeclarativeBase):
    pass


def ensure_schema() -> None:
    """create_all 不会给已有表加列，旧库需要补列。"""
    insp = inspect(engine)
    tables = set(insp.get_table_names())
    if "messages" in tables:
        cols = {c["name"] for c in insp.get_columns("messages")}
        if "rewrite" not in cols:
            with engine.begin() as conn:
                conn.execute(text("ALTER TABLE messages ADD COLUMN rewrite TEXT"))
    if "conversations" in tables:
        cols = {c["name"] for c in insp.get_columns("conversations")}
        with engine.begin() as conn:
            if "summary" not in cols:
                conn.execute(text("ALTER TABLE conversations ADD COLUMN summary TEXT"))
            if "summarized_through_id" not in cols:
                conn.execute(
                    text("ALTER TABLE conversations ADD COLUMN summarized_through_id INTEGER")
                )
    if "poem_keys" in tables:
        cols = {c["name"]: c for c in insp.get_columns("poem_keys")}
        key_col = cols.get("key")
        if key_col is not None and getattr(key_col.get("type"), "length", None) == 64:
            with engine.begin() as conn:
                conn.execute(text("ALTER TABLE poem_keys ALTER COLUMN key TYPE VARCHAR(512)"))
    from app.services.fts import ensure_fts
    from app.services.vector_store import ensure_pgvector

    try:
        ensure_fts()
    except Exception:
        pass
    try:
        ensure_pgvector()
    except Exception:
        pass