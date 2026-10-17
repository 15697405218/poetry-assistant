from app.config import settings
from zhconv import convert

from app.db import SessionLocal
from app.models.author import Author
from app.services.vector_store import search_poems_vector


def _embed_device() -> str:
    try:
        import torch

        if torch.cuda.is_available():
            return "cuda"
    except Exception:
        pass
    return "cpu"


def make_ef():
    """BGE 要归一化；有 GPU 就用 GPU。只用来 embed 问句；ingest 仍可把同一函数喂给 Chroma。"""
    from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction

    return SentenceTransformerEmbeddingFunction(
        model_name=settings.embedding_model,
        device=_embed_device(),
        normalize_embeddings=True,
    )


def search_poems(q: str, top_k: int = 5, where: dict | None = None) -> list[dict]:
    q = to_hans(q)
    author = None
    if where and where.get("author"):
        author = str(where.get("author") or "").strip() or None
    return search_poems_vector(q, top_k=top_k, author=author)


POEM_DISTANCE_MAX = settings.poem_distance_max


def to_hans(text: str) -> str:
    return convert(text or "", "zh-hans")


def to_hant(text: str) -> str:
    return convert(text or "", "zh-hant")


def _name_variants(name: str) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for candidate in (name, to_hans(name), to_hant(name)):
        if candidate and candidate not in seen:
            seen.add(candidate)
            out.append(candidate)
    return out


def get_author_bio(name: str) -> str | None:
    if not name:
        return None
    db = SessionLocal()
    try:
        for candidate in _name_variants(name):
            row = db.query(Author).filter(Author.name == candidate).first()
            if row and row.bio:
                return row.bio
    finally:
        db.close()
    return None


def search_poems_by_author(q: str, author: str, top_k: int) -> list[dict]:
    for candidate in _name_variants(author):
        hits = search_poems(q, top_k=top_k, where={"author": candidate})
        if hits:
            return hits
    return []


def search_poems_by_title(title: str, limit: int = 15) -> list[dict]:
    """标题精确命中，走 Postgres 倒排。"""
    from app.services.fts import search_title

    return search_title(title, limit=limit)


def search_poems_literal(q: str, limit: int = 5) -> list[dict]:
    """正文/异名子串，走 pg_trgm。"""
    from app.services.fts import search_contains

    return search_contains(q, limit=limit)


def _near(poems: list[dict]) -> list[dict]:
    return [
        p for p in poems
        if p.get("distance") is None or p["distance"] <= POEM_DISTANCE_MAX
    ]


def _format_poems(hits: list[dict]) -> str:
    if not hits:
        return "没有检索到足够相近的诗文"
    parts = []
    high = [
        p for p in hits
        if p.get("rerank") is None or float(p.get("rerank") or 0) >= 0.35
    ][:2]
    primary = high or hits[:1]
    primary_ids = {id(p) for p in primary}
    for p in hits:
        src = p.get("id")
        tag = "主出处" if id(p) in primary_ids else "参考"
        head = f"【{tag}#{src} {p.get('author')}·{p.get('title')}】" if src else (
            f"【{tag} {p.get('author')}·{p.get('title')}】"
        )
        extra = []
        if p.get("rerank") is not None:
            extra.append(f"rerank={float(p['rerank']):.3f}")
        if extra:
            head += " " + " ".join(extra)
        parts.append(f"{head}\n{p.get('text')}")
    return (
        "只根据【主出处】回答；【参考】仅在主出处对不上问句时才用，不要并列罗列。"
        "\n\n" + "\n\n".join(parts)
    )
