from pathlib import Path
from app.config import settings
import chromadb
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction
from zhconv import convert


def _embed_device() -> str:
    try:
        import torch

        if torch.cuda.is_available():
            return "cuda"
    except Exception:
        pass
    return "cpu"


def make_ef():
    """BGE 要归一化；有 GPU 就用 GPU。Chroma 集合空间须是 cosine。"""
    return SentenceTransformerEmbeddingFunction(
        model_name=settings.embedding_model,
        device=_embed_device(),
        normalize_embeddings=True,
    )
_client= None
_collection= None
_author_collection= None
def _apply_search_ef(col) -> None:
    """查询 ef 可改、不用重灌。默认 10 对 34 万偏紧。"""
    ef = int(getattr(settings, "hnsw_search_ef", 64) or 64)
    try:
        meta = dict(col.metadata or {})
        if int(meta.get("hnsw:search_ef") or 0) == ef:
            return
        meta.setdefault("hnsw:space", "cosine")
        meta["hnsw:search_ef"] = ef
        col.modify(metadata=meta)
    except Exception:
        pass


def get_collection() -> chromadb.PersistentClient:
    global _client,_collection
    if _collection is None:
        _client = chromadb.PersistentClient(path=str(Path(settings.chroma_persist_dir)))
        _collection = _client.get_collection(name="poetry",embedding_function=make_ef())
        _apply_search_ef(_collection)
    return _collection

def get_author_collection() -> chromadb.PersistentClient:
    global _client,_author_collection
    if _author_collection is None:
        _client = chromadb.PersistentClient(path=str(Path(settings.chroma_persist_dir)))
        _author_collection = _client.get_collection(name="authors",embedding_function=make_ef())
    return _author_collection

def search_poems(q: str, top_k: int = 5, where: dict | None = None) -> list[dict]:
    q = to_hans(q)
    kwargs = {"query_texts": [q], "n_results": top_k}
    if where:
        kwargs["where"] = where
    result = get_collection().query(**kwargs)
    docs = result["documents"][0] if result.get("documents") else []
    metas = result["metadatas"][0] if result.get("metadatas") else []
    dists = result["distances"][0] if result.get("distances") else []
    ids = result["ids"][0] if result.get("ids") else []
    hits = []
    for i, doc in enumerate(docs):
        meta = (metas[i] if i < len(metas) else None) or {}
        dist = float(dists[i]) if i < len(dists) else None
        hits.append({
            "id": ids[i] if i < len(ids) else None,
            "text": doc,
            "author": meta.get("author"),
            "title": meta.get("title"),
            "dynasty": meta.get("dynasty"),
            "distance": dist,
        })
    return hits


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
    col = get_author_collection()
    for candidate in _name_variants(name):
        got = col.get(where={"name": candidate}, limit=1)
        docs = got.get("documents") or []
        if docs:
            return docs[0]
    return None


def search_poems_by_author(q: str, author: str, top_k: int) -> list[dict]:
    for candidate in _name_variants(author):
        hits = search_poems(q, top_k=top_k, where={"author": candidate})
        if hits:
            return hits
    return []


def search_poems_by_title(title: str, limit: int = 15) -> list[dict]:
    """标题精确命中，走 Postgres 倒排，不再扫 Chroma metadata。"""
    from app.services.fts import search_title

    return search_title(title, limit=limit)


def search_poems_literal(q: str, limit: int = 5) -> list[dict]:
    """正文/异名子串，走 pg_trgm，不再 Chroma $contains 全表扫。"""
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