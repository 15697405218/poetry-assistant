"""诗词检索：问句抽片段字面召回 + 原句/消歧句双路向量，再 CrossEncoder 重排。

字面针来自套话剥离和汉字窗口，不写死篇名。消歧在进图之前完成。
"""
from __future__ import annotations

import re

from app.config import settings
from app.observability import get_logger, stage
from app.services.line_index import lookup_index
from app.services.query_span import literal_needles, strip_question_shell
from app.services.rag import (
    _near,
    search_poems,
    search_poems_by_author,
    search_poems_by_title,
    search_poems_literal,
    to_hans,
)
from app.services.redis_store import cache_get, cache_key, cache_set
from app.services.rerank import rerank_hits

log = get_logger("poetry.retrieve")
RETRIEVE_ALGO = "fts.v4"
_EXACT = {"literal", "title", "index", "variant"}
_VERSE = {"literal", "index", "variant"}
# CrossEncoder 默认 sigmoid，0~1。短标题低分不保送（雨霖铃仿作），长诗句略放宽。
TITLE_KEEP_MIN = 0.35
VERSE_KEEP_MIN = 0.15
# 向量召回进来的低分不占 TopK，先保精确率。
VECTOR_KEEP_MIN = 0.22
_CLAUSE_RUN = re.compile(r"[\u4e00-\u9fff]{4,16}")


def _hit_key(hit: dict) -> tuple:
    return (
        hit.get("author") or "",
        hit.get("title") or "",
        (hit.get("text") or "")[:120],
    )


def _merge(groups: list[list[dict]]) -> list[dict]:
    seen: set[tuple] = set()
    out: list[dict] = []
    for hits in groups:
        for hit in hits:
            key = _hit_key(hit)
            if key in seen:
                continue
            seen.add(key)
            out.append(hit)
    return out


def _exact_keep(hit: dict) -> bool:
    """字面保送：只留精排分够的。短标题更严，诗句异文稍松。"""
    kind = hit.get("match")
    if kind not in _EXACT:
        return False
    score = hit.get("rerank")
    if score is None:
        return kind in _VERSE
    floor = VERSE_KEEP_MIN if kind in _VERSE else TITLE_KEEP_MIN
    title = str(hit.get("title") or "")
    if kind == "title" and len(title) < 4:
        floor = max(floor, 0.5)
    return float(score) >= floor


def blend_ranked(ranked: list[dict], top_k: int) -> list[dict]:
    """精排为主；高分字面/首句可保送，低分标题命中不占座。"""
    if top_k <= 0:
        return []
    def _score(hit: dict) -> float:
        return float(hit.get("rerank") or 0.0)

    keep = [h for h in ranked if _exact_keep(h)]
    rest = [
        h for h in ranked
        if not _exact_keep(h) and _score(h) >= VECTOR_KEEP_MIN
    ]
    keep.sort(key=_score, reverse=True)
    rest.sort(key=_score, reverse=True)
    reserve = min(len(keep), 2)
    picked: list[dict] = []
    seen: set[tuple] = set()
    for hit in keep[:reserve] + rest + keep[reserve:]:
        key = _hit_key(hit)
        if key in seen:
            continue
        seen.add(key)
        picked.append(hit)
        if len(picked) >= top_k:
            break
    return picked


def rank_query_of(original: str, rewritten: str) -> str:
    """精排问句用剥壳后的原句，不用消歧扩写（避免去打「白帝城」而不是原诗句）。"""
    core = strip_question_shell(original)
    if len(core) >= 4:
        return core
    return (rewritten or original or "").strip()


def exact_clause_needles(*texts: str) -> list[str]:
    """≥4 字句针：剥壳核心 + 连续汉字，不用滑窗，以免主题问句误跳过 GPU。"""
    seen: set[str] = set()
    out: list[str] = []

    def add(s: str) -> None:
        s = (s or "").strip()
        if s in seen:
            return
        if 4 <= len(s) <= 16 and _CLAUSE_RUN.fullmatch(s):
            seen.add(s)
            out.append(s)

    for text in texts:
        raw = (text or "").strip()
        if not raw:
            continue
        core = strip_question_shell(raw)
        add(core)
        add(raw)
        for run in _CLAUSE_RUN.findall(raw):
            add(run)
        for run in _CLAUSE_RUN.findall(core):
            add(run)
    return out


def is_strong_verse_exact(hit: dict, needles: list[str]) -> bool:
    """字面/倒排/异文命中且明显就是这句针。短标题、意象问句不算。"""
    kind = hit.get("match")
    if kind not in _VERSE:
        return False
    if kind in {"index", "variant"}:
        return True
    text = str(hit.get("text") or "")
    title = str(hit.get("title") or "")
    return any(n in text or n in title for n in needles)


def _distinct_poems(hits: list[dict]) -> set[tuple]:
    return {(h.get("author") or "", h.get("title") or "") for h in hits}


def retrieve_poems(
    q: str,
    original: str | None = None,
    author: str | None = None,
    top_k: int | None = None,
) -> tuple[list[dict], dict]:
    top_k = settings.retrieve_top_k if top_k is None else top_k
    recall_k = settings.recall_k
    q = to_hans(q or "").strip()
    original = to_hans(original or "").strip() or q
    dual = bool(q and original and q != original)
    meta = {
        "original": original,
        "rank_query": rank_query_of(original, q),
        "dual": dual,
        "index_version": settings.index_version,
        "algo": RETRIEVE_ALGO,
    }
    if not q and not original:
        return [], meta

    ck = f"poetry:ret:{cache_key(settings.index_version, RETRIEVE_ALGO, q, original, author or '', top_k, recall_k)}"
    cached = cache_get(ck)
    if isinstance(cached, dict) and isinstance(cached.get("hits"), list):
        log.info("retrieve cache_hit q=%s", q[:40])
        return cached["hits"], cached.get("meta") or meta

    queries: list[str] = []
    for text in (original, q):
        text = (text or "").strip()
        if text and text not in queries:
            queries.append(text)

    groups: list[list[dict]] = []
    seen_n: set[str] = set()
    uniq_needles: list[str] = []
    for n in literal_needles(original) + literal_needles(q):
        if n not in seen_n:
            seen_n.add(n)
            uniq_needles.append(n)
    cores = []
    for text in (original, q):
        core = strip_question_shell(text)
        if core and core not in cores:
            cores.append(core)
    with stage("index_recall", needles=len(uniq_needles) + len(cores)):
        for needle in uniq_needles + cores:
            groups.append(lookup_index(needle, limit=recall_k))
            if 2 <= len(needle) <= 12:
                groups.append(search_poems_by_title(needle, limit=recall_k))
    with stage("literal_recall", needles=len(uniq_needles)):
        for needle in uniq_needles:
            groups.append(search_poems_literal(needle, limit=recall_k))

    exact_merged = _merge(groups)
    clause_needles = exact_clause_needles(original, q)
    strong = [h for h in exact_merged if is_strong_verse_exact(h, clause_needles)]
    skip_vector = bool(strong)
    skip_rerank = bool(strong) and len(_distinct_poems(strong)) <= 1
    meta["skip_vector"] = skip_vector
    meta["skip_rerank"] = skip_rerank

    def _finish(ranked: list[dict]) -> tuple[list[dict], dict]:
        cache_set(ck, {"hits": ranked, "meta": meta})
        log.info(
            "retrieve dual=%s literal=%s hits=%s skip_vector=%s skip_rerank=%s q=%s",
            dual,
            len(uniq_needles),
            len(ranked),
            int(skip_vector),
            int(skip_rerank),
            (q or "")[:40],
        )
        return ranked, meta

    if skip_vector and skip_rerank:
        return _finish(blend_ranked(strong, top_k=top_k))

    if skip_vector:
        merged = strong
    else:
        with stage("vector_recall", n=len(queries), recall_k=recall_k):
            for query in queries:
                if author:
                    hits = search_poems_by_author(query, author, top_k=recall_k)
                else:
                    hits = search_poems(query, top_k=recall_k)
                groups.append(_near(hits))
        merged = _merge(groups)

    with stage("rerank", candidates=len(merged), top_k=top_k):
        ranked = rerank_hits(meta["rank_query"], merged, top_k=max(top_k, recall_k))
        ranked = blend_ranked(ranked, top_k=top_k)
    return _finish(ranked)
