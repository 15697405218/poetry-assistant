"""检索评估用的命中规则。"""
from __future__ import annotations

from app.services.rag import to_hans


def _blob(hit: dict) -> str:
    return to_hans(" ".join([
        str(hit.get("author") or ""),
        str(hit.get("title") or ""),
        str(hit.get("text") or ""),
    ]))


def _verse_keys(gold: dict) -> list[str]:
    """≥4 字才当诗句证据；「明月」「苏轼」这种短词不当精确率命中。"""
    authors = {to_hans(a) for a in (gold.get("gold_authors") or []) if a}
    keys = [to_hans(k) for k in (gold.get("gold_keywords") or []) if k]
    return [k for k in keys if len(k) >= 4 and k not in authors]


def relevant(hit: dict, gold: dict) -> bool:
    blob = _blob(hit)
    authors = [to_hans(a) for a in (gold.get("gold_authors") or []) if a]
    titles = [to_hans(t) for t in (gold.get("gold_titles") or []) if t]
    author = to_hans(str(hit.get("author") or ""))
    title = to_hans(str(hit.get("title") or ""))
    verses = _verse_keys(gold)
    if verses and any(k in blob for k in verses):
        return True
    if titles and any(t and t in title for t in titles):
        return True
    if titles:
        return False
    if authors and any(a and a in author for a in authors):
        return True
    keys = [to_hans(k) for k in (gold.get("gold_keywords") or []) if k]
    if not titles and not verses and keys and any(k and k in blob for k in keys):
        return True
    return False


def gold_recall(hits: list[dict], gold: dict) -> float:
    """召回按「作品」算：诗句对上了，作者/标题有历史异议也算找回。

    朱斌《登楼》和王之涣《登鹳雀楼》是同一首的通行异说，正文有「白日依山尽」即召回为 1。
    """
    authors = [to_hans(a) for a in (gold.get("gold_authors") or []) if a]
    titles = [to_hans(t) for t in (gold.get("gold_titles") or []) if t]
    keys = [to_hans(k) for k in (gold.get("gold_keywords") or []) if k]
    if keys and any(k and k in _blob(h) for h in hits for k in keys):
        return 1.0
    need = authors + titles
    if not need:
        return 1.0 if (hits and any(relevant(h, gold) for h in hits)) else (1.0 if hits else 0.0)
    blob = " ".join(_blob(h) for h in hits)
    hit_n = sum(1 for g in need if g and g in blob)
    return hit_n / len(need)


def precision(hits: list[dict], gold: dict, k: int) -> float:
    top = hits[:k]
    if not top:
        return 0.0
    return sum(1 for h in top if relevant(h, gold)) / len(top)


def f1(p: float, r: float) -> float:
    if p + r == 0:
        return 0.0
    return 2 * p * r / (p + r)
