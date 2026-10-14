"""从 terms.draft.json 加载内存术语表。哈希匹配，不进 Chroma。"""
from __future__ import annotations

import json
from pathlib import Path

from zhconv import convert

from app.observability import get_logger, stage
from app.services.redis_store import cache_get, cache_key, cache_set

log = get_logger("poetry.terms")

_terms = None


def to_hans(text: str) -> str:
    return convert(text or "", "zh-hans")


def draft_path() -> Path:
    return Path(__file__).resolve().parents[2] / "data" / "terms.draft.json"


def load_terms(path: Path | None = None, reload: bool = False) -> dict:
    """返回 {by_id, surface_to_ids}。

    - by_id[id] = {id, canonical, aliases, category, count, source}
    - surface_to_ids[表面] = [id, ...]  一个表面可指向多条（ALIAS 一对多）
    id 用 canonical。
    """
    global _terms
    if _terms is not None and not reload:
        return _terms
    src = path or draft_path()
    rows = json.loads(src.read_text(encoding="utf-8"))
    by_id: dict[str, dict] = {}
    surface_to_ids: dict[str, list[str]] = {}
    note_surfaces: set[str] = set()
    for row in rows:
        if not isinstance(row, dict):
            continue
        if str(row.get("category") or "") != "注解":
            continue
        note = to_hans(str(row.get("canonical") or "")).strip()
        if len(note) >= 2:
            note_surfaces.add(note)

    def add_surface(surface: str, tid: str, *, is_alias: bool = False) -> None:
        s = to_hans(surface).strip()
        if len(s) < 2:
            return
        # 注解词（千里、春风）不当别人的别称，避免欲穷千里目 → 高骈
        if is_alias and s in note_surfaces:
            return
        ids = surface_to_ids.setdefault(s, [])
        if tid not in ids:
            ids.append(tid)

    for row in rows:
        if not isinstance(row, dict):
            continue
        canonical = to_hans(str(row.get("canonical") or "")).strip()
        if len(canonical) < 2:
            continue
        tid = canonical
        if tid not in by_id:
            by_id[tid] = {
                "id": tid,
                "canonical": canonical,
                "aliases": [],
                "category": str(row.get("category") or "未知"),
                "count": int(row.get("count") or 0),
                "source": str(row.get("source") or ""),
            }
        aliases = []
        for a in row.get("aliases") or []:
            a = to_hans(str(a)).strip()
            if a and a != canonical and a not in aliases:
                aliases.append(a)
        for a in aliases:
            if a not in by_id[tid]["aliases"]:
                by_id[tid]["aliases"].append(a)
        add_surface(canonical, tid)
        for a in by_id[tid]["aliases"]:
            add_surface(a, tid, is_alias=True)

    try:
        from app.services.author_aliases import extract_aliases, iter_author_rows

        for name, bio in iter_author_rows():
            if name not in by_id:
                continue
            for alias in extract_aliases(bio):
                if alias == name or alias in note_surfaces:
                    continue
                if alias not in by_id[name]["aliases"]:
                    by_id[name]["aliases"].append(alias)
                add_surface(alias, name, is_alias=True)
    except Exception:
        pass

    _terms = {
        "by_id": by_id,
        "surface_to_ids": surface_to_ids,
        "note_surfaces": note_surfaces,
    }
    return _terms


def lookup(text: str, terms: dict | None = None) -> list[dict]:
    """在问句里找术语，长表面优先，同一 id 只留一次。"""
    bag = terms or load_terms()
    s = to_hans(text)
    if not s:
        return []
    hits = []
    seen: set[str] = set()
    for surface in sorted(bag["surface_to_ids"], key=len, reverse=True):
        if surface not in s:
            continue
        for tid in bag["surface_to_ids"][surface]:
            if tid in seen:
                continue
            seen.add(tid)
            row = dict(bag["by_id"][tid])
            row["matched"] = surface
            hits.append(row)
    return hits


def rewrite_query(text: str, terms: dict | None = None) -> dict:
    """别称换成「本体（别称）」，给第一跳 LLM 和后续检索共用。

    放翁是谁 → 陆游（放翁）是谁。命中的就是本体则原样留下。
    返回 {original, query, hits}。history.content 存 original，rewrite 存 query。
    """
    original = to_hans(text).strip()
    if not original:
        return {"original": "", "query": "", "hits": []}
    ck = f"poetry:term:{cache_key(original)}"
    if terms is None:
        cached = cache_get(ck)
        if isinstance(cached, dict) and "query" in cached:
            return cached
    try:
        with stage("term_lookup", n=len(original)):
            hits = lookup(original, terms)
    except Exception:
        return {"original": original, "query": original, "hits": []}
    bag = terms or load_terms()
    occupied = [False] * len(original)
    replacements: list[tuple[int, int, str]] = []
    for h in hits:
        matched = str(h.get("matched") or "")
        can = str(h.get("canonical") or "").strip()
        if len(matched) < 2:
            continue
        start = original.find(matched)
        if start < 0:
            continue
        end = start + len(matched)
        if any(occupied[start:end]):
            continue
        for i in range(start, end):
            occupied[i] = True
        if matched == can or len(can) < 2:
            continue
        if str(h.get("category") or "") == "注解":
            continue
        ids = bag["surface_to_ids"].get(matched) or []
        if len(ids) > 1:
            continue
        if matched in bag.get("note_surfaces", ()):
            continue
        replacements.append((start, end, f"{can}（{matched}）"))
    replacements.sort(key=lambda x: x[0])
    parts: list[str] = []
    last = 0
    for start, end, new in replacements:
        parts.append(original[last:start])
        parts.append(new)
        last = end
    parts.append(original[last:])
    out = {"original": original, "query": "".join(parts), "hits": [
        {"matched": h.get("matched"), "canonical": h.get("canonical"), "category": h.get("category")}
        for h in hits
    ]}
    if terms is None:
        cache_set(ck, out)
    if out["hits"]:
        log.info("terms hits=%s query=%s", len(out["hits"]), out["query"][:40])
    return {"original": out["original"], "query": out["query"], "hits": hits}
