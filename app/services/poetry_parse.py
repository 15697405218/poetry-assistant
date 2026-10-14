"""诗词 JSON 解析。向量灌库和 Postgres 倒排共用，避免两套规则漂。"""
from __future__ import annotations

import json
import re
from collections.abc import Iterator
from pathlib import Path

from zhconv import convert

SKIP_PARTS = {"rank", "strains", "error", "御定全唐詩"}
_YIZUO = re.compile(r"一作[「『“\"]?([\u4e00-\u9fff]{2,12})")
_SPLIT = re.compile(r"[，。！？、；\n]")


def to_hans(text: str) -> str:
    return convert(text or "", "zh-hans")


def is_skip(path: Path) -> bool:
    if "author" in path.name.lower():
        return True
    return any(part in SKIP_PARTS for part in path.parts)


def item_body(item: dict) -> str:
    raw = item.get("paragraphs") or item.get("para") or item.get("content") or ""
    if isinstance(raw, list):
        return "".join(str(x) for x in raw)
    return str(raw)


def item_title(item: dict) -> str:
    return item.get("title") or item.get("rhythmic") or item.get("chapter") or "未知"


def item_title_aliases(item: dict) -> list[str]:
    seen: list[str] = []
    for key in ("title", "rhythmic", "chapter"):
        title = to_hans(str(item.get(key) or "")).strip()
        if title and title != "未知" and title not in seen:
            seen.append(title)
    return seen


def item_variants(item: dict) -> list[str]:
    notes = item.get("notes") or item.get("note") or ""
    if isinstance(notes, list):
        blob = "".join(str(x) for x in notes)
    else:
        blob = str(notes)
    found: list[str] = []
    for v in _YIZUO.findall(to_hans(blob)):
        if v not in found:
            found.append(v)
    return found


def item_document(author: str, titles: list[str], body: str, variants: list[str]) -> str:
    title = titles[0] if titles else "未知"
    parts = [f"作者：{author}", f"标题：{title}", body]
    extra = [t for t in titles[1:]]
    if extra:
        parts.append("异名：" + " ".join(extra))
    if variants:
        parts.append("异文：" + " ".join(variants))
    return "\n".join(parts)


def first_clause(body: str) -> str:
    text = to_hans(body).strip()
    if not text:
        return ""
    return _SPLIT.split(text, 1)[0].strip()


def hamming(a: str, b: str) -> int:
    if len(a) != len(b):
        return 99
    return sum(x != y for x, y in zip(a, b))


def dedup_key(author: str, title: str, body: str) -> str:
    return f"{author}\t{title}\t{body[:80]}"


def all_clauses(body: str) -> list[str]:
    out: list[str] = []
    for part in _SPLIT.split(to_hans(body)):
        clause = part.strip()
        if 4 <= len(clause) <= 16 and clause not in out:
            out.append(clause)
        if len(out) >= 20:
            break
    return out


def inverted_keys(titles: list[str], body: str, variants: list[str]) -> list[tuple[str, str]]:
    """(key, kind) kind 为 line / title / variant。line 含全文各句，不只有首句。"""
    out: list[tuple[str, str]] = []
    seen: set[tuple[str, str]] = set()

    def add(key: str, kind: str) -> None:
        key = (key or "").strip()
        if len(key) < 2:
            return
        item = (key, kind)
        if item in seen:
            return
        seen.add(item)
        out.append(item)

    for clause in all_clauses(body):
        add(clause, "line")
        if len(clause) >= 5:
            add(clause[:5], "line")
    for y in variants:
        add(y, "variant")
    for t in titles:
        add(t, "title")
    return out


def iter_poems(root: Path) -> Iterator[dict]:
    seen: set[str] = set()
    for file in root.rglob("*.json"):
        if is_skip(file):
            continue
        try:
            data = json.loads(file.read_text(encoding="utf-8"))
        except Exception:
            continue
        if not isinstance(data, list):
            continue
        dynasty = to_hans(file.parent.name)
        for item in data:
            if not isinstance(item, dict):
                continue
            body = to_hans(item_body(item)).strip()
            if len(body) < 2:
                continue
            author = to_hans(str(item.get("author") or "未知"))
            titles = item_title_aliases(item) or [to_hans(str(item_title(item)))]
            title = titles[0]
            key = dedup_key(author, title, body)
            if key in seen:
                continue
            seen.add(key)
            variants = item_variants(item)
            yield {
                "author": author,
                "title": title,
                "title_aliases": " ".join(titles),
                "dynasty": dynasty,
                "body": body,
                "document": item_document(author, titles, body, variants),
                "source_path": str(file),
                "titles": titles,
                "variants": variants,
                "keys": inverted_keys(titles, body, variants),
            }
