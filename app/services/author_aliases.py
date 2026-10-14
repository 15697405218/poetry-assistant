"""从作者简介里抽字/号/居士，叠进术语表。规则通用，不写死评估集别称。"""
from __future__ import annotations

import json
import re
from pathlib import Path

from app.config import settings
from app.services.rag import to_hans

_ALIAS = re.compile(
    r"(?:字|号|小字|别号|自号|又称|又名|亦名|世称|人称)([\u4e00-\u9fff]{2,8})"
)
_HONOR = re.compile(r"([\u4e00-\u9fff]{2,6}(?:居士|野老))")


def extract_aliases(desc: str) -> list[str]:
    text = to_hans(desc or "")
    found: list[str] = []
    for pat in (_ALIAS, _HONOR):
        for m in pat.findall(text):
            alias = m.strip()
            if 2 <= len(alias) <= 8 and alias not in found:
                found.append(alias)
    return found


def iter_author_rows(root: Path | None = None) -> list[tuple[str, str]]:
    root = root or Path(settings.poetry_data_dir)
    rows: list[tuple[str, str]] = []
    if not root.is_dir():
        return rows
    for file in root.rglob("*.json"):
        if "author" not in file.name.lower() or "error" in file.parts:
            continue
        try:
            data = json.loads(file.read_text(encoding="utf-8"))
        except Exception:
            continue
        if not isinstance(data, list):
            continue
        for item in data:
            if not isinstance(item, dict):
                continue
            name = to_hans(str(item.get("name") or "").strip())
            bio = to_hans(str(
                item.get("desc")
                or item.get("description")
                or item.get("short_description")
                or ""
            ).strip())
            if name and bio:
                rows.append((name, bio))
    return rows
