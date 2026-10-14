"""从问句抽出字面检索针。只有套话和汉字跨度，不写死任何篇名/作者。"""
from __future__ import annotations

import re

from app.services.rag import to_hans

# 问句套话，长的在前。不是诗题表。
_TAILS = (
    "最有名的那首",
    "是哪一首诗",
    "是哪一首词",
    "是哪一首",
    "是哪首诗",
    "是哪首词",
    "是哪首",
    "哪一首诗",
    "哪一首词",
    "哪一首",
    "哪首诗",
    "哪首词",
    "哪首",
    "是谁写的",
    "谁写的",
    "是谁的词",
    "是谁的诗",
    "是谁的",
    "写过哪些诗",
    "写过哪些",
    "的出处",
    "出处",
    "下一句是什么",
    "下一句",
    "的意象",
    "那首",
    "是谁",
)
_HEADS = ("请问", "帮我查", "帮我找")
_CJK_RUN = re.compile(r"[\u4e00-\u9fff]{2,}")


def strip_question_shell(text: str) -> str:
    """去掉问答套话，留下可能是诗句/篇名的部分。"""
    s = to_hans(text).strip()
    s = s.replace("《", "").replace("》", "").replace("?", "").replace("？", "")
    changed = True
    while s and changed:
        changed = False
        for head in _HEADS:
            if s.startswith(head):
                s = s[len(head):].strip()
                changed = True
        for tail in _TAILS:
            if s.endswith(tail) and len(s) > len(tail):
                s = s[: -len(tail)].strip()
                changed = True
        if "那首" in s:
            nxt = s.replace("那首", "", 1).strip()
            if nxt and nxt != s:
                s = nxt
                changed = True
        if s.endswith("的") and len(s) > 3:
            s = s[:-1].strip()
            changed = True
    return s


def literal_needles(text: str, limit: int = 6) -> list[str]:
    """字面检索用的针：剥壳后的整段 + 首尾窗口。全库变大也不依赖评估集。"""
    raw = to_hans(text).strip()
    core = strip_question_shell(raw)
    seen: set[str] = set()
    out: list[str] = []

    def add(s: str) -> None:
        s = (s or "").strip()
        if len(s) < 2 or s in seen:
            return
        seen.add(s)
        out.append(s)

    add(core)
    add(raw)
    for src in (core, raw):
        for run in _CJK_RUN.findall(src):
            if len(run) >= 4:
                add(run)
        if len(src) >= 7:
            add(src[:7])
            add(src[-7:])
        if len(src) >= 5:
            add(src[:5])
            add(src[-5:])
    out.sort(key=len, reverse=True)
    return out[:limit]
