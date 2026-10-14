"""对话评测的规则打分：看答案里有没有金标，工具用得对不对。

不调 LLM 当裁判，避免再烧一轮 token。闲聊假装引用本库算失败。
"""
from __future__ import annotations

import re

from app.services.rag import to_hans

_LIB = {"search_poems", "search_poems_literal", "get_author_bio"}
_CITE = re.compile(r"【(?:主出处|参考|出处)|\[\s*[\u4e00-\u9fff].*?·")


def _blob(text: str) -> str:
    return to_hans(text or "")


def used_lib(tools: list[str]) -> bool:
    return any(t in _LIB for t in tools)


def used_web(tools: list[str]) -> bool:
    return "search_web" in tools


def score_case(gold: dict, answer: str, tools: list[str]) -> dict:
    blob = _blob(answer)
    must = [_blob(x) for x in (gold.get("must") or []) if x]
    any_of = [_blob(x) for x in (gold.get("any") or []) if x]
    forbid = [_blob(x) for x in (gold.get("forbid") or []) if x]
    expect = gold.get("expect_tools") or "optional"

    must_ok = all(m in blob for m in must) if must else True
    any_ok = (not any_of) or any(a in blob for a in any_of)
    forbid_ok = not any(f and f in blob for f in forbid)

    lib, web = used_lib(tools), used_web(tools)
    if expect == "required":
        want = gold.get("tool") or "lib"
        if want == "bio":
            tool_ok = "get_author_bio" in tools or lib
        elif want == "web":
            tool_ok = web
        else:
            tool_ok = lib
    elif expect == "forbidden":
        tool_ok = not lib and not web
    elif expect == "forbidden_lib":
        tool_ok = not lib
    else:
        tool_ok = True

    fake_cite = bool(_CITE.search(answer or "")) and not lib
    if gold.get("forbid_cite") or expect in {"forbidden", "forbidden_lib"}:
        cite_ok = not fake_cite
    else:
        cite_ok = True

    ok = must_ok and any_ok and forbid_ok and tool_ok and cite_ok
    return {
        "ok": ok,
        "must_ok": must_ok,
        "any_ok": any_ok,
        "forbid_ok": forbid_ok,
        "tool_ok": tool_ok,
        "cite_ok": cite_ok,
        "tools": tools,
        "fake_cite": fake_cite,
    }
