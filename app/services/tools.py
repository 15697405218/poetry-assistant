import json

from app.services.rag import (
    _format_poems,
    get_author_bio,
    search_poems_literal,
    to_hans,
)
from app.services.retrieve import retrieve_poems
from app.services.web_search import search_web

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "search_poems",
            "description": "检索诗词（消歧+双路向量召回+重排）。已知作者时传入 author，会只搜这个人。",
            "parameters": {
                "type": "object",
                "properties": {
                    "q": {"type": "string", "description": "检索短句：诗句、标题或主题"},
                    "author": {"type": "string", "description": "可选，作者名，简体繁体都行"},
                },
                "required": ["q"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_poems_literal",
            "description": "按正文精确包含检索，适合用户已经说出诗句原文时。",
            "parameters": {
                "type": "object",
                "properties": {
                    "q": {"type": "string", "description": "要在正文里出现的字面片段"},
                },
                "required": ["q"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_author_bio",
            "description": "按作者名取简介。",
            "parameters": {
                "type": "object",
                "properties": {
                    "name": {"type": "string", "description": "作者名"},
                },
                "required": ["name"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_web",
            "description": "公开网页检索。库里没有的当代事实、奖项、通行版本争议、作者近况用这个。不问本库诗词原文。",
            "parameters": {
                "type": "object",
                "properties": {
                    "q": {"type": "string", "description": "检索短句，用中文或专名"},
                },
                "required": ["q"],
            },
        },
    },
]


def run_tool(name: str, arguments: str) -> str:
    args = json.loads(arguments or "{}")
    if name == "search_poems":
        q = to_hans(str(args.get("q") or ""))
        original = to_hans(str(args.get("original") or ""))
        author = (args.get("author") or "").strip()
        hits, meta = retrieve_poems(
            q, original=original or None, author=author or None,
        )
        body = _format_poems(hits)
        if meta.get("dual"):
            return f"双路检索，重排问句：{meta.get('rank_query')}\n\n{body}"
        return body
    if name == "search_poems_literal":
        q = to_hans(str(args.get("q") or ""))
        return _format_poems(search_poems_literal(q, limit=5))
    if name == "get_author_bio":
        bio = get_author_bio(str(args.get("name") or ""))
        return bio or "没有找到该作者简介"
    if name == "search_web":
        return search_web(str(args.get("q") or ""))
    return f"未知工具: {name}"
