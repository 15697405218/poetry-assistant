"""检索前问句消歧。判定不了就原样返回，不要越消越错。"""
from __future__ import annotations

import json
import re

from app.observability import get_logger, stage
from app.services.llm import AuthError, RateLimitError, UpstreamError, llm
from app.services.query_span import strip_question_shell
from app.services.redis_store import cache_get, cache_key, cache_set

log = get_logger("poetry.disambiguate")

_LOOKUP_MARKERS = ("下一句", "出处", "哪首", "谁写的")
_HAN4 = re.compile(r"[\u4e00-\u9fff]{4,}")
_CLAUSE = re.compile(r"^[\u4e00-\u9fff]{4,16}$")
# 剥壳后仍像词牌/意象时走 LLM，不要把「雨霖铃意象」当成诗句。
_KEEP_LLM = ("意象", "词牌", "乐曲", "赏析")


def is_verse_lookup_query(text: str) -> bool:
    """查证诗句：含下一句/出处/哪首/谁写的且剥壳后有≥4 字连续汉字，或核心本身是 4–16 字句。"""
    q = (text or "").strip()
    if not q:
        return False
    core = strip_question_shell(q)
    if any(m in q for m in _LOOKUP_MARKERS) and _HAN4.search(core):
        return True
    if not _CLAUSE.fullmatch(core):
        return False
    return not any(tok in core for tok in _KEEP_LLM)


DISAMBIGUATE_SYSTEM = """你是诗词检索的问句消歧器。只输出一个 JSON 对象，不要 markdown，不要解释。

用户问句里的专名可能有多种互斥义项。经典例子：「雨霖铃」可以是词牌名、乐曲名，或诗句里的意象。

字段：
- ambiguous: 是否存在互斥义项
- keyword: 有歧义的专名，没有则空字符串
- intent: cipai / music / verse / author / create / other / unknown
- query: 给检索用的问句
- confident: 是否能从上下文判定唯一义项

规则：
- 无法确定、没有歧义、创作或闲聊：confident 必须 false，query 必须与输入完全相同。
- 只有能判定唯一义项时，confident 才为 true，并把 query 改写成无歧义检索句（点明词牌/乐曲/诗句/作者等）。
- 已经写成「陆游（放翁）」这种「本体（别称）」的不是歧义，不要再改。
- 不要编造作者、篇名或出处。"""


def _extract_json(text: str) -> dict | None:
    raw = (text or "").strip()
    if not raw:
        return None
    if raw.startswith("```"):
        raw = re.sub(r"^```(?:json)?\s*", "", raw)
        raw = re.sub(r"\s*```$", "", raw)
    start = raw.find("{")
    end = raw.rfind("}")
    if start < 0 or end <= start:
        return None
    try:
        data = json.loads(raw[start : end + 1])
    except json.JSONDecodeError:
        return None
    return data if isinstance(data, dict) else None


def disambiguate_query(text: str) -> dict:
    """返回 {query, confident, ambiguous, keyword, intent}。失败则 confident=false、query=原句。"""
    q = (text or "").strip()
    fallback = {
        "query": q,
        "confident": False,
        "ambiguous": False,
        "keyword": "",
        "intent": "unknown",
    }
    if len(q) < 2:
        return fallback
    ck = f"poetry:dis:{cache_key(q)}"
    cached = cache_get(ck)
    if isinstance(cached, dict) and cached.get("query") is not None:
        log.info("disambiguate cache_hit q=%s", q[:40])
        return cached
    if is_verse_lookup_query(q):
        cache_set(ck, fallback)
        log.info("disambiguate skip_llm verse_lookup q=%s", q[:40])
        return fallback
    try:
        with stage("disambiguate_llm"):
            raw = llm.chat(
                [
                    {"role": "system", "content": DISAMBIGUATE_SYSTEM},
                    {"role": "user", "content": q},
                ],
                temperature=0,
            )
    except (AuthError, RateLimitError, UpstreamError, Exception):
        return fallback
    data = _extract_json(raw) or {}
    intent = str(data.get("intent") or "unknown").strip() or "unknown"
    keyword = str(data.get("keyword") or "").strip()
    query = str(data.get("query") or "").strip() or q
    confident = bool(data.get("confident"))
    if intent in {"unknown", "create"} or not data.get("ambiguous"):
        confident = False
    if not confident:
        query = q
    out = {
        "query": query,
        "confident": confident,
        "ambiguous": bool(data.get("ambiguous")),
        "keyword": keyword,
        "intent": intent,
    }
    cache_set(ck, out)
    log.info("disambiguate confident=%s intent=%s q=%s", out["confident"], out["intent"], out["query"][:40])
    return out
