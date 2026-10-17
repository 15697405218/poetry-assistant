"""检索前问句消歧。判定不了就原样返回，不要越消越错。"""
from __future__ import annotations

import json
import re

from app.config import settings
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


INTENTS = ("cipai", "music", "verse", "author", "create", "other", "unknown")
# 1 次正式 + 2 次因形状不合格重试；超了原句降级。
PARSE_ATTEMPTS = 3
DISAMBIGUATE_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "ambiguous": {"type": "boolean"},
        "keyword": {"type": "string"},
        "intent": {"type": "string", "enum": list(INTENTS)},
        "query": {"type": "string"},
        "confident": {"type": "boolean"},
    },
    "required": ["ambiguous", "keyword", "intent", "query", "confident"],
}
# Ark 对 json_schema 支持因模型而异；json_object 逼它吐 JSON，形状在本地按上面校验。
RESPONSE_FORMAT = {"type": "json_object"}

DISAMBIGUATE_SYSTEM = """你是诗词检索的问句消歧器。只输出一个 JSON 对象，不要 markdown，不要解释。

用户问句里的专名可能有多种互斥义项。经典例子：「雨霖铃」可以是词牌名、乐曲名，或诗句里的意象。

字段（必须全部出现，类型如下）：
- ambiguous: boolean，是否存在互斥义项
- keyword: string，有歧义的专名，没有则空字符串
- intent: 只能是 cipai / music / verse / author / create / other / unknown
- query: string，给检索用的问句
- confident: boolean，是否能从上下文判定唯一义项

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


def validate_disambiguate(data: object) -> dict | None:
    """形状合格才返回规范化字段；多字段、缺字段、intent 不在枚举都算失败。"""
    if not isinstance(data, dict):
        return None
    extra = set(data) - set(DISAMBIGUATE_SCHEMA["properties"])
    if extra:
        return None
    for key in DISAMBIGUATE_SCHEMA["required"]:
        if key not in data:
            return None
    if not isinstance(data["ambiguous"], bool) or not isinstance(data["confident"], bool):
        return None
    if not isinstance(data["keyword"], str) or not isinstance(data["query"], str):
        return None
    intent = str(data["intent"] or "").strip()
    if intent not in INTENTS:
        return None
    return {
        "ambiguous": data["ambiguous"],
        "keyword": data["keyword"].strip(),
        "intent": intent,
        "query": data["query"].strip(),
        "confident": data["confident"],
    }


def _accepted(data: dict, original: str) -> dict:
    intent = data["intent"]
    query = data["query"] or original
    confident = bool(data["confident"])
    if intent in {"unknown", "create"} or not data["ambiguous"]:
        confident = False
    if not confident:
        query = original
    return {
        "query": query,
        "confident": confident,
        "ambiguous": bool(data["ambiguous"]),
        "keyword": data["keyword"],
        "intent": intent,
    }


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
    messages = [
        {"role": "system", "content": DISAMBIGUATE_SYSTEM},
        {"role": "user", "content": q},
    ]
    for attempt in range(PARSE_ATTEMPTS):
        try:
            with stage("disambiguate_llm", attempt=attempt + 1):
                raw = llm.chat(
                    messages,
                    temperature=settings.disambiguate_temperature,
                    max_tokens=settings.disambiguate_max_tokens,
                    response_format=RESPONSE_FORMAT,
                )
        except (AuthError, RateLimitError):
            return fallback
        except (UpstreamError, Exception):
            if attempt + 1 >= PARSE_ATTEMPTS:
                log.info("disambiguate give_up q=%s", q[:40])
                return fallback
            continue
        data = validate_disambiguate(_extract_json(raw))
        if data is None:
            log.info("disambiguate schema_miss attempt=%s q=%s", attempt + 1, q[:40])
            if attempt + 1 >= PARSE_ATTEMPTS:
                cache_set(ck, fallback)
                return fallback
            continue
        out = _accepted(data, q)
        cache_set(ck, out)
        log.info("disambiguate confident=%s intent=%s q=%s", out["confident"], out["intent"], out["query"][:40])
        return out
    return fallback
