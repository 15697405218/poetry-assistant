import json
import urllib.error
import urllib.request

from app.config import settings

TAVILY_URL = "https://api.tavily.com/search"


def search_web(q: str, count: int = 5) -> str:
    q = (q or "").strip()
    if not q:
        return "缺少检索词"
    key = (settings.tavily_api_key or "").strip()
    if not key:
        return "未配置 TAVILY_API_KEY，无法联网检索"

    body = json.dumps({
        "query": q,
        "search_depth": "basic",
        "max_results": count,
        "include_answer": False,
    }).encode("utf-8")
    req = urllib.request.Request(
        TAVILY_URL,
        data=body,
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {key}",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
            payload = json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        return f"联网检索失败：HTTP {e.code}"
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as e:
        return f"联网检索失败：{e}"

    rows = payload.get("results") or []
    if not rows:
        return "没有检索到公开网页结果"

    parts = []
    for i, row in enumerate(rows[:count], 1):
        title = (row.get("title") or "").strip() or "无标题"
        url = (row.get("url") or "").strip()
        snippet = (row.get("content") or "").strip()
        if len(snippet) > 200:
            snippet = snippet[:200] + "…"
        parts.append(f"【{i}】{title}\n{url}\n{snippet}")
    return "\n\n".join(parts)
