"""试四层检索，不走完整对话。先开着能连模型的环境，建议已下载重排模型。

  python scripts/test_retrieve.py 雨霖铃的意象
"""
from __future__ import annotations

import sys
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.services.disambiguate import disambiguate_query
from app.services.retrieve import retrieve_poems
from app.services.terms import rewrite_query


def main() -> None:
    text = " ".join(sys.argv[1:]).strip() or "雨霖铃的意象"
    rw = rewrite_query(text)
    terms_q = rw["query"] or text
    original = rw["original"] or text
    print("原句:", original)
    print("术语改写:", terms_q)
    dis = disambiguate_query(terms_q)
    print("消歧 confident:", dis["confident"], "intent:", dis["intent"], "keyword:", dis["keyword"])
    print("消歧问句:", dis["query"])
    query = dis["query"] if dis["confident"] else terms_q
    hits, meta = retrieve_poems(query, original=original, top_k=5)
    print("双路:", meta["dual"], "重排问句:", meta["rank_query"])
    if not hits:
        print("(无结果)")
        return
    for i, h in enumerate(hits, 1):
        score = h.get("rerank")
        dist = h.get("distance")
        extra = []
        if score is not None:
            extra.append(f"rerank={score:.3f}")
        if dist is not None:
            extra.append(f"L2={dist:.1f}")
        print(f"{i}. {h.get('author')}·{h.get('title')} {' '.join(extra)}")
        body = (h.get("text") or "").replace("\n", " ")
        print("  ", body[:80])


if __name__ == "__main__":
    main()
