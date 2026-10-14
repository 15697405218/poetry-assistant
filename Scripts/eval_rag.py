"""检索评估：对比 raw / 术语改写 / 改写+消歧+字面+双路+重排。

  python scripts/eval_rag.py --k 5 --skip-llm --no-extra
  python scripts/eval_rag.py --k 5 --skip-llm
  python scripts/eval_rag.py --k 5

指标按 query 宏平均。命中规则：作者/标题任一落在 gold 里，或 gold_keywords 出现在标题+正文。
Chroma 未 ingest 时会直接失败。--skip-llm 不做消歧（省钱，full 变成术语+字面+双路+重排）。
默认会把 dataset/eval_extra.jsonl 拼进主集；--no-extra 只跑 --path 那份（用来对比 42 条旧表）。
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.eval_metrics import f1, gold_recall, precision
from app.services.disambiguate import disambiguate_query
from app.services.rag import _near, search_poems
from app.services.retrieve import _merge, retrieve_poems
from app.services.terms import rewrite_query


def load_eval(path: Path, limit: int, no_extra: bool = False) -> list[dict]:
    rows = []
    paths = [path]
    extra = path.parent / "eval_extra.jsonl"
    if not no_extra and extra.is_file() and extra != path:
        paths.append(extra)
    for src in paths:
        for line in src.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
            if limit and len(rows) >= limit:
                return rows
    return rows


def dual_vector(q: str, original: str, k: int) -> list[dict]:
    queries = []
    for text in (original, q):
        text = (text or "").strip()
        if text and text not in queries:
            queries.append(text)
    groups = [_near(search_poems(query, top_k=max(k, 15))) for query in queries]
    return _merge(groups)[:k]


def run_mode(mode: str, query: str, k: int, skip_llm: bool, no_rerank: bool) -> list[dict]:
    rw = rewrite_query(query)
    original = rw["original"] or query
    terms_q = rw["query"] or original
    if mode == "raw":
        return search_poems(original, top_k=k)
    if mode == "terms":
        return search_poems(terms_q, top_k=k)
    q = terms_q
    if not skip_llm:
        dis = disambiguate_query(terms_q)
        if dis.get("confident") and dis.get("query"):
            q = dis["query"]
    if no_rerank:
        return dual_vector(q, original, k)
    hits, _ = retrieve_poems(q, original=original, top_k=k)
    return hits


def summarize(name: str, scores: list[tuple[float, float, float]]) -> dict:
    n = len(scores) or 1
    p = sum(x[0] for x in scores) / n
    r = sum(x[1] for x in scores) / n
    return {"mode": name, "n": len(scores), "precision@k": p, "recall@k": r, "f1": f1(p, r)}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--path", default="dataset/eval.jsonl")
    p.add_argument("--k", type=int, default=5)
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--skip-llm", action="store_true")
    p.add_argument("--no-extra", action="store_true", help="不拼 eval_extra.jsonl，只跑 --path")
    p.add_argument("--no-rerank", action="store_true", help="full 只做双路向量，避开 CrossEncoder 崩进程")
    p.add_argument("--recall-k", type=int, default=0, help="full 粗召条数，0 表示用配置里的 recall_k")
    args = p.parse_args()
    if args.recall_k > 0:
        from app.config import settings as _settings

        object.__setattr__(_settings, "recall_k", args.recall_k)
        print(f"recall_k={_settings.recall_k}", flush=True)
    rows = load_eval(Path(args.path), args.limit, no_extra=args.no_extra)
    if not rows:
        raise SystemExit("评估集为空")
    modes = ["raw", "terms", "full"]
    buckets = {m: [] for m in modes}
    for i, gold in enumerate(rows, 1):
        q = gold["query"]
        print(f"[{i}/{len(rows)}] {q}", flush=True)
        for mode in modes:
            hits = run_mode(mode, q, args.k, args.skip_llm, args.no_rerank)
            prec = precision(hits, gold, args.k)
            rec = gold_recall(hits, gold)
            buckets[mode].append((prec, rec, f1(prec, rec)))
    print()
    print(f"{'mode':<8} {'P@k':>8} {'R@k':>8} {'F1':>8} {'n':>5}")
    table = [summarize(m, buckets[m]) for m in modes]
    for row in table:
        print(
            f"{row['mode']:<8} {row['precision@k']:8.3f} {row['recall@k']:8.3f} "
            f"{row['f1']:8.3f} {row['n']:5d}"
        )
    raw, full = table[0], table[2]
    if raw["recall@k"]:
        lift = (full["recall@k"] - raw["recall@k"]) / raw["recall@k"] * 100
        print(f"\nfull 相对 raw 的 Recall 变化：{lift:+.1f}%")


if __name__ == "__main__":
    main()
