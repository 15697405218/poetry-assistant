"""对话评测：真实口吻问句走 LangGraph（术语+消歧+工具循环），规则打分。

  python scripts/eval_chat.py
  python scripts/eval_chat.py --limit 5
  python scripts/eval_chat.py --start 6

每条问完立刻追加到 --out 并 flush；已有文件按 id 跳过再追加。
跑完才写 --report（默认 dataset/chat_eval_report.md，已 gitignore）。中途杀掉不会丢已完成的答案。

不走 HTTP / worker，避免排队。不写用户会话表。会调大模型和检索，耗 token。
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("OMP_NUM_THREADS", "1")

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from langgraph.errors import GraphRecursionError

from app.chat_eval import score_case
from app.services.chat_run import iter_graph_events
from app.services.disambiguate import disambiguate_query
from app.services import graph as graph_mod
from app.services.terms import rewrite_query

CONV_BASE = 910000


def load_rows(path: Path, limit: int) -> list[dict]:
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        rows.append(json.loads(line))
        if limit and len(rows) >= limit:
            break
    return rows


def prepare_query(text: str) -> tuple[str, str]:
    rw = rewrite_query(text)
    terms_q = rw["query"] or text
    original = rw["original"] or text
    dis = disambiguate_query(terms_q)
    if dis.get("confident") and str(dis.get("query") or "").strip():
        return str(dis["query"]).strip(), original
    return terms_q, original


def run_one(i: int, gold: dict, skip_disambiguate: bool) -> dict:
    tools: list[str] = []
    real = graph_mod.run_tool

    def tap(name: str, args: str) -> str:
        tools.append(name)
        return real(name, args)

    graph_mod.run_tool = tap
    query = gold["query"]
    if skip_disambiguate:
        rw = rewrite_query(query)
        rewrite, original = rw["query"] or query, rw["original"] or query
    else:
        rewrite, original = prepare_query(query)
    parts: list[str] = []
    err = None
    cid = CONV_BASE + i + int(time.time())
    try:
        for ev in iter_graph_events(
            cid,
            rewrite,
            original,
            load_history_fn=lambda _cid, q=query: [{"role": "user", "content": q}],
            save_assistant_fn=lambda *_a: None,
        ):
            if ev.get("type") == "token":
                parts.append(ev.get("data") or "")
            elif ev.get("type") == "error":
                err = ev.get("data") or "error"
    except GraphRecursionError as e:
        err = f"工具循环超限: {e}"
    except Exception as e:
        err = f"{type(e).__name__}: {e}"
    finally:
        graph_mod.run_tool = real
    answer = "".join(parts)
    scored = score_case(gold, answer, tools)
    scored.update({
        "id": gold.get("id"),
        "kind": gold.get("kind"),
        "query": query,
        "rewrite": rewrite,
        "answer": answer,
        "error": err,
    })
    if err:
        scored["ok"] = False
    return scored


def load_done(path: Path) -> dict[str, dict]:
    done: dict[str, dict] = {}
    if not path.is_file():
        return done
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        row = json.loads(line)
        if row.get("id"):
            done[row["id"]] = row
    return done


def append_result(path: Path, row: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(row, ensure_ascii=False) + "\n")
        fh.flush()
        os.fsync(fh.fileno())


def write_report(path: Path, results: list[dict]) -> None:
    n = len(results)
    n_ok = sum(1 for r in results if r.get("ok"))
    by_kind: dict[str, list[bool]] = {}
    for r in results:
        by_kind.setdefault(r.get("kind") or "?", []).append(bool(r.get("ok")))
    lines = ["# 对话评测报告", ""]
    if n:
        lines.append(f"pass {n_ok}/{n} = {n_ok / n:.3f}")
    else:
        lines.append("pass 0/0")
    lines.append("")
    if by_kind:
        lines.extend(["| kind | pass |", "|---|---|"])
        for kind, xs in by_kind.items():
            lines.append(f"| {kind} | {sum(xs)}/{len(xs)} |")
        lines.append("")
    for r in results:
        flag = "PASS" if r.get("ok") else "FAIL"
        rid = r.get("id") or "?"
        kind = r.get("kind") or "?"
        trunc = "（截断预览）" if r.get("truncated") else ""
        lines.append(f"## {rid} {flag} {kind}{trunc}")
        lines.append("")
        lines.append(f"**问：** {r.get('query') or ''}")
        if r.get("rewrite") and r.get("rewrite") != r.get("query"):
            lines.append("")
            lines.append(f"**改写：** {r['rewrite']}")
        lines.append("")
        lines.append(f"**工具：** `{r.get('tools')}`")
        lines.append("")
        if r.get("error"):
            lines.append(f"**错误：** {r['error']}")
            lines.append("")
        lines.append("**答：**")
        lines.append("")
        lines.append((r.get("answer") or "（无）").rstrip() or "（无）")
        lines.append("")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--path", default="dataset/chat_eval.jsonl")
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--start", type=int, default=0, help="跳过前 N 条，接着跑")
    p.add_argument("--skip-disambiguate", action="store_true")
    p.add_argument("--out", default="dataset/chat_eval_last.jsonl")
    p.add_argument("--report", default="dataset/chat_eval_report.md")
    args = p.parse_args()
    rows = load_rows(Path(args.path), 0)
    if args.start:
        rows = rows[args.start:]
    if args.limit:
        rows = rows[: args.limit]
    if not rows:
        raise SystemExit("对话评估集为空")
    print("warmup retrieve", flush=True)
    from app.services.retrieve import retrieve_poems

    retrieve_poems("春眠不觉晓", original="春眠不觉晓", top_k=3)
    out_path = Path(args.out)
    done = load_done(out_path)
    n_ok = 0
    results = []
    for i, gold in enumerate(rows, 1):
        gid = gold.get("id") or str(i)
        if gid in done:
            row = done[gid]
            results.append(row)
            if row.get("ok"):
                n_ok += 1
            print(f"[{i}/{len(rows)}] skip {gold['query']}", flush=True)
            continue
        print(f"[{i}/{len(rows)}] {gold['query']}", flush=True)
        row = run_one(i, gold, args.skip_disambiguate)
        results.append(row)
        if row.get("ok"):
            n_ok += 1
        flag = "PASS" if row.get("ok") else "FAIL"
        preview = (row.get("answer") or row.get("error") or "").replace("\n", " ")[:80]
        print(
            f"  {flag} tools={row.get('tools')} kind={gold.get('kind')} | {preview}",
            flush=True,
        )
        append_result(out_path, row)
    by_kind: dict[str, list[bool]] = {}
    for r in results:
        by_kind.setdefault(r.get("kind") or "?", []).append(bool(r.get("ok")))
    print()
    print(f"pass {n_ok}/{len(results)} = {n_ok / len(results):.3f}")
    for kind, xs in by_kind.items():
        print(f"  {kind}: {sum(xs)}/{len(xs)}")
    report_path = Path(args.report)
    write_report(report_path, results)
    print(f"answers -> {out_path}")
    print(f"report -> {report_path}")


if __name__ == "__main__":
    main()
