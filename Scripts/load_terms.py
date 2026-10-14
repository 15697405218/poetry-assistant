import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.services.terms import draft_path, load_terms, lookup, rewrite_query


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("text", nargs="?", default="", help="试查一句，空则只打印统计")
    args = p.parse_args()
    terms = load_terms()
    n_id = len(terms["by_id"])
    n_surf = len(terms["surface_to_ids"])
    n_multi = sum(1 for ids in terms["surface_to_ids"].values() if len(ids) > 1)
    print("草稿", draft_path())
    print("本体", n_id, "表面", n_surf, "一对多表面", n_multi)
    cats: dict[str, int] = {}
    for row in terms["by_id"].values():
        cats[row["category"]] = cats.get(row["category"], 0) + 1
    for cat, n in sorted(cats.items(), key=lambda x: -x[1]):
        print(n, cat)
    if not args.text:
        print("用法: python scripts/load_terms.py 乐游原晚上")
        return
    hits = lookup(args.text, terms)
    rw = rewrite_query(args.text, terms)
    print("原句:", rw["original"])
    print("改写:", rw["query"])
    if not hits:
        print("(未命中)")
        return
    for h in hits:
        aliases = "、".join(h["aliases"][:6]) or "-"
        print(h["matched"], "→", h["category"], h["canonical"], "别名", aliases)


if __name__ == "__main__":
    main()
