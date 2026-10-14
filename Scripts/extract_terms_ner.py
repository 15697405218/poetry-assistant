"""离线从诗集抽术语草稿（GLiNER）。线上 chat 只读 data/terms.draft.json，不加载本脚本。"""
import argparse
from zhconv import convert
import json
import os
import sys
import faulthandler
from pathlib import Path
from collections import defaultdict
import re

faulthandler.enable()

# 古诗上 gliner-multi 误标多，改用 knowledgator/gliner-x-small

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
SKIP_SURFACES = {"未知", "无名氏", "句","无名氏《宦门子弟错立身》","无名氏《张协状元》"}
SKIP_ALIASES = {"里不详", "不详", "未详", "不详。", "未详。"}

from app.config import settings

# 清洗脚本要下载 GLiNER；config/.env 里的离线开关只给线上 chat 用
os.environ["HF_HUB_OFFLINE"] = "0"
os.environ["TRANSFORMERS_OFFLINE"] = "0"
# 国内 Xet/CAS（us.aws.cdn.hf.co）常失败，改走普通 HTTPS
os.environ["HF_HUB_DISABLE_XET"] = "1"

SKIP_PARTS = {"rank", "strains", "error", "御定全唐詩"}

def to_hans(text:str) -> str: return convert(text or "", "zh-hans")

def is_skip(path:Path) -> bool:
    if "author" in path.name.lower():
        return True
    return any(part in SKIP_PARTS for part in path.parts)

def item_body(item:dict) -> str:
    raw = item.get("paragraphs") or item.get("text") or item.get("content") or ""
    if isinstance(raw,list):
        return "".join(str(x) for x in raw)
    return str(raw)
def item_title(item: dict) -> str:
    return item.get("title") or item.get("rhythmic") or item.get("chapter") or "未知"

def iter_poems(
    limit: int | None = None,
    skip_notes: bool = False,
    path_contains: str | None = None,
    offset: int = 0,
):
    root = Path(settings.poetry_data_dir)
    n = 0
    yielded = 0
    for file in root.rglob("*.json"):
        if is_skip(file):
            continue
        if path_contains and path_contains not in str(file):
            continue
        try:
            data = json.loads(file.read_text(encoding="utf-8"))
        except Exception as e:
            print(f"Error loading {file}: {e}")
            continue
        if not isinstance(data,list):
            continue
        dynasty = to_hans(file.parent.name)
        for item in data:
            if not isinstance(item,dict):
                continue
            body = to_hans(item_body(item))
            if len(body.strip()) < 2:
                continue
            notes = item.get("notes") or []
            if isinstance(notes, str):
                notes = [notes]
            elif not isinstance(notes, list):
                notes = []
            if skip_notes and notes:
                continue
            n += 1
            if n <= offset:
                continue
            yield {
                "author": to_hans(str(item.get("author") or "未知")),
                "title": to_hans(str(item_title(item))),
                "rhythmic": to_hans(str(item.get("rhythmic") or "")),
                "body": body,
                "notes": notes,
                "dynasty": dynasty,
                "source_path": str(file),
            }
            yielded += 1
            if limit is not None and yielded >= limit:
                return
def harvest_fields(poem: dict) -> list[dict]:
    events = []
    author = poem["author"]
    if author and author not in SKIP_SURFACES:
        events.append({
            "surface": author,
            "category_hint": "诗人",
            "source": "field:author",
            "path": poem["source_path"],
        })
    rhythmic = poem["rhythmic"]
    title = poem["title"]
    if rhythmic:
        events.append({
            "surface": rhythmic,
            "category_hint": "词牌",
            "source": "field:rhythmic",
            "path": poem["source_path"],
        })
    elif title and title  not in SKIP_SURFACES and title != rhythmic:
        events.append({
            "surface": title,
            "category_hint": "标题",
            "source": "field:title",
            "path": poem["source_path"],
        })
    return events


NOTE_SPLIT = re.compile(r"--|—|－")
NOTE_HEAD = re.compile(r"^\d+[\.、．]\s*")


def parse_note_line(line: str) -> tuple[str, str] | None:
    s = to_hans(str(line or "")).strip()
    s = NOTE_HEAD.sub("", s)
    if not s:
        return None
    parts = NOTE_SPLIT.split(s, maxsplit=1)
    if len(parts) == 2:
        term, defn = parts[0].strip(), parts[1].strip()
    else:
        parts = re.split(r"[：:]", s, maxsplit=1)
        if len(parts) != 2:
            return None
        term, defn = parts[0].strip(), parts[1].strip()
    if term in SKIP_SURFACES or len(term) < 1 or len(term) > 8:
        return None
    if any(ch in term for ch in "，。；"):
        return None
    return term, defn[:80]


def harvest_notes(poem: dict) -> list[dict]:
    events = []
    for line in poem.get("notes") or []:
        parsed = parse_note_line(line)
        if not parsed:
            continue
        term, defn = parsed
        events.append({
            "surface": term,
            "category_hint": "注解",
            "source": "field:notes",
            "path": poem["source_path"],
            "definition": defn,
        })
    return events


def collect_notes(limit: int | None = None) -> tuple[list[dict], int]:
    buckets: dict[str, dict] = {}
    with_notes = 0
    for poem in iter_poems(limit=limit):
        evs = harvest_notes(poem)
        if evs:
            with_notes += 1
        for ev in evs:
            key = ev["surface"]
            if key not in buckets:
                buckets[key] = {
                    "canonical": key,
                    "aliases": [],
                    "category": "注解",
                    "count": 0,
                    "definition": ev.get("definition") or "",
                    "source": "notes",
                }
            buckets[key]["count"] += 1
            if not buckets[key]["definition"] and ev.get("definition"):
                buckets[key]["definition"] = ev["definition"]
    rows = list(buckets.values())
    rows.sort(key=lambda r: (-r["count"], r["canonical"]))
    return rows, with_notes


def patch_draft_with_notes(draft: list[dict], notes: list[dict]) -> tuple[list[dict], int]:
    by_name = {r["canonical"]: r for r in draft}
    added = 0
    for n in notes:
        name = n["canonical"]
        if name in by_name:
            row = by_name[name]
            if n.get("definition") and not row.get("definition"):
                row["definition"] = n["definition"]
            src = row.get("source") or ""
            if "notes" not in src:
                row["source"] = src + "+notes" if src else "notes"
            continue
        by_name[name] = n
        added += 1
    out = list(by_name.values())
    out.sort(key=lambda r: (-int(r.get("count") or 0), r.get("category") or "", r["canonical"]))
    return out, added


def patch_draft_with_review(draft: list[dict], reviewed: list[dict]) -> tuple[list[dict], int]:
    by_name = {r["canonical"]: r for r in draft}
    added = 0
    for r in reviewed:
        if not r.get("keep"):
            continue
        name = to_hans(str(r.get("canonical") or r.get("surface") or "")).strip()
        if not name or name in SKIP_SURFACES:
            continue
        surf = to_hans(str(r.get("surface") or "")).strip()
        cat = str(r.get("category") or "").strip() or "未知"
        if name in by_name:
            row = by_name[name]
            src = row.get("source") or ""
            if "ner" not in src:
                row["source"] = src + "+ner+llm" if src else "ner+llm"
            aliases = list(row.get("aliases") or [])
            if surf and surf != name and surf not in aliases and surf not in by_name:
                aliases.append(surf)
            row["aliases"] = aliases
            row["count"] = int(row.get("count") or 0) + int(r.get("count") or 1)
            continue
        by_name[name] = {
            "canonical": name,
            "aliases": [surf] if surf and surf != name else [],
            "category": cat,
            "count": int(r.get("count") or 1),
            "source": "ner+llm",
        }
        added += 1
    out = list(by_name.values())
    out.sort(key=lambda r: (-int(r.get("count") or 0), r.get("category") or "", r["canonical"]))
    return out, added


# 词牌走字段。意象/器物不交给 GLiNER：整句乱标，细筛也会丢。
# 试过：multi-v2.1；x-small 六类再收到 地名/人名。真地名能出，官职宫殿被丢。
# 现标签补回专名：地名、人名、官职、宫殿、典故。扩标签后必须 --ner-reset --review-reset。
# 细筛留和丢都进 terms.ner.review.json；--merge-ner 只把 keep 的并进草稿。
# 批量：--ner --review --batch 20。--until-done 循环到没诗；--all-paths 扫整个诗词库。
# 古文编码器微调暂不做。
NER_MODEL = "knowledgator/gliner-x-small"
NER_LABEL_MAP = {
    "地名": "地名",
    "人名": "人物专名",
    "官职": "官职",
    "宫殿": "宫殿",
    "典故": "典故",
}
REVIEW_KEEP_CATS = {"地名", "人物专名", "官职", "宫殿", "典故"}
NER_LABELS = list(NER_LABEL_MAP)
NER_BAD_CHARS = set("，。；、,.!！？?[]【】()（）")
_ner = None
_ner_id = None


def get_ner(model_id: str = NER_MODEL):
    global _ner, _ner_id
    if _ner is not None and _ner_id == model_id:
        return _ner
    os.environ["HF_HUB_OFFLINE"] = "0"
    os.environ["TRANSFORMERS_OFFLINE"] = "0"
    os.environ["HF_HUB_DISABLE_XET"] = "1"
    print(f"正在加载 GLiNER: {model_id}", flush=True)
    from gliner import GLiNER
    print("开始 from_pretrained（装权重/Stanza，崩了也不会有普通 traceback）", flush=True)
    try:
        _ner = GLiNER.from_pretrained(model_id)
    except Exception as e:
        print("加载失败:", type(e).__name__, e, flush=True)
        raise
    _ner_id = model_id
    print("模型已加载", flush=True)
    return _ner


def extract_ner(
    text: str,
    threshold: float = 0.05,
    model_id: str = NER_MODEL,
) -> list[dict]:
    if not (text or "").strip():
        return []
    raw_hits = get_ner(model_id).predict_entities(
        text[:400], NER_LABELS, threshold=threshold,
    )
    print("  raw:", [
        (
            h.get("text"),
            h.get("label"),
            round(float(h.get("score") or 0), 2),
        )
        for h in raw_hits
    ])
    out = []
    for h in raw_hits:
        surface = to_hans(str(h.get("text") or "")).strip()
        mapped = NER_LABEL_MAP.get(str(h.get("label") or "").strip(), "")
        if not mapped:
            continue
        if len(surface) < 2 or len(surface) > 4:
            continue
        if any(ch in NER_BAD_CHARS for ch in surface):
            continue
        out.append({
            "surface": surface,
            "category_hint": mapped,
            "score": round(float(h.get("score") or 0), 3),
        })
    return out


def hit_line(body: str, surface: str) -> str:
    for part in re.split(r"[，。；、！？\n]", body or ""):
        if surface and surface in part:
            return part
    return (body or "")[:40]


def harvest_ner(
    poem: dict,
    threshold: float = 0.05,
    model_id: str = NER_MODEL,
) -> list[dict]:
    events = []
    for hit in extract_ner(poem["body"], threshold=threshold, model_id=model_id):
        events.append({
            "surface": hit["surface"],
            "category_hint": hit["category_hint"],
            "score": hit.get("score") or 0,
            "source": "body:ner",
            "path": poem["source_path"],
            "author": poem["author"],
            "title": poem["title"],
            "line": hit_line(poem["body"], hit["surface"]),
        })
    return events


def uniq_review_items(events: list[dict]) -> list[dict]:
    by: dict[str, dict] = {}
    for ev in events:
        key = ev["surface"]
        if key not in by:
            by[key] = {**ev, "count": 1}
        else:
            by[key]["count"] += 1
    return list(by.values())


def parse_json_list(text: str) -> list:
    s = (text or "").strip()
    if s.startswith("```"):
        s = re.sub(r"^```(?:json)?\s*", "", s)
        s = re.sub(r"\s*```$", "", s)
    start, end = s.find("["), s.rfind("]")
    if start < 0 or end < 0:
        return []
    data = json.loads(s[start : end + 1])
    return data if isinstance(data, list) else []


def review_client():
    # 离线清洗专用，不复用 app.services.llm
    from openai import OpenAI
    return OpenAI(
        api_key=settings.ark_api_key,
        base_url=settings.base_url,
        timeout=180,
        max_retries=0,
    )


REVIEW_SYS = """你是古诗词术语审核员。只审核给定候选，禁止从诗句里新抽词。
对每个候选输出 JSON 数组，元素字段：
surface, keep, category, canonical, reason
规则：
- keep 为 true 或 false；不确定则 false
- 专名才 keep：地名（含古国）、人名、官职/官署、宫殿殿名、典故专名
- 意象、器物泛称、景色、抽象词、残缺片段一律 false
- category 仅在 keep=true 时填 地名、人名、官职、宫殿、典故
- canonical 用通行简体专名，一般与 surface 相同
- reason 不超过 12 字
- 只输出 JSON 数组，不要其它文字"""


def _judge_row(x: dict, r: dict) -> dict:
    keep = bool(r.get("keep"))
    cat = str(r.get("category") or "").strip()
    if cat == "人名":
        cat = "诗人别称"
    if keep and cat not in REVIEW_KEEP_CATS:
        keep = False
    return {
        "surface": x["surface"],
        "canonical": to_hans(str(r.get("canonical") or x["surface"])).strip() or x["surface"],
        "category": cat if keep else "",
        "keep": keep,
        "reason": str(r.get("reason") or "")[:20],
        "count": int(x.get("count") or 1),
        "score": x.get("score") or 0,
        "line": x.get("line") or "",
        "path": x.get("path") or "",
        "source": "ner+llm",
    }


def review_candidates(
    events: list[dict],
    batch_size: int = 10,
    save_path: Path | None = None,
) -> list[dict]:
    from openai import (
        APIConnectionError,
        APITimeoutError,
        RateLimitError as SDKRateLimitError,
    )
    import time
    items = uniq_review_items(events)
    done = {r["surface"]: r for r in load_json(save_path) if r.get("surface")} if save_path else {}
    out = [done[x["surface"]] for x in items if x["surface"] in done]
    pending = [x for x in items if x["surface"] not in done]
    if done:
        print("已有细筛", len(done), "跳过，剩余", len(pending), flush=True)
    client = review_client()
    for i in range(0, len(pending), batch_size):
        chunk = pending[i : i + batch_size]
        payload = [
            {
                "surface": x["surface"],
                "hint": x.get("category_hint") or "",
                "score": x.get("score") or 0,
                "line": x.get("line") or "",
                "title": x.get("title") or "",
                "author": x.get("author") or "",
            }
            for x in chunk
        ]
        print(f"细筛 {len(out) + 1}-{len(out) + len(chunk)} / {len(items)}", flush=True)
        text = ""
        try:
            for attempt in range(3):
                try:
                    resp = client.chat.completions.create(
                        model=settings.chat_model,
                        messages=[
                            {"role": "system", "content": REVIEW_SYS},
                            {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
                        ],
                    )
                    text = (resp.choices[0].message.content or "").strip()
                    break
                except (APITimeoutError, APIConnectionError) as e:
                    if attempt == 2:
                        print("网络/超时，已保存进度，稍后重跑 --review", e, flush=True)
                        if save_path:
                            save_raw(out, save_path)
                        return out
                    print(f"连接失败，重试 {attempt + 2}/3", flush=True)
                    time.sleep(3)
        except SDKRateLimitError as e:
            print("额度/限流，已保存进度，稍后重跑 --review", e, flush=True)
            if save_path:
                save_raw(out, save_path)
            return out
        judged = {str(r.get("surface") or ""): r for r in parse_json_list(text)}
        for x in chunk:
            row = _judge_row(x, judged.get(x["surface"]) or {})
            out.append(row)
            print("留" if row["keep"] else "丢", row["surface"], row["category"] or "-", row["reason"], flush=True)
        if save_path:
            save_raw(out, save_path)
    return out


def aggregate(events: list[dict]) -> list[dict]:
    buckets: dict[str, dict] = {}
    for ev in events:
        surface = ev["surface"]
        if surface not in buckets:
            buckets[surface] = {
                "surface": surface,
                "count": 0,
                "category_hints": defaultdict(int),
                "examples": [],
            }
        row = buckets[surface]
        row["count"] += 1
        row["category_hints"][ev["category_hint"]] += 1
        if len(row["examples"]) < 3:
            row["examples"].append(f"{ev['source']} @ {ev['path']}")
    out = []
    for row in buckets.values():
        out.append({
            "surface": row["surface"],
            "count": row["count"],
            "category_hints": dict(row["category_hints"]),
            "examples": row["examples"],
        })
    out.sort(key=lambda x: x["count"], reverse=True)
    return out
def save_raw(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(rows, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def load_json(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return json.loads(path.read_text(encoding="utf-8"))


def primary_category(hints: dict) -> str:
    if not hints:
        return "未知"
    cat = max(hints, key=hints.get)
    if cat == "标题":
        return "篇名"
    return cat


def merge_draft(raw: list[dict], authors: list[dict]) -> list[dict]:
    by_name: dict[str, dict] = {}
    for a in authors:
        name = a["canonical"]
        by_name[name] = {
            "canonical": name,
            "aliases": list(a.get("aliases") or []),
            "category": "诗人",
            "count": 0,
            "source": "author_file",
        }
    canon_names = set(by_name)

    for row in raw:
        surface = row["surface"]
        cat = primary_category(row.get("category_hints") or {})
        count = int(row.get("count") or 0)
        if cat == "诗人":
            if surface in by_name:
                by_name[surface]["count"] += count
                src = by_name[surface]["source"]
                if "raw" not in src:
                    by_name[surface]["source"] = src + "+raw"
            else:
                by_name[surface] = {
                    "canonical": surface,
                    "aliases": [],
                    "category": "诗人",
                    "count": count,
                    "source": "raw",
                }
                canon_names.add(surface)
            continue
        if surface in by_name:
            continue
        by_name[surface] = {
            "canonical": surface,
            "aliases": [],
            "category": cat,
            "count": count,
            "source": "raw",
        }

    for item in by_name.values():
        item["aliases"] = [
            x for x in item["aliases"]
            if x not in canon_names and x != item["canonical"]
        ]
    out = list(by_name.values())
    out.sort(key=lambda r: (-r["count"], r["category"], r["canonical"]))
    return out


BIO_PATTERNS = [
    ("初名", re.compile(r"初名([^，。；、\s]{1,8})")),
    ("字", re.compile(r"字([^，。；、\s]{1,8})")),
    ("号", re.compile(r"(?<![庙谥年尊])号([^，。；\s]{2,20})")),
]


def split_hao(text: str) -> list[str]:
    return [p for p in re.split(r"[、及与/]", text) if p]


def clean_alias(s: str) -> str | None:
    s = to_hans(s).strip().strip("等")
    if len(s) < 2 or len(s) > 8:
        return None
    if s in SKIP_ALIASES:
        return None
    if any(ch in s for ch in "《》()（）卷"):
        return None
    if s.endswith("人") and ("今" in s or "州" in s):
        return None
    if "自注" in s or "所居" in s:
        return None
    return s


def harvest_author(a: dict) -> list[dict]:
    events = []
    name = a["name"]
    events.append({
        "surface": name,
        "category_hint": "诗人",
        "source": "author:name",
        "path": a["source_path"],
    })
    bio = a["bio"] or ""
    for kind, pat in BIO_PATTERNS:
        m = pat.search(bio)
        if not m:
            continue
        pieces = split_hao(m.group(1)) if kind == "号" else [m.group(1)]
        for piece in pieces:
            surface = clean_alias(piece)
            if not surface or surface == name:
                continue
            events.append({
                "surface": surface,
                "category_hint": "诗人别称",
                "source": f"author:bio:{kind}",
                "path": a["source_path"],
            })
    return events


def author_entry(a: dict) -> dict | None:
    evs = harvest_author(a)
    names = [e["surface"] for e in evs if e["category_hint"] == "诗人"]
    if not names:
        return None
    aliases = []
    seen: set[str] = set()
    for e in evs:
        if e["category_hint"] != "诗人别称":
            continue
        if e["surface"] in seen:
            continue
        seen.add(e["surface"])
        aliases.append(e["surface"])
    return {
        "canonical": names[0],
        "aliases": aliases,
        "category": "诗人",
        "source": "author_file",
        "path": a["source_path"],
    }


def merge_author_entries(rows: list[dict]) -> list[dict]:
    by_name: dict[str, dict] = {}
    for row in rows:
        key = row["canonical"]
        if key not in by_name:
            by_name[key] = {
                "canonical": key,
                "aliases": list(row["aliases"]),
                "category": "诗人",
                "source": row["source"],
                "path": row["path"],
            }
            continue
        seen = set(by_name[key]["aliases"])
        for alias in row["aliases"]:
            if alias not in seen:
                seen.add(alias)
                by_name[key]["aliases"].append(alias)
    out = list(by_name.values())
    out.sort(key=lambda r: (-len(r["aliases"]), r["canonical"]))
    return out
def iter_authors(limit: int | None = None):
    root = Path(settings.poetry_data_dir)
    n = 0
    files = [
        p for p in root.rglob("*.json")
        if "author" in p.name.lower() and "error" not in p.parts
    ]
    for file in files:
        try:
            data = json.loads(file.read_text(encoding="utf-8"))
        except Exception as e:
            print("读失败", file, e)
            continue
        if not isinstance(data, list):
            continue
        for item in data:
            if not isinstance(item, dict):
                continue
            name = to_hans(str(item.get("name") or "").strip())
            bio = to_hans(str(
                item.get("desc")
                or item.get("description")
                or item.get("short_description")
                or ""
            ).strip())
            if not name or len(name) < 2 or name in SKIP_SURFACES or name.startswith("无名氏"):
                continue
            yield {
                "name": name,
                "bio": bio,
                "source_path": str(file),
            }
            n += 1
            if limit is not None and n >= limit:
                return


def cand_key(ev: dict) -> tuple:
    return (ev.get("surface") or "", ev.get("path") or "", ev.get("line") or "")


def merge_candidates(old: list[dict], new: list[dict]) -> list[dict]:
    seen = {cand_key(e) for e in old}
    out = list(old)
    for e in new:
        k = cand_key(e)
        if k in seen:
            continue
        seen.add(k)
        out.append(e)
    return out


def load_cursor(path: Path) -> dict:
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return data if isinstance(data, dict) else {}


def ner_scope(args) -> tuple[str | None, str]:
    if args.all_paths:
        return None, "*"
    return args.ner_path or None, args.ner_path or "*"


def take_poems(gen, n: int) -> list[dict]:
    out = []
    for poem in gen:
        out.append(poem)
        if len(out) >= n:
            break
    return out


def harvest_poems(poems: list[dict], args, offset: int) -> list[dict]:
    events = []
    for i, poem in enumerate(poems, 1):
        evs = harvest_ner(
            poem,
            threshold=args.ner_threshold,
            model_id=args.ner_model,
        )
        events.extend(evs)
        print(f"--- {offset + i} {poem['author']} · {poem['title']} ---")
        print("文件:", poem["source_path"])
        print("正文:", poem["body"][:40], "...")
        if not evs:
            print("(无实体)")
        for ev in evs:
            print(ev["category_hint"], ev["surface"])
    return events


def persist_ner_batch(root: Path, events: list[dict], args, offset: int, got: int, scope: str) -> list[dict]:
    cand_path = root / "terms.ner.candidates.json"
    all_cands = merge_candidates(load_json(cand_path), events)
    save_raw(all_cands, cand_path)
    save_raw({
        "path_contains": scope,
        "threshold": args.ner_threshold,
        "poems_done": offset + got,
        "candidates": len(all_cands),
        "this_batch": got,
        "this_events": len(events),
    }, root / "terms.ner.cursor.json")
    print(
        "本批诗", got, "事件", len(events),
        "累计诗", offset + got, "累计粗筛", len(all_cands),
        "已写入", cand_path, "未进草稿",
        flush=True,
    )
    return all_cands


def prepare_ner(args, root: Path) -> tuple[int, str | None, str]:
    path_contains, scope = ner_scope(args)
    cursor_path = root / "terms.ner.cursor.json"
    cand_path = root / "terms.ner.candidates.json"
    if args.ner_reset:
        cand_path.write_text("[]", encoding="utf-8")
        if cursor_path.exists():
            cursor_path.unlink()
        print("NER 游标和粗筛已重置，细筛记录另用 --review-reset", flush=True)
        return 0, path_contains, scope
    cursor = load_cursor(cursor_path)
    if cursor.get("path_contains") and cursor.get("path_contains") != scope:
        print("范围变了，游标从 0 计，旧候选/细筛仍在", flush=True)
        return 0, path_contains, scope
    return int(cursor.get("poems_done") or 0), path_contains, scope


def run_review(events: list[dict], root: Path, batch_size: int) -> list[dict]:
    out = root / "terms.ner.review.json"
    reviewed = review_candidates(events, batch_size=batch_size, save_path=out)
    kept = [r for r in reviewed if r.get("keep")]
    dropped = [r for r in reviewed if not r.get("keep")]
    print(
        "细筛留", len(kept), "丢", len(dropped),
        "已写入", out, "（含丢弃，重跑会跳过）未进草稿",
        flush=True,
    )
    for r in kept[:15]:
        print("留", r.get("category"), r.get("canonical") or r.get("surface"), r.get("reason") or "")
    return reviewed


def main() -> None:
    # n = sum(1 for _ in iter_poems())
    # print("诗词总数:", n)
    p = argparse.ArgumentParser(description="离线术语挖掘。线上检索不调用本脚本。")
    p.add_argument("--authors", action="store_true")
    p.add_argument("--merge", action="store_true")
    p.add_argument("--ner", action="store_true")
    p.add_argument("--notes", action="store_true")
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--ner-threshold", type=float, default=0.05)
    p.add_argument("--ner-path", default="poet.tang")
    p.add_argument("--ner-model", default=NER_MODEL)
    p.add_argument("--review", action="store_true")
    p.add_argument("--review-batch", type=int, default=10)
    p.add_argument("--batch", type=int, default=20, help="本轮 NER 诗数，接上次游标")
    p.add_argument("--ner-reset", action="store_true", help="NER 游标归零，不删细筛")
    p.add_argument("--review-reset", action="store_true", help="清空细筛记录后重审")
    p.add_argument("--merge-ner", action="store_true", help="细筛 keep 的项并进草稿")
    p.add_argument("--until-done", action="store_true", help="按 batch 循环直到诗词库扫完")
    p.add_argument("--all-paths", action="store_true", help="不限 poet.tang，扫整个诗词库")
    p.add_argument("--max-poems", type=int, default=None, help="本次最多再扫多少首，到了就停")
    # p.add_argument("--limit", type=int, default=345634)
    args = p.parse_args()
    # events = []
    # for poem in iter_poems(limit=args.limit):
    #     events.extend(harvest_fields(poem))
    # raw = aggregate(events)
    # raw = [row for row in raw if row["count"] >= 3]
    
    # for row in raw:
    #     print(row["count"], row["surface"], row["category_hints"])
    # out = Path(__file__).resolve().parents[1] / "data" / "terms.raw.json"
    # save_raw(raw, out)
    # print("已写入", out)
    # print("事件", len(events), "留存", len(raw))

# ...
    if args.authors:
        rows = []
        for a in iter_authors():
            entry = author_entry(a)
            if entry:
                rows.append(entry)
        rows = merge_author_entries(rows)
        out = Path(__file__).resolve().parents[1] / "data" / "terms.authors.json"
        save_raw(rows, out)
        
        for row in rows[:20]:
            print(row["canonical"], row["aliases"] or "(无别称)")
        print("作者人数", len(rows), "已写入", out)
        return
    if args.merge:
        root = Path(__file__).resolve().parents[1] / "data"
        raw = load_json(root / "terms.raw.json")
        authors = load_json(root / "terms.authors.json")
        draft = merge_draft(raw, authors)
        out = root / "terms.draft.json"
        save_raw(draft, out)
        print("草稿", len(draft), "已写入", out)
        for row in draft[:15]:
            print(row["count"], row["category"], row["canonical"], row["aliases"][:4])
        print("草稿", len(draft), "已写入", out)
        return
    if args.merge_ner:
        root = Path(__file__).resolve().parents[1] / "data"
        draft = load_json(root / "terms.draft.json")
        reviewed = load_json(root / "terms.ner.review.json")
        kept = [r for r in reviewed if r.get("keep")]
        if not kept:
            print("没有细筛通过的词，先跑 --ner --review")
            return
        draft, added = patch_draft_with_review(draft, reviewed)
        out = root / "terms.draft.json"
        save_raw(draft, out)
        print("细筛通过", len(kept), "新补进草稿", added, "草稿现", len(draft), "已写入", out)
        for row in kept[:15]:
            print(row.get("category"), row.get("canonical") or row.get("surface"), row.get("reason") or "")
        return
    if args.notes:
        root = Path(__file__).resolve().parents[1] / "data"
        rows, with_notes = collect_notes(limit=args.limit)
        save_raw(rows, root / "terms.notes.json")
        draft = load_json(root / "terms.draft.json")
        draft, added = patch_draft_with_notes(draft, rows)
        save_raw(draft, root / "terms.draft.json")
        print("有注解的诗", with_notes, "注解词条", len(rows), "新补进草稿", added)
        print("草稿现", len(draft))
        for row in rows[:15]:
            print(row["count"], row["canonical"], (row.get("definition") or "")[:30])
        print("有注解的诗", with_notes, "注解词条", len(rows), "新补进草稿", added)
        print("草稿现", len(draft))
        return
    if args.ner or args.review:
        root = Path(__file__).resolve().parents[1] / "data"
        if args.review_reset:
            rp = root / "terms.ner.review.json"
            if rp.exists():
                rp.unlink()
            print("已清空细筛记录", flush=True)
        events = load_json(root / "terms.ner.candidates.json")
        if args.ner:
            offset, path_contains, scope = prepare_ner(args, root)
            batch_n = args.limit if args.limit is not None else args.batch
            remain = args.max_poems
            get_ner(args.ner_model)
            gen = iter_poems(
                skip_notes=True,
                path_contains=path_contains,
                offset=offset,
            )
            while True:
                n = batch_n if remain is None else min(batch_n, remain)
                if n <= 0:
                    print("已达 --max-poems，游标", offset, "范围", scope, flush=True)
                    events = load_json(root / "terms.ner.candidates.json")
                    break
                poems = take_poems(gen, n)
                if not poems:
                    print("没有更多诗，游标", offset, "范围", scope, flush=True)
                    events = load_json(root / "terms.ner.candidates.json")
                    break
                batch_events = harvest_poems(poems, args, offset)
                events = persist_ner_batch(
                    root, batch_events, args, offset, len(poems), scope,
                )
                offset += len(poems)
                if remain is not None:
                    remain -= len(poems)
                if args.review:
                    run_review(events, root, args.review_batch)
                if remain is not None and remain <= 0:
                    print("已达 --max-poems，游标", offset, "范围", scope, flush=True)
                    break
                if not args.until_done:
                    break
            if not args.until_done:
                return
        if args.review and not args.ner:
            if not events:
                print("没有粗筛文件，先跑 --ner --batch 20 --ner-threshold 0.5")
                return
            run_review(events, root, args.review_batch)
        return
if __name__ == "__main__":
    main()