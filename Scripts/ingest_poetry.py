import argparse
import json
import os
import re
import shutil
import sys
from pathlib import Path

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("ANONYMIZED_TELEMETRY", "False")
os.environ.setdefault("OMP_NUM_THREADS", "1")

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from app.config import settings
import chromadb
from app.services.rag import make_ef, to_hans

_YIZUO = re.compile(r"一作[「『“\"]?([\u4e00-\u9fff]{2,12})")

SKIP_PARTS = {"rank", "strains", "error", "御定全唐詩"}


def is_skip(path: Path) -> bool:
    if "author" in path.name.lower():
        return True
    return any(part in SKIP_PARTS for part in path.parts)


def item_body(item: dict) -> str:
    raw = item.get("paragraphs") or item.get("para") or item.get("content") or ""
    if isinstance(raw, list):
        return "".join(str(x) for x in raw)
    return str(raw)


def item_title(item: dict) -> str:
    return item.get("title") or item.get("rhythmic") or item.get("chapter") or "未知"


def item_title_aliases(item: dict) -> list[str]:
    seen: list[str] = []
    for key in ("title", "rhythmic", "chapter"):
        title = to_hans(str(item.get(key) or "")).strip()
        if title and title != "未知" and title not in seen:
            seen.append(title)
    return seen


def item_variants(item: dict) -> list[str]:
    notes = item.get("notes") or item.get("note") or ""
    if isinstance(notes, list):
        blob = "".join(str(x) for x in notes)
    else:
        blob = str(notes)
    found: list[str] = []
    for v in _YIZUO.findall(to_hans(blob)):
        if v not in found:
            found.append(v)
    return found


def item_document(author: str, titles: list[str], body: str, variants: list[str]) -> str:
    title = titles[0] if titles else "未知"
    parts = [f"作者：{author}", f"标题：{title}", body]
    extra = [t for t in titles[1:]]
    if extra:
        parts.append("异名：" + " ".join(extra))
    if variants:
        parts.append("异文：" + " ".join(variants))
    return "\n".join(parts)


BATCH = 64


def _dedup_key(author: str, title: str, body: str) -> str:
    return f"{author}\t{title}\t{body[:80]}"


def ingest_all(resume: bool = False) -> None:
    persist = Path(settings.chroma_persist_dir)
    persist.mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(path=str(persist))
    ef = make_ef()
    if resume:
        collection = client.get_collection(name="poetry", embedding_function=ef)
        next_id = collection.count()
        print("续灌起点", next_id, flush=True)
    else:
        try:
            client.delete_collection("poetry")
        except Exception:
            pass
        collection = client.create_collection(
            name="poetry",
            embedding_function=ef,
            metadata={"hnsw:space": "cosine", "hnsw:search_ef": settings.hnsw_search_ef},
        )
        next_id = 0

    already = next_id
    accepted = 0
    buf: list[dict] = []
    seen: set[str] = set()
    skipped = 0

    def flush():
        nonlocal next_id, buf
        if not buf:
            return
        collection.add(
            ids=[str(next_id + i) for i in range(len(buf))],
            documents=[p["text"] for p in buf],
            metadatas=[{
                "author": p["author"],
                "title": p["title"],
                "dynasty": p["dynasty"],
                "source_path": p["source_path"],
                "title_aliases": p.get("title_aliases") or "",
            } for p in buf],
        )
        next_id += len(buf)
        print("已写入", next_id, "跳过重复", skipped, flush=True)
        buf = []

    root_dir = Path(settings.poetry_data_dir)
    for file in root_dir.rglob("*.json"):
        if is_skip(file):
            continue
        try:
            data = json.loads(file.read_text(encoding="utf-8"))
        except Exception as e:
            print("读失败", file, e)
            continue
        if not isinstance(data, list):
            continue
        dynasty = file.parent.name
        for item in data:
            if not isinstance(item, dict):
                continue
            body = item_body(item)
            if not body.strip():
                continue
            if len(body.strip()) < 2:
                continue
            author = to_hans(str(item.get("author") or "未知"))
            titles = item_title_aliases(item) or [to_hans(str(item_title(item)))]
            title = titles[0]
            body = to_hans(body)
            key = _dedup_key(author, title, body)
            if key in seen:
                skipped += 1
                continue
            seen.add(key)
            accepted += 1
            if resume and accepted <= already:
                if accepted == already:
                    print("已对齐到断点", already, flush=True)
                continue
            variants = item_variants(item)
            buf.append({
                "text": item_document(author, titles, body, variants),
                "author": author,
                "title": title,
                "title_aliases": " ".join(titles),
                "dynasty": to_hans(str(dynasty)),
                "source_path": str(file),
            })
            if len(buf) >= BATCH:
                flush()
        print("已解析", file.name, "累计缓冲", len(buf), flush=True)

    if buf:
        flush()
    print("最终", collection.count(), "重复跳过", skipped)


def ingest_authors() -> None:
    persist = Path(settings.chroma_persist_dir)
    persist.mkdir(parents=True, exist_ok=True)
    ef = make_ef()
    probe = ef(["床前明月光"])
    print("embed probe ok", len(probe[0]), flush=True)
    client = chromadb.PersistentClient(path=str(persist))
    try:
        client.delete_collection("authors")
    except Exception:
        pass
    col = client.create_collection(
        name="authors",
        embedding_function=ef,
        metadata={"hnsw:space": "cosine"},
    )

    root = Path(settings.poetry_data_dir)
    files = [
        p for p in root.rglob("*.json")
        if "author" in p.name.lower() and "error" not in p.parts
    ]
    print("作者文件", [str(p.relative_to(root)) for p in files])

    ids: list[str] = []
    docs: list[str] = []
    metas: list[dict] = []
    n = 0

    def flush() -> None:
        nonlocal ids, docs, metas
        if not ids:
            return
        col.add(ids=ids, documents=docs, metadatas=metas)
        ids, docs, metas = [], [], []

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
            if not name or not bio:
                continue
            ids.append(str(n))
            docs.append(f"姓名：{name}\n简介：{bio}")
            metas.append({
                "name": name,
                "source_path": str(file),
            })
            n += 1
            if len(ids) >= 500:
                flush()
                print("作者已写入", n)
    flush()
    print("作者最终", col.count())


def wipe_persist() -> None:
    persist = Path(settings.chroma_persist_dir)
    if persist.exists():
        print("清空 Chroma", persist, flush=True)
        shutil.rmtree(persist, ignore_errors=False)
    persist.mkdir(parents=True, exist_ok=True)


def write_index_version() -> None:
    dest = Path(settings.chroma_persist_dir).parent / "index_version.json"
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(
        json.dumps({
            "version": settings.index_version,
            "embedding": settings.embedding_model,
            "collection": "poetry",
        }, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print("索引版本", dest, settings.index_version)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--resume", action="store_true", help="从已有 poetry 集合断点续灌，不清空")
    args = parser.parse_args()
    if args.resume:
        ingest_all(resume=True)
    else:
        wipe_persist()
        ingest_authors()
        ingest_all()
    write_index_version()
    from app.services.fts_ingest import ingest_fts

    ingest_fts()