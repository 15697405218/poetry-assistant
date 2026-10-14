"""把 BGE 重排模型拉到本地 HuggingFace 缓存。线上 HF_HUB_OFFLINE=1，必须先跑一次。

官方 HF 不稳时走 ModelScope / hf-mirror。

  python scripts/download_reranker.py
"""
from __future__ import annotations

import os
import shutil
import sys
from pathlib import Path

os.environ["HF_HUB_OFFLINE"] = "0"
os.environ["TRANSFORMERS_OFFLINE"] = "0"
os.environ["HF_HUB_DISABLE_XET"] = "1"
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

from sentence_transformers import CrossEncoder

MODEL = os.environ.get("RERANKER_MODEL", "BAAI/bge-reranker-large")


def _install_into_hf_cache(repo_id: str, src: Path) -> Path:
    name = "models--" + repo_id.replace("/", "--")
    dest = Path.home() / ".cache" / "huggingface" / "hub" / name / "snapshots" / "modelscope"
    dest.mkdir(parents=True, exist_ok=True)
    has_safe = (src / "model.safetensors").is_file()
    for item in src.iterdir():
        if item.name in {".git", ".cache", ".mv"}:
            continue
        if item.name.endswith(".incomplete"):
            continue
        if has_safe and item.name.startswith("pytorch_model.bin"):
            continue
        target = dest / item.name
        if item.is_file():
            shutil.copy2(item, target)
        elif item.is_dir():
            if target.exists():
                shutil.rmtree(target)
            shutil.copytree(item, target)
    refs = dest.parent.parent / "refs"
    refs.mkdir(exist_ok=True)
    (refs / "main").write_text("modelscope", encoding="utf-8")
    return dest


def _via_modelscope(repo_id: str) -> Path:
    from modelscope.hub.snapshot_download import snapshot_download

    print("改走 ModelScope", repo_id, flush=True)
    src = Path(snapshot_download(
        repo_id,
        ignore_file_pattern=["onnx", "*.onnx", "*.onnx_data", "pytorch_model.bin"],
    ))
    dest = _install_into_hf_cache(repo_id, src)
    print("已写入 HF 缓存", dest, flush=True)
    return dest


def main() -> None:
    print("下载", MODEL, flush=True)
    if os.environ.get("PREFER_MODELSCOPE") == "1":
        _via_modelscope(MODEL)
    else:
        try:
            from huggingface_hub import snapshot_download

            snapshot_download(
                repo_id=MODEL,
                ignore_patterns=["onnx/*", "*.onnx", "*.onnx_data", "pytorch_model.bin"],
            )
        except Exception as e:
            print("HuggingFace 失败:", type(e).__name__, e, flush=True)
            _via_modelscope(MODEL)
    CrossEncoder(MODEL, local_files_only=True, device="cpu")
    print("完成，之后 retriever 会走精排", flush=True)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print("失败:", type(e).__name__, e, file=sys.stderr)
        raise SystemExit(1)
