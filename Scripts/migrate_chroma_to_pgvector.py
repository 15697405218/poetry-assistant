"""一次性：Chroma poetry/authors → Postgres pgvector。不重算 embedding，不洗倒排。"""
from __future__ import annotations

import os
import sys
from pathlib import Path

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("ANONYMIZED_TELEMETRY", "False")

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.services.chroma_migrate import match_staged_embeddings, migrate_from_chroma

if __name__ == "__main__":
    if "--match-only" in sys.argv:
        print("只对齐 stage → poems，不重读 Chroma", flush=True)
        result = match_staged_embeddings()
    else:
        print("开始 Chroma → pgvector，不重算向量、不删 poems", flush=True)
        result = migrate_from_chroma()
    print("完成", result, flush=True)
