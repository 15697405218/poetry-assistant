"""只灌 Postgres 倒排 / 正文检索，不重嵌向量。

  python scripts/ingest_fts.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.services.fts_ingest import ingest_fts


if __name__ == "__main__":
    ingest_fts()
