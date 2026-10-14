"""倒排查询入口。实现已持久化到 Postgres，见 app.services.fts。"""
from __future__ import annotations

from app.services.fts import lookup_index
from app.services.poetry_parse import first_clause, hamming

__all__ = ["first_clause", "hamming", "lookup_index"]
