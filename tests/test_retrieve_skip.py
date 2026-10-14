"""Exact-hit GPU skip: mocks only, no Chroma / CrossEncoder / Postgres."""
from app.services.retrieve import retrieve_poems


def _verse_hit(**kwargs):
    hit = {
        "id": "fts-1",
        "text": "床前明月光，疑是地上霜。",
        "author": "李白",
        "title": "静夜思",
        "dynasty": "唐",
        "distance": 0.0,
        "match": "index",
    }
    hit.update(kwargs)
    return hit


def _patch_exact_path(monkeypatch, lookup, stored=None):
    monkeypatch.setattr("app.services.retrieve.cache_get", lambda _k: None)

    def _set(key, value, ttl=None):
        if stored is not None:
            stored[key] = value

    monkeypatch.setattr("app.services.retrieve.cache_set", _set)
    monkeypatch.setattr("app.services.retrieve.lookup_index", lookup)
    monkeypatch.setattr("app.services.retrieve.search_poems_by_title", lambda *_a, **_k: [])
    monkeypatch.setattr("app.services.retrieve.search_poems_literal", lambda *_a, **_k: [])


def test_index_exact_skips_vector_and_rerank(monkeypatch):
    stored = {}

    def lookup(needle, limit=15):
        if "床前明月光" in (needle or ""):
            return [_verse_hit()]
        return []

    _patch_exact_path(monkeypatch, lookup, stored=stored)

    def boom_search(*_a, **_k):
        raise AssertionError("search_poems should not run")

    def boom_rerank(*_a, **_k):
        raise AssertionError("rerank should not run")

    monkeypatch.setattr("app.services.retrieve.search_poems", boom_search)
    monkeypatch.setattr("app.services.retrieve.search_poems_by_author", boom_search)
    monkeypatch.setattr("app.services.retrieve.rerank_hits", boom_rerank)

    hits, meta = retrieve_poems("床前明月光是哪首")
    assert hits
    assert hits[0]["title"] == "静夜思"
    assert meta.get("skip_vector") is True
    assert meta.get("skip_rerank") is True
    assert meta.get("algo") == "fts.v4"
    assert stored
    assert all(k.startswith("poetry:ret:") for k in stored)


def test_conflicting_exacts_rerank_without_vector(monkeypatch):
    called = {"rerank": 0, "vector": 0}

    def lookup(needle, limit=15):
        if "欲穷千里目" not in (needle or ""):
            return []
        return [
            _verse_hit(
                id="fts-a",
                author="王之涣",
                title="登鹳雀楼",
                text="白日依山尽，黄河入海流。欲穷千里目，更上一层楼。",
            ),
            _verse_hit(
                id="fts-b",
                author="朱斌",
                title="登楼",
                text="欲穷千里目，更上一层楼。",
            ),
        ]

    _patch_exact_path(monkeypatch, lookup)

    def fake_search(*_a, **_k):
        called["vector"] += 1
        raise AssertionError("search_poems should not run when verse exacts exist")

    def fake_rerank(query, hits, top_k):
        called["rerank"] += 1
        out = []
        for i, h in enumerate(hits):
            item = dict(h)
            item["rerank"] = 0.9 - i * 0.1
            out.append(item)
        return out

    monkeypatch.setattr("app.services.retrieve.search_poems", fake_search)
    monkeypatch.setattr("app.services.retrieve.search_poems_by_author", fake_search)
    monkeypatch.setattr("app.services.retrieve.rerank_hits", fake_rerank)

    hits, meta = retrieve_poems("欲穷千里目更上一层楼出处")
    assert called["vector"] == 0
    assert called["rerank"] == 1
    assert meta.get("skip_vector") is True
    assert meta.get("skip_rerank") is False
    assert {h["title"] for h in hits} >= {"登鹳雀楼"}


def test_theme_query_still_vectors(monkeypatch):
    called = {"vector": 0, "rerank": 0}
    _patch_exact_path(monkeypatch, lambda *_a, **_k: [])

    def fake_search(q, top_k=5, where=None):
        called["vector"] += 1
        return [{"author": "柳永", "title": "雨霖铃", "text": "寒蝉凄切", "distance": 0.2}]

    def fake_rerank(query, hits, top_k):
        called["rerank"] += 1
        out = []
        for h in hits:
            item = dict(h)
            item["rerank"] = 0.8
            out.append(item)
        return out

    monkeypatch.setattr("app.services.retrieve.search_poems", fake_search)
    monkeypatch.setattr("app.services.retrieve.search_poems_by_author", fake_search)
    monkeypatch.setattr("app.services.retrieve.rerank_hits", fake_rerank)

    hits, meta = retrieve_poems("雨霖铃的意象")
    assert called["vector"] >= 1
    assert called["rerank"] == 1
    assert meta.get("skip_vector") is False
    assert hits
