from app.services.disambiguate import (
    PARSE_ATTEMPTS,
    _extract_json,
    disambiguate_query,
    is_verse_lookup_query,
    validate_disambiguate,
)


def test_extract_json_fence():
    data = _extract_json('```json\n{"confident": false, "query": "雨霖铃的意象"}\n```')
    assert data["query"] == "雨霖铃的意象"


def test_short_query_no_llm():
    out = disambiguate_query("啊")
    assert out["confident"] is False
    assert out["query"] == "啊"


def test_verse_lookup_heuristic():
    assert is_verse_lookup_query("床前明月光是哪首") is True
    assert is_verse_lookup_query("欲穷千里目更上一层楼出处") is True
    assert is_verse_lookup_query("飞流直下三千尺下一句") is True
    assert is_verse_lookup_query("床前明月光") is True
    assert is_verse_lookup_query("雨霖铃") is False
    assert is_verse_lookup_query("雨霖铃的意象") is False
    assert is_verse_lookup_query("雨霖铃意象") is False
    assert is_verse_lookup_query("雨霖铃是哪首") is False
    assert is_verse_lookup_query("将进酒谁写的") is False


def test_verse_query_skips_llm(monkeypatch):
    monkeypatch.setattr("app.services.disambiguate.cache_get", lambda k: None)
    stored = {}
    monkeypatch.setattr("app.services.disambiguate.cache_set", lambda k, v: stored.__setitem__(k, v))

    def boom(*_a, **_k):
        raise AssertionError("llm.chat should not run for verse lookup")

    monkeypatch.setattr("app.services.disambiguate.llm.chat", boom)
    out = disambiguate_query("床前明月光是哪首")
    assert out["confident"] is False
    assert out["query"] == "床前明月光是哪首"
    assert stored
    assert all(k.startswith("poetry:dis:") for k in stored)


def test_clause_core_skips_llm(monkeypatch):
    monkeypatch.setattr("app.services.disambiguate.cache_get", lambda k: None)
    monkeypatch.setattr("app.services.disambiguate.cache_set", lambda *_a, **_k: None)
    monkeypatch.setattr(
        "app.services.disambiguate.llm.chat",
        lambda *_a, **_k: (_ for _ in ()).throw(AssertionError("no llm")),
    )
    out = disambiguate_query("欲穷千里目更上一层楼")
    assert out["confident"] is False
    assert out["query"] == "欲穷千里目更上一层楼"


def test_cipai_still_calls_llm(monkeypatch):
    monkeypatch.setattr("app.services.disambiguate.cache_get", lambda k: None)
    monkeypatch.setattr("app.services.disambiguate.cache_set", lambda *_a, **_k: None)
    called = []

    def fake_chat(messages, temperature=0, **_k):
        called.append(messages[-1]["content"])
        return (
            '{"ambiguous": true, "keyword": "雨霖铃", "intent": "unknown",'
            ' "query": "雨霖铃", "confident": false}'
        )

    monkeypatch.setattr("app.services.disambiguate.llm.chat", fake_chat)
    out = disambiguate_query("雨霖铃的意象")
    assert called == ["雨霖铃的意象"]
    assert out["confident"] is False
    assert out["query"] == "雨霖铃的意象"


def test_validate_disambiguate_shape():
    ok = {
        "ambiguous": True,
        "keyword": "雨霖铃",
        "intent": "cipai",
        "query": "词牌雨霖铃",
        "confident": True,
    }
    assert validate_disambiguate(ok)["intent"] == "cipai"
    assert validate_disambiguate({**ok, "intent": "nope"}) is None
    assert validate_disambiguate({**ok, "extra": 1}) is None
    assert validate_disambiguate({k: v for k, v in ok.items() if k != "query"}) is None


def test_schema_miss_retries_then_original(monkeypatch):
    monkeypatch.setattr("app.services.disambiguate.cache_get", lambda k: None)
    stored = {}
    monkeypatch.setattr("app.services.disambiguate.cache_set", lambda k, v: stored.__setitem__(k, v))
    n = {"i": 0}

    def fake_chat(*_a, **_k):
        n["i"] += 1
        return '{"intent": "cipai"}'

    monkeypatch.setattr("app.services.disambiguate.llm.chat", fake_chat)
    out = disambiguate_query("雨霖铃的意象")
    assert n["i"] == PARSE_ATTEMPTS
    assert out["confident"] is False
    assert out["query"] == "雨霖铃的意象"
    assert stored


def test_schema_ok_on_retry(monkeypatch):
    monkeypatch.setattr("app.services.disambiguate.cache_get", lambda k: None)
    monkeypatch.setattr("app.services.disambiguate.cache_set", lambda *_a, **_k: None)
    n = {"i": 0}

    def fake_chat(*_a, **_k):
        n["i"] += 1
        if n["i"] == 1:
            return "not json"
        return (
            '{"ambiguous": true, "keyword": "雨霖铃", "intent": "verse",'
            ' "query": "诗句雨霖铃", "confident": true}'
        )

    monkeypatch.setattr("app.services.disambiguate.llm.chat", fake_chat)
    out = disambiguate_query("雨霖铃的意象")
    assert n["i"] == 2
    assert out["confident"] is True
    assert out["query"] == "诗句雨霖铃"
