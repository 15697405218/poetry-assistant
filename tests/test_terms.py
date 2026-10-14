from app.services.terms import load_terms, lookup, rewrite_query


def _rw(text: str) -> dict:
    return rewrite_query(text, load_terms(reload=True))


def test_rewrite_alias_fangweng():
    rw = _rw("放翁是谁")
    assert rw["original"] == "放翁是谁"
    assert "陆游" in rw["query"]
    assert "放翁" in rw["query"]
    assert any(h.get("canonical") == "陆游" for h in rw["hits"])


def test_lookup_no_typo_fuzzy():
    assert lookup("放公是谁", load_terms(reload=True)) == []


def test_canonical_stays():
    rw = _rw("陆游是谁")
    assert rw["query"] == "陆游是谁"


def test_empty():
    rw = _rw("  ")
    assert rw["query"] == ""
    assert rw["hits"] == []


def test_verse_qianli_not_gaopian():
    rw = _rw("欲穷千里目更上一层楼出处")
    assert rw["query"] == "欲穷千里目更上一层楼出处"
    assert "高骈" not in rw["query"]
