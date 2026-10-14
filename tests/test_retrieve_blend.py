from app.services.author_aliases import extract_aliases
from app.services.line_index import first_clause, hamming
from app.services.retrieve import blend_ranked, rank_query_of


def test_hamming_one_char_variant():
    assert hamming("床前明月光", "床前看月光") == 1
    assert hamming("床前明月光", "床前明月光") == 0


def test_first_clause_stops_at_comma():
    assert first_clause("床前看月光，疑是地上霜。") == "床前看月光"


def test_extract_hao_jushi():
    aliases = extract_aliases("李白，字太白，号青莲居士。")
    assert "太白" in aliases
    assert "青莲居士" in aliases


def test_blend_drops_low_score_short_title():
    ranked = [
        {"author": "甲", "title": "雨诗", "text": "下雨", "rerank": 0.9, "match": None},
        {"author": "柳永", "title": "雨霖铃", "text": "寒蝉凄切", "rerank": 0.2, "match": "title"},
        {"author": "乙", "title": "听雨", "text": "雨声", "rerank": 0.8, "match": None},
    ]
    out = blend_ranked(ranked, 2)
    titles = [h["title"] for h in out]
    assert titles == ["雨诗", "听雨"]
    assert "雨霖铃" not in titles


def test_blend_keeps_high_score_verse():
    ranked = [
        {"author": "甲", "title": "杂诗", "text": "xxx", "rerank": 0.9, "match": None},
        {"author": "朱斌", "title": "登楼", "text": "欲穷千里目", "rerank": 0.22, "match": "literal"},
        {"author": "乙", "title": "登高", "text": "yyy", "rerank": 0.85, "match": None},
    ]
    out = blend_ranked(ranked, 2)
    assert any(h["title"] == "登楼" for h in out)


def test_blend_drops_low_vector_noise():
    ranked = [
        {"author": "甲", "title": "好诗", "text": "aaa", "rerank": 0.9, "match": None},
        {"author": "乙", "title": "噪声", "text": "bbb", "rerank": 0.05, "match": None},
    ]
    out = blend_ranked(ranked, 2)
    assert [h["title"] for h in out] == ["好诗"]


def test_rank_query_uses_stripped_original():
    assert rank_query_of("欲穷千里目更上一层楼出处", "王之涣登鹳雀楼赏析") == "欲穷千里目更上一层楼"
    assert rank_query_of("放翁是谁", "陆游（放翁）是谁") == "陆游（放翁）是谁"
