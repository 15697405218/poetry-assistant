from app.chat_eval import score_case


def test_provenance_needs_gold_and_lib_tool():
    gold = {"must": ["静夜思"], "expect_tools": "required"}
    ok = score_case(gold, "这是李白的【主出处】静夜思", ["search_poems"])
    assert ok["ok"]
    bad = score_case(gold, "这是将进酒", ["search_poems"])
    assert not bad["must_ok"]
    no_tool = score_case(gold, "静夜思", [])
    assert not no_tool["tool_ok"]


def test_chitchat_fake_cite_fails():
    gold = {"must": [], "expect_tools": "forbidden", "forbid_cite": True}
    hi = score_case(gold, "你好，我是诗词助手。", [])
    assert hi["ok"]
    fake = score_case(gold, "【主出处#1 李白·静夜思】床前明月光", [])
    assert not fake["cite_ok"]
    assert not fake["ok"]


def test_dispute_any_of():
    gold = {"any": ["王之涣", "朱斌"], "expect_tools": "required"}
    hit = score_case(gold, "通行多作王之涣《登鹳雀楼》", ["search_poems_literal"])
    assert hit["ok"]
