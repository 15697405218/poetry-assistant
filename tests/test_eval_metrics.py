from app.eval_metrics import f1, gold_recall, precision, relevant


def test_relevant_author():
    assert relevant({"author": "陆游", "title": "示儿", "text": "死去元知万事空"}, {
        "gold_authors": ["陆游"], "gold_titles": ["示儿"], "gold_keywords": [],
    })


def test_precision_recall():
    hits = [
        {"author": "陆游", "title": "示儿", "text": "死去元知万事空"},
        {"author": "李白", "title": "静夜思", "text": "床前明月光"},
    ]
    gold = {"gold_authors": ["陆游"], "gold_titles": ["示儿"], "gold_keywords": []}
    assert precision(hits, gold, 2) == 0.5
    assert gold_recall(hits, gold) == 1.0
    assert abs(f1(0.5, 1.0) - 2 / 3) < 1e-6


def test_short_keyword_not_enough_when_titled():
    hit = {"author": "苏轼", "title": "定风波", "text": "莫听穿林打叶声 春路雨添花 明月"}
    gold = {"gold_authors": ["苏轼"], "gold_titles": ["水调歌头"], "gold_keywords": ["明月", "苏轼"]}
    assert not relevant(hit, gold)


def test_author_only_when_no_title():
    hit = {"author": "陆游", "title": "游山西村", "text": "山重水复疑无路"}
    gold = {"gold_authors": ["陆游"], "gold_titles": [], "gold_keywords": ["陆游"]}
    assert relevant(hit, gold)


def test_recall_same_poem_disputed_author():
    hits = [
        {"author": "朱斌", "title": "登楼", "text": "白日依山尽，黄河入海流。欲穷千里目，更上一层楼。"},
        {"author": "王之涣", "title": "登鹳雀楼", "text": "白日依山尽，黄河入海流。欲穷千里目，更上一层楼。"},
    ]
    gold = {
        "gold_authors": ["王之涣"],
        "gold_titles": ["登鹳雀楼"],
        "gold_keywords": ["白日依山尽"],
    }
    assert relevant(hits[0], gold)
    assert gold_recall(hits[:1], gold) == 1.0
    assert gold_recall(hits, gold) == 1.0
    assert precision(hits, gold, 2) == 1.0
