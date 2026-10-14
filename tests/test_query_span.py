from app.services.query_span import literal_needles, strip_question_shell


def test_strip_which_poem_suffix():
    assert strip_question_shell("床前明月光是哪首") == "床前明月光"


def test_strip_source_suffix():
    assert strip_question_shell("欲穷千里目更上一层楼出处") == "欲穷千里目更上一层楼"


def test_strip_who_wrote():
    assert strip_question_shell("将进酒谁写的") == "将进酒"


def test_needles_keep_verse_window():
    needles = literal_needles("床前明月光是哪首")
    assert "床前明月光" in needles


def test_needles_not_eval_titles():
    """抽针不能依赖评估集里的通行篇名。"""
    needles = literal_needles("飞流直下三千尺")
    assert "望庐山瀑布" not in needles
    assert any("飞流直下" in n for n in needles)
