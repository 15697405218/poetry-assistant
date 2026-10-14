from app.services.memory import _approx_tokens, _need_compact, RAW_TOKEN_LIMIT, KEEP_LOW


class _M:
    def __init__(self, text, role="user"):
        self.role = role
        self.content = text
        self.rewrite = None


def test_approx_tokens():
    assert _approx_tokens("abcd") == 2


def test_need_compact_hysteresis():
    small = [_M("短")]
    assert _need_compact("", small) is False
    huge = [_M("字" * (RAW_TOKEN_LIMIT * 2 + 10)) for _ in range(KEEP_LOW + 1)]
    assert _need_compact("", huge) is True
