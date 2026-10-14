"""本库引用壳剥离：不调 GPU / 检索 / 灌库。"""
from __future__ import annotations

from app.services.chat_run import iter_graph_events, run_chat_job, strip_library_cites
from app.services.graph import SYSTEM, used_library_tools


class _FakeGraph:
    def __init__(self, events):
        self.events = events
        self.calls = []

    def stream(self, state, config, stream_mode="custom"):
        self.calls.append((state, config, stream_mode))
        yield from self.events


def test_system_forbids_fake_library_cite():
    assert "禁止使用【主出处】【参考】【出处#id】和 [作者·标题]" in SYSTEM
    assert "必须说明不是本库检索结果" in SYSTEM


def test_strip_library_cites_brackets():
    assert strip_library_cites("【主出处#1 李白·静夜思】床前明月光") == "床前明月光"
    assert strip_library_cites("见【参考#2 杜甫·春望】") == "见"
    assert strip_library_cites("【出处#99】") == ""
    assert strip_library_cites("【主出处】只是壳") == "只是壳"


def test_strip_author_title_brackets():
    assert strip_library_cites("出自[李白·静夜思]") == "出自"
    assert strip_library_cites("[余光中·乡愁] 不是本库检索") == "不是本库检索"


def test_strip_keeps_plain_names_and_verses():
    text = "李白的床前明月光，余光中也写过乡愁。"
    assert strip_library_cites(text) == text
    assert strip_library_cites("李白 [杜甫·春望] 诗句") == "李白 诗句"


def test_strip_conservative_other_brackets():
    assert strip_library_cites("见[1]和【注】") == "见[1]和【注】"
    assert strip_library_cites("《静夜思》") == "《静夜思》"
    assert strip_library_cites("[see note]") == "[see note]"


def test_used_library_tools_this_turn_only():
    assert used_library_tools([{"role": "assistant", "content": "【主出处】假装"}]) is False
    messages = [
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [{
                "id": "a",
                "type": "function",
                "function": {"name": "search_poems", "arguments": "{}"},
            }],
        },
        {"role": "tool", "tool_call_id": "a", "content": "没有检索到足够相近的诗文"},
    ]
    assert used_library_tools(messages) is True
    web_only = [
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [{
                "id": "w",
                "type": "function",
                "function": {"name": "search_web", "arguments": "{}"},
            }],
        },
        {"role": "tool", "tool_call_id": "w", "content": "网页"},
    ]
    assert used_library_tools(web_only) is False


def test_iter_graph_events_strips_cites_without_lib_tool():
    g = _FakeGraph([{"token": "【主出处#1 李白·静夜思】床前明月光"}])
    saved: list[tuple[int, str]] = []
    evs = list(iter_graph_events(
        3,
        "q",
        "o",
        graph_impl=g,
        load_history_fn=lambda cid: [{"role": "user", "content": "问"}],
        save_assistant_fn=lambda cid, text: saved.append((cid, text)),
    ))
    assert saved == [(3, "床前明月光")]
    assert evs[-1] == {"type": "done", "data": "床前明月光"}


def test_iter_graph_events_keeps_cites_when_lib_used():
    raw = "【主出处#1 李白·静夜思】床前明月光"
    g = _FakeGraph([{"lib_used": True}, {"token": raw}])
    saved: list[str] = []
    list(iter_graph_events(
        4,
        "q",
        "o",
        graph_impl=g,
        load_history_fn=lambda cid: [],
        save_assistant_fn=lambda cid, text: saved.append(text),
    ))
    assert saved == [raw]


def test_run_chat_job_returns_stripped_text():
    g = _FakeGraph([{"token": "出自[余光中·乡愁]，这不是本库检索结果"}])
    text, err = run_chat_job(
        1,
        "q",
        "o",
        graph_impl=g,
        load_history_fn=lambda cid: [],
        save_assistant_fn=lambda cid, t: None,
    )
    assert err is None
    assert text == "出自，这不是本库检索结果"
    assert "[余光中·乡愁]" not in text
