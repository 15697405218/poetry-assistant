"""Tool-call caps: mock messages only, no GPU / retrieve / ingest."""
from __future__ import annotations

from langgraph.graph import END

from app.services.graph import (
    CAP_MSG,
    after_model,
    call_tools,
    cap_over_budget,
    executed_tool_names,
    over_budget,
    tool_counts,
    tools_for_turn,
)


def _asst(cid_name: list[tuple[str, str]]) -> dict:
    return {
        "role": "assistant",
        "content": None,
        "tool_calls": [
            {
                "id": cid,
                "type": "function",
                "function": {"name": name, "arguments": "{}"},
            }
            for cid, name in cid_name
        ],
    }


def _tool(cid: str, content: str = "ok") -> dict:
    return {"role": "tool", "tool_call_id": cid, "content": content}


def _state(messages: list) -> dict:
    return {
        "messages": messages,
        "answer": "",
        "rewrite": "",
        "original": "",
    }


def test_executed_tool_names_only_completed():
    messages = [
        {"role": "user", "content": "问"},
        _asst([("a", "search_poems"), ("b", "search_web")]),
        _tool("a", "诗"),
        _tool("b", "网"),
        _asst([("c", "search_poems_literal")]),
    ]
    assert executed_tool_names(messages) == ["search_poems", "search_web"]
    counts = tool_counts(messages)
    assert counts["search_poems"] == 1
    assert counts["search_web"] == 1
    assert counts["search_poems_literal"] == 0
    assert over_budget("search_poems", counts) is True
    assert over_budget("search_poems_literal", counts) is False
    assert over_budget("get_author_bio", counts) is False
    assert over_budget("search_web", counts) is False


def test_local_tools_are_independent_not_shared_two():
    messages = [
        _asst([("a", "search_poems")]),
        _tool("a"),
        _asst([("b", "get_author_bio")]),
        _tool("b"),
    ]
    counts = tool_counts(messages)
    assert over_budget("search_poems", counts) is True
    assert over_budget("get_author_bio", counts) is True
    assert over_budget("search_poems_literal", counts) is False
    names = {t["function"]["name"] for t in tools_for_turn(messages)}
    assert "search_poems" not in names
    assert "get_author_bio" not in names
    assert "search_poems_literal" in names
    assert "search_web" in names


def test_web_cap_is_two():
    messages = [
        _asst([("a", "search_web")]),
        _tool("a"),
        _asst([("b", "search_web")]),
        _tool("b"),
    ]
    counts = tool_counts(messages)
    assert counts["search_web"] == 2
    assert over_budget("search_web", counts) is True
    assert over_budget("search_poems", counts) is False
    names = {t["function"]["name"] for t in tools_for_turn(messages)}
    assert "search_web" not in names
    assert "search_poems" in names


def test_over_budget_does_not_call_retrieve(monkeypatch):
    called: list[str] = []

    def fake_run(name, args):
        called.append(name)
        return "hit"

    monkeypatch.setattr("app.services.graph.run_tool", fake_run)
    messages = [
        {"role": "user", "content": "静夜思"},
        _asst([("a", "search_poems")]),
        _tool("a", "李白"),
        _asst([("b", "search_poems")]),
    ]
    out = call_tools(_state(messages))
    assert called == []
    assert out["messages"][-1]["role"] == "tool"
    assert out["messages"][-1]["content"] == CAP_MSG


def test_other_local_tool_still_runs_after_poems(monkeypatch):
    called: list[str] = []
    monkeypatch.setattr(
        "app.services.graph.run_tool",
        lambda name, args: called.append(name) or "bio",
    )
    messages = [
        _asst([("a", "search_poems")]),
        _tool("a"),
        _asst([("b", "get_author_bio")]),
    ]
    out = call_tools(_state(messages))
    assert called == ["get_author_bio"]
    assert out["messages"][-1]["content"] == "bio"


def test_three_different_local_tools_all_run(monkeypatch):
    called: list[str] = []
    monkeypatch.setattr(
        "app.services.graph.run_tool",
        lambda name, args: called.append(name) or name,
    )
    messages = [_asst([
        ("a", "search_poems"),
        ("b", "search_poems_literal"),
        ("c", "get_author_bio"),
    ])]
    out = call_tools(_state(messages))
    assert called == ["search_poems", "search_poems_literal", "get_author_bio"]


def test_batch_second_same_local_is_capped(monkeypatch):
    called: list[str] = []
    monkeypatch.setattr(
        "app.services.graph.run_tool",
        lambda name, args: called.append(name) or "hit",
    )
    messages = [_asst([("a", "search_poems"), ("b", "search_poems")])]
    out = call_tools(_state(messages))
    assert called == ["search_poems"]
    extras = [m for m in out["messages"] if m.get("role") == "tool"]
    assert extras[0]["content"] == "hit"
    assert extras[1]["content"] == CAP_MSG


def test_web_third_call_capped(monkeypatch):
    called: list[str] = []
    monkeypatch.setattr(
        "app.services.graph.run_tool",
        lambda name, args: called.append(name) or "web",
    )
    messages = [
        _asst([("a", "search_web")]),
        _tool("a"),
        _asst([("b", "search_web")]),
        _tool("b"),
        _asst([("c", "search_web")]),
    ]
    out = call_tools(_state(messages))
    assert called == []
    assert out["messages"][-1]["content"] == CAP_MSG


def test_after_model_allows_other_local_tool():
    messages = [
        _asst([("a", "search_poems")]),
        _tool("a"),
        _asst([("b", "get_author_bio")]),
    ]
    assert after_model(_state(messages)) == "tools"


def test_after_model_routes_over_budget_to_cap_not_tools():
    messages = [
        _asst([("a", "search_poems")]),
        _tool("a"),
        _asst([("b", "search_poems")]),
    ]
    assert after_model(_state(messages)) == "cap"


def test_after_model_nudge_then_tool_calls_ends():
    messages = [
        {"role": "user", "content": CAP_MSG},
        _asst([("b", "search_web")]),
    ]
    assert after_model(_state(messages)) == END


def test_cap_over_budget_does_not_retrieve(monkeypatch):
    monkeypatch.setattr(
        "app.services.graph.run_tool",
        lambda name, args: (_ for _ in ()).throw(AssertionError("should not retrieve")),
    )
    messages = [
        _asst([("a", "search_poems")]),
        _tool("a"),
        _asst([("b", "search_poems")]),
    ]
    out = cap_over_budget(_state(messages))
    assert out["messages"][-2]["content"] == CAP_MSG
    assert out["messages"][-1]["role"] == "user"
    assert out["messages"][-1]["content"] == CAP_MSG
