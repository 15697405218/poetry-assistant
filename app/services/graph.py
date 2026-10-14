import json
from collections import Counter
from typing import TypedDict

from langgraph.graph import END, START, StateGraph
from langgraph.types import StreamWriter

from app.services.checkpoint import RedisCheckpointSaver
from app.services.llm import AuthError, RateLimitError, UpstreamError, llm
from app.services.tools import TOOLS, run_tool

SYSTEM = """你是诗词助手。

查证才用工具：问现成诗词、作者、出处、原文、某句是谁写的，必须先调用工具，只根据工具结果回答。用户已给出诗句原文时用 search_poems_literal，否则用 search_poems；问作者生平用 get_author_bio。search_poems 已重排：优先【主出处】，【参考】不要当同等出处并列。不要把字面沾边的乐曲、典故硬往问句上靠。引用写成 [作者·标题] 或工具里的【主出处#id】。工具没查到的不要编成库里有的诗，也不要说库里只有这些。本库每个检索工具每轮最多1次，联网最多2次。

本库没有、或问当代事实、奖项、通行版本争议时用 search_web。引用网页时带标题或 URL，不要标成本库 [作者·标题]。登录墙、个人书架说进不去，不要假装搜到了。

创作、改写、赏析、闲聊可以直接答，不必先检索。仿写某家风格时可以检索几首作参考，不是必须。
未调用 search_poems / search_poems_literal / get_author_bio，或工具结果是没有检索到，禁止使用【主出处】【参考】【出处#id】和 [作者·标题] 这种本库引用格式；观点、对比、闲聊可以谈，不要装成检索命中。库外篇目可以凭记忆回答，但必须说明不是本库检索结果。
创作时如果用户要求避免抄袭，你需要调用工具检索"""

CAP_MSG = "已达本轮检索上限，请只根据已有结果作答，不要再调用工具。"
LOCAL_CAPS = {
    "search_poems": 1,
    "search_poems_literal": 1,
    "get_author_bio": 1,
}
WEB_TOOL = "search_web"
WEB_CAP = 2


def _tool_call_name(c: dict) -> str:
    if c.get("name"):
        return str(c["name"])
    fn = c.get("function") or {}
    return str(fn.get("name") or "")


def used_library_tools(messages: list) -> bool:
    return any(n in LOCAL_CAPS for n in executed_tool_names(messages))


def executed_tool_names(messages: list) -> list[str]:
    """Tools that already ran this graph: assistant tool_calls with a matching role=tool result."""
    id_to_name: dict[str, str] = {}
    for m in messages or []:
        for c in m.get("tool_calls") or []:
            cid = c.get("id")
            name = _tool_call_name(c)
            if cid and name:
                id_to_name[str(cid)] = name
    names: list[str] = []
    for m in messages or []:
        if m.get("role") != "tool":
            continue
        cid = m.get("tool_call_id")
        name = id_to_name.get(str(cid) if cid is not None else "") or m.get("name")
        if name:
            names.append(str(name))
    return names


def tool_counts(messages: list) -> Counter:
    return Counter(executed_tool_names(messages))


def over_budget(name: str, counts: Counter) -> bool:
    if name in LOCAL_CAPS:
        return counts[name] >= LOCAL_CAPS[name]
    if name == WEB_TOOL:
        return counts[name] >= WEB_CAP
    return False


def tools_for_turn(messages: list) -> list[dict] | None:
    counts = tool_counts(messages)
    out = [t for t in TOOLS if not over_budget(t["function"]["name"], counts)]
    return out or None


def _pending_names(last: dict) -> list[str]:
    return [_tool_call_name(c) for c in last.get("tool_calls") or []]


def _all_pending_over_budget(state: dict) -> bool:
    last = (state.get("messages") or [None])[-1] or {}
    names = _pending_names(last)
    if not names:
        return False
    counts = tool_counts(state.get("messages") or [])
    return all(over_budget(n, counts) for n in names)


def _assistant_tool_msg(calls: list[dict]) -> dict:
    return {
        "role": "assistant",
        "content": None,
        "tool_calls": [
            {
                "id": c["id"],
                "type": "function",
                "function": {"name": c["name"], "arguments": c["arguments"]},
            }
            for c in calls
        ],
    }


def _tool_result(call_id: str, content: str) -> dict:
    return {"role": "tool", "tool_call_id": call_id, "content": content}


def _last_is_nudge(messages: list) -> bool:
    if not messages:
        return False
    last = messages[-1]
    return last.get("role") == "user" and last.get("content") == CAP_MSG


def _nudged_then_tool_calls(messages: list) -> bool:
    """No-tools retry still emitted tool_calls — stop the loop."""
    if len(messages) < 2:
        return False
    last, prev = messages[-1], messages[-2]
    return bool(last.get("tool_calls")) and prev.get("role") == "user" and prev.get("content") == CAP_MSG


class ChatState(TypedDict):
    messages: list
    answer: str
    rewrite: str
    original: str


def call_model(state: ChatState, writer: StreamWriter) -> dict:
    writer({"status": "正在思考中…"})
    tools = None if _last_is_nudge(state["messages"]) else tools_for_turn(state["messages"])
    kwargs: dict = {"tools": tools}
    if tools:
        kwargs["tool_choice"] = "auto"
    try:
        parts: list[str] = []
        tool_calls: list[dict] = []
        for ev in llm.chat_stream(state["messages"], **kwargs):
            piece = ev.get("token")
            if piece:
                if not parts:
                    writer({"status": ""})
                parts.append(piece)
                writer({"token": piece})
            if ev.get("tool_calls"):
                names = [c.get("name") or "" for c in ev["tool_calls"]]
                if any(n == "search_web" for n in names) and not any(
                    n.startswith("search_poems") or n == "get_author_bio" for n in names
                ):
                    writer({"status": "正在联网检索…"})
                else:
                    writer({"status": "正在检索库…"})
                tool_calls = ev["tool_calls"]
        if tool_calls:
            return {
                "messages": state["messages"] + [_assistant_tool_msg(tool_calls)],
            }
        writer({"status": ""})
        text = "".join(parts)
        if used_library_tools(state["messages"]):
            writer({"lib_used": True})
        return {
            "messages": state["messages"] + [{"role": "assistant", "content": text}],
            "answer": text,
        }
    except (AuthError, RateLimitError, UpstreamError) as e:
        raise type(e)(f"model: {e}") from e


def _run_one_tool(state: ChatState, name: str, args: str) -> str:
    if name == "search_poems":
        try:
            parsed = json.loads(args or "{}")
        except json.JSONDecodeError:
            parsed = {}
        rewritten = (state.get("rewrite") or "").strip()
        if rewritten:
            parsed["q"] = rewritten
        original = (state.get("original") or "").strip()
        if original:
            parsed["original"] = original
        args = json.dumps(parsed, ensure_ascii=False)
    return run_tool(name, args)


def call_tools(state: ChatState) -> dict:
    last = state["messages"][-1]
    extras = []
    counts = tool_counts(state["messages"])
    for c in last["tool_calls"]:
        fn = c["function"]
        name = fn["name"]
        args = fn["arguments"]
        if over_budget(name, counts):
            extras.append(_tool_result(c["id"], CAP_MSG))
            continue
        out = _run_one_tool(state, name, args)
        extras.append(_tool_result(c["id"], out))
        counts[name] += 1
    remaining_none = tools_for_turn(state["messages"] + extras) is None
    if remaining_none:
        extras.append({"role": "user", "content": CAP_MSG})
    return {"messages": state["messages"] + extras}


def cap_over_budget(state: ChatState) -> dict:
    """Do not retrieve: pending calls are already over their per-tool caps."""
    last = state["messages"][-1]
    extras = [_tool_result(c["id"], CAP_MSG) for c in last.get("tool_calls") or []]
    extras.append({"role": "user", "content": CAP_MSG})
    return {"messages": state["messages"] + extras}


def after_model(state: ChatState) -> str:
    last = state["messages"][-1]
    if not last.get("tool_calls"):
        return END
    if _nudged_then_tool_calls(state["messages"]):
        return END
    if _all_pending_over_budget(state):
        return "cap"
    return "tools"


def graph_config(conversation_id: int) -> dict:
    return {
        "recursion_limit": 12,
        "configurable": {"thread_id": f"conv:{conversation_id}"},
    }


def build_graph():
    g = StateGraph(ChatState)
    g.add_node("model", call_model)
    g.add_node("tools", call_tools)
    g.add_node("cap", cap_over_budget)
    g.add_edge(START, "model")
    g.add_conditional_edges(
        "model", after_model, {"tools": "tools", "cap": "cap", END: END},
    )
    g.add_edge("tools", "model")
    g.add_edge("cap", "model")
    return g.compile(checkpointer=RedisCheckpointSaver())


graph = build_graph()
