import socket
import time

import httpx
from openai import (
    APIConnectionError,
    APIStatusError,
    AuthenticationError,
    InternalServerError,
    OpenAI,
)
from openai import RateLimitError as SDKRateLimitError
from app.config import settings

def _patch_ipv4_first_for_ark() -> None:
    """Ark 在部分 Windows 上 IPv6 不可达；只对 volces 主机优先 A 记录。补丁必须幂等。"""
    if getattr(socket, "_poetry_ipv4_patched", False):
        return
    orig = socket.getaddrinfo

    def ipv4_first(host, port, family=0, type=0, proto=0, flags=0):
        infos = orig(host, port, family, type, proto, flags)
        host_s = host.decode() if isinstance(host, (bytes, bytearray)) else str(host or "")
        if "volces.com" not in host_s and "volcengineapi.com" not in host_s:
            return infos
        if family in (0, socket.AF_UNSPEC):
            v4 = [item for item in infos if item[0] == socket.AF_INET]
            if v4:
                return v4
        return infos

    socket.getaddrinfo = ipv4_first
    socket._poetry_ipv4_patched = True


_patch_ipv4_first_for_ark()

class AuthError(Exception):
    pass
class RateLimitError(Exception):
    pass
class UpstreamError(Exception):
    pass

class LLMClient:
    def __init__(self, api_key: str, base_url: str,model: str):
        self.api_key = api_key
        self.base_url = base_url
        self.model = model
        self.client = OpenAI(
            api_key=api_key,
            base_url=base_url,
            timeout=60,
            max_retries=0,
            http_client=httpx.Client(trust_env=False, timeout=60.0),
        )

    def chat(self, messages: list[dict], **kwargs) -> str:
        for attempt in range(3):
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    **kwargs,
                )
                return response.choices[0].message.content
            except AuthenticationError as e:
                raise AuthError(f"API key无效 或其他认证错误: {e}")
            except SDKRateLimitError  as e: 
                if attempt == 2:
                    raise RateLimitError(f"重试超出: {e}")
                else:
                    time.sleep(1)
            except (InternalServerError, APIConnectionError) as e:
                if attempt == 2:
                    raise UpstreamError(f"上游不可用: {e}")
                else:
                    time.sleep(1)
            except APIStatusError  as e:
                raise UpstreamError(f"请求失败: {e}")
    def chat_turn(
        self,
        messages: list[dict],
        tools: list[dict] | None = None,
        tool_choice: bool | None = None,
    ) -> dict:
        for attempt in range(3):
            try:
                kwargs = {
                    "model": self.model,
                    "messages": messages,
                }
                if tools:
                    kwargs["tools"] = tools
                if tool_choice:
                    kwargs["tool_choice"] = tool_choice
                response = self.client.chat.completions.create(**kwargs)
                msg = response.choices[0].message
                calls = []
                for c in (msg.tool_calls or []):
                    calls.append({
                        "id": c.id,
                        "name": c.function.name,
                        "arguments": c.function.arguments,
                    })
                return {"content": msg.content, "tool_calls": calls}
            except AuthenticationError as e:
                raise AuthError(f"API key无效 或其他认证错误: {e}")
            except SDKRateLimitError  as e: 
                if attempt == 2:
                    raise RateLimitError(f"重试超出: {e}")
                else:
                    time.sleep(1)
            except (InternalServerError, APIConnectionError) as e:
                if attempt == 2:
                    raise UpstreamError(f"上游不可用: {e}")
                else:
                    time.sleep(1)
            except APIStatusError  as e:
                raise UpstreamError(f"请求失败: {e}")
    def chat_stream(
        self,
        messages: list[dict],
        tools: list[dict] | None = None,
        tool_choice: str | None = None,
    ):
        for attempt in range(3):
            try:
                kwargs = {
                    "model": self.model,
                    "messages": messages,
                    "stream": True,
                }
                if tools:
                    kwargs["tools"] = tools
                if tool_choice:
                    kwargs["tool_choice"] = tool_choice
                stream = self.client.chat.completions.create(**kwargs)
                acc: dict[int, dict] = {}
                for chunk in stream:
                    if not chunk.choices:
                        continue
                    delta = chunk.choices[0].delta
                    if delta.content:
                        yield {"token": delta.content}
                    for tc in (delta.tool_calls or []):
                        idx = 0 if tc.index is None else tc.index
                        slot = acc.setdefault(
                            idx, {"id": "", "name": "", "arguments": ""},
                        )
                        if tc.id:
                            slot["id"] += tc.id
                        fn = tc.function
                        if fn is None:
                            continue
                        if fn.name:
                            slot["name"] += fn.name
                        if fn.arguments:
                            slot["arguments"] += fn.arguments
                if acc:
                    yield {"tool_calls": [acc[i] for i in sorted(acc)]}
                return
            except AuthenticationError as e:
                raise AuthError(f"API key无效 或其他认证错误: {e}")
            except SDKRateLimitError  as e:
                if attempt == 2:
                    raise RateLimitError(f"重试超出: {e}")
                else:
                    time.sleep(1)
            except (InternalServerError, APIConnectionError) as e:
                if attempt == 2:
                    raise UpstreamError(f"上游不可用: {e}")
                else:
                    time.sleep(1)
            except APIStatusError as e:
                raise UpstreamError(f"请求失败: {e}")



llm = LLMClient(
    api_key=settings.ark_api_key,
    base_url=settings.base_url,
    model=settings.chat_model,
)

