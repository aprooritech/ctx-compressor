from __future__ import annotations

import json
import logging
import time
import uuid
from collections.abc import AsyncIterator
from typing import Any, ClassVar

import httpx

from src.config import ProviderName, Settings
from src.providers.base import SSE_DONE, ChatProvider, sse_event, translate_transport_errors
from src.schemas import ChatCompletionRequest, ChatMessage

logger = logging.getLogger(__name__)


class OllamaProvider(ChatProvider):
    name: ClassVar[ProviderName] = "ollama"

    def __init__(self, client: httpx.AsyncClient, settings: Settings) -> None:
        super().__init__(client, settings)
        self._base_url = settings.ollama_host

    async def chat(
        self, request: ChatCompletionRequest, *, timeout: float | None = None
    ) -> dict[str, Any]:
        payload = self._build_payload(request, stream=False)
        url = f"{self._base_url}/api/chat"
        with translate_transport_errors(self.name, url):
            response = await self._client.post(url, json=payload, timeout=self._timeout(timeout))
        await self._raise_for_status(response)
        return self._to_openai_response(response.json(), payload["model"])

    async def stream(
        self, request: ChatCompletionRequest, *, timeout: float | None = None
    ) -> AsyncIterator[bytes]:
        payload = self._build_payload(request, stream=True)
        url = f"{self._base_url}/api/chat"
        completion_id = f"chatcmpl-{uuid.uuid4().hex[:24]}"
        created = int(time.time())
        model = str(payload["model"])

        with translate_transport_errors(self.name, url):
            async with self._client.stream(
                "POST", url, json=payload, timeout=self._timeout(timeout)
            ) as response:
                await self._raise_for_status(response)
                first_chunk = True
                async for line in response.aiter_lines():
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        data = json.loads(line)
                    except json.JSONDecodeError:
                        logger.warning("ollama sent a non-JSON stream line: %.120s", line)
                        continue
                    chunk = self._to_openai_chunk(
                        data, completion_id, created, model, include_role=first_chunk
                    )
                    first_chunk = False
                    yield sse_event(chunk)
                    if data.get("done"):
                        break
        yield SSE_DONE

    async def list_models(self) -> list[dict[str, Any]]:
        url = f"{self._base_url}/api/tags"
        with translate_transport_errors(self.name, url):
            response = await self._client.get(url, timeout=self._timeout(None))
        await self._raise_for_status(response)
        payload = response.json()
        entries = payload.get("models", []) if isinstance(payload, dict) else []
        result: list[dict[str, Any]] = []
        for entry in entries:
            if not isinstance(entry, dict):
                continue
            model_name = str(entry.get("name", ""))
            if not model_name:
                continue
            result.append(
                {
                    "id": f"ollama/{model_name}",
                    "object": "model",
                    "owned_by": "ollama",
                    "created": 0,
                    "context_length": _context_length(entry),
                }
            )
        return result

    async def health(self) -> bool:
        try:
            response = await self._client.get(
                f"{self._base_url}/api/tags", timeout=httpx.Timeout(5.0, connect=2.0)
            )
            return response.is_success
        except httpx.HTTPError:
            return False

    def _build_payload(self, request: ChatCompletionRequest, *, stream: bool) -> dict[str, Any]:
        options: dict[str, Any] = {}
        if request.temperature is not None:
            options["temperature"] = request.temperature
        if request.top_p is not None:
            options["top_p"] = request.top_p
        if request.top_k is not None:
            options["top_k"] = request.top_k
        if request.seed is not None:
            options["seed"] = request.seed
        if request.presence_penalty is not None:
            options["presence_penalty"] = request.presence_penalty
        if request.frequency_penalty is not None:
            options["frequency_penalty"] = request.frequency_penalty
        max_tokens = request.effective_max_tokens
        if max_tokens is not None:
            options["num_predict"] = max_tokens
        stop = request.stop_sequences
        if stop:
            options["stop"] = stop

        payload: dict[str, Any] = {
            "model": request.model or self.default_model,
            "messages": [_to_ollama_message(m) for m in request.messages],
            "stream": stream,
            "keep_alive": self._settings.ollama_keep_alive,
        }
        if options:
            payload["options"] = options
        if request.tools:
            payload["tools"] = request.tools
        if request.response_format and request.response_format.get("type") == "json_object":
            payload["format"] = "json"
        return payload

    def _to_openai_response(self, data: dict[str, Any], model: str) -> dict[str, Any]:
        message = data.get("message") or {}
        prompt_tokens = int(data.get("prompt_eval_count") or 0)
        completion_tokens = int(data.get("eval_count") or 0)
        response_message: dict[str, Any] = {
            "role": str(message.get("role", "assistant")),
            "content": message.get("content", ""),
        }
        if message.get("tool_calls"):
            response_message["tool_calls"] = _normalise_tool_calls(message["tool_calls"])
        return {
            "id": f"chatcmpl-{uuid.uuid4().hex[:24]}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": str(data.get("model") or model),
            "choices": [
                {
                    "index": 0,
                    "message": response_message,
                    "finish_reason": _finish_reason(data),
                }
            ],
            "usage": {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": prompt_tokens + completion_tokens,
            },
        }

    def _to_openai_chunk(
        self,
        data: dict[str, Any],
        completion_id: str,
        created: int,
        model: str,
        *,
        include_role: bool,
    ) -> dict[str, Any]:
        message = data.get("message") or {}
        done = bool(data.get("done"))
        delta: dict[str, Any] = {}
        if include_role:
            delta["role"] = str(message.get("role", "assistant"))
        content = message.get("content")
        if content:
            delta["content"] = content
        if message.get("tool_calls"):
            delta["tool_calls"] = _normalise_tool_calls(message["tool_calls"])

        chunk: dict[str, Any] = {
            "id": completion_id,
            "object": "chat.completion.chunk",
            "created": created,
            "model": str(data.get("model") or model),
            "choices": [
                {
                    "index": 0,
                    "delta": delta,
                    "finish_reason": _finish_reason(data) if done else None,
                }
            ],
        }
        if done:
            prompt_tokens = int(data.get("prompt_eval_count") or 0)
            completion_tokens = int(data.get("eval_count") or 0)
            chunk["usage"] = {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": prompt_tokens + completion_tokens,
            }
        return chunk


def _to_ollama_message(message: ChatMessage) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "role": "system" if message.role == "developer" else message.role,
        "content": message.text_content,
    }
    images = _extract_images(message)
    if images:
        payload["images"] = images
    if message.tool_calls:
        payload["tool_calls"] = message.tool_calls
    if message.tool_call_id:
        payload["tool_call_id"] = message.tool_call_id
    return payload


def _extract_images(message: ChatMessage) -> list[str]:
    if not isinstance(message.content, list):
        return []
    images: list[str] = []
    for part in message.content:
        if part.get("type") != "image_url":
            continue
        image_url = part.get("image_url")
        url = image_url.get("url") if isinstance(image_url, dict) else None
        if isinstance(url, str) and url.startswith("data:") and "base64," in url:
            images.append(url.split("base64,", 1)[1])
    return images


def _normalise_tool_calls(tool_calls: Any) -> list[dict[str, Any]]:
    # Ollama omits id and type, which OpenAI clients expect to be present.
    if not isinstance(tool_calls, list):
        return []
    result: list[dict[str, Any]] = []
    for index, call in enumerate(tool_calls):
        if not isinstance(call, dict):
            continue
        function = call.get("function") or {}
        arguments = function.get("arguments", "")
        if not isinstance(arguments, str):
            arguments = json.dumps(arguments, ensure_ascii=False)
        result.append(
            {
                "id": call.get("id") or f"call_{uuid.uuid4().hex[:20]}",
                "type": "function",
                "index": call.get("index", index),
                "function": {"name": function.get("name", ""), "arguments": arguments},
            }
        )
    return result


def _finish_reason(data: dict[str, Any]) -> str | None:
    if not data.get("done"):
        return None
    reason = str(data.get("done_reason") or "stop")
    return {"stop": "stop", "length": "length", "load": "stop"}.get(reason, reason)


def _context_length(entry: dict[str, Any]) -> int | None:
    details = entry.get("details")
    if isinstance(details, dict):
        for key in ("context_length", "num_ctx"):
            value = details.get(key)
            if isinstance(value, int):
                return value
    return None
