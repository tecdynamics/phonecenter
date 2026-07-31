"""Gemma chat client — streaming over the EXISTING llama.cpp server.

Talks to the OpenAI-compatible `/chat/completions` endpoint that llama.cpp
exposes (on the TecAI box: http://127.0.0.1:8084/v1). We do not run or swap the
model — only call it. Streaming is required: the orchestrator begins
synthesizing TTS on the first clause to keep time-to-first-audio low (spec §6).

On any transport/protocol error this raises `LLMError` so the orchestrator can
degrade gracefully (apologize + offer a human) instead of hanging the call.
"""

from __future__ import annotations

import json
import logging
from collections.abc import AsyncIterator, Sequence

import httpx

logger = logging.getLogger(__name__)

Message = dict[str, str]


class LLMError(RuntimeError):
    """The LLM endpoint was unreachable or returned an error."""


class GemmaClient:
    def __init__(
        self,
        endpoint: str,
        model: str,
        *,
        temperature: float = 0.35,
        top_p: float = 0.9,
        max_tokens: int = 220,
        api_key: str = "",
        timeout: float = 30.0,
        path: str = "/chat/completions",
        chat_template_kwargs: dict | None = None,
    ) -> None:
        self._model = model
        self._temperature = temperature
        self._top_p = top_p
        self._max_tokens = max_tokens
        # Engine chat-template options forwarded verbatim in the request body.
        # For the tecai vLLM (Qwen thinking model) this carries
        # {"enable_thinking": False} — WITHOUT it the model streams its reasoning
        # as content and the TTS speaks it aloud on the call. llama.cpp ignores
        # the field, so it is safe on either backend.
        self._chat_template_kwargs = chat_template_kwargs
        # Build the absolute URL ourselves: a leading-slash path with httpx
        # base_url would drop the endpoint's '/v1' prefix (→ 404).
        self._url = endpoint.rstrip("/") + "/" + path.lstrip("/")
        headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
        # trust_env=False: ignore HTTP(S)_PROXY/ALL_PROXY — these are localhost
        # services and must be reached directly, never via a proxy.
        self._client = httpx.AsyncClient(timeout=timeout, headers=headers, trust_env=False)

    async def aclose(self) -> None:
        await self._client.aclose()

    async def stream_chat(self, messages: Sequence[Message]) -> AsyncIterator[str]:
        """Yield assistant text deltas as they arrive (SSE `data:` chunks)."""
        payload = {
            "model": self._model,
            "messages": list(messages),
            "temperature": self._temperature,
            "top_p": self._top_p,
            "max_tokens": self._max_tokens,
            "stream": True,
        }
        if self._chat_template_kwargs:
            payload["chat_template_kwargs"] = self._chat_template_kwargs
        try:
            async with self._client.stream("POST", self._url, json=payload) as resp:
                if resp.status_code >= 400:
                    body = (await resp.aread())[:300]
                    raise LLMError(f"LLM HTTP {resp.status_code}: {body!r}")
                async for line in resp.aiter_lines():
                    if not line or not line.startswith("data:"):
                        continue  # skip keepalives / comments
                    data = line[len("data:"):].strip()
                    if data == "[DONE]":
                        break
                    try:
                        obj = json.loads(data)
                    except json.JSONDecodeError:
                        logger.debug("Non-JSON SSE chunk ignored: %r", data[:120])
                        continue
                    choices = obj.get("choices") or []
                    if not choices:
                        continue
                    choice = choices[0]
                    content = (choice.get("delta") or {}).get("content")
                    if content:
                        yield content
                    if choice.get("finish_reason"):
                        break
        except httpx.HTTPError as exc:  # connect/read/timeout
            raise LLMError(f"LLM request failed: {exc}") from exc
