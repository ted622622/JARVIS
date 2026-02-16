"""Groq chat client for LLM inference (llama-3.3-70b-versatile, etc.).

OpenAI-compatible API via httpx — same pattern as NvidiaClient / OpenRouterClient.
"""

from __future__ import annotations

import asyncio
from typing import Any

from loguru import logger

from .base_client import BaseClient, ChatMessage, ChatResponse


class GroqChatClient(BaseClient):
    """Wraps Groq API (OpenAI-compatible) for fast LLM inference.

    Features:
    - Exponential backoff on 429/5xx (1s → 2s → 4s, max 3 retries)
    - httpx direct (no SDK dependency)
    """

    DEFAULT_MODEL = "llama-3.3-70b-versatile"

    def __init__(
        self,
        api_key: str,
        model: str | None = None,
        timeout: float = 30.0,
    ):
        super().__init__("https://api.groq.com/openai/v1", api_key, timeout)
        self.model = model or self.DEFAULT_MODEL

    def _build_headers(self) -> dict[str, str]:
        return {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

    async def chat(
        self,
        messages: list[ChatMessage],
        *,
        model: str | None = None,
        temperature: float = 0.6,
        max_tokens: int = 4096,
        **kwargs: Any,
    ) -> ChatResponse:
        client = await self._get_client()
        payload = {
            "model": model or self.model,
            "messages": [self._format_msg(m) for m in messages],
            "temperature": temperature,
            "max_tokens": max_tokens,
        }

        backoff = 1.0
        max_retries = 3

        for attempt in range(max_retries + 1):
            try:
                resp = await client.post("/chat/completions", json=payload)

                if resp.status_code == 200:
                    data = resp.json()
                    return ChatResponse(
                        content=data["choices"][0]["message"]["content"],
                        model=data.get("model", payload["model"]),
                        usage=data.get("usage", {}),
                        raw=data,
                    )

                if resp.status_code == 429:
                    logger.warning(f"429 from Groq, backoff {backoff}s (attempt {attempt + 1})")
                elif resp.status_code >= 500:
                    logger.warning(f"{resp.status_code} from Groq, backoff {backoff}s")
                else:
                    raise GroqAPIError(
                        f"{resp.status_code} from Groq: {resp.text[:200]}"
                    )

            except (GroqAPIError, KeyboardInterrupt):
                raise
            except Exception as e:
                if attempt == max_retries:
                    raise
                logger.warning(f"Groq error: {e}, backoff {backoff}s (attempt {attempt + 1})")

            if attempt < max_retries:
                await asyncio.sleep(min(backoff, 30.0))
                backoff *= 2.0

        raise GroqAPIError(f"Failed after {max_retries + 1} attempts")

    @staticmethod
    def _format_msg(msg: ChatMessage) -> dict[str, Any]:
        return {"role": msg.role, "content": msg.content}


class GroqAPIError(Exception):
    """General Groq API error."""
