"""OpenRouter client for backup model access (deepseek-chat, etc.)."""

from __future__ import annotations

import asyncio
from typing import Any

from loguru import logger

from .base_client import BaseClient, ChatMessage, ChatResponse


class OpenRouterClient(BaseClient):
    """Wraps OpenRouter API (OpenAI-compatible).

    Primary backup model: deepseek/deepseek-chat
    Also supports: google/gemini-2.0-flash-001 for vision backup
    """

    DEFAULT_MODEL = "deepseek/deepseek-chat"

    def __init__(
        self,
        api_key: str,
        base_url: str = "https://openrouter.ai/api/v1",
        model: str | None = None,
        timeout: float = 30.0,
    ):
        super().__init__(base_url, api_key, timeout)
        self.model = model or self.DEFAULT_MODEL

    def _build_headers(self) -> dict[str, str]:
        return {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://github.com/ted/jarvis",
            "X-Title": "J.A.R.V.I.S.",
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
            **kwargs,
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
                    logger.warning(f"429 from OpenRouter, backoff {backoff}s")
                elif resp.status_code >= 500:
                    logger.warning(f"{resp.status_code} from OpenRouter, backoff {backoff}s")
                else:
                    resp.raise_for_status()

            except KeyboardInterrupt:
                raise
            except Exception as e:
                if attempt == max_retries:
                    raise
                logger.warning(f"OpenRouter error: {e}, backoff {backoff}s")

            if attempt < max_retries:
                await asyncio.sleep(min(backoff, 30.0))
                backoff *= 2.0

        raise OpenRouterAPIError(f"Failed after {max_retries + 1} attempts")

    async def get_remaining_credits(self) -> float:
        """Query OpenRouter for remaining balance."""
        try:
            client = await self._get_client()
            resp = await client.get("/auth/key")
            if resp.status_code == 200:
                data = resp.json()
                return float(data.get("data", {}).get("limit_remaining", 0.0))
        except Exception as e:
            logger.error(f"Failed to query OpenRouter credits: {e}")
        return -1.0

    @staticmethod
    def _format_msg(msg: ChatMessage) -> dict[str, Any]:
        return {"role": msg.role, "content": msg.content}


class OpenRouterAPIError(Exception):
    """General OpenRouter API error."""
