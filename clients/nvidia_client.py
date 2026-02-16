"""NVIDIA NIM client for Kimi K2.5 with Token Bucket rate limiting."""

from __future__ import annotations

import asyncio
import time
from typing import Any

from loguru import logger

from .base_client import (
    BaseClient,
    ChatMessage,
    ChatResponse,
    RateLimitTracker,
    TokenBucket,
)


class NvidiaClient(BaseClient):
    """Wraps Kimi K2.5 via NVIDIA NIM API.

    Features:
    - Token Bucket rate limiting (40 RPM → 0.667 tokens/sec)
    - Exponential backoff on 429 (1s → 2s → 4s → 8s, max 60s)
    - Silent mode: 15 min cooldown when 429 > 10/hr
    """

    DEFAULT_MODEL = "moonshotai/kimi-k2.5-instruct"

    def __init__(
        self,
        api_key: str,
        base_url: str = "https://integrate.api.nvidia.com/v1",
        model: str | None = None,
        rpm_limit: int = 40,
        timeout: float = 30.0,
    ):
        super().__init__(base_url, api_key, timeout)
        self.model = model or self.DEFAULT_MODEL
        self.bucket = TokenBucket(rate=rpm_limit / 60.0, capacity=rpm_limit)
        self.rate_tracker = RateLimitTracker(threshold_per_hour=10, cooldown_minutes=15)

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
        if self.rate_tracker.is_silent:
            raise RateLimitExceeded(
                "NVIDIA NIM in silent mode — too many 429s. "
                f"Resumes at {time.strftime('%H:%M:%S', time.localtime(self.rate_tracker.silent_until))}"
            )

        # Wait for rate limit token
        wait = await self.bucket.acquire()
        if wait > 0:
            logger.debug(f"Rate limited, waited {wait:.2f}s")

        payload = {
            "model": model or self.model,
            "messages": [self._format_msg(m) for m in messages],
            "temperature": temperature,
            "max_tokens": max_tokens,
            **kwargs,
        }

        return await self._request_with_backoff(payload)

    async def _request_with_backoff(
        self,
        payload: dict[str, Any],
        max_retries: int = 4,
    ) -> ChatResponse:
        """Execute request with exponential backoff on 429/5xx."""
        backoff = 1.0
        client = await self._get_client()

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
                    entered_silent = self.rate_tracker.record_429()
                    if entered_silent:
                        raise RateLimitExceeded("Entered silent mode after repeated 429s")
                    logger.warning(f"429 from NVIDIA NIM, backoff {backoff}s (attempt {attempt + 1})")

                elif resp.status_code >= 500:
                    logger.warning(f"{resp.status_code} from NVIDIA NIM, backoff {backoff}s")

                else:
                    # 4xx (except 429) — not retryable, raise immediately
                    raise NvidiaAPIError(
                        f"{resp.status_code} from NVIDIA NIM: {resp.text[:200]}"
                    )

            except (RateLimitExceeded, NvidiaAPIError, KeyboardInterrupt):
                raise
            except Exception as e:
                if attempt == max_retries:
                    raise
                logger.warning(f"Request error: {e}, backoff {backoff}s (attempt {attempt + 1})")

            if attempt < max_retries:
                await asyncio.sleep(min(backoff, 60.0))
                backoff *= 2.0

        raise NvidiaAPIError(f"Failed after {max_retries + 1} attempts")

    @staticmethod
    def _format_msg(msg: ChatMessage) -> dict[str, Any]:
        return {"role": msg.role, "content": msg.content}


class RateLimitExceeded(Exception):
    """Raised when the client enters silent mode due to excessive 429s."""


class NvidiaAPIError(Exception):
    """General NVIDIA NIM API error."""
