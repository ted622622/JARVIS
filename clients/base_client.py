"""Base client with shared HTTP logic and retry handling."""

from __future__ import annotations

import asyncio
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

import httpx
from loguru import logger


@dataclass
class ChatMessage:
    role: str  # "system" | "user" | "assistant"
    content: str | list[dict[str, Any]]  # str for text, list for multimodal


@dataclass
class ChatResponse:
    content: str
    model: str
    usage: dict[str, int] = field(default_factory=dict)
    raw: dict[str, Any] = field(default_factory=dict)


class TokenBucket:
    """Token bucket rate limiter for API calls."""

    def __init__(self, rate: float, capacity: int):
        self.rate = rate          # tokens added per second
        self.capacity = capacity  # max burst size
        self.tokens = float(capacity)
        self.last_refill = time.monotonic()
        self._lock = asyncio.Lock()

    async def acquire(self) -> float:
        """Wait until a token is available. Returns wait time in seconds."""
        async with self._lock:
            now = time.monotonic()
            elapsed = now - self.last_refill
            self.tokens = min(self.capacity, self.tokens + elapsed * self.rate)
            self.last_refill = now

            if self.tokens >= 1.0:
                self.tokens -= 1.0
                return 0.0

            wait_time = (1.0 - self.tokens) / self.rate
            self.tokens = 0.0
            self.last_refill = now

        await asyncio.sleep(wait_time)
        return wait_time


class RateLimitTracker:
    """Tracks 429 errors to trigger silent mode."""

    def __init__(self, threshold_per_hour: int = 10, cooldown_minutes: int = 15):
        self.threshold = threshold_per_hour
        self.cooldown_seconds = cooldown_minutes * 60
        self.timestamps: list[float] = []
        self.silent_until: float = 0.0

    def record_429(self) -> bool:
        """Record a 429 error. Returns True if entering silent mode."""
        now = time.monotonic()
        self.timestamps = [t for t in self.timestamps if now - t < 3600]
        self.timestamps.append(now)

        if len(self.timestamps) >= self.threshold:
            self.silent_until = now + self.cooldown_seconds
            logger.warning(f"Entering silent mode for {self.cooldown_seconds}s "
                           f"({len(self.timestamps)} 429s in last hour)")
            return True
        return False

    @property
    def is_silent(self) -> bool:
        return time.monotonic() < self.silent_until


class BaseClient(ABC):
    """Abstract base for all API clients."""

    def __init__(self, base_url: str, api_key: str, timeout: float = 30.0):
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.timeout = timeout
        self._client: httpx.AsyncClient | None = None

    async def _get_client(self) -> httpx.AsyncClient:
        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(
                base_url=self.base_url,
                timeout=httpx.Timeout(self.timeout),
                headers=self._build_headers(),
            )
        return self._client

    @abstractmethod
    def _build_headers(self) -> dict[str, str]:
        ...

    @abstractmethod
    async def chat(
        self,
        messages: list[ChatMessage],
        **kwargs: Any,
    ) -> ChatResponse:
        ...

    async def health_check(self) -> bool:
        """Basic connectivity test."""
        try:
            await self.chat(
                [ChatMessage(role="user", content="ping")],
                max_tokens=5,
            )
            return True
        except Exception as e:
            logger.error(f"{self.__class__.__name__} health check failed: {e}")
            return False

    async def close(self) -> None:
        if self._client and not self._client.is_closed:
            await self._client.aclose()
            self._client = None
