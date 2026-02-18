"""Unified model router — all upstream code calls this, never clients directly."""

from __future__ import annotations

import asyncio
import os
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import yaml
from loguru import logger

from clients.base_client import ChatMessage, ChatResponse
from clients.nvidia_client import NvidiaClient, RateLimitExceeded
from clients.openrouter_client import OpenRouterClient
from clients.zhipu_client import ImageResponse, ZhipuClient


class ModelRole(str, Enum):
    CEO = "ceo"
    VISION = "vision"
    IMAGE = "image"


class ProviderStatus(str, Enum):
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    DOWN = "down"


@dataclass
class FailoverEvent:
    """Record of a failover event for diagnostics."""
    timestamp: float = field(default_factory=time.time)
    from_provider: str = ""
    to_provider: str = ""
    reason: str = ""
    role: str = ""


class ModelRouter:
    """Single entry point for all model interactions.

    Responsibilities:
    - Route requests to the correct provider based on role
    - Automatic failover via ordered chain (CEO: zhipu → groq → openrouter)
    - Context bridging when switching between providers with different context windows
    - Health monitoring and recovery probing
    """

    def __init__(
        self,
        nvidia_client: NvidiaClient,
        zhipu_client: ZhipuClient,
        openrouter_client: OpenRouterClient,
        groq_client=None,
        config: dict[str, Any] | None = None,
    ):
        self.nvidia = nvidia_client
        self.zhipu = zhipu_client
        self.openrouter = openrouter_client
        self.groq = groq_client
        self.config = config or {}

        # Failover state
        failover_cfg = self.config.get("failover", {})
        self._consecutive_429: dict[str, int] = {}
        self._consecutive_5xx: dict[str, int] = {}
        self._provider_status: dict[str, ProviderStatus] = {
            "zhipu_ceo": ProviderStatus.HEALTHY,   # zhipu for CEO chain (glm-4.6v / glm-4.5-air lite)
            "zhipu": ProviderStatus.HEALTHY,        # zhipu for VISION/IMAGE
            "groq": ProviderStatus.HEALTHY,
            "openrouter": ProviderStatus.HEALTHY,
            "nvidia": ProviderStatus.HEALTHY,       # kept but not in CEO chain
        }
        self._failover_trigger_429 = failover_cfg.get("trigger", {}).get("consecutive_429", 2)
        self._failover_trigger_5xx = failover_cfg.get("trigger", {}).get("consecutive_5xx", 1)
        self._recovery_interval = failover_cfg.get("recovery", {}).get("check_interval_seconds", 1800)
        self._healthy_checks_required = failover_cfg.get("recovery", {}).get("healthy_checks_required", 3)
        self._last_recovery_check: dict[str, float] = {}
        self._recovery_healthy_count: dict[str, int] = {}

        # Context bridging config
        bridging = failover_cfg.get("context_bridging", {})
        self._keep_recent_turns = bridging.get("keep_recent_turns", 8)
        self._summary_max_tokens = bridging.get("summary_max_tokens", 2000)

        # Failover event history
        self._failover_events: list[FailoverEvent] = []

        # Lock for provider status mutations
        self._status_lock = asyncio.Lock()

    # ── Public API ──────────────────────────────────────────────

    def select_model(self, task_type: str = "ceo") -> str:
        """Select the appropriate Zhipu model based on task type.

        - "template", "format", "cron_message": lightweight tasks → ZHIPU_LITE_MODEL
        - Everything else (including "ceo"): auto-balance 4.6V/4.7 via model_balancer
        """
        if task_type in ("template", "format", "cron_message"):
            return os.getenv("ZHIPU_LITE_MODEL", "glm-4.6v")

        # CEO: auto-balance between 4.6V and 4.7 pools
        ceo_env = os.getenv("ZHIPU_CEO_MODEL", "auto")
        if ceo_env == "auto":
            try:
                from core.model_balancer import select_model as _bal_select
                return _bal_select()
            except Exception:
                return "glm-4.6v"
        return ceo_env

    def _get_chain_for_role(
        self, role: ModelRole, task_type: str = "ceo",
    ) -> list[tuple[str, dict[str, Any]]]:
        """Return ordered provider chain for a given role.

        Each entry is (status_key, extra_kwargs) where status_key maps to
        _provider_status and _get_client_by_name.
        """
        if role == ModelRole.CEO:
            model = self.select_model(task_type)
            chain = [("zhipu_ceo", {"model": model})]
            if self.groq is not None:
                chain.append(("groq", {}))
            chain.append(("openrouter", {}))
            return chain
        elif role == ModelRole.VISION:
            return [("zhipu", {}), ("openrouter", {"model": "google/gemini-2.0-flash-001"})]
        elif role == ModelRole.IMAGE:
            return [("zhipu", {})]
        raise ValueError(f"Unknown role: {role}")

    async def chat(
        self,
        messages: list[ChatMessage],
        *,
        role: ModelRole = ModelRole.CEO,
        task_type: str = "ceo",
        **kwargs: Any,
    ) -> ChatResponse:
        """Send a chat request, automatically routing based on role with chain failover."""
        chain = self._get_chain_for_role(role, task_type=task_type)
        last_error = None
        tried_providers: list[str] = []

        for status_key, extra_kwargs in chain:
            if self._provider_status.get(status_key, ProviderStatus.HEALTHY) == ProviderStatus.DOWN:
                continue

            client = self._get_client_by_name(status_key)
            if client is None:
                continue

            try:
                merged = {**kwargs, **extra_kwargs}
                # Apply context bridging if this is not the first provider in chain
                if tried_providers:
                    messages = await self._bridge_context(messages, role)

                response = await client.chat(messages, **merged)
                await self._on_success(status_key)
                return response
            except RateLimitExceeded:
                logger.warning(f"Provider {status_key} in silent mode for role={role.value}")
                await self._mark_down(status_key)
                last_error = f"{status_key} rate limited"
            except Exception as e:
                logger.warning(f"Provider {status_key} failed for role={role.value}: {e}")
                await self._handle_failure(status_key, e)
                last_error = str(e)

            # Record failover
            tried_providers.append(status_key)
            # Log failover event to next provider
            remaining = [(k, _) for k, _ in chain if k not in tried_providers
                         and self._provider_status.get(k, ProviderStatus.HEALTHY) != ProviderStatus.DOWN
                         and self._get_client_by_name(k) is not None]
            if remaining:
                next_provider = remaining[0][0]
                self._failover_events.append(FailoverEvent(
                    from_provider=status_key,
                    to_provider=next_provider,
                    reason=last_error or "unknown",
                    role=role.value,
                ))
                logger.info(f"Failing over {status_key} → {next_provider} for role={role.value}")

        raise RouterError(f"All providers failed for role={role.value}: {last_error}")

    async def generate_image(self, prompt: str, **kwargs: Any) -> ImageResponse:
        """Generate image via Zhipu CogView."""
        return await self.zhipu.generate_image(prompt, **kwargs)

    async def vision_analyze(
        self,
        image_url: str,
        prompt: str,
        **kwargs: Any,
    ) -> ChatResponse:
        """Analyze image via vision model with failover."""
        if self._provider_status.get("zhipu") != ProviderStatus.DOWN:
            try:
                return await self.zhipu.vision_analyze(image_url, prompt, **kwargs)
            except Exception as e:
                logger.warning(f"Vision primary failed: {e}")
                await self._mark_down("zhipu")

        # Fallback to Gemini via OpenRouter
        messages = [
            ChatMessage(
                role="user",
                content=[
                    {"type": "image_url", "image_url": {"url": image_url}},
                    {"type": "text", "text": prompt},
                ],
            )
        ]
        return await self.openrouter.chat(
            messages, model="google/gemini-2.0-flash-001", **kwargs
        )

    # ── Recovery Probing ────────────────────────────────────────

    async def probe_recovery(self) -> dict[str, str]:
        """Check if downed providers have recovered. Call this periodically."""
        results = {}
        now = time.monotonic()

        for provider_name, status in self._provider_status.items():
            if status != ProviderStatus.DOWN:
                continue

            last_check = self._last_recovery_check.get(provider_name, 0.0)
            if now - last_check < self._recovery_interval:
                continue

            self._last_recovery_check[provider_name] = now
            client = self._get_client_by_name(provider_name)
            if client is None:
                continue

            healthy = await client.health_check()
            if healthy:
                count = self._recovery_healthy_count.get(provider_name, 0) + 1
                self._recovery_healthy_count[provider_name] = count

                if count >= self._healthy_checks_required:
                    self._provider_status[provider_name] = ProviderStatus.HEALTHY
                    self._recovery_healthy_count[provider_name] = 0
                    results[provider_name] = "recovered"
                    logger.info(f"Provider {provider_name} recovered after {count} healthy checks")
                else:
                    results[provider_name] = f"healing ({count}/{self._healthy_checks_required})"
            else:
                self._recovery_healthy_count[provider_name] = 0
                results[provider_name] = "still_down"

        return results

    # ── Health Check ────────────────────────────────────────────

    async def health_check_all(self) -> dict[str, bool]:
        """Run health checks on all providers concurrently."""
        checks = [
            self.nvidia.health_check(),
            self.zhipu.health_check(),
            self.openrouter.health_check(),
        ]
        labels = ["nvidia", "zhipu", "openrouter"]

        if self.groq is not None:
            checks.append(self.groq.health_check())
            labels.append("groq")

        results = await asyncio.gather(*checks, return_exceptions=True)
        return {label: result is True for label, result in zip(labels, results)}

    @property
    def status(self) -> dict[str, str]:
        return {k: v.value for k, v in self._provider_status.items()}

    @property
    def failover_history(self) -> list[FailoverEvent]:
        return list(self._failover_events)

    # ── Internal ────────────────────────────────────────────────

    def _get_client_by_name(self, name: str):
        """Resolve status_key to actual client instance."""
        mapping = {
            "nvidia": self.nvidia,
            "zhipu": self.zhipu,
            "zhipu_ceo": self.zhipu,   # same client, different status tracking
            "openrouter": self.openrouter,
            "groq": self.groq,
        }
        return mapping.get(name)

    async def _on_success(self, provider: str) -> None:
        async with self._status_lock:
            self._consecutive_429[provider] = 0
            self._consecutive_5xx[provider] = 0

    async def _handle_failure(self, provider: str, error: Exception) -> None:
        async with self._status_lock:
            error_str = str(error)
            if "429" in error_str or isinstance(error, RateLimitExceeded):
                count = self._consecutive_429.get(provider, 0) + 1
                self._consecutive_429[provider] = count
                if count >= self._failover_trigger_429:
                    self._do_mark_down(provider)
            elif "404" in error_str:
                # Model endpoint not found — immediately mark down
                self._do_mark_down(provider)
            elif "500" in error_str or "502" in error_str or "503" in error_str:
                count = self._consecutive_5xx.get(provider, 0) + 1
                self._consecutive_5xx[provider] = count
                if count >= self._failover_trigger_5xx:
                    self._do_mark_down(provider)

    async def _mark_down(self, provider: str) -> None:
        async with self._status_lock:
            self._do_mark_down(provider)

    def _do_mark_down(self, provider: str) -> None:
        """Internal mark-down without lock (caller must hold _status_lock)."""
        if self._provider_status.get(provider) != ProviderStatus.DOWN:
            self._provider_status[provider] = ProviderStatus.DOWN
            self._recovery_healthy_count[provider] = 0
            logger.warning(f"Provider {provider} marked DOWN, failover activated")

    async def _bridge_context(
        self,
        messages: list[ChatMessage],
        role: ModelRole,
    ) -> list[ChatMessage]:
        """Truncate and summarize conversation for backup model with smaller context window.

        Strategy (truncate_and_summarize):
        1. Keep the most recent N turns verbatim
        2. Try to LLM-summarize older messages into < 2000 tokens
        3. If LLM summary fails, fall back to simple truncation
        """
        if len(messages) <= self._keep_recent_turns:
            return messages

        older = messages[:-self._keep_recent_turns]
        recent = messages[-self._keep_recent_turns:]

        # Try LLM-powered summarization via the backup provider itself
        summary_text = await self._try_llm_summarize(older)
        if summary_text is None:
            summary_text = self._truncation_fallback(older)

        bridged = [ChatMessage(role="system", content=summary_text)] + recent
        return bridged

    async def _try_llm_summarize(self, messages: list[ChatMessage]) -> str | None:
        """Attempt to summarize older messages using the backup model."""
        try:
            conversation_text = []
            for msg in messages:
                content = msg.content if isinstance(msg.content, str) else "[multimodal content]"
                conversation_text.append(f"[{msg.role}]: {content}")

            full_text = "\n".join(conversation_text)
            # Truncate input if too long (avoid sending massive context to summarizer)
            if len(full_text) > 8000:
                full_text = full_text[-8000:]

            summary_request = [
                ChatMessage(
                    role="system",
                    content="You are a conversation summarizer. Compress the following conversation "
                            "into a concise summary under 500 characters. Preserve key decisions, "
                            "facts, and user preferences. Output only the summary in the same language.",
                ),
                ChatMessage(role="user", content=full_text),
            ]

            resp = await self.openrouter.chat(summary_request, max_tokens=512)
            summary = f"[Earlier conversation summary]\n{resp.content}"
            logger.debug(f"LLM summarization succeeded: {len(summary)} chars")
            return summary

        except Exception as e:
            logger.warning(f"LLM summarization failed, using fallback: {e}")
            return None

    @staticmethod
    def _truncation_fallback(messages: list[ChatMessage]) -> str:
        """Simple truncation when LLM summarization is unavailable."""
        summary_parts = []
        for msg in messages:
            content = msg.content if isinstance(msg.content, str) else "[multimodal content]"
            if len(content) > 200:
                content = content[:200] + "..."
            summary_parts.append(f"[{msg.role}]: {content}")

        return (
            "[Context summary from earlier conversation]\n"
            + "\n".join(summary_parts[-10:])
        )

    async def close(self) -> None:
        tasks = [
            self.nvidia.close(),
            self.zhipu.close(),
            self.openrouter.close(),
        ]
        if self.groq is not None:
            tasks.append(self.groq.close())
        await asyncio.gather(*tasks)


class RouterError(Exception):
    """Raised when routing fails (all providers in chain failed)."""


def create_router_from_config(config_path: str = "config/config.yaml") -> ModelRouter:
    """Factory function to create a ModelRouter from config file.

    Requires environment variables for API keys:
    - NVIDIA_API_KEY
    - ZHIPU_API_KEY
    - OPENROUTER_API_KEY
    - GROQ_API_KEY (optional)
    """
    import os

    from dotenv import load_dotenv

    load_dotenv()

    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    models = config.get("models", {})

    nvidia = NvidiaClient(
        api_key=os.environ["NVIDIA_API_KEY"],
        base_url=os.environ.get("NVIDIA_BASE_URL", "https://integrate.api.nvidia.com/v1"),
        model=models.get("ceo", {}).get("primary", {}).get("model"),
        rpm_limit=models.get("ceo", {}).get("primary", {}).get("rpm_limit", 40),
    )

    zhipu = ZhipuClient(
        api_key=os.environ["ZHIPU_API_KEY"],
        vision_model=models.get("vision", {}).get("primary", {}).get("model"),
        image_model=models.get("image", {}).get("primary", {}).get("model"),
    )

    openrouter = OpenRouterClient(
        api_key=os.environ["OPENROUTER_API_KEY"],
        model=models.get("ceo", {}).get("tertiary", {}).get("model")
            or models.get("ceo", {}).get("backup", {}).get("model"),
    )

    groq_client = None
    groq_key = os.environ.get("GROQ_API_KEY", "")
    if groq_key:
        from clients.groq_chat_client import GroqChatClient
        groq_client = GroqChatClient(
            api_key=groq_key,
            model=models.get("ceo", {}).get("backup", {}).get("model"),
        )

    return ModelRouter(
        nvidia_client=nvidia,
        zhipu_client=zhipu,
        openrouter_client=openrouter,
        groq_client=groq_client,
        config=config,
    )
