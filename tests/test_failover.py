"""Tests for Failover: hot-switch, context bridging, recovery probing.

Run: pytest tests/test_failover.py -v
"""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock

import pytest

from clients.base_client import ChatMessage, ChatResponse
from clients.groq_chat_client import GroqChatClient
from clients.nvidia_client import NvidiaClient, RateLimitExceeded
from clients.openrouter_client import OpenRouterClient
from clients.zhipu_client import ZhipuClient
from core.model_router import FailoverEvent, ModelRole, ModelRouter, ProviderStatus, RouterError


def _make_router(**overrides) -> ModelRouter:
    nvidia = MagicMock(spec=NvidiaClient)
    zhipu = MagicMock(spec=ZhipuClient)
    openrouter = MagicMock(spec=OpenRouterClient)
    groq = MagicMock(spec=GroqChatClient)
    config = overrides.pop("config", {})
    return ModelRouter(
        nvidia_client=overrides.get("nvidia", nvidia),
        zhipu_client=overrides.get("zhipu", zhipu),
        openrouter_client=overrides.get("openrouter", openrouter),
        groq_client=overrides.get("groq", groq),
        config=config,
    )


# ── Failover Trigger: 429 ──────────────────────────────────────


class TestFailoverOn429:
    @pytest.mark.asyncio
    async def test_single_429_no_failover(self):
        """One 429 should NOT trigger failover (threshold is 2)."""
        router = _make_router()
        call_count = 0

        async def flaky_chat(*a, **kw):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise Exception("HTTP 429 Too Many Requests")
            return ChatResponse(content="ok", model="glm-4.5-air")

        router.zhipu.chat = flaky_chat
        router.groq.chat = AsyncMock(
            return_value=ChatResponse(content="backup", model="llama")
        )

        # First call — 429 recorded, but router still tries groq as next in chain
        result = await router.chat([ChatMessage(role="user", content="hi")], role=ModelRole.CEO)
        # Since _handle_failure records 1 consecutive 429 < threshold(2), zhipu_ceo is NOT marked DOWN yet
        # but it falls to groq for this call
        assert result.content == "backup"

    @pytest.mark.asyncio
    async def test_consecutive_429_triggers_failover(self):
        """Two consecutive 429s should mark provider DOWN and failover."""
        router = _make_router()
        router.zhipu.chat = AsyncMock(side_effect=Exception("HTTP 429"))
        router.groq.chat = AsyncMock(
            return_value=ChatResponse(content="backup", model="llama")
        )

        # First call — 429 #1
        await router.chat([ChatMessage(role="user", content="hi")], role=ModelRole.CEO)
        # Second call — 429 #2, should mark down
        await router.chat([ChatMessage(role="user", content="hi")], role=ModelRole.CEO)

        assert router._provider_status["zhipu_ceo"] == ProviderStatus.DOWN

    @pytest.mark.asyncio
    async def test_rate_limit_exceeded_immediate_failover(self):
        """RateLimitExceeded (silent mode) should immediately failover."""
        router = _make_router()
        router.zhipu.chat = AsyncMock(side_effect=RateLimitExceeded("silent mode"))
        router.groq.chat = AsyncMock(
            return_value=ChatResponse(content="backup", model="llama")
        )

        result = await router.chat([ChatMessage(role="user", content="hi")], role=ModelRole.CEO)
        assert result.content == "backup"
        assert router._provider_status["zhipu_ceo"] == ProviderStatus.DOWN


# ── Failover Trigger: 5xx ──────────────────────────────────────


class TestFailoverOn5xx:
    @pytest.mark.asyncio
    async def test_single_5xx_triggers_failover(self):
        """One 5xx should trigger failover (threshold is 1)."""
        router = _make_router()
        router.zhipu.chat = AsyncMock(side_effect=Exception("HTTP 500 Internal Server Error"))
        router.groq.chat = AsyncMock(
            return_value=ChatResponse(content="backup", model="llama")
        )

        result = await router.chat([ChatMessage(role="user", content="hi")], role=ModelRole.CEO)
        assert result.content == "backup"
        assert router._provider_status["zhipu_ceo"] == ProviderStatus.DOWN


# ── Failover Event Logging ──────────────────────────────────────


class TestFailoverEvents:
    @pytest.mark.asyncio
    async def test_failover_event_recorded(self):
        router = _make_router()
        router.zhipu.chat = AsyncMock(side_effect=RateLimitExceeded("silent"))
        router.groq.chat = AsyncMock(
            return_value=ChatResponse(content="backup", model="llama")
        )

        await router.chat([ChatMessage(role="user", content="hi")], role=ModelRole.CEO)

        events = router.failover_history
        assert len(events) == 1
        assert events[0].from_provider == "zhipu_ceo"
        assert events[0].to_provider == "groq"
        assert events[0].role == "ceo"

    @pytest.mark.asyncio
    async def test_double_failover_events(self):
        """zhipu → groq → openrouter should produce 2 failover events."""
        router = _make_router()
        router.zhipu.chat = AsyncMock(side_effect=Exception("zhipu down"))
        router.groq.chat = AsyncMock(side_effect=Exception("groq down"))
        router.openrouter.chat = AsyncMock(
            return_value=ChatResponse(content="openrouter ok", model="deepseek")
        )

        await router.chat([ChatMessage(role="user", content="hi")], role=ModelRole.CEO)

        events = router.failover_history
        assert len(events) == 2
        assert events[0].from_provider == "zhipu_ceo"
        assert events[0].to_provider == "groq"
        assert events[1].from_provider == "groq"
        assert events[1].to_provider == "openrouter"


# ── Context Bridging ───────────────────────────────────────────


class TestContextBridging:
    @pytest.mark.asyncio
    async def test_short_conversation_no_bridging(self):
        router = _make_router()
        router._keep_recent_turns = 8
        msgs = [ChatMessage(role="user", content=f"msg {i}") for i in range(5)]

        bridged = await router._bridge_context(msgs, ModelRole.CEO)
        assert len(bridged) == 5  # Unchanged

    @pytest.mark.asyncio
    async def test_long_conversation_truncated(self):
        router = _make_router()
        router._keep_recent_turns = 3
        # Make LLM summarization fail to test fallback
        router.openrouter.chat = AsyncMock(side_effect=Exception("unavailable"))

        msgs = [ChatMessage(role="user", content=f"message number {i}") for i in range(20)]
        bridged = await router._bridge_context(msgs, ModelRole.CEO)

        # Should be: 1 summary + 3 recent = 4
        assert len(bridged) == 4
        assert bridged[0].role == "system"
        assert "Context summary" in bridged[0].content
        # Recent 3 should be preserved verbatim
        assert bridged[1].content == "message number 17"
        assert bridged[2].content == "message number 18"
        assert bridged[3].content == "message number 19"

    @pytest.mark.asyncio
    async def test_llm_summarization_used_when_available(self):
        router = _make_router()
        router._keep_recent_turns = 3
        router.openrouter.chat = AsyncMock(
            return_value=ChatResponse(content="User discussed weather and schedule.", model="deepseek")
        )

        msgs = [ChatMessage(role="user", content=f"msg {i}") for i in range(10)]
        bridged = await router._bridge_context(msgs, ModelRole.CEO)

        assert len(bridged) == 4
        assert "weather and schedule" in bridged[0].content
        # Verify summarizer was called
        router.openrouter.chat.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_multimodal_messages_handled(self):
        router = _make_router()
        router._keep_recent_turns = 2
        router.openrouter.chat = AsyncMock(side_effect=Exception("fail"))

        msgs = [
            ChatMessage(role="user", content=[{"type": "image_url", "image_url": {"url": "http://img"}}]),
            ChatMessage(role="assistant", content="I see an image"),
            ChatMessage(role="user", content="What's in it?"),
            ChatMessage(role="assistant", content="A cat"),
        ]
        bridged = await router._bridge_context(msgs, ModelRole.CEO)
        assert "[multimodal content]" in bridged[0].content


# ── Recovery Probing ───────────────────────────────────────────


class TestRecoveryProbing:
    @pytest.mark.asyncio
    async def test_recovery_after_required_healthy_checks(self):
        router = _make_router()
        router._provider_status["zhipu_ceo"] = ProviderStatus.DOWN
        router._recovery_interval = 0  # Check immediately
        router._healthy_checks_required = 3
        router.zhipu.health_check = AsyncMock(return_value=True)

        # Need 3 healthy checks
        r1 = await router.probe_recovery()
        assert r1.get("zhipu_ceo") == "healing (1/3)"

        r2 = await router.probe_recovery()
        assert r2.get("zhipu_ceo") == "healing (2/3)"

        r3 = await router.probe_recovery()
        assert r3.get("zhipu_ceo") == "recovered"
        assert router._provider_status["zhipu_ceo"] == ProviderStatus.HEALTHY

    @pytest.mark.asyncio
    async def test_failed_health_check_resets_counter(self):
        router = _make_router()
        router._provider_status["zhipu_ceo"] = ProviderStatus.DOWN
        router._recovery_interval = 0
        router._healthy_checks_required = 2

        router.zhipu.health_check = AsyncMock(return_value=True)
        await router.probe_recovery()  # healing 1/2

        router.zhipu.health_check = AsyncMock(return_value=False)
        result = await router.probe_recovery()  # fails, reset
        assert result.get("zhipu_ceo") == "still_down"
        assert router._recovery_healthy_count.get("zhipu_ceo") == 0

    @pytest.mark.asyncio
    async def test_recovery_respects_interval(self):
        router = _make_router()
        router._provider_status["zhipu_ceo"] = ProviderStatus.DOWN
        router._recovery_interval = 9999  # Very long interval
        router._healthy_checks_required = 1
        router.zhipu.health_check = AsyncMock(return_value=True)

        # First check goes through
        await router.probe_recovery()
        # Second check should be skipped (too soon)
        result = await router.probe_recovery()
        assert result == {}  # No check performed

    @pytest.mark.asyncio
    async def test_groq_recovery(self):
        router = _make_router()
        router._provider_status["groq"] = ProviderStatus.DOWN
        router._recovery_interval = 0
        router._healthy_checks_required = 1
        router.groq.health_check = AsyncMock(return_value=True)

        results = await router.probe_recovery()
        assert results.get("groq") == "recovered"
        assert router._provider_status["groq"] == ProviderStatus.HEALTHY


# ── Full Failover Scenario ──────────────────────────────────────


class TestFullFailoverScenario:
    @pytest.mark.asyncio
    async def test_zhipu_ceo_down_switch_recover(self):
        """Simulate: zhipu_ceo goes down → switch to groq → recover zhipu_ceo."""
        router = _make_router()
        router._recovery_interval = 0
        router._healthy_checks_required = 1

        # Phase 1: zhipu works
        router.zhipu.chat = AsyncMock(
            return_value=ChatResponse(content="zhipu ok", model="glm-4.5-air")
        )
        result = await router.chat([ChatMessage(role="user", content="hello")], role=ModelRole.CEO)
        assert result.content == "zhipu ok"

        # Phase 2: zhipu goes down (silent mode)
        router.zhipu.chat = AsyncMock(side_effect=RateLimitExceeded("silent"))
        router.groq.chat = AsyncMock(
            return_value=ChatResponse(content="groq fallback", model="llama")
        )
        result = await router.chat([ChatMessage(role="user", content="hi again")], role=ModelRole.CEO)
        assert result.content == "groq fallback"
        assert router._provider_status["zhipu_ceo"] == ProviderStatus.DOWN

        # Phase 3: Recovery
        router.zhipu.health_check = AsyncMock(return_value=True)
        recovery = await router.probe_recovery()
        assert recovery.get("zhipu_ceo") == "recovered"

        # Phase 4: zhipu works again
        router.zhipu.chat = AsyncMock(
            return_value=ChatResponse(content="zhipu back", model="glm-4.5-air")
        )
        result = await router.chat([ChatMessage(role="user", content="welcome back")], role=ModelRole.CEO)
        assert result.content == "zhipu back"

    @pytest.mark.asyncio
    async def test_three_provider_cascade(self):
        """zhipu_ceo down → groq down → openrouter handles it."""
        router = _make_router()

        router.zhipu.chat = AsyncMock(side_effect=Exception("zhipu error"))
        router.groq.chat = AsyncMock(side_effect=Exception("groq error"))
        router.openrouter.chat = AsyncMock(
            return_value=ChatResponse(content="last resort", model="deepseek")
        )

        result = await router.chat([ChatMessage(role="user", content="help")], role=ModelRole.CEO)
        assert result.content == "last resort"

    @pytest.mark.asyncio
    async def test_zhipu_ceo_independent_from_zhipu_vision(self):
        """zhipu_ceo DOWN should not affect vision_analyze (separate status)."""
        router = _make_router()

        # Mark zhipu_ceo down
        router._provider_status["zhipu_ceo"] = ProviderStatus.DOWN
        # zhipu (vision) should still be healthy
        assert router._provider_status["zhipu"] == ProviderStatus.HEALTHY

        # Vision should work via zhipu
        router.zhipu.vision_analyze = AsyncMock(
            return_value=ChatResponse(content="I see a cat", model="glm-4v-flash")
        )
        result = await router.vision_analyze("http://img.jpg", "describe this")
        assert result.content == "I see a cat"

        # CEO should skip zhipu_ceo and go to groq
        router.groq.chat = AsyncMock(
            return_value=ChatResponse(content="groq handles ceo", model="llama")
        )
        result = await router.chat([ChatMessage(role="user", content="hi")], role=ModelRole.CEO)
        assert result.content == "groq handles ceo"

    @pytest.mark.asyncio
    async def test_vision_failover_to_gemini(self):
        """Vision failover: GLM-4V down → switch to Gemini via OpenRouter."""
        router = _make_router()
        router.zhipu.vision_analyze = AsyncMock(side_effect=Exception("zhipu down"))
        router.openrouter.chat = AsyncMock(
            return_value=ChatResponse(content="gemini vision result", model="gemini")
        )

        result = await router.vision_analyze("http://img.jpg", "describe this")
        assert result.content == "gemini vision result"
        assert router._provider_status["zhipu"] == ProviderStatus.DOWN
