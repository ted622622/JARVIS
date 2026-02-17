"""Tests for API clients and ModelRouter.

Run: pytest tests/test_api_clients.py -v
For live tests (requires API keys): pytest tests/test_api_clients.py -v -m live
"""

from __future__ import annotations

import asyncio
import time
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import pytest_asyncio

from clients.base_client import ChatMessage, ChatResponse, RateLimitTracker, TokenBucket
from clients.groq_chat_client import GroqChatClient
from clients.nvidia_client import NvidiaClient, RateLimitExceeded
from clients.openrouter_client import OpenRouterClient
from clients.zhipu_client import ZhipuClient
from core.model_router import ModelRole, ModelRouter, ProviderStatus, RouterError


# ── Token Bucket Tests ──────────────────────────────────────────


class TestTokenBucket:
    @pytest.mark.asyncio
    async def test_immediate_acquire_when_full(self):
        bucket = TokenBucket(rate=1.0, capacity=5)
        wait = await bucket.acquire()
        assert wait == 0.0

    @pytest.mark.asyncio
    async def test_bucket_depletes(self):
        bucket = TokenBucket(rate=10.0, capacity=2)
        # Drain the bucket
        await bucket.acquire()
        await bucket.acquire()
        # Third acquire should wait
        start = time.monotonic()
        await bucket.acquire()
        elapsed = time.monotonic() - start
        assert elapsed >= 0.05  # Should have waited ~0.1s at rate=10

    @pytest.mark.asyncio
    async def test_bucket_refills(self):
        bucket = TokenBucket(rate=100.0, capacity=1)
        await bucket.acquire()  # Drain
        await asyncio.sleep(0.02)  # Wait for refill
        wait = await bucket.acquire()
        assert wait == 0.0  # Should be refilled


# ── Rate Limit Tracker Tests ────────────────────────────────────


class TestRateLimitTracker:
    def test_no_silent_under_threshold(self):
        tracker = RateLimitTracker(threshold_per_hour=5, cooldown_minutes=1)
        for _ in range(4):
            result = tracker.record_429()
            assert result is False
        assert not tracker.is_silent

    def test_enters_silent_at_threshold(self):
        tracker = RateLimitTracker(threshold_per_hour=3, cooldown_minutes=1)
        tracker.record_429()
        tracker.record_429()
        entered = tracker.record_429()
        assert entered is True
        assert tracker.is_silent


# ── NvidiaClient Tests ──────────────────────────────────────────


class TestNvidiaClient:
    def _make_client(self) -> NvidiaClient:
        return NvidiaClient(api_key="test-key", rpm_limit=40)

    @pytest.mark.asyncio
    async def test_silent_mode_raises(self):
        client = self._make_client()
        client.rate_tracker.silent_until = time.monotonic() + 9999
        with pytest.raises(RateLimitExceeded, match="silent mode"):
            await client.chat([ChatMessage(role="user", content="test")])

    @pytest.mark.asyncio
    async def test_format_message(self):
        msg = ChatMessage(role="user", content="hello")
        result = NvidiaClient._format_msg(msg)
        assert result == {"role": "user", "content": "hello"}

    @pytest.mark.asyncio
    async def test_successful_chat(self):
        client = self._make_client()
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "choices": [{"message": {"content": "pong"}}],
            "model": "kimi-k2.5",
            "usage": {"prompt_tokens": 5, "completion_tokens": 1, "total_tokens": 6},
        }

        with patch.object(client, "_get_client") as mock_get:
            mock_http = AsyncMock()
            mock_http.post = AsyncMock(return_value=mock_response)
            mock_get.return_value = mock_http

            result = await client.chat([ChatMessage(role="user", content="ping")])
            assert result.content == "pong"
            assert result.model == "kimi-k2.5"

        await client.close()


# ── OpenRouterClient Tests ──────────────────────────────────────


class TestOpenRouterClient:
    @pytest.mark.asyncio
    async def test_successful_chat(self):
        client = OpenRouterClient(api_key="test-key")
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "choices": [{"message": {"content": "hello"}}],
            "model": "deepseek/deepseek-chat",
            "usage": {"prompt_tokens": 3, "completion_tokens": 1},
        }

        with patch.object(client, "_get_client") as mock_get:
            mock_http = AsyncMock()
            mock_http.post = AsyncMock(return_value=mock_response)
            mock_get.return_value = mock_http

            result = await client.chat([ChatMessage(role="user", content="hi")])
            assert result.content == "hello"

        await client.close()

    @pytest.mark.asyncio
    async def test_headers_include_referer(self):
        client = OpenRouterClient(api_key="test-key")
        headers = client._build_headers()
        assert "HTTP-Referer" in headers
        assert "X-Title" in headers
        await client.close()


# ── ModelRouter Tests ───────────────────────────────────────────


class TestModelRouter:
    def _make_router(self, with_groq=True) -> ModelRouter:
        nvidia = MagicMock(spec=NvidiaClient)
        zhipu = MagicMock(spec=ZhipuClient)
        openrouter = MagicMock(spec=OpenRouterClient)
        groq = MagicMock(spec=GroqChatClient) if with_groq else None
        return ModelRouter(nvidia, zhipu, openrouter, groq_client=groq, config={})

    @pytest.mark.asyncio
    async def test_routes_ceo_to_zhipu(self):
        """CEO chain primary is zhipu (glm-4.6v by default)."""
        router = self._make_router()
        expected = ChatResponse(content="response", model="glm-4.6v")
        router.zhipu.chat = AsyncMock(return_value=expected)

        result = await router.chat(
            [ChatMessage(role="user", content="test")],
            role=ModelRole.CEO,
        )
        assert result.content == "response"
        router.zhipu.chat.assert_awaited_once()
        # model override should include the CEO model from env (default: glm-4.6v)
        call_kwargs = router.zhipu.chat.call_args
        assert call_kwargs[1].get("model") == router.select_model("ceo")

    @pytest.mark.asyncio
    async def test_ceo_zhipu_down_falls_to_groq(self):
        """When zhipu_ceo fails, should fallover to groq."""
        router = self._make_router()
        router.zhipu.chat = AsyncMock(side_effect=Exception("zhipu down"))
        router.groq.chat = AsyncMock(
            return_value=ChatResponse(content="groq response", model="llama-3.3-70b")
        )

        result = await router.chat(
            [ChatMessage(role="user", content="test")],
            role=ModelRole.CEO,
        )
        assert result.content == "groq response"
        router.groq.chat.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_ceo_zhipu_groq_down_falls_to_openrouter(self):
        """When both zhipu_ceo and groq fail, should fallover to openrouter."""
        router = self._make_router()
        router.zhipu.chat = AsyncMock(side_effect=Exception("zhipu down"))
        router.groq.chat = AsyncMock(side_effect=Exception("groq down"))
        router.openrouter.chat = AsyncMock(
            return_value=ChatResponse(content="openrouter response", model="deepseek")
        )

        result = await router.chat(
            [ChatMessage(role="user", content="test")],
            role=ModelRole.CEO,
        )
        assert result.content == "openrouter response"
        router.openrouter.chat.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_all_three_fail_raises_router_error(self):
        """All 3 CEO providers fail → RouterError."""
        router = self._make_router()
        router.zhipu.chat = AsyncMock(side_effect=Exception("zhipu down"))
        router.groq.chat = AsyncMock(side_effect=Exception("groq down"))
        router.openrouter.chat = AsyncMock(side_effect=Exception("openrouter down"))

        with pytest.raises(RouterError, match="All providers failed"):
            await router.chat(
                [ChatMessage(role="user", content="test")],
                role=ModelRole.CEO,
            )

    @pytest.mark.asyncio
    async def test_rate_limit_triggers_failover(self):
        router = self._make_router()
        router.zhipu.chat = AsyncMock(side_effect=RateLimitExceeded("silent mode"))
        router.groq.chat = AsyncMock(
            return_value=ChatResponse(content="fallback", model="llama")
        )

        result = await router.chat(
            [ChatMessage(role="user", content="test")],
            role=ModelRole.CEO,
        )
        assert result.content == "fallback"

    @pytest.mark.asyncio
    async def test_zhipu_ceo_down_does_not_affect_vision(self):
        """zhipu_ceo DOWN should not affect zhipu for vision."""
        router = self._make_router()
        # Mark zhipu_ceo down
        router._provider_status["zhipu_ceo"] = ProviderStatus.DOWN
        # zhipu for vision should still be healthy
        assert router._provider_status["zhipu"] == ProviderStatus.HEALTHY

        expected = ChatResponse(content="vision ok", model="glm-4v-flash")
        router.zhipu.chat = AsyncMock(return_value=expected)

        result = await router.chat(
            [ChatMessage(role="user", content="describe")],
            role=ModelRole.VISION,
        )
        assert result.content == "vision ok"

    @pytest.mark.asyncio
    async def test_context_bridging_truncates(self):
        router = self._make_router()
        router._keep_recent_turns = 3

        # Create 10 messages
        messages = [ChatMessage(role="user", content=f"msg {i}") for i in range(10)]

        # _bridge_context is now async; make LLM summarization fail to test fallback
        router.openrouter.chat = AsyncMock(side_effect=Exception("unavailable"))
        bridged = await router._bridge_context(messages, ModelRole.CEO)
        # Should have: 1 summary system msg + 3 recent = 4
        assert len(bridged) == 4
        assert bridged[0].role == "system"
        assert "Context summary" in bridged[0].content

    @pytest.mark.asyncio
    async def test_context_bridging_passthrough_short(self):
        router = self._make_router()
        router._keep_recent_turns = 8

        messages = [ChatMessage(role="user", content=f"msg {i}") for i in range(5)]
        bridged = await router._bridge_context(messages, ModelRole.CEO)
        assert len(bridged) == 5  # No bridging needed

    @pytest.mark.asyncio
    async def test_health_check_all(self):
        router = self._make_router()
        router.nvidia.health_check = AsyncMock(return_value=True)
        router.zhipu.health_check = AsyncMock(return_value=False)
        router.openrouter.health_check = AsyncMock(return_value=True)
        router.groq.health_check = AsyncMock(return_value=True)

        health = await router.health_check_all()
        assert health == {"nvidia": True, "zhipu": False, "openrouter": True, "groq": True}

    @pytest.mark.asyncio
    async def test_health_check_all_without_groq(self):
        router = self._make_router(with_groq=False)
        router.nvidia.health_check = AsyncMock(return_value=True)
        router.zhipu.health_check = AsyncMock(return_value=True)
        router.openrouter.health_check = AsyncMock(return_value=True)

        health = await router.health_check_all()
        assert health == {"nvidia": True, "zhipu": True, "openrouter": True}
        assert "groq" not in health

    @pytest.mark.asyncio
    async def test_recovery_probe(self):
        router = self._make_router()
        router._provider_status["zhipu_ceo"] = ProviderStatus.DOWN
        router._recovery_interval = 0  # Check immediately
        router._healthy_checks_required = 1
        router.zhipu.health_check = AsyncMock(return_value=True)

        results = await router.probe_recovery()
        assert results.get("zhipu_ceo") == "recovered"
        assert router._provider_status["zhipu_ceo"] == ProviderStatus.HEALTHY

    @pytest.mark.asyncio
    async def test_ceo_chain_without_groq(self):
        """CEO chain without groq: zhipu → openrouter (2 providers)."""
        router = self._make_router(with_groq=False)
        router.zhipu.chat = AsyncMock(side_effect=Exception("zhipu down"))
        router.openrouter.chat = AsyncMock(
            return_value=ChatResponse(content="openrouter", model="deepseek")
        )

        result = await router.chat(
            [ChatMessage(role="user", content="test")],
            role=ModelRole.CEO,
        )
        assert result.content == "openrouter"


# ── Live Integration Tests (require API keys) ──────────────────


@pytest.mark.live
class TestLiveIntegration:
    """Run with: pytest tests/test_api_clients.py -v -m live

    Requires .env file with valid API keys.
    """

    @pytest.fixture
    def router(self):
        from core.model_router import create_router_from_config

        return create_router_from_config()

    @pytest.mark.asyncio
    async def test_nvidia_ping(self, router):
        response = await router.chat(
            [ChatMessage(role="user", content="回覆 pong")],
            role=ModelRole.CEO,
            max_tokens=10,
        )
        assert len(response.content) > 0
        await router.close()

    @pytest.mark.asyncio
    async def test_openrouter_ping(self, router):
        response = await router.openrouter.chat(
            [ChatMessage(role="user", content="Reply with pong")],
            max_tokens=10,
        )
        assert len(response.content) > 0
        await router.close()

    @pytest.mark.asyncio
    async def test_zhipu_vision_ping(self, router):
        response = await router.zhipu.chat(
            [ChatMessage(role="user", content="回覆 pong")],
            max_tokens=10,
        )
        assert len(response.content) > 0
        await router.close()

    @pytest.mark.asyncio
    async def test_full_router_chat(self, router):
        """Validates: ModelRouter.chat("你好") routes to primary CEO and returns result."""
        response = await router.chat(
            [ChatMessage(role="user", content="你好")],
            role=ModelRole.CEO,
        )
        assert len(response.content) > 0
        assert response.model is not None
        await router.close()
