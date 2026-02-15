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
    def _make_router(self) -> ModelRouter:
        nvidia = MagicMock(spec=NvidiaClient)
        zhipu = MagicMock(spec=ZhipuClient)
        openrouter = MagicMock(spec=OpenRouterClient)
        return ModelRouter(nvidia, zhipu, openrouter, config={})

    @pytest.mark.asyncio
    async def test_routes_ceo_to_nvidia(self):
        router = self._make_router()
        expected = ChatResponse(content="response", model="kimi")
        router.nvidia.chat = AsyncMock(return_value=expected)

        result = await router.chat(
            [ChatMessage(role="user", content="test")],
            role=ModelRole.CEO,
        )
        assert result.content == "response"
        router.nvidia.chat.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_failover_to_openrouter(self):
        router = self._make_router()
        router.nvidia.chat = AsyncMock(side_effect=Exception("connection refused"))
        router.openrouter.chat = AsyncMock(
            return_value=ChatResponse(content="backup response", model="deepseek")
        )

        result = await router.chat(
            [ChatMessage(role="user", content="test")],
            role=ModelRole.CEO,
        )
        assert result.content == "backup response"
        router.openrouter.chat.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_both_fail_raises_router_error(self):
        router = self._make_router()
        router.nvidia.chat = AsyncMock(side_effect=Exception("nvidia down"))
        router.openrouter.chat = AsyncMock(side_effect=Exception("openrouter down"))

        with pytest.raises(RouterError, match="Both primary and backup failed"):
            await router.chat(
                [ChatMessage(role="user", content="test")],
                role=ModelRole.CEO,
            )

    @pytest.mark.asyncio
    async def test_rate_limit_triggers_failover(self):
        router = self._make_router()
        router.nvidia.chat = AsyncMock(side_effect=RateLimitExceeded("silent mode"))
        router.openrouter.chat = AsyncMock(
            return_value=ChatResponse(content="fallback", model="deepseek")
        )

        result = await router.chat(
            [ChatMessage(role="user", content="test")],
            role=ModelRole.CEO,
        )
        assert result.content == "fallback"

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

        health = await router.health_check_all()
        assert health == {"nvidia": True, "zhipu": False, "openrouter": True}

    @pytest.mark.asyncio
    async def test_recovery_probe(self):
        router = self._make_router()
        router._provider_status["nvidia"] = ProviderStatus.DOWN
        router._recovery_interval = 0  # Check immediately
        router._healthy_checks_required = 1
        router.nvidia.health_check = AsyncMock(return_value=True)

        results = await router.probe_recovery()
        assert results.get("nvidia") == "recovered"
        assert router._provider_status["nvidia"] == ProviderStatus.HEALTHY


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
        """Validates: ModelRouter.chat("你好") routes to Kimi K2.5 and returns result."""
        response = await router.chat(
            [ChatMessage(role="user", content="你好")],
            role=ModelRole.CEO,
        )
        assert len(response.content) > 0
        assert response.model is not None
        await router.close()
