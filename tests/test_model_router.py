"""Tests for ModelRouter select_model + task_type routing.

Run: pytest tests/test_model_router.py -v
"""

from __future__ import annotations

import os
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from clients.base_client import ChatMessage, ChatResponse
from clients.groq_chat_client import GroqChatClient
from clients.nvidia_client import NvidiaClient
from clients.openrouter_client import OpenRouterClient
from clients.zhipu_client import ZhipuClient
from core.model_router import ModelRole, ModelRouter


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


class TestSelectModel:
    def test_ceo_default_returns_env_or_fallback(self):
        """select_model('ceo') returns ZHIPU_CEO_MODEL env var or glm-4.6v."""
        router = _make_router()
        with patch.dict(os.environ, {"ZHIPU_CEO_MODEL": "glm-4.6v"}, clear=False):
            assert router.select_model("ceo") == "glm-4.6v"

    def test_ceo_default_fallback(self):
        """Without env var, select_model('ceo') falls back to glm-4.6v."""
        router = _make_router()
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("ZHIPU_CEO_MODEL", None)
            assert router.select_model("ceo") == "glm-4.6v"

    def test_template_returns_lite_model(self):
        """select_model('template') returns ZHIPU_LITE_MODEL."""
        router = _make_router()
        with patch.dict(os.environ, {"ZHIPU_LITE_MODEL": "glm-4.5-air"}, clear=False):
            assert router.select_model("template") == "glm-4.5-air"

    def test_format_returns_lite_model(self):
        """select_model('format') also returns lite model."""
        router = _make_router()
        with patch.dict(os.environ, {"ZHIPU_LITE_MODEL": "glm-4.5-air"}, clear=False):
            assert router.select_model("format") == "glm-4.5-air"

    def test_cron_message_returns_lite_model(self):
        """select_model('cron_message') also returns lite model."""
        router = _make_router()
        with patch.dict(os.environ, {"ZHIPU_LITE_MODEL": "glm-4.5-air"}, clear=False):
            assert router.select_model("cron_message") == "glm-4.5-air"

    def test_custom_env_overrides(self):
        """Custom env vars override defaults."""
        router = _make_router()
        with patch.dict(os.environ, {
            "ZHIPU_CEO_MODEL": "glm-5-ultra",
            "ZHIPU_LITE_MODEL": "glm-3-mini",
        }, clear=False):
            assert router.select_model("ceo") == "glm-5-ultra"
            assert router.select_model("template") == "glm-3-mini"


class TestChainTaskType:
    def test_chain_default_uses_ceo_model(self):
        """Default chain uses CEO model (glm-4.6v)."""
        router = _make_router()
        with patch.dict(os.environ, {"ZHIPU_CEO_MODEL": "glm-4.6v"}, clear=False):
            chain = router._get_chain_for_role(ModelRole.CEO)
            zhipu_entry = chain[0]
            assert zhipu_entry[0] == "zhipu_ceo"
            assert zhipu_entry[1]["model"] == "glm-4.6v"

    def test_chain_template_uses_lite_model(self):
        """Template task_type chain uses lite model."""
        router = _make_router()
        with patch.dict(os.environ, {"ZHIPU_LITE_MODEL": "glm-4.5-air"}, clear=False):
            chain = router._get_chain_for_role(ModelRole.CEO, task_type="template")
            zhipu_entry = chain[0]
            assert zhipu_entry[1]["model"] == "glm-4.5-air"

    def test_vision_chain_unaffected(self):
        """Vision chain should not be affected by task_type."""
        router = _make_router()
        chain = router._get_chain_for_role(ModelRole.VISION)
        assert chain[0][0] == "zhipu"
        assert "model" not in chain[0][1]  # vision uses client default


class TestChatTaskType:
    @pytest.mark.asyncio
    async def test_chat_passes_task_type_template(self):
        """chat(task_type='template') should use lite model in chain."""
        router = _make_router()
        router.zhipu.chat = AsyncMock(
            return_value=ChatResponse(content="ok", model="glm-4.5-air")
        )

        with patch.dict(os.environ, {"ZHIPU_LITE_MODEL": "glm-4.5-air"}, clear=False):
            result = await router.chat(
                [ChatMessage(role="user", content="classify this")],
                role=ModelRole.CEO,
                task_type="template",
                max_tokens=20,
            )
            assert result.content == "ok"
            # Verify the model kwarg passed to client
            call_kwargs = router.zhipu.chat.call_args
            assert call_kwargs[1]["model"] == "glm-4.5-air"

    @pytest.mark.asyncio
    async def test_chat_default_task_type_uses_ceo(self):
        """chat() without task_type should default to CEO model."""
        router = _make_router()
        router.zhipu.chat = AsyncMock(
            return_value=ChatResponse(content="ok", model="glm-4.6v")
        )

        with patch.dict(os.environ, {"ZHIPU_CEO_MODEL": "glm-4.6v"}, clear=False):
            result = await router.chat(
                [ChatMessage(role="user", content="hello")],
                role=ModelRole.CEO,
            )
            assert result.content == "ok"
            call_kwargs = router.zhipu.chat.call_args
            assert call_kwargs[1]["model"] == "glm-4.6v"
