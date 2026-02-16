"""Tests for KnowledgeWorker — LLM + memory fallback."""

from __future__ import annotations

import pytest
from unittest.mock import AsyncMock, MagicMock

from clients.base_client import ChatResponse
from workers.knowledge_worker import KnowledgeWorker


@pytest.fixture
def mock_router():
    router = AsyncMock()
    router.chat.return_value = ChatResponse(
        content="根據我的知識，台北今天大約15度", model="test", usage={}
    )
    return router


@pytest.fixture
def mock_memory_search():
    search = MagicMock()
    search.search.return_value = [
        {"text": "台北通常冬天10-15度", "source": "memory.md", "score": 1.5},
    ]
    return search


class TestKnowledgeWorker:
    @pytest.mark.asyncio
    async def test_execute_with_memory_and_router(self, mock_router, mock_memory_search):
        worker = KnowledgeWorker(
            model_router=mock_router, memory_search=mock_memory_search,
        )
        result = await worker.execute("台北天氣")
        assert result["source"] == "knowledge"
        assert result["worker"] == "knowledge"
        assert "result" in result
        mock_memory_search.search.assert_called_once()
        mock_router.chat.assert_called_once()

    @pytest.mark.asyncio
    async def test_execute_without_memory(self, mock_router):
        """Should work with just the router, no memory search."""
        worker = KnowledgeWorker(model_router=mock_router)
        result = await worker.execute("台北天氣")
        assert result["source"] == "knowledge"
        assert "result" in result

    @pytest.mark.asyncio
    async def test_execute_with_failed_attempts(self, mock_router):
        """Failed attempts should be included in the prompt."""
        worker = KnowledgeWorker(model_router=mock_router)
        result = await worker.execute(
            "台北天氣",
            failed_attempts=[
                {"worker": "browser", "error": "Connection refused"},
            ],
        )
        assert result["source"] == "knowledge"
        # Check that prompt includes failed attempts info
        call_args = mock_router.chat.call_args[0][0]
        prompt_text = call_args[0].content
        assert "失敗" in prompt_text
        assert "browser" in prompt_text

    @pytest.mark.asyncio
    async def test_execute_no_router(self):
        """Without router, should return error."""
        worker = KnowledgeWorker()
        result = await worker.execute("test")
        assert "error" in result
        assert result["worker"] == "knowledge"

    @pytest.mark.asyncio
    async def test_execute_router_error(self, mock_router):
        """If router.chat raises, should return error dict."""
        mock_router.chat.side_effect = Exception("LLM down")
        worker = KnowledgeWorker(model_router=mock_router)
        result = await worker.execute("test")
        assert "error" in result
        assert "LLM down" in result["error"]
