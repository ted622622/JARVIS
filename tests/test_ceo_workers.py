"""Tests for CEO Agent and Workers."""

from __future__ import annotations

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from clients.base_client import ChatMessage, ChatResponse
from core.ceo_agent import CEOAgent
from workers.code_worker import CodeWorker
from workers.interpreter_worker import InterpreterWorker
from workers.browser_worker import BrowserWorker
from workers.vision_worker import VisionWorker
from workers.selfie_worker import SelfieWorker


# ── Fixtures ────────────────────────────────────────────────────


@pytest.fixture
def mock_router():
    router = AsyncMock()
    router.chat.return_value = ChatResponse(
        content="Test response", model="test-model", usage={}
    )
    return router


@pytest.fixture
def mock_soul():
    soul = MagicMock()
    soul.is_loaded = True
    soul.build_system_prompt.return_value = "你是 J.A.R.V.I.S."
    return soul


@pytest.fixture
def mock_memos():
    memos = AsyncMock()
    memos.working_memory = AsyncMock()
    memos.get_recent_conversation = AsyncMock(return_value=[])
    memos.log_conversation = AsyncMock()
    return memos


@pytest.fixture
def mock_emotion():
    emotion = AsyncMock()
    emotion.classify.return_value = "normal"
    return emotion


@pytest.fixture
def mock_registry():
    reg = MagicMock()
    reg.list_all.return_value = []
    reg.get.return_value = None
    return reg


@pytest.fixture
def ceo(mock_router, mock_soul, mock_emotion, mock_memos, mock_registry):
    return CEOAgent(
        model_router=mock_router,
        soul=mock_soul,
        emotion_classifier=mock_emotion,
        memos=mock_memos,
        skill_registry=mock_registry,
    )


# ── CEO Agent Tests ────────────────────────────────────────────


class TestCEOAgent:
    @pytest.mark.asyncio
    async def test_handle_message_basic(self, ceo, mock_router):
        result = await ceo.handle_message("你好")
        assert result == "Test response"
        assert mock_router.chat.called

    @pytest.mark.asyncio
    async def test_handle_message_with_emotion(self, ceo, mock_emotion, mock_router):
        mock_emotion.classify.return_value = "tired"
        mock_router.chat.return_value = ChatResponse(
            content="辛苦了，要不要休息一下？", model="test", usage={}
        )
        result = await ceo.handle_message("我好累")
        mock_emotion.classify.assert_called_once_with("我好累")
        assert "休息" in result

    @pytest.mark.asyncio
    async def test_handle_message_stores_conversation(self, ceo, mock_memos):
        await ceo.handle_message("測試")
        assert mock_memos.log_conversation.call_count == 2  # user + assistant

    @pytest.mark.asyncio
    async def test_handle_message_with_persona(self, ceo, mock_soul):
        await ceo.handle_message("你好", persona="clawra")
        mock_soul.build_system_prompt.assert_called()
        call_args = mock_soul.build_system_prompt.call_args
        assert call_args[0][0] == "clawra"

    def test_switch_persona(self, ceo):
        ceo.switch_persona("clawra")
        assert ceo.current_persona == "clawra"
        ceo.switch_persona("jarvis")
        assert ceo.current_persona == "jarvis"

    def test_switch_invalid_persona(self, ceo):
        with pytest.raises(ValueError):
            ceo.switch_persona("unknown")

    @pytest.mark.asyncio
    async def test_dispatch_to_worker(self, ceo, mock_router):
        worker = CodeWorker(model_router=mock_router)
        ceo.workers = {"code": worker}

        result = await ceo.dispatch_to_worker("code", "write a function")
        assert "result" in result

    @pytest.mark.asyncio
    async def test_dispatch_to_unknown_worker(self, ceo):
        with pytest.raises(ValueError, match="not registered"):
            await ceo.dispatch_to_worker("nonexistent", "task")

    @pytest.mark.asyncio
    async def test_no_soul_fallback(self, mock_router, mock_emotion, mock_memos, mock_registry):
        ceo = CEOAgent(
            model_router=mock_router,
            emotion_classifier=mock_emotion,
            memos=mock_memos,
            skill_registry=mock_registry,
        )
        result = await ceo.handle_message("hello")
        assert result == "Test response"

    @pytest.mark.asyncio
    async def test_skill_match_none(self, ceo, mock_router):
        """When no skills match, should fall through to normal chat."""
        result = await ceo.handle_message("隨便聊聊")
        assert result == "Test response"


# ── CodeWorker Tests ───────────────────────────────────────────


class TestCodeWorker:
    @pytest.mark.asyncio
    async def test_execute(self, mock_router):
        worker = CodeWorker(model_router=mock_router)
        result = await worker.execute("write a sort function")
        assert result["worker"] == "code"
        assert "result" in result

    @pytest.mark.asyncio
    async def test_execute_with_language(self, mock_router):
        worker = CodeWorker(model_router=mock_router)
        result = await worker.execute("sort", language="rust")
        assert result["worker"] == "code"

    @pytest.mark.asyncio
    async def test_execute_no_router(self):
        worker = CodeWorker()
        result = await worker.execute("test")
        assert "error" in result


# ── InterpreterWorker Tests ────────────────────────────────────


class TestInterpreterWorker:
    @pytest.mark.asyncio
    async def test_execute_python(self):
        worker = InterpreterWorker()
        result = await worker.execute(
            "print('hello')", shell="python", command="print('hello')"
        )
        assert result["returncode"] == 0
        assert "hello" in result["stdout"]

    @pytest.mark.asyncio
    async def test_execute_with_security_block(self):
        security = AsyncMock()
        verdict = MagicMock()
        verdict.action = "BLOCK"
        verdict.reason = "dangerous"
        security.authorize.return_value = verdict

        worker = InterpreterWorker(security_gate=security)
        result = await worker.execute("rm -rf /")
        assert "error" in result
        assert "Blocked" in result["error"]


# ── BrowserWorker Tests ───────────────────────────────────────


class TestBrowserWorker:
    @pytest.mark.asyncio
    async def test_execute_pending(self):
        worker = BrowserWorker()
        result = await worker.execute("search something")
        assert result["status"] == "pending_integration"
        assert result["worker"] == "browser"


# ── VisionWorker Tests ────────────────────────────────────────


class TestVisionWorker:
    @pytest.mark.asyncio
    async def test_execute_success(self, mock_router):
        mock_router.vision_analyze.return_value = ChatResponse(
            content="I see a button", model="glm-4v", usage={}
        )
        worker = VisionWorker(model_router=mock_router)
        result = await worker.execute("找按鈕", image_url="http://img.png")
        assert result["result"] == "I see a button"
        assert result["worker"] == "vision"

    @pytest.mark.asyncio
    async def test_execute_no_image(self, mock_router):
        worker = VisionWorker(model_router=mock_router)
        result = await worker.execute("找按鈕")
        assert "error" in result

    @pytest.mark.asyncio
    async def test_execute_no_router(self):
        worker = VisionWorker()
        result = await worker.execute("找按鈕", image_url="http://img.png")
        assert "error" in result


# ── SelfieWorker Tests ────────────────────────────────────────


class TestSelfieWorker:
    @pytest.mark.asyncio
    async def test_execute_no_registry(self):
        worker = SelfieWorker()
        result = await worker.execute("afternoon coffee")
        assert "error" in result

    @pytest.mark.asyncio
    async def test_execute_with_registry(self):
        registry = AsyncMock()
        registry.invoke.return_value = {
            "image_url": "https://fal.ai/img.jpg",
            "success": True,
        }
        worker = SelfieWorker(skill_registry=registry)
        result = await worker.execute("holding coffee")
        assert result["worker"] == "selfie"
        registry.invoke.assert_called_once()
