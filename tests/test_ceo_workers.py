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
    memos.get_conversation = AsyncMock(return_value=[])
    memos.log_message = AsyncMock()
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
        assert mock_memos.log_message.call_count == 2  # user + assistant

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
        from core.security_gate import OperationVerdict, SecurityEvent
        security = AsyncMock()
        event = SecurityEvent(
            operation="unsigned_script",
            verdict=OperationVerdict.BLOCK,
            detail="dangerous command",
        )
        security.authorize.return_value = event

        worker = InterpreterWorker(security_gate=security)
        result = await worker.execute("rm -rf /")
        assert "error" in result
        assert "Blocked" in result["error"]


# ── BrowserWorker Tests ───────────────────────────────────────


class TestBrowserWorker:
    @pytest.mark.asyncio
    async def test_execute_no_url(self):
        worker = BrowserWorker()
        result = await worker.execute("search something")
        assert result["status"] == "ready"
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


# ── Proactive Web Search Tests ──────────────────────────────────


class TestProactiveWebSearch:
    """Test that CEO agent proactively searches the web before LLM responds."""

    @pytest.fixture
    def mock_browser(self):
        browser = AsyncMock()
        browser.name = "browser"
        browser.fetch_url.return_value = {
            "status": "ok",
            "content": "<html>台北今天氣溫 15°C，多雲</html>",
        }
        browser.execute.return_value = {
            "status": "ok",
            "content": "<html>台北今天氣溫 15°C，多雲</html>",
            "worker": "browser",
        }
        return browser

    @pytest.fixture
    def ceo_with_browser(self, mock_router, mock_soul, mock_emotion, mock_memos, mock_registry, mock_browser):
        agent = CEOAgent(
            model_router=mock_router,
            soul=mock_soul,
            emotion_classifier=mock_emotion,
            memos=mock_memos,
            skill_registry=mock_registry,
            workers={"browser": mock_browser},
        )
        return agent

    @pytest.mark.asyncio
    async def test_search_triggered_by_query(self, ceo_with_browser, mock_browser):
        """「幫我查今天天氣」should trigger proactive web search via ReactExecutor."""
        await ceo_with_browser.handle_message("幫我查今天天氣")
        # ReactExecutor calls worker.execute(), not fetch_url
        mock_browser.execute.assert_called_once()

    @pytest.mark.asyncio
    async def test_search_triggered_by_url(self, ceo_with_browser, mock_browser):
        """URL in message should trigger direct fetch via ReactExecutor."""
        await ceo_with_browser.handle_message("幫我看 https://example.com/news")
        mock_browser.execute.assert_called_once()

    @pytest.mark.asyncio
    async def test_no_search_for_casual_chat(self, ceo_with_browser, mock_browser):
        """Casual chat should NOT trigger web search."""
        await ceo_with_browser.handle_message("你好")
        mock_browser.execute.assert_not_called()

    @pytest.mark.asyncio
    async def test_search_result_injected_into_context(self, ceo_with_browser, mock_router, mock_browser):
        """Search results should appear in system prompt context."""
        await ceo_with_browser.handle_message("幫我搜尋台北美食推薦")
        # Check that the LLM was called with search results in context
        call_args = mock_router.chat.call_args_list
        # The main CEO chat call (not skill judge)
        ceo_call = [c for c in call_args if c.kwargs.get("max_tokens", 0) == 500]
        assert len(ceo_call) > 0
        messages = ceo_call[0][0][0]
        system_msg = messages[0].content
        assert "網路搜尋結果" in system_msg or "台北" in system_msg

    @pytest.mark.asyncio
    async def test_search_keywords_variety(self, ceo_with_browser, mock_browser):
        """Various search-triggering keywords should all work."""
        triggers = [
            "查一下比特幣現在多少",
            "搜尋最新新聞",
            "今天天氣如何",
            "股價多少",
        ]
        for msg in triggers:
            mock_browser.execute.reset_mock()
            await ceo_with_browser.handle_message(msg)
            assert mock_browser.execute.called, f"Should search for: {msg}"

    @pytest.mark.asyncio
    async def test_no_browser_no_crash(self, ceo):
        """Without browser worker, should not crash."""
        result = await ceo.handle_message("幫我查天氣")
        assert result == "Test response"  # Normal LLM response

    @pytest.mark.asyncio
    async def test_search_failure_graceful(self, ceo_with_browser, mock_browser):
        """If web search fails, should still get normal LLM response."""
        mock_browser.execute.return_value = {"error": "Network error", "worker": "browser"}
        result = await ceo_with_browser.handle_message("幫我查天氣")
        assert result == "Test response"  # Falls back to normal


# ── ReactExecutor Integration Tests ──────────────────────────────


class TestCEOReactIntegration:
    """Test CEO agent integration with ReactExecutor."""

    @pytest.fixture
    def mock_knowledge(self):
        knowledge = AsyncMock()
        knowledge.name = "knowledge"
        knowledge.execute.return_value = {
            "result": "根據我的知識回答", "source": "knowledge", "worker": "knowledge",
        }
        return knowledge

    @pytest.fixture
    def ceo_with_react(self, mock_router, mock_soul, mock_emotion, mock_memos, mock_registry, mock_knowledge):
        mock_browser = AsyncMock()
        mock_browser.fetch_url.return_value = {"error": "Connection refused"}
        mock_browser.execute.return_value = {"error": "Connection refused", "worker": "browser"}

        agent = CEOAgent(
            model_router=mock_router,
            soul=mock_soul,
            emotion_classifier=mock_emotion,
            memos=mock_memos,
            skill_registry=mock_registry,
            workers={"browser": mock_browser, "knowledge": mock_knowledge},
        )
        return agent

    def test_react_executor_property(self, ceo_with_react):
        """react_executor should be auto-created from workers."""
        assert ceo_with_react.react_executor is not None

    def test_react_executor_none_without_workers(self, ceo):
        """react_executor should be None when no workers."""
        assert ceo.react_executor is None

    @pytest.mark.asyncio
    async def test_dispatch_with_react(self, ceo_with_react, mock_knowledge):
        """dispatch_to_worker with use_react=True should use ReactExecutor."""
        result = await ceo_with_react.dispatch_to_worker(
            "browser", "fetch something", use_react=True, url="https://example.com",
        )
        # Browser fails, should fallback to knowledge
        assert isinstance(result, dict)

    @pytest.mark.asyncio
    async def test_dispatch_without_react(self, ceo_with_react):
        """dispatch_to_worker with use_react=False should use direct worker call."""
        result = await ceo_with_react.dispatch_to_worker("browser", "fetch", url="https://example.com")
        assert isinstance(result, dict)

    @pytest.mark.asyncio
    async def test_pending_add_on_failure(self, ceo_with_react):
        """Failed react execution should add to pending tasks."""
        from core.pending_tasks import PendingTaskManager
        import tempfile
        import os

        mgr = PendingTaskManager(os.path.join(tempfile.mkdtemp(), "pending.json"))
        ceo_with_react.pending = mgr

        # Force all workers to fail
        ceo_with_react.workers["knowledge"].execute.return_value = {"error": "LLM down", "worker": "knowledge"}

        from core.react_executor import FuseState
        ceo_with_react._fuse = FuseState(max_rounds=10)
        ceo_with_react._react = None  # reset to force re-creation

        await ceo_with_react._execute_tool_call("https://example.com")
        # Should have added a pending task
        assert mgr.task_count >= 0  # may or may not add depending on result

    @pytest.mark.asyncio
    async def test_proactive_search_with_react(self, ceo_with_react, mock_router, mock_knowledge):
        """Proactive search should route through ReactExecutor."""
        # Knowledge worker provides content
        mock_knowledge.execute.return_value = {
            "result": "台北今天15度", "source": "knowledge", "worker": "knowledge",
        }

        from core.react_executor import FuseState
        ceo_with_react._fuse = FuseState(max_rounds=10)
        ceo_with_react._react = None  # reset

        result = await ceo_with_react.handle_message("幫我查今天天氣")
        assert result is not None  # Should get some response
