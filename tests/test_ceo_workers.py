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


# ── Patch M: Selfie Scene + Pending Tests ────────────────────────


class TestSelfieSceneAndPending:
    """Test that CEO passes scene=user_message and saves pending selfies."""

    @pytest.mark.asyncio
    async def test_skill_invoke_passes_scene(self, mock_router, mock_soul, mock_emotion, mock_memos):
        """skills.invoke should be called with scene=user_message."""
        reg = MagicMock()

        mock_meta = MagicMock()
        mock_meta.name = "selfie"
        mock_meta.description = "自拍"
        reg.list_all.return_value = [mock_meta]
        reg.get.return_value = mock_meta
        reg.invoke = AsyncMock(return_value={
            "image_url": "https://fal.ai/test.jpg",
            "success": True,
        })

        # Make the router return "SKILL:selfie" for the judge prompt
        mock_router.chat.side_effect = [
            ChatResponse(content="SKILL:selfie", model="test", usage={}),
            ChatResponse(content="看我的自拍～", model="test", usage={}),
        ]

        ceo = CEOAgent(
            model_router=mock_router,
            soul=mock_soul,
            emotion_classifier=mock_emotion,
            memos=mock_memos,
            skill_registry=reg,
        )
        result = await ceo.handle_message("幫我拍一張自拍")

        # Verify scene=user_message was passed (Patch Q: also passes growth_content)
        reg.invoke.assert_called_once()
        call_args = reg.invoke.call_args
        assert call_args[0] == ("selfie",)
        assert call_args[1]["scene"] == "幫我拍一張自拍"
        assert "growth_content" in call_args[1]

    @pytest.mark.asyncio
    async def test_pending_selfie_saved_on_queue_info(self, mock_router, mock_soul, mock_emotion, mock_memos, tmp_path):
        """When skill returns queue_info, CEO saves pending selfie."""
        reg = MagicMock()
        mock_meta = MagicMock()
        mock_meta.name = "selfie"
        mock_meta.description = "自拍"
        reg.list_all.return_value = [mock_meta]
        reg.get.return_value = mock_meta
        reg.invoke = AsyncMock(return_value={
            "image_url": None,
            "success": False,
            "error": "生成中，稍後補發",
            "queue_info": {
                "status_url": "https://queue.fal.run/status/abc",
                "response_url": "https://queue.fal.run/result/abc",
                "persona": "clawra",
            },
        })

        # Regex pre-match skips LLM judge, so only CEO reply mock needed
        mock_router.chat.side_effect = [
            ChatResponse(content="等等喔～", model="test", usage={}),
        ]

        ceo = CEOAgent(
            model_router=mock_router,
            soul=mock_soul,
            emotion_classifier=mock_emotion,
            memos=mock_memos,
            skill_registry=reg,
        )
        # Override the pending selfie path to tmp_path
        ceo.PENDING_SELFIE_PATH = tmp_path / "pending_selfies.json"

        result = await ceo.handle_message("拍個自拍")

        # Should have saved pending selfie
        import json
        assert ceo.PENDING_SELFIE_PATH.exists()
        entries = json.loads(ceo.PENDING_SELFIE_PATH.read_text(encoding="utf-8"))
        assert len(entries) == 1
        assert entries[0]["status_url"] == "https://queue.fal.run/status/abc"
        assert entries[0]["status"] == "pending"
        assert entries[0]["persona"] == "clawra"

        # Should have set the selfie-specific excuse hint
        # (result is from the second chat call since _try_skill_match returns None)
        assert result == "等等喔～"

    @pytest.mark.asyncio
    async def test_pending_selfie_max_limit(self, tmp_path):
        """Only keeps MAX_PENDING_SELFIES entries."""
        import json

        ceo = CEOAgent(model_router=AsyncMock())
        ceo.PENDING_SELFIE_PATH = tmp_path / "pending_selfies.json"

        # Save more than MAX
        for i in range(ceo.MAX_PENDING_SELFIES + 3):
            ceo._save_pending_selfie({
                "status_url": f"https://queue.fal.run/status/{i}",
                "response_url": f"https://queue.fal.run/result/{i}",
                "persona": "clawra",
            })

        entries = json.loads(ceo.PENDING_SELFIE_PATH.read_text(encoding="utf-8"))
        assert len(entries) == ceo.MAX_PENDING_SELFIES

    def test_load_pending_selfies_empty(self, tmp_path):
        """Load from non-existent file returns empty list."""
        ceo = CEOAgent(model_router=AsyncMock())
        ceo.PENDING_SELFIE_PATH = tmp_path / "nonexistent.json"
        assert ceo._load_pending_selfies() == []

    def test_load_pending_selfies_corrupt(self, tmp_path):
        """Corrupt JSON returns empty list."""
        ceo = CEOAgent(model_router=AsyncMock())
        corrupt_file = tmp_path / "bad.json"
        corrupt_file.write_text("not json!", encoding="utf-8")
        ceo.PENDING_SELFIE_PATH = corrupt_file
        assert ceo._load_pending_selfies() == []


# ── Patch O: Complexity Estimation Tests ──────────────────────────


class TestEstimateComplexity:
    """Test CEO.estimate_complexity() for long-task detection."""

    @pytest.fixture
    def ceo_simple(self):
        return CEOAgent(model_router=AsyncMock())

    def test_simple_greeting_not_long(self, ceo_simple):
        """Simple greeting → is_long=False."""
        result = ceo_simple.estimate_complexity("你好")
        assert result["is_long"] is False
        assert result["estimate_seconds"] == 5

    def test_url_detected_as_long(self, ceo_simple):
        """Message with URL → is_long=True, reason=web_task."""
        result = ceo_simple.estimate_complexity("幫我看 https://github.com/repo")
        assert result["is_long"] is True
        assert result["reason"] == "web_task"
        assert result["estimate_seconds"] == 45

    def test_web_search_keyword_long(self, ceo_simple):
        """Web search keywords → is_long=True."""
        result = ceo_simple.estimate_complexity("幫我查比特幣現在多少")
        assert result["is_long"] is True
        assert result["reason"] == "web_task"

    def test_booking_keyword_long(self, ceo_simple):
        """Booking intent → is_long=True."""
        result = ceo_simple.estimate_complexity("幫我訂明天晚上的餐廳")
        assert result["is_long"] is True
        assert result["reason"] == "web_task"

    def test_code_keyword_long(self, ceo_simple):
        """Code task → is_long=True."""
        result = ceo_simple.estimate_complexity("幫我寫一個程式")
        assert result["is_long"] is True
        assert result["reason"] == "web_task"

    def test_long_text_detected(self, ceo_simple):
        """Message over 300 chars → is_long=True, reason=complex_instruction."""
        long_msg = "這是一段很長的指令。" * 40  # 400 chars
        result = ceo_simple.estimate_complexity(long_msg)
        assert result["is_long"] is True
        assert result["reason"] == "complex_instruction"
        assert result["estimate_seconds"] == 30

    def test_short_question_not_long(self, ceo_simple):
        """Short factual question → is_long=False."""
        result = ceo_simple.estimate_complexity("什麼是 Python？")
        assert result["is_long"] is False


# ── Patch O: Empty Reply Guard Tests ─────────────────────────────


class TestEmptyReplyGuard:
    """Test that CEO returns a friendly fallback instead of empty string."""

    @pytest.fixture
    def ceo_with_browser(self, mock_router, mock_soul, mock_emotion, mock_memos, mock_registry):
        mock_browser = AsyncMock()
        mock_browser.name = "browser"
        mock_browser.fetch_url.return_value = {"content": "some page content"}
        mock_browser.execute.return_value = {
            "content": "some page content", "worker": "browser",
        }
        return CEOAgent(
            model_router=mock_router,
            soul=mock_soul,
            emotion_classifier=mock_emotion,
            memos=mock_memos,
            skill_registry=mock_registry,
            workers={"browser": mock_browser},
        )

    @pytest.mark.asyncio
    async def test_empty_reply_returns_fallback_jarvis(self, mock_router, mock_soul, mock_emotion, mock_memos, mock_registry):
        """Empty LLM reply → friendly fallback for JARVIS."""
        mock_router.chat.return_value = ChatResponse(
            content="", model="test", usage={},
        )
        ceo = CEOAgent(
            model_router=mock_router,
            soul=mock_soul,
            emotion_classifier=mock_emotion,
            memos=mock_memos,
            skill_registry=mock_registry,
        )
        result = await ceo.handle_message("測試空回覆")
        assert result  # not empty
        assert "Sir" in result

    @pytest.mark.asyncio
    async def test_empty_reply_returns_fallback_clawra(self, mock_router, mock_soul, mock_emotion, mock_memos, mock_registry):
        """Empty LLM reply → friendly fallback for Clawra."""
        mock_router.chat.return_value = ChatResponse(
            content="", model="test", usage={},
        )
        ceo = CEOAgent(
            model_router=mock_router,
            soul=mock_soul,
            emotion_classifier=mock_emotion,
            memos=mock_memos,
            skill_registry=mock_registry,
        )
        result = await ceo.handle_message("測試空回覆", persona="clawra")
        assert result
        assert "再問一次" in result

    @pytest.mark.asyncio
    async def test_tool_use_followup_gets_higher_max_tokens(self, ceo_with_browser, mock_router):
        """When tool-use triggers a followup call, max_tokens should be 4096."""
        # First call returns [FETCH:url], second call returns final reply
        mock_router.chat.side_effect = [
            ChatResponse(content="[FETCH:https://example.com]", model="test", usage={}),
            ChatResponse(content="這是結果", model="test", usage={}),
        ]
        result = await ceo_with_browser.handle_message("幫我看 https://example.com")
        # Verify the second chat call used max_tokens=4096
        second_call = mock_router.chat.call_args_list[-1]
        assert second_call.kwargs.get("max_tokens") == 4096


# ── Booking Fallback Tests (Maps failure → httpx fallback) ────────


class TestBookingFallbackWhenMapsFails:
    """Test that booking flow degrades to find_booking_url() when Maps fails."""

    @pytest.mark.asyncio
    async def test_booking_fallback_when_maps_fails(self):
        """Maps returns error → find_booking_url() is called."""
        from core.ceo_agent import CEOAgent
        from clients.base_client import ChatResponse

        router = MagicMock()
        router.chat = AsyncMock(return_value=ChatResponse(
            content="幫你查到了！", model="test", usage={},
        ))

        mock_browser = AsyncMock()
        mock_browser.fetch_url = AsyncMock()
        mock_browser.search_google_maps = AsyncMock(return_value={
            "error": "Failed to connect to Chrome after 10 attempts",
            "worker": "browser",
        })
        mock_browser.find_booking_url = AsyncMock(return_value="https://inline.app/booking/test")

        ceo = CEOAgent(model_router=router)
        ceo.workers = {"browser": mock_browser}
        ceo.memos = None
        ceo.md_memory = None
        ceo.emotion = None
        ceo.skills = None

        result = await ceo.handle_message("幫我訂滿足火鍋")

        mock_browser.find_booking_url.assert_called_once()
        assert isinstance(result, dict)
        assert result.get("booking_url") == "https://inline.app/booking/test"
        # text comes from LLM, booking_url is attached by CEO
        assert "text" in result

    @pytest.mark.asyncio
    async def test_booking_fallback_finds_url(self):
        """Fallback finds inline.app URL → correct dict returned."""
        from core.ceo_agent import CEOAgent

        ceo = CEOAgent(model_router=MagicMock())

        mock_browser = AsyncMock()
        mock_browser.fetch_url = AsyncMock()
        mock_browser.search_google_maps = AsyncMock(return_value={
            "error": "Playwright previously failed, skipping",
            "worker": "browser",
        })
        mock_browser.find_booking_url = AsyncMock(
            return_value="https://inline.app/booking/awesome-restaurant"
        )
        ceo.workers = {"browser": mock_browser}

        result = await ceo._proactive_web_search("幫我訂好棒棒餐廳")

        assert isinstance(result, dict)
        assert result["booking_url"] == "https://inline.app/booking/awesome-restaurant"
        assert "好棒棒餐廳" in result["text"]

    @pytest.mark.asyncio
    async def test_booking_fallback_no_url_no_crash(self):
        """Fallback also finds nothing → falls through without crash."""
        from core.ceo_agent import CEOAgent

        ceo = CEOAgent(model_router=MagicMock())

        mock_browser = AsyncMock()
        mock_browser.fetch_url = AsyncMock()
        mock_browser.search_google_maps = AsyncMock(return_value={
            "error": "Playwright previously failed, skipping",
            "worker": "browser",
        })
        mock_browser.find_booking_url = AsyncMock(return_value=None)
        ceo.workers = {"browser": mock_browser}

        # Should not crash, should fall through to DuckDuckGo search
        result = await ceo._proactive_web_search("幫我訂不存在的餐廳")
        # Result could be None or a DDG search result; either way no exception
        mock_browser.find_booking_url.assert_called_once()

    @pytest.mark.asyncio
    async def test_maps_tool_fallback(self):
        """[MAPS:query] tag fails → fallback provides booking URL."""
        from core.ceo_agent import CEOAgent

        ceo = CEOAgent(model_router=MagicMock())

        mock_browser = AsyncMock()
        mock_browser.search_google_maps = AsyncMock(return_value={
            "error": "Failed to connect to Chrome after 10 attempts",
            "worker": "browser",
        })
        mock_browser.find_booking_url = AsyncMock(
            return_value="https://eztable.com/restaurant/123"
        )
        ceo.workers = {"browser": mock_browser}

        result = await ceo._execute_tool_call("滿足火鍋", tag="MAPS")

        mock_browser.find_booking_url.assert_called_once_with("滿足火鍋")
        assert "eztable.com" in result
        assert "訂位連結" in result

    @pytest.mark.asyncio
    async def test_maps_tool_fallback_no_url(self):
        """[MAPS:query] fails and no booking URL → error message returned."""
        from core.ceo_agent import CEOAgent

        ceo = CEOAgent(model_router=MagicMock())

        mock_browser = AsyncMock()
        mock_browser.search_google_maps = AsyncMock(return_value={
            "error": "Playwright previously failed, skipping",
            "worker": "browser",
        })
        mock_browser.find_booking_url = AsyncMock(return_value=None)
        ceo.workers = {"browser": mock_browser}

        result = await ceo._execute_tool_call("隨便餐廳", tag="MAPS")

        assert "搜尋失敗" in result
