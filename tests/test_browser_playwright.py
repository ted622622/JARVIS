"""Tests for Patch K — Playwright browser automation + restaurant booking flow.

All Playwright interactions are fully mocked (no real browser needed).
"""

from __future__ import annotations

import asyncio
import re
from unittest.mock import AsyncMock, MagicMock, patch, PropertyMock

import pytest

# ═══════════════════════════════════════════════════════════════════
# K1: BrowserWorker Playwright tests
# ═══════════════════════════════════════════════════════════════════


class TestBrowserWorkerPlaywright:
    """Tests for BrowserWorker Playwright methods."""

    def _make_worker(self):
        from workers.browser_worker import BrowserWorker
        return BrowserWorker(security_gate=None, user_data_dir="./data/test_chrome")

    # ── _ensure_context ──────────────────────────────────────────

    @pytest.mark.asyncio
    async def test_ensure_context_lazy_init(self):
        """_ensure_context creates Playwright context on first call (CDP)."""
        import workers.browser_worker as bw
        worker = self._make_worker()

        mock_pw = AsyncMock()
        mock_browser = MagicMock()
        mock_context = AsyncMock()
        mock_browser.contexts = [mock_context]
        mock_pw.chromium.connect_over_cdp = AsyncMock(return_value=mock_browser)

        mock_asp = MagicMock()
        mock_asp.return_value.start = AsyncMock(return_value=mock_pw)

        original = getattr(bw, "async_playwright", None)
        bw.async_playwright = mock_asp
        try:
            with patch.object(bw, "_HAS_PLAYWRIGHT", True):
                ctx = await worker._ensure_context()
                assert ctx is mock_context
                mock_pw.chromium.connect_over_cdp.assert_called_once()
        finally:
            if original is not None:
                bw.async_playwright = original
            elif hasattr(bw, "async_playwright"):
                delattr(bw, "async_playwright")

    @pytest.mark.asyncio
    async def test_ensure_context_reuses_existing(self):
        """_ensure_context does not re-create if already initialized."""
        worker = self._make_worker()
        mock_context = AsyncMock()
        worker._pw_context = mock_context

        with patch("workers.browser_worker._HAS_PLAYWRIGHT", True):
            ctx = await worker._ensure_context()
            assert ctx is mock_context

    @pytest.mark.asyncio
    async def test_ensure_context_no_playwright_raises(self):
        """_ensure_context raises RuntimeError when Playwright not installed."""
        worker = self._make_worker()
        with patch("workers.browser_worker._HAS_PLAYWRIGHT", False):
            with pytest.raises(RuntimeError, match="playwright not installed"):
                await worker._ensure_context()

    # ── search_google_maps ───────────────────────────────────────

    @pytest.mark.asyncio
    async def test_search_google_maps_no_playwright(self):
        """Returns error dict when Playwright not installed."""
        worker = self._make_worker()
        with patch("workers.browser_worker._HAS_PLAYWRIGHT", False):
            result = await worker.search_google_maps("滿築火鍋")
            assert result["error"] == "playwright not installed"
            assert result["worker"] == "browser"

    @pytest.mark.asyncio
    async def test_search_google_maps_success_with_phone(self):
        """Returns structured data with name, phone, address via selectors."""
        worker = self._make_worker()
        worker._router = None  # skip vision, use selectors only

        mock_page = AsyncMock()
        mock_page.goto = AsyncMock()
        mock_page.wait_for_selector = AsyncMock()
        mock_page.wait_for_timeout = AsyncMock()
        mock_page.close = AsyncMock()

        def locator_factory(sel):
            if 'role="feed"' in sel:
                return self._make_locator(0)
            if sel == "h1":
                return self._make_locator(1, inner_text="滿築火鍋")
            if 'phone:tel:' in sel:
                return self._make_locator(1, get_attribute="電話號碼: 02-1234-5678 ")
            if 'data-item-id="address"' in sel:
                return self._make_locator(1, get_attribute="地址: 台北市中山區... ")
            if 'role="img"' in sel:
                return self._make_locator(1, get_attribute="4.5 顆星")
            if '訂位' in sel or '預約' in sel:
                return self._make_locator(0)
            if 'authority' in sel:
                return self._make_locator(0)
            return self._make_locator(0)

        mock_page.locator = MagicMock(side_effect=locator_factory)

        mock_context = AsyncMock()
        mock_context.new_page = AsyncMock(return_value=mock_page)
        worker._pw_context = mock_context

        with patch("workers.browser_worker._HAS_PLAYWRIGHT", True):
            result = await worker.search_google_maps("滿築火鍋")

        assert result["name"] == "滿築火鍋"
        assert result["phone"] == "02-1234-5678"
        assert result["address"] == "台北市中山區..."
        assert result.get("booking_url") is None
        assert result["worker"] == "browser"

    @pytest.mark.asyncio
    async def test_search_google_maps_with_booking_url(self):
        """Returns booking_url when reserve button found."""
        worker = self._make_worker()
        worker._router = None  # skip vision

        mock_page = AsyncMock()
        mock_page.goto = AsyncMock()
        mock_page.wait_for_selector = AsyncMock()
        mock_page.wait_for_timeout = AsyncMock()
        mock_page.close = AsyncMock()

        mock_reserve = self._make_locator(1, get_attribute="https://booking.example.com")

        def locator_factory(sel):
            if 'role="feed"' in sel:
                return self._make_locator(0)
            if sel == "h1":
                return self._make_locator(1, inner_text="好吃餐廳")
            if '訂位' in sel or '預約' in sel:
                return mock_reserve
            if 'authority' in sel:
                return self._make_locator(0)
            return self._make_locator(0)

        mock_page.locator = MagicMock(side_effect=locator_factory)

        mock_context = AsyncMock()
        mock_context.new_page = AsyncMock(return_value=mock_page)
        worker._pw_context = mock_context

        with patch("workers.browser_worker._HAS_PLAYWRIGHT", True):
            result = await worker.search_google_maps("好吃餐廳")

        assert result["booking_url"] == "https://booking.example.com"

    @pytest.mark.asyncio
    async def test_search_google_maps_timeout(self):
        """Returns error on timeout."""
        worker = self._make_worker()

        mock_page = AsyncMock()
        mock_page.goto = AsyncMock(side_effect=Exception("Timeout 15000ms exceeded"))
        mock_page.close = AsyncMock()

        mock_context = AsyncMock()
        mock_context.new_page = AsyncMock(return_value=mock_page)
        worker._pw_context = mock_context

        with patch("workers.browser_worker._HAS_PLAYWRIGHT", True):
            result = await worker.search_google_maps("不存在的店")

        assert "error" in result
        assert "Timeout" in result["error"]

    @pytest.mark.asyncio
    async def test_search_google_maps_no_results(self):
        """Returns error when feed selector not found."""
        worker = self._make_worker()

        mock_page = AsyncMock()
        mock_page.goto = AsyncMock()
        mock_page.wait_for_selector = AsyncMock(side_effect=Exception("waiting for selector"))
        mock_page.close = AsyncMock()

        mock_context = AsyncMock()
        mock_context.new_page = AsyncMock(return_value=mock_page)
        worker._pw_context = mock_context

        with patch("workers.browser_worker._HAS_PLAYWRIGHT", True):
            result = await worker.search_google_maps("xyznotfound")

        assert "error" in result

    # ── navigate_and_click ───────────────────────────────────────

    @pytest.mark.asyncio
    async def test_navigate_and_click_success(self):
        """navigate_and_click returns page content after clicking."""
        worker = self._make_worker()

        mock_page = AsyncMock()
        mock_page.goto = AsyncMock()
        mock_page.wait_for_selector = AsyncMock()
        mock_page.wait_for_timeout = AsyncMock()
        mock_page.inner_text = AsyncMock(return_value="Page content after click")
        mock_page.url = "https://example.com/result"
        mock_page.close = AsyncMock()

        mock_el = AsyncMock()
        mock_locator = MagicMock()
        mock_locator.first = mock_el
        mock_page.locator = MagicMock(return_value=mock_locator)

        mock_context = AsyncMock()
        mock_context.new_page = AsyncMock(return_value=mock_page)
        worker._pw_context = mock_context

        with patch("workers.browser_worker._HAS_PLAYWRIGHT", True):
            result = await worker.navigate_and_click("https://example.com", "button.submit")

        assert result["status"] == "ok"
        assert "Page content" in result["content"]

    @pytest.mark.asyncio
    async def test_navigate_and_click_no_playwright(self):
        """navigate_and_click returns error when Playwright not installed."""
        worker = self._make_worker()
        with patch("workers.browser_worker._HAS_PLAYWRIGHT", False):
            result = await worker.navigate_and_click("https://example.com", "button")
            assert result["error"] == "playwright not installed"

    @pytest.mark.asyncio
    async def test_navigate_and_click_failure(self):
        """navigate_and_click returns error on exception."""
        worker = self._make_worker()

        mock_page = AsyncMock()
        mock_page.goto = AsyncMock(side_effect=Exception("net::ERR_CONNECTION_REFUSED"))
        mock_page.close = AsyncMock()

        mock_context = AsyncMock()
        mock_context.new_page = AsyncMock(return_value=mock_page)
        worker._pw_context = mock_context

        with patch("workers.browser_worker._HAS_PLAYWRIGHT", True):
            result = await worker.navigate_and_click("https://down.example.com", "div")

        assert "error" in result

    # ── close_playwright ─────────────────────────────────────────

    @pytest.mark.asyncio
    async def test_close_playwright_cleans_up(self):
        """close_playwright disconnects CDP browser and stops Playwright."""
        worker = self._make_worker()
        mock_browser = MagicMock()
        mock_pw = AsyncMock()
        worker._browser = mock_browser
        worker._pw_context = MagicMock()
        worker._pw = mock_pw

        await worker.close_playwright()

        mock_browser.close.assert_called_once()
        mock_pw.stop.assert_called_once()
        assert worker._browser is None
        assert worker._pw_context is None
        assert worker._pw is None

    @pytest.mark.asyncio
    async def test_close_playwright_noop_when_not_init(self):
        """close_playwright does nothing when not initialized."""
        worker = self._make_worker()
        await worker.close_playwright()
        assert worker._pw_context is None
        assert worker._pw is None

    @pytest.mark.asyncio
    async def test_close_calls_close_playwright(self):
        """close() also shuts down Playwright."""
        worker = self._make_worker()
        worker._browser = MagicMock()
        worker._pw_context = MagicMock()
        worker._pw = AsyncMock()
        worker._http_client = None

        await worker.close()

        assert worker._browser is None
        assert worker._pw_context is None
        assert worker._pw is None

    # ── Helpers ──────────────────────────────────────────────────

    def _make_text_locator(self, sel):
        """Default locator that returns None (not found)."""
        loc = AsyncMock()
        loc.count = AsyncMock(return_value=0)
        first = AsyncMock()
        first.count = AsyncMock(return_value=0)
        loc.first = first
        return loc

    def _make_count_locator(self, count):
        loc = AsyncMock()
        loc.count = AsyncMock(return_value=count)
        loc.first = AsyncMock()
        return loc

    def _make_text_locator_by_map(self, sel, text_map):
        loc = AsyncMock()
        first = AsyncMock()
        if sel in text_map:
            loc.count = AsyncMock(return_value=1)
            first.count = AsyncMock(return_value=1)
            first.inner_text = AsyncMock(return_value=text_map[sel])
        else:
            loc.count = AsyncMock(return_value=0)
            first.count = AsyncMock(return_value=0)
        loc.first = first
        return loc

    def _make_locator(self, count=0, inner_text=None, get_attribute=None):
        """Make a mock locator with async count/text/attr."""
        loc = AsyncMock()
        loc.count = AsyncMock(return_value=count)
        first = AsyncMock()
        first.count = AsyncMock(return_value=count)
        if inner_text is not None:
            first.inner_text = AsyncMock(return_value=inner_text)
        if get_attribute is not None:
            first.get_attribute = AsyncMock(return_value=get_attribute)
        loc.first = first
        return loc


# ═══════════════════════════════════════════════════════════════════
# K2: CEO Agent booking trigger tests
# ═══════════════════════════════════════════════════════════════════


class TestCEOBookingTrigger:
    """Tests for CEO Agent booking pattern detection and MAPS tag."""

    def test_web_need_patterns_match_booking(self):
        """Booking keywords trigger web need detection."""
        from core.ceo_agent import _WEB_NEED_PATTERNS
        assert _WEB_NEED_PATTERNS.search("幫我訂明天晚上6點的滿足火鍋五個人")
        assert _WEB_NEED_PATTERNS.search("訂位")
        assert _WEB_NEED_PATTERNS.search("預約餐廳")
        assert _WEB_NEED_PATTERNS.search("幫我預定")

    def test_tool_pattern_matches_maps(self):
        """[MAPS:query] tag is recognized by _TOOL_PATTERN."""
        from core.ceo_agent import _TOOL_PATTERN
        m = _TOOL_PATTERN.search("[MAPS:滿築火鍋]")
        assert m is not None
        assert m.group(1) == "滿築火鍋"

    def test_tool_pattern_still_matches_fetch_search(self):
        """FETCH and SEARCH tags still work."""
        from core.ceo_agent import _TOOL_PATTERN
        assert _TOOL_PATTERN.search("[FETCH:https://example.com]")
        assert _TOOL_PATTERN.search("[SEARCH:台北天氣]")

    @pytest.mark.asyncio
    async def test_proactive_booking_calls_maps(self):
        """Booking intent triggers Google Maps search proactively."""
        from core.ceo_agent import CEOAgent
        from unittest.mock import AsyncMock, MagicMock

        router = MagicMock()
        ceo = CEOAgent(model_router=router)

        mock_browser = AsyncMock()
        mock_browser.fetch_url = AsyncMock()
        mock_browser.search_google_maps = AsyncMock(return_value={
            "name": "滿築火鍋",
            "phone": "02-1234-5678",
            "address": "台北市",
            "rating": "4.5",
            "booking_url": None,
            "worker": "browser",
        })
        ceo.workers = {"browser": mock_browser}

        result = await ceo._proactive_web_search("幫我訂滿築火鍋")

        mock_browser.search_google_maps.assert_called_once()
        assert isinstance(result, dict)
        assert result["phone"] == "02-1234-5678"
        assert "滿築火鍋" in result["text"]

    @pytest.mark.asyncio
    async def test_proactive_booking_strips_time_info(self):
        """Booking search strips time/people count from query."""
        from core.ceo_agent import CEOAgent
        from unittest.mock import AsyncMock, MagicMock

        router = MagicMock()
        ceo = CEOAgent(model_router=router)

        mock_browser = AsyncMock()
        mock_browser.fetch_url = AsyncMock()
        mock_browser.search_google_maps = AsyncMock(return_value={
            "name": "火鍋店",
            "phone": "02-0000-0000",
            "address": "somewhere",
            "worker": "browser",
        })
        ceo.workers = {"browser": mock_browser}

        await ceo._proactive_web_search("幫我訂明天晚上6點的滿足火鍋5個人")

        call_args = mock_browser.search_google_maps.call_args[0][0]
        assert "明天" not in call_args
        assert "晚上" not in call_args
        assert "6點" not in call_args
        assert "5個人" not in call_args

    @pytest.mark.asyncio
    async def test_maps_result_with_phone_no_booking(self):
        """MAPS result with phone but no booking_url returns phone key."""
        from core.ceo_agent import CEOAgent
        from unittest.mock import AsyncMock, MagicMock

        router = MagicMock()
        ceo = CEOAgent(model_router=router)

        mock_browser = AsyncMock()
        mock_browser.fetch_url = AsyncMock()
        mock_browser.search_google_maps = AsyncMock(return_value={
            "name": "Test",
            "phone": "02-1111-2222",
            "address": "addr",
            "worker": "browser",
        })
        ceo.workers = {"browser": mock_browser}

        result = await ceo._execute_tool_call("Test Restaurant", tag="MAPS")
        assert "02-1111-2222" in result

    @pytest.mark.asyncio
    async def test_maps_result_with_booking_url(self):
        """MAPS result includes booking URL."""
        from core.ceo_agent import CEOAgent
        from unittest.mock import AsyncMock, MagicMock

        router = MagicMock()
        ceo = CEOAgent(model_router=router)

        mock_browser = AsyncMock()
        mock_browser.fetch_url = AsyncMock()
        mock_browser.search_google_maps = AsyncMock(return_value={
            "name": "Test",
            "phone": "02-1111-2222",
            "booking_url": "https://book.example.com",
            "worker": "browser",
        })
        ceo.workers = {"browser": mock_browser}

        result = await ceo._execute_tool_call("Test Restaurant", tag="MAPS")
        assert "https://book.example.com" in result


# ═══════════════════════════════════════════════════════════════════
# K3: Telegram phone fallback tests
# ═══════════════════════════════════════════════════════════════════


class TestTelegramPhoneFallback:
    """Tests for Telegram phone/booking_url fallback messages."""

    @pytest.mark.asyncio
    async def test_dict_reply_with_phone_sends_two_messages(self):
        """Dict reply with phone → text reply + separate phone message."""
        from clients.telegram_client import TelegramClient

        client = TelegramClient()
        client._BATCH_DELAY = 0
        client._on_message = AsyncMock(return_value={
            "text": "找到了滿築火鍋！",
            "phone": "02-1234-5678",
        })
        client._token_to_persona = {"token123": "jarvis"}

        update = MagicMock()
        update.message.text = "幫我訂滿築"
        update.message.chat_id = 123
        update.message.from_user.id = 1
        update.message.from_user.first_name = "Ted"

        ctx = MagicMock()
        ctx.bot.token = "token123"
        ctx.bot.send_message = AsyncMock()

        client._allowed_user_ids = set()
        await client._handle_text_message(update, ctx)

        # Text + phone both sent via bot.send_message
        calls = [c.kwargs for c in ctx.bot.send_message.call_args_list]
        texts = [c["text"] for c in calls]
        assert "找到了滿築火鍋！" in texts
        assert "02-1234-5678" in texts

    @pytest.mark.asyncio
    async def test_dict_reply_without_phone_no_extra(self):
        """Dict reply without phone → only text reply, no extra messages."""
        from clients.telegram_client import TelegramClient

        client = TelegramClient()
        client._BATCH_DELAY = 0
        client._on_message = AsyncMock(return_value={
            "text": "一般回覆",
        })
        client._token_to_persona = {"tok": "jarvis"}

        update = MagicMock()
        update.message.text = "hello"
        update.message.chat_id = 123
        update.message.from_user.id = 1
        update.message.from_user.first_name = "Ted"

        ctx = MagicMock()
        ctx.bot.token = "tok"
        ctx.bot.send_message = AsyncMock()

        client._allowed_user_ids = set()
        await client._handle_text_message(update, ctx)

        ctx.bot.send_message.assert_called_once_with(chat_id=123, text="一般回覆")

    @pytest.mark.asyncio
    async def test_dict_reply_with_booking_url(self):
        """Dict reply with booking_url → sends booking_url as separate message."""
        from clients.telegram_client import TelegramClient

        client = TelegramClient()
        client._BATCH_DELAY = 0
        client._on_message = AsyncMock(return_value={
            "text": "找到了！",
            "booking_url": "https://book.example.com",
        })
        client._token_to_persona = {"tok": "jarvis"}

        update = MagicMock()
        update.message.text = "訂位"
        update.message.chat_id = 456
        update.message.from_user.id = 1
        update.message.from_user.first_name = "Ted"

        ctx = MagicMock()
        ctx.bot.token = "tok"
        ctx.bot.send_message = AsyncMock()

        client._allowed_user_ids = set()
        await client._handle_text_message(update, ctx)

        calls = [c.kwargs for c in ctx.bot.send_message.call_args_list]
        texts = [c["text"] for c in calls]
        assert "https://book.example.com" in texts


# ═══════════════════════════════════════════════════════════════════
# K4: ReactExecutor + ErrorClassifier + TaskRouter tests
# ═══════════════════════════════════════════════════════════════════


class TestReactExecutorMapsChain:
    """Tests for maps_search fallback chain."""

    def test_maps_search_chain_exists(self):
        from core.react_executor import FALLBACK_CHAINS
        assert "maps_search" in FALLBACK_CHAINS
        assert FALLBACK_CHAINS["maps_search"] == ["browser", "knowledge"]


class TestErrorClassifierPlaywright:
    """Tests for Playwright-specific error patterns."""

    def test_playwright_timeout(self):
        from core.error_classifier import ErrorClassifier, ErrorType
        strategy = ErrorClassifier.classify("playwright Timeout 15000ms exceeded")
        assert strategy.error_type == ErrorType.TIMEOUT
        assert strategy.retry is True

    def test_playwright_not_installed(self):
        from core.error_classifier import ErrorClassifier, ErrorType
        strategy = ErrorClassifier.classify("playwright not installed")
        assert strategy.error_type == ErrorType.DEPENDENCY_MISSING
        assert strategy.retry is False

    def test_selector_not_found(self):
        from core.error_classifier import ErrorClassifier, ErrorType
        strategy = ErrorClassifier.classify("waiting for selector '[role=\"feed\"]'")
        assert strategy.error_type == ErrorType.ELEMENT_NOT_FOUND
        assert strategy.retry is False

    def test_dependency_missing_type_exists(self):
        from core.error_classifier import ErrorType
        assert ErrorType.DEPENDENCY_MISSING == "dependency_missing"


class TestTaskRouterBooking:
    """Tests for restaurant_booking route in TaskRouter."""

    def test_booking_pattern_match(self):
        from core.task_router import TaskRouter
        router = TaskRouter()

        tasks = router.classify("幫我訂明天晚上6點的滿足火鍋五個人")
        types = [t.task_type for t in tasks]
        assert "restaurant_booking" in types

    def test_booking_pattern_reserve(self):
        from core.task_router import TaskRouter
        router = TaskRouter()

        tasks = router.classify("預約今天中午12點的好吃餐廳")
        types = [t.task_type for t in tasks]
        assert "restaurant_booking" in types

    def test_normal_msg_no_booking(self):
        from core.task_router import TaskRouter
        router = TaskRouter()

        tasks = router.classify("今天天氣怎麼樣")
        types = [t.task_type for t in tasks]
        assert "restaurant_booking" not in types


# ═══════════════════════════════════════════════════════════════════
# Integration: end-to-end booking flow (all mocked)
# ═══════════════════════════════════════════════════════════════════


class TestBookingIntegration:
    """End-to-end booking flow with all components mocked."""

    @pytest.mark.asyncio
    async def test_booking_flow_phone_fallback(self):
        """User asks to book → Maps search → phone returned → dict reply with phone."""
        from core.ceo_agent import CEOAgent
        from clients.base_client import ChatResponse

        # Mock router
        router = MagicMock()
        mock_response = MagicMock(spec=ChatResponse)
        mock_response.content = "好的，我幫你找到了滿築火鍋的資訊，電話已經附上囉！"
        router.chat = AsyncMock(return_value=mock_response)

        # Mock browser with Maps — no booking_url, no web booking found
        mock_browser = AsyncMock()
        mock_browser.fetch_url = AsyncMock()
        mock_browser.search_google_maps = AsyncMock(return_value={
            "name": "滿築火鍋",
            "phone": "02-1234-5678",
            "address": "台北市中山區",
            "rating": "4.5",
            "booking_url": None,
            "worker": "browser",
        })
        mock_browser.find_booking_url = AsyncMock(return_value=None)
        mock_browser.complete_booking = AsyncMock(return_value={
            "error": "no_booking_url", "worker": "browser",
        })

        # Setup CEO
        ceo = CEOAgent(model_router=router)
        ceo.workers = {"browser": mock_browser}
        ceo.memos = None
        ceo.md_memory = None
        ceo.emotion = None
        ceo.skills = None

        # Skip Agent SDK dispatch so test exercises the booking flow
        with patch.object(ceo, "_get_agent_executor", return_value=None):
            result = await ceo.handle_message("幫我訂明天晚上6點的滿築火鍋五個人")

        # Should return dict with phone (no booking_url, so goes through LLM)
        assert isinstance(result, dict)
        assert result["phone"] == "02-1234-5678"
        assert "text" in result

    @pytest.mark.asyncio
    async def test_booking_flow_with_booking_url(self):
        """User asks to book → Maps search → booking_url returned."""
        from core.ceo_agent import CEOAgent
        from clients.base_client import ChatResponse

        router = MagicMock()
        mock_response = MagicMock(spec=ChatResponse)
        mock_response.content = "找到訂位連結了！"
        router.chat = AsyncMock(return_value=mock_response)

        mock_browser = AsyncMock()
        mock_browser.fetch_url = AsyncMock()
        mock_browser.search_google_maps = AsyncMock(return_value={
            "name": "好餐廳",
            "phone": "02-9999-0000",
            "address": "台北市",
            "booking_url": "https://booking.example.com",
            "worker": "browser",
        })
        # complete_booking fails → falls back to returning info
        mock_browser.complete_booking = AsyncMock(return_value={
            "error": "test_skip", "worker": "browser",
        })

        ceo = CEOAgent(model_router=router)
        ceo.workers = {"browser": mock_browser}
        ceo.memos = None
        ceo.md_memory = None
        ceo.emotion = None
        ceo.skills = None

        result = await ceo.handle_message("預約好餐廳")

        assert isinstance(result, dict)
        assert result["booking_url"] == "https://booking.example.com"
        assert result["phone"] == "02-9999-0000"
        # Short-circuit: reply assembled directly, not from LLM
        assert "好餐廳" in result["text"]


# ═══════════════════════════════════════════════════════════════════
# Playwright availability cache tests
# ═══════════════════════════════════════════════════════════════════


class TestPlaywrightAvailabilityCache:
    """Tests for _pw_available cache to avoid 20-second retry waste."""

    def _make_worker(self):
        from workers.browser_worker import BrowserWorker
        return BrowserWorker(security_gate=None, user_data_dir="./data/test_chrome")

    @pytest.mark.asyncio
    async def test_pw_available_initially_none(self):
        """_pw_available starts as None (unknown)."""
        worker = self._make_worker()
        assert worker._pw_available is None

    @pytest.mark.asyncio
    async def test_pw_available_set_false_after_failure(self):
        """After _ensure_context fails 10 times, _pw_available is set to False."""
        import workers.browser_worker as bw
        worker = self._make_worker()

        mock_pw = AsyncMock()
        # All connect_over_cdp calls fail
        mock_pw.chromium.connect_over_cdp = AsyncMock(
            side_effect=Exception("Connection refused")
        )

        mock_asp = MagicMock()
        mock_asp.return_value.start = AsyncMock(return_value=mock_pw)

        original = getattr(bw, "async_playwright", None)
        bw.async_playwright = mock_asp
        try:
            with (
                patch.object(bw, "_HAS_PLAYWRIGHT", True),
                patch("subprocess.run"),
                patch("subprocess.Popen"),
                patch("asyncio.sleep", new_callable=AsyncMock),
            ):
                with pytest.raises(RuntimeError, match="Failed to connect"):
                    await worker._ensure_context()
                assert worker._pw_available is False
        finally:
            if original is not None:
                bw.async_playwright = original
            elif hasattr(bw, "async_playwright"):
                delattr(bw, "async_playwright")

    @pytest.mark.asyncio
    async def test_pw_available_false_skips_immediately(self):
        """When _pw_available is False, _ensure_context raises immediately."""
        worker = self._make_worker()
        worker._pw_available = False

        with patch("workers.browser_worker._HAS_PLAYWRIGHT", True):
            with pytest.raises(RuntimeError, match="previously failed"):
                await worker._ensure_context()

    @pytest.mark.asyncio
    async def test_search_google_maps_returns_error_when_cached_fail(self):
        """search_google_maps returns error dict when Playwright cached as failed."""
        worker = self._make_worker()
        worker._pw_available = False

        with patch("workers.browser_worker._HAS_PLAYWRIGHT", True):
            result = await worker.search_google_maps("任何餐廳")
            assert "error" in result
            assert "previously failed" in result["error"]

    @pytest.mark.asyncio
    async def test_pw_available_set_true_on_success(self):
        """_pw_available is set to True on successful CDP connection."""
        import workers.browser_worker as bw
        worker = self._make_worker()

        mock_pw = AsyncMock()
        mock_browser = MagicMock()
        mock_context = AsyncMock()
        mock_browser.contexts = [mock_context]
        mock_pw.chromium.connect_over_cdp = AsyncMock(return_value=mock_browser)

        mock_asp = MagicMock()
        mock_asp.return_value.start = AsyncMock(return_value=mock_pw)

        original = getattr(bw, "async_playwright", None)
        bw.async_playwright = mock_asp
        try:
            with patch.object(bw, "_HAS_PLAYWRIGHT", True):
                await worker._ensure_context()
                assert worker._pw_available is True
        finally:
            if original is not None:
                bw.async_playwright = original
            elif hasattr(bw, "async_playwright"):
                delattr(bw, "async_playwright")
