"""Tests for AssistWorker — 做到 90% 給選項.

Run: pytest tests/test_assist_worker.py -v
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from clients.base_client import ChatResponse
from workers.assist_worker import AssistWorker


# ── Fixtures ────────────────────────────────────────────────────


@pytest.fixture
def mock_router():
    router = MagicMock()
    router.chat = AsyncMock(
        return_value=ChatResponse(content='{"phone": "02-2771-1234", "address": "台北市大安區", "hours": "11:00-22:00", "booking_url": "https://inline.app/booking/test"}', model="test")
    )
    return router


@pytest.fixture
def mock_memory_search():
    search = MagicMock()
    search.search.return_value = [
        {"text": "上次去 Niku Mura 很好吃，牛舌推薦"},
    ]
    return search


@pytest.fixture
def worker(mock_router, mock_memory_search):
    return AssistWorker(
        model_router=mock_router,
        memory_search=mock_memory_search,
    )


@pytest.fixture
def worker_no_deps():
    return AssistWorker()


# ── Execute dispatch tests ──────────────────────────────────────


class TestExecuteDispatch:
    @pytest.mark.asyncio
    async def test_booking_dispatch(self, worker):
        with patch.object(worker, "_web_search", new_callable=AsyncMock, return_value="test search"):
            result = await worker.execute(
                "幫我訂 Niku Mura", task_type="booking",
            )
        assert result["is_partial"] is True
        assert result["worker"] == "assist"
        assert "Niku Mura" in result["result"]

    @pytest.mark.asyncio
    async def test_ticket_dispatch(self, worker):
        with patch.object(worker, "_web_search", new_callable=AsyncMock, return_value=""):
            result = await worker.execute(
                "幫我買高鐵票", task_type="ticket",
            )
        assert result["is_partial"] is True
        assert result["worker"] == "assist"

    @pytest.mark.asyncio
    async def test_generic_dispatch(self, worker):
        with patch.object(worker, "_web_search", new_callable=AsyncMock, return_value=""):
            result = await worker.execute(
                "幫我查一下資料", task_type="generic",
            )
        assert result["is_partial"] is True

    @pytest.mark.asyncio
    async def test_default_generic(self, worker):
        with patch.object(worker, "_web_search", new_callable=AsyncMock, return_value=""):
            result = await worker.execute("隨便什麼")
        assert result["is_partial"] is True


# ── Booking assist tests ────────────────────────────────────────


class TestAssistBooking:
    @pytest.mark.asyncio
    async def test_booking_includes_restaurant_info(self, worker):
        with patch.object(worker, "_web_search", new_callable=AsyncMock, return_value="phone: 02-2771-1234"):
            result = await worker.execute(
                "幫我訂 Niku Mura 明天晚上7點 2位",
                task_type="booking",
                task_context={
                    "restaurant": "Niku Mura",
                    "date": "2026-02-17",
                    "time": "19:00",
                    "people": "2",
                },
            )
        msg = result["result"]
        assert "Niku Mura" in msg
        assert "電話" in msg
        assert "地址" in msg
        assert "2 位" in msg

    @pytest.mark.asyncio
    async def test_booking_includes_options(self, worker):
        with patch.object(worker, "_web_search", new_callable=AsyncMock, return_value="test"):
            result = await worker.execute(
                "幫我訂鼎泰豐", task_type="booking",
                task_context={"restaurant": "鼎泰豐"},
            )
        msg = result["result"]
        # Should have at least one option
        assert "A." in msg or "B." in msg

    @pytest.mark.asyncio
    async def test_booking_with_booking_url(self, worker):
        with patch.object(worker, "_web_search", new_callable=AsyncMock, return_value="test"):
            result = await worker.execute(
                "幫我訂鼎泰豐", task_type="booking",
                task_context={"restaurant": "鼎泰豐"},
            )
        # LLM mock returns booking_url
        assert result.get("booking_url") == "https://inline.app/booking/test"

    @pytest.mark.asyncio
    async def test_booking_extracts_restaurant_from_task(self, worker):
        with patch.object(worker, "_web_search", new_callable=AsyncMock, return_value="test"):
            result = await worker.execute(
                "幫我訂明天晚上Niku Mura 7點2位", task_type="booking",
            )
        assert "Niku Mura" in result["result"]

    @pytest.mark.asyncio
    async def test_booking_phone_in_result(self, worker):
        with patch.object(worker, "_web_search", new_callable=AsyncMock, return_value="test"):
            result = await worker.execute(
                "幫我訂鼎泰豐", task_type="booking",
                task_context={"restaurant": "鼎泰豐"},
            )
        assert result.get("phone") == "02-2771-1234"

    @pytest.mark.asyncio
    async def test_booking_no_router(self, worker_no_deps):
        with patch.object(worker_no_deps, "_web_search", new_callable=AsyncMock, return_value=""):
            result = await worker_no_deps.execute(
                "幫我訂鼎泰豐", task_type="booking",
                task_context={"restaurant": "鼎泰豐"},
            )
        # Should still return partial result, just with empty extracted info
        assert result["is_partial"] is True
        assert "未找到" in result["result"]

    @pytest.mark.asyncio
    async def test_booking_with_failed_attempts(self, worker):
        with patch.object(worker, "_web_search", new_callable=AsyncMock, return_value="test"):
            result = await worker.execute(
                "幫我訂鼎泰豐", task_type="booking",
                task_context={"restaurant": "鼎泰豐"},
                failed_attempts=[
                    {"worker": "browser", "error": "timeout"},
                ],
            )
        assert result["is_partial"] is True


# ── Ticket assist tests ─────────────────────────────────────────


class TestAssistTicket:
    @pytest.mark.asyncio
    async def test_ticket_returns_partial(self, worker):
        with patch.object(worker, "_web_search", new_callable=AsyncMock, return_value="高鐵 1330 班次"):
            result = await worker.execute(
                "幫我買明天台北到高雄的高鐵票", task_type="ticket",
            )
        assert result["is_partial"] is True
        assert result["source"] == "assist"

    @pytest.mark.asyncio
    async def test_ticket_mentions_failed_methods(self, worker):
        with patch.object(worker, "_web_search", new_callable=AsyncMock, return_value=""):
            result = await worker.execute(
                "買高鐵票", task_type="ticket",
                failed_attempts=[
                    {"worker": "browser", "error": "captcha"},
                ],
            )
        # LLM is called with "browser" in the prompt
        call_args = worker.router.chat.call_args
        prompt = call_args[0][0][0].content
        assert "browser" in prompt


# ── Generic assist tests ─────────────────────────────────────────


class TestAssistGeneric:
    @pytest.mark.asyncio
    async def test_generic_returns_partial(self, worker):
        with patch.object(worker, "_web_search", new_callable=AsyncMock, return_value=""):
            result = await worker.execute("幫我做一件事")
        assert result["is_partial"] is True

    @pytest.mark.asyncio
    async def test_generic_no_router_fallback(self, worker_no_deps):
        with patch.object(worker_no_deps, "_web_search", new_callable=AsyncMock, return_value=""):
            result = await worker_no_deps.execute("幫我做一件事")
        assert "暫時無法處理" in result["result"]


# ── Helper tests ─────────────────────────────────────────────────


class TestHelpers:
    def test_extract_restaurant_simple(self):
        w = AssistWorker()
        assert "Niku Mura" in w._extract_restaurant("幫我訂Niku Mura")

    def test_extract_restaurant_with_time(self):
        w = AssistWorker()
        result = w._extract_restaurant("幫我訂明天晚上鼎泰豐 7點2位")
        assert "鼎泰豐" in result

    def test_search_memory_no_search(self):
        w = AssistWorker()
        assert w._search_memory("test") == ""

    def test_search_memory_with_search(self, mock_memory_search):
        w = AssistWorker(memory_search=mock_memory_search)
        result = w._search_memory("Niku Mura")
        assert "Niku Mura" in result

    def test_search_memory_exception(self):
        search = MagicMock()
        search.search.side_effect = RuntimeError("boom")
        w = AssistWorker(memory_search=search)
        assert w._search_memory("test") == ""

    @pytest.mark.asyncio
    async def test_extract_info_no_router(self):
        w = AssistWorker()
        result = await w._extract_info("test", "", "")
        assert result == {}

    @pytest.mark.asyncio
    async def test_extract_info_json_parse_error(self, mock_router):
        mock_router.chat = AsyncMock(
            return_value=ChatResponse(content="not json", model="test")
        )
        w = AssistWorker(model_router=mock_router)
        result = await w._extract_info("test", "search", "memory")
        assert result == {}

    @pytest.mark.asyncio
    async def test_extract_info_with_markdown_code_block(self, mock_router):
        mock_router.chat = AsyncMock(
            return_value=ChatResponse(
                content='```json\n{"phone": "02-1234-5678"}\n```',
                model="test",
            )
        )
        w = AssistWorker(model_router=mock_router)
        result = await w._extract_info("test", "search", "memory")
        assert result["phone"] == "02-1234-5678"

    @pytest.mark.asyncio
    async def test_llm_generate_no_router(self):
        w = AssistWorker()
        result = await w._llm_generate("test prompt")
        assert "暫時無法處理" in result

    @pytest.mark.asyncio
    async def test_llm_generate_success(self, mock_router):
        mock_router.chat = AsyncMock(
            return_value=ChatResponse(content="Sir, 我查到了", model="test")
        )
        w = AssistWorker(model_router=mock_router)
        result = await w._llm_generate("test prompt")
        assert "查到了" in result

    @pytest.mark.asyncio
    async def test_web_search_error(self):
        """Web search should gracefully handle errors."""
        w = AssistWorker()
        # Will fail because DuckDuckGo may not be reachable in tests, but shouldn't raise
        result = await w._web_search("test query that wont work")
        assert isinstance(result, str)  # Empty string on failure is OK

    @pytest.mark.asyncio
    async def test_close(self):
        w = AssistWorker()
        await w.close()  # Should not raise


# ── ReactExecutor integration tests ──────────────────────────────


class TestReactExecutorIntegration:
    @pytest.mark.asyncio
    async def test_booking_chain_includes_assist(self):
        from core.react_executor import FALLBACK_CHAINS
        assert "assist" in FALLBACK_CHAINS["booking"]

    @pytest.mark.asyncio
    async def test_ticket_chain_includes_assist(self):
        from core.react_executor import FALLBACK_CHAINS
        assert "assist" in FALLBACK_CHAINS["ticket"]

    @pytest.mark.asyncio
    async def test_assist_gets_task_type(self):
        """When ReactExecutor calls assist, it should pass task_type."""
        from core.react_executor import ReactExecutor, FuseState

        mock_browser = MagicMock()
        mock_browser.execute = AsyncMock(return_value={"error": "timeout"})

        mock_assist = MagicMock()
        mock_assist.execute = AsyncMock(return_value={
            "result": "partial result",
            "is_partial": True,
            "source": "assist",
        })

        executor = ReactExecutor(
            workers={"browser": mock_browser, "assist": mock_assist},
            fuse=FuseState(),
        )
        result = await executor.execute("booking", "幫我訂鼎泰豐")
        assert result.success is True

        # Verify assist was called with task_type
        call_kwargs = mock_assist.execute.call_args
        assert call_kwargs.kwargs.get("task_type") == "booking" or \
               (len(call_kwargs) > 1 and call_kwargs[1].get("task_type") == "booking")

    @pytest.mark.asyncio
    async def test_assist_gets_failed_attempts(self):
        """AssistWorker should receive failed_attempts from prior workers."""
        from core.react_executor import ReactExecutor, FuseState

        mock_browser = MagicMock()
        mock_browser.execute = AsyncMock(return_value={"error": "captcha detected"})

        mock_assist = MagicMock()
        mock_assist.execute = AsyncMock(return_value={
            "result": "partial result",
            "is_partial": True,
        })

        executor = ReactExecutor(
            workers={"browser": mock_browser, "assist": mock_assist},
            fuse=FuseState(),
        )
        await executor.execute("booking", "訂鼎泰豐")

        call_kwargs = mock_assist.execute.call_args
        failed = call_kwargs.kwargs.get("failed_attempts") or \
                 (call_kwargs[1].get("failed_attempts") if len(call_kwargs) > 1 else [])
        assert len(failed) >= 1
        assert failed[0]["worker"] == "browser"

    @pytest.mark.asyncio
    async def test_booking_falls_through_to_assist(self):
        """If browser fails, booking chain should fall through to assist."""
        from core.react_executor import ReactExecutor, FuseState

        mock_browser = MagicMock()
        # Use captcha error (no retry) so it falls through immediately
        mock_browser.execute = AsyncMock(return_value={"error": "captcha detected"})

        mock_assist = MagicMock()
        mock_assist.execute = AsyncMock(return_value={
            "result": "Sir，我幫你查好了",
            "is_partial": True,
        })

        executor = ReactExecutor(
            workers={"browser": mock_browser, "assist": mock_assist},
            fuse=FuseState(),
        )
        result = await executor.execute("booking", "幫我訂鼎泰豐")
        assert result.success is True
        assert result.result.get("is_partial") is True
        mock_assist.execute.assert_awaited_once()
