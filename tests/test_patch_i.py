"""Tests for Patch I: multi-task architecture modules."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# ── I3: ConversationCompressor ──────────────────────────────────


class TestConversationCompressor:
    def test_add_turn_stores_history(self):
        from core.conversation_compressor import ConversationCompressor

        cc = ConversationCompressor(recent_turns_keep=5)
        cc.add_turn("user", "你好")
        cc.add_turn("assistant", "你好！")
        assert len(cc.full_history) == 2

    def test_no_compression_under_limit(self):
        from core.conversation_compressor import ConversationCompressor

        cc = ConversationCompressor(recent_turns_keep=5)
        for i in range(10):
            cc.add_turn("user" if i % 2 == 0 else "assistant", f"msg {i}")
        # 10 msgs = 5 pairs = exactly at limit, no compression
        assert len(cc.compressed_summary) == 0
        assert len(cc.full_history) == 10

    def test_compression_triggered(self):
        from core.conversation_compressor import ConversationCompressor

        cc = ConversationCompressor(recent_turns_keep=3)
        # Add 10 messages (5 pairs) — keep 3 pairs = 6, compress 4
        for i in range(10):
            cc.add_turn("user" if i % 2 == 0 else "assistant", f"msg {i}")
        assert len(cc.full_history) == 6  # recent 3 pairs
        assert len(cc.compressed_summary) >= 1

    def test_get_context_with_summary(self):
        from core.conversation_compressor import ConversationCompressor

        cc = ConversationCompressor(recent_turns_keep=2)
        for i in range(12):
            cc.add_turn("user" if i % 2 == 0 else "assistant", f"message {i}")

        ctx = cc.get_context_for_ceo()
        # First message should be a system summary
        assert ctx[0]["role"] == "system"
        assert "先前對話摘要" in ctx[0]["content"]
        # Recent messages follow
        assert len(ctx) >= 5  # 1 summary + 4 recent msgs

    def test_get_context_no_summary(self):
        from core.conversation_compressor import ConversationCompressor

        cc = ConversationCompressor(recent_turns_keep=5)
        cc.add_turn("user", "hi")
        cc.add_turn("assistant", "hello")

        ctx = cc.get_context_for_ceo()
        # No summary, just the 2 messages
        assert len(ctx) == 2
        assert ctx[0]["role"] == "user"

    def test_max_summary_lines_cap(self):
        from core.conversation_compressor import ConversationCompressor

        cc = ConversationCompressor(recent_turns_keep=1, max_summary_lines=3)
        for i in range(20):
            cc.add_turn("user" if i % 2 == 0 else "assistant", f"msg {i}")
        assert len(cc.compressed_summary) <= 3

    def test_reset_clears_all(self):
        from core.conversation_compressor import ConversationCompressor

        cc = ConversationCompressor(recent_turns_keep=2)
        for i in range(10):
            cc.add_turn("user" if i % 2 == 0 else "assistant", f"msg {i}")
        cc.reset()
        assert len(cc.full_history) == 0
        assert len(cc.compressed_summary) == 0

    def test_turn_count(self):
        from core.conversation_compressor import ConversationCompressor

        cc = ConversationCompressor(recent_turns_keep=2)
        for i in range(8):
            cc.add_turn("user" if i % 2 == 0 else "assistant", f"msg {i}")
        assert cc.turn_count > 0


# ── I2: TaskRouter ──────────────────────────────────────────────


class TestTaskRouter:
    def test_weather_detection(self):
        from core.task_router import TaskRouter

        router = TaskRouter()
        tasks = router.classify("今天天氣怎麼樣？")
        assert any(t.task_type == "weather" for t in tasks)
        assert not tasks[0].needs_llm

    def test_calendar_detection(self):
        from core.task_router import TaskRouter

        router = TaskRouter()
        tasks = router.classify("我今天有什麼行程？")
        assert any(t.task_type == "calendar" for t in tasks)

    def test_selfie_detection(self):
        from core.task_router import TaskRouter

        router = TaskRouter()
        tasks = router.classify("幫我拍張自拍")
        assert any(t.task_type == "selfie" for t in tasks)

    def test_web_search_detection(self):
        from core.task_router import TaskRouter

        router = TaskRouter()
        tasks = router.classify("幫我查高鐵票")
        assert any(t.task_type == "web_search" for t in tasks)
        assert tasks[0].needs_llm

    def test_url_detection(self):
        from core.task_router import TaskRouter

        router = TaskRouter()
        tasks = router.classify("打開 https://google.com")
        assert any(t.task_type == "web_browse" for t in tasks)

    def test_conversation_fallback(self):
        from core.task_router import TaskRouter

        router = TaskRouter()
        tasks = router.classify("你今天過得怎麼樣？")
        assert len(tasks) == 1
        assert tasks[0].task_type == "conversation"
        assert tasks[0].needs_llm

    def test_multi_task_detection(self):
        from core.task_router import TaskRouter

        router = TaskRouter()
        tasks = router.classify("查天氣順便看行程")
        types = {t.task_type for t in tasks}
        assert "weather" in types
        assert "calendar" in types

    def test_build_ceo_context(self):
        from core.task_router import TaskRouter, RoutedTask

        tasks = [
            RoutedTask(task_type="weather", worker="weather", needs_llm=False, text="天氣"),
        ]
        results = [{"success": True, "summary": "今日 18~25°C"}]
        ctx = TaskRouter.build_ceo_context(tasks, results)
        assert "weather" in ctx
        assert "18~25" in ctx

    def test_build_ceo_context_failure(self):
        from core.task_router import TaskRouter, RoutedTask

        tasks = [
            RoutedTask(task_type="calendar", worker="calendar", needs_llm=False, text="行程"),
        ]
        results = [{"success": False, "error": "no calendar configured"}]
        ctx = TaskRouter.build_ceo_context(tasks, results)
        assert "失敗" in ctx


# ── I1: ParallelDispatcher ──────────────────────────────────────


class TestParallelDispatcher:
    @pytest.mark.asyncio
    async def test_dispatch_parallel(self):
        from core.parallel_dispatcher import ParallelDispatcher
        from core.task_router import RoutedTask

        mock_weather = AsyncMock()
        mock_weather.execute.return_value = {"success": True, "content": "晴天"}

        mock_calendar = AsyncMock()
        mock_calendar.execute.return_value = {"success": True, "content": "10:00 會議"}

        dispatcher = ParallelDispatcher(workers={
            "weather": mock_weather,
            "calendar": mock_calendar,
        })

        tasks = [
            RoutedTask(task_type="weather", worker="weather", needs_llm=False, text="天氣"),
            RoutedTask(task_type="calendar", worker="calendar", needs_llm=False, text="行程"),
        ]

        results = await dispatcher.dispatch(tasks)
        assert len(results) == 2
        assert results[0]["success"]
        assert results[1]["success"]

    @pytest.mark.asyncio
    async def test_one_failure_doesnt_block_others(self):
        from core.parallel_dispatcher import ParallelDispatcher
        from core.task_router import RoutedTask

        mock_ok = AsyncMock()
        mock_ok.execute.return_value = {"success": True, "content": "ok"}

        mock_fail = AsyncMock()
        mock_fail.execute.side_effect = Exception("boom")

        dispatcher = ParallelDispatcher(workers={
            "weather": mock_ok,
            "browser": mock_fail,
        })

        tasks = [
            RoutedTask(task_type="weather", worker="weather", needs_llm=False, text="天氣"),
            RoutedTask(task_type="web_search", worker="browser", needs_llm=True, text="查東西"),
        ]

        results = await dispatcher.dispatch(tasks)
        assert results[0]["success"]
        assert not results[1]["success"]
        assert "boom" in results[1]["error"]

    @pytest.mark.asyncio
    async def test_no_worker_returns_needs_llm(self):
        from core.parallel_dispatcher import ParallelDispatcher
        from core.task_router import RoutedTask

        dispatcher = ParallelDispatcher(workers={})
        tasks = [
            RoutedTask(task_type="conversation", worker=None, needs_llm=True, text="hi"),
        ]
        results = await dispatcher.dispatch(tasks)
        assert results[0]["needs_llm"]

    @pytest.mark.asyncio
    async def test_sequential_tasks(self):
        from core.parallel_dispatcher import ParallelDispatcher
        from core.task_router import RoutedTask

        mock_w = AsyncMock()
        mock_w.execute.return_value = {"success": True}

        dispatcher = ParallelDispatcher(workers={"code": mock_w})
        tasks = [
            RoutedTask(task_type="code", worker="code", needs_llm=True, text="build", depends_on="setup"),
        ]
        results = await dispatcher.dispatch(tasks)
        assert results[0]["success"]


# ── I6: HelpDecisionEngine ──────────────────────────────────────


class TestHelpDecisionEngine:
    def test_login_required_asks_human(self):
        from core.help_decision import HelpDecisionEngine

        assert HelpDecisionEngine.decide("login_required", 0) == "ask_human"

    def test_captcha_asks_human(self):
        from core.help_decision import HelpDecisionEngine

        assert HelpDecisionEngine.decide("captcha_detected", 0) == "ask_human"

    def test_payment_asks_human(self):
        from core.help_decision import HelpDecisionEngine

        assert HelpDecisionEngine.decide("payment_required", 0) == "ask_human"

    def test_network_retries(self):
        from core.help_decision import HelpDecisionEngine

        assert HelpDecisionEngine.decide("network_unreachable", 0) == "retry"
        assert HelpDecisionEngine.decide("network_unreachable", 2) == "retry"

    def test_network_exhausted_asks_human(self):
        from core.help_decision import HelpDecisionEngine

        assert HelpDecisionEngine.decide("network_unreachable", 3) == "ask_human"

    def test_give_up_on_impossible(self):
        from core.help_decision import HelpDecisionEngine

        assert HelpDecisionEngine.decide("site_not_accessible") == "give_up"
        assert HelpDecisionEngine.decide("service_discontinued") == "give_up"
        assert HelpDecisionEngine.decide("region_blocked") == "give_up"

    def test_unknown_retry_then_ask(self):
        from core.help_decision import HelpDecisionEngine

        assert HelpDecisionEngine.decide("something_new", 0) == "retry"
        assert HelpDecisionEngine.decide("something_new", 1) == "ask_human"

    def test_get_message_login(self):
        from core.help_decision import HelpDecisionEngine

        msg = HelpDecisionEngine.get_message("login_required", site_name="高鐵")
        assert "Sir" in msg
        assert "高鐵" in msg

    def test_get_message_give_up(self):
        from core.help_decision import HelpDecisionEngine

        msg = HelpDecisionEngine.get_message("site_not_accessible", attempts=3)
        assert "Sir" in msg

    def test_provider_down_gives_up(self):
        from core.help_decision import HelpDecisionEngine

        assert HelpDecisionEngine.decide("provider_down") == "give_up"


# ── I4: BackgroundTaskManager ───────────────────────────────────


class TestBackgroundTaskManager:
    @pytest.mark.asyncio
    async def test_run_in_background(self):
        from core.background_tasks import BackgroundTaskManager

        sent = []

        async def mock_send(chat_id, text):
            sent.append(text)

        mgr = BackgroundTaskManager(send_fn=mock_send)

        async def slow_task():
            return {"content": "查詢完成"}

        tid = await mgr.run_in_background(
            slow_task(), chat_id=123, immediate_reply="正在查詢...",
        )
        assert tid
        # Immediate reply sent
        assert "正在查詢" in sent[0]
        # Wait for background task
        await mgr.wait_all()
        assert any("查詢完成" in s for s in sent)

    @pytest.mark.asyncio
    async def test_background_task_failure(self):
        from core.background_tasks import BackgroundTaskManager

        sent = []

        async def mock_send(chat_id, text):
            sent.append(text)

        mgr = BackgroundTaskManager(send_fn=mock_send)

        async def failing_task():
            raise RuntimeError("boom")

        await mgr.run_in_background(failing_task(), chat_id=123)
        await mgr.wait_all()
        assert any("異常" in s for s in sent)

    @pytest.mark.asyncio
    async def test_cancel_task(self):
        from core.background_tasks import BackgroundTaskManager

        mgr = BackgroundTaskManager()

        async def forever():
            await asyncio.sleep(9999)

        tid = await mgr.run_in_background(forever())
        assert mgr.active_count >= 1
        assert mgr.cancel(tid)
        await asyncio.sleep(0.05)

    @pytest.mark.asyncio
    async def test_active_count(self):
        from core.background_tasks import BackgroundTaskManager

        mgr = BackgroundTaskManager()
        assert mgr.active_count == 0

    @pytest.mark.asyncio
    async def test_no_send_fn(self):
        from core.background_tasks import BackgroundTaskManager

        mgr = BackgroundTaskManager(send_fn=None)

        async def task():
            return "done"

        tid = await mgr.run_in_background(task())
        await mgr.wait_all()
        assert tid


# ── I5: SessionManager ──────────────────────────────────────────


class TestSessionManager:
    def test_defaults_not_logged_in(self, tmp_path):
        from core.session_manager import SessionManager

        mgr = SessionManager(status_path=str(tmp_path / "status.json"))
        assert not mgr.is_logged_in("thsrc")

    def test_mark_logged_in(self, tmp_path):
        from core.session_manager import SessionManager

        mgr = SessionManager(status_path=str(tmp_path / "status.json"))
        mgr.mark_logged_in("thsrc")
        assert mgr.is_logged_in("thsrc")

    def test_mark_expired(self, tmp_path):
        from core.session_manager import SessionManager

        mgr = SessionManager(status_path=str(tmp_path / "status.json"))
        mgr.mark_logged_in("thsrc")
        mgr.mark_expired("thsrc")
        assert not mgr.is_logged_in("thsrc")

    def test_persistence(self, tmp_path):
        from core.session_manager import SessionManager

        path = str(tmp_path / "status.json")
        mgr = SessionManager(status_path=path)
        mgr.mark_logged_in("google")

        mgr2 = SessionManager(status_path=path)
        assert mgr2.is_logged_in("google")

    def test_get_site_name_known(self):
        from core.session_manager import SessionManager

        mgr = SessionManager(status_path="nonexistent.json")
        assert mgr.get_site_name("thsrc") == "台灣高鐵"

    def test_get_site_name_unknown(self):
        from core.session_manager import SessionManager

        mgr = SessionManager(status_path="nonexistent.json")
        assert mgr.get_site_name("random_site") == "random_site"

    def test_get_login_url(self):
        from core.session_manager import SessionManager

        mgr = SessionManager(status_path="nonexistent.json")
        assert "thsrc" in mgr.get_login_url("thsrc")
        assert mgr.get_login_url("unknown") is None


# ── I5: LoginAssistant ──────────────────────────────────────────


class TestLoginAssistant:
    @pytest.mark.asyncio
    async def test_handle_known_site(self, tmp_path):
        from core.login_assistant import LoginAssistant
        from core.session_manager import SessionManager

        sm = SessionManager(status_path=str(tmp_path / "s.json"))
        la = LoginAssistant(session_manager=sm)

        mock_tg = AsyncMock()
        mock_tg.send_message = AsyncMock()

        result = await la.handle_login_required(
            "thsrc", telegram_client=mock_tg, chat_id=123,
        )
        assert result["status"] == "waiting_for_login"
        assert result["site_key"] == "thsrc"
        mock_tg.send_message.assert_called_once()

    @pytest.mark.asyncio
    async def test_handle_unknown_site(self, tmp_path):
        from core.login_assistant import LoginAssistant
        from core.session_manager import SessionManager

        sm = SessionManager(status_path=str(tmp_path / "s.json"))
        la = LoginAssistant(session_manager=sm)

        result = await la.handle_login_required("unknown_site")
        assert result["status"] == "needs_human_help"

    @pytest.mark.asyncio
    async def test_confirm_login(self, tmp_path):
        from core.login_assistant import LoginAssistant
        from core.session_manager import SessionManager

        sm = SessionManager(status_path=str(tmp_path / "s.json"))
        la = LoginAssistant(session_manager=sm)

        msg = await la.on_user_confirms_login("thsrc")
        assert "確認" in msg
        assert sm.is_logged_in("thsrc")

    def test_detect_site_from_url(self, tmp_path):
        from core.login_assistant import LoginAssistant
        from core.session_manager import SessionManager

        sm = SessionManager(status_path=str(tmp_path / "s.json"))
        la = LoginAssistant(session_manager=sm)

        assert la.detect_site_from_url("https://irs.thsrc.com.tw/foo") == "thsrc"
        assert la.detect_site_from_url("https://random.com") is None

    def test_detect_login_confirmation(self, tmp_path):
        from core.login_assistant import LoginAssistant
        from core.session_manager import SessionManager

        sm = SessionManager(status_path=str(tmp_path / "s.json"))
        # Create an expired session to detect
        sm._status["thsrc"] = {"logged_in": False}
        sm._save()

        la = LoginAssistant(session_manager=sm)
        assert la.detect_login_confirmation("登好了") == "thsrc"
        assert la.detect_login_confirmation("你好") is None
