"""Tests for Heartbeat, SurvivalGate, and TelegramClient.

Run: pytest tests/test_heartbeat.py -v
"""

from __future__ import annotations

import asyncio
import time
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from clients.base_client import ChatMessage, ChatResponse
from core.heartbeat import Heartbeat
from core.survival_gate import CheckResult, HealthReport, SurvivalGate
from memory.memos_manager import MemOS
from memory.token_tracker import TokenSavingTracker


# ── Fixtures ────────────────────────────────────────────────────


@pytest.fixture
async def memos(tmp_path):
    db_path = str(tmp_path / "test.db")
    m = MemOS(db_path)
    await m.init()
    yield m
    await m.close()


@pytest.fixture
async def tracker(memos):
    t = TokenSavingTracker(memos._db)
    await t.init()
    return t


@pytest.fixture
def mock_router():
    router = MagicMock()
    router.chat = AsyncMock(return_value=ChatResponse(content="test response", model="kimi"))
    router.health_check_all = AsyncMock(return_value={
        "nvidia": True, "zhipu": True, "openrouter": True,
    })
    router.probe_recovery = AsyncMock(return_value={})
    router.openrouter = MagicMock()
    router.openrouter.get_remaining_credits = AsyncMock(return_value=5.0)
    return router


@pytest.fixture
def mock_telegram():
    tg = MagicMock()
    tg.send = AsyncMock(return_value=1)
    return tg


# ── HealthReport Tests ──────────────────────────────────────────


class TestHealthReport:
    def test_empty_report(self):
        report = HealthReport()
        assert not report.has_alerts
        assert "Health Report" in report.format()

    def test_report_with_alerts(self):
        report = HealthReport()
        report.add(CheckResult("API: nvidia", "ok", "reachable"))
        report.add(CheckResult("Disk", "warning", "1.5GB free"))
        report.alert("⚠️ 磁碟空間不足")
        assert report.has_alerts
        formatted = report.format()
        assert "nvidia" in formatted
        assert "磁碟" in formatted

    def test_format_icons(self):
        report = HealthReport()
        report.add(CheckResult("Test OK", "ok"))
        report.add(CheckResult("Test Warn", "warning"))
        report.add(CheckResult("Test Crit", "critical"))
        formatted = report.format()
        assert "✅" in formatted
        assert "⚠️" in formatted
        assert "🔴" in formatted


# ── SurvivalGate Tests ──────────────────────────────────────────


class TestSurvivalGate:
    @pytest.mark.asyncio
    async def test_full_check_all_healthy(self, mock_router, tracker, tmp_path):
        # Create a fake backup
        backup_dir = tmp_path / "backups"
        backup_dir.mkdir()
        (backup_dir / "latest.db").write_bytes(b"fake backup data")

        gate = SurvivalGate(
            model_router=mock_router,
            token_tracker=tracker,
            backup_dir=str(backup_dir),
            project_root=str(tmp_path),
        )
        report = await gate.full_check()

        assert len(report.checks) >= 4  # API, quota, token, backup, disk
        # All APIs healthy → no alerts from API check
        api_checks = [c for c in report.checks if c.name.startswith("API:")]
        assert all(c.status == "ok" for c in api_checks)

    @pytest.mark.asyncio
    async def test_api_down_triggers_alert(self, mock_router, tracker, tmp_path):
        mock_router.health_check_all = AsyncMock(return_value={
            "nvidia": False, "zhipu": True, "openrouter": True,
        })
        gate = SurvivalGate(
            model_router=mock_router,
            token_tracker=tracker,
            backup_dir=str(tmp_path / "backups"),
            project_root=str(tmp_path),
        )
        report = await gate.full_check()
        assert report.has_alerts
        assert any("nvidia" in a for a in report.alerts)

    @pytest.mark.asyncio
    async def test_low_openrouter_balance_alert(self, mock_router, tracker, tmp_path):
        mock_router.openrouter.get_remaining_credits = AsyncMock(return_value=0.50)
        gate = SurvivalGate(
            model_router=mock_router,
            token_tracker=tracker,
            backup_dir=str(tmp_path / "backups"),
            project_root=str(tmp_path),
        )
        report = await gate.full_check()
        assert any("充值" in a or "OpenRouter" in a for a in report.alerts)

    @pytest.mark.asyncio
    async def test_missing_backup_alert(self, mock_router, tracker, tmp_path):
        gate = SurvivalGate(
            model_router=mock_router,
            token_tracker=tracker,
            backup_dir=str(tmp_path / "nonexistent_backups"),
            project_root=str(tmp_path),
        )
        report = await gate.full_check()
        assert any("備份" in a for a in report.alerts)

    @pytest.mark.asyncio
    async def test_token_saving_alert(self, mock_router, tracker, tmp_path):
        # Record low saving rate data
        await tracker.record("c1", raw_tokens=1000, memos_tokens=900)  # 10% saving
        gate = SurvivalGate(
            model_router=mock_router,
            token_tracker=tracker,
            backup_dir=str(tmp_path / "backups"),
            project_root=str(tmp_path),
        )
        report = await gate.full_check()
        assert any("節省率" in a or "50%" in a for a in report.alerts)

    @pytest.mark.asyncio
    async def test_disk_check_runs(self, tmp_path):
        gate = SurvivalGate(project_root=str(tmp_path))
        report = HealthReport()
        gate._check_disk(report)
        disk_checks = [c for c in report.checks if c.name == "Disk"]
        assert len(disk_checks) == 1

    @pytest.mark.asyncio
    async def test_no_dependencies_still_works(self):
        """SurvivalGate should work even with no router/tracker."""
        gate = SurvivalGate()
        report = await gate.full_check()
        assert len(report.checks) >= 2  # At least disk + memory


# ── Heartbeat Tests ─────────────────────────────────────────────


class TestHeartbeat:
    @pytest.mark.asyncio
    async def test_start_and_stop(self):
        hb = Heartbeat()
        hb.start()
        assert hb.is_running
        jobs = hb.get_jobs()
        # patrol, brief, health, backup, night_owl, evening_summary, memory_cleanup
        assert len(jobs) == 7
        hb.stop()
        assert not hb.is_running

    @pytest.mark.asyncio
    async def test_job_ids(self):
        hb = Heartbeat()
        hb.start()
        job_ids = [j["id"] for j in hb.get_jobs()]
        assert "hourly_patrol" in job_ids
        assert "morning_brief" in job_ids
        assert "health_check" in job_ids
        assert "nightly_backup" in job_ids
        assert "night_owl" in job_ids
        assert "evening_summary" in job_ids
        assert "memory_cleanup" in job_ids
        hb.stop()

    @pytest.mark.asyncio
    async def test_custom_config(self):
        hb = Heartbeat(config={"heartbeat": {
            "hourly_patrol_interval_minutes": 30,
            "morning_brief_time": "08:00",
            "health_check_interval_hours": 12,
            "backup_time": "04:30",
        }})
        hb.start()
        assert hb.is_running
        hb.stop()


class TestHourlyPatrol:
    @pytest.mark.asyncio
    async def test_patrol_skips_without_deps(self):
        hb = Heartbeat()
        result = await hb.hourly_patrol()
        assert result["action"] == "none"

    @pytest.mark.asyncio
    async def test_patrol_detects_negative_emotion(self, memos, mock_router, mock_telegram):
        await memos.working_memory.set("user_emotion", "tired", agent_id="test")

        hb = Heartbeat(
            model_router=mock_router,
            memos=memos,
            telegram=mock_telegram,
        )
        result = await hb.hourly_patrol()
        assert result["action"] == "sent_caring_message"
        mock_telegram.send.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_patrol_no_outreach_for_happy(self, memos, mock_router, mock_telegram):
        await memos.working_memory.set("user_emotion", "happy", agent_id="test")

        hb = Heartbeat(
            model_router=mock_router,
            memos=memos,
            telegram=mock_telegram,
        )
        result = await hb.hourly_patrol()
        assert result["action"] == "none"

    @pytest.mark.asyncio
    async def test_patrol_updates_timestamp(self, memos, mock_router):
        await memos.working_memory.set("user_emotion", "normal", agent_id="test")
        hb = Heartbeat(model_router=mock_router, memos=memos)
        await hb.hourly_patrol()

        ts = await memos.working_memory.get("last_patrol")
        assert ts is not None
        assert time.time() - ts < 5


class TestMorningBrief:
    @pytest.mark.asyncio
    async def test_brief_format(self, memos, mock_telegram):
        hb = Heartbeat(memos=memos, telegram=mock_telegram)
        brief = await hb.morning_brief()
        assert "早安" in brief
        assert "行程" in brief or "沒有排定" in brief
        mock_telegram.send.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_brief_with_calendar_events(self, memos, mock_telegram):
        from datetime import datetime
        today = datetime.now().strftime("%Y-%m-%d")
        await memos.working_memory.set("calendar_cache", [
            {"date": f"{today}T09:00", "summary": "Team standup"},
            {"date": f"{today}T14:00", "summary": "Code review"},
        ], agent_id="test")

        hb = Heartbeat(memos=memos, telegram=mock_telegram)
        brief = await hb.morning_brief()
        assert "Team standup" in brief
        assert "Code review" in brief

    @pytest.mark.asyncio
    async def test_brief_with_token_report(self, memos, tracker, mock_telegram):
        await tracker.record("c1", raw_tokens=5000, memos_tokens=1500)

        survival = SurvivalGate(token_tracker=tracker)
        hb = Heartbeat(memos=memos, telegram=mock_telegram, survival_gate=survival)
        brief = await hb.morning_brief()
        assert "Token" in brief or "節省" in brief


class TestHealthCheckJob:
    @pytest.mark.asyncio
    async def test_health_check_sends_on_alert(self, mock_router, mock_telegram, tracker, tmp_path):
        mock_router.health_check_all = AsyncMock(return_value={
            "nvidia": False, "zhipu": True, "openrouter": True,
        })
        survival = SurvivalGate(
            model_router=mock_router,
            token_tracker=tracker,
            backup_dir=str(tmp_path / "backups"),
            project_root=str(tmp_path),
        )
        hb = Heartbeat(
            model_router=mock_router,
            telegram=mock_telegram,
            survival_gate=survival,
        )
        result = await hb.health_check()
        assert result["status"] == "completed"
        assert result["alerts"] > 0
        mock_telegram.send.assert_awaited()

    @pytest.mark.asyncio
    async def test_health_check_no_alert_when_healthy(self, mock_router, mock_telegram, tracker, tmp_path):
        # Create backup dir with recent file
        backup_dir = tmp_path / "backups"
        backup_dir.mkdir()
        (backup_dir / "recent.db").write_bytes(b"data")

        survival = SurvivalGate(
            model_router=mock_router,
            token_tracker=tracker,
            backup_dir=str(backup_dir),
            project_root=str(tmp_path),
        )
        hb = Heartbeat(
            model_router=mock_router,
            telegram=mock_telegram,
            survival_gate=survival,
        )
        result = await hb.health_check()
        assert result["status"] == "completed"
        assert result["alerts"] == 0


class TestNightlyBackup:
    @pytest.mark.asyncio
    async def test_backup_creates_file(self, memos, mock_telegram):
        await memos.long_term.set("test", "key", "value")
        hb = Heartbeat(memos=memos, telegram=mock_telegram)
        result = await hb.nightly_backup()
        assert result is not None
        assert Path(result).exists()

    @pytest.mark.asyncio
    async def test_backup_skips_without_memos(self, mock_telegram):
        hb = Heartbeat(telegram=mock_telegram)
        result = await hb.nightly_backup()
        assert result is None


class TestNightOwl:
    @pytest.mark.asyncio
    async def test_no_activity_no_reminder(self, memos, mock_telegram):
        hb = Heartbeat(memos=memos, telegram=mock_telegram)
        result = await hb.night_owl_check()
        assert result is False

    @pytest.mark.asyncio
    async def test_recent_activity_triggers_reminder(self, memos, mock_telegram):
        """Only triggers if current hour is 00:00-04:59."""
        from datetime import datetime
        current_hour = datetime.now().hour

        await memos.working_memory.set("last_user_activity", time.time(), agent_id="test")

        hb = Heartbeat(memos=memos, telegram=mock_telegram)
        result = await hb.night_owl_check()

        if 0 <= current_hour < 5:
            assert result is True
            mock_telegram.send.assert_awaited()
        else:
            # Outside night hours, no reminder
            assert result is False


# ── TelegramClient Unit Tests ──────────────────────────────────


class TestTelegramClient:
    def test_import(self):
        from clients.telegram_client import TelegramClient
        client = TelegramClient()
        assert client.chat_id == 0

    def test_no_bot_returns_none(self):
        from clients.telegram_client import TelegramClient
        client = TelegramClient()
        assert client._get_bot("jarvis") is None
        assert client._get_bot("clawra") is None

    @pytest.mark.asyncio
    async def test_send_without_bot_returns_none(self):
        from clients.telegram_client import TelegramClient
        client = TelegramClient()
        result = await client.send("test")
        assert result is None

    @pytest.mark.asyncio
    async def test_close(self):
        from clients.telegram_client import TelegramClient
        client = TelegramClient()
        await client.close()  # Should not raise

    def test_whitelist_parsing(self):
        from clients.telegram_client import TelegramClient
        client = TelegramClient(allowed_user_ids="123,456,789")
        assert client._allowed_user_ids == {123, 456, 789}

    def test_whitelist_empty_allows_all(self):
        from clients.telegram_client import TelegramClient
        client = TelegramClient(allowed_user_ids="")
        assert client._is_authorized(99999) is True

    def test_whitelist_authorized_user(self):
        from clients.telegram_client import TelegramClient
        client = TelegramClient(allowed_user_ids="123,456")
        assert client._is_authorized(123) is True
        assert client._is_authorized(456) is True

    def test_whitelist_unauthorized_user(self):
        from clients.telegram_client import TelegramClient
        client = TelegramClient(allowed_user_ids="123,456")
        assert client._is_authorized(789) is False
        assert client._is_authorized(None) is False

    @pytest.mark.asyncio
    async def test_typing_delay_clawra_short_text(self):
        """Typing delay should be at least 12s (15 - 3 jitter)."""
        from clients.telegram_client import TelegramClient
        client = TelegramClient()
        mock_bot = AsyncMock()

        start = time.monotonic()
        # Patch asyncio.sleep to skip actual waiting
        with patch("clients.telegram_client.asyncio.sleep", new_callable=AsyncMock) as mock_sleep:
            await client._simulate_typing(mock_bot, 123, "短訊息")
            total_delay = sum(call.args[0] for call in mock_sleep.call_args_list)
            assert 12 <= total_delay <= 60
            mock_bot.send_chat_action.assert_called()

    @pytest.mark.asyncio
    async def test_typing_delay_clawra_long_text(self):
        """Long text should push delay toward 60s cap."""
        from clients.telegram_client import TelegramClient
        client = TelegramClient()
        mock_bot = AsyncMock()

        long_text = "很長的回覆" * 50  # 150 chars → 15 + 150*0.3 = 60, capped
        with patch("clients.telegram_client.asyncio.sleep", new_callable=AsyncMock) as mock_sleep:
            await client._simulate_typing(mock_bot, 123, long_text)
            total_delay = sum(call.args[0] for call in mock_sleep.call_args_list)
            assert 50 <= total_delay <= 63  # near cap with jitter

    @pytest.mark.asyncio
    async def test_typing_action_sent_periodically(self):
        """Chat action should be sent multiple times during delay."""
        from clients.telegram_client import TelegramClient
        client = TelegramClient()
        mock_bot = AsyncMock()

        # Medium text: ~30s delay → should send typing ~7-8 times
        medium_text = "中等長度" * 15  # 60 chars → 15 + 60*0.3 = 33s
        with patch("clients.telegram_client.asyncio.sleep", new_callable=AsyncMock):
            await client._simulate_typing(mock_bot, 123, medium_text)
            assert mock_bot.send_chat_action.call_count >= 4
            mock_bot.send_chat_action.assert_called_with(chat_id=123, action="typing")


# ── Pending Task Retry Tests ──────────────────────────────────────


class TestPendingTaskRetry:
    @pytest.fixture
    def mock_react(self):
        from core.react_executor import TaskResult
        react = AsyncMock()
        react.execute.return_value = TaskResult(success=True, result={"content": "ok"})
        return react

    @pytest.fixture
    def pending_mgr(self, tmp_path):
        from core.pending_tasks import PendingTaskManager
        return PendingTaskManager(str(tmp_path / "pending.json"))

    @pytest.mark.asyncio
    async def test_retry_succeeds(self, mock_telegram, mock_react, pending_mgr):
        tid = pending_mgr.add("web_search", "查天氣")
        hb = Heartbeat(
            telegram=mock_telegram,
            pending_tasks=pending_mgr,
            react_executor=mock_react,
        )
        result = await hb.retry_pending_tasks()
        assert result["retried"] == 1
        assert result["succeeded"] == 1
        mock_telegram.send.assert_called()

    @pytest.mark.asyncio
    async def test_retry_fails(self, mock_telegram, pending_mgr):
        from core.react_executor import TaskResult
        mock_react = AsyncMock()
        mock_react.execute.return_value = TaskResult(
            success=False, gave_up_reason="all_workers_exhausted",
        )
        tid = pending_mgr.add("web_search", "查天氣")
        hb = Heartbeat(
            telegram=mock_telegram,
            pending_tasks=pending_mgr,
            react_executor=mock_react,
        )
        result = await hb.retry_pending_tasks()
        assert result["retried"] == 1
        assert result["failed"] == 1

    @pytest.mark.asyncio
    async def test_retry_no_pending(self, mock_telegram, mock_react, pending_mgr):
        hb = Heartbeat(
            telegram=mock_telegram,
            pending_tasks=pending_mgr,
            react_executor=mock_react,
        )
        result = await hb.retry_pending_tasks()
        assert result["retried"] == 0

    @pytest.mark.asyncio
    async def test_retry_given_up_notifies(self, mock_telegram, pending_mgr):
        from core.react_executor import TaskResult
        mock_react = AsyncMock()
        mock_react.execute.return_value = TaskResult(
            success=False, gave_up_reason="all_workers_exhausted",
        )
        tid = pending_mgr.add("web_search", "查天氣")
        # Set retry_count to max_retries - 1 so next failure gives up
        pending_mgr._tasks[tid].retry_count = 2

        hb = Heartbeat(
            telegram=mock_telegram,
            pending_tasks=pending_mgr,
            react_executor=mock_react,
        )
        await hb.retry_pending_tasks()
        # Should have notified about given-up task
        calls = [str(c) for c in mock_telegram.send.call_args_list]
        assert any("放棄" in c for c in calls)

    @pytest.mark.asyncio
    async def test_heartbeat_has_pending_job(self, pending_mgr, mock_react):
        hb = Heartbeat(
            pending_tasks=pending_mgr,
            react_executor=mock_react,
        )
        hb.start()
        job_ids = [j["id"] for j in hb.get_jobs()]
        assert "pending_tasks" in job_ids
        hb.stop()

    @pytest.mark.asyncio
    async def test_heartbeat_no_pending_job_without_manager(self):
        hb = Heartbeat()
        hb.start()
        job_ids = [j["id"] for j in hb.get_jobs()]
        assert "pending_tasks" not in job_ids
        hb.stop()


# ── K1: Evening Summary Tests ──────────────────────────────────


class TestEveningSummary:
    @pytest.mark.asyncio
    async def test_evening_summary_format(self, mock_telegram):
        hb = Heartbeat(telegram=mock_telegram)
        summary = await hb.evening_summary()
        assert "晚安" in summary
        assert "明日" in summary or "記錄" in summary
        mock_telegram.send.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_evening_summary_without_telegram(self):
        hb = Heartbeat()
        summary = await hb.evening_summary()
        assert "晚安" in summary

    @pytest.mark.asyncio
    async def test_evening_summary_with_gog_events(self, mock_telegram):
        mock_gog = MagicMock()
        mock_gog.is_available = True
        mock_gog.get_events_for_date.return_value = [
            {"summary": "Morning meeting", "start": {"dateTime": "2026-02-17T09:00:00"}},
        ]
        hb = Heartbeat(telegram=mock_telegram, gog_worker=mock_gog)
        summary = await hb.evening_summary()
        assert "Morning meeting" in summary

    @pytest.mark.asyncio
    async def test_evening_summary_no_gog(self, mock_telegram):
        hb = Heartbeat(telegram=mock_telegram)
        summary = await hb.evening_summary()
        assert "行事曆未連線" in summary

    @pytest.mark.asyncio
    async def test_evening_summary_with_reminders(self, mock_telegram, tmp_path):
        from datetime import datetime, timedelta
        from core.reminder_manager import ReminderManager
        rm = ReminderManager(path=str(tmp_path / "rem.json"))
        tomorrow = datetime.now() + timedelta(days=1)
        tomorrow = tomorrow.replace(hour=10, minute=0, second=0, microsecond=0)
        await rm.add("Review PR", tomorrow)
        hb = Heartbeat(telegram=mock_telegram, reminder_manager=rm)
        summary = await hb.evening_summary()
        assert "Review PR" in summary

    @pytest.mark.asyncio
    async def test_evening_summary_config_time(self):
        hb = Heartbeat(config={"heartbeat": {"evening_summary_time": "22:00"}})
        hb.start()
        job_ids = [j["id"] for j in hb.get_jobs()]
        assert "evening_summary" in job_ids
        hb.stop()


# ── K1: Enhanced Morning Brief Tests ───────────────────────────


class TestMorningBriefGog:
    @pytest.mark.asyncio
    async def test_brief_with_gog_events(self, memos, mock_telegram):
        mock_gog = MagicMock()
        mock_gog.is_available = True
        mock_gog.get_today_events.return_value = [
            {"summary": "Standup", "start": {"dateTime": "2026-02-16T09:00:00"}},
            {"summary": "Code review", "start": {"dateTime": "2026-02-16T14:00:00"}},
        ]
        hb = Heartbeat(memos=memos, telegram=mock_telegram, gog_worker=mock_gog)
        brief = await hb.morning_brief()
        assert "Standup" in brief
        assert "Code review" in brief

    @pytest.mark.asyncio
    async def test_brief_gog_unavailable_falls_back(self, memos, mock_telegram):
        from datetime import datetime
        today = datetime.now().strftime("%Y-%m-%d")
        await memos.working_memory.set("calendar_cache", [
            {"date": f"{today}T09:00", "summary": "Cache event"},
        ], agent_id="test")

        hb = Heartbeat(memos=memos, telegram=mock_telegram)
        brief = await hb.morning_brief()
        assert "Cache event" in brief

    @pytest.mark.asyncio
    async def test_brief_no_gog_no_cache(self, memos, mock_telegram):
        hb = Heartbeat(memos=memos, telegram=mock_telegram)
        brief = await hb.morning_brief()
        assert "行事曆未連線" in brief

    @pytest.mark.asyncio
    async def test_brief_includes_reminders(self, memos, mock_telegram, tmp_path):
        from datetime import datetime, timedelta
        from core.reminder_manager import ReminderManager
        rm = ReminderManager(path=str(tmp_path / "rem.json"))
        today_10 = datetime.now().replace(hour=22, minute=0, second=0, microsecond=0)
        if today_10 < datetime.now():
            today_10 += timedelta(days=1)
        await rm.add("Check stocks", today_10)
        hb = Heartbeat(memos=memos, telegram=mock_telegram, reminder_manager=rm)
        brief = await hb.morning_brief()
        # Only passes if reminder is for today
        if today_10.date() == datetime.now().date():
            assert "Check stocks" in brief

    @pytest.mark.asyncio
    async def test_brief_trading_day(self, memos, mock_telegram):
        from datetime import datetime
        hb = Heartbeat(memos=memos, telegram=mock_telegram)
        brief = await hb.morning_brief()
        if datetime.now().weekday() < 5:
            assert "交易日" in brief


# ── K1: Enhanced Patrol Tests ──────────────────────────────────


class TestPatrolUpcoming:
    @pytest.mark.asyncio
    async def test_patrol_with_upcoming_events(self, memos, mock_router, mock_telegram):
        mock_gog = MagicMock()
        mock_gog.is_available = True
        mock_gog.get_upcoming_events.return_value = [
            {"summary": "Meeting", "start": {"dateTime": "2026-02-16T10:30:00"}},
        ]
        await memos.working_memory.set("user_emotion", "normal", agent_id="test")

        hb = Heartbeat(
            model_router=mock_router,
            memos=memos,
            telegram=mock_telegram,
            gog_worker=mock_gog,
        )
        result = await hb.hourly_patrol()
        assert result.get("upcoming_events", 0) >= 1

    @pytest.mark.asyncio
    async def test_patrol_no_gog(self, memos, mock_router, mock_telegram):
        await memos.working_memory.set("user_emotion", "normal", agent_id="test")
        hb = Heartbeat(model_router=mock_router, memos=memos, telegram=mock_telegram)
        result = await hb.hourly_patrol()
        assert result.get("upcoming_events", 0) == 0


# ── K1: Memory Cleanup Tests ──────────────────────────────────


class TestMemoryCleanup:
    @pytest.mark.asyncio
    async def test_cleanup_with_reminder(self, tmp_path):
        from datetime import datetime, timedelta
        from core.reminder_manager import ReminderManager
        rm = ReminderManager(path=str(tmp_path / "rem.json"))
        old = datetime.now() - timedelta(days=10)
        rm._reminders.append({
            "id": "rem_old",
            "content": "old",
            "remind_at": old.isoformat(),
            "source": "user",
            "fired": True,
        })
        rm._save()
        hb = Heartbeat(reminder_manager=rm)
        result = await hb.memory_cleanup()
        assert result["reminders_removed"] == 1

    @pytest.mark.asyncio
    async def test_cleanup_without_reminder(self):
        hb = Heartbeat()
        result = await hb.memory_cleanup()
        assert result["reminders_removed"] == 0

    @pytest.mark.asyncio
    async def test_memory_cleanup_job_exists(self):
        hb = Heartbeat()
        hb.start()
        job_ids = [j["id"] for j in hb.get_jobs()]
        assert "memory_cleanup" in job_ids
        hb.stop()


# ── K1: Gog Helper Tests ──────────────────────────────────────


class TestGogHelpers:
    def test_get_gog_today_events_no_gog(self):
        hb = Heartbeat()
        assert hb._get_gog_today_events() is None

    def test_get_gog_today_events_unavailable(self):
        mock_gog = MagicMock()
        mock_gog.is_available = False
        hb = Heartbeat(gog_worker=mock_gog)
        assert hb._get_gog_today_events() is None

    def test_get_gog_today_events_success(self):
        mock_gog = MagicMock()
        mock_gog.is_available = True
        mock_gog.get_today_events.return_value = [{"summary": "Test"}]
        hb = Heartbeat(gog_worker=mock_gog)
        result = hb._get_gog_today_events()
        assert len(result) == 1

    def test_get_gog_today_events_exception(self):
        mock_gog = MagicMock()
        mock_gog.is_available = True
        mock_gog.get_today_events.side_effect = RuntimeError("fail")
        hb = Heartbeat(gog_worker=mock_gog)
        assert hb._get_gog_today_events() is None

    def test_get_upcoming_gog_events_no_gog(self):
        hb = Heartbeat()
        assert hb._get_upcoming_gog_events(60) == []

    def test_get_upcoming_gog_events_success(self):
        mock_gog = MagicMock()
        mock_gog.is_available = True
        mock_gog.get_upcoming_events.return_value = [{"summary": "Soon"}]
        hb = Heartbeat(gog_worker=mock_gog)
        result = hb._get_upcoming_gog_events(60)
        assert len(result) == 1

    def test_get_gog_events_for_date_no_gog(self):
        from datetime import datetime
        hb = Heartbeat()
        assert hb._get_gog_events_for_date(datetime.now()) is None

    def test_get_gog_events_for_date_success(self):
        from datetime import datetime
        mock_gog = MagicMock()
        mock_gog.is_available = True
        mock_gog.get_events_for_date.return_value = [{"summary": "Event"}]
        hb = Heartbeat(gog_worker=mock_gog)
        result = hb._get_gog_events_for_date(datetime.now())
        assert len(result) == 1
