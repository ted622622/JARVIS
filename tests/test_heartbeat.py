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
        assert len(jobs) == 5  # patrol, brief, health, backup, night_owl
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
