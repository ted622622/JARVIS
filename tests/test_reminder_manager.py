"""Tests for ReminderManager.

Run: pytest tests/test_reminder_manager.py -v
"""

from __future__ import annotations

import json
from datetime import datetime, timedelta
from unittest.mock import AsyncMock, MagicMock

import pytest

from core.reminder_manager import ReminderManager


@pytest.fixture
def rm(tmp_path):
    """ReminderManager with temp file and mock scheduler/telegram."""
    scheduler = MagicMock()
    telegram = MagicMock()
    telegram.send = AsyncMock()
    path = str(tmp_path / "reminders.json")
    return ReminderManager(path=path, scheduler=scheduler, telegram=telegram)


@pytest.fixture
def rm_no_scheduler(tmp_path):
    """ReminderManager without scheduler (for isolated tests)."""
    path = str(tmp_path / "reminders.json")
    return ReminderManager(path=path)


class TestAdd:
    @pytest.mark.asyncio
    async def test_add_creates_reminder(self, rm):
        future = datetime.now() + timedelta(hours=1)
        result = await rm.add("開會", future)
        assert result["content"] == "開會"
        assert result["fired"] is False
        assert result["source"] == "user"
        assert result["id"].startswith("rem_")

    @pytest.mark.asyncio
    async def test_add_persists_to_file(self, rm):
        future = datetime.now() + timedelta(hours=1)
        await rm.add("開會", future)
        # Read the file directly
        data = json.loads(rm._path.read_text(encoding="utf-8"))
        assert len(data) == 1
        assert data[0]["content"] == "開會"

    @pytest.mark.asyncio
    async def test_add_schedules_job(self, rm):
        future = datetime.now() + timedelta(hours=1)
        result = await rm.add("開會", future)
        rm._scheduler.add_job.assert_called_once()
        call_kwargs = rm._scheduler.add_job.call_args
        assert call_kwargs.kwargs["id"] == result["id"]

    @pytest.mark.asyncio
    async def test_add_past_time_does_not_schedule(self, rm):
        past = datetime.now() - timedelta(hours=1)
        await rm.add("已過期", past)
        rm._scheduler.add_job.assert_not_called()

    @pytest.mark.asyncio
    async def test_add_custom_source(self, rm):
        future = datetime.now() + timedelta(hours=1)
        result = await rm.add("chain reminder", future, source="chain")
        assert result["source"] == "chain"

    @pytest.mark.asyncio
    async def test_add_multiple(self, rm):
        future1 = datetime.now() + timedelta(hours=1)
        future2 = datetime.now() + timedelta(hours=2)
        await rm.add("first", future1)
        await rm.add("second", future2)
        assert len(rm.all_reminders) == 2


class TestGetToday:
    @pytest.mark.asyncio
    async def test_get_today_includes_today(self, rm):
        today = datetime.now().replace(hour=23, minute=0, second=0, microsecond=0)
        if today < datetime.now():
            today += timedelta(days=1)
        await rm.add("today event", today)
        result = rm.get_today()
        # Only included if actually today
        if today.date() == datetime.now().date():
            assert len(result) == 1
            assert result[0]["content"] == "today event"

    @pytest.mark.asyncio
    async def test_get_today_excludes_tomorrow(self, rm):
        tomorrow = datetime.now() + timedelta(days=1)
        await rm.add("tomorrow event", tomorrow)
        result = rm.get_today()
        assert len(result) == 0

    @pytest.mark.asyncio
    async def test_get_today_excludes_fired(self, rm):
        now_plus_1h = datetime.now().replace(hour=23, minute=59)
        if now_plus_1h < datetime.now():
            now_plus_1h += timedelta(days=1)
        r = await rm.add("fired event", now_plus_1h)
        # Mark as fired
        rm._reminders[0]["fired"] = True
        result = rm.get_today()
        assert len(result) == 0


class TestGetForDate:
    @pytest.mark.asyncio
    async def test_get_for_date(self, rm):
        target = datetime.now() + timedelta(days=3)
        target = target.replace(hour=10, minute=0)
        await rm.add("future event", target)
        result = rm.get_for_date(target)
        assert len(result) == 1
        assert result[0]["content"] == "future event"

    @pytest.mark.asyncio
    async def test_get_for_date_wrong_date(self, rm):
        target = datetime.now() + timedelta(days=3)
        await rm.add("future event", target)
        other = datetime.now() + timedelta(days=5)
        result = rm.get_for_date(other)
        assert len(result) == 0


class TestFire:
    @pytest.mark.asyncio
    async def test_fire_sends_telegram(self, rm):
        future = datetime.now() + timedelta(hours=1)
        r = await rm.add("test fire", future)
        await rm._fire(r)
        rm._telegram.send.assert_awaited_once()
        msg = rm._telegram.send.call_args[0][0]
        assert "test fire" in msg

    @pytest.mark.asyncio
    async def test_fire_marks_fired(self, rm):
        future = datetime.now() + timedelta(hours=1)
        r = await rm.add("test fire", future)
        await rm._fire(r)
        assert rm._reminders[0]["fired"] is True

    @pytest.mark.asyncio
    async def test_fire_persists(self, rm):
        future = datetime.now() + timedelta(hours=1)
        r = await rm.add("test fire", future)
        await rm._fire(r)
        data = json.loads(rm._path.read_text(encoding="utf-8"))
        assert data[0]["fired"] is True

    @pytest.mark.asyncio
    async def test_fire_without_telegram(self, rm_no_scheduler):
        future = datetime.now() + timedelta(hours=1)
        r = await rm_no_scheduler.add("no tg", future)
        await rm_no_scheduler._fire(r)  # Should not raise
        assert rm_no_scheduler._reminders[0]["fired"] is True


class TestCleanup:
    @pytest.mark.asyncio
    async def test_cleanup_removes_old_fired(self, rm):
        old = datetime.now() - timedelta(days=10)
        rm._reminders.append({
            "id": "rem_old",
            "content": "old",
            "remind_at": old.isoformat(),
            "source": "user",
            "fired": True,
        })
        rm._save()
        removed = rm.cleanup(days=7)
        assert removed == 1
        assert len(rm.all_reminders) == 0

    @pytest.mark.asyncio
    async def test_cleanup_keeps_recent_fired(self, rm):
        recent = datetime.now() - timedelta(days=2)
        rm._reminders.append({
            "id": "rem_recent",
            "content": "recent",
            "remind_at": recent.isoformat(),
            "source": "user",
            "fired": True,
        })
        rm._save()
        removed = rm.cleanup(days=7)
        assert removed == 0
        assert len(rm.all_reminders) == 1

    @pytest.mark.asyncio
    async def test_cleanup_keeps_unfired(self, rm):
        old = datetime.now() - timedelta(days=10)
        rm._reminders.append({
            "id": "rem_unfired",
            "content": "unfired",
            "remind_at": old.isoformat(),
            "source": "user",
            "fired": False,
        })
        rm._save()
        removed = rm.cleanup(days=7)
        assert removed == 0


class TestLoadIntoScheduler:
    @pytest.mark.asyncio
    async def test_load_schedules_future_unfired(self, rm):
        future = datetime.now() + timedelta(hours=2)
        await rm.add("future", future)
        rm._scheduler.reset_mock()
        loaded = rm.load_into_scheduler()
        assert loaded == 1
        rm._scheduler.add_job.assert_called_once()

    @pytest.mark.asyncio
    async def test_load_skips_fired(self, rm):
        future = datetime.now() + timedelta(hours=2)
        await rm.add("fired", future)
        rm._reminders[0]["fired"] = True
        rm._scheduler.reset_mock()
        loaded = rm.load_into_scheduler()
        assert loaded == 0

    @pytest.mark.asyncio
    async def test_load_skips_past(self, rm):
        past = datetime.now() - timedelta(hours=1)
        rm._reminders.append({
            "id": "rem_past",
            "content": "past",
            "remind_at": past.isoformat(),
            "source": "user",
            "fired": False,
        })
        rm._scheduler.reset_mock()
        loaded = rm.load_into_scheduler()
        assert loaded == 0


class TestPersistence:
    @pytest.mark.asyncio
    async def test_reload_from_file(self, tmp_path):
        path = str(tmp_path / "reminders.json")
        rm1 = ReminderManager(path=path)
        future = datetime.now() + timedelta(hours=1)
        await rm1.add("persistent", future)

        rm2 = ReminderManager(path=path)
        assert len(rm2.all_reminders) == 1
        assert rm2.all_reminders[0]["content"] == "persistent"

    def test_empty_file_handled(self, tmp_path):
        path = tmp_path / "reminders.json"
        path.write_text("", encoding="utf-8")
        rm = ReminderManager(path=str(path))
        assert len(rm.all_reminders) == 0

    def test_corrupt_file_handled(self, tmp_path):
        path = tmp_path / "reminders.json"
        path.write_text("{bad json", encoding="utf-8")
        rm = ReminderManager(path=str(path))
        assert len(rm.all_reminders) == 0

    def test_missing_file_handled(self, tmp_path):
        path = str(tmp_path / "nonexistent" / "reminders.json")
        rm = ReminderManager(path=path)
        assert len(rm.all_reminders) == 0
