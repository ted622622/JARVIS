"""Tests for PostActionChain.

Run: pytest tests/test_post_action_chain.py -v
"""

from __future__ import annotations

from datetime import datetime, timedelta
from unittest.mock import AsyncMock, MagicMock

import pytest

from core.post_action_chain import CHAINS, PostActionChain


@pytest.fixture
def mock_gog():
    gog = MagicMock()
    gog.create_event.return_value = {"success": True, "data": {}}
    gog.is_available = True
    return gog


@pytest.fixture
def mock_reminder(tmp_path):
    from core.reminder_manager import ReminderManager
    rm = ReminderManager(path=str(tmp_path / "reminders.json"))
    return rm


@pytest.fixture
def chain(mock_gog, mock_reminder):
    return PostActionChain(gog_worker=mock_gog, reminder_manager=mock_reminder)


class TestChainDefinitions:
    def test_restaurant_booking_exists(self):
        assert "restaurant_booking" in CHAINS
        cfg = CHAINS["restaurant_booking"]
        assert "calendar" in cfg
        assert "reminders" in cfg
        assert len(cfg["reminders"]) >= 2

    def test_ticket_booking_exists(self):
        assert "ticket_booking" in CHAINS

    def test_meeting_exists(self):
        assert "meeting" in CHAINS

    def test_templates_have_placeholders(self):
        for name, cfg in CHAINS.items():
            cal = cfg.get("calendar", {})
            if "title_template" in cal:
                assert "{" in cal["title_template"], f"{name} title_template has no placeholder"


class TestExecuteChain:
    @pytest.mark.asyncio
    async def test_restaurant_booking_full(self, chain, mock_gog, mock_reminder):
        event_time = datetime.now() + timedelta(days=1)
        result = await chain.execute_chain(
            "restaurant_booking",
            event_time=event_time,
            params={"restaurant_name": "鼎泰豐", "address": "台北市信義路"},
        )
        assert result["calendar_added"] is True
        assert result["reminders_set"] == 2
        mock_gog.create_event.assert_called_once()
        call_kwargs = mock_gog.create_event.call_args
        assert "鼎泰豐" in call_kwargs.kwargs["title"]
        assert call_kwargs.kwargs["duration_minutes"] == 120

    @pytest.mark.asyncio
    async def test_unknown_chain_type(self, chain):
        result = await chain.execute_chain(
            "nonexistent",
            event_time=datetime.now() + timedelta(days=1),
        )
        assert result["calendar_added"] is False
        assert result["reminders_set"] == 0

    @pytest.mark.asyncio
    async def test_calendar_failure(self, chain, mock_gog):
        mock_gog.create_event.return_value = {"success": False, "error": "timeout"}
        event_time = datetime.now() + timedelta(days=1)
        result = await chain.execute_chain(
            "restaurant_booking",
            event_time=event_time,
            params={"restaurant_name": "Test"},
        )
        assert result["calendar_added"] is False
        # Reminders should still be set
        assert result["reminders_set"] >= 1

    @pytest.mark.asyncio
    async def test_no_gog_skips_calendar(self, mock_reminder):
        chain = PostActionChain(gog_worker=None, reminder_manager=mock_reminder)
        event_time = datetime.now() + timedelta(days=1)
        result = await chain.execute_chain(
            "restaurant_booking",
            event_time=event_time,
            params={"restaurant_name": "Test"},
        )
        assert result["calendar_added"] is False
        assert result["reminders_set"] >= 1

    @pytest.mark.asyncio
    async def test_no_reminder_skips_reminders(self, mock_gog):
        chain = PostActionChain(gog_worker=mock_gog, reminder_manager=None)
        event_time = datetime.now() + timedelta(days=1)
        result = await chain.execute_chain(
            "restaurant_booking",
            event_time=event_time,
            params={"restaurant_name": "Test"},
        )
        assert result["calendar_added"] is True
        assert result["reminders_set"] == 0

    @pytest.mark.asyncio
    async def test_no_deps(self):
        chain = PostActionChain()
        event_time = datetime.now() + timedelta(days=1)
        result = await chain.execute_chain(
            "restaurant_booking",
            event_time=event_time,
            params={"restaurant_name": "Test"},
        )
        assert result["calendar_added"] is False
        assert result["reminders_set"] == 0

    @pytest.mark.asyncio
    async def test_meeting_chain(self, chain, mock_gog):
        event_time = datetime.now() + timedelta(days=1)
        result = await chain.execute_chain(
            "meeting",
            event_time=event_time,
            params={"meeting_title": "Weekly Sync"},
        )
        assert result["calendar_added"] is True
        assert result["reminders_set"] == 2
        call_kwargs = mock_gog.create_event.call_args
        assert "Weekly Sync" in call_kwargs.kwargs["title"]
        assert call_kwargs.kwargs["duration_minutes"] == 60

    @pytest.mark.asyncio
    async def test_past_reminders_skipped(self, chain, mock_gog):
        # Event in 10 minutes — the 120-min-before reminder is in the past
        event_time = datetime.now() + timedelta(minutes=10)
        result = await chain.execute_chain(
            "restaurant_booking",
            event_time=event_time,
            params={"restaurant_name": "Test"},
        )
        assert result["calendar_added"] is True
        # Only the past-skipped 120-min reminder would be skipped
        # The 30-min-before reminder is also in the past (10 min from now < 30 min before)
        assert result["reminders_set"] == 0

    @pytest.mark.asyncio
    async def test_params_default_empty(self, chain, mock_gog):
        """Chain with missing template vars should handle gracefully."""
        event_time = datetime.now() + timedelta(days=1)
        # restaurant_booking needs restaurant_name — this will fail template
        result = await chain.execute_chain(
            "restaurant_booking",
            event_time=event_time,
            params=None,
        )
        # KeyError caught → calendar_added stays False
        assert result["calendar_added"] is False


class TestCalendarEventDetails:
    @pytest.mark.asyncio
    async def test_location_passed(self, chain, mock_gog):
        event_time = datetime.now() + timedelta(days=1)
        await chain.execute_chain(
            "restaurant_booking",
            event_time=event_time,
            params={"restaurant_name": "鼎泰豐", "address": "台北市信義路二段"},
        )
        call_kwargs = mock_gog.create_event.call_args
        assert call_kwargs.kwargs["location"] == "台北市信義路二段"

    @pytest.mark.asyncio
    async def test_empty_location(self, chain, mock_gog):
        event_time = datetime.now() + timedelta(days=1)
        await chain.execute_chain(
            "restaurant_booking",
            event_time=event_time,
            params={"restaurant_name": "鼎泰豐"},
        )
        call_kwargs = mock_gog.create_event.call_args
        assert call_kwargs.kwargs["location"] == ""

    @pytest.mark.asyncio
    async def test_ticket_chain_duration(self, chain, mock_gog):
        event_time = datetime.now() + timedelta(days=1)
        await chain.execute_chain(
            "ticket_booking",
            event_time=event_time,
            params={"event_name": "Concert"},
        )
        call_kwargs = mock_gog.create_event.call_args
        assert call_kwargs.kwargs["duration_minutes"] == 180
