"""Tests for Google Calendar client — event parsing, conflict detection, caching.

Run: pytest tests/test_calendar.py -v
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from unittest.mock import AsyncMock, MagicMock

import pytest

from clients.google_calendar import (
    CalendarEvent,
    Conflict,
    GoogleCalendarClient,
    _parse_datetime,
)
from memory.memos_manager import MemOS


# ── Fixtures ────────────────────────────────────────────────────


@pytest.fixture
async def memos(tmp_path):
    db_path = str(tmp_path / "test.db")
    m = MemOS(db_path)
    await m.init()
    yield m
    await m.close()


def _make_event(
    summary: str,
    start_hour: int,
    end_hour: int,
    account: str = "personal",
    date: datetime | None = None,
) -> CalendarEvent:
    """Helper to create a CalendarEvent at a given hour today."""
    base = date or datetime.now(timezone.utc).replace(
        hour=0, minute=0, second=0, microsecond=0
    )
    return CalendarEvent(
        summary=summary,
        start=base.replace(hour=start_hour),
        end=base.replace(hour=end_hour),
        account=account,
    )


# ── CalendarEvent Tests ─────────────────────────────────────────


class TestCalendarEvent:
    def test_overlap_true(self):
        a = _make_event("Meeting A", 9, 11, "personal")
        b = _make_event("Meeting B", 10, 12, "work")
        assert a.overlaps(b)
        assert b.overlaps(a)

    def test_no_overlap(self):
        a = _make_event("Morning", 9, 10, "personal")
        b = _make_event("Afternoon", 14, 15, "work")
        assert not a.overlaps(b)
        assert not b.overlaps(a)

    def test_adjacent_no_overlap(self):
        """Events that end exactly when the other starts should NOT overlap."""
        a = _make_event("First", 9, 10, "personal")
        b = _make_event("Second", 10, 11, "work")
        assert not a.overlaps(b)

    def test_contained_overlap(self):
        """Short event entirely within a long event."""
        long = _make_event("All day workshop", 9, 17, "work")
        short = _make_event("Quick sync", 12, 13, "personal")
        assert long.overlaps(short)

    def test_to_dict(self):
        e = _make_event("Test Event", 9, 10, "personal")
        d = e.to_dict()
        assert d["summary"] == "Test Event"
        assert d["account"] == "personal"
        assert "date" in d

    def test_format_brief(self):
        e = _make_event("Team standup", 9, 10, "work")
        e.location = "Room 301"
        brief = e.format_brief()
        assert "09:00" in brief
        assert "10:00" in brief
        assert "Team standup" in brief
        assert "Room 301" in brief
        assert "[work]" in brief

    def test_format_brief_no_location(self):
        e = _make_event("Lunch", 12, 13, "personal")
        brief = e.format_brief()
        assert "Lunch" in brief
        assert "@" not in brief


# ── Conflict Detection Tests ────────────────────────────────────


class TestConflictDetection:
    def test_cross_account_conflict(self):
        events = [
            _make_event("Work meeting", 9, 11, "work"),
            _make_event("Doctor appointment", 10, 11, "personal"),
        ]
        conflicts = GoogleCalendarClient.detect_conflicts(events)
        assert len(conflicts) == 1
        assert "Work meeting" in conflicts[0].event_a.summary or "Work meeting" in conflicts[0].event_b.summary

    def test_same_account_no_conflict(self):
        """Same account overlaps are intentional — don't flag."""
        events = [
            _make_event("Meeting A", 9, 11, "work"),
            _make_event("Meeting B", 10, 12, "work"),
        ]
        conflicts = GoogleCalendarClient.detect_conflicts(events)
        assert len(conflicts) == 0

    def test_no_conflicts(self):
        events = [
            _make_event("Morning work", 9, 12, "work"),
            _make_event("Afternoon personal", 14, 16, "personal"),
        ]
        conflicts = GoogleCalendarClient.detect_conflicts(events)
        assert len(conflicts) == 0

    def test_multiple_conflicts(self):
        events = [
            _make_event("Work A", 9, 11, "work"),
            _make_event("Personal A", 10, 12, "personal"),
            _make_event("Work B", 11, 13, "work"),
        ]
        conflicts = GoogleCalendarClient.detect_conflicts(events)
        # Work A overlaps Personal A, Personal A overlaps Work B
        assert len(conflicts) == 2

    def test_conflict_format(self):
        a = _make_event("Work meeting", 9, 11, "work")
        b = _make_event("Doctor", 10, 11, "personal")
        conflict = Conflict(a, b)
        formatted = conflict.format()
        assert "衝突" in formatted
        assert "work" in formatted
        assert "personal" in formatted

    def test_empty_events(self):
        conflicts = GoogleCalendarClient.detect_conflicts([])
        assert len(conflicts) == 0

    def test_single_event(self):
        events = [_make_event("Solo", 9, 10, "work")]
        conflicts = GoogleCalendarClient.detect_conflicts(events)
        assert len(conflicts) == 0


# ── DateTime Parsing ────────────────────────────────────────────


class TestDateTimeParsing:
    def test_parse_timed_event(self):
        raw = {"dateTime": "2026-02-15T09:00:00+08:00"}
        dt = _parse_datetime(raw)
        assert dt is not None
        assert dt.hour == 9

    def test_parse_all_day_event(self):
        raw = {"date": "2026-02-15"}
        dt = _parse_datetime(raw)
        assert dt is not None
        assert dt.year == 2026
        assert dt.month == 2
        assert dt.day == 15

    def test_parse_empty(self):
        assert _parse_datetime({}) is None

    def test_parse_invalid(self):
        assert _parse_datetime({"dateTime": "not-a-date"}) is None


# ── Event Parsing from API Response ─────────────────────────────


class TestEventParsing:
    def test_parse_full_event(self):
        item = {
            "id": "evt123",
            "summary": "Team meeting",
            "start": {"dateTime": "2026-02-15T09:00:00+08:00"},
            "end": {"dateTime": "2026-02-15T10:00:00+08:00"},
            "location": "Conference Room A",
            "description": "Weekly sync",
            "organizer": {"email": "ted@example.com"},
        }
        event = GoogleCalendarClient._parse_event(item, "work")
        assert event is not None
        assert event.summary == "Team meeting"
        assert event.account == "work"
        assert event.location == "Conference Room A"
        assert event.event_id == "evt123"

    def test_parse_minimal_event(self):
        item = {
            "start": {"date": "2026-02-15"},
            "end": {"date": "2026-02-16"},
        }
        event = GoogleCalendarClient._parse_event(item, "personal")
        assert event is not None
        assert event.summary == "(No title)"

    def test_parse_missing_time(self):
        item = {"summary": "Bad event", "start": {}, "end": {}}
        event = GoogleCalendarClient._parse_event(item, "test")
        assert event is None


# ── MemOS Cache Integration ─────────────────────────────────────


class TestCalendarCache:
    @pytest.mark.asyncio
    async def test_events_cached_to_memos(self, memos):
        cal = GoogleCalendarClient(memos=memos)
        # Manually simulate cached events (since no real API)
        events = [
            _make_event("Meeting", 9, 10, "work"),
            _make_event("Lunch", 12, 13, "personal"),
        ]
        cache = [e.to_dict() for e in events]
        await memos.working_memory.set("calendar_cache", cache, agent_id="calendar")

        cached = await memos.working_memory.get("calendar_cache")
        assert len(cached) == 2
        assert cached[0]["summary"] == "Meeting"
        assert cached[1]["summary"] == "Lunch"


# ── Client Init ─────────────────────────────────────────────────


class TestClientInit:
    def test_init_without_tokens(self, tmp_path):
        cal = GoogleCalendarClient(
            credentials_path=str(tmp_path / "nonexistent.json"),
            accounts={"test": str(tmp_path / "no_token.json")},
        )
        assert not cal.is_available
        assert cal.authorized_accounts == []

    @pytest.mark.asyncio
    async def test_init_skips_missing_tokens(self, tmp_path):
        cal = GoogleCalendarClient(
            accounts={"test": str(tmp_path / "missing.json")},
        )
        await cal.init()
        assert cal.authorized_accounts == []
