"""Google Calendar client — dual-account OAuth with conflict detection.

Features:
- OAuth 2.0 authorization for two Google accounts
- Event fetching with configurable time range
- Cache events to MemOS working_memory
- Detect scheduling conflicts across both accounts
- Format events for Heartbeat morning brief

Setup:
1. Create a Google Cloud project, enable Calendar API
2. Download OAuth client credentials JSON → config/google_credentials.json
3. Run authorize_account() for each account (opens browser for consent)
4. Token files saved to config/ for future use
"""

from __future__ import annotations

import asyncio
import json
import webbrowser
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from loguru import logger

try:
    from google.auth.transport.requests import Request
    from google.oauth2.credentials import Credentials
    from google_auth_oauthlib.flow import InstalledAppFlow
    from googleapiclient.discovery import build

    _HAS_GCAL = True
except ImportError:
    _HAS_GCAL = False
    logger.warning("Google Calendar dependencies not installed")


# Calendar API read-only scope
SCOPES = ["https://www.googleapis.com/auth/calendar.readonly"]

# Chrome path on Windows
_CHROME_PATH = r"C:\Program Files\Google\Chrome\Application\chrome.exe"


def _register_chrome() -> None:
    """Register Chrome as the browser for webbrowser module (skip Edge)."""
    try:
        webbrowser.register(
            "chrome", None,
            webbrowser.BackgroundBrowser(_CHROME_PATH),
        )
    except Exception:
        logger.warning("Could not register Chrome, falling back to system default")


class CalendarEvent:
    """Normalized calendar event."""

    def __init__(
        self,
        summary: str,
        start: datetime,
        end: datetime,
        account: str = "",
        calendar_id: str = "",
        event_id: str = "",
        location: str = "",
        description: str = "",
    ):
        self.summary = summary
        self.start = start
        self.end = end
        self.account = account
        self.calendar_id = calendar_id
        self.event_id = event_id
        self.location = location
        self.description = description

    def overlaps(self, other: CalendarEvent) -> bool:
        """Check if this event overlaps with another."""
        return self.start < other.end and other.start < self.end

    def to_dict(self) -> dict[str, Any]:
        return {
            "summary": self.summary,
            "start": self.start.isoformat(),
            "end": self.end.isoformat(),
            "date": self.start.strftime("%Y-%m-%dT%H:%M"),
            "account": self.account,
            "location": self.location,
        }

    def format_brief(self) -> str:
        """Format for morning brief display."""
        time_str = self.start.strftime("%H:%M")
        end_str = self.end.strftime("%H:%M")
        loc = f" @ {self.location}" if self.location else ""
        tag = f" [{self.account}]" if self.account else ""
        return f"{time_str}-{end_str} {self.summary}{loc}{tag}"

    def __repr__(self) -> str:
        return f"CalendarEvent({self.summary!r}, {self.start}, account={self.account!r})"


class Conflict:
    """Represents a scheduling conflict between two events."""

    def __init__(self, event_a: CalendarEvent, event_b: CalendarEvent):
        self.event_a = event_a
        self.event_b = event_b

    def format(self) -> str:
        return (
            f"⚠️ 時間衝突:\n"
            f"  [{self.event_a.account}] {self.event_a.format_brief()}\n"
            f"  [{self.event_b.account}] {self.event_b.format_brief()}"
        )

    def __repr__(self) -> str:
        return f"Conflict({self.event_a.summary!r} vs {self.event_b.summary!r})"


class GoogleCalendarClient:
    """Dual-account Google Calendar client.

    Usage:
        cal = GoogleCalendarClient(
            credentials_path="config/google_credentials.json",
            accounts={
                "personal": "config/google_token_personal.json",
                "work": "config/google_token_work.json",
            },
        )
        await cal.init()
        events = await cal.get_upcoming(hours=24)
        conflicts = cal.detect_conflicts(events)
    """

    def __init__(
        self,
        credentials_path: str = "config/google_credentials.json",
        accounts: dict[str, str] | None = None,
        memos: Any = None,
    ):
        self.credentials_path = Path(credentials_path)
        self.accounts = accounts or {
            "account1": "config/google_token_account1.json",
            "account2": "config/google_token_account2.json",
        }
        self.memos = memos
        self._services: dict[str, Any] = {}

    async def init(self) -> None:
        """Initialize Calendar API services for authorized accounts."""
        if not _HAS_GCAL:
            logger.warning("Google Calendar not available (missing dependencies)")
            return

        for name, token_path in self.accounts.items():
            token_file = Path(token_path)
            if token_file.exists():
                try:
                    creds = await self._load_credentials(token_file)
                    service = build("calendar", "v3", credentials=creds)
                    self._services[name] = service
                    logger.info(f"Calendar account '{name}' initialized")
                except Exception as e:
                    logger.warning(f"Failed to init calendar account '{name}': {e}")
            else:
                logger.info(f"Calendar token not found for '{name}', skipping (run authorize_account first)")

    async def _load_credentials(self, token_path: Path) -> Credentials:
        """Load and refresh OAuth credentials."""
        def _load():
            creds = Credentials.from_authorized_user_file(str(token_path), SCOPES)
            if creds and creds.expired and creds.refresh_token:
                creds.refresh(Request())
                token_path.write_text(creds.to_json())
            return creds

        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, _load)

    def authorize_account(self, account_name: str) -> bool:
        """Interactive OAuth flow — opens browser for consent.

        Call this once per account during initial setup.
        NOT async — must be run in a terminal context.
        """
        if not _HAS_GCAL:
            logger.error("Cannot authorize: missing Google Calendar dependencies")
            return False

        if not self.credentials_path.exists():
            logger.error(f"Credentials file not found: {self.credentials_path}")
            return False

        token_path = self.accounts.get(account_name)
        if not token_path:
            logger.error(f"Unknown account: {account_name}")
            return False

        try:
            # Force Chrome — never use Edge
            _register_chrome()

            flow = InstalledAppFlow.from_client_secrets_file(
                str(self.credentials_path), SCOPES
            )
            creds = flow.run_local_server(port=0, browser="chrome")

            Path(token_path).parent.mkdir(parents=True, exist_ok=True)
            Path(token_path).write_text(creds.to_json())
            logger.info(f"Account '{account_name}' authorized, token saved to {token_path}")
            return True
        except Exception as e:
            logger.error(f"Authorization failed for '{account_name}': {e}")
            return False

    # ── Event Fetching ──────────────────────────────────────────

    async def get_upcoming(
        self,
        hours: int = 24,
        max_results: int = 50,
    ) -> list[CalendarEvent]:
        """Fetch upcoming events from all accounts.

        Args:
            hours: look-ahead window in hours
            max_results: max events per account

        Returns:
            List of CalendarEvent sorted by start time.
        """
        now = datetime.now(timezone.utc)
        time_max = now + timedelta(hours=hours)

        all_events: list[CalendarEvent] = []

        for account_name, service in self._services.items():
            try:
                events = await self._fetch_events(
                    service, account_name, now, time_max, max_results
                )
                all_events.extend(events)
            except Exception as e:
                logger.warning(f"Failed to fetch events for '{account_name}': {e}")

        all_events.sort(key=lambda e: e.start)

        # Update MemOS cache
        if self.memos and self.memos.working_memory:
            cache = [e.to_dict() for e in all_events]
            await self.memos.working_memory.set(
                "calendar_cache", cache, agent_id="calendar"
            )

        return all_events

    async def _fetch_events(
        self,
        service: Any,
        account_name: str,
        time_min: datetime,
        time_max: datetime,
        max_results: int,
    ) -> list[CalendarEvent]:
        """Fetch events from a single account."""
        def _call():
            return (
                service.events()
                .list(
                    calendarId="primary",
                    timeMin=time_min.isoformat(),
                    timeMax=time_max.isoformat(),
                    maxResults=max_results,
                    singleEvents=True,
                    orderBy="startTime",
                )
                .execute()
            )

        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(None, _call)

        events = []
        for item in result.get("items", []):
            event = self._parse_event(item, account_name)
            if event:
                events.append(event)

        return events

    async def get_today(self) -> list[CalendarEvent]:
        """Shortcut: get today's events (next 24h from midnight)."""
        return await self.get_upcoming(hours=24)

    # ── Conflict Detection ──────────────────────────────────────

    @staticmethod
    def detect_conflicts(events: list[CalendarEvent]) -> list[Conflict]:
        """Detect scheduling conflicts across accounts.

        Only flags conflicts between DIFFERENT accounts.
        Same-account overlaps are assumed intentional.
        """
        conflicts: list[Conflict] = []

        for i, a in enumerate(events):
            for b in events[i + 1:]:
                # Only cross-account conflicts
                if a.account == b.account:
                    continue
                if a.overlaps(b):
                    conflicts.append(Conflict(a, b))

        return conflicts

    # ── Helpers ──────────────────────────────────────────────────

    @staticmethod
    def _parse_event(item: dict[str, Any], account: str) -> CalendarEvent | None:
        """Parse a Google Calendar API event into CalendarEvent."""
        summary = item.get("summary", "(No title)")

        # Handle all-day events vs timed events
        start_raw = item.get("start", {})
        end_raw = item.get("end", {})

        start = _parse_datetime(start_raw)
        end = _parse_datetime(end_raw)

        if start is None or end is None:
            return None

        return CalendarEvent(
            summary=summary,
            start=start,
            end=end,
            account=account,
            calendar_id=item.get("organizer", {}).get("email", ""),
            event_id=item.get("id", ""),
            location=item.get("location", ""),
            description=item.get("description", ""),
        )

    @property
    def authorized_accounts(self) -> list[str]:
        return list(self._services.keys())

    @property
    def is_available(self) -> bool:
        return _HAS_GCAL and len(self._services) > 0


def _parse_datetime(raw: dict[str, str]) -> datetime | None:
    """Parse Google Calendar datetime or date field."""
    if "dateTime" in raw:
        dt_str = raw["dateTime"]
        try:
            return datetime.fromisoformat(dt_str)
        except ValueError:
            return None
    elif "date" in raw:
        # All-day event
        try:
            return datetime.strptime(raw["date"], "%Y-%m-%d").replace(
                tzinfo=timezone.utc
            )
        except ValueError:
            return None
    return None
