"""ReminderManager — JSON-persistent dynamic reminder system.

Stores reminders in data/reminders.json, fires them via APScheduler date jobs.
Integrates with Heartbeat scheduler for startup loading.

Usage:
    rm = ReminderManager(path="./data/reminders.json", scheduler=scheduler, telegram=tg)
    await rm.add("開會", remind_at=datetime(2026, 2, 17, 18, 0))
    rm.load_into_scheduler()  # on startup
"""

from __future__ import annotations

import json
import time
from datetime import datetime
from pathlib import Path
from typing import Any

from loguru import logger


class ReminderManager:
    """JSON-backed reminder system with APScheduler integration."""

    def __init__(
        self,
        path: str = "./data/reminders.json",
        scheduler: Any = None,
        telegram: Any = None,
    ):
        self._path = Path(path)
        self._scheduler = scheduler
        self._telegram = telegram
        self._reminders: list[dict] = []
        self._load_file()

    def _load_file(self) -> None:
        """Load reminders from JSON file."""
        if self._path.exists():
            try:
                self._reminders = json.loads(self._path.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError) as e:
                logger.warning(f"Failed to load reminders: {e}")
                self._reminders = []
        else:
            self._reminders = []

    def _save(self) -> None:
        """Persist reminders to JSON file."""
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._path.write_text(
            json.dumps(self._reminders, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    async def add(
        self,
        content: str,
        remind_at: datetime,
        source: str = "user",
    ) -> dict:
        """Add a reminder and schedule it.

        Args:
            content: Reminder text.
            remind_at: When to fire.
            source: Origin — "user" or "system".

        Returns:
            The created reminder dict.
        """
        reminder = {
            "id": f"rem_{int(time.time() * 1000)}",
            "content": content,
            "remind_at": remind_at.isoformat(),
            "source": source,
            "fired": False,
        }
        self._reminders.append(reminder)
        self._save()

        self._schedule_job(reminder)
        logger.info(f"Reminder added: {reminder['id']} at {remind_at}")
        return reminder

    def _schedule_job(self, reminder: dict) -> None:
        """Add an APScheduler date-trigger job for this reminder."""
        if not self._scheduler:
            return

        remind_at = datetime.fromisoformat(reminder["remind_at"])
        if remind_at <= datetime.now():
            return  # Already past

        try:
            self._scheduler.add_job(
                self._fire,
                "date",
                run_date=remind_at,
                id=reminder["id"],
                name=f"Reminder: {reminder['content'][:30]}",
                args=[reminder],
                replace_existing=True,
            )
        except Exception as e:
            logger.warning(f"Failed to schedule reminder {reminder['id']}: {e}")

    def load_into_scheduler(self) -> int:
        """On startup: load all unfired future reminders into APScheduler.

        Returns:
            Number of reminders loaded.
        """
        now = datetime.now()
        loaded = 0
        for rem in self._reminders:
            if rem.get("fired"):
                continue
            remind_at = datetime.fromisoformat(rem["remind_at"])
            if remind_at > now:
                self._schedule_job(rem)
                loaded += 1
        logger.info(f"Loaded {loaded} reminders into scheduler")
        return loaded

    def get_today(self) -> list[dict]:
        """Get unfired reminders for today."""
        today = datetime.now().strftime("%Y-%m-%d")
        return [
            r for r in self._reminders
            if r["remind_at"].startswith(today) and not r.get("fired")
        ]

    def get_for_date(self, date: datetime) -> list[dict]:
        """Get unfired reminders for a specific date."""
        date_str = date.strftime("%Y-%m-%d")
        return [
            r for r in self._reminders
            if r["remind_at"].startswith(date_str) and not r.get("fired")
        ]

    async def _fire(self, reminder: dict) -> None:
        """Callback when a reminder fires: send via Telegram and mark fired."""
        content = reminder["content"]
        logger.info(f"Firing reminder {reminder['id']}: {content}")

        if self._telegram:
            try:
                await self._telegram.send(f"⏰ 提醒：{content}")
            except Exception as e:
                logger.warning(f"Failed to send reminder via Telegram: {e}")

        # Mark as fired
        for r in self._reminders:
            if r["id"] == reminder["id"]:
                r["fired"] = True
                break
        self._save()

    def cleanup(self, days: int = 7) -> int:
        """Remove fired reminders older than N days.

        Returns:
            Number of reminders removed.
        """
        cutoff = datetime.now().timestamp() - days * 86400
        before = len(self._reminders)
        self._reminders = [
            r for r in self._reminders
            if not r.get("fired")
            or datetime.fromisoformat(r["remind_at"]).timestamp() > cutoff
        ]
        removed = before - len(self._reminders)
        if removed:
            self._save()
            logger.info(f"Cleaned up {removed} old reminders")
        return removed

    @property
    def all_reminders(self) -> list[dict]:
        """All reminders (for testing/debugging)."""
        return list(self._reminders)
