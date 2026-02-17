"""Heartbeat — proactive scheduling engine for J.A.R.V.I.S.

Scheduled jobs:
- hourly_patrol:    every 60 min — check calendar, emotions, upcoming events alert
- morning_brief:    daily 07:30 — weather + calendar (gog) + reminders + agenda
- evening_summary:  daily 23:00 — today recap + tomorrow preview
- health_check:     every 6 hrs — SurvivalGate full diagnostics
- backup:           daily 03:00 — encrypted MemOS + skills backup
- memory_cleanup:   daily 03:15 — purge old fired reminders
- night_owl:        cron check  — detect late-night activity, suggest rest
"""

from __future__ import annotations

import time
from datetime import datetime, timedelta
from typing import Any

from apscheduler.schedulers.asyncio import AsyncIOScheduler
from loguru import logger


class Heartbeat:
    """Life-pulse scheduler for J.A.R.V.I.S.

    Usage:
        hb = Heartbeat(
            model_router=router,
            memos=memos,
            telegram=telegram_client,
            survival_gate=survival_gate,
        )
        hb.start()
        # ... runs in background via asyncio event loop
        hb.stop()
    """

    def __init__(
        self,
        model_router: Any = None,
        memos: Any = None,
        telegram: Any = None,
        survival_gate: Any = None,
        config: dict[str, Any] | None = None,
        weather_client: Any = None,
        pending_tasks: Any = None,
        react_executor: Any = None,
        gog_worker: Any = None,
        reminder_manager: Any = None,
    ):
        self.router = model_router
        self.memos = memos
        self.telegram = telegram
        self.survival = survival_gate
        self.config = config or {}
        self.weather = weather_client
        self.pending = pending_tasks
        self.react = react_executor
        self.gog = gog_worker
        self.reminder = reminder_manager

        self.scheduler = AsyncIOScheduler()
        self._running = False

    def start(self) -> None:
        """Register all jobs and start the scheduler."""
        cfg = self.config.get("heartbeat", {})

        # Hourly patrol
        patrol_interval = cfg.get("hourly_patrol_interval_minutes", 60)
        self.scheduler.add_job(
            self.hourly_patrol,
            "interval",
            minutes=patrol_interval,
            id="hourly_patrol",
            name="Hourly Patrol",
        )

        # Morning brief
        brief_time = cfg.get("morning_brief_time", "07:30")
        hour, minute = map(int, brief_time.split(":"))
        self.scheduler.add_job(
            self.morning_brief,
            "cron",
            hour=hour,
            minute=minute,
            id="morning_brief",
            name="Morning Brief",
        )

        # Health check
        health_interval = cfg.get("health_check_interval_hours", 6)
        self.scheduler.add_job(
            self.health_check,
            "interval",
            hours=health_interval,
            id="health_check",
            name="Health Check",
        )

        # Nightly backup
        backup_time = cfg.get("backup_time", "03:00")
        bh, bm = map(int, backup_time.split(":"))
        self.scheduler.add_job(
            self.nightly_backup,
            "cron",
            hour=bh,
            minute=bm,
            id="nightly_backup",
            name="Nightly Backup",
        )

        # Night owl detection (check every 30 min between 00:00-05:00)
        self.scheduler.add_job(
            self.night_owl_check,
            "cron",
            hour="0-4",
            minute="*/30",
            id="night_owl",
            name="Night Owl Detection",
        )

        # Evening summary (Patch K1)
        evening_time = cfg.get("evening_summary_time", "23:00")
        eh, em = map(int, evening_time.split(":"))
        self.scheduler.add_job(
            self.evening_summary,
            "cron",
            hour=eh,
            minute=em,
            id="evening_summary",
            name="Evening Summary",
        )

        # Memory cleanup — after backup (Patch K1)
        self.scheduler.add_job(
            self.memory_cleanup,
            "cron",
            hour=bh,
            minute=bm + 15 if bm + 15 < 60 else 0,
            id="memory_cleanup",
            name="Memory Cleanup",
        )

        # Pending task retry (Patch H)
        if self.pending:
            self.scheduler.add_job(
                self.retry_pending_tasks,
                "interval",
                minutes=15,
                id="pending_tasks",
                name="Pending Task Retry",
            )

        self.scheduler.start()
        self._running = True
        logger.info(f"Heartbeat started with {len(self.scheduler.get_jobs())} jobs")

    def stop(self) -> None:
        """Stop the scheduler."""
        if self._running:
            self.scheduler.shutdown(wait=False)
            self._running = False
            logger.info("Heartbeat stopped")

    @property
    def is_running(self) -> bool:
        return self._running

    def get_jobs(self) -> list[dict[str, Any]]:
        """List all scheduled jobs with next run time."""
        return [
            {
                "id": job.id,
                "name": job.name,
                "next_run": str(job.next_run_time) if job.next_run_time else "paused",
            }
            for job in self.scheduler.get_jobs()
        ]

    # ── Job Implementations ─────────────────────────────────────

    async def hourly_patrol(self) -> dict[str, Any]:
        """Hourly patrol: check MemOS state, decide if outreach needed.

        Reads:
        - user_emotion from working_memory
        - active_tasks from working_memory
        - calendar_cache from working_memory (future: real calendar)

        Returns patrol result dict (for testing).
        """
        result = {"action": "none", "timestamp": time.time()}

        if not self.memos or not self.router:
            logger.debug("Heartbeat patrol skipped: missing dependencies")
            return result

        # Read current state from MemOS
        emotion = await self.memos.working_memory.get("user_emotion", "unknown")
        active_tasks = await self.memos.working_memory.get("active_tasks", [])
        calendar_cache = await self.memos.working_memory.get("calendar_cache", [])

        # K1: Check gog for upcoming events (30-min-ahead alert)
        upcoming_events = self._get_upcoming_gog_events(60)
        if upcoming_events:
            calendar_cache = upcoming_events  # use real data if available
            result["upcoming_events"] = len(upcoming_events)
            # Send 30-min-ahead alerts for events starting in ≤30 min
            soon_events = self._get_upcoming_gog_events(30)
            if soon_events and self.telegram:
                for ev in soon_events:
                    summary = ev.get("summary", "行程")
                    start = ev.get("start", {}).get("dateTime", "")
                    await self.telegram.send(f"📅 30 分鐘後: {summary} ({start[-5:]})")
                result["action"] = "sent_upcoming_alert"

        # Determine if we should reach out
        should_reach = self._should_reach_out(emotion, active_tasks, calendar_cache)

        if should_reach and result["action"] == "none":
            msg = await self._compose_caring_message(emotion, calendar_cache)
            if msg and self.telegram:
                await self.telegram.send(msg)
                result["action"] = "sent_caring_message"
                result["message"] = msg

        # Update patrol timestamp
        await self.memos.working_memory.set(
            "last_patrol", time.time(), agent_id="heartbeat"
        )

        logger.debug(f"Hourly patrol: emotion={emotion}, action={result['action']}")
        return result

    async def morning_brief(self) -> str:
        """Daily morning briefing: weather + calendar (gog) + reminders + agenda.

        Returns the formatted brief (for testing).
        """
        parts = ["☀️ *早安，Ted！*", ""]

        # Weather
        weather = await self._fetch_weather()
        parts.append(f"🌤 {weather}")
        parts.append("")

        # Today's calendar — prefer gog (real), fallback to MemOS cache
        agenda = self._get_gog_today_events()
        if agenda is not None:
            if agenda:
                parts.append("📋 *今日行程:*")
                for i, ev in enumerate(agenda, 1):
                    summary = ev.get("summary", "Unknown")
                    start = ev.get("start", {}).get("dateTime", "")
                    time_str = start[-5:] if start else ""
                    parts.append(f"  {i}. {time_str} {summary}")
            else:
                parts.append("📋 今日沒有排定的行程")
        else:
            # Fallback to MemOS cache
            cache_agenda = await self._get_today_agenda()
            if cache_agenda:
                parts.append("📋 *今日行程:*")
                for i, event in enumerate(cache_agenda, 1):
                    parts.append(f"  {i}. {event}")
            else:
                parts.append("📋 今日沒有排定的行程（行事曆未連線）")

        # Today's reminders
        if self.reminder:
            today_reminders = self.reminder.get_today()
            if today_reminders:
                parts.append("")
                parts.append("⏰ *今日提醒:*")
                for r in today_reminders:
                    t = r["remind_at"][-5:]  # HH:MM
                    parts.append(f"  - {t} {r['content']}")

        # Trading day hint (weekday < 5)
        if datetime.now().weekday() < 5:
            parts.append("")
            parts.append("📈 今天是交易日")

        # Token saving report
        if self.survival and self.survival.tracker:
            try:
                saving = await self.survival.tracker.daily_report()
                if saving["total_calls"] > 0:
                    parts.append("")
                    parts.append(f"📊 昨日 Token 節省率: {saving['avg_saving_rate']} ({saving['total_calls']} calls)")
            except Exception:
                pass

        brief = "\n".join(parts)

        if self.telegram:
            await self.telegram.send(brief)

        logger.info("Morning brief sent")
        return brief

    async def health_check(self) -> dict[str, Any]:
        """Run SurvivalGate full diagnostics.

        Returns health report dict (for testing).
        """
        if not self.survival:
            logger.warning("Health check skipped: no SurvivalGate configured")
            return {"status": "skipped"}

        report = await self.survival.full_check()

        if report.has_alerts and self.telegram:
            await self.telegram.send(report.format())

        # Also probe recovery for any downed providers
        if self.router:
            recovery = await self.router.probe_recovery()
            if recovery and self.telegram:
                for provider, status in recovery.items():
                    if status == "recovered":
                        await self.telegram.send(f"✅ {provider} 已恢復正常服務")

        return {
            "status": "completed",
            "checks": len(report.checks),
            "alerts": len(report.alerts),
        }

    async def nightly_backup(self) -> str | None:
        """03:00 encrypted backup of MemOS database.

        Returns backup file path (for testing).
        """
        if not self.memos:
            logger.warning("Backup skipped: no MemOS configured")
            return None

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        dest = f"./backups/{timestamp}/memos.db"

        # Ensure backup directory exists
        from pathlib import Path
        Path(dest).parent.mkdir(parents=True, exist_ok=True)

        try:
            result = await self.memos.backup(dest, encrypt=True)
            logger.info(f"Nightly backup completed: {result}")

            if self.telegram:
                await self.telegram.send(f"💾 備份完成: {result}")

            return result
        except Exception as e:
            logger.error(f"Nightly backup failed: {e}")
            if self.telegram:
                await self.telegram.send(f"🔴 備份失敗: {e}")
            return None

    async def night_owl_check(self) -> bool:
        """Detect late-night activity and suggest rest.

        Returns True if a reminder was sent.
        """
        if not self.memos or not self.telegram:
            return False

        # Check if user had recent activity
        last_activity = await self.memos.working_memory.get("last_user_activity")
        if not last_activity:
            return False

        # If last activity was within the last 30 minutes, user is still awake
        if time.time() - last_activity < 1800:
            current_hour = datetime.now().hour
            # Only remind between 00:00-04:59
            if 0 <= current_hour < 5:
                # Check if we already reminded recently
                last_remind = await self.memos.working_memory.get("last_night_remind", 0)
                if time.time() - last_remind > 7200:  # Don't remind more than once per 2 hours
                    await self.telegram.send(
                        "🌙 Ted，現在很晚了，要不要考慮先休息？\n明天的事情我會幫你記著的 💤"
                    )
                    await self.memos.working_memory.set(
                        "last_night_remind", time.time(), agent_id="heartbeat"
                    )
                    return True
        return False

    async def retry_pending_tasks(self) -> dict[str, Any]:
        """Retry pending tasks via ReactExecutor.

        Returns summary dict for testing.
        """
        if not self.pending or not self.react:
            return {"retried": 0, "succeeded": 0, "failed": 0}

        due = self.pending.get_due_tasks()
        if not due:
            return {"retried": 0, "succeeded": 0, "failed": 0}

        succeeded = 0
        failed = 0

        for task in due:
            try:
                result = await self.react.execute(
                    task.task_type, task.task_description, **task.kwargs,
                )
                if result.success:
                    self.pending.mark_completed(task.task_id)
                    succeeded += 1
                    logger.info(f"Pending task {task.task_id} succeeded on retry")
                    if self.telegram:
                        await self.telegram.send(
                            f"✅ 待辦任務完成: {task.task_description[:60]}"
                        )
                else:
                    self.pending.mark_failed(task.task_id, result.gave_up_reason)
                    failed += 1
            except Exception as e:
                self.pending.mark_failed(task.task_id, str(e))
                failed += 1
                logger.warning(f"Pending task {task.task_id} retry error: {e}")

        # Notify about given-up tasks
        given_up = self.pending.get_given_up_tasks()
        if given_up and self.telegram:
            for task in given_up:
                await self.telegram.send(
                    f"❌ 任務放棄: {task.task_description[:60]}\n"
                    f"原因: {task.last_error[:100]}"
                )
            self.pending.clear_given_up()

        self.pending.save()

        summary = {"retried": len(due), "succeeded": succeeded, "failed": failed}
        logger.info(f"Pending task retry: {summary}")
        return summary

    async def evening_summary(self) -> str:
        """Nightly summary: today recap + tomorrow preview.

        Returns the formatted summary (for testing).
        """
        parts = ["🌙 *晚安，Ted*", ""]

        # Today's recap — markdown log line count
        try:
            from memory.markdown_memory import MarkdownMemory
            md = MarkdownMemory("./memory")
            today_log = md.read_daily()
            if today_log:
                line_count = len([l for l in today_log.strip().splitlines() if l.startswith("- ")])
                parts.append(f"📝 今日記錄: {line_count} 條")
            else:
                parts.append("📝 今日沒有特別記錄")
        except Exception:
            parts.append("📝 今日記錄: (無法讀取)")

        # Tomorrow's calendar via gog
        tomorrow = datetime.now() + timedelta(days=1)
        tomorrow_events = self._get_gog_events_for_date(tomorrow)
        parts.append("")
        if tomorrow_events is not None:
            if tomorrow_events:
                parts.append("📋 *明日行程:*")
                for i, ev in enumerate(tomorrow_events, 1):
                    summary = ev.get("summary", "Unknown")
                    start = ev.get("start", {}).get("dateTime", "")
                    time_str = start[-5:] if start else ""
                    parts.append(f"  {i}. {time_str} {summary}")
            else:
                parts.append("📋 明日沒有排定的行程")
        else:
            parts.append("📋 明日行程: (行事曆未連線)")

        # Tomorrow's reminders
        if self.reminder:
            tomorrow_reminders = self.reminder.get_for_date(tomorrow)
            if tomorrow_reminders:
                parts.append("")
                parts.append("⏰ *明日提醒:*")
                for r in tomorrow_reminders:
                    t = r["remind_at"][-5:]
                    parts.append(f"  - {t} {r['content']}")

        parts.append("")
        parts.append("好好休息，明天見 💤")

        summary = "\n".join(parts)

        if self.telegram:
            await self.telegram.send(summary)

        logger.info("Evening summary sent")
        return summary

    async def memory_cleanup(self) -> dict[str, int]:
        """Clean up old fired reminders.

        Returns cleanup result dict (for testing).
        """
        result = {"reminders_removed": 0}
        if self.reminder:
            result["reminders_removed"] = self.reminder.cleanup(days=7)
        logger.info(f"Memory cleanup: {result}")
        return result

    # ── Helpers ──────────────────────────────────────────────────

    def _should_reach_out(
        self,
        emotion: str,
        active_tasks: list,
        calendar_events: list,
    ) -> bool:
        """Decide if proactive outreach is warranted."""
        # Reach out if user seems distressed
        if emotion in ("anxious", "tired", "sad", "frustrated"):
            return True

        # Reach out if there's an upcoming calendar event (within 2 hours)
        # (calendar integration in Task 6)
        if calendar_events:
            return True

        return False

    async def _compose_caring_message(
        self, emotion: str, events: list
    ) -> str | None:
        """Use CEO model to compose a context-appropriate caring message."""
        if not self.router:
            return None

        try:
            from clients.base_client import ChatMessage
            from core.model_router import ModelRole

            prompt_parts = [f"用戶目前的情緒狀態: {emotion}"]
            if events:
                prompt_parts.append(f"即將到來的行程: {events}")

            prompt = (
                "你是 J.A.R.V.I.S.，Tony Stark 的 AI 管家。"
                "請根據以下資訊，用簡短溫暖的中文（繁體）寫一段關懷訊息（不超過 100 字）。\n"
                + "\n".join(prompt_parts)
            )

            response = await self.router.chat(
                [ChatMessage(role="user", content=prompt)],
                role=ModelRole.CEO,
                max_tokens=200,
            )
            return response.content
        except Exception as e:
            logger.warning(f"Failed to compose caring message: {e}")
            return None

    async def _fetch_weather(self) -> str:
        """Fetch weather via Open-Meteo."""
        if self.weather:
            return await self.weather.get_brief()
        return "天氣資訊尚未接入"

    async def _get_today_agenda(self) -> list[str]:
        """Get today's calendar events from MemOS cache."""
        if not self.memos:
            return []

        cache = await self.memos.working_memory.get("calendar_cache", [])
        if not cache:
            return []

        # Filter for today's events
        today = datetime.now().strftime("%Y-%m-%d")
        today_events = [
            e.get("summary", "Unknown event")
            for e in cache
            if isinstance(e, dict) and e.get("date", "").startswith(today)
        ]
        return today_events

    def _get_gog_today_events(self) -> list[dict] | None:
        """Get today's events via gog CLI. Returns None if gog unavailable."""
        if not self.gog or not self.gog.is_available:
            return None
        try:
            return self.gog.get_today_events()
        except Exception as e:
            logger.warning(f"gog get_today_events failed: {e}")
            return None

    def _get_gog_events_for_date(self, date: datetime) -> list[dict] | None:
        """Get events for a date via gog CLI. Returns None if gog unavailable."""
        if not self.gog or not self.gog.is_available:
            return None
        try:
            return self.gog.get_events_for_date(date)
        except Exception as e:
            logger.warning(f"gog get_events_for_date failed: {e}")
            return None

    def _get_upcoming_gog_events(self, minutes: int = 60) -> list[dict]:
        """Get upcoming events via gog CLI. Returns empty list if unavailable."""
        if not self.gog or not self.gog.is_available:
            return []
        try:
            return self.gog.get_upcoming_events(minutes)
        except Exception as e:
            logger.warning(f"gog get_upcoming_events failed: {e}")
            return []
