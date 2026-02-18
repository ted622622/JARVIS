"""Heartbeat — proactive scheduling engine for J.A.R.V.I.S.

Scheduled jobs:
- hourly_patrol:    every 60 min — check calendar, emotions, upcoming events alert
- morning_brief:    daily 07:30 — weather + calendar (gog) + reminders + agenda
- evening_summary:  daily 23:00 — today recap + tomorrow preview
- health_check:     every 6 hrs — SurvivalGate full diagnostics
- backup:           daily 03:00 — encrypted MemOS + skills backup
- memory_cleanup:   daily 03:15 — purge old fired reminders
- night_owl:        cron check  — detect late-night activity, suggest rest
- clawra_morning:   daily 08:30 — Clawra morning greeting
- clawra_daily_share: daily ~15:00 — Clawra shares Seoul life / random thoughts
- clawra_evening:   daily 22:00 — Clawra goodnight
- clawra_missing_check: every 2h — Clawra "你在幹嘛" if silent too long
"""

from __future__ import annotations

import json
import random
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

from apscheduler.schedulers.asyncio import AsyncIOScheduler
from loguru import logger

# ── Clawra proactive behavior constants ──────────────────────────
_CLAWRA_MAX_DAILY = 3           # max proactive messages per day
_CLAWRA_QUIET_START = 0         # no proactive messages 00:00-07:59
_CLAWRA_QUIET_END = 8
_CLAWRA_MIN_SILENCE_HOURS = 4   # hours without user msg before "missing" check
_CLAWRA_COOLDOWN_SECS = 3600    # min gap between proactive messages (1 hr)


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
        fal_client: Any = None,
        soul: Any = None,
        voice_worker: Any = None,
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
        self.fal_client = fal_client
        self.soul = soul
        self._voice_worker = voice_worker

        self.ceo = None  # set externally for Agent SDK usage reporting
        self.scheduler = AsyncIOScheduler()
        self._running = False

        # Clawra proactive behavior state
        self._clawra_daily_count = 0
        self._clawra_daily_date = ""  # "YYYY-MM-DD" for daily reset
        self._clawra_last_sent = 0.0  # timestamp of last proactive msg

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
        cleanup_minute = bm + 15
        cleanup_hour = bh + cleanup_minute // 60
        cleanup_minute = cleanup_minute % 60
        self.scheduler.add_job(
            self.memory_cleanup,
            "cron",
            hour=cleanup_hour,
            minute=cleanup_minute,
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

        # Pending selfie check (Patch M)
        if self.fal_client:
            self.scheduler.add_job(
                self.check_pending_selfies,
                "interval",
                minutes=5,
                id="pending_selfies_check",
                name="Pending Selfie Check",
            )

        # ── Clawra proactive behavior ────────────────────────────
        if self.soul and self.telegram:
            # Morning greeting (08:30)
            self.scheduler.add_job(
                self.clawra_morning,
                "cron",
                hour=8,
                minute=30,
                id="clawra_morning",
                name="Clawra Morning Greeting",
            )
            # Daily share — Seoul life / random thoughts (random 13-17)
            share_hour = random.randint(13, 17)
            share_minute = random.randint(0, 59)
            self.scheduler.add_job(
                self.clawra_daily_share,
                "cron",
                hour=share_hour,
                minute=share_minute,
                id="clawra_daily_share",
                name="Clawra Daily Share",
            )
            # Evening goodnight (22:00)
            self.scheduler.add_job(
                self.clawra_evening,
                "cron",
                hour=22,
                minute=0,
                id="clawra_evening",
                name="Clawra Evening",
            )
            # Missing check — "你在幹嘛" if silent too long (every 2h, 08-22)
            self.scheduler.add_job(
                self.clawra_missing_check,
                "cron",
                hour="8-22/2",
                minute=15,
                id="clawra_missing_check",
                name="Clawra Missing Check",
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

        # Agent SDK token usage
        ceo = getattr(self, "ceo", None)
        if ceo and hasattr(ceo, "_agent_executor") and ceo._agent_executor:
            try:
                parts.append("")
                parts.append(ceo._agent_executor.get_usage_line())
            except Exception:
                pass

        # Token pool balance status
        try:
            from core.model_balancer import get_status, check_alert
            parts.append(f"\n📊 Token: {get_status()}")
            alert = check_alert()
            if alert:
                parts.append(alert)
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

    async def check_pending_selfies(self) -> dict[str, Any]:
        """Check if any delayed selfie generations completed on fal.ai."""
        path = Path("./data/pending_selfies.json")
        if not path.exists():
            return {"checked": 0, "delivered": 0}

        try:
            entries = json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return {"checked": 0, "delivered": 0}

        updated: list[dict] = []
        delivered = 0

        for entry in entries:
            if entry.get("status") != "pending":
                updated.append(entry)
                continue

            # Expire after 1 hour
            if time.time() - entry.get("created_at", 0) > 3600:
                entry["status"] = "expired"
                updated.append(entry)
                continue

            try:
                status = await self.fal_client.check_queue_status(entry["status_url"])
                if status == "COMPLETED":
                    result = await self.fal_client.fetch_queue_result(entry["response_url"])
                    persona = entry.get("persona", "clawra")
                    caption = "欸嘿 剛剛的照片好了！" if persona == "clawra" else "Sir, 照片已備妥。"
                    if self.telegram:
                        await self.telegram.send_photo(result.url, caption=caption, persona=persona)
                    entry["status"] = "delivered"
                    delivered += 1
                elif status == "FAILED":
                    entry["status"] = "failed"
                else:
                    pass  # still pending, keep as-is
            except Exception as e:
                logger.warning(f"Pending selfie check error: {e}")

            updated.append(entry)

        # Save back
        path.write_text(json.dumps(updated, ensure_ascii=False, indent=2), encoding="utf-8")
        if delivered:
            logger.info(f"Delivered {delivered} delayed selfie(s)")
        return {"checked": len(entries), "delivered": delivered}

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

    # ── Clawra Proactive Behavior ────────────────────────────────

    def _clawra_can_send(self) -> bool:
        """Check if Clawra is allowed to send a proactive message now."""
        now = datetime.now()
        # Quiet hours: 00:00-07:59
        if _CLAWRA_QUIET_START <= now.hour < _CLAWRA_QUIET_END:
            return False
        # Daily count reset
        today = now.strftime("%Y-%m-%d")
        if self._clawra_daily_date != today:
            self._clawra_daily_date = today
            self._clawra_daily_count = 0
        # Over daily limit
        if self._clawra_daily_count >= _CLAWRA_MAX_DAILY:
            return False
        # Cooldown
        if time.time() - self._clawra_last_sent < _CLAWRA_COOLDOWN_SECS:
            return False
        return True

    def _clawra_did_send(self) -> None:
        """Record that a Clawra proactive message was sent."""
        self._clawra_daily_count += 1
        self._clawra_last_sent = time.time()

    async def _send_clawra(self, text: str, *, voice_chance: float = 0.2) -> bool:
        """Send a message from the Clawra bot.

        Args:
            text: Message text
            voice_chance: Probability (0-1) of sending as voice message instead of text.
                          Defaults to 20%.

        Returns True on success.
        """
        if not self.telegram or not text:
            return False
        # Apply s2t conversion for consistency
        try:
            from opencc import OpenCC
            cc = OpenCC("s2t")
            text = cc.convert(text)
        except ImportError:
            pass
        # P5: Random chance to send as voice message
        if voice_chance > 0 and random.random() < voice_chance and self._voice_worker:
            try:
                audio_path = await self._voice_worker.text_to_speech(text, persona="clawra")
                if audio_path:
                    await self.telegram.send_voice(audio_path, persona="clawra")
                    self._clawra_did_send()
                    return True
            except Exception as e:
                logger.debug(f"Clawra voice fallback to text: {e}")
        # Fallback: plain text
        await self.telegram.send(text, persona="clawra")
        self._clawra_did_send()
        return True

    async def _compose_clawra_message(self, scenario: str) -> str | None:
        """Use LLM with SOUL_CLAWRA.md to generate a proactive message."""
        if not self.router or not self.soul:
            return None
        try:
            from clients.base_client import ChatMessage
            from core.model_router import ModelRole

            soul_prompt = self.soul.build_system_prompt("clawra")

            prompt = (
                f"你現在要主動傳訊息給 Ted。場景：{scenario}\n"
                "要求：\n"
                "- 用你的說話風格（口語、簡短、不用顏文字）\n"
                "- 就像真的在 LINE/TG 傳訊息一樣\n"
                "- 只回覆訊息內容本身，不要加任何說明\n"
                "- 不超過 50 字\n"
                "- 繁體中文"
            )

            response = await self.router.chat(
                [
                    ChatMessage(role="system", content=soul_prompt),
                    ChatMessage(role="user", content=prompt),
                ],
                role=ModelRole.CEO,
                task_type="cron_message",
                max_tokens=100,
            )
            reply = response.content.strip().strip('"').strip("'")
            # Clean leaked think tags
            import re
            reply = re.sub(r"</?think>", "", reply, flags=re.IGNORECASE)
            reply = re.sub(r"<think>.*?</think>", "", reply, flags=re.DOTALL | re.IGNORECASE)
            return reply if reply else None
        except Exception as e:
            logger.warning(f"Failed to compose Clawra message: {e}")
            return None

    async def clawra_morning(self) -> str | None:
        """08:30 — Clawra morning greeting to Ted."""
        if not self._clawra_can_send():
            return None
        msg = await self._compose_clawra_message(
            "早上好，你剛起床準備傳早安訊息給男友。"
            "首爾今天的天氣或你早上的狀況可以提一下。"
        )
        if msg:
            await self._send_clawra(msg)
            logger.info(f"Clawra morning greeting sent: {msg[:30]}")
        return msg

    async def clawra_daily_share(self) -> str | None:
        """Random time 13-17 — Clawra shares Seoul daily life."""
        if not self._clawra_can_send():
            return None
        scenarios = [
            "你在首爾逛街看到有趣的東西，想跟男友分享",
            "你剛吃完午餐，覺得很好吃想跟他說",
            "你在咖啡廳坐著，突然想到他",
            "首爾今天天氣不錯，你拍了路上的風景想傳給他",
            "你看到一個很可愛的東西想跟他說",
            "你在弘大那邊發現一家新開的店",
        ]
        msg = await self._compose_clawra_message(random.choice(scenarios))
        if msg:
            await self._send_clawra(msg)
            logger.info(f"Clawra daily share sent: {msg[:30]}")
        return msg

    async def clawra_evening(self) -> str | None:
        """22:00 — Clawra goodnight message."""
        if not self._clawra_can_send():
            return None
        msg = await self._compose_clawra_message(
            "晚上了，你準備要睡了。跟男友說晚安。"
            "可以問他今天怎麼樣，或叫他早點睡。"
        )
        if msg:
            await self._send_clawra(msg)
            logger.info(f"Clawra evening sent: {msg[:30]}")
        return msg

    async def clawra_missing_check(self) -> str | None:
        """Every 2h — if Ted hasn't messaged in 4+ hours, Clawra reaches out."""
        if not self._clawra_can_send():
            return None
        if not self.memos:
            return None
        # Check last user activity
        last_activity = await self.memos.working_memory.get("last_user_activity")
        if not last_activity:
            return None
        silence_hours = (time.time() - last_activity) / 3600
        if silence_hours < _CLAWRA_MIN_SILENCE_HOURS:
            return None
        msg = await self._compose_clawra_message(
            f"男友已經 {int(silence_hours)} 小時沒有找你了，你想知道他在幹嘛。"
        )
        if msg:
            await self._send_clawra(msg)
            logger.info(f"Clawra missing check sent (silent {silence_hours:.1f}h): {msg[:30]}")
        return msg

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

            # Use JARVIS soul prompt if available, else fallback
            messages = []
            if self.soul:
                soul_prompt = self.soul.build_system_prompt("jarvis")
                messages.append(ChatMessage(role="system", content=soul_prompt))

            prompt = (
                "請根據以下資訊，用簡短溫暖的中文（繁體）寫一段關懷訊息（不超過 100 字）。\n"
                + "\n".join(prompt_parts)
            )
            messages.append(ChatMessage(role="user", content=prompt))

            response = await self.router.chat(
                messages,
                role=ModelRole.CEO,
                task_type="cron_message",
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
