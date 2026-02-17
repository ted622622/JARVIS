"""PostActionChain — event-driven action automation.

After a booking or event is confirmed, automatically:
1. Create a Google Calendar event (via GogWorker)
2. Set reminders (via ReminderManager)

Usage:
    chain = PostActionChain(gog_worker=gog, reminder_manager=rm)
    result = await chain.execute_chain(
        "restaurant_booking",
        event_time=datetime(2026, 2, 18, 19, 0),
        params={"restaurant_name": "鼎泰豐", "address": "台北市..."},
    )
"""

from __future__ import annotations

from datetime import datetime, timedelta
from typing import Any

from loguru import logger


# Chain definitions: what actions to take per chain type
CHAINS: dict[str, dict[str, Any]] = {
    "restaurant_booking": {
        "calendar": {
            "title_template": "🍽 {restaurant_name}",
            "duration_minutes": 120,
        },
        "reminders": [
            {"before_minutes": 120, "msg": "{restaurant_name} 訂位，準備出門"},
            {"before_minutes": 30, "msg": "再 30 分鐘到 {restaurant_name}"},
        ],
    },
    "ticket_booking": {
        "calendar": {
            "title_template": "🎫 {event_name}",
            "duration_minutes": 180,
        },
        "reminders": [
            {"before_minutes": 1440, "msg": "明天有 {event_name}，記得準備"},
            {"before_minutes": 60, "msg": "{event_name} 還有 1 小時"},
        ],
    },
    "meeting": {
        "calendar": {
            "title_template": "📅 {meeting_title}",
            "duration_minutes": 60,
        },
        "reminders": [
            {"before_minutes": 30, "msg": "{meeting_title} 30 分鐘後開始"},
            {"before_minutes": 5, "msg": "{meeting_title} 即將開始"},
        ],
    },
}


class PostActionChain:
    """Execute post-action chains (booking → calendar + reminders)."""

    def __init__(
        self,
        gog_worker: Any = None,
        reminder_manager: Any = None,
    ):
        self._gog = gog_worker
        self._reminder = reminder_manager

    async def execute_chain(
        self,
        chain_type: str,
        event_time: datetime,
        params: dict[str, str] | None = None,
    ) -> dict[str, Any]:
        """Execute a full action chain.

        Args:
            chain_type: Key into CHAINS dict (e.g. "restaurant_booking").
            event_time: When the event starts.
            params: Template variables (e.g. restaurant_name, address).

        Returns:
            {"calendar_added": bool, "reminders_set": int}
        """
        chain = CHAINS.get(chain_type)
        if not chain:
            logger.warning(f"Unknown chain type: {chain_type}")
            return {"calendar_added": False, "reminders_set": 0}

        params = params or {}
        result: dict[str, Any] = {"calendar_added": False, "reminders_set": 0}

        # 1. Create calendar event
        cal_cfg = chain.get("calendar")
        if cal_cfg and self._gog:
            try:
                title = cal_cfg["title_template"].format(**params)
                duration = cal_cfg.get("duration_minutes", 60)
                location = params.get("address", "")
                gog_result = self._gog.create_event(
                    title=title,
                    start_time=event_time,
                    duration_minutes=duration,
                    location=location,
                )
                if gog_result.get("success"):
                    result["calendar_added"] = True
                    logger.info(f"Calendar event created: {title}")
                else:
                    logger.warning(f"Calendar event failed: {gog_result.get('error')}")
            except (KeyError, Exception) as e:
                logger.warning(f"Calendar creation error: {e}")

        # 2. Set reminders
        rem_specs = chain.get("reminders", [])
        if rem_specs and self._reminder:
            for spec in rem_specs:
                try:
                    before = spec["before_minutes"]
                    remind_at = event_time - timedelta(minutes=before)
                    if remind_at <= datetime.now():
                        continue  # Skip past reminders
                    msg = spec["msg"].format(**params)
                    await self._reminder.add(msg, remind_at, source="chain")
                    result["reminders_set"] += 1
                except (KeyError, Exception) as e:
                    logger.warning(f"Reminder creation error: {e}")

        logger.info(
            f"PostActionChain '{chain_type}': "
            f"calendar={'✓' if result['calendar_added'] else '✗'}, "
            f"reminders={result['reminders_set']}"
        )
        return result
