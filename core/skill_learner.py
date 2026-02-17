"""Skill Learner — observe user patterns, propose automations.

Watches user behavior over a rolling window, detects repetitive
patterns, and proposes learned skills (scheduled automations) via
Telegram.  Approved patterns become skills in ``skills/learned/``.

Usage:
    learner = SkillLearner(scheduler=sched, telegram=tg, model_router=router)
    learner.log_action({"type": "weather_check", "detail": "台北天氣"})
    patterns = await learner.detect_patterns()
    await learner.propose_skills()
"""

from __future__ import annotations

import json
import time
from collections import Counter, defaultdict
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

from loguru import logger


class SkillLearner:
    """Observe user actions, detect patterns, propose automations."""

    LOG_PATH = Path("./data/user_actions.json")
    SKILL_DIR = Path("./skills/learned/")
    PROPOSALS_PATH = Path("./data/skill_proposals.json")
    MIN_REPEAT = 3
    WINDOW_DAYS = 14

    def __init__(
        self,
        scheduler: Any = None,
        telegram: Any = None,
        model_router: Any = None,
    ) -> None:
        self.scheduler = scheduler
        self.telegram = telegram
        self.router = model_router
        self._actions: list[dict[str, Any]] = []
        self._load_actions()

    # ── Action logging ────────────────────────────────────────────

    def log_action(self, action: dict[str, Any]) -> None:
        """Record a user action for pattern analysis.

        Args:
            action: dict with at least ``type`` (str).  Optional keys:
                detail (str), timestamp (float), weekday (int 0-6),
                hour (int 0-23).
        """
        now = datetime.now()
        action.setdefault("timestamp", time.time())
        action.setdefault("weekday", now.weekday())
        action.setdefault("hour", now.hour)
        action.setdefault("date", now.strftime("%Y-%m-%d"))
        self._actions.append(action)

        # Evict old entries
        cutoff = time.time() - self.WINDOW_DAYS * 86400
        self._actions = [a for a in self._actions if a.get("timestamp", 0) > cutoff]

        self._save_actions()

    # ── Pattern detection ─────────────────────────────────────────

    async def detect_patterns(self) -> list[dict[str, Any]]:
        """Analyze logged actions and find repeating patterns.

        Returns:
            list of pattern dicts with type, count, frequency, hours,
            weekdays, detail_sample.
        """
        if not self._actions:
            return []

        # Group by action type
        by_type: dict[str, list[dict]] = defaultdict(list)
        for action in self._actions:
            action_type = action.get("type", "unknown")
            by_type[action_type].append(action)

        patterns: list[dict[str, Any]] = []
        for action_type, items in by_type.items():
            if len(items) < self.MIN_REPEAT:
                continue

            hours = [item.get("hour", 0) for item in items]
            weekdays = [item.get("weekday", 0) for item in items]
            details = [item.get("detail", "") for item in items if item.get("detail")]

            pattern: dict[str, Any] = {
                "type": action_type,
                "count": len(items),
                "frequency": self._guess_frequency(items),
                "peak_hours": self._top_n(hours, 2),
                "peak_weekdays": self._top_n(weekdays, 3),
                "detail_sample": details[:3] if details else [],
            }
            patterns.append(pattern)

        # Sort by count descending
        patterns.sort(key=lambda p: p["count"], reverse=True)
        return patterns

    # ── Skill proposal ────────────────────────────────────────────

    async def propose_skills(self) -> list[str]:
        """Detect patterns and send proposals to Telegram.

        Returns:
            list of proposal message strings sent.
        """
        patterns = await self.detect_patterns()
        if not patterns:
            return []

        proposals: list[str] = []
        for pattern in patterns:
            if self._already_proposed(pattern["type"]):
                continue

            msg = await self._generate_proposal(pattern)
            if msg:
                if self.telegram:
                    try:
                        await self.telegram.send_message(msg)
                    except Exception as e:
                        logger.debug(f"SkillLearner: send proposal failed: {e}")
                self._mark_proposed(pattern["type"])
                proposals.append(msg)

        return proposals

    # ── Skill creation ────────────────────────────────────────────

    async def create_skill_from_pattern(
        self, pattern: dict[str, Any], instructions: str = "",
    ) -> dict[str, Any]:
        """Create a learned skill from a detected pattern.

        Args:
            pattern: Pattern dict from detect_patterns().
            instructions: User's instructions for how the skill should work.

        Returns:
            dict with name, path, schedule info.
        """
        self.SKILL_DIR.mkdir(parents=True, exist_ok=True)

        name = f"auto_{pattern['type']}"
        skill_dir = self.SKILL_DIR / name
        skill_dir.mkdir(parents=True, exist_ok=True)

        # Build skill config
        cron = self._pattern_to_cron(pattern)
        skill_config = {
            "name": name,
            "display_name": f"Auto: {pattern['type']}",
            "version": "1.0",
            "category": "learned",
            "description": instructions or f"自動執行 {pattern['type']} (學習自用戶行為)",
            "pattern": {
                "type": pattern["type"],
                "frequency": pattern["frequency"],
                "count": pattern["count"],
            },
            "schedule": cron,
            "created_at": datetime.now().isoformat(),
        }

        # Save skill.yaml
        config_path = skill_dir / "skill.yaml"
        import yaml
        with open(config_path, "w", encoding="utf-8") as f:
            yaml.dump(skill_config, f, allow_unicode=True, default_flow_style=False)

        # Register APScheduler job if scheduler available
        if self.scheduler and cron:
            try:
                self.scheduler.add_job(
                    self._noop_skill,
                    "cron",
                    **cron,
                    id=f"skill_{name}",
                    name=f"Learned Skill: {name}",
                    replace_existing=True,
                )
                logger.info(f"SkillLearner: scheduled skill '{name}' with cron {cron}")
            except Exception as e:
                logger.warning(f"SkillLearner: schedule failed for '{name}': {e}")

        return {
            "name": name,
            "path": str(skill_dir),
            "schedule": cron,
            "config": skill_config,
        }

    # ── Internal helpers ──────────────────────────────────────────

    def _guess_frequency(self, items: list[dict]) -> str:
        """Guess whether a pattern is daily, weekly, or irregular."""
        if len(items) < 2:
            return "irregular"

        dates = sorted(set(item.get("date", "") for item in items))
        if len(dates) < 2:
            return "irregular"

        # Compute average gap between dates
        try:
            dt_dates = [datetime.strptime(d, "%Y-%m-%d") for d in dates if d]
            if len(dt_dates) < 2:
                return "irregular"
            gaps = [
                (dt_dates[i + 1] - dt_dates[i]).days
                for i in range(len(dt_dates) - 1)
            ]
            avg_gap = sum(gaps) / len(gaps)
            if avg_gap <= 1.5:
                return "daily"
            elif 5 <= avg_gap <= 8:
                return "weekly"
            else:
                return "irregular"
        except (ValueError, ZeroDivisionError):
            return "irregular"

    async def _generate_proposal(self, pattern: dict[str, Any]) -> str:
        """Generate a natural JARVIS-style proposal message."""
        freq_text = {
            "daily": "每天",
            "weekly": "每週",
            "irregular": "不定期",
        }.get(pattern["frequency"], "不定期")

        hour_text = ""
        if pattern.get("peak_hours"):
            hour_text = f"通常在 {', '.join(str(h) for h in pattern['peak_hours'])} 點"

        details = ""
        if pattern.get("detail_sample"):
            details = f"（例如：{'、'.join(pattern['detail_sample'][:2])}）"

        base = (
            f"Sir，我注意到你{freq_text}都會做「{pattern['type']}」{details}"
            f"，已經 {pattern['count']} 次了{' ' + hour_text if hour_text else ''}。\n\n"
            f"要不要我自動幫你處理？我可以設定{freq_text}自動執行。\n"
            f"回覆「好」我就幫你設定。"
        )

        # If LLM available, refine the message
        if self.router:
            try:
                refined = await self._llm_refine(base, pattern)
                if refined and len(refined) > 20:
                    return refined
            except Exception:
                pass

        return base

    async def _llm_refine(self, base_msg: str, pattern: dict) -> str:
        """Optionally refine proposal via LLM."""
        from clients.base_client import ChatMessage
        from core.model_router import ModelRole

        prompt = (
            f"以下是 JARVIS 要發給用戶的技能建議訊息：\n\n"
            f"{base_msg}\n\n"
            f"請用 JARVIS 的語氣（稱呼 Sir，結論先行，自然口語）"
            f"稍微潤飾這段訊息，保持簡短（3-4 行）。只回覆修改後的訊息。"
        )
        response = await self.router.chat(
            [ChatMessage(role="user", content=prompt)],
            role=ModelRole.CEO,
            max_tokens=200,
        )
        return response.content

    def _pattern_to_cron(self, pattern: dict[str, Any]) -> dict[str, Any]:
        """Convert pattern to APScheduler cron kwargs."""
        cron: dict[str, Any] = {}
        freq = pattern.get("frequency", "irregular")
        hours = pattern.get("peak_hours", [])

        if freq == "daily" and hours:
            cron["hour"] = hours[0]
            cron["minute"] = 0
        elif freq == "weekly" and hours:
            weekdays = pattern.get("peak_weekdays", [])
            cron["hour"] = hours[0]
            cron["minute"] = 0
            if weekdays:
                # APScheduler uses 'mon'-'sun' or 0-6
                cron["day_of_week"] = ",".join(str(d) for d in weekdays)
        else:
            # Irregular — default to daily at detected peak hour
            if hours:
                cron["hour"] = hours[0]
                cron["minute"] = 0

        return cron

    def _already_proposed(self, action_type: str) -> bool:
        """Check if this pattern type was already proposed."""
        proposals = self._load_proposals()
        return action_type in proposals

    def _mark_proposed(self, action_type: str) -> None:
        """Mark a pattern type as proposed."""
        proposals = self._load_proposals()
        proposals[action_type] = datetime.now().isoformat()
        self._save_proposals(proposals)

    @staticmethod
    def _top_n(values: list[int], n: int) -> list[int]:
        """Return top N most common values."""
        if not values:
            return []
        counter = Counter(values)
        return [v for v, _ in counter.most_common(n)]

    async def _noop_skill(self) -> None:
        """Placeholder for learned skill execution."""
        logger.debug("SkillLearner: noop skill fired (placeholder)")

    # ── Persistence ───────────────────────────────────────────────

    def _load_actions(self) -> None:
        """Load action log from disk."""
        if self.LOG_PATH.exists():
            try:
                with open(self.LOG_PATH, encoding="utf-8") as f:
                    self._actions = json.load(f)
            except (json.JSONDecodeError, OSError) as e:
                logger.debug(f"SkillLearner: load actions failed: {e}")
                self._actions = []
        else:
            self._actions = []

    def _save_actions(self) -> None:
        """Save action log to disk."""
        self.LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
        try:
            with open(self.LOG_PATH, "w", encoding="utf-8") as f:
                json.dump(self._actions, f, ensure_ascii=False, indent=2)
        except OSError as e:
            logger.debug(f"SkillLearner: save actions failed: {e}")

    def _load_proposals(self) -> dict[str, str]:
        """Load proposal tracker from disk."""
        if self.PROPOSALS_PATH.exists():
            try:
                with open(self.PROPOSALS_PATH, encoding="utf-8") as f:
                    return json.load(f)
            except (json.JSONDecodeError, OSError):
                pass
        return {}

    def _save_proposals(self, proposals: dict[str, str]) -> None:
        """Save proposal tracker to disk."""
        self.PROPOSALS_PATH.parent.mkdir(parents=True, exist_ok=True)
        try:
            with open(self.PROPOSALS_PATH, "w", encoding="utf-8") as f:
                json.dump(proposals, f, ensure_ascii=False, indent=2)
        except OSError as e:
            logger.debug(f"SkillLearner: save proposals failed: {e}")
