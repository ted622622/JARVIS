"""SharedMemory — record and recall shared moments with Clawra.

Patch J4: Tracks anniversaries, inside jokes, nicknames, and
important shared experiences in memory/clawra/SHARED_MOMENTS.md.

Provides context injection for Clawra persona so she can reference
past shared moments naturally.
"""

from __future__ import annotations

import re
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

from loguru import logger


# Patterns that suggest a memorable moment
_MEMORABLE_PATTERNS = re.compile(
    r"今天是.*(?:紀念|生日|周年|節日)|"
    r"(?:第一次|首次).*(?:一起|跟你)|"
    r"(?:我們的|咱們的).*(?:暱稱|綽號|秘密|約定)|"
    r"以後.*叫(?:我|你)|"
    r"記得.*(?:那天|那次|上次)|"
    r"這是我們的|好好笑|太好笑了|"
    r"(?:紀念日|情人節|聖誕|新年|跨年|萬聖節|中秋)",
    re.IGNORECASE,
)

# Date patterns for anniversary detection
_DATE_PATTERN = re.compile(
    r"(\d{1,2})[/月\-.](\d{1,2})[日號]?"
)

# Category tags for different types of moments
MOMENT_TYPES = {
    "anniversary": "紀念日",
    "nickname": "暱稱",
    "joke": "內部笑話",
    "milestone": "里程碑",
    "preference": "共同偏好",
    "memory": "共同回憶",
}


class SharedMemory:
    """Manages shared moments between Ted and Clawra.

    Usage:
        sm = SharedMemory(memory_dir="./memory")
        moment = sm.check_and_remember(user_msg, assistant_msg)
        recent = sm.get_recent(days=30)
        anniversary = sm.get_today_anniversary()
    """

    def __init__(self, memory_dir: str = "./memory"):
        self._memory_dir = Path(memory_dir)
        self._moments_path = self._memory_dir / "clawra" / "SHARED_MOMENTS.md"
        self._moments_path.parent.mkdir(parents=True, exist_ok=True)

    def check_and_remember(
        self,
        user_msg: str,
        assistant_msg: str,
    ) -> str | None:
        """Check if conversation contains a memorable moment and save it.

        Returns the saved moment description, or None.
        """
        if not _MEMORABLE_PATTERNS.search(user_msg):
            logger.debug(f"SharedMemory: no memorable pattern in: {user_msg[:50]}")
            return None

        moment = self._extract_moment(user_msg, assistant_msg)
        if not moment:
            return None

        # Check for duplicates
        existing = self._read_moments()
        moment_text = moment["text"]
        if any(moment_text.lower() in m.lower() or m.lower() in moment_text.lower() for m in existing):
            return None

        self._save_moment(moment)
        logger.info(f"SharedMemory: saved moment — {moment_text[:60]}")
        return moment_text

    def get_recent(self, days: int = 30) -> list[str]:
        """Get moments from the last N days for context injection."""
        cutoff = datetime.now() - timedelta(days=days)
        cutoff_str = cutoff.strftime("%Y-%m-%d")

        moments = []
        for line in self._read_all_lines():
            # Extract date from <!-- YYYY-MM-DD --> comment
            date_match = re.search(r"<!-- (\d{4}-\d{2}-\d{2})", line)
            if date_match and date_match.group(1) >= cutoff_str:
                # Strip the date comment for clean display
                clean = re.sub(r"\s*<!--.*?-->", "", line).strip()
                if clean:
                    moments.append(clean)

        return moments

    def get_today_anniversary(self) -> list[str]:
        """Check if today matches any recorded anniversary dates.

        Returns list of anniversary descriptions, empty if none.
        """
        today = datetime.now()
        today_md = f"{today.month:02d}-{today.day:02d}"

        anniversaries = []
        for line in self._read_all_lines():
            if "紀念日" not in line and "anniversary" not in line.lower():
                continue
            # Look for month-day patterns
            date_match = _DATE_PATTERN.search(line)
            if date_match:
                month = int(date_match.group(1))
                day = int(date_match.group(2))
                if f"{month:02d}-{day:02d}" == today_md:
                    clean = re.sub(r"\s*<!--.*?-->", "", line).strip()
                    anniversaries.append(clean)

        return anniversaries

    def get_context_for_prompt(self) -> str:
        """Build a context string for system prompt injection.

        Returns recent moments + any today's anniversaries.
        """
        parts = []

        # Today's anniversaries (highest priority)
        anniversaries = self.get_today_anniversary()
        if anniversaries:
            parts.append("今天的紀念日：")
            parts.extend(anniversaries)

        # Recent shared moments (last 14 days)
        recent = self.get_recent(days=14)
        if recent:
            parts.append("最近的共同記憶：")
            parts.extend(recent[-5:])  # Last 5 recent moments

        return "\n".join(parts) if parts else ""

    def _extract_moment(
        self, user_msg: str, assistant_msg: str,
    ) -> dict[str, str] | None:
        """Extract a structured moment from conversation."""
        msg = user_msg.strip()

        # Nickname declarations
        if re.search(r"以後.*叫(?:我|你)", msg):
            return {"type": "nickname", "text": f"[暱稱] {msg[:100]}"}

        # Anniversary / date-based events
        if re.search(r"紀念|周年|生日|節日", msg):
            return {"type": "anniversary", "text": f"[紀念日] {msg[:100]}"}

        # Inside jokes
        if re.search(r"好好笑|太好笑了|笑死", msg):
            return {"type": "joke", "text": f"[笑話] {msg[:80]} → {assistant_msg[:40]}"}

        # First-time experiences
        if re.search(r"第一次|首次", msg):
            return {"type": "milestone", "text": f"[里程碑] {msg[:100]}"}

        # General shared memory
        if re.search(r"記得|那天|那次|我們的", msg):
            return {"type": "memory", "text": f"[回憶] {msg[:100]}"}

        # Holiday mentions
        if re.search(r"聖誕|新年|跨年|萬聖節|中秋|情人節", msg):
            return {"type": "memory", "text": f"[節日] {msg[:100]}"}

        return None

    def _save_moment(self, moment: dict[str, str]) -> None:
        """Append a moment to SHARED_MOMENTS.md."""
        path = self._moments_path
        if path.exists():
            content = path.read_text(encoding="utf-8")
        else:
            content = "# 共同記憶\n\n"

        timestamp = datetime.now().strftime("%Y-%m-%d")
        entry = f"{moment['text']}  <!-- {timestamp} -->\n"
        content = content.rstrip("\n") + "\n" + entry

        path.write_text(content, encoding="utf-8")

    def _read_moments(self) -> list[str]:
        """Read moment text entries from the file."""
        return [
            l.strip()
            for l in self._read_all_lines()
            if l.strip().startswith("[")
        ]

    def _read_all_lines(self) -> list[str]:
        """Read all non-header lines from SHARED_MOMENTS.md."""
        if not self._moments_path.exists():
            return []
        content = self._moments_path.read_text(encoding="utf-8")
        return [
            l for l in content.split("\n")
            if l.strip() and not l.startswith("#") and not l.startswith("<!--")
        ]
