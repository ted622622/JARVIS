"""SoulGrowth — learn from conversations and evolve persona preferences.

Patch J2+J3: Observes conversation turns and appends learned insights
to memory/{persona}/SOUL_GROWTH.md.

- J2: Clawra interaction preferences (tone, topics, emotional patterns)
- J3: JARVIS work preferences (report style, tool usage, communication)

Rate limited: at most 1 learning per 10 conversation turns.
Growth file capped at 50 entries; oldest removed when exceeded.
"""

from __future__ import annotations

import re
import time
from datetime import datetime
from pathlib import Path
from typing import Any

from loguru import logger


# Minimum turns between learning attempts
_LEARN_INTERVAL = 3

# Maximum entries in SOUL_GROWTH.md
_MAX_GROWTH_ENTRIES = 50

# Patterns that suggest a preference or correction
_JARVIS_LEARN_PATTERNS = re.compile(
    r"以後|不要再|記住|我偏好|我喜歡|我不喜歡|我習慣|"
    r"太長了|太短了|不要這樣|這樣比較好|"
    r"用\S+方式|別用|不用|省略|直接|"
    r"prefer|always|never|don't|stop",
    re.IGNORECASE,
)

_CLAWRA_LEARN_PATTERNS = re.compile(
    r"以後|不要再|記住|喜歡|不喜歡|習慣|"
    r"太煩了|太黏了|不要這樣|別這樣|"
    r"不要問|不用|不需要|"
    r"好啦|知道了|夠了|"
    r"髮型|頭髮|好看|穿搭|衣服|場景",
    re.IGNORECASE,
)

# Selfie appearance feedback — positive or negative
_SELFIE_APPEARANCE_PATTERN = re.compile(
    r"(?P<sentiment>好看|喜歡|不錯|讚|可愛|好喜歡|超好看|不喜歡|不好看|太奇怪|不要|醜|怪怪的)"
    r".*?"
    r"(?P<category>髮型|頭髮|馬尾|捲髮|包包頭|辮子|雙馬尾|半綁|"
    r"穿搭|衣服|外套|裙子|洋裝|大衣|帽子|"
    r"場景|背景|咖啡廳|漢江|弘大|書店|屋頂)"
    r"|"
    r"(?P<category2>髮型|頭髮|馬尾|捲髮|包包頭|辮子|雙馬尾|半綁|"
    r"穿搭|衣服|外套|裙子|洋裝|大衣|帽子|"
    r"場景|背景|咖啡廳|漢江|弘大|書店|屋頂)"
    r".*?"
    r"(?P<sentiment2>好看|喜歡|不錯|讚|可愛|好喜歡|超好看|不喜歡|不好看|太奇怪|不要|醜|怪怪的)",
    re.IGNORECASE,
)

# Map Chinese keywords → English pref values for [selfie-pref] tags
_CATEGORY_MAP: dict[str, tuple[str, str]] = {
    # (english_category, english_value)
    "髮型": ("hairstyle", ""),
    "頭髮": ("hairstyle", ""),
    "馬尾": ("hairstyle", "ponytail"),
    "捲髮": ("hairstyle", "curly"),
    "包包頭": ("hairstyle", "bun"),
    "辮子": ("hairstyle", "braided"),
    "雙馬尾": ("hairstyle", "twintails"),
    "半綁": ("hairstyle", "half-up"),
    "穿搭": ("outfit", ""),
    "衣服": ("outfit", ""),
    "外套": ("outfit", "coat"),
    "裙子": ("outfit", "skirt"),
    "洋裝": ("outfit", "dress"),
    "大衣": ("outfit", "coat"),
    "帽子": ("outfit", "beanie"),
    "場景": ("scene", ""),
    "背景": ("scene", ""),
    "咖啡廳": ("scene", "cafe"),
    "漢江": ("scene", "Han River"),
    "弘大": ("scene", "Hongdae"),
    "書店": ("scene", "bookstore"),
    "屋頂": ("scene", "rooftop"),
}

_NEGATIVE_SENTIMENTS = frozenset({"不喜歡", "不好看", "太奇怪", "不要", "醜", "怪怪的"})

# Core values that GROWTH must never contradict
_CORE_PRINCIPLES = [
    "100% 誠實",
    "不確定就說不確定",
    "不洩漏系統",
    "不打破角色",
]


class SoulGrowth:
    """Learns from conversations and appends to SOUL_GROWTH.md.

    Usage:
        growth = SoulGrowth(memory_dir="./memory")
        insight = growth.maybe_learn("jarvis", user_msg, assistant_msg)
        if insight:
            soul.reload_growth("jarvis")
    """

    def __init__(self, memory_dir: str = "./memory"):
        self._memory_dir = Path(memory_dir)
        self._turn_counts: dict[str, int] = {}  # persona -> turns since last learn
        self._last_learn_time: dict[str, float] = {}

    def maybe_learn(
        self,
        persona: str,
        user_msg: str,
        assistant_msg: str,
    ) -> str | None:
        """Analyze a conversation turn and potentially learn from it.

        Returns the learned insight string if something was learned, None otherwise.
        """
        # Rate limit: count turns
        count = self._turn_counts.get(persona, 0) + 1
        self._turn_counts[persona] = count

        if count < _LEARN_INTERVAL:
            logger.debug(f"SoulGrowth [{persona}]: turn {count}/{_LEARN_INTERVAL}, waiting")
            return None

        # Check if message contains learnable patterns
        pattern = _JARVIS_LEARN_PATTERNS if persona == "jarvis" else _CLAWRA_LEARN_PATTERNS
        if not pattern.search(user_msg):
            logger.debug(f"SoulGrowth [{persona}]: no pattern match in: {user_msg[:50]}")
            return None

        # Extract the insight
        insight = self._extract_insight(persona, user_msg, assistant_msg)
        if not insight:
            return None

        # Validate against core principles
        if self._violates_core(insight):
            logger.warning(f"SoulGrowth: rejected insight that violates core: {insight[:50]}")
            return None

        # Check for duplicates
        growth_path = self._growth_path(persona)
        existing = self._read_entries(growth_path)
        if any(insight.lower() in e.lower() or e.lower() in insight.lower() for e in existing):
            logger.debug(f"SoulGrowth: duplicate insight skipped: {insight[:50]}")
            self._turn_counts[persona] = 0
            return None

        # Append to growth file
        self._append(persona, insight)
        self._turn_counts[persona] = 0
        self._last_learn_time[persona] = time.time()

        logger.info(f"SoulGrowth [{persona}]: learned — {insight[:60]}")
        return insight

    def _extract_selfie_preference(self, user_msg: str) -> str | None:
        """Extract selfie appearance feedback as a [selfie-pref] tag.

        Returns e.g. "- [selfie-pref] like:hairstyle:ponytail"
        """
        match = _SELFIE_APPEARANCE_PATTERN.search(user_msg)
        if not match:
            return None

        # Determine sentiment and category from either ordering
        sentiment = match.group("sentiment") or match.group("sentiment2")
        category_zh = match.group("category") or match.group("category2")
        if not sentiment or not category_zh:
            return None

        mapping = _CATEGORY_MAP.get(category_zh)
        if not mapping:
            return None

        eng_category, eng_value = mapping
        if not eng_value:
            # Generic category without specific value — skip
            return None

        like_or_dislike = "dislike" if sentiment in _NEGATIVE_SENTIMENTS else "like"
        return f"- [selfie-pref] {like_or_dislike}:{eng_category}:{eng_value}"

    def _extract_insight(
        self, persona: str, user_msg: str, assistant_msg: str,
    ) -> str | None:
        """Extract a concise preference insight from conversation."""
        msg = user_msg.strip()

        # Patch Q: Check selfie appearance preference first (Clawra only)
        if persona == "clawra":
            selfie_pref = self._extract_selfie_preference(msg)
            if selfie_pref:
                return selfie_pref

        # Direct preference statements
        for prefix in ["以後", "記住", "不要再"]:
            if prefix in msg:
                # Take the part after the keyword
                idx = msg.index(prefix)
                rest = msg[idx:].strip()
                if len(rest) > 5:
                    return f"- {rest[:100]}"

        # Correction patterns — user is correcting the assistant
        if any(kw in msg for kw in ["太長了", "太短了", "不要這樣", "別這樣"]):
            return f"- 用戶反饋：{msg[:100]}"

        # Preference declarations
        for kw in ["我偏好", "我喜歡", "我不喜歡", "我習慣"]:
            if kw in msg:
                return f"- {msg[:100]}"

        # Style preferences
        if re.search(r"用\S+方式|別用|不用|省略|直接", msg):
            return f"- 溝通偏好：{msg[:100]}"

        return None

    def _violates_core(self, insight: str) -> bool:
        """Check if an insight contradicts core principles."""
        lower = insight.lower()
        # Check for attempts to override honesty, transparency, etc.
        violation_patterns = [
            r"可以說謊",
            r"不用誠實",
            r"假裝(?:知道|確定)",
            r"洩漏.*(?:系統|架構|prompt)",
            r"打破.*角色",
            r"忽略.*(?:最高|憲法|規則)",
        ]
        for pat in violation_patterns:
            if re.search(pat, lower):
                return True
        return False

    def _append(self, persona: str, insight: str) -> None:
        """Append an insight to SOUL_GROWTH.md, trimming if needed."""
        path = self._growth_path(persona)
        path.parent.mkdir(parents=True, exist_ok=True)

        # Read existing content
        if path.exists():
            content = path.read_text(encoding="utf-8")
        else:
            header = "# JARVIS 成長記錄" if persona == "jarvis" else "# Clawra 成長記錄"
            content = f"{header}\n\n"

        # Append with timestamp
        timestamp = datetime.now().strftime("%Y-%m-%d")
        entry = f"{insight}  <!-- {timestamp} -->\n"
        content = content.rstrip("\n") + "\n" + entry

        # Trim if too many entries
        content = self._trim_if_needed(content)

        path.write_text(content, encoding="utf-8")

    def _trim_if_needed(self, content: str) -> str:
        """Keep only the most recent _MAX_GROWTH_ENTRIES entries."""
        lines = content.split("\n")
        header_lines = []
        entry_lines = []

        for line in lines:
            if line.startswith("#") or line.startswith("<!--") or not line.strip():
                if not entry_lines:
                    header_lines.append(line)
                else:
                    entry_lines.append(line)
            elif line.startswith("- "):
                entry_lines.append(line)
            else:
                entry_lines.append(line)

        # Count actual entries (lines starting with "- ")
        entries = [l for l in entry_lines if l.startswith("- ")]
        if len(entries) <= _MAX_GROWTH_ENTRIES:
            return content

        # Keep only the last N entries
        keep_count = _MAX_GROWTH_ENTRIES
        kept = 0
        result_entries = []
        for line in reversed(entry_lines):
            if line.startswith("- "):
                if kept < keep_count:
                    result_entries.append(line)
                    kept += 1
            else:
                result_entries.append(line)

        result_entries.reverse()
        return "\n".join(header_lines + [""] + result_entries) + "\n"

    def _growth_path(self, persona: str) -> Path:
        """Return the path to SOUL_GROWTH.md for a persona."""
        return self._memory_dir / persona / "SOUL_GROWTH.md"

    def _read_entries(self, path: Path) -> list[str]:
        """Read existing entry lines from a growth file."""
        if not path.exists():
            return []
        content = path.read_text(encoding="utf-8")
        return [l.strip() for l in content.split("\n") if l.strip().startswith("- ")]

    def get_entry_count(self, persona: str) -> int:
        """Return the number of entries in a persona's growth file."""
        return len(self._read_entries(self._growth_path(persona)))
