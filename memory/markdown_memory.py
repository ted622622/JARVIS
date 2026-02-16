"""Markdown Memory — human-readable memory layer.

Manages three types of markdown files:
- MEMORY.md: long-term facts (preferences, decisions, settings)
- daily/YYYY-MM-DD.md: daily journal (append mode)
- sessions/YYYY-MM-DD-slug.md: conversation transcripts
"""

from __future__ import annotations

import re
from datetime import datetime
from pathlib import Path

from loguru import logger


class MarkdownMemory:
    """Human-readable memory layer backed by markdown files.

    Usage:
        mm = MarkdownMemory("./memory")
        mm.remember("用戶喜歡吃拉麵")
        mm.log_daily("今天幫用戶訂了 UberEats")
        mm.save_session("ubereats-order", transcript)
    """

    def __init__(self, memory_dir: str = "./memory"):
        self.root = Path(memory_dir)
        self.root.mkdir(parents=True, exist_ok=True)
        (self.root / "daily").mkdir(exist_ok=True)
        (self.root / "sessions").mkdir(exist_ok=True)

        self.memory_file = self.root / "MEMORY.md"
        if not self.memory_file.exists():
            self.memory_file.write_text(
                "# 長期記憶\n\n> 由 CEO Agent 自動維護。\n",
                encoding="utf-8",
            )

    # ── Long-term memory (MEMORY.md) ─────────────────────────

    def remember(self, fact: str, category: str = "用戶偏好") -> None:
        """Add a fact to long-term memory under a category."""
        content = self.memory_file.read_text(encoding="utf-8")

        # Find or create the category section
        heading = f"## {category}"
        if heading in content:
            # Append under existing heading (before next heading or EOF)
            pattern = rf"(## {re.escape(category)}\n(?:.*\n)*?)((?=\n## )|\Z)"
            match = re.search(pattern, content)
            if match:
                section_end = match.end(1)
                new_content = content[:section_end] + f"- {fact}\n" + content[section_end:]
                self.memory_file.write_text(new_content, encoding="utf-8")
                logger.info(f"Memory: added to [{category}]: {fact[:60]}")
                return

        # Category not found — append new section
        content = content.rstrip() + f"\n\n## {category}\n\n- {fact}\n"
        self.memory_file.write_text(content, encoding="utf-8")
        logger.info(f"Memory: new category [{category}]: {fact[:60]}")

    def read_memory(self) -> str:
        """Read the full MEMORY.md content."""
        if self.memory_file.exists():
            return self.memory_file.read_text(encoding="utf-8")
        return ""

    # ── Daily journal (daily/YYYY-MM-DD.md) ──────────────────

    def _daily_path(self, date: datetime | None = None) -> Path:
        d = date or datetime.now()
        return self.root / "daily" / f"{d.strftime('%Y-%m-%d')}.md"

    def log_daily(self, entry: str, date: datetime | None = None) -> None:
        """Append an entry to today's daily journal."""
        path = self._daily_path(date)
        now = datetime.now()

        if not path.exists():
            header = f"# {now.strftime('%Y-%m-%d')} 日誌\n\n"
            path.write_text(header, encoding="utf-8")

        timestamp = now.strftime("%H:%M")
        with open(path, "a", encoding="utf-8") as f:
            f.write(f"- [{timestamp}] {entry}\n")

        logger.debug(f"Daily log: {entry[:60]}")

    def read_daily(self, date: datetime | None = None) -> str:
        """Read a daily journal. Returns empty string if not found."""
        path = self._daily_path(date)
        if path.exists():
            return path.read_text(encoding="utf-8")
        return ""

    # ── Session transcripts ──────────────────────────────────

    def save_session(
        self,
        slug: str,
        transcript: str,
        date: datetime | None = None,
    ) -> Path:
        """Save a conversation transcript.

        Args:
            slug: short topic identifier (e.g. "ubereats-order")
            transcript: full markdown transcript
            date: override date (defaults to now)

        Returns:
            Path to the saved file
        """
        d = date or datetime.now()
        # Sanitize slug
        safe_slug = re.sub(r"[^\w\-]", "-", slug)[:50].strip("-")
        filename = f"{d.strftime('%Y-%m-%d')}-{safe_slug}.md"
        path = self.root / "sessions" / filename

        path.write_text(transcript, encoding="utf-8")
        logger.info(f"Session saved: {filename} ({len(transcript)} chars)")
        return path

    def list_sessions(self, limit: int = 20) -> list[Path]:
        """List recent session files, newest first."""
        sessions_dir = self.root / "sessions"
        files = sorted(sessions_dir.glob("*.md"), reverse=True)
        return files[:limit]

    def read_session(self, path: Path) -> str:
        """Read a session transcript."""
        if path.exists():
            return path.read_text(encoding="utf-8")
        return ""

    # ── Bulk read (for search indexing) ──────────────────────

    def all_markdown_files(self) -> list[Path]:
        """List all .md files under memory/ for indexing."""
        return list(self.root.rglob("*.md"))
