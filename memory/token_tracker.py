"""TokenSavingTracker — measures actual token savings from MemOS context optimization.

Instead of hardcoding a saving rate, this tracks real usage data per LLM call
and reports observed savings over time.
"""

from __future__ import annotations

import time
from statistics import mean
from typing import Any

import aiosqlite
from loguru import logger

_SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS token_savings (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    call_id      TEXT    NOT NULL,
    raw_tokens   INTEGER NOT NULL,
    memos_tokens INTEGER NOT NULL,
    saving_rate  REAL    NOT NULL,
    timestamp    REAL    NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_ts_timestamp ON token_savings(timestamp);
"""


class TokenSavingTracker:
    """Tracks token consumption with/without MemOS optimization.

    Usage:
        tracker = TokenSavingTracker(db)
        await tracker.init()

        # Before each LLM call, record both counts
        await tracker.record("call-123", raw_tokens=5000, memos_tokens=1500)

        # Get daily report
        report = await tracker.daily_report()
        # {"avg_saving_rate": "70.0%", "total_calls": 42, "alert": False}
    """

    def __init__(self, db: aiosqlite.Connection):
        self._db = db

    async def init(self) -> None:
        """Create the tracking table if not exists."""
        await self._db.executescript(_SCHEMA_SQL)
        await self._db.commit()
        logger.debug("TokenSavingTracker initialized")

    async def record(
        self, call_id: str, raw_tokens: int, memos_tokens: int
    ) -> float:
        """Record a single LLM call's token usage.

        Args:
            call_id: unique identifier for this call
            raw_tokens: token count WITHOUT MemOS optimization
            memos_tokens: token count WITH MemOS optimization

        Returns:
            The saving rate for this call (0.0 to 1.0)
        """
        if raw_tokens <= 0:
            logger.warning(f"Invalid raw_tokens={raw_tokens} for call {call_id}")
            return 0.0

        saving_rate = round(1.0 - (memos_tokens / raw_tokens), 4)
        await self._db.execute(
            "INSERT INTO token_savings (call_id, raw_tokens, memos_tokens, saving_rate, timestamp) "
            "VALUES (?, ?, ?, ?, ?)",
            (call_id, raw_tokens, memos_tokens, saving_rate, time.time()),
        )
        await self._db.commit()
        return saving_rate

    async def daily_report(self, hours: int = 24) -> dict[str, Any]:
        """Return average saving rate over the last N hours.

        Returns:
            {
                "avg_saving_rate": "72.3%",
                "total_calls": 150,
                "total_raw_tokens": 500000,
                "total_memos_tokens": 139000,
                "alert": False  # True if avg < 50%
            }
        """
        cutoff = time.time() - (hours * 3600)
        cursor = await self._db.execute(
            "SELECT saving_rate, raw_tokens, memos_tokens FROM token_savings "
            "WHERE timestamp >= ?",
            (cutoff,),
        )
        rows = await cursor.fetchall()

        if not rows:
            return {
                "avg_saving_rate": "N/A",
                "total_calls": 0,
                "total_raw_tokens": 0,
                "total_memos_tokens": 0,
                "alert": False,
            }

        rates = [r["saving_rate"] for r in rows]
        total_raw = sum(r["raw_tokens"] for r in rows)
        total_memos = sum(r["memos_tokens"] for r in rows)
        avg = mean(rates)

        return {
            "avg_saving_rate": f"{avg:.1%}",
            "total_calls": len(rows),
            "total_raw_tokens": total_raw,
            "total_memos_tokens": total_memos,
            "alert": avg < 0.50,
        }

    async def get_recent(self, limit: int = 20) -> list[dict[str, Any]]:
        """Get the most recent tracking records."""
        cursor = await self._db.execute(
            "SELECT call_id, raw_tokens, memos_tokens, saving_rate, timestamp "
            "FROM token_savings ORDER BY id DESC LIMIT ?",
            (limit,),
        )
        rows = await cursor.fetchall()
        return [
            {
                "call_id": r["call_id"],
                "raw_tokens": r["raw_tokens"],
                "memos_tokens": r["memos_tokens"],
                "saving_rate": f"{r['saving_rate']:.1%}",
                "timestamp": r["timestamp"],
            }
            for r in rows
        ]
