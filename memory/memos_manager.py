"""MemOS — 3-tier memory system for J.A.R.V.I.S.

Architecture:
    short_term    — current conversation context (RAM only, per-session)
    working_memory — cross-agent shared state (RAM cache + SQLite)
    long_term      — persistent storage (SQLite)

Read path: RAM cache → SQLite (read-through)
Write path: write to SQLite + update RAM cache (write-through)
"""

from __future__ import annotations

import asyncio
import json
import time
from pathlib import Path
from typing import Any

import aiosqlite
from loguru import logger

_SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS working_memory (
    key        TEXT    NOT NULL,
    value      TEXT    NOT NULL,
    agent_id   TEXT    NOT NULL,
    updated_at REAL    NOT NULL,
    PRIMARY KEY (key)
);

CREATE TABLE IF NOT EXISTS long_term (
    category   TEXT    NOT NULL,
    key        TEXT    NOT NULL,
    value      TEXT    NOT NULL,
    agent_id   TEXT    NOT NULL DEFAULT '',
    created_at REAL    NOT NULL,
    updated_at REAL    NOT NULL,
    PRIMARY KEY (category, key)
);

CREATE TABLE IF NOT EXISTS conversation_log (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id TEXT    NOT NULL,
    role       TEXT    NOT NULL,
    content    TEXT    NOT NULL,
    timestamp  REAL    NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_conv_session ON conversation_log(session_id);
CREATE INDEX IF NOT EXISTS idx_lt_category ON long_term(category);
"""


class MemOS:
    """Three-tier memory manager.

    Usage:
        memos = MemOS("./data/memos.db")
        await memos.init()

        # Short-term (RAM only)
        memos.short_term.set("last_query", "天氣如何")
        val = memos.short_term.get("last_query")

        # Working memory (cross-agent, cached)
        await memos.working_memory.set("user_emotion", "tired", agent_id="ceo")
        emotion = await memos.working_memory.get("user_emotion")

        # Long-term (persistent)
        await memos.long_term.set("user_preferences", "language", "zh-TW")
        lang = await memos.long_term.get("user_preferences", "language")
    """

    def __init__(self, db_path: str = "./data/memos.db"):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._db: aiosqlite.Connection | None = None

        self.short_term = ShortTermMemory()
        self.working_memory: WorkingMemory | None = None
        self.long_term: LongTermMemory | None = None

    async def init(self) -> None:
        """Initialize database and memory tiers."""
        self._db = await aiosqlite.connect(str(self.db_path))
        self._db.row_factory = aiosqlite.Row
        await self._db.executescript(_SCHEMA_SQL)
        await self._db.commit()

        self.working_memory = WorkingMemory(self._db)
        self.long_term = LongTermMemory(self._db)

        # Pre-load working_memory cache from SQLite
        await self.working_memory._load_cache()
        logger.info(f"MemOS initialized (db: {self.db_path})")

    async def close(self) -> None:
        if self._db:
            await self._db.close()
            self._db = None

    # ── Conversation logging ────────────────────────────────────

    async def log_message(
        self, session_id: str, role: str, content: str
    ) -> None:
        """Append a message to conversation log."""
        await self._db.execute(
            "INSERT INTO conversation_log (session_id, role, content, timestamp) "
            "VALUES (?, ?, ?, ?)",
            (session_id, role, content, time.time()),
        )
        await self._db.commit()

    async def get_conversation(
        self, session_id: str, limit: int = 50
    ) -> list[dict[str, Any]]:
        """Retrieve recent messages for a session."""
        cursor = await self._db.execute(
            "SELECT role, content, timestamp FROM conversation_log "
            "WHERE session_id = ? ORDER BY id DESC LIMIT ?",
            (session_id, limit),
        )
        rows = await cursor.fetchall()
        return [
            {"role": r["role"], "content": r["content"], "timestamp": r["timestamp"]}
            for r in reversed(rows)
        ]

    # ── Backup ──────────────────────────────────────────────────

    async def backup(self, dest_path: str, encrypt: bool = False) -> str:
        """Create a backup of the database.

        Args:
            dest_path: destination file path
            encrypt: if True, AES-256 encrypt (requires BACKUP_ENCRYPTION_KEY env)

        Returns:
            Path to the backup file.
        """
        import shutil

        # Flush WAL to main db
        await self._db.execute("PRAGMA wal_checkpoint(FULL)")

        src = str(self.db_path)
        shutil.copy2(src, dest_path)

        if encrypt:
            dest_path = await self._encrypt_file(dest_path)

        logger.info(f"MemOS backup created: {dest_path}")
        return dest_path

    async def _encrypt_file(self, file_path: str) -> str:
        """AES-256-CBC encrypt a file. Returns path to encrypted file."""
        import os

        from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
        from cryptography.hazmat.primitives.padding import PKCS7

        key_hex = os.environ.get("BACKUP_ENCRYPTION_KEY", "")
        if not key_hex or len(key_hex) < 64:
            logger.warning("BACKUP_ENCRYPTION_KEY not set or too short, skipping encryption")
            return file_path

        key = bytes.fromhex(key_hex)
        iv = os.urandom(16)

        with open(file_path, "rb") as f:
            data = f.read()

        padder = PKCS7(128).padder()
        padded = padder.update(data) + padder.finalize()

        cipher = Cipher(algorithms.AES(key), modes.CBC(iv))
        encryptor = cipher.encryptor()
        encrypted = encryptor.update(padded) + encryptor.finalize()

        enc_path = file_path + ".enc"
        with open(enc_path, "wb") as f:
            f.write(iv + encrypted)

        # Remove unencrypted copy
        os.remove(file_path)
        return enc_path


# ── Short-Term Memory (RAM only) ───────────────────────────────


class ShortTermMemory:
    """In-memory store for current session context. Not persisted."""

    def __init__(self):
        self._store: dict[str, Any] = {}

    def set(self, key: str, value: Any) -> None:
        self._store[key] = value

    def get(self, key: str, default: Any = None) -> Any:
        return self._store.get(key, default)

    def delete(self, key: str) -> bool:
        return self._store.pop(key, None) is not None

    def clear(self) -> None:
        self._store.clear()

    def keys(self) -> list[str]:
        return list(self._store.keys())

    def all(self) -> dict[str, Any]:
        return dict(self._store)


# ── Working Memory (RAM cache + SQLite) ─────────────────────────


class WorkingMemory:
    """Cross-agent shared state with read-through cache.

    Write operations require an agent_id to track ownership.
    Reads are served from RAM cache first, falling back to SQLite.
    """

    def __init__(self, db: aiosqlite.Connection):
        self._db = db
        self._cache: dict[str, Any] = {}
        self._lock = asyncio.Lock()

    async def _load_cache(self) -> None:
        """Pre-load all working memory into RAM cache."""
        cursor = await self._db.execute("SELECT key, value FROM working_memory")
        rows = await cursor.fetchall()
        for row in rows:
            try:
                self._cache[row["key"]] = json.loads(row["value"])
            except (json.JSONDecodeError, TypeError):
                self._cache[row["key"]] = row["value"]
        logger.debug(f"Working memory cache loaded: {len(self._cache)} entries")

    async def set(self, key: str, value: Any, agent_id: str) -> None:
        """Write a value (write-through: SQLite + cache)."""
        async with self._lock:
            serialized = json.dumps(value, ensure_ascii=False)
            await self._db.execute(
                "INSERT INTO working_memory (key, value, agent_id, updated_at) "
                "VALUES (?, ?, ?, ?) "
                "ON CONFLICT(key) DO UPDATE SET value=?, agent_id=?, updated_at=?",
                (key, serialized, agent_id, time.time(),
                 serialized, agent_id, time.time()),
            )
            await self._db.commit()
            self._cache[key] = value

    async def get(self, key: str, default: Any = None) -> Any:
        """Read a value (read-through: cache → SQLite)."""
        # Try cache first
        if key in self._cache:
            return self._cache[key]

        # Fall back to SQLite
        cursor = await self._db.execute(
            "SELECT value FROM working_memory WHERE key = ?", (key,)
        )
        row = await cursor.fetchone()
        if row is None:
            return default

        try:
            value = json.loads(row["value"])
        except (json.JSONDecodeError, TypeError):
            value = row["value"]

        self._cache[key] = value
        return value

    async def delete(self, key: str) -> bool:
        """Remove a key from working memory."""
        async with self._lock:
            cursor = await self._db.execute(
                "DELETE FROM working_memory WHERE key = ?", (key,)
            )
            await self._db.commit()
            self._cache.pop(key, None)
            return cursor.rowcount > 0

    async def get_metadata(self, key: str) -> dict[str, Any] | None:
        """Get value + metadata (agent_id, updated_at) for a key."""
        cursor = await self._db.execute(
            "SELECT value, agent_id, updated_at FROM working_memory WHERE key = ?",
            (key,),
        )
        row = await cursor.fetchone()
        if row is None:
            return None
        try:
            value = json.loads(row["value"])
        except (json.JSONDecodeError, TypeError):
            value = row["value"]
        return {
            "value": value,
            "agent_id": row["agent_id"],
            "updated_at": row["updated_at"],
        }

    async def keys(self) -> list[str]:
        return list(self._cache.keys())

    async def all(self) -> dict[str, Any]:
        return dict(self._cache)


# ── Long-Term Memory (SQLite) ──────────────────────────────────


class LongTermMemory:
    """Persistent storage organized by category.

    Categories: user_preferences, conversation_log, skill_outcomes, etc.
    """

    def __init__(self, db: aiosqlite.Connection):
        self._db = db

    async def set(
        self,
        category: str,
        key: str,
        value: Any,
        agent_id: str = "",
    ) -> None:
        """Store a value under category/key."""
        serialized = json.dumps(value, ensure_ascii=False)
        now = time.time()
        await self._db.execute(
            "INSERT INTO long_term (category, key, value, agent_id, created_at, updated_at) "
            "VALUES (?, ?, ?, ?, ?, ?) "
            "ON CONFLICT(category, key) DO UPDATE SET value=?, agent_id=?, updated_at=?",
            (category, key, serialized, agent_id, now, now,
             serialized, agent_id, now),
        )
        await self._db.commit()

    async def get(
        self, category: str, key: str, default: Any = None
    ) -> Any:
        """Retrieve a value by category/key."""
        cursor = await self._db.execute(
            "SELECT value FROM long_term WHERE category = ? AND key = ?",
            (category, key),
        )
        row = await cursor.fetchone()
        if row is None:
            return default
        try:
            return json.loads(row["value"])
        except (json.JSONDecodeError, TypeError):
            return row["value"]

    async def get_category(self, category: str) -> dict[str, Any]:
        """Retrieve all key-value pairs in a category."""
        cursor = await self._db.execute(
            "SELECT key, value FROM long_term WHERE category = ?",
            (category,),
        )
        rows = await cursor.fetchall()
        result = {}
        for row in rows:
            try:
                result[row["key"]] = json.loads(row["value"])
            except (json.JSONDecodeError, TypeError):
                result[row["key"]] = row["value"]
        return result

    async def delete(self, category: str, key: str) -> bool:
        """Delete a specific entry."""
        cursor = await self._db.execute(
            "DELETE FROM long_term WHERE category = ? AND key = ?",
            (category, key),
        )
        await self._db.commit()
        return cursor.rowcount > 0

    async def delete_category(self, category: str) -> int:
        """Delete all entries in a category. Returns count deleted."""
        cursor = await self._db.execute(
            "DELETE FROM long_term WHERE category = ?", (category,)
        )
        await self._db.commit()
        return cursor.rowcount

    async def search(
        self, category: str | None = None, pattern: str = "%"
    ) -> list[dict[str, Any]]:
        """Search long-term memory by key pattern (SQL LIKE)."""
        if category:
            cursor = await self._db.execute(
                "SELECT category, key, value, updated_at FROM long_term "
                "WHERE category = ? AND key LIKE ? ORDER BY updated_at DESC",
                (category, pattern),
            )
        else:
            cursor = await self._db.execute(
                "SELECT category, key, value, updated_at FROM long_term "
                "WHERE key LIKE ? ORDER BY updated_at DESC",
                (pattern,),
            )
        rows = await cursor.fetchall()
        results = []
        for row in rows:
            try:
                val = json.loads(row["value"])
            except (json.JSONDecodeError, TypeError):
                val = row["value"]
            results.append({
                "category": row["category"],
                "key": row["key"],
                "value": val,
                "updated_at": row["updated_at"],
            })
        return results
