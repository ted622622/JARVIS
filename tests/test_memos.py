"""Tests for MemOS memory system and TokenSavingTracker.

Run: pytest tests/test_memos.py -v
"""

from __future__ import annotations

import asyncio
import os
import tempfile
from pathlib import Path

import pytest

from memory.memos_manager import MemOS
from memory.token_tracker import TokenSavingTracker


@pytest.fixture
async def memos(tmp_path):
    """Create a fresh MemOS instance with a temporary database."""
    db_path = str(tmp_path / "test_memos.db")
    m = MemOS(db_path)
    await m.init()
    yield m
    await m.close()


@pytest.fixture
async def tracker(memos):
    """Create a TokenSavingTracker tied to the MemOS database."""
    t = TokenSavingTracker(memos._db)
    await t.init()
    return t


# ── ShortTermMemory Tests ──────────────────────────────────────


class TestShortTermMemory:
    @pytest.mark.asyncio
    async def test_set_and_get(self, memos):
        memos.short_term.set("key1", "value1")
        assert memos.short_term.get("key1") == "value1"

    @pytest.mark.asyncio
    async def test_get_default(self, memos):
        assert memos.short_term.get("nonexistent", "default") == "default"

    @pytest.mark.asyncio
    async def test_delete(self, memos):
        memos.short_term.set("key1", "value1")
        assert memos.short_term.delete("key1") is True
        assert memos.short_term.get("key1") is None

    @pytest.mark.asyncio
    async def test_clear(self, memos):
        memos.short_term.set("a", 1)
        memos.short_term.set("b", 2)
        memos.short_term.clear()
        assert memos.short_term.keys() == []

    @pytest.mark.asyncio
    async def test_stores_complex_types(self, memos):
        memos.short_term.set("list", [1, 2, 3])
        memos.short_term.set("dict", {"nested": True})
        assert memos.short_term.get("list") == [1, 2, 3]
        assert memos.short_term.get("dict") == {"nested": True}


# ── WorkingMemory Tests ────────────────────────────────────────


class TestWorkingMemory:
    @pytest.mark.asyncio
    async def test_set_and_get(self, memos):
        await memos.working_memory.set("user_emotion", "happy", agent_id="ceo")
        val = await memos.working_memory.get("user_emotion")
        assert val == "happy"

    @pytest.mark.asyncio
    async def test_get_default(self, memos):
        val = await memos.working_memory.get("nonexistent", "default_val")
        assert val == "default_val"

    @pytest.mark.asyncio
    async def test_overwrite_with_different_agent(self, memos):
        await memos.working_memory.set("task", "coding", agent_id="worker_1")
        await memos.working_memory.set("task", "testing", agent_id="worker_2")
        val = await memos.working_memory.get("task")
        assert val == "testing"

    @pytest.mark.asyncio
    async def test_metadata_tracks_agent_id(self, memos):
        await memos.working_memory.set("mood", "tired", agent_id="ceo")
        meta = await memos.working_memory.get_metadata("mood")
        assert meta is not None
        assert meta["value"] == "tired"
        assert meta["agent_id"] == "ceo"

    @pytest.mark.asyncio
    async def test_delete(self, memos):
        await memos.working_memory.set("temp", "data", agent_id="test")
        assert await memos.working_memory.delete("temp") is True
        assert await memos.working_memory.get("temp") is None

    @pytest.mark.asyncio
    async def test_complex_values(self, memos):
        tasks = [{"id": 1, "name": "task1"}, {"id": 2, "name": "task2"}]
        await memos.working_memory.set("active_tasks", tasks, agent_id="ceo")
        val = await memos.working_memory.get("active_tasks")
        assert val == tasks

    @pytest.mark.asyncio
    async def test_persistence_across_restart(self, tmp_path):
        """Write → close → reopen → read back. Validates SQLite persistence."""
        db_path = str(tmp_path / "persist_test.db")

        # Write phase
        m1 = MemOS(db_path)
        await m1.init()
        await m1.working_memory.set("persist_key", {"data": 42}, agent_id="test")
        await m1.close()

        # Read phase (new instance)
        m2 = MemOS(db_path)
        await m2.init()
        val = await m2.working_memory.get("persist_key")
        assert val == {"data": 42}
        await m2.close()

    @pytest.mark.asyncio
    async def test_keys_and_all(self, memos):
        await memos.working_memory.set("a", 1, agent_id="t")
        await memos.working_memory.set("b", 2, agent_id="t")
        keys = await memos.working_memory.keys()
        assert set(keys) == {"a", "b"}
        all_data = await memos.working_memory.all()
        assert all_data == {"a": 1, "b": 2}


# ── LongTermMemory Tests ──────────────────────────────────────


class TestLongTermMemory:
    @pytest.mark.asyncio
    async def test_set_and_get(self, memos):
        await memos.long_term.set("user_preferences", "language", "zh-TW")
        val = await memos.long_term.get("user_preferences", "language")
        assert val == "zh-TW"

    @pytest.mark.asyncio
    async def test_get_default(self, memos):
        val = await memos.long_term.get("nonexistent", "key", "default")
        assert val == "default"

    @pytest.mark.asyncio
    async def test_get_category(self, memos):
        await memos.long_term.set("prefs", "lang", "zh-TW")
        await memos.long_term.set("prefs", "theme", "dark")
        category = await memos.long_term.get_category("prefs")
        assert category == {"lang": "zh-TW", "theme": "dark"}

    @pytest.mark.asyncio
    async def test_delete(self, memos):
        await memos.long_term.set("test", "key1", "val1")
        assert await memos.long_term.delete("test", "key1") is True
        assert await memos.long_term.get("test", "key1") is None

    @pytest.mark.asyncio
    async def test_delete_category(self, memos):
        await memos.long_term.set("temp", "a", 1)
        await memos.long_term.set("temp", "b", 2)
        count = await memos.long_term.delete_category("temp")
        assert count == 2

    @pytest.mark.asyncio
    async def test_search(self, memos):
        await memos.long_term.set("skills", "selfie_skill", {"version": 1})
        await memos.long_term.set("skills", "weather_skill", {"version": 2})
        results = await memos.long_term.search("skills", "%skill%")
        assert len(results) == 2

    @pytest.mark.asyncio
    async def test_persistence_across_restart(self, tmp_path):
        db_path = str(tmp_path / "lt_persist.db")

        m1 = MemOS(db_path)
        await m1.init()
        await m1.long_term.set("user_preferences", "name", "Ted")
        await m1.close()

        m2 = MemOS(db_path)
        await m2.init()
        val = await m2.long_term.get("user_preferences", "name")
        assert val == "Ted"
        await m2.close()


# ── Conversation Log Tests ──────────────────────────────────────


class TestConversationLog:
    @pytest.mark.asyncio
    async def test_log_and_retrieve(self, memos):
        await memos.log_message("session-1", "user", "你好")
        await memos.log_message("session-1", "assistant", "你好！有什麼可以幫你的嗎？")

        conv = await memos.get_conversation("session-1")
        assert len(conv) == 2
        assert conv[0]["role"] == "user"
        assert conv[1]["role"] == "assistant"

    @pytest.mark.asyncio
    async def test_sessions_isolated(self, memos):
        await memos.log_message("s1", "user", "msg1")
        await memos.log_message("s2", "user", "msg2")

        conv1 = await memos.get_conversation("s1")
        conv2 = await memos.get_conversation("s2")
        assert len(conv1) == 1
        assert len(conv2) == 1


# ── TokenSavingTracker Tests ──────────────────────────────────


class TestTokenSavingTracker:
    @pytest.mark.asyncio
    async def test_record_and_report(self, tracker):
        await tracker.record("call-1", raw_tokens=5000, memos_tokens=1500)
        await tracker.record("call-2", raw_tokens=3000, memos_tokens=900)

        report = await tracker.daily_report()
        assert report["total_calls"] == 2
        assert report["total_raw_tokens"] == 8000
        assert report["total_memos_tokens"] == 2400
        assert report["alert"] is False  # 70% saving > 50%

    @pytest.mark.asyncio
    async def test_alert_when_low_saving(self, tracker):
        # 10% saving rate — should trigger alert
        await tracker.record("call-1", raw_tokens=1000, memos_tokens=900)
        report = await tracker.daily_report()
        assert report["alert"] is True

    @pytest.mark.asyncio
    async def test_empty_report(self, tracker):
        report = await tracker.daily_report()
        assert report["total_calls"] == 0
        assert report["avg_saving_rate"] == "N/A"
        assert report["alert"] is False

    @pytest.mark.asyncio
    async def test_saving_rate_return(self, tracker):
        rate = await tracker.record("call-1", raw_tokens=10000, memos_tokens=3000)
        assert abs(rate - 0.7) < 0.001

    @pytest.mark.asyncio
    async def test_get_recent(self, tracker):
        for i in range(5):
            await tracker.record(f"call-{i}", raw_tokens=1000, memos_tokens=500)
        recent = await tracker.get_recent(limit=3)
        assert len(recent) == 3

    @pytest.mark.asyncio
    async def test_invalid_raw_tokens(self, tracker):
        rate = await tracker.record("bad", raw_tokens=0, memos_tokens=100)
        assert rate == 0.0


# ── Backup Test ────────────────────────────────────────────────


class TestBackup:
    @pytest.mark.asyncio
    async def test_unencrypted_backup(self, memos, tmp_path):
        await memos.long_term.set("test", "key", "value")
        dest = str(tmp_path / "backup.db")
        result = await memos.backup(dest, encrypt=False)
        assert Path(result).exists()

        # Verify backup has the data
        backup_memos = MemOS(result)
        await backup_memos.init()
        val = await backup_memos.long_term.get("test", "key")
        assert val == "value"
        await backup_memos.close()
