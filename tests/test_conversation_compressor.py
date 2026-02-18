"""Tests for core.conversation_compressor — Patch T+: Pre-flush callback."""

from __future__ import annotations

import asyncio

import pytest

from core.conversation_compressor import ConversationCompressor


class TestCompressorBasic:
    """Basic compressor behavior (existing functionality)."""

    def test_add_turn(self):
        c = ConversationCompressor(recent_turns_keep=2)
        c.add_turn("user", "Hello")
        assert len(c.full_history) == 1

    def test_compression_triggers(self):
        c = ConversationCompressor(recent_turns_keep=2)
        # 2 pairs = 4 turns kept; 5th should trigger compression
        for i in range(5):
            c.add_turn("user" if i % 2 == 0 else "assistant", f"msg {i}")
        assert len(c.compressed_summary) >= 1
        assert len(c.full_history) <= 4

    def test_reset_clears_all(self):
        c = ConversationCompressor(recent_turns_keep=1)
        for i in range(6):
            c.add_turn("user" if i % 2 == 0 else "assistant", f"msg {i}")
        c.reset()
        assert len(c.full_history) == 0
        assert len(c.compressed_summary) == 0
        assert len(c._pending_flush) == 0


# ── Patch T+: Pre-flush callback ────────────────────────────────


class TestPreFlush:
    """Pre-compaction memory flush."""

    def test_no_flush_without_callback(self):
        """Without callback set, compression should work normally."""
        c = ConversationCompressor(recent_turns_keep=2)
        for i in range(6):
            c.add_turn("user" if i % 2 == 0 else "assistant", f"msg {i}")
        # Should compress without error
        assert not c.has_pending_flush

    def test_pending_flush_staged_after_compression(self):
        """When callback is set, compressed turns are staged."""
        c = ConversationCompressor(recent_turns_keep=2)

        async def dummy_callback(turns):
            pass

        c.set_pre_flush_callback(dummy_callback)

        # Add enough turns to trigger compression (>4 turns)
        for i in range(6):
            c.add_turn("user" if i % 2 == 0 else "assistant", f"msg {i}")

        assert c.has_pending_flush
        assert len(c._pending_flush) > 0

    @pytest.mark.asyncio
    async def test_flush_calls_callback(self):
        """flush_pending() should invoke the callback with staged turns."""
        c = ConversationCompressor(recent_turns_keep=2)
        received_turns: list[dict] = []

        async def capture_callback(turns):
            received_turns.extend(turns)

        c.set_pre_flush_callback(capture_callback)

        for i in range(6):
            c.add_turn("user" if i % 2 == 0 else "assistant", f"msg {i}")

        assert c.has_pending_flush
        await c.flush_pending()

        assert len(received_turns) > 0
        assert not c.has_pending_flush  # cleared after flush

    @pytest.mark.asyncio
    async def test_flush_clears_pending(self):
        """After flush, pending list should be empty."""
        c = ConversationCompressor(recent_turns_keep=2)

        async def noop(turns):
            pass

        c.set_pre_flush_callback(noop)

        for i in range(6):
            c.add_turn("user" if i % 2 == 0 else "assistant", f"msg {i}")

        await c.flush_pending()
        assert not c.has_pending_flush
        assert len(c._pending_flush) == 0

    @pytest.mark.asyncio
    async def test_flush_without_callback_clears_pending(self):
        """If callback was set then removed, flush should still clear pending."""
        c = ConversationCompressor(recent_turns_keep=2)

        async def dummy(turns):
            pass

        c.set_pre_flush_callback(dummy)

        for i in range(6):
            c.add_turn("user" if i % 2 == 0 else "assistant", f"msg {i}")

        # Remove callback
        c._pre_flush_callback = None
        # Pending was already staged
        assert c.has_pending_flush

        await c.flush_pending()
        assert not c.has_pending_flush

    @pytest.mark.asyncio
    async def test_flush_no_pending_is_noop(self):
        """Flushing when nothing is pending should be safe."""
        c = ConversationCompressor(recent_turns_keep=10)
        c.add_turn("user", "hello")

        await c.flush_pending()  # should not raise
        assert not c.has_pending_flush

    def test_reset_clears_pending_flush(self):
        """reset() should also clear pending flush."""
        c = ConversationCompressor(recent_turns_keep=2)

        async def dummy(turns):
            pass

        c.set_pre_flush_callback(dummy)

        for i in range(6):
            c.add_turn("user" if i % 2 == 0 else "assistant", f"msg {i}")

        assert c.has_pending_flush
        c.reset()
        assert not c.has_pending_flush
