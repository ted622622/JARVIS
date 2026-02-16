"""Tests for ReactExecutor — fallback chain execution with fuse protection."""

from __future__ import annotations

import asyncio
import time

import pytest
from unittest.mock import AsyncMock, MagicMock

from core.react_executor import (
    FALLBACK_CHAINS,
    FuseState,
    ReactExecutor,
    TaskResult,
    _ErrorDeduplicator,
    _LoopDetector,
)


# ── Helpers ─────────────────────────────────────────────────────


def make_worker(name: str, result: dict | None = None, side_effect=None):
    """Create a mock worker with given result or side effect."""
    w = AsyncMock()
    w.name = name
    if side_effect:
        w.execute.side_effect = side_effect
    else:
        w.execute.return_value = result or {"status": "ok", "worker": name}
    return w


def make_failing_worker(name: str, error: str):
    return make_worker(name, result={"error": error, "worker": name})


# ── ErrorDeduplicator Tests ──────────────────────────────────────


class TestErrorDeduplicator:
    def test_first_occurrence_not_duplicate(self):
        d = _ErrorDeduplicator()
        assert d.is_duplicate("err1") is False

    def test_recorded_is_duplicate(self):
        d = _ErrorDeduplicator()
        d.record("err1")
        assert d.is_duplicate("err1") is True

    def test_different_key_not_duplicate(self):
        d = _ErrorDeduplicator()
        d.record("err1")
        assert d.is_duplicate("err2") is False

    def test_expired_not_duplicate(self):
        d = _ErrorDeduplicator()
        d._seen["err1"] = time.time() - 2000  # expired
        assert d.is_duplicate("err1") is False


# ── LoopDetector Tests ───────────────────────────────────────────


class TestLoopDetector:
    def test_no_loop_on_first(self):
        ld = _LoopDetector()
        assert ld.record("A") is False

    def test_a_a_loop(self):
        ld = _LoopDetector()
        ld.record("A")
        assert ld.record("A") is True

    def test_a_b_a_loop(self):
        ld = _LoopDetector()
        ld.record("A")
        ld.record("B")
        assert ld.record("A") is True

    def test_a_b_c_no_loop(self):
        ld = _LoopDetector()
        ld.record("A")
        ld.record("B")
        assert ld.record("C") is False


# ── FuseState Tests ──────────────────────────────────────────────


class TestFuseState:
    def test_window_initially_open(self):
        fuse = FuseState()
        assert fuse.check_window() is True

    def test_window_closes_after_max(self):
        fuse = FuseState(window_max_tasks=2, window_seconds=60)
        fuse.record_window()
        fuse.record_window()
        assert fuse.check_window() is False

    def test_window_reopens_after_timeout(self):
        fuse = FuseState(window_max_tasks=1, window_seconds=0.1)
        fuse.record_window()
        assert fuse.check_window() is False
        time.sleep(0.15)
        assert fuse.check_window() is True

    def test_daily_budget_initially_open(self):
        fuse = FuseState(daily_token_budget=1000)
        assert fuse.check_daily(500) is True

    def test_daily_budget_exceeded(self):
        fuse = FuseState(daily_token_budget=100)
        fuse.record_daily(100)
        assert fuse.check_daily(1) is False

    def test_daily_budget_resets_on_new_day(self):
        fuse = FuseState(daily_token_budget=100)
        fuse.record_daily(100)
        fuse._budget_date = "2020-01-01"  # force stale date
        assert fuse.check_daily(50) is True  # should reset


# ── ReactExecutor Core Tests ─────────────────────────────────────


class TestReactExecutorSuccess:
    @pytest.mark.asyncio
    async def test_first_worker_succeeds(self):
        browser = make_worker("browser", {"content": "data", "worker": "browser"})
        knowledge = make_worker("knowledge")
        executor = ReactExecutor(workers={"browser": browser, "knowledge": knowledge})

        result = await executor.execute("web_search", "test query")
        assert result.success is True
        assert result.result["content"] == "data"
        browser.execute.assert_called_once()
        knowledge.execute.assert_not_called()

    @pytest.mark.asyncio
    async def test_fallback_to_second_worker(self):
        browser = make_failing_worker("browser", "Connection refused")
        knowledge = make_worker("knowledge", {"result": "answer", "source": "knowledge", "worker": "knowledge"})
        executor = ReactExecutor(
            workers={"browser": browser, "knowledge": knowledge},
            fuse=FuseState(max_rounds=10),
        )

        result = await executor.execute("web_search", "test query")
        assert result.success is True
        assert result.result["source"] == "knowledge"
        assert len(result.attempts) >= 1
        assert result.attempts[0]["worker"] == "browser"

    @pytest.mark.asyncio
    async def test_knowledge_receives_failed_attempts(self):
        browser = make_failing_worker("browser", "Connection refused")
        knowledge = make_worker("knowledge", {"result": "answer", "source": "knowledge", "worker": "knowledge"})
        executor = ReactExecutor(
            workers={"browser": browser, "knowledge": knowledge},
            fuse=FuseState(max_rounds=10),
        )

        await executor.execute("web_search", "test query")
        # knowledge should have been called with failed_attempts
        call_kwargs = knowledge.execute.call_args
        assert "failed_attempts" in call_kwargs.kwargs


class TestReactExecutorRetry:
    @pytest.mark.asyncio
    async def test_timeout_triggers_retry(self):
        """Timeout error should retry once then fallback."""
        call_count = 0

        async def timeout_then_fail(task, **kw):
            nonlocal call_count
            call_count += 1
            return {"error": "Request timed out", "worker": "browser"}

        browser = AsyncMock()
        browser.name = "browser"
        browser.execute.side_effect = timeout_then_fail
        knowledge = make_worker("knowledge", {"result": "ok", "source": "knowledge", "worker": "knowledge"})

        executor = ReactExecutor(
            workers={"browser": browser, "knowledge": knowledge},
            fuse=FuseState(max_rounds=10),
        )
        result = await executor.execute("web_search", "test")
        assert result.success is True
        # browser called: 1 original + 1 retry = 2
        assert call_count == 2

    @pytest.mark.asyncio
    async def test_captcha_no_retry(self):
        """CAPTCHA should not retry, go directly to next worker."""
        browser = make_failing_worker("browser", "Page requires CAPTCHA verification")
        knowledge = make_worker("knowledge", {"result": "ok", "source": "knowledge", "worker": "knowledge"})
        executor = ReactExecutor(workers={"browser": browser, "knowledge": knowledge})

        result = await executor.execute("web_search", "test")
        assert result.success is True
        browser.execute.assert_called_once()  # No retry


class TestReactExecutorFuse:
    @pytest.mark.asyncio
    async def test_max_rounds_fuse(self):
        """Should stop after max_rounds."""
        browser = make_failing_worker("browser", "some error")
        knowledge = make_failing_worker("knowledge", "also error")
        fuse = FuseState(max_rounds=1)
        executor = ReactExecutor(
            workers={"browser": browser, "knowledge": knowledge},
            fuse=fuse,
        )
        result = await executor.execute("web_search", "test")
        assert result.success is False
        assert "max_rounds" in result.gave_up_reason

    @pytest.mark.asyncio
    async def test_time_limit_fuse(self):
        """Should stop when time limit is reached."""

        async def slow_worker(task, **kw):
            await asyncio.sleep(0.3)
            return {"error": "still failing"}

        browser = AsyncMock()
        browser.execute.side_effect = slow_worker
        fuse = FuseState(max_time_seconds=0.1, max_rounds=10)
        executor = ReactExecutor(workers={"browser": browser}, fuse=fuse)

        result = await executor.execute("web_search", "test")
        assert result.success is False
        assert "time_limit" in result.gave_up_reason

    @pytest.mark.asyncio
    async def test_daily_budget_fuse(self):
        """Should reject when daily budget is exceeded."""
        fuse = FuseState(daily_token_budget=0)
        fuse.record_daily(1)
        browser = make_worker("browser")
        executor = ReactExecutor(workers={"browser": browser}, fuse=fuse)

        result = await executor.execute("web_search", "test")
        assert result.success is False
        assert "daily_budget" in result.gave_up_reason

    @pytest.mark.asyncio
    async def test_sliding_window_rejection(self):
        """Should reject when sliding window is full."""
        fuse = FuseState(window_max_tasks=1, window_seconds=60)
        fuse.record_window()  # fill the window
        browser = make_worker("browser")
        executor = ReactExecutor(workers={"browser": browser}, fuse=fuse)

        result = await executor.execute("web_search", "test")
        assert result.success is False
        assert "sliding_window" in result.gave_up_reason


class TestReactExecutorEdgeCases:
    @pytest.mark.asyncio
    async def test_missing_worker_skipped(self):
        """Workers not in the dict should be skipped."""
        # web_search chain is ["browser", "knowledge"], only knowledge exists
        knowledge = make_worker("knowledge", {"result": "ok", "source": "knowledge", "worker": "knowledge"})
        executor = ReactExecutor(workers={"knowledge": knowledge})

        result = await executor.execute("web_search", "test")
        assert result.success is True
        knowledge.execute.assert_called_once()

    @pytest.mark.asyncio
    async def test_unknown_chain_uses_general(self):
        """Unknown chain name should fall back to 'general' chain."""
        knowledge = make_worker("knowledge", {"result": "ok", "source": "knowledge", "worker": "knowledge"})
        executor = ReactExecutor(workers={"knowledge": knowledge})

        result = await executor.execute("nonexistent_chain", "test")
        assert result.success is True

    @pytest.mark.asyncio
    async def test_gave_up_reason_includes_details(self):
        """gave_up_reason should mention last worker and error type."""
        browser = make_failing_worker("browser", "Connection refused")
        executor = ReactExecutor(workers={"browser": browser})

        result = await executor.execute("web_search", "test")
        assert result.success is False
        assert "browser" in result.gave_up_reason

    @pytest.mark.asyncio
    async def test_error_dedup_skips_same_error(self):
        """Same error within 30min window should be skipped."""
        browser = make_failing_worker("browser", "Connection refused")
        knowledge = make_worker("knowledge", {"result": "ok", "source": "knowledge", "worker": "knowledge"})
        executor = ReactExecutor(
            workers={"browser": browser, "knowledge": knowledge},
            fuse=FuseState(max_rounds=10),
        )

        # First call: browser fails, falls to knowledge
        r1 = await executor.execute("web_search", "test query")
        assert r1.success is True

        # Second call: browser should be skipped (dedup), goes straight to knowledge
        browser.execute.reset_mock()
        r2 = await executor.execute("web_search", "test query")
        assert r2.success is True
        browser.execute.assert_not_called()  # skipped by dedup

    @pytest.mark.asyncio
    async def test_worker_exception_caught(self):
        """Worker raising an exception should be handled gracefully."""

        async def exploding(task, **kw):
            raise RuntimeError("unexpected crash")

        browser = AsyncMock()
        browser.execute.side_effect = exploding
        knowledge = make_worker("knowledge", {"result": "ok", "source": "knowledge", "worker": "knowledge"})
        executor = ReactExecutor(workers={"browser": browser, "knowledge": knowledge})

        result = await executor.execute("web_search", "test")
        assert result.success is True

    @pytest.mark.asyncio
    async def test_no_workers_available(self):
        """Empty worker dict should return failure."""
        executor = ReactExecutor(workers={})
        result = await executor.execute("web_search", "test")
        assert result.success is False

    @pytest.mark.asyncio
    async def test_attempts_list_populated(self):
        """Attempts list should log each failed attempt."""
        browser = make_failing_worker("browser", "timeout")
        knowledge = make_failing_worker("knowledge", "LLM down")
        executor = ReactExecutor(
            workers={"browser": browser, "knowledge": knowledge},
            fuse=FuseState(max_rounds=10),
        )

        result = await executor.execute("web_search", "test")
        assert result.success is False
        assert len(result.attempts) >= 2
        workers_tried = [a["worker"] for a in result.attempts]
        assert "browser" in workers_tried
        assert "knowledge" in workers_tried
