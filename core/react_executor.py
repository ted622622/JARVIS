"""ReactExecutor — fallback chain executor with fuse protection.

Tries workers in a predefined chain order. Each failure is classified
by ErrorClassifier, retried if appropriate, then falls through to the
next worker. Three-layer fuse (rounds, time, daily budget) prevents
runaway token consumption.

Internal helpers (_ErrorDeduplicator, _LoopDetector, FuseState) are
co-located here since they're only used by ReactExecutor.
"""

from __future__ import annotations

import asyncio
import time
from collections import deque
from dataclasses import dataclass, field
from datetime import date
from typing import Any

from loguru import logger

from core.error_classifier import ErrorClassifier


# ── Internal helpers ──────────────────────────────────────────────


class _ErrorDeduplicator:
    """Skip retries for identical errors within a 30-minute window."""

    TTL = 1800  # 30 minutes

    def __init__(self) -> None:
        self._seen: dict[str, float] = {}

    def is_duplicate(self, key: str) -> bool:
        ts = self._seen.get(key)
        if ts is None:
            return False
        return (time.time() - ts) < self.TTL

    def record(self, key: str) -> None:
        self._seen[key] = time.time()
        # Evict stale entries
        cutoff = time.time() - self.TTL
        self._seen = {k: v for k, v in self._seen.items() if v > cutoff}


class _LoopDetector:
    """Detect A->A or A->B->A circular fallback patterns."""

    def __init__(self) -> None:
        self._actions: list[str] = []

    def record(self, action: str) -> bool:
        """Record an action. Returns True if a loop is detected."""
        if self._actions and self._actions[-1] == action:
            return True  # A -> A
        if len(self._actions) >= 2 and self._actions[-2] == action:
            return True  # A -> B -> A
        self._actions.append(action)
        return False

    def reset(self) -> None:
        self._actions.clear()


# ── FuseState ─────────────────────────────────────────────────────


@dataclass
class FuseState:
    """Three-layer fuse to prevent runaway token consumption.

    Layer 1: Per-task limits (max_rounds, max_time_seconds)
    Layer 2: Sliding window (max N tasks in M seconds)
    Layer 3: Daily token budget
    """

    # Per-task limits
    max_rounds: int = 3
    max_tokens: int = 2000
    max_time_seconds: float = 60.0

    # Sliding window: 5 tasks in 5 minutes
    window_max_tasks: int = 5
    window_seconds: float = 300.0
    _window_timestamps: deque = field(default_factory=deque)

    # Daily budget: 10K tokens
    daily_token_budget: int = 10_000
    _daily_tokens_used: int = 0
    _budget_date: str = field(default_factory=lambda: str(date.today()))

    def check_window(self) -> bool:
        """Return True if within sliding window limit."""
        self._evict_window()
        return len(self._window_timestamps) < self.window_max_tasks

    def record_window(self) -> None:
        """Record a task execution in the sliding window."""
        self._window_timestamps.append(time.time())

    def _evict_window(self) -> None:
        cutoff = time.time() - self.window_seconds
        while self._window_timestamps and self._window_timestamps[0] < cutoff:
            self._window_timestamps.popleft()

    def check_daily(self, tokens: int = 0) -> bool:
        """Return True if within daily budget."""
        self._reset_if_new_day()
        return (self._daily_tokens_used + tokens) <= self.daily_token_budget

    def record_daily(self, tokens: int) -> None:
        """Record token usage against daily budget."""
        self._reset_if_new_day()
        self._daily_tokens_used += tokens

    def _reset_if_new_day(self) -> None:
        today = str(date.today())
        if self._budget_date != today:
            self._daily_tokens_used = 0
            self._budget_date = today


# ── TaskResult ────────────────────────────────────────────────────


@dataclass
class TaskResult:
    """Result of a ReactExecutor.execute() call."""

    success: bool
    result: Any = None
    error: str = ""
    attempts: list[dict] = field(default_factory=list)
    tokens_used: int = 0
    gave_up_reason: str = ""


# ── Fallback chains ──────────────────────────────────────────────

FALLBACK_CHAINS: dict[str, list[str]] = {
    "web_browse":     ["browser", "knowledge"],
    "web_search":     ["browser", "search", "knowledge"],
    "maps_search":    ["browser", "knowledge"],
    "file_operation": ["interpreter", "code", "knowledge"],
    "code_task":      ["code", "knowledge"],
    "calendar":       ["gog", "knowledge"],    # H1 v2: gog CLI first
    "email":          ["gog", "knowledge"],    # H1 v2: gog CLI first
    "booking":        ["browser", "assist"],   # H2: assist does 90% + options
    "ticket":         ["browser", "assist"],   # H2: assist does 90% + options
    "general":        ["knowledge"],
}


# ── ReactExecutor ─────────────────────────────────────────────────


class ReactExecutor:
    """Execute tasks with automatic fallback and fuse protection.

    Usage:
        executor = ReactExecutor(workers={"browser": bw, "knowledge": kw})
        result = await executor.execute("web_search", "查高鐵票")
    """

    def __init__(
        self,
        workers: dict[str, Any],
        fuse: FuseState | None = None,
    ):
        self.workers = workers
        self.fuse = fuse or FuseState()
        self._dedup = _ErrorDeduplicator()

    async def execute(
        self,
        chain_name: str,
        task: str,
        **kwargs: Any,
    ) -> TaskResult:
        """Execute a task through a fallback chain.

        Args:
            chain_name: Key into FALLBACK_CHAINS (e.g. "web_search").
            task: Task description.
            **kwargs: Passed to worker.execute().

        Returns:
            TaskResult with success status and attempts log.
        """
        chain = FALLBACK_CHAINS.get(chain_name, FALLBACK_CHAINS["general"])
        loop_detector = _LoopDetector()
        attempts: list[dict] = []
        start_time = time.time()
        total_tokens = 0

        # Fuse check: sliding window
        if not self.fuse.check_window():
            logger.warning("ReactExecutor: sliding window exceeded, rejecting task")
            return TaskResult(
                success=False,
                gave_up_reason="sliding_window_exceeded",
                attempts=attempts,
            )

        self.fuse.record_window()

        round_count = 0

        for worker_name in chain:
            worker = self.workers.get(worker_name)
            if not worker:
                logger.debug(f"ReactExecutor: worker '{worker_name}' not found, skipping")
                continue

            # Loop detection
            if loop_detector.record(worker_name):
                logger.warning(f"ReactExecutor: loop detected at '{worker_name}', breaking")
                break

            # Attempt with retries
            strategy = None
            retry_count = 0
            max_retries = 0

            while True:
                # Fuse: round limit
                round_count += 1
                if round_count > self.fuse.max_rounds:
                    logger.warning("ReactExecutor: max rounds exceeded")
                    return TaskResult(
                        success=False,
                        gave_up_reason="max_rounds_exceeded",
                        attempts=attempts,
                        tokens_used=total_tokens,
                    )

                # Fuse: time limit
                elapsed = time.time() - start_time
                if elapsed > self.fuse.max_time_seconds:
                    logger.warning(f"ReactExecutor: time limit exceeded ({elapsed:.1f}s)")
                    return TaskResult(
                        success=False,
                        gave_up_reason="time_limit_exceeded",
                        attempts=attempts,
                        tokens_used=total_tokens,
                    )

                # Fuse: daily budget
                if not self.fuse.check_daily():
                    logger.warning("ReactExecutor: daily token budget exceeded")
                    return TaskResult(
                        success=False,
                        gave_up_reason="daily_budget_exceeded",
                        attempts=attempts,
                        tokens_used=total_tokens,
                    )

                # Error dedup check
                dedup_key = f"{worker_name}:{task[:80]}"
                if self._dedup.is_duplicate(dedup_key):
                    logger.debug(f"ReactExecutor: dedup skip '{worker_name}' for this task")
                    break  # skip to next worker

                # Execute worker
                try:
                    timeout = min(
                        self.fuse.max_time_seconds - elapsed,
                        30.0,  # per-worker timeout cap
                    )
                    exec_kwargs = dict(kwargs)
                    if worker_name in ("knowledge", "assist") and attempts:
                        exec_kwargs["failed_attempts"] = attempts
                    if worker_name == "assist":
                        exec_kwargs["task_type"] = chain_name

                    result = await asyncio.wait_for(
                        worker.execute(task, **exec_kwargs),
                        timeout=max(timeout, 5.0),
                    )
                except asyncio.TimeoutError:
                    result = {"error": f"Worker '{worker_name}' timed out"}
                except Exception as e:
                    result = {"error": str(e)}

                # Classify result
                if isinstance(result, dict):
                    strategy = ErrorClassifier.classify_worker_result(result, worker_name)
                else:
                    strategy = None

                # Success
                if strategy is None:
                    # Estimate tokens used (rough: 500 per knowledge call)
                    if worker_name == "knowledge":
                        total_tokens += 500
                        self.fuse.record_daily(500)

                    logger.info(f"ReactExecutor: '{worker_name}' succeeded for '{task[:40]}'")
                    return TaskResult(
                        success=True,
                        result=result,
                        attempts=attempts,
                        tokens_used=total_tokens,
                    )

                # Record attempt
                error_str = result.get("error", "unknown") if isinstance(result, dict) else str(result)
                attempts.append({
                    "worker": worker_name,
                    "error": error_str,
                    "error_type": strategy.error_type.value,
                    "retry": retry_count,
                })
                logger.info(
                    f"ReactExecutor: '{worker_name}' failed ({strategy.error_type.value}): "
                    f"{error_str[:80]}"
                )

                # Should retry?
                if strategy.retry and retry_count < strategy.max_retries:
                    retry_count += 1
                    if strategy.delay_seconds > 0:
                        await asyncio.sleep(strategy.delay_seconds)
                    continue

                # Record dedup for this error
                self._dedup.record(dedup_key)
                break  # Move to next worker in chain

        # All workers exhausted
        gave_up = "all_workers_exhausted"
        if attempts:
            last = attempts[-1]
            gave_up = f"all_workers_exhausted: last={last['worker']}({last['error_type']})"

        logger.warning(f"ReactExecutor gave up: {gave_up}")
        return TaskResult(
            success=False,
            error=attempts[-1]["error"] if attempts else "no workers available",
            gave_up_reason=gave_up,
            attempts=attempts,
            tokens_used=total_tokens,
        )
