"""Parallel Dispatcher — execute independent sub-tasks concurrently.

Uses ``asyncio.gather`` with ``return_exceptions=True`` so one failing
worker never blocks the others.

NOTE: Currently not wired into any live flow. Kept for future Phase 3
      multi-task parallel dispatch. See jarvis_diagnostic_20260219.md A1-3.
"""

from __future__ import annotations

import asyncio
from typing import Any

from loguru import logger

from core.task_router import RoutedTask


class ParallelDispatcher:
    """Dispatch a list of RoutedTask objects to workers in parallel."""

    def __init__(self, workers: dict[str, Any] | None = None):
        self.workers = workers or {}

    async def dispatch(self, tasks: list[RoutedTask]) -> list[dict]:
        """Execute *tasks*, running independent ones concurrently.

        Tasks with ``depends_on`` set are executed sequentially after
        all independent tasks finish.

        Returns one result dict per task (same order as input).
        """
        parallel: list[tuple[int, RoutedTask]] = []
        sequential: list[tuple[int, RoutedTask]] = []

        for idx, task in enumerate(tasks):
            if task.depends_on is not None:
                sequential.append((idx, task))
            else:
                parallel.append((idx, task))

        results: list[dict | None] = [None] * len(tasks)

        # Run independent tasks concurrently
        if parallel:
            coros = [self._execute_single(t) for _, t in parallel]
            raw = await asyncio.gather(*coros, return_exceptions=True)
            for (idx, _), result in zip(parallel, raw):
                if isinstance(result, BaseException):
                    logger.warning(f"Parallel task {idx} raised: {result}")
                    results[idx] = {"error": str(result), "success": False}
                else:
                    results[idx] = result

        # Run dependent tasks sequentially
        for idx, task in sequential:
            try:
                results[idx] = await self._execute_single(task)
            except Exception as exc:
                logger.warning(f"Sequential task {idx} raised: {exc}")
                results[idx] = {"error": str(exc), "success": False}

        return results  # type: ignore[return-value]

    async def _execute_single(self, task: RoutedTask) -> dict:
        """Execute one task via its designated worker."""
        worker = self.workers.get(task.worker) if task.worker else None
        if worker is None:
            # No worker — this task should be handled by CEO LLM
            return {"needs_llm": True, "task_type": task.task_type}

        try:
            result = await worker.execute(task.text)
            if isinstance(result, dict):
                result.setdefault("success", "error" not in result)
                return result
            return {"success": True, "content": str(result)}
        except Exception as exc:
            logger.warning(f"Worker {task.worker} failed: {exc}")
            return {"error": str(exc), "success": False}
