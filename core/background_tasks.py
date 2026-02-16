"""Background Task Manager — run long tasks without blocking replies.

When a task is expected to take a while (e.g. web scraping), the
manager sends an immediate reply ("正在查詢"), executes the task
in the background, and sends a follow-up message when it finishes.
"""

from __future__ import annotations

import asyncio
import uuid
from typing import Any, Awaitable, Callable

from loguru import logger


class BackgroundTaskManager:
    """Execute coroutines in the background and notify via callback.

    Args:
        send_fn: Async callable ``(chat_id, text) -> None`` used to
            push results back to the user once the task completes.
    """

    def __init__(
        self,
        send_fn: Callable[..., Awaitable[None]] | None = None,
    ):
        self._send = send_fn
        self._running: dict[str, asyncio.Task] = {}

    # ── Public API ──────────────────────────────────────────────

    async def run_in_background(
        self,
        coroutine: Awaitable[Any],
        *,
        chat_id: int | str = 0,
        immediate_reply: str = "Sir，正在查詢，稍後回報。",
        task_id: str | None = None,
    ) -> str:
        """Send *immediate_reply*, then execute *coroutine* in the background.

        Returns the task_id (useful for tracking / cancellation).
        """
        tid = task_id or uuid.uuid4().hex[:8]

        # Send immediate reply
        if self._send:
            try:
                await self._send(chat_id, immediate_reply)
            except Exception as exc:
                logger.warning(f"BG immediate reply failed: {exc}")

        async def _wrapper() -> None:
            try:
                result = await coroutine
                if self._send:
                    if isinstance(result, dict):
                        text = result.get("content") or result.get("text", "任務完成。")
                    elif isinstance(result, str):
                        text = result
                    else:
                        text = "Sir，背景任務已完成。"
                    await self._send(chat_id, str(text))
            except Exception as exc:
                logger.warning(f"Background task {tid} failed: {exc}")
                if self._send:
                    await self._send(
                        chat_id, f"Sir，背景任務異常：{str(exc)[:100]}",
                    )
            finally:
                self._running.pop(tid, None)

        task = asyncio.create_task(_wrapper())
        self._running[tid] = task
        return tid

    def cancel(self, task_id: str) -> bool:
        """Cancel a running background task. Returns True if cancelled."""
        task = self._running.pop(task_id, None)
        if task and not task.done():
            task.cancel()
            return True
        return False

    @property
    def active_count(self) -> int:
        """Number of currently running background tasks."""
        # Clean up finished tasks
        done = [k for k, t in self._running.items() if t.done()]
        for k in done:
            self._running.pop(k, None)
        return len(self._running)

    async def wait_all(self) -> None:
        """Wait for all running tasks to finish (useful at shutdown)."""
        if self._running:
            await asyncio.gather(
                *self._running.values(), return_exceptions=True,
            )
            self._running.clear()
