"""Memory file watcher — auto re-index when memory files change.

Uses watchdog to monitor memory/ directory for .md file changes,
then triggers a debounced rebuild of the HybridSearch index.

Inspired by OpenClaw's chokidar-based live sync pattern.
"""

from __future__ import annotations

import asyncio
import threading
from typing import Any

from loguru import logger

try:
    from watchdog.observers import Observer
    from watchdog.events import FileSystemEventHandler, FileModifiedEvent

    _WATCHDOG_OK = True
except ImportError:
    _WATCHDOG_OK = False
    Observer = None  # type: ignore[assignment, misc]
    FileSystemEventHandler = object  # type: ignore[assignment, misc]


DEBOUNCE_SECONDS = 2.0


class _MemoryHandler(FileSystemEventHandler):
    """Handles file change events in the memory/ directory."""

    def __init__(self, rebuild_callback: Any, loop: asyncio.AbstractEventLoop):
        super().__init__()
        self._callback = rebuild_callback
        self._loop = loop
        self._timer: threading.Timer | None = None
        self._lock = threading.Lock()

    def on_modified(self, event: Any) -> None:
        if event.is_directory:
            return
        src = getattr(event, "src_path", "")
        if not src.endswith(".md"):
            return
        self._schedule_rebuild()

    def on_created(self, event: Any) -> None:
        self.on_modified(event)

    def _schedule_rebuild(self) -> None:
        """Debounce: reset timer on each change, rebuild after DEBOUNCE_SECONDS."""
        with self._lock:
            if self._timer is not None:
                self._timer.cancel()
            self._timer = threading.Timer(
                DEBOUNCE_SECONDS,
                self._trigger_rebuild,
            )
            self._timer.daemon = True
            self._timer.start()

    def _trigger_rebuild(self) -> None:
        """Called from timer thread — schedule async rebuild on the event loop."""
        try:
            asyncio.run_coroutine_threadsafe(
                self._async_rebuild(), self._loop,
            )
        except Exception as e:
            logger.warning(f"Memory watcher rebuild trigger failed: {e}")

    async def _async_rebuild(self) -> None:
        try:
            await self._callback()
            logger.info("Memory index rebuilt (file watcher trigger)")
        except Exception as e:
            logger.warning(f"Memory index rebuild failed: {e}")


class MemoryWatcher:
    """Watch memory/ directory and auto-rebuild search index on changes."""

    def __init__(
        self,
        memory_dir: str,
        rebuild_callback: Any,
        loop: asyncio.AbstractEventLoop | None = None,
    ):
        self._dir = memory_dir
        self._callback = rebuild_callback
        self._loop = loop or asyncio.get_event_loop()
        self._observer: Any = None

    def start(self) -> bool:
        """Start watching. Returns True if successful, False if watchdog unavailable."""
        if not _WATCHDOG_OK:
            logger.warning("watchdog not installed — memory file watcher disabled")
            return False

        handler = _MemoryHandler(self._callback, self._loop)
        self._observer = Observer()
        self._observer.schedule(handler, self._dir, recursive=True)
        self._observer.daemon = True
        self._observer.start()
        logger.info(f"Memory watcher started on {self._dir}")
        return True

    def stop(self) -> None:
        """Stop watching."""
        if self._observer is not None:
            self._observer.stop()
            self._observer = None
            logger.info("Memory watcher stopped")
