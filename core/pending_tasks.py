"""Pending Task Manager — JSON-backed task queue for retry.

Failed tasks are stored to disk and retried periodically by Heartbeat.
Max 20 tasks, 15-min cooldown between retries, max 3 retries before giving up.
"""

from __future__ import annotations

import json
import time
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from loguru import logger


@dataclass
class PendingTask:
    """A task that failed and is awaiting retry."""

    task_id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    task_type: str = ""          # chain name (e.g. "web_search")
    task_description: str = ""
    kwargs: dict = field(default_factory=dict)
    retry_count: int = 0
    max_retries: int = 3
    created_at: float = field(default_factory=time.time)
    last_attempt_at: float = 0.0
    last_error: str = ""
    status: str = "pending"      # pending / completed / given_up


class PendingTaskManager:
    """JSON file-backed pending task queue.

    Usage:
        mgr = PendingTaskManager("./data/pending_tasks.json")
        mgr.load()
        task_id = mgr.add("web_search", "查高鐵票", url="...")
        # ... later in heartbeat ...
        for task in mgr.get_due_tasks():
            # retry via ReactExecutor
            ...
        mgr.save()
    """

    MAX_TASKS = 20
    RETRY_COOLDOWN = 900  # 15 minutes

    def __init__(self, path: str = "./data/pending_tasks.json"):
        self.path = Path(path)
        self._tasks: dict[str, PendingTask] = {}

    def load(self) -> int:
        """Load tasks from JSON file.

        Returns:
            Number of tasks loaded.
        """
        if not self.path.exists():
            return 0

        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
            if not isinstance(data, list):
                logger.warning("pending_tasks.json: expected list, got %s", type(data).__name__)
                return 0

            self._tasks = {}
            for item in data:
                if not isinstance(item, dict):
                    continue
                task = PendingTask(
                    task_id=item.get("task_id", uuid.uuid4().hex[:12]),
                    task_type=item.get("task_type", ""),
                    task_description=item.get("task_description", ""),
                    kwargs=item.get("kwargs", {}),
                    retry_count=item.get("retry_count", 0),
                    max_retries=item.get("max_retries", 3),
                    created_at=item.get("created_at", time.time()),
                    last_attempt_at=item.get("last_attempt_at", 0.0),
                    last_error=item.get("last_error", ""),
                    status=item.get("status", "pending"),
                )
                self._tasks[task.task_id] = task

            logger.info(f"Loaded {len(self._tasks)} pending tasks")
            return len(self._tasks)

        except (json.JSONDecodeError, OSError) as e:
            logger.warning(f"Failed to load pending tasks: {e}")
            return 0

    def save(self) -> None:
        """Persist tasks to JSON file."""
        self.path.parent.mkdir(parents=True, exist_ok=True)
        data = [asdict(t) for t in self._tasks.values()]
        self.path.write_text(
            json.dumps(data, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    def add(
        self,
        task_type: str,
        description: str,
        **kwargs: Any,
    ) -> str | None:
        """Add a pending task. Returns task_id, or None if at capacity.

        Args:
            task_type: The fallback chain name (e.g. "web_search").
            description: Human-readable task description.
            **kwargs: Worker-specific keyword arguments.

        Returns:
            task_id string, or None if MAX_TASKS reached.
        """
        # Evict completed/given_up tasks first
        self._tasks = {
            tid: t for tid, t in self._tasks.items()
            if t.status == "pending"
        }

        if len(self._tasks) >= self.MAX_TASKS:
            logger.warning("Pending task queue full (%d), rejecting new task", self.MAX_TASKS)
            return None

        task = PendingTask(
            task_type=task_type,
            task_description=description,
            kwargs=kwargs,
        )
        self._tasks[task.task_id] = task
        logger.info(f"Added pending task {task.task_id}: {description[:60]}")
        return task.task_id

    def get_due_tasks(self) -> list[PendingTask]:
        """Return tasks whose cooldown has elapsed and are still pending."""
        now = time.time()
        due = []
        for task in self._tasks.values():
            if task.status != "pending":
                continue
            if now - task.last_attempt_at >= self.RETRY_COOLDOWN:
                due.append(task)
        return due

    def mark_completed(self, task_id: str) -> None:
        """Mark a task as completed."""
        task = self._tasks.get(task_id)
        if task:
            task.status = "completed"
            logger.info(f"Pending task {task_id} completed")

    def mark_failed(self, task_id: str, error: str) -> None:
        """Record a failed retry attempt. Marks given_up if max retries reached."""
        task = self._tasks.get(task_id)
        if not task:
            return

        task.retry_count += 1
        task.last_attempt_at = time.time()
        task.last_error = error

        if task.retry_count >= task.max_retries:
            task.status = "given_up"
            logger.warning(
                f"Pending task {task_id} given up after {task.retry_count} retries"
            )
        else:
            logger.info(
                f"Pending task {task_id} failed (retry {task.retry_count}/{task.max_retries})"
            )

    def get_given_up_tasks(self) -> list[PendingTask]:
        """Return tasks that have exceeded max retries."""
        return [t for t in self._tasks.values() if t.status == "given_up"]

    def clear_given_up(self) -> None:
        """Remove all given_up tasks."""
        self._tasks = {
            tid: t for tid, t in self._tasks.items()
            if t.status != "given_up"
        }

    @property
    def task_count(self) -> int:
        """Number of pending tasks (excludes completed/given_up)."""
        return sum(1 for t in self._tasks.values() if t.status == "pending")

    @property
    def all_tasks(self) -> list[PendingTask]:
        """All tasks including completed and given_up."""
        return list(self._tasks.values())
