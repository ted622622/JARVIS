"""Tests for PendingTaskManager — JSON-backed retry queue."""

from __future__ import annotations

import json
import time

import pytest

from core.pending_tasks import PendingTask, PendingTaskManager


@pytest.fixture
def task_file(tmp_path):
    return str(tmp_path / "pending.json")


@pytest.fixture
def mgr(task_file):
    return PendingTaskManager(task_file)


class TestPendingTaskManager:
    def test_add_and_count(self, mgr):
        tid = mgr.add("web_search", "查高鐵票")
        assert tid is not None
        assert mgr.task_count == 1

    def test_add_returns_task_id(self, mgr):
        tid = mgr.add("web_search", "查高鐵票")
        assert isinstance(tid, str)
        assert len(tid) > 0

    def test_save_and_load(self, task_file):
        mgr1 = PendingTaskManager(task_file)
        mgr1.add("web_search", "查高鐵票", url="https://thsr.com")
        mgr1.add("code_task", "寫排序函數")
        mgr1.save()

        mgr2 = PendingTaskManager(task_file)
        loaded = mgr2.load()
        assert loaded == 2
        assert mgr2.task_count == 2

    def test_max_tasks_limit(self, mgr):
        for i in range(20):
            tid = mgr.add("web_search", f"task {i}")
            assert tid is not None

        # 21st should fail
        tid = mgr.add("web_search", "overflow")
        assert tid is None
        assert mgr.task_count == 20

    def test_cooldown_not_due(self, mgr):
        tid = mgr.add("web_search", "test")
        # Simulate recent attempt
        task = mgr._tasks[tid]
        task.last_attempt_at = time.time()

        due = mgr.get_due_tasks()
        assert len(due) == 0

    def test_cooldown_elapsed(self, mgr):
        tid = mgr.add("web_search", "test")
        # Simulate old attempt (cooldown elapsed)
        task = mgr._tasks[tid]
        task.last_attempt_at = time.time() - 1000  # > 900s cooldown

        due = mgr.get_due_tasks()
        assert len(due) == 1
        assert due[0].task_id == tid

    def test_new_task_is_immediately_due(self, mgr):
        """New task with last_attempt_at=0 should be immediately due."""
        mgr.add("web_search", "test")
        due = mgr.get_due_tasks()
        assert len(due) == 1

    def test_mark_completed(self, mgr):
        tid = mgr.add("web_search", "test")
        mgr.mark_completed(tid)
        task = mgr._tasks[tid]
        assert task.status == "completed"
        assert mgr.task_count == 0  # completed doesn't count

    def test_mark_failed_increments_retry(self, mgr):
        tid = mgr.add("web_search", "test")
        mgr.mark_failed(tid, "Connection error")
        task = mgr._tasks[tid]
        assert task.retry_count == 1
        assert task.last_error == "Connection error"
        assert task.status == "pending"

    def test_mark_failed_gives_up_after_max_retries(self, mgr):
        tid = mgr.add("web_search", "test")
        for i in range(3):
            mgr.mark_failed(tid, f"Error {i}")
        task = mgr._tasks[tid]
        assert task.status == "given_up"
        assert task.retry_count == 3

    def test_get_given_up_tasks(self, mgr):
        tid1 = mgr.add("web_search", "task1")
        tid2 = mgr.add("code_task", "task2")
        for i in range(3):
            mgr.mark_failed(tid1, "fail")
        given_up = mgr.get_given_up_tasks()
        assert len(given_up) == 1
        assert given_up[0].task_id == tid1

    def test_clear_given_up(self, mgr):
        tid1 = mgr.add("web_search", "task1")
        tid2 = mgr.add("code_task", "task2")
        for i in range(3):
            mgr.mark_failed(tid1, "fail")

        mgr.clear_given_up()
        assert len(mgr.get_given_up_tasks()) == 0
        assert mgr.task_count == 1  # task2 still pending

    def test_load_empty_file(self, task_file):
        # Create empty file
        from pathlib import Path
        Path(task_file).parent.mkdir(parents=True, exist_ok=True)
        Path(task_file).write_text("[]", encoding="utf-8")

        mgr = PendingTaskManager(task_file)
        loaded = mgr.load()
        assert loaded == 0

    def test_load_bad_json(self, task_file):
        from pathlib import Path
        Path(task_file).parent.mkdir(parents=True, exist_ok=True)
        Path(task_file).write_text("{invalid json!!!", encoding="utf-8")

        mgr = PendingTaskManager(task_file)
        loaded = mgr.load()
        assert loaded == 0

    def test_load_nonexistent_file(self, task_file):
        mgr = PendingTaskManager(task_file)
        loaded = mgr.load()
        assert loaded == 0

    def test_add_evicts_completed_before_checking_limit(self, mgr):
        """Adding should evict completed tasks before checking MAX_TASKS."""
        for i in range(20):
            tid = mgr.add("web_search", f"task {i}")
        # Complete 5 tasks
        for tid in list(mgr._tasks.keys())[:5]:
            mgr.mark_completed(tid)

        # Now we should be able to add more
        new_tid = mgr.add("web_search", "new task")
        assert new_tid is not None

    def test_persistence_roundtrip(self, task_file):
        """Full roundtrip: add → save → load → verify."""
        mgr1 = PendingTaskManager(task_file)
        tid = mgr1.add("web_search", "查天氣", url="https://weather.com")
        mgr1.mark_failed(tid, "timeout")
        mgr1.save()

        mgr2 = PendingTaskManager(task_file)
        mgr2.load()
        tasks = mgr2.all_tasks
        assert len(tasks) == 1
        assert tasks[0].retry_count == 1
        assert tasks[0].last_error == "timeout"
        assert tasks[0].kwargs.get("url") == "https://weather.com"
