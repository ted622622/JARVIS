"""Tests for GogWorker (H0 v2) and TASK_RESOLUTION_CHAINS (H1 v2)."""

from __future__ import annotations

import json
import subprocess
from datetime import datetime, timedelta
from unittest.mock import MagicMock, patch

import pytest

from workers.gog_worker import GogWorker


# ── H0: GogWorker ─────────────────────────────────────────────────


class TestGogWorkerInit:
    """Verify GogWorker initialization and health check."""

    def test_init_with_gog_available(self):
        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(
                returncode=0, stdout="v0.11.0 (abc123)\n",
            )
            gog = GogWorker(account="test@gmail.com")
            assert gog.is_available is True
            assert gog.account == "test@gmail.com"
            assert gog.name == "gog"

    def test_init_gog_not_found(self):
        with patch("subprocess.run", side_effect=FileNotFoundError):
            gog = GogWorker()
            assert gog.is_available is False

    def test_init_gog_nonzero_exit(self):
        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(
                returncode=1, stdout="", stderr="error",
            )
            gog = GogWorker()
            assert gog.is_available is False


class TestRunGog:
    """Verify _run_gog wrapper handles all subprocess outcomes."""

    def _make_gog(self) -> GogWorker:
        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=0, stdout="v0.11.0\n")
            return GogWorker(account="test@gmail.com")

    def test_run_gog_success_json(self):
        gog = self._make_gog()
        data = [{"summary": "Meeting", "start": "10:00"}]
        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(
                returncode=0, stdout=json.dumps(data),
            )
            result = gog._run_gog(["calendar", "events", "primary"])
            assert result["success"] is True
            assert result["data"] == data
            # Verify command includes account, --json, --no-input
            cmd = mock_run.call_args[0][0]
            assert "--account" in cmd
            assert "test@gmail.com" in cmd
            assert "--json" in cmd
            assert "--no-input" in cmd

    def test_run_gog_success_empty_output(self):
        gog = self._make_gog()
        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=0, stdout="")
            result = gog._run_gog(["calendar", "events", "primary"])
            assert result["success"] is True
            assert result["data"] == {}

    def test_run_gog_nonzero_exit(self):
        gog = self._make_gog()
        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(
                returncode=1, stdout="", stderr="auth failed",
            )
            result = gog._run_gog(["calendar", "events", "primary"])
            assert result["success"] is False
            assert "auth failed" in result["error"]

    def test_run_gog_timeout(self):
        gog = self._make_gog()
        with patch("subprocess.run", side_effect=subprocess.TimeoutExpired("gog", 30)):
            result = gog._run_gog(["calendar", "events", "primary"])
            assert result["success"] is False
            assert "timeout" in result["error"]

    def test_run_gog_file_not_found(self):
        gog = self._make_gog()
        with patch("subprocess.run", side_effect=FileNotFoundError):
            result = gog._run_gog(["calendar", "events", "primary"])
            assert result["success"] is False
            assert gog._available is False

    def test_run_gog_invalid_json(self):
        gog = self._make_gog()
        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(
                returncode=0, stdout="not json at all",
            )
            result = gog._run_gog(["calendar", "events", "primary"])
            assert result["success"] is True
            assert result["data"]["raw"] == "not json at all"

    def test_run_gog_not_available(self):
        gog = self._make_gog()
        gog._available = False
        result = gog._run_gog(["test"])
        assert result["success"] is False
        assert "not installed" in result["error"]


class TestCalendarMethods:
    """Verify Calendar helper methods."""

    def _make_gog(self) -> GogWorker:
        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=0, stdout="v0.11.0\n")
            return GogWorker()

    def test_get_today_events(self):
        gog = self._make_gog()
        events = [{"summary": "Standup"}, {"summary": "Lunch"}]
        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(
                returncode=0, stdout=json.dumps(events),
            )
            result = gog.get_today_events()
            assert result == events
            cmd = mock_run.call_args[0][0]
            assert "calendar" in cmd
            assert "events" in cmd
            assert "--from" in cmd
            assert "--to" in cmd

    def test_get_today_events_failure(self):
        gog = self._make_gog()
        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(
                returncode=1, stdout="", stderr="error",
            )
            result = gog.get_today_events()
            assert result == []

    def test_get_events_for_date(self):
        gog = self._make_gog()
        target = datetime(2026, 2, 17)
        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(
                returncode=0, stdout=json.dumps([{"summary": "Dinner"}]),
            )
            result = gog.get_events_for_date(target)
            assert len(result) == 1
            cmd = mock_run.call_args[0][0]
            assert "2026-02-17" in " ".join(cmd)

    def test_get_upcoming_events(self):
        gog = self._make_gog()
        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(
                returncode=0, stdout=json.dumps([]),
            )
            result = gog.get_upcoming_events(minutes=30)
            assert result == []

    def test_create_event(self):
        gog = self._make_gog()
        start = datetime(2026, 2, 17, 19, 0)
        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(
                returncode=0, stdout=json.dumps({"id": "evt123"}),
            )
            result = gog.create_event(
                title="Dinner", start_time=start,
                duration_minutes=90, location="Niku Mura",
            )
            assert result["success"] is True
            cmd = mock_run.call_args[0][0]
            assert "create" in cmd
            assert "--summary" in cmd
            assert "Dinner" in cmd
            assert "--location" in cmd
            assert "Niku Mura" in cmd

    def test_create_event_no_location(self):
        gog = self._make_gog()
        start = datetime(2026, 2, 17, 19, 0)
        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(
                returncode=0, stdout=json.dumps({"id": "evt456"}),
            )
            result = gog.create_event(title="Meeting", start_time=start)
            assert result["success"] is True
            cmd = mock_run.call_args[0][0]
            assert "--location" not in cmd


class TestGmailMethods:
    """Verify Gmail helper methods."""

    def _make_gog(self) -> GogWorker:
        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=0, stdout="v0.11.0\n")
            return GogWorker()

    def test_search_inbox(self):
        gog = self._make_gog()
        emails = [{"subject": "Hello", "from": "test@example.com"}]
        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(
                returncode=0, stdout=json.dumps(emails),
            )
            result = gog.search_inbox("newer_than:1d", max_results=5)
            assert result == emails
            cmd = mock_run.call_args[0][0]
            assert "gmail" in cmd
            assert "search" in cmd
            assert "--max" in cmd
            assert "5" in cmd

    def test_send_email(self):
        gog = self._make_gog()
        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(
                returncode=0, stdout=json.dumps({"status": "sent"}),
            )
            result = gog.send_email(
                to="friend@example.com",
                subject="Hi",
                body="How are you?",
            )
            assert result["success"] is True
            cmd = mock_run.call_args[0][0]
            assert "--to" in cmd
            assert "friend@example.com" in cmd
            assert "--subject" in cmd
            assert "--body" in cmd


class TestDriveMethods:
    """Verify Drive helper methods."""

    def _make_gog(self) -> GogWorker:
        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=0, stdout="v0.11.0\n")
            return GogWorker()

    def test_search_drive(self):
        gog = self._make_gog()
        files = [{"name": "report.pdf", "id": "abc123"}]
        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(
                returncode=0, stdout=json.dumps(files),
            )
            result = gog.search_drive("report", max_results=3)
            assert result == files


class TestGogExecute:
    """Verify worker interface (execute method)."""

    @pytest.mark.asyncio
    async def test_execute_available(self):
        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=0, stdout="v0.11.0\n")
            gog = GogWorker()
            result = await gog.execute("check calendar")
            assert result["worker"] == "gog"
            assert "status" in result

    @pytest.mark.asyncio
    async def test_execute_not_available(self):
        with patch("subprocess.run", side_effect=FileNotFoundError):
            gog = GogWorker()
            result = await gog.execute("check calendar")
            assert "error" in result


# ── H1 v2: TASK_RESOLUTION_CHAINS ─────────────────────────────────


class TestTaskResolutionChains:
    """Verify TASK_RESOLUTION_CHAINS structure and completeness."""

    def test_chains_exist(self):
        from core.ceo_agent import TASK_RESOLUTION_CHAINS
        assert isinstance(TASK_RESOLUTION_CHAINS, dict)
        assert len(TASK_RESOLUTION_CHAINS) >= 5

    def test_calendar_chain_uses_gog(self):
        from core.ceo_agent import TASK_RESOLUTION_CHAINS
        chain = TASK_RESOLUTION_CHAINS["calendar"]["chain"]
        assert chain[0]["worker"] == "gog"
        assert chain[0]["method"] == "gog_cli"

    def test_email_chain_uses_gog(self):
        from core.ceo_agent import TASK_RESOLUTION_CHAINS
        chain = TASK_RESOLUTION_CHAINS["email"]["chain"]
        assert chain[0]["worker"] == "gog"

    def test_booking_chain_browser_not_first(self):
        from core.ceo_agent import TASK_RESOLUTION_CHAINS
        chain = TASK_RESOLUTION_CHAINS["booking"]["chain"]
        # Browser should NOT be the first method
        assert chain[0]["method"] != "browser"
        # But should be somewhere in the chain
        methods = [s["method"] for s in chain]
        assert "browser" in methods or "partial_assist" in methods

    def test_all_chains_have_timeout(self):
        from core.ceo_agent import TASK_RESOLUTION_CHAINS
        for name, cfg in TASK_RESOLUTION_CHAINS.items():
            for step in cfg["chain"]:
                assert "timeout" in step, f"Chain {name} step missing timeout"

    def test_chain_structure(self):
        from core.ceo_agent import TASK_RESOLUTION_CHAINS
        for name, cfg in TASK_RESOLUTION_CHAINS.items():
            assert "chain" in cfg
            for step in cfg["chain"]:
                assert "method" in step
                assert "worker" in step


class TestTaskRouterGogIntegration:
    """Verify TaskRouter routes calendar/email to gog worker."""

    def test_calendar_routes_to_gog(self):
        from core.task_router import TaskRouter
        router = TaskRouter()
        tasks = router.classify("明天有什麼行程")
        calendar_tasks = [t for t in tasks if t.task_type == "calendar"]
        assert len(calendar_tasks) == 1
        assert calendar_tasks[0].worker == "gog"

    def test_email_routes_to_gog(self):
        from core.task_router import TaskRouter
        router = TaskRouter()
        tasks = router.classify("幫我看一下gmail有沒有新信")
        email_tasks = [t for t in tasks if t.task_type == "email"]
        assert len(email_tasks) == 1
        assert email_tasks[0].worker == "gog"

    def test_schedule_routes_to_gog(self):
        from core.task_router import TaskRouter
        router = TaskRouter()
        tasks = router.classify("今天的會議是幾點")
        assert any(t.worker == "gog" for t in tasks)


class TestReactExecutorGogChain:
    """Verify ReactExecutor FALLBACK_CHAINS include gog."""

    def test_calendar_chain_has_gog(self):
        from core.react_executor import FALLBACK_CHAINS
        assert "calendar" in FALLBACK_CHAINS
        assert "gog" in FALLBACK_CHAINS["calendar"]

    def test_email_chain_has_gog(self):
        from core.react_executor import FALLBACK_CHAINS
        assert "email" in FALLBACK_CHAINS
        assert "gog" in FALLBACK_CHAINS["email"]
