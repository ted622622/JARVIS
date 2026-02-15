"""Tests for Security Gate.

Run: pytest tests/test_security.py -v
"""

from __future__ import annotations

import asyncio

import pytest

from core.security_gate import (
    OperationType,
    OperationVerdict,
    SecurityGate,
)


@pytest.fixture
def gate(tmp_path) -> SecurityGate:
    """Create a SecurityGate rooted at a temp directory."""
    # Create SOUL.md to test protection
    config_dir = tmp_path / "config"
    config_dir.mkdir()
    (config_dir / "SOUL.md").write_text("identity")

    return SecurityGate(project_root=str(tmp_path))


# ── Path Validation ─────────────────────────────────────────────


class TestPathValidation:
    def test_normal_path_allowed(self, gate, tmp_path):
        test_file = tmp_path / "data" / "test.txt"
        test_file.parent.mkdir(parents=True, exist_ok=True)
        test_file.write_text("test")
        assert gate.check_path(str(test_file)) == OperationVerdict.ALLOW

    def test_path_traversal_blocked(self, gate):
        assert gate.check_path("../../../etc/passwd") == OperationVerdict.BLOCK

    def test_path_traversal_backslash_blocked(self, gate):
        assert gate.check_path("..\\..\\secret") == OperationVerdict.BLOCK

    def test_path_outside_project_blocked(self, gate):
        assert gate.check_path("C:\\Windows\\System32\\cmd.exe") == OperationVerdict.BLOCK

    def test_soul_md_blocked(self, gate, tmp_path):
        soul_path = str(tmp_path / "config" / "SOUL.md")
        assert gate.check_path(soul_path) == OperationVerdict.BLOCK

    def test_normal_config_allowed(self, gate, tmp_path):
        cfg_path = str(tmp_path / "config" / "config.yaml")
        assert gate.check_path(cfg_path) == OperationVerdict.ALLOW


# ── API Whitelist ───────────────────────────────────────────────


class TestAPIWhitelist:
    def test_nvidia_allowed(self, gate):
        assert gate.check_api("integrate.api.nvidia.com") == OperationVerdict.ALLOW

    def test_zhipu_allowed(self, gate):
        assert gate.check_api("open.bigmodel.cn") == OperationVerdict.ALLOW

    def test_openrouter_allowed(self, gate):
        assert gate.check_api("openrouter.ai") == OperationVerdict.ALLOW

    def test_telegram_allowed(self, gate):
        assert gate.check_api("api.telegram.org") == OperationVerdict.ALLOW

    def test_unknown_api_blocked(self, gate):
        assert gate.check_api("evil-api.example.com") == OperationVerdict.BLOCK

    def test_subdomain_of_whitelisted_allowed(self, gate):
        assert gate.check_api("maps.googleapis.com") == OperationVerdict.ALLOW


# ── Operation Authorization ─────────────────────────────────────


class TestAuthorization:
    @pytest.mark.asyncio
    async def test_auto_block_operations(self, gate):
        event = await gate.authorize(OperationType.PATH_TRAVERSAL, "attempted ../")
        assert event.verdict == OperationVerdict.BLOCK

        event = await gate.authorize(OperationType.UNAUTHORIZED_API, "evil.com")
        assert event.verdict == OperationVerdict.BLOCK

        event = await gate.authorize(OperationType.IDENTITY_MODIFICATION, "change SOUL.md")
        assert event.verdict == OperationVerdict.BLOCK

    @pytest.mark.asyncio
    async def test_auto_allow_safe_operations(self, gate):
        event = await gate.authorize(OperationType.READ_FILE, "reading test.txt")
        assert event.verdict == OperationVerdict.ALLOW

    @pytest.mark.asyncio
    async def test_confirm_op_denied_without_callback(self, gate):
        """Without Telegram callback, confirm-needed ops are auto-denied."""
        event = await gate.authorize(OperationType.PAYMENT, "pay $100")
        assert event.verdict == OperationVerdict.BLOCK

    @pytest.mark.asyncio
    async def test_confirm_op_approved_via_callback(self, tmp_path):
        async def always_approve(prompt: str) -> bool:
            return True

        gate = SecurityGate(
            project_root=str(tmp_path),
            confirm_callback=always_approve,
        )
        event = await gate.authorize(OperationType.PAYMENT, "pay $5")
        assert event.verdict == OperationVerdict.ALLOW
        assert event.confirmed_by == "telegram_user"

    @pytest.mark.asyncio
    async def test_confirm_op_rejected_via_callback(self, tmp_path):
        async def always_reject(prompt: str) -> bool:
            return False

        gate = SecurityGate(
            project_root=str(tmp_path),
            confirm_callback=always_reject,
        )
        event = await gate.authorize(OperationType.BULK_DELETE, "delete 50 files")
        assert event.verdict == OperationVerdict.BLOCK

    @pytest.mark.asyncio
    async def test_confirm_timeout_auto_denies(self, tmp_path):
        async def slow_confirm(prompt: str) -> bool:
            await asyncio.sleep(10)
            return True

        gate = SecurityGate(
            project_root=str(tmp_path),
            confirm_callback=slow_confirm,
            confirmation_timeout=1,  # 1 second timeout
        )
        event = await gate.authorize(OperationType.SYSTEM_CONFIG, "modify env vars")
        assert event.verdict == OperationVerdict.BLOCK


# ── Bulk Delete ─────────────────────────────────────────────────


class TestBulkDelete:
    @pytest.mark.asyncio
    async def test_small_delete_allowed(self, gate):
        event = await gate.check_bulk_delete(5, "cleaning temp files")
        assert event.verdict == OperationVerdict.ALLOW

    @pytest.mark.asyncio
    async def test_large_delete_needs_confirm(self, gate):
        """Without callback, large delete is blocked."""
        event = await gate.check_bulk_delete(15, "removing old logs")
        assert event.verdict == OperationVerdict.BLOCK


# ── Audit Log ───────────────────────────────────────────────────


class TestAuditLog:
    @pytest.mark.asyncio
    async def test_events_logged(self, gate):
        gate.check_path("../hack")
        gate.check_api("evil.com")
        await gate.authorize(OperationType.READ_FILE, "test")

        log = gate.get_audit_log()
        assert len(log) == 3
        assert log[0].verdict == OperationVerdict.BLOCK
        assert log[1].verdict == OperationVerdict.BLOCK
        assert log[2].verdict == OperationVerdict.ALLOW

    @pytest.mark.asyncio
    async def test_clear_audit_log(self, gate):
        gate.check_path("../hack")
        gate.clear_audit_log()
        assert len(gate.get_audit_log()) == 0
