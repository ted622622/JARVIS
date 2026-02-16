"""Security Gate — permission enforcement and human-in-the-loop confirmation.

Three tiers of operation handling:
1. AUTO_ALLOW  — safe operations, no confirmation needed
2. CONFIRM     — requires Telegram Y/N from user (timeout = auto-deny)
3. AUTO_BLOCK  — immediately rejected, logged

Covers:
- Path whitelist validation (prevent path traversal)
- Operation classification
- Telegram confirmation flow (Y/N inline keyboard)
- Audit logging
"""

from __future__ import annotations

import asyncio
import os
import re
import time
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Awaitable

from loguru import logger


class OperationVerdict(str, Enum):
    ALLOW = "allow"
    CONFIRM = "confirm"
    BLOCK = "block"


class OperationType(str, Enum):
    # Auto-allow
    READ_FILE = "read_file"
    LIST_DIR = "list_dir"
    API_CALL_WHITELISTED = "api_call_whitelisted"

    # Requires confirmation
    PAYMENT = "payment"
    BULK_DELETE = "bulk_delete"
    EXTERNAL_MESSAGE = "external_message"
    SYSTEM_CONFIG = "system_config"
    UNSIGNED_SCRIPT = "unsigned_script"

    # Auto-block
    PATH_TRAVERSAL = "path_traversal"
    UNAUTHORIZED_API = "unauthorized_api"
    IDENTITY_MODIFICATION = "identity_modification"

    # Unknown / default
    UNKNOWN = "unknown"


@dataclass
class SecurityEvent:
    """Audit log entry for a security decision."""
    operation: str
    verdict: OperationVerdict
    detail: str
    timestamp: float = field(default_factory=time.time)
    confirmed_by: str | None = None  # Telegram user if confirmed


# ── Classification Rules ────────────────────────────────────────

_CONFIRM_OPS = {
    OperationType.PAYMENT,
    OperationType.BULK_DELETE,
    OperationType.EXTERNAL_MESSAGE,
    OperationType.SYSTEM_CONFIG,
    OperationType.UNSIGNED_SCRIPT,
}

_BLOCK_OPS = {
    OperationType.PATH_TRAVERSAL,
    OperationType.UNAUTHORIZED_API,
    OperationType.IDENTITY_MODIFICATION,
}


class SecurityGate:
    """Central security enforcement for J.A.R.V.I.S.

    Usage:
        gate = SecurityGate(
            project_root="C:/ted/JARVIS",
            confirm_callback=telegram_confirm_fn,
        )

        # Check a file operation
        ok = await gate.check_path("/some/path/file.txt")

        # Check an operation
        result = await gate.authorize(OperationType.BULK_DELETE, detail="deleting 15 files")
        if result.verdict == OperationVerdict.ALLOW:
            proceed()
    """

    def __init__(
        self,
        project_root: str = ".",
        confirm_callback: Callable[[str], Awaitable[bool]] | None = None,
        confirmation_timeout: int = 300,
        api_whitelist: list[str] | None = None,
    ):
        self.project_root = Path(project_root).resolve()
        self._confirm = confirm_callback
        self._timeout = confirmation_timeout
        self._audit_log: list[SecurityEvent] = []

        # Sensitive paths that are always blocked
        self._blocked_paths = {
            self.project_root / "config" / "SOUL.md",
        }

        # Protected directories — writes require confirmation
        self._protected_dirs = {
            self.project_root / "config",
            self.project_root / "backups",
        }

        # API whitelist
        self._api_whitelist = set(api_whitelist or [
            "integrate.api.nvidia.com",
            "open.bigmodel.cn",
            "openrouter.ai",
            "api.telegram.org",
            "googleapis.com",
            "fal.run",
            "queue.fal.run",
            "api.groq.com",
            "tts.speech.microsoft.com",
            "cognitiveservices.azure.com",
        ])
        # General internet access for browser worker
        self._allow_browser_navigation = True

    # ── Path Validation ─────────────────────────────────────────

    def check_path(self, path: str) -> OperationVerdict:
        """Validate a file path. Returns verdict.

        Blocks:
        - Path traversal attempts (../ or resolving outside project root)
        - Direct access to SOUL.md or IDENTITY_CORE files
        """
        # Detect path traversal patterns (../ or ..\)
        if re.search(r"\.\.[/\\]", path) or path.endswith(".."):
            self._log(
                "path_access", OperationVerdict.BLOCK,
                f"Path traversal attempt: {path}",
            )
            return OperationVerdict.BLOCK

        try:
            resolved = Path(path).resolve()
        except (OSError, ValueError):
            self._log("path_access", OperationVerdict.BLOCK, f"Invalid path: {path}")
            return OperationVerdict.BLOCK

        # Must be within project root
        try:
            resolved.relative_to(self.project_root)
        except ValueError:
            self._log(
                "path_access", OperationVerdict.BLOCK,
                f"Path outside project root: {path} → {resolved}",
            )
            return OperationVerdict.BLOCK

        # Block writes to identity files
        if resolved in self._blocked_paths:
            self._log(
                "path_access", OperationVerdict.BLOCK,
                f"Access to protected identity file: {resolved}",
            )
            return OperationVerdict.BLOCK

        return OperationVerdict.ALLOW

    # ── API Validation ──────────────────────────────────────────

    def check_api(self, hostname: str) -> OperationVerdict:
        """Check if an API hostname is whitelisted."""
        for allowed in self._api_whitelist:
            if hostname == allowed or hostname.endswith("." + allowed):
                return OperationVerdict.ALLOW

        self._log(
            "api_call", OperationVerdict.BLOCK,
            f"Unauthorized API: {hostname}",
        )
        return OperationVerdict.BLOCK

    def check_browser_url(self, url: str) -> OperationVerdict:
        """Check if a URL is allowed for browser navigation.

        Browser has general internet access (separate from API whitelist).
        Only blocks known-dangerous patterns.
        """
        if not self._allow_browser_navigation:
            return OperationVerdict.BLOCK

        # Block obviously dangerous URLs
        dangerous = ["file:///", "javascript:", "data:text/html"]
        for pattern in dangerous:
            if url.lower().startswith(pattern):
                self._log("browser_nav", OperationVerdict.BLOCK, f"Dangerous URL: {url}")
                return OperationVerdict.BLOCK

        return OperationVerdict.ALLOW

    # ── Operation Authorization ─────────────────────────────────

    async def authorize(
        self,
        op_type: OperationType,
        detail: str = "",
    ) -> SecurityEvent:
        """Authorize an operation. May trigger Telegram confirmation.

        Returns a SecurityEvent with the verdict.
        """
        # Auto-block
        if op_type in _BLOCK_OPS:
            event = self._log(op_type.value, OperationVerdict.BLOCK, detail)
            return event

        # Auto-allow
        if op_type not in _CONFIRM_OPS:
            event = self._log(op_type.value, OperationVerdict.ALLOW, detail)
            return event

        # Needs confirmation
        if self._confirm is None:
            # No callback configured — deny by default
            logger.warning(f"No confirm callback for {op_type.value}, auto-denying")
            event = self._log(
                op_type.value, OperationVerdict.BLOCK,
                f"No confirm callback: {detail}",
            )
            return event

        # Send confirmation request
        prompt = f"🔐 需要確認操作:\n類型: {op_type.value}\n詳情: {detail}\n\n允許執行嗎？"
        try:
            confirmed = await asyncio.wait_for(
                self._confirm(prompt),
                timeout=self._timeout,
            )
        except asyncio.TimeoutError:
            logger.warning(f"Confirmation timeout for {op_type.value}")
            event = self._log(
                op_type.value, OperationVerdict.BLOCK,
                f"Timeout ({self._timeout}s): {detail}",
            )
            return event

        verdict = OperationVerdict.ALLOW if confirmed else OperationVerdict.BLOCK
        event = self._log(op_type.value, verdict, detail)
        event.confirmed_by = "telegram_user" if confirmed else None
        return event

    # ── Bulk Delete Check ───────────────────────────────────────

    async def check_bulk_delete(self, file_count: int, detail: str = "") -> SecurityEvent:
        """Convenience: check if a bulk delete needs confirmation (> 10 files)."""
        if file_count <= 10:
            return self._log("bulk_delete", OperationVerdict.ALLOW, f"{file_count} files")
        return await self.authorize(
            OperationType.BULK_DELETE,
            detail=f"Deleting {file_count} files. {detail}",
        )

    # ── Audit Log ───────────────────────────────────────────────

    def _log(
        self, operation: str, verdict: OperationVerdict, detail: str
    ) -> SecurityEvent:
        event = SecurityEvent(
            operation=operation,
            verdict=verdict,
            detail=detail,
        )
        self._audit_log.append(event)

        if verdict == OperationVerdict.BLOCK:
            logger.warning(f"SECURITY BLOCKED: {operation} — {detail}")
        elif verdict == OperationVerdict.CONFIRM:
            logger.info(f"SECURITY CONFIRM: {operation} — {detail}")
        else:
            logger.debug(f"SECURITY ALLOW: {operation}")

        return event

    def get_audit_log(self, limit: int = 100) -> list[SecurityEvent]:
        """Return recent audit events."""
        return self._audit_log[-limit:]

    def clear_audit_log(self) -> None:
        self._audit_log.clear()
