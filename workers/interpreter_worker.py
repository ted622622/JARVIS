"""Interpreter Worker — OS-level operations via local Python/PowerShell.

Handles file operations, system commands, and automation tasks
that require local OS access.
"""

from __future__ import annotations

import asyncio
import subprocess
from typing import Any

from loguru import logger


class InterpreterWorker:
    """Worker for OS-level operations.

    Usage:
        worker = InterpreterWorker()
        result = await worker.execute("列出桌面上的檔案")
    """

    def __init__(self, security_gate: Any = None):
        self.security = security_gate
        self.name = "interpreter"

    async def execute(self, task: str, **kwargs: Any) -> dict[str, Any]:
        """Execute an OS-level task.

        Args:
            task: command or description to execute
            **kwargs: shell ("powershell"|"python"), timeout, etc.

        Returns:
            dict with output and metadata
        """
        shell = kwargs.get("shell", "powershell")
        timeout = kwargs.get("timeout", 30)
        command = kwargs.get("command", task)

        # Security check
        if self.security:
            verdict = await self.security.authorize(
                operation="unsigned_script",
                detail=command[:200],
            )
            if verdict.action == "BLOCK":
                return {"error": f"Blocked by SecurityGate: {verdict.reason}"}
            if verdict.action == "CONFIRM" and not verdict.approved:
                return {"error": "Operation not approved by user"}

        try:
            result = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: self._run_command(command, shell, timeout),
            )
            return result
        except Exception as e:
            logger.error(f"InterpreterWorker failed: {e}")
            return {"error": str(e), "worker": self.name}

    def _run_command(
        self, command: str, shell: str, timeout: int
    ) -> dict[str, Any]:
        """Run a command in subprocess."""
        if shell == "powershell":
            args = ["powershell", "-NoProfile", "-Command", command]
        elif shell == "python":
            args = ["python", "-c", command]
        else:
            args = command.split()

        proc = subprocess.run(
            args,
            capture_output=True,
            text=True,
            timeout=timeout,
        )

        return {
            "stdout": proc.stdout,
            "stderr": proc.stderr,
            "returncode": proc.returncode,
            "worker": self.name,
        }
