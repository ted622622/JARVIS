"""gog CLI 封裝 — Google Workspace 操作統一入口.

透過 steipete/gogcli 操作 Gmail / Calendar / Drive / Contacts。
為什麼不自己寫 Google API？
  - gog 處理 OAuth token refresh、pagination、error handling
  - 社群維護，比自己寫更穩定
  - ~120 行取代 500+ 行

風險緩解：所有呼叫包在 _run_gog()，gog 壞了只改一處。
"""

from __future__ import annotations

import json
import os
import subprocess
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

from loguru import logger

# Default gog binary — project-local first, then PATH
_GOG_BIN = str(Path(__file__).resolve().parent.parent / "bin" / "gog.exe")
if not Path(_GOG_BIN).exists():
    _GOG_BIN = "gog"  # fall back to PATH


class GogWorker:
    """透過 gog CLI 操作 Google Workspace。

    Usage:
        gog = GogWorker()
        events = gog.get_today_events()
    """

    def __init__(
        self,
        account: str | None = None,
        gog_bin: str = _GOG_BIN,
        timeout: int = 30,
    ):
        self.account = account or os.getenv("GOG_ACCOUNT", "")
        self.gog_bin = gog_bin
        self.default_timeout = timeout
        self.name = "gog"
        self._available = self._verify_installed()

    @property
    def is_available(self) -> bool:
        return self._available

    def _verify_installed(self) -> bool:
        """Check gog CLI is installed and reachable."""
        try:
            result = subprocess.run(
                [self.gog_bin, "--version"],
                capture_output=True, text=True, timeout=5,
            )
            if result.returncode == 0:
                ver = result.stdout.strip()
                logger.info(f"gog CLI ready: {ver}")
                return True
            logger.warning(f"gog CLI exit code {result.returncode}: {result.stderr}")
            return False
        except FileNotFoundError:
            logger.warning(f"gog CLI not found at {self.gog_bin}")
            return False
        except Exception as e:
            logger.warning(f"gog CLI check failed: {e}")
            return False

    def _run_gog(
        self,
        args: list[str],
        timeout: int | None = None,
    ) -> dict[str, Any]:
        """統一呼叫入口 — gog 壞了只需改這裡。"""
        if not self._available:
            return {"success": False, "error": "gog CLI not installed"}

        cmd = [self.gog_bin] + args
        if self.account:
            cmd += ["--account", self.account]
        cmd += ["--json", "--no-input"]

        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=timeout or self.default_timeout,
            )
            if result.returncode == 0:
                data = {}
                if result.stdout.strip():
                    try:
                        data = json.loads(result.stdout)
                    except json.JSONDecodeError:
                        data = {"raw": result.stdout.strip()}
                return {"success": True, "data": data}
            return {
                "success": False,
                "error": result.stderr.strip() or f"exit code {result.returncode}",
            }
        except subprocess.TimeoutExpired:
            return {"success": False, "error": "gog call timeout"}
        except FileNotFoundError:
            self._available = False
            return {"success": False, "error": "gog CLI not found"}

    # ── Calendar ──────────────────────────────────────────────────

    def get_today_events(self) -> list[dict]:
        """Get today's calendar events."""
        today = datetime.now().strftime("%Y-%m-%dT00:00:00")
        end = datetime.now().strftime("%Y-%m-%dT23:59:59")
        result = self._run_gog([
            "calendar", "events", "primary",
            "--from", today, "--to", end,
        ])
        return result.get("data", []) if result["success"] else []

    def get_events_for_date(self, date: datetime) -> list[dict]:
        """Get events for a specific date."""
        start = date.strftime("%Y-%m-%dT00:00:00")
        end = date.strftime("%Y-%m-%dT23:59:59")
        result = self._run_gog([
            "calendar", "events", "primary",
            "--from", start, "--to", end,
        ])
        return result.get("data", []) if result["success"] else []

    def get_upcoming_events(self, minutes: int = 60) -> list[dict]:
        """Get events in the next N minutes."""
        now = datetime.now()
        end = now + timedelta(minutes=minutes)
        result = self._run_gog([
            "calendar", "events", "primary",
            "--from", now.isoformat(),
            "--to", end.isoformat(),
        ])
        return result.get("data", []) if result["success"] else []

    def create_event(
        self,
        title: str,
        start_time: datetime,
        duration_minutes: int = 60,
        location: str = "",
    ) -> dict[str, Any]:
        """Create a calendar event."""
        end_time = start_time + timedelta(minutes=duration_minutes)
        cmd = [
            "calendar", "create", "primary",
            "--summary", title,
            "--from", start_time.isoformat(),
            "--to", end_time.isoformat(),
        ]
        if location:
            cmd += ["--location", location]
        return self._run_gog(cmd)

    # ── Gmail ─────────────────────────────────────────────────────

    def search_inbox(
        self, query: str = "newer_than:1d", max_results: int = 10,
    ) -> list[dict]:
        """Search Gmail inbox."""
        result = self._run_gog([
            "gmail", "search", query, "--max", str(max_results),
        ])
        return result.get("data", []) if result["success"] else []

    def send_email(
        self, to: str, subject: str, body: str,
    ) -> dict[str, Any]:
        """Send an email."""
        return self._run_gog([
            "gmail", "send",
            "--to", to,
            "--subject", subject,
            "--body", body,
        ])

    # ── Drive ─────────────────────────────────────────────────────

    def search_drive(
        self, query: str, max_results: int = 10,
    ) -> list[dict]:
        """Search Google Drive."""
        result = self._run_gog([
            "drive", "search", query, "--max", str(max_results),
        ])
        return result.get("data", []) if result["success"] else []

    # ── Generic execute (worker interface) ────────────────────────

    async def execute(self, task: str, **kwargs: Any) -> dict[str, Any]:
        """Worker interface — dispatch by task description.

        Parses the task string to route to the appropriate specific method.
        """
        if not self._available:
            return {"error": "gog CLI not installed", "worker": self.name}

        task_lower = task.lower()

        # ── Calendar ──
        if any(kw in task_lower for kw in (
            "行事曆", "日程", "calendar", "schedule", "會議", "meeting",
            "約", "預約", "appointment", "event", "agenda",
        )):
            if any(kw in task_lower for kw in ("新增", "加入", "create", "add", "建立")):
                return {
                    "content": "請使用 create_event() 方法建立事件（需要標題和時間）。",
                    "worker": self.name,
                }
            if "明天" in task_lower:
                tomorrow = datetime.now() + timedelta(days=1)
                events = self.get_events_for_date(tomorrow)
            elif any(kw in task_lower for kw in ("upcoming", "接下來", "下一個")):
                events = self.get_upcoming_events(minutes=120)
            else:
                events = self.get_today_events()
            return {"content": json.dumps(events, ensure_ascii=False), "worker": self.name}

        # ── Email ──
        if any(kw in task_lower for kw in (
            "email", "信", "mail", "寄", "發信", "收件", "inbox", "gmail",
        )):
            if any(kw in task_lower for kw in ("寄", "發", "send")):
                return {
                    "content": "請使用 send_email(to, subject, body) 方法發信。",
                    "worker": self.name,
                }
            query = kwargs.get("query", "newer_than:1d")
            results = self.search_inbox(query=query)
            return {"content": json.dumps(results, ensure_ascii=False), "worker": self.name}

        # ── Drive ──
        if any(kw in task_lower for kw in (
            "drive", "雲端", "檔案", "file", "document", "文件",
        )):
            query = kwargs.get("query", task[:50])
            results = self.search_drive(query=query)
            return {"content": json.dumps(results, ensure_ascii=False), "worker": self.name}

        # ── Fallback: try calendar (most common use case) ──
        events = self.get_today_events()
        if events:
            return {"content": json.dumps(events, ensure_ascii=False), "worker": self.name}

        return {
            "content": f"無法辨識任務類型: {task[:80]}",
            "worker": self.name,
        }
