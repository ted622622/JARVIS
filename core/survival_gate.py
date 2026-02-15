"""SurvivalGate — health monitoring and self-protection.

Runs every 6 hours (triggered by Heartbeat scheduler).
Checks:
1. API connectivity (NVIDIA, Zhipu, OpenRouter)
2. Quota/balance monitoring (Zhipu image credits, OpenRouter balance)
3. MemOS token saving rate (via TokenSavingTracker)
4. Backup integrity
5. Disk and memory usage
"""

from __future__ import annotations

import os
import shutil
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from loguru import logger


@dataclass
class CheckResult:
    """Single check result."""
    name: str
    status: str  # "ok", "warning", "critical"
    detail: str = ""


@dataclass
class HealthReport:
    """Aggregated health report from SurvivalGate."""
    checks: list[CheckResult] = field(default_factory=list)
    alerts: list[str] = field(default_factory=list)
    timestamp: float = field(default_factory=time.time)

    def add(self, check: CheckResult) -> None:
        self.checks.append(check)

    def alert(self, message: str) -> None:
        self.alerts.append(message)

    @property
    def has_alerts(self) -> bool:
        return len(self.alerts) > 0

    def format(self) -> str:
        """Format report for Telegram notification."""
        lines = ["🏥 *Health Report*", ""]

        for check in self.checks:
            icon = {"ok": "✅", "warning": "⚠️", "critical": "🔴"}.get(check.status, "❓")
            line = f"{icon} {check.name}: {check.detail}" if check.detail else f"{icon} {check.name}"
            lines.append(line)

        if self.alerts:
            lines.append("")
            lines.append("*Alerts:*")
            for a in self.alerts:
                lines.append(f"  {a}")

        return "\n".join(lines)


class SurvivalGate:
    """Health monitoring — called by Heartbeat every 6 hours.

    Usage:
        gate = SurvivalGate(
            model_router=router,
            token_tracker=tracker,
            backup_dir="./backups",
        )
        report = await gate.full_check()
        if report.has_alerts:
            await telegram.send(report.format())
    """

    def __init__(
        self,
        model_router: Any = None,
        token_tracker: Any = None,
        fal_client: Any = None,
        backup_dir: str = "./backups",
        project_root: str = ".",
        disk_alert_gb: float = 2.0,
        memory_alert_percent: float = 90.0,
    ):
        self.router = model_router
        self.tracker = token_tracker
        self.fal_client = fal_client
        self.backup_dir = Path(backup_dir)
        self.project_root = Path(project_root)
        self.disk_alert_gb = disk_alert_gb
        self.memory_alert_pct = memory_alert_percent

    async def full_check(self) -> HealthReport:
        """Execute all health checks. Returns aggregated report."""
        report = HealthReport()

        # 1. API connectivity
        await self._check_apis(report)

        # 2. Quota monitoring
        await self._check_quotas(report)

        # 3. Token saving rate
        await self._check_token_savings(report)

        # 4. Backup integrity
        self._check_backup(report)

        # 5. Disk usage
        self._check_disk(report)

        # 6. Memory usage
        self._check_memory(report)

        logger.info(
            f"SurvivalGate: {len(report.checks)} checks, "
            f"{len(report.alerts)} alerts"
        )
        return report

    # ── Individual Checks ───────────────────────────────────────

    async def _check_apis(self, report: HealthReport) -> None:
        """Ping all API providers."""
        if not self.router:
            report.add(CheckResult("API connectivity", "warning", "No router configured"))
            return

        health = await self.router.health_check_all()
        for provider, is_healthy in health.items():
            if is_healthy:
                report.add(CheckResult(f"API: {provider}", "ok", "reachable"))
            else:
                report.add(CheckResult(f"API: {provider}", "critical", "UNREACHABLE"))
                report.alert(f"🔴 {provider} API is unreachable")

    async def _check_quotas(self, report: HealthReport) -> None:
        """Check API quotas and balances."""
        if not self.router:
            return

        # OpenRouter balance
        try:
            balance = await self.router.openrouter.get_remaining_credits()
            if balance >= 0:
                if balance < 1.0:
                    report.add(CheckResult("OpenRouter balance", "warning", f"${balance:.2f}"))
                    report.alert(f"⚠️ OpenRouter 餘額不足 $1.00 (剩 ${balance:.2f})，請充值")
                else:
                    report.add(CheckResult("OpenRouter balance", "ok", f"${balance:.2f}"))
            else:
                report.add(CheckResult("OpenRouter balance", "warning", "Unable to query"))
        except Exception as e:
            report.add(CheckResult("OpenRouter balance", "warning", f"Query failed: {e}"))

        # fal.ai connectivity (balance API not available, check reachability)
        if self.fal_client:
            try:
                fal_ok = await self.fal_client.health_check()
                if fal_ok:
                    report.add(CheckResult("fal.ai", "ok", "reachable"))
                else:
                    report.add(CheckResult("fal.ai", "warning", "unreachable"))
                    report.alert("⚠️ fal.ai API 無法連線，自拍功能暫停")
            except Exception as e:
                report.add(CheckResult("fal.ai", "warning", f"Check failed: {e}"))

    async def _check_token_savings(self, report: HealthReport) -> None:
        """Check MemOS token saving efficiency."""
        if not self.tracker:
            report.add(CheckResult("Token savings", "warning", "No tracker configured"))
            return

        try:
            saving = await self.tracker.daily_report()
            rate_str = saving["avg_saving_rate"]
            calls = saving["total_calls"]

            if calls == 0:
                report.add(CheckResult("Token savings", "ok", "No calls in last 24h"))
            elif saving["alert"]:
                report.add(CheckResult("Token savings", "warning", f"{rate_str} ({calls} calls)"))
                report.alert(f"⚠️ MemOS 節省率跌破 50% ({rate_str})，需排查")
            else:
                report.add(CheckResult("Token savings", "ok", f"{rate_str} ({calls} calls)"))
        except Exception as e:
            report.add(CheckResult("Token savings", "warning", f"Check failed: {e}"))

    def _check_backup(self, report: HealthReport) -> None:
        """Verify most recent backup exists and is recent."""
        if not self.backup_dir.exists():
            report.add(CheckResult("Backup", "warning", "Backup directory missing"))
            report.alert("⚠️ 備份目錄不存在")
            return

        # Find most recent backup file
        backups = sorted(self.backup_dir.glob("**/*"), key=lambda p: p.stat().st_mtime if p.is_file() else 0, reverse=True)
        backup_files = [b for b in backups if b.is_file()]

        if not backup_files:
            report.add(CheckResult("Backup", "warning", "No backup files found"))
            report.alert("⚠️ 找不到任何備份檔案")
            return

        latest = backup_files[0]
        age_hours = (time.time() - latest.stat().st_mtime) / 3600
        size_mb = latest.stat().st_size / (1024 * 1024)

        if age_hours > 48:
            report.add(CheckResult("Backup", "warning", f"Last backup {age_hours:.0f}h ago ({size_mb:.1f}MB)"))
            report.alert(f"⚠️ 最近備份已超過 48 小時 ({age_hours:.0f}h)")
        else:
            report.add(CheckResult("Backup", "ok", f"{age_hours:.0f}h ago ({size_mb:.1f}MB)"))

    def _check_disk(self, report: HealthReport) -> None:
        """Check available disk space."""
        try:
            usage = shutil.disk_usage(str(self.project_root))
            free_gb = usage.free / (1024 ** 3)
            total_gb = usage.total / (1024 ** 3)
            used_pct = (usage.used / usage.total) * 100

            if free_gb < self.disk_alert_gb:
                report.add(CheckResult("Disk", "warning", f"{free_gb:.1f}GB free / {total_gb:.0f}GB total"))
                report.alert(f"⚠️ 磁碟空間不足 {self.disk_alert_gb}GB (剩 {free_gb:.1f}GB)")
            else:
                report.add(CheckResult("Disk", "ok", f"{free_gb:.1f}GB free ({used_pct:.0f}% used)"))
        except Exception as e:
            report.add(CheckResult("Disk", "warning", f"Check failed: {e}"))

    def _check_memory(self, report: HealthReport) -> None:
        """Check process/system memory usage."""
        try:
            import psutil
            mem = psutil.virtual_memory()
            used_pct = mem.percent
            avail_gb = mem.available / (1024 ** 3)

            if used_pct > self.memory_alert_pct:
                report.add(CheckResult("Memory", "warning", f"{used_pct:.0f}% used ({avail_gb:.1f}GB available)"))
                report.alert(f"⚠️ 記憶體使用率 {used_pct:.0f}% 超過閾值 {self.memory_alert_pct:.0f}%")
            else:
                report.add(CheckResult("Memory", "ok", f"{used_pct:.0f}% used ({avail_gb:.1f}GB available)"))
        except ImportError:
            # psutil not installed, skip memory check
            report.add(CheckResult("Memory", "ok", "psutil not available, skipped"))
        except Exception as e:
            report.add(CheckResult("Memory", "warning", f"Check failed: {e}"))
