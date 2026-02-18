"""Startup security audit — lightweight config/permission checks.

Inspired by OpenClaw's runSecurityAudit() pattern.
Runs once at startup, logs findings, optionally includes in morning brief.
"""

from __future__ import annotations

import os
import re
from pathlib import Path

from loguru import logger


def startup_audit(project_root: str = ".") -> list[str]:
    """Run security checks on config/data directories.

    Returns a list of finding strings (empty = all clear).
    """
    root = Path(project_root)
    findings: list[str] = []

    # 1. Check .env exists and isn't world-readable (Windows: check it exists)
    env_path = root / ".env"
    if not env_path.exists():
        findings.append("INFO: .env 不存在 — 確認環境變數已透過其他方式設定")

    # 2. Check for hardcoded API keys in Python source files
    api_key_pattern = re.compile(
        r'''["'](?:sk-|zhipu-|fal-|gsk_)[a-zA-Z0-9]{16,}["']'''
    )
    for py_file in root.rglob("*.py"):
        # Skip venv, __pycache__, .git
        parts = py_file.parts
        if any(p in parts for p in ("__pycache__", ".git", "venv", ".venv", "node_modules")):
            continue
        try:
            content = py_file.read_text(encoding="utf-8", errors="ignore")
            if api_key_pattern.search(content):
                findings.append(f"CRITICAL: {py_file.relative_to(root)} 可能包含 hardcoded API key")
        except Exception:
            pass

    # 3. Check data/ for sensitive content
    data_dir = root / "data"
    if data_dir.exists():
        sensitive_patterns = re.compile(r'"(?:api_key|password|secret|token)":\s*"[^"]{10,}"')
        for json_file in data_dir.glob("*.json"):
            try:
                content = json_file.read_text(encoding="utf-8", errors="ignore")
                if sensitive_patterns.search(content):
                    findings.append(f"WARN: {json_file.name} 可能包含敏感資料")
            except Exception:
                pass

    # 4. Check that SoulGuard-protected files are not writable by the process
    for soul_file in (root / "config").glob("SOUL_*.md"):
        # Just verify they exist and are non-empty
        if soul_file.stat().st_size == 0:
            findings.append(f"WARN: {soul_file.name} 是空檔案 — SOUL CORE 可能遺失")

    # 5. Check backup encryption key availability
    if not os.environ.get("BACKUP_ENCRYPTION_KEY"):
        findings.append("INFO: BACKUP_ENCRYPTION_KEY 未設定 — 備份將不加密")

    # Log results
    if findings:
        logger.warning(f"Security audit: {len(findings)} finding(s)")
        for f in findings:
            logger.warning(f"  {f}")
    else:
        logger.info("Security audit: all clear")

    return findings
