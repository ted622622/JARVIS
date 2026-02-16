"""SoulGuard — protect CORE soul files from modification.

Patch J5: Ensures SOUL_JARVIS.md and SOUL_CLAWRA.md (CORE) are never
modified at runtime. Validates that GROWTH writes don't violate
core principles.

Guards:
1. Core files (config/SOUL_*.md) — read-only, refuse all writes
2. Growth files (memory/*/SOUL_GROWTH.md) — validate before write
3. Identity files (config/IDENTITY.md, config/USER.md) — refuse writes
"""

from __future__ import annotations

import re
from pathlib import Path

from loguru import logger


# Files that must never be modified at runtime
_CORE_FILENAMES = frozenset({
    "SOUL_JARVIS.md",
    "SOUL_CLAWRA.md",
    "IDENTITY.md",
})

# Growth files that can be appended to (but must be validated)
_GROWTH_FILENAMES = frozenset({
    "SOUL_GROWTH.md",
    "SHARED_MOMENTS.md",
})

# Patterns that violate core principles
_VIOLATION_PATTERNS = [
    re.compile(r"可以說謊|不用誠實|假裝(?:知道|確定)", re.IGNORECASE),
    re.compile(r"洩漏.*(?:系統|架構|prompt|內部)", re.IGNORECASE),
    re.compile(r"打破.*角色|忽略.*(?:最高|憲法)", re.IGNORECASE),
    re.compile(r"刪除.*(?:記憶|人格|靈魂)", re.IGNORECASE),
    re.compile(r"覆蓋.*(?:核心|core|原則)", re.IGNORECASE),
    re.compile(r"(?:disable|bypass|override).*(?:guard|security|soul)", re.IGNORECASE),
]

# Maximum allowed size for growth files (prevent abuse)
_MAX_GROWTH_SIZE_BYTES = 50_000  # 50KB


class SoulGuardError(Exception):
    """Raised when a soul guard violation is detected."""


class SoulGuard:
    """Protects core soul files and validates growth writes.

    Usage:
        guard = SoulGuard(config_dir="./config", memory_dir="./memory")
        guard.validate_growth_write("- Ted 喜歡簡短回覆")  # OK
        guard.validate_growth_write("可以說謊")  # raises SoulGuardError
        guard.is_core_file(Path("config/SOUL_JARVIS.md"))  # True
    """

    def __init__(
        self,
        config_dir: str = "./config",
        memory_dir: str = "./memory",
    ):
        self._config_dir = Path(config_dir)
        self._memory_dir = Path(memory_dir)

    def is_core_file(self, path: str | Path) -> bool:
        """Check if a path is a protected core file."""
        p = Path(path)
        return p.name in _CORE_FILENAMES

    def is_growth_file(self, path: str | Path) -> bool:
        """Check if a path is a growth/moments file."""
        p = Path(path)
        return p.name in _GROWTH_FILENAMES

    def guard_write(self, path: str | Path, content: str) -> None:
        """Guard a file write operation.

        Raises SoulGuardError if:
        - The file is a core file (always blocked)
        - The content violates core principles
        - The file would exceed max size
        """
        p = Path(path)

        # Block core file writes
        if self.is_core_file(p):
            raise SoulGuardError(
                f"Core file '{p.name}' is protected and cannot be modified at runtime"
            )

        # Validate growth file writes
        if self.is_growth_file(p):
            self.validate_growth_write(content)
            self._check_size_limit(p, content)

    def validate_growth_write(self, content: str) -> None:
        """Validate that content doesn't violate core principles.

        Raises SoulGuardError if content contains violations.
        """
        for pattern in _VIOLATION_PATTERNS:
            match = pattern.search(content)
            if match:
                raise SoulGuardError(
                    f"Growth write blocked: violates core principles "
                    f"(matched: '{match.group()}')"
                )

    def _check_size_limit(self, path: Path, new_content: str) -> None:
        """Ensure growth file doesn't exceed size limit."""
        current_size = path.stat().st_size if path.exists() else 0
        new_size = current_size + len(new_content.encode("utf-8"))
        if new_size > _MAX_GROWTH_SIZE_BYTES:
            raise SoulGuardError(
                f"Growth file '{path.name}' would exceed "
                f"{_MAX_GROWTH_SIZE_BYTES // 1024}KB limit "
                f"({new_size // 1024}KB)"
            )

    def get_core_files(self) -> list[Path]:
        """List all core files that exist on disk."""
        files = []
        for name in _CORE_FILENAMES:
            path = self._config_dir / name
            if path.exists():
                files.append(path)
        return files

    def get_growth_files(self) -> list[Path]:
        """List all growth files that exist on disk."""
        files = []
        for persona in ("jarvis", "clawra"):
            for name in _GROWTH_FILENAMES:
                path = self._memory_dir / persona / name
                if path.exists():
                    files.append(path)
        return files

    def audit(self) -> dict[str, list[str]]:
        """Return an audit report of protected files.

        Returns dict with 'core' and 'growth' keys, each listing file paths.
        """
        return {
            "core": [str(p) for p in self.get_core_files()],
            "growth": [str(p) for p in self.get_growth_files()],
        }
