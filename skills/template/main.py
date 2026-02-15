"""Template skill — copy and modify to create new skills.

Every skill must have an async execute() function as its entry point.
"""

from __future__ import annotations

import time
from typing import Any


async def execute(message: str = "hello", **kwargs: Any) -> dict[str, Any]:
    """Skill entry point.

    Args:
        message: text to echo back

    Returns:
        dict with result
    """
    return {
        "result": f"[template] {message}",
        "timestamp": time.time(),
    }
