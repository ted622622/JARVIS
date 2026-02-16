"""Selfie Worker — Clawra self-portrait via fal.ai FLUX Kontext.

Generates consistent character images using a reference anchor image
and the CORE_DNA_PROMPT. Uses fal.ai FLUX Kontext [pro] as primary,
Google Gemini as backup.

Supports two modes:
- direct: close-up selfie (default for cafe, smile, 近照 scenes)
- mirror: full-body mirror selfie (for outfit, 穿搭 scenes)

Cost: ~$0.04/image (FLUX Kontext [pro])
"""

from __future__ import annotations

import re
from typing import Any

from loguru import logger

_MIRROR_PATTERN = re.compile(
    r"outfit|wearing|clothes|dress|穿|穿搭|全身|鏡子|洋裝|裙|外套|大衣|衣服",
    re.IGNORECASE,
)
_DIRECT_PATTERN = re.compile(
    r"cafe|coffee|beach|smile|近照|自拍|臉|咖啡|餐廳|公園|早安|晚安|街|日落|sunset",
    re.IGNORECASE,
)


def detect_mode(user_context: str) -> str:
    """Auto-detect selfie mode from user context."""
    if _DIRECT_PATTERN.search(user_context):
        return "direct"
    if _MIRROR_PATTERN.search(user_context):
        return "mirror"
    return "direct"  # default to close-up


def build_prompt(user_context: str, mode: str) -> str:
    """Build the scene prompt for image generation."""
    if mode == "mirror":
        return (
            f"make a pic of this person, but {user_context}. "
            "the person is taking a mirror selfie"
        )
    # direct mode
    return (
        f"a close-up selfie taken by herself at {user_context}, "
        "direct eye contact with the camera, looking straight into the lens, "
        "eyes centered and clearly visible, not a mirror selfie, "
        "phone held at arm's length, face fully visible"
    )


class SelfieWorker:
    """Worker for Clawra selfie generation.

    Wraps the selfie skill for use by the CEO Agent and Heartbeat.

    Usage:
        worker = SelfieWorker(skill_registry=registry)
        result = await worker.execute("afternoon coffee, warm light")
    """

    def __init__(self, skill_registry: Any = None):
        self.skills = skill_registry
        self.name = "selfie"

    async def execute(self, task: str, **kwargs: Any) -> dict[str, Any]:
        """Generate a Clawra selfie.

        Auto-detects mirror vs direct mode from the task description.

        Args:
            task: scene description (e.g. "holding coffee, afternoon light")
            **kwargs: additional params for the selfie skill

        Returns:
            dict with image URL and metadata
        """
        if not self.skills:
            return {"error": "No skill registry configured"}

        mode = kwargs.pop("mode", None) or detect_mode(task)
        scene = build_prompt(task, mode)
        logger.info(f"SelfieWorker: mode={mode}, scene={scene[:80]}...")

        try:
            result = await self.skills.invoke("selfie", scene=scene, **kwargs)
            return {
                "worker": self.name,
                "mode": mode,
                **result,
            }
        except Exception as e:
            logger.error(f"SelfieWorker failed: {e}")
            return {"error": str(e), "worker": self.name}
