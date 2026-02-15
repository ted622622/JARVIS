"""Selfie Worker — Clawra self-portrait via fal.ai FLUX Kontext.

Generates consistent character images using a reference anchor image
and the CORE_DNA_PROMPT. Uses fal.ai FLUX Kontext [pro] as primary,
Google Gemini as backup.

Cost: ~$0.04/image (FLUX Kontext [pro])
"""

from __future__ import annotations

from typing import Any

from loguru import logger


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

        Args:
            task: scene description (e.g. "holding coffee, afternoon light")
            **kwargs: additional params for the selfie skill

        Returns:
            dict with image URL and metadata
        """
        if not self.skills:
            return {"error": "No skill registry configured"}

        try:
            result = await self.skills.invoke("selfie", scene=task, **kwargs)
            return {
                "worker": self.name,
                **result,
            }
        except Exception as e:
            logger.error(f"SelfieWorker failed: {e}")
            return {"error": str(e), "worker": self.name}
