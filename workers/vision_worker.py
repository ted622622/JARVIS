"""Vision Worker — image analysis via GLM-4V / Gemini.

Handles screenshot analysis, UI element detection,
and general image understanding tasks.
"""

from __future__ import annotations

from typing import Any

from loguru import logger

from core.model_router import ModelRouter


class VisionWorker:
    """Worker for visual analysis tasks.

    Uses GLM-4V (primary) or Gemini Flash (backup) for image understanding.
    Supports: UI element location, screenshot analysis, image QA.

    Usage:
        worker = VisionWorker(model_router=router)
        result = await worker.execute(
            "找到登入按鈕的位置",
            image_url="screenshot.png"
        )
    """

    def __init__(self, model_router: ModelRouter | None = None):
        self.router = model_router
        self.name = "vision"

    async def execute(self, task: str, **kwargs: Any) -> dict[str, Any]:
        """Execute a vision analysis task.

        Args:
            task: description of what to analyze
            **kwargs: image_url (required), format, etc.

        Returns:
            dict with analysis result
        """
        image_url = kwargs.get("image_url")
        if not image_url:
            return {"error": "image_url is required for vision tasks"}

        if not self.router:
            return {"error": "No model router configured"}

        try:
            response = await self.router.vision_analyze(
                image_url=image_url,
                prompt=task,
            )
            return {
                "result": response.content,
                "model": response.model,
                "worker": self.name,
            }
        except Exception as e:
            logger.error(f"VisionWorker failed: {e}")
            return {"error": str(e), "worker": self.name}

    async def locate_element(
        self,
        screenshot_url: str,
        element_description: str,
    ) -> dict[str, Any]:
        """Locate a UI element in a screenshot.

        Returns approximate coordinates for pyautogui integration.
        """
        prompt = (
            f"在這張截圖中找到以下元素：{element_description}\n"
            "回覆格式：x=數字, y=數字（像素座標）\n"
            "如果找不到，回覆「NOT_FOUND」"
        )
        return await self.execute(prompt, image_url=screenshot_url)
