"""Selfie Skill — Generate Clawra selfies via fal.ai FLUX Kontext [pro].

Pipeline:
1. Compose prompt: CORE_DNA_PROMPT + scene description
2. Generate image via fal.ai FLUX Kontext (primary) with reference image
3. Fallback: Google Gemini image generation (backup)
4. Verify consistency against anchor reference image via vision model
5. Return image URL + metadata

Cost: ~$0.04/image (FLUX Kontext [pro])

Usage (as skill):
    result = await execute(scene="holding coffee, afternoon light")

Usage (as class):
    skill = SelfieSkill(fal_api_key="...", anchor_image_url="https://...")
    result = await skill.generate("holding coffee, afternoon light")
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass
from typing import Any

from loguru import logger

from core.soul import CORE_DNA_PROMPT


@dataclass
class SelfieResult:
    """Result of a selfie generation attempt."""
    image_url: str | None = None
    width: int = 0
    height: int = 0
    consistency_score: float = 0.0
    attempts: int = 0
    success: bool = False
    error: str | None = None
    cost_usd: float = 0.0


class SelfieSkill:
    """Generate Clawra selfies with FLUX Kontext character consistency.

    Primary: fal.ai FLUX Kontext [pro] — uses reference image for consistency
    Backup: Google Gemini image generation (free tier)
    Verification: Vision model compares against anchor reference
    """

    CONSISTENCY_THRESHOLD = 0.6
    MAX_RETRIES = 2
    COST_PER_IMAGE = 0.04  # USD for FLUX Kontext [pro]

    def __init__(
        self,
        fal_api_key: str | None = None,
        anchor_image_url: str | None = None,
        model_router: Any = None,
        google_api_key: str | None = None,
    ):
        self._fal_key = fal_api_key or os.environ.get("FAL_KEY", "")
        self._google_key = google_api_key or os.environ.get("GOOGLE_API_KEY", "")
        self._anchor_url = anchor_image_url or os.environ.get(
            "CLAWRA_ANCHOR_URL", ""
        )
        self.router = model_router
        self._fal_client = None

    def _get_fal_client(self):
        """Lazy-init FalClient."""
        if self._fal_client is None:
            from clients.fal_client import FalClient
            self._fal_client = FalClient(api_key=self._fal_key)
        return self._fal_client

    async def generate(
        self,
        scene: str,
        *,
        verify: bool = True,
    ) -> SelfieResult:
        """Generate a Clawra selfie.

        Args:
            scene: Scene description, e.g. "holding coffee, afternoon light"
            verify: Whether to run consistency check

        Returns:
            SelfieResult with image URL and metadata
        """
        full_prompt = f"{CORE_DNA_PROMPT} {scene}"
        result = SelfieResult()

        for attempt in range(1, self.MAX_RETRIES + 2):
            result.attempts = attempt

            try:
                img_response = await self._generate_via_fal(full_prompt)
                result.image_url = img_response.url
                result.width = img_response.width
                result.height = img_response.height
                result.cost_usd = self.COST_PER_IMAGE
            except Exception as e:
                logger.warning(f"fal.ai generation failed (attempt {attempt}): {e}")
                # Try Gemini backup
                try:
                    img_url = await self._generate_via_gemini(full_prompt)
                    result.image_url = img_url
                    result.cost_usd = 0.0  # Gemini free tier
                except Exception as e2:
                    logger.warning(f"Gemini backup also failed: {e2}")
                    result.error = f"All generators failed: fal={e}, gemini={e2}"
                    if attempt > self.MAX_RETRIES:
                        return result
                    continue

            # Consistency check (optional)
            if verify and self.router and result.image_url:
                try:
                    score = await self._check_consistency(result.image_url)
                    result.consistency_score = score
                    if score >= self.CONSISTENCY_THRESHOLD:
                        result.success = True
                        return result
                    else:
                        logger.warning(
                            f"Consistency {score:.2f} < {self.CONSISTENCY_THRESHOLD}, "
                            f"retry {attempt}/{self.MAX_RETRIES + 1}"
                        )
                        if attempt > self.MAX_RETRIES:
                            result.error = f"Consistency too low: {score:.2f}"
                            return result
                        continue
                except Exception as e:
                    logger.warning(f"Consistency check error: {e}, accepting image")
                    result.success = True
                    return result
            else:
                result.success = True
                return result

        return result

    async def _generate_via_fal(self, prompt: str):
        """Generate image via fal.ai FLUX Kontext [pro]."""
        if not self._fal_key:
            raise ValueError("FAL_KEY not set")

        fal = self._get_fal_client()

        kwargs: dict[str, Any] = {}
        if self._anchor_url:
            kwargs["image_url"] = self._anchor_url

        return await fal.generate_image(prompt=prompt, **kwargs)

    async def _generate_via_gemini(self, prompt: str) -> str:
        """Backup: generate image via Google Gemini."""
        if not self._google_key:
            raise ValueError("GOOGLE_API_KEY not set")

        import asyncio

        import google.generativeai as genai
        genai.configure(api_key=self._google_key)

        def _call():
            model = genai.GenerativeModel("gemini-2.0-flash-exp")
            response = model.generate_content(
                f"Generate a photo: {prompt}",
                generation_config=genai.GenerationConfig(
                    response_mime_type="image/png",
                ),
            )
            for part in response.parts:
                if hasattr(part, "inline_data") and part.inline_data:
                    # Save locally and return a file path as URL
                    import tempfile
                    with tempfile.NamedTemporaryFile(
                        suffix=".png", delete=False, dir="./data/selfies"
                    ) as f:
                        f.write(part.inline_data.data)
                        return f.name
            raise ValueError("No image in Gemini response")

        os.makedirs("./data/selfies", exist_ok=True)
        return await asyncio.get_event_loop().run_in_executor(None, _call)

    async def _check_consistency(self, image_url: str) -> float:
        """Compare generated image against anchor using vision model."""
        if not self.router or not self._anchor_url:
            return 1.0

        from clients.base_client import ChatMessage
        from core.model_router import ModelRole

        messages = [
            ChatMessage(
                role="user",
                content=[
                    {"type": "text", "text": (
                        "Compare these two images of a person. "
                        "Rate visual similarity 0.0-1.0 (1.0 = same person). "
                        "Focus on: face, eyes, smile, appearance. "
                        "Reply with ONLY a number."
                    )},
                    {"type": "image_url", "image_url": {"url": self._anchor_url}},
                    {"type": "image_url", "image_url": {"url": image_url}},
                ],
            )
        ]

        try:
            response = await self.router.chat(
                messages, role=ModelRole.VISION, max_tokens=10,
            )
            return max(0.0, min(1.0, float(response.content.strip())))
        except (ValueError, TypeError):
            return 0.5


# ── Skill entry point ────────────────────────────────────────────

async def execute(scene: str = "casual selfie, natural light", **kwargs: Any) -> dict[str, Any]:
    """Skill entry point for SkillRegistry.invoke()."""
    skill = SelfieSkill(
        fal_api_key=kwargs.get("fal_api_key"),
        anchor_image_url=kwargs.get("anchor_image_url"),
        model_router=kwargs.get("model_router"),
        google_api_key=kwargs.get("google_api_key"),
    )

    result = await skill.generate(scene, verify=kwargs.get("verify", False))

    return {
        "image_url": result.image_url,
        "width": result.width,
        "height": result.height,
        "consistency_score": result.consistency_score,
        "attempts": result.attempts,
        "success": result.success,
        "cost_usd": result.cost_usd,
        "error": result.error,
    }
