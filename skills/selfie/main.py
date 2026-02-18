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
import re
import time
from dataclasses import dataclass, field
from typing import Any

from loguru import logger

from core.soul import CORE_DNA_PROMPT

# Patch Q: Location keywords — if user specifies a location, skip auto-scene
_LOCATION_PATTERN = re.compile(
    r"在.{1,6}(?:邊|旁|裡|上|下|前|後|附近)|"
    r"漢江|弘大|江南|明洞|梨泰院|咖啡廳|書店|屋頂|公園|海邊|"
    r"辦公室|家裡|房間|地鐵站|餐廳|bar|cafe|rooftop|park|beach|office|room",
    re.IGNORECASE,
)


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
    queue_info: dict[str, Any] | None = None  # status_url, response_url for delayed check


class SelfieSkill:
    """Generate Clawra selfies with FLUX Kontext character consistency.

    Primary: fal.ai FLUX Kontext [pro] — uses reference image for consistency
    Backup: Google Gemini image generation (free tier)
    Verification: Vision model compares against anchor reference

    Patch M: Single request (no retry) + queue API with delayed check.
    """

    CONSISTENCY_THRESHOLD = 0.6
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
        persona: str = "clawra",
        growth_content: str = "",
        season: str | None = None,
        framing: str | None = None,
    ) -> SelfieResult:
        """Generate a Clawra selfie (single attempt, no retry).

        Args:
            scene: Scene description, e.g. "holding coffee, afternoon light"
            verify: Whether to run consistency check
            persona: "clawra" or "jarvis" — affects delayed delivery caption
            growth_content: raw SOUL_GROWTH.md for preference parsing (Patch Q)
            season: override Seoul season (default: auto-detect)
            framing: framing type (mirror/full_body/medium/closeup) — Patch T+

        Returns:
            SelfieResult with image URL and metadata
        """
        # Patch Q + T+: Randomize appearance (hairstyle + outfit + scene)
        from core.appearance import AppearanceBuilder

        builder = AppearanceBuilder()
        # Include auto-scene when user didn't specify a location
        _has_user_location = bool(scene and _LOCATION_PATTERN.search(scene))
        appearance = builder.build(
            growth_content=growth_content,
            season=season,
            include_scene=not _has_user_location,
            framing=framing,
        )

        # Build full prompt: when framing is specified, use framing prompt
        if framing:
            from workers.selfie_worker import build_framing_prompt
            framing_prompt = build_framing_prompt(scene, framing)
            full_prompt = f"{CORE_DNA_PROMPT} {appearance}. {framing_prompt}"
        else:
            full_prompt = f"{CORE_DNA_PROMPT} {appearance}. {scene}"
        logger.info(f"Selfie prompt appearance: {appearance[:80]}")
        result = SelfieResult(attempts=1)

        # 1. Try fal.ai queue (30s wait)
        try:
            img_response = await self._generate_via_fal_queued(full_prompt, max_wait=30.0)
            result.image_url = img_response.url
            result.width = img_response.width
            result.height = img_response.height
            result.cost_usd = self.COST_PER_IMAGE
        except _FalQueueTimeout as e:
            # fal accepted the job but didn't finish in 30s
            result.error = "生成中，稍後補發"
            result.queue_info = {
                "status_url": e.status_url,
                "response_url": e.response_url,
                "persona": persona,
            }
            return result
        except Exception as e:
            logger.warning(f"fal.ai failed: {e}")
            # Try Gemini backup
            try:
                img_url = await self._generate_via_gemini(full_prompt)
                result.image_url = img_url
                result.cost_usd = 0.0
            except Exception as e2:
                result.error = f"fal={e}, gemini={e2}"
                return result

        # Optional consistency check (single, no retry)
        if verify and self.router and result.image_url:
            try:
                score = await self._check_consistency(result.image_url)
                result.consistency_score = score
            except Exception:
                pass

        result.success = True
        return result

    async def _generate_via_fal_queued(self, prompt: str, max_wait: float = 30.0):
        """Submit to fal.ai queue, poll up to max_wait seconds."""
        if not self._fal_key:
            raise ValueError("FAL_KEY not set")

        from clients.fal_client import FalQueueTimeoutError

        fal = self._get_fal_client()
        kwargs: dict[str, Any] = {}
        if self._anchor_url:
            kwargs["image_url"] = self._anchor_url
        try:
            return await fal.generate_image_queued(prompt=prompt, max_wait=max_wait, **kwargs)
        except FalQueueTimeoutError as e:
            raise _FalQueueTimeout(status_url=e.status_url, response_url=e.response_url) from e

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


# ── Internal exception ────────────────────────────────────────────


class _FalQueueTimeout(Exception):
    """Internal: fal.ai queue job timed out but may still complete."""

    def __init__(self, status_url: str = "", response_url: str = ""):
        super().__init__("fal.ai queue timeout")
        self.status_url = status_url
        self.response_url = response_url


# ── Skill entry point ────────────────────────────────────────────

async def execute(scene: str = "casual selfie, natural light", **kwargs: Any) -> dict[str, Any]:
    """Skill entry point for SkillRegistry.invoke()."""
    skill = SelfieSkill(
        fal_api_key=kwargs.get("fal_api_key"),
        anchor_image_url=kwargs.get("anchor_image_url"),
        model_router=kwargs.get("model_router"),
        google_api_key=kwargs.get("google_api_key"),
    )

    result = await skill.generate(
        scene,
        verify=kwargs.get("verify", False),
        persona=kwargs.get("persona", "clawra"),
        growth_content=kwargs.get("growth_content", ""),
        season=kwargs.get("season"),
        framing=kwargs.get("framing"),
    )

    return {
        "image_url": result.image_url,
        "width": result.width,
        "height": result.height,
        "consistency_score": result.consistency_score,
        "attempts": result.attempts,
        "success": result.success,
        "cost_usd": result.cost_usd,
        "error": result.error,
        "queue_info": result.queue_info,
    }
