"""Selfie Worker — Clawra self-portrait via fal.ai FLUX Kontext.

Generates consistent character images using a reference anchor image
and the CORE_DNA_PROMPT. Uses fal.ai FLUX Kontext [pro] as primary,
Google Gemini as backup.

Patch T+: 4-framing system replaces old 2-mode (mirror/direct).
- mirror:    full-body mirror selfie (穿搭, outfit, 鏡子)
- full_body: outdoor full-body shot (全身照, 全身)
- medium:    half-body / waist-up (default — cafe, bookstore, etc.)
- closeup:   face close-up (近照, 臉, bokeh)

Cost: ~$0.04/image (FLUX Kontext [pro])
"""

from __future__ import annotations

import random
import re
from typing import Any

from loguru import logger

# ── Framing detection patterns ────────────────────────────────────

_FRAMING_PATTERNS: dict[str, re.Pattern] = {
    "mirror": re.compile(
        r"鏡子|mirror|鏡前|穿搭|outfit|wearing|clothes|dress|洋裝|裙|外套|大衣|衣服",
        re.IGNORECASE,
    ),
    "full_body": re.compile(
        r"全身照|全身|full.?body|站著|走路|standing|walking",
        re.IGNORECASE,
    ),
    "closeup": re.compile(
        r"近照|近拍|close.?up|臉|face|特寫|大頭照|bokeh",
        re.IGNORECASE,
    ),
    "medium": re.compile(
        r"cafe|coffee|咖啡|餐廳|公園|書店|早安|晚安|街|日落|sunset|beach|smile|自拍|拍照",
        re.IGNORECASE,
    ),
}

_FRAMING_PRIORITY = ["mirror", "full_body", "closeup", "medium"]


def detect_framing(context: str) -> str:
    """Auto-detect framing from user context.

    Checks patterns in priority order; first match wins.
    Default: "medium" (half-body shot).
    """
    for framing in _FRAMING_PRIORITY:
        if _FRAMING_PATTERNS[framing].search(context):
            return framing
    return "medium"


# ── Framing prompt templates ─────────────────────────────────────

_FRAMING_PROMPTS: dict[str, str] = {
    "mirror": (
        "make a pic of this person, but {scene}. "
        "the person is taking a mirror selfie, full body visible in the reflection"
    ),
    "full_body": (
        "a full-body photo of this person at {scene}, "
        "taken by a friend from a few steps away, "
        "entire body from head to shoes visible, natural candid pose"
    ),
    "medium": (
        "a half-body selfie taken by herself at {scene}, "
        "waist up visible, direct eye contact with the camera, "
        "phone held at arm's length, face clearly visible"
    ),
    "closeup": (
        "a close-up selfie taken by herself, {scene}, "
        "{expression}, "
        "direct eye contact with the camera, looking straight into the lens, "
        "eyes centered and clearly visible, face filling the frame"
    ),
}

_CLOSEUP_EXPRESSIONS: list[str] = [
    "soft smile with eyes slightly squinted",
    "playful wink with a grin",
    "gentle smile with tilted head",
    "bright wide smile showing teeth",
    "calm serene expression with soft lips",
    "cute pouting lips",
    "laughing naturally with closed eyes",
    "confident smirk with one raised eyebrow",
    "surprised expression with wide eyes",
    "peaceful expression with eyes half-closed",
]


def build_framing_prompt(user_context: str, framing: str) -> str:
    """Build a framing-specific scene prompt for image generation.

    Args:
        user_context: the user's scene description or message
        framing: one of "mirror", "full_body", "medium", "closeup"

    Returns:
        Complete prompt string with {scene} filled in.
    """
    template = _FRAMING_PROMPTS.get(framing, _FRAMING_PROMPTS["medium"])
    result = template.replace("{scene}", user_context)
    if framing == "closeup":
        expression = random.choice(_CLOSEUP_EXPRESSIONS)
        result = result.replace("{expression}", expression)
    return result


# ── Backward compatibility aliases ────────────────────────────────

def detect_mode(user_context: str) -> str:
    """Backward compat: maps framing to old mode names."""
    framing = detect_framing(user_context)
    return "mirror" if framing == "mirror" else "direct"


def build_prompt(user_context: str, mode: str) -> str:
    """Backward compat: maps old mode to framing prompt."""
    framing = "mirror" if mode == "mirror" else "medium"
    return build_framing_prompt(user_context, framing)


# ── Worker class ──────────────────────────────────────────────────


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

        Auto-detects framing from the task description.

        Args:
            task: scene description (e.g. "holding coffee, afternoon light")
            **kwargs: additional params for the selfie skill
                - framing: override framing type
                - mode: legacy alias for framing

        Returns:
            dict with image URL and metadata
        """
        if not self.skills:
            return {"error": "No skill registry configured"}

        # Support both "framing" and legacy "mode" kwarg
        framing = kwargs.pop("framing", None) or kwargs.pop("mode", None)
        if not framing:
            framing = detect_framing(task)

        scene = build_framing_prompt(task, framing)
        logger.info(f"SelfieWorker: framing={framing}, scene={scene[:80]}...")

        try:
            result = await self.skills.invoke(
                "selfie", scene=scene, framing=framing, **kwargs,
            )
            return {
                "worker": self.name,
                "framing": framing,
                "mode": "mirror" if framing == "mirror" else "direct",  # compat
                **result,
            }
        except Exception as e:
            logger.error(f"SelfieWorker failed: {e}")
            return {"error": str(e), "worker": self.name}
