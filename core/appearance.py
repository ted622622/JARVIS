"""Appearance variation for Clawra selfies.

Patch Q: Randomize hairstyle + seasonal outfit + optional Seoul scene
in each selfie prompt. Preferences from SOUL_GROWTH.md bias selections.

Usage:
    builder = AppearanceBuilder()
    snippet = builder.build(
        growth_content="- [selfie-pref] like:hairstyle:ponytail",
        season="winter",
    )
    # → "with a high ponytail, wearing a wool coat with scarf, at a Seoul cafe"
"""

from __future__ import annotations

import random
import re
from datetime import datetime
from typing import Any


# ── Hair ─────────────────────────────────────────────────────────

HAIRSTYLES: list[str] = [
    "with a high ponytail",
    "with long hair down",
    "with loose curly hair",
    "with a messy bun",
    "with braided hair",
    "with side-parted hair",
    "with twintails",
    "with half-up half-down hair",
]

# ── Seasonal outfits (Seoul weather) ─────────────────────────────

OUTFITS: dict[str, list[str]] = {
    "spring": [
        "wearing a light cardigan over a floral blouse",
        "wearing a denim jacket and white tee",
        "wearing a pastel hoodie and pleated skirt",
        "wearing a trench coat with a striped shirt",
    ],
    "summer": [
        "wearing a sleeveless top and shorts",
        "wearing a linen dress with sandals",
        "wearing an oversized tee and denim shorts",
        "wearing a crop top and high-waisted jeans",
    ],
    "autumn": [
        "wearing a knit sweater and plaid skirt",
        "wearing a leather jacket over a turtleneck",
        "wearing an oversized flannel shirt and jeans",
        "wearing a long cardigan with boots",
    ],
    "winter": [
        "wearing a wool coat with a scarf",
        "wearing a padded puffer jacket and beanie",
        "wearing a cozy oversized hoodie and leggings",
        "wearing a long down coat with earmuffs",
    ],
}

# ── Seoul scenes ─────────────────────────────────────────────────

SCENES: list[str] = [
    "in her cozy room with warm lighting",
    "at a Seoul cafe with coffee",
    "by the Han River with city lights",
    "on a Hongdae street with neon signs",
    "near the office in Gangnam",
    "at a Seoul subway station",
    "on a rooftop with the city skyline",
    "in a quiet bookstore corner",
]

# ── Preference tags in SOUL_GROWTH.md ────────────────────────────

_PREF_PATTERN = re.compile(
    r"\[selfie-pref\]\s*(like|dislike):(hairstyle|outfit|scene):([^<\n]+)",
    re.IGNORECASE,
)


def get_seoul_season(dt: datetime | None = None) -> str:
    """Return the current Seoul season based on month.

    Dec-Feb → winter, Mar-May → spring, Jun-Aug → summer, Sep-Nov → autumn.
    """
    month = (dt or datetime.now()).month
    if month in (12, 1, 2):
        return "winter"
    if month <= 5:
        return "spring"
    if month <= 8:
        return "summer"
    return "autumn"


def parse_preferences(growth_content: str) -> dict[str, Any]:
    """Parse [selfie-pref] tags from SOUL_GROWTH.md content.

    Returns:
        {
            "hairstyle_likes": ["ponytail", ...],
            "hairstyle_dislikes": ["twintails", ...],
            "outfit_likes": [...],
            "outfit_dislikes": [...],
            "scene_likes": [...],
            "scene_dislikes": [...],
        }
    """
    prefs: dict[str, list[str]] = {
        "hairstyle_likes": [],
        "hairstyle_dislikes": [],
        "outfit_likes": [],
        "outfit_dislikes": [],
        "scene_likes": [],
        "scene_dislikes": [],
    }
    if not growth_content:
        return prefs

    for match in _PREF_PATTERN.finditer(growth_content):
        sentiment = match.group(1).lower()  # like or dislike
        category = match.group(2).lower()   # hairstyle, outfit, scene
        value = match.group(3).strip()
        key = f"{category}_{'likes' if sentiment == 'like' else 'dislikes'}"
        if key in prefs:
            prefs[key].append(value)

    return prefs


class AppearanceBuilder:
    """Build randomized appearance snippets for selfie prompts.

    Preferences from SOUL_GROWTH bias the selection:
    - liked items get 2x weight
    - disliked items are excluded
    """

    def build(
        self,
        growth_content: str = "",
        season: str | None = None,
        include_scene: bool = True,
    ) -> str:
        """Compose a full appearance string: hairstyle + outfit + (scene).

        Args:
            growth_content: raw SOUL_GROWTH.md content for preference parsing
            season: override season (default: auto-detect from current date)
            include_scene: whether to append a Seoul scene
        """
        season = season or get_seoul_season()
        prefs = parse_preferences(growth_content)

        parts: list[str] = []
        parts.append(self._pick_hairstyle(prefs))
        parts.append(self._pick_outfit(season, prefs))
        if include_scene:
            parts.append(self._pick_scene(prefs))

        return ", ".join(parts)

    def _pick_hairstyle(self, prefs: dict[str, Any]) -> str:
        """Pick a random hairstyle, weighted by preferences."""
        return _weighted_pick(
            HAIRSTYLES,
            likes=prefs.get("hairstyle_likes", []),
            dislikes=prefs.get("hairstyle_dislikes", []),
        )

    def _pick_outfit(self, season: str, prefs: dict[str, Any]) -> str:
        """Pick a random seasonal outfit, weighted by preferences."""
        pool = OUTFITS.get(season, OUTFITS["spring"])
        return _weighted_pick(
            pool,
            likes=prefs.get("outfit_likes", []),
            dislikes=prefs.get("outfit_dislikes", []),
        )

    def _pick_scene(self, prefs: dict[str, Any]) -> str:
        """Pick a random Seoul scene, weighted by preferences."""
        return _weighted_pick(
            SCENES,
            likes=prefs.get("scene_likes", []),
            dislikes=prefs.get("scene_dislikes", []),
        )


def _weighted_pick(
    items: list[str],
    likes: list[str],
    dislikes: list[str],
) -> str:
    """Pick a random item with preference weighting.

    - Items matching a 'like' keyword get weight 2
    - Items matching a 'dislike' keyword are excluded
    - Everything else gets weight 1
    """
    likes_lower = [l.lower() for l in likes]
    dislikes_lower = [d.lower() for d in dislikes]

    candidates: list[str] = []
    weights: list[int] = []

    for item in items:
        item_lower = item.lower()
        # Exclude disliked
        if any(d in item_lower for d in dislikes_lower):
            continue
        candidates.append(item)
        # Boost liked
        if any(l in item_lower for l in likes_lower):
            weights.append(2)
        else:
            weights.append(1)

    if not candidates:
        # All excluded — fall back to full pool
        return random.choice(items)

    return random.choices(candidates, weights=weights, k=1)[0]
