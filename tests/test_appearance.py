"""Tests for core.appearance — Patch Q: Selfie appearance variation."""

from __future__ import annotations

import re
from datetime import datetime
from unittest.mock import patch

import pytest

from core.appearance import (
    HAIRSTYLES,
    OUTFITS,
    SCENES,
    AppearanceBuilder,
    get_seoul_season,
    parse_preferences,
    _weighted_pick,
)


# ── get_seoul_season ─────────────────────────────────────────────


class TestGetSeoulSeason:
    """Season detection from month."""

    @pytest.mark.parametrize("month,expected", [
        (1, "winter"), (2, "winter"), (12, "winter"),
        (3, "spring"), (4, "spring"), (5, "spring"),
        (6, "summer"), (7, "summer"), (8, "summer"),
        (9, "autumn"), (10, "autumn"), (11, "autumn"),
    ])
    def test_all_months(self, month: int, expected: str):
        dt = datetime(2025, month, 15)
        assert get_seoul_season(dt) == expected

    def test_defaults_to_now(self):
        """Without argument, should return a valid season."""
        season = get_seoul_season()
        assert season in ("spring", "summer", "autumn", "winter")


# ── parse_preferences ────────────────────────────────────────────


class TestParsePreferences:
    """Parsing [selfie-pref] tags from SOUL_GROWTH content."""

    def test_empty_content(self):
        prefs = parse_preferences("")
        assert prefs["hairstyle_likes"] == []
        assert prefs["hairstyle_dislikes"] == []

    def test_single_like(self):
        content = "- [selfie-pref] like:hairstyle:ponytail  <!-- 2025-02-18 -->"
        prefs = parse_preferences(content)
        assert prefs["hairstyle_likes"] == ["ponytail"]

    def test_single_dislike(self):
        content = "- [selfie-pref] dislike:hairstyle:twintails"
        prefs = parse_preferences(content)
        assert prefs["hairstyle_dislikes"] == ["twintails"]

    def test_multiple_preferences(self):
        content = (
            "- [selfie-pref] like:hairstyle:ponytail\n"
            "- [selfie-pref] dislike:outfit:crop top\n"
            "- [selfie-pref] like:scene:cafe\n"
        )
        prefs = parse_preferences(content)
        assert prefs["hairstyle_likes"] == ["ponytail"]
        assert prefs["outfit_dislikes"] == ["crop top"]
        assert prefs["scene_likes"] == ["cafe"]

    def test_non_pref_lines_ignored(self):
        content = (
            "# Clawra 成長記錄\n"
            "- 以後不要太黏  <!-- 2025-01-15 -->\n"
            "- [selfie-pref] like:hairstyle:bun\n"
        )
        prefs = parse_preferences(content)
        assert prefs["hairstyle_likes"] == ["bun"]
        assert len(prefs["hairstyle_dislikes"]) == 0

    def test_case_insensitive_tag(self):
        content = "- [SELFIE-PREF] Like:Hairstyle:curly"
        prefs = parse_preferences(content)
        assert prefs["hairstyle_likes"] == ["curly"]


# ── _weighted_pick ───────────────────────────────────────────────


class TestWeightedPick:
    """Weighted random selection with like/dislike bias."""

    def test_returns_item_from_pool(self):
        items = ["a", "b", "c"]
        result = _weighted_pick(items, likes=[], dislikes=[])
        assert result in items

    def test_dislike_excludes(self):
        items = ["with a high ponytail", "with twintails", "with braided hair"]
        # Exclude twintails
        for _ in range(50):
            result = _weighted_pick(items, likes=[], dislikes=["twintails"])
            assert "twintails" not in result

    def test_like_boosts_probability(self):
        """Liked item should appear more frequently (statistical test)."""
        items = ["with a high ponytail", "with long hair down"]
        counts = {"ponytail": 0, "hair down": 0}
        for _ in range(200):
            result = _weighted_pick(items, likes=["ponytail"], dislikes=[])
            if "ponytail" in result:
                counts["ponytail"] += 1
            else:
                counts["hair down"] += 1
        # ponytail should be ~2x more frequent (weight 2 vs 1)
        assert counts["ponytail"] > counts["hair down"]

    def test_all_excluded_falls_back(self):
        """If all items are disliked, fallback to full pool."""
        items = ["a", "b"]
        result = _weighted_pick(items, likes=[], dislikes=["a", "b"])
        assert result in items


# ── AppearanceBuilder ────────────────────────────────────────────


class TestAppearanceBuilder:
    """Full appearance snippet generation."""

    def setup_method(self):
        self.builder = AppearanceBuilder()

    def test_build_returns_string(self):
        result = self.builder.build(season="winter")
        assert isinstance(result, str)
        assert len(result) > 10

    def test_build_contains_hairstyle(self):
        result = self.builder.build(season="summer", include_scene=False)
        # Should contain at least one hairstyle keyword
        has_hair = any(h.lower() in result.lower() for h in HAIRSTYLES)
        assert has_hair, f"No hairstyle found in: {result}"

    def test_build_winter_outfit(self):
        """Winter build should use winter outfits."""
        result = self.builder.build(season="winter", include_scene=False)
        # Should contain a winter outfit keyword
        winter_keywords = ["wool", "puffer", "hoodie", "down coat"]
        has_winter = any(k in result.lower() for k in winter_keywords)
        assert has_winter, f"No winter outfit in: {result}"

    def test_build_summer_outfit(self):
        result = self.builder.build(season="summer", include_scene=False)
        summer_keywords = ["sleeveless", "linen", "shorts", "crop"]
        has_summer = any(k in result.lower() for k in summer_keywords)
        assert has_summer, f"No summer outfit in: {result}"

    def test_build_with_scene(self):
        result = self.builder.build(season="spring", include_scene=True)
        # 3 parts: hairstyle, outfit, scene
        parts = result.split(", ")
        assert len(parts) >= 3, f"Expected >=3 parts, got {len(parts)}: {result}"

    def test_build_without_scene(self):
        result = self.builder.build(season="spring", include_scene=False)
        # Should NOT contain scene keywords
        scene_keywords = ["room", "cafe", "Han River", "Hongdae", "Gangnam", "subway", "rooftop", "bookstore"]
        has_scene = any(k.lower() in result.lower() for k in scene_keywords)
        assert not has_scene, f"Scene found when include_scene=False: {result}"

    def test_location_pattern_detection(self):
        """Verify the _LOCATION_PATTERN in selfie/main.py detects locations."""
        from skills.selfie.main import _LOCATION_PATTERN
        # Should detect locations
        assert _LOCATION_PATTERN.search("在漢江邊拍張照")
        assert _LOCATION_PATTERN.search("在咖啡廳拍一張")
        assert _LOCATION_PATTERN.search("去弘大拍照")
        assert _LOCATION_PATTERN.search("at the cafe")
        # Should NOT detect in generic requests
        assert not _LOCATION_PATTERN.search("拍張照")
        assert not _LOCATION_PATTERN.search("自拍一下")
        assert not _LOCATION_PATTERN.search("幫我拍個照")

    def test_build_with_preferences(self):
        """Preferences should influence output."""
        growth = (
            "- [selfie-pref] like:hairstyle:ponytail\n"
            "- [selfie-pref] dislike:hairstyle:twintails\n"
        )
        # Run many times — twintails should never appear
        for _ in range(30):
            result = self.builder.build(
                growth_content=growth,
                season="spring",
                include_scene=False,
            )
            assert "twintails" not in result.lower()

    def test_build_auto_season(self):
        """Without explicit season, should auto-detect."""
        result = self.builder.build()
        assert isinstance(result, str)
        assert len(result) > 10

    def test_variation_across_calls(self):
        """Multiple calls should produce different results (most of the time)."""
        results = set()
        for _ in range(20):
            results.add(self.builder.build(season="autumn"))
        # Should have at least 2 different results out of 20 calls
        assert len(results) >= 2, "No variation in 20 calls"


# ── Data integrity ───────────────────────────────────────────────


class TestDataIntegrity:
    """Verify data pools have expected sizes and formats."""

    def test_hairstyle_count(self):
        assert len(HAIRSTYLES) == 8

    def test_outfit_seasons(self):
        assert set(OUTFITS.keys()) == {"spring", "summer", "autumn", "winter"}

    def test_outfit_count_per_season(self):
        for season, outfits in OUTFITS.items():
            assert len(outfits) == 4, f"{season} has {len(outfits)} outfits, expected 4"

    def test_scene_count(self):
        assert len(SCENES) == 8

    def test_hairstyles_are_strings(self):
        for h in HAIRSTYLES:
            assert isinstance(h, str) and len(h) > 5

    def test_outfits_are_strings(self):
        for season, outfits in OUTFITS.items():
            for o in outfits:
                assert isinstance(o, str) and len(o) > 10
