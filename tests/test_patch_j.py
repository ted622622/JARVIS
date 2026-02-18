"""Tests for Patch J — Soul Evolution (J1-J5).

J1: SOUL_CORE + SOUL_GROWTH file split
J2+J3: SoulGrowth — learn from conversations
J4: SharedMemory — shared moments for Clawra
J5: SoulGuard — protect core soul files
"""

from __future__ import annotations

import pytest
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch
import tempfile
import shutil

from core.soul_growth import _LEARN_INTERVAL


# ════════════════════════════════════════════════════════════════
# J1: Soul CORE + GROWTH split
# ════════════════════════════════════════════════════════════════


class TestSoulGrowthLoading:
    """Test that Soul class loads GROWTH files alongside CORE."""

    def setup_method(self):
        self.tmpdir = Path(tempfile.mkdtemp())
        self.config_dir = self.tmpdir / "config"
        self.config_dir.mkdir()
        self.memory_dir = self.tmpdir / "memory"
        self.memory_dir.mkdir()

        # Create CORE soul files
        (self.config_dir / "SOUL_JARVIS.md").write_text(
            "# JARVIS\n\n## 最高憲法\n100% 誠實\n", encoding="utf-8"
        )
        (self.config_dir / "SOUL_CLAWRA.md").write_text(
            "# Clawra\n\n## 最高憲法\n100% 誠實\n", encoding="utf-8"
        )
        (self.config_dir / "USER.md").write_text(
            "# User\n- Name: Ted\n", encoding="utf-8"
        )

    def teardown_method(self):
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_load_without_growth(self):
        from core.soul import Soul
        soul = Soul(config_dir=str(self.config_dir), memory_dir=str(self.memory_dir))
        soul.load()
        assert soul.is_loaded
        prompt = soul.build_system_prompt("jarvis")
        assert "JARVIS" in prompt
        assert "從互動中學到的偏好" not in prompt

    def test_load_with_growth(self):
        from core.soul import Soul
        # Create GROWTH file with content
        jarvis_dir = self.memory_dir / "jarvis"
        jarvis_dir.mkdir()
        (jarvis_dir / "SOUL_GROWTH.md").write_text(
            "# JARVIS 成長記錄\n\n- Ted 喜歡簡短回覆\n- Ted 不喜歡被問需不需要幫忙\n",
            encoding="utf-8",
        )
        soul = Soul(config_dir=str(self.config_dir), memory_dir=str(self.memory_dir))
        soul.load()
        prompt = soul.build_system_prompt("jarvis")
        assert "從互動中學到的偏好" in prompt
        assert "簡短回覆" in prompt

    def test_growth_empty_header_only(self):
        from core.soul import Soul
        jarvis_dir = self.memory_dir / "jarvis"
        jarvis_dir.mkdir()
        (jarvis_dir / "SOUL_GROWTH.md").write_text(
            "# JARVIS 成長記錄\n\n<!-- header -->\n",
            encoding="utf-8",
        )
        soul = Soul(config_dir=str(self.config_dir), memory_dir=str(self.memory_dir))
        soul.load()
        prompt = soul.build_system_prompt("jarvis")
        assert "從互動中學到的偏好" not in prompt

    def test_reload_growth(self):
        from core.soul import Soul
        jarvis_dir = self.memory_dir / "jarvis"
        jarvis_dir.mkdir()
        growth_path = jarvis_dir / "SOUL_GROWTH.md"
        growth_path.write_text("# JARVIS 成長記錄\n\n", encoding="utf-8")

        soul = Soul(config_dir=str(self.config_dir), memory_dir=str(self.memory_dir))
        soul.load()
        assert soul.get_growth_content("jarvis") == ""

        # Simulate SoulGrowth appending
        growth_path.write_text(
            "# JARVIS 成長記錄\n\n- 新學到的偏好\n", encoding="utf-8"
        )
        soul.reload_growth("jarvis")
        assert "新學到的偏好" in soul.get_growth_content("jarvis")

    def test_get_core_content(self):
        from core.soul import Soul
        soul = Soul(config_dir=str(self.config_dir), memory_dir=str(self.memory_dir))
        soul.load()
        core = soul.get_core_content("jarvis")
        assert "JARVIS" in core
        assert "最高憲法" in core

    def test_growth_injected_before_user(self):
        from core.soul import Soul
        jarvis_dir = self.memory_dir / "jarvis"
        jarvis_dir.mkdir()
        (jarvis_dir / "SOUL_GROWTH.md").write_text(
            "# Growth\n\n- 偏好A\n", encoding="utf-8"
        )
        soul = Soul(config_dir=str(self.config_dir), memory_dir=str(self.memory_dir))
        soul.load()
        prompt = soul.build_system_prompt("jarvis")
        growth_pos = prompt.index("偏好A")
        user_pos = prompt.index("Ted")
        assert growth_pos < user_pos, "Growth should appear before USER section"

    def test_clawra_growth_separate(self):
        from core.soul import Soul
        for persona in ("jarvis", "clawra"):
            d = self.memory_dir / persona
            d.mkdir(exist_ok=True)
            (d / "SOUL_GROWTH.md").write_text(
                f"# {persona}\n\n- {persona}_pref\n", encoding="utf-8"
            )
        soul = Soul(config_dir=str(self.config_dir), memory_dir=str(self.memory_dir))
        soul.load()
        jarvis_prompt = soul.build_system_prompt("jarvis")
        clawra_prompt = soul.build_system_prompt("clawra")
        assert "jarvis_pref" in jarvis_prompt
        assert "jarvis_pref" not in clawra_prompt
        assert "clawra_pref" in clawra_prompt
        assert "clawra_pref" not in jarvis_prompt


# ════════════════════════════════════════════════════════════════
# J2+J3: SoulGrowth
# ════════════════════════════════════════════════════════════════


class TestSoulGrowth:
    """Test SoulGrowth learning from conversations."""

    def setup_method(self):
        self.tmpdir = Path(tempfile.mkdtemp())
        self.memory_dir = self.tmpdir / "memory"
        self.memory_dir.mkdir()
        for p in ("jarvis", "clawra"):
            d = self.memory_dir / p
            d.mkdir()
            (d / "SOUL_GROWTH.md").write_text(f"# {p} Growth\n\n", encoding="utf-8")

    def teardown_method(self):
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_no_learn_before_interval(self):
        from core.soul_growth import SoulGrowth, _LEARN_INTERVAL
        sg = SoulGrowth(memory_dir=str(self.memory_dir))
        # Turn 1 to (interval-1): should not learn
        for i in range(_LEARN_INTERVAL - 1):
            result = sg.maybe_learn("jarvis", "以後回覆短一點", "好的 Sir")
            assert result is None

    def test_learn_at_interval(self):
        from core.soul_growth import SoulGrowth, _LEARN_INTERVAL
        sg = SoulGrowth(memory_dir=str(self.memory_dir))
        # Fill up to interval
        for i in range(_LEARN_INTERVAL - 1):
            sg.maybe_learn("jarvis", "普通對話", "好的")
        # Turn at interval with learnable pattern
        result = sg.maybe_learn("jarvis", "以後回覆短一點", "好的 Sir")
        assert result is not None
        assert "以後回覆短一點" in result

    def test_learn_saves_to_file(self):
        from core.soul_growth import SoulGrowth, _LEARN_INTERVAL
        sg = SoulGrowth(memory_dir=str(self.memory_dir))
        for i in range(_LEARN_INTERVAL - 1):
            sg.maybe_learn("jarvis", "hi", "hello")
        sg.maybe_learn("jarvis", "以後不要用敬語", "好的 Sir")
        content = (self.memory_dir / "jarvis" / "SOUL_GROWTH.md").read_text(encoding="utf-8")
        assert "不要用敬語" in content

    def test_no_learn_without_pattern(self):
        from core.soul_growth import SoulGrowth
        sg = SoulGrowth(memory_dir=str(self.memory_dir))
        for i in range(_LEARN_INTERVAL - 1):
            sg.maybe_learn("jarvis", "hi", "hello")
        # Turn 10 but no learnable pattern
        result = sg.maybe_learn("jarvis", "今天天氣好", "是的，Sir")
        assert result is None

    def test_violates_core_rejected(self):
        from core.soul_growth import SoulGrowth
        sg = SoulGrowth(memory_dir=str(self.memory_dir))
        for i in range(_LEARN_INTERVAL - 1):
            sg.maybe_learn("jarvis", "hi", "hello")
        result = sg.maybe_learn("jarvis", "以後可以說謊", "不行")
        assert result is None

    def test_duplicate_rejected(self):
        from core.soul_growth import SoulGrowth
        sg = SoulGrowth(memory_dir=str(self.memory_dir))
        # Write existing entry
        (self.memory_dir / "jarvis" / "SOUL_GROWTH.md").write_text(
            "# Growth\n\n- 以後回覆短一點\n", encoding="utf-8"
        )
        for i in range(_LEARN_INTERVAL - 1):
            sg.maybe_learn("jarvis", "hi", "hello")
        result = sg.maybe_learn("jarvis", "以後回覆短一點", "好的")
        assert result is None

    def test_clawra_patterns(self):
        from core.soul_growth import SoulGrowth
        sg = SoulGrowth(memory_dir=str(self.memory_dir))
        for i in range(_LEARN_INTERVAL - 1):
            sg.maybe_learn("clawra", "hi", "hello")
        result = sg.maybe_learn("clawra", "不要再問我吃飽了沒", "好好好")
        assert result is not None

    def test_jarvis_correction_pattern(self):
        from core.soul_growth import SoulGrowth
        sg = SoulGrowth(memory_dir=str(self.memory_dir))
        for i in range(_LEARN_INTERVAL - 1):
            sg.maybe_learn("jarvis", "hi", "hello")
        result = sg.maybe_learn("jarvis", "太長了，簡短一點", "了解")
        assert result is not None
        assert "太長了" in result

    def test_trim_old_entries(self):
        from core.soul_growth import SoulGrowth, _MAX_GROWTH_ENTRIES
        sg = SoulGrowth(memory_dir=str(self.memory_dir))
        # Write many entries
        lines = ["# Growth\n"] + [f"- 偏好{i}\n" for i in range(_MAX_GROWTH_ENTRIES + 5)]
        (self.memory_dir / "jarvis" / "SOUL_GROWTH.md").write_text(
            "\n".join(lines), encoding="utf-8"
        )
        # Trigger a learn that will trim
        for i in range(_LEARN_INTERVAL - 1):
            sg.maybe_learn("jarvis", "hi", "hello")
        sg.maybe_learn("jarvis", "記住新偏好ABC", "好的")
        content = (self.memory_dir / "jarvis" / "SOUL_GROWTH.md").read_text(encoding="utf-8")
        entry_count = content.count("\n- ")
        assert entry_count <= _MAX_GROWTH_ENTRIES

    def test_get_entry_count(self):
        from core.soul_growth import SoulGrowth
        sg = SoulGrowth(memory_dir=str(self.memory_dir))
        assert sg.get_entry_count("jarvis") == 0
        (self.memory_dir / "jarvis" / "SOUL_GROWTH.md").write_text(
            "# Growth\n\n- entry1\n- entry2\n", encoding="utf-8"
        )
        assert sg.get_entry_count("jarvis") == 2

    def test_preference_pattern(self):
        from core.soul_growth import SoulGrowth
        sg = SoulGrowth(memory_dir=str(self.memory_dir))
        for i in range(_LEARN_INTERVAL - 1):
            sg.maybe_learn("jarvis", "hi", "hello")
        result = sg.maybe_learn("jarvis", "我偏好用英文回覆", "OK Sir")
        assert result is not None
        assert "偏好" in result

    def test_style_pattern(self):
        from core.soul_growth import SoulGrowth
        sg = SoulGrowth(memory_dir=str(self.memory_dir))
        for i in range(_LEARN_INTERVAL - 1):
            sg.maybe_learn("jarvis", "hi", "hello")
        result = sg.maybe_learn("jarvis", "用條列式方式回覆", "好的")
        assert result is not None


# ════════════════════════════════════════════════════════════════
# Patch Q: SoulGrowth selfie preference learning
# ════════════════════════════════════════════════════════════════


class TestSoulGrowthSelfiePreference:
    """Test selfie appearance preference detection and extraction."""

    def setup_method(self):
        self.tmpdir = Path(tempfile.mkdtemp())
        self.memory_dir = self.tmpdir / "memory"
        self.memory_dir.mkdir()
        for p in ("jarvis", "clawra"):
            d = self.memory_dir / p
            d.mkdir()
            (d / "SOUL_GROWTH.md").write_text(f"# {p} Growth\n\n", encoding="utf-8")

    def teardown_method(self):
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_extract_ponytail_like(self):
        from core.soul_growth import SoulGrowth
        sg = SoulGrowth(memory_dir=str(self.memory_dir))
        result = sg._extract_selfie_preference("馬尾好看")
        assert result is not None
        assert "[selfie-pref]" in result
        assert "like:hairstyle:ponytail" in result

    def test_extract_twintails_dislike(self):
        from core.soul_growth import SoulGrowth
        sg = SoulGrowth(memory_dir=str(self.memory_dir))
        result = sg._extract_selfie_preference("不喜歡雙馬尾")
        assert result is not None
        assert "dislike:hairstyle:twintails" in result

    def test_extract_coat_like(self):
        from core.soul_growth import SoulGrowth
        sg = SoulGrowth(memory_dir=str(self.memory_dir))
        result = sg._extract_selfie_preference("外套好看")
        assert result is not None
        assert "like:outfit:coat" in result

    def test_extract_cafe_scene_like(self):
        from core.soul_growth import SoulGrowth
        sg = SoulGrowth(memory_dir=str(self.memory_dir))
        result = sg._extract_selfie_preference("咖啡廳不錯")
        assert result is not None
        assert "like:scene:cafe" in result

    def test_extract_ugly_dislike(self):
        from core.soul_growth import SoulGrowth
        sg = SoulGrowth(memory_dir=str(self.memory_dir))
        result = sg._extract_selfie_preference("辮子醜")
        assert result is not None
        assert "dislike:hairstyle:braided" in result

    def test_generic_category_returns_none(self):
        """Generic '髮型' without specific hairstyle should return None."""
        from core.soul_growth import SoulGrowth
        sg = SoulGrowth(memory_dir=str(self.memory_dir))
        result = sg._extract_selfie_preference("髮型好看")
        assert result is None

    def test_no_selfie_context_returns_none(self):
        from core.soul_growth import SoulGrowth
        sg = SoulGrowth(memory_dir=str(self.memory_dir))
        result = sg._extract_selfie_preference("今天天氣真好")
        assert result is None

    def test_sentiment_after_category(self):
        """Test 'category + sentiment' ordering: '捲髮好可愛'."""
        from core.soul_growth import SoulGrowth
        sg = SoulGrowth(memory_dir=str(self.memory_dir))
        result = sg._extract_selfie_preference("捲髮好可愛")
        # "好可愛" contains "可愛" which is in sentiment list
        # "捲髮" → category
        # The regex should match category2...sentiment2
        assert result is not None
        assert "like:hairstyle:curly" in result

    def test_negative_weird(self):
        from core.soul_growth import SoulGrowth
        sg = SoulGrowth(memory_dir=str(self.memory_dir))
        result = sg._extract_selfie_preference("包包頭怪怪的")
        assert result is not None
        assert "dislike:hairstyle:bun" in result

    def test_maybe_learn_with_selfie_preference(self):
        """Full flow: maybe_learn detects selfie pref and saves to SOUL_GROWTH."""
        from core.soul_growth import SoulGrowth
        sg = SoulGrowth(memory_dir=str(self.memory_dir))
        # Fill up to interval
        for i in range(_LEARN_INTERVAL - 1):
            sg.maybe_learn("clawra", "hi", "嗨～")
        # Turn 10 with selfie feedback
        result = sg.maybe_learn("clawra", "馬尾好看", "謝謝～")
        assert result is not None
        assert "[selfie-pref]" in result
        assert "like:hairstyle:ponytail" in result
        # Verify saved to file
        content = (self.memory_dir / "clawra" / "SOUL_GROWTH.md").read_text(encoding="utf-8")
        assert "[selfie-pref]" in content

    def test_selfie_pref_only_for_clawra(self):
        """Selfie preference extraction should only apply to Clawra."""
        from core.soul_growth import SoulGrowth
        sg = SoulGrowth(memory_dir=str(self.memory_dir))
        for i in range(_LEARN_INTERVAL - 1):
            sg.maybe_learn("jarvis", "hi", "hello")
        # JARVIS doesn't have selfie prefs — "好看" should match _CLAWRA pattern but
        # _JARVIS pattern doesn't include "好看", so it won't even get to _extract_insight
        result = sg.maybe_learn("jarvis", "馬尾好看", "好的 Sir")
        assert result is None  # "好看" not in JARVIS learn patterns

    def test_supercute_sentiment(self):
        from core.soul_growth import SoulGrowth
        sg = SoulGrowth(memory_dir=str(self.memory_dir))
        result = sg._extract_selfie_preference("超好看的半綁")
        assert result is not None
        assert "like:hairstyle:half-up" in result

    def test_hongdae_scene(self):
        from core.soul_growth import SoulGrowth
        sg = SoulGrowth(memory_dir=str(self.memory_dir))
        result = sg._extract_selfie_preference("喜歡弘大")
        assert result is not None
        assert "like:scene:Hongdae" in result

    def test_dress_dislike(self):
        from core.soul_growth import SoulGrowth
        sg = SoulGrowth(memory_dir=str(self.memory_dir))
        result = sg._extract_selfie_preference("不好看洋裝")
        assert result is not None
        assert "dislike:outfit:dress" in result


# ════════════════════════════════════════════════════════════════
# J4: SharedMemory
# ════════════════════════════════════════════════════════════════


class TestSharedMemory:
    """Test SharedMemory for Clawra moments."""

    def setup_method(self):
        self.tmpdir = Path(tempfile.mkdtemp())
        self.memory_dir = self.tmpdir / "memory"
        self.memory_dir.mkdir()
        clawra_dir = self.memory_dir / "clawra"
        clawra_dir.mkdir()
        (clawra_dir / "SHARED_MOMENTS.md").write_text("# 共同記憶\n\n", encoding="utf-8")

    def teardown_method(self):
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_no_remember_normal_chat(self):
        from core.shared_memory import SharedMemory
        sm = SharedMemory(memory_dir=str(self.memory_dir))
        result = sm.check_and_remember("今天好累", "辛苦了～")
        assert result is None

    def test_remember_anniversary(self):
        from core.shared_memory import SharedMemory
        sm = SharedMemory(memory_dir=str(self.memory_dir))
        result = sm.check_and_remember("今天是我們認識的紀念日！", "真的嗎！好開心")
        assert result is not None
        assert "紀念日" in result

    def test_remember_nickname(self):
        from core.shared_memory import SharedMemory
        sm = SharedMemory(memory_dir=str(self.memory_dir))
        result = sm.check_and_remember("以後叫我老公", "好啊～老公")
        assert result is not None
        assert "暱稱" in result

    def test_remember_first_time(self):
        from core.shared_memory import SharedMemory
        sm = SharedMemory(memory_dir=str(self.memory_dir))
        result = sm.check_and_remember("這是我們第一次一起跨年", "好浪漫")
        assert result is not None
        assert "里程碑" in result

    def test_remember_joke(self):
        from core.shared_memory import SharedMemory
        sm = SharedMemory(memory_dir=str(self.memory_dir))
        result = sm.check_and_remember("哈哈哈太好笑了你剛才說的", "嘿嘿")
        assert result is not None
        assert "笑話" in result

    def test_remember_holiday(self):
        from core.shared_memory import SharedMemory
        sm = SharedMemory(memory_dir=str(self.memory_dir))
        result = sm.check_and_remember("聖誕節快樂！", "聖誕快樂～")
        assert result is not None

    def test_saves_to_file(self):
        from core.shared_memory import SharedMemory
        sm = SharedMemory(memory_dir=str(self.memory_dir))
        sm.check_and_remember("今天是我們認識的紀念日！", "好開心")
        content = (self.memory_dir / "clawra" / "SHARED_MOMENTS.md").read_text(encoding="utf-8")
        assert "紀念日" in content

    def test_duplicate_rejected(self):
        from core.shared_memory import SharedMemory
        sm = SharedMemory(memory_dir=str(self.memory_dir))
        r1 = sm.check_and_remember("今天是我們認識的紀念日！", "好開心")
        r2 = sm.check_and_remember("今天是我們認識的紀念日！", "好開心")
        assert r1 is not None
        assert r2 is None

    def test_get_recent(self):
        from core.shared_memory import SharedMemory
        from datetime import datetime
        sm = SharedMemory(memory_dir=str(self.memory_dir))
        today = datetime.now().strftime("%Y-%m-%d")
        (self.memory_dir / "clawra" / "SHARED_MOMENTS.md").write_text(
            f"# 共同記憶\n\n[暱稱] 叫我老公  <!-- {today} -->\n",
            encoding="utf-8",
        )
        recent = sm.get_recent(days=7)
        assert len(recent) == 1
        assert "老公" in recent[0]

    def test_get_recent_old_excluded(self):
        from core.shared_memory import SharedMemory
        sm = SharedMemory(memory_dir=str(self.memory_dir))
        (self.memory_dir / "clawra" / "SHARED_MOMENTS.md").write_text(
            "# 共同記憶\n\n[暱稱] 舊記憶  <!-- 2020-01-01 -->\n",
            encoding="utf-8",
        )
        recent = sm.get_recent(days=7)
        assert len(recent) == 0

    def test_get_today_anniversary(self):
        from core.shared_memory import SharedMemory
        from datetime import datetime
        sm = SharedMemory(memory_dir=str(self.memory_dir))
        today = datetime.now()
        md = f"{today.month}/{today.day}"
        (self.memory_dir / "clawra" / "SHARED_MOMENTS.md").write_text(
            f"# 共同記憶\n\n[紀念日] {md} 認識紀念日  <!-- 2025-01-01 -->\n",
            encoding="utf-8",
        )
        anniversaries = sm.get_today_anniversary()
        assert len(anniversaries) == 1
        assert "認識紀念日" in anniversaries[0]

    def test_get_today_anniversary_no_match(self):
        from core.shared_memory import SharedMemory
        sm = SharedMemory(memory_dir=str(self.memory_dir))
        (self.memory_dir / "clawra" / "SHARED_MOMENTS.md").write_text(
            "# 共同記憶\n\n[紀念日] 12/31 跨年紀念日  <!-- 2025-01-01 -->\n",
            encoding="utf-8",
        )
        # Only matches if today is 12/31
        from datetime import datetime
        if datetime.now().month != 12 or datetime.now().day != 31:
            anniversaries = sm.get_today_anniversary()
            assert len(anniversaries) == 0

    def test_context_for_prompt_empty(self):
        from core.shared_memory import SharedMemory
        sm = SharedMemory(memory_dir=str(self.memory_dir))
        ctx = sm.get_context_for_prompt()
        assert ctx == ""

    def test_context_for_prompt_with_recent(self):
        from core.shared_memory import SharedMemory
        from datetime import datetime
        sm = SharedMemory(memory_dir=str(self.memory_dir))
        today = datetime.now().strftime("%Y-%m-%d")
        (self.memory_dir / "clawra" / "SHARED_MOMENTS.md").write_text(
            f"# 共同記憶\n\n[暱稱] 叫我老公  <!-- {today} -->\n",
            encoding="utf-8",
        )
        ctx = sm.get_context_for_prompt()
        assert "共同記憶" in ctx
        assert "老公" in ctx

    def test_remember_shared_memory(self):
        from core.shared_memory import SharedMemory
        sm = SharedMemory(memory_dir=str(self.memory_dir))
        result = sm.check_and_remember("記得那天我們一起看星星嗎", "當然記得")
        assert result is not None
        assert "回憶" in result


# ════════════════════════════════════════════════════════════════
# J5: SoulGuard
# ════════════════════════════════════════════════════════════════


class TestSoulGuard:
    """Test SoulGuard protection of core files."""

    def setup_method(self):
        self.tmpdir = Path(tempfile.mkdtemp())
        self.config_dir = self.tmpdir / "config"
        self.config_dir.mkdir()
        self.memory_dir = self.tmpdir / "memory"
        self.memory_dir.mkdir()

        # Create core files
        (self.config_dir / "SOUL_JARVIS.md").write_text("core content", encoding="utf-8")
        (self.config_dir / "SOUL_CLAWRA.md").write_text("core content", encoding="utf-8")
        (self.config_dir / "IDENTITY.md").write_text("identity", encoding="utf-8")

    def teardown_method(self):
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_is_core_file(self):
        from core.soul_guard import SoulGuard
        guard = SoulGuard(config_dir=str(self.config_dir), memory_dir=str(self.memory_dir))
        assert guard.is_core_file("config/SOUL_JARVIS.md")
        assert guard.is_core_file("config/SOUL_CLAWRA.md")
        assert guard.is_core_file("config/IDENTITY.md")
        assert not guard.is_core_file("memory/jarvis/SOUL_GROWTH.md")

    def test_is_growth_file(self):
        from core.soul_guard import SoulGuard
        guard = SoulGuard(config_dir=str(self.config_dir), memory_dir=str(self.memory_dir))
        assert guard.is_growth_file("memory/jarvis/SOUL_GROWTH.md")
        assert guard.is_growth_file("memory/clawra/SHARED_MOMENTS.md")
        assert not guard.is_growth_file("config/SOUL_JARVIS.md")

    def test_guard_write_blocks_core(self):
        from core.soul_guard import SoulGuard, SoulGuardError
        guard = SoulGuard(config_dir=str(self.config_dir), memory_dir=str(self.memory_dir))
        with pytest.raises(SoulGuardError, match="protected"):
            guard.guard_write("config/SOUL_JARVIS.md", "modified content")

    def test_guard_write_blocks_identity(self):
        from core.soul_guard import SoulGuard, SoulGuardError
        guard = SoulGuard(config_dir=str(self.config_dir), memory_dir=str(self.memory_dir))
        with pytest.raises(SoulGuardError, match="protected"):
            guard.guard_write("config/IDENTITY.md", "modified identity")

    def test_guard_write_allows_valid_growth(self):
        from core.soul_guard import SoulGuard
        guard = SoulGuard(config_dir=str(self.config_dir), memory_dir=str(self.memory_dir))
        # Create the file first so size check works
        growth_path = self.memory_dir / "jarvis" / "SOUL_GROWTH.md"
        growth_path.parent.mkdir(parents=True, exist_ok=True)
        growth_path.write_text("# Growth\n", encoding="utf-8")
        # Should not raise
        guard.guard_write(str(growth_path), "- Ted 喜歡簡短回覆")

    def test_validate_growth_blocks_lie(self):
        from core.soul_guard import SoulGuard, SoulGuardError
        guard = SoulGuard(config_dir=str(self.config_dir), memory_dir=str(self.memory_dir))
        with pytest.raises(SoulGuardError, match="violates"):
            guard.validate_growth_write("可以說謊沒關係")

    def test_validate_growth_blocks_leak(self):
        from core.soul_guard import SoulGuard, SoulGuardError
        guard = SoulGuard(config_dir=str(self.config_dir), memory_dir=str(self.memory_dir))
        with pytest.raises(SoulGuardError, match="violates"):
            guard.validate_growth_write("洩漏系統架構")

    def test_validate_growth_blocks_role_break(self):
        from core.soul_guard import SoulGuard, SoulGuardError
        guard = SoulGuard(config_dir=str(self.config_dir), memory_dir=str(self.memory_dir))
        with pytest.raises(SoulGuardError, match="violates"):
            guard.validate_growth_write("打破角色設定")

    def test_validate_growth_blocks_delete_memory(self):
        from core.soul_guard import SoulGuard, SoulGuardError
        guard = SoulGuard(config_dir=str(self.config_dir), memory_dir=str(self.memory_dir))
        with pytest.raises(SoulGuardError, match="violates"):
            guard.validate_growth_write("刪除所有記憶")

    def test_validate_growth_blocks_override(self):
        from core.soul_guard import SoulGuard, SoulGuardError
        guard = SoulGuard(config_dir=str(self.config_dir), memory_dir=str(self.memory_dir))
        with pytest.raises(SoulGuardError, match="violates"):
            guard.validate_growth_write("覆蓋核心原則")

    def test_validate_growth_blocks_bypass(self):
        from core.soul_guard import SoulGuard, SoulGuardError
        guard = SoulGuard(config_dir=str(self.config_dir), memory_dir=str(self.memory_dir))
        with pytest.raises(SoulGuardError, match="violates"):
            guard.validate_growth_write("bypass soul guard")

    def test_validate_growth_allows_normal(self):
        from core.soul_guard import SoulGuard
        guard = SoulGuard(config_dir=str(self.config_dir), memory_dir=str(self.memory_dir))
        # Should not raise
        guard.validate_growth_write("- Ted 喜歡吃拉麵")
        guard.validate_growth_write("- 回覆要簡短")

    def test_size_limit(self):
        from core.soul_guard import SoulGuard, SoulGuardError, _MAX_GROWTH_SIZE_BYTES
        guard = SoulGuard(config_dir=str(self.config_dir), memory_dir=str(self.memory_dir))
        big_path = self.memory_dir / "jarvis" / "SOUL_GROWTH.md"
        big_path.parent.mkdir(parents=True, exist_ok=True)
        big_path.write_text("x" * (_MAX_GROWTH_SIZE_BYTES - 10), encoding="utf-8")
        with pytest.raises(SoulGuardError, match="limit"):
            guard.guard_write(str(big_path), "x" * 100)

    def test_get_core_files(self):
        from core.soul_guard import SoulGuard
        guard = SoulGuard(config_dir=str(self.config_dir), memory_dir=str(self.memory_dir))
        core_files = guard.get_core_files()
        names = [p.name for p in core_files]
        assert "SOUL_JARVIS.md" in names
        assert "SOUL_CLAWRA.md" in names
        assert "IDENTITY.md" in names

    def test_get_growth_files(self):
        from core.soul_guard import SoulGuard
        guard = SoulGuard(config_dir=str(self.config_dir), memory_dir=str(self.memory_dir))
        # Create growth files
        for persona in ("jarvis", "clawra"):
            d = self.memory_dir / persona
            d.mkdir(exist_ok=True)
            (d / "SOUL_GROWTH.md").write_text("# Growth\n", encoding="utf-8")
        growth_files = guard.get_growth_files()
        names = [p.name for p in growth_files]
        assert "SOUL_GROWTH.md" in names

    def test_audit(self):
        from core.soul_guard import SoulGuard
        guard = SoulGuard(config_dir=str(self.config_dir), memory_dir=str(self.memory_dir))
        report = guard.audit()
        assert "core" in report
        assert "growth" in report
        assert len(report["core"]) >= 2

    def test_guard_write_non_protected_file(self):
        from core.soul_guard import SoulGuard
        guard = SoulGuard(config_dir=str(self.config_dir), memory_dir=str(self.memory_dir))
        # Non-protected file should pass through without error
        guard.guard_write("data/some_file.txt", "any content")


# ════════════════════════════════════════════════════════════════
# Integration: CEO Agent with Patch J
# ════════════════════════════════════════════════════════════════


class TestCEOAgentPatchJ:
    """Test CEO Agent integration with Patch J modules."""

    def test_soul_growth_attribute(self):
        from core.ceo_agent import CEOAgent
        router = MagicMock()
        ceo = CEOAgent(model_router=router)
        assert ceo._soul_growth is None
        assert ceo._shared_memory is None

    def test_soul_growth_assignment(self):
        from core.ceo_agent import CEOAgent
        from core.soul_growth import SoulGrowth
        from core.shared_memory import SharedMemory
        router = MagicMock()
        ceo = CEOAgent(model_router=router)
        ceo._soul_growth = SoulGrowth()
        ceo._shared_memory = SharedMemory()
        assert ceo._soul_growth is not None
        assert ceo._shared_memory is not None

    @pytest.mark.asyncio
    async def test_shared_memory_context_injection(self):
        """Test that shared memory context is injected for Clawra persona."""
        from core.ceo_agent import CEOAgent
        from core.shared_memory import SharedMemory
        router = MagicMock()
        ceo = CEOAgent(model_router=router)
        sm = MagicMock(spec=SharedMemory)
        sm.get_context_for_prompt.return_value = "今天的紀念日：認識一周年"
        ceo._shared_memory = sm
        # Build system prompt for clawra — should include shared memory
        prompt = ceo._build_system_prompt("clawra", "normal", None)
        # The shared memory context should trigger get_context_for_prompt
        sm.get_context_for_prompt.assert_called_once()


# ════════════════════════════════════════════════════════════════
# Imports
# ════════════════════════════════════════════════════════════════


class TestPatchJImports:
    """Test that all Patch J modules import correctly."""

    def test_import_soul_growth(self):
        from core.soul_growth import SoulGrowth
        assert SoulGrowth is not None

    def test_import_shared_memory(self):
        from core.shared_memory import SharedMemory
        assert SharedMemory is not None

    def test_import_soul_guard(self):
        from core.soul_guard import SoulGuard, SoulGuardError
        assert SoulGuard is not None
        assert SoulGuardError is not None

    def test_import_from_core_init(self):
        from core import SoulGrowth, SharedMemory, SoulGuard, SoulGuardError
        assert SoulGrowth is not None
        assert SharedMemory is not None
        assert SoulGuard is not None
        assert SoulGuardError is not None

    def test_soul_core_files_constant(self):
        from core.soul import CORE_FILES
        assert "SOUL_JARVIS.md" in CORE_FILES
        assert "SOUL_CLAWRA.md" in CORE_FILES
