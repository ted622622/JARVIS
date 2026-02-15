"""Tests for SkillRegistry and skill framework.

Run: pytest tests/test_skills.py -v
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

from skills.registry import SkillExecutionError, SkillMeta, SkillNotFoundError, SkillRegistry


# ── Fixtures ────────────────────────────────────────────────────


@pytest.fixture
def skills_dir(tmp_path) -> Path:
    """Create a temp skills directory with test skills."""
    # Skill A: complete skill
    skill_a = tmp_path / "category_a" / "skill_a"
    skill_a.mkdir(parents=True)
    (skill_a / "skill.yaml").write_text(yaml.dump({
        "name": "skill_a",
        "display_name": "Skill Alpha",
        "version": "1.0",
        "category": "testing",
        "description": "A test skill for unit testing",
        "input": {"msg": "A message"},
        "output": {"result": "Echoed message"},
    }))
    (skill_a / "__init__.py").write_text("")
    (skill_a / "main.py").write_text(
        "async def execute(msg='hello', **kw):\n"
        "    return {'result': f'echo: {msg}'}\n"
    )

    # Skill B: another skill
    skill_b = tmp_path / "category_b" / "skill_b"
    skill_b.mkdir(parents=True)
    (skill_b / "skill.yaml").write_text(yaml.dump({
        "name": "skill_b",
        "display_name": "搜尋技能",
        "version": "2.1",
        "category": "search",
        "description": "Search for information",
    }))
    (skill_b / "__init__.py").write_text("")
    (skill_b / "main.py").write_text(
        "async def execute(**kw):\n"
        "    return {'found': True}\n"
    )

    # Skill C: broken (no execute function)
    skill_c = tmp_path / "broken" / "skill_c"
    skill_c.mkdir(parents=True)
    (skill_c / "skill.yaml").write_text(yaml.dump({
        "name": "skill_c",
        "version": "0.1",
        "category": "broken",
        "description": "This skill has no execute()",
    }))
    (skill_c / "__init__.py").write_text("")
    (skill_c / "main.py").write_text("def helper(): pass\n")

    return tmp_path


@pytest.fixture
def registry(skills_dir) -> SkillRegistry:
    reg = SkillRegistry(str(skills_dir))
    reg.scan()
    return reg


# ── SkillMeta Tests ─────────────────────────────────────────────


class TestSkillMeta:
    def test_from_dict(self, tmp_path):
        data = {
            "name": "test_skill",
            "display_name": "Test Skill",
            "version": "1.0",
            "category": "testing",
            "description": "A test",
        }
        meta = SkillMeta(data, tmp_path)
        assert meta.name == "test_skill"
        assert meta.display_name == "Test Skill"
        assert meta.version == "1.0"

    def test_matches_by_name(self, tmp_path):
        meta = SkillMeta({"name": "selfie", "description": ""}, tmp_path)
        assert meta.matches("selfie")
        assert meta.matches("Selfie")
        assert not meta.matches("weather")

    def test_matches_by_description(self, tmp_path):
        meta = SkillMeta({"name": "x", "description": "Generate 自拍 images"}, tmp_path)
        assert meta.matches("自拍")

    def test_matches_by_category(self, tmp_path):
        meta = SkillMeta({"name": "x", "description": "", "category": "identity"}, tmp_path)
        assert meta.matches("identity")

    def test_to_dict(self, tmp_path):
        meta = SkillMeta({"name": "test", "version": "1.0"}, tmp_path)
        d = meta.to_dict()
        assert d["name"] == "test"
        assert "version" in d


# ── Registry Scan Tests ─────────────────────────────────────────


class TestRegistryScan:
    def test_scan_finds_skills(self, registry):
        assert registry.count == 3  # skill_a, skill_b, skill_c

    def test_scan_empty_dir(self, tmp_path):
        reg = SkillRegistry(str(tmp_path / "empty"))
        count = reg.scan()
        assert count == 0

    def test_scan_nonexistent_dir(self, tmp_path):
        reg = SkillRegistry(str(tmp_path / "nonexistent"))
        count = reg.scan()
        assert count == 0

    def test_rescan_refreshes(self, registry, skills_dir):
        assert registry.count == 3
        # Add a new skill
        new_skill = skills_dir / "new" / "skill_d"
        new_skill.mkdir(parents=True)
        (new_skill / "skill.yaml").write_text(yaml.dump({
            "name": "skill_d", "version": "1.0",
        }))
        registry.scan()
        assert registry.count == 4


# ── Registry Query Tests ────────────────────────────────────────


class TestRegistryQuery:
    def test_get_by_name(self, registry):
        meta = registry.get("skill_a")
        assert meta is not None
        assert meta.name == "skill_a"
        assert meta.version == "1.0"

    def test_get_nonexistent(self, registry):
        assert registry.get("nonexistent") is None

    def test_search_by_name(self, registry):
        results = registry.search("skill_a")
        assert len(results) == 1
        assert results[0].name == "skill_a"

    def test_search_by_display_name(self, registry):
        results = registry.search("搜尋")
        assert len(results) == 1
        assert results[0].name == "skill_b"

    def test_search_by_description(self, registry):
        results = registry.search("unit testing")
        assert len(results) == 1

    def test_search_no_results(self, registry):
        results = registry.search("zzz_nonexistent_zzz")
        assert len(results) == 0

    def test_list_all(self, registry):
        all_skills = registry.list_all()
        assert len(all_skills) == 3

    def test_list_by_category(self, registry):
        testing = registry.list_by_category("testing")
        assert len(testing) == 1
        assert testing[0].name == "skill_a"

    def test_list_by_category_empty(self, registry):
        assert len(registry.list_by_category("nonexistent")) == 0


# ── Registry Invoke Tests ───────────────────────────────────────


class TestRegistryInvoke:
    @pytest.mark.asyncio
    async def test_invoke_skill(self, registry):
        result = await registry.invoke("skill_a", msg="hello world")
        assert result == {"result": "echo: hello world"}

    @pytest.mark.asyncio
    async def test_invoke_with_defaults(self, registry):
        result = await registry.invoke("skill_a")
        assert result == {"result": "echo: hello"}

    @pytest.mark.asyncio
    async def test_invoke_skill_b(self, registry):
        result = await registry.invoke("skill_b")
        assert result == {"found": True}

    @pytest.mark.asyncio
    async def test_invoke_nonexistent_raises(self, registry):
        with pytest.raises(SkillNotFoundError):
            await registry.invoke("nonexistent_skill")

    @pytest.mark.asyncio
    async def test_invoke_broken_skill_raises(self, registry):
        with pytest.raises(SkillExecutionError, match="no execute"):
            await registry.invoke("skill_c")


# ── Registry Management ─────────────────────────────────────────


class TestRegistryManagement:
    def test_register_skill(self, registry, tmp_path):
        meta = SkillMeta({"name": "dynamic", "version": "1.0"}, tmp_path)
        registry.register_skill(meta)
        assert registry.get("dynamic") is not None
        assert registry.count == 4

    def test_unregister_skill(self, registry):
        assert registry.unregister_skill("skill_a")
        assert registry.get("skill_a") is None
        assert registry.count == 2

    def test_unregister_nonexistent(self, registry):
        assert not registry.unregister_skill("nonexistent")


# ── Index File ──────────────────────────────────────────────────


class TestIndexFile:
    def test_index_saved_on_scan(self, registry, skills_dir):
        index_path = skills_dir / "skills_index.json"
        assert index_path.exists()
        with open(index_path, encoding="utf-8") as f:
            index = json.load(f)
        assert "skill_a" in index
        assert "skill_b" in index
        assert index["skill_a"]["version"] == "1.0"


# ── Real Skills (selfie + template) ────────────────────────────


class TestRealSkills:
    def test_scan_real_skills_dir(self):
        """Test that the actual skills/ directory loads correctly."""
        reg = SkillRegistry("./skills")
        count = reg.scan()
        assert count >= 2  # At least selfie + template

    def test_selfie_skill_found(self):
        reg = SkillRegistry("./skills")
        reg.scan()
        results = reg.search("自拍")
        assert len(results) >= 1
        selfie = results[0]
        assert selfie.name == "selfie"
        assert selfie.category == "identity"

    def test_template_skill_found(self):
        reg = SkillRegistry("./skills")
        reg.scan()
        meta = reg.get("template")
        assert meta is not None
        assert meta.category == "example"

    @pytest.mark.asyncio
    async def test_invoke_template_skill(self):
        reg = SkillRegistry("./skills")
        reg.scan()
        result = await reg.invoke("template", message="test123")
        assert result["result"] == "[template] test123"
        assert "timestamp" in result
