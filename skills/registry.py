"""SkillRegistry — loading, querying, and invoking solidified skills.

A "skill" is a reusable task that was once an ad-hoc script,
recognized as repeatable by the CEO Agent, and packaged into:

    skills/{category}/{name}/
        skill.yaml   — metadata (name, version, input/output schema, deps)
        main.py      — entry point with async execute() function
        test.py      — unit tests (optional)

The Registry scans the skills directory, loads metadata, and provides
search + invocation for the CEO Agent.
"""

from __future__ import annotations

import importlib
import importlib.util
import inspect
import sys
from pathlib import Path
from typing import Any

import yaml
from loguru import logger


class SkillMeta:
    """Parsed metadata from a skill.yaml file."""

    def __init__(self, data: dict[str, Any], skill_dir: Path):
        self.name: str = data.get("name", "")
        self.display_name: str = data.get("display_name", self.name)
        self.version: str = data.get("version", "0.1")
        self.category: str = data.get("category", "general")
        self.description: str = data.get("description", "")
        self.input_schema: dict = data.get("input", {})
        self.output_schema: dict = data.get("output", {})
        self.dependencies: list[str] = data.get("dependencies", [])
        self.skill_dir: Path = skill_dir
        self._raw = data

    @property
    def module_path(self) -> str:
        """Python import path for the skill's main module."""
        parts = self.skill_dir.relative_to(Path(".")).parts
        return ".".join(parts) + ".main"

    def matches(self, query: str) -> bool:
        """Check if this skill matches a search query (fuzzy)."""
        q = query.lower()
        return (
            q in self.name.lower()
            or q in self.display_name.lower()
            or q in self.description.lower()
            or q in self.category.lower()
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "display_name": self.display_name,
            "version": self.version,
            "category": self.category,
            "description": self.description,
            "input": self.input_schema,
            "output": self.output_schema,
            "dir": str(self.skill_dir),
        }

    def __repr__(self) -> str:
        return f"SkillMeta({self.name!r} v{self.version})"


class SkillRegistry:
    """Central registry for all solidified skills.

    Usage:
        registry = SkillRegistry("./skills")
        registry.scan()

        # Search
        results = registry.search("自拍")
        # → [SkillMeta(name='selfie', ...)]

        # List all
        all_skills = registry.list_all()

        # Invoke
        result = await registry.invoke("selfie", scene="holding coffee")
    """

    def __init__(self, skills_dir: str = "./skills"):
        self.skills_dir = Path(skills_dir)
        self._skills: dict[str, SkillMeta] = {}
        self._index_path = self.skills_dir / "skills_index.json"

    def scan(self) -> int:
        """Scan skills directory and load all skill.yaml files.

        Returns number of skills loaded.
        """
        self._skills.clear()

        if not self.skills_dir.exists():
            logger.warning(f"Skills directory not found: {self.skills_dir}")
            return 0

        for yaml_path in self.skills_dir.rglob("skill.yaml"):
            try:
                with open(yaml_path, "r", encoding="utf-8") as f:
                    data = yaml.safe_load(f)

                if not data or not isinstance(data, dict):
                    continue

                skill_dir = yaml_path.parent
                meta = SkillMeta(data, skill_dir)

                if not meta.name:
                    logger.warning(f"Skill at {yaml_path} has no name, skipping")
                    continue

                self._skills[meta.name] = meta
                logger.debug(f"Loaded skill: {meta.name} v{meta.version}")

            except Exception as e:
                logger.warning(f"Failed to load skill from {yaml_path}: {e}")

        logger.info(f"SkillRegistry: {len(self._skills)} skills loaded")
        self._save_index()
        return len(self._skills)

    def get(self, name: str) -> SkillMeta | None:
        """Get a skill by exact name."""
        return self._skills.get(name)

    def search(self, query: str) -> list[SkillMeta]:
        """Fuzzy search skills by name, display_name, description, or category."""
        return [s for s in self._skills.values() if s.matches(query)]

    def list_all(self) -> list[SkillMeta]:
        """List all registered skills."""
        return list(self._skills.values())

    def list_by_category(self, category: str) -> list[SkillMeta]:
        """List skills in a specific category."""
        return [s for s in self._skills.values() if s.category == category]

    async def invoke(self, name: str, **kwargs: Any) -> Any:
        """Dynamically load and invoke a skill's execute() function.

        The skill module must have an async execute(**kwargs) function.
        """
        meta = self._skills.get(name)
        if not meta:
            raise SkillNotFoundError(f"Skill '{name}' not registered")

        # Ensure the skill directory is importable
        skill_parent = str(meta.skill_dir.parent)
        if skill_parent not in sys.path:
            sys.path.insert(0, skill_parent)

        try:
            main_file = meta.skill_dir / "main.py"
            module_name = f"_skill_{name}_main"

            spec = importlib.util.spec_from_file_location(module_name, main_file)
            if spec is None or spec.loader is None:
                raise SkillExecutionError(
                    f"Skill '{name}' has no main.py at {main_file}"
                )

            # Always load fresh (path may differ between invocations)
            module = importlib.util.module_from_spec(spec)
            sys.modules[module_name] = module
            spec.loader.exec_module(module)

            if not hasattr(module, "execute"):
                raise SkillExecutionError(
                    f"Skill '{name}' has no execute() function in main.py"
                )

            execute_fn = module.execute
            if inspect.iscoroutinefunction(execute_fn):
                return await execute_fn(**kwargs)
            else:
                import asyncio
                loop = asyncio.get_event_loop()
                return await loop.run_in_executor(None, lambda: execute_fn(**kwargs))

        except (SkillNotFoundError, SkillExecutionError):
            raise
        except Exception as e:
            raise SkillExecutionError(f"Skill '{name}' execution failed: {e}") from e

    def register_skill(self, meta: SkillMeta) -> None:
        """Manually register a skill (for dynamically created skills)."""
        self._skills[meta.name] = meta
        logger.info(f"Skill registered: {meta.name} v{meta.version}")
        self._save_index()

    def unregister_skill(self, name: str) -> bool:
        """Remove a skill from the registry."""
        if name in self._skills:
            del self._skills[name]
            self._save_index()
            return True
        return False

    def _save_index(self) -> None:
        """Write skills_index.json for quick reference."""
        import json

        index = {name: meta.to_dict() for name, meta in self._skills.items()}
        try:
            self._index_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self._index_path, "w", encoding="utf-8") as f:
                json.dump(index, f, ensure_ascii=False, indent=2)
        except Exception as e:
            logger.warning(f"Failed to save skills index: {e}")

    @property
    def count(self) -> int:
        return len(self._skills)


class SkillNotFoundError(Exception):
    """Raised when a requested skill is not in the registry."""


class SkillExecutionError(Exception):
    """Raised when a skill fails during execution."""
