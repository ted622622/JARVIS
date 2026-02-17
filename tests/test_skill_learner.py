"""Tests for SkillLearner — observe user patterns, propose automations.

Run: pytest tests/test_skill_learner.py -v
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from clients.base_client import ChatResponse
from core.skill_learner import SkillLearner


# ── Fixtures ────────────────────────────────────────────────────


@pytest.fixture
def mock_router():
    router = MagicMock()
    router.chat = AsyncMock(
        return_value=ChatResponse(
            content="Sir，我注意到你每天都會查天氣，要我自動幫你嗎？",
            model="test",
        )
    )
    return router


@pytest.fixture
def mock_telegram():
    tg = MagicMock()
    tg.send_message = AsyncMock()
    return tg


@pytest.fixture
def learner(tmp_path, mock_router, mock_telegram):
    """SkillLearner with temp paths for isolation."""
    sl = SkillLearner(
        scheduler=None,
        telegram=mock_telegram,
        model_router=mock_router,
    )
    sl.LOG_PATH = tmp_path / "user_actions.json"
    sl.SKILL_DIR = tmp_path / "learned"
    sl.PROPOSALS_PATH = tmp_path / "skill_proposals.json"
    sl._actions = []
    return sl


@pytest.fixture
def learner_no_deps(tmp_path):
    """SkillLearner without dependencies."""
    sl = SkillLearner()
    sl.LOG_PATH = tmp_path / "user_actions.json"
    sl.SKILL_DIR = tmp_path / "learned"
    sl.PROPOSALS_PATH = tmp_path / "skill_proposals.json"
    sl._actions = []
    return sl


# ── Action logging ────────────────────────────────────────────────


class TestLogAction:
    def test_log_single_action(self, learner):
        learner.log_action({"type": "weather_check", "detail": "台北天氣"})
        assert len(learner._actions) == 1
        assert learner._actions[0]["type"] == "weather_check"

    def test_log_adds_defaults(self, learner):
        learner.log_action({"type": "test"})
        action = learner._actions[0]
        assert "timestamp" in action
        assert "weekday" in action
        assert "hour" in action
        assert "date" in action

    def test_log_preserves_custom_fields(self, learner):
        learner.log_action({"type": "test", "custom": "value"})
        assert learner._actions[0]["custom"] == "value"

    def test_log_evicts_old_entries(self, learner):
        old_ts = time.time() - (learner.WINDOW_DAYS + 1) * 86400
        learner._actions = [
            {"type": "old", "timestamp": old_ts},
        ]
        learner.log_action({"type": "new"})
        assert len(learner._actions) == 1
        assert learner._actions[0]["type"] == "new"

    def test_log_persists_to_file(self, learner):
        learner.log_action({"type": "persist_test"})
        assert learner.LOG_PATH.exists()
        data = json.loads(learner.LOG_PATH.read_text(encoding="utf-8"))
        assert len(data) == 1
        assert data[0]["type"] == "persist_test"


# ── Pattern detection ─────────────────────────────────────────────


class TestDetectPatterns:
    @pytest.mark.asyncio
    async def test_no_actions_empty(self, learner):
        patterns = await learner.detect_patterns()
        assert patterns == []

    @pytest.mark.asyncio
    async def test_below_min_repeat(self, learner):
        for _ in range(2):  # Below MIN_REPEAT=3
            learner.log_action({"type": "rare_action"})
        patterns = await learner.detect_patterns()
        assert len(patterns) == 0

    @pytest.mark.asyncio
    async def test_meets_min_repeat(self, learner):
        for _ in range(3):
            learner.log_action({"type": "frequent_action", "detail": "test"})
        patterns = await learner.detect_patterns()
        assert len(patterns) == 1
        assert patterns[0]["type"] == "frequent_action"
        assert patterns[0]["count"] == 3

    @pytest.mark.asyncio
    async def test_multiple_patterns(self, learner):
        for _ in range(5):
            learner.log_action({"type": "action_a"})
        for _ in range(3):
            learner.log_action({"type": "action_b"})
        for _ in range(1):
            learner.log_action({"type": "action_c"})

        patterns = await learner.detect_patterns()
        types = [p["type"] for p in patterns]
        assert "action_a" in types
        assert "action_b" in types
        assert "action_c" not in types  # below MIN_REPEAT

    @pytest.mark.asyncio
    async def test_sorted_by_count(self, learner):
        for _ in range(3):
            learner.log_action({"type": "less"})
        for _ in range(5):
            learner.log_action({"type": "more"})
        patterns = await learner.detect_patterns()
        assert patterns[0]["type"] == "more"
        assert patterns[1]["type"] == "less"

    @pytest.mark.asyncio
    async def test_pattern_has_peak_hours(self, learner):
        for i in range(5):
            learner.log_action({"type": "morning", "hour": 8})
        patterns = await learner.detect_patterns()
        assert 8 in patterns[0]["peak_hours"]

    @pytest.mark.asyncio
    async def test_pattern_has_detail_sample(self, learner):
        for i in range(3):
            learner.log_action({"type": "search", "detail": f"query_{i}"})
        patterns = await learner.detect_patterns()
        assert len(patterns[0]["detail_sample"]) == 3


# ── Frequency guessing ────────────────────────────────────────────


class TestGuessFrequency:
    def test_daily(self, learner):
        items = [
            {"date": "2026-02-10"},
            {"date": "2026-02-11"},
            {"date": "2026-02-12"},
            {"date": "2026-02-13"},
        ]
        assert learner._guess_frequency(items) == "daily"

    def test_weekly(self, learner):
        items = [
            {"date": "2026-02-03"},
            {"date": "2026-02-10"},
            {"date": "2026-02-17"},
        ]
        assert learner._guess_frequency(items) == "weekly"

    def test_irregular(self, learner):
        items = [
            {"date": "2026-01-01"},
            {"date": "2026-01-05"},
            {"date": "2026-01-20"},
        ]
        assert learner._guess_frequency(items) == "irregular"

    def test_single_item(self, learner):
        items = [{"date": "2026-02-10"}]
        assert learner._guess_frequency(items) == "irregular"

    def test_same_date(self, learner):
        items = [
            {"date": "2026-02-10"},
            {"date": "2026-02-10"},
        ]
        assert learner._guess_frequency(items) == "irregular"


# ── Proposal management ──────────────────────────────────────────


class TestProposals:
    @pytest.mark.asyncio
    async def test_propose_sends_to_telegram(self, learner, mock_telegram):
        for _ in range(5):
            learner.log_action({"type": "weather_check"})
        proposals = await learner.propose_skills()
        assert len(proposals) >= 1
        mock_telegram.send_message.assert_awaited()

    @pytest.mark.asyncio
    async def test_propose_marks_proposed(self, learner):
        for _ in range(5):
            learner.log_action({"type": "weather_check"})
        await learner.propose_skills()
        assert learner._already_proposed("weather_check")

    @pytest.mark.asyncio
    async def test_no_duplicate_proposals(self, learner, mock_telegram):
        for _ in range(5):
            learner.log_action({"type": "weather_check"})
        await learner.propose_skills()
        mock_telegram.send_message.reset_mock()

        # Second call should not re-propose
        proposals = await learner.propose_skills()
        assert len(proposals) == 0
        mock_telegram.send_message.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_propose_no_telegram(self, learner_no_deps):
        for _ in range(3):
            learner_no_deps.log_action({"type": "test_action"})
        # Should not raise even without telegram
        proposals = await learner_no_deps.propose_skills()
        assert isinstance(proposals, list)

    @pytest.mark.asyncio
    async def test_generate_proposal_no_router(self, learner_no_deps):
        pattern = {"type": "test", "count": 5, "frequency": "daily", "peak_hours": [9], "detail_sample": ["q1"]}
        msg = await learner_no_deps._generate_proposal(pattern)
        assert "test" in msg
        assert "5" in msg


# ── Skill creation ────────────────────────────────────────────────


class TestCreateSkill:
    @pytest.mark.asyncio
    async def test_create_skill_basic(self, learner):
        pattern = {
            "type": "weather_check",
            "count": 10,
            "frequency": "daily",
            "peak_hours": [8],
            "peak_weekdays": [0, 1, 2, 3, 4],
            "detail_sample": ["台北天氣"],
        }
        result = await learner.create_skill_from_pattern(pattern, "每天早上查天氣")
        assert result["name"] == "auto_weather_check"
        assert Path(result["path"]).exists()

        # Check skill.yaml was created
        import yaml
        config_path = Path(result["path"]) / "skill.yaml"
        assert config_path.exists()
        with open(config_path, encoding="utf-8") as f:
            config = yaml.safe_load(f)
        assert config["name"] == "auto_weather_check"
        assert config["category"] == "learned"

    @pytest.mark.asyncio
    async def test_create_skill_with_scheduler(self, learner):
        mock_scheduler = MagicMock()
        learner.scheduler = mock_scheduler

        pattern = {
            "type": "stock_check",
            "count": 5,
            "frequency": "daily",
            "peak_hours": [9],
            "peak_weekdays": [0, 1, 2, 3, 4],
        }
        await learner.create_skill_from_pattern(pattern)
        mock_scheduler.add_job.assert_called_once()


# ── Pattern to cron ───────────────────────────────────────────────


class TestPatternToCron:
    def test_daily_cron(self, learner):
        pattern = {"frequency": "daily", "peak_hours": [8]}
        cron = learner._pattern_to_cron(pattern)
        assert cron["hour"] == 8
        assert cron["minute"] == 0

    def test_weekly_cron(self, learner):
        pattern = {"frequency": "weekly", "peak_hours": [9], "peak_weekdays": [0, 2, 4]}
        cron = learner._pattern_to_cron(pattern)
        assert cron["hour"] == 9
        assert "day_of_week" in cron

    def test_irregular_with_peak(self, learner):
        pattern = {"frequency": "irregular", "peak_hours": [14]}
        cron = learner._pattern_to_cron(pattern)
        assert cron["hour"] == 14

    def test_no_peak_hours(self, learner):
        pattern = {"frequency": "irregular", "peak_hours": []}
        cron = learner._pattern_to_cron(pattern)
        assert cron == {}


# ── Top N helper ──────────────────────────────────────────────────


class TestTopN:
    def test_top_n_basic(self):
        result = SkillLearner._top_n([1, 1, 2, 2, 2, 3], 2)
        assert result == [2, 1]

    def test_top_n_empty(self):
        assert SkillLearner._top_n([], 3) == []


# ── Persistence ───────────────────────────────────────────────────


class TestPersistence:
    def test_load_missing_file(self, learner):
        learner.LOG_PATH = Path("/tmp/nonexistent_xyz.json")
        learner._load_actions()
        assert learner._actions == []

    def test_load_corrupt_json(self, learner):
        learner.LOG_PATH.write_text("not json", encoding="utf-8")
        learner._load_actions()
        assert learner._actions == []

    def test_round_trip(self, learner):
        learner.log_action({"type": "round_trip_test"})
        # Reload
        learner2 = SkillLearner()
        learner2.LOG_PATH = learner.LOG_PATH
        learner2._load_actions()
        assert len(learner2._actions) == 1
        assert learner2._actions[0]["type"] == "round_trip_test"

    def test_proposals_round_trip(self, learner):
        learner._mark_proposed("test_type")
        assert learner._already_proposed("test_type")
        # Reload proposals from disk
        proposals = learner._load_proposals()
        assert "test_type" in proposals
