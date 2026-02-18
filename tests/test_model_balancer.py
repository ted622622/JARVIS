"""Tests for core.model_balancer — token pool auto-balancing."""

import json

import pytest

from core.model_balancer import (
    _DEFAULT_POOLS,
    _load_pools,
    _save_pools,
    check_alert,
    get_status,
    record_usage,
    select_model,
)


@pytest.fixture(autouse=True)
def _clean_pool_file(tmp_path, monkeypatch):
    """Use a temp pool file for every test."""
    pool_file = tmp_path / "token_pools.json"
    import core.model_balancer as mb
    monkeypatch.setattr(mb, "_POOL_FILE", pool_file)
    yield


class TestSelectModel:
    def test_selects_higher_remaining(self):
        # 4.6v has more initial tokens → should be selected first
        model = select_model()
        assert model == "glm-4.6v"

    def test_switches_when_46v_used_more(self):
        record_usage("glm-4.6v", 3_000_000)
        # Now 4.6v remaining ~3M, 4.7 remaining ~4M → picks 4.7
        model = select_model()
        assert model == "glm-4.7"

    def test_returns_default_when_both_exhausted(self):
        record_usage("glm-4.6v", 6_000_000)
        record_usage("glm-4.7", 4_000_000)
        model = select_model()
        assert model == "glm-4.6v"


class TestRecordUsage:
    def test_records_and_persists(self):
        record_usage("glm-4.6v", 1000)
        pools = _load_pools()
        assert pools["glm-4.6v"]["estimated_used"] == 1000

    def test_accumulates(self):
        record_usage("glm-4.7", 500)
        record_usage("glm-4.7", 300)
        pools = _load_pools()
        assert pools["glm-4.7"]["estimated_used"] == 800

    def test_ignores_unknown_model(self):
        record_usage("unknown-model", 999)
        pools = _load_pools()
        assert "unknown-model" not in pools


class TestGetStatus:
    def test_shows_both_models(self):
        status = get_status()
        assert "glm-4.6v" in status
        assert "glm-4.7" in status
        assert "100%" in status

    def test_reflects_usage(self):
        record_usage("glm-4.6v", 2_997_151)  # ~50%
        status = get_status()
        assert "50%" in status


class TestCheckAlert:
    def test_no_alert_when_healthy(self):
        assert check_alert() is None

    def test_alert_below_20_pct(self):
        # Use 85% of 4.7 pool
        record_usage("glm-4.7", int(3_990_469 * 0.85))
        alert = check_alert()
        assert alert is not None
        assert "glm-4.7" in alert
        assert "⚠️" in alert

    def test_no_alert_at_21_pct(self):
        record_usage("glm-4.7", int(3_990_469 * 0.78))
        alert = check_alert()
        assert alert is None
