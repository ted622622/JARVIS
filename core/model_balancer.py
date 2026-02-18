"""Token pool auto-balancing — 4.6V / 4.7 round-robin by remaining quota.

Both models have similar capability; auto-select the one with more
remaining tokens to consume both pools evenly.
"""

from __future__ import annotations

import json
from pathlib import Path

from loguru import logger

_POOL_FILE = Path("data/token_pools.json")

_DEFAULT_POOLS = {
    "glm-4.6v": {"initial": 5_950_978, "estimated_used": 0},
    "glm-4.7":  {"initial": 3_990_469, "estimated_used": 0},
}


def _load_pools() -> dict:
    if _POOL_FILE.exists():
        try:
            return json.loads(_POOL_FILE.read_text(encoding="utf-8"))
        except Exception:
            pass
    return {k: dict(v) for k, v in _DEFAULT_POOLS.items()}


def _save_pools(pools: dict) -> None:
    _POOL_FILE.parent.mkdir(parents=True, exist_ok=True)
    _POOL_FILE.write_text(
        json.dumps(pools, indent=2, ensure_ascii=False), encoding="utf-8",
    )


def select_model() -> str:
    """Pick the model with more remaining tokens."""
    pools = _load_pools()
    remaining = {
        m: info["initial"] - info["estimated_used"]
        for m, info in pools.items()
    }
    best = max(remaining, key=remaining.get)
    if remaining[best] <= 0:
        return "glm-4.6v"  # both exhausted → default (pay-as-you-go)
    return best


def record_usage(model: str, tokens: int) -> None:
    """Record estimated token consumption."""
    pools = _load_pools()
    if model in pools:
        pools[model]["estimated_used"] += tokens
        _save_pools(pools)


def get_status() -> str:
    """One-line status for morning brief."""
    pools = _load_pools()
    parts = []
    for model, info in pools.items():
        remaining = info["initial"] - info["estimated_used"]
        pct = remaining / info["initial"] * 100 if info["initial"] else 0
        parts.append(f"{model}: {remaining:,} ({pct:.0f}%)")
    return " | ".join(parts)


def check_alert() -> str | None:
    """Return alert string if any pool is below 20%."""
    pools = _load_pools()
    alerts = []
    for model, info in pools.items():
        remaining = info["initial"] - info["estimated_used"]
        pct = remaining / info["initial"] * 100 if info["initial"] else 0
        if pct < 20:
            alerts.append(f"{model} 剩 {remaining:,} ({pct:.0f}%)")
    if alerts:
        return "⚠️ Token 低額度預警：" + "、".join(alerts)
    return None
