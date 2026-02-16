"""Session Manager — track login state for external websites.

Persists a simple JSON status file so JARVIS knows which sites the
user has already logged into and which need a first-time login flow.
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

from loguru import logger

# ── Known sites and their login metadata ────────────────────────

KNOWN_SITES: dict[str, dict[str, str]] = {
    "thsrc": {
        "name": "台灣高鐵",
        "login_url": "https://irs.thsrc.com.tw/IMINT/",
        "cookie_domain": ".thsrc.com.tw",
    },
    "inline": {
        "name": "inline 訂位",
        "login_url": "https://inline.app/",
        "cookie_domain": ".inline.app",
    },
    "google": {
        "name": "Google",
        "login_url": "https://accounts.google.com/",
        "cookie_domain": ".google.com",
    },
}


class SessionManager:
    """Track which external sites the user has logged into.

    State is persisted to *status_path* as a JSON file.
    """

    def __init__(self, status_path: str = "./data/session_status.json"):
        self.status_path = Path(status_path)
        self._status: dict[str, dict[str, Any]] = self._load()

    # ── Queries ─────────────────────────────────────────────────

    def is_logged_in(self, site_key: str) -> bool:
        """Check if the user has an active session for *site_key*."""
        return self._status.get(site_key, {}).get("logged_in", False)

    def get_site_name(self, site_key: str) -> str:
        """Human-readable name for *site_key*."""
        site = KNOWN_SITES.get(site_key)
        if site:
            return site["name"]
        return self._status.get(site_key, {}).get("name", site_key)

    def get_login_url(self, site_key: str) -> str | None:
        """Login URL for *site_key*, or None if unknown."""
        site = KNOWN_SITES.get(site_key)
        return site["login_url"] if site else None

    def all_status(self) -> dict[str, dict[str, Any]]:
        """Return a copy of all session statuses."""
        return dict(self._status)

    # ── Mutations ───────────────────────────────────────────────

    def mark_logged_in(self, site_key: str, name: str | None = None) -> None:
        """Record that the user has logged in to *site_key*."""
        self._status[site_key] = {
            "logged_in": True,
            "ts": datetime.now().isoformat(),
            "name": name or self.get_site_name(site_key),
        }
        self._save()
        logger.info(f"Session marked logged-in: {site_key}")

    def mark_expired(self, site_key: str) -> None:
        """Record that the session for *site_key* has expired."""
        entry = self._status.get(site_key, {})
        entry["logged_in"] = False
        entry["expired_at"] = datetime.now().isoformat()
        self._status[site_key] = entry
        self._save()
        logger.info(f"Session marked expired: {site_key}")

    # ── Persistence ─────────────────────────────────────────────

    def _load(self) -> dict[str, dict[str, Any]]:
        if self.status_path.exists():
            try:
                return json.loads(self.status_path.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError) as exc:
                logger.warning(f"Session status load failed: {exc}")
        return {}

    def _save(self) -> None:
        self.status_path.parent.mkdir(parents=True, exist_ok=True)
        self.status_path.write_text(
            json.dumps(self._status, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
