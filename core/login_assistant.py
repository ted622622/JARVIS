"""Login Assistant — guide the user through first-time login.

When a task fails because a website requires login, the assistant
asks the user for help (JARVIS style) instead of ghost-looping.
"""

from __future__ import annotations

from typing import Any

from loguru import logger

from core.session_manager import KNOWN_SITES, SessionManager


class LoginAssistant:
    """Handle login-required situations by requesting user help.

    Works with SessionManager to track which sites need login and
    which are already authenticated.
    """

    def __init__(self, session_manager: SessionManager):
        self.sessions = session_manager

    async def handle_login_required(
        self,
        site_key: str,
        *,
        telegram_client: Any = None,
        chat_id: int | str = 0,
    ) -> dict[str, Any]:
        """Detect login needed → ask user for help (JARVIS style).

        Returns:
            dict with status "waiting_for_login" or "needs_human_help".
        """
        site = KNOWN_SITES.get(site_key)
        site_name = site["name"] if site else site_key
        login_url = site["login_url"] if site else None

        if not site:
            # Unknown site — ask user to tell us how
            message = (
                f"Sir，{site_key} 的登入流程我尚未學習。\n"
                "請提供登入頁面網址，或由您先手動完成。\n"
                "我記錄後下次即可自行處理。"
            )
            if telegram_client:
                await _send(telegram_client, chat_id, message)
            return {"success": False, "status": "needs_human_help", "site_key": site_key}

        # Known site — provide login URL and ask for help
        message = (
            f"Sir，{site_name}需要您登入一次。\n"
            f"登入頁面: {login_url}\n\n"
            "請選擇：\n"
            "1. 在瀏覽器中手動完成登入\n"
            "2. 完成後回覆「登好了」即可\n\n"
            "登入後 session 會被記住，後續我可直接操作。"
        )
        if telegram_client:
            await _send(telegram_client, chat_id, message)

        return {
            "success": False,
            "status": "waiting_for_login",
            "site_key": site_key,
            "site_name": site_name,
        }

    async def on_user_confirms_login(self, site_key: str) -> str:
        """User said '登好了' — mark session as logged in.

        Returns a confirmation message.
        """
        site_name = self.sessions.get_site_name(site_key)
        self.sessions.mark_logged_in(site_key)
        logger.info(f"User confirmed login for {site_key}")
        return f"Sir，{site_name}登入已確認。此 session 已記錄，後續我可直接操作。"

    def detect_site_from_url(self, url: str) -> str | None:
        """Try to identify a known site from a URL.

        Returns site_key if matched, None otherwise.
        """
        url_lower = url.lower()
        for key, site in KNOWN_SITES.items():
            domain = site.get("cookie_domain", "").lstrip(".")
            if domain and domain in url_lower:
                return key
        return None

    def detect_login_confirmation(self, message: str) -> str | None:
        """Check if the user's message confirms a login.

        Returns the site_key the user is confirming, or None.
        """
        triggers = ["登好了", "登入好了", "已登入", "登完了", "好了已經登入"]
        msg = message.strip()
        for trigger in triggers:
            if trigger in msg:
                # Find the most recent pending login
                for key, info in self.sessions.all_status().items():
                    if not info.get("logged_in", False):
                        return key
                return None
        return None


async def _send(telegram_client: Any, chat_id: int | str, text: str) -> None:
    """Helper to send a Telegram message (tolerates missing method)."""
    try:
        if hasattr(telegram_client, "send_message"):
            await telegram_client.send_message(text, chat_id=chat_id)
    except Exception as exc:
        logger.warning(f"LoginAssistant send failed: {exc}")
