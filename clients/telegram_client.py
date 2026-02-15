"""Telegram Bot client — dual bot support (JARVIS / Clawra).

Features:
- Send text messages (Markdown)
- Send photos with captions
- Inline keyboard Y/N confirmation (for SecurityGate)
- Callback query handling
"""

from __future__ import annotations

import asyncio
from typing import Any

from loguru import logger

try:
    from telegram import Bot, InlineKeyboardButton, InlineKeyboardMarkup, Update
    from telegram.constants import ParseMode
    from telegram.ext import (
        Application,
        CallbackQueryHandler,
        CommandHandler,
        MessageHandler,
        filters,
    )

    _HAS_PTB = True
except ImportError:
    _HAS_PTB = False
    logger.warning("python-telegram-bot not installed, TelegramClient will be non-functional")


class TelegramClient:
    """Dual-bot Telegram client for J.A.R.V.I.S. and Clawra.

    Usage:
        client = TelegramClient(
            jarvis_token="...",
            clawra_token="...",
            chat_id=123456789,
        )
        await client.init()
        await client.send("Hello from JARVIS!")
        await client.send("Hello from Clawra!", persona="clawra")
    """

    def __init__(
        self,
        jarvis_token: str = "",
        clawra_token: str = "",
        chat_id: int | str = 0,
    ):
        self._jarvis_token = jarvis_token
        self._clawra_token = clawra_token
        self.chat_id = int(chat_id) if chat_id else 0

        self._jarvis_bot: Bot | None = None
        self._clawra_bot: Bot | None = None
        self._app: Application | None = None

        # Pending confirmation callbacks: {callback_id: asyncio.Future}
        self._pending_confirms: dict[str, asyncio.Future] = {}
        self._confirm_counter = 0

    async def init(self) -> None:
        """Initialize bot instances."""
        if not _HAS_PTB:
            logger.warning("Telegram bots not initialized (missing python-telegram-bot)")
            return

        if self._jarvis_token:
            self._jarvis_bot = Bot(token=self._jarvis_token)
            logger.info("JARVIS Telegram bot initialized")

        if self._clawra_token:
            self._clawra_bot = Bot(token=self._clawra_token)
            logger.info("Clawra Telegram bot initialized")

    def _get_bot(self, persona: str = "jarvis") -> Bot | None:
        if persona == "clawra" and self._clawra_bot:
            return self._clawra_bot
        return self._jarvis_bot

    # ── Send Messages ───────────────────────────────────────────

    async def send(
        self,
        text: str,
        *,
        persona: str = "jarvis",
        chat_id: int | None = None,
        parse_mode: str | None = None,
    ) -> int | None:
        """Send a text message. Returns message_id on success."""
        bot = self._get_bot(persona)
        if not bot:
            logger.warning(f"No bot available for persona={persona}")
            return None

        target = chat_id or self.chat_id
        if not target:
            logger.warning("No chat_id configured")
            return None

        try:
            msg = await bot.send_message(
                chat_id=target,
                text=text,
                parse_mode=parse_mode,
            )
            return msg.message_id
        except Exception as e:
            logger.error(f"Failed to send Telegram message: {e}")
            return None

    async def send_photo(
        self,
        photo_url: str,
        caption: str = "",
        *,
        persona: str = "jarvis",
        chat_id: int | None = None,
    ) -> int | None:
        """Send a photo with optional caption."""
        bot = self._get_bot(persona)
        if not bot:
            return None

        target = chat_id or self.chat_id
        if not target:
            return None

        try:
            msg = await bot.send_photo(
                chat_id=target,
                photo=photo_url,
                caption=caption,
            )
            return msg.message_id
        except Exception as e:
            logger.error(f"Failed to send Telegram photo: {e}")
            return None

    # ── Confirmation Flow (for SecurityGate) ────────────────────

    async def request_confirmation(
        self,
        prompt: str,
        *,
        persona: str = "jarvis",
        chat_id: int | None = None,
    ) -> bool:
        """Send a Y/N inline keyboard and wait for response.

        This is designed to be used as SecurityGate's confirm_callback.
        Returns True if user approved, False if denied.
        """
        bot = self._get_bot(persona)
        if not bot:
            logger.warning("No bot for confirmation, auto-denying")
            return False

        target = chat_id or self.chat_id
        if not target:
            return False

        self._confirm_counter += 1
        callback_id = f"confirm_{self._confirm_counter}"

        keyboard = InlineKeyboardMarkup([
            [
                InlineKeyboardButton("✅ 允許", callback_data=f"{callback_id}_yes"),
                InlineKeyboardButton("❌ 拒絕", callback_data=f"{callback_id}_no"),
            ]
        ])

        future: asyncio.Future[bool] = asyncio.get_event_loop().create_future()
        self._pending_confirms[callback_id] = future

        try:
            await bot.send_message(
                chat_id=target,
                text=prompt,
                reply_markup=keyboard,
            )
            # The caller (SecurityGate) handles timeout via asyncio.wait_for
            return await future
        except asyncio.CancelledError:
            return False
        finally:
            self._pending_confirms.pop(callback_id, None)

    async def handle_callback_query(self, update: Update, context: Any) -> None:
        """Process inline keyboard button presses."""
        query = update.callback_query
        if not query or not query.data:
            return

        await query.answer()
        data = query.data  # e.g. "confirm_1_yes" or "confirm_1_no"

        # Parse callback_id and decision
        parts = data.rsplit("_", 1)
        if len(parts) != 2:
            return

        callback_id = parts[0]
        decision = parts[1]  # "yes" or "no"

        future = self._pending_confirms.get(callback_id)
        if future and not future.done():
            future.set_result(decision == "yes")
            status = "✅ 已允許" if decision == "yes" else "❌ 已拒絕"
            await query.edit_message_text(
                text=f"{query.message.text}\n\n{status}",
            )

    # ── Polling (for standalone operation) ──────────────────────

    def build_application(self, token: str | None = None) -> Application | None:
        """Build a telegram Application with handlers for polling mode.

        Call this if you want to run the bot with long-polling.
        """
        if not _HAS_PTB:
            return None

        tok = token or self._jarvis_token
        if not tok:
            return None

        app = Application.builder().token(tok).build()
        app.add_handler(CallbackQueryHandler(self.handle_callback_query))
        self._app = app
        return app

    async def close(self) -> None:
        """Clean up pending confirmations."""
        for future in self._pending_confirms.values():
            if not future.done():
                future.cancel()
        self._pending_confirms.clear()
