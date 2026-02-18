"""Telegram Bot client — dual bot support (JARVIS / Clawra).

Features:
- Send text messages (Markdown)
- Send photos with captions
- Send / receive voice messages (TTS + STT)
- Inline keyboard Y/N confirmation (for SecurityGate)
- Callback query handling
- Clawra typing delay for humanization
"""

from __future__ import annotations

import asyncio
import random
import tempfile
from pathlib import Path
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
        allowed_user_ids: str = "",
    ):
        self._jarvis_token = jarvis_token
        self._clawra_token = clawra_token
        self.chat_id = int(chat_id) if chat_id else 0

        # Whitelist: only these user IDs can interact
        self._allowed_user_ids: set[int] = set()
        if allowed_user_ids:
            for uid in allowed_user_ids.split(","):
                uid = uid.strip()
                if uid.isdigit():
                    self._allowed_user_ids.add(int(uid))

        self._jarvis_bot: Bot | None = None
        self._clawra_bot: Bot | None = None
        self._jarvis_app: Application | None = None
        self._clawra_app: Application | None = None

        # Pending confirmation callbacks: {callback_id: asyncio.Future}
        self._pending_confirms: dict[str, asyncio.Future] = {}
        self._confirm_counter = 0

        # Message handler callback: async fn(user_message: str, chat_id: int, persona: str) -> str
        self._on_message = None
        # Map bot token → persona for incoming message routing
        self._token_to_persona: dict[str, str] = {}

        # Patch O: CEO reference for complexity estimation
        self._ceo_ref = None

        # Patch R: Message batching — accumulate rapid messages
        self._msg_batch: dict[int, dict[str, Any]] = {}  # chat_id → batch info
        self._BATCH_DELAY = 6.0  # seconds to wait for more messages

        # Voice components (injected after init)
        self._voice_worker = None
        self._stt_client = None
        self._transcribe_worker = None
        self._voice_cache_dir = Path("./data/voice_cache")
        self._voice_cache_dir.mkdir(parents=True, exist_ok=True)

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

    @property
    def voice_worker(self):
        return self._voice_worker

    @voice_worker.setter
    def voice_worker(self, worker) -> None:
        self._voice_worker = worker

    @property
    def stt_client(self):
        return self._stt_client

    @stt_client.setter
    def stt_client(self, client) -> None:
        self._stt_client = client

    @property
    def transcribe_worker(self):
        return self._transcribe_worker

    @transcribe_worker.setter
    def transcribe_worker(self, worker) -> None:
        self._transcribe_worker = worker

    def set_ceo_ref(self, ceo) -> None:
        """Set reference to CEO agent for task complexity estimation."""
        self._ceo_ref = ceo

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

    async def send_voice(
        self,
        audio_path: str,
        *,
        persona: str = "jarvis",
        chat_id: int | None = None,
    ) -> int | None:
        """Send a voice message from a local audio file.

        Args:
            audio_path: Path to the audio file (MP3/OGA)
            persona: Which bot sends it
            chat_id: Target chat (defaults to self.chat_id)

        Returns:
            message_id on success, None on failure
        """
        bot = self._get_bot(persona)
        if not bot:
            return None

        target = chat_id or self.chat_id
        if not target:
            return None

        try:
            with open(audio_path, "rb") as f:
                msg = await bot.send_voice(chat_id=target, voice=f)
            return msg.message_id
        except Exception as e:
            logger.error(f"Failed to send voice message: {e}")
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

        future: asyncio.Future[bool] = asyncio.get_running_loop().create_future()
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

        # Whitelist check
        user_id = query.from_user.id if query.from_user else None
        if not self._is_authorized(user_id):
            logger.warning(f"Unauthorized callback from user {user_id} blocked")
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

    def set_message_handler(self, callback) -> None:
        """Set the callback for incoming user messages.

        Args:
            callback: async fn(user_message: str, chat_id: int) -> str
        """
        self._on_message = callback

    def _is_authorized(self, user_id: int | None) -> bool:
        """Check if a user ID is in the whitelist."""
        if not self._allowed_user_ids:
            return True  # No whitelist configured → allow all
        return user_id in self._allowed_user_ids

    _TEXT_REPLY_KEYWORDS = (
        "用文字", "文字回", "打字回", "不要語音", "不要用語音",
        "用打字", "text reply", "用 text", "傳文字", "回文字",
    )

    @staticmethod
    def _wants_text_reply(transcribed: str) -> bool:
        """Detect if user's voice message requests a text reply."""
        lower = transcribed.lower()
        return any(kw in lower for kw in TelegramClient._TEXT_REPLY_KEYWORDS)

    async def _simulate_typing(
        self, bot: Any, chat_id: int, text: str,
    ) -> None:
        """Simulate human typing delay for Clawra persona.

        Delay is 5–45 seconds, scaled by response length.
        Sends 'typing' chat action every 4s so the indicator stays visible.
        """
        # ~0.3s per character, clamped to [5, 45], with a small random jitter
        base = min(5 + len(text) * 0.3, 45)
        delay = base + random.uniform(-2, 2)
        delay = max(5, min(delay, 45))
        logger.debug(f"Clawra typing delay: {delay:.1f}s for {len(text)} chars")

        elapsed = 0.0
        while elapsed < delay:
            try:
                await bot.send_chat_action(chat_id=chat_id, action="typing")
            except Exception:
                pass  # non-critical
            wait = min(4.0, delay - elapsed)
            await asyncio.sleep(wait)
            elapsed += wait

    async def _send_long_text(self, update: Update, text: str) -> None:
        """Split long replies into <=4000-char chunks for Telegram."""
        if len(text) > 4000:
            for i in range(0, len(text), 4000):
                await update.message.reply_text(text[i:i + 4000])
        else:
            await update.message.reply_text(text)

    async def _handle_text_message(self, update: Update, context: Any) -> None:
        """Accumulate rapid messages into a batch, then process together.

        Patch R: Instead of processing each message immediately, we wait
        _BATCH_DELAY seconds for more messages. This creates natural
        couple-chat flow where 2-3 rapid messages are handled as one.
        """
        if not update.message or not update.message.text:
            return

        # Whitelist check — must be first
        user_id = update.message.from_user.id if update.message.from_user else None
        if not self._is_authorized(user_id):
            logger.warning(f"Unauthorized user {user_id} blocked")
            return

        chat_id = update.message.chat_id
        user_text = update.message.text
        user_name = update.message.from_user.first_name if update.message.from_user else "User"

        bot_token = context.bot.token
        persona = self._token_to_persona.get(bot_token, "jarvis")

        logger.info(f"[{persona}] Telegram received from {user_name}: {user_text[:80]}")

        if not self._on_message:
            await update.message.reply_text("收到，但 CEO Agent 尚未就緒。")
            return

        # ── Patch R: Batch accumulation ──
        if chat_id in self._msg_batch:
            batch = self._msg_batch[chat_id]
            batch["messages"].append(user_text)
        else:
            batch = {
                "messages": [user_text],
                "task": None,
                "persona": persona,
                "bot": context.bot,
            }
            self._msg_batch[chat_id] = batch

        # Cancel previous batch timer, start fresh
        if batch["task"] and not batch["task"].done():
            batch["task"].cancel()
        batch["task"] = asyncio.create_task(self._process_batch(chat_id))

        # Immediate mode (no batching): await inline — used for testing
        if self._BATCH_DELAY <= 0:
            await batch["task"]

    async def _process_batch(self, chat_id: int) -> None:
        """Wait for batch delay, then process accumulated messages."""
        try:
            await asyncio.sleep(self._BATCH_DELAY)
        except asyncio.CancelledError:
            return  # Timer reset by new incoming message

        batch = self._msg_batch.pop(chat_id, None)
        if not batch or not batch["messages"]:
            return

        persona = batch["persona"]
        bot = batch["bot"]
        messages = batch["messages"]
        combined = "\n".join(messages)

        if len(messages) > 1:
            logger.info(f"[{persona}] Batch: {len(messages)} messages combined")

        # ── Patch O: Long-task acknowledgement ──
        if self._ceo_ref and hasattr(self._ceo_ref, "estimate_complexity"):
            try:
                complexity = self._ceo_ref.estimate_complexity(combined)
                if complexity["is_long"]:
                    est = complexity["estimate_seconds"]
                    if persona == "clawra":
                        ack = f"欸收到！讓我看看...大概要等個 {est} 秒左右喔"
                    else:
                        ack = f"收到，正在處理中。預計需要 {est} 秒。"
                    await bot.send_message(chat_id=chat_id, text=ack)
            except Exception as e:
                logger.debug(f"Complexity estimation failed: {e}")

        try:
            reply = await self._on_message(combined, chat_id, persona)

            # Clawra typing delay — humanize response timing
            if persona == "clawra":
                reply_text = reply.get("text", "") if isinstance(reply, dict) else (reply or "")
                await self._simulate_typing(bot, chat_id, reply_text)

            await self._send_reply(bot, chat_id, persona, reply)
        except Exception as e:
            logger.error(f"[{persona}] Batch handler error: {e}")
            fallback = (
                "欸...我這邊好像出了點小狀況，等我一下喔"
                if persona == "clawra"
                else "Sir, 系統暫時出了點問題，我正在處理中。"
            )
            try:
                await bot.send_message(chat_id=chat_id, text=fallback)
            except Exception:
                pass

    async def _send_reply(
        self, bot: Any, chat_id: int, persona: str, reply,
    ) -> None:
        """Send CEO reply. Clawra splits long text into multiple messages."""
        _empty_fallback = (
            "嗯...我想了一下但不太確定怎麼回，你可以換個方式問問看嗎"
            if persona == "clawra"
            else "Sir, 我已處理完成，但未能產生有效回覆。請換個方式描述您的需求。"
        )

        if isinstance(reply, dict):
            photo_url = reply.get("photo_url")
            text = reply.get("text", "")
            phone = reply.get("phone")
            booking_url = reply.get("booking_url")

            if photo_url:
                try:
                    await bot.send_photo(
                        chat_id=chat_id, photo=photo_url, caption=text or None,
                    )
                except Exception as e:
                    logger.warning(f"Failed to send photo, sending as text: {e}")
                    await self._send_split_text(bot, chat_id, persona, text or _empty_fallback)
            else:
                await self._send_split_text(bot, chat_id, persona, text or _empty_fallback)

            if phone:
                await bot.send_message(chat_id=chat_id, text=phone)
            if booking_url:
                await bot.send_message(chat_id=chat_id, text=booking_url)
        else:
            await self._send_split_text(bot, chat_id, persona, reply or _empty_fallback)

    async def _send_split_text(
        self, bot: Any, chat_id: int, persona: str, text: str,
    ) -> None:
        """Send text with paragraph splitting for Clawra, plain for JARVIS.

        Clawra messages with multiple paragraphs are split into separate
        Telegram messages with short typing delays between them, mimicking
        natural chat behavior.
        """
        # Clawra: split at paragraph breaks for natural multi-message flow
        if persona == "clawra" and "\n\n" in text and len(text) > 60:
            parts = [p.strip() for p in text.split("\n\n") if p.strip()]

            # Merge into max 3 groups if too many paragraphs
            if len(parts) > 3:
                third = max(1, len(parts) // 3)
                parts = [
                    "\n\n".join(parts[:third]),
                    "\n\n".join(parts[third : 2 * third]),
                    "\n\n".join(parts[2 * third :]),
                ]

            for i, part in enumerate(parts):
                if i > 0:
                    delay = random.uniform(2, 4)
                    try:
                        await bot.send_chat_action(chat_id=chat_id, action="typing")
                    except Exception:
                        pass
                    await asyncio.sleep(delay)

                for chunk_start in range(0, len(part), 4000):
                    await bot.send_message(
                        chat_id=chat_id, text=part[chunk_start : chunk_start + 4000],
                    )
        else:
            # JARVIS or short Clawra reply: single message
            if len(text) > 4000:
                for i in range(0, len(text), 4000):
                    await bot.send_message(chat_id=chat_id, text=text[i : i + 4000])
            else:
                await bot.send_message(chat_id=chat_id, text=text)

    async def _handle_voice_message(self, update: Update, context: Any) -> None:
        """Process incoming voice messages: STT → CEO → TTS → reply voice."""
        if not update.message or not update.message.voice:
            return

        # Whitelist check
        user_id = update.message.from_user.id if update.message.from_user else None
        if not self._is_authorized(user_id):
            logger.warning(f"Unauthorized voice from user {user_id} blocked")
            return

        chat_id = update.message.chat_id
        bot_token = context.bot.token
        persona = self._token_to_persona.get(bot_token, "jarvis")
        user_name = update.message.from_user.first_name if update.message.from_user else "User"

        logger.info(f"[{persona}] Voice message received from {user_name}")

        # Check STT client availability
        if not self._stt_client:
            if persona == "clawra":
                await update.message.reply_text("欸我現在還聽不懂語音啦，先打字給我好不好")
            else:
                await update.message.reply_text("Sir, 語音辨識模組尚未啟用，請先設定 GROQ_API_KEY。")
            return

        # Download voice file
        try:
            voice_file = await update.message.voice.get_file()
            oga_path = self._voice_cache_dir / f"incoming_{update.message.message_id}.ogg"
            await voice_file.download_to_drive(str(oga_path))
            logger.debug(f"Voice downloaded: {oga_path}")
        except Exception as e:
            logger.error(f"Failed to download voice: {e}")
            if persona == "clawra":
                await update.message.reply_text("啊...語音好像沒收到耶，再傳一次給我好嗎")
            else:
                await update.message.reply_text("Sir, 語音下載失敗，請再試一次。")
            return

        # STT: transcribe voice to text
        try:
            transcribed = await self._stt_client.transcribe(str(oga_path))
        except Exception as e:
            logger.error(f"STT failed: {e}")
            if persona == "clawra":
                await update.message.reply_text("嗚嗚我沒聽清楚啦，再說一次好不好")
            else:
                await update.message.reply_text("Sir, 語音辨識失敗，請再試一次。")
            return
        finally:
            # Clean up downloaded file
            oga_path.unlink(missing_ok=True)

        if not transcribed.strip():
            if persona == "clawra":
                await update.message.reply_text("欸？好像什麼都沒聽到耶，你有在說話嗎")
            else:
                await update.message.reply_text("Sir, 未偵測到語音內容。")
            return

        logger.info(f"[{persona}] STT result: {transcribed[:80]}")

        # Detect if user requests text reply in their voice message
        force_text = self._wants_text_reply(transcribed)

        # CEO Agent processes the transcribed text
        if not self._on_message:
            return

        try:
            reply = await self._on_message(transcribed, chat_id, persona)

            reply_text = ""
            emotion = None
            phone = None
            booking_url = None
            photo_url = None
            reply_mode = None
            if isinstance(reply, dict):
                reply_text = reply.get("text", "")
                emotion = reply.get("emotion")
                phone = reply.get("phone")
                booking_url = reply.get("booking_url")
                photo_url = reply.get("photo_url")
                reply_mode = reply.get("reply_mode")
            else:
                reply_text = reply or ""

            # Selfie photo — send photo with caption, skip TTS
            if photo_url:
                try:
                    bot = update.message.get_bot()
                    await bot.send_photo(
                        chat_id=chat_id, photo=photo_url,
                        caption=reply_text or None,
                    )
                except Exception as e:
                    logger.warning(f"Failed to send photo via voice handler: {e}")
                    if reply_text:
                        await update.message.reply_text(reply_text)
                return

            if not reply_text:
                return

            # Determine reply mode: text or voice
            use_text = force_text or reply_mode == "text"

            if use_text or not self._voice_worker:
                # Text reply (user requested or no TTS available)
                await update.message.reply_text(reply_text)
            else:
                # TTS: reply with voice, fallback to text
                try:
                    audio_path = await self._voice_worker.text_to_speech(
                        reply_text, persona=persona, emotion=emotion,
                    )
                    await self.send_voice(
                        audio_path, persona=persona, chat_id=chat_id
                    )
                except Exception as e:
                    logger.warning(f"TTS reply failed, falling back to text: {e}")
                    await update.message.reply_text(reply_text)

            # K3: Send phone/booking_url as separate clickable messages
            if phone:
                await self.send(phone, persona=persona, chat_id=chat_id)
            if booking_url:
                await self.send(booking_url, persona=persona, chat_id=chat_id)

        except Exception as e:
            logger.error(f"[{persona}] Voice message handler error: {e}")
            if persona == "clawra":
                await update.message.reply_text("欸...語音處理好像出了點小問題，等我一下喔")
            else:
                await update.message.reply_text("Sir, 語音處理暫時出了點問題，我正在處理中。")

    async def _handle_audio_document(self, update: Update, context: Any) -> None:
        """Process incoming audio documents (long recordings) via TranscribeWorker."""
        if not update.message:
            return

        # Determine which attachment we got
        audio = update.message.audio or update.message.document
        if not audio:
            return

        user_id = update.message.from_user.id if update.message.from_user else None
        if not self._is_authorized(user_id):
            logger.warning(f"Unauthorized audio from user {user_id} blocked")
            return

        chat_id = update.message.chat_id
        bot_token = context.bot.token
        persona = self._token_to_persona.get(bot_token, "jarvis")

        if not self._transcribe_worker:
            await update.message.reply_text(
                "Sir，長音檔轉錄模組尚未啟用。" if persona == "jarvis"
                else "欸我現在還沒辦法處理長音檔耶"
            )
            return

        # Check file extension
        file_name = audio.file_name or ""
        suffix = Path(file_name).suffix.lower() if file_name else ""
        mime = getattr(audio, "mime_type", "") or ""
        is_audio = suffix in {".ogg", ".mp3", ".m4a", ".wav", ".flac", ".opus"} or mime.startswith("audio/")

        if not is_audio:
            return  # Not an audio file, ignore

        await update.message.reply_text(
            "收到音檔，正在轉錄中，請稍候 ..." if persona == "jarvis"
            else "收到！讓我聽聽看，等我一下喔"
        )

        # Download audio file
        try:
            dl_file = await audio.get_file()
            ext = suffix or ".ogg"
            local_path = self._voice_cache_dir / f"transcribe_{update.message.message_id}{ext}"
            await dl_file.download_to_drive(str(local_path))
        except Exception as e:
            logger.error(f"Failed to download audio document: {e}")
            await update.message.reply_text("Sir，音檔下載失敗，請再試一次。")
            return

        # Transcribe + summarize
        try:
            caption = update.message.caption or ""
            result = await self._transcribe_worker.process_audio(
                str(local_path), context=caption,
            )

            if result.get("error"):
                await update.message.reply_text(f"轉錄失敗: {result['error']}")
            else:
                summary = result.get("result", "")
                if summary:
                    # Split long messages (Telegram 4096 char limit)
                    for i in range(0, len(summary), 4000):
                        await update.message.reply_text(summary[i:i + 4000])
                else:
                    await update.message.reply_text("轉錄完成但未產生摘要。")
        except Exception as e:
            logger.error(f"Transcribe worker failed: {e}")
            await update.message.reply_text("Sir，轉錄處理出了點問題。")
        finally:
            local_path.unlink(missing_ok=True)

    def build_applications(self) -> list[Application]:
        """Build telegram Applications for all configured bots.

        Returns a list of Applications ready for polling.
        """
        if not _HAS_PTB:
            return []

        apps = []

        for token, persona in [
            (self._jarvis_token, "jarvis"),
            (self._clawra_token, "clawra"),
        ]:
            if not token:
                continue
            self._token_to_persona[token] = persona
            app = Application.builder().token(token).build()
            app.add_handler(MessageHandler(
                filters.TEXT & ~filters.COMMAND, self._handle_text_message,
            ))
            app.add_handler(MessageHandler(
                filters.VOICE, self._handle_voice_message,
            ))
            app.add_handler(MessageHandler(
                filters.AUDIO | filters.Document.AUDIO, self._handle_audio_document,
            ))
            app.add_handler(CallbackQueryHandler(self.handle_callback_query))
            if persona == "jarvis":
                self._jarvis_app = app
            else:
                self._clawra_app = app
            apps.append(app)
            logger.info(f"Telegram Application built for {persona}")

        return apps

    async def close(self) -> None:
        """Clean up pending confirmations."""
        for future in self._pending_confirms.values():
            if not future.done():
                future.cancel()
        self._pending_confirms.clear()
