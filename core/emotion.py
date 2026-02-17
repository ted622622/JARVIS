"""Emotion classifier — lightweight sentiment detection via LLM.

Classifies user messages into emotion labels and writes to MemOS.
"""

from __future__ import annotations

from typing import Any

from loguru import logger

from clients.base_client import ChatMessage, ChatResponse

# Valid emotion labels
EMOTION_LABELS = ("anxious", "tired", "sad", "frustrated", "normal", "happy", "excited")

_CLASSIFY_PROMPT = (
    "你是一個情緒分類器。根據以下用戶訊息，輸出一個情緒標籤。\n"
    "只能輸出以下其中一個（不要輸出任何其他文字）：\n"
    "anxious, tired, sad, frustrated, normal, happy, excited\n\n"
    "用戶訊息：{message}"
)


class EmotionClassifier:
    """Classify user messages into emotion labels.

    Usage:
        classifier = EmotionClassifier(model_router, memos)
        label = await classifier.classify("我好累喔，今天加班到很晚")
        # → "tired"
    """

    def __init__(self, model_router: Any = None, memos: Any = None):
        self.router = model_router
        self.memos = memos

    async def classify(self, user_message: str) -> str:
        """Classify a user message into an emotion label.

        Falls back to "normal" if classification fails.
        """
        if not self.router:
            return "normal"

        try:
            from core.model_router import ModelRole

            prompt = _CLASSIFY_PROMPT.format(message=user_message)
            response = await self.router.chat(
                [ChatMessage(role="user", content=prompt)],
                role=ModelRole.CEO,
                task_type="template",
                max_tokens=20,
                temperature=0.1,
            )

            label = response.content.strip().lower()

            # Validate — only accept known labels
            if label not in EMOTION_LABELS:
                # Try to extract a valid label from the response
                for valid in EMOTION_LABELS:
                    if valid in label:
                        label = valid
                        break
                else:
                    logger.debug(f"Unknown emotion label '{label}', defaulting to normal")
                    label = "normal"

            # Write to MemOS
            if self.memos and self.memos.working_memory:
                await self.memos.working_memory.set(
                    "user_emotion", label, agent_id="emotion_classifier"
                )

            logger.debug(f"Emotion classified: '{user_message[:30]}...' → {label}")
            return label

        except Exception as e:
            logger.warning(f"Emotion classification failed: {e}")
            return "normal"

    async def get_current_emotion(self) -> str:
        """Read the last classified emotion from MemOS."""
        if self.memos and self.memos.working_memory:
            return await self.memos.working_memory.get("user_emotion", "unknown")
        return "unknown"
