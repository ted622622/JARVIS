"""Code Worker — deep development tasks via Claude Code / LLM.

Handles complex code generation, refactoring, and analysis
by routing through the CEO model.
"""

from __future__ import annotations

from typing import Any

from loguru import logger

from clients.base_client import ChatMessage
from core.model_router import ModelRole


class CodeWorker:
    """Worker for deep programming tasks.

    Usage:
        worker = CodeWorker(model_router=router)
        result = await worker.execute("寫一個排序函數")
    """

    def __init__(self, model_router: Any = None):
        self.router = model_router
        self.name = "code"

    async def execute(self, task: str, **kwargs: Any) -> dict[str, Any]:
        """Execute a code-related task.

        Args:
            task: description of the coding task
            **kwargs: language, context, etc.

        Returns:
            dict with code output and metadata
        """
        if not self.router:
            return {"error": "No model router configured"}

        language = kwargs.get("language", "python")
        context = kwargs.get("context", "")

        system = (
            f"你是一位專業的 {language} 程式開發者。\n"
            "寫出乾淨、可維護的程式碼。\n"
            "回覆格式：先簡要說明方案，再附上程式碼。"
        )

        messages = [
            ChatMessage(role="system", content=system),
        ]
        if context:
            messages.append(ChatMessage(role="user", content=f"上下文:\n{context}"))
        messages.append(ChatMessage(role="user", content=task))

        response = await self.router.chat(
            messages, role=ModelRole.CEO, **kwargs
        )

        logger.debug(f"CodeWorker completed: {task[:50]}...")
        return {
            "result": response.content,
            "model": response.model,
            "worker": self.name,
        }
