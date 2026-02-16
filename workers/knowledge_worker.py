"""Knowledge Worker — last-resort LLM + memory fallback.

When all other workers fail, this worker answers using the LLM's
built-in knowledge plus any relevant memories from MemorySearch.
"""

from __future__ import annotations

from typing import Any

from loguru import logger


class KnowledgeWorker:
    """Final fallback: answer using LLM knowledge + stored memories.

    Usage:
        worker = KnowledgeWorker(model_router=router, memos=memos, memory_search=search)
        result = await worker.execute("台北天氣", failed_attempts=[...])
    """

    def __init__(
        self,
        model_router: Any = None,
        memos: Any = None,
        memory_search: Any = None,
    ):
        self.router = model_router
        self.memos = memos
        self.memory_search = memory_search
        self.name = "knowledge"

    async def execute(self, task: str, **kwargs: Any) -> dict[str, Any]:
        """Answer a question using LLM knowledge and memory.

        Args:
            task: The question or task description.
            **kwargs:
                failed_attempts: list of prior failed attempt dicts.

        Returns:
            dict with result, source, and worker fields.
        """
        if not self.router:
            return {"error": "No model router available", "worker": self.name}

        failed_attempts = kwargs.get("failed_attempts", [])

        # 1. Search memory for relevant context
        memory_context = ""
        if self.memory_search:
            try:
                results = self.memory_search.search(task, top_k=3)
                if results:
                    memory_context = "\n".join(
                        r["text"][:200] for r in results
                    )
            except Exception as e:
                logger.debug(f"KnowledgeWorker memory search failed: {e}")

        # 2. Build prompt
        parts = []
        if failed_attempts:
            attempts_text = "\n".join(
                f"- {a.get('worker', '?')}: {a.get('error', 'unknown')}"
                for a in failed_attempts
            )
            parts.append(f"以下方法都失敗了:\n{attempts_text}\n")

        if memory_context:
            parts.append(f"相關記憶:\n{memory_context}\n")

        parts.append(
            f"請用你的知識回答以下問題，簡潔明瞭:\n{task}"
        )

        prompt = "\n".join(parts)

        # 3. Call LLM
        try:
            from clients.base_client import ChatMessage
            from core.model_router import ModelRole

            response = await self.router.chat(
                [ChatMessage(role="user", content=prompt)],
                role=ModelRole.CEO,
                max_tokens=500,
            )
            return {
                "result": response.content,
                "source": "knowledge",
                "worker": self.name,
            }
        except Exception as e:
            logger.warning(f"KnowledgeWorker LLM call failed: {e}")
            return {"error": str(e), "worker": self.name}
