"""CEO Agent — top-level dispatcher for J.A.R.V.I.S.

Responsibilities:
- Parse user intent and dispatch to appropriate workers
- Emotion detection → empathetic response path
- Inject SOUL.md persona into all interactions
- Skill invocation via SkillRegistry (Task 8.3)
- Memory integration for context continuity
"""

from __future__ import annotations

from typing import Any

from loguru import logger

from clients.base_client import ChatMessage, ChatResponse
from core.model_router import ModelRole, ModelRouter


class CEOAgent:
    """Central orchestrator — all user interactions flow through here.

    Usage:
        ceo = CEOAgent(
            model_router=router,
            soul=soul,
            emotion_classifier=emotion,
            memos=memos,
            skill_registry=registry,
            security_gate=security,
        )
        response = await ceo.handle_message("幫我查一下明天行程")
    """

    def __init__(
        self,
        model_router: ModelRouter,
        soul: Any = None,
        emotion_classifier: Any = None,
        memos: Any = None,
        skill_registry: Any = None,
        security_gate: Any = None,
        workers: dict[str, Any] | None = None,
    ):
        self.router = model_router
        self.soul = soul
        self.emotion = emotion_classifier
        self.memos = memos
        self.skills = skill_registry
        self.security = security_gate
        self.workers = workers or {}
        self._persona = "jarvis"

    # ── Public API ──────────────────────────────────────────────

    async def handle_message(
        self,
        user_message: str,
        *,
        persona: str | None = None,
        context: dict[str, Any] | None = None,
    ) -> str:
        """Process a user message end-to-end.

        Steps:
        1. Classify emotion
        2. Check if a skill can handle it
        3. Build system prompt with persona + context
        4. Route to CEO model
        5. Store conversation in MemOS

        Returns the assistant's reply text.
        """
        active_persona = persona or self._persona

        # 1. Emotion detection
        emotion_label = "normal"
        if self.emotion:
            emotion_label = await self.emotion.classify(user_message)
            logger.debug(f"Emotion: {emotion_label}")

        # 2. Check skill registry for matching skill
        skill_result = await self._try_skill_match(user_message)
        if skill_result is not None:
            return skill_result

        # 3. Build system prompt
        system_prompt = self._build_system_prompt(
            active_persona, emotion_label, context
        )

        # 4. Build message list with conversation history
        messages = await self._build_messages(system_prompt, user_message)

        # 5. Route to CEO model
        response = await self.router.chat(
            messages,
            role=ModelRole.CEO,
            max_tokens=500,
        )

        # 6. Store to MemOS
        await self._store_conversation(user_message, response.content)

        return response.content

    async def dispatch_to_worker(
        self,
        worker_name: str,
        task: str,
        **kwargs: Any,
    ) -> Any:
        """Dispatch a task to a specific worker.

        Args:
            worker_name: "code", "interpreter", "browser", "vision", "selfie"
            task: task description or instruction
            **kwargs: worker-specific parameters
        """
        worker = self.workers.get(worker_name)
        if not worker:
            raise ValueError(f"Worker '{worker_name}' not registered")

        # Security check
        if self.security:
            verdict = await self.security.authorize(
                operation=f"worker_{worker_name}",
                detail=task[:200],
            )
            if verdict.action == "BLOCK":
                return f"操作被安全閘門拒絕: {verdict.reason}"

        return await worker.execute(task, **kwargs)

    def switch_persona(self, persona: str) -> None:
        """Switch between 'jarvis' and 'clawra' persona."""
        if persona in ("jarvis", "clawra"):
            self._persona = persona
            logger.info(f"Persona switched to: {persona}")
        else:
            raise ValueError(f"Unknown persona: {persona}")

    @property
    def current_persona(self) -> str:
        return self._persona

    # ── Skill Invocation (Task 8.3) ─────────────────────────────

    async def _try_skill_match(self, user_message: str) -> str | None:
        """Check if a registered skill can handle this message.

        Returns skill output as string if matched, None otherwise.
        """
        if not self.skills:
            return None

        # Ask CEO model to determine if a skill should be invoked
        skill_list = self.skills.list_all()
        if not skill_list:
            return None

        skill_info = ", ".join(
            f"{s.name}({s.description[:40]})" for s in skill_list
        )

        judge_prompt = (
            f"可用技能: [{skill_info}]\n"
            f"用戶訊息: {user_message}\n\n"
            "如果這個訊息明確需要調用某個技能，回覆「SKILL:技能名稱」。\n"
            "如果不需要調用技能，回覆「NONE」。\n"
            "只回覆 SKILL:xxx 或 NONE，不要有其他文字。"
        )

        try:
            response = await self.router.chat(
                [ChatMessage(role="user", content=judge_prompt)],
                role=ModelRole.CEO,
                max_tokens=30,
                temperature=0.1,
            )
            answer = response.content.strip()

            if answer.startswith("SKILL:"):
                skill_name = answer[6:].strip()
                meta = self.skills.get(skill_name)
                if meta:
                    logger.info(f"CEO invoking skill: {skill_name}")
                    result = await self.skills.invoke(skill_name)
                    return f"[技能 {skill_name} 執行結果]\n{result}"

        except Exception as e:
            logger.debug(f"Skill matching failed: {e}")

        return None

    # ── Internal ────────────────────────────────────────────────

    def _build_system_prompt(
        self,
        persona: str,
        emotion: str,
        context: dict[str, Any] | None,
    ) -> str:
        """Construct the full system prompt."""
        extra_parts = []

        if emotion != "normal":
            extra_parts.append(f"用戶當前情緒: {emotion}")

        if context:
            for k, v in context.items():
                extra_parts.append(f"{k}: {v}")

        extra = "\n".join(extra_parts)

        if self.soul and self.soul.is_loaded:
            return self.soul.build_system_prompt(persona, extra)

        # Fallback if SOUL not loaded
        return (
            "你是 J.A.R.V.I.S.，Ted 的 AI 管家。"
            "結論先行，回覆不超過 500 Token。"
            f"\n{extra}" if extra else ""
        )

    async def _build_messages(
        self,
        system_prompt: str,
        user_message: str,
    ) -> list[ChatMessage]:
        """Build message list with system prompt + recent history + new message."""
        messages = [ChatMessage(role="system", content=system_prompt)]

        # Load recent conversation history from MemOS
        if self.memos:
            try:
                history = await self.memos.get_recent_conversation(limit=6)
                for entry in history:
                    messages.append(ChatMessage(
                        role=entry.get("role", "user"),
                        content=entry.get("content", ""),
                    ))
            except Exception:
                pass  # No history available

        messages.append(ChatMessage(role="user", content=user_message))
        return messages

    async def _store_conversation(self, user_msg: str, assistant_msg: str) -> None:
        """Store the conversation turn in MemOS."""
        if not self.memos:
            return

        try:
            await self.memos.log_conversation(
                role="user", content=user_msg, agent_id="user"
            )
            await self.memos.log_conversation(
                role="assistant", content=assistant_msg, agent_id="ceo_agent"
            )
        except Exception as e:
            logger.debug(f"Failed to store conversation: {e}")
