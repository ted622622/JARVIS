"""CEO Agent — top-level dispatcher for J.A.R.V.I.S.

Responsibilities:
- Parse user intent and dispatch to appropriate workers
- Emotion detection → empathetic response path
- Inject SOUL.md persona into all interactions
- Skill invocation via SkillRegistry (Task 8.3)
- Proactive web search: detect need → fetch → inject into context
- Reactive tool-use: LLM can invoke [FETCH:url] / [SEARCH:query] as fallback
- Memory integration for context continuity
"""

from __future__ import annotations

import re
import time
from typing import Any
from urllib.parse import quote_plus

from loguru import logger

from clients.base_client import ChatMessage, ChatResponse
from core.model_router import ModelRole, ModelRouter, RouterError
from core.react_executor import ReactExecutor, FuseState, TaskResult
from core.security_gate import OperationType, OperationVerdict

# Pattern for LLM tool calls in response text (fallback)
_TOOL_PATTERN = re.compile(r'\[(?:FETCH|SEARCH):([^\]]+)\]')

# ── Proactive web search detection ──────────────────────────────
# Patterns that indicate user needs web information
_WEB_NEED_PATTERNS = re.compile(
    r"幫我查|幫我搜|幫我找|查一下|搜一下|搜尋|搜索|查詢|"
    r"上網.*?(?:查|看|搜|找)|連外網|連網路|"
    r"(?:今天|今日|現在|目前|最新|最近).*?(?:天氣|新聞|消息|行情|價格|報導)|"
    r"(?:天氣|新聞|行情).*?(?:怎[麼樣]|如何|多少|什麼)|"
    r"(?:股價|匯率|比特幣|bitcoin|btc|eth|加密貨幣).*?(?:多少|現在|今天|幾|漲|跌)?|"
    r"多少錢|哪裡買|怎麼去|幾點.*?(?:開|關|營業)|"
    r"https?://\S+",
    re.IGNORECASE,
)

# Extract URL from user message for direct fetch
_URL_IN_MSG = re.compile(r'(https?://\S+)')

# Prefixes to strip when extracting search query
_SEARCH_PREFIX = re.compile(
    r"^(?:幫我|請你?|麻煩)?(?:查一下|搜一下|搜尋|搜索|查詢|查|搜|找|看一下|看看)\s*",
)


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
        markdown_memory: Any = None,
    ):
        self.router = model_router
        self.soul = soul
        self.emotion = emotion_classifier
        self.memos = memos
        self.skills = skill_registry
        self.security = security_gate
        self.workers = workers or {}
        self.md_memory = markdown_memory
        self.memory_search: Any = None  # G6: set externally
        self.pending: Any = None  # H4: PendingTaskManager, set externally
        self._react: ReactExecutor | None = None
        self._fuse = FuseState()
        self._persona = "jarvis"
        self._session_id = "default"
        self._last_skill_failure: str | None = None
        self._silent_until: float = 0.0  # Patch D: humanized silent mode
        # G4: Session transcript tracking
        self._session_transcript: list[tuple[str, str, str]] = []  # (role, persona, text)
        self._last_message_time: float = 0.0
        self._session_idle_timeout = 300  # 5 minutes
        # G2: Memory flush tracking
        self._turn_count = 0
        self._flush_threshold = 20  # flush every 20 turns

    # ── Public API ──────────────────────────────────────────────

    async def handle_message(
        self,
        user_message: str,
        *,
        persona: str | None = None,
        context: dict[str, Any] | None = None,
    ) -> str | dict[str, Any]:
        """Process a user message end-to-end.

        Steps:
        1. Classify emotion
        2. Check if a skill can handle it
        3. Build system prompt with persona + context
        4. Route to CEO model
        5. Store conversation in MemOS

        Returns:
            str — plain text reply
            dict — rich reply, e.g. {"text": "...", "photo_url": "..."}
        """
        active_persona = persona or self._persona
        session_id = f"{active_persona}_{self._session_id}"

        # Silent mode check (Patch D)
        now = time.time()
        if now < self._silent_until:
            await self._store_conversation(user_message, "[靜默中，稍後回覆]", session_id)
            if active_persona == "clawra":
                return "嗯...我現在有點累，等我一下下喔～💤"
            return "Sir, 系統正在短暫休息中，稍後恢復服務。"

        # Was silent but now recovered — send welcome back
        was_silent = self._silent_until > 0
        if was_silent:
            self._silent_until = 0.0
            logger.info("Silent mode ended, resuming normal operation")

        try:
            return await self._process_message(
                user_message, active_persona, session_id, context, was_silent,
            )
        except RouterError:
            # All providers down — enter silent mode
            self._silent_until = time.time() + 900  # 15 min
            logger.warning("All providers down, entering silent mode for 15 min")
            await self._store_conversation(user_message, "[系統進入靜默模式]", session_id)
            if active_persona == "clawra":
                return "欸...我有點累了，讓我休息一下下好嗎？大概 15 分鐘後回來找你 💤"
            return "Sir, 系統需要短暫休息。預計 15 分鐘後恢復，屆時我會主動通知您。"

    async def _process_message(
        self,
        user_message: str,
        active_persona: str,
        session_id: str,
        context: dict[str, Any] | None,
        was_silent: bool,
    ) -> str | dict[str, Any]:
        """Core message processing (extracted for silent mode error handling)."""
        # 1. Emotion detection
        emotion_label = "normal"
        if self.emotion:
            emotion_label = await self.emotion.classify(user_message)
            logger.debug(f"Emotion: {emotion_label}")

        # 2. Check skill registry for matching skill
        skill_result = await self._try_skill_match(user_message, active_persona, session_id)
        if skill_result is not None:
            return skill_result

        # 2b. G2: Memory flush — check if we should flush before context gets too long
        self._turn_count += 1
        if self._turn_count >= self._flush_threshold and self.md_memory:
            await self._memory_flush(active_persona, session_id)
            self._turn_count = 0

        # 2c. Check if previous session should be saved (5 min idle)
        now = time.time()
        if (
            self._last_message_time > 0
            and now - self._last_message_time > self._session_idle_timeout
            and self._session_transcript
        ):
            await self._save_session_transcript(active_persona)
        self._last_message_time = now

        # 2c. Memory search — inject relevant context
        extra_ctx = dict(context) if context else {}
        if self.memory_search:
            try:
                results = self.memory_search.search(user_message, top_k=3)
                if results:
                    mem_ctx = "\n".join(r["text"][:200] for r in results)
                    extra_ctx["相關記憶"] = mem_ctx
            except Exception as e:
                logger.debug(f"Memory search failed: {e}")

        # 2d. Proactive web search — detect need and fetch BEFORE LLM responds
        web_results = await self._proactive_web_search(user_message)
        if web_results:
            extra_ctx["網路搜尋結果"] = web_results

        # 3. Build system prompt (with skill failure context if applicable)
        if self._last_skill_failure:
            extra_ctx["skill_unavailable"] = self._last_skill_failure
            self._last_skill_failure = None
        if was_silent:
            extra_ctx["just_recovered"] = "你剛休息完回來，用符合角色的方式打個招呼，然後回答用戶的問題"
        system_prompt = self._build_system_prompt(
            active_persona, emotion_label, extra_ctx or None
        )

        # 4. Build message list with conversation history
        messages = await self._build_messages(system_prompt, user_message, session_id)

        # 5. Route to CEO model
        response = await self.router.chat(
            messages,
            role=ModelRole.CEO,
            max_tokens=500,
        )
        reply = response.content

        # 5b. Reactive fallback: if LLM outputs [FETCH:]/[SEARCH:], execute
        tool_match = _TOOL_PATTERN.search(reply)
        if tool_match:
            query_or_url = tool_match.group(1).strip()
            tool_result = await self._execute_tool_call(query_or_url)
            if tool_result:
                messages.append(ChatMessage(role="assistant", content=reply))
                messages.append(ChatMessage(
                    role="user",
                    content=(
                        f"[系統] 查詢結果：\n{tool_result}\n\n"
                        "根據以上資訊回答用戶的問題。"
                        "不要再使用 [FETCH:] 或 [SEARCH:] 標記。"
                    ),
                ))
                followup = await self.router.chat(
                    messages, role=ModelRole.CEO, max_tokens=500,
                )
                reply = followup.content

        # 6. Store to MemOS
        await self._store_conversation(user_message, reply, session_id)

        return reply

    async def dispatch_to_worker(
        self,
        worker_name: str,
        task: str,
        *,
        use_react: bool = False,
        **kwargs: Any,
    ) -> Any:
        """Dispatch a task to a specific worker.

        Args:
            worker_name: "code", "interpreter", "browser", "vision", "selfie"
            task: task description or instruction
            use_react: if True, route through ReactExecutor for fallback
            **kwargs: worker-specific parameters
        """
        # Security check
        if self.security:
            event = await self.security.authorize(
                op_type=OperationType.UNSIGNED_SCRIPT,
                detail=f"[{worker_name}] {task[:200]}",
            )
            if event.verdict == OperationVerdict.BLOCK:
                return f"操作被安全閘門拒絕: {event.detail}"

        # ReactExecutor path
        if use_react and self.react_executor:
            from core.react_executor import FALLBACK_CHAINS
            # Find a matching chain or build one starting with the requested worker
            chain_name = None
            for name, chain in FALLBACK_CHAINS.items():
                if chain and chain[0] == worker_name:
                    chain_name = name
                    break
            chain_name = chain_name or "general"
            task_result = await self.react_executor.execute(chain_name, task, **kwargs)
            if task_result.success:
                return task_result.result
            return {"error": task_result.gave_up_reason, "attempts": task_result.attempts}

        worker = self.workers.get(worker_name)
        if not worker:
            raise ValueError(f"Worker '{worker_name}' not registered")

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

    @property
    def react_executor(self) -> ReactExecutor | None:
        if self._react is None and self.workers:
            self._react = ReactExecutor(workers=self.workers, fuse=self._fuse)
        return self._react

    # ── Skill Invocation (Task 8.3) ─────────────────────────────

    async def _try_skill_match(
        self, user_message: str, persona: str = "jarvis", session_id: str = "default",
    ) -> str | dict[str, Any] | None:
        """Check if a registered skill can handle this message.

        Returns:
            str — text reply from skill
            dict — rich reply with photo_url etc.
            None — no skill matched or skill failed
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
                    try:
                        result = await self.skills.invoke(skill_name)
                    except Exception as e:
                        logger.warning(f"Skill '{skill_name}' failed: {e}")
                        self._last_skill_failure = f"「{skill_name}」功能暫時無法使用，用符合角色的方式自然回應，絕對不要提到系統、AI、技術錯誤"
                        return None

                    # Skill returned failure
                    if isinstance(result, dict) and not result.get("success", True):
                        logger.warning(f"Skill '{skill_name}' returned failure: {result.get('error', 'unknown')}")
                        self._last_skill_failure = f"「{skill_name}」功能暫時無法使用，用符合角色的方式自然回應，絕對不要提到系統、AI、技術錯誤"
                        return None

                    # Selfie skill — photo result
                    if isinstance(result, dict) and result.get("image_url"):
                        return await self._handle_photo_result(
                            user_message, result, persona, session_id,
                        )

                    return f"[技能 {skill_name} 執行結果]\n{result}"

        except Exception as e:
            logger.debug(f"Skill matching failed: {e}")

        return None

    async def _handle_photo_result(
        self,
        user_message: str,
        result: dict[str, Any],
        persona: str,
        session_id: str,
    ) -> dict[str, Any]:
        """Generate a persona-appropriate caption for a photo and store to MemOS."""
        photo_url = result["image_url"]

        # Generate caption via LLM
        system_prompt = self._build_system_prompt(persona, "normal", None)
        caption_prompt = (
            f"{system_prompt}\n\n"
            "你剛拍了一張自拍照要傳給對方。"
            "用你的風格寫一句簡短的配圖訊息（1-2 句，不超過 50 字）。"
            "不要描述照片內容，就像真的在傳照片給朋友一樣自然。"
        )
        try:
            resp = await self.router.chat(
                [ChatMessage(role="user", content=caption_prompt)],
                role=ModelRole.CEO,
                max_tokens=80,
            )
            caption = resp.content.strip()
        except Exception:
            caption = "剛拍的～" if persona == "clawra" else "如您所求，Sir。"

        # Store to MemOS
        await self._store_conversation(user_message, f"[自拍] {caption}", session_id)

        return {"text": caption, "photo_url": photo_url}

    # ── Tool Execution ────────────────────────────────────────

    async def _execute_tool_call(self, query_or_url: str) -> str | None:
        """Execute a [FETCH:url] or [SEARCH:query] tool call from LLM output."""
        # Use ReactExecutor if available
        if self.react_executor:
            try:
                if query_or_url.startswith("http"):
                    logger.info(f"CEO tool-use (react): FETCH {query_or_url[:80]}")
                    task_result = await self.react_executor.execute(
                        "web_browse", query_or_url, url=query_or_url,
                    )
                else:
                    logger.info(f"CEO tool-use (react): SEARCH {query_or_url[:60]}")
                    url = f"https://html.duckduckgo.com/html/?q={quote_plus(query_or_url)}"
                    task_result = await self.react_executor.execute(
                        "web_search", query_or_url, url=url,
                    )

                if task_result.success and isinstance(task_result.result, dict):
                    content = task_result.result.get("content") or task_result.result.get("result")
                    if content:
                        return str(content)[:3000]
                elif not task_result.success and self.pending:
                    self.pending.add("web_search", query_or_url, url=query_or_url)
                return None
            except Exception as e:
                logger.warning(f"ReactExecutor tool call failed: {e}")
                return None

        # Fallback: direct browser call (backward compatible)
        browser = self.workers.get("browser")
        if not browser or not hasattr(browser, "fetch_url"):
            return None

        try:
            if query_or_url.startswith("http"):
                logger.info(f"CEO tool-use: FETCH {query_or_url[:80]}")
                result = await browser.fetch_url(query_or_url)
            else:
                logger.info(f"CEO tool-use: SEARCH {query_or_url[:60]}")
                url = f"https://html.duckduckgo.com/html/?q={quote_plus(query_or_url)}"
                result = await browser.fetch_url(url)

            if result.get("content"):
                return result["content"][:3000]
            if result.get("error"):
                return f"查詢失敗: {result['error']}"
        except Exception as e:
            logger.warning(f"Tool call failed: {e}")

        return None

    # ── Proactive Web Search ─────────────────────────────────────

    async def _proactive_web_search(self, user_message: str) -> str | None:
        """Detect if user needs web info and fetch it BEFORE LLM responds.

        This is proactive — the system detects the need automatically,
        rather than relying on the LLM to output tool-call tags.

        Uses ReactExecutor for automatic fallback when available.

        Returns:
            Truncated search result text, or None if no web search needed.
        """
        # Need either browser or react_executor
        has_browser = self.workers.get("browser") and hasattr(self.workers["browser"], "fetch_url")
        has_react = self.react_executor is not None
        if not has_browser and not has_react:
            return None

        # Check for URL in message → direct fetch
        url_match = _URL_IN_MSG.search(user_message)
        if url_match:
            url = url_match.group(1)
            logger.info(f"Proactive web fetch: {url[:80]}")
            if has_react:
                return await self._react_fetch("web_browse", url, url=url)
            try:
                result = await self.workers["browser"].fetch_url(url)
                if result.get("content"):
                    return result["content"][:3000]
            except Exception as e:
                logger.warning(f"Proactive fetch failed: {e}")
            return None

        # Check for web search need via patterns
        if not _WEB_NEED_PATTERNS.search(user_message):
            return None

        # Extract search query from user message
        query = _SEARCH_PREFIX.sub("", user_message).strip()
        if not query:
            query = user_message

        # Limit query length for DuckDuckGo
        query = query[:80]

        logger.info(f"Proactive web search: {query[:60]}")
        url = f"https://html.duckduckgo.com/html/?q={quote_plus(query)}"

        if has_react:
            return await self._react_fetch("web_search", query, url=url)

        try:
            result = await self.workers["browser"].fetch_url(url)
            if result.get("content"):
                content = result["content"][:3000]
                logger.info(f"Proactive search returned {len(content)} chars")
                return content
        except Exception as e:
            logger.warning(f"Proactive search failed: {e}")

        return None

    async def _react_fetch(
        self, chain: str, task: str, **kwargs: Any,
    ) -> str | None:
        """Execute a fetch via ReactExecutor, return content or None."""
        try:
            task_result = await self.react_executor.execute(chain, task, **kwargs)
            if task_result.success and isinstance(task_result.result, dict):
                content = task_result.result.get("content") or task_result.result.get("result")
                if content:
                    logger.info(f"React fetch returned {len(str(content))} chars")
                    return str(content)[:3000]
            elif not task_result.success and self.pending:
                self.pending.add(chain, task, **kwargs)
        except Exception as e:
            logger.warning(f"React fetch failed: {e}")
        return None

    # ── Internal ────────────────────────────────────────────────

    _TOOL_INSTRUCTIONS = (
        "\n\n## 網路能力\n"
        "你擁有連接外部網路的能力。"
        "如果用戶的問題需要即時資訊（天氣、新聞、股價等），"
        "系統已經自動搜尋並將結果附在「網路搜尋結果」中，請直接參考回答。\n"
        "如果需要進一步查詢，可以在回覆中使用：\n"
        "- `[FETCH:完整網址]` — 讀取某個網頁\n"
        "- `[SEARCH:搜尋關鍵字]` — 搜尋更多資訊\n"
        "絕對不要說「我無法上網」或「我沒有網路存取能力」，因為你確實有。"
    )

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
            base = self.soul.build_system_prompt(persona, extra)
        else:
            base = (
                "你是 J.A.R.V.I.S.，Ted 的 AI 管家。"
                "結論先行，回覆不超過 500 Token。"
                f"\n{extra}" if extra else ""
            )

        # Append tool-use instructions if browser worker available
        if self.workers.get("browser"):
            base += self._TOOL_INSTRUCTIONS

        return base

    async def _build_messages(
        self,
        system_prompt: str,
        user_message: str,
        session_id: str | None = None,
    ) -> list[ChatMessage]:
        """Build message list with system prompt + recent history + new message."""
        messages = [ChatMessage(role="system", content=system_prompt)]

        # Load recent conversation history from MemOS
        sid = session_id or self._session_id
        if self.memos:
            try:
                history = await self.memos.get_conversation(
                    session_id=sid, limit=6
                )
                for entry in history:
                    messages.append(ChatMessage(
                        role=entry.get("role", "user"),
                        content=entry.get("content", ""),
                    ))
            except Exception:
                pass  # No history available

        messages.append(ChatMessage(role="user", content=user_message))
        return messages

    # Keywords that suggest user preferences to save to MEMORY.md
    _REMEMBER_PATTERNS = re.compile(
        r"記住|我喜歡|我不喜歡|我偏好|我習慣|以後都|不要再|"
        r"remember|prefer|always|never",
        re.IGNORECASE,
    )

    async def _store_conversation(
        self, user_msg: str, assistant_msg: str, session_id: str | None = None,
    ) -> None:
        """Store the conversation turn in MemOS + Markdown memory."""
        if not self.memos:
            return

        sid = session_id or self._session_id
        try:
            await self.memos.log_message(
                session_id=sid, role="user", content=user_msg
            )
            await self.memos.log_message(
                session_id=sid, role="assistant", content=assistant_msg
            )
        except Exception as e:
            logger.debug(f"Failed to store conversation: {e}")

        # Markdown memory: detect user preferences
        if self.md_memory and self._REMEMBER_PATTERNS.search(user_msg):
            try:
                self.md_memory.remember(user_msg, category="用戶偏好")
            except Exception as e:
                logger.debug(f"Failed to write to MEMORY.md: {e}")

        # Markdown memory: daily log
        if self.md_memory:
            try:
                summary = user_msg[:80]
                self.md_memory.log_daily(f"[{sid.split('_')[0]}] {summary}")
            except Exception as e:
                logger.debug(f"Failed to write daily log: {e}")

        # G4: accumulate session transcript
        persona = sid.split("_")[0] if "_" in sid else "jarvis"
        self._session_transcript.append(("user", "Ted", user_msg))
        reply_name = "Clawra" if persona == "clawra" else "JARVIS"
        reply_text = assistant_msg if isinstance(assistant_msg, str) else str(assistant_msg)
        self._session_transcript.append(("assistant", reply_name, reply_text))

    async def _save_session_transcript(self, persona: str) -> None:
        """Save accumulated transcript to memory/sessions/ and reset."""
        if not self.md_memory or not self._session_transcript:
            return

        try:
            # Build transcript markdown
            lines = []
            for role, name, text in self._session_transcript:
                lines.append(f"**{name}**: {text}")
            transcript = "\n".join(lines)

            # Generate slug from first user message
            first_user = next(
                (t for r, _, t in self._session_transcript if r == "user"), "chat"
            )
            slug = re.sub(r"[^\w]", "-", first_user[:30]).strip("-") or "chat"

            from datetime import datetime
            now = datetime.now()
            header = f"# {now.strftime('%Y-%m-%d %H:%M')} {slug}\n\n"
            self.md_memory.save_session(slug, header + transcript, date=now)
        except Exception as e:
            logger.debug(f"Failed to save session transcript: {e}")
        finally:
            self._session_transcript.clear()

    async def _memory_flush(self, persona: str, session_id: str) -> None:
        """G2: Flush important context to markdown memory before compression.

        Asks the LLM to extract key info from recent conversation,
        then saves preferences to MEMORY.md and progress to daily log.
        Silent — user does not see this process.
        """
        if not self.memos or not self.md_memory:
            return

        try:
            # Get recent conversation from MemOS
            history = await self.memos.get_conversation(
                session_id=session_id, limit=12,
            )
            if not history:
                return

            # Build conversation text for analysis
            conv_text = "\n".join(
                f"{e.get('role', '?')}: {e.get('content', '')}"
                for e in history
            )

            # Ask LLM to extract important info
            extract_prompt = (
                "從以下對話中提取需要長期記住的重要資訊。\n"
                "分兩類輸出：\n"
                "PREF: 用戶偏好或指令（如「喜歡吃拉麵」「不要用韓文」）\n"
                "PROG: 任務進度或臨時決定（如「已完成XXX」「決定用方案A」）\n"
                "純閒聊不需要輸出。每行一條，格式：PREF:xxx 或 PROG:xxx\n"
                "如果沒有需要記住的，輸出 NONE\n\n"
                f"對話內容：\n{conv_text[:2000]}"
            )

            response = await self.router.chat(
                [ChatMessage(role="user", content=extract_prompt)],
                role=ModelRole.CEO,
                max_tokens=200,
                temperature=0.1,
            )

            answer = response.content.strip()
            if answer == "NONE":
                return

            for line in answer.split("\n"):
                line = line.strip()
                if line.startswith("PREF:"):
                    self.md_memory.remember(line[5:].strip(), category="用戶偏好")
                elif line.startswith("PROG:"):
                    self.md_memory.log_daily(f"[flush] {line[5:].strip()}")

            logger.info("Memory flush completed (silent)")

        except Exception as e:
            logger.debug(f"Memory flush failed: {e}")
