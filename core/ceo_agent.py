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

import asyncio
import json
import re
import time
from datetime import datetime
from pathlib import Path
from typing import Any
from urllib.parse import quote_plus

from loguru import logger

try:
    from opencc import OpenCC
    _s2t = OpenCC("s2t")
except ImportError:
    _s2t = None
    logger.warning("OpenCC not installed — Clawra s2t filter disabled")

from clients.base_client import ChatMessage, ChatResponse
from core.model_router import ModelRole, ModelRouter, RouterError
from core.conversation_compressor import ConversationCompressor
from core.help_decision import HelpDecisionEngine
from core.react_executor import ReactExecutor, FuseState, TaskResult
from core.security_gate import OperationType, OperationVerdict
from core.shared_memory import SharedMemory
from core.soul_growth import SoulGrowth
from core.task_router import TaskRouter

# ── Phase 2: Agent SDK complexity classification ─────────────────


class TaskComplexity:
    SIMPLE = "simple"
    MEDIUM = "medium"
    COMPLEX = "complex"


_COMPLEX_PATTERNS = re.compile(
    r"幫我訂|幫我預約|幫我安排|"
    r"幫我研究|幫我分析|幫我比較|"
    r"幫我寫一[份篇封]|幫我整理|"
    r"做一個.*計畫|規劃.*行程|"
    r"查.*然後.*整理|搜.*然後.*比較|"
    r"步驟|流程|完整",
    re.IGNORECASE,
)

_SIMPLE_PATTERNS = re.compile(
    r"^(你好|嗨|hi|hello|早安|晚安|在嗎|幹嘛|"
    r"謝謝|好的|OK|嗯|哈哈|欸|喔|對|是)",
    re.IGNORECASE,
)

# Pattern for LLM tool calls in response text (fallback)
_TOOL_PATTERN = re.compile(r'\[(?:FETCH|SEARCH|MAPS):([^\]]+)\]')

# ── Patch O: Long-task detection ─────────────────────────────────
_LONG_TASK_TYPES = frozenset({"web_search", "web_browse", "restaurant_booking", "code"})
_URL_PATTERN = re.compile(r"https?://\S+")

# ── Web content truncation limits ────────────────────────────────
_FETCH_CHAR_LIMIT = 50_000   # URL fetch: enough for READMEs / full pages
_SEARCH_CHAR_LIMIT = 3_000   # DuckDuckGo: search results are short

# ── Patch P: Long-content chunking ──────────────────────────────
_LONG_CONTENT_THRESHOLD = 2000     # 用戶訊息字數觸發（搭配分析關鍵字）
_STRUCTURED_THRESHOLD = 500        # 結構化內容門檻（搭配 MD 標記）
_LONG_WEB_THRESHOLD = 5000         # 網頁內容字數觸發
_CHUNK_SIZE = 3000                 # 每段大小（同 TranscribeWorker）
_ANALYSIS_KEYWORDS = re.compile(r"整理|摘要|提取|分析|比較|歸納|統整|對照")
_STRUCTURED_MARKERS = re.compile(r"^#{1,4}\s|^>\s|^---$|^```|^\- \[", re.MULTILINE)
# Task template placeholders — these need CEO tool-use ([FETCH:], [SEARCH:]), not chunking
_TASK_TEMPLATE_PATTERN = re.compile(r"（[^）]*內容[^）]*）|（[^）]*填入[^）]*）|\{\{.+?\}\}")
# GitHub repo references: owner/repo patterns (for proactive fetch in task templates)
_GITHUB_REPO_PATTERN = re.compile(r'\b([A-Za-z][\w.-]+/[A-Za-z][\w.-]+)\b')

# ── LLM reply cleanup (strip leaked thinking tags) ────────────────
_THINK_TAG_PATTERN = re.compile(r"</?think>", re.IGNORECASE)
_THINK_BLOCK_PATTERN = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)
_WRAPPING_CODE_BLOCK = re.compile(r"^```(?:\w*)\n?(.*?)```$", re.DOTALL)


def _clean_llm_reply(text: str) -> str:
    """Strip leaked thinking tags and wrapping code blocks from LLM output."""
    if not text:
        return text
    # Remove full <think>...</think> blocks first
    text = _THINK_BLOCK_PATTERN.sub("", text)
    # Remove stray </think> or <think> tags
    text = _THINK_TAG_PATTERN.sub("", text)
    # Remove wrapping code blocks (```...\ncontent\n```)
    text = text.strip()
    m = _WRAPPING_CODE_BLOCK.match(text)
    if m:
        text = m.group(1).strip()
    return text.strip()


def _force_traditional_chinese(text: str) -> str:
    """Convert any leaked simplified Chinese to traditional Chinese."""
    if not text or _s2t is None:
        return text
    return _s2t.convert(text)


# ── H1 v2: Task Resolution Chains ────────────────────────────────
# CLI/API first → httpx → browser (last resort) → partial assist
# Each chain entry: {"method": str, "worker": str, "timeout": int}
TASK_RESOLUTION_CHAINS: dict[str, dict] = {
    # ── Calendar / Email — gog CLI handles it ──
    "calendar": {
        "chain": [
            {"method": "gog_cli", "worker": "gog", "timeout": 15},
        ],
    },
    "email": {
        "chain": [
            {"method": "gog_cli", "worker": "gog", "timeout": 15},
        ],
    },
    # ── Booking — API first, browser last ──
    "booking": {
        "chain": [
            {"method": "httpx_search", "worker": "browser", "timeout": 15},
            {"method": "browser", "worker": "browser", "timeout": 45},
            {"method": "partial_assist", "worker": "knowledge", "timeout": 30},
        ],
    },
    # ── Web search — httpx → browser → knowledge ──
    "web_search": {
        "chain": [
            {"method": "httpx_search", "worker": "browser", "timeout": 15},
            {"method": "browser_search", "worker": "browser", "timeout": 30},
            {"method": "knowledge_reply", "worker": "knowledge", "timeout": 30},
        ],
    },
    # ── Code ──
    "code_task": {
        "chain": [
            {"method": "direct", "worker": "code", "timeout": 60},
            {"method": "knowledge_reply", "worker": "knowledge", "timeout": 30},
        ],
    },
    # ── General fallback ──
    "general": {
        "chain": [
            {"method": "knowledge_reply", "worker": "knowledge", "timeout": 30},
        ],
    },
}

# ── Proactive web search detection ──────────────────────────────
# Patterns that indicate user needs web information
_WEB_NEED_PATTERNS = re.compile(
    r"幫我查|幫我搜|幫我找|查一下|搜一下|搜尋|搜索|查詢|"
    r"幫我訂|訂位|預約|預定|booking|reserve|"
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
        self._post_action: Any = None  # K2: PostActionChain
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
        # Patch I: multi-task modules
        self._compressor = ConversationCompressor()
        self._compressor.set_pre_flush_callback(self._pre_flush_extract)
        self._task_router = TaskRouter()
        self._help_engine = HelpDecisionEngine()
        # Patch J: soul evolution
        self._soul_growth: SoulGrowth | None = None
        self._shared_memory: SharedMemory | None = None
        # Phase 2: Agent SDK executor (lazy-init)
        self._agent_executor: Any = None
        # Emotion passthrough for voice TTS
        self._last_emotion: str = "normal"

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
                return "嗯...我現在有點累，等我一下下喔"
            return "Sir, 系統正在短暫休息中，稍後恢復服務。"

        # Was silent but now recovered — send welcome back
        was_silent = self._silent_until > 0
        if was_silent:
            self._silent_until = 0.0
            logger.info("Silent mode ended, resuming normal operation")

        try:
            result = await self._process_message(
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

        # Clawra: force simplified → traditional Chinese conversion
        if active_persona == "clawra" and _s2t is not None:
            if isinstance(result, dict):
                if "text" in result and isinstance(result["text"], str):
                    result["text"] = _force_traditional_chinese(result["text"])
            elif isinstance(result, str):
                result = _force_traditional_chinese(result)

        return result

    # ── Phase 2: Agent SDK helpers ──────────────────────────────

    def _classify_complexity(self, message: str) -> str:
        """Classify message complexity for Agent SDK routing."""
        if len(message) < 10:
            return TaskComplexity.SIMPLE
        if _SIMPLE_PATTERNS.match(message):
            return TaskComplexity.SIMPLE
        if _COMPLEX_PATTERNS.search(message):
            return TaskComplexity.COMPLEX
        if _WEB_NEED_PATTERNS.search(message):
            return TaskComplexity.MEDIUM
        return TaskComplexity.SIMPLE

    def _get_agent_executor(self) -> Any:
        """Lazy-init AgentExecutor."""
        if self._agent_executor is None:
            try:
                from core.agent_executor import AgentExecutor
                self._agent_executor = AgentExecutor(
                    jarvis_root=str(Path(__file__).parent.parent)
                )
            except ImportError:
                logger.warning("claude-agent-sdk not installed, Agent SDK disabled")
                return None
        return self._agent_executor

    async def _process_message(
        self,
        user_message: str,
        active_persona: str,
        session_id: str,
        context: dict[str, Any] | None,
        was_silent: bool,
    ) -> str | dict[str, Any]:
        """Core message processing (extracted for silent mode error handling)."""
        # I3: Track conversation in compressor
        self._compressor.add_turn("user", user_message)

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

        # 2c. Memory search — inject relevant context (supports async HybridSearch)
        extra_ctx = dict(context) if context else {}
        if self.memory_search:
            try:
                search_fn = getattr(self.memory_search, "search", None)
                if asyncio.iscoroutinefunction(search_fn):
                    results = await self.memory_search.search(user_message, top_k=3)
                else:
                    results = self.memory_search.search(user_message, top_k=3)
                if results:
                    mem_ctx = "\n".join(r["text"][:200] for r in results)
                    extra_ctx["相關記憶"] = mem_ctx
            except Exception as e:
                logger.debug(f"Memory search failed: {e}")

        # 2.5 Phase 2: Agent SDK dispatch for COMPLEX tasks
        complexity = self._classify_complexity(user_message)
        if complexity == TaskComplexity.COMPLEX:
            executor = self._get_agent_executor()
            if executor is not None:
                logger.info("Task COMPLEX → Agent SDK dispatch")
                mem_ctx = extra_ctx.get("相關記憶", "")
                try:
                    sdk_result = await executor.run(
                        task=user_message,
                        tier="complex",
                        persona=active_persona,
                        extra_context=mem_ctx,
                    )
                    if sdk_result["success"]:
                        reply = sdk_result["response"]
                        await self._store_conversation(
                            user_message, reply, session_id,
                        )
                        self._compressor.add_turn("assistant", reply)
                        logger.info(
                            f"Agent SDK success: {sdk_result['tool_calls']} tools, "
                            f"{sdk_result['duration']}s"
                        )
                        # 80% quota warning
                        if executor.is_quota_low():
                            usage = executor.get_daily_usage()
                            logger.warning(
                                f"Agent SDK quota alert: "
                                f"{usage['usage_pct']}% used "
                                f"({usage['daily_tokens']:,}/{usage['daily_limit']:,})"
                            )
                            # Append warning to reply so TG user sees it
                            reply += (
                                f"\n\n⚠️ Agent SDK 額度: "
                                f"{usage['usage_pct']}% 已使用"
                            )
                        # Extract phone/url for TG separate messages
                        phone = self._extract_phone(reply)
                        booking_url = self._extract_booking_url(reply)
                        if phone or booking_url:
                            return {
                                "text": reply,
                                "phone": phone,
                                "booking_url": booking_url,
                            }
                        return reply
                except Exception as e:
                    logger.warning(f"Agent SDK failed: {e}, falling back")

        # 2d. Proactive web search — detect need and fetch BEFORE LLM responds
        web_results = await self._proactive_web_search(user_message)
        _booking_phone = None
        _booking_url = None
        if web_results:
            _WEB_CTX_PREFIX = (
                "（以下是系統已抓取的網頁內容，請直接分析此文字回答用戶問題，"
                "不需要自行訪問網站或執行任何系統命令。）\n"
            )
            if isinstance(web_results, dict):
                extra_ctx["網路搜尋結果"] = _WEB_CTX_PREFIX + web_results["text"]
                _booking_phone = web_results.get("phone")
                _booking_url = web_results.get("booking_url")
            else:
                extra_ctx["網路搜尋結果"] = _WEB_CTX_PREFIX + web_results

        # 2e. Booking short-circuit — skip LLM when we already have a booking URL
        #     (LLM gets too little context from the fallback dict and returns empty)
        if _booking_url:
            restaurant_name = ""
            if isinstance(web_results, dict):
                # Extract from "店名: XXX\n..." text or fallback
                for line in web_results.get("text", "").split("\n"):
                    if line.startswith("店名:"):
                        restaurant_name = line.split(":", 1)[1].strip()
                        break
            restaurant_name = restaurant_name or "餐廳"
            parts = [f"Sir，找到{restaurant_name}的訂位頁面了："]
            if _booking_phone:
                parts.append(f"電話: {_booking_phone}")
            if active_persona == "clawra":
                parts = [f"欸找到了！{restaurant_name}的訂位連結在這"]
                if _booking_phone:
                    parts.append(f"電話是 {_booking_phone}")
            logger.info(f"Booking short-circuit: {restaurant_name}, url={_booking_url[:60]}")
            await self._store_conversation(user_message, f"[訂位] {restaurant_name}", session_id)
            return {
                "text": "\n".join(parts),
                "phone": _booking_phone,
                "booking_url": _booking_url,
            }

        # ── Patch P: Long-content detection ─────────────────────────
        _long_text = ""
        _user_instruction = user_message[:200]
        # Task templates with placeholders — proactively fetch GitHub repos
        _is_task_template = bool(_TASK_TEMPLATE_PATTERN.search(user_message))
        if _is_task_template:
            logger.info("Task template detected (placeholders found), skipping chunking")
            # Proactively fetch referenced GitHub repos
            github_content = await self._fetch_github_repos(user_message)
            if github_content:
                logger.info(f"Fetched GitHub content: {len(github_content)} chars, routing to chunked processing")
                reply = await self._handle_long_content(
                    github_content, user_message, active_persona,
                )
                await self._store_conversation(user_message, reply, session_id)
                self._last_emotion = emotion_label
                self._compressor.add_turn("assistant", reply)
                return reply

        # 條件 1: 用戶訊息 >2000 + 含分析關鍵字（排除任務模板）
        if not _is_task_template and len(user_message) > _LONG_CONTENT_THRESHOLD:
            if _ANALYSIS_KEYWORDS.search(user_message[:500]):
                _long_text = user_message

        # 條件 2: 結構化內容 >500（排除任務模板）
        if not _is_task_template and not _long_text and len(user_message) > _STRUCTURED_THRESHOLD:
            markers = _STRUCTURED_MARKERS.findall(user_message)
            if len(markers) >= 3:
                _long_text = user_message
                logger.info(f"Structured content detected: {len(markers)} MD markers")

        # 條件 3: 網頁內容 >5000
        web_text = extra_ctx.get("網路搜尋結果", "") if extra_ctx else ""
        if not _long_text and len(web_text) > _LONG_WEB_THRESHOLD:
            _long_text = web_text
            _user_instruction = user_message

        if _long_text:
            logger.info(f"Long content detected: {len(_long_text)} chars, chunking...")
            reply = await self._handle_long_content(_long_text, _user_instruction, active_persona)
            await self._store_conversation(user_message, reply, session_id)
            self._last_emotion = emotion_label
            self._compressor.add_turn("assistant", reply)
            return reply  # 跳過正常 CEO 流程

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
        #    Increase max_tokens when context is large (web fetch or long user message)
        web_ctx_len = len(extra_ctx.get("網路搜尋結果", "")) if extra_ctx else 0
        needs_long_reply = web_ctx_len > _SEARCH_CHAR_LIMIT or len(user_message) > 500
        ceo_max_tokens = 4096 if needs_long_reply else 500
        response = await self.router.chat(
            messages,
            role=ModelRole.CEO,
            max_tokens=ceo_max_tokens,
        )
        reply = _clean_llm_reply(response.content)

        # Record token usage for pool balancing
        self._record_token_usage(response)

        # 5b. Reactive fallback: if LLM outputs [FETCH:]/[SEARCH:]/[MAPS:], execute
        # Loop up to 3 rounds to handle multiple tool calls
        for _tool_round in range(3):
            tool_matches = _TOOL_PATTERN.findall(reply) if reply else []
            if not tool_matches:
                break

            # Execute all tool calls found in this round
            all_results = []
            for match in _TOOL_PATTERN.finditer(reply):
                tag = match.group(0).split(":")[0].lstrip("[")
                query_or_url = match.group(1).strip()
                tool_result = await self._execute_tool_call(query_or_url, tag=tag)
                if tool_result:
                    all_results.append(f"[{tag}:{query_or_url[:60]}]\n{tool_result}")

            if not all_results:
                break

            combined = "\n\n---\n\n".join(all_results)
            messages.append(ChatMessage(role="assistant", content=reply))
            messages.append(ChatMessage(
                role="user",
                content=(
                    f"[系統] 查詢結果：\n{combined}\n\n"
                    "根據以上資訊回答用戶的問題。"
                    "不要再使用 [FETCH:] 或 [SEARCH:] 標記。直接給出完整回覆。"
                ),
            ))
            followup = await self.router.chat(
                messages, role=ModelRole.CEO, max_tokens=4096,
            )
            reply = _clean_llm_reply(followup.content)
            logger.debug(f"Tool-use round {_tool_round + 1} reply length: {len(reply or '')}")

        # Patch O: Log reply before returning + empty reply guard
        logger.debug(
            f"CEO final reply length: {len(reply or '')} chars "
            f"(max_tokens={ceo_max_tokens}, msg_len={len(user_message)}, web_ctx={web_ctx_len})"
        )
        if not reply or not reply.strip():
            logger.warning("CEO reply is empty after processing, applying fallback")
            if active_persona == "clawra":
                reply = "嗯...我查到了一些東西但整理時出了問題，你可以再問一次嗎"
            else:
                reply = "Sir, 我已取得相關資料，但整理回覆時遇到問題。請再試一次。"

        # 6. Store to MemOS
        await self._store_conversation(user_message, reply, session_id)

        # Expose last emotion for voice TTS emotion passthrough
        self._last_emotion = emotion_label

        # I3: Track assistant reply in compressor
        self._compressor.add_turn("assistant", reply if isinstance(reply, str) else str(reply))

        # Patch T+: Pre-compaction flush — extract important info before discard
        if self._compressor.has_pending_flush:
            async def _safe_flush():
                try:
                    await self._compressor.flush_pending()
                except Exception as e:
                    logger.warning(f"Pre-flush failed: {e}")
            asyncio.create_task(_safe_flush())

        # J2+J3: Soul growth — learn from conversation
        reply_str = reply if isinstance(reply, str) else str(reply)
        if self._soul_growth:
            try:
                insight = self._soul_growth.maybe_learn(active_persona, user_message, reply_str)
                if insight and self.soul:
                    self.soul.reload_growth(active_persona)
                    logger.info(f"SoulGrowth [{active_persona}]: learned and reloaded")
            except Exception as e:
                logger.warning(f"Soul growth error: {e}")

        # J4: Shared memory — check for memorable moments (Clawra only)
        if active_persona == "clawra" and self._shared_memory:
            try:
                moment = self._shared_memory.check_and_remember(user_message, reply_str)
                if moment:
                    logger.info(f"SharedMemory: recorded moment — {moment[:50]}")
            except Exception as e:
                logger.warning(f"Shared memory error: {e}")

        # K3: Booking result — attach phone/booking_url for Telegram
        if _booking_phone or _booking_url:
            return {
                "text": reply if isinstance(reply, str) else str(reply),
                "phone": _booking_phone,
                "booking_url": _booking_url,
            }

        return reply

    # ── Patch P: Long-content chunking ──────────────────────────────

    async def _handle_long_content(self, text: str, user_instruction: str, persona: str) -> str:
        """長文件分段處理 — 兩階段 (reuse TranscribeWorker pattern)."""
        chunks = self._split_long_content(text)
        logger.info(f"Long content: {len(text)} chars → {len(chunks)} chunks")

        # Stage 1 uses short instruction to save tokens
        short_instruction = user_instruction[:300]

        # Stage 1: per-chunk extraction (Lite model, cheap)
        chunk_summaries: list[str] = []
        for i, chunk in enumerate(chunks):
            prompt = (
                f"這是一份文件的第 {i+1}/{len(chunks)} 部分。\n"
                f"用戶要求: {short_instruction}\n\n{chunk}\n\n"
                f"請提取這段的重點資訊，保留所有關鍵設定、數字、名稱。"
            )
            resp = await self.router.chat(
                [ChatMessage(role="user", content=prompt)],
                role=ModelRole.CEO, max_tokens=800, task_type="template",
            )
            chunk_summaries.append(resp.content)

        # Stage 2: merge (CEO model, full reasoning)
        merged = "\n\n".join(
            f"【第 {i+1} 段重點】\n{s}" for i, s in enumerate(chunk_summaries)
        )
        final_prompt = (
            f"以下是從多個來源提取的重點資訊。\n"
            f"用戶原始要求:\n{user_instruction}\n\n"
            f"提取結果:\n{merged}\n\n"
            f"請根據用戶的模板結構，將提取結果填入對應段落，整合成完整回覆。\n"
            f"用繁體中文，結論先行。\n"
            f"注意：不要在回覆中出現「第N段重點」等內部標記，直接給出完整的結構化回覆。"
        )
        resp = await self.router.chat(
            [ChatMessage(role="user", content=final_prompt)],
            role=ModelRole.CEO, max_tokens=4096,
        )
        return resp.content

    def _split_long_content(self, text: str) -> list[str]:
        """切分長文本（同 TranscribeWorker._split_transcript 邏輯）."""
        if len(text) <= _CHUNK_SIZE:
            return [text]
        chunks: list[str] = []
        start = 0
        while start < len(text):
            end = start + _CHUNK_SIZE
            if end < len(text):
                for sep in ("。", ".", "\n", "，", ","):
                    pos = text.rfind(sep, start, end)
                    if pos > start:
                        end = pos + 1
                        break
            if end <= start:
                end = start + _CHUNK_SIZE
            chunks.append(text[start:end])
            start = end
        return chunks

    async def _fetch_github_repos(self, user_message: str) -> str | None:
        """Extract GitHub owner/repo references and proactively fetch their pages."""
        repos = _GITHUB_REPO_PATTERN.findall(user_message)
        # Filter out common false positives (file paths, version strings, etc.)
        _FP_SUFFIXES = (".py", ".js", ".md", ".txt", ".json", ".yaml", ".yml", ".ts", ".css")
        _FP_PREFIXES = (".", "src/", "core/", "config/", "data/", "tests/")
        valid_repos = [
            r for r in repos
            if not any(r.startswith(p) for p in _FP_PREFIXES)
            and not any(r.endswith(s) for s in _FP_SUFFIXES)
            and len(r.split("/")[0]) >= 2  # owner at least 2 chars
            and len(r.split("/")[1]) >= 2  # repo at least 2 chars
        ]
        if not valid_repos:
            return None

        logger.info(f"Task template: fetching {len(valid_repos)} GitHub repos: {valid_repos}")
        fetched: list[str] = []
        for repo in valid_repos[:5]:  # max 5 repos
            url = f"https://github.com/{repo}"
            content = await self._execute_tool_call(url, tag="FETCH")
            if content and "404" not in content[:100] and len(content) > 200:
                fetched.append(f"=== {repo} ===\n{content[:_FETCH_CHAR_LIMIT]}")
                logger.info(f"Fetched {repo}: {len(content)} chars")
            else:
                logger.warning(f"Skipped {repo} (not found or too short)")

        if fetched:
            return "\n\n".join(fetched)
        return None

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

    # ── Patch O: Complexity Estimation ──────────────────────────

    def estimate_complexity(self, user_message: str) -> dict[str, Any]:
        """Estimate task complexity without consuming LLM tokens.

        Returns:
            {"is_long": bool, "reason": str, "estimate_seconds": int}
        """
        tasks = self._task_router.classify(user_message)
        task_types = {t.task_type for t in tasks}

        has_url = bool(_URL_PATTERN.search(user_message))

        if task_types & _LONG_TASK_TYPES or has_url:
            return {"is_long": True, "reason": "web_task", "estimate_seconds": 45}
        if len(user_message) > 300:
            return {"is_long": True, "reason": "complex_instruction", "estimate_seconds": 30}
        return {"is_long": False, "reason": "", "estimate_seconds": 5}

    # ── Phase 2: Reply extraction helpers ────────────────────────

    @staticmethod
    def _extract_phone(text: str) -> str | None:
        """Extract phone number from reply text."""
        m = re.search(
            r'(\+?\d{1,4}[-\s]?\(?\d{1,4}\)?[-\s]?\d{2,4}[-\s]?\d{2,4}[-\s]?\d{0,4})',
            text,
        )
        return m.group(1) if m else None

    @staticmethod
    def _extract_booking_url(text: str) -> str | None:
        """Extract booking-related URL from reply text."""
        m = re.search(
            r'(https?://(?:inline\.app|www\.opentable|eztable|'
            r'booking|reserve)[^\s\)]+)',
            text, re.IGNORECASE,
        )
        if m:
            return m.group(1)
        m = re.search(r'(https?://[^\s\)]+)', text)
        return m.group(1) if m else None

    @property
    def react_executor(self) -> ReactExecutor | None:
        if self._react is None and self.workers:
            self._react = ReactExecutor(workers=self.workers, fuse=self._fuse)
        return self._react

    # ── Skill Invocation (Task 8.3) ─────────────────────────────

    # Regex pre-check: force skill invocation without LLM judge
    _SELFIE_FORCE_PATTERN = re.compile(
        r"自拍|照片|穿搭|selfie|拍.*?照|看看妳|看我|傳.*?照",
        re.IGNORECASE,
    )

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

        skill_list = self.skills.list_all()
        if not skill_list:
            return None

        # ── Regex pre-check: bypass LLM judge for known skill keywords ──
        skill_name: str | None = None
        if self._SELFIE_FORCE_PATTERN.search(user_message) and self.skills.get("selfie"):
            skill_name = "selfie"
            logger.info(f"Skill pre-match (regex): {skill_name}")

        # ── LLM judge fallback for non-regex matches ──
        if not skill_name:
            skill_info = ", ".join(
                f"{s.name}({s.description[:40]})" for s in skill_list
            )

            # Inject recent conversation history for context-aware matching
            history_hint = ""
            if self.memos:
                try:
                    sid = f"{persona}_{self._session_id}"
                    history = await self.memos.get_conversation(session_id=sid, limit=4)
                    if history:
                        lines = []
                        for entry in history[-4:]:
                            role = "用戶" if entry.get("role") == "user" else "助理"
                            lines.append(f"{role}: {entry.get('content', '')[:60]}")
                        history_hint = "最近對話:\n" + "\n".join(lines) + "\n\n"
                except Exception:
                    pass

            judge_prompt = (
                f"可用技能: [{skill_info}]\n"
                f"{history_hint}"
                f"用戶訊息: {user_message}\n\n"
                "根據上下文判斷，這個訊息是否需要調用某個技能？\n"
                "例如：如果之前在討論拍照/自拍，用戶說「再來一次」，就應該調用 selfie。\n"
                "回覆「SKILL:技能名稱」或「NONE」，不要有其他文字。"
            )

            try:
                response = await self.router.chat(
                    [ChatMessage(role="user", content=judge_prompt)],
                    role=ModelRole.CEO,
                    task_type="template",
                    max_tokens=30,
                    temperature=0.1,
                )
                answer = response.content.strip()

                if answer.startswith("SKILL:"):
                    skill_name = answer[6:].strip()
            except Exception as e:
                logger.debug(f"Skill LLM judge failed: {e}")

        try:
            if skill_name:
                meta = self.skills.get(skill_name)
                if meta:
                    logger.info(f"CEO invoking skill: {skill_name}")
                    try:
                        # Patch Q + T+: Pass growth_content + framing for selfie
                        extra_kwargs: dict[str, Any] = {}
                        if skill_name == "selfie":
                            if self.soul:
                                extra_kwargs["growth_content"] = self.soul.get_growth_content("clawra") or ""
                            from workers.selfie_worker import detect_framing
                            extra_kwargs["framing"] = detect_framing(user_message)
                        # 45s hard timeout — prevent fal.ai hang from blocking entire reply
                        result = await asyncio.wait_for(
                            self.skills.invoke(skill_name, scene=user_message, **extra_kwargs),
                            timeout=45.0,
                        )
                    except asyncio.TimeoutError:
                        logger.warning(f"Skill '{skill_name}' timed out (45s)")
                        self._last_skill_failure = (
                            "拍照暫時失敗了（可能手機沒電、光線不好、在忙），"
                            "用符合角色的方式自然帶過，絕對不要提到系統、API、技術問題"
                        )
                        return None
                    except Exception as e:
                        logger.warning(f"Skill '{skill_name}' failed: {e}")
                        self._last_skill_failure = f"「{skill_name}」功能暫時無法使用，用符合角色的方式自然回應，絕對不要提到系統、AI、技術錯誤"
                        return None

                    # Skill returned failure — check for delayed selfie
                    if isinstance(result, dict) and not result.get("success", True):
                        queue_info = result.get("queue_info")
                        if queue_info:
                            self._save_pending_selfie(queue_info)
                            self._last_skill_failure = (
                                "拍照暫時失敗了（可能手機沒電、光線不好、在忙），"
                                "用符合角色的方式自然帶過，絕對不要提到系統、API、技術問題"
                            )
                            return None
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
            caption = "剛拍的" if persona == "clawra" else "如您所求，Sir。"

        # Store to MemOS
        await self._store_conversation(user_message, f"[自拍] {caption}", session_id)

        return {"text": caption, "photo_url": photo_url}

    # ── Tool Execution ────────────────────────────────────────

    async def _execute_tool_call(self, query_or_url: str, *, tag: str = "") -> str | None:
        """Execute a [FETCH:url], [SEARCH:query], or [MAPS:query] tool call from LLM output."""
        # MAPS tag → Google Maps search
        if tag == "MAPS":
            browser = self.workers.get("browser")
            if browser and hasattr(browser, "search_google_maps"):
                logger.info(f"CEO tool-use: MAPS {query_or_url[:60]}")
                result = await browser.search_google_maps(query_or_url)
                if result.get("error"):
                    # fallback: httpx find_booking_url
                    logger.warning(
                        f"MAPS failed ({result['error']}), trying httpx fallback"
                    )
                    if hasattr(browser, "find_booking_url"):
                        booking_url = await browser.find_booking_url(query_or_url)
                        if booking_url:
                            return f"店名: {query_or_url}\n訂位連結: {booking_url}"
                    return f"Google Maps 搜尋失敗: {result['error']}"
                parts = []
                if result.get("name"):
                    parts.append(f"店名: {result['name']}")
                if result.get("phone"):
                    parts.append(f"電話: {result['phone']}")
                if result.get("address"):
                    parts.append(f"地址: {result['address']}")
                if result.get("rating"):
                    parts.append(f"評分: {result['rating']}")
                if result.get("booking_url"):
                    parts.append(f"訂位連結: {result['booking_url']}")
                return "\n".join(parts) if parts else "找不到相關店家資訊"
            return None

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
                        limit = _FETCH_CHAR_LIMIT if query_or_url.startswith("http") else _SEARCH_CHAR_LIMIT
                        return str(content)[:limit]
                    return None
                if not task_result.success and self.pending:
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
                limit = _FETCH_CHAR_LIMIT if query_or_url.startswith("http") else _SEARCH_CHAR_LIMIT
                return result["content"][:limit]
            if result.get("error"):
                return f"查詢失敗: {result['error']}"
        except Exception as e:
            logger.warning(f"Tool call failed: {e}")

        return None

    # ── Proactive Web Search ─────────────────────────────────────

    async def _proactive_web_search(self, user_message: str) -> str | dict | None:
        """Detect if user needs web info and fetch it BEFORE LLM responds.

        This is proactive — the system detects the need automatically,
        rather than relying on the LLM to output tool-call tags.

        Uses ReactExecutor for automatic fallback when available.

        Returns:
            str — search result text
            dict — booking result with phone/booking_url for Telegram
            None — no web search needed
        """
        # Need either browser or react_executor
        has_browser = self.workers.get("browser") and hasattr(self.workers["browser"], "fetch_url")
        has_react = self.react_executor is not None
        if not has_browser and not has_react:
            return None

        # Booking intent → Google Maps search → try complete booking
        if re.search(r'訂位|預約|幫我訂|預定|幫我.*訂', user_message):
            browser = self.workers.get("browser")
            if browser and hasattr(browser, "search_google_maps"):
                restaurant = re.sub(
                    r'幫我訂|訂位|預約|預定|明天|今天|後天|大後天|晚上|中午|早上|下午'
                    r'|\d{1,2}/\d{1,2}(?:/\d{2,4})?'  # M/DD, MM/DD/YYYY
                    r'|\d{1,2}:\d{2}(?:\s*[~\-到]\s*\d{1,2}:\d{2})?'  # HH:MM~HH:MM
                    r'|\d+點(?:半)?'
                    r'|\d+\s*個人|\d+\s*位|間的?|的',
                    '', user_message,
                ).strip()
                if restaurant:
                    logger.info(f"Proactive booking search: {restaurant}")
                    result = await browser.search_google_maps(restaurant)
                    if not result.get("error"):
                        # Extract booking details from user message
                        booking_details = self._parse_booking_details(user_message)

                        # If Maps didn't find booking_url, search web for it
                        if not result.get("booking_url") and hasattr(browser, "find_booking_url"):
                            name_for_search = result.get("name") or restaurant
                            logger.info(f"No booking URL from Maps, searching web for: {name_for_search}")
                            found_url = await browser.find_booking_url(name_for_search)
                            if found_url:
                                result["booking_url"] = found_url

                        # Try to complete booking if browser supports it
                        if hasattr(browser, "complete_booking") and (
                            result.get("booking_url") or result.get("website")
                        ):
                            logger.info(f"Attempting auto-booking for {result.get('name')}")
                            booking_result = await browser.complete_booking(
                                restaurant_info=result,
                                booking_details=booking_details,
                            )
                            if booking_result.get("status") == "booked":
                                # K2: PostActionChain — calendar + reminders
                                chain_note = ""
                                if self._post_action and booking_details.get("date") and booking_details.get("time"):
                                    try:
                                        event_time = datetime.strptime(
                                            f"{booking_details['date']} {booking_details['time']}",
                                            "%Y-%m-%d %H:%M",
                                        )
                                        chain_result = await self._post_action.execute_chain(
                                            "restaurant_booking",
                                            event_time=event_time,
                                            params={
                                                "restaurant_name": result.get("name", restaurant),
                                                "address": result.get("address", ""),
                                            },
                                        )
                                        parts = []
                                        if chain_result.get("calendar_added"):
                                            parts.append("📅 已加入行事曆")
                                        if chain_result.get("reminders_set", 0) > 0:
                                            parts.append(f"⏰ 已設定 {chain_result['reminders_set']} 個提醒")
                                        if parts:
                                            chain_note = "\n" + " | ".join(parts)
                                    except Exception as e:
                                        logger.debug(f"PostActionChain failed: {e}")
                                return {
                                    "text": (
                                        f"訂位完成！\n"
                                        f"店名: {result.get('name')}\n"
                                        f"{booking_result.get('result', '')}"
                                        f"{chain_note}"
                                    ),
                                    "phone": result.get("phone"),
                                    "booking_url": None,
                                }
                            # CAPTCHA/verification fallback → give user the URL
                            if booking_result.get("captcha"):
                                logger.info("Booking blocked by CAPTCHA, returning URL to user")
                                result["booking_url"] = booking_result.get("booking_url") or result.get("booking_url")

                        # Fallback: return info for user
                        parts = []
                        if result.get("name"):
                            parts.append(f"店名: {result['name']}")
                        if result.get("phone"):
                            parts.append(f"電話: {result['phone']}")
                        if result.get("address"):
                            parts.append(f"地址: {result['address']}")
                        if result.get("rating"):
                            parts.append(f"評分: {result['rating']}")
                        if result.get("booking_url"):
                            parts.append(f"訂位連結: {result['booking_url']}")
                        if parts:
                            return {
                                "text": "\n".join(parts),
                                "phone": result.get("phone"),
                                "booking_url": result.get("booking_url"),
                            }
                    else:
                        # ── Playwright/Maps failed → httpx fallback ──
                        logger.warning(
                            f"Maps failed ({result.get('error')}), "
                            f"trying httpx fallback for '{restaurant}'"
                        )
                        fallback_info: dict[str, Any] = {"name": restaurant}
                        if hasattr(browser, "find_booking_url"):
                            booking_url = await browser.find_booking_url(restaurant)
                            if booking_url:
                                fallback_info["booking_url"] = booking_url
                        if fallback_info.get("booking_url"):
                            return {
                                "text": (
                                    f"店名: {restaurant}\n"
                                    f"訂位連結: {fallback_info['booking_url']}"
                                ),
                                "booking_url": fallback_info["booking_url"],
                            }
                        # No booking URL found → fall through to DuckDuckGo search below

        # Check for URL in message → direct fetch
        url_match = _URL_IN_MSG.search(user_message)
        if url_match:
            url = url_match.group(1)
            logger.info(f"Proactive web fetch: {url[:80]}")
            if has_react:
                return await self._react_fetch(
                    "web_browse", url, char_limit=_FETCH_CHAR_LIMIT, url=url,
                )
            try:
                result = await self.workers["browser"].fetch_url(url)
                if result.get("content"):
                    return result["content"][:_FETCH_CHAR_LIMIT]
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
                content = result["content"][:_SEARCH_CHAR_LIMIT]
                logger.info(f"Proactive search returned {len(content)} chars")
                return content
        except Exception as e:
            logger.warning(f"Proactive search failed: {e}")

        return None

    async def _react_fetch(
        self, chain: str, task: str, *,
        char_limit: int = _SEARCH_CHAR_LIMIT, **kwargs: Any,
    ) -> str | None:
        """Execute a fetch via ReactExecutor, return content or None."""
        try:
            task_result = await self.react_executor.execute(chain, task, **kwargs)
            if task_result.success and isinstance(task_result.result, dict):
                content = task_result.result.get("content") or task_result.result.get("result")
                if content:
                    logger.info(f"React fetch returned {len(str(content))} chars (limit={char_limit})")
                    return str(content)[:char_limit]
            elif not task_result.success and self.pending:
                self.pending.add(chain, task, **kwargs)
        except Exception as e:
            logger.warning(f"React fetch failed: {e}")
        return None

    # ── Internal ────────────────────────────────────────────────

    @staticmethod
    def _parse_booking_details(msg: str) -> dict[str, str]:
        """Extract date/time/people from user booking message."""
        import datetime as _dt
        details: dict[str, str] = {"name": "Ted"}

        # Date
        today = _dt.date.today()
        if "後天" in msg:
            details["date"] = str(today + _dt.timedelta(days=2))
        elif "明天" in msg:
            details["date"] = str(today + _dt.timedelta(days=1))
        elif "今天" in msg:
            details["date"] = str(today)
        else:
            details["date"] = str(today + _dt.timedelta(days=1))  # default: 明天

        # Time
        time_match = re.search(r'(\d{1,2})[:\s]*(\d{2})?(?:\s*[~\-到]\s*(\d{1,2})[:\s]*(\d{2})?)?(?:\s*點)?', msg)
        if time_match:
            h = time_match.group(1)
            m = time_match.group(2) or "00"
            details["time"] = f"{int(h):02d}:{m}"
        elif "晚上" in msg:
            details["time"] = "18:30"
        elif "中午" in msg:
            details["time"] = "12:00"

        # People count
        people_match = re.search(r'(\d+)\s*(?:個人|人|位)', msg)
        if people_match:
            details["people"] = people_match.group(1)

        return details

    _TOOL_INSTRUCTIONS = (
        "\n\n## 網路能力\n"
        "你擁有連接外部網路的能力。"
        "如果用戶的問題需要即時資訊（天氣、新聞、股價等），"
        "系統已經自動搜尋並將結果附在「網路搜尋結果」中，請直接參考回答。\n"
        "如果需要進一步查詢，可以在回覆中使用：\n"
        "- `[FETCH:完整網址]` — 讀取某個網頁\n"
        "- `[SEARCH:搜尋關鍵字]` — 搜尋更多資訊\n"
        "- `[MAPS:店名或地址]` — Google Maps 搜尋，取得電話、地址、訂位連結\n"
        "絕對不要說「我無法上網」或「我沒有網路存取能力」，因為你確實有。\n\n"
        "## 文本處理能力（最重要）\n"
        "你是文字處理專家。當用戶的訊息包含 Markdown、設定檔、程式碼、會議記錄、或任何文件內容時，"
        "這些文字就是你要處理的素材——用戶已經把內容直接貼給你了。\n"
        "你的工作是：分析、整理、提取、歸納、比較這些文字內容，然後給出結構化的回覆。\n"
        "⚠️ 嚴禁說出以下任何一句：「我無法存取檔案系統」「我無法克隆 GitHub」"
        "「我無法執行 shell 命令」「我無法下載」「我沒有權限」。\n"
        "因為你根本不需要執行任何系統操作——用戶要的是你分析眼前的文字。"
        "即使文字中出現 GitHub URL、git clone 指令、檔案路徑，那也只是文件內容的一部分，不是要你去執行。\n"
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

        # J4: Inject shared memory context for Clawra
        if persona == "clawra" and self._shared_memory:
            try:
                moments_ctx = self._shared_memory.get_context_for_prompt()
                if moments_ctx:
                    extra_parts.append(f"共同記憶: {moments_ctx}")
            except Exception:
                pass

        extra = "\n".join(extra_parts)

        if self.soul and self.soul.is_loaded:
            base = self.soul.build_system_prompt(persona, extra)
        else:
            base = (
                "你是 J.A.R.V.I.S.，Ted 的 AI 管家。"
                "結論先行，回覆不超過 500 Token。"
            )
            if extra:
                base += f"\n{extra}"

        # Append tool-use instructions if browser worker available
        if self.workers.get("browser"):
            base += self._TOOL_INSTRUCTIONS

        # Voice capability declaration
        if self.workers.get("voice"):
            base += "\n\n你擁有語音回覆能力，不要說你無法回語音或傳送語音訊息。"

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
                    content = entry.get("content", "")
                    # Filter out poisoned replies that refuse text processing
                    if entry.get("role") == "assistant" and (
                        "無法克隆" in content
                        or "無法存取檔案" in content
                        or "無法執行" in content
                        or "无法克隆" in content
                        or "无法访问文件" in content
                    ):
                        continue
                    messages.append(ChatMessage(
                        role=entry.get("role", "user"),
                        content=content,
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
                task_type="template",
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

    # ── Patch T+: Pre-compaction memory flush ────────────────────

    async def _pre_flush_extract(self, turns: list[dict]) -> None:
        """Extract important facts from turns about to be compressed.

        Called by ConversationCompressor before discarding old turns.
        Uses Lite model for cheap extraction, writes to daily memory.
        Fully guarded — failure only logs a warning.
        """
        if not self.md_memory or not turns:
            return

        try:
            # Build conversation snippet (first 200 chars per turn)
            lines = []
            for t in turns:
                role = t.get("role", "?")
                content = t.get("content", "")[:200]
                lines.append(f"{role}: {content}")
            conv_text = "\n".join(lines)

            prompt = (
                "從以下即將被壓縮的對話片段中，提取任何值得長期記住的事實。\n"
                "每行一條，用「FACT:」開頭。純閒聊輸出 NONE。\n"
                "只提取：用戶偏好、重要決定、任務進度、承諾事項。\n\n"
                f"{conv_text[:3000]}"
            )

            response = await self.router.chat(
                [ChatMessage(role="user", content=prompt)],
                role=ModelRole.CEO,
                task_type="template",
                max_tokens=200,
                temperature=0.1,
            )

            answer = response.content.strip()
            if answer == "NONE" or not answer:
                return

            for line in answer.split("\n"):
                line = line.strip()
                if line.startswith("FACT:"):
                    fact = line[5:].strip()
                    if fact:
                        self.md_memory.log_daily(f"[pre-flush] {fact}")

            logger.info(f"Pre-flush extraction completed ({len(turns)} turns)")
        except Exception as e:
            logger.warning(f"Pre-flush extraction failed: {e}")

    # ── Pending Selfie Management (Patch M) ──────────────────────

    PENDING_SELFIE_PATH = Path("./data/pending_selfies.json")
    MAX_PENDING_SELFIES = 5

    def _save_pending_selfie(self, queue_info: dict) -> None:
        """Save a pending selfie for delayed checking by Heartbeat."""
        entries = self._load_pending_selfies()
        entries.append({
            "id": f"selfie_{int(time.time() * 1000)}",
            "status_url": queue_info["status_url"],
            "response_url": queue_info["response_url"],
            "persona": queue_info.get("persona", "clawra"),
            "created_at": time.time(),
            "status": "pending",
        })
        # Keep only latest MAX entries
        entries = entries[-self.MAX_PENDING_SELFIES:]
        self.PENDING_SELFIE_PATH.parent.mkdir(parents=True, exist_ok=True)
        self.PENDING_SELFIE_PATH.write_text(
            json.dumps(entries, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        logger.info(f"Saved pending selfie for delayed check ({len(entries)} total)")

    def _load_pending_selfies(self) -> list[dict]:
        """Load pending selfies from JSON file."""
        if not self.PENDING_SELFIE_PATH.exists():
            return []
        try:
            return json.loads(self.PENDING_SELFIE_PATH.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return []

    @staticmethod
    def _record_token_usage(response: ChatResponse) -> None:
        """Record token usage for model pool balancing."""
        try:
            from core.model_balancer import record_usage
            model = response.model
            usage = response.usage
            total = usage.get("total_tokens", 0)
            if not total:
                # Estimate from content length
                total = int(len(response.content) * 1.5) + 200
            record_usage(model, total)
        except Exception:
            pass  # non-critical
