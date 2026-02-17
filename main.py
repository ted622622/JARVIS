"""J.A.R.V.I.S. — Entry point.

Startup sequence (per Blueprint v2.1 Section 11):
 1. Load config.yaml + .env
 2. Initialize Security Gate
 3. Initialize API Clients (Nvidia / Zhipu / OpenRouter / fal.ai)
 4. Initialize Model Router + Failover
 5. Initialize MemOS
 6. Initialize Telegram Client
 7. Initialize CEO Agent (inject SOUL.md + Router + MemOS)
 8. Initialize Workers (Code / Interpreter / Browser / Vision / Selfie)
 9. Start Heartbeat Scheduler
10. Start Telegram Polling / Webhook
11. [Optional] Initialize FalClient + Selfie Worker (if FAL_KEY present)
12. Log: "J.A.R.V.I.S. is alive. All systems nominal."
"""

from __future__ import annotations

import asyncio
import os
import sys
from pathlib import Path

import yaml
from dotenv import load_dotenv
from loguru import logger


async def main() -> None:
    # ── Step 1: Load config + env ─────────────────────────────────
    load_dotenv()

    logger.remove()
    logger.add(sys.stderr, level="INFO")
    Path("data/logs").mkdir(parents=True, exist_ok=True)
    logger.add(
        "data/logs/jarvis.log",
        rotation="50 MB",
        retention="5 days",
        level="DEBUG",
    )

    logger.info("Initializing J.A.R.V.I.S. ...")

    with open("config/config.yaml", "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    # ── Step 2: Initialize Security Gate ──────────────────────────
    from core.security_gate import SecurityGate

    security = SecurityGate(project_root=str(Path.cwd()))
    logger.info("  [2/12] Security Gate initialized")

    # ── Step 3: Initialize API Clients ────────────────────────────
    from clients.groq_chat_client import GroqChatClient
    from clients.nvidia_client import NvidiaClient
    from clients.openrouter_client import OpenRouterClient
    from clients.zhipu_client import ZhipuClient

    models = config.get("models", {})

    nvidia = NvidiaClient(
        api_key=os.environ.get("NVIDIA_API_KEY", ""),
        base_url=os.environ.get(
            "NVIDIA_BASE_URL", "https://integrate.api.nvidia.com/v1"
        ),
        model=models.get("ceo", {}).get("primary", {}).get("model"),
        rpm_limit=models.get("ceo", {}).get("primary", {}).get("rpm_limit", 40),
    )

    zhipu = ZhipuClient(
        api_key=os.environ.get("ZHIPU_API_KEY", ""),
        vision_model=models.get("vision", {}).get("primary", {}).get("model"),
        image_model="cogview-4-250304",
    )

    openrouter = OpenRouterClient(
        api_key=os.environ.get("OPENROUTER_API_KEY", ""),
        model=models.get("ceo", {}).get("tertiary", {}).get("model")
            or models.get("ceo", {}).get("backup", {}).get("model"),
    )

    groq_chat = None
    groq_key_chat = os.environ.get("GROQ_API_KEY", "")
    if groq_key_chat and groq_key_chat != "gsk_your-groq-key-here":
        groq_chat = GroqChatClient(
            api_key=groq_key_chat,
            model=models.get("ceo", {}).get("backup", {}).get("model", "llama-3.3-70b-versatile"),
        )

    logger.info("  [3/12] API Clients initialized (Nvidia, Zhipu, OpenRouter%s)", ", Groq" if groq_chat else "")

    # ── Step 4: Initialize Model Router + Failover ────────────────
    from core.model_router import ModelRouter

    router = ModelRouter(
        nvidia_client=nvidia,
        zhipu_client=zhipu,
        openrouter_client=openrouter,
        groq_client=groq_chat,
        config=config,
    )

    # Quick health check — mark failed providers DOWN immediately
    from core.model_router import ModelRole, ProviderStatus

    health = await router.health_check_all()
    for provider, is_healthy in health.items():
        status = "OK" if is_healthy else "UNREACHABLE"
        logger.info(f"    {provider}: {status}")
        if not is_healthy:
            router._provider_status[provider] = ProviderStatus.DOWN
            # If zhipu health check fails, also mark zhipu_ceo down
            if provider == "zhipu":
                router._provider_status["zhipu_ceo"] = ProviderStatus.DOWN
            logger.warning(f"    {provider} marked DOWN at startup (skip on first request)")

    ceo_chain = " → ".join(
        name for name, _ in router._get_chain_for_role(ModelRole.CEO)
    )
    logger.info(f"    CEO chain: {ceo_chain}")

    logger.info("  [4/12] Model Router initialized")

    # ── Step 5: Initialize MemOS ──────────────────────────────────
    from memory.memos_manager import MemOS

    memos_db = config.get("memos", {}).get("database_path", "./data/memos.db")
    Path(memos_db).parent.mkdir(parents=True, exist_ok=True)
    memos = MemOS(db_path=memos_db)
    await memos.init()
    logger.info("  [5/12] MemOS initialized")

    # ── Step 6: Initialize Telegram Client ────────────────────────
    from clients.telegram_client import TelegramClient

    telegram = TelegramClient(
        jarvis_token=os.environ.get("TELEGRAM_JARVIS_BOT_TOKEN", ""),
        clawra_token=os.environ.get("TELEGRAM_CLAWRA_BOT_TOKEN", ""),
        chat_id=os.environ.get("TELEGRAM_CHAT_ID", ""),
        allowed_user_ids=os.environ.get("TELEGRAM_ALLOWED_USER_IDS", ""),
    )
    await telegram.init()
    logger.info("  [6/12] Telegram Client initialized")

    # ── Step 7: Initialize CEO Agent ──────────────────────────────
    from core.ceo_agent import CEOAgent
    from core.emotion import EmotionClassifier
    from core.soul import Soul
    from memory.markdown_memory import MarkdownMemory
    from skills.registry import SkillRegistry

    soul = Soul(config_dir="./config", memory_dir="./memory")
    soul.load()

    md_memory = MarkdownMemory("./memory")
    logger.info("  [7a/12] Markdown memory initialized")

    emotion = EmotionClassifier(model_router=router, memos=memos)

    registry = SkillRegistry("./skills")
    registry.scan()

    from core.memory_search import MemorySearch

    memory_search = MemorySearch("./memory")
    chunk_count = memory_search.build_index()
    logger.info(f"  [7b/12] Memory search index built ({chunk_count} chunks)")

    ceo = CEOAgent(
        model_router=router,
        soul=soul,
        emotion_classifier=emotion,
        memos=memos,
        skill_registry=registry,
        security_gate=security,
        markdown_memory=md_memory,
    )

    # 7c: Hybrid search (BM25 + Gemini Embedding) if GEMINI_API_KEY available
    gemini_key = os.environ.get("GEMINI_API_KEY", "")
    if gemini_key:
        from core.embedding_search import EmbeddingIndex, HybridSearch

        embedding_index = EmbeddingIndex(
            memory_dir="./memory",
            cache_path="./data/embedding_index.json",
            api_key=gemini_key,
        )
        try:
            await embedding_index.build_index()
            hybrid_search = HybridSearch(bm25=memory_search, embedding=embedding_index)
            ceo.memory_search = hybrid_search
            logger.info("  [7c/12] Hybrid search initialized (BM25 + Gemini Embedding)")
        except Exception as e:
            logger.warning(f"  [7c/12] Embedding init failed, using BM25 only: {e}")
            ceo.memory_search = memory_search
    else:
        ceo.memory_search = memory_search
        logger.info("  [7c/12] Gemini Embedding skipped (no GEMINI_API_KEY)")

    logger.info("  [7/12] CEO Agent initialized (persona: jarvis)")

    # ── Step 8: Initialize Workers ────────────────────────────────
    from workers import (
        BrowserWorker,
        CodeWorker,
        InterpreterWorker,
        SelfieWorker,
        VisionWorker,
        VoiceWorker,
    )

    voice_cache_dir = config.get("voice", {}).get("cache_dir", "./data/voice_cache")
    voice_worker = VoiceWorker(
        cache_dir=voice_cache_dir,
        azure_key=os.environ.get("AZURE_SPEECH_KEY", ""),
        azure_region=os.environ.get("AZURE_SPEECH_REGION", ""),
        zhipu_key=os.environ.get("ZHIPU_API_KEY", ""),
        zhipu_voice=os.environ.get("ZHIPU_TTS_VOICE", "tongtong"),
    )

    workers = {
        "code": CodeWorker(model_router=router),
        "interpreter": InterpreterWorker(security_gate=security),
        "browser": BrowserWorker(security_gate=security, model_router=router),
        "vision": VisionWorker(model_router=router),
        "selfie": SelfieWorker(skill_registry=registry),
        "voice": voice_worker,
    }
    # H2: Knowledge Worker
    from workers.knowledge_worker import KnowledgeWorker

    workers["knowledge"] = KnowledgeWorker(
        model_router=router, memos=memos, memory_search=memory_search,
    )

    # H0 v2: GogWorker (Google Workspace via gog CLI)
    from workers.gog_worker import GogWorker

    gog_worker = GogWorker(account=os.environ.get("GOG_ACCOUNT", ""))
    if gog_worker.is_available:
        workers["gog"] = gog_worker
        logger.info("    gog CLI: ready")
    else:
        logger.warning("    gog CLI: not available (Calendar/Gmail via gog disabled)")

    # H2 v2: Assist Worker — 做到 90% 給選項
    from workers.assist_worker import AssistWorker

    workers["assist"] = AssistWorker(
        model_router=router,
        memory_search=memory_search,
        gog_worker=gog_worker if gog_worker.is_available else None,
    )

    # S3: SearchWorker — consolidated DuckDuckGo search
    from workers.search_worker import SearchWorker

    workers["search"] = SearchWorker()

    # S2: TranscribeWorker — long audio ASR + meeting summary
    from workers.transcribe_worker import TranscribeWorker

    transcribe_worker = TranscribeWorker(
        model_router=router,
        zhipu_key=os.environ.get("ZHIPU_API_KEY", ""),
    )
    workers["transcribe"] = transcribe_worker

    ceo.workers = workers
    logger.info(f"  [8/12] Workers initialized ({len(workers)} workers)")

    # H1: ReactExecutor
    from core.react_executor import ReactExecutor, FuseState

    shared_fuse = FuseState()
    react_exec = ReactExecutor(workers=workers, fuse=shared_fuse)
    ceo._react = react_exec
    ceo._fuse = shared_fuse

    # H4: PendingTaskManager
    from core.pending_tasks import PendingTaskManager

    pending_mgr = PendingTaskManager("./data/pending_tasks.json")
    pending_mgr.load()
    ceo.pending = pending_mgr
    logger.info("  [8c/12] ReactExecutor + PendingTasks initialized")

    # I5: SessionManager + LoginAssistant
    from core.session_manager import SessionManager
    from core.login_assistant import LoginAssistant

    session_mgr = SessionManager("./data/session_status.json")
    login_assistant = LoginAssistant(session_manager=session_mgr)
    ceo._session_mgr = session_mgr
    ceo._login_assistant = login_assistant
    logger.info("  [8d/12] Patch I modules initialized (TaskRouter, Compressor, SessionManager)")

    # Patch J: Soul evolution
    from core.soul_growth import SoulGrowth
    from core.shared_memory import SharedMemory
    from core.soul_guard import SoulGuard

    soul_growth = SoulGrowth(memory_dir="./memory")
    shared_memory = SharedMemory(memory_dir="./memory")
    soul_guard = SoulGuard(config_dir="./config", memory_dir="./memory")
    ceo._soul_growth = soul_growth
    ceo._shared_memory = shared_memory
    logger.info("  [8e/12] Patch J modules initialized (SoulGrowth, SharedMemory, SoulGuard)")

    # Inject voice + transcribe workers into Telegram client
    telegram.voice_worker = voice_worker
    telegram.transcribe_worker = transcribe_worker

    # Initialize Groq STT client if key present
    groq_key = os.environ.get("GROQ_API_KEY", "")
    if groq_key and groq_key != "gsk_your-groq-key-here":
        from clients.groq_stt_client import GroqSTTClient

        stt_model = config.get("voice", {}).get("stt", {}).get("model", "whisper-large-v3-turbo")
        telegram.stt_client = GroqSTTClient(api_key=groq_key, model=stt_model)
        logger.info("  [8b/12] Groq STT client initialized (Whisper)")
    else:
        logger.warning("  [8b/12] Groq STT skipped (no GROQ_API_KEY)")

    # ── Step 9: Start Heartbeat Scheduler ─────────────────────────
    from core.heartbeat import Heartbeat
    from core.survival_gate import SurvivalGate
    from memory.token_tracker import TokenSavingTracker

    token_tracker = TokenSavingTracker(memos._db)
    await token_tracker.init()

    # Step 11 (early): Initialize FalClient if FAL_KEY present
    fal_client = None
    fal_key = os.environ.get("FAL_KEY", "")
    if fal_key and fal_key != "your-fal-key-here":
        from clients.fal_client import FalClient
        fal_client = FalClient(api_key=fal_key)
        logger.info("  [11/12] FalClient initialized (FLUX Kontext)")

    survival = SurvivalGate(
        model_router=router,
        token_tracker=token_tracker,
        fal_client=fal_client,
        backup_dir=config.get("backup", {}).get("destination", "./backups/"),
    )

    from clients.weather_client import WeatherClient

    weather = WeatherClient(
        latitude=float(os.environ.get("WEATHER_LATITUDE", "25.0143")),
        longitude=float(os.environ.get("WEATHER_LONGITUDE", "121.4673")),
    )

    # K1: ReminderManager
    from core.reminder_manager import ReminderManager

    reminder_mgr = ReminderManager(
        path="./data/reminders.json",
        scheduler=None,  # set after heartbeat.start()
        telegram=telegram,
    )

    heartbeat = Heartbeat(
        model_router=router,
        memos=memos,
        telegram=telegram,
        survival_gate=survival,
        config=config,
        weather_client=weather,
        pending_tasks=pending_mgr,
        react_executor=react_exec,
        gog_worker=gog_worker if gog_worker.is_available else None,
        reminder_manager=reminder_mgr,
    )
    heartbeat.start()

    # Wire scheduler into ReminderManager and load existing reminders
    reminder_mgr._scheduler = heartbeat.scheduler
    loaded_reminders = reminder_mgr.load_into_scheduler()
    logger.info(f"  [9/12] Heartbeat Scheduler started ({loaded_reminders} reminders loaded)")

    # K2: PostActionChain → CEO
    from core.post_action_chain import PostActionChain

    post_action = PostActionChain(
        gog_worker=gog_worker if gog_worker.is_available else None,
        reminder_manager=reminder_mgr,
    )
    ceo._post_action = post_action
    logger.info("  [9b/12] PostActionChain initialized")

    # S4: SkillLearner — observe user patterns, propose automations
    from core.skill_learner import SkillLearner

    skill_learner = SkillLearner(
        scheduler=heartbeat.scheduler,
        telegram=telegram,
        model_router=router,
    )
    ceo._skill_learner = skill_learner
    # Daily cron job at 03:30 to propose learned skills
    heartbeat.scheduler.add_job(
        skill_learner.propose_skills,
        "cron",
        hour=3,
        minute=30,
        id="skill_learner_propose",
        name="Skill Learner Proposal",
    )
    logger.info("  [9c/12] SkillLearner initialized (daily 03:30 proposals)")

    # ── Step 10: Telegram polling ─────────────────────────────────
    async def on_telegram_message(user_text: str, chat_id: int, persona: str = "jarvis") -> dict | str:
        """Handle incoming Telegram messages via CEO Agent."""
        result = await ceo.handle_message(user_text, persona=persona)
        emotion = getattr(ceo, "_last_emotion", None)
        if isinstance(result, str):
            return {"text": result, "emotion": emotion}
        if isinstance(result, dict):
            result.setdefault("emotion", emotion)
        return result

    telegram.set_message_handler(on_telegram_message)
    tg_apps = telegram.build_applications()
    for tg_app in tg_apps:
        await tg_app.initialize()
        await tg_app.start()
        await tg_app.updater.start_polling(drop_pending_updates=True)
    if tg_apps:
        logger.info(f"  [10/12] Telegram polling started ({len(tg_apps)} bots)")
    else:
        logger.warning("  [10/12] Telegram polling skipped (no bot token)")

    # ── Step 12: All systems go ───────────────────────────────────
    logger.info("")
    logger.info("=" * 50)
    logger.info("  J.A.R.V.I.S. is alive. All systems nominal.")
    logger.info("=" * 50)

    # First-boot greeting: send morning brief to Telegram
    logger.info("Sending first-boot morning brief to Telegram ...")
    try:
        brief = await heartbeat.morning_brief()
        logger.info(f"Morning brief sent ({len(brief)} chars)")
    except Exception as e:
        logger.error(f"Morning brief failed: {e}")

    # Keep running until interrupted
    try:
        while True:
            await asyncio.sleep(60)
    except (KeyboardInterrupt, asyncio.CancelledError):
        logger.info("Shutting down J.A.R.V.I.S. ...")
    finally:
        heartbeat.stop()
        for tg_app in tg_apps:
            await tg_app.updater.stop()
            await tg_app.stop()
            await tg_app.shutdown()
        await telegram.close()
        await memos.close()
        await router.close()  # also closes groq if present
        if fal_client:
            await fal_client.close()
        await voice_worker.close()
        await workers["browser"].close()  # also calls close_playwright()
        logger.info("J.A.R.V.I.S. offline. Good night, Sir.")


if __name__ == "__main__":
    asyncio.run(main())
