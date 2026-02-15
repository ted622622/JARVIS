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

    security = SecurityGate(config=config)
    logger.info("  [2/12] Security Gate initialized")

    # ── Step 3: Initialize API Clients ────────────────────────────
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
        model=models.get("ceo", {}).get("backup", {}).get("model"),
    )

    logger.info("  [3/12] API Clients initialized (Nvidia, Zhipu, OpenRouter)")

    # ── Step 4: Initialize Model Router + Failover ────────────────
    from core.model_router import ModelRouter

    router = ModelRouter(
        nvidia_client=nvidia,
        zhipu_client=zhipu,
        openrouter_client=openrouter,
        config=config,
    )

    # Quick health check
    health = await router.health_check_all()
    for provider, is_healthy in health.items():
        status = "OK" if is_healthy else "UNREACHABLE"
        logger.info(f"    {provider}: {status}")

    logger.info("  [4/12] Model Router initialized")

    # ── Step 5: Initialize MemOS ──────────────────────────────────
    from memory.memos_manager import MemOS

    memos_db = config.get("memos", {}).get("database_path", "./data/memos.db")
    Path(memos_db).parent.mkdir(parents=True, exist_ok=True)
    memos = MemOS(db_path=memos_db)
    await memos.initialize()
    logger.info("  [5/12] MemOS initialized")

    # ── Step 6: Initialize Telegram Client ────────────────────────
    from clients.telegram_client import TelegramClient

    telegram = TelegramClient(
        jarvis_token=os.environ.get("TELEGRAM_JARVIS_BOT_TOKEN", ""),
        clawra_token=os.environ.get("TELEGRAM_CLAWRA_BOT_TOKEN", ""),
        chat_id=os.environ.get("TELEGRAM_CHAT_ID", ""),
    )
    logger.info("  [6/12] Telegram Client initialized")

    # ── Step 7: Initialize CEO Agent ──────────────────────────────
    from core.ceo_agent import CEOAgent
    from core.emotion import EmotionClassifier
    from core.soul import Soul
    from skills.registry import SkillRegistry

    soul = Soul(config.get("identity", {}).get("soul_file", "./config/SOUL.md"))
    soul.load()

    emotion = EmotionClassifier(model_router=router, memos=memos)

    registry = SkillRegistry("./skills")
    registry.scan()

    ceo = CEOAgent(
        model_router=router,
        soul=soul,
        emotion_classifier=emotion,
        memos=memos,
        skill_registry=registry,
        security_gate=security,
    )
    logger.info("  [7/12] CEO Agent initialized (persona: jarvis)")

    # ── Step 8: Initialize Workers ────────────────────────────────
    from workers import (
        BrowserWorker,
        CodeWorker,
        InterpreterWorker,
        SelfieWorker,
        VisionWorker,
    )

    workers = {
        "code": CodeWorker(model_router=router),
        "interpreter": InterpreterWorker(security_gate=security),
        "browser": BrowserWorker(security_gate=security),
        "vision": VisionWorker(model_router=router),
        "selfie": SelfieWorker(skill_registry=registry),
    }
    ceo.workers = workers
    logger.info(f"  [8/12] Workers initialized ({len(workers)} workers)")

    # ── Step 9: Start Heartbeat Scheduler ─────────────────────────
    from core.heartbeat import Heartbeat
    from core.survival_gate import SurvivalGate
    from memory.token_tracker import TokenSavingTracker

    token_tracker = TokenSavingTracker(db_path=memos_db)

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

    heartbeat = Heartbeat(
        model_router=router,
        memos=memos,
        telegram=telegram,
        survival_gate=survival,
        config=config,
    )
    heartbeat.start()
    logger.info("  [9/12] Heartbeat Scheduler started")

    # ── Step 10: Telegram polling ─────────────────────────────────
    # Note: actual polling requires webhook or long-polling loop
    # This will be activated when the system runs as a service
    logger.info("  [10/12] Telegram ready (polling starts on demand)")

    # ── Step 12: All systems go ───────────────────────────────────
    logger.info("")
    logger.info("=" * 50)
    logger.info("  J.A.R.V.I.S. is alive. All systems nominal.")
    logger.info("=" * 50)

    # Keep running until interrupted
    try:
        while True:
            await asyncio.sleep(60)
    except (KeyboardInterrupt, asyncio.CancelledError):
        logger.info("Shutting down J.A.R.V.I.S. ...")
    finally:
        heartbeat.stop()
        await router.close()
        if fal_client:
            await fal_client.close()
        logger.info("J.A.R.V.I.S. offline. Good night, Sir.")


if __name__ == "__main__":
    asyncio.run(main())
