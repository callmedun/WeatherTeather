import asyncio
import sys
import os

# Ensure src is in python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils import logger, send_telegram_message
from src.scheduler import BotScheduler
from config.settings import config
from src.telegram_bot import start_telegram_bot

async def start_bot():
    logger.info("Initializing Polymarket Weather Bot...")
    
    if config.dry_run:
        logger.warning(f"Starting in DRY_RUN mode. No real capital will be risked.")
    else:
        logger.warning(f"Starting in LIVE mode. Capital is at risk.")
        
    await send_telegram_message("🚀 Polymarket bot started! " + ("(DRY RUN)" if config.dry_run else "(LIVE)"))

    # Launch background PTB polling first so it isn't blocked by initial scans
    await start_telegram_bot()
    
    from src.scheduler import bot_scheduler
    await bot_scheduler.start()

    # Keep the main thread alive
    try:
        while True:
            await asyncio.sleep(3600)
    except (KeyboardInterrupt, SystemExit):
        logger.info("Shutting down bot...")
        await send_telegram_message("🛑 Bot shutting down.")

if __name__ == "__main__":
    try:
        if sys.platform == 'win32':
            asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
        asyncio.run(start_bot())
    except KeyboardInterrupt:
        logger.info("Interrupted by user. Exiting.")
