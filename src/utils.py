import sys
from loguru import logger
import httpx
from config.settings import config
from pathlib import Path

# Setup Loguru Logger
log_dir = Path("logs")
log_dir.mkdir(exist_ok=True)

logger.remove()
logger.add(
    sys.stdout, 
    colorize=True, 
    format="<green>{time:HH:mm:ss}</green> | <level>{level: <7}</level> | <level>{message}</level>"
)
logger.add(
    log_dir / "bot_{time:YYYY-MM-DD}.log", 
    rotation="00:00", # Rotate daily at midnight
    retention="30 days",
    level="INFO"
)

async def send_telegram_message(message: str) -> bool:
    """
    Sends a message securely to the configured Telegram chat.
    Doesn't raise exceptions, just logs heavily on failure.
    """
    if not config.telegram_bot_token or not config.telegram_chat_id:
        logger.debug("Telegram credentials not set. Skipping message.")
        return False
        
    url = f"https://api.telegram.org/bot{config.telegram_bot_token}/sendMessage"
    payload = {
        "chat_id": config.telegram_chat_id,
        "text": message,
        "parse_mode": "HTML"
    }
    
    try:
        async with httpx.AsyncClient() as client:
            response = await client.post(url, json=payload, timeout=10.0)
            response.raise_for_status()
            return True
    except Exception as e:
        logger.error(f"Failed to send Telegram message: {e}")
        return False
