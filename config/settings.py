import os
from dotenv import load_dotenv
from pydantic import Field
from pydantic_settings import BaseSettings
from typing import Dict

load_dotenv()

class Settings(BaseSettings):
    # API Config
    clob_api_key: str = ""
    clob_api_secret: str = ""
    clob_api_passphrase: str = ""
    polymarket_private_key: str = "" # Fixed from poly_private_key
    funder_address: str = "" # Added back
    chain_id: int = 137 # Added back
    
    # AI Config
    gemini_api_keys_str: str = "" # Comma separated list of keys
    gemini_model: str = "gemini-2.5-flash-lite"
    
    # Notifications
    telegram_bot_token: str = ""
    telegram_chat_id: str = ""
    
    # Trading Params
    dry_run: bool = True
    max_trade_size_usd: float = 20.0
    scan_interval_hours: int = 1
    scan_days_ahead: int = 2
    resolution_check_interval_minutes: int = 15
    
    # Kelly & EV Thresholds
    kelly_fraction: float = 0.1
    # EV thresholds per city (default + specific overrides)
    ev_threshold: Dict[str, float] = {
        "default": 0.08,
        "London": 0.12 # Example override
    }
    
    # City-ICAO Mapping for Weather
    city_icao_mapping: Dict[str, str] = {
        "London": "EGLL",
        "Seoul": "RKSS",
        "Chicago": "KORD",
        "Dallas": "KDFW",
        "Atlanta": "KATL",
        "Tokyo": "RJTT",
        "Shanghai": "ZSSS",
        "Singapore": "WSSS"
    }

    class Config:
        env_file = ".env"
        extra = "ignore"

config = Settings()
