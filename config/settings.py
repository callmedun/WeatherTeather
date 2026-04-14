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
    polymarket_private_key: str = ""
    funder_address: str = ""
    chain_id: int = 137
    
    # AI Config
    gemini_api_keys_str: str = "" # Comma separated list of keys
    gemini_model: str = "gemini-flash-lite-latest"
    
    # Notifications
    telegram_bot_token: str = ""
    telegram_chat_id: str = ""
    telegram_menu_enabled: bool = True # Required by telegram_bot.py
    
    # Trading Params
    dry_run: bool = True
    max_trade_size_usd: float = 20.0
    default_trade_size: float = 15.0 # Required by PortfolioManager
    min_trade_usd: float = 5.0
    min_shares: int = 1
    max_decimals_amount: int = 2
    scan_interval_hours: int = 1
    scan_days_ahead: int = 2
    resolution_check_interval_minutes: int = 15
    
    # Risk & Exposure
    max_total_exposure: float = 0.5 # Required by PortfolioManager (50% of bankroll)
    max_city_exposure: float = 0.1 # Required by PortfolioManager (10% of bankroll)
    
    # Calibration
    calibration_min_trades: int = 3
    
    # Kelly & EV Thresholds
    kelly_fraction: float = 0.1
    # EV thresholds per city (default + specific overrides)
    ev_threshold: Dict[str, float] = {
        "default": 0.08,
        "London": 0.12
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
