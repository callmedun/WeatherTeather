import os
from pydantic_settings import BaseSettings, SettingsConfigDict
from typing import Dict

class Settings(BaseSettings):
    # Polymarket Config
    polymarket_private_key: str = ""
    funder_address: str = ""
    chain_id: int = 137
    
    # AI Config
    gemini_api_keys_str: str = "" # Comma separated list of keys
    gemini_model: str = "gemini-1.5-flash"
    
    # Notifications
    telegram_bot_token: str = ""
    telegram_chat_id: str = ""
    telegram_menu_enabled: bool = True
    
    # Trading Config
    scan_days_ahead: int = 2
    scan_interval_hours: int = 1
    max_total_exposure: float = 0.20
    max_city_exposure: float = 0.05
    default_trade_size: float = 500.0
    dry_run: bool = True
    
    # Polymarket Order Constraints
    min_trade_usd: float = 2.0
    min_shares: int = 1
    max_decimals_amount: int = 2
    
    # Quantitative parameters
    kelly_fraction: float = 0.35
    ev_threshold: Dict[str, float] = {
        "default": 0.08,
        "London": 0.10,
        "EGLL": 0.10
    }
    
    # Self-Calibration Module
    calibration_min_trades: int = 30
    resolution_check_interval_minutes: int = 60

    # Dictionary of supported cities to their ICAO codes
    # This allows mapping the text "Highest temperature in [CITY]" to METAR data.
    city_icao_mapping: Dict[str, str] = {
        "Shanghai": "ZSPD", # PVG / ZSPD
        "Seoul": "RKSI",
        "London": "EGLL",
        "Miami": "KMIA",
        "Dallas": "KDFW",
        "Atlanta": "KATL",
        "Madrid": "LEMD",
        "Singapore": "WSSS",
        # "Mexico City": "MMMX",
        # "New York": "KJFK",
        # "Berlin": "EDDB",
        # "Paris": "LFPG",
        # "Tokyo": "RJTT",
        # "Chicago": "KORD",
        # "Los Angeles": "KLAX",
        # "Toronto": "CYYZ",
        # "Sydney": "YSSY",
        # "Dubai": "OMDB",
        # "Mumbai": "VABB",
        # "Sao Paulo": "SBGR"
    }

    model_config = SettingsConfigDict(
        env_file=".env", 
        env_file_encoding="utf-8", 
        extra="ignore"
    )

config = Settings()
