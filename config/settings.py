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
    gemini_model: str = "gemma-4-31b-it"
    
    # Notifications
    telegram_bot_token: str = ""
    telegram_chat_id: str = ""
    telegram_menu_enabled: bool = True # Required by telegram_bot.py
    
    # Trading Params
    dry_run: bool = True
    is_paused: bool = False
    max_trade_size_usd: float = 20.0
    default_trade_size: float = 15.0 # Required by PortfolioManager
    min_trade_usd: float = 1.0
    min_shares: float = 0.01
    max_decimals_amount: int = 2
    scan_interval_hours: int = 1
    scan_days_ahead: int = 2
    resolution_check_interval_minutes: int = 15
    
    # Risk & Exposure
    max_total_exposure: float = 0.2 # Required by PortfolioManager (20% of bankroll)
    max_city_exposure: float = 0.1 # Required by PortfolioManager (10% of bankroll)
    max_single_trade_exposure: float = 0.05 # Hard cap per transaction (5% of bankroll)
    
    # Calibration
    calibration_min_trades: int = 3
    
    # Analysis engine
    # "tsas" enables the new Thermo-Stochastic analysis core.
    # "legacy" keeps the previous deterministic blended model available as a rollback.
    analysis_model: str = "tsas"
    tsas_min_confidence: float = 0.35
    tsas_max_taf_inflation: float = 3.5
    tsas_metar_alpha: float = 0.08
    tsas_circuit_breaker_hours: float = 3.0
    tsas_entry_min_hours_to_close: float = 12.0
    tsas_entry_max_hours_to_close: float = 72.0
    tsas_min_daily_volume_usd: float = 200.0
    tsas_target_daily_volume_usd: float = 2500.0
    tsas_max_spread: float = 0.20
    tsas_ladder_enabled: bool = True
    tsas_ladder_max_price: float = 0.35
    tsas_ladder_max_positions: int = 3
    tsas_max_city_positions: int = 3
    tsas_ladder_package_cap: float = 0.06
    tsas_ladder_min_package_ev: float = 0.10
    tsas_ladder_min_price: float = 0.02
    tsas_min_executable_price: float = 0.02
    tsas_max_executable_price: float = 0.98
    tsas_min_ev_for_min_trade: float = 0.18
    tsas_reentry_cooldown_minutes: int = 180
    tsas_hold_tail_max_entry_price: float = 0.10
    tsas_hold_tail_min_prob: float = 0.12
    tsas_exit_prob_drop_points: float = 10.0
    tsas_exit_edge_floor: float = 2.0
    tsas_exit_confidence_floor: float = 0.25
    tsas_exit_confidence_edge_floor: float = 4.0
    tsas_exit_probability_collapse_points: float = 35.0
    tsas_exit_probability_floor: float = 0.35
    tsas_exit_model_flip_floor: float = 0.30
    tsas_exit_market_divergence_loss_pct: float = 35.0
    tsas_exit_market_divergence_prob_stability_points: float = 5.0
    tsas_exit_market_divergence_min_entry_price: float = 0.25
    tsas_entry_market_conflict_divergence: float = 0.35
    tsas_open_meteo_cache_ttl_seconds: int = 1800
    tsas_intraday_window_hours: float = 10.0
    tsas_monitor_adjustment_multiplier: float = 1.20
    
    # Kelly & EV Thresholds
    kelly_fraction: float = 0.1
    # EV thresholds per city (default + specific overrides)
    ev_threshold: Dict[str, float] = {
        "default": 0.08,
        "London": 0.12
    }
    
    # City-ICAO Mapping for Weather
    city_icao_mapping: Dict[str, str] = {
        "Austin": "KAUS",
        "Miami": "KMIA",
        "Wellington": "NZWN",
        "Chicago": "KORD",
        "Dallas": "KDAL",
        "Atlanta": "KATL",
        "San Francisco": "KSFO",
        "London": "EGLC",
        "Shanghai": "ZSPD",
        "Milan": "LIMC",
        "Munich": "EDDM",
        "Beijing": "ZBAA",
        "Taipei": "RCSS",
        "Los Angeles": "KLAX",
        "Singapore": "WSSS",
    }

    class Config:
        env_file = ".env"
        extra = "ignore"

config = Settings()
