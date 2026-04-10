import asyncio
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.interval import IntervalTrigger
from apscheduler.triggers.cron import CronTrigger
from src.utils import logger, send_telegram_message
from src.market_discovery import MarketDiscoverer
from src.weather_data import weather_fetcher
from src.ai_analyzer import ai_analyzer
from src.trading_engine import trading_engine
from src.calibration import calibration_engine
from config.settings import config
from datetime import datetime

class BotScheduler:
    def __init__(self):
        self.scheduler = AsyncIOScheduler()
        self.discoverer = MarketDiscoverer()

    async def scan_and_trade(self):
        logger.info("=== Starting Full Scan & Trade Cycle ===")
        try:
            # 1. Fetch active weather markets
            markets = await self.discoverer.get_active_weather_markets()
            if not markets:
                logger.info("No relevant weather markets found.")
                return

            # Group unique ICAO codes needed
            icao_codes = list(set([m['icao_code'] for m in markets if m.get('icao_code')]))
            if not icao_codes:
                logger.warning("No valid ICAO codes resolved from active markets.")
                return

            # 2. Fetch bulk weather data
            logger.info(f"Fetching METAR/TAF for {len(icao_codes)} stations...")
            weather_data_map = await weather_fetcher.fetch_weather_for_icao(icao_codes)

            # --- Fetch Open-Meteo Multi-Source Data ---
            logger.info("Enriching data with Open-Meteo ECMWF/GFS Ensemble streams...")
            for icao in icao_codes:
                try:
                    om_data = await weather_fetcher.fetch_open_meteo(icao)
                    if icao in weather_data_map:
                        weather_data_map[icao].update(om_data)
                except Exception as e:
                    logger.warning(f"Failed to fetch open_meteo for {icao}: {e}")

            # 3. Analyze each market
            for market in markets:
                icao = market["icao_code"]
                city = market["city"]
                w_data = weather_data_map.get(icao)
                
                # Check cache fallback if empty
                if not w_data or (not w_data["metar"] and not w_data["taf"]):
                    w_data = weather_fetcher.get_weather_for_icao(icao)
                    
                if not w_data or (not w_data["metar"] and not w_data["taf"]):
                    logger.warning(f"No weather data available for {city} ({icao}). Skipping.")
                    continue
                    
                q_text = str(market.get('question', '')).replace('\n', ' ').replace('\r', '').strip()
                logger.debug(f"Checking {city} - {market['event_title']} | Q: {q_text}")
                
                # Pre-filter out purely guaranteed or dead outcomes
                valid_outcomes = []
                for out in market.get("outcomes", []):
                    # Gamma cross-routed price correctly accounts for No-Bid / Yes-Ask inversions
                    if 0.02 <= out["current_price"] <= 0.98:
                        valid_outcomes.append(out)
                
                if not valid_outcomes:
                    logger.debug(f"   -> Skipped (Prices out of bounds or dead market)")
                    continue # Skip AI evaluation completely, no valid trades available
                    
                market["outcomes"] = valid_outcomes

                logger.debug(f"   -> Valid constraints. Sending to AI...")
                analysis = await ai_analyzer.analyze_market(market, w_data)
                
                if analysis:
                    logger.success(
                        f"Trade signal! {city} -> BUY '{analysis.get('outcome_slug')}' "
                        f"(EV: {analysis.get('ev', 0.0):+.3f}, Edge: {analysis.get('edge', 0.0):+.1f}%, Conf: {analysis.get('confidence')}/100, Kelly frac: {analysis.get('kelly_frac', 0.0):.2f})"
                    )
                    
                    # Merge city info for the trading engine
                    analysis["city"] = city
                    await trading_engine.execute_trade(analysis)
                
                # Optimized speed for flash-lite limit (15 RPM = ~4s delay)
                await asyncio.sleep(4)
                    
            logger.info("=== Scan & Trade Cycle Completed ===")
            
        except Exception as e:
            logger.error(f"Error during scan cycle: {e}")
            await send_telegram_message(f"⚠️ Bot Exception in Scan Cycle:\n<pre>{e}</pre>")

    async def cleanup_daily(self):
        # We can implement cleanup of portfolio DB or exports here
        logger.info("Running daily cleanup...")
        await send_telegram_message("📅 Daily Bot Status: Running smoothly. PnL Sync pending.")

    async def start(self):
        # Initial boot actions (dry run logs etc. done by main.py usually)
        logger.info("Starting Full Scan & Trade Cycle")
        
        # Schedule the primary scanning sequence
        self.scheduler.add_job(
            self.scan_and_trade,
            "interval",
            hours=config.scan_interval_hours,
            id="market_scan",
            replace_existing=True
        )
        
        # Schedule the 30-minute active trade monitor
        logger.info("[SCHEDULER] monitor_open_trades started every 30 minutes")
        self.scheduler.add_job(
            self.monitor_open_trades_task,
            "interval",
            minutes=30,
            id="monitor_open_trades",
            replace_existing=True
        )
        
        # Schedule the resolution checker to update the calibration module
        self.scheduler.add_job(
            calibration_engine.check_resolutions,
            "interval",
            minutes=config.resolution_check_interval_minutes,
            id="calibration_checker",
            replace_existing=True
        )
        
        # Every 24 hours at midnight
        self.scheduler.add_job(
            self.cleanup_daily,
            'cron', hour=0, minute=0,
            id='daily_cleanup',
            name='Perform daily portfolio cleanups'
        )

        self.scheduler.start()
        logger.info("Scheduler started successfully.")
        
        # Asyncly run calibration checker once at boot
        asyncio.create_task(calibration_engine.check_resolutions())
        
        # Run standard scan independently of cron for first boot
        await self.scan_and_trade()
        
        # Give API a breather before checking positions immediately on boot
        await asyncio.sleep(5)
        await self.monitor_open_trades_task()

    async def monitor_open_trades_task(self):
        from src.portfolio_manager import portfolio_manager
        logger.info("[SCHEDULER] Running scheduled 30-minute open trades monitor...")
        await portfolio_manager.monitor_open_trades(clob_client=trading_engine.client)
