import asyncio
from typing import Optional
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.interval import IntervalTrigger
from apscheduler.triggers.cron import CronTrigger
from src.utils import logger, send_telegram_message
from src.market_discovery import MarketDiscoverer
from src.weather_data import weather_fetcher
from src.ai_analyzer import ai_analyzer
from src.trading_engine import trading_engine
from src.portfolio_manager import portfolio_manager
from src.calibration import calibration_engine
from config.settings import config
from datetime import datetime

class BotScheduler:
    def __init__(self):
        self.scheduler = AsyncIOScheduler()
        self.discoverer = MarketDiscoverer()
        self.city_lock = asyncio.Lock() # For thread-safe traded_cities access
        self.last_scan_time = None
        self.last_monitor_time = None
        self._master_cycle_lock = asyncio.Lock()
        
    def get_system_status(self) -> str:
        """Generates a text report about bot health and scheduling."""
        mode_str = "DRY RUN (виртуальный баланс)" if config.dry_run else "LIVE (реальный баланс!)"
        pause_str = "⏸ ПАУЗА (новые рынки не ищутся)" if config.is_paused else "▶️ АКТИВЕН"
        
        def fmt_time(dt_obj):
            if dt_obj is None: return "Еще не запускался"
            return dt_obj.strftime("%Y-%m-%d %H:%M:%S UTC")
            
        return (
            f"⚙️ СТАТУС БОТА:\n\n"
            f"Режим: {mode_str}\n"
            f"Состояние: {pause_str}\n\n"
            f"Ошибки ИИ (подряд): {ai_analyzer.consecutive_failures}\n\n"
            f"⏱ Последний часовой скан:\n{fmt_time(self.last_scan_time)}\n\n"
            f"⏱ Последний 10-мин мониторинг:\n{fmt_time(self.last_monitor_time)}"
        )

    async def scan_and_trade(self):
        if config.is_paused:
            logger.info("⏸ Bot is currently PAUSED. Skipping market scan and new trades.")
            return

        if self._master_cycle_lock.locked():
            logger.info("[SCHEDULER] scan_and_trade is waiting for other processes to finish...")
            
        async with self._master_cycle_lock:
            logger.info("=== Starting Full Scan & Trade Cycle (Best EV Mode) ===")
            ai_analyzer.consecutive_failures = 0
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
                logger.info(f"Fetching weather for {len(icao_codes)} stations (with retries)...")
                weather_data_map = await weather_fetcher.fetch_weather_for_icao(icao_codes)
                
                if weather_data_map is None:
                    logger.error("!!! [SAFETY ABORT] Failed to fetch critical weather data after 3 retries. Skipping THIS whole scan cycle to prevent 'blind' trading.")
                    await send_telegram_message("⚠️ <b>[SAFETY ABORT]</b> Scan cycle skipped: Weather API (METAR/TAF) is unreachable after retries.")
                    return

                for icao in icao_codes:
                    try:
                        om_data = await weather_fetcher.fetch_open_meteo(icao)
                        if icao in weather_data_map:
                            weather_data_map[icao].update(om_data)
                    except Exception as e:
                        logger.warning(f"Failed to fetch open_meteo for {icao}: {e}")

                # 3. Process cities using Batch AI Analysis
                semaphore = asyncio.Semaphore(len(ai_analyzer.clients))
                
                async def process_city(city: str, icao: str) -> list[dict]:
                    async with semaphore:
                        city_markets = [m for m in markets if m["city"] == city]
                        if not city_markets:
                            return []
                        
                        w_data = weather_data_map.get(icao)
                        if not w_data: 
                            return []
                            
                        # Pre-filter outcomes
                        for m in city_markets:
                            m["outcomes"] = [o for o in m.get("outcomes", []) if 0.02 <= o["current_price"] <= 0.98]
                        city_markets = [m for m in city_markets if m["outcomes"]]
                        
                        if not city_markets:
                            return []
                            
                        logger.info(f"[{city}] Sending batch request for {len(city_markets)} markets...")
                        signals = await ai_analyzer.analyze_city_batch(city, city_markets, w_data)
                        return signals

                # Process all cities in parallel (semaphore limited)
                city_tasks = [process_city(city, icao) for city, icao in config.city_icao_mapping.items()]
                all_city_signals = await asyncio.gather(*city_tasks)
                
                # 4. Execute trades based on signals
                for i, city in enumerate(config.city_icao_mapping.keys()):
                    signals_list = all_city_signals[i]
                    if not signals_list:
                        continue

                    # Sort by EV descending
                    signals_list.sort(key=lambda x: x["ev"], reverse=True)
                    
                    # Fetch currently open trades for this city
                    open_trades = portfolio_manager.get_open_trades_for_city(city)
                    open_sentiments = [t.sentiment for t in open_trades if t.sentiment]
                    open_tokens = [t.token_id for t in open_trades]
                    
                    trades_to_execute = []
                    for sig in signals_list:
                        if len(open_trades) + len(trades_to_execute) >= 2:
                            break
                        if sig["token_id"] in open_tokens:
                            # Update prediction memory but skip buying
                            calibration_engine.save_prediction(sig)
                            continue
                        if sig["sentiment"] in open_sentiments and sig["sentiment"] != "NEUTRAL":
                            continue
                        
                        # Conflict check within the city batch
                        if any(existing["market_id"] == sig["market_id"] for existing in trades_to_execute):
                            continue

                        trades_to_execute.append(sig)
                        open_sentiments.append(sig["sentiment"])

                    for sig in trades_to_execute:
                        logger.success(
                            f"Ranked Signal! {city} -> BUY {sig['sentiment']} '{sig['outcome_slug']}' "
                            f"(EV: {sig['ev']:+.3f}, Edge: {sig['edge']:+.1f}%)"
                        )
                        await trading_engine.execute_trade(sig)

                logger.info("=== Scan & Trade Cycle Completed ===")
            
            except Exception as e:
                logger.error(f"Error during scan cycle: {e}")
                import traceback
                traceback.print_exc()
            finally:
                self.last_scan_time = datetime.utcnow()

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
        
        # Schedule the 10-minute active trade monitor
        logger.info("[SCHEDULER] monitor_open_trades started every 10 minutes")
        self.scheduler.add_job(
            self.monitor_open_trades_task,
            "interval",
            minutes=10,
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
        if self._master_cycle_lock.locked():
            logger.info("[SCHEDULER] monitor_open_trades skipped because scan_and_trade is running.")
            return
            
        async with self._master_cycle_lock:
            from src.portfolio_manager import portfolio_manager
            logger.info("[SCHEDULER] Running scheduled 10-minute open trades monitor...")
            await portfolio_manager.monitor_open_trades(clob_client=trading_engine.client)
            self.last_monitor_time = datetime.utcnow()

bot_scheduler = BotScheduler()
