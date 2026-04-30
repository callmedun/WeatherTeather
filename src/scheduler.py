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
try:
    from src.model_skill_tracker import model_skill_tracker
    _SKILL_TRACKER_OK = True
except Exception as _e:
    _SKILL_TRACKER_OK = False
    model_skill_tracker = None
try:
    from src.batch_price_fetcher import batch_price_fetcher
    _BATCH_PRICE_OK = True
except Exception as _e:
    _BATCH_PRICE_OK = False
    batch_price_fetcher = None
try:
    from src.backtest.paper_trader import paper_trader
    _PAPER_TRADER_OK = True
except Exception as _e:
    _PAPER_TRADER_OK = False
    paper_trader = None
try:
    from src.health_monitor import (
        circuit_breaker, health_monitor, grib_cleaner,
        retry_with_backoff, CircuitOpenError
    )
    _HEALTH_OK = True
except Exception as _e:
    _HEALTH_OK = False
    circuit_breaker = None
    health_monitor = None
    grib_cleaner = None

class BotScheduler:
    def __init__(self):
        self.scheduler = AsyncIOScheduler()
        self.discoverer = MarketDiscoverer()
        self.city_lock = asyncio.Lock()
        self.last_scan_time = None
        self.last_monitor_time = None
        # [Phase 6] Register circuit breaker pause callback
        if _HEALTH_OK and circuit_breaker:
            circuit_breaker.register_pause_callback(self._circuit_pause_callback)

    def get_system_status(self) -> str:
        """[Phase 6] Full bot status using health_monitor."""
        if _HEALTH_OK and health_monitor:
            return health_monitor.get_full_status(
                scheduler=self,
                paper_trader_ref=paper_trader if _PAPER_TRADER_OK else None,
                skill_tracker_ref=model_skill_tracker if _SKILL_TRACKER_OK else None,
            )
        # Fallback (original)
        mode_str = "DRY RUN" if config.dry_run else "LIVE"
        pause_str = "⏸ ПАУЗА" if config.is_paused else "▶️ АКТИВЕН"
        def fmt_time(dt_obj):
            if dt_obj is None: return "Не запускался"
            return dt_obj.strftime("%Y-%m-%d %H:%M:%S UTC")
        return (
            f"⚙️ СТАТУС: {mode_str} | {pause_str}\n"
            f"Ошибки: {ai_analyzer.consecutive_failures}\n"
            f"Скан: {fmt_time(self.last_scan_time)}\n"
            f"Монитор: {fmt_time(self.last_monitor_time)}"
        )

    async def scan_and_trade(self):
        if config.is_paused:
            logger.info("⏸ Bot is currently PAUSED. Skipping market scan and new trades.")
            return

        logger.info("=== Starting Full Scan & Trade Cycle (Best EV Mode) ===")
        ai_analyzer.consecutive_failures = 0
        # [Phase 6] Record scan in health monitor
        if _HEALTH_OK and health_monitor:
            health_monitor.record_scan()
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

            # 2b. [Phase 4] Record forecasts for skill tracking
            if _SKILL_TRACKER_OK:
                for icao in icao_codes:
                    w = weather_data_map.get(icao, {})
                    fd = w.get("forecast_daily", {})
                    if fd:
                        try:
                            model_skill_tracker.record_forecasts_from_herbie(icao, fd)
                        except Exception as _sk_e:
                            logger.debug(f"[SkillTracker] record failed for {icao}: {_sk_e}")

            # 3. Process cities using Batch AI Analysis
            # Math model has no API clients — use a generous semaphore for parallelism
            semaphore = asyncio.Semaphore(max(1, len(ai_analyzer.clients)) if ai_analyzer.clients else 8)
            
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
                    try:
                        signals = await asyncio.wait_for(
                            ai_analyzer.analyze_city_batch(city, city_markets, w_data),
                            timeout=600  # 10 min per-city hard cap
                        )
                        return signals
                    except asyncio.TimeoutError:
                        logger.warning(f"[{city}] AI batch timed out after 10 minutes. Skipping city.")
                        return []
                    except Exception as e:
                        logger.error(f"[{city}] process_city error: {e}")
                        return []

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
                open_market_ids = {t.market_id for t in open_trades if t.market_id}
                max_city_positions = int(getattr(config, "tsas_max_city_positions", 2)) if getattr(config, "analysis_model", "tsas").lower() == "tsas" else 2
                
                trades_to_execute = []
                for sig in signals_list:
                    if len(open_trades) + len(trades_to_execute) >= max_city_positions:
                        break
                    if sig["token_id"] in open_tokens:
                        # Update prediction memory but skip buying
                        calibration_engine.save_prediction(sig)
                        continue
                    if sig["market_id"] in open_market_ids:
                        logger.info(
                            f"[TSAS] Open-position skip for {city}: market {sig['market_id'][:8]} "
                            f"already has an active position."
                        )
                        calibration_engine.save_prediction(sig)
                        continue
                    if (
                        getattr(config, "analysis_model", "tsas").lower() == "tsas"
                        and portfolio_manager.was_market_closed_recently(
                            sig["market_id"],
                            int(getattr(config, "tsas_reentry_cooldown_minutes", 0))
                        )
                    ):
                        logger.info(
                            f"[TSAS] Cooldown skip for {city}: market {sig['market_id'][:8]} "
                            f"was closed recently."
                        )
                        calibration_engine.save_prediction(sig)
                        continue
                    ladder_group = sig.get("ladder_group")
                    allow_same_sentiment = bool(ladder_group)
                    if sig["sentiment"] in open_sentiments and sig["sentiment"] != "NEUTRAL" and not allow_same_sentiment:
                        continue
                    
                    # Conflict check within the city batch
                    if any(existing["market_id"] == sig["market_id"] for existing in trades_to_execute):
                        continue

                    trades_to_execute.append(sig)
                    open_sentiments.append(sig["sentiment"])
                    open_market_ids.add(sig["market_id"])

                for sig in trades_to_execute:
                    ladder_suffix = ""
                    if sig.get("ladder_group"):
                        ladder_suffix = (
                            f" | Ladder {sig.get('ladder_rank')}/{sig.get('ladder_size')}"
                            f" pkgEV={sig.get('ladder_package_ev', 0.0):+.3f}"
                        )
                    logger.success(
                        f"Ranked Signal! {city} -> BUY {sig['sentiment']} '{sig['outcome_slug']}' "
                        f"(EV: {sig['ev']:+.3f}, Edge: {sig['edge']:+.1f}%){ladder_suffix}"
                    )
                    # [Phase 5] Record in paper trader before live execution
                    if _PAPER_TRADER_OK and paper_trader is not None:
                        try:
                            paper_trader.record_signal(sig)
                        except Exception as _pt_e:
                            logger.debug(f"[PaperTrader] record_signal error: {_pt_e}")
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
        
        # Schedule the primary scanning sequence (max_instances=1 = self-protection only)
        self.scheduler.add_job(
            self.scan_and_trade,
            "interval",
            hours=config.scan_interval_hours,
            id="market_scan",
            replace_existing=True,
            max_instances=1
        )
        
        # Schedule the 2-minute active trade monitor
        # max_instances=2 allows it to run alongside the hourly scan
        logger.info("[SCHEDULER] monitor_open_trades started every 2 minutes")
        self.scheduler.add_job(
            self.monitor_open_trades_task,
            "interval",
            minutes=2,
            id="monitor_open_trades",
            replace_existing=True,
            max_instances=2
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

        # [Phase 4] Daily skill score update at 03:00 UTC
        if _SKILL_TRACKER_OK:
            self.scheduler.add_job(
                self._update_skill_scores_task,
                'cron', hour=3, minute=0,
                id='skill_score_update',
                name='Update model skill scores from resolved trades'
            )
            logger.info("[SCHEDULER] model_skill_tracker daily job scheduled at 03:00 UTC")

        # [Phase 6] GRIB cleanup at 04:00 UTC
        if _HEALTH_OK:
            self.scheduler.add_job(
                self._grib_cleanup_task,
                'cron', hour=4, minute=0,
                id='grib_cleanup',
                name='Clean stale Herbie GRIB files'
            )
            logger.info("[SCHEDULER] GRIB cleanup job scheduled at 04:00 UTC")

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
        logger.info("[SCHEDULER] Running scheduled 2-minute open trades monitor...")
        try:
            await asyncio.wait_for(
                portfolio_manager.monitor_open_trades(clob_client=trading_engine.client),
                timeout=110
            )
            self.last_monitor_time = datetime.utcnow()
        except asyncio.TimeoutError:
            logger.warning("[SCHEDULER] monitor_open_trades TIMED OUT. Releasing for next cycle.")
            self.last_monitor_time = datetime.utcnow()
        except Exception as e:
            logger.error(f"[SCHEDULER] monitor_open_trades error: {e}")
            if _HEALTH_OK and health_monitor:
                health_monitor.record_error("monitor", str(e))

    async def _update_skill_scores_task(self):
        """[Phase 4] Daily skill score update — runs at 03:00 UTC."""
        logger.info("[SCHEDULER][Phase4] Running daily model skill score update...")
        try:
            if not _SKILL_TRACKER_OK or model_skill_tracker is None:
                return
            from datetime import date, timedelta
            yesterday = date.today() - timedelta(days=1)
            n = model_skill_tracker.compute_errors_for_date(yesterday)
            skills = model_skill_tracker.update_skill_scores()
            logger.info(f"[Phase4] Skill update: {n} errors, {len(skills)} skill records")
            if skills:
                await send_telegram_message(
                    f"📊 <b>[Phase 4] Model Skill Update</b>\n"
                    f"Date: {yesterday}\n"
                    f"Errors computed: {n}\n"
                    f"Skill records updated: {len(skills)}"
                )
        except Exception as e:
            logger.error(f"[SCHEDULER][Phase4] skill update error: {e}")

    async def _grib_cleanup_task(self):
        """[Phase 6] Daily GRIB cleanup — runs at 04:00 UTC."""
        logger.info("[SCHEDULER][Phase6] Running GRIB file cleanup...")
        try:
            if not _HEALTH_OK or grib_cleaner is None:
                return
            stats = grib_cleaner.cleanup()
            if stats["files_removed"] > 0:
                await send_telegram_message(
                    f"🗑️ <b>[Phase 6] GRIB Cleanup</b>\n"
                    f"Files removed: {stats['files_removed']}\n"
                    f"Freed: {stats['mb_freed']:.1f}MB"
                )
            logger.info(f"[Phase6] GRIB cleanup: {stats}")
        except Exception as e:
            logger.error(f"[SCHEDULER][Phase6] GRIB cleanup error: {e}")

    async def _circuit_pause_callback(self, circuit_name: str, error: str) -> None:
        """[Phase 6] Called when a circuit trips — notifies Telegram."""
        msg = (
            f"🔴 <b>[Circuit Breaker TRIPPED]</b>\n"
            f"Circuit: <code>{circuit_name}</code>\n"
            f"Error: {error[:200]}\n"
            f"Trading paused for 120s, then auto-recovery."
        )
        logger.error(f"[CB] Circuit '{circuit_name}' tripped: {error}")
        try:
            await send_telegram_message(msg)
        except Exception:
            pass

    def reset_circuit(self, name: Optional[str] = None) -> str:
        """Manually reset a circuit breaker (callable from Telegram)."""
        if not _HEALTH_OK or circuit_breaker is None:
            return "Health monitor not available."
        circuit_breaker.reset(name)
        label = name or "all circuits"
        return f"✅ Circuit breaker reset: {label}"

bot_scheduler = BotScheduler()
