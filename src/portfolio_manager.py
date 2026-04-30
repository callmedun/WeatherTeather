from sqlalchemy import create_engine, Column, Integer, String, Float, Boolean, DateTime
from sqlalchemy.orm import declarative_base, sessionmaker
from datetime import datetime, timezone
import json
import httpx
import traceback
from src.utils import logger, send_telegram_message
from config.settings import config
from py_clob_client.clob_types import OrderArgs
import os
import re
import asyncio
from src.probability_calculator import parse_temperature_bin

Base = declarative_base()

class TradePosition(Base):
    __tablename__ = 'trade_positions'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    market_id = Column(String, index=True)
    token_id = Column(String)
    city = Column(String, index=True)
    outcome_name = Column(String)
    entry_price = Column(Float)
    size_usd = Column(Float)
    status = Column(String) # OPEN, RESOLVED, SOLD
    sentiment = Column(String, nullable=True) # BULLISH, BEARISH, NEUTRAL
    created_at = Column(DateTime, default=datetime.utcnow)
    resolved_at = Column(DateTime, nullable=True)

class RiskSetting(Base):
    __tablename__ = 'risk_settings'
    key = Column(String, primary_key=True)
    value = Column(Float)

class PortfolioManager:
    def __init__(self):
        db_dir = "data"
        os.makedirs(db_dir, exist_ok=True)
        self.engine = create_engine(f"sqlite:///{db_dir}/portfolio.db")
        Base.metadata.create_all(self.engine)
        self.Session = sessionmaker(bind=self.engine)
        
        # Initialize default settings if not exists
        self.init_risk_settings()
        
        self.assumed_bankroll = 10000.0 
        
        # Cache for live prices (token_id -> {'price': float, 'time': timestamp})
        self._price_cache = {}
        self.cache_ttl = 30 # seconds

    def _format_temp_value(self, value: float) -> str:
        if abs(value - round(value)) < 0.05:
            return str(int(round(value)))
        return f"{value:.1f}"

    def _extract_market_date(self, question: str) -> str:
        match = re.search(r'\bon\s+([A-Za-z]+\s+\d+)', question or "")
        return match.group(1) if match else "N/A"

    def _format_market_bin(self, question: str) -> str:
        parsed = parse_temperature_bin(question or "")
        if parsed is None:
            return "N/A"

        low, high, unit = parsed
        if low <= -999:
            return f"<= {self._format_temp_value(high - 0.5)}{unit}"
        if high >= 999:
            return f">= {self._format_temp_value(low + 0.5)}{unit}"
        if abs((high - low) - 1.0) < 0.05:
            return f"{self._format_temp_value(low + 0.5)}{unit}"
        return f"{self._format_temp_value(low + 0.5)}-{self._format_temp_value(high - 0.5)}{unit}"

    def get_current_prices(self, clob_client, token_ids: list[str]) -> dict[str, float]:
        """Пакетное получение цен (POST /prices). С защитой от ошибок SDK."""
        if not clob_client or not token_ids:
            return {}
            
        try:
            # Try SDK first (but with a format that bypasses attribute errors if possible)
            res = clob_client.get_prices(token_ids)
            return self._parse_prices_response(res)
        except Exception as e:
            if "'str' object has no attribute 'token_id'" in str(e):
                # Fallback to direct HTTP if SDK is bugged
                try:
                    host = "https://clob.polymarket.com" if config.chain_id == 137 else "https://clob.amoy.polymarket.com"
                    url = f"{host}/prices"
                    with httpx.Client() as client:
                        r = client.post(url, json={"token_ids": token_ids}, timeout=10.0)
                        if r.status_code == 200:
                            return self._parse_prices_response(r.json())
                except Exception as http_e:
                    logger.warning(f"[Portfolio] Direct HTTP Price fetch failed: {http_e}")
            else:
                logger.warning(f"[Portfolio] SDK get_prices failed: {e}")
            return {}

    def _parse_prices_response(self, res) -> dict[str, float]:
        """Helper to parse various response formats from Polymarket prices endpoint."""
        now = datetime.now().timestamp()
        results = {}
        
        if isinstance(res, dict):
            # Format: {"token_id": "price", ...}
            for tid, p in res.items():
                try:
                    val = float(p)
                    self._price_cache[tid] = {'price': val, 'time': now}
                    results[tid] = val
                except: continue
        elif isinstance(res, list):
            # Format: [{"token_id": "...", "price": "..."}, ...] or ["price1", "price2"]?
            for item in res:
                if isinstance(item, dict):
                    tid = item.get("token_id")
                    p = item.get("price")
                    if tid and p:
                        try:
                            val = float(p)
                            self._price_cache[tid] = {'price': val, 'time': now}
                            results[tid] = val
                        except: continue
        return results

    def get_current_price(self, clob_client, token_id: str) -> float | None:
        """Получает текущую цену токена, используя кэш или одиночный запрос."""
        if not token_id:
            return None
            
        now = datetime.now().timestamp()
        
        # 1. Check Cache
        if token_id in self._price_cache:
            if now - self._price_cache[token_id]['time'] < self.cache_ttl:
                return self._price_cache[token_id]['price']
                
        if clob_client is None:
            return None
            
        try:
            # 2. Try Batch (even for one) as it's more stable
            res_dict = self.get_current_prices(clob_client, [token_id])
            if token_id in res_dict:
                return res_dict[token_id]
                
            # 3. Last resort fallback
            result = clob_client.get_price(token_id, side="BUY")
            if isinstance(result, (int, float, str)):
                val = float(result)
                self._price_cache[token_id] = {'price': val, 'time': now}
                return val
            elif isinstance(result, dict) and "price" in result:
                val = float(result["price"])
                self._price_cache[token_id] = {'price': val, 'time': now}
                return val
            return None
        except Exception as e:
            logger.debug(f"[WARN] Failed to get price for {token_id}: {e}")
            return None

    def record_trade(self, market_id: str, token_id: str, city: str, outcome: str, price: float, size: float, sentiment: str = "NEUTRAL"):
        session = self.Session()
        try:
            trade = TradePosition(
                market_id=market_id,
                token_id=token_id,
                city=city,
                outcome_name=outcome,
                entry_price=price,
                size_usd=size,
                status="OPEN",
                sentiment=sentiment
            )
            session.add(trade)
            session.commit()
            logger.info(f"Recorded trade for {city} in DB. Size: ${size:.2f}")
        except Exception as e:
            session.rollback()
            logger.error(f"Failed to record trade: {e}")
        finally:
            session.close()

    def get_open_exposure_for_city(self, city: str) -> float:
        session = self.Session()
        try:
            open_trades = session.query(TradePosition).filter_by(city=city, status="OPEN").all()
            return sum(t.size_usd for t in open_trades)
        finally:
            session.close()

    def get_open_trades_for_city(self, city: str) -> list:
        session = self.Session()
        try:
            return session.query(TradePosition).filter_by(city=city, status="OPEN").all()
        finally:
            session.close()
            
    def get_total_open_exposure(self) -> float:
        session = self.Session()
        try:
            open_trades = session.query(TradePosition).filter_by(status="OPEN").all()
            return sum(t.size_usd for t in open_trades)
        finally:
            session.close()

    def get_active_market_ids(self) -> list[str]:
        """Returns a list of all market_ids where we currently have an OPEN trade, to prevent duplicate scanning."""
        session = self.Session()
        try:
            open_trades = session.query(TradePosition).filter_by(status="OPEN").all()
            return list(set(t.market_id for t in open_trades if t.market_id))
        finally:
            session.close()

    def was_market_closed_recently(self, market_id: str, cooldown_minutes: int) -> bool:
        if not market_id or cooldown_minutes <= 0:
            return False

        session = self.Session()
        try:
            recent_trade = (
                session.query(TradePosition)
                .filter(
                    TradePosition.market_id == market_id,
                    TradePosition.status.in_(["SOLD", "RESOLVED"]),
                    TradePosition.resolved_at.isnot(None),
                )
                .order_by(TradePosition.resolved_at.desc())
                .first()
            )
            if not recent_trade or recent_trade.resolved_at is None:
                return False

            elapsed_seconds = (datetime.utcnow() - recent_trade.resolved_at).total_seconds()
            return elapsed_seconds < cooldown_minutes * 60
        finally:
            session.close()

    def can_trade_city(self, city: str, intended_size: float = None) -> bool:
        if intended_size is None:
            intended_size = config.default_trade_size
            
        current_city_exposure = self.get_open_exposure_for_city(city)
        total_exposure = self.get_total_open_exposure()
        
        # Max bankroll % used per day total 
        max_total = self.assumed_bankroll * config.max_total_exposure
        # Max city %
        max_city = self.assumed_bankroll * config.max_city_exposure
        max_single = self.assumed_bankroll * config.max_single_trade_exposure

        if intended_size > max_single:
            logger.warning(f"Trade rejected: Exceeds single-trade cap ({intended_size} > {max_single})")
            return False
        
        if total_exposure + intended_size > max_total:
            logger.warning(f"Trade rejected: Exceeds max total exposure limit ({total_exposure} + {intended_size} > {max_total})")
            return False
            
        if current_city_exposure + intended_size > max_city:
            logger.warning(f"Trade rejected: Exceeds max city exposure for {city} ({current_city_exposure} + {intended_size} > {max_city})")
            return False
            
        return True

    def init_risk_settings(self):
        session = self.Session()
        try:
            defaults = {
                "tp_edge": 0.0,         # Take Profit Edge (0%)
                "strong_tp_pnl": 60.0,   # Strong PnL TP (60%)
                "sl_edge": -12.0,        # Stop Loss Edge (-12%)
                "sl_pnl": -70.0,         # Stop Loss PnL (-70%)
                "time_exit_h": 6.0       # Time-based exit (6 hours)
            }
            for k, v in defaults.items():
                existing = session.query(RiskSetting).filter_by(key=k).first()
                if not existing:
                    session.add(RiskSetting(key=k, value=v))
                else:
                    # Auto-migration of old defaults to new standards
                    if k == "tp_edge" and existing.value in [5.0, 10.0]:
                        existing.value = v
                    elif k == "strong_tp_pnl" and existing.value == 30.0:
                        existing.value = v
                    elif k == "sl_pnl" and existing.value in [-15.0, -30.0]:
                        existing.value = v
            session.commit()
        except Exception as e:
            logger.error(f"Failed to init risk settings: {e}")
            session.rollback()
        finally:
            session.close()

    def get_risk_setting(self, key: str, default: float) -> float:
        session = self.Session()
        try:
            res = session.query(RiskSetting).filter_by(key=key).first()
            return res.value if res else default
        finally:
            session.close()

    def set_risk_setting(self, key: str, value: float):
        session = self.Session()
        try:
            res = session.query(RiskSetting).filter_by(key=key).first()
            if res:
                res.value = value
            else:
                session.add(RiskSetting(key=key, value=value))
            session.commit()
        except Exception as e:
            logger.error(f"Failed to update risk setting {key}: {e}")
            session.rollback()
        finally:
            session.close()

        return True

    def _get_effective_exit_price(self, clob_client, token_id: str, target_shares: float) -> tuple[float, float, float]:
        """Calculates Weighted Average Price (WAP) of bids for a given amount of shares."""
        if not clob_client or target_shares <= 0:
            return 0.0, 0.0, 0.0
            
        try:
            ob = clob_client.get_order_book(token_id)
            bids = getattr(ob, "bids", [])
            if not bids:
                return 0.0, 0.0, 0.0
            
            # Sort BIDS descending (highest to lowest) for selling
            sorted_bids = sorted(bids, key=lambda x: float(getattr(x, 'price', 0.0)), reverse=True)
            
            total_shares_filled = 0.0
            total_receive_usd = 0.0
            
            for bid in sorted_bids:
                p = float(getattr(bid, 'price', 0.0))
                s = float(getattr(bid, 'size', 0.0))
                
                remaining_shares = target_shares - total_shares_filled
                if remaining_shares <= 0:
                    break
                    
                if s <= remaining_shares:
                    total_shares_filled += s
                    total_receive_usd += s * p
                else:
                    total_receive_usd += remaining_shares * p
                    total_shares_filled += remaining_shares
                    break
            
            if total_shares_filled == 0:
                return 0.0, 0.0, 0.0
                
            avg_price = total_receive_usd / total_shares_filled
            return avg_price, total_shares_filled, total_receive_usd
        except Exception as e:
            logger.warning(f"[Portfolio] Exit depth check failed for {token_id}: {e}")
            return 0.0, 0.0, 0.0

    async def monitor_open_trades(self, clob_client) -> None:
        """Проверяет ТОЛЬКО открытые позиции каждые 10 минут и решает закрывать или нет.
        ТЕПЕРЬ: Сначала перепроверяет погоду через ИИ для всех открытых позиций!"""
        if not clob_client:
            logger.warning("[MONITOR 10min] clob_client is None. Skipping.")
            return

        from src.weather_data import weather_fetcher
        from src.ai_analyzer import ai_analyzer
        from src.calibration import calibration_engine

        session = self.Session()
        try:
            open_trades = session.query(TradePosition).filter_by(status="OPEN").all()
            if not open_trades:
                logger.info("[MONITOR 10min] No open trades to monitor.")
                return
            
            logger.info(f"[MONITOR 10min] Started full re-analysis for {len(open_trades)} open trades...")
            
            # 1. Load current AI memory (to get context: questions, target city etc)
            ai_memory = {}
            mem_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "historical_predictions.jsonl")
            if os.path.exists(mem_path):
                with open(mem_path, 'r', encoding='utf-8') as f:
                    for line in f:
                        if not line.strip(): continue
                        try:
                            r = json.loads(line)
                            key = f"{r.get('market_id')}_{r.get('token_id')}"
                            ai_memory[key] = r
                        except: pass

            # 2. GROUP TRADES BY CITY for efficient bulk weather & AI calls
            city_groups = {}
            for t in open_trades:
                if t.city not in city_groups: city_groups[t.city] = []
                city_groups[t.city].append(t)
            
            # 3. BULK AI RE-ANALYSIS
            icao_codes = list(set([config.city_icao_mapping.get(city) for city in city_groups.keys() if config.city_icao_mapping.get(city)]))
            weather_data_map = {}
            if icao_codes:
                logger.info(f"[MONITOR] Fetching weather for {len(icao_codes)} stations (with retries)...")
                weather_data_map = await weather_fetcher.fetch_weather_for_icao(icao_codes)
                
                if weather_data_map is None:
                    logger.warning("[MONITOR] [SAFETY ABORT] Missing METAR/TAF after retries. Skipping AI re-analysis for this 10m cycle.")
                    # We don't return here because we might still want to check basic Stop-Loss-Price
                    weather_data_map = {} # Keep it empty but set flag or handle below
                else:
                    # Supplement with OpenMeteo
                    for icao in icao_codes:
                        try:
                            om_data = await weather_fetcher.fetch_open_meteo(icao, force_refresh=True)
                            if icao in weather_data_map:
                                weather_data_map[icao].update(om_data)
                        except: pass

            # Re-analyze all cities in parallel (semaphore limits concurrent AI calls)
            semaphore = asyncio.Semaphore(len(ai_analyzer.clients) if hasattr(ai_analyzer, 'clients') and ai_analyzer.clients else 1)

            async def process_monitor_city(city: str, trades: list, all_city_markets: list):
                async with semaphore:
                    icao = config.city_icao_mapping.get(city)
                    w_data = weather_data_map.get(icao)
                    if not w_data:
                        logger.warning(f"[MONITOR] No weather for {city}, skipping re-analysis.")
                        return
                    
                    # Use ALL active markets for this city — same context as the hourly scan.
                    # This prevents AI from giving different answers due to narrower context.
                    city_markets_for_ai = [
                        m for m in all_city_markets
                        if m.get("city") == city
                        and m.get("outcomes")
                        and any(0.02 <= o["current_price"] <= 0.98 for o in m["outcomes"])
                    ]
                    
                    # Filter outcomes to tradeable range
                    for m in city_markets_for_ai:
                        m["outcomes"] = [o for o in m["outcomes"] if 0.02 <= o["current_price"] <= 0.98]
                    city_markets_for_ai = [m for m in city_markets_for_ai if m["outcomes"]]

                    if not city_markets_for_ai:
                        logger.warning(f"[MONITOR] No active markets found for {city}, skipping re-analysis.")
                        return
                    
                    logger.info(f"[MONITOR] [{city}] Re-analyzing {len(city_markets_for_ai)} markets with fresh weather...")
                    fresh_signals = await ai_analyzer.analyze_city_batch(city, city_markets_for_ai, w_data, return_all=True)
                    
                    # Update memory — but only log/use results for OUR open trades
                    for t in trades:
                        mem_key = f"{t.market_id}_{t.token_id}"
                        old_mem = ai_memory.get(mem_key, {})
                        old_prob = old_mem.get('predicted_prob')
                        
                        matching_sig = next((s for s in fresh_signals if s.get("token_id") == t.token_id), None)
                        
                        if matching_sig:
                            new_prob = matching_sig.get('predicted_prob')
                            if new_prob is not None:
                                q_text = matching_sig.get("question", "")
                                date_str = self._extract_market_date(q_text)
                                
                                if old_prob is not None:
                                    shift = (new_prob - old_prob) * 100
                                    logger.info(f"[MONITOR] 🔄 {city} ({date_str}) \"{t.outcome_name}\" | Вероятность: {old_prob*100:.1f}% ➔ {new_prob*100:.1f}% | Изменение: {shift:+.1f}%")
                                    # Log reasoning for significant shifts (with math model this shows formula values)
                                    if abs(shift) >= 15:
                                        reasoning = matching_sig.get("reasoning", "")
                                        if reasoning:
                                            logger.warning(
                                                f"[MONITOR] 🧮 ПРИЧИНА ИЗМЕНЕНИЯ для {city} ({date_str}) \"{t.outcome_name}\":\n"
                                                f"           {reasoning}"
                                            )
                                        else:
                                            logger.warning(f"[MONITOR] 🧮 Изменение {city} ({date_str}) \"{t.outcome_name}\": данные расчёта недоступны.")
                                else:
                                    logger.info(f"[MONITOR] 🔄 {city} ({date_str}) \"{t.outcome_name}\" | Вероятность: [первый расчёт] ➔ {new_prob*100:.1f}%")
                            
                            calibration_engine.update_prediction_prob(t.token_id, new_prob)
                            ai_memory[mem_key] = matching_sig
                        else:
                            logger.debug(f"[MONITOR] No fresh signal found for {t.city} token {t.token_id[:8]}...")

            # 3b. Fetch ALL active markets for relevant cities (same as hourly scan)
            from src.market_discovery import MarketDiscoverer
            discoverer = MarketDiscoverer()
            try:
                all_markets = await discoverer.get_active_weather_markets()
            except Exception as e:
                logger.warning(f"[MONITOR] Failed to fetch full city markets: {e}. Using empty context.")
                all_markets = []

            # Run all city batches concurrently
            tasks = [process_monitor_city(city, trades, all_markets) for city, trades in city_groups.items()]
            if tasks:
                await asyncio.gather(*tasks)

            # 4. DECISION LOOP (Now with fresh probs)
            trades_sold = 0
            async with httpx.AsyncClient() as http_client:
                for trade in open_trades:
                    mem_key = f"{trade.market_id}_{trade.token_id}"
                    mem = ai_memory.get(mem_key, {})
                    predicted_prob = mem.get('predicted_prob', None)
                    
                    if predicted_prob is None:
                        logger.debug(f"[MONITOR 10min] Skipping {trade.city} - still no predicted_prob.")
                        continue
                        
                    shares = trade.size_usd / trade.entry_price if trade.entry_price > 0 else 0
                    starting_edge = (predicted_prob - trade.entry_price) * 100 

                    # 1. SMART EXIT DEPTH: Calculate actual exit price
                    exit_price, filled_sh, total_received = self._get_effective_exit_price(clob_client, trade.token_id, shares)
                    
                    if exit_price == 0:
                        logger.debug(f"[MONITOR 10min] No liquidity to exit {trade.city}")
                        continue

                    # Rate limit relief
                    await asyncio.sleep(0.05)

                    # Compute PnL and Edge with REAL exit price
                    unrealized_pnl = total_received - trade.size_usd
                    unrealized_pnl_percent = (unrealized_pnl / trade.size_usd) * 100 if trade.size_usd > 0 else 0
                    new_edge = (predicted_prob - exit_price) * 100
                    
                    # Log depth evaluation
                    # We compare against the actual best bid to show slippage in logs
                    try:
                        # (We can't easily get best_bid without extra call or parsing whole book again, 
                        # so we'll just log the WAP)
                        pass
                    except: pass

                    # Fetch end date to check Time Exit
                    hours_to_resolve = 999
                    try:
                        gamma_url = f"https://gamma-api.polymarket.com/markets?condition_id={trade.market_id}"
                        r = await http_client.get(gamma_url, timeout=10.0)
                        if r.status_code == 200:
                            m_data = r.json()
                            if m_data and len(m_data) > 0:
                                end_date_str = m_data[0].get("endDate")
                                if end_date_str:
                                    dt_obj = datetime.strptime(end_date_str, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc)
                                    diff = dt_obj - datetime.now(timezone.utc)
                                    hours_to_resolve = diff.total_seconds() / 3600
                    except:
                        pass

                    # Fetch current thresholds
                    tp_edge_limit = self.get_risk_setting("tp_edge", 0.0)
                    strong_tp_limit = self.get_risk_setting("strong_tp_pnl", 60.0)
                    sl_edge_limit = self.get_risk_setting("sl_edge", -12.0)
                    sl_pnl_limit = self.get_risk_setting("sl_pnl", -70.0)
                    time_exit_limit = self.get_risk_setting("time_exit_h", 6.0)
                    analysis_model = str(mem.get("analysis_model") or getattr(config, "analysis_model", "legacy")).lower()
                    is_tsas_trade = analysis_model == "tsas"
                    hold_tail_trade = (
                        is_tsas_trade
                        and trade.entry_price <= float(getattr(config, "tsas_hold_tail_max_entry_price", 0.05))
                        and predicted_prob >= float(getattr(config, "tsas_hold_tail_min_prob", 0.12))
                    )

                    exit_reason = None
                    sl_edge_threshold = -abs(sl_edge_limit)
                    sl_pnl_threshold = -abs(sl_pnl_limit)
                    
                    # DECISION LOGIC
                    if unrealized_pnl_percent <= sl_pnl_threshold:
                        exit_reason = "STOP_LOSS_PRICE"
                    elif new_edge <= sl_edge_threshold:
                        exit_reason = "STOP_LOSS_EDGE"
                    elif exit_price >= 0.98:
                        exit_reason = "TARGET_REACHED"
                    elif unrealized_pnl_percent >= strong_tp_limit:
                        exit_reason = "TAKE_PROFIT_STRONG"
                    elif new_edge <= tp_edge_limit and unrealized_pnl_percent > 0 and not hold_tail_trade:
                        exit_reason = "TAKE_PROFIT_EDGE"
                    elif hours_to_resolve < time_exit_limit and not hold_tail_trade:
                        exit_reason = "TIME_EXIT"

                    if exit_reason:
                        # Format info strings
                        q_text = mem.get("question", "")
                        date_str = self._extract_market_date(q_text)
                        temp_str = self._format_market_bin(q_text)
                        header_str = f"{trade.city} ({date_str}) [{temp_str}] {trade.outcome_name}"

                        logger.info(f"[MONITOR 10min] {header_str} | old_edge +{starting_edge:.1f}% → new_edge {new_edge:+.1f}% → {exit_reason} SELL {shares:.2f} shares @ WAP {exit_price:.3f} (Entry: {trade.entry_price:.3f}) | PnL {unrealized_pnl:+.2f}$ ({unrealized_pnl_percent:+.1f}%)")
                        
                        if not config.dry_run:
                            try:
                                order_args = OrderArgs(
                                    price=exit_price,
                                    size=shares,
                                    side="SELL",
                                    token_id=trade.token_id
                                )
                                resp = clob_client.create_and_post_order(order_args)
                                if resp and resp.get("success"):
                                    msg = f"🔔 <b>[MONITOR EXIT]</b>\n<b>Market:</b> {header_str}\n<b>Reason:</b> {exit_reason}\n<b>Shares:</b> {shares:.2f}\n<b>Entry Price:</b> {trade.entry_price}\n<b>Exit Price:</b> {exit_price}\n<b>PnL:</b> {unrealized_pnl:+.2f}$"
                                    await send_telegram_message(msg)
                                    
                                    # Update DB
                                    trade.status = "SOLD"
                                    trade.resolved_at = datetime.utcnow()
                                    from src.calibration import calibration_engine
                                    calibration_engine.mark_trade_closed(
                                        trade.token_id, 
                                        actual_outcome=None, 
                                        exit_price=exit_price, 
                                        realized_pnl=unrealized_pnl
                                    )
                                        
                                    session.commit()
                                    trades_sold += 1
                                else:
                                    logger.error(f"[MONITOR 10min] Sell order failed: {resp}")
                            except Exception as e:
                                logger.error(f"[MONITOR 10min] Execution exception: {e}")
                                traceback.print_exc()
                        else:
                            # Dry run logging
                            msg = f"🔔 <b>[DRY_RUN EXIT]</b>\n<b>Market:</b> {header_str}\n<b>Reason:</b> {exit_reason}\n<b>Shares:</b> {shares:.2f}\n<b>Entry Price:</b> {trade.entry_price}\n<b>Exit Price:</b> {exit_price}\n<b>PnL:</b> {unrealized_pnl:+.2f}$"
                            await send_telegram_message(msg)
                            trade.status = "SOLD"
                            trade.resolved_at = datetime.utcnow()
                            from src.calibration import calibration_engine
                            calibration_engine.mark_trade_closed(
                                trade.token_id, 
                                actual_outcome=None, 
                                exit_price=exit_price, 
                                realized_pnl=unrealized_pnl
                            )
                            session.commit()
                            trades_sold += 1
                
                if trades_sold == 0:
                    logger.info(f"[MONITOR 10min] Finished check of {len(open_trades)} trades. No positions reached exit thresholds.")
                else:
                    logger.info(f"[MONITOR 10min] Finished check. Sold {trades_sold} positions.")

        except Exception as e:
            logger.error(f"[MONITOR 10min] Fatal error: {e}")
            traceback.print_exc()
        finally:
            session.close()

    async def monitor_open_trades(self, clob_client) -> None:
        """Re-check open positions, refresh probabilities, and decide whether to exit."""
        if not clob_client:
            logger.warning("[MONITOR 10min] clob_client is None. Skipping.")
            return

        from src.weather_data import weather_fetcher
        from src.ai_analyzer import ai_analyzer
        from src.calibration import calibration_engine
        from src.market_discovery import MarketDiscoverer

        session = self.Session()
        try:
            open_trades = session.query(TradePosition).filter_by(status="OPEN").all()
            if not open_trades:
                logger.info("[MONITOR 10min] No open trades to monitor.")
                return

            logger.info(f"[MONITOR 10min] Started full re-analysis for {len(open_trades)} open trades...")

            ai_memory = {}
            mem_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "historical_predictions.jsonl")
            if os.path.exists(mem_path):
                with open(mem_path, 'r', encoding='utf-8') as f:
                    for line in f:
                        if not line.strip():
                            continue
                        try:
                            record = json.loads(line)
                            key = f"{record.get('market_id')}_{record.get('token_id')}"
                            ai_memory[key] = record
                        except Exception:
                            pass

            city_groups = {}
            for trade in open_trades:
                city_groups.setdefault(trade.city, []).append(trade)

            icao_codes = list({
                config.city_icao_mapping.get(city)
                for city in city_groups.keys()
                if config.city_icao_mapping.get(city)
            })
            weather_data_map = {}
            if icao_codes:
                logger.info(f"[MONITOR] Fetching weather for {len(icao_codes)} stations (with retries)...")
                weather_data_map = await weather_fetcher.fetch_weather_for_icao(icao_codes)

                if weather_data_map is None:
                    logger.warning("[MONITOR] [SAFETY ABORT] Missing METAR/TAF after retries. Skipping model re-analysis for this cycle.")
                    weather_data_map = {}
                else:
                    for icao in icao_codes:
                        try:
                            om_data = await weather_fetcher.fetch_open_meteo(icao, force_refresh=True)
                            if icao in weather_data_map:
                                weather_data_map[icao].update(om_data)
                        except Exception:
                            pass

            semaphore = asyncio.Semaphore(len(ai_analyzer.clients) if hasattr(ai_analyzer, 'clients') and ai_analyzer.clients else 1)

            discoverer = MarketDiscoverer()
            try:
                all_markets = await discoverer.get_active_weather_markets()
            except Exception as e:
                logger.warning(f"[MONITOR] Failed to fetch full city markets: {e}. Using empty context.")
                all_markets = []

            async def process_monitor_city(city: str, trades: list, all_city_markets: list):
                async with semaphore:
                    icao = config.city_icao_mapping.get(city)
                    weather = weather_data_map.get(icao)
                    if not weather:
                        logger.warning(f"[MONITOR] No weather for {city}, skipping re-analysis.")
                        return

                    city_markets = [
                        market for market in all_city_markets
                        if market.get("city") == city and market.get("outcomes")
                    ]

                    if not city_markets:
                        logger.warning(f"[MONITOR] No active markets found for {city}, skipping re-analysis.")
                        return

                    logger.info(f"[MONITOR] [{city}] Re-analyzing {len(city_markets)} markets with fresh weather...")
                    fresh_signals = await ai_analyzer.analyze_city_batch(city, city_markets, weather, return_all=True)
                    fresh_by_token = {
                        str(signal.get("token_id")): signal
                        for signal in fresh_signals
                        if signal.get("token_id")
                    }

                    for trade in trades:
                        mem_key = f"{trade.market_id}_{trade.token_id}"
                        old_mem = ai_memory.get(mem_key, {})
                        old_prob = old_mem.get("predicted_prob")
                        trade_token = str(trade.token_id)
                        matching_sig = fresh_by_token.get(trade_token)

                        if not matching_sig:
                            fallback_market = next(
                                (
                                    market for market in city_markets
                                    if str(market.get("market_id")) == str(trade.market_id)
                                    or any(str(outcome.get("token_id")) == trade_token for outcome in market.get("outcomes", []))
                                ),
                                None,
                            )
                            if fallback_market:
                                fallback_signals = await ai_analyzer.analyze_city_batch(
                                    city,
                                    [fallback_market],
                                    weather,
                                    return_all=True,
                                )
                                matching_sig = next(
                                    (
                                        signal for signal in fallback_signals
                                        if str(signal.get("token_id")) == trade_token
                                    ),
                                    None,
                                )

                        if not matching_sig:
                            logger.debug(
                                f"[MONITOR] No fresh signal found for {trade.city} "
                                f"market {str(trade.market_id)[:8]} token {trade_token[:8]}..."
                            )
                            continue

                        new_prob = matching_sig.get("predicted_prob")
                        if new_prob is not None:
                            question = matching_sig.get("question", "")
                            date_str = self._extract_market_date(question)
                            temp_str = self._format_market_bin(question)

                            if old_prob is not None:
                                shift = (new_prob - old_prob) * 100
                                logger.info(
                                    f"[MONITOR] UPDATE {city} ({date_str}) [{temp_str}] \"{trade.outcome_name}\" | "
                                    f"Probability: {old_prob*100:.1f}% -> {new_prob*100:.1f}% | Change: {shift:+.1f}%"
                                )
                                family_distribution = matching_sig.get("family_distribution")
                                if family_distribution and abs(shift) >= 0.05:
                                    family_id = matching_sig.get("family_id") or f"{city} ({date_str})"
                                    family_raw_sum = matching_sig.get("family_raw_sum")
                                    family_norm_sum = matching_sig.get("family_norm_sum")
                                    market_divergence = float(matching_sig.get("family_market_divergence") or 0.0)
                                    coverage_suffix = (
                                        f" | coverage={family_raw_sum*100:.1f}%"
                                        if isinstance(family_raw_sum, (int, float))
                                        else ""
                                    )
                                    norm_suffix = (
                                        f" | norm_sum={family_norm_sum*100:.1f}%"
                                        if isinstance(family_norm_sum, (int, float))
                                        else ""
                                    )
                                    divergence_suffix = f" | market_div={market_divergence*100:.1f}%" if market_divergence > 0 else ""
                                    logger.info(
                                        f"[MONITOR] MODEL DISTRIBUTION {family_id}{coverage_suffix}{norm_suffix}{divergence_suffix} | {family_distribution}"
                                    )
                                    market_distribution = matching_sig.get("family_market_distribution")
                                    if market_distribution:
                                        logger.info(f"[MONITOR] MARKET DISTRIBUTION {family_id} | {market_distribution}")
                                if abs(shift) >= 15:
                                    reasoning = matching_sig.get("reasoning", "")
                                    if reasoning:
                                        logger.warning(
                                            f"[MONITOR] REASONING SHIFT for {city} ({date_str}) [{temp_str}] "
                                            f"\"{trade.outcome_name}\":\n           {reasoning}"
                                        )
                                    else:
                                        logger.warning(
                                            f"[MONITOR] Shift details unavailable for {city} ({date_str}) [{temp_str}] "
                                            f"\"{trade.outcome_name}\"."
                                        )
                            else:
                                logger.info(
                                    f"[MONITOR] UPDATE {city} ({date_str}) [{temp_str}] \"{trade.outcome_name}\" | "
                                    f"Probability: [first calculation] -> {new_prob*100:.1f}%"
                                )
                                family_distribution = matching_sig.get("family_distribution")
                                if family_distribution:
                                    family_id = matching_sig.get("family_id") or f"{city} ({date_str})"
                                    family_raw_sum = matching_sig.get("family_raw_sum")
                                    family_norm_sum = matching_sig.get("family_norm_sum")
                                    market_divergence = float(matching_sig.get("family_market_divergence") or 0.0)
                                    coverage_suffix = (
                                        f" | coverage={family_raw_sum*100:.1f}%"
                                        if isinstance(family_raw_sum, (int, float))
                                        else ""
                                    )
                                    norm_suffix = (
                                        f" | norm_sum={family_norm_sum*100:.1f}%"
                                        if isinstance(family_norm_sum, (int, float))
                                        else ""
                                    )
                                    divergence_suffix = f" | market_div={market_divergence*100:.1f}%" if market_divergence > 0 else ""
                                    logger.info(
                                        f"[MONITOR] MODEL DISTRIBUTION {family_id}{coverage_suffix}{norm_suffix}{divergence_suffix} | {family_distribution}"
                                    )
                                    market_distribution = matching_sig.get("family_market_distribution")
                                    if market_distribution:
                                        logger.info(f"[MONITOR] MARKET DISTRIBUTION {family_id} | {market_distribution}")

                        calibration_engine.update_prediction_prob(trade.token_id, new_prob)
                        ai_memory[mem_key] = matching_sig

            tasks = [process_monitor_city(city, trades, all_markets) for city, trades in city_groups.items()]
            if tasks:
                await asyncio.gather(*tasks)

            trades_sold = 0
            async with httpx.AsyncClient() as http_client:
                for trade in open_trades:
                    mem_key = f"{trade.market_id}_{trade.token_id}"
                    mem = ai_memory.get(mem_key, {})
                    predicted_prob = mem.get("predicted_prob")

                    if predicted_prob is None:
                        logger.debug(f"[MONITOR 10min] Skipping {trade.city} - still no predicted_prob.")
                        continue

                    shares = trade.size_usd / trade.entry_price if trade.entry_price > 0 else 0

                    exit_price, filled_shares, total_received = self._get_effective_exit_price(clob_client, trade.token_id, shares)
                    if exit_price == 0:
                        logger.debug(f"[MONITOR 10min] No liquidity to exit {trade.city}")
                        continue

                    await asyncio.sleep(0.05)

                    unrealized_pnl = total_received - trade.size_usd
                    unrealized_pnl_percent = (unrealized_pnl / trade.size_usd) * 100 if trade.size_usd > 0 else 0
                    new_edge = (predicted_prob - exit_price) * 100

                    hours_to_resolve = 999
                    try:
                        gamma_url = f"https://gamma-api.polymarket.com/markets?condition_id={trade.market_id}"
                        response = await http_client.get(gamma_url, timeout=10.0)
                        if response.status_code == 200:
                            market_data = response.json()
                            if market_data:
                                end_date_str = market_data[0].get("endDate")
                                if end_date_str:
                                    dt_obj = datetime.strptime(end_date_str, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc)
                                    diff = dt_obj - datetime.now(timezone.utc)
                                    hours_to_resolve = diff.total_seconds() / 3600
                    except Exception:
                        pass

                    tp_edge_limit = self.get_risk_setting("tp_edge", 0.0)
                    strong_tp_limit = self.get_risk_setting("strong_tp_pnl", 60.0)
                    sl_edge_limit = self.get_risk_setting("sl_edge", -12.0)
                    sl_pnl_limit = self.get_risk_setting("sl_pnl", -70.0)
                    time_exit_limit = self.get_risk_setting("time_exit_h", 6.0)
                    analysis_model = str(mem.get("analysis_model") or getattr(config, "analysis_model", "legacy")).lower()
                    is_tsas_trade = analysis_model == "tsas"
                    entry_predicted_prob = float(mem.get("entry_predicted_prob", predicted_prob))
                    starting_edge = (entry_predicted_prob - trade.entry_price) * 100
                    prob_drop_points = (entry_predicted_prob - predicted_prob) * 100
                    prob_shift_points = abs(predicted_prob - entry_predicted_prob) * 100
                    model_flip_floor = float(getattr(config, "tsas_exit_model_flip_floor", 0.30))
                    crossed_model_flip_floor = (
                        entry_predicted_prob > model_flip_floor
                        and predicted_prob <= model_flip_floor
                    )
                    current_tsas_confidence = float(mem.get("tsas_confidence", 1.0))
                    hold_tail_trade = (
                        is_tsas_trade
                        and trade.entry_price <= float(getattr(config, "tsas_hold_tail_max_entry_price", 0.05))
                        and predicted_prob >= float(getattr(config, "tsas_hold_tail_min_prob", 0.12))
                    )

                    exit_reason = None
                    sl_edge_threshold = -abs(sl_edge_limit)
                    sl_pnl_threshold = -abs(sl_pnl_limit)

                    if unrealized_pnl_percent <= sl_pnl_threshold:
                        exit_reason = "STOP_LOSS_PRICE"
                    elif new_edge <= sl_edge_threshold:
                        exit_reason = "STOP_LOSS_EDGE"
                    elif exit_price >= 0.98:
                        exit_reason = "TARGET_REACHED"
                    elif (
                        is_tsas_trade
                        and crossed_model_flip_floor
                    ):
                        exit_reason = "TSAS_MODEL_FLIP"
                    elif (
                        is_tsas_trade
                        and prob_drop_points >= float(getattr(config, "tsas_exit_probability_collapse_points", 35.0))
                        and predicted_prob <= float(getattr(config, "tsas_exit_probability_floor", 0.35))
                    ):
                        exit_reason = "TSAS_PROBABILITY_COLLAPSE"
                    elif unrealized_pnl_percent >= strong_tp_limit:
                        exit_reason = "TAKE_PROFIT_STRONG"
                    elif (
                        is_tsas_trade
                        and not hold_tail_trade
                        and current_tsas_confidence <= float(getattr(config, "tsas_exit_confidence_floor", 0.25))
                        and new_edge <= float(getattr(config, "tsas_exit_confidence_edge_floor", 4.0))
                    ):
                        exit_reason = "TSAS_CONFIDENCE_DECAY"
                    elif (
                        is_tsas_trade
                        and not hold_tail_trade
                        and prob_drop_points >= float(getattr(config, "tsas_exit_prob_drop_points", 10.0))
                        and new_edge <= float(getattr(config, "tsas_exit_edge_floor", 2.0))
                    ):
                        exit_reason = "TSAS_THESIS_DECAY"
                    elif new_edge <= tp_edge_limit and unrealized_pnl_percent > 0 and not hold_tail_trade:
                        exit_reason = "TAKE_PROFIT_EDGE"
                    elif hours_to_resolve < time_exit_limit and not hold_tail_trade:
                        exit_reason = "TIME_EXIT"

                    if not exit_reason:
                        continue

                    question = mem.get("question", "")
                    date_str = self._extract_market_date(question)
                    temp_str = self._format_market_bin(question)
                    header_str = f"{trade.city} ({date_str}) [{temp_str}] {trade.outcome_name}"

                    logger.info(
                        f"[MONITOR 10min] {header_str} | old_edge +{starting_edge:.1f}% -> "
                        f"new_edge {new_edge:+.1f}% -> {exit_reason} SELL {shares:.2f} shares @ WAP {exit_price:.3f} "
                        f"(Entry: {trade.entry_price:.3f}) | PnL {unrealized_pnl:+.2f}$ ({unrealized_pnl_percent:+.1f}%)"
                    )

                    if not config.dry_run:
                        try:
                            order_args = OrderArgs(
                                price=exit_price,
                                size=shares,
                                side="SELL",
                                token_id=trade.token_id
                            )
                            resp = clob_client.create_and_post_order(order_args)
                            if resp and resp.get("success"):
                                msg = (
                                    f"<b>[MONITOR EXIT]</b>\n"
                                    f"<b>Market:</b> {header_str}\n"
                                    f"<b>Reason:</b> {exit_reason}\n"
                                    f"<b>Shares:</b> {shares:.2f}\n"
                                    f"<b>Entry Price:</b> {trade.entry_price}\n"
                                    f"<b>Exit Price:</b> {exit_price}\n"
                                    f"<b>PnL:</b> {unrealized_pnl:+.2f}$"
                                )
                                await send_telegram_message(msg)

                                trade.status = "SOLD"
                                trade.resolved_at = datetime.utcnow()
                                calibration_engine.mark_trade_closed(
                                    trade.token_id,
                                    actual_outcome=None,
                                    exit_price=exit_price,
                                    realized_pnl=unrealized_pnl
                                )
                                session.commit()
                                trades_sold += 1
                            else:
                                logger.error(f"[MONITOR 10min] Sell order failed: {resp}")
                        except Exception as e:
                            logger.error(f"[MONITOR 10min] Execution exception: {e}")
                            traceback.print_exc()
                    else:
                        msg = (
                            f"<b>[DRY RUN EXIT]</b>\n"
                            f"<b>Market:</b> {header_str}\n"
                            f"<b>Reason:</b> {exit_reason}\n"
                            f"<b>Shares:</b> {shares:.2f}\n"
                            f"<b>Entry Price:</b> {trade.entry_price}\n"
                            f"<b>Exit Price:</b> {exit_price}\n"
                            f"<b>PnL:</b> {unrealized_pnl:+.2f}$"
                        )
                        await send_telegram_message(msg)
                        trade.status = "SOLD"
                        trade.resolved_at = datetime.utcnow()
                        calibration_engine.mark_trade_closed(
                            trade.token_id,
                            actual_outcome=None,
                            exit_price=exit_price,
                            realized_pnl=unrealized_pnl
                        )
                        session.commit()
                        trades_sold += 1

                if trades_sold == 0:
                    logger.info(f"[MONITOR 10min] Finished check of {len(open_trades)} trades. No positions reached exit thresholds.")
                else:
                    logger.info(f"[MONITOR 10min] Finished check. Sold {trades_sold} positions.")

        except Exception as e:
            logger.error(f"[MONITOR 10min] Fatal error: {e}")
            traceback.print_exc()
        finally:
            session.close()

    async def liquidate_all_trades(self, clob_client) -> None:
        """Экстренно закрывает все открытые позиции по рыночным ценам."""
        from py_clob_client.clob_types import OrderArgs
        if not clob_client:
            logger.warning("[LIQUIDATE] clob_client is None. Skipping.")
            return

        session = self.Session()
        try:
            open_trades = session.query(TradePosition).filter_by(status="OPEN").all()
            if not open_trades:
                logger.info("[LIQUIDATE] No open trades to liquidate.")
                return

            logger.info(f"[LIQUIDATE] Starting EMERGENCY LIQUIDATION for {len(open_trades)} trades...")
            
            for trade in open_trades:
                shares = trade.size_usd / trade.entry_price if trade.entry_price > 0 else 0
                exit_price, _, total_received = self._get_effective_exit_price(clob_client, trade.token_id, shares)
                
                if exit_price == 0:
                    logger.warning(f"[LIQUIDATE] No liquidity for {trade.token_id}. Skipping.")
                    continue
                
                # To guarantee execution, drop price slightly if needed, but not below 0.01 (WAP handles it mostly)
                safe_exit = max(0.01, round(exit_price - 0.01, 3))
                unrealized_pnl = total_received - trade.size_usd

                if not config.dry_run:
                    try:
                        order_args = OrderArgs(
                            price=safe_exit,
                            size=shares,
                            side="SELL",
                            token_id=trade.token_id
                        )
                        resp = clob_client.create_and_post_order(order_args)
                        if resp and resp.get("success"):
                            trade.status = "SOLD"
                            trade.resolved_at = datetime.utcnow()
                            from src.calibration import calibration_engine
                            calibration_engine.mark_trade_closed(trade.token_id, actual_outcome=None, exit_price=safe_exit, realized_pnl=unrealized_pnl)
                            logger.info(f"[LIQUIDATE] Sold {trade.city} shares at {safe_exit}")
                        else:
                            logger.error(f"[LIQUIDATE] Failed to sell {trade.city}: {resp}")
                    except Exception as e:
                        logger.error(f"[LIQUIDATE] Exception selling {trade.city}: {e}")
                else:
                    trade.status = "SOLD"
                    trade.resolved_at = datetime.utcnow()
                    from src.calibration import calibration_engine
                    calibration_engine.mark_trade_closed(trade.token_id, actual_outcome=None, exit_price=safe_exit, realized_pnl=unrealized_pnl)
                    logger.info(f"[DRY_RUN LIQUIDATE] Sold {trade.city} shares at {safe_exit}")

            session.commit()
            logger.info("[LIQUIDATE] Emergency liquidation complete.")
        except Exception as e:
            logger.error(f"Error during liquidate_all_trades: {e}")
            import traceback
            traceback.print_exc()
        finally:
            session.close()

portfolio_manager = PortfolioManager()
