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

    def can_trade_city(self, city: str, intended_size: float = None) -> bool:
        if intended_size is None:
            intended_size = config.default_trade_size
            
        current_city_exposure = self.get_open_exposure_for_city(city)
        total_exposure = self.get_total_open_exposure()
        
        # Max bankroll % used per day total 
        max_total = self.assumed_bankroll * config.max_total_exposure
        # Max city %
        max_city = self.assumed_bankroll * config.max_city_exposure
        
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
                "tp_edge": 5.0,         # Take Profit Edge (5%)
                "strong_tp_pnl": 30.0,   # Strong PnL TP (30%)
                "sl_edge": -12.0,        # Stop Loss Edge (-12%)
                "sl_pnl": -30.0,         # Stop Loss PnL (-30%)
                "time_exit_h": 6.0       # Time-based exit (6 hours)
            }
            for k, v in defaults.items():
                existing = session.query(RiskSetting).filter_by(key=k).first()
                if not existing:
                    session.add(RiskSetting(key=k, value=v))
                else:
                    # Auto-migration of old defaults to new standards
                    if k == "tp_edge" and existing.value == 10.0:
                        existing.value = v
                    elif k == "sl_pnl" and existing.value == -15.0:
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
                            om_data = await weather_fetcher.fetch_open_meteo(icao)
                            if icao in weather_data_map:
                                weather_data_map[icao].update(om_data)
                        except: pass

            # Re-analyze each city
            for city, trades in city_groups.items():
                icao = config.city_icao_mapping.get(city)
                w_data = weather_data_map.get(icao)
                if not w_data:
                    logger.warning(f"[MONITOR] No weather for {city}, skipping re-analysis.")
                    continue
                
                # Construct "market" objects for AI analyzer from memory, grouped by market_id
                city_markets_dict = {}
                for t in trades:
                    mem_key = f"{t.market_id}_{t.token_id}"
                    mem = ai_memory.get(mem_key)
                    if not mem:
                        logger.debug(f"[MONITOR] Missing metadata for {t.city} {t.token_id}. Skipping re-analysis.")
                        continue
                        
                    if t.market_id not in city_markets_dict:
                        city_markets_dict[t.market_id] = {
                            "market_id": t.market_id,
                            "question": mem.get("question"),
                            "event_title": mem.get("event_title"),
                            "city": city,
                            "outcomes": []
                        }
                    
                    curr_price = self.get_current_price(clob_client, t.token_id)
                    curr_price = curr_price if curr_price is not None else 0.5
                    
                    city_markets_dict[t.market_id]["outcomes"].append({
                        "name": t.outcome_name, 
                        "token_id": t.token_id, 
                        "current_price": curr_price
                    })
                    alt_name = "No" if t.outcome_name.lower() == "yes" else "Yes"
                    city_markets_dict[t.market_id]["outcomes"].append({
                        "name": alt_name,
                        "token_id": "dummy_" + alt_name,
                        "current_price": max(0.01, 1.0 - curr_price)
                    })
                
                city_markets_for_ai = list(city_markets_dict.values())
                if city_markets_for_ai:
                    logger.info(f"[MONITOR] [{city}] Re-analyzing {len(city_markets_for_ai)} markets with fresh weather...")
                    fresh_signals = await ai_analyzer.analyze_city_batch(city, city_markets_for_ai, w_data, return_all=True)
                    # Update memory with fresh probs and log shifts for active trades
                    for t in trades:
                        mem_key = f"{t.market_id}_{t.token_id}"
                        old_mem = ai_memory.get(mem_key, {})
                        old_prob = old_mem.get('predicted_prob')
                        
                        # Find the fresh signal for this specific token
                        matching_sig = next((s for s in fresh_signals if s.get("token_id") == t.token_id), None)
                        
                        if matching_sig:
                            new_prob = matching_sig.get('predicted_prob')
                            if new_prob is not None:
                                # Extract clean name
                                q_text = matching_sig.get("question", "")
                                date_match = re.search(r'on\s+([A-Za-z]+\s+\d+)', q_text)
                                date_str = date_match.group(1) if date_match else "N/A"
                                
                                if old_prob is not None:
                                    shift = (new_prob - old_prob) * 100
                                    logger.info(f"[MONITOR] 🔄 {city} ({date_str}) \"{t.outcome_name}\" | ИИ: {old_prob*100:.1f}% ➔ {new_prob*100:.1f}% | Изменение: {shift:+.1f}%")
                                else:
                                    logger.info(f"[MONITOR] 🔄 {city} ({date_str}) \"{t.outcome_name}\" | ИИ: [Нет старого] ➔ {new_prob*100:.1f}%")
                            
                            # Update probability for open trade in history
                            calibration_engine.update_prediction_prob(t.token_id, new_prob)
                            # Update local dict for current loop
                            ai_memory[mem_key] = matching_sig


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
                    tp_edge_limit = self.get_risk_setting("tp_edge", 10.0)
                    strong_tp_limit = self.get_risk_setting("strong_tp_pnl", 30.0)
                    sl_edge_limit = self.get_risk_setting("sl_edge", -12.0)
                    sl_pnl_limit = self.get_risk_setting("sl_pnl", -15.0)
                    time_exit_limit = self.get_risk_setting("time_exit_h", 6.0)

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
                    elif new_edge <= tp_edge_limit and unrealized_pnl_percent > 0:
                        exit_reason = "TAKE_PROFIT_EDGE"
                    elif hours_to_resolve < time_exit_limit:
                        exit_reason = "TIME_EXIT"

                    if exit_reason:
                        # Format info strings
                        q_text = mem.get("question", "")
                        date_match = re.search(r'on\s+([A-Za-z]+\s+\d+)', q_text)
                        temp_match = re.search(r'be\s+(.*?)(?:\s+or\s+|\?$|$)', q_text)
                        date_str = date_match.group(1) if date_match else "N/A"
                        temp_str = temp_match.group(1).strip() if temp_match else "N/A"
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
