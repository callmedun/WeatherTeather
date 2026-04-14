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
                "tp_edge": 10.0,         # Take Profit Edge (10%)
                "strong_tp_pnl": 30.0,   # Strong PnL TP (30%)
                "sl_edge": -12.0,        # Stop Loss Edge (-12%)
                "sl_pnl": -15.0,         # Stop Loss PnL (-15%)
                "time_exit_h": 6.0       # Time-based exit (6 hours)
            }
            for k, v in defaults.items():
                existing = session.query(RiskSetting).filter_by(key=k).first()
                if not existing:
                    session.add(RiskSetting(key=k, value=v))
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

    async def monitor_open_trades(self, clob_client) -> None:
        """Проверяет ТОЛЬКО открытые позиции каждые 30 минут и решает закрывать или нет.
        НЕ сканирует новые рынки!"""
        if not clob_client:
            logger.warning("[MONITOR 10min] clob_client is None. Skipping.")
            return

        session = self.Session()
        try:
            open_trades = session.query(TradePosition).filter_by(status="OPEN").all()
            if not open_trades:
                logger.info("[MONITOR 10min] No open trades to monitor.")
                return
            
            logger.info(f"[MONITOR 10min] Started checking {len(open_trades)} open trades...")
            
            # --- NEW: Batch Sync Prices once to avoid 429 later ---
            unique_tokens = list(set([t.token_id for t in open_trades if t.token_id]))
            self.get_current_prices(clob_client, unique_tokens)
            
            trades_sold = 0
            
            # Load AI prediction memory
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
                        except:
                            pass

            async with httpx.AsyncClient() as http_client:
                for trade in open_trades:
                    mem_key = f"{trade.market_id}_{trade.token_id}"
                    mem = ai_memory.get(mem_key, {})
                    old_edge = mem.get('edge', 0.0) # might not be saved directly, fallback calc
                    predicted_prob = mem.get('predicted_prob', None)
                    
                    if predicted_prob is None:
                        logger.debug(f"[MONITOR 10min] Skipping {trade.city} - no predicted_prob in memory.")
                        continue
                        
                    # Calculate true starting edge manually in case it wasn't saved in json
                    starting_edge = (predicted_prob - trade.entry_price) * 100 

                    # Fetch live orderbook to compute spread and best bid
                    try:
                        ob = clob_client.get_order_book(trade.token_id)
                        bids = getattr(ob, "bids", [])
                        asks = getattr(ob, "asks", [])
                        
                        best_bid = 0.0
                        if bids:
                            best_bid = max([float(getattr(b, 'price', b.get('price', 0.0) if isinstance(b, dict) else 0.0)) for b in bids])
                            
                        best_ask = 1.0
                        if asks:
                            best_ask = min([float(getattr(a, 'price', a.get('price', 1.0) if isinstance(a, dict) else 1.0)) for a in asks])
                    except Exception as e:
                        logger.warning(f"[MONITOR 10min] Orderbook fetch failed for {trade.token_id}: {e}")
                        continue

                    # Rate limit relief
                    await asyncio.sleep(0.1)

                    if best_bid == 0.0:
                        continue # No liquidity to exit
                        
                    spread = best_ask - best_bid
                    if spread > 0.06:
                        logger.debug(f"[MONITOR 10min] Spread too high ({spread*100:.1f}%) for {trade.city}. Skipping.")
                        continue

                    # Current best_bid is our exit price
                    exit_price = best_bid
                    shares = trade.size_usd / trade.entry_price if trade.entry_price > 0 else 0
                    
                    unrealized_pnl = (shares * exit_price) - trade.size_usd
                    unrealized_pnl_percent = (unrealized_pnl / trade.size_usd) * 100 if trade.size_usd > 0 else 0
                    
                    new_edge = (predicted_prob - exit_price) * 100
                    
                    # Fetch end date
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

                    # Fetch current thresholds from DB
                    tp_edge_limit = self.get_risk_setting("tp_edge", 10.0)
                    strong_tp_limit = self.get_risk_setting("strong_tp_pnl", 30.0)
                    sl_edge_limit = self.get_risk_setting("sl_edge", -12.0)
                    sl_pnl_limit = self.get_risk_setting("sl_pnl", -15.0)
                    time_exit_limit = self.get_risk_setting("time_exit_h", 6.0)

                    exit_reason = None
                    sell_shares = shares

                    # Ensure Stop Loss limits are treated as negative (loss) thresholds
                    sl_edge_threshold = -abs(sl_edge_limit)
                    sl_pnl_threshold = -abs(sl_pnl_limit)
                    
                    # PRIORITY ORDER: Stop Loss -> Strong TP -> Normal TP -> Time Exit
                    if new_edge <= sl_edge_threshold or unrealized_pnl_percent <= sl_pnl_threshold:
                        exit_reason = "stop_loss"
                    elif unrealized_pnl_percent >= strong_tp_limit:
                        exit_reason = "strong_take_profit"
                        sell_shares = int(shares / 2) # Partial sell
                    elif new_edge <= tp_edge_limit and unrealized_pnl_percent > 0:
                        exit_reason = "take_profit"
                    elif hours_to_resolve < time_exit_limit:
                        exit_reason = "time_based"

                    if exit_reason and sell_shares > 0:
                        sell_shares = int(sell_shares)
                        if sell_shares == 0:
                            continue # Too small to partial sell
                            
                        # Format Date/Temp
                        q_text = mem.get("question", "")
                        date_match = re.search(r'on\s+([A-Za-z]+\s+\d+)', q_text)
                        temp_match = re.search(r'be\s+(.*?)(?:\s+or\s+|\?$|$)', q_text)
                        date_str = date_match.group(1) if date_match else "N/A"
                        temp_str = temp_match.group(1).strip() if temp_match else "N/A"
                        
                        header_str = f"{trade.city} ({date_str}) [{temp_str}] {trade.outcome_name}"

                        logger.info(f"[MONITOR 10min] {header_str} | old_edge +{starting_edge:.1f}% → new_edge {new_edge:+.1f}% → {exit_reason.upper()} SELL {sell_shares} shares @ {exit_price} (Entry: {trade.entry_price}) | PnL {unrealized_pnl:+.2f}$")
                        
                        if not config.dry_run:
                            try:
                                order_args = OrderArgs(
                                    price=exit_price,
                                    size=sell_shares,
                                    side="SELL",
                                    token_id=trade.token_id
                                )
                                resp = clob_client.create_and_post_order(order_args)
                                if resp and resp.get("success"):
                                    msg = f"🔔 <b>[MONITOR 30min EXIT]</b>\n<b>Market:</b> {header_str}\n<b>Reason:</b> {exit_reason.upper()}\n<b>Shares:</b> {sell_shares}\n<b>Entry Price:</b> {trade.entry_price}\n<b>Exit Price:</b> {exit_price}\n<b>PnL:</b> {unrealized_pnl:+.2f}$"
                                    await send_telegram_message(msg)
                                    
                                    # Update DB
                                    if sell_shares >= int(shares):
                                        trade.status = "SOLD"
                                        trade.resolved_at = datetime.utcnow()
                                        from src.calibration import calibration_engine
                                        calibration_engine.mark_trade_closed(
                                            trade.token_id, 
                                            actual_outcome=None, 
                                            exit_price=exit_price, 
                                            realized_pnl=unrealized_pnl
                                        )
                                    else:
                                        # Reduce size
                                        trade.size_usd -= (sell_shares * trade.entry_price)
                                        
                                    session.commit()
                                    trades_sold += 1
                                else:
                                    logger.error(f"[MONITOR 10min] Sell order failed: {resp}")
                            except Exception as e:
                                logger.error(f"[MONITOR 10min] Execution exception: {e}")
                                traceback.print_exc()
                        else:
                            # Dry run logging
                            msg = f"🔔 <b>[DRY_RUN EXIT]</b>\n<b>Market:</b> {header_str}\n<b>Reason:</b> {exit_reason.upper()}\n<b>Shares:</b> {sell_shares}\n<b>Entry Price:</b> {trade.entry_price}\n<b>Exit Price:</b> {exit_price}\n<b>PnL:</b> {unrealized_pnl:+.2f}$"
                            await send_telegram_message(msg)
                            if sell_shares >= int(shares):
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

portfolio_manager = PortfolioManager()
