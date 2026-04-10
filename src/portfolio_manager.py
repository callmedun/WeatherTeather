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
    created_at = Column(DateTime, default=datetime.utcnow)
    resolved_at = Column(DateTime, nullable=True)

class PortfolioManager:
    def __init__(self):
        db_dir = "data"
        os.makedirs(db_dir, exist_ok=True)
        self.engine = create_engine(f"sqlite:///{db_dir}/portfolio.db")
        Base.metadata.create_all(self.engine)
        self.Session = sessionmaker(bind=self.engine)
        
        self.assumed_bankroll = 10000.0 
        
        # Cache for live prices (token_id -> {'price': float, 'time': timestamp})
        self._price_cache = {}
        self.cache_ttl = 30 # seconds

    def get_current_price(self, clob_client, token_id: str) -> float | None:
        """Получает текущую midpoint цену токена"""
        if not token_id:
            print(f"[ERROR] Некорректный token_id: {token_id}")
            return None
            
        print(f"[DEBUG] Запрос цены для token: {str(token_id)[:20]}...")
        if clob_client is None:
            print("[WARN] clob_client is None")
            return None
            
        now = datetime.now().timestamp()
        
        # Check cache
        if token_id in self._price_cache:
            if now - self._price_cache[token_id]['time'] < self.cache_ttl:
                print(f"[DEBUG] Cached price = {self._price_cache[token_id]['price']}")
                return self._price_cache[token_id]['price']
                
        try:
            # Основной способ (цена моментальной продажи = Best Bid, сторона покупателей "BUY")
            result = clob_client.get_price(token_id, side="BUY")
            
            price_val = None
            if isinstance(result, dict):
                if "price" in result:
                    price_val = float(result["price"])
                else:
                    print(f"[WARN] dict без ключа price: {result}")
            elif isinstance(result, (int, float, str)) and result is not None:
                price_val = float(result)
            else:
                print(f"[WARN] Неизвестный тип ответа get_price: {type(result)}")
                
            if price_val is not None:
                self._price_cache[token_id] = {'price': price_val, 'time': now}
                print(f"[SUCCESS] Instant sell price (Best Bid) = {price_val}")
                return price_val
            
            # Fallback на midpoint, если стакан пуст с одной стороны
            print("[WARN] Нет покупателей (Best Bid = None), пробуем Midpoint")
            mid_result = clob_client.get_midpoint(token_id)
            if isinstance(mid_result, dict):
                if "price" in mid_result:
                    val = float(mid_result["price"])
                    self._price_cache[token_id] = {'price': val, 'time': now}
                    print(f"[SUCCESS] Fallback Midpoint = {val}")
                    return val
                elif "mid" in mid_result:
                    val = float(mid_result["mid"])
                    self._price_cache[token_id] = {'price': val, 'time': now}
                    print(f"[SUCCESS] Fallback Midpoint = {val}")
                    return val
                    
            print("[WARN] Книга ордеров полностью пуста (None)")
            return None
        except Exception as e:
            logger.debug(f"[WARN] Не удалось получить цену для {token_id}: {e}")
            print(f"[ERROR] get_current_price failed: {e}")
            return None

    def record_trade(self, market_id: str, token_id: str, city: str, outcome: str, price: float, size: float):
        session = self.Session()
        try:
            trade = TradePosition(
                market_id=market_id,
                token_id=token_id,
                city=city,
                outcome_name=outcome,
                entry_price=price,
                size_usd=size,
                status="OPEN"
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

        return True

    async def monitor_open_trades(self, clob_client) -> None:
        """Проверяет ТОЛЬКО открытые позиции каждые 30 минут и решает закрывать или нет.
        НЕ сканирует новые рынки!"""
        if not clob_client:
            logger.warning("[MONITOR 30min] clob_client is None. Skipping.")
            return

        session = self.Session()
        try:
            open_trades = session.query(TradePosition).filter_by(status="OPEN").all()
            if not open_trades:
                logger.info("[MONITOR 30min] No open trades to monitor.")
                return
            
            logger.info(f"[MONITOR 30min] Started checking {len(open_trades)} open trades...")
            
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
                        logger.debug(f"[MONITOR 30min] Skipping {trade.city} - no predicted_prob in memory.")
                        continue
                        
                    bought_outcome = mem.get('bought_outcome', '').lower()
                    if bought_outcome == 'no':
                        predicted_prob = 1.0 - predicted_prob
                        
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
                        logger.warning(f"[MONITOR 30min] Orderbook fetch failed for {trade.token_id}: {e}")
                        continue

                    if best_bid == 0.0:
                        continue # No liquidity to exit
                        
                    spread = best_ask - best_bid
                    if spread > 0.06:
                        logger.debug(f"[MONITOR 30min] Spread too high ({spread*100:.1f}%) for {trade.city}. Skipping.")
                        continue

                    # Current best_bid is our exit price
                    exit_price = best_bid
                    shares = trade.size_usd / trade.entry_price if trade.entry_price > 0 else 0
                    
                    unrealized_pnl = (shares * exit_price) - trade.size_usd
                    unrealized_pnl_percent = (unrealized_pnl / trade.size_usd) * 100 if trade.size_usd > 0 else 0
                    
                    new_edge = (predicted_prob - exit_price) * 100
                    new_ev = (predicted_prob * exit_price) + ((1 - predicted_prob) * -1) # Simplified EV
                    
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

                    exit_reason = None
                    sell_shares = shares

                    if new_edge <= 3.0 or new_ev <= 0:
                        exit_reason = "take_profit"
                    elif new_edge >= 30.0:
                        exit_reason = "strong_take_profit"
                        sell_shares = int(shares / 2) # Partial sell
                    elif new_edge <= -12.0 or unrealized_pnl_percent <= -8.0:
                        exit_reason = "stop_loss"
                    elif hours_to_resolve < 6.0:
                        exit_reason = "time_based"

                    if exit_reason and sell_shares > 0:
                        sell_shares = int(sell_shares)
                        if sell_shares == 0:
                            continue # Too small to partial sell

                        logger.info(f"[MONITOR 30min] {trade.city} {trade.outcome_name} | old_edge +{starting_edge:.1f}% → new_edge {new_edge:+.1f}% → {exit_reason.upper()} SELL {sell_shares} shares @ {exit_price} | PnL {unrealized_pnl:+.2f}$")
                        
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
                                    msg = f"🔔 <b>[MONITOR 30min EXIT]</b>\n<b>City:</b> {trade.city} {trade.outcome_name}\n<b>Reason:</b> {exit_reason.upper()}\n<b>Old Edge:</b> {starting_edge:+.1f}%\n<b>New Edge:</b> {new_edge:+.1f}%\n<b>Shares:</b> {sell_shares}\n<b>PnL:</b> {unrealized_pnl:+.2f}$"
                                    await send_telegram_message(msg)
                                    
                                    # Update DB
                                    if sell_shares >= int(shares):
                                        trade.status = "SOLD"
                                        trade.resolved_at = datetime.utcnow()
                                    else:
                                        # Reduce size
                                        trade.size_usd -= (sell_shares * trade.entry_price)
                                        
                                    session.commit()
                                else:
                                    logger.error(f"[MONITOR 30min] Sell order failed: {resp}")
                            except Exception as e:
                                logger.error(f"[MONITOR 30min] Execution exception: {e}")
                                traceback.print_exc()
                        else:
                            # Dry run logging
                            msg = f"🔔 <b>[DRY_RUN EXIT]</b>\n<b>City:</b> {trade.city} {trade.outcome_name}\n<b>Reason:</b> {exit_reason.upper()}\n<b>PnL:</b> {unrealized_pnl:+.2f}$"
                            await send_telegram_message(msg)
                            if sell_shares >= int(shares):
                                trade.status = "SOLD"
                                trade.resolved_at = datetime.utcnow()
                            session.commit()

        except Exception as e:
            logger.error(f"[MONITOR 30min] Fatal error: {e}")
            traceback.print_exc()
        finally:
            session.close()

portfolio_manager = PortfolioManager()
