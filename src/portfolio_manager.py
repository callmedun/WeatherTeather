from sqlalchemy import create_engine, Column, Integer, String, Float, Boolean, DateTime
from sqlalchemy.orm import declarative_base, sessionmaker
from datetime import datetime
from src.utils import logger
from config.settings import config
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

portfolio_manager = PortfolioManager()
