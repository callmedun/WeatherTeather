import httpx
import asyncio
from typing import List, Dict, Any
from src.utils import logger
from config.settings import config
from datetime import datetime, timedelta, timezone
from src.portfolio_manager import portfolio_manager

class MarketDiscoverer:
    def __init__(self):
        self.gamma_api_url = "https://gamma-api.polymarket.com"

    async def get_active_weather_markets(self) -> List[Dict[str, Any]]:
        """
        Fetches all active Polymarket events/markets and filters for "Highest temperature" items.
        Returns a structured list of markets ready for analysis.
        """
        markets = []
        # Polymarket's Gamma API often requires pagination or specific tag filtering.
        # We will fetch by pulling the latest active markets.
        # Alternatively, we could filter by tags or search queries if the API supports it.
        # For robustness, we will hit the events endpoint.
        
        limit = 100
        offset = 0
        has_more = True
        
        target_cities = list(config.city_icao_mapping.keys())
        logger.info(f"Scanning Polymarket for weather markets across {len(target_cities)} cities (Horizon: {config.scan_days_ahead} days)...")
        
        active_market_ids = set(portfolio_manager.get_active_market_ids())
        if active_market_ids:
            logger.info(f"Loaded {len(active_market_ids)} open market IDs. Will skip processing to prevent overlapping trades.")
        
        async with httpx.AsyncClient() as client:
            # We'll reasonably bound the search to recent active markets to avoid huge overhead
            while has_more:
                url = f"{self.gamma_api_url}/events?limit={limit}&offset={offset}&tag_slug=weather&active=true&closed=false"
                try:
                    response = await client.get(url, timeout=15.0)
                    response.raise_for_status()
                    data = response.json()
                    
                    if not data:
                        has_more = False
                        break
                        
                    for event in data:
                        title = event.get("title", "")
                        
                        # Look for weather/temperature titles
                        if "highest temperature" in title.lower() or "temperature in" in title.lower():
                            for market in event.get("markets", []):
                                if market.get("closed") or not market.get("active"):
                                    continue
                                
                                # Filter Date Horizon
                                end_date_str = market.get("endDate", "")
                                if end_date_str:
                                    try:
                                        # Example: "2024-04-10T12:00:00Z"
                                        dt_obj = datetime.strptime(end_date_str, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc)
                                        limit_date = datetime.now(timezone.utc) + timedelta(days=config.scan_days_ahead)
                                        if dt_obj > limit_date:
                                            continue
                                    except Exception:
                                        pass
                                
                                # Find the city match
                                matched_city = next((city for city in target_cities if city.lower() in title.lower()), None)
                                
                                if matched_city:
                                    market_info = self._parse_market(market, matched_city, title)
                                    if market_info:
                                        if market_info["market_id"] in active_market_ids:
                                            # Skip to prevent duplicates
                                            continue
                                        markets.append(market_info)

                    offset += limit
                    await asyncio.sleep(0.1) # Respect rate limits when scanning everything
                except Exception as e:
                    logger.error(f"Error fetching markets at offset {offset}: {e}")
                    break
                    
        logger.success(f"Found {len(markets)} active weather markets.")
        return markets

    async def get_clob_best_ask(self, token_id: str) -> float:
        """Fetch exact orderbook minimum ASKING price for a specific token"""
        url = f"https://clob.polymarket.com/book?token_id={token_id}"
        try:
            async with httpx.AsyncClient() as client:
                res = await client.get(url, timeout=5.0)
                if res.status_code == 200:
                    asks = res.json().get("asks", [])
                    if asks:
                        return float(asks[0]["price"])
        except Exception:
            pass
        return None

    def _parse_market(self, market: Dict[str, Any], city: str, event_title: str) -> Dict[str, Any]:
        try:
            # Needs to extract tokens and outcome prices
            outcomes = market.get("outcomes", "[]") # usually a stringified list or list
            if isinstance(outcomes, str):
                import json
                outcomes = json.loads(outcomes)
                
            token_ids = market.get("clobTokenIds", "[]")
            if isinstance(token_ids, str):
                import json
                token_ids = json.loads(token_ids)
                
            prices = market.get("outcomePrices", "[]")
            if isinstance(prices, str):
                import json
                prices = json.loads(prices)
            
            # Map Yes/No or range outcomes
            parsed_outcomes = []
            for i, outcome in enumerate(outcomes):
                if i < len(token_ids) and i < len(prices):
                    try:
                        base_price = float(prices[i])
                    except (ValueError, TypeError):
                        base_price = 0.0
                        
                    final_price = base_price
                    
                    # Capture exact executable Orderbook Spread via Gamma's bestAsk/bestBid
                    if i == 0:
                        gamma_ask = float(market.get("bestAsk") or 0.0)
                        if gamma_ask > 0:
                            final_price = gamma_ask
                    elif i == 1:
                        gamma_bid = float(market.get("bestBid") or 0.0)
                        if gamma_bid > 0:
                            # The best Ask for 'No' is the inverse of the best Bid for 'Yes'
                            final_price = round(1.0 - gamma_bid, 4)
                            
                    parsed_outcomes.append({
                        "name": outcome,
                        "token_id": token_ids[i],
                        "current_price": final_price
                    })

            # Check resolution date
            end_date = market.get("endDate", "")
            
            return {
                "market_id": market.get("conditionId"),
                "question": market.get("question"),
                "event_title": event_title,
                "city": city,
                "icao_code": config.city_icao_mapping.get(city),
                "resolution_date": end_date,
                "outcomes": parsed_outcomes
            }
        except Exception as e:
            logger.warning(f"Failed to parse market {market.get('conditionId')}: {e}")
            return None
