"""
batch_price_fetcher.py -- Phase 4: Batch Price Fetching for Polymarket
=======================================================================
Replaces individual per-market price fetches with a single batch call
to the Gamma API /markets?ids=... endpoint.

Benefits:
  - 1 HTTP request instead of N requests for N markets
  - ~10x faster price refresh during monitoring cycles
  - Includes spread, bestBid, bestAsk, volume for each market
  - Results cached in memory with configurable TTL

Usage:
    fetcher = BatchPriceFetcher()
    prices = await fetcher.get_prices(token_ids)  # {token_id: PriceInfo}

    # Or for full market data:
    markets = await fetcher.get_markets_batch(market_ids)
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field
from typing import Optional

import httpx

logger = logging.getLogger(__name__)

GAMMA_API = "https://gamma-api.polymarket.com"
CLOB_API  = "https://clob.polymarket.com"

# Cache TTL in seconds
_DEFAULT_TTL = 60       # 1 min for prices during monitoring
_MARKET_TTL  = 300      # 5 min for full market data


@dataclass
class PriceInfo:
    """Price data for a single token."""
    token_id: str
    best_bid: float
    best_ask: float
    last_price: float
    spread: float
    volume_24h: float
    timestamp: float = field(default_factory=time.time)

    @property
    def mid_price(self) -> float:
        return (self.best_bid + self.best_ask) / 2.0 if self.best_ask > 0 else self.last_price

    @property
    def is_stale(self) -> bool:
        return time.time() - self.timestamp > _DEFAULT_TTL


@dataclass
class MarketInfo:
    """Full market snapshot including both outcomes."""
    market_id: str
    question: str
    active: bool
    closed: bool
    outcome_prices: dict[str, float]      # token_id -> price
    best_asks: dict[str, float]            # token_id -> best ask
    best_bids: dict[str, float]            # token_id -> best bid
    volume_24h: float
    liquidity_usd: float
    spread: float
    timestamp: float = field(default_factory=time.time)

    @property
    def is_stale(self) -> bool:
        return time.time() - self.timestamp > _MARKET_TTL


class BatchPriceFetcher:
    """
    Fetches prices for multiple Polymarket tokens in a single HTTP request.

    Primary method: get_prices(token_ids) -> dict[str, PriceInfo]
    Batch size: up to 100 token IDs per request (Gamma API limit).
    """

    BATCH_SIZE = 100  # Gamma API limit per request

    def __init__(self):
        self._price_cache: dict[str, PriceInfo] = {}
        self._market_cache: dict[str, MarketInfo] = {}
        self._lock = asyncio.Lock()

    def _cache_hit_count(self) -> int:
        return sum(1 for p in self._price_cache.values() if not p.is_stale)

    # ------------------------------------------------------------------
    # 1. Batch price fetch from CLOB /prices endpoint
    # ------------------------------------------------------------------

    async def get_prices(
        self,
        token_ids: list[str],
        force_refresh: bool = False,
    ) -> dict[str, PriceInfo]:
        """
        Return PriceInfo for each token_id.
        Uses cache; fetches only stale/missing tokens.
        """
        if not token_ids:
            return {}

        # Identify which tokens need a refresh
        to_fetch = [
            tid for tid in token_ids
            if force_refresh or tid not in self._price_cache or self._price_cache[tid].is_stale
        ]

        if to_fetch:
            async with self._lock:
                # Re-check under lock (another coroutine may have fetched already)
                to_fetch = [
                    tid for tid in to_fetch
                    if tid not in self._price_cache or self._price_cache[tid].is_stale
                ]
                if to_fetch:
                    await self._fetch_prices_batch(to_fetch)

        return {tid: self._price_cache[tid] for tid in token_ids if tid in self._price_cache}

    async def _fetch_prices_batch(self, token_ids: list[str]) -> None:
        """
        Internal: fetch prices for token_ids in batches of BATCH_SIZE.
        Populates self._price_cache.
        """
        for i in range(0, len(token_ids), self.BATCH_SIZE):
            batch = token_ids[i : i + self.BATCH_SIZE]
            try:
                await self._fetch_clob_prices(batch)
            except Exception as e:
                logger.warning(f"[BatchPrice] CLOB price fetch failed for batch {i//self.BATCH_SIZE}: {e}")
                # Fall back to Gamma API for this batch
                try:
                    await self._fetch_gamma_prices_fallback(batch)
                except Exception as e2:
                    logger.error(f"[BatchPrice] Gamma fallback also failed: {e2}")

    async def _fetch_clob_prices(self, token_ids: list[str]) -> None:
        """Fetch prices from CLOB /prices endpoint (fastest path)."""
        # CLOB /prices accepts comma-separated token IDs
        params = {"token_ids": ",".join(token_ids)}
        async with httpx.AsyncClient(timeout=10.0) as client:
            r = await client.get(f"{CLOB_API}/prices", params=params)
            if r.status_code != 200:
                raise RuntimeError(f"CLOB prices HTTP {r.status_code}")

            data = r.json()
            # Response format: {token_id: {"bid": float, "ask": float, "price": float}}
            now = time.time()
            for tid, info in data.items():
                if not isinstance(info, dict):
                    continue
                bid  = float(info.get("bid", 0.0))
                ask  = float(info.get("ask", bid))
                last = float(info.get("price", (bid + ask) / 2.0 if ask > 0 else 0.0))
                self._price_cache[tid] = PriceInfo(
                    token_id=tid,
                    best_bid=bid,
                    best_ask=ask,
                    last_price=last,
                    spread=ask - bid if ask > bid else 0.0,
                    volume_24h=0.0,  # not available from /prices
                    timestamp=now,
                )
            logger.debug(f"[BatchPrice] CLOB: fetched {len(data)} prices for {len(token_ids)} tokens")

    async def _fetch_gamma_prices_fallback(self, token_ids: list[str]) -> None:
        """
        Fallback: use Gamma /markets endpoint by market ID to get prices.
        Less efficient but always available.
        """
        # Gamma /prices endpoint by token IDs
        params = {"token_id": token_ids}  # Gamma accepts list
        async with httpx.AsyncClient(timeout=15.0) as client:
            r = await client.get(f"{GAMMA_API}/prices-history", params={"token_id": ",".join(token_ids), "interval": "1h", "fidelity": 1})
            # Actually use the simpler /markets?clob_token_ids= approach
            clob_query = ",".join(token_ids)
            r = await client.get(f"{GAMMA_API}/markets", params={"clob_token_ids": clob_query}, timeout=15.0)
            if r.status_code != 200:
                raise RuntimeError(f"Gamma fallback HTTP {r.status_code}")

            now = time.time()
            markets_data = r.json()
            if not isinstance(markets_data, list):
                markets_data = [markets_data]

            for market in markets_data:
                token_ids_json = market.get("clobTokenIds", "[]")
                prices_json    = market.get("outcomePrices", "[]")
                if isinstance(token_ids_json, str):
                    import json
                    token_ids_json = json.loads(token_ids_json)
                if isinstance(prices_json, str):
                    import json
                    prices_json = json.loads(prices_json)

                for tid, price_str in zip(token_ids_json, prices_json):
                    price = float(price_str)
                    if tid in token_ids:
                        self._price_cache[tid] = PriceInfo(
                            token_id=tid,
                            best_bid=max(0.0, price - 0.01),
                            best_ask=min(1.0, price + 0.01),
                            last_price=price,
                            spread=0.02,
                            volume_24h=float(market.get("volume24hr", 0.0)),
                            timestamp=now,
                        )

    # ------------------------------------------------------------------
    # 2. Batch market fetch (full metadata for scanner)
    # ------------------------------------------------------------------

    async def get_markets_batch(
        self,
        market_ids: list[str],
        force_refresh: bool = False,
    ) -> dict[str, MarketInfo]:
        """
        Return MarketInfo for each market_id.
        Includes outcome prices, spreads, volume, liquidity.
        """
        if not market_ids:
            return {}

        to_fetch = [
            mid for mid in market_ids
            if force_refresh or mid not in self._market_cache or self._market_cache[mid].is_stale
        ]

        if to_fetch:
            async with self._lock:
                to_fetch = [
                    mid for mid in to_fetch
                    if mid not in self._market_cache or self._market_cache[mid].is_stale
                ]
                if to_fetch:
                    await self._fetch_markets_batch(to_fetch)

        return {mid: self._market_cache[mid] for mid in market_ids if mid in self._market_cache}

    async def _fetch_markets_batch(self, market_ids: list[str]) -> None:
        """Fetch full market data from Gamma API in batches."""
        import json as _json

        for i in range(0, len(market_ids), self.BATCH_SIZE):
            batch = market_ids[i : i + self.BATCH_SIZE]
            try:
                ids_str = ",".join(batch)
                async with httpx.AsyncClient(timeout=15.0) as client:
                    r = await client.get(
                        f"{GAMMA_API}/markets",
                        params={"id": ids_str},
                        timeout=15.0
                    )
                    if r.status_code != 200:
                        logger.warning(f"[BatchPrice] Gamma markets HTTP {r.status_code}")
                        continue

                    now = time.time()
                    markets = r.json()
                    if not isinstance(markets, list):
                        markets = [markets]

                    for m in markets:
                        mid = m.get("condition_id") or m.get("conditionId", "")
                        if not mid:
                            continue

                        token_ids_raw = m.get("clobTokenIds", "[]")
                        prices_raw    = m.get("outcomePrices", "[]")
                        outcomes_raw  = m.get("outcomes", "[]")

                        for attr in [token_ids_raw, prices_raw, outcomes_raw]:
                            pass  # will parse below

                        if isinstance(token_ids_raw, str):
                            token_ids_parsed = _json.loads(token_ids_raw)
                        else:
                            token_ids_parsed = list(token_ids_raw)

                        if isinstance(prices_raw, str):
                            prices_parsed = _json.loads(prices_raw)
                        else:
                            prices_parsed = list(prices_raw)

                        outcome_prices = {}
                        best_asks = {}
                        best_bids = {}
                        for tid, price_str in zip(token_ids_parsed, prices_parsed):
                            p = float(price_str)
                            outcome_prices[tid] = p
                            best_asks[tid] = min(1.0, p + 0.01)
                            best_bids[tid] = max(0.0, p - 0.01)

                        spread = float(m.get("spread", 0.02))
                        self._market_cache[mid] = MarketInfo(
                            market_id=mid,
                            question=m.get("question", ""),
                            active=bool(m.get("active", True)),
                            closed=bool(m.get("closed", False)),
                            outcome_prices=outcome_prices,
                            best_asks=best_asks,
                            best_bids=best_bids,
                            volume_24h=float(m.get("volume24hr", 0.0)),
                            liquidity_usd=float(m.get("liquidityNum", 0.0)),
                            spread=spread,
                            timestamp=now,
                        )

                        # Also populate price cache from market data
                        for tid, p in outcome_prices.items():
                            self._price_cache[tid] = PriceInfo(
                                token_id=tid,
                                best_bid=max(0.0, p - spread / 2),
                                best_ask=min(1.0, p + spread / 2),
                                last_price=p,
                                spread=spread,
                                volume_24h=float(m.get("volume24hr", 0.0)),
                                timestamp=now,
                            )

                    logger.info(f"[BatchPrice] Fetched {len(markets)} market snapshots in batch {i//self.BATCH_SIZE}")

            except Exception as e:
                logger.error(f"[BatchPrice] Market batch fetch error: {e}")

    # ------------------------------------------------------------------
    # 3. Synchronous helper for portfolio_manager compatibility
    # ------------------------------------------------------------------

    def get_price_sync(self, token_id: str) -> Optional[float]:
        """
        Return cached price synchronously (used by portfolio_manager).
        Returns None if not in cache or stale.
        """
        info = self._price_cache.get(token_id)
        if info and not info.is_stale:
            return info.best_ask  # Return ask price for buy-side
        return None

    def inject_price(self, token_id: str, price: float, spread: float = 0.02) -> None:
        """
        Manually inject a price (used by portfolio_manager from CLOB book data).
        """
        self._price_cache[token_id] = PriceInfo(
            token_id=token_id,
            best_bid=max(0.0, price - spread / 2),
            best_ask=min(1.0, price + spread / 2),
            last_price=price,
            spread=spread,
            volume_24h=0.0,
            timestamp=time.time(),
        )

    def get_cache_stats(self) -> dict:
        return {
            "total_cached": len(self._price_cache),
            "fresh_prices": self._cache_hit_count(),
            "markets_cached": len(self._market_cache),
        }


# Singleton
batch_price_fetcher = BatchPriceFetcher()
