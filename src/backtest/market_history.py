from __future__ import annotations

import json
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Optional

import httpx

from config.settings import config
from src.backtest.models import BacktestMarket
from src.market_discovery import MarketDiscoverer
from src.utils import logger


def _parse_date(value: Any) -> Optional[date]:
    if not value:
        return None
    if isinstance(value, date) and not isinstance(value, datetime):
        return value
    try:
        return datetime.fromisoformat(str(value).replace("Z", "+00:00")).date()
    except Exception:
        pass
    try:
        return datetime.strptime(str(value), "%Y-%m-%d").date()
    except Exception:
        return None


def _iter_records(path: str | Path) -> Iterable[dict[str, Any]]:
    p = Path(path)
    text = p.read_text(encoding="utf-8")
    if p.suffix.lower() == ".jsonl":
        for line in text.splitlines():
            if line.strip():
                yield json.loads(line)
        return

    payload = json.loads(text)
    if isinstance(payload, list):
        yield from payload
    elif isinstance(payload, dict):
        for key in ("markets", "data", "results"):
            if isinstance(payload.get(key), list):
                yield from payload[key]
                return
        yield payload


def _from_market_dict(record: dict[str, Any]) -> Optional[BacktestMarket]:
    resolution = _parse_date(
        record.get("resolution_date")
        or record.get("end_date_iso")
        or record.get("endDateIso")
        or record.get("endDate")
        or record.get("end_date")
    )
    city = record.get("city")
    icao = record.get("icao_code") or (config.city_icao_mapping.get(city) if city else None)
    market_id = record.get("market_id") or record.get("conditionId") or record.get("id")
    question = record.get("question")
    outcomes = record.get("outcomes") or []
    if isinstance(outcomes, str):
        try:
            outcomes = json.loads(outcomes)
        except Exception:
            outcomes = []

    if not (resolution and city and icao and market_id and question and outcomes):
        return None

    return BacktestMarket(
        market_id=str(market_id),
        question=str(question),
        city=str(city),
        icao_code=str(icao),
        resolution_date=resolution,
        outcomes=list(outcomes),
        event_title=str(record.get("event_title") or record.get("title") or ""),
        best_ask=_maybe_float(record.get("best_ask") or record.get("bestAsk")),
        best_bid=_maybe_float(record.get("best_bid") or record.get("bestBid")),
        spread=_maybe_float(record.get("spread")),
        midpoint=_maybe_float(record.get("midpoint")),
        daily_volume=_maybe_float(record.get("daily_volume") or record.get("volume")),
        actual_high_c=_maybe_float(record.get("actual_high_c")),
        actual_low_c=_maybe_float(record.get("actual_low_c")),
        metadata=record,
    )


def _maybe_float(value: Any) -> Optional[float]:
    if value in (None, "", "null"):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def load_markets_file(path: str | Path) -> list[BacktestMarket]:
    markets = []
    for record in _iter_records(path):
        market = _from_market_dict(record)
        if market:
            markets.append(market)
    return markets


async def fetch_gamma_weather_markets(start: date, end: date) -> list[BacktestMarket]:
    """Best-effort Gamma loader for historical weather markets.

    Gamma's historical schema changes more often than the active endpoint, so
    this loader is intentionally permissive. For reproducible backtests,
    prefer saving its output to JSONL and reusing that file.
    """
    discoverer = MarketDiscoverer()
    target_cities = list(config.city_icao_mapping.keys())
    parsed: list[BacktestMarket] = []
    limit = 100
    offset = 0

    async with httpx.AsyncClient() as client:
        while True:
            url = (
                f"{discoverer.gamma_api_url}/events"
                f"?limit={limit}&offset={offset}&tag_slug=weather&closed=true&active=false"
            )
            try:
                response = await client.get(url, timeout=20.0)
                response.raise_for_status()
                events = response.json()
            except Exception as exc:
                logger.warning(f"[BACKTEST] Gamma history fetch failed at offset {offset}: {exc}")
                break

            if not events:
                break

            for event in events:
                title = event.get("title", "")
                if "temperature" not in title.lower():
                    continue
                city = next((name for name in target_cities if name.lower() in title.lower()), None)
                if not city:
                    continue
                for raw_market in event.get("markets", []) or []:
                    market_info = discoverer._parse_market(raw_market, city, title)
                    if not market_info:
                        continue
                    market = _from_market_dict(market_info)
                    if not market or not (start <= market.resolution_date <= end):
                        continue
                    market.metadata["gamma_event"] = event
                    parsed.append(market)

            if len(events) < limit:
                break
            offset += limit

    return parsed

