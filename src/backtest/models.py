from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime
from typing import Any, Optional


@dataclass
class BacktestMarket:
    market_id: str
    question: str
    city: str
    icao_code: str
    resolution_date: date
    outcomes: list[dict[str, Any]]
    event_title: str = ""
    best_ask: Optional[float] = None
    best_bid: Optional[float] = None
    spread: Optional[float] = None
    midpoint: Optional[float] = None
    daily_volume: Optional[float] = None
    actual_high_c: Optional[float] = None
    actual_low_c: Optional[float] = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_tsas_market(self) -> dict[str, Any]:
        return {
            "market_id": self.market_id,
            "question": self.question,
            "event_title": self.event_title,
            "city": self.city,
            "icao_code": self.icao_code,
            "resolution_date": f"{self.resolution_date.isoformat()}T12:00:00+00:00",
            "best_ask": self.best_ask,
            "best_bid": self.best_bid,
            "spread": self.spread,
            "midpoint": self.midpoint,
            "daily_volume": self.daily_volume,
            "outcomes": self.outcomes,
        }


@dataclass
class ForecastSnapshot:
    icao_code: str
    target_date: date
    as_of: datetime
    weather_data: dict[str, Any]
    source: str


@dataclass
class PredictionRecord:
    run_id: str
    market_id: str
    city: str
    icao_code: str
    question: str
    target_date: date
    as_of: datetime
    outcome_name: str
    token_id: str
    predicted_prob: float
    bucket_probability_yes: float
    actual_outcome: Optional[bool]
    brier: Optional[float]
    reasoning: str = ""
    family_id: str = ""
    family_distribution: str = ""
    family_market_distribution: str = ""
    family_market_divergence: float = 0.0
