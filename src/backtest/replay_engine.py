from __future__ import annotations

from datetime import datetime
from typing import Iterable, Optional

from src.backtest.models import BacktestMarket, ForecastSnapshot, PredictionRecord
from src.probability_calculator import parse_temperature_bin
from src.tsas_model import analyze_city_tsas


def _actual_yes(market: BacktestMarket) -> Optional[bool]:
    parsed = parse_temperature_bin(market.question)
    if parsed is None:
        return None
    low, high, unit = parsed
    is_low_market = "lowest" in market.question.lower() or "minimum" in market.question.lower()
    actual_c = market.actual_low_c if is_low_market else market.actual_high_c
    if actual_c is None:
        return None
    actual = actual_c * 9.0 / 5.0 + 32.0 if unit == "F" else actual_c
    return low <= actual < high


class BacktestReplayEngine:
    def __init__(self, run_id: str):
        self.run_id = run_id

    def replay_snapshot(self, market: BacktestMarket, snapshot: ForecastSnapshot) -> list[PredictionRecord]:
        tsas_market = market.to_tsas_market()
        tsas_market["_backtest_now"] = snapshot.as_of.isoformat()
        signals = analyze_city_tsas(
            market.city,
            [tsas_market],
            snapshot.weather_data,
            return_all=True,
            monitor_mode=True,
        )
        actual_yes = _actual_yes(market)
        rows: list[PredictionRecord] = []

        for signal in signals:
            outcome_name = str(signal.get("outcome_name") or "")
            if actual_yes is None:
                actual_outcome = None
            elif outcome_name.lower() == "yes":
                actual_outcome = actual_yes
            else:
                actual_outcome = not actual_yes

            probability = float(signal.get("predicted_prob") or 0.0)
            brier = None if actual_outcome is None else (probability - (1.0 if actual_outcome else 0.0)) ** 2
            rows.append(
                PredictionRecord(
                    run_id=self.run_id,
                    market_id=market.market_id,
                    city=market.city,
                    icao_code=market.icao_code,
                    question=market.question,
                    target_date=market.resolution_date,
                    as_of=snapshot.as_of,
                    outcome_name=outcome_name,
                    token_id=str(signal.get("token_id") or ""),
                    predicted_prob=probability,
                    bucket_probability_yes=float(signal.get("bucket_probability_yes") or 0.0),
                    actual_outcome=actual_outcome,
                    brier=brier,
                    reasoning=str(signal.get("reasoning") or ""),
                    family_id=str(signal.get("family_id") or ""),
                    family_distribution=str(signal.get("family_distribution") or ""),
                    family_market_distribution=str(signal.get("family_market_distribution") or ""),
                    family_market_divergence=float(signal.get("family_market_divergence") or 0.0),
                )
            )
        return rows
