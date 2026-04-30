from __future__ import annotations

import argparse
import asyncio
import csv
from datetime import date, datetime
from pathlib import Path

from src.backtest.market_history import fetch_gamma_weather_markets, load_markets_file
from src.backtest.metrics import summarize_predictions
from src.backtest.models import BacktestMarket, PredictionRecord
from src.backtest.report import write_markdown_report
from src.backtest.replay_engine import BacktestReplayEngine
from src.backtest.storage import BacktestStore
from src.backtest.weather_archive import WeatherArchive, snapshot_times
from src.utils import logger


def _parse_date(value: str) -> date:
    return datetime.strptime(value, "%Y-%m-%d").date()


def _parse_leads(value: str) -> list[int]:
    return [int(part.strip()) for part in value.split(",") if part.strip()]


def _write_predictions_csv(path: Path, rows: list[PredictionRecord]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(
            fh,
            fieldnames=[
                "run_id",
                "market_id",
                "city",
                "icao_code",
                "question",
                "target_date",
                "as_of",
                "outcome_name",
                "token_id",
                "predicted_prob",
                "bucket_probability_yes",
                "actual_outcome",
                "brier",
                "family_market_divergence",
                "reasoning",
            ],
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "run_id": row.run_id,
                    "market_id": row.market_id,
                    "city": row.city,
                    "icao_code": row.icao_code,
                    "question": row.question,
                    "target_date": row.target_date.isoformat(),
                    "as_of": row.as_of.isoformat(),
                    "outcome_name": row.outcome_name,
                    "token_id": row.token_id,
                    "predicted_prob": row.predicted_prob,
                    "bucket_probability_yes": row.bucket_probability_yes,
                    "actual_outcome": row.actual_outcome,
                    "brier": row.brier,
                    "family_market_divergence": row.family_market_divergence,
                    "reasoning": row.reasoning,
                }
            )


async def _load_markets(args) -> list[BacktestMarket]:
    if args.markets_file:
        markets = load_markets_file(args.markets_file)
    else:
        markets = await fetch_gamma_weather_markets(args.start, args.end)

    markets = [
        market
        for market in markets
        if args.start <= market.resolution_date <= args.end
        and (not args.city or market.city.lower() == args.city.lower())
        and (not args.icao or market.icao_code.upper() == args.icao.upper())
    ]
    if args.max_markets:
        markets = markets[: args.max_markets]
    return markets


async def run_backtest(args) -> list[PredictionRecord]:
    markets = await _load_markets(args)
    logger.info(f"[BACKTEST] Loaded {len(markets)} markets.")

    archive = WeatherArchive()
    engine = BacktestReplayEngine(args.run_id)
    rows: list[PredictionRecord] = []

    try:
        for index, market in enumerate(markets, start=1):
            if market.actual_high_c is None or market.actual_low_c is None:
                actuals = await archive.fetch_actual_daily(market.icao_code, market.resolution_date)
                market.actual_high_c = actuals.get("actual_high_c")
                market.actual_low_c = actuals.get("actual_low_c")

            for as_of in snapshot_times(market.resolution_date, args.leads):
                snapshot = await archive.fetch_forecast_snapshot(market.icao_code, market.resolution_date, as_of)
                rows.extend(engine.replay_snapshot(market, snapshot))

            if index % 25 == 0:
                logger.info(f"[BACKTEST] Processed {index}/{len(markets)} markets.")
    finally:
        await archive.close()

    return rows


async def async_main() -> None:
    parser = argparse.ArgumentParser(description="Run offline TSAS probability backtest.")
    parser.add_argument("--from", dest="start", required=True, type=_parse_date, help="Start date YYYY-MM-DD")
    parser.add_argument("--to", dest="end", required=True, type=_parse_date, help="End date YYYY-MM-DD")
    parser.add_argument("--markets-file", help="Optional JSON/JSONL market file. Prefer this for reproducible runs.")
    parser.add_argument("--city", help="Optional city filter")
    parser.add_argument("--icao", help="Optional ICAO filter")
    parser.add_argument("--leads", default="48,24,12,6,2", type=_parse_leads, help="Comma-separated lead hours")
    parser.add_argument("--max-markets", type=int, default=0, help="Limit markets for smoke tests")
    parser.add_argument("--run-id", default=None, help="Run id. Defaults to timestamp.")
    parser.add_argument("--store", default="data/backtest/backtest_runs.sqlite")
    parser.add_argument("--report", default=None)
    parser.add_argument("--csv", default=None)
    args = parser.parse_args()

    if args.run_id is None:
        args.run_id = f"bt_{args.start}_{args.end}_{datetime.utcnow().strftime('%Y%m%d%H%M%S')}"

    rows = await run_backtest(args)

    store = BacktestStore(args.store)
    try:
        store.write_predictions(rows)
    finally:
        store.close()

    report_path = Path(args.report or f"reports/backtest_{args.run_id}.md")
    csv_path = Path(args.csv or f"data/backtest/predictions_{args.run_id}.csv")
    _write_predictions_csv(csv_path, rows)
    write_markdown_report(
        report_path,
        args.run_id,
        rows,
        notes=[
            "MVP probability backtest: evaluates TSAS probabilities against observed station temperatures.",
            "Execution PnL and L2 order book replay are intentionally out of scope for this first version.",
            "Forecast snapshots use Open-Meteo Previous Runs with Historical Forecast fallback.",
        ],
    )

    summary = summarize_predictions(rows)
    logger.info(
        "[BACKTEST] Complete | predictions={} resolved={} brier={} log_loss={} report={} csv={}".format(
            len(rows),
            summary.count,
            "N/A" if summary.brier is None else f"{summary.brier:.4f}",
            "N/A" if summary.log_loss is None else f"{summary.log_loss:.4f}",
            report_path,
            csv_path,
        )
    )


def main() -> None:
    asyncio.run(async_main())


if __name__ == "__main__":
    main()

