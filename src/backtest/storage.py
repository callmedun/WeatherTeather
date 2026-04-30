from __future__ import annotations

import json
import sqlite3
from dataclasses import asdict, is_dataclass
from datetime import date, datetime
from pathlib import Path
from typing import Iterable

from src.backtest.models import PredictionRecord


def _json_default(value):
    if isinstance(value, (datetime, date)):
        return value.isoformat()
    if is_dataclass(value):
        return asdict(value)
    return str(value)


class BacktestStore:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(self.path)
        self.conn.execute("PRAGMA journal_mode=WAL")
        self._init_schema()

    def close(self) -> None:
        self.conn.close()

    def _init_schema(self) -> None:
        self.conn.execute(
            """
            CREATE TABLE IF NOT EXISTS predictions (
                run_id TEXT NOT NULL,
                market_id TEXT NOT NULL,
                city TEXT,
                icao_code TEXT,
                question TEXT,
                target_date TEXT,
                as_of TEXT,
                outcome_name TEXT,
                token_id TEXT,
                predicted_prob REAL,
                bucket_probability_yes REAL,
                actual_outcome INTEGER,
                brier REAL,
                reasoning TEXT,
                family_id TEXT,
                family_distribution TEXT,
                family_market_distribution TEXT,
                family_market_divergence REAL,
                PRIMARY KEY (run_id, market_id, token_id, as_of)
            )
            """
        )
        self.conn.commit()

    def write_predictions(self, rows: Iterable[PredictionRecord]) -> None:
        payload = []
        for row in rows:
            payload.append(
                (
                    row.run_id,
                    row.market_id,
                    row.city,
                    row.icao_code,
                    row.question,
                    row.target_date.isoformat(),
                    row.as_of.isoformat(),
                    row.outcome_name,
                    row.token_id,
                    row.predicted_prob,
                    row.bucket_probability_yes,
                    None if row.actual_outcome is None else int(row.actual_outcome),
                    row.brier,
                    row.reasoning,
                    row.family_id,
                    row.family_distribution,
                    row.family_market_distribution,
                    row.family_market_divergence,
                )
            )

        self.conn.executemany(
            """
            INSERT OR REPLACE INTO predictions VALUES (
                ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?
            )
            """,
            payload,
        )
        self.conn.commit()

    def write_jsonl(self, path: str | Path, rows: Iterable[object]) -> None:
        output = Path(path)
        output.parent.mkdir(parents=True, exist_ok=True)
        with output.open("w", encoding="utf-8") as fh:
            for row in rows:
                fh.write(json.dumps(row, default=_json_default, ensure_ascii=False) + "\n")

