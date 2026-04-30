from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path
from typing import Iterable, Optional

from src.backtest.metrics import MetricSummary, calibration_bins, summarize_predictions
from src.backtest.models import PredictionRecord
from src.probability_calculator import parse_temperature_bin


@dataclass
class ParsedRow:
    record: PredictionRecord
    low: float
    high: float
    unit: str
    is_open: bool
    center: float


def _parse_date(value: str) -> date:
    return datetime.strptime(value, "%Y-%m-%d").date()


def _load_rows(path: str | Path) -> list[ParsedRow]:
    rows: list[ParsedRow] = []
    with Path(path).open("r", encoding="utf-8", newline="") as fh:
        for raw in csv.DictReader(fh):
            parsed = parse_temperature_bin(raw["question"])
            if parsed is None:
                continue
            low, high, unit = parsed
            is_open = low <= -999 or high >= 999
            center = 0.0 if is_open else (low + high) / 2.0
            rows.append(
                ParsedRow(
                    record=PredictionRecord(
                        run_id=raw["run_id"],
                        market_id=raw["market_id"],
                        city=raw["city"],
                        icao_code=raw["icao_code"],
                        question=raw["question"],
                        target_date=_parse_date(raw["target_date"]),
                        as_of=datetime.fromisoformat(raw["as_of"]),
                        outcome_name=raw["outcome_name"],
                        token_id=raw["token_id"],
                        predicted_prob=float(raw["predicted_prob"]),
                        bucket_probability_yes=float(raw["bucket_probability_yes"]),
                        actual_outcome=None if raw["actual_outcome"] in ("", "None") else raw["actual_outcome"] == "True",
                        brier=None if raw["brier"] in ("", "None") else float(raw["brier"]),
                        reasoning=raw.get("reasoning", ""),
                        family_market_divergence=float(raw.get("family_market_divergence") or 0.0),
                    ),
                    low=low,
                    high=high,
                    unit=unit,
                    is_open=is_open,
                    center=center,
                )
            )
    return rows


def _nearest_focus_questions(rows: Iterable[ParsedRow], neighbor_buckets: int = 1) -> tuple[set[str], dict[tuple[str, date], str]]:
    by_family: dict[tuple[str, date], list[ParsedRow]] = defaultdict(list)
    for row in rows:
        by_family[(row.record.city, row.record.target_date)].append(row)

    selected_questions: set[str] = set()
    realized_labels: dict[tuple[str, date], str] = {}

    for family_key, family_rows in by_family.items():
        yes_true = [
            row for row in family_rows
            if row.record.outcome_name.lower() == "yes"
            and row.record.actual_outcome is True
            and not row.is_open
        ]
        if not yes_true:
            continue

        realized = yes_true[0]
        bounded_questions = sorted(
            {row.record.question: row for row in family_rows if not row.is_open and row.record.outcome_name.lower() == "yes"}.values(),
            key=lambda row: (row.low, row.high),
        )
        question_order = [row.record.question for row in bounded_questions]
        try:
            idx = question_order.index(realized.record.question)
        except ValueError:
            continue

        lo = max(0, idx - neighbor_buckets)
        hi = min(len(question_order), idx + neighbor_buckets + 1)
        for question in question_order[lo:hi]:
            selected_questions.add(question)
        realized_labels[family_key] = realized.record.question

    return selected_questions, realized_labels


def _filter_rows(rows: Iterable[ParsedRow], allowed_questions: set[str]) -> list[PredictionRecord]:
    return [row.record for row in rows if row.record.question in allowed_questions]


def _group_summary(rows: Iterable[PredictionRecord], attr: str) -> dict[str, MetricSummary]:
    groups: dict[str, list[PredictionRecord]] = defaultdict(list)
    for row in rows:
        groups[str(getattr(row, attr))].append(row)
    return {key: summarize_predictions(value) for key, value in sorted(groups.items())}


def _lead_hours(row: PredictionRecord) -> int:
    target_dt = datetime.combine(row.target_date, datetime.min.time(), tzinfo=row.as_of.tzinfo).replace(hour=12)
    return round((target_dt - row.as_of).total_seconds() / 3600.0)


def _lead_summary(rows: Iterable[PredictionRecord]) -> dict[int, MetricSummary]:
    groups: dict[int, list[PredictionRecord]] = defaultdict(list)
    for row in rows:
        groups[_lead_hours(row)].append(row)
    return {key: summarize_predictions(value) for key, value in sorted(groups.items())}


def _fmt(value: Optional[float], digits: int = 3) -> str:
    if value is None:
        return "N/A"
    return f"{value:.{digits}f}"


def write_focused_report(csv_path: str | Path, out_path: str | Path, neighbor_buckets: int = 1) -> Path:
    parsed_rows = _load_rows(csv_path)
    selected_questions, realized_labels = _nearest_focus_questions(parsed_rows, neighbor_buckets=neighbor_buckets)
    focused_rows = _filter_rows(parsed_rows, selected_questions)

    overall = summarize_predictions(focused_rows)
    by_city = _group_summary(focused_rows, "city")
    by_lead = _lead_summary(focused_rows)
    cal = calibration_bins(focused_rows)

    output = Path(out_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    lines = [
        f"# Focused Near-Threshold Report: `{Path(csv_path).stem}`",
        "",
        "## Scope",
        "",
        f"- Families analyzed: `{len(realized_labels)}`",
        f"- Neighbor buckets on each side: `{neighbor_buckets}`",
        f"- Focused questions selected: `{len(selected_questions)}`",
        f"- Focused predictions: `{len(focused_rows)}`",
        "",
        "## Summary",
        "",
        f"- Resolved predictions: `{overall.count}`",
        f"- Brier score: `{_fmt(overall.brier)}`",
        f"- Log loss: `{_fmt(overall.log_loss)}`",
        f"- Mean probability: `{_fmt(overall.mean_probability)}`",
        f"- Hit rate: `{_fmt(overall.hit_rate)}`",
        "",
        "## By City",
        "",
        "| City | N | Brier | Log Loss | Mean P | Hit Rate |",
        "|---|---:|---:|---:|---:|---:|",
    ]

    for city, summary in by_city.items():
        lines.append(
            f"| {city} | {summary.count} | {_fmt(summary.brier)} | {_fmt(summary.log_loss)} | "
            f"{_fmt(summary.mean_probability)} | {_fmt(summary.hit_rate)} |"
        )

    lines.extend([
        "",
        "## By Lead",
        "",
        "| Lead | N | Brier | Log Loss | Mean P | Hit Rate |",
        "|---|---:|---:|---:|---:|---:|",
    ])
    for lead, summary in by_lead.items():
        lines.append(
            f"| {lead}h | {summary.count} | {_fmt(summary.brier)} | {_fmt(summary.log_loss)} | "
            f"{_fmt(summary.mean_probability)} | {_fmt(summary.hit_rate)} |"
        )

    lines.extend(["", "## Calibration", "", "| Bin | N | Mean P | Hit Rate |", "|---|---:|---:|---:|"])
    for bucket in cal:
        label = f"{bucket['low']:.1f}-{bucket['high']:.1f}"
        lines.append(f"| {label} | {bucket['n']} | {_fmt(bucket['mean_probability'])} | {_fmt(bucket['hit_rate'])} |")

    lines.extend(["", "## Realized Buckets", ""])
    for (city, target_date), question in sorted(realized_labels.items()):
        lines.append(f"- {city} {target_date.isoformat()}: `{question}`")

    output.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description="Write focused near-threshold backtest report.")
    parser.add_argument("--csv", required=True, help="Predictions CSV produced by src.backtest.run")
    parser.add_argument("--out", required=True, help="Markdown output path")
    parser.add_argument("--neighbor-buckets", type=int, default=1, help="How many adjacent buckets to include on each side")
    args = parser.parse_args()
    write_focused_report(args.csv, args.out, neighbor_buckets=args.neighbor_buckets)


if __name__ == "__main__":
    main()
