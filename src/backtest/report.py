"""
report.py -- Phase 5: Extended Backtest Report
===============================================
Generates a full Markdown report with:
  - Probability calibration metrics (Brier, Log Loss, CRPS)
  - Trading performance metrics (Sharpe, PF, MDD, Win Rate, PnL)
  - Per-city and per-ICAO breakdowns
  - Calibration curve table
  - Model quality assessment
"""

from __future__ import annotations

import math
from datetime import datetime
from pathlib import Path
from typing import Iterable

from src.backtest.metrics import (
    MetricSummary,
    TradeSimResult,
    calibration_bins,
    group_summaries,
    phase5_trade_simulation,
    summarize_predictions,
)
from src.backtest.models import PredictionRecord


def _fmt(value, digits: int = 3, suffix: str = "") -> str:
    if value is None:
        return "N/A"
    if math.isinf(value):
        return "∞" if value > 0 else "-∞"
    return f"{value:.{digits}f}{suffix}"


def _grade(brier: Optional[float]) -> str:
    """Quality grade based on Brier score."""
    if brier is None:
        return "?"
    if brier < 0.08:  return "🏆 Excellent"
    if brier < 0.12:  return "✅ Good"
    if brier < 0.18:  return "⚠️ Acceptable"
    return "❌ Poor"


from typing import Optional


def _trading_grade(sharpe: Optional[float]) -> str:
    if sharpe is None: return "?"
    if sharpe >= 2.0:  return "🏆 Excellent"
    if sharpe >= 1.5:  return "✅ Good"
    if sharpe >= 1.0:  return "⚠️ Marginal"
    return "❌ Poor"


def write_markdown_report(
    path: str | Path,
    run_id: str,
    rows: Iterable[PredictionRecord],
    notes: list[str] | None = None,
    bankroll: float = 1000.0,
    kelly_fraction: float = 0.10,
) -> Path:
    """
    Write a full Phase 5 Markdown report to 'path'.
    Returns the Path of the written file.
    """
    rows = list(rows)
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)

    overall = summarize_predictions(rows)
    by_city = group_summaries(rows, "city")
    by_icao = group_summaries(rows, "icao_code")
    cal      = calibration_bins(rows)
    sim      = phase5_trade_simulation(rows, bankroll=bankroll, kelly_fraction=kelly_fraction)
    generated_at = datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC")

    lines: list[str] = [
        f"# 📊 Backtest Report: `{run_id}`",
        "",
        f"*Generated: {generated_at}*",
        "",
        "---",
        "",
        "## 🎯 Probability Calibration",
        "",
        f"| Metric | Value | Target | Grade |",
        f"|---|---|---|---|",
        f"| Predictions | `{len(rows)}` | — | — |",
        f"| Resolved | `{overall.count}` | — | — |",
        f"| **Brier Score** | `{_fmt(overall.brier, 4)}` | `< 0.15` | {_grade(overall.brier)} |",
        f"| Log Loss | `{_fmt(overall.log_loss, 4)}` | `< 0.40` | — |",
        f"| CRPS | `{_fmt(overall.crps, 4)}` | minimise | — |",
        f"| Mean Probability | `{_fmt(overall.mean_probability, 3)}` | — | — |",
        f"| Hit Rate | `{_fmt(overall.hit_rate, 3)}` | — | — |",
        "",
        "---",
        "",
        "## 💰 Trading Simulation",
        f"*(bankroll=${bankroll:.0f}, Kelly fraction={kelly_fraction:.0%}, min edge=20%)*",
        "",
        f"| Metric | Value | Target | Grade |",
        f"|---|---|---|---|",
        f"| Total Trades | `{sim.n_trades}` | — | — |",
        f"| Wins / Losses | `{sim.n_wins}W / {sim.n_losses}L` | — | — |",
        f"| **Win Rate** | `{_fmt(sim.win_rate * 100, 1, '%')}` | `> 55%` | {'✅' if sim.win_rate > 0.55 else '❌'} |",
        f"| **Profit Factor** | `{_fmt(sim.profit_factor, 2)}` | `> 2.0` | {'✅' if sim.profit_factor > 2.0 else '❌'} |",
        f"| **Sharpe Ratio** | `{_fmt(sim.sharpe, 2)}` | `> 1.5` | {_trading_grade(sim.sharpe)} |",
        f"| **Max Drawdown** | `{_fmt(sim.max_drawdown_pct, 1, '%')}` | `< 15%` | {'✅' if sim.max_drawdown_pct < 15 else '❌'} |",
        f"| Total PnL | `${_fmt(sim.total_pnl, 2)}` | `> 0` | {'✅' if sim.total_pnl > 0 else '❌'} |",
        f"| Avg PnL / Trade | `${_fmt(sim.avg_pnl_per_trade, 2)}` | `> 0` | — |",
        f"| Gross Profit | `${_fmt(sim.gross_profit, 2)}` | — | — |",
        f"| Gross Loss | `${_fmt(sim.gross_loss, 2)}` | — | — |",
        "",
    ]

    if notes:
        lines.extend(["---", "", "## 📝 Notes", ""])
        lines.extend(f"- {note}" for note in notes)
        lines.append("")

    # Per-city breakdown
    lines.extend([
        "---",
        "",
        "## 🏙️ By City",
        "",
        "| City | N | Brier | CRPS | Win Rate | Sharpe | Total PnL |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ])
    for city, s in by_city.items():
        lines.append(
            f"| {city} | {s.count} | {_fmt(s.brier, 4)} | {_fmt(s.crps, 4)} | "
            f"{_fmt(s.win_rate * 100 if s.win_rate else None, 1, '%')} | "
            f"{_fmt(s.sharpe, 2)} | ${_fmt(s.total_pnl, 2)} |"
        )

    # Per-ICAO breakdown
    lines.extend([
        "",
        "---",
        "",
        "## ✈️ By ICAO",
        "",
        "| ICAO | N | Brier | Hit Rate | Win Rate | PnL |",
        "|---|---:|---:|---:|---:|---:|",
    ])
    for icao, s in by_icao.items():
        lines.append(
            f"| {icao} | {s.count} | {_fmt(s.brier, 4)} | {_fmt(s.hit_rate, 3)} | "
            f"{_fmt(s.win_rate * 100 if s.win_rate else None, 1, '%')} | "
            f"${_fmt(s.total_pnl, 2)} |"
        )

    # Calibration curve
    lines.extend([
        "",
        "---",
        "",
        "## 📈 Calibration Curve",
        "",
        "| Prob Bin | N | Mean Predicted P | Actual Hit Rate | ECE Gap |",
        "|---|---:|---:|---:|---:|",
    ])
    for bucket in cal:
        if bucket["n"] == 0:
            continue
        mean_p = bucket["mean_probability"]
        hit    = bucket["hit_rate"]
        ece_gap = abs(mean_p - hit) if (mean_p is not None and hit is not None) else None
        label = f"{bucket['low']:.1f}–{bucket['high']:.1f}"
        lines.append(
            f"| {label} | {bucket['n']} | {_fmt(mean_p, 3)} | "
            f"{_fmt(hit, 3)} | {_fmt(ece_gap, 3)} |"
        )

    # ECE (Expected Calibration Error)
    ece_vals = []
    for bucket in cal:
        n = bucket["n"]
        if n == 0 or bucket["mean_probability"] is None or bucket["hit_rate"] is None:
            continue
        ece_vals.append(n * abs(bucket["mean_probability"] - bucket["hit_rate"]))
    ece = sum(ece_vals) / overall.count if (overall.count > 0 and ece_vals) else None

    lines.extend([
        "",
        f"**Expected Calibration Error (ECE):** `{_fmt(ece, 4)}`",
        *(["*(ECE < 0.05 = well calibrated)*"] if ece is not None else []),
        "",
        "---",
        "",
        f"*Report generated by Polymarket Weather Bot v2 Phase 5*",
    ])

    output.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return output
