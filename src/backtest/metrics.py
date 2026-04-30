"""
metrics.py -- Phase 5: Extended Backtest Metrics
=================================================
Extends the original summarize_predictions() with full trading metrics:
  - Brier Score (original)
  - Log Loss (original)
  - Calibration bins (original)
  - NEW: CRPS (Continuous Ranked Probability Score)
  - NEW: Sharpe Ratio (risk-adjusted returns)
  - NEW: Profit Factor (gross profit / gross loss)
  - NEW: Max Drawdown (peak-to-trough)
  - NEW: Win Rate per trade
  - NEW: Trade simulation (entry/exit PnL from Phase 3 rules)
"""

from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Iterable, Optional

from src.backtest.models import PredictionRecord


# ---------------------------------------------------------------------------
# Original metric summary (kept for backward compatibility)
# ---------------------------------------------------------------------------

@dataclass
class MetricSummary:
    count: int
    brier: Optional[float]
    log_loss: Optional[float]
    mean_probability: Optional[float]
    hit_rate: Optional[float]

    # Phase 5 additions
    crps: Optional[float] = None
    sharpe: Optional[float] = None
    profit_factor: Optional[float] = None
    max_drawdown_pct: Optional[float] = None
    win_rate: Optional[float] = None
    total_pnl: Optional[float] = None
    avg_pnl_per_trade: Optional[float] = None
    n_trades: int = 0
    n_wins: int = 0
    n_losses: int = 0


# ---------------------------------------------------------------------------
# Phase 5: Extended trading simulation dataclass
# ---------------------------------------------------------------------------

@dataclass
class TradeSimResult:
    """Result of simulating entry/exit on a set of prediction records."""
    n_trades: int
    n_wins: int
    n_losses: int
    gross_profit: float
    gross_loss: float
    total_pnl: float
    max_drawdown_pct: float
    sharpe: float
    profit_factor: float
    win_rate: float
    avg_pnl_per_trade: float
    pnl_curve: list[float] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def _log_loss(probability: float, actual: bool) -> float:
    p = min(0.999, max(0.001, probability))
    return -(math.log(p) if actual else math.log(1.0 - p))


def _crps_bernoulli(probability: float, actual: bool) -> float:
    """
    CRPS for a Bernoulli outcome (binary market).
    CRPS = (p - 1)^2 if actual else p^2, equivalent to Brier score for binary.
    For continuous approximation: CRPS ≈ Brier for binary outcomes.
    """
    outcome = 1.0 if actual else 0.0
    return (probability - outcome) ** 2


def _sharpe(returns: list[float], risk_free: float = 0.0) -> float:
    """Annualized Sharpe ratio from a list of trade returns (as fractions)."""
    if len(returns) < 2:
        return 0.0
    n = len(returns)
    mean = sum(returns) / n
    variance = sum((r - mean) ** 2 for r in returns) / (n - 1)
    std = math.sqrt(variance) if variance > 0 else 1e-9
    # Annualize assuming ~250 trades/year
    return (mean - risk_free) / std * math.sqrt(min(250, n))


def _max_drawdown(pnl_curve: list[float]) -> float:
    """
    Maximum peak-to-trough drawdown as a percentage of peak equity.
    pnl_curve: cumulative PnL values (not returns).
    Returns drawdown as a positive fraction (e.g. 0.15 = -15%).
    """
    if not pnl_curve:
        return 0.0
    peak = pnl_curve[0]
    max_dd = 0.0
    for val in pnl_curve:
        if val > peak:
            peak = val
        if peak > 0:
            dd = (peak - val) / peak
            if dd > max_dd:
                max_dd = dd
    return max_dd


def _simulate_trades(
    rows: Iterable[PredictionRecord],
    bankroll: float = 1000.0,
    kelly_fraction: float = 0.10,
    min_edge_pct: float = 20.0,
    min_ev: float = 0.05,
) -> TradeSimResult:
    """
    Simulate fractional Kelly trades from resolved prediction records.

    For each YES outcome with sufficient edge, simulate:
      - Entry at market_price (from record or estimated)
      - Exit at 1.0 if won, 0.0 if lost
      - Size = fractional Kelly

    Only trades YES outcomes to avoid double-counting markets.
    """
    pnl_list: list[float] = []
    equity = bankroll
    equity_curve: list[float] = [bankroll]
    gross_profit = 0.0
    gross_loss = 0.0

    for row in rows:
        # Only trade YES outcomes to avoid counting both YES and NO for same market
        if str(row.outcome_name).lower() != "yes":
            continue
        if row.actual_outcome is None:
            continue

        prob = float(row.predicted_prob)
        # Estimate market price from predicted_prob (in backtest we use market midpoint when available)
        market_price = getattr(row, "market_price", None) or prob * 0.85  # Conservative: assume 15% worse pricing

        edge = prob - market_price
        if edge < (min_edge_pct / 100.0):
            continue

        ev = prob * (1 - market_price) - (1 - prob) * market_price
        if ev < min_ev:
            continue

        # Fractional Kelly sizing
        odds = (1 - market_price) / market_price if market_price > 0 else 1.0
        full_kelly = edge / odds if odds > 0 else 0.0
        size_frac = full_kelly * kelly_fraction
        size_usd = min(size_frac * equity, 20.0)  # Hard cap at $20

        if size_usd < 0.50:
            continue

        # P&L calculation
        if row.actual_outcome:
            # Win: buy at market_price, resolve at 1.0
            shares = size_usd / market_price
            pnl = shares * (1.0 - market_price)
            gross_profit += pnl
        else:
            # Loss: lose cost basis
            pnl = -size_usd
            gross_loss += abs(pnl)

        equity += pnl
        pnl_list.append(pnl)
        equity_curve.append(equity)

    n_wins = sum(1 for p in pnl_list if p > 0)
    n_losses = sum(1 for p in pnl_list if p < 0)
    n_trades = len(pnl_list)
    total_pnl = sum(pnl_list)

    # Returns as fractions of bankroll for Sharpe
    returns = [p / bankroll for p in pnl_list]
    sharpe = _sharpe(returns)
    max_dd = _max_drawdown(equity_curve)
    pf = gross_profit / gross_loss if gross_loss > 0 else float("inf")
    wr = n_wins / n_trades if n_trades > 0 else 0.0
    avg_pnl = total_pnl / n_trades if n_trades > 0 else 0.0

    return TradeSimResult(
        n_trades=n_trades,
        n_wins=n_wins,
        n_losses=n_losses,
        gross_profit=gross_profit,
        gross_loss=gross_loss,
        total_pnl=total_pnl,
        max_drawdown_pct=max_dd * 100.0,
        sharpe=sharpe,
        profit_factor=pf,
        win_rate=wr,
        avg_pnl_per_trade=avg_pnl,
        pnl_curve=equity_curve,
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def summarize_predictions(rows: Iterable[PredictionRecord]) -> MetricSummary:
    """
    Compute full metric summary including Phase 5 trading metrics.
    """
    rows_list = list(rows)
    resolved = [row for row in rows_list if row.actual_outcome is not None]

    if not resolved:
        return MetricSummary(0, None, None, None, None)

    brier    = sum(float(row.brier or 0.0) for row in resolved) / len(resolved)
    log_loss = sum(_log_loss(row.predicted_prob, bool(row.actual_outcome)) for row in resolved) / len(resolved)
    crps     = sum(_crps_bernoulli(row.predicted_prob, bool(row.actual_outcome)) for row in resolved) / len(resolved)
    mean_p   = sum(row.predicted_prob for row in resolved) / len(resolved)
    hit_rate = sum(1 for row in resolved if row.actual_outcome) / len(resolved)

    # Trading simulation
    sim = _simulate_trades(resolved)

    return MetricSummary(
        count=len(resolved),
        brier=brier,
        log_loss=log_loss,
        mean_probability=mean_p,
        hit_rate=hit_rate,
        crps=crps,
        sharpe=sim.sharpe,
        profit_factor=sim.profit_factor if math.isfinite(sim.profit_factor) else None,
        max_drawdown_pct=sim.max_drawdown_pct,
        win_rate=sim.win_rate,
        total_pnl=sim.total_pnl,
        avg_pnl_per_trade=sim.avg_pnl_per_trade,
        n_trades=sim.n_trades,
        n_wins=sim.n_wins,
        n_losses=sim.n_losses,
    )


def group_summaries(rows: Iterable[PredictionRecord], attr: str) -> dict[str, MetricSummary]:
    groups: dict[str, list[PredictionRecord]] = defaultdict(list)
    for row in rows:
        groups[str(getattr(row, attr))].append(row)
    return {key: summarize_predictions(value) for key, value in sorted(groups.items())}


def calibration_bins(rows: Iterable[PredictionRecord], bins: int = 10) -> list[dict]:
    buckets = [{"n": 0, "prob_sum": 0.0, "hit_sum": 0.0} for _ in range(bins)]
    for row in rows:
        if row.actual_outcome is None:
            continue
        idx = min(bins - 1, max(0, int(row.predicted_prob * bins)))
        buckets[idx]["n"] += 1
        buckets[idx]["prob_sum"] += row.predicted_prob
        buckets[idx]["hit_sum"] += 1.0 if row.actual_outcome else 0.0

    result = []
    for idx, bucket in enumerate(buckets):
        n = bucket["n"]
        result.append(
            {
                "bin": idx,
                "low": idx / bins,
                "high": (idx + 1) / bins,
                "n": n,
                "mean_probability": None if n == 0 else bucket["prob_sum"] / n,
                "hit_rate": None if n == 0 else bucket["hit_sum"] / n,
            }
        )
    return result


def phase5_trade_simulation(
    rows: Iterable[PredictionRecord],
    bankroll: float = 1000.0,
    kelly_fraction: float = 0.10,
    min_edge_pct: float = 20.0,
) -> TradeSimResult:
    """Public entry point for standalone trade simulation."""
    return _simulate_trades(list(rows), bankroll, kelly_fraction, min_edge_pct)
