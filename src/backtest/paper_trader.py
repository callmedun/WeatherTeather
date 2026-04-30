"""
paper_trader.py -- Phase 5: Parallel Paper Trading Engine
==========================================================
Shadows every live scan cycle but records to a separate SQLite DB.
Compares paper signals vs live trades to validate BMA improvements.

Architecture:
  - PaperTrader.record_signal()  → called during scan for every signal
  - PaperTrader.record_outcome() → called when a market resolves
  - PaperTrader.get_report()     → Telegram-formatted performance summary
  - PaperTrader.run_comparison() → compare paper vs live stats

Paper trading uses Phase 3 entry gates but does NOT execute real orders.
All positions are tracked in paper_trades.sqlite.

Usage in scheduler.py:
    # After generating signals in scan_and_trade:
    for signal in signals:
        paper_trader.record_signal(signal)
"""

from __future__ import annotations

import json
import logging
import math
import os
import sqlite3
import time
from dataclasses import dataclass, field
from datetime import date, datetime, timezone
from typing import Optional

from src.backtest.metrics import (
    MetricSummary,
    TradeSimResult,
    _log_loss,
    _crps_bernoulli,
    _sharpe,
    _max_drawdown,
    _simulate_trades,
    phase5_trade_simulation,
)
from src.backtest.models import PredictionRecord

logger = logging.getLogger(__name__)

_DB_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
    "data", "paper_trades.sqlite"
)
os.makedirs(os.path.dirname(_DB_PATH), exist_ok=True)


@dataclass
class PaperPosition:
    """A single paper trade position."""
    signal_id: int
    market_id: str
    token_id: str
    city: str
    question: str
    outcome_name: str          # 'Yes' or 'No'
    entry_price: float         # market_price at signal time
    true_prob: float           # our predicted probability
    ev: float
    edge_pct: float
    usd_size: float            # simulated position size
    htc: float                 # hours to close at entry
    bma_ensemble_prob: float
    bma_n_members: int
    analysis_model: str        # 'tsas' or 'tsas+bma'
    opened_at: str             # ISO datetime
    status: str = "OPEN"       # OPEN | WIN | LOSS | SKIP
    exit_price: Optional[float] = None
    pnl: Optional[float] = None
    closed_at: Optional[str] = None
    actual_outcome: Optional[bool] = None


class PaperTrader:
    """
    Parallel paper trading engine.
    Records every BMA signal that passes Phase 3 entry gates.
    Tracks paper PnL vs live bot performance.
    """

    BANKROLL = 1000.0           # Paper bankroll ($)
    MAX_POSITIONS = 20          # Max concurrent paper positions
    MIN_USD = 0.50              # Minimum paper trade size

    def __init__(self, db_path: str = _DB_PATH):
        self.db_path = db_path
        self._init_db()

    def _get_conn(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path, timeout=10)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        return conn

    def _init_db(self) -> None:
        with self._get_conn() as conn:
            conn.executescript("""
                CREATE TABLE IF NOT EXISTS paper_signals (
                    id              INTEGER PRIMARY KEY AUTOINCREMENT,
                    market_id       TEXT NOT NULL,
                    token_id        TEXT NOT NULL,
                    city            TEXT NOT NULL,
                    question        TEXT,
                    outcome_name    TEXT,
                    entry_price     REAL,
                    true_prob       REAL,
                    ev              REAL,
                    edge_pct        REAL,
                    usd_size        REAL,
                    htc             REAL,
                    bma_ensemble_prob REAL DEFAULT 0.0,
                    bma_n_members   INTEGER DEFAULT 0,
                    analysis_model  TEXT DEFAULT 'tsas',
                    opened_at       TEXT DEFAULT (datetime('now')),
                    status          TEXT DEFAULT 'OPEN',
                    exit_price      REAL,
                    pnl             REAL,
                    closed_at       TEXT,
                    actual_outcome  INTEGER,
                    UNIQUE(market_id, token_id)
                );

                CREATE TABLE IF NOT EXISTS paper_daily_stats (
                    id          INTEGER PRIMARY KEY AUTOINCREMENT,
                    stat_date   TEXT NOT NULL UNIQUE,
                    n_signals   INTEGER DEFAULT 0,
                    n_entered   INTEGER DEFAULT 0,
                    n_wins      INTEGER DEFAULT 0,
                    n_losses    INTEGER DEFAULT 0,
                    pnl         REAL DEFAULT 0.0,
                    brier       REAL,
                    sharpe      REAL,
                    updated_at  TEXT DEFAULT (datetime('now'))
                );

                CREATE INDEX IF NOT EXISTS idx_signals_market
                    ON paper_signals(market_id, status);
                CREATE INDEX IF NOT EXISTS idx_signals_city
                    ON paper_signals(city, status);
            """)

    # ------------------------------------------------------------------
    # 1. Record a signal during scan (Phase 3 gate applied here)
    # ------------------------------------------------------------------

    def record_signal(self, signal: dict, p3_result: Optional[dict] = None) -> bool:
        """
        Record a signal from scan_and_trade.
        Returns True if we paper-entered the trade, False if skipped.

        Args:
            signal    : full signal dict from tsas_model.analyze_city_tsas
            p3_result : result dict from phase3_rules.evaluate_entry (optional)
        """
        try:
            market_id    = str(signal.get("market_id", ""))
            token_id     = str(signal.get("token_id", ""))
            outcome_name = str(signal.get("outcome_name", ""))

            if not market_id or not token_id:
                return False

            # Only paper-trade YES outcomes (mirror live bot)
            if outcome_name.lower() != "yes":
                return False

            entry_price   = float(signal.get("market_price", 0.0))
            true_prob     = float(signal.get("true_probability", signal.get("predicted_prob", 0.0)))
            ev            = float(signal.get("ev", 0.0))
            edge_pct      = (true_prob - entry_price) * 100.0
            htc           = float(signal.get("htc", 0.0))
            bma_ens_prob  = float(signal.get("bma_ensemble_prob", 0.0))
            bma_members   = int(signal.get("bma_n_members", 0))
            analysis_model = str(signal.get("analysis_model", "tsas"))
            city          = str(signal.get("city", ""))
            question      = str(signal.get("question", ""))

            # Use Phase 3 size if provided, else estimate with Kelly
            if p3_result and p3_result.get("enter") and p3_result.get("usd_size", 0) > 0:
                usd_size = float(p3_result["usd_size"])
                entered  = True
            else:
                # Apply basic Phase 3 gate manually
                entered = (
                    edge_pct >= 20.0
                    and ev >= 0.05
                    and htc >= 12.0
                    and htc <= 72.0
                )
                # Simple Kelly sizing
                odds = (1 - entry_price) / entry_price if entry_price > 0 else 1.0
                full_kelly = (edge_pct / 100.0) / odds if odds > 0 else 0.0
                usd_size = min(full_kelly * 0.10 * self.BANKROLL, 20.0)

            if not entered or usd_size < self.MIN_USD:
                # Record as a skipped signal (still useful for calibration)
                self._upsert_signal(
                    market_id=market_id, token_id=token_id, city=city,
                    question=question, outcome_name=outcome_name,
                    entry_price=entry_price, true_prob=true_prob,
                    ev=ev, edge_pct=edge_pct, usd_size=0.0, htc=htc,
                    bma_ens_prob=bma_ens_prob, bma_members=bma_members,
                    analysis_model=analysis_model, status="SKIP",
                )
                return False

            # Check position cap
            with self._get_conn() as conn:
                n_open = conn.execute(
                    "SELECT COUNT(*) FROM paper_signals WHERE status='OPEN'"
                ).fetchone()[0]
                if n_open >= self.MAX_POSITIONS:
                    logger.debug(f"[PaperTrader] Max positions ({self.MAX_POSITIONS}) reached, skipping {market_id}")
                    return False

            self._upsert_signal(
                market_id=market_id, token_id=token_id, city=city,
                question=question, outcome_name=outcome_name,
                entry_price=entry_price, true_prob=true_prob,
                ev=ev, edge_pct=edge_pct, usd_size=usd_size, htc=htc,
                bma_ens_prob=bma_ens_prob, bma_members=bma_members,
                analysis_model=analysis_model, status="OPEN",
            )
            logger.info(
                f"[PaperTrader] ENTER ${usd_size:.2f} | {city} {outcome_name} | "
                f"edge={edge_pct:.1f}% ev={ev:.3f} bma_ens={bma_ens_prob:.1%}"
            )
            return True

        except Exception as e:
            logger.error(f"[PaperTrader] record_signal error: {e}")
            return False

    def _upsert_signal(self, **kwargs) -> None:
        with self._get_conn() as conn:
            conn.execute("""
                INSERT OR IGNORE INTO paper_signals
                    (market_id, token_id, city, question, outcome_name,
                     entry_price, true_prob, ev, edge_pct, usd_size, htc,
                     bma_ensemble_prob, bma_n_members, analysis_model, status)
                VALUES
                    (:market_id, :token_id, :city, :question, :outcome_name,
                     :entry_price, :true_prob, :ev, :edge_pct, :usd_size, :htc,
                     :bma_ens_prob, :bma_members, :analysis_model, :status)
            """, kwargs)

    # ------------------------------------------------------------------
    # 2. Record market outcome (called from calibration.check_resolutions)
    # ------------------------------------------------------------------

    def record_outcome(
        self,
        token_id: str,
        actual_win: bool,
        exit_price: float = 1.0,
    ) -> Optional[float]:
        """
        Close an open paper position with its actual outcome.
        Returns realized PnL or None if position not found.
        """
        try:
            with self._get_conn() as conn:
                row = conn.execute("""
                    SELECT id, entry_price, usd_size, outcome_name
                    FROM paper_signals
                    WHERE token_id = ? AND status = 'OPEN'
                    LIMIT 1
                """, (token_id,)).fetchone()

                if not row:
                    return None

                entry_price = float(row["entry_price"])
                usd_size    = float(row["usd_size"])
                shares      = usd_size / entry_price if entry_price > 0 else 0.0

                if actual_win:
                    pnl = shares * (exit_price - entry_price)
                    status = "WIN"
                else:
                    pnl = -usd_size
                    status = "LOSS"

                conn.execute("""
                    UPDATE paper_signals
                    SET status=?, exit_price=?, pnl=?, closed_at=datetime('now'), actual_outcome=?
                    WHERE id=?
                """, (status, exit_price, pnl, 1 if actual_win else 0, row["id"]))

            logger.info(f"[PaperTrader] CLOSE {status} | token={token_id[:12]}... pnl=${pnl:.2f}")
            return pnl

        except Exception as e:
            logger.error(f"[PaperTrader] record_outcome error: {e}")
            return None

    # ------------------------------------------------------------------
    # 3. Monitor open positions (update if BMA signal changed)
    # ------------------------------------------------------------------

    def update_position(self, token_id: str, new_true_prob: float) -> None:
        """Update predicted probability for monitoring (e.g. from BMA re-scan)."""
        try:
            with self._get_conn() as conn:
                conn.execute("""
                    UPDATE paper_signals SET true_prob=?
                    WHERE token_id=? AND status='OPEN'
                """, (new_true_prob, token_id))
        except Exception as e:
            logger.debug(f"[PaperTrader] update_position error: {e}")

    # ------------------------------------------------------------------
    # 4. Performance report (Telegram-friendly)
    # ------------------------------------------------------------------

    def get_report(self, days: int = 30) -> str:
        """
        Returns a concise Telegram-formatted performance report.
        """
        try:
            with self._get_conn() as conn:
                rows = conn.execute("""
                    SELECT status, entry_price, true_prob, ev, pnl, usd_size,
                           edge_pct, bma_n_members, analysis_model
                    FROM paper_signals
                    WHERE opened_at >= datetime('now', '-' || ? || ' days')
                    ORDER BY opened_at
                """, (days,)).fetchall()

            if not rows:
                return f"📋 *Paper Trader* — No signals in last {days} days."

            total        = len(rows)
            entered      = [r for r in rows if r["status"] in ("OPEN","WIN","LOSS")]
            skipped      = [r for r in rows if r["status"] == "SKIP"]
            closed       = [r for r in rows if r["status"] in ("WIN","LOSS")]
            wins         = [r for r in rows if r["status"] == "WIN"]
            losses       = [r for r in rows if r["status"] == "LOSS"]
            open_pos     = [r for r in rows if r["status"] == "OPEN"]

            pnl_vals     = [float(r["pnl"]) for r in closed if r["pnl"] is not None]
            total_pnl    = sum(pnl_vals)
            gross_profit = sum(p for p in pnl_vals if p > 0)
            gross_loss   = abs(sum(p for p in pnl_vals if p < 0))
            pf           = gross_profit / gross_loss if gross_loss > 0 else float("inf")
            wr           = len(wins) / len(closed) if closed else 0.0

            # Brier score from closed positions
            brier_vals = []
            for r in closed:
                prob = float(r["true_prob"])
                act  = r["status"] == "WIN"
                brier_vals.append((prob - (1.0 if act else 0.0)) ** 2)
            brier = sum(brier_vals) / len(brier_vals) if brier_vals else None

            # Sharpe from closed PnL
            returns = [p / self.BANKROLL for p in pnl_vals]
            sharpe  = _sharpe(returns) if len(returns) >= 2 else 0.0

            # BMA vs TSAS split
            bma_signals  = [r for r in entered if "bma" in str(r["analysis_model"])]
            tsas_signals = [r for r in entered if "bma" not in str(r["analysis_model"])]

            pf_str = f"{pf:.2f}" if math.isfinite(pf) else "∞"

            return (
                f"📋 *Paper Trader* (last {days}d)\n\n"
                f"📊 Signals: {total} total | {len(entered)} entered | {len(skipped)} skipped\n"
                f"🟢 Open positions: {len(open_pos)}\n\n"
                f"🎯 Closed: {len(closed)} | ✅{len(wins)}W / ❌{len(losses)}L\n"
                f"📈 Win Rate: {wr:.1%}\n"
                f"💰 Total PnL: ${total_pnl:+.2f}\n"
                f"🔢 Profit Factor: {pf_str}\n"
                f"📉 Sharpe: {sharpe:.2f}\n"
                f"🎲 Brier Score: {f'{brier:.4f}' if brier else 'N/A'}\n\n"
                f"🧠 BMA signals: {len(bma_signals)} | TSAS-only: {len(tsas_signals)}"
            )

        except Exception as e:
            return f"📋 Paper Trader error: {e}"

    def get_open_positions(self) -> str:
        """Returns open paper positions for monitoring."""
        try:
            with self._get_conn() as conn:
                rows = conn.execute("""
                    SELECT city, outcome_name, entry_price, true_prob,
                           edge_pct, usd_size, htc, bma_n_members, opened_at
                    FROM paper_signals
                    WHERE status='OPEN'
                    ORDER BY opened_at DESC
                """).fetchall()

            if not rows:
                return "📋 Paper Trader: No open positions."

            lines = ["📋 *Paper Trader — Open Positions*\n"]
            for r in rows:
                lines.append(
                    f"• {r['city']} {r['outcome_name']} @ {r['entry_price']:.3f}\n"
                    f"  AI={r['true_prob']:.1%} edge={r['edge_pct']:.1f}% "
                    f"size=${r['usd_size']:.2f} BMA_n={r['bma_n_members']}"
                )
            return "\n".join(lines)

        except Exception as e:
            return f"Paper Trader positions error: {e}"

    def get_comparison_report(self) -> str:
        """
        Compare paper bot performance vs live bot calibration data.
        Shows if BMA improves edge over pure TSAS.
        """
        try:
            with self._get_conn() as conn:
                # BMA vs TSAS model comparison
                for_comp = conn.execute("""
                    SELECT analysis_model, COUNT(*) as n,
                           AVG(CASE WHEN status='WIN' THEN 1.0 ELSE 0.0 END) as win_rate,
                           SUM(pnl) as total_pnl,
                           AVG(edge_pct) as avg_edge,
                           AVG(bma_n_members) as avg_members
                    FROM paper_signals
                    WHERE status IN ('WIN','LOSS')
                    GROUP BY analysis_model
                """).fetchall()

            if not for_comp:
                return "📊 No closed paper trades for comparison yet."

            lines = ["📊 *Paper Trader: Model Comparison*\n"]
            for r in for_comp:
                model = r["analysis_model"] or "tsas"
                wr    = float(r["win_rate"] or 0.0)
                pnl   = float(r["total_pnl"] or 0.0)
                edge  = float(r["avg_edge"] or 0.0)
                n     = int(r["n"])
                lines.append(
                    f"**{model}** (n={n})\n"
                    f"  WR={wr:.1%} | PnL=${pnl:+.2f} | AvgEdge={edge:.1f}%"
                )
            return "\n".join(lines)

        except Exception as e:
            return f"Comparison error: {e}"

    # ------------------------------------------------------------------
    # 5. Convert paper signals to PredictionRecord for backtest metrics
    # ------------------------------------------------------------------

    def to_prediction_records(self, days: int = 90) -> list[PredictionRecord]:
        """
        Convert closed paper signals to PredictionRecord for metric computation.
        Allows using the full Phase 5 metrics pipeline.
        """
        try:
            with self._get_conn() as conn:
                rows = conn.execute("""
                    SELECT market_id, token_id, city, question, outcome_name,
                           entry_price, true_prob, ev, pnl, status, actual_outcome,
                           opened_at, closed_at, analysis_model
                    FROM paper_signals
                    WHERE status IN ('WIN','LOSS')
                    AND opened_at >= datetime('now', '-' || ? || ' days')
                """, (days,)).fetchall()

            records = []
            for r in rows:
                prob   = float(r["true_prob"] or 0.0)
                actual = bool(r["actual_outcome"]) if r["actual_outcome"] is not None else None
                brier  = None if actual is None else (prob - (1.0 if actual else 0.0)) ** 2
                try:
                    as_of = datetime.fromisoformat(str(r["opened_at"]))
                except Exception:
                    as_of = datetime.now(timezone.utc)

                rec = PredictionRecord(
                    run_id="paper_trader",
                    market_id=str(r["market_id"]),
                    city=str(r["city"]),
                    icao_code="",
                    question=str(r["question"] or ""),
                    target_date=as_of.date(),
                    as_of=as_of,
                    outcome_name=str(r["outcome_name"]),
                    token_id=str(r["token_id"]),
                    predicted_prob=prob,
                    bucket_probability_yes=prob,
                    actual_outcome=actual,
                    brier=brier,
                )
                # Attach market_price for trade simulation
                rec.market_price = float(r["entry_price"] or 0.0)
                records.append(rec)

            return records
        except Exception as e:
            logger.error(f"[PaperTrader] to_prediction_records error: {e}")
            return []

    def get_full_metrics_report(self, days: int = 90) -> str:
        """Full metrics report using Phase 5 metrics pipeline."""
        records = self.to_prediction_records(days)
        if not records:
            return f"📊 Paper Trader: No closed trades in last {days} days for metrics."

        from src.backtest.metrics import summarize_predictions
        summary = summarize_predictions(records)

        pf_str = f"{summary.profit_factor:.2f}" if summary.profit_factor and math.isfinite(summary.profit_factor) else "∞"

        return (
            f"📊 *Paper Trader Full Metrics* (last {days}d)\n\n"
            f"🎯 **Calibration**\n"
            f"  Brier: {f'{summary.brier:.4f}' if summary.brier else 'N/A'}\n"
            f"  CRPS:  {f'{summary.crps:.4f}' if summary.crps else 'N/A'}\n"
            f"  Hit Rate: {f'{summary.hit_rate:.1%}' if summary.hit_rate else 'N/A'}\n\n"
            f"💰 **Trading** ({summary.n_trades} trades)\n"
            f"  Win Rate:      {f'{summary.win_rate:.1%}' if summary.win_rate else 'N/A'}\n"
            f"  Profit Factor: {pf_str}\n"
            f"  Sharpe:        {f'{summary.sharpe:.2f}' if summary.sharpe else 'N/A'}\n"
            f"  Max Drawdown:  {f'{summary.max_drawdown_pct:.1f}%' if summary.max_drawdown_pct is not None else 'N/A'}\n"
            f"  Total PnL:     ${summary.total_pnl:+.2f}" if summary.total_pnl is not None else "  Total PnL: N/A"
        )


# Singleton
paper_trader = PaperTrader()
