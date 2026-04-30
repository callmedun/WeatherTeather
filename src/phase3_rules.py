"""
phase3_rules.py -- Phase 3: BMA-Aware Entry / Exit / Position Sizing Rules
===========================================================================

This module provides decision logic that uses BMA data (from Phase 2) to
make smarter entry, exit, and sizing decisions than the original TSAS rules.

Design principles:
  - Fully backward compatible: all functions have safe defaults when BMA
    fields are absent (falls back to TSAS-only behaviour).
  - No new DB schema required: BMA fields live in the signal/trade dict.
  - Three public surfaces:
      1. should_enter(signal)         -> (bool, str)   entry gate
      2. compute_position_size(sig)   -> float          USD size
      3. should_exit(trade, context)  -> (bool, str)    exit trigger
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Configuration (mirrors config/settings.py where possible)
# ---------------------------------------------------------------------------

@dataclass
class Phase3Config:
    # --- Entry thresholds ---
    min_edge_pct: float = 20.0          # minimum edge % (true_prob - market_price)*100
    min_ev: float = 0.10                # minimum EV after slippage
    min_confidence: int = 35            # minimum tsas_confidence * 100

    # BMA-specific entry gates (active only when bma_n_members > 0)
    bma_min_ensemble_prob: float = 0.15  # ensemble must agree at least this much
    bma_max_model_spread: float = 3.5    # reject if models disagree > 3.5°C
    bma_min_members: int = 20            # require at least 20 ensemble members
    bma_correction_boost: float = 0.05   # lower EV threshold when BMA confirms

    # Fat-tail hunting (target underpriced buckets)
    fat_tail_max_price: float = 0.15     # max market price for fat-tail entry
    fat_tail_min_edge_pct: float = 25.0  # higher edge required for fat tails
    fat_tail_min_members: int = 30       # need more ensemble members for fat tails

    # Hours-to-close gates
    min_htc: float = 12.0               # don't enter < 12h to resolution
    max_htc: float = 72.0               # don't enter > 72h out

    # --- Position sizing ---
    kelly_fraction: float = 0.10        # fractional Kelly
    max_single_usd: float = 20.0        # hard cap per trade USD
    max_city_exposure_pct: float = 0.08 # 8% of bankroll per city
    max_total_exposure_pct: float = 0.20# 20% of bankroll total

    # BMA confidence scaling: multiply position by this when BMA active + high confidence
    bma_high_conf_multiplier: float = 1.20  # up to 20% larger when BMA is very confident
    bma_low_conf_multiplier: float = 0.75   # scale down when BMA uncertain

    # --- Exit thresholds ---
    exit_prob_drop_pct: float = 10.0    # exit if true_prob drops > 10pp from entry
    exit_edge_floor_pct: float = 2.0    # exit if edge < 2%
    exit_model_flip: float = 0.30       # exit if prob drops below 30%
    exit_trailing_profit_pct: float = 50.0  # set trailing stop at 50% unrealized PnL
    exit_trailing_stop_pct: float = 30.0   # trailing stop level

    # BMA-specific exit: exit when ensemble disagrees with our position
    bma_exit_ensemble_flip: float = 0.10   # exit if ensemble_prob < 10% for YES
    bma_exit_spread_spike: float = 4.0     # exit if model spread > 4°C (uncertainty spike)


# Default config instance (can be overridden in tests)
p3_config = Phase3Config()


# ---------------------------------------------------------------------------
# 1. ENTRY GATE
# ---------------------------------------------------------------------------

def should_enter(signal: dict, bankroll: float = 1000.0) -> tuple[bool, str]:
    """
    BMA-enhanced entry gate.

    Returns (True, reason) to enter, or (False, reason) to skip.
    Falls back to TSAS-only logic when BMA fields are absent.
    """
    htc = float(signal.get("htc", 24.0))
    market_price = float(signal.get("market_price", 0.5))
    true_prob = float(signal.get("true_probability", 0.0))
    ev = float(signal.get("ev", 0.0) or signal.get("raw_ev", 0.0))
    confidence = int(signal.get("confidence", 0))
    analysis_model = str(signal.get("analysis_model", "tsas"))

    # BMA diagnostic fields
    bma_active = "bma" in analysis_model and signal.get("bma_n_members", 0) > 0
    n_members = int(signal.get("bma_n_members", 0))
    ensemble_prob = float(signal.get("bma_ensemble_prob", -1.0))
    model_spread = float(signal.get("bma_model_spread", 0.0))
    bma_correction = bool(signal.get("bma_correction", False))

    edge_pct = (true_prob - market_price) * 100.0

    # --- Gate 1: Hours-to-close ---
    if htc < p3_config.min_htc:
        return False, f"htc_too_close: {htc:.1f}h < {p3_config.min_htc}h"
    if htc > p3_config.max_htc:
        return False, f"htc_too_far: {htc:.1f}h > {p3_config.max_htc}h"

    # --- Gate 2: Edge ---
    min_edge = p3_config.min_edge_pct
    is_fat_tail = market_price <= p3_config.fat_tail_max_price
    if is_fat_tail:
        # Fat tail markets require a higher minimum edge to compensate for tail risk
        min_edge = max(min_edge, p3_config.fat_tail_min_edge_pct)
    if edge_pct < min_edge:
        return False, f"edge_too_low: {edge_pct:.1f}% < {min_edge:.1f}% (fat_tail={is_fat_tail})"

    # --- Gate 3: EV ---
    ev_threshold = p3_config.min_ev
    if bma_active and bma_correction:
        # BMA confirmed — slightly lower EV bar
        ev_threshold = max(0.05, ev_threshold - p3_config.bma_correction_boost)
    if ev < ev_threshold:
        return False, f"ev_too_low: {ev:.3f} < {ev_threshold:.3f}"

    # --- Gate 4: Confidence ---
    if confidence < p3_config.min_confidence:
        return False, f"confidence_too_low: {confidence} < {p3_config.min_confidence}"

    # --- Gate 5: BMA ensemble minimum (when BMA active) ---
    if bma_active:
        if n_members < p3_config.bma_min_members:
            return False, f"bma_too_few_members: {n_members} < {p3_config.bma_min_members}"

        if model_spread > p3_config.bma_max_model_spread:
            return False, f"bma_model_spread_too_high: {model_spread:.2f}C > {p3_config.bma_max_model_spread}C"

        if ensemble_prob >= 0:
            # For YES trades: ensemble must not strongly disagree
            outcome = str(signal.get("outcome_name", "")).lower()
            if outcome == "yes" and ensemble_prob < p3_config.bma_min_ensemble_prob:
                return False, f"bma_ensemble_disagrees: ens_prob={ensemble_prob:.1%} < {p3_config.bma_min_ensemble_prob:.1%}"
            # For NO trades: ensemble must not strongly agree with YES
            if outcome == "no" and ensemble_prob > (1.0 - p3_config.bma_min_ensemble_prob):
                return False, f"bma_ensemble_disagrees_no: ens_prob={ensemble_prob:.1%}"

        # Fat tail BMA gate: require more members
        if market_price <= p3_config.fat_tail_max_price:
            if n_members < p3_config.fat_tail_min_members:
                return False, f"fat_tail_bma_members: {n_members} < {p3_config.fat_tail_min_members}"

    return True, f"OK edge={edge_pct:.1f}% ev={ev:.3f} conf={confidence} bma={bma_active} n={n_members}"


# ---------------------------------------------------------------------------
# 2. POSITION SIZING
# ---------------------------------------------------------------------------

def compute_position_size(
    signal: dict,
    bankroll: float,
    current_city_exposure: float = 0.0,
    current_total_exposure: float = 0.0,
) -> tuple[float, str]:
    """
    BMA-enhanced fractional Kelly position sizing.

    Returns (usd_size, reasoning_str).
    Scales position up when BMA has high confidence, down when uncertain.
    """
    market_price = float(signal.get("market_price", 0.5))
    true_prob = float(signal.get("true_probability", 0.0))
    kelly_raw = float(signal.get("kelly", 0.0))

    # BMA confidence adjustment
    bma_active = "bma" in str(signal.get("analysis_model", ""))
    n_members = int(signal.get("bma_n_members", 0))
    model_spread = float(signal.get("bma_model_spread", 0.0))
    bma_correction = bool(signal.get("bma_correction", False))

    # Base Kelly size
    odds = (1.0 - market_price) / market_price if market_price > 0 else 0.0
    raw_edge = true_prob - market_price
    full_kelly = raw_edge / odds if odds > 0 else 0.0
    frac_kelly = full_kelly * p3_config.kelly_fraction

    # Base USD size from Kelly
    kelly_usd = frac_kelly * bankroll

    # BMA confidence scalar
    bma_scalar = 1.0
    reason_parts = [f"kelly={frac_kelly:.4f}"]

    if bma_active and n_members > 0:
        # High confidence: large ensemble, tight spread, correction confirmed
        if n_members >= 40 and model_spread <= 1.5 and bma_correction:
            bma_scalar = p3_config.bma_high_conf_multiplier
            reason_parts.append(f"bma_boost={bma_scalar:.2f}(n={n_members},spread={model_spread:.1f}C)")
        # Low confidence: few members or wide spread
        elif n_members < 25 or model_spread > 3.0:
            bma_scalar = p3_config.bma_low_conf_multiplier
            reason_parts.append(f"bma_penalty={bma_scalar:.2f}(n={n_members},spread={model_spread:.1f}C)")
        else:
            reason_parts.append(f"bma_neutral(n={n_members},spread={model_spread:.1f}C)")

    adjusted_usd = kelly_usd * bma_scalar

    # Exposure caps
    max_city = p3_config.max_city_exposure_pct * bankroll - current_city_exposure
    max_total = p3_config.max_total_exposure_pct * bankroll - current_total_exposure
    hard_cap = p3_config.max_single_usd

    final_usd = min(adjusted_usd, max_city, max_total, hard_cap)
    final_usd = max(0.0, final_usd)

    reason_parts.append(
        f"base_usd={kelly_usd:.2f} adj={adjusted_usd:.2f} "
        f"caps=[city={max_city:.1f},total={max_total:.1f},hard={hard_cap:.1f}] "
        f"final={final_usd:.2f}"
    )

    return final_usd, " | ".join(reason_parts)


# ---------------------------------------------------------------------------
# 3. EXIT RULES
# ---------------------------------------------------------------------------

@dataclass
class ExitContext:
    """Current state of an open trade for exit evaluation."""
    entry_price: float
    entry_prob: float
    current_price: float          # latest market price (bestAsk or bestBid)
    current_prob: float           # latest model probability
    unrealized_pnl_pct: float     # as fraction, e.g. 0.45 = +45%
    hours_to_close: float
    outcome_name: str             # "Yes" or "No"

    # BMA refresh fields (from latest model run, if available)
    bma_ensemble_prob: float = -1.0
    bma_model_spread: float = 0.0
    bma_n_members: int = 0


def should_exit(trade_dict: dict, ctx: ExitContext) -> tuple[bool, str]:
    """
    BMA-enhanced exit decision.

    Returns (True, reason) to close position, or (False, "hold") to keep.
    Evaluates in priority order: resolution, TSAS drops, BMA signals, trailing.
    """
    outcome = ctx.outcome_name.lower()
    is_yes = outcome == "yes"

    # --- Exit 1: Resolution imminent + profitable ---
    if ctx.hours_to_close <= 4.0 and ctx.unrealized_pnl_pct >= 0.0:
        return True, f"resolution_imminent: htc={ctx.hours_to_close:.1f}h pnl={ctx.unrealized_pnl_pct:.1%}"

    # --- Exit 2: Probability collapse ---
    if ctx.current_prob < p3_config.exit_model_flip:
        return True, f"prob_collapse: prob={ctx.current_prob:.1%} < {p3_config.exit_model_flip:.1%}"

    # --- Exit 3: Probability drop from entry ---
    prob_drop = (ctx.entry_prob - ctx.current_prob) * 100.0
    if prob_drop > p3_config.exit_prob_drop_pct:
        return True, f"prob_drop: {prob_drop:.1f}pp > {p3_config.exit_prob_drop_pct}pp"

    # --- Exit 4: Edge below floor ---
    current_edge_pct = (ctx.current_prob - ctx.current_price) * 100.0
    if current_edge_pct < p3_config.exit_edge_floor_pct:
        return True, f"edge_floor: edge={current_edge_pct:.1f}% < {p3_config.exit_edge_floor_pct}%"

    # --- Exit 5: BMA ensemble flip (when BMA data is fresh) ---
    if ctx.bma_n_members >= p3_config.bma_min_members and ctx.bma_ensemble_prob >= 0:
        if is_yes and ctx.bma_ensemble_prob < p3_config.bma_exit_ensemble_flip:
            return True, (
                f"bma_ensemble_flip_yes: ens_prob={ctx.bma_ensemble_prob:.1%} "
                f"< {p3_config.bma_exit_ensemble_flip:.1%}"
            )
        if not is_yes and ctx.bma_ensemble_prob > (1.0 - p3_config.bma_exit_ensemble_flip):
            return True, (
                f"bma_ensemble_flip_no: ens_prob={ctx.bma_ensemble_prob:.1%} "
                f"> {1.0 - p3_config.bma_exit_ensemble_flip:.1%}"
            )

    # --- Exit 6: BMA model spread spike (models became uncertain) ---
    if ctx.bma_model_spread > p3_config.bma_exit_spread_spike:
        if ctx.unrealized_pnl_pct > 0.10:  # only exit with profit on uncertainty spike
            return True, f"bma_spread_spike: spread={ctx.bma_model_spread:.2f}C > {p3_config.bma_exit_spread_spike:.2f}C (protecting profit)"

    # --- Exit 7: Trailing profit stop ---
    if ctx.unrealized_pnl_pct >= p3_config.exit_trailing_profit_pct / 100.0:
        # Once we hit target profit, protect at trailing_stop level
        if ctx.unrealized_pnl_pct <= p3_config.exit_trailing_stop_pct / 100.0:
            return True, f"trailing_stop: pnl={ctx.unrealized_pnl_pct:.1%} dropped below {p3_config.exit_trailing_stop_pct:.0f}%"

    return False, "hold"


# ---------------------------------------------------------------------------
# Utility: build ExitContext from trade dict + fresh signal
# ---------------------------------------------------------------------------

def build_exit_context(
    trade: dict,
    current_market_price: float,
    fresh_signal: Optional[dict] = None,
) -> ExitContext:
    """
    Build ExitContext from a stored trade dict and fresh market data.

    trade        : dict from portfolio_manager.get_open_trades() row
    current_market_price : latest bestAsk (for YES) or bestBid (for NO)
    fresh_signal : latest analyze_city_tsas output for this market (optional)
    """
    entry_price = float(trade.get("entry_price", 0.5))
    entry_prob  = float(trade.get("predicted_prob", 0.5))
    outcome     = str(trade.get("outcome_name", "Yes"))
    htc         = float(trade.get("hours_to_close", 24.0))
    shares      = float(trade.get("shares", 0.0))

    # Unrealized PnL: (current_price - entry_price) / entry_price
    if entry_price > 0:
        unreal_pnl = (current_market_price - entry_price) / entry_price
    else:
        unreal_pnl = 0.0

    # Fresh model probability
    current_prob = entry_prob
    bma_ensemble = -1.0
    bma_spread = 0.0
    bma_members = 0
    if fresh_signal:
        current_prob   = float(fresh_signal.get("true_probability", entry_prob))
        bma_ensemble   = float(fresh_signal.get("bma_ensemble_prob", -1.0))
        bma_spread     = float(fresh_signal.get("bma_model_spread", 0.0))
        bma_members    = int(fresh_signal.get("bma_n_members", 0))

    return ExitContext(
        entry_price=entry_price,
        entry_prob=entry_prob,
        current_price=current_market_price,
        current_prob=current_prob,
        unrealized_pnl_pct=unreal_pnl,
        hours_to_close=htc,
        outcome_name=outcome,
        bma_ensemble_prob=bma_ensemble,
        bma_model_spread=bma_spread,
        bma_n_members=bma_members,
    )


# ---------------------------------------------------------------------------
# Composite: full entry decision with logging
# ---------------------------------------------------------------------------

def evaluate_entry(signal: dict, bankroll: float = 1000.0,
                   city_exposure: float = 0.0,
                   total_exposure: float = 0.0) -> dict:
    """
    Run all Phase 3 entry checks and return a decision dict.

    Returns:
        {
          "enter": bool,
          "reason": str,
          "usd_size": float,
          "sizing_reason": str,
        }
    """
    ok, entry_reason = should_enter(signal, bankroll)
    if not ok:
        return {"enter": False, "reason": entry_reason, "usd_size": 0.0, "sizing_reason": ""}

    usd, sizing_reason = compute_position_size(signal, bankroll, city_exposure, total_exposure)
    if usd < 1.0:
        return {"enter": False, "reason": f"size_too_small: {usd:.2f}USD", "usd_size": 0.0, "sizing_reason": sizing_reason}

    return {
        "enter": True,
        "reason": entry_reason,
        "usd_size": usd,
        "sizing_reason": sizing_reason,
    }
