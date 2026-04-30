"""
bma_engine.py -- Phase 2: Bayesian Model Averaging Probability Engine
======================================================================
Replaces the single-normal-distribution CDF in tsas_model.py with a
proper multi-model ensemble probability calculation.

Architecture:
  1. ENSEMBLE_COUNT:  Direct member counting from GEFS + ICON ensembles
                      (no distribution assumption)
  2. PARAMETRIC_BMA:  Weighted Gaussian mixture from deterministic models
                      (GFS, ECMWF, HRRR, NAM, NBM) using static skill weights
  3. TSAS_NORMAL:     Original TSAS normal CDF (kept as fallback)

Final probability:
  P = w_ens * P_ensemble + w_bma * P_bma + w_tsas * P_tsas

Weights by hours-to-close:
  - >48h:  ensemble=0.55, bma=0.35, tsas=0.10
  - 12-48h: ensemble=0.45, bma=0.35, tsas=0.20
  - <12h:  ensemble=0.25, bma=0.30, tsas=0.45  (METAR/TAF dominate via TSAS)

This module is intentionally decoupled from herbie_fetcher.py --
it only needs the pre-computed forecast_daily dict already present
in weather_data.
"""

from __future__ import annotations

import math
import logging
from dataclasses import dataclass
from typing import Optional

import numpy as np
from scipy.stats import norm

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Static model skill weights (equal until Phase 2b ML calibration is ready)
# Source: NOAA/ECMWF verification studies, 2-metre temperature, 24h forecast
# Will be replaced by learned BMA weights in Phase 2b
# ---------------------------------------------------------------------------
_SKILL_WEIGHTS: dict[str, float] = {
    "gefs_mean":  0.30,   # GFS ensemble mean (31 members)
    "ecmwf_ens":  0.25,   # ICON/secondary ensemble mean
    "ecmwf":      0.20,   # ECMWF IFS deterministic
    "gfs":        0.12,   # GFS deterministic
    "hrrr":       0.08,   # HRRR (US only, short-range)
    "nam":        0.03,   # NAM (US only)
    "nbm":        0.02,   # NBM (US only)
}

# Short-range boost for HRRR (hours < 18h)
_HRRR_SHORT_RANGE_BOOST = 0.15


@dataclass
class BmaResult:
    """Result from BMA probability engine."""
    probability: float          # Final blended probability P(bin)
    ensemble_prob: float        # Raw ensemble counting probability
    bma_prob: float             # Parametric BMA probability
    tsas_prob: float            # TSAS normal CDF probability (fallback)
    n_gefs_members: int         # Number of GEFS members used
    n_icon_members: int         # Number of ICON/secondary ensemble members
    n_det_models: int           # Number of deterministic models
    model_spread_c: float       # Spread across model means (Celsius)
    blend_weights: dict[str, float]  # Final layer weights used
    correction_applied: bool    # Whether ensemble correction was applied
    reasoning: str


# ---------------------------------------------------------------------------
# Layer 1: Direct ensemble member counting
# ---------------------------------------------------------------------------

def _ensemble_count_probability(
    forecast_daily: dict,
    day_idx: int,
    bin_low_c: float,
    bin_high_c: float,
    use_max: bool,
) -> tuple[float | None, int, int]:
    """
    Count ensemble members falling in [bin_low_c, bin_high_c).
    Returns (probability, n_gefs_members, n_icon_members).
    Combines GEFS + ICON members for larger effective ensemble.
    """
    max_or_min = "max" if use_max else "min"
    all_members: list[float] = []

    # GEFS members
    gefs_key = f"gefs_members_{max_or_min}"
    gefs_by_day = forecast_daily.get(gefs_key) or []
    n_gefs = 0
    if gefs_by_day and day_idx < len(gefs_by_day):
        gefs_members = gefs_by_day[day_idx] or []
        all_members.extend(gefs_members)
        n_gefs = len(gefs_members)

    # ICON/secondary ensemble members
    icon_key = "ecmwf_ens_members_max"  # currently ICON, named ecmwf_ens for compat
    icon_by_day = forecast_daily.get(icon_key) or []
    n_icon = 0
    if use_max and icon_by_day and day_idx < len(icon_by_day):
        icon_members = icon_by_day[day_idx] or []
        all_members.extend(icon_members)
        n_icon = len(icon_members)

    if not all_members:
        return None, 0, 0

    in_bin = sum(1 for t in all_members if bin_low_c <= t < bin_high_c)
    prob = in_bin / len(all_members)
    return prob, n_gefs, n_icon


# ---------------------------------------------------------------------------
# Layer 2: Parametric BMA from deterministic model means
# ---------------------------------------------------------------------------

def _extract_model_means(
    forecast_daily: dict,
    day_idx: int,
    use_max: bool,
    hours_to_close: float,
) -> dict[str, float]:
    """
    Extract available model point forecasts for the target day.
    Returns dict of {model_name: temperature_celsius}.
    """
    suffix = "max" if use_max else "min"
    models: dict[str, float] = {}

    def safe(key: str) -> float | None:
        vals = forecast_daily.get(key) or []
        if vals and day_idx < len(vals) and vals[day_idx] is not None:
            return float(vals[day_idx])
        # Fallback to day 0 if target day missing
        if vals and vals[0] is not None:
            return float(vals[0])
        return None

    for model_key in [
        f"gefs_mean_{suffix}",
        f"ecmwf_ens_mean_{suffix}",
        f"ecmwf_{suffix}",
        f"gfs_{suffix}",
        f"hrrr_{suffix}",
        f"nam_{suffix}",
        f"nbm_{suffix}",
    ]:
        val = safe(model_key)
        if val is not None:
            # Map key to skill weight key
            skill_key = model_key.replace(f"_{suffix}", "").replace("_mean", "")
            models[skill_key] = val

    return models


def _bma_gaussian_mixture_probability(
    model_means: dict[str, float],
    bin_low_c: float,
    bin_high_c: float,
    hours_to_close: float,
    ensemble_spread_c: float | None,
) -> tuple[float | None, float]:
    """
    Weighted Gaussian mixture probability.
    Each model contributes N(model_mean, sigma) with weight proportional
    to its skill score. Sigma estimated from ensemble spread + horizon.

    Returns (probability, effective_spread_across_models).
    """
    if not model_means:
        return None, 0.0

    # Per-component sigma: blend of ensemble spread and horizon uncertainty
    horizon_base = 0.65 + 0.4 * min(3, max(0, (hours_to_close - 12) / 24))
    ens_spread = ensemble_spread_c if ensemble_spread_c and ensemble_spread_c > 0 else 1.0
    sigma = max(0.5, 0.6 * ens_spread + 0.4 * horizon_base)

    # Boost HRRR weight for short-range
    weights = dict(_SKILL_WEIGHTS)
    if hours_to_close <= 18 and "hrrr" in model_means:
        weights["hrrr"] = weights.get("hrrr", 0.08) + _HRRR_SHORT_RANGE_BOOST

    # Normalize weights to available models only
    avail_w = {k: weights.get(k, 0.05) for k in model_means}
    total_w = sum(avail_w.values())
    if total_w <= 0:
        return None, 0.0
    norm_w = {k: v / total_w for k, v in avail_w.items()}

    # BMA probability = sum of weighted Gaussian CDFs
    prob = 0.0
    for model_key, mean_c in model_means.items():
        w = norm_w.get(model_key, 0.0)
        p_component = norm.cdf(bin_high_c, loc=mean_c, scale=sigma) - \
                      norm.cdf(bin_low_c, loc=mean_c, scale=sigma)
        prob += w * max(0.0, min(1.0, p_component))

    # Model spread (useful for confidence estimation)
    vals = list(model_means.values())
    spread = max(vals) - min(vals) if len(vals) > 1 else 0.0

    return max(0.0, min(1.0, prob)), spread


# ---------------------------------------------------------------------------
# Layer 3: Blend weights by hours-to-close
# ---------------------------------------------------------------------------

def _blend_weights(hours_to_close: float, n_ensemble_members: int) -> dict[str, float]:
    """
    Determine how to blend ensemble counting, BMA, and TSAS.
    More ensemble members and longer horizon -> more weight to ensemble.
    """
    has_ensemble = n_ensemble_members >= 10

    if hours_to_close > 48:
        if has_ensemble:
            w = {"ensemble": 0.55, "bma": 0.35, "tsas": 0.10}
        else:
            w = {"ensemble": 0.00, "bma": 0.60, "tsas": 0.40}
    elif hours_to_close > 12:
        if has_ensemble:
            w = {"ensemble": 0.45, "bma": 0.35, "tsas": 0.20}
        else:
            w = {"ensemble": 0.00, "bma": 0.55, "tsas": 0.45}
    else:
        # Close to resolution: METAR/TAF (captured in TSAS) are most relevant
        if has_ensemble:
            w = {"ensemble": 0.25, "bma": 0.25, "tsas": 0.50}
        else:
            w = {"ensemble": 0.00, "bma": 0.40, "tsas": 0.60}

    return w


# ---------------------------------------------------------------------------
# Public API: compute_bma_probability()
# ---------------------------------------------------------------------------

def compute_bma_probability(
    forecast_daily: dict,
    day_idx: int,
    bin_low_c: float,
    bin_high_c: float,
    use_max: bool,
    hours_to_close: float,
    tsas_prob: float,
) -> BmaResult:
    """
    Compute final probability using BMA blend of:
      - Layer 1: ensemble member counting (GEFS + ICON)
      - Layer 2: parametric Gaussian BMA from deterministic models
      - Layer 3: original TSAS normal CDF result (passed in as tsas_prob)

    Parameters
    ----------
    forecast_daily  : weather_data['forecast_daily'] dict (enriched by herbie_fetcher)
    day_idx         : 0=today, 1=tomorrow, etc.
    bin_low_c       : bucket lower bound in Celsius
    bin_high_c      : bucket upper bound in Celsius
    use_max         : True=daily max, False=daily min market
    hours_to_close  : hours until market resolution
    tsas_prob       : probability from existing TSAS normal CDF (fallback)

    Returns
    -------
    BmaResult with final blended probability and diagnostic fields
    """
    # --- Layer 1: Ensemble counting ---
    ens_prob, n_gefs, n_icon = _ensemble_count_probability(
        forecast_daily, day_idx, bin_low_c, bin_high_c, use_max
    )
    n_total_members = n_gefs + n_icon

    # --- Layer 2: Parametric BMA ---
    model_means = _extract_model_means(forecast_daily, day_idx, use_max, hours_to_close)

    # Get ensemble spread for sigma estimation
    spread_key = "gefs_spread_max" if use_max else "gefs_spread_min"
    spread_by_day = forecast_daily.get(spread_key) or []
    ens_spread = None
    if spread_by_day and day_idx < len(spread_by_day):
        ens_spread = spread_by_day[day_idx]
    # Fallback to model_spread_max
    if ens_spread is None:
        ms_key = "model_spread_max" if use_max else "model_spread_min"
        ms_by_day = forecast_daily.get(ms_key) or []
        if ms_by_day and day_idx < len(ms_by_day):
            ens_spread = ms_by_day[day_idx]

    bma_prob_raw, model_spread_c = _bma_gaussian_mixture_probability(
        model_means, bin_low_c, bin_high_c, hours_to_close, ens_spread
    )
    bma_prob = bma_prob_raw if bma_prob_raw is not None else tsas_prob

    # --- Layer 3: Blend ---
    blend_w = _blend_weights(hours_to_close, n_total_members)

    final_prob = (
        blend_w["ensemble"] * (ens_prob if ens_prob is not None else tsas_prob) +
        blend_w["bma"]      * bma_prob +
        blend_w["tsas"]     * tsas_prob
    )
    final_prob = max(0.001, min(0.999, final_prob))

    # Correction applied if ensemble meaningfully differs from TSAS
    correction_applied = False
    if ens_prob is not None and abs(ens_prob - tsas_prob) > 0.08:
        correction_applied = True

    # Build reasoning string
    parts = [
        f"BMAv1 day={day_idx} htc={hours_to_close:.1f}h",
        f"bin=[{bin_low_c:.1f},{bin_high_c:.1f}]C",
        f"ens={ens_prob:.2%}" if ens_prob is not None else "ens=N/A",
        f"(n={n_total_members}:{n_gefs}gefs+{n_icon}icon)",
        f"bma={bma_prob:.2%}",
        f"(n_det={len(model_means)})",
        f"tsas={tsas_prob:.2%}",
        f"w=[ens={blend_w['ensemble']:.2f},bma={blend_w['bma']:.2f},tsas={blend_w['tsas']:.2f}]",
        f"final={final_prob:.2%}",
        f"spread={model_spread_c:.2f}C",
        f"corr={'Y' if correction_applied else 'N'}",
    ]
    reasoning = " ".join(parts)

    return BmaResult(
        probability=final_prob,
        ensemble_prob=ens_prob if ens_prob is not None else -1.0,
        bma_prob=bma_prob,
        tsas_prob=tsas_prob,
        n_gefs_members=n_gefs,
        n_icon_members=n_icon,
        n_det_models=len(model_means),
        model_spread_c=model_spread_c,
        blend_weights=blend_w,
        correction_applied=correction_applied,
        reasoning=reasoning,
    )


# ---------------------------------------------------------------------------
# Convenience: compute BMA for a Fahrenheit bucket (handles unit conversion)
# ---------------------------------------------------------------------------

def compute_bma_probability_fahrenheit(
    forecast_daily: dict,
    day_idx: int,
    bin_low_f: float,
    bin_high_f: float,
    use_max: bool,
    hours_to_close: float,
    tsas_prob: float,
) -> BmaResult:
    """Same as compute_bma_probability but with Fahrenheit bucket bounds."""
    bin_low_c  = (bin_low_f  - 32.0) * 5.0 / 9.0
    bin_high_c = (bin_high_f - 32.0) * 5.0 / 9.0
    return compute_bma_probability(
        forecast_daily, day_idx, bin_low_c, bin_high_c,
        use_max, hours_to_close, tsas_prob
    )


# ---------------------------------------------------------------------------
# Confidence scoring
# ---------------------------------------------------------------------------

def bma_confidence(result: BmaResult, taf_inflation: float = 1.0) -> float:
    """
    Compute confidence score [0, 1] based on:
    - Ensemble size (more members = more confidence)
    - Model spread (tighter = more confidence)
    - Agreement between layers (ensemble vs BMA vs TSAS)
    - TAF inflation penalty
    """
    # Base from ensemble size
    if result.n_gefs_members + result.n_icon_members >= 40:
        ens_conf = 0.90
    elif result.n_gefs_members + result.n_icon_members >= 20:
        ens_conf = 0.75
    elif result.n_gefs_members + result.n_icon_members >= 10:
        ens_conf = 0.60
    else:
        ens_conf = 0.35

    # Penalty from model spread
    spread_penalty = min(0.30, result.model_spread_c / 10.0)

    # Agreement penalty: if layers disagree strongly
    probs = [p for p in [result.ensemble_prob, result.bma_prob, result.tsas_prob] if p >= 0]
    if len(probs) >= 2:
        layer_spread = max(probs) - min(probs)
        agreement_penalty = min(0.25, layer_spread * 0.8)
    else:
        agreement_penalty = 0.10

    # TAF inflation penalty
    taf_penalty = min(0.20, (taf_inflation - 1.0) * 0.15)

    confidence = ens_conf - spread_penalty - agreement_penalty - taf_penalty
    return max(0.05, min(1.0, confidence))
