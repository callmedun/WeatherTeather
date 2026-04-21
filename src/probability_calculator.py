"""
Pure mathematical probability calculator for Polymarket weather temperature markets.

Uses a normal distribution over weather ensemble data.
Fully deterministic — same inputs always produce the same output. No AI involved.
"""

import re
import statistics
from typing import Optional
from scipy.stats import norm

from src.utils import logger


def celsius_to_fahrenheit(c: float) -> float:
    """Convert Celsius to Fahrenheit."""
    return c * 9 / 5 + 32


def parse_temperature_bin(question: str) -> Optional[tuple[float, float, str]]:
    """
    Parse bin boundaries and temperature unit from a Polymarket market question.

    Supported formats:
      - "between X and Y °F/°C"       → bin [X, Y]
      - "between X-Y"                  → bin [X, Y]
      - "X to Y degrees"               → bin [X, Y]
      - "be X°C or higher/above/more"  → bin [X, +999]  (open upper)
      - "be X°C or lower/below/less"   → bin [-999, X+1] (open lower)
      - "be X°C" (exact point)         → bin [X, X+1]
      - "above X°" / "higher than X°"  → bin [X, +999]
      - "below X°" / "lower than X°"   → bin [-999, X]

    Returns:
        (bin_low, bin_high, unit) — unit is 'F' or 'C'
        None if all patterns fail.
    """
    q_lower = question.lower()

    # --- 1. Detect temperature unit ---
    if '°f' in q_lower or 'fahrenheit' in q_lower:
        unit = 'F'
    elif '°c' in q_lower or 'celsius' in q_lower:
        unit = 'C'
    else:
        # Heuristic: first number < 50 → likely Celsius
        preview = re.search(r'([\d.]+)', question)
        if preview:
            try:
                unit = 'C' if float(preview.group(1)) < 50 else 'F'
            except ValueError:
                unit = 'F'
        else:
            unit = 'F'

    # --- 2. Range patterns: "between X and Y" ---
    m = re.search(
        r'between\s+([\d.]+)\s*(?:°[FCfc])?\s+and\s+([\d.]+)',
        question, re.IGNORECASE
    )
    if m:
        try:
            return float(m.group(1)), float(m.group(2)), unit
        except ValueError:
            pass

    # "between X-Y" or "between X–Y"
    m = re.search(r'between\s+([\d.]+)\s*[-–]\s*([\d.]+)', question, re.IGNORECASE)
    if m:
        try:
            return float(m.group(1)), float(m.group(2)), unit
        except ValueError:
            pass

    # "X to Y degrees"
    m = re.search(
        r'([\d.]+)\s+to\s+([\d.]+)\s*(?:degrees?\s*)?(?:°[FCfc])?',
        question, re.IGNORECASE
    )
    if m:
        try:
            return float(m.group(1)), float(m.group(2)), unit
        except ValueError:
            pass

    # --- 3. Open-upper: "be X or higher/above/more" → [X, +inf) ---
    m = re.search(
        r'be\s+([\d.]+)\s*(?:°[FCfc])?\s+or\s+(?:higher|above|more)',
        question, re.IGNORECASE
    )
    if m:
        try:
            return float(m.group(1)), 999.0, unit
        except ValueError:
            pass

    # --- 4. Open-lower: "be X or lower/below/less" → (-inf, X] ---
    m = re.search(
        r'be\s+([\d.]+)\s*(?:°[FCfc])?\s+or\s+(?:lower|below|less)',
        question, re.IGNORECASE
    )
    if m:
        try:
            return -999.0, float(m.group(1)) + 1.0, unit
        except ValueError:
            pass

    # --- 5. Exact point: "be X°C" or "be X°F" → treat as [X, X+1) ---
    # Must come AFTER open-upper/lower patterns to avoid false matches
    m = re.search(r'be\s+([\d.]+)\s*°[FCfc]', question, re.IGNORECASE)
    if m:
        try:
            t = float(m.group(1))
            return t, t + 1.0, unit
        except ValueError:
            pass

    # --- 6. Directional: "above X" / "higher than X" → open upper ---
    m = re.search(
        r'(?:above|higher\s+than)\s+([\d.]+)\s*(?:°[FCfc])?',
        question, re.IGNORECASE
    )
    if m:
        try:
            return float(m.group(1)), 999.0, unit
        except ValueError:
            pass

    # --- 7. Directional: "below X" / "lower than X" → open lower ---
    m = re.search(
        r'(?:below|lower\s+than)\s+([\d.]+)\s*(?:°[FCfc])?',
        question, re.IGNORECASE
    )
    if m:
        try:
            return -999.0, float(m.group(1)), unit
        except ValueError:
            pass

    logger.debug(f"[MATH] Cannot parse temperature bin from: '{question[:80]}'")
    return None


def extract_metar_high_c(metar_list: list) -> Optional[float]:
    """
    Extract the maximum observed temperature (°C) from a list of METAR observations.

    Aviation Weather METAR JSON has a 'temp' field in Celsius.
    We take the maximum across all available observations (up to 24h).

    Returns:
        Max temperature in °C, or None if no valid data found.
    """
    temps = []
    for obs in metar_list:
        temp = obs.get('temp')
        if temp is not None:
            try:
                temps.append(float(temp))
            except (ValueError, TypeError):
                pass

    return max(temps) if temps else None


def calculate_bin_probability(
    ensemble_mean: float,       # °F or °C  — must match bin units
    ensemble_std: float,        # same unit — standard deviation of ensemble spread
    historical_bias: float,     # same unit — positive = model runs warm (default 0.0)
    metar_current_high: float,  # same unit — current observed daily maximum
    bin_low: float,             # lower bound of temperature bin (inclusive)
    bin_high: float,            # upper bound of temperature bin (exclusive)
    calibration_factor: float = 1.0,
) -> float:
    """
    Pure mathematical probability that the daily high temperature falls
    within [bin_low, bin_high], using a bias-corrected normal distribution.

    Formula:
        adjusted_mean = ensemble_mean - historical_bias
                        + (metar_current_high - ensemble_mean) * 0.4
        raw_prob      = norm.cdf(bin_high, adjusted_mean, ensemble_std)
                        - norm.cdf(bin_low, adjusted_mean, ensemble_std)
        result        = clip(raw_prob * calibration_factor, 0.01, 0.99)

    The 0.4 METAR blending weight nudges the forecast toward the current
    observation without fully abandoning the model consensus.

    Args:
        ensemble_mean:      Blended model temperature forecast
        ensemble_std:       Ensemble spread (uncertainty); minimum clamped to 0.1
        historical_bias:    Known warm/cold bias of the models at this station
        metar_current_high: Observed daily max from aviation weather METAR
        bin_low / bin_high: Temperature bin boundaries
        calibration_factor: Historical accuracy multiplier (from SelfCalibration)

    Returns:
        Probability in [0.01, 0.99]
    """
    ensemble_std = max(0.1, ensemble_std)  # Prevent degenerate distribution

    adjusted_mean = (
        ensemble_mean
        - historical_bias
        + (metar_current_high - ensemble_mean) * 0.4
    )

    raw_prob = (
        norm.cdf(bin_high, loc=adjusted_mean, scale=ensemble_std)
        - norm.cdf(bin_low, loc=adjusted_mean, scale=ensemble_std)
    )

    return max(0.01, min(0.99, raw_prob * calibration_factor))
