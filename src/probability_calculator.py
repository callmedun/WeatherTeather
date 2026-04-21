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

    Handles formats such as:
      - "Will the high be between 72°F and 73°F on April 20?"
      - "Will the high be between 20 and 21°C on April 20?"
      - "between 72 and 73 degrees Fahrenheit"
      - "between 20-21°C"

    Returns:
        (bin_low, bin_high, unit) — unit is 'F' or 'C'
        None if parsing fails.
    """
    q_lower = question.lower()

    # --- 1. Detect temperature unit ---
    if '°f' in q_lower or 'fahrenheit' in q_lower:
        unit = 'F'
    elif '°c' in q_lower or 'celsius' in q_lower:
        unit = 'C'
    else:
        # Heuristic: if both bounds < 50 it's almost certainly Celsius
        preview = re.search(r'between\s*([\d.]+)', question, re.IGNORECASE)
        if preview:
            try:
                unit = 'C' if float(preview.group(1)) < 50 else 'F'
            except ValueError:
                unit = 'F'
        else:
            unit = 'F'

    # --- 2. Try multiple "between X and Y" patterns ---
    # Pattern A: "between 72°F and 73°F"  /  "between 20 and 21°C"
    m = re.search(
        r'between\s+([\d.]+)\s*(?:°[FCfc])?\s+and\s+([\d.]+)',
        question, re.IGNORECASE
    )
    if m:
        try:
            return float(m.group(1)), float(m.group(2)), unit
        except ValueError:
            pass

    # Pattern B: "between 72-73" or "between 20–21"
    m = re.search(r'between\s+([\d.]+)\s*[-–]\s*([\d.]+)', question, re.IGNORECASE)
    if m:
        try:
            return float(m.group(1)), float(m.group(2)), unit
        except ValueError:
            pass

    # Pattern C: "72 to 73 degrees" / "72 to 73°F"
    m = re.search(
        r'([\d.]+)\s+to\s+([\d.]+)\s*(?:degrees?\s*)?(?:°[FCfc])?',
        question, re.IGNORECASE
    )
    if m:
        try:
            return float(m.group(1)), float(m.group(2)), unit
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
