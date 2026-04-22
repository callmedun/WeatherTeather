"""
Pure mathematical probability calculator for Polymarket weather temperature markets.

Uses a normal distribution over blended forecast sources:
  - Open-Meteo ensemble (ECMWF, GFS, 51-member)
  - TAF (Terminal Aerodrome Forecast — official airport temperature forecast)
  - METAR (current observations — weighted by how close the market is to resolution)

All weights are adjusted by hours_to_close so that:
  - Far-future markets lean on ensemble + TAF
  - Same-day markets lean heavily on live METAR observations
"""

import re
from typing import Optional
from scipy.stats import norm

from src.utils import logger


# ---------------------------------------------------------------------------
# Unit conversion
# ---------------------------------------------------------------------------

def celsius_to_fahrenheit(c: float) -> float:
    """Convert Celsius to Fahrenheit."""
    return c * 9 / 5 + 32


# ---------------------------------------------------------------------------
# Question parser — all Polymarket temperature market formats
# ---------------------------------------------------------------------------

def parse_temperature_bin(question: str) -> Optional[tuple[float, float, str]]:
    """
    Parse bin boundaries and temperature unit from a Polymarket market question.

    Supported formats:
      - "between X and Y °F/°C"       → bin [X, Y]
      - "between X-Y"                  → bin [X, Y]
      - "X to Y degrees"               → bin [X, Y]
      - "be X°C or higher/above/more"  → bin [X, +999]  open upper
      - "be X°C or lower/below/less"   → bin [-999, X+1] open lower
      - "be X°C" (exact point)         → bin [X, X+1]
      - "above X°" / "higher than X°"  → bin [X, +999]
      - "below X°" / "lower than X°"   → bin [-999, X]

    Returns:
        (bin_low, bin_high, unit) where unit is 'F' or 'C',
        or None if all patterns fail.
    """
    q_lower = question.lower()

    # --- 1. Detect temperature unit ---
    if '°f' in q_lower or 'fahrenheit' in q_lower:
        unit = 'F'
    elif '°c' in q_lower or 'celsius' in q_lower:
        unit = 'C'
    else:
        preview = re.search(r'([\d.]+)', question)
        if preview:
            try:
                unit = 'C' if float(preview.group(1)) < 50 else 'F'
            except ValueError:
                unit = 'F'
        else:
            unit = 'F'

    # --- 2. Range: "between X and Y" ---
    m = re.search(
        r'between\s+([\d.]+)\s*(?:°[FCfc])?\s+and\s+([\d.]+)',
        question, re.IGNORECASE
    )
    if m:
        try:
            return float(m.group(1)), float(m.group(2)), unit
        except ValueError:
            pass

    m = re.search(r'between\s+([\d.]+)\s*[-–]\s*([\d.]+)', question, re.IGNORECASE)
    if m:
        try:
            return float(m.group(1)), float(m.group(2)), unit
        except ValueError:
            pass

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

    # --- 5. Exact point: "be X°C" → [X, X+1) — AFTER open patterns ---
    m = re.search(r'be\s+([\d.]+)\s*°[FCfc]', question, re.IGNORECASE)
    if m:
        try:
            t = float(m.group(1))
            return t, t + 1.0, unit
        except ValueError:
            pass

    # --- 6. "above X" / "higher than X" → open upper ---
    m = re.search(
        r'(?:above|higher\s+than)\s+([\d.]+)\s*(?:°[FCfc])?',
        question, re.IGNORECASE
    )
    if m:
        try:
            return float(m.group(1)), 999.0, unit
        except ValueError:
            pass

    # --- 7. "below X" / "lower than X" → open lower ---
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


# ---------------------------------------------------------------------------
# METAR temperature extraction
# ---------------------------------------------------------------------------

def extract_metar_high_c(metar_list: list) -> Optional[float]:
    """
    Return the maximum observed temperature (°C) from recent METAR observations.
    Uses the 'temp' field which the Aviation Weather API provides in Celsius.
    """
    temps = []
    for obs in (metar_list or []):
        temp = obs.get('temp')
        if temp is not None:
            try:
                temps.append(float(temp))
            except (ValueError, TypeError):
                pass
    return max(temps) if temps else None


# ---------------------------------------------------------------------------
# TAF temperature extraction
# ---------------------------------------------------------------------------

def extract_taf_max_c(taf_list: list) -> Optional[float]:
    """
    Extract the forecasted maximum temperature (°C) from TAF rawTAF strings.

    TAF encodes max/min temps as:
        TX18/2215Z  → max 18°C expected at day-22, 15:00 UTC
        TXM05/1512Z → max -5°C (M = Minus)

    We parse all TX entries and return the maximum forecast value found.
    The latest TAF issued is processed first (list is newest-first from API).
    """
    all_tx: list[float] = []

    for taf in (taf_list or []):
        raw = taf.get("rawTAF", "")
        if not raw:
            continue
        # Match both positive (TX18) and negative (TXM05) formats
        for hit in re.finditer(r'TX(M?)(\d+)/', raw, re.IGNORECASE):
            sign = -1 if hit.group(1).upper() == 'M' else 1
            try:
                all_tx.append(sign * float(hit.group(2)))
            except ValueError:
                pass

    if not all_tx:
        return None

    # Return the highest TX found (most optimistic ceiling for the period)
    # For markets asking "will it reach X?", the peak matters most.
    return max(all_tx)


def extract_taf_min_c(taf_list: list) -> Optional[float]:
    """
    Extract the forecasted minimum temperature (°C) from TAF TN lines.
    Useful for markets about overnight lows.
    """
    all_tn: list[float] = []
    for taf in (taf_list or []):
        raw = taf.get("rawTAF", "")
        for hit in re.finditer(r'TN(M?)(\d+)/', raw, re.IGNORECASE):
            sign = -1 if hit.group(1).upper() == 'M' else 1
            try:
                all_tn.append(sign * float(hit.group(2)))
            except ValueError:
                pass
    return min(all_tn) if all_tn else None


# ---------------------------------------------------------------------------
# Core probability calculator — now with TAF + time-weighted blending
# ---------------------------------------------------------------------------

def calculate_bin_probability(
    ensemble_mean: float,           # °same unit as bin
    ensemble_std: float,            # °same unit — standard deviation
    historical_bias: float,         # °same unit — positive = model runs warm
    metar_current_high: float,      # °same unit — observed daily maximum
    bin_low: float,
    bin_high: float,
    calibration_factor: float = 1.0,
    taf_max: Optional[float] = None,  # °same unit — TAF TX forecast
    hours_to_close: float = 24.0,     # hours until market resolves
) -> float:
    """
    Probability that daily high temperature falls in [bin_low, bin_high].

    Blending weights by forecast horizon (hours_to_close):

        ≤ 6h  (same-day):   METAR 50% + TAF 20% + Ensemble 30%
        ≤ 24h (tomorrow):   METAR 10% + TAF 50% + Ensemble 40%
        > 24h (future):     METAR  0% + TAF 30% + Ensemble 70%

    Missing sources are dropped and remaining weights renormalized.
    Bias correction applied to the blended mean before distribution lookup.

    Returns probability in [0.01, 0.99].
    """
    ensemble_std = max(0.5, ensemble_std)  # minimum uncertainty of 0.5°

    # --- Time-weighted source blending ---
    if hours_to_close <= 6:
        w_metar, w_taf, w_ens = 0.50, 0.20, 0.30
    elif hours_to_close <= 24:
        w_metar, w_taf, w_ens = 0.10, 0.50, 0.40
    else:
        w_metar, w_taf, w_ens = 0.00, 0.30, 0.70

    sources: list[tuple[float, float]] = [(ensemble_mean, w_ens)]
    if taf_max is not None:
        sources.append((taf_max, w_taf))
    if metar_current_high is not None:
        sources.append((metar_current_high, w_metar))

    total_w = sum(w for _, w in sources)
    blended = sum(v * w for v, w in sources) / total_w

    adjusted_mean = blended - historical_bias

    raw_prob = (
        norm.cdf(bin_high, loc=adjusted_mean, scale=ensemble_std)
        - norm.cdf(bin_low,  loc=adjusted_mean, scale=ensemble_std)
    )

    return max(0.01, min(0.99, raw_prob * calibration_factor))
