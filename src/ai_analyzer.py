"""
Mathematical AI Analyzer — drop-in replacement for the Gemini-based analyzer.

Uses a three-source blended normal distribution:
  - Open-Meteo ensemble (ECMWF + GFS + 51-member consensus)
  - TAF (Terminal Aerodrome Forecast — official airport temperature ceiling)
  - METAR (current observations — weighted by hours until resolution)

Fully deterministic: same inputs → same output. No AI involved.

Public interface identical to original Gemini AIAnalyzer.
"""

from datetime import datetime, timezone
from typing import Optional

from config.settings import config
from src.utils import logger
from src.calibration import calibration_engine
from src.probability_calculator import (
    calculate_bin_probability,
    parse_temperature_bin,
    extract_metar_high_c,
    extract_taf_max_c,
    extract_taf_min_c,
    celsius_to_fahrenheit,
)


def _hours_to_close(market: dict) -> float:
    """Compute hours until market resolution from any date field available."""
    res = (
        market.get("resolution_date")
        or market.get("end_date_iso")
        or market.get("endDateIso")
        or market.get("end_date")
    )
    if not res:
        return 24.0
    try:
        if isinstance(res, (int, float)):
            target = datetime.fromtimestamp(float(res), timezone.utc)
        else:
            target = datetime.fromisoformat(str(res).replace("Z", "+00:00"))
        delta = (target - datetime.now(timezone.utc)).total_seconds() / 3600
        return max(0.1, delta)
    except Exception:
        return 24.0


def _day_idx(hours_to_close: float) -> int:
    """
    Return the forecast day index (0=today, 1=tomorrow, 2=D+2, 3=D+3)
    from hours until market closes.

    Boundaries (aligned with typical daily-cycle data):
        0-18h  → today    (day 0)  — METAR + current observations are very relevant
        18-42h → tomorrow (day 1)  — TAF + ensemble dominate
        42-66h → D+2      (day 2)  — ensemble+GFS, no METAR weight
        66h+   → D+3      (day 3)  — ensemble only
    """
    if hours_to_close <= 18:
        return 0
    elif hours_to_close <= 42:
        return 1
    elif hours_to_close <= 66:
        return 2
    return 3


def _safe_day(arr: list, idx: int) -> "float | None":
    """Safely get day idx from a forecast array; None if missing."""
    if arr and idx < len(arr) and arr[idx] is not None:
        return float(arr[idx])
    # Walk backwards to find nearest available day
    for j in range(idx - 1, -1, -1):
        if arr and j < len(arr) and arr[j] is not None:
            return float(arr[j])
    return None


class AIAnalyzer:
    """
    Drop-in replacement for the original Gemini-based AIAnalyzer.

    Backward-compatible attributes:
        clients              — always empty (no API keys needed)
        consecutive_failures — reset to 0 on each successful batch
    """

    def __init__(self):
        self.clients = []
        self.consecutive_failures = 0
        logger.info("[MATH] Initialized: TAF + METAR + Ensemble blended probability model.")

    async def analyze_city_batch(
        self,
        city: str,
        markets: list[dict],
        weather_data: dict,
        return_all: bool = False,
    ) -> list[dict]:
        """
        Calculate probabilities for all markets in a city.

        Args:
            city:         City name (e.g. "Chicago")
            markets:      Market dicts from MarketDiscoverer
            weather_data: Dict with metar, taf, forecast_daily, etc.
            return_all:   False = scan mode (BUY signals only)
                          True  = monitor mode (all outcomes with predicted_prob)

        Returns:
            List of signal dicts compatible with the rest of the pipeline.
        """
        try:
            # ── 1. Pull per-source daily forecast arrays ─────────────────────
            forecast_daily = weather_data.get("forecast_daily", {})
            ecmwf_arr  = forecast_daily.get("ecmwf",        [])
            gfs_arr    = forecast_daily.get("gfs",           [])
            ens_arr    = forecast_daily.get("ensemble",      [])
            std_arr    = forecast_daily.get("ensemble_std",  [])

            # Require at least one model has day-0 data
            has_day0 = any(
                arr and len(arr) > 0 and arr[0] is not None
                for arr in [ecmwf_arr, gfs_arr, ens_arr]
            )
            if not has_day0:
                logger.warning(
                    f"[MATH] {city}: No numerical forecast data available — skipping."
                )
                return []

            # ── 2. METAR current high (city-level, same for all markets) ──────
            metar_list   = weather_data.get("metar", [])
            metar_high_c = extract_metar_high_c(metar_list)

            # ── 3. TAF — airport temperature forecast ─────────────────────────
            taf_list  = weather_data.get("taf", [])
            taf_max_c = extract_taf_max_c(taf_list)   # TX
            taf_min_c = extract_taf_min_c(taf_list)   # TN

            calibration_factor = calibration_engine.calculate_calibration_factor(city)
            historical_bias_c  = 0.0

            ev_threshold = config.ev_threshold.get(
                city, config.ev_threshold.get("default", 0.08)
            )

            # ── 4. Process each market ────────────────────────────────────────
            final_signals = []

            for m in markets:
                question = m.get("question", "")
                parsed   = parse_temperature_bin(question)
                if parsed is None:
                    continue

                bin_low, bin_high, unit = parsed
                is_low_market = "lowest" in question.lower() or "minimum" in question.lower()
                htc     = _hours_to_close(m)
                day_idx = _day_idx(htc)

                # ── 4a. Pick the correct forecast day for each model ──────────
                ecmwf_c = _safe_day(ecmwf_arr, day_idx)
                gfs_c   = _safe_day(gfs_arr,   day_idx)
                ens_c   = _safe_day(ens_arr,   day_idx)
                ens_std = _safe_day(std_arr,   day_idx)

                model_means = [v for v in [ens_c, ecmwf_c, gfs_c] if v is not None]
                if not model_means:
                    continue  # no forecast at all for this day

                blended_mean_c = sum(model_means) / len(model_means)

                # ── 4b. Uncertainty (grows with forecast horizon) ─────────────
                # Base: 0.5°C same-day, +0.5°C per day out
                base_std = 0.5 + day_idx * 0.5
                if ens_std is not None:
                    # Real 51-member ensemble spread for this specific day
                    std_c = max(base_std, ens_std)
                elif len(model_means) >= 2:
                    # Use inter-model spread + growing base
                    model_spread = max(abs(v - blended_mean_c) for v in model_means)
                    std_c = max(base_std + day_idx * 0.5, model_spread)
                else:
                    # Single model only
                    std_c = base_std + day_idx * 0.5 + 1.0

                # ── 4c. METAR relevance at this horizon ───────────────────────
                # Beyond 30h, today's observed high is not meaningful
                metar_high_for_calc = metar_high_c if htc <= 30 else blended_mean_c
                if metar_high_for_calc is None:
                    metar_high_for_calc = blended_mean_c

                # Convert all Celsius values to match the market's unit
                if unit == "F":
                    em  = celsius_to_fahrenheit(blended_mean_c)
                    es  = std_c * 9 / 5
                    mh  = celsius_to_fahrenheit(metar_high_for_calc)
                    hb  = 0.0
                    taf_val = celsius_to_fahrenheit(taf_min_c) if (is_low_market and taf_min_c is not None) else (
                              celsius_to_fahrenheit(taf_max_c) if (not is_low_market and taf_max_c is not None) else None)
                else:
                    em  = blended_mean_c
                    es  = std_c
                    mh  = metar_high_for_calc
                    hb  = historical_bias_c
                    taf_val = taf_min_c if is_low_market else taf_max_c

                true_prob_yes = calculate_bin_probability(
                    ensemble_mean     = em,
                    ensemble_std      = es,
                    historical_bias   = hb,
                    metar_current_high= mh,
                    bin_low           = bin_low,
                    bin_high          = bin_high,
                    calibration_factor= calibration_factor,
                    taf_max           = taf_val,
                    hours_to_close    = htc,
                )

                uncertainty_score = min(1.0, round(es / 5.0, 2))

                # Human-readable formula trace (now includes day index)
                taf_str = f"TAF={'min' if is_low_market else 'max'}={taf_val:.1f}°{unit}" if taf_val is not None else "TAF=N/A"
                reasoning = (
                    f"day={day_idx} ens={em:.1f}°{unit} std={es:.2f} "
                    f"metar={mh:.1f}°{unit} {taf_str} "
                    f"htc={htc:.1f}h bin=[{bin_low:.1f},{bin_high:.1f}]°{unit} "
                    f"YES={true_prob_yes:.3f}"
                )

                if return_all:
                    # Monitor mode: emit one record per outcome
                    for out in m.get("outcomes", []):
                        out_name = out["name"]
                        p = (
                            true_prob_yes
                            if out_name.lower() == "yes"
                            else (1.0 - true_prob_yes)
                        )
                        final_signals.append({
                            "market_id":         m["market_id"],
                            "question":          question,
                            "token_id":          out["token_id"],
                            "outcome_name":      out_name,
                            "outcome_slug":      out_name,
                            "predicted_prob":    p,
                            "city":              city,
                            "reasoning":         reasoning,
                            "uncertainty_score": uncertainty_score,
                        })

                else:
                    # Scan mode: find best-EV BUY signal for this market
                    best_signal = None
                    best_ev     = -999.0

                    for out in m.get("outcomes", []):
                        out_name     = out["name"]
                        market_price = out.get("current_price", 0.5)

                        p = (
                            true_prob_yes
                            if out_name.lower() == "yes"
                            else (1.0 - true_prob_yes)
                        )

                        edge         = p - market_price
                        ev           = (p * (1 - market_price)) - ((1 - p) * market_price)
                        odds         = (1 - market_price) / market_price if market_price > 0 else 0
                        full_kelly   = (edge / odds) if odds > 0 else 0
                        frac_kelly   = (
                            full_kelly * config.kelly_fraction
                            if full_kelly > 0 else 0.0
                        )

                        if ev > ev_threshold and frac_kelly > 0 and ev > best_ev:
                            best_ev = ev
                            sentiment = "BULLISH" if out_name.lower() == "yes" else "BEARISH"
                            best_signal = {
                                "market_id":         m["market_id"],
                                "question":          question,
                                "token_id":          out["token_id"],
                                "outcome_name":      out["name"],
                                "outcome_slug":      out_name,
                                "market_price":      market_price,
                                "true_probability":  p,
                                "predicted_prob":    p,
                                "ev":                ev,
                                "edge":              edge * 100,
                                "kelly":             frac_kelly,
                                "confidence":        int(min(99, (1.0 - uncertainty_score) * 100)),
                                "uncertainty_score": uncertainty_score,
                                "sentiment":         sentiment,
                                "city":              city,
                                "icao_code":         m.get("icao_code", ""),
                                "reasoning":         reasoning,
                            }

                    if best_signal is not None:
                        final_signals.append(best_signal)

            self.consecutive_failures = 0

            if not return_all:
                taf_status = f"TAF={taf_max_c:.1f}°C" if taf_max_c else "TAF=N/A"
                # Log a summary of what day-index range we saw
                day_range = set(
                    _day_idx(_hours_to_close(m)) for m in markets
                )
                logger.info(
                    f"[MATH] {city}: {len(markets)} markets → "
                    f"{len(final_signals)} BUY signals "
                    f"[{taf_status} | days={sorted(day_range)}]"
                )

            return final_signals

        except Exception as e:
            logger.error(f"[MATH] Batch analysis error for {city}: {e}")
            self.consecutive_failures += 1
            return []


ai_analyzer = AIAnalyzer()
