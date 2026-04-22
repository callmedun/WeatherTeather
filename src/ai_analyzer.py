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
            weather_data: Dict with metar, taf, ensemble_mean_c, etc.
            return_all:   False = scan mode (BUY signals only)
                          True  = monitor mode (all outcomes with predicted_prob)

        Returns:
            List of signal dicts compatible with the rest of the pipeline.
        """
        try:
            # ── 1. Extract numerical Open-Meteo forecasts ────────────────────
            ensemble_mean_c = weather_data.get("ensemble_mean_c")
            ensemble_std_c  = weather_data.get("ensemble_std_c")
            gfs_mean_c      = weather_data.get("gfs_mean_c")
            ecmwf_mean_c    = weather_data.get("ecmwf_mean_c")

            model_means = [
                v for v in [ensemble_mean_c, gfs_mean_c, ecmwf_mean_c]
                if v is not None
            ]
            if not model_means:
                logger.warning(
                    f"[MATH] {city}: No numerical forecast data available — skipping."
                )
                return []

            blended_mean_c = sum(model_means) / len(model_means)

            if ensemble_std_c is not None and ensemble_std_c > 0.1:
                std_c = ensemble_std_c
            elif len(model_means) >= 2:
                std_c = max(abs(v - blended_mean_c) for v in model_means) or 2.0
            else:
                std_c = 2.0

            # ── 2. METAR — latest observed high ──────────────────────────────
            metar_list   = weather_data.get("metar", [])
            metar_high_c = extract_metar_high_c(metar_list)
            if metar_high_c is None:
                metar_high_c = blended_mean_c

            # ── 3. TAF — official airport temperature forecast ───────────────
            taf_list    = weather_data.get("taf", [])
            taf_max_c   = extract_taf_max_c(taf_list)   # TX — daily high forecast
            taf_min_c   = extract_taf_min_c(taf_list)   # TN — overnight low

            taf_log = f"TX={taf_max_c:.1f}°C" if taf_max_c else "TX=N/A"
            logger.debug(
                f"[MATH] {city} weather: "
                f"ens={blended_mean_c:.1f}°C std={std_c:.2f} "
                f"metar_high={metar_high_c:.1f}°C {taf_log}"
            )

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

                # Detect if market is about LOWEST temperature (min)
                is_low_market = "lowest" in question.lower() or "minimum" in question.lower()

                # hours until this specific market closes
                htc = _hours_to_close(m)

                # Convert all Celsius values to match the market's unit
                if unit == "F":
                    em  = celsius_to_fahrenheit(blended_mean_c)
                    es  = std_c * 9 / 5
                    mh  = celsius_to_fahrenheit(metar_high_c)
                    hb  = 0.0
                    # For TAF: if low market use TN, else TX
                    if is_low_market:
                        taf_val = celsius_to_fahrenheit(taf_min_c) if taf_min_c is not None else None
                    else:
                        taf_val = celsius_to_fahrenheit(taf_max_c) if taf_max_c is not None else None
                else:
                    em  = blended_mean_c
                    es  = std_c
                    mh  = metar_high_c
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

                # Human-readable formula trace
                taf_str = f"TAF={'min' if is_low_market else 'max'}={taf_val:.1f}°{unit}" if taf_val is not None else "TAF=N/A"
                reasoning = (
                    f"ens={em:.1f}°{unit} std={es:.2f}°{unit} "
                    f"metar_high={mh:.1f}°{unit} {taf_str} "
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
                logger.info(
                    f"[MATH] {city}: {len(markets)} markets → "
                    f"{len(final_signals)} BUY signals [{taf_status}]"
                )

            return final_signals

        except Exception as e:
            logger.error(f"[MATH] Batch analysis error for {city}: {e}")
            self.consecutive_failures += 1
            return []


ai_analyzer = AIAnalyzer()
