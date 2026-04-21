"""
Mathematical AI Analyzer — drop-in replacement for the Gemini-based analyzer.

Uses scipy normal distribution over weather ensemble data.
Fully deterministic: same inputs → same output every time.

Public interface is identical to the original Gemini AIAnalyzer so that
scheduler.py and portfolio_manager.py require no changes.
"""

from config.settings import config
from src.utils import logger
from src.calibration import calibration_engine
from src.probability_calculator import (
    calculate_bin_probability,
    parse_temperature_bin,
    extract_metar_high_c,
    celsius_to_fahrenheit,
)


class AIAnalyzer:
    """
    Drop-in replacement for the original Gemini-based AIAnalyzer.

    Attributes kept for backward compatibility with scheduler status reporting:
        clients            — always empty list (no API keys needed)
        consecutive_failures — reset to 0 on each successful batch
    """

    def __init__(self):
        self.clients = []           # No API clients — kept for scheduler compat
        self.consecutive_failures = 0
        logger.info("[MATH] Initialized mathematical probability analyzer (Gemini disabled).")

    async def analyze_city_batch(
        self,
        city: str,
        markets: list[dict],
        weather_data: dict,
        return_all: bool = False,
    ) -> list[dict]:
        """
        Calculate probabilities for all markets of a city using the normal
        distribution model over ensemble weather data.

        Args:
            city:         City name (e.g. "Chicago")
            markets:      List of market dicts from MarketDiscoverer
            weather_data: Dict with METAR list + Open-Meteo numerical fields
            return_all:   False = scan mode (returns BUY signals only)
                          True  = monitor mode (returns all outcomes with predicted_prob)

        Returns:
            List of signal dicts — same format as the original Gemini analyzer.
        """
        try:
            # ── 1. Extract numerical weather data ───────────────────────────
            ensemble_mean_c = weather_data.get("ensemble_mean_c")
            ensemble_std_c  = weather_data.get("ensemble_std_c")
            gfs_mean_c      = weather_data.get("gfs_mean_c")
            ecmwf_mean_c    = weather_data.get("ecmwf_mean_c")

            # Require at least one model forecast
            model_means = [
                m for m in [ensemble_mean_c, gfs_mean_c, ecmwf_mean_c]
                if m is not None
            ]
            if not model_means:
                logger.warning(f"[MATH] {city}: No numerical forecast data available — skipping.")
                return []

            # Blend available model means (equal weight)
            blended_mean_c = sum(model_means) / len(model_means)

            # Ensemble std: prefer real ensemble spread, fall back to model spread
            if ensemble_std_c is not None and ensemble_std_c > 0.1:
                std_c = ensemble_std_c
            elif len(model_means) >= 2:
                # Estimate from max model spread as a rough proxy
                std_c = max(abs(m - blended_mean_c) for m in model_means) or 2.0
            else:
                std_c = 2.0  # Default: 2 °C — reasonable short-term uncertainty

            # METAR current observed high (fallback: use blended model mean)
            metar_high_c = extract_metar_high_c(weather_data.get("metar", []))
            if metar_high_c is None:
                metar_high_c = blended_mean_c

            calibration_factor = calibration_engine.calculate_calibration_factor(city)
            historical_bias_c  = 0.0  # No historical bias data yet

            ev_threshold = config.ev_threshold.get(
                city, config.ev_threshold.get("default", 0.08)
            )

            # ── 2. Process each market ───────────────────────────────────────
            final_signals = []

            for m in markets:
                question = m.get("question", "")
                parsed = parse_temperature_bin(question)
                if parsed is None:
                    # If we can't determine the bin boundaries, skip safely
                    continue

                bin_low, bin_high, unit = parsed

                # Convert Celsius weather data to match the market's unit
                if unit == "F":
                    em = celsius_to_fahrenheit(blended_mean_c)
                    es = std_c * 9 / 5          # Scale std (no offset)
                    mh = celsius_to_fahrenheit(metar_high_c)
                    hb = 0.0                    # 0 °C bias = 0 °F difference
                else:
                    em = blended_mean_c
                    es = std_c
                    mh = metar_high_c
                    hb = historical_bias_c

                true_prob_yes = calculate_bin_probability(
                    em, es, hb, mh, bin_low, bin_high, calibration_factor
                )

                # Uncertainty proxy: higher ensemble spread → higher uncertainty
                uncertainty_score = min(1.0, round(es / 5.0, 2))

                adj_mean = em - hb + (mh - em) * 0.4
                reasoning = (
                    f"adj_mean={adj_mean:.1f}°{unit}, "
                    f"bin=[{bin_low:.1f}-{bin_high:.1f}]°{unit}, "
                    f"std={es:.2f}°{unit}, "
                    f"metar_high={mh:.1f}°{unit}, "
                    f"YES_prob={true_prob_yes:.3f}"
                )

                if return_all:
                    # ── Monitor mode: all outcomes with fresh predicted_prob ──
                    for out in m.get("outcomes", []):
                        out_name = out["name"]
                        p = (
                            true_prob_yes
                            if out_name.lower() == "yes"
                            else (1.0 - true_prob_yes)
                        )
                        final_signals.append({
                            "market_id":        m["market_id"],
                            "question":         question,
                            "token_id":         out["token_id"],
                            "outcome_name":     out_name,
                            "outcome_slug":     out_name,
                            "predicted_prob":   p,
                            "city":             city,
                            "reasoning":        reasoning,
                            "uncertainty_score": uncertainty_score,
                        })

                else:
                    # ── Scan mode: find best-EV BUY signal for this market ───
                    best_signal = None
                    best_ev = -999.0

                    for out in m.get("outcomes", []):
                        out_name     = out["name"]
                        market_price = out.get("current_price", 0.5)

                        p = (
                            true_prob_yes
                            if out_name.lower() == "yes"
                            else (1.0 - true_prob_yes)
                        )

                        edge  = p - market_price
                        ev    = (p * (1 - market_price)) - ((1 - p) * market_price)
                        odds  = (1 - market_price) / market_price if market_price > 0 else 0
                        full_kelly       = (edge / odds) if odds > 0 else 0
                        fractional_kelly = (
                            full_kelly * config.kelly_fraction
                            if full_kelly > 0
                            else 0.0
                        )

                        if ev > ev_threshold and fractional_kelly > 0 and ev > best_ev:
                            best_ev = ev
                            sentiment = "BULLISH" if out_name.lower() == "yes" else "BEARISH"
                            best_signal = {
                                "market_id":        m["market_id"],
                                "question":         question,
                                "token_id":         out["token_id"],
                                "outcome_name":     out["name"],
                                "outcome_slug":     out_name,
                                "market_price":     market_price,
                                "true_probability": p,
                                "predicted_prob":   p,
                                "ev":               ev,
                                "edge":             edge * 100,
                                "kelly":            fractional_kelly,
                                "confidence":       int(min(99, (1.0 - uncertainty_score) * 100)),
                                "uncertainty_score": uncertainty_score,
                                "sentiment":        sentiment,
                                "city":             city,
                                "icao_code":        m.get("icao_code", ""),
                                "reasoning":        reasoning,
                            }

                    if best_signal is not None:
                        final_signals.append(best_signal)

            self.consecutive_failures = 0

            if not return_all:
                logger.info(
                    f"[MATH] {city}: {len(markets)} markets → "
                    f"{len(final_signals)} BUY signals"
                )

            return final_signals

        except Exception as e:
            logger.error(f"[MATH] Batch analysis error for {city}: {e}")
            self.consecutive_failures += 1
            return []


ai_analyzer = AIAnalyzer()
