"""
TSAS v1 probability engine.

This module implements the first production-safe layer of the new strategy:
it keeps the bot's existing public signal format, but replaces the simple
single-normal blend with a conservative stochastic model using:
  - Open-Meteo ensemble disagreement
  - TAF TX/TN plus uncertainty markers
  - METAR same-day lower-bound and short-horizon bias pressure
  - confidence and circuit-breaker penalties

Full MOS/BMA/analog ensembles require historical training data. TSAS v1 is
designed as a compatible bridge toward that architecture without breaking the
current scheduler, Telegram bot, portfolio DB, or execution engine.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Optional

from scipy.stats import norm

from config.settings import config
from src.probability_calculator import (
    celsius_to_fahrenheit,
    extract_metar_high_c,
    extract_taf_max_c,
    extract_taf_min_c,
    parse_temperature_bin,
)
try:
    from src.bma_engine import compute_bma_probability, compute_bma_probability_fahrenheit, bma_confidence
    _BMA_AVAILABLE = True
except ImportError:
    _BMA_AVAILABLE = False
    logger_tmp = __import__('logging').getLogger(__name__)
    logger_tmp.warning("[TSAS] bma_engine not available, falling back to pure TSAS")


CONVECTIVE_MARKERS = ("TSRA", "+TSRA", "VCTS", "SHRA", "+SHRA", "CB")
COASTAL_SKY_ICAOS = {"KLAX", "KSFO", "EGLC", "NZWN"}
TROPICAL_HEAT_ICAOS = {"WSSS", "KMIA"}
CONTINENTAL_CONVECTIVE_ICAOS = {"KAUS", "KDAL", "KATL", "KORD", "ZBAA"}
TEMPERATE_STABLE_ICAOS = {"LIMC", "EDDM", "ZSPD", "RCSS"}


@dataclass
class TsasDistribution:
    mean: float
    std: float
    confidence: float
    liquidity_factor: float
    taf_inflation: float
    circuit_breaker: bool
    reasoning: str


def _extract_raw_metar_text(metar_list: list) -> str:
    parts: list[str] = []
    for item in metar_list or []:
        if isinstance(item, dict):
            raw = item.get("rawOb") or item.get("raw_text") or item.get("rawMETAR") or ""
            if raw:
                parts.append(str(raw).upper())
        elif item:
            parts.append(str(item).upper())
    return " ".join(parts)


def _latest_metar_temp_c(metar_list: list) -> Optional[float]:
    for item in metar_list or []:
        if not isinstance(item, dict):
            continue
        temp = item.get("temp")
        if temp is None:
            continue
        try:
            return float(temp)
        except (TypeError, ValueError):
            continue
    return None


def _latest_metar_dewpoint_c(metar_list: list) -> Optional[float]:
    for item in metar_list or []:
        if not isinstance(item, dict):
            continue
        for key in ("dewp", "dewpoint", "dewpoint_c", "dewpointC"):
            value = item.get(key)
            if value is None:
                continue
            try:
                return float(value)
            except (TypeError, ValueError):
                continue
    return None


def _hourly_forecast_max_c(weather_data: dict, lookahead_hours: int = 8) -> Optional[float]:
    hourly = weather_data.get("forecast_hourly", {}) or {}
    temps = hourly.get("temperature_2m", []) or []
    if not temps:
        return None
    trimmed = [float(v) for v in temps[: max(1, lookahead_hours)] if v is not None]
    if not trimmed:
        return None
    return max(trimmed)


def _hourly_average(weather_data: dict, key: str, lookahead_hours: int = 8) -> Optional[float]:
    hourly = weather_data.get("forecast_hourly", {}) or {}
    values = hourly.get(key, []) or []
    if not values:
        return None
    trimmed = [float(v) for v in values[: max(1, lookahead_hours)] if v is not None]
    if not trimmed:
        return None
    return sum(trimmed) / len(trimmed)


def fahrenheit_to_celsius(f: float) -> float:
    return (f - 32.0) * 5.0 / 9.0


def _taf_groups(taf_list: list) -> list[str]:
    raw = taf_text(taf_list)
    if not raw:
        return []
    return [group.strip() for group in re.split(r"\bFM\d{6}\b", raw) if group.strip()]


def _has_low_cloud(group: str) -> bool:
    heights = [int(height) for height in re.findall(r"\b(?:FEW|SCT|BKN|OVC)(\d{3})\b", group)]
    return any(height < 50 for height in heights)


def _is_clear_or_high_only(group: str) -> bool:
    if re.search(r"\b(?:SKC|CLR|NSC)\b", group):
        return True
    heights = [int(height) for height in re.findall(r"\b(?:FEW|SCT|BKN|OVC)(\d{3})\b", group)]
    return bool(heights) and min(heights) >= 120


def _group_has_convection(group: str) -> bool:
    upper = group.upper()
    return any(marker in upper for marker in CONVECTIVE_MARKERS)


def coastal_taf_adjustment_c(
    market: dict,
    taf_list: list,
    metar_list: list,
    hours_to_close_value: float,
) -> tuple[float, list[str]]:
    icao = str(market.get("icao_code") or "").upper()
    if icao not in COASTAL_SKY_ICAOS or hours_to_close_value > 36:
        return 0.0, []

    groups = _taf_groups(taf_list)
    if not groups:
        return 0.0, []

    low_cloud_groups = sum(1 for group in groups if _has_low_cloud(group))
    clear_high_groups = sum(1 for group in groups if _is_clear_or_high_only(group))
    bonus_c = 0.0
    tags: list[str] = []

    if clear_high_groups >= 2 and low_cloud_groups == 0:
        bonus_c += 1.8
        tags.append("COAST_CLEAR")
    elif clear_high_groups >= 1 and low_cloud_groups == 0:
        bonus_c += 1.1
        tags.append("COAST_SUN")
    elif low_cloud_groups >= 2:
        bonus_c -= 1.0
        tags.append("MARINE_LOW")
    elif low_cloud_groups >= 1:
        bonus_c -= 0.45
        tags.append("LOW_CLOUD")

    metar_raw = _extract_raw_metar_text(metar_list)
    if metar_raw:
        if re.search(r"\b(?:FEW|SCT)0(?:0\d|1\d|2\d)\b", metar_raw) and not re.search(r"\b(?:BKN|OVC)0(?:0\d|1\d|2\d)\b", metar_raw):
            bonus_c += 0.4
            tags.append("METAR_CLEARING")
        elif re.search(r"\b(?:BKN|OVC)0(?:0\d|1\d|2\d)\b", metar_raw):
            bonus_c -= 0.4
            tags.append("METAR_MARINE")

    return bonus_c, tags


def _extract_winds(text: str) -> list[tuple[Optional[int], int]]:
    winds: list[tuple[Optional[int], int]] = []
    for direction, speed in re.findall(r"\b(\d{3}|VRB)(\d{2})(?:G\d{2})?KT\b", text.upper()):
        dir_value = None if direction == "VRB" else int(direction)
        winds.append((dir_value, int(speed)))
    return winds


def _classify_coastal_wind(icao: str, direction: Optional[int], speed: int) -> Optional[str]:
    if direction is None or speed < 4:
        return None

    icao = icao.upper()
    if icao == "KLAX":
        if 20 <= direction <= 120 or direction >= 330:
            return "OFFSHORE"
        if 220 <= direction <= 310:
            return "ONSHORE"
        return None

    if icao == "KSFO":
        if 20 <= direction <= 110 or direction >= 330:
            return "OFFSHORE"
        if 220 <= direction <= 320:
            return "ONSHORE"
        return None

    if icao in {"EGLC", "NZWN"}:
        if 290 <= direction <= 360 or 0 <= direction <= 40:
            return "OFFSHORE"
        if 90 <= direction <= 180:
            return "ONSHORE"
        return None

    return None


def coastal_wind_adjustment_c(
    market: dict,
    taf_list: list,
    metar_list: list,
    hours_to_close_value: float,
) -> tuple[float, list[str]]:
    icao = str(market.get("icao_code") or "").upper()
    if icao not in {"KLAX", "KSFO", "EGLC", "NZWN"} or hours_to_close_value > 36:
        return 0.0, []

    bonus_c = 0.0
    tags: list[str] = []

    taf_raw = taf_text(taf_list)
    taf_winds = _extract_winds(taf_raw)
    if taf_winds:
        offshore_count = 0
        onshore_count = 0
        for direction, speed in taf_winds:
            regime = _classify_coastal_wind(icao, direction, speed)
            if regime == "OFFSHORE":
                offshore_count += 1
            elif regime == "ONSHORE":
                onshore_count += 1

        if offshore_count >= 2 and offshore_count > onshore_count:
            bonus_c += 0.9
            tags.append("OFFSHORE_TAF")
        elif onshore_count >= 2 and onshore_count > offshore_count:
            bonus_c -= 0.9
            tags.append("ONSHORE_TAF")

    metar_raw = _extract_raw_metar_text(metar_list)
    metar_winds = _extract_winds(metar_raw)
    if metar_winds:
        direction, speed = metar_winds[0]
        regime = _classify_coastal_wind(icao, direction, speed)
        if regime == "OFFSHORE":
            bonus_c += 0.3
            tags.append("OFFSHORE_METAR")
        elif regime == "ONSHORE":
            bonus_c -= 0.3
            tags.append("ONSHORE_METAR")

    return bonus_c, tags


def tropical_intraday_adjustment_c(
    market: dict,
    taf_list: list,
    metar_list: list,
    hours_to_close_value: float,
    low: float,
    unit: str,
    is_low_market: bool,
    baseline_mean_c: float,
) -> tuple[float, list[str]]:
    icao = str(market.get("icao_code") or "").upper()
    if icao not in TROPICAL_HEAT_ICAOS or is_low_market or not (2 <= hours_to_close_value <= 12):
        return 0.0, []

    latest_temp_c = _latest_metar_temp_c(metar_list)
    if latest_temp_c is None:
        return 0.0, []

    threshold_c = low if unit == "C" else fahrenheit_to_celsius(low)
    gap_c = threshold_c - latest_temp_c
    bonus_c = 0.0
    tags: list[str] = []

    if gap_c <= 0.25:
        bonus_c += 1.6
        tags.append("HEAT_AT_THRESHOLD")
    elif gap_c <= 1.0:
        bonus_c += 1.1
        tags.append("HEAT_NEAR_THRESHOLD")
    elif gap_c <= 2.0:
        bonus_c += 0.7
        tags.append("HEAT_RUNWAY")
    elif gap_c <= 3.0:
        bonus_c += 0.35
        tags.append("HEAT_BUILD")

    if hours_to_close_value <= 8 and gap_c <= 2.0 and latest_temp_c >= baseline_mean_c - 0.4:
        bonus_c += 0.45
        tags.append("TROPICAL_RUNUP")
    elif hours_to_close_value <= 6 and latest_temp_c >= baseline_mean_c + 0.4:
        bonus_c += 0.35
        tags.append("TROPICAL_ACCEL")

    taf_raw = taf_text(taf_list)
    if taf_raw:
        if re.search(r"\b(?:FEW|SCT)0(?:0\d|1\d|2\d)\b", taf_raw) and not re.search(r"\b(?:BKN|OVC)0(?:0\d|1\d|2\d)\b", taf_raw):
            bonus_c += 0.5
            tags.append("TROPICAL_CLEAR")
        elif re.search(r"\b(?:BKN|OVC)0(?:0\d|1\d|2\d)\b", taf_raw):
            bonus_c -= 0.45
            tags.append("TROPICAL_CLOUD")
        elif re.search(r"\b(?:BKN|OVC)0(?:3\d|4\d|5\d|6\d|7\d)\b", taf_raw):
            bonus_c -= 0.2
            tags.append("TROPICAL_MIDCLOUD")

    metar_raw = _extract_raw_metar_text(metar_list)
    if metar_raw:
        if re.search(r"\b(?:FEW|SCT)0(?:0\d|1\d|2\d)\b", metar_raw) and not re.search(r"\b(?:BKN|OVC)0(?:0\d|1\d|2\d)\b", metar_raw):
            bonus_c += 0.35
            tags.append("METAR_SUN")
        elif re.search(r"\b(?:BKN|OVC)0(?:0\d|1\d|2\d)\b", metar_raw):
            bonus_c -= 0.35
            tags.append("METAR_SHADE")

        if any(marker in taf_raw for marker in CONVECTIVE_MARKERS) and "METAR_SUN" in tags and gap_c <= 1.5:
            bonus_c += 0.25
            tags.append("POST_CONVECTION_CLEAR")

    winds = _extract_winds(metar_raw)
    if winds:
        _, speed = winds[0]
        if speed <= 8:
            bonus_c += 0.25
            tags.append("LIGHT_WIND")
        elif speed >= 16:
            bonus_c -= 0.15
            tags.append("WIND_MIX")

    dewpoint_c = _latest_metar_dewpoint_c(metar_list)
    if dewpoint_c is not None:
        dew_gap_c = latest_temp_c - dewpoint_c
        if dew_gap_c >= 6.0:
            bonus_c += 0.25
            tags.append("DRY_MIXING")
        elif dew_gap_c <= 3.0:
            bonus_c -= 0.2
            tags.append("MUGGY")

    bonus_c = max(-1.0, min(2.8, bonus_c))
    return bonus_c, tags


def same_day_threshold_adjustment_c(
    market: dict,
    metar_list: list,
    hours_to_close_value: float,
    low: float,
    unit: str,
    is_low_market: bool,
    baseline_mean_c: float,
) -> tuple[float, list[str]]:
    if is_low_market or not (1.0 <= hours_to_close_value <= float(getattr(config, "tsas_intraday_window_hours", 10.0))):
        return 0.0, []

    latest_temp_c = _latest_metar_temp_c(metar_list)
    if latest_temp_c is None:
        return 0.0, []

    threshold_c = low if unit == "C" else fahrenheit_to_celsius(low)
    gap_c = threshold_c - latest_temp_c
    bonus_c = 0.0
    tags: list[str] = []

    if gap_c <= 0.25:
        bonus_c += 1.0
        tags.append("THRESHOLD_TOUCH")
    elif gap_c <= 1.0:
        bonus_c += 0.7
        tags.append("THRESHOLD_NEAR")
    elif gap_c <= 2.0:
        bonus_c += 0.35
        tags.append("THRESHOLD_RUNUP")

    if latest_temp_c >= baseline_mean_c + 0.8:
        bonus_c += 0.35
        tags.append("METAR_HOTTER")
    elif latest_temp_c <= baseline_mean_c - 1.2:
        bonus_c -= 0.25
        tags.append("METAR_COOLER")

    return bonus_c, tags


def same_day_runrate_adjustment_c(
    metar_list: list,
    hours_to_close_value: float,
    low: float,
    unit: str,
    is_low_market: bool,
    baseline_mean_c: float,
) -> tuple[float, list[str]]:
    if is_low_market or not (1.0 <= hours_to_close_value <= 8.0):
        return 0.0, []

    latest_temp_c = _latest_metar_temp_c(metar_list)
    if latest_temp_c is None:
        return 0.0, []

    threshold_c = low if unit == "C" else fahrenheit_to_celsius(low)
    delta_vs_baseline_c = latest_temp_c - baseline_mean_c
    gap_c = threshold_c - latest_temp_c
    bonus_c = 0.0
    tags: list[str] = []

    if hours_to_close_value <= 6 and delta_vs_baseline_c >= 0.8:
        bonus_c += 0.55
        tags.append("MODEL_LAG")
    elif hours_to_close_value <= 4 and delta_vs_baseline_c >= 0.3:
        bonus_c += 0.3
        tags.append("MODEL_CATCHUP")

    if hours_to_close_value <= 6 and gap_c <= 1.5 and delta_vs_baseline_c >= -0.2:
        bonus_c += 0.35
        tags.append("EARLY_PEAK_TRACK")
    elif hours_to_close_value <= 4 and gap_c <= 0.5:
        bonus_c += 0.2
        tags.append("PEAK_CLOSE")

    if hours_to_close_value <= 6 and delta_vs_baseline_c <= -1.8:
        bonus_c -= 0.35
        tags.append("COOL_RUNRATE")

    bonus_c = max(-0.5, min(1.2, bonus_c))
    return bonus_c, tags


def regional_regime_adjustment_c(
    market: dict,
    taf_list: list,
    metar_list: list,
    hours_to_close_value: float,
    low: float,
    unit: str,
    is_low_market: bool,
    baseline_mean_c: float,
) -> tuple[float, list[str]]:
    icao = str(market.get("icao_code") or "").upper()
    latest_temp_c = _latest_metar_temp_c(metar_list)
    threshold_c = low if unit == "C" else fahrenheit_to_celsius(low)
    taf_raw = taf_text(taf_list)
    metar_raw = _extract_raw_metar_text(metar_list)
    groups = _taf_groups(taf_list)
    bonus_c = 0.0
    tags: list[str] = []

    if icao in CONTINENTAL_CONVECTIVE_ICAOS and not is_low_market and 2 <= hours_to_close_value <= 18:
        gap_c = threshold_c - latest_temp_c if latest_temp_c is not None else 99.0
        convective_groups = [idx for idx, group in enumerate(groups) if _group_has_convection(group)]
        later_clear = any(_is_clear_or_high_only(group) for idx, group in enumerate(groups) if idx > 0)

        if convective_groups:
            first_convective_idx = convective_groups[0]
            clear_after_convection = any(
                _is_clear_or_high_only(group) for idx, group in enumerate(groups) if idx > first_convective_idx
            )
            if gap_c <= 2.0 and clear_after_convection:
                bonus_c += 0.4
                tags.append("POST_STORM_HEAT")
            elif gap_c <= 2.0 and not later_clear:
                bonus_c -= 0.45
                tags.append("PEAK_CONVECTION_RISK")
            elif gap_c > 2.0:
                bonus_c -= 0.2
                tags.append("CONVECTION_DRAG")

        if latest_temp_c is not None:
            if latest_temp_c >= baseline_mean_c + 1.0 and gap_c <= 2.5:
                bonus_c += 0.25
                tags.append("CONTINENTAL_HEAT")
            elif latest_temp_c <= baseline_mean_c - 1.5 and gap_c > 1.5:
                bonus_c -= 0.2
                tags.append("CONTINENTAL_COOL")

        winds = _extract_winds(metar_raw)
        if winds:
            _, speed = winds[0]
            if speed <= 10 and gap_c <= 2.0:
                bonus_c += 0.15
                tags.append("LIGHT_MIXING")
            elif speed >= 18 and gap_c > 1.0:
                bonus_c -= 0.15
                tags.append("GUST_MIXING")

    elif icao in TEMPERATE_STABLE_ICAOS and not is_low_market and 2 <= hours_to_close_value <= 18:
        gap_c = threshold_c - latest_temp_c if latest_temp_c is not None else 99.0
        stable_taf = bool(taf_raw) and not any(marker in taf_raw for marker in ("TEMPO", "PROB30", "PROB40"))
        convective_taf = any(marker in taf_raw for marker in CONVECTIVE_MARKERS)
        low_cloud_taf = bool(re.search(r"\b(?:BKN|OVC)0(?:0\d|1\d|2\d|3\d|4\d)\b", taf_raw))

        if latest_temp_c is not None and stable_taf and not convective_taf:
            if latest_temp_c >= baseline_mean_c + 0.7 and gap_c <= 2.0 and not low_cloud_taf:
                bonus_c += 0.2
                tags.append("STABLE_WARM_TRACK")
            elif latest_temp_c <= baseline_mean_c - 1.0 and low_cloud_taf:
                bonus_c -= 0.25
                tags.append("STABLE_COOL_TRACK")

        if groups:
            if any(_is_clear_or_high_only(group) for group in groups) and not low_cloud_taf:
                bonus_c += 0.1
                tags.append("TEMPERATE_SUN")
            elif low_cloud_taf:
                bonus_c -= 0.1
                tags.append("TEMPERATE_CLOUD")

    bonus_c = max(-0.8, min(0.8, bonus_c))
    return bonus_c, tags


def convection_timing_adjustment_c(
    taf_list: list,
    metar_list: list,
    hours_to_close_value: float,
    low: float,
    unit: str,
    is_low_market: bool,
) -> tuple[float, list[str]]:
    if is_low_market or not (2.0 <= hours_to_close_value <= 18.0):
        return 0.0, []

    latest_temp_c = _latest_metar_temp_c(metar_list)
    if latest_temp_c is None:
        return 0.0, []

    threshold_c = low if unit == "C" else fahrenheit_to_celsius(low)
    gap_c = threshold_c - latest_temp_c
    groups = _taf_groups(taf_list)
    if not groups:
        return 0.0, []

    convective_idxs = [idx for idx, group in enumerate(groups) if _group_has_convection(group)]
    if not convective_idxs:
        return 0.0, []

    bonus_c = 0.0
    tags: list[str] = []
    first_conv_idx = convective_idxs[0]
    last_conv_idx = convective_idxs[-1]
    clear_after = any(_is_clear_or_high_only(group) for idx, group in enumerate(groups) if idx > last_conv_idx)
    low_cloud_after = any(_has_low_cloud(group) for idx, group in enumerate(groups) if idx > last_conv_idx)
    convective_late = last_conv_idx >= max(1, len(groups) - 2)
    tempo_prob = any(("TEMPO" in group or "PROB30" in group or "PROB40" in group) for group in groups)

    if first_conv_idx == 0 and clear_after and gap_c <= 2.0:
        bonus_c += 0.35
        tags.append("POST_CONVECTION_WINDOW")

    if convective_late and gap_c <= 2.0:
        bonus_c -= 0.45
        tags.append("MIDDAY_CONVECTION_RISK")
    elif convective_late:
        bonus_c -= 0.2
        tags.append("LATE_CONVECTION")

    if tempo_prob and gap_c <= 1.5 and not clear_after:
        bonus_c -= 0.25
        tags.append("TEMPO_HEAT_RISK")

    if low_cloud_after and gap_c <= 1.5:
        bonus_c -= 0.15
        tags.append("POST_STORM_CLOUD")

    bonus_c = max(-0.8, min(0.5, bonus_c))
    return bonus_c, tags


def hourly_projection_adjustment_c(
    weather_data: dict,
    hours_to_close_value: float,
    low: float,
    unit: str,
    is_low_market: bool,
    baseline_mean_c: float,
) -> tuple[float, list[str]]:
    if is_low_market or not (1.0 <= hours_to_close_value <= 12.0):
        return 0.0, []

    threshold_c = low if unit == "C" else fahrenheit_to_celsius(low)
    hourly_max_c = _hourly_forecast_max_c(weather_data, lookahead_hours=8 if hours_to_close_value <= 8 else 12)
    if hourly_max_c is None:
        return 0.0, []

    hourly = weather_data.get("forecast_hourly", {}) or {}
    cloud_cover = [float(v) for v in (hourly.get("cloud_cover", []) or [])[:8] if v is not None]
    wind_speed = [float(v) for v in (hourly.get("wind_speed_10m", []) or [])[:8] if v is not None]

    bonus_c = 0.0
    tags: list[str] = []
    gap_c = threshold_c - hourly_max_c
    delta_vs_baseline_c = hourly_max_c - baseline_mean_c

    if gap_c <= 0.0:
        bonus_c += 0.55
        tags.append("HOURLY_CROSS")
    elif gap_c <= 0.8:
        bonus_c += 0.35
        tags.append("HOURLY_NEAR")
    elif gap_c >= 2.0 and hours_to_close_value <= 6:
        bonus_c -= 0.25
        tags.append("HOURLY_SHORTFALL")

    if delta_vs_baseline_c >= 0.8:
        bonus_c += 0.25
        tags.append("HOURLY_HOTTER")
    elif delta_vs_baseline_c <= -1.2:
        bonus_c -= 0.2
        tags.append("HOURLY_COOLER")

    if cloud_cover:
        avg_cloud = sum(cloud_cover) / len(cloud_cover)
        if avg_cloud <= 35:
            bonus_c += 0.15
            tags.append("HOURLY_SUN")
        elif avg_cloud >= 75:
            bonus_c -= 0.15
            tags.append("HOURLY_CLOUD")

    if wind_speed:
        avg_wind = sum(wind_speed) / len(wind_speed)
        if avg_wind <= 12:
            bonus_c += 0.1
            tags.append("HOURLY_LIGHT_WIND")
        elif avg_wind >= 24:
            bonus_c -= 0.1
            tags.append("HOURLY_WINDY")

    bonus_c = max(-0.5, min(0.8, bonus_c))
    return bonus_c, tags


def exact_bucket_cap_adjustment_c(
    weather_data: dict,
    taf_list: list,
    metar_list: list,
    hours_to_close_value: float,
    low: float,
    unit: str,
    market_kind: str,
    is_low_market: bool,
    baseline_mean_c: float,
) -> tuple[float, list[str]]:
    if market_kind != "exact_bucket" or is_low_market or not (1.0 <= hours_to_close_value <= 12.0):
        return 0.0, []

    latest_temp_c = _latest_metar_temp_c(metar_list)
    threshold_c = low if unit == "C" else fahrenheit_to_celsius(low)
    if latest_temp_c is None:
        return 0.0, []

    taf_groups = _taf_groups(taf_list)
    if not taf_groups:
        return 0.0, []

    gap_c = threshold_c - latest_temp_c
    bonus_c = 0.0
    tags: list[str] = []
    convective_idxs = [idx for idx, group in enumerate(taf_groups) if _group_has_convection(group)]
    later_convection = bool(convective_idxs) and convective_idxs[-1] >= max(1, len(taf_groups) - 2)
    clear_before_convection = any(
        _is_clear_or_high_only(group) or not _has_low_cloud(group)
        for idx, group in enumerate(taf_groups)
        if not convective_idxs or idx < convective_idxs[0]
    )

    hourly_max_c = _hourly_forecast_max_c(weather_data, lookahead_hours=8)
    hourly_gap_c = threshold_c - hourly_max_c if hourly_max_c is not None else 99.0

    if -0.25 <= gap_c <= 1.25 and later_convection and clear_before_convection:
        bonus_c += 0.9
        tags.append("BUCKET_TOUCH_CAP")
    elif -0.5 <= gap_c <= 1.5 and later_convection:
        bonus_c += 0.55
        tags.append("CONVECTION_CAP")

    if hourly_max_c is not None and -0.25 <= hourly_gap_c <= 1.0 and later_convection:
        bonus_c += 0.35
        tags.append("HOURLY_BUCKET_TOUCH")

    if baseline_mean_c >= threshold_c + 2.0 and later_convection and gap_c <= 1.5:
        bonus_c += 0.25
        tags.append("OVERSHOOT_CAPPED")

    bonus_c = max(0.0, min(1.4, bonus_c))
    return bonus_c, tags


def hours_to_close(market: dict) -> float:
    res = (
        market.get("resolution_date")
        or market.get("end_date_iso")
        or market.get("endDateIso")
        or market.get("end_date")
    )
    if not res:
        return 24.0
    try:
        now_override = market.get("_backtest_now")
        if now_override:
            now = datetime.fromisoformat(str(now_override).replace("Z", "+00:00"))
            if now.tzinfo is None:
                now = now.replace(tzinfo=timezone.utc)
        else:
            now = datetime.now(timezone.utc)
        if isinstance(res, (int, float)):
            target = datetime.fromtimestamp(float(res), timezone.utc)
        else:
            target = datetime.fromisoformat(str(res).replace("Z", "+00:00"))
            if target.tzinfo is None:
                target = target.replace(tzinfo=timezone.utc)
        return max(0.1, (target - now).total_seconds() / 3600)
    except Exception:
        return 24.0


def day_index(hours: float) -> int:
    if hours <= 18:
        return 0
    if hours <= 42:
        return 1
    if hours <= 66:
        return 2
    return 3


def safe_day(values: list, idx: int) -> Optional[float]:
    if values and idx < len(values) and values[idx] is not None:
        return float(values[idx])
    for j in range(idx - 1, -1, -1):
        if values and j < len(values) and values[j] is not None:
            return float(values[j])
    return None


def taf_text(taf_list: list) -> str:
    return " ".join(str(item.get("rawTAF", "")) for item in (taf_list or [])).upper()


def taf_uncertainty_multiplier(taf_list: list) -> tuple[float, bool, list[str]]:
    raw = taf_text(taf_list)
    if not raw:
        return 1.25, False, ["TAF_MISSING"]

    multiplier = 1.0
    tags: list[str] = []

    if "TEMPO" in raw:
        multiplier += 0.35
        tags.append("TEMPO")
    if "PROB30" in raw:
        multiplier += 0.25
        tags.append("PROB30")
    if "PROB40" in raw:
        multiplier += 0.35
        tags.append("PROB40")
    if any(marker in raw for marker in CONVECTIVE_MARKERS):
        multiplier += 0.95
        tags.append("CONVECTION")
    if re.search(r"\bBECMG\b", raw):
        multiplier += 0.15
        tags.append("BECMG")

    multiplier = min(float(config.tsas_max_taf_inflation), multiplier)
    circuit_breaker = "TEMPO" in tags and "CONVECTION" in tags
    return multiplier, circuit_breaker, tags or ["TAF_STABLE"]


def source_weights(hours: float, taf_available: bool, metar_available: bool, monitor_mode: bool = False) -> dict[str, float]:
    if monitor_mode and hours <= 6:
        weights = {"ensemble": 0.25, "taf": 0.25, "metar": 0.50}
    elif monitor_mode and hours <= 24:
        weights = {"ensemble": 0.35, "taf": 0.45, "metar": 0.20}
    elif monitor_mode and hours <= 72:
        weights = {"ensemble": 0.65, "taf": 0.30, "metar": 0.05}
    elif hours <= 6:
        weights = {"ensemble": 0.35, "taf": 0.25, "metar": 0.40}
    elif hours <= 24:
        weights = {"ensemble": 0.45, "taf": 0.40, "metar": 0.15}
    elif hours <= 72:
        weights = {"ensemble": 0.70, "taf": 0.25, "metar": 0.05}
    else:
        weights = {"ensemble": 0.85, "taf": 0.15, "metar": 0.0}

    if not taf_available:
        weights["taf"] = 0.0
    if not metar_available:
        weights["metar"] = 0.0

    total = sum(weights.values())
    if total <= 0:
        return {"ensemble": 1.0, "taf": 0.0, "metar": 0.0}
    return {k: v / total for k, v in weights.items()}


def build_distribution(
    city: str,
    market: dict,
    weather_data: dict,
    is_low_market: bool,
    unit: str,
    low: float,
    high: float,
    monitor_mode: bool = False,
    family_mode: bool = False,
) -> Optional[TsasDistribution]:
    forecast_daily = weather_data.get("forecast_daily", {})
    htc = hours_to_close(market)
    d_idx = day_index(htc)

    if is_low_market:
        model_values_c = [
            safe_day(forecast_daily.get("ensemble_min", []), d_idx),
            safe_day(forecast_daily.get("ecmwf_min", []), d_idx),
            safe_day(forecast_daily.get("gfs_min", []), d_idx),
        ]
        taf_value_c = extract_taf_min_c(weather_data.get("taf", []))
    else:
        model_values_c = [
            safe_day(forecast_daily.get("ensemble_max", []), d_idx),
            safe_day(forecast_daily.get("ecmwf_max", []), d_idx),
            safe_day(forecast_daily.get("gfs_max", []), d_idx),
        ]
        taf_value_c = extract_taf_max_c(weather_data.get("taf", []))

    model_values_c = [v for v in model_values_c if v is not None]
    if not model_values_c:
        return None

    ensemble_mean_c = sum(model_values_c) / len(model_values_c)
    model_spread_c = max(model_values_c) - min(model_values_c) if len(model_values_c) > 1 else 1.5

    metar_high_c = extract_metar_high_c(weather_data.get("metar", []))
    metar_component_c = None
    if metar_high_c is not None:
        if is_low_market:
            metar_component_c = metar_high_c if d_idx == 0 else None
        else:
            # Same-day max cannot be below the observed high. For future days,
            # METAR must not be mixed as a direct source; today's observed high
            # can otherwise inflate tomorrow's source spread and create huge
            # false tails in the bucket family.
            metar_component_c = max(metar_high_c, ensemble_mean_c) if d_idx == 0 else None

    taf_inflation, taf_circuit, taf_tags = taf_uncertainty_multiplier(weather_data.get("taf", []))
    weights = source_weights(htc, taf_value_c is not None, metar_component_c is not None, monitor_mode=monitor_mode)

    weighted_sources: list[tuple[float, float, str]] = [(ensemble_mean_c, weights["ensemble"], "ENS")]
    if taf_value_c is not None and weights["taf"] > 0:
        weighted_sources.append((taf_value_c, weights["taf"], "TAF"))
    if metar_component_c is not None and weights["metar"] > 0:
        weighted_sources.append((metar_component_c, weights["metar"], "METAR"))

    mean_c = sum(value * weight for value, weight, _ in weighted_sources)
    baseline_mean_c = mean_c
    market_kind = temperature_market_kind(str(market.get("question", "")), low, high)
    coast_bonus_c, coast_tags = coastal_taf_adjustment_c(
        market=market,
        taf_list=weather_data.get("taf", []),
        metar_list=weather_data.get("metar", []),
        hours_to_close_value=htc,
    )
    wind_bonus_c, wind_tags = coastal_wind_adjustment_c(
        market=market,
        taf_list=weather_data.get("taf", []),
        metar_list=weather_data.get("metar", []),
        hours_to_close_value=htc,
    )
    if family_mode:
        # A city/date family must be one temperature distribution. Bucket- or
        # threshold-specific adjustments are useful as diagnostics, but they
        # cannot be mixed into the family curve without producing impossible
        # non-unimodal bucket probabilities.
        tropical_bonus_c, tropical_tags = 0.0, []
        threshold_bonus_c, threshold_tags = 0.0, []
        runrate_bonus_c, runrate_tags = 0.0, []
        regime_bonus_c, regime_tags = 0.0, []
        convection_bonus_c, convection_tags = 0.0, []
        hourly_bonus_c, hourly_tags = 0.0, []
        exact_bucket_bonus_c, exact_bucket_tags = 0.0, []
    else:
        tropical_bonus_c, tropical_tags = tropical_intraday_adjustment_c(
            market=market,
            taf_list=weather_data.get("taf", []),
            metar_list=weather_data.get("metar", []),
            hours_to_close_value=htc,
            low=low,
            unit=unit,
            is_low_market=is_low_market,
            baseline_mean_c=baseline_mean_c,
        )
        threshold_bonus_c, threshold_tags = same_day_threshold_adjustment_c(
            market=market,
            metar_list=weather_data.get("metar", []),
            hours_to_close_value=htc,
            low=low,
            unit=unit,
            is_low_market=is_low_market,
            baseline_mean_c=baseline_mean_c,
        )
        runrate_bonus_c, runrate_tags = same_day_runrate_adjustment_c(
            metar_list=weather_data.get("metar", []),
            hours_to_close_value=htc,
            low=low,
            unit=unit,
            is_low_market=is_low_market,
            baseline_mean_c=baseline_mean_c,
        )
        regime_bonus_c, regime_tags = regional_regime_adjustment_c(
            market=market,
            taf_list=weather_data.get("taf", []),
            metar_list=weather_data.get("metar", []),
            hours_to_close_value=htc,
            low=low,
            unit=unit,
            is_low_market=is_low_market,
            baseline_mean_c=baseline_mean_c,
        )
        convection_bonus_c, convection_tags = convection_timing_adjustment_c(
            taf_list=weather_data.get("taf", []),
            metar_list=weather_data.get("metar", []),
            hours_to_close_value=htc,
            low=low,
            unit=unit,
            is_low_market=is_low_market,
        )
        hourly_bonus_c, hourly_tags = hourly_projection_adjustment_c(
            weather_data=weather_data,
            hours_to_close_value=htc,
            low=low,
            unit=unit,
            is_low_market=is_low_market,
            baseline_mean_c=baseline_mean_c,
        )
        exact_bucket_bonus_c, exact_bucket_tags = exact_bucket_cap_adjustment_c(
            weather_data=weather_data,
            taf_list=weather_data.get("taf", []),
            metar_list=weather_data.get("metar", []),
            hours_to_close_value=htc,
            low=low,
            unit=unit,
            market_kind=market_kind,
            is_low_market=is_low_market,
            baseline_mean_c=baseline_mean_c,
        )
    adjustment_multiplier = float(getattr(config, "tsas_monitor_adjustment_multiplier", 1.0)) if monitor_mode else 1.0
    mean_c += (
        coast_bonus_c * adjustment_multiplier
        + wind_bonus_c * adjustment_multiplier
        + tropical_bonus_c * adjustment_multiplier
        + threshold_bonus_c * adjustment_multiplier
        + runrate_bonus_c * adjustment_multiplier
        + regime_bonus_c * adjustment_multiplier
        + convection_bonus_c * adjustment_multiplier
        + hourly_bonus_c * adjustment_multiplier
        + exact_bucket_bonus_c * adjustment_multiplier
    )
    source_spread_c = max(value for value, _, _ in weighted_sources) - min(value for value, _, _ in weighted_sources)

    ens_std_c = safe_day(forecast_daily.get("ensemble_std", []), d_idx)
    horizon_std_c = 0.65 + d_idx * 0.55
    raw_std_c = max(horizon_std_c, ens_std_c or 0.0, model_spread_c / 1.35, source_spread_c / 1.15)
    std_c = max(0.55, raw_std_c * taf_inflation)

    disagreement_c = max(model_spread_c, source_spread_c)
    confidence = 1.0 / (1.0 + disagreement_c / 3.0)
    confidence *= 1.0 / math.sqrt(taf_inflation)
    if htc > 72:
        confidence *= 0.75
    confidence = max(0.05, min(1.0, confidence))
    liquidity_factor = liquidity_factor_for_market(market)

    circuit_breaker = taf_circuit and htc <= float(config.tsas_circuit_breaker_hours)

    if unit == "F":
        mean = celsius_to_fahrenheit(mean_c)
        std = std_c * 9.0 / 5.0
        model_spread = model_spread_c * 9.0 / 5.0
    else:
        mean = mean_c
        std = std_c
        model_spread = model_spread_c

    src = ",".join(f"{name}:{value:.1f}C@{weight:.2f}" for value, weight, name in weighted_sources)
    scaled_coast_bonus_c = coast_bonus_c * adjustment_multiplier
    scaled_wind_bonus_c = wind_bonus_c * adjustment_multiplier
    scaled_tropical_bonus_c = tropical_bonus_c * adjustment_multiplier
    scaled_threshold_bonus_c = threshold_bonus_c * adjustment_multiplier
    scaled_runrate_bonus_c = runrate_bonus_c * adjustment_multiplier
    scaled_regime_bonus_c = regime_bonus_c * adjustment_multiplier
    scaled_convection_bonus_c = convection_bonus_c * adjustment_multiplier
    scaled_hourly_bonus_c = hourly_bonus_c * adjustment_multiplier
    scaled_exact_bucket_bonus_c = exact_bucket_bonus_c * adjustment_multiplier

    all_tags = (
        taf_tags
        + coast_tags
        + wind_tags
        + tropical_tags
        + threshold_tags
        + runrate_tags
        + regime_tags
        + convection_tags
        + hourly_tags
        + exact_bucket_tags
    )
    reasoning = (
        f"TSASv1 day={d_idx} htc={htc:.1f}h mean={mean:.1f}{unit} std={std:.2f} "
        f"spread={model_spread:.2f}{unit} taf_x={taf_inflation:.2f} "
        f"conf={confidence:.2f} liq={liquidity_factor:.2f} coast_dx={scaled_coast_bonus_c:+.2f}C "
        f"wind_dx={scaled_wind_bonus_c:+.2f}C tropical_dx={scaled_tropical_bonus_c:+.2f}C "
        f"intraday_dx={scaled_threshold_bonus_c:+.2f}C runrate_dx={scaled_runrate_bonus_c:+.2f}C "
        f"regime_dx={scaled_regime_bonus_c:+.2f}C conv_dx={scaled_convection_bonus_c:+.2f}C "
        f"hourly_dx={scaled_hourly_bonus_c:+.2f}C bucket_dx={scaled_exact_bucket_bonus_c:+.2f}C "
        f"kind={market_kind} monitor={int(monitor_mode)} family={int(family_mode)} "
        f"tags={'+'.join(all_tags)} src=[{src}]"
    )

    return TsasDistribution(
        mean=mean,
        std=std,
        confidence=confidence,
        liquidity_factor=liquidity_factor,
        taf_inflation=taf_inflation,
        circuit_breaker=circuit_breaker,
        reasoning=reasoning,
    )


def bin_probability(dist: TsasDistribution, low: float, high: float) -> float:
    return max(0.01, min(0.99, raw_bin_probability(dist, low, high)))


def raw_bin_probability(dist: TsasDistribution, low: float, high: float) -> float:
    p = norm.cdf(high, loc=dist.mean, scale=dist.std) - norm.cdf(low, loc=dist.mean, scale=dist.std)
    return max(0.0, min(1.0, p))


def temperature_market_kind(question: str, low: float, high: float) -> str:
    q = question.lower()
    if high >= 999:
        return "open_upper"
    if low <= -999:
        return "open_lower"
    if re.search(r"\bor\s+(?:higher|above|more|lower|below|less)\b", q):
        return "open"
    if re.search(r"\b(?:above|higher\s+than|below|lower\s+than)\b", q):
        return "open"
    if re.search(r"\bbetween\b|\bto\b", q):
        return "range"
    return "exact_bucket"


def outcome_probability(dist: TsasDistribution, question: str, low: float, high: float, outcome_name: str) -> float:
    yes_probability = bin_probability(dist, low, high)
    if temperature_market_kind(question, low, high) == "exact_bucket":
        # Exact weather buckets resolve Yes only when the maximum lands inside
        # the integer bucket. No wins on both sides: below or above the bucket.
        yes_probability = max(0.01, min(0.99, yes_probability))

    if str(outcome_name).lower() == "yes":
        return yes_probability
    return max(0.01, min(0.99, 1.0 - yes_probability))


def raw_yes_probability(dist: TsasDistribution, low: float, high: float) -> float:
    return raw_bin_probability(dist, low, high)


def outcome_probability_from_yes(yes_probability: float, outcome_name: str) -> float:
    yes_probability = max(0.01, min(0.99, yes_probability))
    if str(outcome_name).lower() == "yes":
        return yes_probability
    return max(0.01, min(0.99, 1.0 - yes_probability))


def _market_date_key(question: str, market: dict) -> str:
    match = re.search(r"\bon\s+([A-Za-z]+\s+\d+)\b", question or "", re.IGNORECASE)
    if match:
        return match.group(1)
    return str(market.get("resolution_date") or market.get("end_date_iso") or market.get("endDateIso") or "")


def _bucket_label(low: float, high: float, unit: str) -> str:
    if low <= -999:
        return f"<= {high - 0.5:g}{unit}"
    if high >= 999:
        return f">= {low + 0.5:g}{unit}"
    if abs((high - low) - 1.0) < 0.001:
        return f"{low + 0.5:g}{unit}"
    return f"{low + 0.5:g}-{high - 0.5:g}{unit}"


def _base_family_key(city: str, market: dict, question: str, unit: str, is_low_market: bool) -> Optional[tuple]:
    date_key = _market_date_key(question, market)
    if not date_key:
        return None
    market_type = "low" if is_low_market else "high"
    return (city, date_key, market_type, unit)


def _family_key(city: str, market: dict, question: str, low: float, high: float, unit: str, is_low_market: bool) -> Optional[tuple]:
    return _base_family_key(city, market, question, unit, is_low_market)


def _yes_market_price(item: dict) -> Optional[float]:
    for outcome in item.get("market", {}).get("outcomes", []) or []:
        if str(outcome.get("name", "")).lower() == "yes":
            try:
                return max(0.0, min(1.0, float(outcome.get("current_price"))))
            except (TypeError, ValueError):
                return None
    return None


def _apply_family_distributions(items: list[dict], city: str, weather_data: dict, monitor_mode: bool) -> None:
    groups: dict[tuple, list[dict]] = {}
    for item in items:
        key = item.get("family_key")
        if key is not None:
            groups.setdefault(key, []).append(item)

    for _, group in groups.items():
        if len(group) < 3:
            continue

        representative = next(
            (
                item for item in group
                if float(item.get("low", 0.0)) > -999 and float(item.get("high", 0.0)) < 999
            ),
            group[0],
        )
        family_dist = build_distribution(
            city,
            representative["market"],
            weather_data,
            bool(representative["is_low_market"]),
            str(representative["unit"]),
            float(representative["low"]),
            float(representative["high"]),
            monitor_mode=monitor_mode,
            family_mode=True,
        )
        if family_dist is None:
            continue

        for item in group:
            low = float(item["low"])
            high = float(item["high"])
            item["dist"] = family_dist
            item["raw_yes_probability"] = raw_yes_probability(family_dist, low, high)
            item["coherent_yes_probability"] = outcome_probability(
                family_dist,
                str(item["question"]),
                low,
                high,
                "Yes",
            )


def _attach_family_coherence(items: list[dict]) -> None:
    groups: dict[tuple, list[dict]] = {}
    for item in items:
        key = item.get("family_key")
        if key is not None:
            groups.setdefault(key, []).append(item)

    for key, group in groups.items():
        bounded_group = [
            item for item in group
            if item.get("market_kind") in {"exact_bucket", "range"}
            and float(item.get("low", 0.0)) > -999
            and float(item.get("high", 0.0)) < 999
        ]
        if len(bounded_group) < 3:
            continue

        min_low = min(float(item["low"]) for item in bounded_group)
        max_high = max(float(item["high"]) for item in bounded_group)
        edge_open_group = []
        for item in group:
            low = float(item.get("low", 0.0))
            high = float(item.get("high", 0.0))
            kind = item.get("market_kind")
            if kind == "open_lower" and high <= min_low + 0.001:
                edge_open_group.append(item)
            elif kind == "open_upper" and low >= max_high - 0.001:
                edge_open_group.append(item)

        family_group = bounded_group + edge_open_group
        raw_sum = sum(float(item.get("raw_yes_probability", 0.0)) for item in family_group)
        if raw_sum <= 0:
            continue

        covers_open_tail = bool(edge_open_group)
        near_complete = 0.80 <= raw_sum <= 1.20
        should_normalize = covers_open_tail or near_complete

        market_prices = [_yes_market_price(item) for item in family_group]
        has_market_family = all(price is not None for price in market_prices)
        market_sum = sum(float(price or 0.0) for price in market_prices)
        has_market_family = has_market_family and market_sum > 0

        model_probs: list[float] = []
        market_probs: list[float] = []
        for idx, item in enumerate(family_group):
            raw_probability = max(0.0, min(1.0, float(item.get("raw_yes_probability", 0.0))))
            model_probability = raw_probability / raw_sum if should_normalize else raw_probability
            model_probs.append(max(0.0, min(1.0, model_probability)))
            if has_market_family:
                market_probs.append(max(0.0, min(1.0, float(market_prices[idx] or 0.0) / market_sum)))

        divergence = 0.0
        if has_market_family and should_normalize:
            divergence = 0.5 * sum(abs(model - market) for model, market in zip(model_probs, market_probs))

        # Normalize only when the visible family is a credible partition of the
        # same market. If the scanned markets miss tail buckets, keep raw model
        # probabilities so we do not inflate a partial set to 100%.
        for idx, item in enumerate(family_group):
            item["coherent_yes_probability"] = max(0.0, min(1.0, model_probs[idx]))
            item["family_raw_sum"] = raw_sum
            item["family_normalized"] = should_normalize
            item["family_market_sum"] = market_sum if has_market_family else 0.0
            item["family_market_divergence"] = divergence

        ordered = sorted(family_group, key=lambda item: (float(item["low"]), float(item["high"])))
        distribution_parts = [
            f"{_bucket_label(float(item['low']), float(item['high']), str(item['unit']))}:{float(item['coherent_yes_probability']) * 100:.1f}%"
            for item in ordered
        ]
        distribution = " | ".join(distribution_parts)
        market_distribution = ""
        if has_market_family and market_probs:
            market_by_id = {id(item): market_probs[idx] for idx, item in enumerate(family_group)}
            market_distribution = " | ".join(
                f"{_bucket_label(float(item['low']), float(item['high']), str(item['unit']))}:"
                f"{market_by_id.get(id(item), 0.0) * 100:.1f}%"
                for item in ordered
            )
        norm_sum = sum(float(item.get("coherent_yes_probability", 0.0)) for item in family_group)
        family_id = f"{key[0]} {key[1]} {key[2]} {key[3]}"
        for item in group:
            item["family_distribution"] = distribution
            item["family_market_distribution"] = market_distribution
            item["family_id"] = family_id
            item["family_raw_sum"] = raw_sum
            item["family_norm_sum"] = norm_sum
            item["family_market_sum"] = market_sum if has_market_family else 0.0
            item["family_market_divergence"] = divergence


def _item_yes_probability(item: dict) -> float:
    return float(item.get("coherent_yes_probability", item.get("raw_yes_probability", 0.0)))


def liquidity_factor_for_market(market: dict) -> float:
    spread = market.get("spread")

    spread_factor = 1.0
    if spread is not None and float(config.tsas_max_spread) > 0:
        spread_ratio = max(0.0, min(1.0, float(spread) / float(config.tsas_max_spread)))
        spread_factor = max(0.25, 1.0 - 0.75 * spread_ratio)

    return max(0.2, min(1.0, spread_factor))


def ladder_bucket_key(signal: dict) -> Optional[tuple]:
    parsed = parse_temperature_bin(signal.get("question", ""))
    if parsed is None:
        return None

    low, high, unit = parsed
    if low <= -999 or high >= 999:
        return None

    outcome_name = str(signal.get("outcome_name", "")).lower()
    sentiment = str(signal.get("sentiment", "")).upper()
    if outcome_name != "yes" or sentiment != "BULLISH":
        return None

    market_price = float(signal.get("market_price", 1.0))
    if market_price > float(config.tsas_ladder_max_price):
        return None

    date_match = re.search(r"\bon\s+([A-Za-z]+\s+\d+)\b", signal.get("question", ""))
    date_key = date_match.group(1) if date_match else signal.get("resolution_date", "")
    return (signal.get("city"), date_key, unit, low, high)


def apply_laddering(signals: list[dict]) -> list[dict]:
    if not getattr(config, "tsas_ladder_enabled", True):
        return signals

    keyed: list[tuple[tuple, dict]] = []
    for signal in signals:
        key = ladder_bucket_key(signal)
        if key is not None:
            keyed.append((key, signal))

    if not keyed:
        return signals

    groups: dict[tuple, list[tuple[tuple, dict]]] = {}
    for key, signal in keyed:
        family = key[:3]
        groups.setdefault(family, []).append((key, signal))

    laddered_ids: set[str] = set()
    ladder_counter = 0

    for family, items in groups.items():
        ordered = sorted(items, key=lambda item: (item[0][3], item[0][4]))
        run: list[tuple[tuple, dict]] = []

        def flush_run(current_run: list[tuple[tuple, dict]]):
            nonlocal ladder_counter
            if len(current_run) < 2:
                return

            selected = sorted(
                current_run,
                key=lambda item: (item[1].get("ev", -999.0), -item[1].get("market_price", 1.0)),
                reverse=True,
            )[: int(config.tsas_ladder_max_positions)]
            selected = sorted(selected, key=lambda item: item[0][3])
            package_ev = sum(float(sig.get("ev", 0.0)) for _, sig in selected)
            package_raw_ev = sum(float(sig.get("raw_ev", 0.0)) for _, sig in selected)
            package_probability = sum(float(sig.get("true_probability", 0.0)) for _, sig in selected)
            if package_ev < float(config.tsas_ladder_min_package_ev):
                return

            package_kelly = sum(float(sig.get("kelly", 0.0)) for _, sig in selected)
            ladder_cap = float(config.tsas_ladder_package_cap)
            scaling = 1.0
            if package_kelly > 0 and package_kelly > ladder_cap:
                scaling = ladder_cap / package_kelly

            ladder_counter += 1
            ladder_group = f"{family[0]}|{family[1]}|{family[2]}|ladder{ladder_counter}"

            for rank, (_, sig) in enumerate(selected, start=1):
                original_kelly = float(sig.get("kelly", 0.0))
                scaled_kelly = original_kelly * scaling
                ladder_weight = (original_kelly / package_kelly) if package_kelly > 0 else (1.0 / len(selected))
                sig["kelly"] = scaled_kelly
                sig["ladder_group"] = ladder_group
                sig["ladder_rank"] = rank
                sig["ladder_size"] = len(selected)
                sig["ladder_package_ev"] = package_ev
                sig["ladder_package_raw_ev"] = package_raw_ev
                sig["ladder_package_probability"] = package_probability
                sig["ladder_package_kelly"] = package_kelly * scaling
                sig["ladder_weight"] = ladder_weight
                sig["ladder_scaling"] = scaling
                sig["reasoning"] += (
                    f" ladder={rank}/{len(selected)}"
                    f" pkg_ev={package_ev:.3f}"
                    f" pkg_prob={package_probability:.3f}"
                    f" pkg_scale={scaling:.2f}"
                )
                laddered_ids.add(sig["token_id"])

        for item in ordered:
            key, signal = item
            if not run:
                run = [item]
                continue

            prev_key = run[-1][0]
            prev_high = prev_key[4]
            curr_low = key[3]
            if abs(curr_low - prev_high) <= 1.01:
                run.append(item)
            else:
                flush_run(run)
                run = [item]

        flush_run(run)

    return signals


def analyze_city_tsas(
    city: str,
    markets: list[dict],
    weather_data: dict,
    return_all: bool = False,
    monitor_mode: bool = False,
) -> list[dict]:
    ev_threshold = config.ev_threshold.get(city, config.ev_threshold.get("default", 0.08))
    final_signals: list[dict] = []
    parsed_items: list[dict] = []
    skipped = {
        "unparsed": 0,
        "distribution": 0,
        "entry_window": 0,
        "spread": 0,
        "circuit_breaker": 0,
        "confidence": 0,
        "market_conflict": 0,
    }

    for market in markets:
        question = market.get("question", "")
        parsed = parse_temperature_bin(question)
        if parsed is None:
            skipped["unparsed"] += 1
            continue

        low, high, unit = parsed
        is_low_market = "lowest" in question.lower() or "minimum" in question.lower()
        dist = build_distribution(
            city,
            market,
            weather_data,
            is_low_market,
            unit,
            low,
            high,
            monitor_mode=monitor_mode,
        )
        if dist is None:
            skipped["distribution"] += 1
            continue

        htc = hours_to_close(market)
        market_kind = temperature_market_kind(question, low, high)
        raw_prob = raw_yes_probability(dist, low, high)
        trade_yes_probability = outcome_probability(dist, question, low, high, "Yes")
        parsed_items.append({
            "market": market,
            "question": question,
            "low": low,
            "high": high,
            "unit": unit,
            "is_low_market": is_low_market,
            "dist": dist,
            "htc": htc,
            "market_kind": market_kind,
            "raw_yes_probability": raw_prob,
            "coherent_yes_probability": trade_yes_probability,
            "family_key": _family_key(city, market, question, low, high, unit, is_low_market),
            "family_raw_sum": 1.0,
            "family_normalized": False,
            "family_distribution": "",
            "family_market_distribution": "",
            "family_id": "",
        })

    _apply_family_distributions(parsed_items, city, weather_data, monitor_mode)
    _attach_family_coherence(parsed_items)

    # Determine if BMA enrichment is available for this city
    forecast_daily = weather_data.get("forecast_daily", {})
    herbie_ok = weather_data.get("herbie_available", False) and _BMA_AVAILABLE

    for item in parsed_items:
        market = item["market"]
        question = item["question"]
        dist = item["dist"]
        htc = float(item["htc"])
        low = float(item["low"])
        high = float(item["high"])
        unit = str(item["unit"])
        yes_probability = _item_yes_probability(item)
        tsas_yes_probability = yes_probability  # keep original for logging
        bma_result = None
        bma_reasoning = ""

        # --- Phase 2: BMA enrichment ---
        if herbie_ok:
            try:
                d_idx = day_index(htc)
                is_low = bool(item.get("is_low_market", False))
                if unit == "F":
                    bma_result = compute_bma_probability_fahrenheit(
                        forecast_daily, d_idx, low, high, not is_low, htc, yes_probability
                    )
                else:
                    bma_result = compute_bma_probability(
                        forecast_daily, d_idx, low, high, not is_low, htc, yes_probability
                    )
                if bma_result is not None:
                    yes_probability = bma_result.probability
                    bma_reasoning = bma_result.reasoning
            except Exception as _bma_err:
                import logging as _log
                _log.getLogger(__name__).warning(f"[BMA] {city}: {_bma_err}")
                bma_result = None

        family_distribution = str(item.get("family_distribution") or "")
        family_market_distribution = str(item.get("family_market_distribution") or "")
        family_id = str(item.get("family_id") or "")
        family_raw_sum = float(item.get("family_raw_sum", 1.0))
        family_norm_sum = float(item.get("family_norm_sum", 1.0))
        family_market_sum = float(item.get("family_market_sum", 0.0))
        family_market_divergence = float(item.get("family_market_divergence", 0.0))

        if return_all:
            for out in market.get("outcomes", []):
                out_name = out["name"]
                p = outcome_probability_from_yes(yes_probability, out_name)
                final_signals.append({
                    "market_id": market["market_id"],
                    "question": question,
                    "token_id": out["token_id"],
                    "outcome_name": out_name,
                    "outcome_slug": out_name,
                    "predicted_prob": p,
                    "true_probability": p,
                    "bucket_probability_yes": yes_probability,
                    "family_distribution": family_distribution,
                    "family_market_distribution": family_market_distribution,
                    "family_id": family_id,
                    "family_raw_sum": family_raw_sum,
                    "family_norm_sum": family_norm_sum,
                    "family_market_sum": family_market_sum,
                    "family_market_divergence": family_market_divergence,
                    "family_normalized": bool(item.get("family_normalized")),
                    "city": city,
                    "reasoning": (dist.reasoning + " || " + bma_reasoning) if bma_reasoning else dist.reasoning,
                    "uncertainty_score": 1.0 - dist.confidence,
                    "tsas_confidence": dist.confidence,
                    "tsas_probability": tsas_yes_probability,
                    "bma_ensemble_prob": bma_result.ensemble_prob if bma_result else -1.0,
                    "bma_det_prob": bma_result.bma_prob if bma_result else -1.0,
                    "bma_n_members": (bma_result.n_gefs_members + bma_result.n_icon_members) if bma_result else 0,
                    "bma_model_spread": bma_result.model_spread_c if bma_result else 0.0,
                    "bma_correction": bma_result.correction_applied if bma_result else False,
                    "analysis_model": "tsas+bma" if bma_result else "tsas",
                })
            continue

        if htc < float(config.tsas_entry_min_hours_to_close) or htc > float(config.tsas_entry_max_hours_to_close):
            skipped["entry_window"] += 1
            continue

        spread = market.get("spread")
        if spread is not None and spread > float(config.tsas_max_spread):
            skipped["spread"] += 1
            continue

        if dist.circuit_breaker:
            skipped["circuit_breaker"] += 1
            continue
        if dist.confidence < float(config.tsas_min_confidence):
            skipped["confidence"] += 1
            continue
        if family_market_divergence >= float(getattr(config, "tsas_entry_market_conflict_divergence", 0.35)):
            skipped["market_conflict"] = skipped.get("market_conflict", 0) + 1
            continue

        best_signal = None
        best_score = -999.0
        confidence_penalty = max(0.0, min(1.0, dist.confidence))
        liquidity_penalty = max(0.0, min(1.0, dist.liquidity_factor))
        total_penalty = confidence_penalty * liquidity_penalty

        for out in market.get("outcomes", []):
            out_name = out["name"]
            market_price = float(out.get("current_price", 0.5))
            p = outcome_probability_from_yes(yes_probability, out_name)

            raw_edge = p - market_price
            adjusted_edge = raw_edge * total_penalty
            ev = (p * (1.0 - market_price)) - ((1.0 - p) * market_price)
            adjusted_ev = ev * total_penalty
            odds = (1.0 - market_price) / market_price if market_price > 0 else 0.0
            full_kelly = raw_edge / odds if odds > 0 else 0.0
            frac_kelly = full_kelly * config.kelly_fraction * total_penalty if full_kelly > 0 else 0.0

            if adjusted_ev > ev_threshold and frac_kelly > 0 and adjusted_ev > best_score:
                best_score = adjusted_ev
                sentiment = "BULLISH" if out_name.lower() == "yes" else "BEARISH"
                best_signal = {
                    "market_id": market["market_id"],
                    "question": question,
                    "token_id": out["token_id"],
                    "outcome_name": out["name"],
                    "outcome_slug": out_name,
                    "market_price": market_price,
                    "true_probability": p,
                    "predicted_prob": p,
                    "bucket_probability_yes": yes_probability,
                    "family_distribution": family_distribution,
                    "family_market_distribution": family_market_distribution,
                    "family_id": family_id,
                    "family_raw_sum": family_raw_sum,
                    "family_norm_sum": family_norm_sum,
                    "family_market_sum": family_market_sum,
                    "family_market_divergence": family_market_divergence,
                    "family_normalized": bool(item.get("family_normalized")),
                    "ev": adjusted_ev,
                    "raw_ev": ev,
                    "edge": adjusted_edge * 100.0,
                    "raw_edge": raw_edge * 100.0,
                    "kelly": frac_kelly,
                    "confidence": int(min(99, max(1, dist.confidence * 100))),
                    "liquidity_factor": dist.liquidity_factor,
                    "uncertainty_score": 1.0 - dist.confidence,
                    "sentiment": sentiment,
                    "city": city,
                    "icao_code": market.get("icao_code", ""),
                    "resolution_date": market.get("resolution_date"),
                    "reasoning": dist.reasoning,
                    "analysis_model": "tsas",
                }

        if best_signal is not None:
            # Attach BMA diagnostics to winning signal
            if bma_result is not None:
                best_signal["tsas_probability"] = tsas_yes_probability
                best_signal["bma_ensemble_prob"] = bma_result.ensemble_prob
                best_signal["bma_det_prob"] = bma_result.bma_prob
                best_signal["bma_n_members"] = bma_result.n_gefs_members + bma_result.n_icon_members
                best_signal["bma_model_spread"] = bma_result.model_spread_c
                best_signal["bma_correction"] = bma_result.correction_applied
                best_signal["analysis_model"] = "tsas+bma"
                best_signal["reasoning"] = best_signal["reasoning"] + " || " + bma_result.reasoning
            final_signals.append(best_signal)

    if not return_all and final_signals:
        final_signals = apply_laddering(final_signals)

    if not return_all:
        skipped_summary = ", ".join(f"{k}={v}" for k, v in skipped.items() if v)
        if skipped_summary:
            from src.utils import logger
            logger.info(f"[TSAS] {city}: filtered markets [{skipped_summary}]")

    return final_signals
