"""
herbie_fetcher.py — Phase 1: Multi-Model Data Pipeline
========================================================
Загружает сырые данные NWP-моделей через Herbie (прямой GRIB-доступ).
Обогащает существующие данные Open-Meteo дополнительными источниками:
  - GEFS: 31-член GFS ensemble (почасовые прогнозы T2m)
  - GFS:  deterministic (контрольный прогноз)
  - HRRR: высокоразрешённый (только США, до 48h)
  - NAM:  сетка 12km (только США, до 84h)
  - ECMWF: через open-data (00Z и 12Z run)

Выходной формат совместим с weather_data.py:
  - forecast_daily.*_max / ._min  (list[float|None], 4 дня)
  - herbie_members_max / _min     (list[list[float]], GEFS члены × дни)
  - herbie_available              (bool)
"""

from __future__ import annotations

import asyncio
import logging
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone, timedelta
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

# ─── Координаты станций (lat, lon) ─────────────────────────────────────────
ICAO_COORDS: dict[str, tuple[float, float]] = {
    "KAUS": (30.1975, -97.6664),   # Austin
    "KMIA": (25.7959, -80.2870),   # Miami
    "KORD": (41.9742, -87.9073),   # Chicago
    "KDAL": (32.8473, -96.8517),   # Dallas Love Field
    "KATL": (33.6407, -84.4277),   # Atlanta
    "KSFO": (37.6213, -122.3790),  # San Francisco
    "KLAX": (33.9416, -118.4085),  # Los Angeles
    "EGLC": (51.5048,   0.0495),   # London City
    "ZSPD": (31.1443, 121.8083),   # Shanghai
    "LIMC": (45.6301,   8.7281),   # Milan
    "EDDM": (48.3538,  11.7861),   # Munich
    "ZBAA": (40.0799, 116.6031),   # Beijing
    "RCSS": (25.0694, 121.5525),   # Taipei
    "WSSS": ( 1.3644, 103.9915),   # Singapore
    "NZWN": (-41.3272, 174.8050),  # Wellington
}

# Только US-ICAO поддерживают HRRR/NAM
US_ICAO = {k for k in ICAO_COORDS if k.startswith("K")}

FORECAST_DAYS = 4
GEFS_MEMBERS  = 31   # member01 … member30 (+ control = 31 total in Open-Meteo API)

# ─── Herbie cache directory ─────────────────────────────────────────────────
_CACHE_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "herbie_cache")
os.makedirs(_CACHE_DIR, exist_ok=True)

# ─── In-memory result cache (TTL = 2h) ─────────────────────────────────────
_mem_cache: dict[str, dict[str, Any]] = {}
_CACHE_TTL = 7200  # seconds


def _cache_key(icao: str) -> str:
    return icao


def _cached(icao: str) -> dict[str, Any] | None:
    entry = _mem_cache.get(_cache_key(icao))
    if entry and time.time() - entry.get("_ts", 0) < _CACHE_TTL:
        return entry
    return None


def _store(icao: str, data: dict[str, Any]) -> None:
    data["_ts"] = time.time()
    _mem_cache[_cache_key(icao)] = data


# ─── Helpers ────────────────────────────────────────────────────────────────

def _latest_model_run(model: str, max_age_h: int = 6) -> datetime:
    """Возвращает последний run модели (кратный max_age_h часам, UTC)."""
    now = datetime.now(timezone.utc)
    run_hour = (now.hour // max_age_h) * max_age_h
    run_dt = now.replace(hour=run_hour, minute=0, second=0, microsecond=0)
    # Если run был слишком недавно — берём предыдущий (данные ещё не готовы)
    if (now - run_dt).total_seconds() < 3600:
        run_dt -= timedelta(hours=max_age_h)
    return run_dt


def _day_temp_range(hourly_temps: list[float], valid_times: list[datetime],
                    target_date: datetime.date) -> tuple[float | None, float | None]:
    """Вычисляет max/min температуры за конкретный день из почасового ряда."""
    day_vals = [
        t for t, vt in zip(hourly_temps, valid_times)
        if vt.date() == target_date
    ]
    if not day_vals:
        return None, None
    return max(day_vals), min(day_vals)


# ─── Синхронные функции для запуска в ThreadPoolExecutor ───────────────────

def _fetch_gefs_point(icao: str, lat: float, lon: float) -> dict[str, Any]:
    """
    Загружает GEFS (31 member) и вычисляет:
      - gefs_max_daily[day][member] — все члены
      - gefs_mean_max / gefs_mean_min — среднее по членам
      - gefs_spread_max / _min — стандартное отклонение
    Использует Open-Meteo Ensemble API (бесплатно, без GRIB).
    """
    import httpx

    result: dict[str, Any] = {
        "gefs_members_max": [[] for _ in range(FORECAST_DAYS)],
        "gefs_members_min": [[] for _ in range(FORECAST_DAYS)],
        "gefs_mean_max":    [None] * FORECAST_DAYS,
        "gefs_mean_min":    [None] * FORECAST_DAYS,
        "gefs_spread_max":  [None] * FORECAST_DAYS,
        "gefs_spread_min":  [None] * FORECAST_DAYS,
        "gefs_available":   False,
    }
    try:
        url = (
            f"https://ensemble-api.open-meteo.com/v1/ensemble"
            f"?latitude={lat}&longitude={lon}"
            f"&daily=temperature_2m_max,temperature_2m_min"
            f"&models=gfs_seamless"           # GFS ensemble = 31 members
            f"&forecast_days={FORECAST_DAYS}"
            f"&timezone=UTC"
        )
        with httpx.Client(timeout=20.0) as client:
            r = client.get(url)
            if r.status_code != 200:
                logger.warning(f"[Herbie/GEFS] HTTP {r.status_code} for {icao}")
                return result

            daily = r.json().get("daily", {})
            max_keys = sorted([k for k in daily if "temperature_2m_max" in k])
            min_keys = sorted([k for k in daily if "temperature_2m_min" in k])

            for d in range(FORECAST_DAYS):
                max_vals = [float(daily[k][d]) for k in max_keys
                            if d < len(daily.get(k, [])) and daily[k][d] is not None]
                min_vals = [float(daily[k][d]) for k in min_keys
                            if d < len(daily.get(k, [])) and daily[k][d] is not None]

                result["gefs_members_max"][d] = max_vals
                result["gefs_members_min"][d] = min_vals

                if max_vals:
                    result["gefs_mean_max"][d] = round(float(np.mean(max_vals)), 2)
                    result["gefs_spread_max"][d] = round(float(np.std(max_vals)), 2)
                if min_vals:
                    result["gefs_mean_min"][d] = round(float(np.mean(min_vals)), 2)
                    result["gefs_spread_min"][d] = round(float(np.std(min_vals)), 2)

            n_members = len(max_keys)
            result["gefs_available"] = n_members > 0
            result["gefs_n_members"] = n_members
            logger.info(f"[Herbie/GEFS] {icao}: {n_members} members, day0 max={result['gefs_mean_max'][0]:.1f}°C spread={result['gefs_spread_max'][0]:.2f}°C")

    except Exception as exc:
        logger.warning(f"[Herbie/GEFS] {icao} failed: {exc}")
    return result


def _fetch_ecmwf_ens_point(icao: str, lat: float, lon: float) -> dict[str, Any]:
    """
    Loads an additional ensemble model via Open-Meteo.
    Tries icon_seamless (40 members, free global) since ecmwf_ensemble
    requires a paid Open-Meteo subscription.
    """
    import httpx

    result: dict[str, Any] = {
        "ecmwf_ens_mean_max":    [None] * FORECAST_DAYS,
        "ecmwf_ens_mean_min":    [None] * FORECAST_DAYS,
        "ecmwf_ens_spread_max":  [None] * FORECAST_DAYS,
        "ecmwf_ens_spread_min":  [None] * FORECAST_DAYS,
        "ecmwf_ens_members_max": [[] for _ in range(FORECAST_DAYS)],
        "ecmwf_ens_available":   False,
    }
    # icon_seamless = ICON EU + ICON Global ensemble (40 members, free)
    candidate_models = ["icon_seamless", "gem_global"]
    for model_name in candidate_models:
        try:
            url = (
                f"https://ensemble-api.open-meteo.com/v1/ensemble"
                f"?latitude={lat}&longitude={lon}"
                f"&daily=temperature_2m_max,temperature_2m_min"
                f"&models={model_name}"
                f"&forecast_days={FORECAST_DAYS}"
                f"&timezone=UTC"
            )
            with httpx.Client(timeout=20.0) as client:
                r = client.get(url)
                if r.status_code != 200:
                    continue
                daily = r.json().get("daily", {})
                max_keys = sorted([k for k in daily
                                   if "temperature_2m_max" in k and k != "time"])
                min_keys = sorted([k for k in daily
                                   if "temperature_2m_min" in k and k != "time"])
                if not max_keys:
                    continue

                for d in range(FORECAST_DAYS):
                    mx = [float(daily[k][d]) for k in max_keys
                          if d < len(daily.get(k, [])) and daily[k][d] is not None]
                    mn = [float(daily[k][d]) for k in min_keys
                          if d < len(daily.get(k, [])) and daily[k][d] is not None]
                    result["ecmwf_ens_members_max"][d] = mx
                    if mx:
                        result["ecmwf_ens_mean_max"][d]   = round(float(np.mean(mx)), 2)
                        result["ecmwf_ens_spread_max"][d] = round(float(np.std(mx)), 2)
                    if mn:
                        result["ecmwf_ens_mean_min"][d]   = round(float(np.mean(mn)), 2)
                        result["ecmwf_ens_spread_min"][d] = round(float(np.std(mn)), 2)

                result["ecmwf_ens_available"]  = True
                result["ecmwf_ens_n_members"]  = len(max_keys)
                result["ecmwf_ens_model"]      = model_name
                logger.info(f"[Herbie/ENS2] {icao}: {model_name} {len(max_keys)} members, "
                            f"day0 max={result['ecmwf_ens_mean_max'][0]}C")
                return result  # success
        except Exception as exc:
            logger.debug(f"[Herbie/ENS2] {icao} {model_name}: {exc}")

    logger.warning(f"[Herbie/ENS2] {icao}: all secondary ensemble models failed")
    return result


def _fetch_deterministic_models(icao: str, lat: float, lon: float) -> dict[str, Any]:
    """
    Загружает детерминированные модели через Open-Meteo:
      GFS, ECMWF IFS, HRRR (US only), NAM (US only), NBM (US only)
    Возвращает daily max/min для каждой модели на 4 дня вперёд.
    """
    import httpx

    is_us = icao in US_ICAO
    result: dict[str, Any] = {
        "gfs_max":  [None] * FORECAST_DAYS,
        "gfs_min":  [None] * FORECAST_DAYS,
        "ecmwf_max": [None] * FORECAST_DAYS,
        "ecmwf_min": [None] * FORECAST_DAYS,
        "hrrr_max": [None] * FORECAST_DAYS,
        "hrrr_min": [None] * FORECAST_DAYS,
        "nam_max":  [None] * FORECAST_DAYS,
        "nam_min":  [None] * FORECAST_DAYS,
        "nbm_max":  [None] * FORECAST_DAYS,
        "nbm_min":  [None] * FORECAST_DAYS,
    }

    models_global = [
        ("gfs_seamless",  "https://api.open-meteo.com/v1/forecast", "gfs"),
        ("ecmwf_ifs04",   "https://api.open-meteo.com/v1/forecast", "ecmwf"),
    ]
    models_us = [
        ("hrrr_conus", "https://api.open-meteo.com/v1/gfs", "hrrr"),
        ("nam_conus",  "https://api.open-meteo.com/v1/gfs", "nam"),
        ("nbm_conus",  "https://api.open-meteo.com/v1/gfs", "nbm"),
    ]

    to_fetch = models_global + (models_us if is_us else [])

    with httpx.Client(timeout=15.0) as client:
        for model_id, base_url, key in to_fetch:
            try:
                url = (
                    f"{base_url}?latitude={lat}&longitude={lon}"
                    f"&daily=temperature_2m_max,temperature_2m_min"
                    f"&models={model_id}"
                    f"&forecast_days={FORECAST_DAYS}"
                    f"&timezone=UTC"
                )
                r = client.get(url)
                if r.status_code == 200:
                    daily = r.json().get("daily", {})
                    maxs = daily.get("temperature_2m_max", [])
                    mins = daily.get("temperature_2m_min", [])
                    for d in range(min(FORECAST_DAYS, len(maxs))):
                        if maxs[d] is not None:
                            result[f"{key}_max"][d] = round(float(maxs[d]), 2)
                        if d < len(mins) and mins[d] is not None:
                            result[f"{key}_min"][d] = round(float(mins[d]), 2)
            except Exception as exc:
                logger.debug(f"[Herbie/det] {icao} {model_id}: {exc}")

    return result


# ─── Основная функция ───────────────────────────────────────────────────────

def fetch_herbie_data_sync(icao: str) -> dict[str, Any]:
    """
    Синхронная точка входа для получения всех данных Herbie.
    Запускает 3 параллельных задачи в ThreadPoolExecutor.
    """
    cached = _cached(icao)
    if cached:
        logger.debug(f"[Herbie] Cache hit for {icao}")
        return cached

    coords = ICAO_COORDS.get(icao)
    if not coords:
        logger.warning(f"[Herbie] No coordinates for {icao}")
        return {"herbie_available": False}

    lat, lon = coords
    logger.info(f"[Herbie] Fetching all models for {icao} ({lat}, {lon})")

    result: dict[str, Any] = {"herbie_available": False, "icao": icao}

    # Параллельный запуск 3 задач
    tasks = {
        "gefs":  (_fetch_gefs_point,             icao, lat, lon),
        "ecmwf": (_fetch_ecmwf_ens_point,         icao, lat, lon),
        "det":   (_fetch_deterministic_models,    icao, lat, lon),
    }

    with ThreadPoolExecutor(max_workers=3) as pool:
        futures = {pool.submit(fn, *args): name for name, (fn, *args) in tasks.items()}
        for future in as_completed(futures):
            name = futures[future]
            try:
                data = future.result(timeout=30)
                result.update(data)
            except Exception as exc:
                logger.error(f"[Herbie] Task '{name}' failed for {icao}: {exc}")

    # Вычисляем multi-model ensemble consensus
    result.update(_compute_consensus(result))
    result["herbie_available"] = result.get("gefs_available", False) or result.get("ecmwf_ens_available", False)
    result["fetched_at"] = time.time()

    _store(icao, result)
    return result


def _compute_consensus(r: dict[str, Any]) -> dict[str, Any]:
    """
    Вычисляет взвешенный консенсус по всем доступным моделям.
    Возвращает consensus_max/min и model_spread.
    """
    consensus: dict[str, Any] = {
        "consensus_max": [None] * FORECAST_DAYS,
        "consensus_min": [None] * FORECAST_DAYS,
        "model_spread_max": [None] * FORECAST_DAYS,
        "model_spread_min": [None] * FORECAST_DAYS,
        "n_models_max": [0] * FORECAST_DAYS,
    }

    # Модели с одинаковым весом (позже заменим на BMA из Фазы 2)
    model_keys_max = ["gfs_max", "ecmwf_max", "hrrr_max", "nam_max",
                      "gefs_mean_max", "ecmwf_ens_mean_max"]
    model_keys_min = ["gfs_min", "ecmwf_min", "hrrr_min", "nam_min",
                      "gefs_mean_min", "ecmwf_ens_mean_min"]

    for d in range(FORECAST_DAYS):
        max_vals = [r[k][d] for k in model_keys_max if r.get(k) and r[k][d] is not None]
        min_vals = [r[k][d] for k in model_keys_min if r.get(k) and r[k][d] is not None]

        if max_vals:
            consensus["consensus_max"][d] = round(float(np.mean(max_vals)), 2)
            consensus["model_spread_max"][d] = round(float(np.std(max_vals)), 2)
            consensus["n_models_max"][d] = len(max_vals)
        if min_vals:
            consensus["consensus_min"][d] = round(float(np.mean(min_vals)), 2)
            consensus["model_spread_min"][d] = round(float(np.std(min_vals)), 2)

    return consensus


async def fetch_herbie_data(icao: str) -> dict[str, Any]:
    """Async обёртка для использования в существующем async коде."""
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, fetch_herbie_data_sync, icao)


# ─── Интеграция с weather_data.py ──────────────────────────────────────────

def enrich_weather_data(weather_data: dict, herbie_data: dict) -> dict:
    """
    Enriches existing weather_data dict with Herbie data.
    Adds new keys to forecast_daily without overwriting existing ones.
    Handles list values correctly (previous version skipped them).
    """
    if not herbie_data.get("herbie_available"):
        return weather_data

    fd = weather_data.setdefault("forecast_daily", {})

    # Keys to add (explicit list so lists are handled correctly)
    keys_to_add = [
        "gefs_mean_max", "gefs_mean_min",
        "gefs_spread_max", "gefs_spread_min",
        "gefs_members_max", "gefs_members_min",
        "gefs_n_members",
        "ecmwf_ens_mean_max", "ecmwf_ens_mean_min",
        "ecmwf_ens_spread_max",
        "ecmwf_ens_members_max",
        "ecmwf_ens_n_members",
        "consensus_max", "consensus_min",
        "model_spread_max", "model_spread_min",
        "n_models_max",
    ]
    for k in keys_to_add:
        if k not in fd and k in herbie_data and herbie_data[k] is not None:
            fd[k] = herbie_data[k]

    weather_data["forecast_daily"] = fd
    weather_data["herbie_available"] = True
    weather_data["herbie_fetched_at"] = herbie_data.get("fetched_at")
    return weather_data


# ─── Точечный тест вероятностей по членам GEFS ─────────────────────────────

def gefs_bucket_probability(herbie_data: dict, day_idx: int,
                             bin_low_c: float, bin_high_c: float,
                             use_max: bool = True) -> float | None:
    """
    Прямой подсчёт вероятности попадания в бакет по членам GEFS.
    Возвращает float [0, 1] или None если данных нет.

    Параметры:
        day_idx    — индекс дня (0=сегодня, 1=завтра, …)
        bin_low_c  — нижняя граница бакета в Цельсиях
        bin_high_c — верхняя граница бакета в Цельсиях
        use_max    — True=по дневному максимуму, False=по минимуму
    """
    key = "gefs_members_max" if use_max else "gefs_members_min"
    members_by_day = herbie_data.get(key, [])
    if not members_by_day or day_idx >= len(members_by_day):
        return None
    members = members_by_day[day_idx]
    if not members:
        return None

    in_bin = sum(1 for t in members if bin_low_c <= t < bin_high_c)
    return round(in_bin / len(members), 4)


def ecmwf_bucket_probability(herbie_data: dict, day_idx: int,
                              bin_low_c: float, bin_high_c: float,
                              use_max: bool = True) -> float | None:
    """Аналогично gefs_bucket_probability, но для ECMWF ensemble."""
    key = "ecmwf_ens_members_max" if use_max else "ecmwf_ens_members_min"
    # ecmwf_ens_members_min пока не реализован отдельно — добавим в фазе 2
    members_by_day = herbie_data.get(key, [])
    if not members_by_day or day_idx >= len(members_by_day):
        return None
    members = members_by_day[day_idx]
    if not members:
        return None

    in_bin = sum(1 for t in members if bin_low_c <= t < bin_high_c)
    return round(in_bin / len(members), 4)
