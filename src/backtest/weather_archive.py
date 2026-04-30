from __future__ import annotations

import csv
from datetime import date, datetime, timedelta, timezone
from io import StringIO
from typing import Any, Optional

import httpx

from src.backtest.models import ForecastSnapshot
from src.probability_calculator import celsius_to_fahrenheit
from src.utils import logger
from src.weather_data import ICAO_COORDS


def fahrenheit_to_celsius(value: float) -> float:
    return (value - 32.0) * 5.0 / 9.0


class WeatherArchive:
    def __init__(self):
        self.client = httpx.AsyncClient(timeout=30.0)
        self._actual_daily_cache: dict[tuple[str, date], dict[str, Optional[float]]] = {}
        self._iem_rows_cache: dict[tuple[str, date], list[dict[str, str]]] = {}
        self._forecast_cache: dict[tuple[str, date, str], dict[str, Any]] = {}

    async def close(self) -> None:
        await self.client.aclose()

    async def fetch_actual_daily(self, icao: str, target_date: date) -> dict[str, Optional[float]]:
        """Fetch observed daily high/low, preferring Open-Meteo archive for stability."""
        cache_key = (icao.upper(), target_date)
        cached = self._actual_daily_cache.get(cache_key)
        if cached is not None:
            return dict(cached)

        coords = ICAO_COORDS.get(icao)
        if coords:
            lat, lon = coords
            actuals = await self._fetch_open_meteo_actual_daily(lat, lon, target_date)
            if actuals.get("actual_high_c") is not None or actuals.get("actual_low_c") is not None:
                self._actual_daily_cache[cache_key] = actuals
                return dict(actuals)

        url = (
            "https://mesonet.agron.iastate.edu/cgi-bin/request/asos.py"
            f"?station={icao}"
            "&data=tmpf"
            f"&year1={target_date.year}&month1={target_date.month}&day1={target_date.day}"
            f"&year2={target_date.year}&month2={target_date.month}&day2={target_date.day}"
            "&tz=Etc/UTC&format=onlycomma&latlon=no&elev=no&missing=M&trace=T"
            "&direct=no&report_type=1&report_type=2"
        )
        try:
            response = await self.client.get(url)
            response.raise_for_status()
        except Exception as exc:
            logger.warning(f"[BACKTEST] IEM actual fetch failed for {icao} {target_date}: {exc}")
            result = {"actual_high_c": None, "actual_low_c": None}
            self._actual_daily_cache[cache_key] = result
            return dict(result)

        temps_f: list[float] = []
        reader = csv.DictReader(StringIO(response.text))
        for row in reader:
            raw = row.get("tmpf")
            if raw in (None, "", "M"):
                continue
            try:
                temps_f.append(float(raw))
            except ValueError:
                continue

        if not temps_f:
            result = {"actual_high_c": None, "actual_low_c": None}
            self._actual_daily_cache[cache_key] = result
            return dict(result)

        result = {
            "actual_high_c": fahrenheit_to_celsius(max(temps_f)),
            "actual_low_c": fahrenheit_to_celsius(min(temps_f)),
        }
        self._actual_daily_cache[cache_key] = result
        return dict(result)

    async def _fetch_open_meteo_actual_daily(self, lat: float, lon: float, target_date: date) -> dict[str, Optional[float]]:
        url = (
            "https://archive-api.open-meteo.com/v1/archive"
            f"?latitude={lat}&longitude={lon}"
            f"&start_date={target_date.isoformat()}&end_date={target_date.isoformat()}"
            "&hourly=temperature_2m"
            "&timezone=UTC"
        )
        try:
            response = await self.client.get(url)
            response.raise_for_status()
            hourly = response.json().get("hourly", {}) or {}
            temps = [float(v) for v in (hourly.get("temperature_2m") or []) if v is not None]
        except Exception as exc:
            logger.warning(f"[BACKTEST] Open-Meteo actual fetch failed for {target_date}: {exc}")
            return {"actual_high_c": None, "actual_low_c": None}

        if not temps:
            return {"actual_high_c": None, "actual_low_c": None}

        return {"actual_high_c": max(temps), "actual_low_c": min(temps)}

    async def fetch_forecast_snapshot(self, icao: str, target_date: date, as_of: datetime) -> ForecastSnapshot:
        coords = ICAO_COORDS.get(icao)
        if not coords:
            return ForecastSnapshot(icao, target_date, as_of, {"forecast_daily": {}}, "missing_coords")

        lat, lon = coords
        previous_days = max(0, min(7, (target_date - as_of.date()).days))
        cache_key = (icao.upper(), target_date, f"prev:{previous_days}")
        cached_weather = self._forecast_cache.get(cache_key)
        if cached_weather is None:
            cached_weather = await self._fetch_open_meteo_previous_runs(lat, lon, target_date, previous_days)
            self._forecast_cache[cache_key] = cached_weather
        weather_data = dict(cached_weather)
        source = "open_meteo_previous_runs"
        if not _has_forecast_daily(weather_data):
            hist_key = (icao.upper(), target_date, "hist")
            cached_hist = self._forecast_cache.get(hist_key)
            if cached_hist is None:
                cached_hist = await self._fetch_open_meteo_historical_forecast(lat, lon, target_date)
                self._forecast_cache[hist_key] = cached_hist
            weather_data = dict(cached_hist)
            source = "open_meteo_historical_forecast"

        weather_data["forecast_hourly"] = _trim_hourly_forecast(weather_data.get("forecast_hourly", {}), as_of)
        weather_data["metar"] = await self._fetch_metar_snapshot(icao, as_of)
        return ForecastSnapshot(icao, target_date, as_of, weather_data, source)

    async def _fetch_open_meteo_previous_runs(
        self,
        lat: float,
        lon: float,
        target_date: date,
        previous_days: int,
    ) -> dict[str, Any]:
        base = "https://previous-runs-api.open-meteo.com/v1/forecast"
        common = (
            f"?latitude={lat}&longitude={lon}"
            f"&start_date={target_date.isoformat()}&end_date={target_date.isoformat()}"
            "&daily=temperature_2m_max,temperature_2m_min"
            "&hourly=temperature_2m,dew_point_2m,cloud_cover,wind_speed_10m,wind_direction_10m"
            f"&previous_days={previous_days}"
            "&timezone=auto"
        )
        return await self._fetch_open_meteo_models(base, common)

    async def _fetch_open_meteo_historical_forecast(self, lat: float, lon: float, target_date: date) -> dict[str, Any]:
        base = "https://historical-forecast-api.open-meteo.com/v1/forecast"
        common = (
            f"?latitude={lat}&longitude={lon}"
            f"&start_date={target_date.isoformat()}&end_date={target_date.isoformat()}"
            "&daily=temperature_2m_max,temperature_2m_min"
            "&hourly=temperature_2m,dew_point_2m,cloud_cover,wind_speed_10m,wind_direction_10m"
            "&timezone=auto"
        )
        return await self._fetch_open_meteo_models(base, common)

    async def _fetch_open_meteo_models(self, base: str, common_query: str) -> dict[str, Any]:
        daily = {
            "ecmwf_max": [None],
            "ecmwf_min": [None],
            "gfs_max": [None],
            "gfs_min": [None],
            "ensemble_max": [None],
            "ensemble_min": [None],
            "ensemble_std": [None],
        }
        hourly: dict[str, list[Any]] = {
            "time": [],
            "temperature_2m": [],
            "dew_point_2m": [],
            "cloud_cover": [],
            "wind_speed_10m": [],
            "wind_direction_10m": [],
        }
        for model_name, max_key, min_key in (
            ("ecmwf_ifs04", "ecmwf_max", "ecmwf_min"),
            ("gfs_seamless", "gfs_max", "gfs_min"),
            ("ecmwf_ensemble", "ensemble_max", "ensemble_min"),
        ):
            try:
                response = await self.client.get(f"{base}{common_query}&models={model_name}")
                if response.status_code != 200:
                    continue
                block = response.json().get("daily", {})
                maxs = block.get("temperature_2m_max", [])
                mins = block.get("temperature_2m_min", [])
                if maxs:
                    daily[max_key] = [float(maxs[0]) if maxs[0] is not None else None]
                if mins:
                    daily[min_key] = [float(mins[0]) if mins[0] is not None else None]
                hourly_block = response.json().get("hourly", {}) or {}
                if not hourly["time"] and hourly_block.get("time"):
                    for key in hourly.keys():
                        values = hourly_block.get(key)
                        if isinstance(values, list):
                            hourly[key] = list(values)
            except Exception as exc:
                logger.debug(f"[BACKTEST] Open-Meteo model fetch failed for {model_name}: {exc}")

        return {"forecast_daily": daily, "metar": [], "taf": [], "forecast_hourly": hourly}

    async def _fetch_iem_rows(self, icao: str, obs_date: date) -> list[dict[str, str]]:
        cache_key = (icao.upper(), obs_date)
        cached = self._iem_rows_cache.get(cache_key)
        if cached is not None:
            return cached

        url = (
            "https://mesonet.agron.iastate.edu/cgi-bin/request/asos.py"
            f"?station={icao}"
            "&data=tmpf&data=dwpf&data=drct&data=sknt"
            "&data=skyc1&data=skyc2&data=skyc3&data=skyc4"
            "&data=skyl1&data=skyl2&data=skyl3&data=skyl4"
            "&data=metar"
            f"&year1={obs_date.year}&month1={obs_date.month}&day1={obs_date.day}"
            f"&year2={obs_date.year}&month2={obs_date.month}&day2={obs_date.day}"
            "&tz=Etc/UTC&format=onlycomma&latlon=no&elev=no&missing=M&trace=T"
            "&direct=no&report_type=1&report_type=2"
        )
        try:
            response = await self.client.get(url)
            response.raise_for_status()
            rows = list(csv.DictReader(StringIO(response.text)))
        except Exception as exc:
            logger.warning(f"[BACKTEST] IEM obs fetch failed for {icao} {obs_date}: {exc}")
            rows = []

        self._iem_rows_cache[cache_key] = rows
        return rows

    async def _fetch_metar_snapshot(self, icao: str, as_of: datetime) -> list[dict[str, Any]]:
        rows = await self._fetch_iem_rows(icao, as_of.date())
        latest: Optional[dict[str, str]] = None
        latest_ts: Optional[datetime] = None

        for row in rows:
            valid = row.get("valid")
            if not valid:
                continue
            try:
                ts = datetime.fromisoformat(valid.replace("Z", "+00:00"))
                if ts.tzinfo is None:
                    ts = ts.replace(tzinfo=timezone.utc)
            except Exception:
                continue
            if ts > as_of:
                continue
            if latest_ts is None or ts > latest_ts:
                latest = row
                latest_ts = ts

        if latest is None:
            return []

        temp_c = _maybe_f_to_c(latest.get("tmpf"))
        dew_c = _maybe_f_to_c(latest.get("dwpf"))
        return [
            {
                "time": latest_ts.isoformat() if latest_ts else "",
                "temp": temp_c,
                "dewp": dew_c,
                "rawOb": latest.get("metar") or "",
                "drct": latest.get("drct"),
                "sknt": latest.get("sknt"),
            }
        ]


def _has_forecast_daily(weather_data: dict[str, Any]) -> bool:
    daily = weather_data.get("forecast_daily", {})
    for key in ("ecmwf_max", "gfs_max", "ensemble_max"):
        values = daily.get(key) or []
        if values and values[0] is not None:
            return True
    return False


def _maybe_f_to_c(value: Any) -> Optional[float]:
    if value in (None, "", "M"):
        return None
    try:
        return fahrenheit_to_celsius(float(value))
    except (TypeError, ValueError):
        return None


def _trim_hourly_forecast(hourly: dict[str, Any], as_of: datetime) -> dict[str, list[Any]]:
    if not hourly:
        return {}

    times = hourly.get("time") or []
    if not times:
        return hourly

    start_idx = 0
    parsed_times: list[datetime] = []
    for raw in times:
        try:
            dt = datetime.fromisoformat(str(raw).replace("Z", "+00:00"))
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            parsed_times.append(dt)
        except Exception:
            parsed_times.append(as_of)

    for idx, dt in enumerate(parsed_times):
        if dt >= as_of:
            start_idx = idx
            break
    else:
        start_idx = max(0, len(parsed_times) - 1)

    trimmed: dict[str, list[Any]] = {}
    for key, values in hourly.items():
        if isinstance(values, list):
            trimmed[key] = values[start_idx:]
    return trimmed


def snapshot_times(target_date: date, leads_hours: list[int]) -> list[datetime]:
    target_dt = datetime.combine(target_date, datetime.min.time(), tzinfo=timezone.utc) + timedelta(hours=12)
    return [target_dt - timedelta(hours=hours) for hours in leads_hours]
