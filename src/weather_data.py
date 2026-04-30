import httpx
import time
from typing import Dict, Any, List, Optional
from config.settings import config
from src.utils import logger

# Hardcoded coordinates mapping for precise Open-Meteo queries based on config standard
ICAO_COORDS = {
    "ZSPD": (31.1443, 121.8083), # Shanghai
    "KAUS": (30.1975, -97.6664), # Austin
    "NZWN": (-41.3272, 174.8050),# Wellington
    "KSFO": (37.6213, -122.3790),# San Francisco
    "KMIA": (25.7959, -80.2870), # Miami
    "KATL": (33.6407, -84.4277), # Atlanta
    "WSSS": (1.3644, 103.9915),  # Singapore
    "KORD": (41.9742, -87.9073), # Chicago
    "KLAX": (33.9416, -118.4085),# Los Angeles
    "EGLC": (51.5048,  0.0495),  # London City
    "KDAL": (32.8473, -96.8517), # Dallas Love Field
    "LIMC": (45.6301, 8.7281),   # Milan Malpensa
    "EDDM": (48.3538, 11.7861),  # Munich
    "ZBAA": (40.0799, 116.6031), # Beijing
    "RCSS": (25.0694, 121.5525), # Taipei Songshan
}


US_ICAO_PREFIXES = ("K",)


def _is_us_icao(icao: str) -> bool:
    return str(icao or "").upper().startswith(US_ICAO_PREFIXES)

class WeatherFetcher:
    def __init__(self):
        self.base_url_metar = "https://aviationweather.gov/api/data/metar"
        self.base_url_taf = "https://aviationweather.gov/api/data/taf"
        self.cache: Dict[str, Dict[str, Any]] = {}
        self.cache_ttl = 1200  # 20 minutes — METAR updates every 20-30 min
        self.open_meteo_cache: Dict[str, Any] = {}
        self.open_meteo_cache_ttl = int(getattr(config, "tsas_open_meteo_cache_ttl_seconds", 1800))
        self.open_meteo_cache_ttl = 10800  # 3 hours — models update every 6h

    async def fetch_weather_for_icao(self, icao_codes: List[str]) -> Optional[Dict[str, Dict[str, Any]]]:
        """
        Fetches METAR and TAF data for a list of ICAO codes with 3 retries.
        - METAR: Restored 'hours=24' for full 24h context.
        - TAF: Fetched in one batch for speed (site issues resolved).
        """
        import asyncio
        ids_str = ",".join(icao_codes)
        results = {code: {"metar": [], "taf": [], "last_updated": time.time()} for code in icao_codes}
        
        # 1. Fetch METAR (Up to 3 attempts)
        metar_success = False
        for attempt in range(1, 4):
            try:
                async with httpx.AsyncClient() as client:
                    # Restored &hours=24 for deeper AI context
                    res_metar = await client.get(
                        f"{self.base_url_metar}?ids={ids_str}&format=json&hours=24", 
                        timeout=25.0
                    )
                    res_metar.raise_for_status()
                    metar_data = res_metar.json()
                    
                    found_any = False
                    for item in metar_data:
                        icao = item.get("icaoId")
                        if icao in results:
                            results[icao]["metar"].append(item)
                            found_any = True
                    
                    if found_any:
                        metar_success = True
                        break
                    else:
                        logger.warning(f"[WEATHER] METAR empty response for {ids_str} (Attempt {attempt})")
            except Exception as e:
                logger.error(f"[WEATHER] METAR fetch error (Attempt {attempt}): {e}")
            
            if attempt < 3:
                await asyncio.sleep(attempt * 5)
        
        if not metar_success:
            logger.error(f"[CRITICAL] Failed to fetch METAR after 3 attempts for {ids_str}")
            return None

        # 2. Fetch TAF (Up to 3 attempts - Batch mode restored)
        taf_success = False
        for attempt in range(1, 4):
            try:
                async with httpx.AsyncClient() as client:
                    res_taf = await client.get(
                        f"{self.base_url_taf}?ids={ids_str}&format=json", 
                        timeout=25.0
                    )
                    res_taf.raise_for_status()
                    taf_data = res_taf.json()
                    
                    found_any = False
                    for item in taf_data:
                        icao = item.get("icaoId")
                        if icao in results:
                            results[icao]["taf"].append(item)
                            found_any = True
                    
                    if found_any:
                        taf_success = True
                        break
                    else:
                        logger.warning(f"[WEATHER] TAF empty response for {ids_str} (Attempt {attempt})")
            except Exception as e:
                logger.error(f"[WEATHER] TAF fetch error (Attempt {attempt}): {e}")
            
            if attempt < 3:
                await asyncio.sleep(attempt * 5)

        if not taf_success:
            logger.error(f"[CRITICAL] Failed to fetch TAF after 3 attempts for {ids_str}")
            return None

        # Update cache for successful results
        for icao, data in results.items():
            if data["metar"] or data["taf"]:
                self.cache[icao] = data
                
        return results

    async def fetch_open_meteo(self, icao: str, force_refresh: bool = False) -> Dict[str, Any]:
        """
        Fetches ECMWF, GFS and Ensemble forecasts from Open-Meteo.
        Results are cached for 3 hours (models update every 6h).
        Returns a dict with both summary strings and numerical fields.
        """
        import time as time_module
        cache_ttl = int(getattr(config, "tsas_open_meteo_cache_ttl_seconds", self.open_meteo_cache_ttl))
        cached = self.open_meteo_cache.get(icao)
        if (not force_refresh) and cached and time_module.time() - cached.get("_fetched_at", 0) < cache_ttl:
            return cached
        res_data: Dict[str, Any] = {}
        coords = ICAO_COORDS.get(icao)
        if not coords:
            return {}

        lat, lon = coords
        FORECAST_DAYS = 4  # cover today + 3 future days for markets resolving up to ~4 days out
        use_us_short_range = _is_us_icao(icao)

        # ── 1. ECMWF IFS — 4 days (Max & Min) ────────────────────────────────
        ecmwf_max_daily: list[float | None] = [None] * FORECAST_DAYS
        ecmwf_min_daily: list[float | None] = [None] * FORECAST_DAYS
        try:
            url = (
                f"https://api.open-meteo.com/v1/forecast"
                f"?latitude={lat}&longitude={lon}"
                f"&daily=temperature_2m_max,temperature_2m_min"
                f"&models=ecmwf_ifs04"
                f"&forecast_days={FORECAST_DAYS}"
                f"&timezone=auto"
            )
            async with httpx.AsyncClient() as client:
                r = await client.get(url, timeout=12.0)
                if r.status_code == 200:
                    daily = r.json().get("daily", {})
                    maxs = daily.get("temperature_2m_max", [])
                    mins = daily.get("temperature_2m_min", [])
                    for i in range(min(FORECAST_DAYS, len(maxs))):
                        if maxs[i] is not None: ecmwf_max_daily[i] = float(maxs[i])
                        if i < len(mins) and mins[i] is not None: ecmwf_min_daily[i] = float(mins[i])
        except Exception:
            pass

        # ── 2. GFS Seamless — 4 days (Max & Min) ─────────────────────────────
        gfs_max_daily: list[float | None] = [None] * FORECAST_DAYS
        gfs_min_daily: list[float | None] = [None] * FORECAST_DAYS
        try:
            url = (
                f"https://api.open-meteo.com/v1/forecast"
                f"?latitude={lat}&longitude={lon}"
                f"&daily=temperature_2m_max,temperature_2m_min"
                f"&models=gfs_seamless"
                f"&forecast_days={FORECAST_DAYS}"
                f"&timezone=auto"
            )
            async with httpx.AsyncClient() as client:
                r = await client.get(url, timeout=12.0)
                if r.status_code == 200:
                    daily = r.json().get("daily", {})
                    maxs = daily.get("temperature_2m_max", [])
                    mins = daily.get("temperature_2m_min", [])
                    for i in range(min(FORECAST_DAYS, len(maxs))):
                        if maxs[i] is not None: gfs_max_daily[i] = float(maxs[i])
                        if i < len(mins) and mins[i] is not None: gfs_min_daily[i] = float(mins[i])
        except Exception:
            pass

        # ── 2b. US short-range models (HRRR + NAM) ──────────────────────────
        hrrr_max_daily: list[float | None] = [None] * FORECAST_DAYS
        hrrr_min_daily: list[float | None] = [None] * FORECAST_DAYS
        nam_max_daily: list[float | None] = [None] * FORECAST_DAYS
        nam_min_daily: list[float | None] = [None] * FORECAST_DAYS
        nbm_max_daily: list[float | None] = [None] * FORECAST_DAYS
        nbm_min_daily: list[float | None] = [None] * FORECAST_DAYS

        if use_us_short_range:
            for model_name, max_target, min_target in (
                ("hrrr_conus", hrrr_max_daily, hrrr_min_daily),
                ("nam_conus", nam_max_daily, nam_min_daily),
                ("nbm_conus", nbm_max_daily, nbm_min_daily),
            ):
                try:
                    url = (
                        f"https://api.open-meteo.com/v1/gfs"
                        f"?latitude={lat}&longitude={lon}"
                        f"&daily=temperature_2m_max,temperature_2m_min"
                        f"&models={model_name}"
                        f"&forecast_days={FORECAST_DAYS}"
                        f"&timezone=auto"
                    )
                    async with httpx.AsyncClient() as client:
                        r = await client.get(url, timeout=12.0)
                        if r.status_code == 200:
                            daily = r.json().get("daily", {})
                            maxs = daily.get("temperature_2m_max", [])
                            mins = daily.get("temperature_2m_min", [])
                            for i in range(min(FORECAST_DAYS, len(maxs))):
                                if maxs[i] is not None:
                                    max_target[i] = float(maxs[i])
                                if i < len(mins) and mins[i] is not None:
                                    min_target[i] = float(mins[i])
                except Exception:
                    pass

        # ── 3. ECMWF Ensemble (51 members) — per-day mean & std (Max & Min) ──
        ens_max_daily:     list[float | None] = [None] * FORECAST_DAYS
        ens_max_std_daily: list[float | None] = [None] * FORECAST_DAYS
        ens_min_daily:     list[float | None] = [None] * FORECAST_DAYS
        try:
            url = (
                f"https://ensemble-api.open-meteo.com/v1/ensemble"
                f"?latitude={lat}&longitude={lon}"
                f"&daily=temperature_2m_max,temperature_2m_min"
                f"&models=ecmwf_ensemble"
                f"&forecast_days={FORECAST_DAYS}"
                f"&timezone=auto"
            )
            async with httpx.AsyncClient() as client:
                r = await client.get(url, timeout=15.0)
                if r.status_code == 200:
                    daily_block = r.json().get("daily", {})
                    # Max stats
                    max_members = [v for k, v in daily_block.items() if "temperature_2m_max" in k and isinstance(v, list)]
                    # Min stats
                    min_members = [v for k, v in daily_block.items() if "temperature_2m_min" in k and isinstance(v, list)]
                    
                    import statistics as stats_module
                    for d in range(FORECAST_DAYS):
                        # Max
                        m_vals = [float(arr[d]) for arr in max_members if len(arr) > d and arr[d] is not None]
                        if m_vals:
                            ens_max_daily[d] = round(sum(m_vals)/len(m_vals), 1)
                            ens_max_std_daily[d] = round(stats_module.stdev(m_vals) if len(m_vals) > 1 else 0.0, 2)
                        # Min
                        n_vals = [float(arr[d]) for arr in min_members if len(arr) > d and arr[d] is not None]
                        if n_vals:
                            ens_min_daily[d] = round(sum(n_vals)/len(n_vals), 1)
        except Exception:
            pass

        # ── 4. Package results ───────────────────────────────────────────────
        hourly_temps: list[float] = []
        hourly_cloud_cover: list[float] = []
        hourly_wind_speed: list[float] = []
        current_snapshot: Dict[str, Any] = {}
        try:
            url = (
                f"https://api.open-meteo.com/v1/forecast"
                f"?latitude={lat}&longitude={lon}"
                f"&current=temperature_2m,dew_point_2m,cloud_cover,wind_speed_10m,wind_direction_10m"
                f"&hourly=temperature_2m,cloud_cover,wind_speed_10m"
                f"&forecast_hours=18"
                f"&timezone=auto"
            )
            async with httpx.AsyncClient() as client:
                r = await client.get(url, timeout=12.0)
                if r.status_code == 200:
                    payload = r.json()
                    current_block = payload.get("current", {})
                    hourly_block = payload.get("hourly", {})
                    current_snapshot = {
                        "temperature_2m": current_block.get("temperature_2m"),
                        "dew_point_2m": current_block.get("dew_point_2m"),
                        "cloud_cover": current_block.get("cloud_cover"),
                        "wind_speed_10m": current_block.get("wind_speed_10m"),
                        "wind_direction_10m": current_block.get("wind_direction_10m"),
                    }
                    hourly_temps = [float(v) for v in (hourly_block.get("temperature_2m", []) or []) if v is not None]
                    hourly_cloud_cover = [float(v) for v in (hourly_block.get("cloud_cover", []) or []) if v is not None]
                    hourly_wind_speed = [float(v) for v in (hourly_block.get("wind_speed_10m", []) or []) if v is not None]
        except Exception:
            pass

        hrrr_hourly_temps: list[float] = []
        hrrr_hourly_cloud_cover: list[float] = []
        hrrr_hourly_wind_speed: list[float] = []
        hrrr_hourly_wind_dir: list[float] = []
        hrrr_hourly_dew_point: list[float] = []
        nam_hourly_temps: list[float] = []
        nam_hourly_cloud_cover: list[float] = []
        nam_hourly_wind_speed: list[float] = []
        nam_hourly_wind_dir: list[float] = []
        nam_hourly_dew_point: list[float] = []
        nbm_hourly_temps: list[float] = []

        if use_us_short_range:
            for model_name, sinks in (
                ("hrrr_conus", ("hrrr",)),
                ("nam_conus", ("nam",)),
                ("nbm_conus", ("nbm",)),
            ):
                try:
                    url = (
                        f"https://api.open-meteo.com/v1/gfs"
                        f"?latitude={lat}&longitude={lon}"
                        f"&hourly=temperature_2m,dew_point_2m,cloud_cover,wind_speed_10m,wind_direction_10m"
                        f"&forecast_hours=18"
                        f"&models={model_name}"
                        f"&timezone=auto"
                    )
                    async with httpx.AsyncClient() as client:
                        r = await client.get(url, timeout=12.0)
                        if r.status_code == 200:
                            hourly_block = r.json().get("hourly", {})
                            temps = [float(v) for v in (hourly_block.get("temperature_2m", []) or []) if v is not None]
                            dew = [float(v) for v in (hourly_block.get("dew_point_2m", []) or []) if v is not None]
                            cloud = [float(v) for v in (hourly_block.get("cloud_cover", []) or []) if v is not None]
                            wind = [float(v) for v in (hourly_block.get("wind_speed_10m", []) or []) if v is not None]
                            wind_dir = [float(v) for v in (hourly_block.get("wind_direction_10m", []) or []) if v is not None]

                            if model_name == "hrrr_conus":
                                hrrr_hourly_temps = temps
                                hrrr_hourly_dew_point = dew
                                hrrr_hourly_cloud_cover = cloud
                                hrrr_hourly_wind_speed = wind
                                hrrr_hourly_wind_dir = wind_dir
                            elif model_name == "nam_conus":
                                nam_hourly_temps = temps
                                nam_hourly_dew_point = dew
                                nam_hourly_cloud_cover = cloud
                                nam_hourly_wind_speed = wind
                                nam_hourly_wind_dir = wind_dir
                            elif model_name == "nbm_conus":
                                nbm_hourly_temps = temps
                except Exception:
                    pass

        res_data["forecast_daily"] = {
            "ecmwf_max":    ecmwf_max_daily,
            "ecmwf_min":    ecmwf_min_daily,
            "gfs_max":      gfs_max_daily,
            "gfs_min":      gfs_min_daily,
            "hrrr_max":     hrrr_max_daily,
            "hrrr_min":     hrrr_min_daily,
            "nam_max":      nam_max_daily,
            "nam_min":      nam_min_daily,
            "nbm_max":      nbm_max_daily,
            "nbm_min":      nbm_min_daily,
            "ensemble_max": ens_max_daily,
            "ensemble_std": ens_max_std_daily,
            "ensemble_min": ens_min_daily,
        }

        # Backward-compat single-value fields (day-0 = today)
        res_data["ecmwf_mean_c"]    = ecmwf_max_daily[0]
        res_data["gfs_mean_c"]      = gfs_max_daily[0]
        res_data["hrrr_mean_c"]     = hrrr_max_daily[0]
        res_data["nam_mean_c"]      = nam_max_daily[0]
        res_data["nbm_mean_c"]      = nbm_max_daily[0]
        res_data["ensemble_mean_c"] = ens_max_daily[0]
        res_data["ensemble_std_c"]  = ens_max_std_daily[0]
        res_data["forecast_hourly"] = {
            "temperature_2m": hourly_temps,
            "hrrr_temperature_2m": hrrr_hourly_temps,
            "hrrr_dew_point_2m": hrrr_hourly_dew_point,
            "hrrr_cloud_cover": hrrr_hourly_cloud_cover,
            "hrrr_wind_speed_10m": hrrr_hourly_wind_speed,
            "hrrr_wind_direction_10m": hrrr_hourly_wind_dir,
            "nam_temperature_2m": nam_hourly_temps,
            "nam_dew_point_2m": nam_hourly_dew_point,
            "nam_cloud_cover": nam_hourly_cloud_cover,
            "nam_wind_speed_10m": nam_hourly_wind_speed,
            "nam_wind_direction_10m": nam_hourly_wind_dir,
            "nbm_temperature_2m": nbm_hourly_temps,
            "cloud_cover": hourly_cloud_cover,
            "wind_speed_10m": hourly_wind_speed,
        }
        res_data["current_model"] = current_snapshot

        res_data["_fetched_at"] = time_module.time()
        self.open_meteo_cache[icao] = res_data
        return res_data


    def get_weather_for_icao(self, icao: str) -> Dict[str, Any]:
        """
        Returns cached data if valid, otherwise returns empty struct.
        """
        data = self.cache.get(icao)
        if data and time.time() - data.get("last_updated", 0) < self.cache_ttl:
            return data
        
        logger.warning(f"No valid fresh cache for {icao}. Fallback needed.")
        return {"metar": [], "taf": [], "last_updated": 0}

weather_fetcher = WeatherFetcher()
