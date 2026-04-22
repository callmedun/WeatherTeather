import httpx
import time
from typing import Dict, Any, List, Optional
from src.utils import logger

# Hardcoded coordinates mapping for precise Open-Meteo queries based on config standard
ICAO_COORDS = {
    "ZSPD": (31.1443, 121.8083), # Shanghai
    "RKSI": (37.4602, 126.4407), # Seoul
    "EGLL": (51.4700, -0.4543),  # London
    "KMIA": (25.7959, -80.2870), # Miami
    "KDFW": (32.8998, -97.0403), # Dallas
    "KATL": (33.6407, -84.4277), # Atlanta
    "LEMD": (40.4839, -3.5680),  # Madrid
    "WSSS": (1.3644, 103.9915),  # Singapore
    "MMMX": (19.4361, -99.0719), # Mexico City
    "KJFK": (40.6413, -73.7781), # New York
    "EDDB": (52.3667, 13.5033),  # Berlin
    "LFPG": (49.0097, 2.5479),   # Paris
    "RJTT": (35.5494, 139.7798), # Tokyo
    "KORD": (41.9742, -87.9073), # Chicago
    "KLAX": (33.9416, -118.4085),# Los Angeles
    "CYYZ": (43.6777, -79.6248), # Toronto
    "YSSY": (-33.9399, 151.1753),# Sydney
    "OMDB": (25.2532, 55.3657),  # Dubai
    "VABB": (19.0896, 72.8656),  # Mumbai
    "SBGR": (-23.4356, -46.4731),# Sao Paulo
    "KHOU": (29.6454, -95.2789), # Houston
    "VHHH": (22.3080, 113.9185), # Hong Kong
    "EGLC": (51.5048,  0.0495),  # London City
    "KDAL": (32.8473, -96.8517), # Dallas Love Field
}

class WeatherFetcher:
    def __init__(self):
        self.base_url_metar = "https://aviationweather.gov/api/data/metar"
        self.base_url_taf = "https://aviationweather.gov/api/data/taf"
        self.cache: Dict[str, Dict[str, Any]] = {}
        self.cache_ttl = 1200  # 20 minutes — METAR updates every 20-30 min
        self.open_meteo_cache: Dict[str, Any] = {}
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

    async def fetch_open_meteo(self, icao: str) -> Dict[str, Any]:
        """
        Fetches ECMWF, GFS and Ensemble forecasts from Open-Meteo.
        Results are cached for 3 hours (models update every 6h).
        Returns a dict with both summary strings and numerical fields.
        """
        import time as time_module
        cached = self.open_meteo_cache.get(icao)
        if cached and time_module.time() - cached.get("_fetched_at", 0) < self.open_meteo_cache_ttl:
            return cached
        res_data: Dict[str, Any] = {}
        coords = ICAO_COORDS.get(icao)
        if not coords:
            return {}

        lat, lon = coords
        FORECAST_DAYS = 4  # cover today + 3 future days for markets resolving up to ~4 days out

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
        res_data["forecast_daily"] = {
            "ecmwf_max":    ecmwf_max_daily,
            "ecmwf_min":    ecmwf_min_daily,
            "gfs_max":      gfs_max_daily,
            "gfs_min":      gfs_min_daily,
            "ensemble_max": ens_max_daily,
            "ensemble_std": ens_max_std_daily,
            "ensemble_min": ens_min_daily,
        }

        # Backward-compat single-value fields (day-0 = today)
        res_data["ecmwf_mean_c"]    = ecmwf_max_daily[0]
        res_data["gfs_mean_c"]      = gfs_max_daily[0]
        res_data["ensemble_mean_c"] = ens_max_daily[0]
        res_data["ensemble_std_c"]  = ens_max_std_daily[0]

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
