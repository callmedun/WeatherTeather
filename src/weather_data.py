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
        self.cache_ttl = 3600  # 1 hour cache

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

        # Update cache for successful results
        for icao, data in results.items():
            if data["metar"] or data["taf"]:
                self.cache[icao] = data
                
        return results

    async def fetch_open_meteo(self, icao: str) -> Dict[str, str]:
        """
        Fetches latitude/longitude mapping from ICAO and hits Open-Meteo models
        Specifically formats for the AI Multi-Source analysis prompt.
        """
        res_data = {"ecmwf_summary": "", "gfs_hrrr_summary": "", "ensemble_summary": ""}
        coords = ICAO_COORDS.get(icao)
        if not coords:
            return {"ecmwf_summary": "N/A", "gfs_hrrr_summary": "N/A", "ensemble_summary": "N/A"}
            
        lat, lon = coords
        
        # 1. ECMWF IFS
        try:
            url_ecmwf = f"https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}&daily=temperature_2m_max&models=ecmwf_ifs04&timezone=auto"
            async with httpx.AsyncClient() as client:
                r = await client.get(url_ecmwf, timeout=10.0)
                if r.status_code == 200:
                    data = r.json()
                    temps = data.get("daily", {}).get("temperature_2m_max", [])
                    if temps:
                        res_data["ecmwf_summary"] = f"Max temp forecast: {temps[0]}°C, trend stable / ECMWF IFS 0.4"
                        res_data["ecmwf_mean_c"] = float(temps[0])  # Numerical field for math model
                    else:
                        res_data["ecmwf_summary"] = "Data unavailable"
        except Exception as e:
            res_data["ecmwf_summary"] = f"Error: {e}"

        # 2. GFS + HRRR (Seamless)
        try:
            url_gfs = f"https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}&daily=temperature_2m_max&models=gfs_seamless&timezone=auto"
            async with httpx.AsyncClient() as client:
                r = await client.get(url_gfs, timeout=10.0)
                if r.status_code == 200:
                    data = r.json()
                    temps = data.get("daily", {}).get("temperature_2m_max", [])
                    if temps:
                        res_data["gfs_hrrr_summary"] = f"Max temp forecast: {temps[0]}°C / GFS Seamless"
                        res_data["gfs_mean_c"] = float(temps[0])  # Numerical field for math model
                    else:
                        res_data["gfs_hrrr_summary"] = "Data unavailable"
        except Exception as e:
            res_data["gfs_hrrr_summary"] = f"Error: {e}"

        # 3. Ensemble Consensus (51+ members) via ecmwf_ensemble
        try:
            url_ens = f"https://ensemble-api.open-meteo.com/v1/ensemble?latitude={lat}&longitude={lon}&daily=temperature_2m_max&models=ecmwf_ensemble&timezone=auto"
            async with httpx.AsyncClient() as client:
                r = await client.get(url_ens, timeout=10.0)
                if r.status_code == 200:
                    data = r.json()
                    members = []
                    # Filter all keys that are temperature_2m_max arrays
                    for key in data.get("daily", {}):
                        if "temperature_2m_max" in key:
                            val = data["daily"][key]
                            if val and val[0] is not None:
                                members.append(val[0])
                    
                    if members:
                        import statistics as stats_module
                        avg = round(sum(members) / len(members), 1)
                        std = round(stats_module.stdev(members), 2) if len(members) > 1 else 2.0
                        res_data["ensemble_summary"] = f"Average ensemble max: {avg}°C across {len(members)} members."
                        # Numerical fields for math model
                        res_data["ensemble_mean_c"] = avg
                        res_data["ensemble_std_c"] = std
        except Exception as e:
            res_data["ensemble_summary"] = f"Error: {e}"

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
