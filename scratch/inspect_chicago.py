import asyncio
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.weather_data import weather_fetcher

async def inspect_chicago_weather():
    icao = "KORD"
    print(f"--- Fetching real-time data for {icao} (Chicago) ---")
    
    # 1. Fetch METAR/TAF
    weather_map = await weather_fetcher.fetch_weather_for_icao([icao])
    if not weather_map or icao not in weather_map:
        print("Failed to fetch METAR/TAF")
        return

    data = weather_map[icao]
    
    print("\n[METAR - Last 3 reports]")
    for m in data.get("metar", [])[:3]:
        print(f"  {m.get('reportTime')}: {m.get('rawOb')}")
        
    print("\n[TAF - Current]")
    for t in data.get("taf", []):
        print(f"  {t.get('rawTaf')}")

    # 2. Fetch Open-Meteo
    print("\n--- Fetching Open-Meteo Models ---")
    om = await weather_fetcher.fetch_open_meteo(icao)
    print(f"  ECMWF: {om['ecmwf_summary']}")
    print(f"  GFS/HRRR: {om['gfs_hrrr_summary']}")
    print(f"  Ensemble: {om['ensemble_summary']}")

if __name__ == "__main__":
    asyncio.run(inspect_chicago_weather())
