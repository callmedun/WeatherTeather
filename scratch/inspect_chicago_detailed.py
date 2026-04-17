import asyncio
import sys
import os
import httpx

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.weather_data import ICAO_COORDS

async def inspect_chicago_detailed():
    icao = "KORD"
    coords = ICAO_COORDS.get(icao)
    lat, lon = coords
    
    print(f"--- Fetching 7-day forecast for {icao} (Chicago) ---")
    url = f"https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}&daily=temperature_2m_max&timezone=auto"
    
    async with httpx.AsyncClient() as client:
        r = await client.get(url, timeout=10.0)
        if r.status_code == 200:
            data = r.json()
            days = data.get("daily", {}).get("time", [])
            temps = data.get("daily", {}).get("temperature_2m_max", [])
            
            for d, t in zip(days, temps):
                f_temp = (t * 9/5) + 32
                print(f"  Date: {d} | Max: {t}°C ({f_temp:.1f}°F)")
        else:
            print(f"Error: {r.status_code}")

if __name__ == "__main__":
    asyncio.run(inspect_chicago_detailed())
