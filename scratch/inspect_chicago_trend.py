import asyncio
import sys
import os
import httpx

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

async def get_kord_detailed_weather():
    icao = "KORD"
    # Get last 48 hours of METARs to see the heatwave trend
    url_metar = f"https://aviationweather.gov/api/data/metar?ids={icao}&format=json&hours=48"
    
    async with httpx.AsyncClient() as client:
        r = await client.get(url_metar, timeout=10.0)
        if r.status_code == 200:
            data = r.json()
            print(f"--- METAR History for {icao} ---")
            for m in data[:10]: # Just first 10
                time = m.get('reportTime')
                temp = m.get('temp')
                print(f"  {time} | Temp: {temp}°C ({(temp*9/5)+32:.1f}°F)")
        
        # Get TAF
        url_taf = f"https://aviationweather.gov/api/data/taf?ids={icao}&format=json"
        r = await client.get(url_taf, timeout=10.0)
        if r.status_code == 200:
            data = r.json()
            print("\n--- Current TAF ---")
            for t in data:
                print(f"  {t.get('rawTaf')}")

if __name__ == "__main__":
    asyncio.run(get_kord_detailed_weather())
