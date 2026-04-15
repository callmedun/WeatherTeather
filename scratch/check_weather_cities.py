import httpx
import json
import sys

# Ensure UTF-8 output even on Windows
sys.stdout.reconfigure(encoding='utf-8')

def fetch_weather_events():
    url = "https://gamma-api.polymarket.com/events?limit=50&active=true&closed=false&tag_slug=weather"
    response = httpx.get(url)
    data = response.json()
    
    found_cities = {}
    
    for event in data:
        title = event.get("title", "")
        print(f"Event: {title}")
        
        # Look for "High temperature in [City]"
        import re
        city_match = re.search(r'in (.*) on', title)
        if not city_match:
             city_match = re.search(r'temperature in (.*)', title)
             
        if city_match:
            city = city_match.group(1).strip()
            # Clean city name (e.g. remove "the specified amount of")
            if "case" in city.lower() or "storm" in city.lower():
                continue
                
            # Check markets for ICAO
            for market in event.get("markets", []):
                desc = market.get("description", "")
                icaos = re.findall(r'\b([K|E|Z|R|V|W|L][A-Z]{3})\b', desc)
                if icaos:
                    found_cities[city] = icaos[0]
                    break
    
    print("\n--- DETECTED CITIES & ICAOs ---")
    for c, i in found_cities.items():
        print(f"{c}: {i}")

if __name__ == "__main__":
    fetch_weather_events()
