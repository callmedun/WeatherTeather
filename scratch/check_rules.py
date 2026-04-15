import httpx
import json

def fetch_market_rules():
    url = "https://gamma-api.polymarket.com/events?limit=50&active=true&closed=false&tag_slug=weather"
    response = httpx.get(url)
    data = response.json()
    
    for event in data:
        print(f"--- Event: {event['title']} ---")
        for market in event.get("markets", []):
            desc = market.get("description", "")
            print(f"Market ID: {market.get('conditionId')}")
            print(f"Description snippet: {desc[:200]}...")
            # Search for 4-letter uppercase codes (ICAO)
            import re
            icaos = re.findall(r'\b[K|E|Z|R|V|W|L][A-Z]{3}\b', desc)
            if icaos:
                print(f"Potential ICAOs in rules: {icaos}")
        print("\n")

if __name__ == "__main__":
    fetch_market_rules()
