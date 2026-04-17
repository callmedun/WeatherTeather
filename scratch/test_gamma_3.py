import asyncio
import httpx
import json

async def test_gamma_market_events():
    async with httpx.AsyncClient() as client:
        r = await client.get("https://gamma-api.polymarket.com/events/23784")
        data = r.json()
        print(f"Event: {data.get('title')}")
        markets = data.get("markets", [])
        print(f"Found {len(markets)} markets in this event.")
        for m in markets[:2]:
            print(f"Market: {m.get('question')} | Condition: {m.get('conditionId')}")

if __name__ == "__main__":
    asyncio.run(test_gamma_market_events())
