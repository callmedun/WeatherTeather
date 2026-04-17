import asyncio
import httpx

async def test_gamma_event():
    # Use any known condition ID. We have in logs: 
    # market_id or condition_id. Let's find one.
    # We can just fetch the first weather event.
    async with httpx.AsyncClient() as client:
        r = await client.get("https://gamma-api.polymarket.com/events?limit=1&tag_slug=weather&active=true&closed=false")
        data = r.json()
        event = data[0]
        event_id = event["id"]
        title = event["title"]
        print(f"Event: {title} (ID: {event_id})")
        
        markets = event.get("markets", [])
        print(f"Markets count: {len(markets)}")
        for m in markets:
            print(f"  - Condition: {m.get('conditionId')}, Question: {m.get('question')}")

if __name__ == "__main__":
    asyncio.run(test_gamma_event())
