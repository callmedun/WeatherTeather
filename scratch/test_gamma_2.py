import asyncio
import httpx
import json

async def test_gamma_market():
    async with httpx.AsyncClient() as client:
        r = await client.get("https://gamma-api.polymarket.com/markets?condition_id=0xe67f4d31f0eba1b009b49d7a7739b2a5f53a2d7d979a0902b465a510492c33bb")
        data = r.json()
        print(json.dumps(data[0].get("events", []), indent=2))

if __name__ == "__main__":
    asyncio.run(test_gamma_market())
