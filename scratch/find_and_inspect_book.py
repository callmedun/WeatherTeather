import httpx
import json
import os
from py_clob_client.client import ClobClient
from dotenv import load_dotenv

load_dotenv()

def find_token_and_inspect():
    # 1. Get an active weather market token
    url = "https://gamma-api.polymarket.com/events?limit=5&active=true&closed=false&tag_slug=weather"
    r = httpx.get(url)
    data = r.json()
    
    token_id = None
    for event in data:
        for market in event.get("markets", []):
            tokens = market.get("clobTokenIds")
            if tokens:
                if isinstance(tokens, str): tokens = json.loads(tokens)
                token_id = tokens[0] # Use 'Yes' token
                break
        if token_id: break
        
    if not token_id:
        print("No active weather tokens found.")
        return

    print(f"Inspecting active token: {token_id}")
    
    # 2. Inspect Book
    client = ClobClient(
        host="https://clob.polymarket.com",
        key=os.getenv("POLYMARKET_PRIVATE_KEY", "0x" + "0"*64),
        chain_id=137
    )
    
    try:
        book = client.get_order_book(token_id)
        print(f"Book type: {type(book)}")
        # Check if it has dict-like access or attr access
        if hasattr(book, "asks"):
            print(f"Asks: {book.asks[:3]}")
            if book.asks:
                first = book.asks[0]
                # Check if ask is a dict or object
                print(f"First ask dict/attr test: {first}")
        else:
            print(f"Raw book: {book}")
            
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    find_token_and_inspect()
