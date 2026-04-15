import os
from py_clob_client.client import ClobClient
from dotenv import load_dotenv

load_dotenv()

def inspect_book():
    host = "https://clob.polymarket.com"
    # We can use a public token or just any active one
    # Let's try to get a market first or use a known one from the bot's logs if available
    # Actually, the client might need credentials to even fetch a book? 
    # Usually public endpoints don't need full auth but the SDK might require it.
    
    client = ClobClient(
        host=host,
        key=os.getenv("POLYMARKET_PRIVATE_KEY", "0x" + "0"*64), # Dummy key if not set
        chain_id=137
    )
    
    # Example token (London April 16 [17C] No)
    token_id = "11252119102450375936306560945934149021484964645224376510301138240032943715873" # This is a dummy example
    
    try:
        # Note: In our code we use clob_client.get_order_book(token_id)
        # Let's see what it returns.
        print(f"Fetching book for {token_id}...")
        book = client.get_order_book(token_id)
        
        print("\n--- BOOK STRUCTURE ---")
        print(type(book))
        # It's usually an object with .bids and .asks attributes
        if hasattr(book, "asks"):
            print(f"Asks count: {len(book.asks)}")
            if book.asks:
                print(f"First ask: {book.asks[0]}")
                print(f"Type of ask item: {type(book.asks[0])}")
        else:
            print("No .asks attribute found. Printing raw:")
            print(book)
            
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    inspect_book()
