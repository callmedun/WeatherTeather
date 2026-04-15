import httpx
import json

def find_cities():
    url = "https://gamma-api.polymarket.com/events?limit=100&active=true&closed=false&tag_slug=weather"
    response = httpx.get(url)
    data = response.json()
    
    cities = set()
    for event in data:
        title = event.get("title", "")
        # Titile example: "Highest temperature in London on April 15"
        # We can try to extract the city after 'in ' or 'for '
        print(f"Found event: {title}")
        
    # Standard list of cities often used:
    # NYC (KJFK), Chicago (KORD), Dallas (KDFW), Los Angeles (KLAX), Atlanta (KATL),
    # Seattle (KSEA), Denver (KDEN), Houston (KIAH), Miami (KMIA), Phoenix (KPHX),
    # Minneapolis (KMSP), Philadelphia (KPHL), DC (KIAD), SF (KSFO), Boston (KBOS),
    # London (EGLL), Paris (LFPG), Tokyo (RJTT), Seoul (RKSS), Shanghai (ZSSS),
    # Hong Kong (VHHH), Singapore (WSSS)

if __name__ == "__main__":
    find_cities()
