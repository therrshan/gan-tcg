import os
import requests
import time
import json
import random
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

API_KEY = 'bfeb5824-08e9-4238-97e6-96d1d90e05fb'
PAGE_SIZE = 100
MAX_RETRIES = 3
BASE_URL = 'https://api.pokemontcg.io/v2/cards'
HEADERS = {'X-Api-Key': API_KEY}
TIMEOUT = 30

IMAGE_DIR = "raw_pokemon_data/pokemon_image"
DATA_DIR = "raw_pokemon_data/pokemon_data"
JSON_PATH = os.path.join(DATA_DIR, "pokemon_cards.json")

os.makedirs(IMAGE_DIR, exist_ok=True)
os.makedirs(DATA_DIR, exist_ok=True)

def create_session():
    session = requests.Session()
    retry_strategy = Retry(
        total=3,
        backoff_factor=1,
        status_forcelist=[429, 500, 502, 503, 504],
    )
    adapter = HTTPAdapter(max_retries=retry_strategy)
    session.mount("http://", adapter)
    session.mount("https://", adapter)
    return session

def load_existing_data():
    if os.path.exists(JSON_PATH):
        with open(JSON_PATH, "r") as f:
            data = json.load(f)
        existing_ids = {card["id"] for card in data}
        return data, existing_ids
    return [], set()

def save_data_incrementally(metadata):
    with open(JSON_PATH, "w") as f:
        json.dump(metadata, f, indent=2)

def fetch_page_safe(session, page, existing_ids):
    for attempt in range(MAX_RETRIES):
        try:
            print(f"Fetching page {page}, attempt {attempt + 1}")
            
            params = {
                'page': page,
                'pageSize': PAGE_SIZE,
                'q': 'supertype:Pokémon'
            }
            
            response = session.get(BASE_URL, headers=HEADERS, params=params, timeout=TIMEOUT)
            
            if response.status_code == 429:
                wait_time = 60
                print(f"Rate limited. Waiting {wait_time} seconds...")
                time.sleep(wait_time)
                continue
                
            response.raise_for_status()
            cards = response.json().get("data", [])
            
            new_cards = []
            for card in cards:
                if card.get("id") not in existing_ids:
                    new_cards.append(card)
            
            return new_cards
            
        except requests.exceptions.Timeout:
            wait = min(30, 5 * (2 ** attempt))
            print(f"Timeout on page {page}. Retrying in {wait} seconds...")
            time.sleep(wait)
            
        except requests.exceptions.ConnectionError:
            wait = min(60, 10 * (2 ** attempt))
            print(f"Connection error on page {page}. Retrying in {wait} seconds...")
            time.sleep(wait)
            
        except requests.exceptions.RequestException as e:
            wait = min(30, 5 * (2 ** attempt))
            print(f"Request error on page {page}: {e}. Retrying in {wait} seconds...")
            time.sleep(wait)
    
    print(f"Failed to fetch page {page} after {MAX_RETRIES} attempts")
    return []

def download_image(session, card_id, image_url):
    image_path = os.path.join(IMAGE_DIR, f"{card_id}.jpg")
    
    if os.path.exists(image_path):
        return image_path
    
    for attempt in range(2):
        try:
            response = session.get(image_url, timeout=20)
            if response.status_code == 200:
                with open(image_path, "wb") as f:
                    f.write(response.content)
                return image_path
            else:
                print(f"Failed to download image for {card_id}: HTTP {response.status_code}")
                return None
                
        except Exception as e:
            if attempt == 0:
                time.sleep(2)
                continue
            print(f"Error downloading image for {card_id}: {e}")
            return None
    
    return None

def main():
    session = create_session()
    metadata, existing_ids = load_existing_data()
    
    start_page = (len(metadata) // PAGE_SIZE) + 1 if metadata else 1
    page = start_page
    consecutive_empty_pages = 0
    processed_this_session = 0
    
    print(f"Starting from page {start_page}. Already have {len(metadata)} cards.")
    
    while consecutive_empty_pages < 3:
        cards = fetch_page_safe(session, page, existing_ids)
        
        if not cards:
            consecutive_empty_pages += 1
            print(f"Empty page {page}. Consecutive empty: {consecutive_empty_pages}")
            page += 1
            continue
        
        consecutive_empty_pages = 0
        page_processed = 0
        
        for card in cards:
            card_id = card.get("id")
            name = card.get("name", "Unknown")
            hp = card.get("hp", "N/A")
            types = card.get("types", [])
            image_url = card.get("images", {}).get("large")
            
            if not image_url:
                continue
            
            image_path = download_image(session, card_id, image_url)
            if not image_path:
                continue
            
            metadata.append({
                "id": card_id,
                "name": name,
                "hp": hp,
                "types": types,
                "image_path": image_path,
                "rarity": card.get("rarity", "Unknown"),
                "set": card.get("set", {}).get("name", "Unknown"),
                "attacks": card.get("attacks", []),
                "abilities": card.get("abilities", [])
            })
            
            existing_ids.add(card_id)
            page_processed += 1
            processed_this_session += 1
            
            if processed_this_session % 20 == 0:
                save_data_incrementally(metadata)
                print(f"Saved progress: {len(metadata)} total cards")
        
        print(f"Page {page}: processed {page_processed} new cards")
        
        page += 1
        time.sleep(random.uniform(1, 3))
        
        if processed_this_session >= 500:
            print(f"Processed {processed_this_session} cards this session. Taking a break.")
            break
    
    save_data_incrementally(metadata)
    print(f"Final save: {len(metadata)} total Pokemon cards")
    print(f"Images saved in: {IMAGE_DIR}/")
    print(f"Metadata saved in: {JSON_PATH}")

if __name__ == "__main__":
    main()