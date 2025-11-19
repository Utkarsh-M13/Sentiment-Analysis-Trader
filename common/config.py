from dotenv import load_dotenv
import os

load_dotenv()

POLYGON_API_KEY = os.getenv("POLYGON_API_KEY")
TZ              = os.getenv("TZ", "America/New_York")
POSTGRES_PASSWORD = os.getenv("POSTGRES_PASSWORD")
POSTGRES_URL      = os.getenv("POSTGRES_URL")
APCA_API_KEY_ID = os.getenv("APCA_API_KEY_ID")
APCA_API_SECRET_KEY = os.getenv("APCA_API_SECRET_KEY")
APCA_API_BASE_URL = os.getenv("APCA_API_BASE_URL", "https://paper-api.alpaca.markets/v2")
