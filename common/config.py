from dotenv import load_dotenv
import os

load_dotenv()

POLYGON_API_KEY = os.getenv("POLYGON_API_KEY")
TZ              = os.getenv("TZ", "America/New_York")
POSTGRES_PASSWORD = os.getenv("POSTGRES_PASSWORD")
POSTGRES_URL      = os.getenv("POSTGRES_URL")
