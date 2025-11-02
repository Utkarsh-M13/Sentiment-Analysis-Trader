import time
import httpx
import os

from sqlalchemy import text
from common.db import SessionLocal
from .watermark import get_watermark, set_watermark
from common.logger import get_logger
log = get_logger("ingest")

def pull_and_process_data():
  POLYGON_API_KEY = os.getenv("POLYGON_API_KEY")
  headers = {"Authorization": f"Bearer {POLYGON_API_KEY}"}
  since_timestamp = None
  with SessionLocal() as db, db.begin():
    since_timestamp = get_watermark(db, "last_processed_timestamp", default="2025-10-01T00:00:00Z")
  with httpx.Client() as client:
    try:
        response = client.get(f"https://api.massive.com/v2/reference/news?ticker=SPY&published_utc.gte={since_timestamp}&order=asc&limit=10&sort=published_utc&apiKey={POLYGON_API_KEY}", headers=headers)
        data = response.json()
        articles = data.get("results", [])
        with SessionLocal() as db, db.begin():
            set_watermark(db, "last_processed_timestamp", time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))
        add_articles_to_db(articles)
        
        log.info(f"Processed {len(articles)} articles from Polygon API.")
    except httpx.TimeoutException:
        log.error("Polygon API timed out.")
    except httpx.HTTPStatusError as e:
        log.error(f"Polygon returned {e.response.status_code}: {e.response.text}")
    except httpx.RequestError as e:
        log.error(f"Network error: {e}")
  
  
def add_articles_to_db(articles):
    with SessionLocal() as db, db.begin():
        for article in articles:
          db.execute(text("""
                          INSERT INTO news_headlines(provider, provider_id, tickers, published_utc, url, title, description, source_name)
                          VALUES (:provider, :provider_id, :tickers, :published_utc, :url, :title, :description, :source_name)
                          ON CONFLICT (provider, provider_id) DO NOTHING
                          """ ),{
                              "provider": "Polygon",
                              "provider_id": article["id"],
                              "tickers": article["tickers"],
                              "published_utc": article["published_utc"],
                              "url": article["article_url"],
                              "title": article["title"],
                              "description": article["description"],
                              "source_name": article["publisher"]["name"]
                          })               
