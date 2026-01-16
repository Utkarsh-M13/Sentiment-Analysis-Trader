import time
import httpx
import os

from sqlalchemy import text
from common.db import SessionLocal
from .watermark import get_watermark, set_watermark
from common.logger import get_logger
from common.scoring import score_articles_regression
log = get_logger("ingest")

def pull_and_process_data():
  MASSIVE_API_KEY = os.getenv("MASSIVE_API_KEY")
  headers = {"Authorization": f"Bearer {MASSIVE_API_KEY}"}
  since_timestamp = None
  with SessionLocal() as db, db.begin():
    since_timestamp = get_watermark(db, "last_processed_timestamp", default="2025-10-01T00:00:00Z")

  print(f"Pulling data since {since_timestamp}")

  with httpx.Client() as client:
    try:
        params = {
            "ticker": "SPY",
            "published_utc.gte": since_timestamp,
            "order": "asc",
            "limit": 100,
            "sort": "published_utc",
        }

        response = client.get(
            "https://api.massive.com/v2/reference/news",
            headers=headers,
            params=params,
        )
        print(f"Response status code: {response}")
        data = response.json()
        print(data)
        articles = data.get("results", [])
        with SessionLocal() as db, db.begin():
            set_watermark(db, "last_processed_timestamp", time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))
        articles = add_articles_to_db(articles)
        scores = score_articles_regression(articles)
        add_scores_to_db(scores)
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
            res = db.execute(text("""
                          INSERT INTO news_headlines(provider, provider_id, tickers, published_utc, url, title, description, source_name)
                          VALUES (:provider, :provider_id, :tickers, :published_utc, :url, :title, :description, :source_name)
                          ON CONFLICT (provider, provider_id) 
                          DO UPDATE SET title = news_headlines.title
                          RETURNING id;
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
            row = res.fetchone()
            article["provider_id"] = article["id"]
            article["id"] = row[0] if row else None
    return articles

def add_scores_to_db(scores):
    with SessionLocal() as db, db.begin():
        for article in scores:
          db.execute(text("""
                          INSERT INTO headline_scores(headline_id, provider_id, model_name, model_version, score, p_up)
                          VALUES (:headline_id, :provider_id, :model_name, :model_version, :score, :p_up)
                          ON CONFLICT (headline_id, model_name, model_version) DO NOTHING
                          """ ),{
                              "headline_id": article["id"],
                              "provider_id": article["provider_id"],
                              "model_name": "finbert-regression-v1",
                              "model_version": "2025-10-30",
                              "score": article["score_raw"],
                              "p_up": None,
                          })

if __name__ == "__main__":
    log.info("Starting pull_and_process_data()")
    pull_and_process_data()
    log.info("Finished pull_and_process_data()")
