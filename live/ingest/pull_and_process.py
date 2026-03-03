import time
import httpx
import os

from sqlalchemy import text
from common.db import SessionLocal
from .watermark import get_watermark, set_watermark
from common.logger import get_logger
from common.scoring import score_articles_regression
from alpaca_trade_api import REST

log = get_logger("ingest")

APCA_API_BASE_URL = os.getenv("APCA_API_BASE_URL", "https://paper-api.alpaca.markets/v2")

def get_alpaca_client() -> REST:
    return REST(
        key_id=os.getenv("APCA_API_KEY_ID"),
        secret_key=os.getenv("APCA_API_SECRET_KEY"),
        base_url=APCA_API_BASE_URL,
    )

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
        
        articles = add_articles_to_db(articles)
        scores = score_articles_regression(articles)
        add_scores_to_db(scores)
        log.info(f"Processed {len(articles)} articles from Massive API.")
        with SessionLocal() as db, db.begin():
            set_watermark(db, "last_processed_timestamp", time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))
    except httpx.TimeoutException:
        log.error("Massive API timed out.")
    except httpx.HTTPStatusError as e:
        log.error(f"Massive returned {e.response.status_code}: {e.response.text}")
    except httpx.RequestError as e:
        log.error(f"Network error: {e}")
        
  with httpx.Client() as client:
    try:
        response = client.get(
            "https://api.massive.com/v2/last/trade/SPY",
            headers=headers,
        )
        response.raise_for_status()
        data = response.json()

        price = data["results"]["p"]  
        update_benchmark_prices_in_db("SPY", price)
        log.info(f"Updated SPY benchmark price: {price}")
    except httpx.RequestError as e:
        log.error(f"Network error while fetching equity: {e}")
    except Exception as e:
        log.error(f"Error while fetching/updating equity: {e}")
  
  # ---- Alpaca connection and sizing ----
  api = get_alpaca_client()
  log.info("Connected to Alpaca (paper).")

  # Account equity
  account = api.get_account()
  equity = float(account.equity) 
  update_equity_in_db("SPY", equity)
  log.info(f"Updated account equity: {equity}")
  
  
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
                              "provider": "Massive",
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
          
def update_equity_in_db(ticker, equity):
    with SessionLocal() as db, db.begin():
        db.execute(text("""
                        INSERT INTO equity_history(ticker, equity, as_of_date)
                        VALUES (:ticker, :equity, :as_of_date)
                        """ ),{
                            "ticker": ticker,
                            "equity": equity,
                            "as_of_date": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
                        })
        
def update_benchmark_prices_in_db(ticker, price):
    with SessionLocal() as db, db.begin():
        db.execute(text("""
                        INSERT INTO benchmark_prices(ticker, price, as_of_date)
                        VALUES (:ticker, :price, :as_of_date)
                        """ ),{
                            "ticker": ticker,
                            "price": price,
                            "as_of_date": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
                        })
         

if __name__ == "__main__":
    log.info("Starting pull_and_process_data()")
    pull_and_process_data()
    log.info("Finished pull_and_process_data()")
