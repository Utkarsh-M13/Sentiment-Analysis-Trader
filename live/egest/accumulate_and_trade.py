import json
from datetime import datetime, timezone, timedelta
from pathlib import Path
import os

import numpy as np
import pandas as pd
from sqlalchemy import text

from common.db import SessionLocal
from common.logger import get_logger
from live.ingest.watermark import get_watermark, set_watermark

from alpaca.trading.client import TradingClient
from alpaca.trading.requests import MarketOrderRequest
from alpaca.trading.enums import OrderSide, TimeInForce

from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockLatestTradeRequest
from alpaca.data.timeframe import TimeFrame
from alpaca.data.requests import StockBarsRequest

log = get_logger("trade")

MIN_NOTIONAL = 100.0
MAX_DELTA_SIG = 0.8

SYMBOL = "SPY"


def get_trading_client() -> TradingClient:
    key = os.environ["APCA_API_KEY_ID"]
    secret = os.environ["APCA_API_SECRET_KEY"]

    base = os.getenv("APCA_API_BASE_URL", "https://paper-api.alpaca.markets").rstrip("/")
    paper = "paper" in base
    return TradingClient(key, secret, paper=paper)


def get_data_client() -> StockHistoricalDataClient:
    key = os.environ["APCA_API_KEY_ID"]
    secret = os.environ["APCA_API_SECRET_KEY"]
    return StockHistoricalDataClient(key, secret)


def safe_parse_iso_date(s: str) -> datetime.date:
    # Supports YYYY-MM-DD and ...Z timestamps
    return datetime.fromisoformat(s.replace("Z", "+00:00")).date()


def fetch_spy_price(data_client: StockHistoricalDataClient) -> float:
    # Prefer latest trade, fallback to last daily close if entitlement fails.
    try:
        resp = data_client.get_stock_latest_trade(
            StockLatestTradeRequest(symbol_or_symbols=SYMBOL)
        )
        trade = resp[SYMBOL]
        price = float(trade.price)
        if price > 0:
            return price
    except Exception as e:
        log.warning(f"Latest trade unavailable, falling back to bars: {e}")

    bars = data_client.get_stock_bars(
        StockBarsRequest(symbol_or_symbols=SYMBOL, timeframe=TimeFrame.Day, limit=1)
    )
    df = bars.df
    # MultiIndex (symbol, timestamp)
    price = float(df["close"].iloc[-1])
    return price


def accumulate_and_trade():
    # get last_traded watermark
    with SessionLocal() as db, db.begin():
        last_traded_raw = get_watermark(db, "last_traded", default="2025-10-01")
        last_traded_date = safe_parse_iso_date(last_traded_raw)

    LOOKBACK_DAYS = 15
    since_timestamp = last_traded_date - timedelta(days=LOOKBACK_DAYS)
    log.info(f"Starting accumulate_and_trade. Since: {since_timestamp}")

    # --- FETCH SCORES ---
    with SessionLocal() as db, db.begin():
        results = db.execute(
            text(
                """
                SELECT 
                  scored_at::date AS date,
                  score
                FROM headline_scores
                WHERE scored_at::date > :since_timestamp
                  AND model_name = :model_name
                ORDER BY date ASC
                """
            ),
            {
                "since_timestamp": since_timestamp,
                "model_name": "finbert-regression-v1",
            },
        ).fetchall()

    if not results:
        log.info("No new scores since last_traded watermark.")
        return

    per_headline_df = pd.DataFrame([{"date": row.date, "score": row.score} for row in results])
    per_headline_df["date"] = pd.to_datetime(per_headline_df["date"])

    per_headline_df["trade_date"] = per_headline_df["date"].dt.date
    daily_df = (
        per_headline_df.groupby("trade_date", as_index=False)
        .agg(score=("score", "mean"))
        .rename(columns={"trade_date": "date"})
    )

    cfg = load_cfg()
    pred_col = cfg["pred_col"]
    if pred_col != "score":
        daily_df[pred_col] = daily_df["score"]

    today_signal, today_pred_use, full_series_df = compute_daily_signal(daily_df, cfg)

    as_of_date = full_series_df["date"].iloc[-1].date()
    num_headlines_today = int((per_headline_df["date"].dt.date == as_of_date).sum())

    log.info(
        f"Signal={today_signal:.4f} | pred_use={today_pred_use:.4f} | headlines={num_headlines_today}"
    )

    # ---- Alpaca connection and sizing ----
    trading_client = get_trading_client()
    data_client = get_data_client()

    account = trading_client.get_account()
    equity = float(account.equity)

    # Current SPY position
    try:
        pos = trading_client.get_open_position(SYMBOL)
        current_shares = int(float(pos.qty))
    except Exception:
        current_shares = 0

    # Market price for SPY
    price = fetch_spy_price(data_client)

    if price <= 0:
        log.info("No valid SPY price; aborting.")
        return

    target_notional = today_signal * equity
    target_shares = int(target_notional // price)

    current_sig = (current_shares * price) / equity if equity > 0 else 0.0
    raw_delta_sig = today_signal - current_sig

    if abs(raw_delta_sig) > MAX_DELTA_SIG:
        capped_sig = current_sig + np.sign(raw_delta_sig) * MAX_DELTA_SIG
        capped_notional = capped_sig * equity
        target_shares = int(capped_notional // price)

    delta_shares = target_shares - current_shares
    trade_notional = abs(delta_shares) * price

    log.info(
        f"equity={equity:.2f} price={price:.2f} current={current_shares} "
        f"target={target_shares} delta={delta_shares}"
    )

    if delta_shares == 0 or trade_notional < MIN_NOTIONAL:
        log.info("No trade (already at target or too small).")
        return

    side_enum = OrderSide.BUY if delta_shares > 0 else OrderSide.SELL
    side_str = "buy" if delta_shares > 0 else "sell"
    qty = abs(delta_shares)

    order_req = MarketOrderRequest(
        symbol=SYMBOL,
        qty=qty,
        side=side_enum,
        time_in_force=TimeInForce.DAY,
    )
    order = trading_client.submit_order(order_req)
    alpaca_order_id = str(order.id)

    # Try to get fill price (may not be immediate)
    fill_price = price
    try:
        filled = trading_client.get_order_by_id(order.id)
        if getattr(filled, "filled_avg_price", None):
            fill_price = float(filled.filled_avg_price)
    except Exception:
        pass

    log.info(f"Placed {side_str.upper()} {qty} {SYMBOL} at ~{fill_price:.2f}, orderId={alpaca_order_id}")

    new_shares = current_shares + (qty if side_str == "buy" else -qty)

    log_trade(
        ticker=SYMBOL,
        side=side_str,
        qty=qty,
        price=fill_price,
        order_id=alpaca_order_id,
        prev_shares=current_shares,
        new_shares=new_shares,
        equity=equity,
        signal=today_signal,
        pred_use=today_pred_use,
        num_headlines=num_headlines_today,
        as_of_date=as_of_date,
        cfg=cfg,
    )

    with SessionLocal() as db, db.begin():
        set_watermark(db, "last_traded", as_of_date.isoformat())

    log.info("Trade logged and watermark updated.")


BASE_DIR = Path(__file__).resolve().parent.parent
CONFIG_DIR = BASE_DIR.parent / "common" / "best" / "best_threshold.json"


def load_cfg(path=CONFIG_DIR):
    with open(path, "r") as f:
        return json.load(f)


def compute_daily_signal(daily_df: pd.DataFrame, cfg: dict):
    pred_col = cfg["pred_col"]
    dir_ = float(cfg["dir"])
    lo = float(cfg["band"]["lo"])
    hi = float(cfg["band"]["hi"])
    span = int(cfg["smooth_span"])

    df = daily_df.copy()
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date")

    if pred_col not in df.columns:
        raise ValueError(f"pred_col '{pred_col}' not found")

    df = df.dropna(subset=[pred_col])

    x = df[pred_col].ewm(span=span, adjust=False).mean()
    x = dir_ * x

    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        lo, hi = np.percentile(x.values, [40, 80])

    sig = ((x - lo) / (hi - lo)).clip(0.0, 1.0)

    df["pred_use"] = x
    df["signal"] = sig

    last = df.iloc[-1]
    return float(last["signal"]), float(last["pred_use"]), df


def log_trade(
    *,
    ticker: str,
    side: str,
    qty: int,
    price: float | None,
    order_id: str | None,
    prev_shares: int,
    new_shares: int,
    equity: float,
    signal: float,
    pred_use: float,
    num_headlines: int,
    as_of_date,
    cfg: dict,
):
    as_of_date = pd.to_datetime(as_of_date).date()

    with SessionLocal() as db, db.begin():
        db.execute(
            text(
                """
                INSERT INTO trade_logs (
                  traded_at,
                  ticker,
                  side,
                  qty,
                  price,
                  order_id,
                  prev_shares,
                  new_shares,
                  equity,
                  signal,
                  pred_use,
                  num_headlines,
                  as_of_date,
                  cfg_json
                )
                VALUES (
                  :traded_at,
                  :ticker,
                  :side,
                  :qty,
                  :price,
                  :order_id,
                  :prev_shares,
                  :new_shares,
                  :equity,
                  :signal,
                  :pred_use,
                  :num_headlines,
                  :as_of_date,
                  :cfg_json
                )
                """
            ),
            {
                "traded_at": datetime.now(timezone.utc),
                "ticker": ticker,
                "side": side,
                "qty": qty,
                "price": price,
                "order_id": order_id,
                "prev_shares": prev_shares,
                "new_shares": new_shares,
                "equity": equity,
                "signal": signal,
                "pred_use": pred_use,
                "num_headlines": num_headlines,
                "as_of_date": as_of_date,
                "cfg_json": json.dumps(cfg),
            },
        )
    log.info("Trade inserted into trade_logs.")


if __name__ == "__main__":
    accumulate_and_trade()
    log.info("Done accumulate_and_trade()")