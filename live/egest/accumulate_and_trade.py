import json
from datetime import datetime, timezone, timedelta
import pandas as pd
import numpy as np
from sqlalchemy import text
from pathlib import Path
from common.db import SessionLocal
from common.logger import get_logger
from live.ingest.watermark import get_watermark, set_watermark
import os
from alpaca_trade_api import REST

log = get_logger("trade")

MIN_NOTIONAL = 100.0    
MAX_DELTA_SIG = 0.8

APCA_API_BASE_URL = os.getenv("APCA_API_BASE_URL", "https://paper-api.alpaca.markets/v2")

def get_alpaca_client() -> REST:
    return REST(
        key_id=os.getenv("APCA_API_KEY_ID"),
        secret_key=os.getenv("APCA_API_SECRET_KEY"),
        base_url=APCA_API_BASE_URL,
    )


def accumulate_and_trade():
    scores = []

    # get last_traded watermark
    with SessionLocal() as db, db.begin():
        last_traded_raw = get_watermark(db, "last_traded", default="2025-10-01")
        last_traded_date = datetime.fromisoformat(last_traded_raw).date()

    LOOKBACK_DAYS = 15
    since_timestamp = last_traded_date - timedelta(days=LOOKBACK_DAYS)
    log.info(f"Starting accumulate_and_trade. Since: {since_timestamp}")

    # --- FETCH SCORES ---
    with SessionLocal() as db, db.begin():
        results = db.execute(
            text("""
            SELECT 
              scored_at::date AS date,
              score
            FROM headline_scores
            WHERE scored_at::date > :since_timestamp
              AND model_name = :model_name
            ORDER BY date ASC
            """),
            {
                "since_timestamp": since_timestamp,
                "model_name": "finbert-regression-v1",
            },
        ).fetchall()

    if not results:
        log.info("No new scores since last_traded watermark.")
        return

    # per-headline df
    per_headline_df = pd.DataFrame(
        [{"date": row.date, "score": row.score} for row in results]
    )
    # keep this as datetime64 so .dt works later
    per_headline_df["date"] = pd.to_datetime(per_headline_df["date"])

    # daily aggregation: create a separate trade_date column for grouping
    per_headline_df["trade_date"] = per_headline_df["date"].dt.date
    daily_df = (
        per_headline_df
        .groupby("trade_date", as_index=False)
        .agg(score=("score", "mean"))
        .rename(columns={"trade_date": "date"})  # now daily_df has ['date', 'score']
    )

    cfg = load_cfg()
    pred_col = cfg["pred_col"]
    if pred_col != "score":
        daily_df[pred_col] = daily_df["score"]

    today_signal, today_pred_use, full_series_df = compute_daily_signal(daily_df, cfg)

    # metadata for today's trade
    as_of_date = full_series_df["date"].iloc[-1].date()
    # per_headline_df["date"] is still datetime64, so .dt is valid here
    num_headlines_today = (per_headline_df["date"].dt.date == as_of_date).sum()

    log.info(
        f"Signal={today_signal:.4f} | pred_use={today_pred_use:.4f} | headlines={num_headlines_today}"
    )

    # ---- Alpaca connection and sizing ----
    api = get_alpaca_client()
    log.info("Connected to Alpaca (paper).")

    # Account equity
    account = api.get_account()
    equity = float(account.equity)

    # Current SPY position
    try:
        pos = api.get_position("SPY")
        current_shares = int(float(pos.qty))
    except Exception:
        # No current position
        current_shares = 0

    # Market price for SPY (latest trade)
    latest_trade = api.get_latest_trade("SPY")
    price = float(latest_trade.price)

    if price <= 0:
        log.info("No valid SPY price; aborting.")
        return

    # Target notional & shares
    target_notional = today_signal * equity
    target_shares = int(target_notional // price)

    current_sig = (current_shares * price) / equity if equity > 0 else 0.0
    raw_delta_sig = today_signal - current_sig

    # Cap allocation change
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

    side = "buy" if delta_shares > 0 else "sell"
    qty = abs(delta_shares)

    # Place market order
    order = api.submit_order(
        symbol="SPY",
        qty=qty,
        side=side,
        type="market",
        time_in_force="day",
    )

    # Alpaca order ID / fills
    alpaca_order_id = order.id

    # Try to get fill price (may not be immediate)
    try:
        filled_order = api.get_order(alpaca_order_id)
        fill_price = float(filled_order.filled_avg_price or price)
    except Exception:
        fill_price = price

    log.info(f"Placed {side.upper()} {qty} SPY at ~{fill_price:.2f}, orderId={alpaca_order_id}")

    new_shares = current_shares + (qty if side == "buy" else -qty)

    # ---- log trade to DB ----
    log_trade(
        ticker="SPY",
        side=side,
        qty=qty,
        price=fill_price,
        order_id=alpaca_order_id,
        prev_shares=current_shares,
        new_shares=new_shares,
        equity=equity,
        signal=today_signal,
        pred_use=today_pred_use,
        num_headlines=int(num_headlines_today),
        as_of_date=as_of_date,
        cfg=cfg,
    )

    # ---- advance last_traded watermark ----
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
    order_id: int | None,
    prev_shares: int,
    new_shares: int,
    equity: float,
    signal: float,
    pred_use: float,
    num_headlines: int,
    as_of_date,
    cfg: dict
):
    as_of_date = pd.to_datetime(as_of_date).date()

    with SessionLocal() as db, db.begin():
        db.execute(
            text("""
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
            """),
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
