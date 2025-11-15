import json
import time
import pandas as pd
import numpy as np

from common import db
from live.ingest.watermark import get_watermark, set_watermark # you may also want set_watermark
from ib_insync import *

MIN_NOTIONAL = 100.0    
MAX_DELTA_SIG = 0.8


def accumulate_and_trade():
    # 1) Load new scores since last trade
    since_timestamp = get_watermark(db, "last_traded", default="2025-10-01T00:00:00Z")

    with db.SessionLocal() as session:
        results = session.execute(
            """
            SELECT date, score
            FROM headline_scores
            WHERE date > :since_timestamp
            ORDER BY date ASC
            """,
            {"since_timestamp": since_timestamp}
        ).fetchall()

    if not results:
        print("No new scores since last_traded watermark.")
        return

    scores_df = pd.DataFrame(
        [{"date": row.date, "score": row.score} for row in results]
    )

    # 2) Compute signal in line with backtest
    cfg = load_cfg()
    pred_col = cfg["pred_col"]
    if pred_col != "score":
        scores_df[pred_col] = scores_df["score"]

    today_signal, today_pred_use, full_series_df = compute_daily_signal(scores_df, cfg)

    print(f"today_signal={today_signal:.4f}, pred_use={today_pred_use:.4f}")

    # 3) Connect to IBKR
    ib = IB()
    ib.connect('127.0.0.1', 4002, clientId=1)

    contract = Stock('SPY', 'SMART', 'USD')

    # 4) Get portfolio equity (NetLiquidation)
    acct_summary = ib.accountSummary()
    net_liq = next((a for a in acct_summary if a.tag == 'NetLiquidation'), None)
    if net_liq is None:
        print("Could not find NetLiquidation in accountSummary; aborting.")
        return

    equity = float(net_liq.value)
    print("Portfolio equity:", equity)

    # 5) Get current SPY position
    positions = ib.positions()
    current_shares = 0
    for p in positions:
        if isinstance(p.contract, Contract) and p.contract.symbol == "SPY" and p.contract.secType == "STK":
            current_shares = int(p.position)
            break
    print("Current SPY shares:", current_shares)

    # 6) Get a price to size the trade
    # snapshot market data; requires market data permissions
    ticker = ib.reqMktData(contract, '', snapshot=True)
    ib.sleep(2)  # give IB a moment to populate
    price = ticker.last or ticker.close or ticker.marketPrice()
    if price is None or price <= 0:
        print("Could not get a valid SPY price; aborting.")
        return
    print("SPY price:", price)

    # 7) Convert signal into target shares (0..1 of equity in SPY)
    target_notional = today_signal * equity
    target_shares = int(target_notional // price)
    delta_shares = target_shares - current_shares
    trade_notional = abs(delta_shares) * price

    print(f"Target shares={target_shares}, delta={delta_shares}, trade_notional={trade_notional:.2f}")

    # Safety rails
    # a) Do nothing if target change is tiny
    if delta_shares == 0 or trade_notional < MIN_NOTIONAL:
        print("No trade: either already at target or trade too small.")
        return

    # b) Limit daily change in position as fraction of equity
    current_sig = (current_shares * price) / equity if equity > 0 else 0.0
    raw_delta_sig = today_signal - current_sig
    if abs(raw_delta_sig) > MAX_DELTA_SIG:
        capped_sig = current_sig + np.sign(raw_delta_sig) * MAX_DELTA_SIG
        capped_notional = capped_sig * equity
        capped_shares = int(capped_notional // price)
        delta_shares = capped_shares - current_shares
        trade_notional = abs(delta_shares) * price
        print(f"Capping delta_sig from {raw_delta_sig:.3f} to {np.sign(raw_delta_sig) * MAX_DELTA_SIG:.3f}")
        print(f"New delta_shares={delta_shares}, trade_notional={trade_notional:.2f}")
        if delta_shares == 0 or trade_notional < MIN_NOTIONAL:
            print("No trade after capping.")
            return

    side = "BUY" if delta_shares > 0 else "SELL"
    qty = abs(delta_shares)

    # 8) Place order to move towards target
    order = MarketOrder(side, qty)
    trade = ib.placeOrder(contract, order)
    print(f"Placed {side} {qty} SPY")

    ib.sleep(2)
    print("Order status:", trade.orderStatus.status)
    
    # 9) Update watermark
    set_watermark(db, "last_processed_timestamp", time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))
    

def load_cfg(path="experiments/best_threshold.json"):
    with open(path, "r") as f:
        cfg = json.load(f)
    return cfg


def compute_daily_signal(daily_df: pd.DataFrame, cfg: dict):
    """
    daily_df: DataFrame with columns ["date", <pred_col>]
    cfg: JSON from best_threshold.json
    Returns: (today_signal, today_pred_use, full_series_df)
    """
    pred_col = cfg["pred_col"]           # e.g. "score", "score_std"
    dir_ = float(cfg["dir"])             # +1 or -1
    lo = float(cfg["band"]["lo"])
    hi = float(cfg["band"]["hi"])
    span = int(cfg["smooth_span"])

    df = daily_df.copy()
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date")

    # ensure the column exists
    if pred_col not in df.columns:
        raise ValueError(f"pred_col '{pred_col}' not found in daily_df columns {df.columns.tolist()}")

    df = df.dropna(subset=[pred_col])

    # EMA smoothing (same as backtest)
    x = df[pred_col].ewm(span=span, adjust=False).mean()

    # apply direction
    x = dir_ * x

    # band to 0..1
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        lo, hi = np.percentile(x.values, [40, 80])

    sig = ((x - lo) / (hi - lo)).clip(0.0, 1.0)

    df["pred_use"] = x
    df["signal"] = sig

    today_row = df.iloc[-1]
    today_sig = float(today_row["signal"])
    today_pred_use = float(today_row["pred_use"])

    return today_sig, today_pred_use, df
