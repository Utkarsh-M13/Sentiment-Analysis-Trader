# src/nst/eval/sweep_thresholds.py
import numpy as np, pandas as pd
from nst.backtests.backtest_simple import backtest_daily

VAL_CSV = "experiments/val_predictions.csv"  # create like test_predictions

def main():
    ths = np.linspace(0.50, 0.75, 21)
    rows = []
    for t in ths:
        _, kpis = backtest_daily(VAL_CSV, best_t=float(t), long_short=False, cost_bps=5, use_hysteresis=False)
        rows.append({"t": float(t), **kpis})
    df = pd.DataFrame(rows)
    print(df.sort_values("Sharpe", ascending=False).head(10))
    best = df.loc[df["Sharpe"].idxmax()]
    print("\nBest by Sharpe:", best.to_dict())

if __name__ == "__main__":
    main()
