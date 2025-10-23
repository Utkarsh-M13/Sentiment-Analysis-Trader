# build_splits_nextday.py  (drop-in replacement)
from datasets import load_dataset, Dataset
import pandas as pd, ast, numpy as np
from pathlib import Path

HF_NAME = "baptle/financial_headlines_market_based"
ds = load_dataset(HF_NAME)
split_name = list(ds.keys())[0]             # usually "train" for HF community sets
raw_df = ds[split_name].to_pandas()

# --- 1) Basic parsing & cleaning ---
raw_df["Date"] = pd.to_datetime(raw_df["Date"], errors="coerce")
raw_df = raw_df.dropna(subset=["Date", "Title"]).copy()
raw_df["Date"] = raw_df["Date"].dt.date

# Deduplicate identical headlines within a day
raw_df = raw_df.drop_duplicates(subset=["Date", "Title"]).copy()

def safe_literal(x):
    try:
        return ast.literal_eval(x)
    except Exception:
        return x

# Many columns in this HF dataset are dict-like strings
for col in ["Sentiment", "Pct_Change", "Global Sentiment"]:
    if col in raw_df.columns:
        raw_df[col] = raw_df[col].apply(safe_literal)

# --- 2) Extract headline text, same-day return, and global sentiment candidate ---
raw_df["headline"] = raw_df["Title"].astype(str)
raw_df = raw_df.rename(columns={"Date": "date"})  # normalize

def pick_spy_or_first(d):
    if not isinstance(d, dict) or len(d) == 0:
        return np.nan
    if "SPY" in d:
        return d["SPY"]
    # fallbacks that sometimes appear
    for k in ["ACWI", "Global", "SPX", "IVV"]:
        if k in d:
            return d[k]
    # else first key
    return next(iter(d.values()))

# same-day move (we will shift this to next day below)
raw_df["pct_val_same"] = raw_df["Pct_Change"].apply(pick_spy_or_first)

def extract_global_sent(x):
    # Prefer an explicit "Global Sentiment" column if present and scalar
    if isinstance(x, (int, float, np.floating)):
        return float(x)
    if isinstance(x, dict):
        for k in ["Global", "global", "SPY", "ACWI"]:
            if k in x and isinstance(x[k], (int, float, np.floating)):
                return float(x[k])
        # fallback: mean of numeric values
        vals = [v for v in x.values() if isinstance(v, (int, float, np.floating))]
        return float(np.mean(vals)) if len(vals) else np.nan
    return np.nan

if "Global Sentiment" in raw_df.columns:
    raw_df["global_sentiment"] = raw_df["Global Sentiment"].apply(extract_global_sent)
elif "Sentiment" in raw_df.columns:
    raw_df["global_sentiment"] = raw_df["Sentiment"].apply(extract_global_sent)
else:
    raw_df["global_sentiment"] = np.nan  # column will exist even if missing

# --- 3) Convert same-day returns to next-day returns by date ---
raw_df = raw_df.sort_values("date").reset_index(drop=True)

# Make a date-level table of same-day returns (averaged across headlines if multiple)
daily = (
    raw_df.groupby("date", as_index=False)["pct_val_same"]
    .mean()
    .sort_values("date")
    .rename(columns={"pct_val_same": "pct_val_same_day"})
)

# Shift -1 so each date maps to *next day's* return
daily["pct_val_next"] = daily["pct_val_same_day"].shift(-1)

# Map next-day return back to each headline row by its date
map_next = dict(zip(daily["date"], daily["pct_val_next"]))
raw_df["pct_val"] = raw_df["date"].map(map_next)

# Drop rows that cannot have a next-day label (e.g., last day)
raw_df = raw_df.dropna(subset=["pct_val"]).copy()

# --- 4) Chronological day-based split (train/val/test by unique dates) ---
TRAIN_PCT, VAL_PCT, TEST_PCT = 0.70, 0.15, 0.15

unique_days = sorted(pd.Series(raw_df["date"].unique()))
n = len(unique_days)
i = int(TRAIN_PCT * n)
j = int((TRAIN_PCT + VAL_PCT) * n)

train_days = set(unique_days[:i])
val_days   = set(unique_days[i:j])
test_days  = set(unique_days[j:])

def filter_by_days(df, days):
    return df[df["date"].isin(days)].copy()

train_df = filter_by_days(raw_df, train_days)
val_df   = filter_by_days(raw_df, val_days)
test_df  = filter_by_days(raw_df, test_days)

print("Train:", min(train_df["date"]), "→", max(train_df["date"]), "| rows:", len(train_df))
print("Val:  ", min(val_df["date"]),   "→", max(val_df["date"]),   "| rows:", len(val_df))
print("Test: ", min(test_df["date"]),  "→", max(test_df["date"]),  "| rows:", len(test_df))

# --- 5) Save in the exact schema your training code expects ---
Path("data/processed").mkdir(parents=True, exist_ok=True)

keep_cols = ["date", "headline", "pct_val", "global_sentiment"]

# Ensure types are friendly
for df_ in (train_df, val_df, test_df):
    df_["date"] = pd.to_datetime(df_["date"]).dt.date
    # keep headline as str, pct_val/global_sentiment as float
    df_["headline"] = df_["headline"].astype(str)
    df_["pct_val"] = pd.to_numeric(df_["pct_val"], errors="coerce")
    df_["global_sentiment"] = pd.to_numeric(df_["global_sentiment"], errors="coerce")

train_df[keep_cols].to_parquet("data/processed/train.parquet", index=False)
val_df[keep_cols].to_parquet("data/processed/val.parquet", index=False)
test_df[keep_cols].to_parquet("data/processed/test.parquet", index=False)

print("Wrote data/processed/{train,val,test}.parquet with columns:", keep_cols)
