from datasets import load_dataset, Dataset
import pandas as pd, ast
from nst.common.io import read_yaml
import numpy as np
from pathlib import Path

HF_NAME = "baptle/financial_headlines_market_based"
ds = load_dataset(HF_NAME)           
split_name = list(ds.keys())[0]      
raw_df = ds[split_name].to_pandas()  

# Parse date safely
raw_df["Date"] = pd.to_datetime(raw_df["Date"], errors="coerce").dt.date
raw_df = raw_df.dropna(subset=["Date", "Title"])

# Remove duplicate headlines per day
before = len(raw_df)
raw_df = raw_df.drop_duplicates(subset=["Date", "Title"])


def safe_literal(x):
    try: return ast.literal_eval(x)
    except Exception: return x


for col in ["Sentiment", "Pct_Change"]:
    raw_df[col] = raw_df[col].apply(safe_literal)

    
def pick_spy_or_first(d):
    if not isinstance(d, dict) or len(d) == 0:
        return np.nan
    if "SPY" in d:
        return d["SPY"]
    first_key = list(d.keys())[0]
    return d[first_key]

raw_df["Pct_Val"] = raw_df["Pct_Change"].apply(pick_spy_or_first)

raw_df["label"] = (raw_df["Pct_Val"] > 0).astype(int)

# Sort by date so we can slice chronologically
raw_df = raw_df.sort_values("Date").reset_index(drop=True)


# Choose your split ratios
TRAIN_PCT = 0.70
VAL_PCT   = 0.15
TEST_PCT  = 0.15

# Extract unique trading days
unique_days = sorted(raw_df["Date"].unique())
n = len(unique_days)
i = int(TRAIN_PCT * n)
j = int((TRAIN_PCT + VAL_PCT) * n)

train_days = set(unique_days[:i])
val_days   = set(unique_days[i:j])
test_days  = set(unique_days[j:])

# Filter the main dataframe
train_df = raw_df[raw_df["Date"].isin(train_days)].copy()
val_df   = raw_df[raw_df["Date"].isin(val_days)].copy()
test_df  = raw_df[raw_df["Date"].isin(test_days)].copy()

print("Train:", train_df["Date"].min(), "→", train_df["Date"].max(), "| rows:", len(train_df))
print("Val:  ", val_df["Date"].min(),   "→", val_df["Date"].max(),   "| rows:", len(val_df))
print("Test: ", test_df["Date"].min(),  "→", test_df["Date"].max(),  "| rows:", len(test_df))

Path("data/processed").mkdir(parents=True, exist_ok=True)

cols = ["Date","Title","label","Pct_Val","Global Sentiment"]  # adjust to your rename
train_df[cols].to_parquet("data/processed/total_train.parquet", index=False)
val_df[cols].to_parquet("data/processed/total_val.parquet", index=False)
test_df[cols].to_parquet("data/processed/total_test.parquet", index=False)




