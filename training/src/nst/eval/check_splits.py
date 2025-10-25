import pandas as pd

splits = {
    "train": "data/processed/total_train.parquet",
    "val":   "data/processed/total_val.parquet",
    "test":  "data/processed/total_test.parquet",
}

for name, path in splits.items():
    df = pd.read_parquet(path)
    print(f"\n{name.upper()}  —  rows: {len(df)}")
    print("Date range:", df["Date"].min(), "→", df["Date"].max())
