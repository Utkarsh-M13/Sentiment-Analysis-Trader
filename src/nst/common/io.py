from pathlib import Path
import pandas as pd
import yaml

def read_yaml(path):
    with open(path, "r") as f:
        return yaml.safe_load(f)

def ensure_dir(p):
    Path(p).mkdir(parents=True, exist_ok=True)
    

def load_split(split):
    path = f"data/processed/total_{split}.parquet"
    df = pd.read_parquet(path)
    # standardize column names expected by the tokenizer/trainer
    df = df.rename(columns={
        "Title": "headline",
        "Date": "date",
        "Global Sentiment": "global_sentiment",
        "Pct_Val": "pct_val"
    })
    # keep only what Trainer needs (+ anything you want to log)
    return df[["headline","label","date","pct_val","global_sentiment"]]

