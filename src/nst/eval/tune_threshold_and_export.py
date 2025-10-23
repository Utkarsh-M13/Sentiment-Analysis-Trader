# src/nst/eval/tune_threshold.py
import json
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.metrics import f1_score
from transformers import (
    AutoTokenizer, AutoModelForSequenceClassification,
    Trainer, DataCollatorWithPadding
)
from nst.train.train_stage1 import load_split, BASE_MODEL, MAX_LEN, OUTPUT_DIR

def to_hfds_text_only(df, text_col, tok, max_len):
    """Tokenize text only. No labels required for prediction."""
    from datasets import Dataset
    ds = Dataset.from_pandas(df[[text_col]].copy(), preserve_index=False)

    def tok_fn(batch):
        return tok(batch[text_col], truncation=True, max_length=max_len)

    ds = ds.map(tok_fn, batched=True)
    ds = ds.remove_columns([text_col])
    ds.set_format(type="torch", columns=["input_ids", "attention_mask"])
    return ds

def percentile_thresholds(scores, lo=5, hi=95, n=41):
    """Evenly spaced thresholds between lo and hi percentiles of scores."""
    lo_v = np.percentile(scores, lo)
    hi_v = np.percentile(scores, hi)
    if lo_v == hi_v:
        # fallback if predictions are nearly constant
        lo_v, hi_v = scores.min(), scores.max()
        if lo_v == hi_v:
            lo_v, hi_v = -1.0, 1.0
    return np.linspace(lo_v, hi_v, n)

def main():
    Path("experiments").mkdir(exist_ok=True)

    # 1) Load model + tokenizer
    tok = AutoTokenizer.from_pretrained(BASE_MODEL, use_fast=True)
    model = AutoModelForSequenceClassification.from_pretrained(Path(OUTPUT_DIR) / "best")
    collator = DataCollatorWithPadding(tokenizer=tok)
    trainer = Trainer(model=model, tokenizer=tok, data_collator=collator)

    # 2) VAL predictions
    val_df = load_split("val")
    val_ds = to_hfds_text_only(val_df, "headline", tok, MAX_LEN)
    val_pred = trainer.predict(val_ds)

    # Regression head returns shape (N, 1). Squeeze to (N,)
    v_scores = np.array(val_pred.predictions).reshape(-1).astype(float)

    # Ground truth for threshold tuning:
    # If a binary "label" exists, use it. Else use sign of pct_val.
    if "label" in val_df.columns:
        y_val = val_df["label"].astype(int).to_numpy()
    else:
        if "pct_val" not in val_df.columns:
            raise ValueError("val split must include 'label' or 'pct_val' for threshold tuning.")
        y_val = (val_df["pct_val"].values > 0).astype(int)

    # Search threshold over prediction percentiles
    ths = percentile_thresholds(v_scores, lo=5, hi=95, n=41)
    best = max(((f1_score(y_val, v_scores > t, average="macro"), t) for t in ths), key=lambda x: x[0])
    best_f1, best_t = best
    print(f"[VAL] best threshold by F1: {best_t:.6f} (F1={best_f1:.3f})")

    # Export val predictions
    val_out = val_df[["date", "headline"]].copy()
    if "pct_val" in val_df.columns:
        val_out["pct_val"] = val_df["pct_val"].values
    if "global_sentiment" in val_df.columns:
        val_out["global_sentiment"] = val_df["global_sentiment"].values

    # Provide both raw score and a 0..1 rank-normalized version for compatibility
    val_out["score"] = v_scores
    ranks = pd.Series(v_scores).rank(method="average")
    val_out["p_up"] = (ranks - 1) / (len(ranks) - 1) if len(ranks) > 1 else 0.5

    val_out.to_csv("experiments/val_predictions.csv", index=False)

    with open("experiments/best_threshold.json", "w") as f:
        json.dump({"best_t": float(best_t), "best_f1": float(best_f1)}, f, indent=2)

    # 3) TEST predictions
    test_df = load_split("test")
    test_ds = to_hfds_text_only(test_df, "headline", tok, MAX_LEN)
    test_pred = trainer.predict(test_ds)
    t_scores = np.array(test_pred.predictions).reshape(-1).astype(float)

    test_out = test_df[["date", "headline"]].copy()
    for col in ["pct_val", "global_sentiment"]:
        if col in test_df.columns:
            test_out[col] = test_df[col].values

    test_out["score"] = t_scores
    tranks = pd.Series(t_scores).rank(method="average")
    test_out["p_up"] = (tranks - 1) / (len(tranks) - 1) if len(tranks) > 1 else 0.5

    test_out.to_csv("experiments/test_predictions.csv", index=False)

    print(f"Saved VAL to experiments/val_predictions.csv ({len(val_out)})")
    print(f"Saved TEST to experiments/test_predictions.csv ({len(test_out)})")
    print(f"Use BEST_T = {best_t:.6f} on 'score' for binary signals.")

if __name__ == "__main__":
    main()
