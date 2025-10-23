import numpy as np
import pandas as pd
from pathlib import Path
from nst.common.io import load_split

from datasets import Dataset
from transformers import (
    AutoTokenizer, AutoModelForSequenceClassification,
    Trainer, TrainingArguments, DataCollatorWithPadding, set_seed
)
from sklearn.metrics import mean_absolute_error, r2_score, root_mean_squared_error


TRAIN_PATH = "data/processed/train.parquet"
VAL_PATH   = "data/processed/val.parquet"
TEST_PATH  = "data/processed/test.parquet"

TEXT_COL = "headline"
LABEL_COL = "pct_val"            # or "global_sentiment"

BASE_MODEL   = "yiyanghkust/finbert-tone"
MAX_LEN      = 64
BATCH_SIZE   = 16
EPOCHS       = 6
LR_HEAD      = 2e-5
WARMUP_RATIO = 0.10
WEIGHT_DECAY = 0.01
FREEZE_ENCODER = True
SEED = 42
TAU = 0.0                        # no dead-zone for regression
UNFREEZE_LAYERS = 1

OUTPUT_DIR = "data/artifacts/stage1_from_splits"

set_seed(SEED)

def to_hfds(df, text_col, label_col, tok, max_len):
    df = df[[text_col, label_col]].copy()
    df[label_col] = df[label_col].astype("float32")
    ds = Dataset.from_pandas(df, preserve_index=False)

    def tok_fn(batch):
        return tok(batch[text_col], truncation=True, max_length=max_len)
    ds = ds.map(tok_fn, batched=True)

    ds = ds.rename_column(label_col, "labels")
    ds = ds.remove_columns([text_col])
    ds.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])
    return ds

def compute_metrics(eval_pred):
    preds, labels = eval_pred
    preds = np.array(preds).reshape(-1).astype(float)
    labels = np.array(labels).reshape(-1).astype(float)

    rmse = root_mean_squared_error(labels, preds)
    mae  = mean_absolute_error(labels, preds)
    r2   = r2_score(labels, preds)

    # Correlations
    pearson = float(np.corrcoef(labels, preds)[0, 1])
    spearman = float(pd.Series(labels).rank().corr(pd.Series(preds).rank()))

    return {
        "rmse": rmse,
        "mae": mae,
        "r2": r2,
        "pearson": pearson,
        "spearman": spearman,
    }

def run():
    # 1) load processed splits
    train_df = load_split("train")
    val_df   = load_split("val")
    test_df  = load_split("test")

    # 2) sanity: required columns exist
    for name, df_ in [("train", train_df), ("val", val_df), ("test", test_df)]:
        if TEXT_COL not in df_.columns or LABEL_COL not in df_.columns:
            raise ValueError(f"{name}.parquet must have columns: '{TEXT_COL}', '{LABEL_COL}'")

    # 3) tokenizer & datasets
    tok = AutoTokenizer.from_pretrained(BASE_MODEL, use_fast=True)
    train_ds = to_hfds(train_df, TEXT_COL, LABEL_COL, tok, MAX_LEN)
    val_ds   = to_hfds(val_df,   TEXT_COL, LABEL_COL, tok, MAX_LEN)
    test_ds  = to_hfds(test_df,  TEXT_COL, LABEL_COL, tok, MAX_LEN)

    # 4) model as regression head (1 output)
    model = AutoModelForSequenceClassification.from_pretrained(
        BASE_MODEL,
        num_labels=1,
        ignore_mismatched_sizes=True,
        problem_type="regression",
    )

    # 5) freeze encoder then unfreeze last N layers
    if FREEZE_ENCODER:
        for p in model.base_model.parameters():
            p.requires_grad = False
    for name, param in model.named_parameters():
        if any(f"encoder.layer.{i}." in name for i in range(12 - UNFREEZE_LAYERS, 12)):
            param.requires_grad = True

    # 6) training args
    args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="pearson",
        greater_is_better=True,
        num_train_epochs=EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        learning_rate=LR_HEAD,
        weight_decay=WEIGHT_DECAY,
        warmup_ratio=WARMUP_RATIO,
        logging_steps=50,
        report_to="none",
        seed=SEED,
    )
    collator = DataCollatorWithPadding(tokenizer=tok)

    # 7) train
    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        tokenizer=tok,
        data_collator=collator,
        compute_metrics=compute_metrics,
    )
    trainer.train()

    # 8) save and test
    Path(f"{OUTPUT_DIR}/best").mkdir(parents=True, exist_ok=True)
    trainer.save_model(f"{OUTPUT_DIR}/best")
    tok.save_pretrained(f"{OUTPUT_DIR}/best")

    print("TEST:", trainer.evaluate(test_ds))

if __name__ == "__main__":
    run()
