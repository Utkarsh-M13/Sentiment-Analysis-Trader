import pandas as pd
df = pd.read_parquet("data/processed/total_test.parquet")
print("Pearson corr(Global Sentiment, Pct_Val):", end=" ")
print(df["Global Sentiment"].corr(df["Pct_Val"]))

print("Spearman corr(Global Sentiment, Pct_Val):", end=" ")
print(df["Global Sentiment"].corr(df["Pct_Val"], method="spearman"))

df = pd.read_csv("experiments/test_predictions.csv")

print("Pearson corr(p_up, Global Sentiment):", df["p_up"].corr(df["global_sentiment"]))
print("Spearman corr(p_up, Global Sentiment):", df["p_up"].corr(df["global_sentiment"], method="spearman"))



