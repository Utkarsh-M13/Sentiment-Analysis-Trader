import pandas as pd
val = pd.read_csv("experiments/val_predictions.csv")
test = pd.read_csv("experiments/test_predictions.csv")
print(val.columns)
print("VAL rows:", len(val), "| TEST rows:", len(test))
print(val.head(3))
