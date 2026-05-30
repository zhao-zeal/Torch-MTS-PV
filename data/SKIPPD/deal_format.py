import pandas as pd

df = pd.read_csv("SKIPPD.csv")
df["date"] = pd.to_datetime(df["date"]).dt.strftime("%Y/%m/%d %H:%M:%S")
df.to_csv("SKIPPD.csv", index=False)