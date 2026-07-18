import pandas as pd
import glob
import os

files = glob.glob("data/*.csv")
for f in files:
    print(f"File: {f}")
    df = pd.read_csv(f)
    print(f"Columns: {list(df.columns)}")
    print(f"Rows: {len(df)}")
    # Combine Date and Time
    df["datetime"] = pd.to_datetime(df["Date"].astype(str) + " " + df["Time"].astype(str))
    print(f"Start Date: {df['datetime'].min()}")
    print(f"End Date: {df['datetime'].max()}")
    print("-" * 50)
