import pandas as pd
import numpy as np
from datetime import datetime, timedelta, timezone
from feast import FeatureStore

def calculate_user_features(transactions):
    return transactions.groupby("user_id").agg(
        avg_transaction = ("amount", "mean"),
        transaction_count_30d = ("timestamp", lambda x: (((transactions["timestamp"].max())-x) < timedelta(days = 30)).sum()),
        decline_rate_60d = ("is_fraud", lambda x: x[((transactions["timestamp"].max())- transactions.loc[x.index, "timestamp"]) < timedelta(days = 60)].mean()),
        
    ).reset_index()

def calculate_merchant_features(transactions):
    return transactions.groupby("merchant_id").agg(
        fraud_rate = ("is_fraud", "mean"),
        avg_transaction_value = ("amount", "mean"),
    ).reset_index()

def compute_features():

    df = pd.read_parquet("./features/data/transactions.parquet")
    df["timestamp"] = pd.to_datetime(df["timestamp"])

    user_features = calculate_user_features(df)
    merchant_features = calculate_merchant_features(df)

    user_features["event_timestamp"] = df["timestamp"]
    merchant_features["event_timestamp"] = df["timestamp"]

    user_features.to_parquet("./features/data/user_features.parquet")
    merchant_features.to_parquet("./features/data/merchant_features.parquet")

    store = FeatureStore("./features")
    store.materialize_incremental(end_date = df["timestamp"].max())

if __name__ =="__main__":
    compute_features()