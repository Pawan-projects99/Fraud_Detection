# import numpy as np
import pandas as pd
import torch
# from datetime import datetime,timedelta
# print(datetime.now())
# print(timedelta(minutes = np.random.randint(1,1440)))
# print(datetime.now()-timedelta(minutes = np.random.randint(1,1440)))
# print("Hello")

df = pd.read_parquet("features/data/user_features.parquet")

# print(df[df["timestamp"]== df["timestamp"].max()])
# print(df.head(10))
# print(df["timestamp"].dtype)
# df1 = pd.read_parquet("features/data/user_features.parquet")
# print(df1["event_timestamp"].min(), df1["event_timestamp"].max())
# df1["event_timestamp"] = df1["event_timestamp"].astype("datetime64[ns]")
# print(df1["event_timestamp"].dtype)
print(df.head(10))
print (df.shape)
print(df.columns)
print(df.describe())
print(df["amount_spike_indicator"])

# print(df1.isna().sum())
# df2 = pd.read_parquet("features/data/merchant_features.parquet")
# print(df2["event_timestamp"].min(), df1["event_timestamp"].max())
# df2["event_timestamp"] = df2["event_timestamp"].astype("datetime64[ns]")
# print(df2["event_timestamp"].dtype)
# print(df2.head(10))
# print(df2.shape)
# print(df2.isna().sum())
# from datetime import datetime, timezone, timedelta
# start_date = datetime.now(timezone.utc)-timedelta(days=90)
# end_date = datetime.now(timezone.utc)
# new_date = pd.to_datetime(end_date)
# print(start_date, ",", end_date, ",", new_date)
# print(torch.__version__)
# print(df["payment_type"].unique())
# print(df.shape)

# pos_weight = int(df["is_fraud"].sum())
# print(pos_weight)
