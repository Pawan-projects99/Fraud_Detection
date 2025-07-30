import numpy as np
import pandas as pd
import os
from datetime import datetime, timedelta, timezone

def generate_transactions(num_records):
    np.random.seed(42)
    start_date = datetime.now(timezone.utc)-timedelta(days=90)
    end_date = datetime.now(timezone.utc)
    timestamps = [
        start_date + timedelta(seconds = np.random.randint(0, (end_date - start_date).total_seconds()))
        for _ in range(num_records)
    ]

    return pd.DataFrame({
        "transaction_id": [f"TX_{i}" for i in range(num_records)],
        "user_id": [f"UR_{np.random.randint(2000)}" for _ in range(num_records)],
        "merchant_id": [f"MR_{np.random.randint(100)}" for _ in range(num_records)],
        "amount": np.round(np.random.lognormal(3,1.5, num_records),2),
        "location": np.random.choice(["IN", "US", "UK", "GE", "CN"], num_records),
        "device": np.random.choice(["Mobile", "Laptop", "Tablet"], num_records),
        "timestamp": timestamps,
        "is_fraud": np.random.binomial(1, 0.02, num_records),
        "payment_type":np.random.choice(["VISA", "MASTER", "AMEX"])
    })

df = generate_transactions(200000)
os.makedirs("./data", exist_ok = True)
df.to_parquet("./data/transactions.parquet")
