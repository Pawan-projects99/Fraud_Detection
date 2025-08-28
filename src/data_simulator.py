import numpy as np
import pandas as pd
import os
from datetime import datetime, timedelta, timezone

def create_synthetic_fraud_data(n_samples: int, fraud_ratio: float):
    """Create synthetic fraud detection dataset with mixed features"""
    
    np.random.seed(42)
    n_fraud = int(n_samples * fraud_ratio)
    n_normal = n_samples - n_fraud

    start_date = datetime.now(timezone.utc)-timedelta(days=90)
    end_date = datetime.now(timezone.utc)
    timestamps = [
        start_date + timedelta(seconds = np.random.randint(0, (end_date - start_date).total_seconds()))
        for _ in range(n_samples)
    ]
    
    # Create labels
    labels = np.concatenate([np.zeros(n_normal), np.ones(n_fraud)])
    
    # Numerical features
    numerical_data = []
    
    for i in range(n_samples):
        if labels[i] == 0:  # Normal transaction
            transaction_amount = np.random.lognormal(3, 1)  # Typical amounts
            account_balance = np.random.lognormal(7, 1.5)
            p = np.array([0.15] * 14)
            p = p / p.sum()   # makes it sum to 1
            transaction_hour = np.random.choice(range(8, 22), p=p)
            days_since_last = np.random.exponential(2)
            merchant_risk_score = np.random.beta(2, 8) * 10  # Lower risk
            velocity_1h = np.random.poisson(0.5)  # Low velocity
            velocity_24h = np.random.poisson(3)
            
        else:  # Fraudulent transaction
            transaction_amount = np.random.choice([
                np.random.lognormal(6, 1),  # High amounts
                np.random.lognormal(1, 0.5)  # Micro-transactions
            ], p=[0.7, 0.3])
            account_balance = np.random.lognormal(6, 2)
            transaction_hour = np.random.choice(range(24))  # Any time
            days_since_last = np.random.exponential(0.1)  # Recent activity
            merchant_risk_score = np.random.beta(8, 2) * 10  # Higher risk
            velocity_1h = np.random.poisson(5)  # High velocity
            velocity_24h = np.random.poisson(15)
            
        
        numerical_data.append([
            transaction_amount,
            account_balance,
            transaction_hour,
            days_since_last,
            merchant_risk_score,
            velocity_1h,
            velocity_24h
        ])
    
    numerical_data = np.array(numerical_data)
    
    # Categorical features
    categorical_data = []
    
    for i in range(n_samples):
        if labels[i] == 0:  # Normal
            payment_method = np.random.choice([0, 1, 2, 3], p=[0.4, 0.3, 0.2, 0.1])  # Credit, debit, cash, other
            merchant_category = np.random.choice(range(10), p=[0.2, 0.15, 0.1, 0.1, 0.1, 0.08, 0.08, 0.07, 0.07, 0.05])
            device_type = np.random.choice([0, 1, 2], p=[0.5, 0.3, 0.2])  # Desktop, mobile, tablet
            location_risk = np.random.choice([0, 1, 2], p=[0.7, 0.2, 0.1])  # Low, medium, high
        else:  # Fraudulent
            payment_method = np.random.choice([0, 1, 2, 3], p=[0.6, 0.25, 0.05, 0.1])  # More credit cards
            merchant_category = np.random.choice(range(10), p=[0.05, 0.05, 0.3, 0.2, 0.15, 0.1, 0.05, 0.05, 0.025, 0.025])
            device_type = np.random.choice([0, 1, 2], p=[0.3, 0.6, 0.1])  # More mobile
            location_risk = np.random.choice([0, 1, 2], p=[0.3, 0.4, 0.3])  # Higher risk locations
        
        categorical_data.append([payment_method, merchant_category, device_type, location_risk])
    
    categorical_data = np.array(categorical_data)
    
    # Shuffle the data
    indices = np.random.permutation(n_samples)
    numerical_data = numerical_data[indices]
    categorical_data = categorical_data[indices]
    labels = labels[indices]
    
    # Create DataFrame
    num_cols = ['amount', 'balance', 'hour', 'days_since_last', 'merchant_risk', 'velocity_1h', 'velocity_24h']
    cat_cols = ['payment_method', 'merchant_category', 'device_type', 'location_risk']
    
    df = pd.DataFrame(numerical_data, columns=num_cols)
    for i, col in enumerate(cat_cols):
        df[col] = categorical_data[:, i]
    df['is_fraud'] = labels.astype(int)
    df["timestamp"] = timestamps
    df["transaction_id"] = [f"TX_{i}" for i in range(n_samples)]
    df["user_id"] = [f"UR_{np.random.randint(2000)}" for _ in range(n_samples)]
    df["merchant_id"] = [f"M_{np.random.randint(200)}" for _ in range(n_samples)]
    return df

if __name__ == "__main__":
    df = create_synthetic_fraud_data(200000, 0.02)
    df.to_parquet("./features/data/transactions.parquet")
