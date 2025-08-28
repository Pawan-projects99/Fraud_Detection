# import pandas as pd
# import numpy as np
# from datetime import datetime, timedelta, timezone
# from feast import FeatureStore

# def calculate_user_features(transactions):
#     return transactions.groupby("user_id").agg(
#         avg_transaction = ("amount", "mean"),
#         transaction_count_30d = ("timestamp", lambda x: (((transactions["timestamp"].max())-x) < timedelta(days = 30)).sum()),
#         decline_rate_60d = ("is_fraud", lambda x: x[((transactions["timestamp"].max())- transactions.loc[x.index, "timestamp"]) < timedelta(days = 60)].mean()),
        
#     ).reset_index()

# def calculate_merchant_features(transactions):
#     return transactions.groupby("merchant_id").agg(
#         fraud_rate = ("is_fraud", "mean"),
#         avg_transaction_value = ("amount", "mean"),
#     ).reset_index()

# def compute_features():

#     df = pd.read_parquet("./features/data/transactions.parquet")
#     df["timestamp"] = pd.to_datetime(df["timestamp"])

#     user_features = calculate_user_features(df)
#     merchant_features = calculate_merchant_features(df)

#     user_features["event_timestamp"] = df["timestamp"]
#     merchant_features["event_timestamp"] = df["timestamp"]

#     user_features.to_parquet("./features/data/user_features.parquet")
#     merchant_features.to_parquet("./features/data/merchant_features.parquet")

#     store = FeatureStore("./features")
#     store.materialize_incremental(end_date = df["timestamp"].max())

# if __name__ =="__main__":
#     compute_features()

# feature_engineering.py
# feature_engineering.py
import pandas as pd
import numpy as np
from datetime import datetime, timedelta, timezone
from feast import FeatureStore
from feast.data_source import PushMode
from typing import Dict, List, Tuple
import uuid
import os
import warnings
warnings.filterwarnings('ignore')

class FraudFeatureEngineer:
    """Feature engineering pipeline for fraud detection using Feast"""
    
    def __init__(self, feature_store_path: str = "./features"):
        """Initialize feature store connection"""
        # Initialize Feast with local configuration
        self.fs = FeatureStore(repo_path=feature_store_path)
        self.feature_store_path = feature_store_path
        
        # Create data directory if it doesn't exist
        os.makedirs(os.path.join(feature_store_path, "data"), exist_ok=True)
        
        
    def compute_user_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Compute user-level aggregated features"""
        
        user_features_list = []
        
        for user_id in df['user_id'].unique():
            user_df = df[df['user_id'] == user_id].sort_values('timestamp')
            
            # Basic account features (using latest values)
            latest_balance = user_df['balance'].iloc[-1]
            
            # Historical transaction patterns
            user_amounts = user_df['amount'].values
            user_timestamps = pd.to_datetime(user_df['timestamp'])
            
            # Time-based features
            now = user_timestamps.max()
            last_7d = now - timedelta(days=7)
            last_30d = now - timedelta(days=30)
            
            recent_7d = user_df[user_timestamps >= last_7d]
            recent_30d = user_df[user_timestamps >= last_30d]
            
            # Days since last transaction
            if len(user_df) > 1:
                days_since_last = (user_timestamps.iloc[-1] - user_timestamps.iloc[-2]).total_seconds() / 86400
            else:
                days_since_last = 0
            
            # Velocity features (simplified - using recent transactions)
            velocity_1h = len(recent_7d) // 7 // 24  # Approximate hourly rate
            velocity_24h = len(recent_7d) // 7  # Approximate daily rate
            
            # Amount aggregations
            avg_amount_7d = recent_7d['amount'].mean() if len(recent_7d) > 0 else 0
            avg_amount_30d = recent_30d['amount'].mean() if len(recent_30d) > 0 else 0
            
            # Transaction counts
            count_7d = len(recent_7d)
            count_30d = len(recent_30d)
            
            # Unique merchants
            unique_merchants_7d = recent_7d['merchant_id'].nunique() if len(recent_7d) > 0 else 0
            unique_merchants_30d = recent_30d['merchant_id'].nunique() if len(recent_30d) > 0 else 0
            
            # Time pattern features
            if len(recent_7d) > 0:
                night_hours = recent_7d['hour'].apply(lambda x: 1 if x >= 22 or x <= 6 else 0)
                night_ratio_7d = night_hours.mean()
                
                # Weekend ratio (simplified - using day of week from timestamp)
                weekend_ratio_7d = 0.2  # Placeholder
                
                # High amount transactions (>95th percentile of user's history)
                if len(user_amounts) > 5:
                    high_threshold = np.percentile(user_amounts, 95)
                    high_amount_ratio_7d = (recent_7d['amount'] > high_threshold).mean()
                else:
                    high_amount_ratio_7d = 0
            else:
                night_ratio_7d = 0
                weekend_ratio_7d = 0  
                high_amount_ratio_7d = 0
            
            # Risk indicators
            cross_border_ratio_7d = np.random.beta(1, 10)  # Placeholder
            failed_transactions_7d = np.random.poisson(0.5)  # Placeholder
            
            # Account age (days since first transaction)
            account_age_days = (now - user_timestamps.min()).days
            
            # User risk score based on patterns
            risk_factors = [
                night_ratio_7d * 2,
                (velocity_24h > 10) * 3,
                (high_amount_ratio_7d > 0.3) * 2,
                (unique_merchants_7d > 20) * 1.5
            ]
            user_risk_score = min(sum(risk_factors), 10)
            
            user_features_list.append({
                'user_id': user_id,
                'account_balance': latest_balance,
                'days_since_last_transaction': days_since_last,
                'velocity_1h': max(velocity_1h, 0),
                'velocity_24h': max(velocity_24h, 0),
                'avg_transaction_amount_7d': avg_amount_7d,
                'avg_transaction_amount_30d': avg_amount_30d,
                'transaction_count_7d': count_7d,
                'transaction_count_30d': count_30d,
                'unique_merchants_7d': unique_merchants_7d,
                'unique_merchants_30d': unique_merchants_30d,
                'night_transaction_ratio_7d': night_ratio_7d,
                'weekend_transaction_ratio_7d': weekend_ratio_7d,
                'high_amount_transaction_ratio_7d': high_amount_ratio_7d,
                'cross_border_transaction_ratio_7d': cross_border_ratio_7d,
                'failed_transaction_count_7d': failed_transactions_7d,
                'account_age_days': account_age_days,
                'user_risk_score': user_risk_score,
                'timestamp': now
            })
        
        return pd.DataFrame(user_features_list)
    
    def compute_merchant_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Compute merchant-level aggregated features"""
        
        merchant_features_list = []
        
        for merchant_id in df['merchant_id'].unique():
            merchant_df = df[df['merchant_id'] == merchant_id].sort_values('timestamp')
            
            # Basic merchant info
            merchant_category = merchant_df['merchant_category'].iloc[-1]
            merchant_risk_base = merchant_df['merchant_risk'].iloc[-1]
            
            # Time-based aggregations
            now = merchant_df['timestamp'].max()
            last_7d = now - timedelta(days=7)
            last_30d = now - timedelta(days=30)
            
            recent_7d = merchant_df[merchant_df['timestamp'] >= last_7d]
            recent_30d = merchant_df[merchant_df['timestamp'] >= last_30d]
            
            # Fraud rates
            if len(recent_7d) > 0:
                fraud_rate_7d = recent_7d['is_fraud'].mean()
            else:
                fraud_rate_7d = 0
                
            if len(recent_30d) > 0:
                fraud_rate_30d = recent_30d['is_fraud'].mean()
            else:
                fraud_rate_30d = 0
            
            # Transaction patterns
            transaction_count_7d = len(recent_7d)
            avg_amount_7d = recent_7d['amount'].mean() if len(recent_7d) > 0 else 0
            unique_users_7d = recent_7d['user_id'].nunique() if len(recent_7d) > 0 else 0
            
            # Risk indicators
            chargeback_rate_30d = fraud_rate_30d * 0.1  # Simplified assumption
            
            # Operating hours match (simplified)
            if len(recent_7d) > 0:
                business_hours_transactions = recent_7d['hour'].apply(lambda x: 1 if 9 <= x <= 17 else 0)
                operating_hours_match = business_hours_transactions.mean()
            else:
                operating_hours_match = 0.8  # Default assumption
            
            # Merchant reputation (based on fraud rates and patterns)
            reputation_score = max(0, 10 - fraud_rate_30d * 50 - (transaction_count_7d > 1000) * 2)
            
            # Merchant age
            merchant_age_days = (now - merchant_df['timestamp'].min()).days
            
            merchant_features_list.append({
                'merchant_id': merchant_id,
                'merchant_category': merchant_category,
                'merchant_risk_score': merchant_risk_base,
                'merchant_fraud_rate_7d': fraud_rate_7d,
                'merchant_fraud_rate_30d': fraud_rate_30d,
                'merchant_transaction_count_7d': transaction_count_7d,
                'merchant_avg_amount_7d': avg_amount_7d,
                'merchant_unique_users_7d': unique_users_7d,
                'merchant_chargeback_rate_30d': chargeback_rate_30d,
                'merchant_operating_hours_match': operating_hours_match,
                'merchant_reputation_score': reputation_score,
                'merchant_age_days': merchant_age_days,
                'timestamp': now
            })
        
        return pd.DataFrame(merchant_features_list)
    
    def compute_real_time_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Compute real-time features for each transaction"""
        
        df = df.sort_values(['user_id', 'timestamp']).reset_index(drop=True)
        real_time_features_list = []
        
        for idx, row in df.iterrows():
            user_id = row['user_id']
            current_time = row['timestamp']
            
            # Get user's transaction history up to current transaction
            user_history = df[(df['user_id'] == user_id) & (df['timestamp'] < current_time)]
            
            if len(user_history) == 0:
                # First transaction for user
                real_time_features = {
                    'user_id': user_id,
                    'transactions_last_5min': 0,
                    'transactions_last_15min': 0,
                    'amount_last_5min': 0.0,
                    'amount_last_15min': 0.0,
                    'unique_merchants_last_hour': 0,
                    'location_changes_last_hour': 0,
                    'device_changes_last_hour': 0,
                    'velocity_spike_indicator': 0.0,
                    'amount_spike_indicator': 0.0,
                    'time_since_last_transaction_seconds': 86400,  # Default to 1 day
                    'consecutive_failed_attempts': 0,
                    'timestamp': current_time
                }
            else:
                # Time windows
                time_5min = current_time - timedelta(minutes=5)
                time_15min = current_time - timedelta(minutes=15)  
                time_1hour = current_time - timedelta(hours=1)
                
                # Transactions in time windows
                recent_5min = user_history[user_history['timestamp'] >= time_5min]
                recent_15min = user_history[user_history['timestamp'] >= time_15min]
                recent_1hour = user_history[user_history['timestamp'] >= time_1hour]
                
                # Count and amount features
                transactions_last_5min = len(recent_5min)
                transactions_last_15min = len(recent_15min)
                amount_last_5min = recent_5min['amount'].sum()
                amount_last_15min = recent_15min['amount'].sum()
                
                # Diversity features
                unique_merchants_last_hour = recent_1hour['merchant_id'].nunique() if len(recent_1hour) > 0 else 0
                location_changes_last_hour = recent_1hour['location_risk'].nunique() if len(recent_1hour) > 0 else 0
                device_changes_last_hour = recent_1hour['device_type'].nunique() if len(recent_1hour) > 0 else 0
                
                # Spike indicators (compared to user's normal behavior)
                if len(user_history) >= 5:
                    normal_velocity = len(user_history) / max((user_history['timestamp'].max() - user_history['timestamp'].min()).total_seconds() / 3600, 1)
                    current_velocity = transactions_last_15min * 4  # Scale to hourly
                    velocity_spike_indicator = current_velocity / max(normal_velocity, 0.1)
                    
                    normal_amount = user_history['amount'].mean()
                    amount_spike_indicator = row['amount'] / max(normal_amount, 1)
                else:
                    velocity_spike_indicator = 1.0
                    amount_spike_indicator = 1.0
                
                # Time since last transaction
                if len(user_history) > 0:
                    time_since_last = (current_time - user_history['timestamp'].max()).total_seconds()
                else:
                    time_since_last = 86400
                
                # Consecutive failed attempts (placeholder)
                consecutive_failed_attempts = 0
                
                real_time_features = {
                    'user_id': user_id,
                    'transactions_last_5min': transactions_last_5min,
                    'transactions_last_15min': transactions_last_15min,
                    'amount_last_5min': amount_last_5min,
                    'amount_last_15min': amount_last_15min,
                    'unique_merchants_last_hour': unique_merchants_last_hour,
                    'location_changes_last_hour': location_changes_last_hour,
                    'device_changes_last_hour': device_changes_last_hour,
                    'velocity_spike_indicator': velocity_spike_indicator,
                    'amount_spike_indicator': amount_spike_indicator,
                    'time_since_last_transaction_seconds': int(time_since_last),
                    'consecutive_failed_attempts': consecutive_failed_attempts,
                    'timestamp': current_time
                }
            
            real_time_features_list.append(real_time_features)
        
        return pd.DataFrame(real_time_features_list)
    
    def prepare_transaction_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Prepare transaction-level features for Feast"""
        
        transaction_features = df[['transaction_id', 'user_id', 'merchant_id', 'amount', 'hour', 'payment_method', 
                                   'device_type', 'location_risk', 'is_fraud', 'timestamp']].copy()
        
        return transaction_features
    
    def save_features_locally(self, 
                             transaction_features: pd.DataFrame,
                             user_features: pd.DataFrame, 
                             merchant_features: pd.DataFrame,
                             real_time_features: pd.DataFrame):
        """Save feature dataframes locally as parquet files"""
        
        # data_dir = os.path.join(self.feature_store_path, "data")
        # user_features.to_parquet("./features/data/user_features.parquet")
        # Save transaction features
        transaction_features.to_parquet("./features/data/transaction_features.parquet")
        print(f"Saved {len(transaction_features)} transaction features to data/transaction_features.parquet")
        
        # Save user features
        user_features.to_parquet("./features/data/user_features.parquet")
        print(f"Saved {len(user_features)} user features to data/user_features.parquet")
        
        # Save merchant features
        merchant_features.to_parquet("./features/data/merchant_features.parquet")
        print(f"Saved {len(merchant_features)} merchant features to data/merchant_features.parquet")
        
        # Save real-time features
        real_time_features.to_parquet("./features/data/real_time_features.parquet")
        print(f"Saved {len(real_time_features)} real-time features to data/real_time_features.parquet")
        
        # Create a summary file
        summary = {
            'transaction_count': len(transaction_features),
            'user_count': len(user_features),
            'merchant_count': len(merchant_features),
            'real_time_feature_count': len(real_time_features),
            'fraud_rate': transaction_features['is_fraud'].mean(),
            'generated_at': datetime.now().isoformat()
        }
        
        summary_df = pd.DataFrame([summary])
        summary_df.to_parquet("./features/data/summary.parquet")
        print("Saved feature summary to data/feature_summary.parquet")
    
    def load_features_locally(self) -> Dict[str, pd.DataFrame]:
        """Load feature dataframes from local parquet files"""
        
        features = {}
        
        try:
            features['transaction_features'] = pd.read_parquet("./features/data/transaction_features.parquet")
            features['user_features'] = pd.read_parquet("./features/data/user_features.parquet")
            features['merchant_features'] = pd.read_parquet("./features/data/merchant_features.parquet")
            features['real_time_features'] = pd.read_parquet("./features/data/real_time_features.parquet")
            
            print("Successfully loaded all feature datasets")
            
            # Print summary
            for name, df in features.items():
                print(f"{name}: {len(df)} rows")
                
        except Exception as e:
            print(f"Error loading features: {e}")
            
        return features
    
    def push_features_to_feast_online_store(self, 
                                           transaction_features: pd.DataFrame,
                                           user_features: pd.DataFrame, 
                                           merchant_features: pd.DataFrame,
                                           real_time_features: pd.DataFrame):
        """Push features to Feast online store for real-time serving"""
        
        try:
            print("Pushing features to Feast online store...")
            
            # Materialize features to online store
            # This works with your local SQLite configuration
            
            # For transaction features - push latest per transaction_id
            if not transaction_features.empty:
                # Convert timestamp to event_timestamp for Feast
                transaction_features_feast = transaction_features.copy()
                transaction_features_feast['event_timestamp'] = pd.to_datetime(transaction_features_feast['timestamp'])
                
                # Write to online store via feature store
                self.fs.write_to_online_store(
                    feature_view_name="transaction_features",
                    df=transaction_features_feast,
                )
                print(f"Pushed {len(transaction_features_feast)} transaction features to online store")
            
            # For user features
            if not user_features.empty:
                user_features_feast = user_features.copy()
                user_features_feast['event_timestamp'] = pd.to_datetime(user_features_feast['timestamp'])
                
                self.fs.write_to_online_store(
                    feature_view_name="user_features",
                    df=user_features_feast,
                )
                print(f"Pushed {len(user_features_feast)} user features to online store")
            
            # For merchant features
            if not merchant_features.empty:
                merchant_features_feast = merchant_features.copy()
                merchant_features_feast['event_timestamp'] = pd.to_datetime(merchant_features_feast['timestamp'])
                
                self.fs.write_to_online_store(
                    feature_view_name="merchant_features",
                    df=merchant_features_feast,
                )
                print(f"Pushed {len(merchant_features_feast)} merchant features to online store")
            
            # For real-time features - use latest per user
            if not real_time_features.empty:
                latest_rt_features = real_time_features.groupby('user_id').tail(1).copy()
                latest_rt_features['event_timestamp'] = pd.to_datetime(latest_rt_features['timestamp'])
                
                self.fs.write_to_online_store(
                    feature_view_name="real_time_features",
                    df=latest_rt_features,
                )
                print(f"Pushed {len(latest_rt_features)} real-time features to online store")
                
        except Exception as e:
            print(f"Error pushing features to online store: {e}")
            print("Note: Make sure to run 'feast apply' first to create the feature views")
    def get_online_features(self, entity_rows: List[Dict]) -> pd.DataFrame:
        """Retrieve online features for real-time inference"""
        
        try:
            # Get features from the fraud detection service
            feature_vector = self.fs.get_online_features(
                features=[
                    "transaction_features:amount",
                    "transaction_features:hour", 
                    "transaction_features:payment_method",
                    "transaction_features:device_type",
                    "transaction_features:location_risk",
                    "user_features:account_balance",
                    "user_features:velocity_1h",
                    "user_features:velocity_24h",
                    "user_features:user_risk_score",
                    "merchant_features:merchant_risk_score",
                    "merchant_features:merchant_fraud_rate_7d",
                    "real_time_features:transactions_last_5min",
                    "real_time_features:velocity_spike_indicator",
                ],
                entity_rows=entity_rows,
            )
            
            return feature_vector.to_df()
            
        except Exception as e:
            print(f"Error retrieving online features: {e}")
            return pd.DataFrame()
    
    def create_training_dataset(self) -> pd.DataFrame:
        """Create a complete training dataset by joining all features"""
        
        features = self.load_features_locally()
        
        if not features:
            print("No features loaded. Please generate features first.")
            return pd.DataFrame()
        
        # Start with transaction features as base
        training_df = features['transaction_features'].copy()
        
        # Add user features
        training_df = training_df.merge(
            features['user_features'], 
            on='user_id', 
            how='left',
            suffixes=('', '_user')
        )
        
        # Add merchant features
        training_df = training_df.merge(
            features['merchant_features'], 
            on='merchant_id', 
            how='left',
            suffixes=('', '_merchant')
        )
        
        # Add real-time features (match by user_id and closest timestamp)
        rt_features = features['real_time_features'].copy()
        
        # For simplicity, merge on user_id (in production, you'd want time-based matching)
        training_df = training_df.merge(
            rt_features.groupby('user_id').tail(1)[['user_id'] + [col for col in rt_features.columns if col not in ['user_id', 'timestamp']]], 
            on='user_id', 
            how='left',
            suffixes=('', '_rt')
        )
        
        print(f"Created training dataset with {len(training_df)} rows and {len(training_df.columns)} features")
        
        # Save training dataset
        data_dir = os.path.join(self.feature_store_path, "data")
        training_df.to_parquet(
            os.path.join(data_dir, "training_dataset.parquet"),
            index=False
        )
        print("Saved complete training dataset to data/training_dataset.parquet")
        
        return training_df

def main():
    """Main pipeline to process fraud data and populate Feast feature store"""
    
    # Generate synthetic data
    print("Generating synthetic fraud data...")
    df = pd.read_parquet("./features/data/transactions.parquet")
    print(f"Generated {len(df)} transactions")
    
    # Initialize feature engineer
    print("Initializing Feast feature store...")
    fe = FraudFeatureEngineer()
    
    # Compute all feature types
    print("Computing user features...")
    user_features = fe.compute_user_features(df)
    
    print("Computing merchant features...")
    merchant_features = fe.compute_merchant_features(df)
    
    print("Computing real-time features...")
    real_time_features = fe.compute_real_time_features(df)
    
    print("Preparing transaction features...")
    transaction_features = fe.prepare_transaction_features(df)
    
    # Save features locally
    print("Saving features locally...")
    fe.save_features_locally(
        transaction_features=transaction_features,
        user_features=user_features,
        merchant_features=merchant_features,
        real_time_features=real_time_features
    )
    
    # Create training dataset
    print("Creating complete training dataset...")
    training_df = fe.create_training_dataset()
    
    # Print summary statistics
    print("\n=== FEATURE ENGINEERING SUMMARY ===")
    print(f"Total transactions: {len(transaction_features)}")
    print(f"Total users: {len(user_features)}")
    print(f"Total merchants: {len(merchant_features)}")
    print(f"Fraud rate: {df['is_fraud'].mean():.3f}")
    print(f"Training dataset shape: {training_df.shape}")
    print("\nFeature files saved in ./data/ directory:")
    print("- transaction_features.parquet")
    print("- user_features.parquet") 
    print("- merchant_features.parquet")
    print("- real_time_features.parquet")
    print("- training_dataset.parquet")
    print("- feature_summary.parquet")
    
    return fe, training_df

if __name__ == "__main__":
    # feature_engineer, training_dataset = main()

    #push features to online store
    fe = FraudFeatureEngineer(feature_store_path="./features")
    transaction_df = pd.read_parquet("./features/data/transaction_features.parquet")
    user_df = pd.read_parquet("./features/data/user_features.parquet")
    merchant_df = pd.read_parquet("./features/data/merchant_features.parquet")
    real_time_df = pd.read_parquet("./features/data/real_time_features.parquet")
    fe.push_features_to_feast_online_store(transaction_df, user_df, merchant_df, real_time_df)