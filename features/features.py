# from datetime import timedelta
# from feast import Entity, FeatureView, ValueType, Field
# from feast.types import Float32, Int64
# from feast.infra.offline_stores.file_source import FileSource

# #Defining the Entities
# user = Entity(name = "user_id", join_keys = ["user_id"])
# merchant = Entity(name = "merchant_id", join_keys = ["merchant_id"])

# # Defining the datasource for features
# user_source= FileSource(
#     path = "./data/user_features.parquet",
#     event_timestamp_column = "event_timestamp"
# )
# merchant_source= FileSource(
#     path = "./data/merchant_features.parquet",
#     event_timestamp_column = "event_timestamp"
# )
# #User_behaviour features
# user_features = FeatureView(
#     name = "user_behaviour",
#     entities = [user],
#     ttl = timedelta(days = 60),
#     schema = [
#         Field(name = "avg_transaction", dtype = Float32),
#         Field(name = "transaction_count_30d", dtype = Int64),
#         Field(name = "decline_rate_60d", dtype = Float32)
#     ],
#     source = user_source
# )
# merchant_features = FeatureView(
#     name = "merchant_risk_profile",
#     entities = [merchant],
#     ttl = timedelta(days = 90),
#     schema = [
#         Field(name = "fraud_rate", dtype = Float32),
#         Field(name = "avg_transaction_value", dtype = Float32)
#     ],
#     source = merchant_source
# )
# features.py
from feast import Entity, FeatureView, Field, PushSource, ValueType
from feast.types import Float32, Float64, Int32, Int64, String, UnixTimestamp
from datetime import timedelta
from feast.infra.offline_stores.file_source import FileSource

# Define entities
user_entity = Entity(
    name="user_id",
    description="User identifier",
    value_type=ValueType.STRING,
)

merchant_entity = Entity(
    name="merchant_id", 
    description="Merchant identifier",
    value_type=ValueType.STRING,
)

transaction_entity = Entity(
    name="transaction_id",
    description="Transaction identifier", 
    value_type=ValueType.STRING,
)

# Define data sources
transaction_source = FileSource(
    path = "./data/transaction_features.parquet",
    event_timestamp_column = "timestamp"
)

user_features_source = FileSource(
    path = "./data/user_features.parquet",
    event_timestamp_column = "timestamp"
)

merchant_features_source = FileSource(
    path = "./data/merchant_features.parquet",
    event_timestamp_column = "timestamp"
)
# real_time_feature_source = FileSource(
#     path = "./data/real_time_features.parquet",
#     event_timestamp_column = "event_timestamp"
# )

# Transaction-level features
transaction_features = FeatureView(
    name="transaction_features",
    description="Core transaction features",
    entities=[transaction_entity],
    ttl=timedelta(days=30),
    schema=[
        Field(name="amount", dtype=Float64, description="Transaction amount"),
        Field(name="hour", dtype=Int32, description="Hour of transaction (0-23)"),
        Field(name="payment_method", dtype=Int32, description="Payment method type (0=credit, 1=debit, 2=cash, 3=other)"),
        Field(name="device_type", dtype=Int32, description="Device type (0=desktop, 1=mobile, 2=tablet)"),
        Field(name="location_risk", dtype=Int32, description="Location risk level (0=low, 1=medium, 2=high)"),
        Field(name="is_fraud", dtype=Int32, description="Fraud label (0=normal, 1=fraud)"),
    ],
    source=transaction_source,
)

# User-level features (account and historical patterns)
user_features = FeatureView(
    name="user_features",
    description="User account and behavioral features",
    entities=[user_entity],
    ttl=timedelta(days=90),
    schema=[
        Field(name="account_balance", dtype=Float64, description="Current account balance"),
        Field(name="days_since_last_transaction", dtype=Float64, description="Days since last transaction"),
        Field(name="velocity_1h", dtype=Int32, description="Number of transactions in last hour"),
        Field(name="velocity_24h", dtype=Int32, description="Number of transactions in last 24 hours"),
        Field(name="avg_transaction_amount_7d", dtype=Float64, description="Average transaction amount in last 7 days"),
        Field(name="avg_transaction_amount_30d", dtype=Float64, description="Average transaction amount in last 30 days"),
        Field(name="transaction_count_7d", dtype=Int32, description="Transaction count in last 7 days"),
        Field(name="transaction_count_30d", dtype=Int32, description="Transaction count in last 30 days"),
        Field(name="unique_merchants_7d", dtype=Int32, description="Unique merchants in last 7 days"),
        Field(name="unique_merchants_30d", dtype=Int32, description="Unique merchants in last 30 days"),
        Field(name="night_transaction_ratio_7d", dtype=Float64, description="Ratio of night transactions (10PM-6AM) in last 7 days"),
        Field(name="weekend_transaction_ratio_7d", dtype=Float64, description="Ratio of weekend transactions in last 7 days"),
        Field(name="high_amount_transaction_ratio_7d", dtype=Float64, description="Ratio of high amount transactions (>95th percentile) in last 7 days"),
        Field(name="cross_border_transaction_ratio_7d", dtype=Float64, description="Ratio of cross-border transactions in last 7 days"),
        Field(name="failed_transaction_count_7d", dtype=Int32, description="Number of failed transactions in last 7 days"),
        Field(name="account_age_days", dtype=Int32, description="Account age in days"),
        Field(name="user_risk_score", dtype=Float64, description="User risk score based on historical behavior"),
    ],
    source=user_features_source,
)

# Merchant-level features
merchant_features = FeatureView(
    name="merchant_features", 
    description="Merchant characteristics and risk indicators",
    entities=[merchant_entity],
    ttl=timedelta(days=90),
    schema=[
        Field(name="merchant_category", dtype=Int32, description="Merchant category code"),
        Field(name="merchant_risk_score", dtype=Float64, description="Merchant risk score"),
        Field(name="merchant_fraud_rate_7d", dtype=Float64, description="Merchant fraud rate in last 7 days"),
        Field(name="merchant_fraud_rate_30d", dtype=Float64, description="Merchant fraud rate in last 30 days"),
        Field(name="merchant_transaction_count_7d", dtype=Int32, description="Merchant transaction count in last 7 days"),
        Field(name="merchant_avg_amount_7d", dtype=Float64, description="Merchant average transaction amount in last 7 days"),
        Field(name="merchant_unique_users_7d", dtype=Int32, description="Unique users transacting with merchant in last 7 days"),
        Field(name="merchant_chargeback_rate_30d", dtype=Float64, description="Merchant chargeback rate in last 30 days"),
        Field(name="merchant_operating_hours_match", dtype=Float64, description="How well transaction time matches merchant operating hours"),
        Field(name="merchant_reputation_score", dtype=Float64, description="Merchant reputation score"),
        Field(name="merchant_age_days", dtype=Int32, description="Merchant age in days"),
    ],
    source=merchant_features_source,
)

# Real-time aggregation features (computed on-the-fly)
# real_time_features = FeatureView(
#     name="real_time_features",
#     description="Real-time computed features for immediate fraud detection",
#     entities=[user_entity],
#     ttl=timedelta(hours=1),
#     schema=[
#         Field(name="transactions_last_5min", dtype=Int32, description="Transaction count in last 5 minutes"),
#         Field(name="transactions_last_15min", dtype=Int32, description="Transaction count in last 15 minutes"),
#         Field(name="amount_last_5min", dtype=Float64, description="Total amount in last 5 minutes"),
#         Field(name="amount_last_15min", dtype=Float64, description="Total amount in last 15 minutes"),
#         Field(name="unique_merchants_last_hour", dtype=Int32, description="Unique merchants in last hour"),
#         Field(name="location_changes_last_hour", dtype=Int32, description="Location changes in last hour"),
#         Field(name="device_changes_last_hour", dtype=Int32, description="Device changes in last hour"),
#         Field(name="velocity_spike_indicator", dtype=Float64, description="Velocity spike compared to user's normal behavior"),
#         Field(name="amount_spike_indicator", dtype=Float64, description="Amount spike compared to user's normal behavior"),
#         Field(name="time_since_last_transaction_seconds", dtype=Int32, description="Seconds since last transaction"),
#         Field(name="consecutive_failed_attempts", dtype=Int32, description="Consecutive failed transaction attempts"),
#     ],
#     online = True,
#     source= real_time_feature_source,
# )

# Feature service for model serving
from feast import FeatureService

fraud_detection_service = FeatureService(
    name="fraud_detection_v1",
    description="Feature service for real-time fraud detection",
    features=[
        transaction_features,
        user_features,
        merchant_features
    ],
)

# On-demand feature views for derived features
from feast import RequestSource # request source is the for at that moment features
from feast.on_demand_feature_view import on_demand_feature_view

request_source = RequestSource(
    name="request_source",
    schema=[
        Field(name="current_amount", dtype=Float64),
        Field(name="current_hour", dtype=Int32),
        Field(name="current_location_risk", dtype=Int32),
        
    ],
)

@on_demand_feature_view(
    sources=[
        request_source,
        user_features[["avg_transaction_amount_7d", "avg_transaction_amount_30d"]],
        merchant_features[["merchant_avg_amount_7d"]],
    ],
    schema=[
        Field(name="amount_vs_user_avg_7d", dtype=Float64),
        Field(name="amount_vs_user_avg_30d", dtype=Float64), 
        Field(name="amount_vs_merchant_avg_7d", dtype=Float64),
        Field(name="is_night_transaction", dtype=Int64),
        Field(name="is_weekend", dtype=Int64),
        Field(name="location_risk_score", dtype=Float64),
    ],
)
def derived_features(inputs):
    """On-demand feature transformations"""
    import pandas as pd
    from datetime import datetime
    
    output = pd.DataFrame()
    
    # Amount comparison features
    output["amount_vs_user_avg_7d"] = inputs["current_amount"] / (inputs["avg_transaction_amount_7d"] + 1e-8)
    output["amount_vs_user_avg_30d"] = inputs["current_amount"] / (inputs["avg_transaction_amount_30d"] + 1e-8)
    output["amount_vs_merchant_avg_7d"] = inputs["current_amount"] / (inputs["merchant_avg_amount_7d"] + 1e-8)
    
    # Time-based features
    output["is_night_transaction"] = ((inputs["current_hour"] >= 22) | (inputs["current_hour"] <= 6)).astype(int)
    output["is_weekend"] = 0  # Would need timestamp to compute properly
    
    # Location risk scoring
    risk_multipliers = {0: 1.0, 1: 2.5, 2: 5.0}
    output["location_risk_score"] = inputs["current_location_risk"].map(risk_multipliers)
    
    return output

