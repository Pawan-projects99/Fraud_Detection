from datetime import timedelta
from feast import Entity, FeatureView, ValueType, Field
from feast.types import Float32, Int64
from feast.infra.offline_stores.file_source import FileSource

#Defining the Entities
user = Entity(name = "user_id", join_keys = ["user_id"])
merchant = Entity(name = "merchant_id", join_keys = ["merchant_id"])

# Defining the datasource for features
user_source= FileSource(
    path = "./data/user_features.parquet",
    event_timestamp_column = "event_timestamp"
)
merchant_source= FileSource(
    path = "./data/merchant_features.parquet",
    event_timestamp_column = "event_timestamp"
)
#User_behaviour features
user_features = FeatureView(
    name = "user_behaviour",
    entities = [user],
    ttl = timedelta(days = 60),
    schema = [
        Field(name = "avg_transaction", dtype = Float32),
        Field(name = "transaction_count_30d", dtype = Int64),
        Field(name = "decline_rate_60d", dtype = Float32)
    ],
    source = user_source
)
merchant_features = FeatureView(
    name = "merchant_risk_profile",
    entities = [merchant],
    ttl = timedelta(days = 90),
    schema = [
        Field(name = "fraud_rate", dtype = Float32),
        Field(name = "avg_transaction_value", dtype = Float32)
    ],
    source = merchant_source
)
