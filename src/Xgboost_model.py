import xgboost as xgb
import pandas as pd
import numpy as np
from feast import FeatureStore
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score, precision_score, recall_score, f1_score, average_precision_score, precision_recall_curve
import matplotlib.pyplot as plt
import shap
import joblib
from imblearn.over_sampling import SMOTENC
store = FeatureStore(repo_path="./features")

def get_training_data():
    df = pd.read_parquet("./features/data/transactions.parquet")
    print("loaded_df_shape :",df.shape)
    entity_df = df.rename(columns={"timestamp": "event_timestamp"})
    entity_df = entity_df[["transaction_id", "user_id", "merchant_id","location", "amount", "event_timestamp", "is_fraud"]]
    print(entity_df["event_timestamp"].min(), entity_df["event_timestamp"].max())
    print("entity_df_shape :", entity_df.shape)
    feature_df = store.get_historical_features(
        entity_df= entity_df,
        features=[
            "user_behaviour:avg_transaction",
            "user_behaviour:transaction_count_30d",
            "user_behaviour:decline_rate_60d",
            "merchant_risk_profile:fraud_rate",
            "merchant_risk_profile:avg_transaction_value"
        ]
    ).to_df()
    print(feature_df.columns)
    print(feature_df.shape)
    print(feature_df.head(10))
    return feature_df

def preprocessing(df):
    df["decline_rate_60d"].fillna(0, inplace=True)
    df["fraud_rate"].fillna(df["fraud_rate"].median(), inplace=True)
    df["amount_vs_avg"] = df["amount"] / df["avg_transaction"]
    df["amount_vs_merchant_avg"] = df["amount"] / df["avg_transaction_value"]
    
    df["transactions_1hr"] = 0
    for user_id, group in df.groupby("user_id"):
        times = group["event_timestamp"]
        counts = []
        for i in range(len(times)):
            current_time = times.iloc[i]
            past_window = times[(times >= current_time - pd.Timedelta(hours=1)) & (times <= current_time)]
            counts.append(len(past_window))
        df.loc[group.index, "transactions_1hr"] = counts

    df["amount_deviation"] = df["amount"]-df["avg_transaction"]/df["avg_transaction"]
    df["amount_merchant_risk"]= df["amount"] * df["fraud_rate"]
    df["hour"] = pd.to_datetime(df["event_timestamp"]).dt.hour
    df["is_night"] = ((df["hour"] >= 22) | (df["hour"] <= 5)).astype(int)
    df = df.sort_values(by=["user_id", "event_timestamp"])
    df["prev_location"] = df.groupby("user_id")["location"].shift(1)
    df["is_location_changed"] = (df["location"] != df["prev_location"]).astype(int)
    df = df.drop(columns = ["prev_location"])
    features = [
        "amount",
        "avg_transaction",
        "transaction_count_30d",
        "decline_rate_60d",
        "fraud_rate",
        "avg_transaction_value",
        "amount_vs_avg",
        "amount_vs_merchant_avg",
        "is_location_changed",
        "transactions_1hr",
        "amount_deviation",
        "amount_merchant_risk",
        "is_night"
    ]
    return df[features], df["is_fraud"]

def train_model(X, y):
    cat_features = [X.columns.get_loc(col) for col in ["is_night", "is_location_changed"]]
    smote = SMOTENC(categorical_features = cat_features, sampling_strategy=0.25, random_state = 42)
    X,y = smote.fit_resample(X,y)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, train_size=0.8, shuffle=False)
    
    pos_class_weight = (len(y) - np.sum(y)) / np.sum(y)


    model = xgb.XGBClassifier(
        objective="binary:logistic",
        eta=0.1,
        max_depth=6,
        grow_policy="lossguide",
        sampling_method="uniform",
        subsample=0.8,
        n_estimators=1000,
        colsample_bytree=0.7,
        scale_pos_weight=pos_class_weight,
        max_delta_step=1,
        eval_metric=["aucpr", "map"],
        early_stopping_rounds = 50
    )

    model.fit(
        X_train, y_train,
        eval_set=[(X_test, y_test)],
        verbose=10
    )

    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    optimal_threshold = 0.1337
    y_pred = (y_proba >= optimal_threshold).astype(int)
    precisions, recalls, thresholds = precision_recall_curve(y_test, y_proba)

    # F1 calculation
    f1_scores = 2 * (precisions * recalls) / (precisions + recalls + 1e-8)
    optimal_idx = np.argmax(f1_scores)
    optimal_threshold = thresholds[optimal_idx]

    print(classification_report(y_test, y_pred))
    print(f"ROC AUC: {roc_auc_score(y_test, y_proba):.4f}")
    print(f"Average Precision Score: {average_precision_score(y_test, y_proba):.4f}")
    print("Precision:", precision_score(y_test, y_pred))
    print("Recall:", recall_score(y_test, y_pred))
    print("F1 Score:", f1_score(y_test, y_pred))
    print(f"Optimal Threshold: {optimal_threshold:.4f}, F1 Score: {f1_scores[optimal_idx]:.4f}")

    # Feature importance
    sorted_idx = model.feature_importances_.argsort()
    plt.barh(X.columns[sorted_idx], model.feature_importances_[sorted_idx], color="blue")
    plt.xlabel("XGBoost Feature Importance")
    plt.tight_layout()
    plt.savefig("./reports/feature_importance.png")
    plt.close()

    return model

def interpret_model(model, X):
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)
    shap.summary_plot(shap_values, X, plot_type="bar")

    sample_idx = 42
    shap.force_plot(
        explainer.expected_value,
        shap_values[sample_idx],
        X.iloc[sample_idx],
        matplotlib=True
    )
    plt.savefig("./reports/shap_summary.png")

if __name__ == "__main__":
    print("Retrieving training data from feature store")
    training_data = get_training_data()

    print("Preprocessing the data")
    X, y = preprocessing(training_data)
    print(f"features: {X.shape}, is_fraud: {y.shape}")
    print(f"features: {X.describe}, is_fraud: {y.describe}")
    print("is_fraud_sum: ", y[y==1].count())

    print("Training the model")
    actual_model = train_model(X, y)

    print("Interpreting the model")
    interpret_model(actual_model, X.sample(1000))

    print("Saving the model to ./models")
    joblib.dump(actual_model, "./models/xgboost_v1.pkl")
    actual_model.get_booster().save_model("./models/xgboost.json")