# import xgboost as xgb
# import pandas as pd
# import numpy as np
# from feast import FeatureStore
# from sklearn.model_selection import train_test_split
# from sklearn.metrics import classification_report, roc_auc_score, precision_score, recall_score, f1_score, average_precision_score, precision_recall_curve
# import matplotlib.pyplot as plt
# import shap
# import joblib
# from imblearn.over_sampling import SMOTENC

# store = FeatureStore(repo_path="./features")

# def get_training_data():
#     df = pd.read_parquet("./features/data/transactions.parquet")
#     print("loaded_df_shape :",df.shape)
#     entity_df = df.rename(columns={"timestamp": "event_timestamp"})
#     entity_df = entity_df[["transaction_id", "user_id", "merchant_id","location", "amount", "event_timestamp", "is_fraud"]]
#     print(entity_df["event_timestamp"].min(), entity_df["event_timestamp"].max())
#     print("entity_df_shape :", entity_df.shape)
#     feature_df = store.get_historical_features(
#         entity_df= entity_df,
#         features=[
#             "user_behaviour:avg_transaction",
#             "user_behaviour:transaction_count_30d",
#             "user_behaviour:decline_rate_60d",
#             "merchant_risk_profile:fraud_rate",
#             "merchant_risk_profile:avg_transaction_value"
#         ]
#     ).to_df()
#     print(feature_df.columns)
#     print(feature_df.shape)
#     print(feature_df.head(10))
#     return feature_df

# def preprocessing(df):
#     df["decline_rate_60d"].fillna(0, inplace=True)
#     df["fraud_rate"].fillna(df["fraud_rate"].median(), inplace=True)
#     df["amount_vs_avg"] = df["amount"] / df["avg_transaction"]
#     df["amount_vs_merchant_avg"] = df["amount"] / df["avg_transaction_value"]
    
#     df["transactions_1hr"] = 0
#     for user_id, group in df.groupby("user_id"):
#         times = group["event_timestamp"]
#         counts = []
#         for i in range(len(times)):
#             current_time = times.iloc[i]
#             past_window = times[(times >= current_time - pd.Timedelta(hours=1)) & (times <= current_time)]
#             counts.append(len(past_window))
#         df.loc[group.index, "transactions_1hr"] = counts

#     df["amount_deviation"] = df["amount"]-df["avg_transaction"]/df["avg_transaction"]
#     df["amount_merchant_risk"]= df["amount"] * df["fraud_rate"]
#     df["hour"] = pd.to_datetime(df["event_timestamp"]).dt.hour
#     df["is_night"] = ((df["hour"] >= 22) | (df["hour"] <= 5)).astype(int)
#     df = df.sort_values(by=["user_id", "event_timestamp"])
#     df["prev_location"] = df.groupby("user_id")["location"].shift(1)
#     df["is_location_changed"] = (df["location"] != df["prev_location"]).astype(int)
#     df = df.drop(columns = ["prev_location"])
#     features = [
#         "amount",
#         "avg_transaction",
#         "transaction_count_30d",
#         "decline_rate_60d",
#         "fraud_rate",
#         "avg_transaction_value",
#         "amount_vs_avg",
#         "amount_vs_merchant_avg",
#         "is_location_changed",
#         "transactions_1hr",
#         "amount_deviation",
#         "amount_merchant_risk",
#         "is_night"
#     ]
#     return df[features], df["is_fraud"]

# def train_model(X, y):
#     cat_features = [X.columns.get_loc(col) for col in ["is_night", "is_location_changed"]]
#     smote = SMOTENC(categorical_features = cat_features, sampling_strategy=0.25, random_state = 42)
#     X,y = smote.fit_resample(X,y)
#     X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, train_size=0.8, shuffle=False)
    
#     pos_class_weight = (len(y) - np.sum(y)) / np.sum(y)


#     model = xgb.XGBClassifier(
#         objective="binary:logistic",
#         eta=0.1,
#         max_depth=6,
#         grow_policy="lossguide",
#         sampling_method="uniform",
#         subsample=0.8,
#         n_estimators=1000,
#         colsample_bytree=0.7,
#         scale_pos_weight=pos_class_weight,
#         max_delta_step=1,
#         eval_metric=["aucpr", "map"],
#         early_stopping_rounds = 50
#     )

#     model.fit(
#         X_train, y_train,
#         eval_set=[(X_test, y_test)],
#         verbose=10
#     )

#     y_pred = model.predict(X_test)
#     y_proba = model.predict_proba(X_test)[:, 1]
#     optimal_threshold = 0.1337
#     y_pred = (y_proba >= optimal_threshold).astype(int)
#     precisions, recalls, thresholds = precision_recall_curve(y_test, y_proba)

#     # F1 calculation
#     f1_scores = 2 * (precisions * recalls) / (precisions + recalls + 1e-8)
#     optimal_idx = np.argmax(f1_scores)
#     optimal_threshold = thresholds[optimal_idx]

#     print(classification_report(y_test, y_pred))
#     print(f"ROC AUC: {roc_auc_score(y_test, y_proba):.4f}")
#     print(f"Average Precision Score: {average_precision_score(y_test, y_proba):.4f}")
#     print("Precision:", precision_score(y_test, y_pred))
#     print("Recall:", recall_score(y_test, y_pred))
#     print("F1 Score:", f1_score(y_test, y_pred))
#     print(f"Optimal Threshold: {optimal_threshold:.4f}, F1 Score: {f1_scores[optimal_idx]:.4f}")

#     # Feature importance
#     sorted_idx = model.feature_importances_.argsort()
#     plt.barh(X.columns[sorted_idx], model.feature_importances_[sorted_idx], color="blue")
#     plt.xlabel("XGBoost Feature Importance")
#     plt.tight_layout()
#     plt.savefig("./reports/feature_importance.png")
#     plt.close()

#     return model

# def interpret_model(model, X):
#     explainer = shap.TreeExplainer(model)
#     shap_values = explainer.shap_values(X)
#     shap.summary_plot(shap_values, X, plot_type="bar")

#     sample_idx = 42
#     shap.force_plot(
#         explainer.expected_value,
#         shap_values[sample_idx],
#         X.iloc[sample_idx],
#         matplotlib=True
#     )
#     plt.savefig("./reports/shap_summary.png")

# if __name__ == "__main__":
#     print("Retrieving training data from feature store")
#     training_data = get_training_data()

#     print("Preprocessing the data")
#     X, y = preprocessing(training_data)
#     print(f"features: {X.shape}, is_fraud: {y.shape}")
#     print(f"features: {X.describe}, is_fraud: {y.describe}")
#     print("is_fraud_sum: ", y[y==1].count())

#     print("Training the model")
#     actual_model = train_model(X, y)

#     print("Interpreting the model")
#     interpret_model(actual_model, X.sample(1000))

#     print("Saving the model to ./models")
#     joblib.dump(actual_model, "./models/xgboost_v1.pkl")
#     actual_model.get_booster().save_model("./models/xgboost.json")








import xgboost as xgb
import pandas as pd
import numpy as np
from feast import FeatureStore
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (classification_report, roc_auc_score, precision_score, 
                           recall_score, f1_score, average_precision_score, 
                           precision_recall_curve, roc_curve, confusion_matrix)
import matplotlib.pyplot as plt
import seaborn as sns
import shap
import joblib
from imblearn.over_sampling import SMOTENC
import warnings
import os
warnings.filterwarnings('ignore')

# Create directories if they don't exist
os.makedirs("./models", exist_ok=True)
os.makedirs("./reports", exist_ok=True)

class FraudXGBoostModel:
    """XGBoost model for fraud detection using comprehensive feature set"""
    
    def __init__(self, feature_store_path="./features"):
        """Initialize the model with feature store connection"""
        self.fs = FeatureStore(repo_path=feature_store_path)
        self.model = None
        self.feature_columns = None
        self.scaler = None
        self.optimal_threshold = 0.5
        
    def load_training_data(self, use_feast=False):
        """Load training data either from Feast or directly from parquet"""
        
        # if use_feast:
        #     # Load from Feast feature store (if you have it set up)
        #     try:
        #         df = pd.read_parquet("./features/data/training_dataset.parquet")
        #         print("Loaded df shape:", df.shape)
                
        #         entity_df = df.rename(columns={"timestamp": "event_timestamp"})
        #         entity_df = entity_df[["transaction_id", "user_id", "merchant_id", "amount", "event_timestamp", "is_fraud"]]
                
        #         feature_df = self.fs.get_historical_features(
        #             entity_df=entity_df,
        #             features=[
        #                 "transaction_features:amount",
        #                 "transaction_features:hour",
        #                 "transaction_features:payment_method",
        #                 "transaction_features:device_type",
        #                 "transaction_features:location_risk",
        #                 "user_features:account_balance",
        #                 "user_features:velocity_1h",
        #                 "user_features:velocity_24h",
        #                 "user_features:user_risk_score",
        #                 "merchant_features:merchant_risk_score",
        #                 "merchant_features:merchant_fraud_rate_7d",
        #                 "real_time_features:transactions_last_5min",
        #                 "real_time_features:velocity_spike_indicator",
        #             ]
        #         ).to_df()
                
        #         return feature_df
                
        #     except Exception as e:
        #         print(f"Error loading from Feast: {e}")
        #         print("Falling back to direct parquet loading...")
        
        # Load directly from training dataset parquet
        try:
            df = pd.read_parquet("./features/data/training_dataset.parquet")
            print(f"Loaded training dataset shape: {df.shape}")
            return df
        except FileNotFoundError:
            print("Training dataset not found. Please run feature_engineering.py first.")
            return None
    
    def preprocessing(self, df):
        """Comprehensive preprocessing for the new feature set"""
        
        df = df.copy()
        
        # Handle missing values
        numerical_columns = df.select_dtypes(include=[np.number]).columns
        for col in numerical_columns:
            if col != 'is_fraud':
                df[col].fillna(df[col].median(), inplace=True)
        
        # Create derived features
        print("Creating derived features...")
        
        # Amount-based features
        if 'amount' in df.columns and 'avg_transaction_amount_7d' in df.columns:
            df['amount_vs_user_avg_7d'] = df['amount'] / (df['avg_transaction_amount_7d'] + 1e-8)
            df['amount_vs_user_avg_30d'] = df['amount'] / (df['avg_transaction_amount_30d'] + 1e-8)
            df['amount_deviation_7d'] = (df['amount'] - df['avg_transaction_amount_7d']) / (df['avg_transaction_amount_7d'] + 1e-8)
        
        # Risk-based interactions
        if 'merchant_risk_score' in df.columns and 'amount' in df.columns:
            df['amount_merchant_risk'] = df['amount'] * df['merchant_risk_score']
        
        if 'user_risk_score' in df.columns and 'merchant_risk_score' in df.columns:
            df['combined_risk_score'] = df['user_risk_score'] * df['merchant_risk_score']
        
        # Time-based features
        if 'hour' in df.columns:
            df['is_night'] = ((df['hour'] >= 22) | (df['hour'] <= 6)).astype(int)
            df['is_business_hours'] = ((df['hour'] >= 9) & (df['hour'] <= 17)).astype(int)
            df['is_weekend_hours'] = ((df['hour'] >= 18) | (df['hour'] <= 8)).astype(int)
        
        # Velocity features
        if 'velocity_1h' in df.columns and 'velocity_24h' in df.columns:
            df['velocity_ratio'] = df['velocity_1h'] / (df['velocity_24h'] + 1e-8)
            df['velocity_spike'] = (df['velocity_1h'] > df['velocity_24h'] * 0.5).astype(int)
        
        # Account balance features
        if 'account_balance' in df.columns and 'amount' in df.columns:
            df['amount_vs_balance'] = df['amount'] / (df['account_balance'] + 1e-8)
            df['high_amount_vs_balance'] = (df['amount_vs_balance'] > 0.1).astype(int)
        
        # Merchant-specific features
        if 'merchant_fraud_rate_7d' in df.columns:
            df['high_risk_merchant'] = (df['merchant_fraud_rate_7d'] > 0.05).astype(int)
        
        # Location and device risk
        if 'location_risk' in df.columns:
            df['high_location_risk'] = (df['location_risk'] >= 2).astype(int)
        
        if 'device_type' in df.columns:
            df['mobile_device'] = (df['device_type'] == 1).astype(int)
        
        # Transaction frequency features
        if 'transaction_count_7d' in df.columns and 'transaction_count_30d' in df.columns:
            df['recent_activity_ratio'] = df['transaction_count_7d'] / (df['transaction_count_30d'] + 1e-8)
        
        # Define feature categories for better organization
        core_features = [
            'amount', 'hour', 'payment_method', 'device_type', 'location_risk'
        ]
        
        user_features = [
            'account_balance', 'velocity_1h', 'velocity_24h', 'user_risk_score',
            'avg_transaction_amount_7d', 'avg_transaction_amount_30d',
            'transaction_count_7d', 'transaction_count_30d',
            'unique_merchants_7d', 'unique_merchants_30d',
            'night_transaction_ratio_7d', 'high_amount_transaction_ratio_7d'
        ]
        
        merchant_features = [
            'merchant_risk_score', 'merchant_fraud_rate_7d', 'merchant_fraud_rate_30d',
            'merchant_transaction_count_7d', 'merchant_avg_amount_7d',
            'merchant_reputation_score'
        ]
        
        real_time_features = [
            'transactions_last_5min', 'transactions_last_15min',
            'amount_last_5min', 'amount_last_15min',
            'velocity_spike_indicator', 'amount_spike_indicator',
            'time_since_last_transaction_seconds'
        ]
        
        derived_features = [
            'amount_vs_user_avg_7d', 'amount_vs_user_avg_30d', 'amount_deviation_7d',
            'amount_merchant_risk', 'combined_risk_score', 'is_night',
            'is_business_hours', 'velocity_ratio', 'velocity_spike',
            'amount_vs_balance', 'high_amount_vs_balance', 'high_risk_merchant',
            'high_location_risk', 'mobile_device', 'recent_activity_ratio'
        ]
        
        # Combine all features
        all_features = core_features + user_features + merchant_features + real_time_features + derived_features
        
        # Select features that exist in the dataframe
        available_features = [f for f in all_features if f in df.columns]
        
        print(f"Using {len(available_features)} features out of {len(all_features)} defined features")
        
        # Final feature selection and target extraction
        X = df[available_features].copy()
        y = df['is_fraud'].copy()
        
        # Handle any remaining missing values
        X.fillna(X.median(), inplace=True)
        
        print(f"Final feature matrix shape: {X.shape}")
        print(f"Features used: {list(X.columns)}")
        
        return X, y
    
    def train_model(self, X, y, use_smote=True, test_size=0.2):
        """Train XGBoost model with enhanced configuration"""
        
        print(f"Training data shape: {X.shape}")
        print(f"Fraud ratio: {y.mean():.4f}")
        
        # Identify categorical features
        categorical_features = []
        categorical_columns = ['payment_method', 'device_type', 'location_risk', 'is_night', 
                              'is_business_hours', 'velocity_spike', 'high_amount_vs_balance',
                              'high_risk_merchant', 'high_location_risk', 'mobile_device']
        
        for col in categorical_columns:
            if col in X.columns:
                categorical_features.append(X.columns.get_loc(col))
        
        # Apply SMOTE if requested
        if use_smote and categorical_features:
            print("Applying SMOTENC for class balancing...")
            smote = SMOTENC(categorical_features=categorical_features, 
                          sampling_strategy=0.3, random_state=42)
            X_resampled, y_resampled = smote.fit_resample(X, y)
            print(f"After SMOTE - Shape: {X_resampled.shape}, Fraud ratio: {y_resampled.mean():.4f}")
        else:
            X_resampled, y_resampled = X, y
        
        # Train-test split (stratified to maintain class distribution)
        X_train, X_test, y_train, y_test = train_test_split(
            X_resampled, y_resampled, 
            test_size=test_size, 
            stratify=y_resampled,
            random_state=42
        )
        
        # Calculate class weight
        pos_class_weight = (len(y_train) - np.sum(y_train)) / np.sum(y_train)
        print(f"Positive class weight: {pos_class_weight:.2f}")
        
        # Define XGBoost parameters optimized for fraud detection
        params = {
            'objective': 'binary:logistic',
            'eval_metric': ['aucpr', 'auc'],
            'eta': 0.05,  # Lower learning rate for better generalization
            'max_depth': 6,
            'min_child_weight': 5,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'colsample_bylevel': 0.8,
            'scale_pos_weight': pos_class_weight,
            'max_delta_step': 1,  # Helps with imbalanced datasets
            'gamma': 0.1,  # Regularization
            'reg_alpha': 0.1,  # L1 regularization
            'reg_lambda': 1.0,  # L2 regularization
            'n_estimators': 1000,
            'random_state': 42,
            'n_jobs': -1,
            'tree_method': 'hist',  # Faster training
            'grow_policy': 'lossguide',
            'early_stopping_rounds':50
        }
        
        # Initialize model
        self.model = xgb.XGBClassifier(**params)
        
        # Train with early stopping
        print("Training XGBoost model...")
        self.model.fit(
            X_train, y_train,
            eval_set=[(X_train, y_train), (X_test, y_test)],
            verbose=50
        )
        
        # Store feature columns for later use
        self.feature_columns = X.columns.tolist()
        
        # Predictions and probability scores
        y_pred_proba = self.model.predict_proba(X_test)[:, 1]
        
        # Find optimal threshold using F1 score
        precisions, recalls, thresholds = precision_recall_curve(y_test, y_pred_proba)
        f1_scores = 2 * (precisions * recalls) / (precisions + recalls + 1e-8)
        optimal_idx = np.argmax(f1_scores)
        self.optimal_threshold = thresholds[optimal_idx]
        
        # Make predictions with optimal threshold
        y_pred = (y_pred_proba >= self.optimal_threshold).astype(int)
        
        # Print evaluation metrics
        print("\n=== MODEL EVALUATION ===")
        print(f"Optimal Threshold: {self.optimal_threshold:.4f}")
        print(f"F1 Score: {f1_scores[optimal_idx]:.4f}")
        print(classification_report(y_test, y_pred))
        print(f"ROC AUC: {roc_auc_score(y_test, y_pred_proba):.4f}")
        print(f"Average Precision Score: {average_precision_score(y_test, y_pred_proba):.4f}")
        
        # Plot evaluation metrics
        self._plot_evaluation_metrics(y_test, y_pred_proba, y_pred)
        self._plot_feature_importance(X)
        
        return X_test, y_test, y_pred_proba
    
    def _plot_evaluation_metrics(self, y_test, y_pred_proba, y_pred):
        """Plot ROC curve, PR curve, and confusion matrix"""
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # ROC Curve
        fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
        roc_auc = roc_auc_score(y_test, y_pred_proba)
        
        axes[0, 0].plot(fpr, tpr, label=f'ROC Curve (AUC = {roc_auc:.4f})')
        axes[0, 0].plot([0, 1], [0, 1], 'k--')
        axes[0, 0].set_xlabel('False Positive Rate')
        axes[0, 0].set_ylabel('True Positive Rate')
        axes[0, 0].set_title('ROC Curve')
        axes[0, 0].legend()
        axes[0, 0].grid(True)
        
        # Precision-Recall Curve
        precision, recall, _ = precision_recall_curve(y_test, y_pred_proba)
        avg_precision = average_precision_score(y_test, y_pred_proba)
        
        axes[0, 1].plot(recall, precision, label=f'PR Curve (AP = {avg_precision:.4f})')
        axes[0, 1].set_xlabel('Recall')
        axes[0, 1].set_ylabel('Precision')
        axes[0, 1].set_title('Precision-Recall Curve')
        axes[0, 1].legend()
        axes[0, 1].grid(True)
        
        # Confusion Matrix
        cm = confusion_matrix(y_test, y_pred)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[1, 0])
        axes[1, 0].set_title('Confusion Matrix')
        axes[1, 0].set_xlabel('Predicted')
        axes[1, 0].set_ylabel('Actual')
        
        # Probability Distribution
        axes[1, 1].hist(y_pred_proba[y_test == 0], bins=50, alpha=0.7, label='Normal', density=True)
        axes[1, 1].hist(y_pred_proba[y_test == 1], bins=50, alpha=0.7, label='Fraud', density=True)
        axes[1, 1].axvline(self.optimal_threshold, color='red', linestyle='--', label='Optimal Threshold')
        axes[1, 1].set_xlabel('Fraud Probability')
        axes[1, 1].set_ylabel('Density')
        axes[1, 1].set_title('Fraud Probability Distribution')
        axes[1, 1].legend()
        
        plt.tight_layout()
        plt.savefig('./reports/model_evaluation.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def _plot_feature_importance(self, X):
        """Plot feature importance"""
        
        # Get feature importance
        importance = self.model.feature_importances_
        feature_names = X.columns
        
        # Sort by importance
        sorted_idx = np.argsort(importance)
        sorted_features = feature_names[sorted_idx]
        sorted_importance = importance[sorted_idx]
        
        # Plot top 20 features
        top_n = min(20, len(sorted_features))
        
        plt.figure(figsize=(10, 8))
        plt.barh(range(top_n), sorted_importance[-top_n:], color='skyblue')
        plt.yticks(range(top_n), sorted_features[-top_n:])
        plt.xlabel('Feature Importance')
        plt.title('Top 20 XGBoost Feature Importances')
        plt.tight_layout()
        plt.savefig('./reports/feature_importance.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def interpret_model(self, X_sample):
        """Generate SHAP interpretations"""
        
        if X_sample.shape[0] > 1000:
            X_sample = X_sample.sample(1000, random_state=42)
        
        print("Generating SHAP explanations...")
        explainer = shap.TreeExplainer(self.model)
        shap_values = explainer.shap_values(X_sample)
        
        # Summary plot
        plt.figure(figsize=(12, 8))
        shap.summary_plot(shap_values, X_sample, plot_type="bar", show=False)
        plt.tight_layout()
        plt.savefig('./reports/shap_summary_bar.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        # Detailed summary plot
        plt.figure(figsize=(12, 8))
        shap.summary_plot(shap_values, X_sample, show=False)
        plt.tight_layout()
        plt.savefig('./reports/shap_summary.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        return explainer, shap_values
    
    def save_model(self, model_path="./models/"):
        """Save the trained model"""
        
        if self.model is None:
            print("No model to save. Please train the model first.")
            return
        
        # Save with joblib
        joblib.dump(self.model, f"{model_path}/xgboost_fraud_model.pkl")
        
        # Save XGBoost native format
        self.model.get_booster().save_model(f"{model_path}/xgboost_fraud_model.json")
        
        # Save feature columns and threshold
        model_metadata = {
            'feature_columns': self.feature_columns,
            'optimal_threshold': self.optimal_threshold,
            'model_version': 'v2_comprehensive_features'
        }
        pd.Series(model_metadata).to_json(f"{model_path}/model_metadata.json")
        
        print(f"Model saved to {model_path}")
        print(f"- Model file: xgboost_fraud_model.pkl")
        print(f"- Native format: xgboost_fraud_model.json") 
        print(f"- Metadata: model_metadata.json")
    
    def load_model(self, model_path="./models/xgboost_fraud_model.pkl"):
        """Load a saved model"""
        
        self.model = joblib.load(model_path)
        
        # Load metadata if available
        try:
            metadata_path = model_path.replace('.pkl', '_metadata.json')
            metadata = pd.read_json(metadata_path, typ='series')
            self.feature_columns = metadata['feature_columns']
            self.optimal_threshold = metadata['optimal_threshold']
            print(f"Model loaded successfully with optimal threshold: {self.optimal_threshold:.4f}")
        except:
            print("Model loaded but metadata not found. Using default threshold 0.5")
            self.optimal_threshold = 0.5

def main():
    """Main execution function"""
    
    # Initialize model
    fraud_model = FraudXGBoostModel()
    
    # Load training data
    print("Loading training data...")
    training_data = fraud_model.load_training_data(use_feast=False)
    
    if training_data is None:
        print("Failed to load training data. Exiting.")
        return
    
    print(f"Loaded training data shape: {training_data.shape}")
    print(f"Columns: {list(training_data.columns)}")
    print(f"Fraud ratio: {training_data['is_fraud'].mean():.4f}")
    
    # Preprocessing
    print("\nPreprocessing data...")
    X, y = fraud_model.preprocessing(training_data)
    
    print(f"Features shape: {X.shape}")
    print(f"Target shape: {y.shape}")
    print(f"Fraud cases: {y.sum()}/{len(y)} ({y.mean():.4f})")
    
    # Train model
    print("\nTraining XGBoost model...")
    X_test, y_test, y_pred_proba = fraud_model.train_model(X, y, use_smote=True)
    
    # Model interpretation
    print("\nGenerating model interpretations...")
    explainer, shap_values = fraud_model.interpret_model(X_test)
    
    # Save model
    print("\nSaving model...")
    fraud_model.save_model()
    
    print("\n=== TRAINING COMPLETE ===")
    print("Model files saved in ./models/")
    print("Reports saved in ./reports/")
    
    return fraud_model, X, y

if __name__ == "__main__":
    model, features, target = main()