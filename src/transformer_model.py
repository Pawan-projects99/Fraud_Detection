import numpy as np
import pandas as pd
import joblib
import torch
import torch.nn as nn
from feast import FeatureStore 
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score, precision_recall_curve, recall_score, f1_score, average_precision_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from torch.utils.data import DataLoader, TensorDataset
from datetime import timedelta
import os
from imblearn.over_sampling import SMOTENC
import torch.nn.functional as F
import math

transactions = pd.read_parquet("./features/data/transactions.parquet")
entity_df = transactions[["transaction_id", "user_id", "merchant_id", "amount", "timestamp", "location", "is_fraud"]].rename(columns = {"timestamp":"event_timestamp"})

store = FeatureStore("./features")
features_to_pull = [ 
            "user_behaviour:avg_transaction",
            "user_behaviour:transaction_count_30d",
            "user_behaviour:decline_rate_60d",
            "merchant_risk_profile:fraud_rate",
            "merchant_risk_profile:avg_transaction_value"
            ]

df = store.get_historical_features(
    entity_df = entity_df,
    features = features_to_pull
).to_df()
# df = feature_df.merge(transactions, left_on=["user_id", "merchant_id", "event_timestamp"],
#                           right_on=["user_id", "merchant_id", "timestamp"], how="left")
print(df.shape)
print(df.head(10))

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

categorical_columns = ["is_night","is_location_changed"]
numerical_columns = ["amount", "avg_transaction", "transaction_count_30d", "decline_rate_60d", "fraud_rate", "avg_transaction_value", "amount_vs_avg", "amount_vs_merchant_avg", "transactions_1hr", "amount_deviation", "amount_merchant_risk"]
label_columns = ["is_fraud"]

df = df.dropna(subset = categorical_columns + numerical_columns + label_columns)
df.to_parquet("./features/data/transactions_features.parquet")
df.reset_index(drop = True, inplace = True)
print("After dropna:", df.shape)

#Time based split instaed of random split
df["event_timestamp"] = pd.to_datetime(df["event_timestamp"])
df = df.sort_values("event_timestamp").reset_index(drop=True)

cut_off = int(len(df) * 0.8)
train_df = df.iloc[:cut_off]
test_df  = df.iloc[cut_off:]
# train_df, test_df = train_test_split(df, test_size=0.2, random_state=42, stratify=df[label_columns])
print(train_df.shape)
print(test_df.shape)

# Transforming the data using label encoders for categ data and standard scalar for numeric data
label_encoders = {}
for col in categorical_columns:
    le = LabelEncoder()
    
    # Fit only on training data
    train_df[col] = le.fit_transform(train_df[col].astype(str))
    
    # Handle unseen categories in test set
    unseen_mask = ~test_df[col].isin(le.classes_)
    le_classes = np.append(le.classes_, "Unknown")
    le.classes_ = le_classes
    
    test_df.loc[unseen_mask, col] = "Unknown"
    test_df[col] = le.transform(test_df[col].astype(str))
    
    label_encoders[col] = le

categorical_cardinalities = [len(le.classes_) for le in label_encoders.values()]
print("categorical_cardinalities: ", categorical_cardinalities)

scaler = StandardScaler()
train_df[numerical_columns] = scaler.fit_transform(train_df[numerical_columns])
test_df[numerical_columns]  = scaler.transform(test_df[numerical_columns])

X_train = train_df[categorical_columns+numerical_columns]
y_train = train_df[label_columns]

X_test = test_df[categorical_columns+numerical_columns]
y_test = test_df[label_columns]

#Resampling the data using imblearn SMOTENC only training data
cat_features = [X_train.columns.get_loc(col) for col in categorical_columns]
smote = SMOTENC(categorical_features = cat_features, sampling_strategy= 0.25, random_state = 42)
X_resampled, y_resampled = smote.fit_resample(X_train,y_train)
resampled_df = pd.concat([X_resampled, y_resampled], axis=1).sample(frac=1, random_state=42).reset_index(drop=True)

print(resampled_df.shape)
print(resampled_df.head(10))
print(resampled_df.isna().sum())
print(resampled_df.describe().transpose())

#Converting into tensors
Xc_train = torch.tensor(resampled_df[categorical_columns].values, dtype = torch.long)
Xn_train = torch.tensor(resampled_df[numerical_columns].values, dtype = torch.float32)
y_train = torch.tensor(resampled_df[label_columns].values, dtype = torch.float32).view(-1,1)

Xc_test = torch.tensor(X_test[categorical_columns].values, dtype = torch.long)
Xn_test = torch.tensor(X_test[numerical_columns].values, dtype = torch.float32)
y_test = torch.tensor(y_test.values, dtype = torch.float32).view(-1,1)

train_ds = TensorDataset(Xc_train, Xn_train, y_train)
train_loader = DataLoader(train_ds, batch_size = 32, shuffle = True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#Transformer model Architecture
# class TransformerEncoderWithAttention(nn.Module):
#     def __init__(self, categ_cards, num_numeric, emb_dim, num_heads, num_layers, ff_hidden, dropout):
#         super().__init__()
#         self.num_categ = len(categ_cards)
#         self.num_tokens = self.num_categ+1
#         self.emb_dim = emb_dim

#         self.cat_embeddings = nn.ModuleList([nn.Embedding(card, emb_dim) for card in categ_cards]) # make a list of embeddings for each categorical feature with their respective categorical_cardinality shape ([2000, 48], [100, 48], [5,48], [3,48], [3,48]) --> list of 5 embedding vectors, 1 for each categorical feature
#         self.numeric_bn = nn.BatchNorm1d(num_numeric) 
#         self.num_projections = nn.Linear(num_numeric, emb_dim) # no embeddings for numeric features because they are already a dense vector, all numeric features (each row) form a single dense vector, now made into shape (num_rows, 48) if batch_size, the it will become (batch_size, 48)--> 1 embedding vector, for all 6 numeric features
#         self.token_type = nn.Parameter(torch.randn(self.num_tokens+1, emb_dim)*0.02) # create a learnable matrix of shape[4,48] nneded to be converted into the same shape of input sequence so that it can be added to the input tokens. this will learn about the positions of different token types(categ, num). consider it similar to positional encoding vector in the NLP, but for tabular data with different type of data(categ, numeric)
#         self.cls_token  = nn.Parameter(torch.randn(1, 1, emb_dim) * 0.02) 
        
#         #Transformer Encoder 
#         self.layers = nn.ModuleList([
#             nn.ModuleDict({
#                 "mha":nn.MultiheadAttention(embed_dim = emb_dim, num_heads = num_heads, dropout = dropout, batch_first = True),
#                 "ln1": nn.LayerNorm(emb_dim),
#                 "ffn": nn.Sequential(
#                     nn.Linear(emb_dim, ff_hidden),
#                     nn.ReLU(),
#                     nn.Dropout(dropout),
#                     nn.Linear(ff_hidden, emb_dim),
#                     nn.Dropout(dropout)
#                 ),
#                 "ln2": nn.LayerNorm(emb_dim)
#             }) for _ in range(num_layers)
#         ])
#         # classifier output logits
#         self.classifier = nn.Sequential(
#             nn.LayerNorm(emb_dim),
#             nn.Linear(emb_dim, 32),
#             nn.ReLU(),
#             nn.Dropout(dropout),
#             nn.Linear(32, 1)
#         )
#     def forward(self, X_categ, X_numeric, return_attn = False):
#         categ_embs = [emb(X_categ[:,i]) for i,emb in enumerate(self.cat_embeddings)] # each categorical feature from X_categ will map to the corresponding row of embedded table from cat_embeddings -> here i = [0,1,2,3,4,5] we have 6 categ_columns in total
#         categ_embs = torch.stack(categ_embs, dim = 1)# stacking the categ_embs at dim=1, i.e, shape = [batch_size, 6, 48], stacked at dim=1 to become 6.
#         X_numeric = self.numeric_bn(X_numeric)
#         num_embs = self.num_projections(X_numeric).unsqueeze(1) # to become [batch_size, 1, 48] added a new dimension at position 1.

#         tokens = torch.cat([categ_embs, num_embs], dim=1) # shape = [batch_size, 6+1, 48] = [batch_size, 7, 48]
#         #tokens = tokens + self.token_type.unsqueeze(0)# unsqueezed in the dim=0 to create an extra dimension to match the tokens dimension to perform addition.
#         cls_tokens = self.cls_token.expand(tokens.size(0), -1, -1)
#         x = torch.cat([cls_tokens, tokens], dim=1)
#         x = x + self.token_type.unsqueeze(0)

#         attn_matrices = []
#         for layer in self.layers:
#             mha = layer["mha"]
#             attn_out, attn_weights = mha(x,x,x, need_weights= True, average_attn_weights = False)
#             x = layer["ln1"](x+attn_out)
#             ff = layer["ffn"](x)
#             x = layer["ln2"](x+ff)

#             if return_attn:
#                 attn_matrices.append(attn_weights.detach())# No gradients
#         pooled = x[:, 0, :] # mean pooling on dim=1 to converst all 7 tokensof 48 dim into single 48 dim, Now shape(batch_size, 48)
#         out = self.classifier(pooled)

#         if return_attn:
#             return out, attn_matrices
#         else:
#             return out

class MultiHeadSelfAttention(nn.Module):
    """Custom Multi-Head Attention matching Keras implementation"""
    def __init__(self, embed_size, num_heads):
        super().__init__()
        self.embed_size = embed_size
        self.num_heads = num_heads
        assert embed_size % num_heads == 0
        self.projection_dim = embed_size // num_heads
        
        self.query_dense = nn.Linear(embed_size, embed_size)
        self.key_dense = nn.Linear(embed_size, embed_size)
        self.value_dense = nn.Linear(embed_size, embed_size)
        self.combine_heads = nn.Linear(embed_size, embed_size)
        
    def attention(self, query, key, value):
        # query, key, value shape: (batch_size, num_heads, seq_len, projection_dim)
        score = torch.matmul(query, key.transpose(-2, -1))
        dim_key = torch.tensor(key.shape[-1], dtype=torch.float32, device=key.device)
        scaled_score = score / torch.sqrt(dim_key)
        weights = F.softmax(scaled_score, dim=-1)
        output = torch.matmul(weights, value)
        return output, weights
    
    def separate_heads(self, x, batch_size):
        # x shape: (batch_size, seq_len, embed_size)
        x = x.reshape(batch_size, -1, self.num_heads, self.projection_dim)
        return x.permute(0, 2, 1, 3)  # (batch_size, num_heads, seq_len, projection_dim)
    
    def forward(self, inputs):
        batch_size = inputs.shape[0]
        
        query = self.query_dense(inputs)
        key = self.key_dense(inputs)
        value = self.value_dense(inputs)
        
        query = self.separate_heads(query, batch_size)
        key = self.separate_heads(key, batch_size)
        value = self.separate_heads(value, batch_size)
        
        attention, weights = self.attention(query, key, value)
        attention = attention.permute(0, 2, 1, 3)  # (batch_size, seq_len, num_heads, projection_dim)
        concat_attention = attention.reshape(batch_size, -1, self.embed_size)
        output = self.combine_heads(concat_attention)
        
        return output


class TransformerBlock(nn.Module):
    """Transformer block matching Keras implementation with pre-norm architecture"""
    def __init__(self, embed_size, num_heads, ff_dim, dropout_rate=0.1):
        super().__init__()
        self.att = MultiHeadSelfAttention(embed_size, num_heads)
        self.ffn = nn.Sequential(
            nn.Linear(embed_size, ff_dim),
            nn.ReLU(),
            nn.Linear(ff_dim, embed_size),
        )
        self.layernorm1 = nn.LayerNorm(embed_size, eps=1e-6)
        self.layernorm2 = nn.LayerNorm(embed_size, eps=1e-6)
        self.dropout1 = nn.Dropout(dropout_rate)
        self.dropout2 = nn.Dropout(dropout_rate)
    
    def forward(self, inputs, training=True):
        # Pre-norm architecture (LayerNorm before operations)
        attn_output = self.att(inputs)
        attn_output = self.dropout1(attn_output) if training else attn_output
        out1 = self.layernorm1(inputs + attn_output)
        
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output) if training else ffn_output
        return self.layernorm2(out1 + ffn_output)

class TransformerEncoderWithAttention(nn.Module):
    """
    Keras-style transformer adapted for tabular data with categorical and numerical features
    """
    def __init__(self, categ_cards, num_numeric, embed_size, num_heads, ff_dim, dropout_rate):
        super().__init__()
        self.num_categ = len(categ_cards)
        self.embed_size = embed_size
        
        # Categorical embeddings
        self.cat_embeddings = nn.ModuleList([
            nn.Embedding(card, embed_size) for card in categ_cards
        ])
        
        # Numerical feature processing
        self.numeric_projection = nn.Linear(num_numeric, embed_size)
        
        # Transformer block (Keras style)
        self.transformer_block = TransformerBlock(embed_size, num_heads, ff_dim, dropout_rate)
        
        # Output layers matching Keras
        self.global_dropout = nn.Dropout(0.1)
        self.dense_20 = nn.Linear(embed_size, 20)
        self.output_layer = nn.Linear(20, 1)
        
    def forward(self, X_categ, X_numeric):
        batch_size = X_categ.shape[0]
        
        # Process categorical features
        categ_embs = []
        for i, emb in enumerate(self.cat_embeddings):
            categ_embs.append(emb(X_categ[:, i]))
        categ_embs = torch.stack(categ_embs, dim=1)  # (batch_size, num_categ, embed_size)
        
        # Process numerical features
        num_emb = self.numeric_projection(X_numeric).unsqueeze(1)  # (batch_size, 1, embed_size)
        
        # Combine all features
        x = torch.cat([categ_embs, num_emb], dim=1)  # (batch_size, num_categ + 1, embed_size)
        
        # Transformer block
        x = self.transformer_block(x, training=self.training)
        
        # Global average pooling (Keras style)
        x = torch.mean(x, dim=1)  # (batch_size, embed_size)
        
        # Classifier head (matching Keras)
        x = self.global_dropout(x)
        x = F.relu(self.dense_20(x))
        x = self.output_layer(x)  # No sigmoid here - use BCEWithLogitsLoss
        
        return x


# Loss Crieterion
# n_pos = int(resampled_df[label_columns].sum())# transactions which are fraud (minority class)
# n_neg = int(len(resampled_df) - n_pos)# transactions which are not fraud (majority class)
# pos_weight = torch.tensor([max(1.0, n_neg/max(1.0, n_pos))], dtype = torch.float).to(device)


# class FocalLoss(nn.Module):
#     def __init__(self, alpha=1, gamma=2):
#         super().__init__()
#         self.alpha = alpha
#         self.gamma = gamma
    
#     def forward(self, inputs, targets):
#         bce_loss = nn.functional.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
#         pt = torch.exp(-bce_loss)
#         focal_loss = self.alpha * (1-pt)**self.gamma * bce_loss
#         return focal_loss.mean()

#Model creation
model = TransformerEncoderWithAttention(
    # categ_cards = categorical_cardinalities,
    # num_numeric = len(numerical_columns),
    # emb_dim = 64,
    # num_heads = 4,
    # num_layers = 3,
    # ff_hidden = 64,
    # dropout = 0.1

    categ_cards=categorical_cardinalities,
        num_numeric= len(numerical_columns),
        embed_size=32, 
        num_heads=4,   
        ff_dim=32,     
        dropout_rate=0.1
)
criterion = nn.BCEWithLogitsLoss()

# criterion = FocalLoss(alpha=2, gamma=2)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
# print("pos_weight:", pos_weight.item())   

# Model Training
epochs = 25
for epoch in range(epochs):
    model.train()
    total_loss = 0.0
    total_samples = 0
    for Xc_train, Xn_train, y_train in train_loader:
        Xc_train, Xn_train, y_train = Xc_train.to(device), Xn_train.to(device), y_train.to(device)
        optimizer.zero_grad()
        logits = model(Xc_train, Xn_train)
        loss = criterion(logits, y_train)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        total_loss += loss.item()
        total_samples +=1
    epoch_loss = total_loss/total_samples
    print(f"epoch{epoch}: {epoch_loss}")
    
# Model Evaluation
model.eval()
with torch.no_grad():
    test_logits = model(Xc_test.to(device), Xn_test.to(device))
    test_probs = torch.sigmoid(test_logits).cpu().numpy().ravel()
    y_true = y_test.cpu().numpy().ravel()
    
precisions, recalls, thresholds = precision_recall_curve(y_true, test_probs)
pr_auc = average_precision_score(y_true, test_probs)
# F1 calculation
f1_scores = 2 * (precisions * recalls) / (precisions + recalls + 1e-8)
optimal_idx = np.argmax(f1_scores)
optimal_threshold = thresholds[max(optimal_idx, 0)] if thresholds.size > 0 else 0.5

print(f"Optimal Threshold: {optimal_threshold:.4f}, F1 Score: {f1_scores[optimal_idx]:.4f}")
test_preds = (test_probs > optimal_threshold).astype(int)
print("ROC-AUC:", roc_auc_score(y_true, test_probs))
print(classification_report(y_true, test_preds, digits = 4))




        




