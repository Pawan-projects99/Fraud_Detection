import numpy as np
import pandas as pd
import joblib
import torch
import torch.nn as nn
from feast import FeatureStore 
import torch.optim as optim
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
from datetime import datetime, timedelta, timezone

store = FeatureStore(repo_path="./features")


#Transformer model Architecture
class TransformerEncoderWithAttention(nn.Module):
    def __init__(self, categ_cards, num_numeric, emb_dim, num_heads, num_layers, ff_hidden, dropout):
        super().__init__()
        self.num_categ = len(categ_cards)
        self.num_tokens = self.num_categ+1
        self.emb_dim = emb_dim

        self.cat_embeddings = nn.ModuleList([nn.Embedding(card, emb_dim) for card in categ_cards]) # make a list of embeddings for each categorical feature with their respective categorical_cardinality shape ([2000, 48], [100, 48], [5,48], [3,48], [3,48]) --> list of 5 embedding vectors, 1 for each categorical feature
        self.numeric_bn = nn.BatchNorm1d(num_numeric) 
        self.num_projections = nn.Linear(num_numeric, emb_dim) # no embeddings for numeric features because they are already a dense vector, all numeric features (each row) form a single dense vector, now made into shape (num_rows, 48) if batch_size, the it will become (batch_size, 48)--> 1 embedding vector, for all 6 numeric features
        self.token_type = nn.Parameter(torch.randn(self.num_tokens+1, emb_dim)*0.02) # create a learnable matrix of shape[4,48] nneded to be converted into the same shape of input sequence so that it can be added to the input tokens. this will learn about the positions of different token types(categ, num). consider it similar to positional encoding vector in the NLP, but for tabular data with different type of data(categ, numeric)
        self.cls_token  = nn.Parameter(torch.randn(1, 1, emb_dim) * 0.02) 
        
        #Transformer Encoder 
        self.layers = nn.ModuleList([
            nn.ModuleDict({
                "mha":nn.MultiheadAttention(embed_dim = emb_dim, num_heads = num_heads, dropout = dropout, batch_first = True),
                "ln1": nn.LayerNorm(emb_dim),
                "ffn": nn.Sequential(
                    nn.Linear(emb_dim, ff_hidden),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                    nn.Linear(ff_hidden, emb_dim),
                    nn.Dropout(dropout)
                ),
                "ln2": nn.LayerNorm(emb_dim)
            }) for _ in range(num_layers)
        ])
        # classifier output logits
        self.classifier = nn.Sequential(
            nn.LayerNorm(emb_dim),
            nn.Linear(emb_dim, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 1)
        )
    def forward(self, X_categ, X_numeric, return_attn = False):
        categ_embs = [emb(X_categ[:,i]) for i,emb in enumerate(self.cat_embeddings)] # each categorical feature from X_categ will map to the corresponding row of embedded table from cat_embeddings -> here i = [0,1,2,3,4,5] we have 6 categ_columns in total
        categ_embs = torch.stack(categ_embs, dim = 1)# stacking the categ_embs at dim=1, i.e, shape = [batch_size, 6, 48], stacked at dim=1 to become 6.
        X_numeric = self.numeric_bn(X_numeric)
        num_embs = self.num_projections(X_numeric).unsqueeze(1) # to become [batch_size, 1, 48] added a new dimension at position 1.

        tokens = torch.cat([categ_embs, num_embs], dim=1) # shape = [batch_size, 6+1, 48] = [batch_size, 7, 48]
        #tokens = tokens + self.token_type.unsqueeze(0)# unsqueezed in the dim=0 to create an extra dimension to match the tokens dimension to perform addition.
        cls_tokens = self.cls_token.expand(tokens.size(0), -1, -1)
        x = torch.cat([cls_tokens, tokens], dim=1)
        x = x + self.token_type.unsqueeze(0)

        attn_matrices = []
        for layer in self.layers:
            mha = layer["mha"]
            attn_out, attn_weights = mha(x,x,x, need_weights= True, average_attn_weights = False)
            x = layer["ln1"](x+attn_out)
            ff = layer["ffn"](x)
            x = layer["ln2"](x+ff)

            if return_attn:
                attn_matrices.append(attn_weights.detach())# No gradients
        pooled = x[:, 0, :] # mean pooling on dim=1 to converst all 7 tokensof 48 dim into single 48 dim, Now shape(batch_size, 48)
        out = self.classifier(pooled)

        if return_attn:
            return out, attn_matrices
        else:
            return out

def train_model(model, train_loader, eval_loader, device, epochs, patience = 15):

    #Loss Crieterion
    # n_pos = int(train_loader[label_columns].sum())# transactions which are fraud (minority class)
    # n_neg = int(len(y_resampled) - n_pos)# transactions which are not fraud (majority class)
    # pos_weight = torch.tensor([max(1.0, n_neg/max(1.0, n_pos))], dtype = torch.float).to(device)

    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience = 5, factor = 0.5)

    best_f1 = 0
    patience_counter = 0
    train_losses = []
    eval_losses = []
    eval_f1_scores = []

    
    for epoch in range(epochs):
        #Training phase
        model.train()
        total_train_loss = 0.0
        total_samples = 0
        for Xc_train, Xn_train, y_train in train_loader:
            Xc_train, Xn_train, y_train = Xc_train.to(device), Xn_train.to(device), y_train.to(device)
            optimizer.zero_grad()
            logits = model(Xc_train, Xn_train)
            loss = criterion(logits, y_train)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            total_train_loss += loss.item()
            total_samples += 1
        epoch_loss = total_train_loss/total_samples

        #Evaluating phase
        model.eval()
        total_eval_loss = 0.0
        all_eval_probs = []
        all_y_true = []

        with torch.no_grad():
            for Xc_eval, Xn_eval, y_eval in eval_loader:
                eval_logits = model(Xc_eval.to(device), Xn_eval.to(device))
                loss = criterion(eval_logits, y_eval.to(device))
                total_eval_loss += loss.item()

                eval_probs = torch.sigmoid(eval_logits).cpu().numpy().ravel()
                y_true = y_eval.cpu().numpy().ravel()

                all_eval_probs.extend(eval_probs)
                all_y_true.extend(y_true)

        precisions, recalls, thresholds = precision_recall_curve(all_y_true, all_eval_probs)
        pr_auc = average_precision_score(all_y_true, all_eval_probs)

        # F1 calculation
        f1_scores = 2 * (precisions * recalls) / (precisions + recalls + 1e-8)
        optimal_idx = np.argmax(f1_scores)
        optimal_threshold = thresholds[max(optimal_idx, 0)] if thresholds.size > 0 else 0.5
        print("optimal_threshold", optimal_threshold)
        eval_preds = ( all_eval_probs > optimal_threshold).astype(int)

        avg_train_loss = total_train_loss / len(train_loader)
        avg_eval_loss = total_eval_loss / len(eval_loader)
        eval_f1 = f1_score(all_y_true, eval_preds)

        train_losses.append(avg_train_loss)
        eval_losses.append(avg_eval_loss)
        eval_f1_scores.append(eval_f1)

        print(f'Epoch {epoch:3d} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_eval_loss:.4f} | Val F1: {eval_f1_scores[-1]:.4f}')

        # Early stopping
        if(f1_scores[optimal_idx] > best_f1):
            best_f1 = f1_scores[optimal_idx]
            patience_counter = 0
            torch.save(model.state_dict(), "./models/transformer_best_fraud_model.pth")
        else:
            patience+=1

        if patience_counter >= patience:
            print(f'Early stopping at epoch {epoch}')
            break
    
    model.load_state_dict(torch.load("./models/transformer_best_fraud_model.pth"))
    return train_losses, eval_losses, eval_f1_scores, optimal_threshold
def evaluate_model(model, test_loader, device, threshold):
    """Evaluate model with optimized threshold for high recall"""
    model.eval()
    all_preds = []
    all_probs = []
    all_labels = []
    
    with torch.no_grad():
         for Xc_test, Xn_test, y_test in test_loader:
            test_logits = model(Xc_test.to(device), Xn_test.to(device))
            test_probs = torch.sigmoid(test_logits).squeeze().cpu().numpy()
            
            # Use custom threshold for better recall
            predicted = (test_probs > threshold).astype(int)
            
            all_preds.extend(predicted)
            all_probs.extend(test_probs)
            all_labels.extend(y_test.cpu().numpy())
    
    return all_labels, all_preds, all_probs

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


# criterion = nn.BCEWithLogitsLoss()

# criterion = FocalLoss(alpha=2, gamma=2)
# optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
# print("pos_weight:", pos_weight.item())   


def main():
    df = pd.read_parquet("./features/data/training_dataset.parquet")

    print("df_shape: ",df.shape)
    print(df.columns)
    print(df.head(10))
    print(df.describe())

    #Time based split instaed of random split
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.sort_values("timestamp").reset_index(drop=True)

    cut_off1 = int(len(df) * 0.7)
    cut_off2 = int(len(df)*0.85)

    train_df = df.iloc[:cut_off1]
    eval_df = df.iloc[cut_off1:cut_off2]
    test_df  = df.iloc[cut_off2:]

    # train_df, test_df = train_test_split(df, test_size=0.2, random_state=42, stratify=df[label_columns])
    print(train_df.shape)
    print(eval_df.shape)
    print(test_df.shape)

    numerical_columns = ['amount', 'hour',  'account_balance', 'velocity_1h', 'velocity_24h', 'user_risk_score', 'avg_transaction_amount_7d', 'avg_transaction_amount_30d',
        'transaction_count_7d', 'transaction_count_30d', 'unique_merchants_7d', 'unique_merchants_30d', 'night_transaction_ratio_7d', 'high_amount_transaction_ratio_7d',
        'merchant_risk_score', 'merchant_fraud_rate_7d', 'merchant_fraud_rate_30d', 'merchant_transaction_count_7d', 'merchant_avg_amount_7d', 'merchant_reputation_score']

    categorical_columns = ['payment_method', 'device_type', 'location_risk', 'merchant_category']
    label_columns = ['is_fraud']

    # Transforming the data using label encoders for categ data and standard scaler for numeric data
    label_encoders = {}
    for col in categorical_columns:
        le = LabelEncoder()
        
        # Fit only on training data
        train_df[col] = le.fit_transform(train_df[col].astype(str))
        eval_df[col] = le.transform(eval_df[col].astype(str))
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
    eval_df[numerical_columns] = scaler.transform(eval_df[numerical_columns])
    test_df[numerical_columns]  = scaler.transform(test_df[numerical_columns])

    X_train = train_df[categorical_columns+numerical_columns]
    y_train = train_df[label_columns]
    X_eval = eval_df[categorical_columns+numerical_columns]
    y_eval = eval_df[label_columns]
    X_test = test_df[categorical_columns+numerical_columns]
    y_test = test_df[label_columns]
    #Resampling the data using imblearn SMOTENC only training data
    cat_features = [X_train.columns.get_loc(col) for col in categorical_columns]
    smote = SMOTENC(categorical_features = cat_features, sampling_strategy= 0.25, random_state = 42)
    X_resampled, y_resampled = smote.fit_resample(X_train,y_train)
    
    print(X_resampled.shape)
   
    #Converting into tensors
    Xc_train = torch.tensor(X_resampled[categorical_columns].values, dtype = torch.long)
    Xn_train = torch.tensor(X_resampled[numerical_columns].values, dtype = torch.float32)
    y_train = torch.tensor(y_resampled[label_columns].values, dtype = torch.float32).view(-1,1)

    Xc_eval = torch.tensor(X_eval[categorical_columns].values, dtype = torch.long)
    Xn_eval = torch.tensor(X_eval[numerical_columns].values, dtype = torch.float32)
    y_eval = torch.tensor(y_eval.values, dtype = torch.float32).view(-1,1)


    Xc_test = torch.tensor(X_test[categorical_columns].values, dtype = torch.long)
    Xn_test = torch.tensor(X_test[numerical_columns].values, dtype = torch.float32)
    y_test = torch.tensor(y_test.values, dtype = torch.float32).view(-1,1)

    train_ds = TensorDataset(Xc_train, Xn_train, y_train)
    eval_ds = TensorDataset(Xc_eval, Xn_eval, y_eval)
    test_ds = TensorDataset(Xc_test, Xn_test, y_test)

    train_loader = DataLoader(train_ds, batch_size = 256, shuffle = True)
    eval_loader = DataLoader(eval_ds, batch_size = 256, shuffle = False)
    test_loader = DataLoader(test_ds, batch_size = 256, shuffle = False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = TransformerEncoderWithAttention(
        categ_cards = categorical_cardinalities,
        num_numeric = len(numerical_columns),
        emb_dim = 64,
        num_heads = 4,
        num_layers = 3,
        ff_hidden = 128,
        dropout = 0.1
    ).to(device)

    print("\nStarting training...")
    train_losses, val_losses, val_f1_scores, optimal_threshold = train_model(
        model, train_loader, eval_loader, device, epochs=10, patience=15
    )
    y_true, y_pred, y_probs = evaluate_model(model, test_loader, device, optimal_threshold)
    print("\nClassification Report:")
    print(classification_report(y_true, y_pred, target_names=['Normal', 'Fraud']))

if __name__ == "__main__":
    main()

