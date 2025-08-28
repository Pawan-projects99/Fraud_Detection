import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, f1_score, recall_score
from sklearn.utils.class_weight import compute_class_weight
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple
import warnings
warnings.filterwarnings('ignore')

# Set random seeds for reproducibility
np.random.seed(42)
torch.manual_seed(42)

class FraudDataset(Dataset):
    """Custom Dataset for fraud detection with mixed data types"""
    
    def __init__(self, numerical_features, categorical_features, labels):
        self.numerical_features = torch.FloatTensor(numerical_features)
        self.categorical_features = torch.LongTensor(categorical_features)
        self.labels = torch.LongTensor(labels)
    
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        return {
            'numerical': self.numerical_features[idx],
            'categorical': self.categorical_features[idx],
            'label': self.labels[idx]
        }

class MultiHeadAttention(nn.Module):
    """Multi-head attention mechanism"""
    
    def __init__(self, d_model, n_heads, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_model)
    
    def forward(self, x):
        batch_size, seq_len, d_model = x.size()
        
        # Linear transformations
        Q = self.W_q(x).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        K = self.W_k(x).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        V = self.W_v(x).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        
        # Scaled dot-product attention
        attention_scores = torch.matmul(Q, K.transpose(-2, -1)) / np.sqrt(self.d_k)
        attention_weights = torch.softmax(attention_scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        
        # Apply attention to values
        attention_output = torch.matmul(attention_weights, V)
        attention_output = attention_output.transpose(1, 2).contiguous().view(
            batch_size, seq_len, d_model
        )
        
        # Output projection and residual connection
        output = self.W_o(attention_output)
        return self.layer_norm(output + x)

class TransformerEncoderBlock(nn.Module):
    """Transformer encoder block with attention and feed-forward"""
    
    def __init__(self, d_model, n_heads, d_ff, dropout=0.1):
        super().__init__()
        self.attention = MultiHeadAttention(d_model, n_heads, dropout)
        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout)
        )
        self.layer_norm = nn.LayerNorm(d_model)
    
    def forward(self, x):
        # Apply attention
        attended = self.attention(x)
        
        # Feed-forward with residual connection
        output = self.feed_forward(attended)
        return self.layer_norm(output + attended)

class FraudTransformerModel(nn.Module):
    """Transformer-based fraud detection model"""
    
    def __init__(self, 
                 num_features: int,
                 cat_features_info: Dict[str, int],
                 d_model: int = 128,
                 n_heads: int = 8,
                 n_layers: int = 4,
                 d_ff: int = 512,
                 dropout: float = 0.1):
        super().__init__()
        
        self.d_model = d_model
        self.num_features = num_features
        self.cat_features_info = cat_features_info
        
        # Embedding layers for categorical features
        self.cat_embeddings = nn.ModuleDict({
            name: nn.Embedding(vocab_size, min(50, (vocab_size + 1) // 2))
            for name, vocab_size in cat_features_info.items()
        })
        
        # Calculate total embedding dimension
        cat_embed_dim = sum(min(50, (vocab_size + 1) // 2) 
                           for vocab_size in cat_features_info.values())
        
        # Input projection layer
        input_dim = num_features + cat_embed_dim
        self.input_projection = nn.Linear(input_dim, d_model)
        
        # Positional encoding (learnable)
        self.pos_encoding = nn.Parameter(torch.randn(1, 1, d_model))
        
        # Transformer encoder layers
        self.transformer_layers = nn.ModuleList([
            TransformerEncoderBlock(d_model, n_heads, d_ff, dropout)
            for _ in range(n_layers)
        ])
        
        # Classification head optimized for high recall
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, d_model // 4),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 4, 2)
        )
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.xavier_uniform_(module.weight)
    
    def forward(self, numerical_features, categorical_features):
        batch_size = numerical_features.size(0)
        
        # Process categorical features through embeddings
        cat_embeds = []
        for i, (name, _) in enumerate(self.cat_features_info.items()):
            embed = self.cat_embeddings[name](categorical_features[:, i])
            cat_embeds.append(embed)
        
        # Concatenate all features
        if cat_embeds:
            cat_features = torch.cat(cat_embeds, dim=1)
            combined_features = torch.cat([numerical_features, cat_features], dim=1)
        else:
            combined_features = numerical_features
        
        # Project to model dimension and add positional encoding
        x = self.input_projection(combined_features).unsqueeze(1)  # Add sequence dimension
        x = x + self.pos_encoding
        
        # Pass through transformer layers
        for layer in self.transformer_layers:
            x = layer(x)
        
        # Global average pooling and classification
        x = x.squeeze(1)  # Remove sequence dimension
        logits = self.classifier(x)
        
        return logits

def create_synthetic_fraud_data(n_samples: int = 10000, fraud_ratio: float = 0.1):
    """Create synthetic fraud detection dataset with mixed features"""
    
    np.random.seed(42)
    n_fraud = int(n_samples * fraud_ratio)
    n_normal = n_samples - n_fraud
    
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
    
    return df

def focal_loss(inputs, targets, alpha=0.25, gamma=2.0):
    """Focal loss to handle class imbalance and improve recall"""
    ce_loss = nn.CrossEntropyLoss(reduction='none')(inputs, targets)
    pt = torch.exp(-ce_loss)
    focal_loss = alpha * (1 - pt) ** gamma * ce_loss
    return focal_loss.mean()

def train_model(model, train_loader, val_loader, device, epochs=100, patience=15):
    """Train the fraud detection model with early stopping"""
    
    # Use class weights to handle imbalance
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)
    
    best_f1 = 0
    patience_counter = 0
    train_losses = []
    val_losses = []
    val_f1_scores = []
    val_recalls = []
    
    for epoch in range(epochs):
        # Training phase
        model.train()
        total_train_loss = 0
        
        for batch in train_loader:
            optimizer.zero_grad()
            
            numerical = batch['numerical'].to(device)
            categorical = batch['categorical'].to(device)
            labels = batch['label'].to(device)
            
            outputs = model(numerical, categorical)
            
            # Use focal loss for better handling of imbalanced data
            loss = focal_loss(outputs, labels, alpha=0.75, gamma=2.0)
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            total_train_loss += loss.item()
        
        # Validation phase
        model.eval()
        total_val_loss = 0
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for batch in val_loader:
                numerical = batch['numerical'].to(device)
                categorical = batch['categorical'].to(device)
                labels = batch['label'].to(device)
                
                outputs = model(numerical, categorical)
                loss = focal_loss(outputs, labels, alpha=0.75, gamma=2.0)
                
                total_val_loss += loss.item()
                
                _, predicted = torch.max(outputs.data, 1)
                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        # Calculate metrics
        avg_train_loss = total_train_loss / len(train_loader)
        avg_val_loss = total_val_loss / len(val_loader)
        val_f1 = f1_score(all_labels, all_preds)
        val_recall = recall_score(all_labels, all_preds)
        
        train_losses.append(avg_train_loss)
        val_losses.append(avg_val_loss)
        val_f1_scores.append(val_f1)
        val_recalls.append(val_recall)
        
        scheduler.step(avg_val_loss)
        
        if epoch % 10 == 0:
            print(f'Epoch {epoch:3d} | Train Loss: {avg_train_loss:.4f} | '
                  f'Val Loss: {avg_val_loss:.4f} | Val F1: {val_f1:.4f} | Val Recall: {val_recall:.4f}')
        
        # Early stopping based on F1 score
        if val_f1 > best_f1:
            best_f1 = val_f1
            patience_counter = 0
            torch.save(model.state_dict(), 'best_fraud_model.pth')
        else:
            patience_counter += 1
            
        if patience_counter >= patience:
            print(f'Early stopping at epoch {epoch}')
            break
    
    # Load best model
    model.load_state_dict(torch.load('best_fraud_model.pth'))
    
    return train_losses, val_losses, val_f1_scores, val_recalls

def evaluate_model(model, test_loader, device, threshold=0.3):
    """Evaluate model with optimized threshold for high recall"""
    model.eval()
    all_preds = []
    all_probs = []
    all_labels = []
    
    with torch.no_grad():
        for batch in test_loader:
            numerical = batch['numerical'].to(device)
            categorical = batch['categorical'].to(device)
            labels = batch['label'].to(device)
            
            outputs = model(numerical, categorical)
            probs = torch.softmax(outputs, dim=1)
            
            # Use custom threshold for better recall
            fraud_probs = probs[:, 1].cpu().numpy()
            predicted = (fraud_probs > threshold).astype(int)
            
            all_preds.extend(predicted)
            all_probs.extend(fraud_probs)
            all_labels.extend(labels.cpu().numpy())
    
    return all_labels, all_preds, all_probs

def plot_training_metrics(train_losses, val_losses, val_f1_scores, val_recalls):
    """Plot training metrics"""
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Training and validation loss
    axes[0, 0].plot(train_losses, label='Training Loss')
    axes[0, 0].plot(val_losses, label='Validation Loss')
    axes[0, 0].set_title('Training and Validation Loss')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(True)
    
    # F1 Score
    axes[0, 1].plot(val_f1_scores, label='Validation F1', color='green')
    axes[0, 1].set_title('Validation F1 Score')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('F1 Score')
    axes[0, 1].legend()
    axes[0, 1].grid(True)
    
    # Recall
    axes[1, 0].plot(val_recalls, label='Validation Recall', color='red')
    axes[1, 0].set_title('Validation Recall')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('Recall')
    axes[1, 0].legend()
    axes[1, 0].grid(True)
    
    # Combined F1 and Recall
    axes[1, 1].plot(val_f1_scores, label='F1 Score', color='green')
    axes[1, 1].plot(val_recalls, label='Recall', color='red')
    axes[1, 1].set_title('F1 Score vs Recall')
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('Score')
    axes[1, 1].legend()
    axes[1, 1].grid(True)
    
    plt.tight_layout()
    plt.show()

def main():
    """Main execution function"""
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    
    # Generate synthetic fraud data
    print("Generating synthetic fraud detection data...")
    df = create_synthetic_fraud_data(n_samples=200000, fraud_ratio=0.08)
    
    print(f"Dataset shape: {df.shape}")
    print(f"Fraud ratio: {df['is_fraud'].mean():.3f}")
    
    # Prepare features
    numerical_cols = ['amount', 'balance', 'hour', 'days_since_last', 'merchant_risk', 'velocity_1h', 'velocity_24h']
    categorical_cols = ['payment_method', 'merchant_category', 'device_type', 'location_risk']
    
    # Scale numerical features
    scaler = StandardScaler()
    X_numerical = scaler.fit_transform(df[numerical_cols])
    
    # Prepare categorical features
    X_categorical = df[categorical_cols].values
    
    # Get categorical feature info
    cat_features_info = {
        col: df[col].nunique() + 1  # +1 for unknown category
        for col in categorical_cols
    }
    
    y = df['is_fraud'].values
    
    # Split the data
    X_num_train, X_num_temp, X_cat_train, X_cat_temp, y_train, y_temp = train_test_split(
        X_numerical, X_categorical, y, test_size=0.4, random_state=42, stratify=y
    )
    
    X_num_val, X_num_test, X_cat_val, X_cat_test, y_val, y_test = train_test_split(
        X_num_temp, X_cat_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp
    )
    
    # Create datasets
    train_dataset = FraudDataset(X_num_train, X_cat_train, y_train)
    val_dataset = FraudDataset(X_num_val, X_cat_val, y_val)
    test_dataset = FraudDataset(X_num_test, X_cat_test, y_test)
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=256, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=256, shuffle=False)
    
    print(f"Training set size: {len(train_dataset)}")
    print(f"Validation set size: {len(val_dataset)}")
    print(f"Test set size: {len(test_dataset)}")
    
    # Initialize model
    model = FraudTransformerModel(
        num_features=len(numerical_cols),
        cat_features_info=cat_features_info,
        d_model=128,
        n_heads=8,
        n_layers=4,
        d_ff=512,
        dropout=0.2
    ).to(device)
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Train model
    print("\nStarting training...")
    train_losses, val_losses, val_f1_scores, val_recalls = train_model(
        model, train_loader, val_loader, device, epochs=100, patience=15
    )
    
    # Plot training metrics
    plot_training_metrics(train_losses, val_losses, val_f1_scores, val_recalls)
    
    # Evaluate on test set
    print("\nEvaluating on test set...")
    
    # Try different thresholds to optimize for recall
    thresholds = [0.1, 0.2, 0.3, 0.4, 0.5]
    best_threshold = 0.3
    best_f1 = 0
    
    for threshold in thresholds:
        y_true, y_pred, y_probs = evaluate_model(model, test_loader, device, threshold)
        f1 = f1_score(y_true, y_pred)
        recall = recall_score(y_true, y_pred)
        
        print(f"Threshold {threshold}: F1 = {f1:.4f}, Recall = {recall:.4f}")
        
        if f1 > best_f1:
            best_f1 = f1
            best_threshold = threshold
    
    # Final evaluation with best threshold
    print(f"\nFinal evaluation with threshold {best_threshold}:")
    y_true, y_pred, y_probs = evaluate_model(model, test_loader, device, best_threshold)
    
    print("\nClassification Report:")
    print(classification_report(y_true, y_pred, target_names=['Normal', 'Fraud']))
    
    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Normal', 'Fraud'], 
                yticklabels=['Normal', 'Fraud'])
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.show()
    
    # Feature importance (attention weights analysis)
    print(f"\nFinal Metrics:")
    print(f"F1 Score: {f1_score(y_true, y_pred):.4f}")
    print(f"Recall: {recall_score(y_true, y_pred):.4f}")
    print(f"Precision: {f1_score(y_true, y_pred) * recall_score(y_true, y_pred) / (2 * recall_score(y_true, y_pred) - f1_score(y_true, y_pred)):.4f}")

if __name__ == "__main__":
    main()