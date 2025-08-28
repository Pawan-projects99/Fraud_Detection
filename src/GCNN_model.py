import numpy as np
import pandas as pd
from feast import FeatureStore
import torch.nn.functional as F
import torch
from torch_geometric.data import HeteroData
from torch_geometric.loader import LinkNeighborLoader
from torch.nn import Linear, ReLU, ModuleList
from torch_geometric.nn import HeteroConv, GCNConv, SAGEConv, to_hetero
from sklearn.metrics import classification_report, roc_auc_score, f1_score, average_precision_score

def build_fraud_graph():
    store = FeatureStore(repo_path = "./features")

    transactions = pd.read_parquet("./features/data/transactions.parquet")
    transactions = transactions[["user_id", "merchant_id", "amount", "timestamp", "is_fraud"]]
    transactions = transactions.rename(columns = {"timestamp": "event_timestamp"})

    graph_features = store.get_historical_features(
        entity_df = transactions,
        features = [
            "user_behaviour:avg_transaction",
            "user_behaviour:transaction_count_30d",
            "user_behaviour:decline_rate_60d",
            "merchant_risk_profile:fraud_rate",
            "merchant_risk_profile:avg_transaction_value"
        ]
    ).to_df()

    graph_data = pd.merge(transactions, graph_features, on=["user_id", "merchant_id", "event_timestamp"])
    print(graph_data)

    graph = HeteroData()

    user_ids = graph_data["user_id"].unique()
    merchant_ids = graph_data["merchant_id"].unique()
    user_ids_map = {uid:i for i, uid in enumerate(user_ids)}
    merchant_ids_map = {mid: i+ len(user_ids) for i,mid in enumerate(merchant_ids)}

    user_features = []
    for uid in user_ids:
        user_data = graph_data[graph_data["user_id"]==uid].iloc[0]
        user_features.append([
            user_data["avg_transaction"],
            user_data["transaction_count_30d"],
            user_data["decline_rate_60d"]
        ])
    graph["user"].x = torch.tensor(user_features, dtype = torch.float)

    merchant_features = []
    for mid in merchant_ids:
        merchant_data = graph_data[graph_data["merchant_id"]==mid].iloc[0]
        merchant_features.append([
            merchant_data["fraud_rate"],
            merchant_data["avg_transaction_value"]
        ])
    graph["merchant"].x = torch.tensor(merchant_features, dtype = torch.float)

    edge_index = []
    edge_attr = []
    for _,row in graph_data.iterrows():
        src = user_ids_map[row["user_id"]]
        dst = merchant_ids_map[row["merchant_id"]]

        hour = pd.to_datetime(row["event_timestamp"]).hour
        is_night = 1 if (hour>=22 or hour<=4) else 0

        edge_index.append([src, dst])
        edge_attr.append([row["amount_x"], is_night])

    graph["user", "transacts_with", "merchant"].edge_index = torch.tensor(edge_index, dtype = torch.long).t().contiguous()
    graph["user", "transacts_with", "merchant"].edge_attr = torch.tensor(edge_attr, dtype = torch.float)
    graph["user", "transacts_with", "merchant"].y = torch.tensor(graph_data["is_fraud_x"].values, dtype = torch.float)
    graph["user", "transacts_with", "merchant"].time = torch.tensor(graph_data["event_timestamp"].values.astype("datetime64[s]").astype(np.int64), dtype = torch.long)
    graph["user", "transacts_with", "merchant"].edge_index = torch.tensor(edge_index, dtype = torch.long)[:,[1,0]].t().contiguous()

    return graph, graph_data


class FraudGNN(torch.nn.Module):
    print("Fraud_GNN model executing")
    def __init__(self, hidden_channels, num_layers):
        super().__init__()

        self.user_lin = torch.nn.Linear(3, hidden_channels)
        self.merchant_lin = torch.nn.Linear(2, hidden_channels)
        self.edge_lin = torch.nn.Linear(2, hidden_channels)

        self.convs = torch.nn.ModuleList()
        for _ in range(num_layers):
            conv = HeteroConv({
                ("user", "transacts_with", "merchant"): SAGEConv((-1,-1), 128),
                ("merchant", "transacts_with", "user"): SAGEConv((-1,-1), 128)
            }, aggr="mean")
            self.convs.append(conv)
        
        self.classifier = torch.nn.Sequential(
            Linear(3*hidden_channels, hidden_channels),
            RelU(),
            torch.nn.Dropout(0.5),
            Linear(hidden_channels, 1)
        )
    def forward(self, data : HeteroData):
        x_dict = {
            "user": F.relu(self.user_lin(data["user"].x)),
            "merchant": F.relu(self.merchant_lin(data["merchant"].x))
        }
        edge_attr_dict = {
            ("user", "transacts_with", "merchant"): F.relu(self.edge_lin(data["user", "transacts_with", "merchant"].edge_attr))
        }
    # Message passing through Conv layers
        for conv in self.convs:
            x_dict = conv(x_dict, data.edge_index_dict , edge_attr_dict)
            x_dict = { key: F.relu(x) for key,x in x_dict.items()}
    # Updated edge_features(with relation between nodes) after passing through conv layers
        edge_features = []
        edge_index = data["user", "transacts_with", "merchant"].edge_index.t() # here the (2, num_edges) will converst back to (num_edges, 2)
        edge_attr = edge_attr_dict["user", "transacts_with", "merchant"].edge_attr

        for i in range(edge_index.shape[0]):
            src, dst = edge_index[i]
            user_feat = x_dict["user"][src]
            merchant_feat = x_dict["merchant"][dst]
            edge_feat = edge_attr[i]

            edge_features.append(torch.cat([user_feat, merchant_feat, edge_feat], dim = -1))
        
        return self.classifier(torch.stack(edge_features)).squeeze()
        print(((torch.stack(edge_features)).squeeze()).shape)

def train_gnn(graph : HeteroData, graph_data : pd.DataFrame):
    print("Excecuting train_gnn")
    timestamps = graph_data["event_timestamp"]
    sorted_indices = np.argsort(timestamps.to_numpy())

    train_cutoff = int(0.7 * len(sorted_indices))
    eval_cutoff = int(0.85 * len(sorted_indices))

    train_mask = torch.zeros(len(graph_data), dtype = torch.bool)
    eval_mask = torch.zeros(len(graph_data), dtype = torch.bool)
    test_mask = torch.zeros(len(graph_data), dtype = torch.bool)

    train_mask[sorted_indices[:train_cutoff]] = True
    eval_mask[sorted_indices[train_cutoff:eval_cutoff]]= True
    test_mask[sorted_indices[eval_cutoff:]]= True
    #Load data into the masks
    train_loader = LinkNeighborLoader(
        data= graph,
        num_neighbors = [10,5],
        edge_label_index = (("user","transacts_with", "merchant"), graph["user", "transacts_with", "merchant"].edge_index),
        edge_label = graph["user", "transacts_with", "merchant"].y,
        edge_label_time = graph["user", "transacts_with", "merchant"].time,
        time_attr = "time",
        batch_size = 32,
        shuffle = True,
        temporal_strategy = "last"
    )
    eval_loader = LinkNeighborLoader(
        data= graph,
        num_neighbors = [10,5],
        edge_label_index = (("user","transacts_with", "merchant"), graph["user", "transacts_with", "merchant"].edge_index),
        edge_label = graph["user", "transacts_with", "merchant"].y,
        edge_label_time = graph["user", "transacts_with", "merchant"].time,
        time_attr = "time",
        batch_size = 32,
        shuffle = False,
        temporal_strategy = "last"
    )
    test_loader = LinkNeighborLoader(
        data= graph,
        num_neighbors = [10,5],
        edge_label_index =(("user","transacts_with", "merchant"), graph["user", "transacts_with", "merchant"].edge_index),
        edge_label = graph["user", "transacts_with", "merchant"].y,
        edge_label_time = graph["user", "transacts_with", "merchant"].time,
        time_attr = "time",
        batch_size = 32,
        shuffle = False,
        temporal_strategy = "last"
    )
    #initialize the model 
    device = torch.device("cpu")    
    model = FraudGNN(hidden_channels = 128, num_layers = 3)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay = 1e-5)

    def focal_loss(pred, target, alpha = 0.25, gamma = 2):
        bce = F.binary_cross_entropy_with_logits(pred, target, reduction = "none")
        pt = torch.exp(-bce) # to get the probability from logits, log(probability)
        loss = alpha * (1-pt) ** gamma * bce
        return loss.mean()

    # training loop
    def train():
        model.train()
        total_loss = 0
        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            pred = model(batch)
            loss = focal_loss(pred, batch["user", "transacts_with", "merchant"].y)
            loss.backward()
            optimizer.step()
            total_loss += float(loss)
        return total_loss/len(train_loader)

    @torch.no_grad() # do not calculate gradients for test data
    def test(loader):
        model.eval()
        preds, targets = [], []
        for batch in loader:
            pred = model(batch).sigmoid # tensors from each batch
            preds.append(pred) # list of tensors from each batch
            targets.append(batch["user", "transacts_with", "merchant"].y)

        preds = torch.cat(preds).numpy()# concat the tensors into a single tensor and convert into numpy array for metric calculations.
        targets = torch.cat(targets).numpy()

        auroc = roc_auc_score(targets, preds)
        aps = average_precision_score(targets, preds)
        topk_precision = targets[ap.argsort(preds)[-1000:]].mean()
        return auroc, aps, topk_precision

    for epoch in range(1,101):
        loss = train()
        val_auroc, val_aps, val_topk_precision = test(eval_loader)
        best_aps = 0
        print(f" epoch:{epoch:03d}, auroc_score: {val_auroc:.4f}, average_precision_score: {val_aps:.4f}, top_100_precision: {val_topk_precision:.4f}")
        if (val_aps > best_aps):
            best_aps = val_aps
            torch.save(model.state_dict(), "./models/fraud_gnn.pt")
    
    model.load_state_dict(torch.load("./models/fraud_gnn.pt"))
    test_auroc, test_aps, test_topk_precision = test(test_loader)

    print(f"auroc_score: {test_auroc:.4f}, average_precision_score: {test_aps:.4f}, top_100_precision: {test_topk_precision:.4f}")

if __name__ == "__main__":
    graph, graph_data = build_fraud_graph()
    print("garph builed successfully")
    model  = train_gnn(graph, graph_data)
