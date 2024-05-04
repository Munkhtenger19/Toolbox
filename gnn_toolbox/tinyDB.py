from sacred.observers import TinyDbObserver
from sacred import Experiment
ex = Experiment("Run_experiments")
# ex.observers.append(TinyDbObserver('my_runs'))
from torch_geometric.nn import GCN
from torch_geometric.datasets import KarateClub, Planetoid
import mlflow
from mlflow import MlflowClient
import torch.nn.functional as F
from torch.optim import Adam
import torch
from mlflow.models.signature import infer_signature

# client = MlflowClient("http://127.0.0.1:8080")
mlflow.set_tracking_uri("http://127.0.0.1:8080")
def train(model, data, epochs=200, lr=0.01, weight_decay=5e-4):
    model.train()
    optimizer = Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    for _ in range(epochs):
        optimizer.zero_grad()
        pred = model(data.x, data.edge_index, data.edge_weight)
        loss = F.cross_entropy(pred[data.train_mask], data.y[data.train_mask])
        loss.backward()
        optimizer.step()

def accuracy(pred, y, mask):
    return (pred.argmax(-1)[mask] == y[mask]).float().mean()

@torch.no_grad()
def test(model, data):
    model.eval()
    sample_input = data[data.train_mask][:1]
    pred = model(data.x, data.edge_index, data.edge_weight)
    signature = infer_signature(sample_input.numpy(), pred.numpy())
    return float(accuracy(pred, data.y, data.test_mask)), signature

@ex.automain
def main():
    experiment_tags = {
        "mlflow.runName": "Apple_Models",
        "mlflow.user": "mlflow",
        "mlflow.source.type": "LOCAL",
        "mlflow.source.name": "main",
        "mlflow.log-model.history": "true",
        "mlflow.log-model.meta": "true",
        "mlflow.source.git.commit": "main",
    }
    # produce_apples_experiment = client.create_experiment(
    # name="Apple_Models2", tags=experiment_tags
    # )
    
    # set current active experiment
    apple_experiment = mlflow.set_experiment("Apple_Models2")
    
    # 
    
    run_name = "apples_rf_test"
    with mlflow.start_run(run_name=run_name) as run:
        dataset = Planetoid(root='datasets', name='Cora')
        data = dataset[0]
        GNN = GCN(in_channels = data.num_features, hidden_channels=16, num_layers=2, dropout=0.5, out_channels=dataset.num_classes)
        # train(GNN, data)
        GNN.train()
        optimizer = Adam(GNN.parameters(), lr=0.0001, weight_decay=5e-4)
        for epoch in range(300):
            optimizer.zero_grad()
            pred = GNN(data.x, data.edge_index, data.edge_weight)
            loss = F.cross_entropy(pred[data.train_mask], data.y[data.train_mask])
            loss.backward()
            optimizer.step()
            mlflow.log_metric("loss", loss, step=epoch)
            # mlflow.log_metric("accuracy", avg_accuracy, step=epoch)
        GNN.eval()
        sample_input = data.x[data.train_mask][:1]
        with torch.no_grad():
            pred = GNN(data.x, data.edge_index, data.edge_weight)
        signature = infer_signature(sample_input.numpy(), pred.numpy())
        acc = float(accuracy(pred, data.y, data.test_mask))
        mlflow.pytorch.log_model(GNN, "GNNN", signature=signature)
        mlflow.log_metric("accuracy", acc.item())
        print('accuracy:', acc.item())
    
    