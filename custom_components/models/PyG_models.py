import torch
import torch.nn as nn
import os
import torch.nn.functional as F
from torch_geometric.nn import GATConv, GCNConv, GCN, GAT
from torch_geometric.nn.conv.gcn_conv import gcn_norm

from gnn_toolbox.registry import register_model
import os

@register_model("GCN")
class GCN(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, hidden_channels: int, num_layers: int = 2, dropout: float = 0.5, **kwargs):
        super().__init__()
        self.GCNConv1 = GCNConv(in_channels=in_channels, out_channels=hidden_channels)
        self.layers = nn.ModuleList([GCNConv(hidden_channels, hidden_channels) for _ in range(num_layers)])
        self.dropout = nn.Dropout(dropout)
        self.GCNConv2 = GCNConv(in_channels=hidden_channels, out_channels=out_channels)
        
    def forward(self, x, edge_index, edge_weight, **kwargs):
        x = self.GCNConv1(x, edge_index, edge_weight).relu()
        x = F.dropout(x, training=self.training)
        x = self.GCNConv2(x, edge_index, edge_weight)
        return x
    

@register_model('GCN_PyG')
class GCNWrapper(torch.nn.Module):
    def __init__(self, in_channels, out_channels, hidden_channels):
        super().__init__()
        self.gcn = GCN(in_channels, hidden_channels, out_channels)

    def forward(self, x, edge_index, edge_weight=None):
        return self.gcn(x, edge_index, edge_weight)

@register_model('GAT_PyG')
class GATWrapper(torch.nn.Module):
    def __init__(self, in_channels, out_channels, hidden_channels):
        super().__init__()
        self.gat = GAT(in_channels, hidden_channels, out_channels)

    def forward(self, x, edge_index):
        return self.gat(x, edge_index)


@register_model('GCN2')
class GCN2(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super().__init__()
        self.norm = gcn_norm
        self.conv1 = GCNConv(in_channels, hidden_channels, normalize=False)
        self.conv2 = GCNConv(hidden_channels, out_channels, normalize=False)

    def reset_parameters(self):
        self.conv1.reset_parameters()
        self.conv2.reset_parameters()

    def forward(self, x, edge_index, edge_weight=None, **kwargs):
        edge_index, edge_weight = self.norm(
            edge_index,
            edge_weight,
            num_nodes=x.size(0),
            add_self_loops=True,
        )

        x = self.conv1(x, edge_index, edge_weight).relu()
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index, edge_weight)
        return x