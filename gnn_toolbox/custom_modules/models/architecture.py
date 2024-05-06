import torch
import torch.nn as nn
import os
print("Current Working Directory:", os.getcwd())
import torch.nn.functional as F
from torch_geometric.nn import GATConv, GCNConv
from torch_geometric.nn.conv.gcn_conv import gcn_norm
# from gnn_toolbox.custom_modules.models.model
# import gnn_toolbox
from gnn_toolbox.registry import register_architecture, registry
import os
from typing import Any, Dict, Union


@register_architecture("GCN")
class GCN(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, hidden_channels: int, num_layers: int = 2, dropout: float = 0.5, **kwargs):
        super().__init__()
        self.GCNConv = GCNConv(in_channels=in_channels, out_channels=hidden_channels)
        self.layers = nn.ModuleList([GCNConv(hidden_channels, hidden_channels) for _ in range(num_layers)])
        self.dropout = nn.Dropout(dropout)
        self.linear = nn.Linear(hidden_channels, out_channels)
        
    def forward(self, x, edge_index, **kwargs):
        x = self.GCNConv(x, edge_index)
        for layer in self.layers:
            x = F.relu(layer(x, edge_index))
            x = self.dropout(x)
        return self.linear(x)
    
    # def compute_loss(self, pred, true):
    #     return F.binary_cross_entropy_with_logits(pred, true), torch.sigmoid(pred)


@register_architecture('gcn2')
class GCN2(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers=2):
        super().__init__()
        self.norm = gcn_norm
        self.conv1 = GCNConv(in_channels, hidden_channels, normalize=False)
        self.conv2 = GCNConv(hidden_channels, out_channels, normalize=False)

    def reset_parameters(self):
        self.conv1.reset_parameters()
        self.conv2.reset_parameters()

    def forward(self, x, edge_index, edge_weight=None, **kwargs):
        # Normalize edge indices only once:
        if not kwargs.get('skip_norm', False):
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
    
    def _extract_model_info(self):
        layers = list(self.modules())
        for idx, m in enumerate(self.modules()):
            print(idx, '->', m)
    
# model = GCN(16, 16, 16)
# print(model.hparams)

# parameters = list(self.modules())
# model._extract_model_info()

MODEL_TYPE = Union[GCN, GCN2]

print(registry['architecture'])