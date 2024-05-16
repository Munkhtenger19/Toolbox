import torch
import torch.nn as nn
import os
import torch.nn.functional as F
from torch_geometric.nn import GATConv, GCNConv, GCN, GAT
from torch_geometric.nn.conv.gcn_conv import gcn_norm
# from gnn_toolbox.custom_modules.models.model
# import gnn_toolbox
from gnn_toolbox.registry import register_model, registry
import os
from typing import Any, Dict, Union



@register_model("GCN")
class GCNWH(nn.Module):
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
    
    # def compute_loss(self, pred, true):
    #     return F.binary_cross_entropy_with_logits(pred, true), torch.sigmoid(pred)



@register_model('gcn2')
def GCN_(in_channels, out_channels, hidden_channels):
    return GCN(in_channels, out_channels, hidden_channels)

# @register_model('gat')
# def GAT

@register_model('gcn3')
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

# MODEL_TYPE = Union[GCN, GCN2]
from typing import Union, Tuple
class DenseGraphConvolution(nn.Module):
    """Dense GCN convolution layer for the FGSM attack that requires a gradient towards the adjacency matrix.
    """

    def __init__(self, in_channels: int, out_channels: int):
        """
        Parameters
        ----------
        in_channels : int
            Number of channels of the input
        out_channels : int
            Desired number of channels for the output (for trainable linear transform)
        """
        super().__init__()
        self._linear = nn.Linear(in_channels, out_channels, bias=False)

    def forward(self, arguments: Tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
        """Prediction based on input.

        Parameters
        ----------
        arguments : Tuple[torch.Tensor, torch.Tensor]
            Tuple with two elements of the attributes and dense adjacency matrix

        Returns
        -------
        torch.Tensor
            The new embeddings
        """
        x, adj_matrix = arguments

        x_trans = self._linear(x)
        return adj_matrix @ x_trans

import collections
from torch_sparse import coalesce, SparseTensor 
@register_model('DenseGCN')
class DenseGCN(nn.Module):
    """Dense two layer GCN for the FGSM attack that requires a gradient towards the adjacency matrix.
    """

    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 hidden_channels: int = 64,
                 activation: nn.Module = nn.ReLU(),
                 dropout: float = 0.5,
                 ** kwargs):
        """
        Parameters
        ----------
        n_features : int
            Number of attributes for each node
        n_classes : int
            Number of classes for prediction
        n_filters : int, optional
            number of dimensions for the hidden units, by default 80
        activation : nn.Module, optional
            Arbitrary activation function for the hidden layer, by default nn.ReLU()
        dropout : int, optional
            Dropout rate, by default 0.5
        """
        super().__init__()
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.out_channels = out_channels
        self.activation = activation
        self.dropout = dropout
        self.layers = nn.ModuleList([
            nn.Sequential(collections.OrderedDict([
                ('gcn_0', DenseGraphConvolution(in_channels=in_channels,
                                                out_channels=hidden_channels)),
                ('activation_0', self.activation),
                ('dropout_0', nn.Dropout(p=dropout))
            ])),
            nn.Sequential(collections.OrderedDict([
                ('gcn_1', DenseGraphConvolution(in_channels=hidden_channels,
                                                out_channels=out_channels)),
                ('softmax_1', nn.LogSoftmax(dim=1))
            ]))
        ])

    @ staticmethod
    def normalize_dense_adjacency_matrix(adj: torch.Tensor) -> torch.Tensor:
        """Normalizes the adjacency matrix as proposed for a GCN by Kipf et al. Moreover, it only uses the upper triangular
        matrix of the input to obtain the right gradient towards the undirected adjacency matrix.

        Parameters
        ----------
        adj: torch.Tensor
            The weighted undirected [n x n] adjacency matrix.

        Returns
        -------
        torch.Tensor
            Normalized [n x n] adjacency matrix.
        """
        adj_norm = torch.triu(adj, diagonal=1) + torch.triu(adj, diagonal=1).T
        adj_norm.data[torch.arange(adj.shape[0]), torch.arange(adj.shape[0])] = 1
        deg = torch.diag(torch.pow(adj_norm.sum(axis=1), - 1 / 2))
        adj_norm = deg @ adj_norm @ deg
        return adj_norm

    def forward(self, x: torch.Tensor, adjacency_matrix: Union[torch.Tensor, SparseTensor]) -> torch.Tensor:
        """Prediction based on input.

        Parameters
        ----------
        x : torch.Tensor
            Dense [n, d] tensor holding the attributes
        adjacency_matrix : torch.Tensor
            Dense [n, n] tensor for the adjacency matrix

        Returns
        -------
        torch.Tensor
            The predictions (after applying the softmax)
        """
        if isinstance(adjacency_matrix, SparseTensor):
            adjacency_matrix = adjacency_matrix.to_dense()
        adjacency_matrix = DenseGCN.normalize_dense_adjacency_matrix(adjacency_matrix)
        for layer in self.layers:
            x = layer((x, adjacency_matrix))
        return x