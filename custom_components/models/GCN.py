"""
Adapted from DeepRobust project: https://github.com/DSE-MSU/DeepRobust/blob/master/deeprobust/graph/defense_pyg/gcn.py
"""


import torch.nn as nn
import torch.nn.functional as F
import math
import torch
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module
from torch_geometric.nn import GCNConv
# from .base_model import BaseModel
from torch_sparse import coalesce, SparseTensor, matmul
from custom_components.utils import ensure_contiguousness

from gnn_toolbox.registration_handler.register_components import register_model

@register_model("GCN_DPR")
class GCN(nn.Module):

    def __init__(self, in_channels, hidden_channels, out_channels, nlayers=2, dropout=0.5, lr=0.01,
                with_bn=False, weight_decay=5e-4, with_bias=True):

        super(GCN, self).__init__()

        self.layers = nn.ModuleList([])
        if with_bn:
            self.bns = nn.ModuleList()

        if nlayers == 1:
            self.layers.append(GCNConv(in_channels, out_channels, bias=with_bias))
        else:
            self.layers.append(GCNConv(in_channels, hidden_channels, bias=with_bias))
            if with_bn:
                self.bns.append(nn.BatchNorm1d(hidden_channels))
            for i in range(nlayers-2):
                self.layers.append(GCNConv(hidden_channels, hidden_channels, bias=with_bias))
                if with_bn:
                    self.bns.append(nn.BatchNorm1d(hidden_channels))
            self.layers.append(GCNConv(hidden_channels, out_channels, bias=with_bias))

        self.dropout = dropout
        self.weight_decay = weight_decay
        self.lr = lr
        self.output = None
        self.best_model = None
        self.best_output = None
        self.with_bn = with_bn
        self.name = 'GCN'

    def forward(self, x, edge_index, edge_weight=None):
        x, edge_index, edge_weight = ensure_contiguousness(x, edge_index, edge_weight)
        for ii, layer in enumerate(self.layers):
            if edge_weight is not None:
                adj = SparseTensor.from_edge_index(edge_index, edge_weight, sparse_sizes=2 * x.shape[:1]).t()
                x = layer(x, adj)
            else:
                x = layer(x, edge_index)
            if ii != len(self.layers) - 1:
                if self.with_bn:
                    x = self.bns[ii](x)
                x = F.relu(x)
                x = F.dropout(x, p=self.dropout, training=self.training)
        return F.log_softmax(x, dim=1)

    def get_embed(self, x, edge_index, edge_weight=None):
        x, edge_index, edge_weight = ensure_contiguousness(x, edge_index, edge_weight)
        for ii, layer in enumerate(self.layers):
            if ii == len(self.layers) - 1:
                return x
            if edge_weight is not None:
                adj = SparseTensor.from_edge_index(edge_index, edge_weight, sparse_sizes=2 * x.shape[:1]).t()
                x = layer(x, adj)
            else:
                x = layer(x, edge_index)
            if ii != len(self.layers) - 1:
                if self.with_bn:
                    x = self.bns[ii](x)
                x = F.relu(x)
        return x

    def initialize(self):
        for m in self.layers:
            m.reset_parameters()
        if self.with_bn:
            for bn in self.bns:
                bn.reset_parameters()
