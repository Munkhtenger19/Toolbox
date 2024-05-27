"""
Adapted from DeepRobust project: https://github.com/DSE-MSU/DeepRobust/blob/master/deeprobust/graph/defense_pyg/gat.py
"""


import torch.nn as nn
import torch.nn.functional as F
import math
import torch
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module
from torch_geometric.nn import GATConv
# from .mygat_conv import GATConv
# from .base_model import BaseModel

from gnn_toolbox.registration_handler.register_components import register_model

@register_model("GAT_DPR")
class GAT(nn.Module):

    def __init__(self, in_channels, hidden_channels, out_channels, heads=8, output_heads=1, dropout=0.5, lr=0.01,
            nlayers=2, with_bn=False, weight_decay=5e-4, with_bias=True):

        super(GAT, self).__init__()

        # assert device is not None, "Please specify 'device'!"

        self.convs = nn.ModuleList([])
        if with_bn:
            self.bns = nn.ModuleList([])
            self.bns.append(nn.BatchNorm1d(hidden_channels*heads))

        self.convs.append(GATConv(
            in_channels,
            hidden_channels,
            heads=heads,
            dropout=dropout,
            bias=with_bias))

        for i in range(nlayers-2):
            self.convs.append(GATConv(hidden_channels*heads,
                hidden_channels, heads=heads, dropout=dropout, bias=with_bias))
            if with_bn:
                self.bns.append(nn.BatchNorm1d(hidden_channels*heads))

        self.convs.append(GATConv(
            hidden_channels * heads,
            out_channels,
            heads=output_heads,
            concat=False,
            dropout=dropout,
            bias=with_bias))

        self.dropout = dropout
        self.weight_decay = weight_decay
        self.lr = lr
        self.output = None
        self.best_model = None
        self.best_output = None
        self.name = 'GAT'
        self.with_bn = with_bn

    def forward(self, x, edge_index, edge_weight=None):
        for ii, conv in enumerate(self.convs[:-1]):
            x = F.dropout(x, p=self.dropout, training=self.training)
            x = conv(x, edge_index, edge_weight)
            if self.with_bn:
                x = self.bns[ii](x)
                x = F.elu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.convs[-1](x, edge_index, edge_weight)
        return F.log_softmax(x, dim=1)

    def get_embed(self, x, edge_index, edge_weight=None):
        for ii, conv in enumerate(self.convs[:-1]):
            x = F.dropout(x, p=self.dropout, training=self.training)
            x = conv(x, edge_index, edge_weight)
            if self.with_bn:
                x = self.bns[ii](x)
                x = F.elu(x)
        return x

    def initialize(self):
        for conv in self.convs:
            conv.reset_parameters()
        if self.with_bn:
            for bn in self.bns:
                bn.reset_parameters()