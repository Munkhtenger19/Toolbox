import torch
from torch_geometric.nn import GATConv, GCNConv
from torch_geometric.nn.conv.gcn_conv import gcn_norm
from model import BaseModel
from register import register_architecture

# class ModelArchitecture(nn.Module):
#     def __init__(self, model_name, num_classes, pretrained=True):
#         self.model_name = model_name
#         self.num_classes = num_classes
#         self.pretrained = pretrained

@register_architecture('gcn')
class GCN(BaseModel):
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
    
model = GCN(16, 16, 16)
print(model.hparams)
# parameters = list(self.modules())

# model._extract_model_info()