import torch_geometric.transforms as T
from torch_geometric.transforms import (
    AddSelfLoops,
    Constant,
    NormalizeFeatures,
    ToDevice,
    ToUndirected,
)
from gnn_toolbox.registry import register_transform

register_transform("AddSelfLoops", AddSelfLoops)
register_transform("Constant", Constant)
register_transform("NormalizeFeatures", NormalizeFeatures)
register_transform("ToDevice", ToDevice)
register_transform("ToUndirected", ToUndirected)
