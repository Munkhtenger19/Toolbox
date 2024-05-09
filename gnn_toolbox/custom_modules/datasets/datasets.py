from torch_geometric.datasets import (
    Planetoid,
    TUDataset,
    CoraFull,
    Amazon,
    Coauthor,
    PPI,
    Reddit,
    GNNBenchmarkDataset
)
from ogb.nodeproppred import PygNodePropPredDataset
from gnn_toolbox.registry import register_dataset


register_dataset('Cora', Planetoid)
register_dataset('Citeseer', Planetoid)
register_dataset('Pubmed', Planetoid)
register_dataset('CoraFull', CoraFull)
register_dataset('PPI', PPI)
register_dataset('Amazon', Amazon)
register_dataset('Coauthor', Coauthor)
register_dataset('Reddit', Reddit)
register_dataset('GNNBenchmarkDataset', GNNBenchmarkDataset)
register_dataset('ogb-arxiv', PygNodePropPredDataset)



