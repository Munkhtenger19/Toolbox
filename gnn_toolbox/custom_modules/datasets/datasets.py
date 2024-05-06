from torch_geometric.datasets import (
    Planetoid,
    TUDataset,
    CoraFull,
    Amazon,
    Coauthor,
    PPI,
    Reddit,
    Reddit2,
    WikiCS,
    GNNBenchmarkDataset
)

from gnn_toolbox.registry import register_dataset


# dataset_registry = {  
#     "cora": Planetoid,
#     "citeseer": Planetoid,
#     "pubmed": Planetoid,
#     "mutag": TUDataset,
#     "proteins": TUDataset,
#     "PPI" : PPI,
#     "cora_full": CoraFull,
#     "amazon": Amazon,
#     "coauthor": Coauthor,
    
# }

register_dataset('Cora', Planetoid)
register_dataset('Citeseer', Planetoid)
register_dataset('Pubmed', Planetoid)
register_dataset('CoraFull', CoraFull)
register_dataset('PPI', PPI)
register_dataset('Amazon', Amazon)
register_dataset('Coauthor', Coauthor)
register_dataset('Reddit', Reddit)
register_dataset('gnn_benchmark_dataset', GNNBenchmarkDataset)



