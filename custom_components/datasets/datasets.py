from torch_geometric.datasets import (
    Planetoid,
    CoraFull,
    Amazon,
    Coauthor,
    CitationFull,
    Reddit,
    GNNBenchmarkDataset,
)
from ogb.nodeproppred import PygNodePropPredDataset
from gnn_toolbox.registration_handler.register_components import register_dataset


register_dataset('Cora', lambda root, transform=None, **kwargs: Planetoid(root, name='Cora', transform=transform, **kwargs))
register_dataset('Citeseer', lambda root: Planetoid(root, name='Citeseer'))
register_dataset('PubMed', lambda root: Planetoid(root, name='PubMed'))
register_dataset('CoraFull', CoraFull)
register_dataset('CS', lambda root: Coauthor(root, name='CS'))
register_dataset('Physics', lambda root: Coauthor(root, name='Physics'))
register_dataset('Computers', lambda root: Amazon(root, name='Computers'))
register_dataset('Photo', lambda root: Amazon(root, name='Photo'))
register_dataset('CoraCitationFull',  lambda root: CitationFull(root, name='Cora'))
register_dataset('DBLP', lambda root: CitationFull(root, name='Cora'))
register_dataset('PubMedCitationFull', lambda root: CitationFull(root, name='PubMed'))
register_dataset('Reddit', Reddit)
register_dataset('GNNBenchmarkDataset', GNNBenchmarkDataset)
register_dataset('ogbn-arxiv', PygNodePropPredDataset)



