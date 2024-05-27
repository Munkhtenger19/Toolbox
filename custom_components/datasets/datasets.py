from torch_geometric.datasets import (
    Planetoid,
    CoraFull,
    Amazon,
    Coauthor,
    CitationFull,
    Reddit,
    # GNNBenchmarkDataset,
)
from ogb.nodeproppred import PygNodePropPredDataset
from gnn_toolbox.registration_handler.register_components import register_dataset

# Registering datasets: register_dataset(name, dataset)

register_dataset('Cora', lambda root, transform, **kwargs: Planetoid(root, name='Cora', transform=transform, **kwargs))

register_dataset('Citeseer', lambda root, transform, **kwargs: Planetoid(root, name='Citeseer', transform=transform, **kwargs))

register_dataset('PubMed', lambda root, transform, **kwargs: Planetoid(root, name='PubMed', transform=transform, **kwargs))

register_dataset('CoraFull', lambda root, transform, **kwargs: CoraFull(root, transform=transform, **kwargs))

register_dataset('CS', lambda root, transform, **kwargs: Coauthor(root, name='CS', transform=transform, **kwargs))

register_dataset('Physics', lambda root, transform, **kwargs: Coauthor(root, name='Physics', transform=transform, **kwargs))

register_dataset('Computers', lambda root, transform, **kwargs: Amazon(root, name='Computers', transform=transform, **kwargs))

register_dataset('Photo', lambda root, transform, **kwargs: Amazon(root, name='Photo', transform=transform, **kwargs))

register_dataset('CoraCitationFull',  lambda root, transform, **kwargs: CitationFull(root, name='Cora', transform=transform, **kwargs))

register_dataset('DBLP', lambda root, transform, **kwargs: CitationFull(root, name='Cora', transform=transform, **kwargs))

register_dataset('PubMedCitationFull', lambda root, transform, **kwargs: CitationFull(root, name='PubMed', transform=transform, **kwargs))

register_dataset('Reddit', lambda root, transform, **kwargs: Reddit(root, transform=transform, **kwargs))

# register_dataset('GNNBenchmarkDataset', lambda root, transform, **kwargs: GNNBenchmarkDataset(root, transform=transform, **kwargs))

register_dataset('PygNodePropPredDataset', lambda root, transform, **kwargs: PygNodePropPredDataset(root, name = 'ogbn-arxiv', transform=transform, **kwargs))



