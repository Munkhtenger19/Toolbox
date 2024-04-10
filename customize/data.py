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
    gnn_benchmark_dataset
)
import torch_geometric.transforms as T
from config_def import cfg


dataset_registry = {  
    "Cora": Planetoid,
    "CiteSeer": Planetoid,
    "PubMed": Planetoid,
    "MUTAG": TUDataset,
    "PROTEINS": TUDataset,
}

transforms_registry = {
    # 'ToDevice': T.ToDevice(),
    'AddSelfLoops': T.AddSelfLoops(),
    'ToSparseTensor': T.ToSparseTensor(),
    'Constant': T.Constant(),
    'NormalizeFeatures': T.NormalizeFeatures(),
    # 'SVDFeatureReduction': T.SVDFeatureReduction(),
    # 'RemoveTrainingClasses': T.RemoveTrainingClasses(),
    # 'RandomNodeSplit': T.RandomNodeSplit(),
    # 'RandomLinkSplit': T.RandomLinkSplit(),
    # 'NodePropertySplit': T.NodePropertySplit(),
    # 'IndexToMask': T.IndexToMask(),
    # 'MaskToIndex': T.MaskToIndex(),
    # 'Pad': T.Pad(),
}

def register_dataset(dataset_name: str): 
    """Loads a dataset by name.

    Args:
        dataset_name (str): Name of the dataset to load.
        root (str): Root directory where the dataset is stored.

    Returns:
        Dataset: An instance of the requested dataset.

    Raises:
        ValueError: If an unknown dataset name is provided.
    """
    #should allow single dataset to be loaded
    if dataset_name not in dataset_registry:
        raise ValueError(f"Unknown dataset name: {dataset_name}")
        
    dataset_class = dataset_registry[dataset_name]
    # should support other params
    return lambda root: dataset_class(root, dataset_name)

def load_dataset_from_cfg():
    dataset_func = register_dataset(cfg.dataset.name)
    if cfg.dataset.transforms == 'None':
        return dataset_func(cfg.dataset.dir)
    transform_list = [transforms_registry[name] for name in cfg.dataset.transforms if name in transforms_registry]
    if not transform_list:
        raise ValueError(f"Unknown transform names: {set(cfg.dataset.transforms) - set(transforms_registry.keys())}")
    transform = T.Compose(transform_list)
    # TODO support other params such as pre_transform, pre_filter 
    # return dataset_func(cfg.dataset.dir, transform=transform)
    return dataset_func(cfg.dataset.dir)




# def register_loader(loader_name: str, dataset, **kwargs):
#     """Registers and creates a DataLoader based on the loader name and parameters."""

#     if loader_name not in loader_classes:
#         raise ValueError(f"Unknown loader name: {loader_name}")

#     loader_class = loader_classes[loader_name]

#     if loader_name == "ClusterData":
#         # Special handling for ClusterData
#         cluster_data = loader_class(dataset[0], **kwargs)
#         return ClusterLoader(cluster_data, **kwargs)  # Create ClusterLoader
#     else:
#         return loader_class(dataset, **kwargs)  # Create other loaders directly