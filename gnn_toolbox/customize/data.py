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
# from customize.config_def import cfg


dataset_registry = {  
    "cora": Planetoid,
    "citeseer": Planetoid,
    "pubmed": Planetoid,
    "mutag": TUDataset,
    "proteins": TUDataset,
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
    if dataset_name not in dataset_registry.keys():
        raise ValueError(f"Unknown dataset name: {dataset_name}")
        
    dataset_class = dataset_registry[dataset_name]
    # should support other params
    return lambda root: dataset_class(root = root, name = dataset_name)
    # return dataset_class(root = root, name = dataset_name)

    
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

def load_dataset(dataset):
    dataset_func = register_dataset(dataset.name)
    return dataset_func(dataset.path)

def set_dataset_info(dataset):
    r"""Set global dataset information.

    Args:
        dataset: PyG dataset object

    """
    # get dim_in and dim_out
    try:
        cfg.model.params.in_channels = dataset.num_features
        # cfg.model.params.in_channels = dataset._data.x.shape[1]
    except Exception:
        cfg.model.params.in_channels = 1
    try:
        if cfg.dataset.task_type == 'classification':
            cfg.model.params.out_channels = dataset.num_classes
            # cfg.model.params.out_channels = torch.unique(dataset._data.y).shape[0]
        else:
            cfg.model.params.out_channels = dataset._data.y.shape[1]
            # cfg.share.dim_out = dataset._data.y.shape[1]
    except Exception:
        cfg.model.params.out_channels = 1

    # count number of dataset splits
    cfg.trainer.num_splits += sum('val' in key or 'test' in key for key in dataset._data.keys())
    print('cfg.trainer.num_splits', cfg.trainer.num_splits)


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

register_dataset