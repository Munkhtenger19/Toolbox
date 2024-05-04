from config_def import cfg
from customize.data import load_dataset_from_cfg, set_dataset_info
from config_def import cfg
from torch_geometric.loader import (
    DataLoader, 
    DataListLoader, 
    NeighborSampler, 
    ClusterLoader, 
    GraphSAINTNodeSampler, 
    GraphSAINTEdgeSampler, 
    GraphSAINTRandomWalkSampler, 
    ShaDowKHopSampler
)


loader_registry = {
    "DataLoader": DataLoader,
    "DataListLoader": DataListLoader,
    "NeighborSampler": NeighborSampler,
    "ClusterLoader": ClusterLoader,
    "GraphSAINTNodeSampler": GraphSAINTNodeSampler,
    "GraphSAINTEdgeSampler": GraphSAINTEdgeSampler,
    "GraphSAINTRandomWalkSampler": GraphSAINTRandomWalkSampler,
    "ShaDowKHopSampler": ShaDowKHopSampler,
}

def register_loader(loader_name: str, dataset, **kwargs):
    """Registers a DataLoader by name.

    Args:
        loader_name (str): Name of the DataLoader to register.
        dataset (Dataset): The dataset to load.
        **kwargs: Additional arguments for the DataLoader.

    Returns:
        DataLoader: An instance of the DataLoader.
    """
    if loader_name not in loader_registry:
        raise ValueError(f"Unknown loader name: {loader_name}")

    loader_class = loader_registry[loader_name]
    return loader_class(dataset, **kwargs)

# def load_loader_from_cfg():
#     """Loads a DataLoader from the configuration.

#     Returns:
#         DataLoader: The DataLoader instance.
#     """
#     dataset = load_dataset_from_cfg()
#     loader = register_loader(cfg.loader.name, dataset)
#     cfg.model.params.in_channels = dataset.num_features
#     cfg.model.params.out_channels = dataset.num_classes
#     return loader

def create_loader():
    # ! not good loaders, need to get correct dataset masks,
    # ! check if the dataset is a OGB dataset or a custom dataset
    """Creates data loaders for train, validation, and test splits."""

    dataset = load_dataset_from_cfg()
    set_dataset_info(dataset)
    
    # train loader
    if cfg.dataset.prediction_target == 'graph':
        id = dataset.data['train_graph_index']
        loaders = [
            # get_loader(dataset[id], cfg.train.sampler, cfg.train.batch_size, shuffle=True)
            register_loader(cfg.loader.name, dataset[id], **cfg.loader.params, shuffle=True)
        ]
        delattr(dataset.data, 'train_graph_index')
    else:
        loaders = [
            register_loader(cfg.loader.name, dataset, **cfg.loader.params, shuffle=True)
        ]

    # val and test loaders
    for i in range(cfg.trainer.num_splits - 1):
        if cfg.dataset.prediction_target == 'graph':
            split_names = ['val_graph_index', 'test_graph_index']
            id = dataset.data[split_names[i]]
            loaders.append(
                register_loader(cfg.loader.name, dataset, **cfg.loader.params, shuffle=False))
            delattr(dataset.data, split_names[i])
        else:
            loaders.append(
                register_loader(cfg.loader.name, dataset, **cfg.loader.params, shuffle=False))

    # for split in ["train", "val", "test"]:
    #     shuffle = split == "train"

    #     if cfg.dataset.task == "graph":
    #         # Graph-level: Assume split_cfg has "graph_index" attribute
    #         data = dataset[dataset.data[f"{split}_graph_index"]]
    #     else:
    #         # Node-level or custom: Assume full dataset is used
    #         data = dataset

    #     loader = register_loader(cfg.loader.name, data, **cfg.loader.params, shuffle=shuffle)
    #     loaders.append(loader)

    return loaders

