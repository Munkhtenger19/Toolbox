import logging
from inspect import signature
from typing import Dict
from torch_geometric.transforms import Compose
from torch_geometric.data import Dataset

from gnn_toolbox.registry import registry, get_from_registry


def create_transforms(configs: list):
    """Create a list of transforms from a configuration.

    Args:
        configs (str): Given configuration for transforms
    Raises:
        ValueError: Raised when transform configurations are not provided as a list, do not have a 'name' key, or the transform is not found in the registry.

    Returns:
        Compose: Composed transforms
    """
    if not isinstance(configs, list):
        raise ValueError("Transform configurations should be provided as a list.")

    transforms = []
    for cfg in configs:
        name = cfg.get("name")
        if not name:
            raise ValueError("Each transform configuration must have a 'name' key.")

        params = cfg.get("params", {})
        transform_cls = get_from_registry("transform", name, registry)
        if transform_cls is None:
            raise ValueError(f"Transform '{name}' not found in the registry.")
        logging.debug(f"For dataset, creating transform '{name}' with parameters: {params}")
        transforms.append(transform_cls(**params))

    return Compose(transforms)

def create_dataset(name: str, root: str, transforms: list = None, **kwargs) -> Dataset:
    """Create a dataset from a given configuration.

    Args:
        name (str): name of dataset
        root (str): path for storing the dataset
        transforms (optional): transforms given in a list. Defaults to None.

    Returns:
        Dataset: instance of the dataset
    """
    dataset = get_from_registry("dataset", name, registry)
    transforms = create_transforms(transforms) if transforms else None
    params = kwargs.get("params", {})
    if 'name' in signature(dataset).parameters.keys():
        return dataset(name = name, root=root, transform=transforms, **params)
    return dataset(root=root, transform=transforms, **params)

def create_model(model: Dict):
    """
    Create a model from a given configuration.

    Args:
        model (Dict): configuration for the model

    Returns:
        model: instance of the model
    """
    architecture = get_from_registry("model", model['name'], registry)
    model = architecture(**model['params'])
    return model

def create_global_attack(attack_name: str):
    """
    Create a global attack from a given configuration.

    Args:
        attack_name (str): name of the attack

    Returns:
        attack: instance of the global attack
    """
    attack = get_from_registry("global_attack", attack_name, registry)
    return attack

def create_local_attack(attack_name: str):
    """
    Create a local attack from a given configuration.

    Args:
        attack_name (str): name of the attack

    Returns:
        attack: instance of the local attack
    """
    attack = get_from_registry("local_attack", attack_name, registry)
    return attack

def create_optimizer(optimizer_name, model, **kwargs):
    """
    Create an optimizer from a given configuration.

    Args:
        optimizer_name (str): name of the optimizer
        model: instance of the model
        **kwargs: additional keyword arguments for the optimizer

    Returns:
        optimizer: instance of the optimizer
    """
    optimizer = get_from_registry("optimizer", optimizer_name, registry)
    return optimizer(model.parameters(), **kwargs)

def create_loss(loss_name: str, **kwargs):
    """
    Create a loss function from a given configuration.

    Args:
        loss_name (str): name of the loss function
        **kwargs: additional keyword arguments for the loss function

    Returns:
        loss: instance of the loss function
    """
    loss = get_from_registry("loss", loss_name, registry)
    if 'logits' and 'labels' in signature(loss).parameters.keys():
        return loss
    return loss(**kwargs)