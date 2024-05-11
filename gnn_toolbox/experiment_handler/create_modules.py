import logging
from inspect import signature

from torch_geometric.transforms import Compose
from torch_geometric.data import Dataset

from gnn_toolbox.registry import registry, get_from_registry


def create_transforms(configs):
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
        logging.info(f"Creating transform '{name}' with parameters: {params}")
        transforms.append(transform_cls(**params))

    return Compose(transforms)

def create_dataset(name: str, root: str, transforms=None, **kwargs) -> Dataset:
    dataset = get_from_registry("dataset", name, registry)
    transforms = create_transforms(transforms) if transforms else None
    if 'name' in signature(dataset).parameters.keys():
        return dataset(name = name, root=root, transform=transforms)
    return dataset(root=root, transform=transforms, **kwargs)

def create_model(model):
    architecture = get_from_registry("model", model['name'], registry)
    model = architecture(**model['params'])
    return model

def create_global_attack(attack_name):
    attack = get_from_registry("global_attack", attack_name, registry)
    return attack

def create_local_attack(attack_name):
    attack = get_from_registry("local_attack", attack_name, registry)
    return attack

def create_optimizer(optimizer_name, model, **kwargs):
    optimizer = get_from_registry("optimizer", optimizer_name, registry)
    return optimizer(model.parameters(), **kwargs)

def create_loss(loss_name, **kwargs):
    loss = get_from_registry("loss", loss_name, registry)
    if 'logits' and 'labels' in signature(loss).parameters.keys():
        return loss
    return loss(**kwargs)