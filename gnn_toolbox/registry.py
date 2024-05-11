from typing import Any, Callable, Dict, Union
from functools import partial
from gnn_toolbox.custom_modules import *
ModuleType = Any

registry: Dict[str, Dict[str, ModuleType]] = {
    "model": {},
    "global_attack":{},
    "local_attack":{},
    "dataset": {},
    "transform": {},
    "optimizer": {},
    "loss": {},
}

def register_module(category: str, key: str, module: ModuleType = None) -> Union[Callable, None]:
    """
    Registers a module.

    Args:
        category (str): The category of the module (e.g., "act", "node_encoder").
        key (str): The name of the module.
        module (any, optional): The module. If set to None, will return a decorator.
    """
    if category not in registry:
        raise ValueError(f"Category '{category}' is not valid. Please choose from {list(registry.keys())}.")
        
    if module is not None:
        if key in registry[category]:
            raise KeyError(f"Module with '{key}' already defined in category '{category}'")
        registry[category][key] = module
        return

    def register_by_decorator(module):
        register_module(category, key, module)
        return module

    return register_by_decorator

def get_from_registry(category: str, key: str, registry: Dict[str, Dict[str, ModuleType]], default: Any = None) -> Any:
    """Retrieve a module from the registry safely with a fallback."""
    if category not in registry:
        raise ValueError(f"Category '{category}' is not recognized. Available categories: {list(registry.keys())}")

    category_registry = registry[category]
    if key in category_registry:
        return category_registry[key]
    else:
        if default is not None:
            return default
        else:
            raise KeyError(f"Module '{key}' not found in category '{category}'. Available options: {list(category_registry.keys())}")

register_model = partial(register_module, "model")
register_global_attack = partial(register_module, "global_attack")
register_local_attack = partial(register_module, "local_attack")
register_dataset = partial(register_module, "dataset")
register_transform = partial(register_module, "transform")
register_optimizer = partial(register_module, "optimizer")
register_loss = partial(register_module, "loss")