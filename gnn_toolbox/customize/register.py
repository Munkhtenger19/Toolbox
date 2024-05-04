from typing import Any, Callable, Dict, Union
from functools import partial
from utils import DotDict
ModuleType = Any

registry: Dict[str, Dict[str, ModuleType]] = {
    "architecture": {},
    "attack":{},
    "dataset": {},
    "activation": {},
    "optimizer": {},
    "scheduler": {},
    "loss": {},
    "metric": {},
}

registry = DotDict(registry)

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


register_act = partial(register_module, "activation")
register_attack = partial(register_module, "attack")
register_optimizer = partial(register_module, "optimizer")
register_architecture = partial(register_module, "architecture")
