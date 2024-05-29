from functools import partial
from typing import Any, Callable, Dict, Union
import inspect
import torch

MODULE_TYPE = Any

registry: Dict[str, Dict[str, MODULE_TYPE]] = {
    "model": {},
    "global_attack": {},
    "local_attack": {},
    "dataset": {},
    "transform": {},
    "optimizer": {},
    "loss": {},
}

def register_component(
    type: str, key: str, component: MODULE_TYPE = None
) -> Union[Callable, None]:
    """
    Registers a component to the registry.

    Args:
        type (str): The type of the component (e.g., "model", "global_attack").
        key (str): The name of the component.
        component (MODULE_TYPE, optional): The component. If set to None, will return a decorator.
    """
    if type not in registry:
        raise ValueError(
            f"Type '{type}' is not valid. Please choose from {list(registry.keys())}."
        )

    if component is not None:
        if key in registry[type]:
            raise KeyError(
                f"Component with '{key}' already defined in type '{type}'"
            )
        if type == 'model':
            try:
                check_model_forward_signature(component)
                check_model_init_signature(component)
            except Exception as e:
                raise ValueError(f"Failed to validate model signature of model '{component.__name__}'") from e
        registry[type][key] = component
        return

    def register_by_decorator(component):
        register_component(type, key, component)
        return component

    return register_by_decorator


register_model = partial(register_component, "model")
register_dataset = partial(register_component, "dataset")
register_global_attack = partial(register_component, "global_attack")
register_local_attack = partial(register_component, "local_attack")
register_transform = partial(register_component, "transform")
register_optimizer = partial(register_component, "optimizer")
register_loss = partial(register_component, "loss")


def get_from_registry(
    type: str,
    key: str,
    registry: Dict[str, Dict[str, MODULE_TYPE]]
) -> MODULE_TYPE:
    """Get a component from the registry.

    Args:
        type (str): component type (e.g., "model", "global_attack").
        key (str): component alias.
        registry (Dict[str, Dict[str, MODULE_TYPE]]): registry of components.

    Raises:
        ValueError: raised when the given type is not recognized.
        KeyError: raised when the component is not found in the registry.
    Returns:
        Any: component from the registry.
    """
    if type not in registry:
        raise ValueError(
            f"Given type '{type}' is not recognized. Available types: {list(registry.keys())}"
        )

    registry_type = registry[type]
    if key in registry_type:
        return registry_type[key]
    else:
        raise KeyError(
            f"Component '{key}' not found in the type '{type}'. Available options: {list(registry_type.keys())}"
        )

def check_model_init_signature(model: torch.nn.Module):
    """Method to check if the model has the required parameters in the '__init__' method.

    Args:
        model (torch.nn.Module): Model to be checked.

    Raises:
        TypeError: Raised when the model does not have the required parameters 'in_channels' and 'out_channels' in the '__init__' method and raised when the model does not have an '__init__' method.
    """
    init_method = has_required_params(model)
    if not init_method:
        raise TypeError(f"The class {model.__name__} or its ancestors do not define a '__init__' method.")
    
    sig = inspect.signature(init_method)
    params = sig.parameters
    if not ('in_channels' in params and 'out_channels' in params):
        raise TypeError(f"The class {model.__name__} or its ancestors do not define the required parameters 'in_channels' and 'out_channels' in the '__init__' method.")

def has_required_params(model):
    for cls in inspect.getmro(model):
        if '__init__' in cls.__dict__:
            return cls.__dict__['__init__']
    return None

def check_model_forward_signature(model: torch.nn.Module):
    """Method to check if the model has the required parameters in the 'forward' method.

    Args:
        model (torch.nn.Module): 

    Raises:
        TypeError: Raised when the model does not have the required parameters 'x' and 'edge_index' in the 'forward' method and raised when the model does not have an 'forward' method.
    """
    forward_method = check_forward_in_inheritance_chain(model)
    if not forward_method:
        raise TypeError(f"The class {model.__name__} or its ancestors do not define a 'forward' method.")
    
    sig = inspect.signature(forward_method)
    parameters = [
        param for param in sig.parameters.values() if param.name != "self" and param.name != "kwargs"
    ]

    required_params = ["x", "edge_index"]
    optional_params = ["edge_weight"]
    allowed_signatures = [
        ["x", "edge_index"],
        ["x", "edge_index", "edge_weight"],
    ]

    for param in required_params:
        if param not in [p.name for p in parameters]:
            raise TypeError(f"Missing required parameter '{param}' in 'forward' method of class {model.__name__}.")

    for param in parameters:
        if param.name not in required_params + optional_params:
            raise TypeError(f"Invalid parameter '{param.name}' in 'forward' method of model {model.__name__}. Only {required_params + optional_params} are allowed.")

    params_names = [param.name for param in parameters]
    if params_names not in allowed_signatures:
        raise TypeError(
            f"Invalid forward parameters. Allowed parameters are {allowed_signatures}."
        )

def check_forward_in_inheritance_chain(cls):
    for base in inspect.getmro(cls):
        if 'forward' in base.__dict__:
            return base.__dict__['forward']
    return None
