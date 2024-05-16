from typing import Any, Callable, Dict, Union
from functools import partial
from gnn_toolbox.custom_components import * # noqa
import inspect

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
    category: str, key: str, component: MODULE_TYPE = None
) -> Union[Callable, None]:
    """
    Registers a component.

    Args:
        category (str): The category of the component (e.g., "act", "node_encoder").
        key (str): The name of the component.
        component (any, optional): The component. If set to None, will return a decorator.
    """
    if category not in registry:
        raise ValueError(
            f"Category '{category}' is not valid. Please choose from {list(registry.keys())}."
        )

    if component is not None:
        if key in registry[category]:
            raise KeyError(
                f"Component with '{key}' already defined in category '{category}'"
            )
        if key == 'model':
            try:
                check_model_signature(component)
            except Exception as e:
                raise ValueError(f"Failed to validate model signature: {e}")
        registry[category][key] = component
        return

    def register_by_decorator(component):
        register_component(category, key, component)
        return component

    return register_by_decorator


def get_from_registry(
    category: str,
    key: str,
    registry: Dict[str, Dict[str, MODULE_TYPE]],
    default: Any = None,
) -> Any:
    """Retrieve a component from the registry safely with a fallback."""
    if category not in registry:
        raise ValueError(
            f"Category '{category}' is not recognized. Available categories: {list(registry.keys())}"
        )

    category_registry = registry[category]
    if key in category_registry:
        return category_registry[key]
    else:
        if default is not None:
            return default
        else:
            raise KeyError(
                f"Component '{key}' not found in category '{category}'. Available options: {list(category_registry.keys())}"
            )


def check_model_signature(model):
    sig = inspect.signature(model.forward)
    parameters = [
        param.name for param in sig.parameters.values() if param.name != "self"
    ]

    allowed_signatures = [
        ["x", "edge_index"],
        ["x", "edge_index", "edge_weight"]
    ]

    if parameters not in allowed_signatures:
        raise TypeError(
            f"Invalid forward parameters. Allowed parameters are {allowed_signatures}."
        )


# def register_model(func):
#     @wraps(func)
#     def wrapper(*args, **kwargs):
#         # Check the forward method's signature
#         sig = inspect.signature(args[0].forward)
#         parameters = [
#             param.name for param in sig.parameters.values() if param.name != "self"
#         ]

#         # Define allowed signatures
#         allowed_signatures = [
#             ["x", "edge_index"],
#             ["x", "edge_index", "edge_weight"],
#             ["x", "edge_index", "edge_attr"],
#         ]

#         # Check if the model's parameters match any of the allowed signatures
#         if parameters not in allowed_signatures:
#             raise TypeError(
#                 f"Invalid forward parameters. Allowed parameters are {allowed_signatures}."
#             )

#         # If valid, call the original function (usually the model's initializer)
#         # return func(*args, **kwargs)

#     return wrapper
#     return

register_model = partial(register_component, "model")
register_global_attack = partial(register_component, "global_attack")
register_local_attack = partial(register_component, "local_attack")
register_dataset = partial(register_component, "dataset")
register_transform = partial(register_component, "transform")
register_optimizer = partial(register_component, "optimizer")
register_loss = partial(register_component, "loss")
