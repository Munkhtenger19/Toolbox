from typing import Any, Callable, Dict, Union
# from custom_components import * # noqa
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
        if category == 'model':
            try:
                check_model_signature(component)
            except Exception as e:
                raise ValueError(f"Failed to validate model signature of model '{component.__name__}': {e}")
        registry[category][key] = component
        return

    def register_by_decorator(component):
        register_component(category, key, component)
        return component

    return register_by_decorator


def get_from_registry(
    category: str,
    key: str,
    registry: Dict[str, Dict[str, MODULE_TYPE]]
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
        raise KeyError(
            f"Component '{key}' not found in category '{category}'. Available options: {list(category_registry.keys())}"
        )


def check_model_signature(model):
    forward_method = check_forward_in_inheritance_chain(model)
    if not forward_method:
        raise TypeError(f"The class {model.__name__} or its ancestors do not define a 'forward' method.")
    
    sig = inspect.signature(forward_method)
    parameters = [
        param for param in sig.parameters.values() if param.name != "self" and param.name != "kwargs"
    ]

    # allowed_signatures = [
    #     ["x", "edge_index"],
    #     ["x", "edge_index", "edge_weight"],
    #     ["x", "edge_index", "edge_attr"]
    # ]

    # if parameters not in allowed_signatures:
    #     raise TypeError(
    #         f"Invalid forward parameters. Allowed parameters are {allowed_signatures}."
    #     )
    required_params = ["x", "edge_index"]
    optional_params = ["edge_weight", "edge_attr"]
    allowed_signatures = [
        ["x", "edge_index"],
        ["x", "edge_index", "edge_weight"],
        ["x", "edge_index", "edge_attr"]
    ]

    for param in required_params:
        if param not in [p.name for p in parameters]:
            raise TypeError(f"Missing required parameter '{param}' in 'forward' method of class {model.__name__}.")

    for param in parameters:
        if param.name not in required_params + optional_params:
            # if param.default is inspect.Parameter.empty:
            raise TypeError(f"Invalid parameter '{param.name}' in 'forward' method of model {model.__name__}. Only {required_params + optional_params} are allowed.")

    # if not any([set(required_params + [opt]).issubset([p.name for p in parameters]) for opt in optional_params]):
    #     raise TypeError(f"The 'forward' method of class {model.__name__} does not match any of the allowed signatures {allowed_signatures}.")
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

