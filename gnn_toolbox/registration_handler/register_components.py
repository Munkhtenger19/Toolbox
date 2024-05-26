from functools import partial

from gnn_toolbox.registration_handler.registry import register_component

register_model = partial(register_component, "model")
register_global_attack = partial(register_component, "global_attack")
register_local_attack = partial(register_component, "local_attack")
register_dataset = partial(register_component, "dataset")
register_transform = partial(register_component, "transform")
register_optimizer = partial(register_component, "optimizer")
register_loss = partial(register_component, "loss")