import torch
from config_def import cfg
from customize.register import registry, register_architecture
from customize.architecture import register_architecture

def create_model():
    # model_params = {key: value for key, value in cfg.model.items() if key != 'name'}
    model = registry['architecture'][cfg.model.name](**cfg.model.params)
    # if to_device:
    model.to(torch.device(cfg.device))
    return model