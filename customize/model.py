from pytorch_lightning import LightningModule
from config_def import cfg
from register import registry

class BaseModel(LightningModule):
    def __init__(self, **kwargs):
        super().__init__()
        self.save_hyperparameters()
        self.model = registry['architecture'][cfg.model.name](**cfg.model.params)
    
    def forward(self, *args, **kwargs):
        return self.model(*args, **kwargs)
        
def create_model():
    # model_params = {key: value for key, value in cfg.model.items() if key != 'name'}
    model = registry['architecture'][cfg.model.name](**cfg.model.params)
    if to_device:
        model.to(torch.device(cfg.device))
    return model

