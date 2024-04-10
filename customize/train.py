import pytorch_lightning as L
from torch_geometric.data.lightning.datamodule import LightningDataModule

from config_def import cfg
from customize.model import BaseModel
from customize.loader import create_loader

class BaseDataModule(LightningDataModule):
    r"""A :class:`pytorch_lightning.LightningDataModule` for handling data
    loading.

    This class provides data loaders for training, validation, and testing, and
    can be accessed through the :meth:`train_dataloader`,
    :meth:`val_dataloader`, and :meth:`test_dataloader` methods, respectively.
    """
    def __init__(self):
        self.loaders = create_loader()
        super().__init__(has_val=True, has_test=True)

    def train_dataloader(self):
        return self.loaders[0]

    def val_dataloader(self):
        return self.loaders[1]

    def test_dataloader(self):
        return self.loaders[2]
    
def train(model: BaseModel, loader:BaseDataModule):
    trainer = L.Trainer(logger=True, enable_checkpointing=True, max_epochs=cfg.trainer.epochs, log_every_n_steps=1)
    trainer.fit(model=model, datamodule=loader)
    trainer.test(model=model, datamodule= loader)