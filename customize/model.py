from typing import Any, Dict, Tuple
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from pytorch_lightning import LightningModule

from config_def import cfg
from customize.optimizer import register_optimizer
from customize.register import registry

class BaseModel(LightningModule):
    def __init__(self, **kwargs):
        super().__init__()
        self.save_hyperparameters()
        
        # self.train_acc = Accuracy(task=cfg., num_classes=out_channels)
        # self.val_acc = Accuracy(task='multiclass', num_classes=out_channels)
        # self.test_acc = Accuracy(task='multiclass', num_classes=out_channels)
        
        # self.model = registry['architecture'][cfg.model.name](**cfg.model.params)
    
    def forward(self, *args, **kwargs):
        raise NotImplementedError
        # return self.model(*args, **kwargs)
    
    
    def configure_optimizers(self):
        return registry['optimizer'][cfg.optimizer.name](self.parameters(), **cfg.optimizer.params)
        
    # def compute_loss(self, pred, true):
    #     raise NotImplementedError
    
    def _shared_step(self, batch, split: str) -> Dict:
        batch.split = split
        pred, true = self(batch)
        loss, pred_score = compute_loss(pred, true)
        step_end_time = time.time()
        return dict(loss=loss, true=true, pred_score=pred_score.detach(),
                    step_end_time=step_end_time)

    def training_step(self, batch, *args, **kwargs):
        return self._shared_step(batch, split="train")

    def validation_step(self, batch, *args, **kwargs):
        return self._shared_step(batch, split="val")

    def test_step(self, batch, *args, **kwargs):
        return self._shared_step(batch, split="test")


def compute_loss(pred, true):
    """Compute loss and prediction score.

    Args:
        pred (torch.tensor): Unnormalized prediction
        true (torch.tensor): Grou

    Returns: Loss, normalized prediction score

    """
    bce_loss = torch.nn.BCEWithLogitsLoss(reduction=cfg.model.size_average)
    mse_loss = torch.nn.MSELoss(reduction=cfg.model.size_average)

    # default manipulation for pred and true
    # can be skipped if special loss computation is needed
    pred = pred.squeeze(-1) if pred.ndim > 1 else pred
    true = true.squeeze(-1) if true.ndim > 1 else true

    if cfg.model.loss_fun == 'cross_entropy':
        # multiclass
        if pred.ndim > 1 and true.ndim == 1:
            pred = F.log_softmax(pred, dim=-1)
            return F.nll_loss(pred, true), pred
        # binary or multilabel
        else:
            true = true.float()
            return bce_loss(pred, true), torch.sigmoid(pred)
    elif cfg.model.loss_fun == 'mse':
        true = true.float()
        return mse_loss(pred, true), pred
    else:
        raise ValueError(f"Loss function '{cfg.model.loss_fun}' not supported")