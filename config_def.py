from yacs.config import CfgNode as CN
import torch
from torch_geometric.data import InMemoryDataset
from dataclasses import dataclass
import os

# TODO perform config sanity checks

cfg = CN()
cfg.device = 'cuda' if torch.cuda.is_available() else 'cpu'
def load_dataset_config(cfg: CN):
    """Configures dataset-related settings."""
    cfg.dataset = CN()
    cfg.dataset.name = 'Coras'
    cfg.dataset.format = 'PyGs'
    cfg.dataset.task = 'node'
    cfg.dataset.dir = './datasets'
    cfg.dataset.transforms = 'None'
    
def load_optimizer_config(cfg):
    """Configures optimizer-related settings."""
    cfg.optimizer = CN()
    cfg.optimizer.name = 'adam'
    cfg.optimizer.params = CN()
    cfg.optimizer.params.base_lr = 0.01
    cfg.optimizer.params.weight_decay = 0.00051
    
def load_model_config(cfg):
    """Configures model-related settings."""
    cfg.model = CN()
    cfg.model.name = 'GCN'
    cfg.model.loss_fun = 'cross_entropy'
    cfg.model.size_average = 'mean'
    cfg.model.params = CN()
    cfg.model.params.hidden_channels = 16
    cfg.model.params.num_layers = 2
    cfg.model.params.dropout = 0.5
    cfg.model.params.out_channels = 1
    
    
def load_trainer_config(cfg):
    """Configures trainer-related settings."""
    cfg.trainer = CN()
    cfg.trainer.epochs = 100
    cfg.trainer.batch_size = 64
    cfg.trainer.num_workers = os.cpu_count()
    cfg.trainer.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    cfg.trainer.log_every = 1
    cfg.trainer.eval_every = 1
    cfg.trainer.save_every = 1
    cfg.trainer.out_dir = './out'
    cfg.trainer.log_dir = './runs'
    cfg.trainer.checkpoint_dir = './checkpoints'
    cfg.trainer.load_checkpoint = ''
    cfg.trainer.seed = 42

def load_loader_config(cfg):
    """Configures loader-related settings."""
    cfg.loader = CN()
    cfg.loader.name = 'DataLoader'
    cfg.loader.params = CN()
    cfg.loader.params.batch_size = 64
    cfg.loader.params.num_workers = 0
    cfg.loader.params.pin_memory = True
    # cfg.loader.params.shuffle = True
    cfg.loader.params.drop_last = False

load_dataset_config(cfg)
load_optimizer_config(cfg)
load_model_config(cfg)
load_loader_config(cfg)
load_trainer_config(cfg)
