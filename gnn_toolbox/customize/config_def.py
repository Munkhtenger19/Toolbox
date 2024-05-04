from yacs.config import CfgNode as CN
import torch
from torch_geometric.data import InMemoryDataset
from dataclasses import dataclass
import os
import os.path as osp
from torch_geometric.io import fs

# TODO perform config sanity checks

cfg = CN()
cfg.device = 'auto'
cfg.device_num = 1
cfg.seed = 42
cfg.custom_metrics = []
cfg.print = 'file'
cfg.run_dir = 'result'
cfg.rounding = 4
cfg.tensorboard_each_run = True
cfg.gpu_usage = True

def load_dataset_config(cfg: CN):
    # ! Not using all dataset configs
    """Configures dataset-related settings."""
    cfg.dataset = CN()
    cfg.dataset.name = 'cora'
    cfg.dataset.format = 'PyG'
    cfg.dataset.task_type = 'classification'
    cfg.dataset.prediction_target = 'node'
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
    cfg.model.threshold = 0.5
    cfg.model.params = CN()
    cfg.model.params.hidden_channels = 16
    cfg.model.params.num_layers = 2
    cfg.model.params.dropout = 0.5
    cfg.model.params.out_channels = 1
    
    
def load_trainer_config(cfg):
    # ! Not using all trainer configs
    """Configures trainer-related settings."""
    cfg.trainer = CN()
    cfg.trainer.epochs = 100
    cfg.trainer.batch_size = 64
    cfg.trainer.num_splits = 1
    cfg.trainer.non_train_splits = ["val", "test"]
    # cfg.share.num_splits = 1 + sum(key in dataset._data for key in split_keys)
    cfg.trainer.num_workers = os.cpu_count()
    cfg.trainer.log_every = 1
    cfg.trainer.enable_checkpointing = True
    cfg.trainer.checkpoint_dir = './checkpoints'
    cfg.trainer.auto_resume = False


def load_loader_config(cfg):
    """Configures loader-related settings."""
    cfg.loader = CN()
    cfg.loader.name = 'DataLoader'
    cfg.loader.params = CN()
    cfg.loader.params.batch_size = 64
    cfg.loader.params.num_workers = 0
    cfg.loader.params.pin_memory = True
    cfg.loader.params.drop_last = False
    # cfg.loader.params.shuffle = True

def set_run_dir(out_dir):
    r"""Create the directory for each random seed experiment run.

    Args:
        out_dir (str): Directory for output, specified in :obj:`cfg.out_dir`
    """
    cfg.run_dir = os.path.join(out_dir, str(cfg.seed))
    # Make output directory
    if cfg.trainer.auto_resume:
        os.makedirs(cfg.run_dir, exist_ok=True)
    else:
        if osp.isdir(cfg.run_dir):
            fs.rm(cfg.run_dir)
        os.makedirs(cfg.run_dir, exist_ok=True)

load_dataset_config(cfg)
load_optimizer_config(cfg)
load_model_config(cfg)
load_loader_config(cfg)
load_trainer_config(cfg)
