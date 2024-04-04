from yacs.config import CfgNode as CN
import torch
 
cfg = CN()
cfg.device = 'cuda' if torch.cuda.is_available() else 'cpu'
def load_dataset_config(cfg: CN):
    """Configures dataset-related settings."""
    cfg.dataset = CN()
    cfg.dataset.name = 'Coras'
    cfg.dataset.format = 'PyGs'
    cfg.dataset.task = 'node'
    cfg.dataset.dir = './datasets'
    
def load_optimizer_config(cfg):
    """Configures optimizer-related settings."""
    cfg.optimizer = CN()
    cfg.optimizer.name = 'adam'
    cfg.optimizer.lr = 0.01
    cfg.optimizer.weight_decay = 0.00051
    
def load_model_config(cfg):
    """Configures model-related settings."""
    cfg.model = CN()
    cfg.model.name = 'GCN'
    cfg.model.params = CN(new_allowed=True)
    cfg.model.params.hidden_size = 16
    cfg.model.params.num_layers = 2
    # cfg.model.params.dropout = 0.5
    
    
def load_trainer_config(cfg):
    """Configures trainer-related settings."""
    cfg.trainer = CN()
    cfg.trainer.epochs = 100
    cfg.trainer.batch_size = 64
    cfg.trainer.num_workers = 0
    cfg.trainer.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    cfg.trainer.log_every = 1
    cfg.trainer.eval_every = 1
    cfg.trainer.save_every = 1
    cfg.trainer.out_dir = './out'
    cfg.trainer.log_dir = './runs'
    cfg.trainer.checkpoint_dir = './checkpoints'
    cfg.trainer.load_checkpoint = ''
    cfg.trainer.seed = 42
    
load_dataset_config(cfg)
load_optimizer_config(cfg)
load_model_config(cfg)

print(cfg.model.params)
args= dict(cfg.model)

# print(type(cfg.model))