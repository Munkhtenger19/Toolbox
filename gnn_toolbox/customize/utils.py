import torch
from customize.config_def import cfg
from customize.register import register_attack
import customize.register as register
from customize.architecture import register_architecture
from attacks.robustness_of_gnns_at_scale.dice import register_attack
from attacks.robustness_of_gnns_at_scale.local_dice import LocalDICE
import os
import os.path as osp
import json
from typing import Tuple
from torch_sparse import SparseTensor, SparseStorage, coalesce

# import resource
# _resource_module_available = True
import scipy.sparse as sp
from torchtyping import TensorType

def create_model(experiment):
    # model_params = {key: value for key, value in cfg.model.items() if key != 'name'}
    model = register.registry.architecture[experiment.model.name](**experiment.model.params)
    return model.to(experiment.device)

def create_attack(attack_name):
    attack = register.registry.attack[attack_name]
    return attack

def auto_select_device(experiment):
    r"""Auto select device for the current experiment."""
    if experiment.device == 'auto':
        if torch.cuda.is_available():
            experiment.device = 'cuda'
            experiment.devices = 1
        else:
            experiment.device = 'cpu'
            experiment.devices = None
            
def get_current_gpu_usage():
    """Get the current GPU memory usage."""
    if torch.cuda.is_available():
        return torch.cuda.memory_allocated()
    else:
        return -1
    
def dict_to_json(dict, fname):
    """Dump a :python:`Python` dictionary to a JSON file.

    Args:
        dict (dict): The :python:`Python` dictionary.
        fname (str): The output file name.
    """
    with open(fname, 'a') as f:
        json.dump(dict, f)
        f.write('\n')
        
def dict_to_json(dict, fname):
    """Dump a :python:`Python` dictionary to a JSON file.

    Args:
        dict (dict): The :python:`Python` dictionary.
        fname (str): The output file name.
    """
    with open(fname, 'a') as f:
        json.dump(dict, f)
        f.write('\n')
        
def dict_to_tb(dict, writer, epoch):
    """Add a dictionary of statistics to a Tensorboard writer.

    Args:
        dict (dict): Statistics of experiments, the keys are attribute names,
        the values are the attribute values
        writer: Tensorboard writer object
        epoch (int): The current epoch
    """
    for key in dict:
        writer.add_scalar(key, dict[key], epoch)
        
def get_ckpt_dir() -> str:
    return osp.join(cfg.run_dir, 'my_ckpt')

def get_max_memory_bytes():
    # if _resource_module_available:
    #     return 1024 * resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    # return np.nan
    if torch.cuda.is_available():
        return torch.cuda.max_memory_allocated()

def to_symmetric_scipy(adjacency: sp.csr_matrix):
    sym_adjacency = (adjacency + adjacency.T).astype(bool).astype(float)

    sym_adjacency.tocsr().sort_indices()

    return sym_adjacency

def row_norm(A: TensorType["a", "b"]):
    rowsum = A.sum(-1)
    norm_mask = rowsum != 0
    A[norm_mask] = A[norm_mask] / rowsum[norm_mask][:, None]
    return A / A.sum(-1)[:, None]

