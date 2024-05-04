import torch
import scipy.sparse as sp
from abc import ABC, abstractmethod
from torch_sparse import SparseTensor

class BaseAttack(ABC):
    """
    Abstract base class for graph attacks, compatible with both PyTorch tensors and SciPy sparse matrices.
    """
    def __init__(self, model, device='cpu'):
        self.model = model.to(device)
        self.device = device

    @abstractmethod
    def attack(self, n_perturbations, **kwargs):
        """
        Method to execute the attack. Must be implemented by specific attack classes.
        """
        pass

    def convert_data_and_attack(self, data, format_required):
        """
        Convert data between PyTorch tensor and SciPy sparse matrix based on the model's requirement.
        """
        if format_required == 'scipy' and isinstance(data, torch.Tensor):
            data = torch_to_scipy(data)
        elif format_required == 'torch' and isinstance(data, sp.csr_matrix):
            data = scipy_to_torch(data)
        
        return self.attack(n_perturbations, **kwargs)

    @staticmethod
    def _torch_to_scipy(tensor):
        if isinstance(tensor, SparseTensor):
            tensor = tensor.to_torch_tensor()
        coo = tensor.coalesce()
        return sp.csr_matrix((coo.values().cpu().numpy(), coo.indices().cpu().numpy()), shape=tensor.shape)

    @staticmethod
    def _scipy_to_torch(matrix):
        coo = sp.coo_matrix(matrix)
        indices = torch.LongTensor([coo.row, coo.col])
        values = torch.FloatTensor(coo.data)
        return SparseTensor(row=indices[0], col=indices[1], value=values, sparse_sizes=(matrix.shape[0], matrix.shape[1]))