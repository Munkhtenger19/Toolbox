import torch
from torch_sparse import SparseTensor, SparseStorage, coalesce
from typing import Tuple, Union, Sequence

class DotDict(dict):     
    """
    dot.notation access to dictionary attributes
    https://stackoverflow.com/questions/2352181/how-to-use-a-dot-to-access-members-of-dictionary
    """      
    def __getattr__(*args):         
        val = dict.get(*args)         
        return DotDict(val) if type(val) is dict else val      
    __setattr__ = dict.__setitem__     
    __delattr__ = dict.__delitem__ 
 
    
def to_symmetric(edge_index: torch.Tensor, edge_weight: torch.Tensor,
                 n: int, op='mean') -> Tuple[torch.Tensor, torch.Tensor]:
    symmetric_edge_index = torch.cat(
        (edge_index, edge_index.flip(0)), dim=-1
    )

    symmetric_edge_weight = edge_weight.repeat(2)

    symmetric_edge_index, symmetric_edge_weight = coalesce(
        symmetric_edge_index,
        symmetric_edge_weight,
        m=n,
        n=n,
        op=op
    )
    return symmetric_edge_index, symmetric_edge_weight

def grad_with_checkpoint(outputs: Union[torch.Tensor, Sequence[torch.Tensor]],
                         inputs: Union[torch.Tensor, Sequence[torch.Tensor]]) -> Tuple[torch.Tensor, ...]:
    inputs = (inputs,) if isinstance(inputs, torch.Tensor) else tuple(inputs)

    for input in inputs:
        if not input.is_leaf:
            input.retain_grad()

    torch.autograd.backward(outputs)

    grad_outputs = []
    for input in inputs:
        grad_outputs.append(input.grad.clone())
        input.grad.zero_()
    return grad_outputs

def project(
    n_perturbations: int,
    values: torch.Tensor,
    eps: float = 0,
    inplace: bool = False,
):
    if not inplace:
        values = values.clone()

    if torch.clamp(values, 0, 1).sum() > n_perturbations:
        left = (values - 1).min()
        right = values.max()
        miu = bisection(values, left, right, n_perturbations)
        values.data.copy_(torch.clamp(values - miu, min=eps, max=1 - eps))
    else:
        values.data.copy_(torch.clamp(values, min=eps, max=1 - eps))
    return values

def bisection(edge_weights, a, b, n_perturbations, epsilon=1e-5, iter_max=1e5):
    def func(x):
        return torch.clamp(edge_weights - x, 0, 1).sum() - n_perturbations

    miu = a
    for _ in range(int(iter_max)):
        miu = (a + b) / 2
        # Check if middle point is root
        if func(miu) == 0.0:
            break
        # Decide the side to repeat the steps
        if func(miu) * func(a) < 0:
            b = miu
        else:
            a = miu
        if (b - a) <= epsilon:
            break
    return miu