from torch.optim import Adam
from torch.nn.parameter import Parameter
from gnn_toolbox.registry import register_optimizer
from typing import Iterator, Dict

@register_optimizer('adam')
def adam_optimizer(params: Iterator[Parameter], lr: float, weight_decay: float) -> Adam:
    return Adam(params, lr=lr, weight_decay=weight_decay)