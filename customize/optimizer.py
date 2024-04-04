from torch.optim import Adam
from torch.nn.parameter import Parameter
from register import register_optimizer
from typing import Iterator, Dict

@register_optimizer('adam')
def adam_optimizer(params: Iterator[Parameter], base_lr: float, weight_decay: float) -> Adam:
    return Adam(params, lr=base_lr, weight_decay=weight_decay)

