from torch.optim import (
    Adam,
    AdamW,
    RAdam,
    NAdam,
    Adadelta,
    Adagrad,
    Adamax,
    RMSprop,
    SGD,
    ASGD,
    Rprop,
)
from torch.nn.parameter import Parameter
from gnn_toolbox.registration_handler.register_components import register_optimizer
from typing import Iterator


@register_optimizer("Adam")
def adam_optimizer(params: Iterator[Parameter], **kwargs) -> Adam:
    return Adam(params, **kwargs)


@register_optimizer("AdamW")
def adamW_optimizer(params: Iterator[Parameter], **kwargs) -> AdamW:
    return AdamW(params, **kwargs)


@register_optimizer("SGD")
def sgd_optimizer(params: Iterator[Parameter], **kwargs) -> SGD:
    return SGD(params, **kwargs)


@register_optimizer("RAdam")
def radam_optimizer(params: Iterator[Parameter], **kwargs) -> RAdam:
    return RAdam(params, **kwargs)


@register_optimizer("NAdam")
def nadam_optimizer(params: Iterator[Parameter], **kwargs) -> NAdam:
    return NAdam(params, **kwargs)


@register_optimizer("Adadelta")
def adadelta_optimizer(params: Iterator[Parameter], **kwargs) -> Adadelta:
    return Adadelta(params, **kwargs)


@register_optimizer("Adagrad")
def adagrad_optimizer(params: Iterator[Parameter], **kwargs) -> Adagrad:
    return Adagrad(params, **kwargs)


@register_optimizer("Adamax")
def adamax_optimizer(params: Iterator[Parameter], **kwargs) -> Adamax:
    return Adamax(params, **kwargs)


@register_optimizer("RMSprop")
def rmsprop_optimizer(params: Iterator[Parameter], **kwargs) -> RMSprop:
    return RMSprop(params, **kwargs)


@register_optimizer("ASGD")
def asgd_optimizer(params: Iterator[Parameter], **kwargs) -> ASGD:
    return ASGD(params, **kwargs)


@register_optimizer("Rprop")
def rprop_optimizer(params: Iterator[Parameter], **kwargs) -> Rprop:
    return Rprop(params, **kwargs)
