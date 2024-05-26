import torch
import numpy as np
import torch.nn.functional as F

from gnn_toolbox.registration_handler.register_components import register_loss


@register_loss("CE")
def CE(logits, labels):
    return F.cross_entropy(logits, labels)


@register_loss("CW")
def CW(logits, labels):
    sorted = logits.argsort(-1)
    best_non_target_class = sorted[sorted != labels[:, None]].reshape(
        logits.size(0), -1
    )[:, -1]
    margin = (
        logits[np.arange(logits.size(0)), labels]
        - logits[np.arange(logits.size(0)), best_non_target_class]
    )
    return -torch.clamp(margin, min=0).mean()


@register_loss("LCW")
def LCW(logits, labels):
    sorted = logits.argsort(-1)
    best_non_target_class = sorted[sorted != labels[:, None]].reshape(
        logits.size(0), -1
    )[:, -1]
    margin = (
        logits[np.arange(logits.size(0)), labels]
        - logits[np.arange(logits.size(0)), best_non_target_class]
    )
    return -F.leaky_relu(margin, negative_slope=0.1).mean()


@register_loss("tanhMargin")
def tanhMargin(logits, labels):
    sorted = logits.argsort(-1)
    best_non_target_class = sorted[sorted != labels[:, None]].reshape(
        logits.size(0), -1
    )[:, -1]
    margin = (
        logits[np.arange(logits.size(0)), labels]
        - logits[np.arange(logits.size(0)), best_non_target_class]
    )
    loss = torch.tanh(-margin).mean()


@register_loss("Margin")
def Margin(logits, labels):
    sorted = logits.argsort(-1)
    best_non_target_class = sorted[sorted != labels[:, None]].reshape(
        logits.size(0), -1
    )[:, -1]
    margin = (
        logits[np.arange(logits.size(0)), labels]
        - logits[np.arange(logits.size(0)), best_non_target_class]
    )
    return -margin.mean()


@register_loss("eluMargin")
def eluMargin(logits, labels):
    sorted = logits.argsort(-1)
    best_non_target_class = sorted[sorted != labels[:, None]].reshape(
        logits.size(0), -1
    )[:, -1]
    margin = (
        logits[np.arange(logits.size(0)), labels]
        - logits[np.arange(logits.size(0)), best_non_target_class]
    )
    return -F.elu(margin).mean()


@register_loss("MCE")
def MCE(logits, labels):
    not_flipped = logits.argmax(-1) == labels
    return F.cross_entropy(logits[not_flipped], labels[not_flipped])


@register_loss("NCE")
def NCE(logits, labels):
    sorted = logits.argsort(-1)
    best_non_target_class = sorted[sorted != labels[:, None]].reshape(
        logits.size(0), -1
    )[:, -1]
    return -F.cross_entropy(logits, best_non_target_class)


