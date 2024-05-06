import warnings
from abc import ABC, abstractmethod
from copy import deepcopy
from typing import Optional, Tuple, Union, List, Dict
from torchtyping import TensorType, patch_typeguard
from typeguard import typechecked
import logging

import numpy as np
import scipy.sparse as sp
import torch
from torch import Tensor
from torch.nn import functional as F

from torch_sparse import SparseTensor

# from rgnn_at_scale.models import MODEL_TYPE, DenseGCN, GCN, RGNN, BATCHED_PPR_MODELS
# from rgnn_at_scale.helper.utils import accuracy
from custom_modules.architecture import MODEL_TYPE

# from .real_base import BaseAttack
# patch_typeguard()


def accuracy(
    logits: torch.Tensor, labels: torch.Tensor, split_idx: np.ndarray
) -> float:
    """Returns the accuracy for a tensor of logits, a list of lables and and a split indices.

    Parameters
    ----------
    prediction : torch.Tensor
        [n x c] tensor of logits (`.argmax(1)` should return most probable class).
    labels : torch.Tensor
        [n x 1] target label.
    split_idx : np.ndarray
        [?] array with indices for current split.

    Returns
    -------
    float
        the Accuracy
    """
    return (logits.argmax(1)[split_idx] == labels[split_idx]).float().mean().item()


# @typechecked
class BaseAttack(ABC):
    """
    Base class for adversarial attacks

    Parameters
    ----------
    adj : SparseTensor or torch.Tensor
        [n, n] (sparse) adjacency matrix.
    attr : torch.Tensor
        [n, d] feature/attribute matrix.
    labels : torch.Tensor
        Labels vector of shape [n].
    idx_attack : np.ndarray
        Indices of the nodes which are to be attacked.
    model : MODEL_TYPE
        Model to be attacked.
    device : Union[str, int, torch.device]
        The cuda device to use for the attack
    data_device : Union[str, int, torch.device]
        The cuda device to use for storing the dataset.
        For batched models (like PPRGo) this may differ from the device parameter.
        Other models require the dataset and model to be on the same device.
    make_undirected: bool
        Wether the perturbed adjacency matrix should be made undirected (symmetric degree normalization)
    binary_attr: bool
        If true the perturbed attributes are binarized (!=0)
    loss_type: str
        The loss to be used by a gradient based attack, can be one of the following loss types:
            - CW: Carlini-Wagner
            - LCW: Leaky Carlini-Wagner
            - Margin: Negative classification margin
            - tanhMargin: Negative TanH of classification margin
            - eluMargin: Negative Exponential Linear Unit (ELU) of classification margin
            - CE: Cross Entropy
            - MCE: Masked Cross Entropy
            - NCE: Negative Cross Entropy
    """

    def __init__(
        self,
        adj: Union[SparseTensor, TensorType["n_nodes", "n_nodes"]],
        attr: TensorType["n_nodes", "n_features"],
        labels: TensorType["n_nodes"],
        idx_attack: np.ndarray,
        model: MODEL_TYPE,
        device: Union[str, int, torch.device],
        data_device: Union[str, int, torch.device],
        make_undirected: bool = False,
        loss_type: str = "CE",  # 'CW', 'LeakyCW'  # 'CE', 'MCE', 'Margin'
        #  attack_structure=True,
        #  attack_features=False,
        #  num
        **kwargs,
    ):
        # if not (isinstance(model, GCN) or isinstance(model, DenseGCN) or isinstance(model, RGNN)):
        #     warnings.warn("The attack will fail if the gradient w.r.t. the adjacency can't be computed.")

        # if isinstance(model, GCN) or isinstance(model, RGNN):
        #     assert (
        #         model.gdc_params is None
        #         or 'use_cpu' not in model.gdc_params
        #         or not model.gdc_params['use_cpu']
        #     ), "GDC doesn't support a gradient w.r.t. the adjacency"
        #     assert model.svd_params is None, "SVD preproc. doesn't support a gradient w.r.t. the adjacency"
        #     assert model.jaccard_params is None, "Jaccard preproc. doesn't support a gradient w.r.t. the adjacency"
        # if isinstance(model, RGNN):
        #     assert model._mean in ['dimmedian', 'medoid', 'soft_median'],\
        #         "Agg. doesn't support a gradient w.r.t. the adjacency"

        self.device = device
        self.data_device = data_device
        self.idx_attack = idx_attack
        self.loss_type = loss_type

        self.make_undirected = make_undirected
        # self.binary_attr = binary_attr

        self.attacked_model = deepcopy(model).to(self.device)
        self.attacked_model.eval()
        for p in self.attacked_model.parameters():
            p.requires_grad = False
        self.eval_model = self.attacked_model

        self.labels = labels.to(torch.long).to(self.device)
        self.labels_attack = self.labels[self.idx_attack]
        self.attr = attr.to(self.data_device)
        self.adj = adj.to(self.data_device)

        self.attr_adversary = self.attr
        self.adj_adversary = self.adj

    @abstractmethod
    def _attack(self, n_perturbations: int, **kwargs):
        pass

    def attack(self, n_perturbations: int, **kwargs):
        """
        Executes the attack on the model updating the attributes
        self.adj_adversary and self.attr_adversary accordingly.

        Parameters
        ----------
        n_perturbations : int
            number of perturbations (attack budget in terms of node additions/deletions) that constrain the atack
        """
        if n_perturbations > 0:
            return self._attack(n_perturbations, **kwargs)
        else:
            self.attr_adversary = self.attr
            self.adj_adversary = self.adj

    def set_pertubations(
        self,
        adj_perturbed: Union[SparseTensor, TensorType["n_nodes", "n_nodes"]],
        attr_perturbed: TensorType["n_nodes", "n_features"],
    ):
        self.adj_adversary = adj_perturbed.to(self.data_device)
        self.attr_adversary = attr_perturbed.to(self.data_device)

    def get_perturbations(self):
        adj_adversary, attr_adversary = self.adj_adversary, self.attr_adversary

        if isinstance(self.adj_adversary, torch.Tensor):
            # * might need to do to_dense() here since torch_geometric uses torch tensor
            adj_adversary = SparseTensor.to_torch_sparse_coo_tensor(self.adj_adversary)

        if isinstance(self.attr_adversary, SparseTensor):
            attr_adversary = self.attr_adversary.to_dense()

        return adj_adversary, attr_adversary

    @staticmethod
    @torch.no_grad()
    def evaluate_global(
        model,
        attr: TensorType["n_nodes", "n_features"],
        adj: Union[SparseTensor, TensorType["n_nodes", "n_nodes"]],
        labels: TensorType["n_nodes"],
        eval_idx: Union[List[int], np.ndarray],
    ):
        """
        Evaluates any model w.r.t. accuracy for a given (perturbed) adjacency and attribute matrix.
        """
        model.eval()
        
        pred_logits_target = model(attr, adj)[eval_idx]

        acc_test_target = accuracy(
            pred_logits_target.cpu(),
            labels.cpu()[eval_idx],
            np.arange(pred_logits_target.shape[0]),
        )

        return pred_logits_target, acc_test_target



    def calculate_loss(self, logits, labels):
        """
        TODO: maybe add formal definition for all losses? or maybe don't
        """
        if self.loss_type == "CW":
            sorted = logits.argsort(-1)
            best_non_target_class = sorted[sorted != labels[:, None]].reshape(
                logits.size(0), -1
            )[:, -1]
            margin = (
                logits[np.arange(logits.size(0)), labels]
                - logits[np.arange(logits.size(0)), best_non_target_class]
            )
            loss = -torch.clamp(margin, min=0).mean()
        elif self.loss_type == "LCW":
            sorted = logits.argsort(-1)
            best_non_target_class = sorted[sorted != labels[:, None]].reshape(
                logits.size(0), -1
            )[:, -1]
            margin = (
                logits[np.arange(logits.size(0)), labels]
                - logits[np.arange(logits.size(0)), best_non_target_class]
            )
            loss = -F.leaky_relu(margin, negative_slope=0.1).mean()
        elif self.loss_type == "tanhMargin":
            sorted = logits.argsort(-1)
            best_non_target_class = sorted[sorted != labels[:, None]].reshape(
                logits.size(0), -1
            )[:, -1]
            margin = (
                logits[np.arange(logits.size(0)), labels]
                - logits[np.arange(logits.size(0)), best_non_target_class]
            )
            loss = torch.tanh(-margin).mean()
        elif self.loss_type == "Margin":
            sorted = logits.argsort(-1)
            best_non_target_class = sorted[sorted != labels[:, None]].reshape(
                logits.size(0), -1
            )[:, -1]
            margin = (
                logits[np.arange(logits.size(0)), labels]
                - logits[np.arange(logits.size(0)), best_non_target_class]
            )
            loss = -margin.mean()
        elif self.loss_type.startswith("tanhMarginCW-"):
            alpha = float(self.loss_type.split("-")[-1])
            assert alpha >= 0, f"Alpha {alpha} must be greater or equal 0"
            assert alpha <= 1, f"Alpha {alpha} must be less or equal 1"
            sorted = logits.argsort(-1)
            best_non_target_class = sorted[sorted != labels[:, None]].reshape(
                logits.size(0), -1
            )[:, -1]
            margin = (
                logits[np.arange(logits.size(0)), labels]
                - logits[np.arange(logits.size(0)), best_non_target_class]
            )
            loss = (
                alpha * torch.tanh(-margin) - (1 - alpha) * torch.clamp(margin, min=0)
            ).mean()
        elif self.loss_type.startswith("tanhMarginMCE-"):
            alpha = float(self.loss_type.split("-")[-1])
            assert alpha >= 0, f"Alpha {alpha} must be greater or equal 0"
            assert alpha <= 1, f"Alpha {alpha} must be less or equal 1"

            sorted = logits.argsort(-1)
            best_non_target_class = sorted[sorted != labels[:, None]].reshape(
                logits.size(0), -1
            )[:, -1]
            margin = (
                logits[np.arange(logits.size(0)), labels]
                - logits[np.arange(logits.size(0)), best_non_target_class]
            )

            not_flipped = logits.argmax(-1) == labels

            loss = alpha * torch.tanh(-margin).mean() + (1 - alpha) * F.cross_entropy(
                logits[not_flipped], labels[not_flipped]
            )
        elif self.loss_type == "eluMargin":
            sorted = logits.argsort(-1)
            best_non_target_class = sorted[sorted != labels[:, None]].reshape(
                logits.size(0), -1
            )[:, -1]
            margin = (
                logits[np.arange(logits.size(0)), labels]
                - logits[np.arange(logits.size(0)), best_non_target_class]
            )
            loss = -F.elu(margin).mean()
        elif self.loss_type == "MCE":
            not_flipped = logits.argmax(-1) == labels
            loss = F.cross_entropy(logits[not_flipped], labels[not_flipped])
        elif self.loss_type == "NCE":
            sorted = logits.argsort(-1)
            best_non_target_class = sorted[sorted != labels[:, None]].reshape(
                logits.size(0), -1
            )[:, -1]
            loss = -F.cross_entropy(logits, best_non_target_class)
        else:
            loss = F.cross_entropy(logits, labels)
        return loss


# @typechecked
class SparseAttack(BaseAttack):
    """
    Base class for all sparse attacks.
    Just like the base attack class but automatically casting the adjacency to sparse format.
    """

    def __init__(
        self,
        adj: Union[SparseTensor, TensorType["n_nodes", "n_nodes"], sp.csr_matrix],
        make_undirected: bool = True,
        **kwargs,
    ):

        if isinstance(adj, torch.Tensor):
            adj = SparseTensor.from_dense(adj)
        elif isinstance(adj, sp.csr_matrix):
            adj = SparseTensor.from_scipy(adj)

        super().__init__(adj, make_undirected = make_undirected, **kwargs)

        edge_index_rows, edge_index_cols, edge_weight = adj.coo()
        self.edge_index = torch.stack([edge_index_rows, edge_index_cols], dim=0).to(
            self.data_device
        )
        self.edge_weight = edge_weight.to(self.data_device)
        self.n = adj.size(0)
        self.d = self.attr.shape[1]


# @typechecked
class SparseLocalAttack(SparseAttack):
    """
    Base class for all local sparse attacks
    """

    def get_perturbed_edges(self) -> torch.Tensor:
        """
        returns the edge (in coo format) that should be perturbed (added/deleted)
        """
        if not hasattr(self, "perturbed_edges"):
            return torch.tensor([])
        return self.perturbed_edges

    def get_logits(
        self,
        model: MODEL_TYPE,
        node_idx: int,
        perturbed_graph: SparseTensor = None,
    ):
        if perturbed_graph is None:
            perturbed_graph = self.adj
        return model(self.attr.to(self.device), perturbed_graph.to(self.device))[node_idx:node_idx + 1]

    def get_surrogate_logits(
        self,
        node_idx: int,
        perturbed_graph: SparseTensor = None,
    ) -> torch.Tensor:
        return self.get_logits(self.attacked_model, node_idx, perturbed_graph)

    def get_eval_logits(
        self,
        node_idx: int,
        perturbed_graph: SparseTensor = None,
    ) -> torch.Tensor:
        return self.get_logits(self.eval_model, node_idx, perturbed_graph)

    @torch.no_grad()
    def evaluate_local(self, node_idx: int):
        self.eval_model.eval()

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            memory = torch.cuda.memory_allocated() / (1024**3)
            logging.info(
                f"Cuda Memory before local evaluation on clean adjacency {memory}"
            )

        initial_logits = self.get_eval_logits(node_idx)

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            memory = torch.cuda.memory_allocated() / (1024**3)
            logging.info(
                f"Cuda Memory before local evaluation on perturbed adjacency {memory}"
            )

        logits = self.get_eval_logits(node_idx, self.adj_adversary)
        return logits, initial_logits

    @staticmethod
    def evaluate_global(model, attr, adj, labels: torch.Tensor, eval_idx: List[int]):
        raise NotImplementedError("Can't evaluate globally for a local attack")

    def set_eval_model(self, model):
        self.eval_model = deepcopy(model).to(self.device)

    @staticmethod
    def classification_statistics(
        logits: TensorType[1, "n_classes"], label: TensorType[()]
    ) -> Dict[str, float]:
        logits, label = F.log_softmax(logits.cpu(), dim=-1), label.cpu()
        logits = logits[0]
        logit_target = logits[label].item()
        print(logits[label])
        sorted = logits.argsort()
        logit_best_non_target = (logits[sorted[sorted != label][-1]]).item()
        confidence_target = np.exp(logit_target)
        confidence_non_target = np.exp(logit_best_non_target)
        margin = confidence_target - confidence_non_target
        return {
            "logit_target": logit_target,
            "logit_best_non_target": logit_best_non_target,
            "confidence_target": confidence_target,
            "confidence_non_target": confidence_non_target,
            "margin": margin,
        }
        
    @staticmethod
    def _margin_loss(score: Tensor, labels: Tensor,
                     idx_mask: Optional[Tensor] = None,
                     reduce: Optional[str] = None) -> Tensor:
        r"""Margin loss between true score and highest non-target score.

        .. math::
            m = - s_{y} + max_{y' \ne y} s_{y'}

        where :math:`m` is the margin :math:`s` the score and :math:`y` the
        labels.

        Args:
            score (Tensor): Some score (*e.g.*, logits) of shape
                :obj:`[n_elem, dim]`.
            labels (LongTensor): The labels of shape :obj:`[n_elem]`.
            idx_mask (Tensor, optional): To select subset of `score` and
                `labels` of shape :obj:`[n_select]`. Defaults to None.
            reduce (str, optional): if :obj:`mean` the result is aggregated.
                Otherwise, return element wise margin.

        :rtype: (Tensor)
        """
        if idx_mask is not None:
            score = score[idx_mask]
            labels = labels[idx_mask]

        linear_idx = torch.arange(score.size(0), device=score.device)
        true_score = score[linear_idx, labels]

        score = score.clone()
        score[linear_idx, labels] = float('-Inf')
        best_non_target_score = score.amax(dim=-1)

        margin_ = best_non_target_score - true_score
        if reduce is None:
            return margin_
        return margin_.mean()
    
    @staticmethod
    def _probability_margin_loss(prediction: Tensor, labels: Tensor,
                                 idx_mask: Optional[Tensor] = None) -> Tensor:
        """Calculate probability margin loss, a node-classification loss that
        focuses  on nodes next to decision boundary. See `Are Defenses for
        Graph Neural Networks Robust?
        <https://www.cs.cit.tum.de/daml/are-gnn-defenses-robust>`_ for details.

        Args:
            prediction (Tensor): Prediction of shape :obj:`[n_elem, dim]`.
            labels (LongTensor): The labels of shape :obj:`[n_elem]`.
            idx_mask (Tensor, optional): To select subset of `score` and
                `labels` of shape :obj:`[n_select]`. Defaults to None.

        :rtype: (Tensor)
        """
        prob = F.softmax(prediction, dim=-1)
        margin_ = SparseLocalAttack._margin_loss(prob, labels, idx_mask)
        return margin_.mean()
    
    def adj_adversary_for_poisoning(self):
        return self.adj_adversary


class DenseAttack(BaseAttack):

    @typechecked
    def __init__(
        self,
        adj: Union[SparseTensor, TensorType["n_nodes", "n_nodes"]],
        attr: TensorType["n_nodes", "n_features"],
        labels: TensorType["n_nodes"],
        idx_attack: np.ndarray,
        model,
        device: Union[str, int, torch.device],
        data_device: Union[str, int, torch.device],
        make_undirected: bool = True,
        loss_type: str = "CE",
        **kwargs,
    ):
        # assert isinstance(
        #     model, DenseGCN
        # ), "DenseAttacks can only attack the DenseGCN model"

        if isinstance(adj, SparseTensor):
            adj = adj.to_dense()

        super().__init__(
            adj,
            attr,
            labels,
            idx_attack,
            model,
            device,
            data_device,
            loss_type=loss_type,
            make_undirected=make_undirected,
            **kwargs,
        )

        self.n = adj.shape[0]
