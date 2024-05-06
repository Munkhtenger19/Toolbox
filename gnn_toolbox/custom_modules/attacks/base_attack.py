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

from gnn_toolbox.registry import registry, get_from_registry

# from custom_modules.architecture import MODEL_TYPE

# from .real_base import BaseAttack
# patch_typeguard()


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
        Other models require the dataset and model to be on the same device.
    make_undirected: bool
        Wether the perturbed adjacency matrix should be made undirected (symmetric degree normalization)
    binary_attr: bool
        If true the perturbed attributes are binarized (!=0)
    loss_type: str
        The loss to be used by a gradient based attack
    """

    def __init__(
        self,
        adj: Union[SparseTensor, TensorType["n_nodes", "n_nodes"]],
        attr: TensorType["n_nodes", "n_features"],
        labels: TensorType["n_nodes"],
        idx_attack: np.ndarray,
        model
        # : MODEL_TYPE
        ,
        device: Union[str, int, torch.device],
        data_device: Union[str, int, torch.device],
        make_undirected: bool = False,
        loss_func: str = "CE",  # 'CW', 'LeakyCW'  # 'CE', 'MCE', 'Margin'
        **kwargs,
    ):

        self.device = device
        self.data_device = data_device
        self.idx_attack = idx_attack
        self.loss_func = loss_func

        self.make_undirected = make_undirected

        self.attacked_model = deepcopy(model).to(self.device)
        self.attacked_model.eval()
        for params in self.attacked_model.parameters():
            params.requires_grad = False
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

    def calculate_loss(self, logits, labels):
        loss = get_from_registry("losses", self.loss_func, registry)
        return loss(logits, labels)


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

        super().__init__(adj, make_undirected=make_undirected, **kwargs)

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
        model
        # : MODEL_TYPE
        ,
        node_idx: int,
        perturbed_graph: SparseTensor = None,
    ):
        if perturbed_graph is None:
            perturbed_graph = self.adj
        return model(self.attr.to(self.device), perturbed_graph.to(self.device))[
            node_idx : node_idx + 1
        ]

    def get_surrogate_logits(
        self,
        node_idx: int,
        perturbed_graph: SparseTensor = None,
    ) -> torch.Tensor:
        return self.get_logits(self.attacked_model, node_idx, perturbed_graph)

    def get_eval_logits(
        self,
        node_idx: int,
        perturbed_graph: Optional[
            Union[SparseTensor, Tuple[TensorType[2, "nnz"], TensorType["nnz"]]]
        ] = None,
    ) -> torch.Tensor:
        return self.get_logits(self.eval_model, node_idx, perturbed_graph)

    @torch.no_grad()
    def evaluate_local(self, node_idx: int):
        self.attacked_model.eval()

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

    def set_eval_model(self, model):
        self.eval_model = model.to(self.device)

    @staticmethod
    def _margin_loss(
        score: Tensor,
        labels: Tensor,
        idx_mask: Optional[Tensor] = None,
        reduce: Optional[str] = None,
    ) -> Tensor:
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
        score[linear_idx, labels] = float("-Inf")
        best_non_target_score = score.amax(dim=-1)

        margin_ = best_non_target_score - true_score
        if reduce is None:
            return margin_
        return margin_.mean()

    @staticmethod
    def _probability_margin_loss(
        prediction: Tensor, labels: Tensor, idx_mask: Optional[Tensor] = None
    ) -> Tensor:
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
            loss_func=loss_type,
            make_undirected=make_undirected,
            **kwargs,
        )

        self.n = adj.shape[0]
