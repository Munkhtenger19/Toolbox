"""
Code in this file is modified from: https://github.com/sigeisler/robustness_of_gnns_at_scale/tree/main/rgnn_at_scale
"""

import warnings
from abc import ABC, abstractmethod
from copy import deepcopy
from typing import Optional, Tuple, Union, List, Dict
from torchtyping import TensorType, patch_typeguard
from typeguard import typechecked
import logging
import inspect

import numpy as np
import scipy.sparse as sp
import torch
from torch import Tensor
from torch.nn import functional as F
from torch_sparse import SparseTensor

from gnn_toolbox.registration_handler.registry import registry, get_from_registry

@typechecked
class BaseAttack(ABC):
    """
    Base class for adversarial attacks

    Parameters
    ----------
    adj : SparseTensor or torch.Tensor
        [n, n] (sparse) adjacency matrix.
    attr : torch.Tensor
        [n, d] feature/attribute matrix.
    labels : torch.Tensor
        Labels vector of shape [n].
    idx_attack : np.ndarray
        Indices of the nodes which are to be attacked.
    model : MODEL_TYPE
        Model to be attacked.
    device : Union[str, int, torch.device]
        The cuda device to use for the attack
    make_undirected: bool
        Wether the perturbed adjacency matrix should be made undirected (symmetric degree normalization)
    binary_attr: bool
        If true the perturbed attributes are binarized (!=0)
    loss_type: str
        The loss to be used by a gradient based attack
    """

    def __init__(
        self,
        adj: Union[SparseTensor, TensorType],
        attr: TensorType,
        labels: TensorType,
        idx_attack: np.ndarray,
        model,
        device: Union[str, int, torch.device],
        make_undirected: bool = False,
        loss_type: str = "CE",
        **kwargs,
    ):
        self.device = device
        self.attr = attr.to(self.device) # unperturbed attributes
        self.adj = adj.to(self.device) # unperturbed adjacency 
        
        self.attr_adversary = self.attr # perturbed attributes
        self.adj_adversary = self.adj # adjacency matrix that can be perturbed
        
        self.idx_attack = idx_attack    
        self.labels = labels.to(torch.long).to(self.device)
        self.labels_attack = self.labels[self.idx_attack]
        
        self.loss_type = loss_type
        self.make_undirected = make_undirected

        self.attacked_model = deepcopy(model).to(self.device)
        self.attacked_model.eval()
        for params in self.attacked_model.parameters():
            params.requires_grad = False
            
        self.eval_model = self.attacked_model     


    @abstractmethod
    def attack(self, n_perturbations: int, **kwargs):
        pass

    # def attack(self, n_perturbations: int, **kwargs):
    #     """
    #     Executes the attack on the model updating the attributes
    #     self.adj_adversary and self.attr_adversary accordingly.

    #     Parameters
    #     ----------
    #     n_perturbations : int
    #         number of perturbations (attack budget in terms of node additions/deletions) that constrain the atack
    #     """
    #     if n_perturbations > 0:
    #         return self._attack(n_perturbations, **kwargs)
    #     else:
    #         self.attr_adversary = self.attr
    #         self.adj_adversary = self.adj
        

    def get_perturbations(self):
        adj_adversary, attr_adversary = self.adj_adversary, self.attr_adversary

        # if isinstance(self.adj_adversary, torch.Tensor):
        #     adj_adversary = SparseTensor.to_torch_sparse_coo_tensor(self.adj_adversary)

        if isinstance(self.attr_adversary, SparseTensor):
            attr_adversary = self.attr_adversary.to_dense()
        return adj_adversary, attr_adversary

    def calculate_loss(self, logits, labels):
        loss = get_from_registry("loss", self.loss_type, registry)
        return loss(logits, labels)
    
    def from_sparsetensor_to_edge_index(self, adj):
        if isinstance(adj, SparseTensor):
            edge_index_rows, edge_index_cols, edge_weight = adj.coo()
            edge_index = torch.stack([edge_index_rows, edge_index_cols], dim=0).to(self.device)
            return edge_index, edge_weight.to(self.device)
        raise ValueError("Adjacency matrix is not a SparseTensor from torch_sparse")
    
    def from_edge_index_to_sparsetensor(self, edge_index, edge_weight):
        return SparseTensor(row=edge_index[0], col=edge_index[1], value=edge_weight).to(self.device)

    
@typechecked
class GlobalAttack(BaseAttack):
    """
    Base class for all sparse attacks.
    Just like the base attack class but automatically casting the adjacency to sparse format.
    """

    def __init__(
        self,
        adj: SparseTensor,
        make_undirected: bool = True,
        **kwargs,
    ):

        super().__init__(adj, make_undirected=make_undirected, **kwargs)

        self.n = adj.size(0)
        self.d = self.attr.shape[1]
        self.num_nodes = self.attr.shape[0]

@typechecked
class LocalAttack(GlobalAttack):
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
        model,
        node_idx: int,
        perturbed_graph: Union[SparseTensor, None] = None,
    ):
        if perturbed_graph is None:
            perturbed_graph = self.adj
    
        sig = inspect.signature(model.forward)
        if "edge_weight" in sig.parameters or "edge_attr" in sig.parameters:
            edge_index, edge_weight = self.from_sparsetensor_to_edge_index(perturbed_graph)
            if edge_index is not None and edge_weight is not None:
                return model(self.attr, edge_index, edge_weight)[node_idx : node_idx + 1]
            raise ValueError("Model requires edge_weight or edge_attr but none provided")
        return model(self.attr.to(self.device), perturbed_graph.to(self.device))[node_idx : node_idx + 1]

    def get_surrogate_logits(
        self,
        node_idx: int,
        perturbed_graph: Union[SparseTensor, None] = None,
    ) -> torch.Tensor:
        return self.get_logits(self.attacked_model, node_idx, perturbed_graph)

    def get_eval_logits(
        self,
        node_idx: int,
        perturbed_graph: Optional[
            Union[SparseTensor, Tuple[TensorType, TensorType], None]
        ] = None,
    ) -> torch.Tensor:
        return self.get_logits(self.eval_model, node_idx, perturbed_graph)

    @torch.no_grad()
    def evaluate_node(self, node_idx: int):
        self.attacked_model.eval()

        if self.device == "cuda":
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            memory = torch.cuda.memory_allocated() / (1024**3)
            logging.info(
                f"Cuda Memory before local evaluation on clean adjacency {memory}"
            )

        initial_logits = self.get_eval_logits(node_idx)

        if self.device == "cuda":
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            memory = torch.cuda.memory_allocated() / (1024**3)
            logging.info(
                f"Cuda Memory before local evaluation on perturbed adjacency {memory}"
            )

        logits = self.get_eval_logits(node_idx, self.adj_adversary)
        return logits, initial_logits

    @staticmethod
    def classification_statistics(
        logits: TensorType, label: TensorType
    ) -> Dict[str, float]:
        logits, label = F.log_softmax(logits.cpu(), dim=-1), label.cpu()
        logits = logits[0]
        logit_target = logits[label].item()
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
        margin_ = LocalAttack._margin_loss(prob, labels, idx_mask)
        return margin_.mean()

    def adj_adversary_for_poisoning(self):
        return self.adj_adversary

# from torch_geometric.datasets import Planetoid
# from torch_geometric.transforms import ToSparseTensor
# from torch_geometric.utils import dense_to_sparse
# class DenseAttack(BaseAttack):
#     @typechecked
#     def __init__(
#         self,
#         adj: Union[SparseTensor, TensorType],
#         attr: TensorType,
#         labels: TensorType,
#         idx_attack: np.ndarray,
#         model,
#         device: Union[str, int, torch.device],
#         make_undirected: bool = True,
#         loss_type: str = "CE",
#         **kwargs,
#     ):
#         if isinstance(adj, SparseTensor):
#             adj = adj.to_dense()
#             # adj = dense_to_sparse(adj)
#             # cora = Planetoid(root='datasets', name='Cora',transform=ToSparseTensor(remove_edge_index=False))
#             # data = cora[0]
#             # row, col, edge_attr = data.adj_t.t().coo()
#             # edge_index = torch.stack([row, col], dim=0)
#             # adj = edge_index
#             # adj = data.adj_t.to_dense()
#             # ad

#         super().__init__(
#             adj,
#             attr,
#             labels,
#             idx_attack,
#             model,
#             device,
#             loss_type=loss_type,
#             make_undirected=make_undirected,
#             **kwargs,
#         )

#         self.n = adj.shape[0]
