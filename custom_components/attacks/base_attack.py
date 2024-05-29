"""
Code in this file is adapted from: https://github.com/sigeisler/robustness_of_gnns_at_scale/tree/main/rgnn_at_scale
"""

from abc import ABC, abstractmethod
from copy import deepcopy
from typing import Optional, Tuple, Union, Dict
from typeguard import typechecked
import logging
import inspect

import numpy as np
import torch
from torch import Tensor
from torch.nn import functional as F
from torch_sparse import SparseTensor

from gnn_toolbox.registry import registry, get_from_registry

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
        adj: SparseTensor,
        attr: Tensor,
        labels: Tensor,
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
        self.n = adj.size(0)
        self.d = self.attr.shape[1]
        self.num_nodes = self.attr.shape[0]

        self.attacked_model = deepcopy(model).to(self.device)
        self.attacked_model.eval()
        for params in self.attacked_model.parameters():
            params.requires_grad = False
            
        self.eval_model = self.attacked_model     


    @abstractmethod
    def attack(self, n_perturbations: int, **kwargs):
        pass

    def get_perturbations(self):
        adj_adversary, attr_adversary = self.adj_adversary, self.attr_adversary

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
    Base class for all global attacks.
    """

    def __init__(
        self,
        adj: SparseTensor,
        make_undirected: bool = True,
        **kwargs,
    ):

        super().__init__(adj, make_undirected=make_undirected, **kwargs)

@typechecked
class LocalAttack(BaseAttack):
    """
    Base class for all local attacks
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
            Union[SparseTensor, Tuple[Tensor, Tensor], None]
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
                f"Cuda Memory before local evaluation on perturbed adjacency: {memory}"
            )

        logits = self.get_eval_logits(node_idx, self.adj_adversary)
        return logits, initial_logits

    @staticmethod
    def classification_statistics(
        logits: Tensor, label: Tensor
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

    def adj_adversary_for_poisoning(self):
        return self.adj_adversary