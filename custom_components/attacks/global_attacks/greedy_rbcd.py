from tqdm import tqdm
import torch

from custom_components import utils
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple, Union
from torch import Tensor
from torch_geometric.utils import coalesce, to_undirected
from custom_components.attacks.global_attacks.new_prbcd import PRBCDAttack
from gnn_toolbox.registration_handler.register_components import register_global_attack

LOSS_TYPE = Callable[[Tensor, Tensor, Optional[Tensor]], Tensor]

# @register_global_attack("GreedyRBCD")
# class GreedyRBCD(PRBCD):
#     """Sampled and hence scalable PGD attack for graph data.
#     """

#     def __init__(self, epochs: int = 500, **kwargs):
#         super().__init__(**kwargs)

#         rows, cols, self.edge_weight = self.adj.coo()
#         self.edge_index = torch.stack([rows, cols], dim=0)

#         self.edge_index = self.edge_index.to(self.device)
#         self.edge_weight = self.edge_weight.float().to(self.device)
#         self.attr = self.attr.to(self.device)
#         self.epochs = epochs

#         self.n_perturbations = 0

#     def _greedy_update(self, step_size: int, gradient: torch.Tensor):
#         _, topk_edge_index = torch.topk(gradient, step_size)

#         add_edge_index = self.modified_edge_index[:, topk_edge_index]
#         add_edge_weight = torch.ones_like(add_edge_index[0], dtype=torch.float32)

#         if self.make_undirected:
#             add_edge_index, add_edge_weight = utils.to_symmetric(add_edge_index, add_edge_weight, self.n)
#         add_edge_index = torch.cat((self.edge_index, add_edge_index.to(self.device)), dim=-1)
#         add_edge_weight = torch.cat((self.edge_weight, add_edge_weight.to(self.device)))
#         edge_index, edge_weight = torch_sparse.coalesce(
#             add_edge_index, add_edge_weight, m=self.n, n=self.n, op='sum'
#         )

#         is_one_mask = torch.isclose(edge_weight, torch.tensor(1.))
#         self.edge_index = edge_index[:, is_one_mask]
#         self.edge_weight = edge_weight[is_one_mask]
#         # self.edge_weight = torch.ones_like(self.edge_weight)
#         assert self.edge_index.size(1) == self.edge_weight.size(0)

#     def attack(self, n_perturbations: int):
#         """Perform attack

#         Parameters
#         ----------
#         n_perturbations : int
#             Number of edges to be perturbed (assuming an undirected graph)
#         """
#         assert n_perturbations > self.n_perturbations, (
#             f'Number of perturbations must be bigger as this attack is greedy (current {n_perturbations}, '
#             f'previous {self.n_perturbations})'
#         )
#         n_perturbations -= self.n_perturbations
#         self.n_perturbations += n_perturbations

#         # To assert the number of perturbations later on
#         clean_edges = self.edge_index.shape[1]

#         # Determine the number of edges to be flipped in each attach step / epoch
#         step_size = n_perturbations // self.epochs
#         if step_size > 0:
#             steps = self.epochs * [step_size]
#             for i in range(n_perturbations % self.epochs):
#                 steps[i] += 1
#         else:
#             steps = [1] * n_perturbations

#         for step_size in tqdm(steps):
#             # Sample initial search space (Algorithm 2, line 3-4)
#             self.sample_random_block(step_size)
#             # Retreive sparse perturbed adjacency matrix `A \oplus p_{t-1}` (Algorithm 2, line 7)
#             edge_index, edge_weight = self.get_modified_adj()

#             if torch.cuda.is_available() and self.do_synchronize:
#                 torch.cuda.empty_cache()
#                 torch.cuda.synchronize()

#             # Calculate logits for each node (Algorithm 2, line 7)
#             logits = self._get_logits(self.attr, edge_index, edge_weight)
#             # Calculate loss combining all each node (Algorithm 2, line 8)
#             loss = self.calculate_loss(logits[self.idx_attack], self.labels[self.idx_attack])
#             # Retreive gradient towards the current block (Algorithm 2, line 8)
#             gradient = utils.grad_with_checkpoint(loss, self.perturbed_edge_weight)[0]

#             if torch.cuda.is_available() and self.do_synchronize:
#                 torch.cuda.empty_cache()
#                 torch.cuda.synchronize()

#             with torch.no_grad():
#                 # Greedy update of edges (Algorithm 2, line 8)
#                 self._greedy_update(step_size, gradient)

#             del logits
#             del loss
#             del gradient

#         allowed_perturbations = 2 * n_perturbations if self.make_undirected else n_perturbations
#         edges_after_attack = self.edge_index.shape[1]
#         assert (edges_after_attack >= clean_edges - allowed_perturbations
#                 and edges_after_attack <= clean_edges + allowed_perturbations), \
#             f'{edges_after_attack} out of range with {clean_edges} clean edges and {n_perturbations} pertutbations'

#         self.adj_adversary = SparseTensor.from_edge_index(
#             self.edge_index, self.edge_weight, (self.n, self.n)
#         ).coalesce().detach()

#         self.attr_adversary = self.attr

# @register_global_attack("GRBCD")
class GRBCDAttack(PRBCDAttack):
    r"""The Greedy Randomized Block Coordinate Descent (GRBCD) adversarial
    attack from the `Robustness of Graph Neural Networks at Scale
    <https://www.cs.cit.tum.de/daml/robustness-of-gnns-at-scale>`_ paper.

    GRBCD shares most of the properties and requirements with
    :class:`PRBCDAttack`. It also uses an efficient gradient based approach.
    However, it greedily flips edges based on the gradient towards the
    adjacency matrix.

    Args:
        model (torch.nn.Module): The GNN module to assess.
        block_size (int): Number of randomly selected elements in the
            adjacency matrix to consider.
        epochs (int, optional): Number of epochs (aborts early if
            :obj:`mode='greedy'` and budget is satisfied) (default: :obj:`125`)
        loss (str or callable, optional): A loss to quantify the "strength" of
            an attack. Note that this function must match the output format of
            :attr:`model`. By default, it is assumed that the task is
            classification and that the model returns raw predictions (*i.e.*,
            no output activation) or uses :obj:`logsoftmax`. Moreover, and the
            number of predictions should match the number of labels passed to
            :attr:`attack`. Either pass Callable or one of: :obj:`'masked'`,
            :obj:`'margin'`, :obj:`'prob_margin'`, :obj:`'tanh_margin'`.
            (default: :obj:`'masked'`)
        is_undirected (bool, optional): If :obj:`True` the graph is
            assumed to be undirected. (default: :obj:`True`)
        log (bool, optional): If set to :obj:`False`, will not log any learning
            progress. (default: :obj:`True`)
    """
    coeffs = {'max_trials_sampling': 20, 'eps': 1e-7}

    def __init__(
        self,
        block_size: int = 200_000,
        epochs: int = 125,
        loss: Optional[Union[str, LOSS_TYPE]] = 'masked',
        make_undirected: bool = True,
        log: bool = True,
        **kwargs,
    ):
        super().__init__(block_size, epochs, loss_type=loss,
                         make_undirected=make_undirected, log=log, **kwargs)

    @torch.no_grad()
    def _prepare(self, budget: int) -> List[int]:
        """Prepare attack."""
        self.flipped_edges = self.edge_index.new_empty(2, 0).to(self.device)

        # Determine the number of edges to be flipped in each attach step/epoch
        step_size = budget // self.epochs
        if step_size > 0:
            steps = self.epochs * [step_size]
            for i in range(budget % self.epochs):
                steps[i] += 1
        else:
            steps = [1] * budget

        # Sample initial search space (Algorithm 2, line 3-4)
        self._sample_random_block(step_size)

        return steps

    @torch.no_grad()
    def _update(self, step_size: int, gradient: Tensor, *args,
                **kwargs) -> Dict[str, Any]:
        """Update edge weights given gradient."""
        _, topk_edge_index = torch.topk(gradient, step_size)

        flip_edge_index = self.block_edge_index[:, topk_edge_index]
        flip_edge_weight = torch.ones_like(flip_edge_index[0],
                                           dtype=torch.float32)

        self.flipped_edges = torch.cat((self.flipped_edges, flip_edge_index),
                                       axis=-1)

        if self.is_undirected:
            flip_edge_index, flip_edge_weight = to_undirected(
                flip_edge_index, flip_edge_weight, num_nodes=self.num_nodes,
                reduce='mean')
        edge_index = torch.cat(
            (self.edge_index.to(self.device), flip_edge_index.to(self.device)),
            dim=-1)
        edge_weight = torch.cat((self.edge_weight.to(self.device),
                                 flip_edge_weight.to(self.device)))
        edge_index, edge_weight = coalesce(edge_index, edge_weight,
                                           num_nodes=self.num_nodes,
                                           reduce='sum')

        is_one_mask = torch.isclose(edge_weight, torch.tensor(1.))
        self.edge_index = edge_index[:, is_one_mask]
        self.edge_weight = edge_weight[is_one_mask]
        assert self.edge_index.size(1) == self.edge_weight.size(0)

        # Sample initial search space (Algorithm 2, line 3-4)
        self._sample_random_block(step_size)

        # Return debug information
        scalars = {
            'number_positive_entries_in_gradient': (gradient > 0).sum().item()
        }
        return scalars

    def _close(self, *args, **kwargs) -> Tuple[Tensor, Tensor]:
        """Clean up and prepare return argument."""
        return self.edge_index, self.flipped_edges