"""
Code in this file is modified from: https://github.com/sigeisler/robustness_of_gnns_at_scale/tree/main/rgnn_at_scale

@inproceedings{geisler2021_robustness_of_gnns_at_scale,
    title = {Robustness of Graph Neural Networks at Scale},
    author = {Geisler, Simon and Schmidt, Tobias and \c{S}irin, Hakan and Z\"ugner, Daniel and Bojchevski, Aleksandar and G\"unnemann, Stephan},
    booktitle={Neural Information Processing Systems, {NeurIPS}},
    year = {2021},
}
"""

import logging
import inspect
import numpy as np
import torch
import torch.nn.functional as F
import scipy.sparse as sp
from tqdm.auto import tqdm
from typing import Tuple, Union, Optional, List, Dict, Any
from torchtyping import TensorType
from torch_sparse import SparseTensor
from torch_geometric.utils import to_undirected

def train(
    model,
    attr,
    adj,
    labels,
    idx_train,
    idx_val,
    idx_test,
    optimizer,
    loss,
    max_epochs,
    patience,
):
    """Train a model using either standard training.
    Parameters
    ----------
    model: torch.nn.Module
        Model which we want to train.
    attr: torch.Tensor [n, d]
        Dense attribute matrix.
    adj: torch.Tensor [n, n]
        Dense adjacency matrix.
    labels: torch.Tensor [n]
        Ground-truth labels of all nodes,
    idx_train: array-like [?]
        Indices of the training nodes.
    idx_val: array-like [?]
        Indices of the validation nodes.
    lr: float
        Learning rate.
    weight_decay : float
        Weight decay.
    patience: int
        The number of epochs to wait for the validation loss to improve before stopping early.
    max_epochs: int
        Maximum number of epochs for training.
    display_step : int
        How often to print information.
    Returns
    -------
    train_val, trace_val: list
        A tupole of lists of values of the validation loss during training.
    """
    # trace_loss_train = []
    # trace_loss_val = []
    # trace_acc_train = []
    # trace_acc_val = []
    # trace_acc_test = []
    results = []
    best_loss = np.inf
    edge_index_rows, edge_index_cols, edge_weight = adj.coo()
    edge_index = torch.stack([edge_index_rows, edge_index_cols], dim=0)
    # edge_weight2 = edge_weight2.float()
    # attr = attr.float()
    # .to(self.device)
    model.train()
    for epoch in tqdm(range(max_epochs), desc="Training"):
        optimizer.zero_grad()

        logits = get_logits(model, attr, edge_index, edge_weight)

        loss_train = loss(logits[idx_train], labels[idx_train])
        loss_val = loss(logits[idx_val], labels[idx_val])

        loss_train.backward()
        optimizer.step()

        # trace_loss_train.append(loss_train.detach().item())
        # trace_loss_val.append(loss_val.detach().item())

        # * we can run through the different metrics and append them to the results array
        train_acc = accuracy(logits, labels, idx_train)
        val_acc = accuracy(logits, labels, idx_val)
        test_acc = accuracy(logits, labels, idx_test)
        # trace_acc_train.append(train_acc)
        # trace_acc_val.append(val_acc)
        # trace_acc_test.append(test_acc)

        if loss_val < best_loss:
            best_loss = loss_val
            best_epoch = epoch
            best_state = {key: value.cpu() for key, value in model.state_dict().items()}
        else:
            if epoch >= best_epoch + patience:
                break

        result = {
            "epoch": epoch,
            "loss_train": loss_train.item(),
            "loss_val": loss_val.item(),
            "accuracy_train": train_acc,
            "accuracy_val": val_acc,
            "accuracy_test": test_acc,
        }
        results.append(result)

    # restore the best validation state
    model.load_state_dict(best_state)
    # save the model

    return results

def get_logits(model, attr, edge_index, edge_weight, idx=None):
    sig = inspect.signature(model.forward)
    # print('sig.parameters', sig.parameters)
    if "edge_weight" in sig.parameters or "edge_attr" in sig.parameters:
        logits = model(attr, edge_index, edge_weight)
    else:
        logits = model(attr, edge_index)

    if idx is not None:
        logits = logits[idx]
    return logits


def gen_local_attack_nodes(
    attr, adj, labels, model, idx_test, device, topk=10, min_node_degree=2
):
    logits, acc = evaluate_model(model, attr, adj, labels, idx_test, device)

    logging.info(f"Sample Attack Nodes for model with accuracy {acc:.4}")

    max_confidence_nodes_idx, min_confidence_nodes_idx, rand_nodes_idx = (
        sample_attack_nodes(
            logits, labels[idx_test], idx_test, adj, topk, min_node_degree
        )
    )
    tmp_nodes = np.concatenate(
        (max_confidence_nodes_idx, min_confidence_nodes_idx, rand_nodes_idx)
    )
    logging.info(
        f"Sample the following attack nodes:\n{max_confidence_nodes_idx}\n{min_confidence_nodes_idx}\n{rand_nodes_idx}"
    )
    return tmp_nodes


def sample_attack_nodes(
    logits: torch.Tensor,
    labels: torch.Tensor,
    nodes_idx,
    adj: SparseTensor,
    topk: int,
    min_node_degree: int,
):
    assert logits.shape[0] == labels.shape[0]
    if isinstance(nodes_idx, torch.Tensor):
        nodes_idx = nodes_idx.cpu()
    node_degrees = adj[nodes_idx.tolist()].sum(-1)
    print("len(node_degrees)", len(node_degrees))
    suitable_nodes_mask = (node_degrees >= min_node_degree).cpu()

    labels = labels.cpu()[suitable_nodes_mask]
    confidences = F.softmax(logits.cpu()[suitable_nodes_mask], dim=-1)

    correctly_classifed = confidences.max(-1).indices == labels

    logging.info(
        f"Found {sum(suitable_nodes_mask)} suitable '{min_node_degree}+ degree' nodes out of {len(nodes_idx)} "
        f"candidate nodes to be sampled from for the attack of which {correctly_classifed.sum().item()} have the "
        "correct class label"
    )
    print(
        f"Found {sum(suitable_nodes_mask)} suitable '{min_node_degree}+ degree' nodes out of {len(nodes_idx)} "
        f"candidate nodes to be sampled from for the attack of which {correctly_classifed.sum().item()} have the "
        "correct class label"
    )
    print(sum(suitable_nodes_mask))
    assert sum(suitable_nodes_mask) >= (
        topk * 4
    ), f"There are not enough suitable nodes to sample {(topk*4)} nodes from"

    _, max_confidence_nodes_idx = torch.topk(
        confidences[correctly_classifed].max(-1).values, k=topk
    )
    _, min_confidence_nodes_idx = torch.topk(
        -confidences[correctly_classifed].max(-1).values, k=topk
    )

    rand_nodes_idx = np.arange(correctly_classifed.sum().item())
    rand_nodes_idx = np.setdiff1d(rand_nodes_idx, max_confidence_nodes_idx)
    rand_nodes_idx = np.setdiff1d(rand_nodes_idx, min_confidence_nodes_idx)
    rnd_sample_size = min((topk * 2), len(rand_nodes_idx))
    rand_nodes_idx = np.random.choice(
        rand_nodes_idx, size=rnd_sample_size, replace=False
    )

    return (
        np.array(
            nodes_idx[suitable_nodes_mask][correctly_classifed][
                max_confidence_nodes_idx
            ]
        )[None].flatten(),
        np.array(
            nodes_idx[suitable_nodes_mask][correctly_classifed][
                min_confidence_nodes_idx
            ]
        )[None].flatten(),
        np.array(nodes_idx[suitable_nodes_mask][correctly_classifed][rand_nodes_idx])[
            None
        ].flatten(),
    )

def to_edge_index(adj, device):
    if isinstance(adj, SparseTensor):
        edge_index_rows, edge_index_cols, edge_weight = adj.coo()
        edge_index = torch.stack([edge_index_rows, edge_index_cols], dim=0).to(device)
        return edge_index, edge_weight.to(device)

@torch.no_grad()
def evaluate_model(
    model,
    attr: TensorType["n_nodes", "n_features"],
    adj: Union[SparseTensor, TensorType["n_nodes", "n_nodes"]],
    labels: TensorType["n_nodes"],
    idx_test: Union[List[int], np.ndarray],
    device: str,
):
    """
    Evaluates any model w.r.t. accuracy for a given (perturbed) adjacency and attribute matrix.
    """
    model.eval()
    
    edge_index, edge_weight = to_edge_index(adj, device)
    
    
    # pred_logits_target = model(attr, adj, edge_weight)[idx_test]
    pred_logits_target = get_logits(model, attr, edge_index, edge_weight, idx_test)
    
    acc_test_target = accuracy(
        pred_logits_target.cpu(),
        labels.cpu()[idx_test],
        np.arange(pred_logits_target.shape[0]),
    )

    return pred_logits_target, acc_test_target


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


def random_splitter(labels, n_per_class=20, seed=None):
    """
    Randomly split the training data.

    Parameters
    ----------
    labels: array-like [num_nodes]
        The class labels
    n_per_class : int
        Number of samples per class
    seed: int
        Seed

    Returns
    -------
    split_train: array-like [n_per_class * nc]
        The indices of the training nodes
    split_val: array-like [n_per_class * nc]
        The indices of the validation nodes
    split_test array-like [num_nodes - 2*n_per_class * nc]
        The indices of the test nodes
    """
    if seed is not None:
        np.random.seed(seed)
    nc = labels.max() + 1

    split_train, split_val = [], []
    for label in range(nc):
        perm = np.random.permutation((labels == label).nonzero()[0])
        split_train.append(perm[:n_per_class])
        split_val.append(perm[n_per_class : 2 * n_per_class])

    split_train = np.random.permutation(np.concatenate(split_train))
    split_val = np.random.permutation(np.concatenate(split_val))

    assert split_train.shape[0] == split_val.shape[0] == n_per_class * nc

    split_test = np.setdiff1d(
        np.arange(len(labels)), np.concatenate((split_train, split_val))
    )

    return dict(split_train, split_val, split_test)


def prepare_dataset(
    dataset,
    experiment: Dict[str, Any],
    graph_index: int,
    make_undirected: bool,
) -> Tuple[
    TensorType["num_nodes", "num_features"],
    SparseTensor,
    TensorType["num_nodes"],
    Optional[Dict[str, np.ndarray]],
]:
    """Prepares and normalizes the desired dataset

    Parameters
    ----------
    name : str
        Name of the data set
    device : Union[int, torch.device]
        `cpu` or GPU id, by default 0
    binary_attr : bool, optional
        If true the attributes are binarized (!=0), by default False
    dataset_root : str, optional
        Path where to find/store the dataset, by default "datasets"
    return_original_split: bool, optional
        If true (and the split is available for the choice of dataset) additionally the original split is returned.

    Returns
    -------
    Tuple[torch.Tensor, torch_sparse.SparseTensor, torch.Tensor]
        dense attribute tensor, sparse adjacency matrix (normalized) and labels tensor.
    """

    logging.debug("Memory Usage before loading the dataset:")
    logging.debug(torch.cuda.memory_allocated(device=None) / (1024**3))

    if graph_index is None:
        graph_index = 0

    data = dataset[graph_index]

    if hasattr(data, "num_nodes"):
        num_nodes = data.num_nodes
    else:
        num_nodes = data.x.shape[0]

    # converting to numpy arrays, so we don't have to handle different array types (tensor/numpy/list) later on.
    # Also we need numpy arrays because Numba cant determine type of torch.Tensor

    device = experiment["device"]
    # edge_index = data.edge_index
    if data.edge_attr is not None:
        edge_weight = data.edge_attr
    elif data.edge_weight is not None:
        edge_weight = data.edge_weight
    else:
        edge_weight = torch.ones(data.edge_index.shape[1])
    
    num_edges = data.edge_index.size(1)
    edge_index = data.edge_index
    if make_undirected:
        edge_index, edge_weight = to_undirected(edge_index, edge_weight, num_nodes, reduce="mean")
        num_edges = edge_index.shape[1]
        logging.debug("Memory Usage after making the graph undirected:")
        logging.debug(torch.cuda.memory_allocated(device=None) / (1024**3))
        
    adj = SparseTensor(row=edge_index[0], col=edge_index[1], value=edge_weight).t().to(device)
    
    
    # edge_weight = edge_weight.cpu()

    # adj = sp.csr_matrix((edge_weight, edge_index), (num_nodes, num_nodes))

    # del edge_index
    # # del edge_weight

    # # make unweighted
    # adj.data = np.ones_like(adj.data)

    # if make_undirected:
    #     adj = to_symmetric_scipy(adj)
    #     num_edges = adj.nnz / 2
    #     logging.debug("Memory Usage after making the graph undirected:")
    #     logging.debug(torch.cuda.memory_allocated(device=None) / (1024**3))
    # else:
    #     num_edges = adj.nnz

    # adj = SparseTensor.from_scipy(adj).coalesce().to(device)
    # if edge_weight.dtype != torch.float32:
    #     edge_weight = edge_weight.float()
    # if make_undirected:
    #     num_edges = adj.nnz() / 2
    # else:
    #     num_edges = adj.nnz()
    attr_matrix = data.x.cpu().numpy()

    attr = torch.from_numpy(attr_matrix).to(device)
    # edge_weight = edge_weight.cpu().numpy()
    # edge_weight = torch.from_numpy(edge_weight).to(device)
    
    labels = data.y.squeeze().to(device)

    split = splitter(dataset, data, labels, experiment["seed"])
    split = {k: v.numpy() for k, v in split.items()}

    experiment["model"]["params"].update(
        {
            "in_channels": attr.shape[1],
            "out_channels": int(labels[~labels.isnan()].max() + 1),
        }
    )

    return attr, adj, labels, split, num_edges
# , edge_weight


def splitter(dataset, data, labels, seed):
    """
    Splits the dataset into train, validation, and test sets.

    Args:
        dataset: The dataset object.
        data: The data object containing train, validation, and test masks.
        labels: The labels for the dataset.
        seed: The seed value for randomization.

    Returns:
        A dictionary containing the indices of the train, validation, and test sets.
    """
    if hasattr(dataset, "get_idx_split"):
        split = dataset.get_idx_split()
        logging.debug(f"Using the provided split from get_idx_split().")
        return split
    else:
        try:
            split = dict(
                train=data.train_mask.nonzero().squeeze(),
                valid=data.val_mask.nonzero().squeeze(),
                test=data.test_mask.nonzero().squeeze(),
            )
            logging.debug(f"Using the provided split with train, val, test mask.")
            return split
        except AttributeError:
            logging.debug(
                f"Dataset doesn't provide train, val, test splits. Using random_splitter() for the splitting."
            )
            return random_splitter(labels=labels.cpu().numpy(), seed=seed)


def to_symmetric_scipy(adjacency: sp.csr_matrix):
    sym_adjacency = (adjacency + adjacency.T).astype(bool).astype(float)

    sym_adjacency.tocsr().sort_indices()

    return sym_adjacency


def is_directed(adj_matrix) -> bool:
    """Check if the graph is directed (adjacency matrix is not symmetric)."""
    return (adj_matrix != adj_matrix.t()).sum() != 0
